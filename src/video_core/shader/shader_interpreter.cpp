// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <boost/container/static_vector.hpp>
#include <nihstro/shader_bytecode.h>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/vector_math.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"
#include "video_core/pica_types.h"
#include "video_core/shader/shader_interpreter.h"
#ifdef __PSVITA__
#include <atomic>
#include <cstring>
#include <string>
#include <fmt/format.h>
#include "common/vita_diag.h"
#include "video_core/shader/shader_interpreter_fast.h"
#include "video_core/shader/shader_neon_jit.h"
#endif

using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

#ifdef __PSVITA__
namespace Pica::Shader::Fast {

namespace {

/// Una fuente, con el mismo criterio que LookupSourceRegister del interprete.
void DecodeSource(Op& op, u32 n, SourceRegister reg, u32 addr_reg, bool negate,
                  const std::array<u32, 4>& selectors) {
    Src& src = op.src[n];
    switch (reg.GetRegisterType()) {
    case RegisterType::Input:
        src.type = SrcType::Input;
        break;
    case RegisterType::Temporary:
        src.type = SrcType::Temporary;
        break;
    default:
        src.type = SrcType::Uniform;
        break;
    }
    src.index = static_cast<u8>(reg.GetIndex());
    src.addr_reg = static_cast<u8>(addr_reg);
    op.negate[n] = negate ? 0x80000000u : 0u;
    // Lo que LoadSource usa desde 0.1.6.0 (ver Src).
    src.negated = negate;
    src.identity =
        selectors[0] == 0 && selectors[1] == 1 && selectors[2] == 2 && selectors[3] == 3;
    switch (src.type) {
    case SrcType::Input:
        src.offset = static_cast<u16>(ShaderUnit::InputOffset(src.index));
        break;
    case SrcType::Temporary:
        src.offset = static_cast<u16>(ShaderUnit::TemporaryOffset(src.index));
        break;
    default:
        src.offset = static_cast<u16>(src.index * sizeof(Common::Vec4<f24>));
        break;
    }
    for (u32 lane = 0; lane < 4; lane++) {
        for (u32 byte = 0; byte < 4; byte++) {
            op.swizzle[n][lane * 4 + byte] = static_cast<u8>(selectors[lane] * 4 + byte);
        }
    }
}

std::array<u32, 4> Selectors1(const SwizzlePattern& s) {
    return {static_cast<u32>(s.src1_selector_0.Value()),
            static_cast<u32>(s.src1_selector_1.Value()),
            static_cast<u32>(s.src1_selector_2.Value()),
            static_cast<u32>(s.src1_selector_3.Value())};
}
std::array<u32, 4> Selectors2(const SwizzlePattern& s) {
    return {static_cast<u32>(s.src2_selector_0.Value()),
            static_cast<u32>(s.src2_selector_1.Value()),
            static_cast<u32>(s.src2_selector_2.Value()),
            static_cast<u32>(s.src2_selector_3.Value())};
}
std::array<u32, 4> Selectors3(const SwizzlePattern& s) {
    return {static_cast<u32>(s.src3_selector_0.Value()),
            static_cast<u32>(s.src3_selector_1.Value()),
            static_cast<u32>(s.src3_selector_2.Value()),
            static_cast<u32>(s.src3_selector_3.Value())};
}

/// Destino y mascara de escritura. False si el destino es el "registro
/// ficticio" (>= 0x20): el interprete escribe ahi en un array estatico que
/// tambien se LEE como fuente invalida, y eso se deja en sus manos.
bool DecodeDest(Op& op, u32 dest_value, const SwizzlePattern& swizzle) {
    if (dest_value >= 0x20) {
        return false;
    }
    op.dest_is_output = dest_value < 0x10;
    op.dest_index = static_cast<u8>(op.dest_is_output ? dest_value : dest_value - 0x10);
    op.full_mask = true;
    for (u32 lane = 0; lane < 4; lane++) {
        op.write_mask[lane] = swizzle.DestComponentEnabled(lane) ? 0xFFFFFFFFu : 0u;
        op.full_mask = op.full_mask && op.write_mask[lane] != 0;
    }
    // Ver Op::dest_offset (0.1.6.0).
    op.dest_offset = static_cast<u16>(op.dest_is_output
                                          ? ShaderUnit::OutputOffset(op.dest_index)
                                          : ShaderUnit::TemporaryOffset(op.dest_index));
    return true;
}

/// Decodifica una instruccion. False = no entra en tramos (la hace el
/// interprete). Sigue al pie de la letra el switch de RunInterpreter.
bool DecodeInstruction(Instruction instr, const SwizzleData& swizzle_data, Op& op) {
    const auto info = instr.opcode.Value().GetInfo();
    const auto id = instr.opcode.Value().EffectiveOpCode();

    if (info.type == OpCode::Type::Arithmetic) {
        const bool inverted = (info.subtype & OpCode::Info::SrcInversed) != 0;
        const SwizzlePattern swizzle = {swizzle_data[instr.common.operand_desc_id]};
        switch (id) {
        case OpCode::Id::ADD:
            op.kind = Kind::Add;
            break;
        case OpCode::Id::MUL:
            op.kind = Kind::Mul;
            break;
        case OpCode::Id::FLR:
            op.kind = Kind::Flr;
            break;
        case OpCode::Id::MAX:
            op.kind = Kind::Max;
            break;
        case OpCode::Id::MIN:
            op.kind = Kind::Min;
            break;
        case OpCode::Id::DP3:
            op.kind = Kind::Dp3;
            break;
        case OpCode::Id::DP4:
            op.kind = Kind::Dp4;
            break;
        case OpCode::Id::DPH:
        case OpCode::Id::DPHI:
            op.kind = Kind::Dph;
            break;
        case OpCode::Id::RCP:
            op.kind = Kind::Rcp;
            break;
        case OpCode::Id::RSQ:
            op.kind = Kind::Rsq;
            break;
        case OpCode::Id::MOVA:
            op.kind = Kind::Mova;
            break;
        case OpCode::Id::MOV:
            op.kind = Kind::Mov;
            break;
        case OpCode::Id::SGE:
        case OpCode::Id::SGEI:
            op.kind = Kind::Sge;
            break;
        case OpCode::Id::SLT:
        case OpCode::Id::SLTI:
            op.kind = Kind::Slt;
            break;
        case OpCode::Id::CMP: {
            op.kind = Kind::Cmp;
            const u32 x = static_cast<u32>(instr.common.compare_op.x.Value());
            const u32 y = static_cast<u32>(instr.common.compare_op.y.Value());
            // 6 y 7: el interprete avisa y no toca el codigo de condicion.
            if (x > 5 || y > 5) {
                return false;
            }
            op.cmp_x = static_cast<u8>(x);
            op.cmp_y = static_cast<u8>(y);
            break;
        }
        case OpCode::Id::EX2:
            op.kind = Kind::Ex2;
            break;
        case OpCode::Id::LG2:
            op.kind = Kind::Lg2;
            break;
        default:
            // DST, LIT y compania: el interprete no las implementa (avisa y no
            // escribe nada). Se quedan con el.
            return false;
        }
        const u32 addr = instr.common.address_register_index;
        DecodeSource(op, 0, instr.common.GetSrc1(inverted), inverted ? 0 : addr,
                     swizzle.negate_src1.Value() != 0, Selectors1(swizzle));
        DecodeSource(op, 1, instr.common.GetSrc2(inverted), inverted ? addr : 0,
                     swizzle.negate_src2.Value() != 0, Selectors2(swizzle));
        if (op.kind == Kind::Mova) {
            op.mova_mask = static_cast<u8>((swizzle.DestComponentEnabled(0) ? 1 : 0) |
                                           (swizzle.DestComponentEnabled(1) ? 2 : 0));
            return true; // MOVA no escribe en ningun registro de datos.
        }
        if (op.kind == Kind::Cmp) {
            return true; // CMP tampoco: solo el codigo de condicion.
        }
        return DecodeDest(op, static_cast<u32>(instr.common.dest.Value()), swizzle);
    }

    if (info.type == OpCode::Type::MultiplyAdd) {
        if (id != OpCode::Id::MAD && id != OpCode::Id::MADI) {
            return false;
        }
        const bool inverted = id == OpCode::Id::MADI;
        const SwizzlePattern swizzle = {swizzle_data[instr.mad.operand_desc_id]};
        op.kind = Kind::Mad;
        const u32 addr = instr.mad.address_register_index;
        DecodeSource(op, 0, instr.mad.GetSrc1(inverted), 0, swizzle.negate_src1.Value() != 0,
                     Selectors1(swizzle));
        DecodeSource(op, 1, instr.mad.GetSrc2(inverted), inverted ? 0 : addr,
                     swizzle.negate_src2.Value() != 0, Selectors2(swizzle));
        DecodeSource(op, 2, instr.mad.GetSrc3(inverted), inverted ? addr : 0,
                     swizzle.negate_src3.Value() != 0, Selectors3(swizzle));
        return DecodeDest(op, static_cast<u32>(instr.mad.dest.Value()), swizzle);
    }

    return false;
}

} // Anonymous namespace

void Build(Program& program, const ProgramCode& code, const SwizzleData& swizzle_data,
           u32 entry_point) {
    // El interprete trata la ultima direccion como END: no entra en tramos.
    constexpr u32 kLast = MAX_PROGRAM_CODE_LENGTH - 1;
    program.entry = entry_point;

    std::vector<Op> decoded(kLast);
    std::vector<bool> ok(kLast, false);
    // +2: los destinos se marcan con dest+num y dest+1, que pueden pasarse.
    std::vector<bool> boundary(MAX_PROGRAM_CODE_LENGTH + 2, false);
    const auto mark = [&boundary](u32 address) {
        if (address < boundary.size()) {
            boundary[address] = true;
        }
    };
    mark(entry_point);

    for (u32 address = 0; address < kLast; address++) {
        Instruction instr{};
        instr.hex = code[address];
        if (DecodeInstruction(instr, swizzle_data, decoded[address])) {
            ok[address] = true;
            continue;
        }
        // Cualquier otra instruccion corta el tramo, y TODAS las direcciones
        // que su control de flujo pueda poner en una pila o usar de destino
        // pasan a ser principio de tramo: destino (IF, CALL, JMP, y else del
        // IF), final (IF, CALL), final del cuerpo de LOOP (destino + 1) y la
        // siguiente (vuelta de CALL, entrada de LOOP). Si la instruccion no es
        // de control de flujo, sobran cortes, que solo cuestan velocidad.
        const u32 dest = instr.flow_control.dest_offset;
        const u32 count = instr.flow_control.num_instructions;
        mark(address + 1);
        mark(dest);
        mark(dest + count);
        mark(dest + 1);
    }

    program.run_at.fill(kNoRun);
    program.ops.clear();
    program.runs.clear();
    for (u32 address = 0; address < kLast; address++) {
        if (!ok[address]) {
            continue;
        }
        const bool starts_run = address == 0 || !ok[address - 1] || boundary[address];
        if (starts_run) {
            program.run_at[address] = static_cast<u16>(program.runs.size());
            program.runs.push_back({address + 1, static_cast<u32>(program.ops.size()), 0});
        }
        Run& run = program.runs.back();
        program.ops.push_back(decoded[address]);
        run.count++;
        run.end = address + 1;
    }
}

} // namespace Pica::Shader::Fast
#endif // __PSVITA__

namespace Pica::Shader {

struct IfStackElement {
    u32 else_address;
    u32 end_address;
};

struct CallStackElement {
    u32 end_address;
    u32 return_address;
};

struct LoopStackElement {
    u32 entry_address;
    u32 end_address;
    u8 loop_downcounter;
    u8 address_increment;
    u8 previous_aL;
};

/**
 * Pila de capacidad fija con la MISMA semantica que boost::circular_buffer,
 * pero sin memoria dinamica.
 *
 * POR QUE. RunInterpreter se llama una vez POR VERTICE, y las tres pilas eran
 * boost::circular_buffer construidos con capacidad: cada uno reserva su
 * almacenamiento en el heap al construirse y lo libera al salir. Son tres
 * malloc y tres free por vertice -- con ~19.000 vertices por fotograma en la
 * cinematica de Rubi Omega, mas de cien mil idas y vueltas al asignador de
 * newlib, que ademas coge un cerrojo en cada una. Todo para pilas de 4 y 8
 * elementos que caben de sobra en la pila del hilo.
 *
 * LA SEMANTICA ES LA MISMA, incluido el caso raro: circular_buffer, lleno,
 * NO rechaza el push_back -- sobrescribe el elemento MAS ANTIGUO (el del
 * frente) y el nuevo pasa a ser back(). El hardware se comporta igual con los
 * anidamientos que se pasan de hondo, y es lo que el interprete reproduce, asi
 * que aqui se hace exactamente eso: 'head' marca el mas antiguo y avanza al
 * sobrescribirlo.
 *
 * Solo se usan empty(), size(), back(), push_back() y pop_back(); pop_back con
 * la pila vacia no ocurre (el interprete comprueba empty() antes), igual que
 * con circular_buffer, donde seria comportamiento indefinido.
 */
template <typename T, u32 N>
class FixedRingStack {
    static_assert((N & (N - 1)) == 0, "el indice circular usa una mascara");

public:
    bool empty() const {
        return count == 0;
    }
    u32 size() const {
        return count;
    }
    T& back() {
        return items[(head + count - 1) & (N - 1)];
    }
    void push_back(const T& value) {
        if (count == N) {
            // Llena: el nuevo ocupa el hueco del mas antiguo, que se pierde.
            items[head] = value;
            head = (head + 1) & (N - 1);
        } else {
            items[(head + count) & (N - 1)] = value;
            count++;
        }
    }
    void pop_back() {
        count--;
    }

private:
    std::array<T, N> items;
    u32 head = 0;
    u32 count = 0;
};

#ifdef __PSVITA__
/**
 * CONTROL DE FLUJO DEL PROGRAMA ENTERO COMPILADO (0.1.7.2).
 *
 * El codigo NEON de shader_neon_jit.cpp ejecuta los tramos aritmeticos en
 * linea y, para cada instruccion de control de flujo, llama aqui. Todo lo de
 * abajo es COPIA LITERAL de RunInterpreter (mismo orden, mismas pilas con la
 * misma FixedRingStack, mismas rarezas del hardware: el cuarto CALL que no
 * vuelve, LOOP que guarda como aL anterior el propio y, END que aun hace las
 * comprobaciones de pila), para que el resultado sea el del interprete bit a
 * bit. La autocomprobacion de InterpreterEngine::Run lo compara igualmente.
 */
namespace Fast {

struct FlowContext {
    ShaderUnit* state = nullptr;
    const ShaderSetup* setup = nullptr;
    FixedRingStack<IfStackElement, 8> if_stack;
    FixedRingStack<CallStackElement, 4> call_stack;
    FixedRingStack<LoopStackElement, 4> loop_stack;
    bool trapped = false;
};

namespace {

/// Las comprobaciones de pila que el interprete hace DESPUES de cada
/// instruccion. 'program_counter' llega ya incrementado.
u32 CheckStacks(FlowContext& f, u32 old_program_counter, u32 program_counter, bool is_break) {
    ShaderUnit& state = *f.state;
    u32 next_program_counter = old_program_counter + 1;
    for (u32 i = 0; i < 4; i++) {
        if (f.call_stack.empty() || f.call_stack.back().end_address != next_program_counter)
            break;
        if (i < 3) {
            program_counter = f.call_stack.back().return_address;
            next_program_counter = program_counter;
        }
        f.call_stack.pop_back();
    }
    if (!f.if_stack.empty() && f.if_stack.back().else_address == old_program_counter + 1) {
        program_counter = f.if_stack.back().end_address;
        f.if_stack.pop_back();
    }
    if (!f.loop_stack.empty() &&
        (f.loop_stack.back().end_address == old_program_counter + 1 || is_break)) {
        auto& loop = f.loop_stack.back();
        state.address_registers[2] += loop.address_increment;
        if (!is_break && loop.loop_downcounter--) {
            program_counter = loop.entry_address;
        } else {
            program_counter = loop.end_address;
            if (f.loop_stack.size() > 1)
                state.address_registers[2] = loop.previous_aL;
            f.loop_stack.pop_back();
        }
    }
    return program_counter;
}

} // Anonymous namespace

u32 FlowPostCheck(FlowContext* flow, u32 old_pc) {
    flow->state->slow_instrs++;
    return CheckStacks(*flow, old_pc, old_pc + 1, false);
}

u32 FlowTrap(FlowContext* flow, u32) {
    flow->trapped = true;
    return kFlowEnd;
}

u32 FlowStep(FlowContext* flow, u32 pc) {
    FlowContext& f = *flow;
    ShaderUnit& state = *f.state;
    const auto& uniforms = f.setup->uniforms;
    const auto& program_code = f.setup->GetProgramCode();
    state.slow_instrs++;

    u32 program_counter = pc;
    const u32 old_program_counter = pc;
    bool is_break = false;
    bool should_stop = false;

    Instruction instr{};
    if (program_counter < MAX_PROGRAM_CODE_LENGTH - 1) {
        instr.hex = program_code[program_counter];
    } else {
        instr.opcode.Assign(OpCode::Id::END);
    }

    const auto evaluate_condition = [&state](Instruction::FlowControlType flow_control) {
        using Op = Instruction::FlowControlType::Op;
        bool result_x = flow_control.refx.Value() == state.conditional_code[0];
        bool result_y = flow_control.refy.Value() == state.conditional_code[1];
        switch (flow_control.op) {
        case Op::Or:
            return result_x || result_y;
        case Op::And:
            return result_x && result_y;
        case Op::JustX:
            return result_x;
        case Op::JustY:
            return result_y;
        default:
            UNREACHABLE();
            return false;
        }
    };
    const auto do_if = [&](Instruction i, bool condition) {
        if (condition) {
            f.if_stack.push_back({
                .else_address = i.flow_control.dest_offset,
                .end_address = i.flow_control.dest_offset + i.flow_control.num_instructions,
            });
        } else {
            program_counter = i.flow_control.dest_offset - 1;
        }
    };
    const auto do_call = [&](Instruction i) {
        f.call_stack.push_back({
            .end_address = i.flow_control.dest_offset + i.flow_control.num_instructions,
            .return_address = program_counter + 1,
        });
        program_counter = i.flow_control.dest_offset - 1;
    };
    const auto do_loop = [&](Instruction i, const Common::Vec4<u8>& loop_param) {
        const u8 previous_aL = static_cast<u8>(state.address_registers[2]);
        f.loop_stack.push_back({
            .entry_address = program_counter + 1,
            .end_address = i.flow_control.dest_offset + 1,
            .loop_downcounter = loop_param.x,
            .address_increment = loop_param.z,
            .previous_aL = previous_aL,
        });
        state.address_registers[2] = loop_param.y;
    };

    switch (instr.opcode.Value()) {
    case OpCode::Id::END:
        should_stop = true;
        break;
    case OpCode::Id::JMPC:
        if (evaluate_condition(instr.flow_control)) {
            program_counter = instr.flow_control.dest_offset - 1;
        }
        break;
    case OpCode::Id::JMPU:
        if (uniforms.b[instr.flow_control.bool_uniform_id] ==
            !(instr.flow_control.num_instructions & 1)) {
            program_counter = instr.flow_control.dest_offset - 1;
        }
        break;
    case OpCode::Id::CALL:
        do_call(instr);
        break;
    case OpCode::Id::CALLU:
        if (uniforms.b[instr.flow_control.bool_uniform_id]) {
            do_call(instr);
        }
        break;
    case OpCode::Id::CALLC:
        if (evaluate_condition(instr.flow_control)) {
            do_call(instr);
        }
        break;
    case OpCode::Id::NOP:
        break;
    case OpCode::Id::IFU:
        do_if(instr, uniforms.b[instr.flow_control.bool_uniform_id]);
        break;
    case OpCode::Id::IFC:
        do_if(instr, evaluate_condition(instr.flow_control));
        break;
    case OpCode::Id::LOOP: {
        const Common::Vec4<u8>& loop_param = uniforms.i[instr.flow_control.int_uniform_id];
        state.address_registers[2] = loop_param.y;
        do_loop(instr, loop_param);
        break;
    }
    case OpCode::Id::BREAK:
        is_break = true;
        break;
    case OpCode::Id::BREAKC:
        if (evaluate_condition(instr.flow_control)) {
            is_break = true;
        }
        break;
    default:
        // El compilador solo manda aqui las instrucciones de arriba (ver
        // IsFlowInstruction en shader_neon_jit.cpp).
        f.trapped = true;
        return kFlowEnd;
    }

    ++program_counter;
    program_counter = CheckStacks(f, old_program_counter, program_counter, is_break);
    return should_stop ? kFlowEnd : program_counter;
}

/// Ejecuta un vertice con el programa entero compilado.
void RunWhole(const ShaderSetup& setup, ShaderUnit& state, const Program& program) {
    FlowContext flow;
    flow.state = &state;
    flow.setup = &setup;
    program.whole(&state, &setup.uniforms, &flow);
    if (flow.trapped) [[unlikely]] {
        // Llego a codigo sin compilar: no deberia pasar. Este programa vuelve
        // a la ruta de antes (el vertice lo repasa la comprobacion si toca).
        program.disabled.store(true, std::memory_order_relaxed);
    }
}

} // namespace Fast
#endif

template <bool Debug>
static void RunInterpreter(const ShaderSetup& setup, ShaderUnit& state,
                           DebugData<Debug>& debug_data, unsigned entry_point,
                           [[maybe_unused]] const void* fast_program = nullptr) {
#ifdef __PSVITA__
    // Programa pre-decodificado (ver shader_interpreter_fast.h), o null para el
    // interprete puro -- que es lo que usan la depuracion y la comprobacion.
    const auto* fast = static_cast<const Fast::Program*>(fast_program);
#endif
    FixedRingStack<IfStackElement, 8> if_stack;
    FixedRingStack<CallStackElement, 4> call_stack;
    FixedRingStack<LoopStackElement, 4> loop_stack;
    u32 program_counter = entry_point;

    const auto do_if = [&](Instruction instr, bool condition) {
        if (condition) {
            if_stack.push_back({
                .else_address = instr.flow_control.dest_offset,
                .end_address = instr.flow_control.dest_offset + instr.flow_control.num_instructions,
            });
        } else {
            program_counter = instr.flow_control.dest_offset - 1;
        }
    };

    const auto do_call = [&](Instruction instr) {
        call_stack.push_back({
            .end_address = instr.flow_control.dest_offset + instr.flow_control.num_instructions,
            .return_address = program_counter + 1,
        });
        program_counter = instr.flow_control.dest_offset - 1;
    };

    const auto do_loop = [&](Instruction instr, const Common::Vec4<u8>& loop_param) {
        const u8 previous_aL = static_cast<u8>(state.address_registers[2]);
        loop_stack.push_back({
            .entry_address = program_counter + 1,
            .end_address = instr.flow_control.dest_offset + 1,
            .loop_downcounter = loop_param.x,
            .address_increment = loop_param.z,
            .previous_aL = previous_aL,
        });
        state.address_registers[2] = loop_param.y;
    };

    auto evaluate_condition = [&state](Instruction::FlowControlType flow_control) {
        using Op = Instruction::FlowControlType::Op;

        bool result_x = flow_control.refx.Value() == state.conditional_code[0];
        bool result_y = flow_control.refy.Value() == state.conditional_code[1];

        switch (flow_control.op) {
        case Op::Or:
            return result_x || result_y;
        case Op::And:
            return result_x && result_y;
        case Op::JustX:
            return result_x;
        case Op::JustY:
            return result_y;
        default:
            UNREACHABLE();
            return false;
        }
    };

    const auto& uniforms = setup.uniforms;
    const auto& swizzle_data = setup.GetSwizzleData();
    const auto& program_code = setup.GetProgramCode();

    // Constants for handling invalid inputs
    static f24 dummy_vec4_float24_zeros[4] = {f24::Zero(), f24::Zero(), f24::Zero(), f24::Zero()};
    static f24 dummy_vec4_float24_ones[4] = {f24::One(), f24::One(), f24::One(), f24::One()};

    u32 iteration = 0;
    bool should_stop = false;
    while (!should_stop) {
        bool is_break = false;
        u32 old_program_counter = program_counter;

#ifdef __PSVITA__
        /**
         * Tramo pre-decodificado: se ejecuta entero y se sigue como si se
         * hubiera interpretado instruccion a instruccion hasta su ultima.
         *
         * old_program_counter pasa a ser la ULTIMA instruccion del tramo, que
         * es contra la que el codigo de abajo hace las comprobaciones de pila:
         * por como se cortan los tramos (ver Fast::Build), ninguna de las
         * instrucciones anteriores del tramo podia disparar ninguna.
         */
        bool ran_fast = false;
        if constexpr (!Debug) {
            if (fast != nullptr && program_counter < MAX_PROGRAM_CODE_LENGTH - 1) {
                const u16 run_id = fast->run_at[program_counter];
                if (run_id != Fast::kNoRun) {
                    const Fast::Run& run = fast->runs[run_id];
                    if (run.code != nullptr) {
                        run.code(&state, &setup.uniforms); // NEON (0.1.7.0)
                    } else {
                        Fast::ExecuteRun(*fast, run, state, setup.uniforms);
                    }
                    state.fast_ops += run.count;
                    old_program_counter = run.end - 1;
                    program_counter = run.end - 1;
                    ran_fast = true;
                }
            }
        }
        if (!ran_fast) {
            if constexpr (!Debug) {
                state.slow_instrs++;
            }
#endif
        // Always treat the last instruction of the program code as an
        // end instruction. This fixes some games such as Thunder Blade
        // or After Burner II which have malformed geo shaders without an
        // end instruction crashing the emulator due to the program counter
        // growing uncontrollably.
        // TODO(PabloMK7): Find how real HW reacts to this, most likely the
        // program counter wraps around after reaching the last instruction,
        // but more testing is needed.
        Instruction instr{};
        if (program_counter < MAX_PROGRAM_CODE_LENGTH - 1) {
            instr.hex = program_code[program_counter];
        } else {
            instr.opcode.Assign(OpCode::Id::END);
        }

        const SwizzlePattern swizzle = {swizzle_data[instr.common.operand_desc_id]};

        Record<DebugDataRecord::CUR_INSTR>(debug_data, iteration, program_counter);
        if (iteration > 0)
            Record<DebugDataRecord::NEXT_INSTR>(debug_data, iteration - 1, program_counter);

        debug_data.max_offset = std::max<u32>(debug_data.max_offset, 1 + program_counter);

#ifdef __PSVITA__
        /**
         * 0.1.5.2 (4.2a): este bloque solo se alcanza cuando NO ha corrido la
         * ruta rapida (esta dentro de if (!ran_fast)), asi que aqui "slow" es
         * de verdad lento. Se cuenta por OPCODE para saber cual de ellas meter
         * a la ruta rapida; el total ya lo daba slow_instrs. El overlay anota
         * las 5 mas frecuentes del intervalo en crash.txt una vez por segundo.
         */
        // QUITADO en 0.1.5.8: un fetch_add atomico por instruccion lenta, desde
        // los tres nucleos a la vez sobre la misma linea de cache, subio "sh"
        // de 92 a 103 ms en 0.1.5.7. Ya dio su dato (crash.txt de 0.1.5.7: las
        // lentas son DP4, MOV, ADD, DP3 y MUL, o sea programas enteros fuera de
        // la ruta rapida, no instrucciones sueltas).
#endif

        auto LookupSourceRegister = [&](const SourceRegister& source_reg,
                                        int address_register_index) -> const f24* {
            int index = source_reg.GetIndex();
            switch (source_reg.GetRegisterType()) {
            case RegisterType::Input:
                return &state.input[index].x;

            case RegisterType::Temporary:
                return &state.temporary[index].x;

            case RegisterType::FloatUniform:
                if (address_register_index != 0) {
                    int offset = state.address_registers[address_register_index - 1];
                    if (offset < std::numeric_limits<s8>::min() ||
                        offset > std::numeric_limits<s8>::max()) [[unlikely]] {
                        offset = 0;
                    }
                    index = (index + offset) & 0x7F;
                    // If the index is above 96, the result is all one.
                    if (index >= 96) [[unlikely]] {
                        return dummy_vec4_float24_ones;
                    }
                }
                return &uniforms.f[index].x;

            default:
                return dummy_vec4_float24_zeros;
            }
        };

        switch (instr.opcode.Value().GetInfo().type) {
        case OpCode::Type::Arithmetic: {
            const bool is_inverted =
                (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));

            const f24* src1_ =
                LookupSourceRegister(instr.common.GetSrc1(is_inverted),
                                     !is_inverted * instr.common.address_register_index);
            const f24* src2_ =
                LookupSourceRegister(instr.common.GetSrc2(is_inverted),
                                     is_inverted * instr.common.address_register_index);

            const bool negate_src1 = swizzle.negate_src1.Value() != 0;
            const bool negate_src2 = swizzle.negate_src2.Value() != 0;

            f24 src1[4] = {
                src1_[(int)swizzle.src1_selector_0.Value()],
                src1_[(int)swizzle.src1_selector_1.Value()],
                src1_[(int)swizzle.src1_selector_2.Value()],
                src1_[(int)swizzle.src1_selector_3.Value()],
            };
            if (negate_src1) {
                src1[0] = -src1[0];
                src1[1] = -src1[1];
                src1[2] = -src1[2];
                src1[3] = -src1[3];
            }
            f24 src2[4] = {
                src2_[(int)swizzle.src2_selector_0.Value()],
                src2_[(int)swizzle.src2_selector_1.Value()],
                src2_[(int)swizzle.src2_selector_2.Value()],
                src2_[(int)swizzle.src2_selector_3.Value()],
            };
            if (negate_src2) {
                src2[0] = -src2[0];
                src2[1] = -src2[1];
                src2[2] = -src2[2];
                src2[3] = -src2[3];
            }

            f24* dest =
                (instr.common.dest.Value() < 0x10)
                    ? &state.output[state.output_bank][instr.common.dest.Value().GetIndex()][0]
                : (instr.common.dest.Value() < 0x20)
                    ? &state.temporary[instr.common.dest.Value().GetIndex()][0]
                    : dummy_vec4_float24_zeros;

            debug_data.max_opdesc_id =
                std::max<u32>(debug_data.max_opdesc_id, 1 + instr.common.operand_desc_id);

            switch (instr.opcode.Value().EffectiveOpCode()) {
            case OpCode::Id::ADD: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i] + src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::MUL: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i] * src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::FLR:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = f24::FromFloat32(std::floor(src1[i].ToFloat32()));
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::MAX:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    // NOTE: Exact form required to match NaN semantics to hardware:
                    //   max(0, NaN) -> NaN
                    //   max(NaN, 0) -> 0
                    dest[i] = (src1[i] > src2[i]) ? src1[i] : src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::MIN:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    // NOTE: Exact form required to match NaN semantics to hardware:
                    //   min(0, NaN) -> NaN
                    //   min(NaN, 0) -> 0
                    dest[i] = (src1[i] < src2[i]) ? src1[i] : src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::DP3:
            case OpCode::Id::DP4:
            case OpCode::Id::DPH:
            case OpCode::Id::DPHI: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);

                OpCode::Id opcode = instr.opcode.Value().EffectiveOpCode();
                if (opcode == OpCode::Id::DPH || opcode == OpCode::Id::DPHI)
                    src1[3] = f24::One();

                int num_components = (opcode == OpCode::Id::DP3) ? 3 : 4;
                f24 dot = std::inner_product(src1, src1 + num_components, src2, f24::Zero());

                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = dot;
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            // Reciprocal
            case OpCode::Id::RCP: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                f24 rcp_res = f24::FromFloat32(1.0f / src1[0].ToFloat32());
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = rcp_res;
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            // Reciprocal Square Root
            case OpCode::Id::RSQ: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                f24 rsq_res = f24::FromFloat32(1.0f / std::sqrt(src1[0].ToFloat32()));
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = rsq_res;
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::MOVA: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                for (int i = 0; i < 2; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    // TODO: Figure out how the rounding is done on hardware
                    state.address_registers[i] = static_cast<s32>(src1[i].ToFloat32());
                }
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::MOV: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::SGE:
            case OpCode::Id::SGEI:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = (src1[i] >= src2[i]) ? f24::One() : f24::Zero();
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::SLT:
            case OpCode::Id::SLTI:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = (src1[i] < src2[i]) ? f24::One() : f24::Zero();
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::CMP:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                for (int i = 0; i < 2; ++i) {
                    // TODO: Can you restrict to one compare via dest masking?

                    auto compare_op = instr.common.compare_op;
                    auto op = (i == 0) ? compare_op.x.Value() : compare_op.y.Value();

                    switch (op) {
                    case Instruction::Common::CompareOpType::Equal:
                        state.conditional_code[i] = (src1[i] == src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::NotEqual:
                        state.conditional_code[i] = (src1[i] != src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::LessThan:
                        state.conditional_code[i] = (src1[i] < src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::LessEqual:
                        state.conditional_code[i] = (src1[i] <= src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::GreaterThan:
                        state.conditional_code[i] = (src1[i] > src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::GreaterEqual:
                        state.conditional_code[i] = (src1[i] >= src2[i]);
                        break;

                    default:
                        LOG_ERROR(HW_GPU, "Unknown compare mode {:x}", static_cast<int>(op));
                        break;
                    }
                }
                Record<DebugDataRecord::CMP_RESULT>(debug_data, iteration, state.conditional_code);
                break;

            case OpCode::Id::EX2: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);

                // EX2 only takes first component exp2 and writes it to all dest components
                f24 ex2_res = f24::FromFloat32(std::exp2(src1[0].ToFloat32()));
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = ex2_res;
                }

                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::LG2: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);

                // LG2 only takes the first component log2 and writes it to all dest components
                f24 lg2_res = f24::FromFloat32(std::log2(src1[0].ToFloat32()));
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = lg2_res;
                }

                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            default:
                LOG_ERROR(HW_GPU, "Unhandled arithmetic instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                DEBUG_ASSERT(false);
                break;
            }

            break;
        }

        case OpCode::Type::MultiplyAdd: {
            if ((instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MAD) ||
                (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI)) {
                const SwizzlePattern& mad_swizzle = *reinterpret_cast<const SwizzlePattern*>(
                    &swizzle_data[instr.mad.operand_desc_id]);

                bool is_inverted = (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI);

                const f24* src1_ = LookupSourceRegister(instr.mad.GetSrc1(is_inverted), 0);
                const f24* src2_ =
                    LookupSourceRegister(instr.mad.GetSrc2(is_inverted),
                                         !is_inverted * instr.mad.address_register_index);
                const f24* src3_ = LookupSourceRegister(
                    instr.mad.GetSrc3(is_inverted), is_inverted * instr.mad.address_register_index);

                const bool negate_src1 = mad_swizzle.negate_src1.Value() != 0;
                const bool negate_src2 = mad_swizzle.negate_src2.Value() != 0;
                const bool negate_src3 = mad_swizzle.negate_src3.Value() != 0;

                f24 src1[4] = {
                    src1_[(int)mad_swizzle.src1_selector_0.Value()],
                    src1_[(int)mad_swizzle.src1_selector_1.Value()],
                    src1_[(int)mad_swizzle.src1_selector_2.Value()],
                    src1_[(int)mad_swizzle.src1_selector_3.Value()],
                };
                if (negate_src1) {
                    src1[0] = -src1[0];
                    src1[1] = -src1[1];
                    src1[2] = -src1[2];
                    src1[3] = -src1[3];
                }
                f24 src2[4] = {
                    src2_[(int)mad_swizzle.src2_selector_0.Value()],
                    src2_[(int)mad_swizzle.src2_selector_1.Value()],
                    src2_[(int)mad_swizzle.src2_selector_2.Value()],
                    src2_[(int)mad_swizzle.src2_selector_3.Value()],
                };
                if (negate_src2) {
                    src2[0] = -src2[0];
                    src2[1] = -src2[1];
                    src2[2] = -src2[2];
                    src2[3] = -src2[3];
                }
                f24 src3[4] = {
                    src3_[(int)mad_swizzle.src3_selector_0.Value()],
                    src3_[(int)mad_swizzle.src3_selector_1.Value()],
                    src3_[(int)mad_swizzle.src3_selector_2.Value()],
                    src3_[(int)mad_swizzle.src3_selector_3.Value()],
                };
                if (negate_src3) {
                    src3[0] = -src3[0];
                    src3[1] = -src3[1];
                    src3[2] = -src3[2];
                    src3[3] = -src3[3];
                }

                f24* dest =
                    (instr.mad.dest.Value() < 0x10)
                        ? &state.output[state.output_bank][instr.mad.dest.Value().GetIndex()][0]
                    : (instr.mad.dest.Value() < 0x20)
                        ? &state.temporary[instr.mad.dest.Value().GetIndex()][0]
                        : dummy_vec4_float24_zeros;

                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::SRC3>(debug_data, iteration, src3);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!mad_swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i] * src2[i] + src3[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
            } else {
                LOG_ERROR(HW_GPU, "Unhandled multiply-add instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
            }
            break;
        }

        default: {
            // Handle each instruction on its own
            switch (instr.opcode.Value()) {
            case OpCode::Id::END:
                should_stop = true;
                break;

            case OpCode::Id::JMPC:
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                if (evaluate_condition(instr.flow_control)) {
                    program_counter = instr.flow_control.dest_offset - 1;
                }
                break;

            case OpCode::Id::JMPU:
                Record<DebugDataRecord::COND_BOOL_IN>(
                    debug_data, iteration, uniforms.b[instr.flow_control.bool_uniform_id]);

                if (uniforms.b[instr.flow_control.bool_uniform_id] ==
                    !(instr.flow_control.num_instructions & 1)) {
                    program_counter = instr.flow_control.dest_offset - 1;
                }
                break;

            case OpCode::Id::CALL:
                do_call(instr);
                break;

            case OpCode::Id::CALLU:
                Record<DebugDataRecord::COND_BOOL_IN>(
                    debug_data, iteration, uniforms.b[instr.flow_control.bool_uniform_id]);
                if (uniforms.b[instr.flow_control.bool_uniform_id]) {
                    do_call(instr);
                }
                break;

            case OpCode::Id::CALLC:
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                if (evaluate_condition(instr.flow_control)) {
                    do_call(instr);
                }
                break;

            case OpCode::Id::NOP:
                break;

            case OpCode::Id::IFU: {
                Record<DebugDataRecord::COND_BOOL_IN>(
                    debug_data, iteration, uniforms.b[instr.flow_control.bool_uniform_id]);
                const bool cond = uniforms.b[instr.flow_control.bool_uniform_id];
                do_if(instr, cond);
                break;
            }

            case OpCode::Id::IFC: {
                // TODO: Do we need to consider swizzlers here?
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                const bool cond = evaluate_condition(instr.flow_control);
                do_if(instr, cond);
                break;
            }

            case OpCode::Id::LOOP: {
                const Common::Vec4<u8>& loop_param = uniforms.i[instr.flow_control.int_uniform_id];
                state.address_registers[2] = loop_param.y;

                Record<DebugDataRecord::LOOP_INT_IN>(debug_data, iteration, loop_param);
                do_loop(instr, loop_param);
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::BREAK: {
                is_break = true;
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::BREAKC: {
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                if (evaluate_condition(instr.flow_control)) {
                    is_break = true;
                }
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::EMIT: {
                auto* emitter = state.emitter_ptr;
                ASSERT_MSG(emitter, "execute EMIT on VS");
                emitter->Emit(state.output[state.output_bank]);
                state.output_bank = !state.output_bank;
                break;
            }

            case OpCode::Id::SETEMIT: {
                auto* emitter = state.emitter_ptr;
                ASSERT_MSG(emitter, "execute SETEMIT on VS");
                emitter->emit_state.vertex_id = instr.setemit.vertex_id;
                emitter->emit_state.prim_emit = instr.setemit.prim_emit != 0;
                emitter->emit_state.winding = instr.setemit.winding != 0;
                break;
            }

            default:
                LOG_ERROR(HW_GPU, "Unhandled instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                break;
            }

            break;
        }
        }

#ifdef __PSVITA__
        } // if (!ran_fast)
#endif

        ++program_counter;
        ++iteration;

        // Stacks are checked in the order CALL -> IF -> LOOP. The CALL stack
        // can be popped multiple times per instruction. A JMP at the end of a
        // scope is never taken, this is why we compare against
        // old_program_counter + 1 here.
        u32 next_program_counter = old_program_counter + 1;
        for (u32 i = 0; i < 4; i++) {
            if (call_stack.empty() || call_stack.back().end_address != next_program_counter)
                break;
            // Hardware bug: when popping four CALL scopes at once, the last
            // one doesn't update the program counter
            if (i < 3) {
                program_counter = call_stack.back().return_address;
                next_program_counter = program_counter;
            }
            call_stack.pop_back();
        }

        // The other two stacks can only pop one entry per instruction. They
        // are checked against the original program counter before any CALL
        // scopes were closed and they overwrite any previous program counter
        // updates.
        if (!if_stack.empty() && if_stack.back().else_address == old_program_counter + 1) {
            program_counter = if_stack.back().end_address;
            if_stack.pop_back();
        }

        if (!loop_stack.empty() &&
            (loop_stack.back().end_address == old_program_counter + 1 || is_break)) {
            auto& loop = loop_stack.back();
            state.address_registers[2] += loop.address_increment;
            if (!is_break && loop.loop_downcounter--) {
                program_counter = loop.entry_address;
            } else {
                program_counter = loop.end_address;
                // Only restore previous value if there is a surrounding LOOP scope.
                if (loop_stack.size() > 1)
                    state.address_registers[2] = loop.previous_aL;
                loop_stack.pop_back();
            }
        }
    }
}

InterpreterEngine::InterpreterEngine() = default;
InterpreterEngine::~InterpreterEngine() = default;

void InterpreterEngine::SetupBatch(ShaderSetup& setup, unsigned int entry_point) {
    ASSERT(entry_point < MAX_PROGRAM_CODE_LENGTH);
    setup.DoProgramCodeFixup();
    setup.entry_point = entry_point;
#ifdef __PSVITA__
    /**
     * El programa pre-decodificado de este lote, por (codigo, swizzles,
     * entrada). Los dos hashes los guarda ShaderSetup y solo los recalcula
     * cuando el juego escribe codigo o swizzles nuevos, asi que en el caso
     * normal esto es una busqueda en una tabla por lote.
     *
     * Tope de programas: cada uno son unos kilobytes, y un juego no usa
     * cientos. Pasado el tope, los nuevos se quedan en el interprete puro
     * (cached_shader = null) en vez de vaciar la tabla: vaciarla dejaria
     * colgando el puntero que tenga guardado el OTRO ShaderSetup (el de
     * geometria o el de vertices).
     */
    constexpr std::size_t kMaxPrograms = 256;
    const u64 code_hash = setup.GetProgramCodeHash();
    const u64 swizzle_hash = setup.GetSwizzleDataHash();
    const u64 key = code_hash ^ ((swizzle_hash << 21) | (swizzle_hash >> 43)) ^
                    (static_cast<u64>(entry_point) * 0x9E3779B97F4A7C15ull);
    const auto it = fast_programs.find(key);
    if (it != fast_programs.end()) {
        setup.cached_shader = it->second.get();
        return;
    }
    if (fast_programs.size() >= kMaxPrograms) {
        setup.cached_shader = nullptr;
        return;
    }
    auto program = std::make_unique<Fast::Program>();
    program->key = key;
    Fast::Build(*program, setup.GetProgramCode(), setup.GetSwizzleData(), entry_point);
    // Programa entero a NEON (0.1.7.2); si no se puede, por tramos (0.1.7.0).
    Fast::CompileWhole(*program, setup.GetProgramCode());
    if (program->whole == nullptr) {
        Fast::CompileRuns(*program);
    }
    Common::FrameStats::fast_programs.fetch_add(1, std::memory_order_relaxed);
    setup.cached_shader = program.get();
    fast_programs.emplace(key, std::move(program));
#endif
}

MICROPROFILE_DEFINE(GPU_Shader, "GPU", "Shader", MP_RGB(50, 50, 240));

#ifdef __PSVITA__
namespace {

/// Primer sitio donde dos estados difieren, para crash.txt. False si son
/// iguales bit a bit en todo lo que un shader puede escribir.
bool FindDifference(const ShaderUnit& fast, const ShaderUnit& reference, std::string& where) {
    const auto compare_regs = [&where](const auto& a, const auto& b, const char* name) {
        for (std::size_t reg = 0; reg < a.size(); reg++) {
            for (u32 lane = 0; lane < 4; lane++) {
                u32 va;
                u32 vb;
                std::memcpy(&va, &a[reg][lane], sizeof(u32));
                std::memcpy(&vb, &b[reg][lane], sizeof(u32));
                if (va != vb) {
                    where = fmt::format("{}{}.{} rapido {:#010x} referencia {:#010x}", name, reg,
                                        "xyzw"[lane], va, vb);
                    return true;
                }
            }
        }
        return false;
    };
    if (compare_regs(fast.output[0], reference.output[0], "o") ||
        compare_regs(fast.output[1], reference.output[1], "o(banco 1)") ||
        compare_regs(fast.temporary, reference.temporary, "r")) {
        return true;
    }
    for (u32 i = 0; i < 3; i++) {
        if (fast.address_registers[i] != reference.address_registers[i]) {
            where = fmt::format("a{} rapido {} referencia {}", i, fast.address_registers[i],
                                reference.address_registers[i]);
            return true;
        }
    }
    for (u32 i = 0; i < 2; i++) {
        if (fast.conditional_code[i] != reference.conditional_code[i]) {
            where = fmt::format("cc{} rapido {} referencia {}", i, fast.conditional_code[i],
                                reference.conditional_code[i]);
            return true;
        }
    }
    if (fast.output_bank != reference.output_bank) {
        where = "banco de salida";
        return true;
    }
    return false;
}

/// El codigo del programa en crash.txt, para poder reproducir la diferencia
/// fuera de la consola. Solo la primera vez en toda la sesion.
void NoteProgram(const ShaderSetup& setup) {
    static std::atomic<bool> noted{false};
    if (noted.exchange(true)) {
        return;
    }
    const auto& code = setup.GetProgramCode();
    const auto& swizzles = setup.GetSwizzleData();
    const u32 code_size = std::min<u32>(setup.GetBiggestProgramSize(), MAX_PROGRAM_CODE_LENGTH);
    const u32 swizzle_size =
        std::min<u32>(setup.GetBiggestSwizzleSize(), MAX_SWIZZLE_DATA_LENGTH);
    Common::VitaNote("vs rapido",
                     fmt::format("entrada {} codigo {} swizzles {}", setup.entry_point, code_size,
                                 swizzle_size)
                         .c_str());
    for (u32 base = 0; base < code_size; base += 8) {
        std::string line = fmt::format("c{:04x}:", base);
        for (u32 i = base; i < std::min(base + 8, code_size); i++) {
            line += fmt::format(" {:08x}", code[i]);
        }
        Common::VitaNote("vs rapido", line.c_str());
    }
    for (u32 base = 0; base < swizzle_size; base += 8) {
        std::string line = fmt::format("s{:04x}:", base);
        for (u32 i = base; i < std::min(base + 8, swizzle_size); i++) {
            line += fmt::format(" {:08x}", swizzles[i]);
        }
        Common::VitaNote("vs rapido", line.c_str());
    }
}

} // Anonymous namespace
#endif

void InterpreterEngine::Run(const ShaderSetup& setup, ShaderUnit& state) const {

    MICROPROFILE_SCOPE(GPU_Shader);

    DebugData<false> dummy_debug_data;
#ifdef __PSVITA__
    const auto* fast = static_cast<const Fast::Program*>(setup.cached_shader);
    if (fast != nullptr && !fast->disabled.load(std::memory_order_relaxed)) {
        /**
         * AUTOCOMPROBACION. Los primeros kFullCheckVertices vertices de cada
         * programa, y despues uno de cada kSampleEvery, se ejecutan TAMBIEN con
         * el interprete puro sobre una copia del estado de entrada, y se
         * comparan bit a bit. Ver la cabecera de shader_interpreter_fast.h.
         */
        const bool check =
            fast->verified.load(std::memory_order_relaxed) < Fast::kFullCheckVertices ||
            (state.fast_check_counter++ & (Fast::kSampleEvery - 1)) == 0;
        // El programa entero compilado a NEON (0.1.7.2) si lo hay; si no, la
        // ruta rapida por tramos de siempre.
        const auto run_fast = [&] {
            if (fast->whole != nullptr) {
                Fast::RunWhole(setup, state, *fast);
            } else {
                RunInterpreter(setup, state, dummy_debug_data, setup.entry_point, fast);
            }
        };
        if (!check) [[likely]] {
            run_fast();
            return;
        }
        ShaderUnit reference = state;
        run_fast();
        RunInterpreter(setup, reference, dummy_debug_data, setup.entry_point, nullptr);
        fast->verified.fetch_add(1, std::memory_order_relaxed);
        Common::FrameStats::fast_checks.fetch_add(1, std::memory_order_relaxed);
        std::string where;
        if (FindDifference(state, reference, where)) [[unlikely]] {
            // El vertice se queda con el resultado de referencia, y el programa
            // vuelve al interprete para siempre. Solo el primer hilo que lo
            // detecte lo anota.
            state = reference;
            if (!fast->disabled.exchange(true)) {
                Common::FrameStats::fast_mismatches.fetch_add(1, std::memory_order_relaxed);
                Common::VitaNote(
                    "vs rapido",
                    fmt::format("DIFERENCIA en el programa {:016x}: {} -- vuelve al interprete",
                                fast->key, where)
                        .c_str());
                NoteProgram(setup);
            }
        }
        return;
    }
#endif
    RunInterpreter(setup, state, dummy_debug_data, setup.entry_point);
}

DebugData<true> InterpreterEngine::ProduceDebugInfo(const ShaderSetup& setup,
                                                    const AttributeBuffer& input,
                                                    const ShaderRegs& config) const {
    ShaderUnit state;
    DebugData<true> debug_data;

    // Setup input register table
    state.input.fill(Common::Vec4<f24>::AssignToAll(f24::Zero()));
    state.LoadInput(config, input);
    RunInterpreter(setup, state, debug_data, setup.entry_point);
    return debug_data;
}

} // namespace Pica::Shader
