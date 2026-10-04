// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

/**
 * RUTA RAPIDA DEL INTERPRETE DE SHADERS DE LA PICA200 (solo PS Vita).
 *
 * POR QUE. En 0.1.0.43 (cinematica de Rubi Omega) sombrear vertices costaba
 * ~103 ms de cada vblank de 353, ya repartido en los tres nucleos: unos 23 us
 * por vertice y nucleo, del orden de 10.000 ciclos. El interprete, por cada
 * instruccion y cada vertice, vuelve a sacar de sus campos de bits el tipo y
 * el indice de cada registro, los ocho selectores de swizzle, las negaciones y
 * la mascara de escritura, pasa por un switch para localizar cada fuente y
 * recorre las cuatro componentes una a una. Todo eso depende SOLO del programa,
 * que no cambia en todo el lote: se estaba decodificando decenas de miles de
 * veces lo mismo.
 *
 * QUE HACE. Al preparar el lote (SetupBatch) cada instruccion aritmetica se
 * decodifica UNA vez a un Op con todo resuelto: de donde sale cada fuente, el
 * swizzle convertido en una tabla de bytes para vtbl de NEON, la negacion como
 * mascara de signo y la mascara de escritura como mascara de carriles. Las
 * instrucciones aritmeticas seguidas se agrupan en TRAMOS, y el interprete, al
 * llegar al principio de uno, lo ejecuta entero de golpe. El control de flujo
 * (IF, CALL, LOOP, JMP, BREAK, END...) NO se toca: sigue pasando por el codigo
 * de siempre del interprete, instruccion a instruccion.
 *
 * POR QUE LOS TRAMOS NO ROMPEN EL CONTROL DE FLUJO. El interprete, DESPUES de
 * cada instruccion, compara la direccion siguiente con lo que haya en lo alto
 * de sus pilas de CALL, IF y LOOP. Si un tramo se saltara una de esas
 * comparaciones, cambiaria el flujo. Por eso los tramos se cortan en TODA
 * direccion que pueda acabar en esas pilas o ser destino de un salto: el
 * destino y el final de cada IF/CALL/LOOP/JMP del programa, el final de cada
 * cuerpo de LOOP, la instruccion que sigue a cada instruccion de control y el
 * punto de entrada. Dentro de un tramo, por construccion, ninguna comparacion
 * puede dar positivo, asi que ejecutarlo de golpe y hacer la comprobacion solo
 * al final es lo mismo que hacerla en cada instruccion.
 *
 * POR QUE EL RESULTADO ES EL MISMO BIT A BIT. Los tres nucleos corren con FZ
 * y DN activados en el FPSCR (Common::VitaEnableFastFloatMode). Con esos dos
 * bits, VFP descarta los denormales de entrada y salida y devuelve siempre el
 * NaN por defecto, que es exactamente lo que hace NEON siempre; los dos
 * redondean al mas cercano. Asi que un vmul.f32 de NEON da los mismos bits que
 * el de VFP que genera el compilador para el interprete. Encima de eso:
 *   - MUL, MAD y los DP usan el producto "saneado" de f24::operator* (un NaN que
 *     sale de dos operandos que no lo eran -- 0 x inf -- pasa a ser +0).
 *   - MAD es producto y DESPUES suma, con dos redondeos, como en el interprete.
 *   - Los DP suman en el mismo orden que std::inner_product, empezando por +0
 *     (0 + -0 = +0, y eso cambia bits).
 *   - FLR, EX2, LG2, RCP y RSQ llaman a las MISMAS funciones escalares.
 *   - MAX/MIN usan la misma forma exacta "a > b ? a : b" con sus NaN.
 *
 * Y AUN ASI SE COMPRUEBA EN LA CONSOLA. InterpreterEngine::Run ejecuta, para
 * los primeros kFullCheckVertices vertices de cada programa y despues para uno
 * de cada 64, TAMBIEN el interprete de siempre sobre una copia del estado, y
 * compara bit a bit salidas, temporales, registros de direccion y codigos de
 * condicion. Si algo difiere, el programa se desactiva para siempre (vuelve al
 * interprete), el vertice se queda con el resultado de referencia y crash.txt
 * dice que registro, que carril y que dos valores. Es la red de seguridad de
 * no poder probar esto en un PC.
 *
 * LO QUE NO ENTRA EN TRAMOS (y sigue en el interprete): DST, LIT y las demas
 * aritmeticas que el interprete no implementa, CMP con operadores 6 y 7, y
 * cualquier instruccion en la ultima direccion (el interprete la trata como
 * END).
 */

#ifdef __PSVITA__

#include <arm_neon.h>
#include <array>
#include <atomic>
#include <cmath>
#include <vector>
#include <nihstro/shader_bytecode.h>
#include "common/common_types.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"

namespace Pica::Shader::Fast {

/// Vertices comprobados enteros contra el interprete antes de pasar a
/// comprobar solo por muestreo.
constexpr u32 kFullCheckVertices = 256;
/// Despues, uno de cada kSampleEvery (potencia de dos). 0.1.6.0: 512 en vez
/// de 64. El vertice comprobado se sombrea DOS veces, una con el interprete
/// puro (varias veces mas lento que esta ruta): a 1/64 era un coste fijo de
/// varios ms por fotograma. Los primeros kFullCheckVertices siguen enteros.
constexpr u32 kSampleEvery = 512;

enum class Kind : u8 {
    Add,
    Mul,
    Dp3,
    Dp4,
    Dph,
    Max,
    Min,
    Sge,
    Slt,
    Flr,
    Mov,
    Rcp,
    Rsq,
    Ex2,
    Lg2,
    Mova,
    Cmp,
    Mad,
};

enum class SrcType : u8 {
    Input,
    Temporary,
    Uniform,
};

struct Src {
    SrcType type = SrcType::Input;
    u8 index = 0;
    /// 0 = direccionamiento directo; 1, 2, 3 = a0.x, a0.y, aL (como en el
    /// interprete, solo tiene efecto con uniforms).
    u8 addr_reg = 0;
    /**
     * 0.1.6.0: resuelto al decodificar, para no pasar por un switch en cada
     * lectura. Entrada y temporal: desplazamiento en bytes dentro de
     * ShaderUnit. Uniform sin registro de direccion: desplazamiento dentro de
     * Uniforms::f. Con registro de direccion se sigue por el camino de antes.
     */
    u16 offset = 0;
    /// El swizzle es .xyzw (la tabla no cambia nada y se salta).
    bool identity = false;
    /// La fuente va negada (ver Op::negate).
    bool negated = false;
};

struct alignas(16) Op {
    /// Indices de byte para vtbl: carril k lee los bytes 4*sel_k .. 4*sel_k+3.
    std::array<std::array<u8, 16>, 3> swizzle{};
    /// Mascara de carriles escritos (0xFFFFFFFF o 0), en el orden x, y, z, w.
    std::array<u32, 4> write_mask{};
    /// 0x80000000 si la fuente va negada: invertir el bit de signo es
    /// exactamente lo que hace el '-' de f24 (tambien con NaN y con cero).
    std::array<u32, 3> negate{};
    std::array<Src, 3> src{};
    Kind kind = Kind::Mov;
    bool dest_is_output = false;
    u8 dest_index = 0;
    /// Se escriben las cuatro componentes: se guarda sin leer ni mezclar
    /// (0.1.6.0). Mismo resultado que la mezcla con la mascara entera.
    bool full_mask = false;
    /// Desplazamiento del destino en ShaderUnit: el temporal, o la salida del
    /// banco 0 (el banco 1 esta OutputBankSize bytes despues).
    u16 dest_offset = 0;
    /// Solo MOVA: que componentes (x, y) se escriben.
    u8 mova_mask = 0;
    /// Solo CMP: operador para x y para y.
    u8 cmp_x = 0;
    u8 cmp_y = 0;
};

/// Codigo nativo de un tramo (0.1.7.0, ver shader_neon_jit.h).
using RunFn = void (*)(ShaderUnit* state, const Uniforms* uniforms);

struct Run {
    u32 end = 0;      ///< Primera direccion DESPUES del tramo.
    u32 first_op = 0; ///< Indice en Program::ops.
    u32 count = 0;
    /// El tramo compilado a NEON, o nullptr (se interpreta con ExecuteRun).
    RunFn code = nullptr;
};

constexpr u16 kNoRun = 0xFFFF;

/**
 * PROGRAMA ENTERO COMPILADO (0.1.7.2, ver shader_neon_jit.h).
 *
 * FlowContext son las tres pilas del interprete (IF, CALL, LOOP) de UN
 * vertice, con la misma clase FixedRingStack; esta definido en
 * shader_interpreter.cpp. El codigo generado llama a:
 *   FlowStep:      ejecuta la instruccion de control de flujo en 'pc' y las
 *                  comprobaciones de pila de despues, COPIADAS del bucle del
 *                  interprete; devuelve el pc siguiente o kFlowEnd.
 *   FlowPostCheck: solo las comprobaciones de pila tras un tramo que acaba en
 *                  old_pc (las mismas que hace el interprete tras el tramo).
 *   FlowTrap:      el codigo llego a una direccion sin compilar (no deberia
 *                  pasar nunca); apaga el programa y termina el vertice.
 */
struct FlowContext;
constexpr u32 kFlowEnd = 0xFFFFFFFFu;
u32 FlowStep(FlowContext* flow, u32 pc);
u32 FlowPostCheck(FlowContext* flow, u32 old_pc);
u32 FlowTrap(FlowContext* flow, u32 pc);
using ProgramFn = void (*)(ShaderUnit* state, const Uniforms* uniforms, FlowContext* flow);

struct Program {
    std::vector<Op> ops;
    std::vector<Run> runs;
    /// Tramo que EMPIEZA en cada direccion, o kNoRun.
    std::array<u16, MAX_PROGRAM_CODE_LENGTH> run_at{};
    u64 key = 0;
    /// Una diferencia contra el interprete lo apaga para siempre.
    mutable std::atomic<bool> disabled{false};
    /// Vertices ya comprobados enteros.
    mutable std::atomic<u32> verified{0};
    /// Punto de entrada (0.1.7.2).
    u32 entry = 0;
    /// El programa entero compilado a NEON, o nullptr (0.1.7.2).
    ProgramFn whole = nullptr;
    /// Direccion del codigo de cada pc del programa (para los saltos del
    /// codigo generado); las que no tienen codigo apuntan a la trampa.
    std::vector<u32> jump_table;
};

/// Decodifica el programa entero. Nunca falla: lo que no sabe hacer lo deja
/// fuera de los tramos y lo ejecuta el interprete.
void Build(Program& program, const ProgramCode& code, const SwizzleData& swizzle_data,
           u32 entry_point);

// ---------------------------------------------------------------------------
// Ejecucion
// ---------------------------------------------------------------------------

inline float32x4_t LoadSource(const Op& op, u32 n, const ShaderUnit& state,
                              const Uniforms& uniforms) {
    alignas(16) static constexpr float kOnes[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const Src& src = op.src[n];
    const u8* base;
    if (src.type != SrcType::Uniform) [[likely]] {
        // Entrada o temporal: desplazamiento ya resuelto (0.1.6.0).
        base = reinterpret_cast<const u8*>(&state) + src.offset;
    } else if (src.addr_reg == 0) [[likely]] {
        base = reinterpret_cast<const u8*>(uniforms.f.data()) + src.offset;
    } else {
        // Mismo calculo que LookupSourceRegister del interprete.
        int index = src.index;
        int offset = state.address_registers[src.addr_reg - 1];
        if (offset < -128 || offset > 127) [[unlikely]] {
            offset = 0;
        }
        index = (index + offset) & 0x7F;
        base = index >= 96 ? reinterpret_cast<const u8*>(kOnes)
                           : reinterpret_cast<const u8*>(&uniforms.f[index]);
    }
    uint8x16_t value = vld1q_u8(base);
    if (!src.identity) {
        // La tabla de bytes del swizzle. Con .xyzw es la identidad y se salta:
        // da exactamente los mismos bytes.
        const uint8x8x2_t table = {{vget_low_u8(value), vget_high_u8(value)}};
        const uint8x16_t indices = vld1q_u8(op.swizzle[n].data());
        value = vcombine_u8(vtbl2_u8(table, vget_low_u8(indices)),
                            vtbl2_u8(table, vget_high_u8(indices)));
    }
    if (src.negated) {
        value = vreinterpretq_u8_u32(
            veorq_u32(vreinterpretq_u32_u8(value), vdupq_n_u32(0x80000000u)));
    }
    return vreinterpretq_f32_u8(value);
}

/// f24::operator* en cuatro carriles: si el producto es NaN y NINGUNO de los
/// dos operandos lo era (0 x inf), el resultado es +0.
inline float32x4_t SanitizedMul(float32x4_t a, float32x4_t b) {
    const float32x4_t product = vmulq_f32(a, b);
    const uint32x4_t product_ok = vceqq_f32(product, product);
    const uint32x4_t inputs_ok = vandq_u32(vceqq_f32(a, a), vceqq_f32(b, b));
    const uint32x4_t make_zero = vbicq_u32(inputs_ok, product_ok);
    return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(product), make_zero));
}

inline void Store(const Op& op, ShaderUnit& state, float32x4_t value) {
    // Destino ya resuelto (0.1.6.0); la salida depende del banco activo.
    u8* dest = reinterpret_cast<u8*>(&state) + op.dest_offset;
    if (op.dest_is_output && state.output_bank) {
        dest += ShaderUnit::OutputBankSize;
    }
    float* dest_f = reinterpret_cast<float*>(dest);
    if (op.full_mask) [[likely]] {
        vst1q_f32(dest_f, value);
        return;
    }
    const float32x4_t old = vld1q_f32(dest_f);
    vst1q_f32(dest_f, vbslq_f32(vld1q_u32(op.write_mask.data()), value, old));
}

/// Suma en el orden de std::inner_product(src1, src1 + n, src2, +0).
template <int N>
inline float DotInOrder(float32x4_t products) {
    float acc = 0.0f;
    acc = acc + vgetq_lane_f32(products, 0);
    acc = acc + vgetq_lane_f32(products, 1);
    acc = acc + vgetq_lane_f32(products, 2);
    if constexpr (N == 4) {
        acc = acc + vgetq_lane_f32(products, 3);
    }
    return acc;
}

inline bool Compare(u8 op, float a, float b) {
    using CompareOp = nihstro::Instruction::Common::CompareOpType;
    switch (op) {
    case CompareOp::Equal:
        return a == b;
    case CompareOp::NotEqual:
        return a != b;
    case CompareOp::LessThan:
        return a < b;
    case CompareOp::LessEqual:
        return a <= b;
    case CompareOp::GreaterThan:
        return a > b;
    default: // GreaterEqual: los 6 y 7 no entran en tramos (ver Build).
        return a >= b;
    }
}

inline void ExecuteOp(const Op& op, ShaderUnit& state, const Uniforms& uniforms) {
    switch (op.kind) {
    case Kind::Add:
        Store(op, state,
              vaddq_f32(LoadSource(op, 0, state, uniforms), LoadSource(op, 1, state, uniforms)));
        break;
    case Kind::Mul:
        Store(op, state,
              SanitizedMul(LoadSource(op, 0, state, uniforms), LoadSource(op, 1, state, uniforms)));
        break;
    case Kind::Mad: {
        const float32x4_t product =
            SanitizedMul(LoadSource(op, 0, state, uniforms), LoadSource(op, 1, state, uniforms));
        Store(op, state, vaddq_f32(product, LoadSource(op, 2, state, uniforms)));
        break;
    }
    case Kind::Dp3:
    case Kind::Dp4:
    case Kind::Dph: {
        float32x4_t a = LoadSource(op, 0, state, uniforms);
        const float32x4_t b = LoadSource(op, 1, state, uniforms);
        if (op.kind == Kind::Dph) {
            // Despues de negar, como en el interprete (src1[3] = One()).
            a = vsetq_lane_f32(1.0f, a, 3);
        }
        const float32x4_t products = SanitizedMul(a, b);
        const float dot =
            op.kind == Kind::Dp3 ? DotInOrder<3>(products) : DotInOrder<4>(products);
        Store(op, state, vdupq_n_f32(dot));
        break;
    }
    case Kind::Max: {
        const float32x4_t a = LoadSource(op, 0, state, uniforms);
        const float32x4_t b = LoadSource(op, 1, state, uniforms);
        Store(op, state, vbslq_f32(vcgtq_f32(a, b), a, b));
        break;
    }
    case Kind::Min: {
        const float32x4_t a = LoadSource(op, 0, state, uniforms);
        const float32x4_t b = LoadSource(op, 1, state, uniforms);
        Store(op, state, vbslq_f32(vcltq_f32(a, b), a, b));
        break;
    }
    case Kind::Sge:
    case Kind::Slt: {
        const float32x4_t a = LoadSource(op, 0, state, uniforms);
        const float32x4_t b = LoadSource(op, 1, state, uniforms);
        const uint32x4_t pass = op.kind == Kind::Sge ? vcgeq_f32(a, b) : vcltq_f32(a, b);
        // 1.0f donde se cumple y +0 donde no (tambien con NaN: la comparacion
        // da falso, igual que el operador del interprete).
        Store(op, state,
              vreinterpretq_f32_u32(vandq_u32(pass, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
        break;
    }
    case Kind::Flr: {
        alignas(16) float lanes[4];
        vst1q_f32(lanes, LoadSource(op, 0, state, uniforms));
        for (float& lane : lanes) {
            lane = std::floor(lane);
        }
        Store(op, state, vld1q_f32(lanes));
        break;
    }
    case Kind::Mov:
        Store(op, state, LoadSource(op, 0, state, uniforms));
        break;
    case Kind::Rcp:
        Store(op, state,
              vdupq_n_f32(1.0f / vgetq_lane_f32(LoadSource(op, 0, state, uniforms), 0)));
        break;
    case Kind::Rsq:
        Store(op, state,
              vdupq_n_f32(1.0f /
                          std::sqrt(vgetq_lane_f32(LoadSource(op, 0, state, uniforms), 0))));
        break;
    case Kind::Ex2:
        Store(op, state,
              vdupq_n_f32(std::exp2(vgetq_lane_f32(LoadSource(op, 0, state, uniforms), 0))));
        break;
    case Kind::Lg2:
        Store(op, state,
              vdupq_n_f32(std::log2(vgetq_lane_f32(LoadSource(op, 0, state, uniforms), 0))));
        break;
    case Kind::Mova: {
        const float32x4_t a = LoadSource(op, 0, state, uniforms);
        if (op.mova_mask & 1) {
            state.address_registers[0] = static_cast<s32>(vgetq_lane_f32(a, 0));
        }
        if (op.mova_mask & 2) {
            state.address_registers[1] = static_cast<s32>(vgetq_lane_f32(a, 1));
        }
        break;
    }
    case Kind::Cmp: {
        const float32x4_t a = LoadSource(op, 0, state, uniforms);
        const float32x4_t b = LoadSource(op, 1, state, uniforms);
        state.conditional_code[0] =
            Compare(op.cmp_x, vgetq_lane_f32(a, 0), vgetq_lane_f32(b, 0));
        state.conditional_code[1] =
            Compare(op.cmp_y, vgetq_lane_f32(a, 1), vgetq_lane_f32(b, 1));
        break;
    }
    }
}

inline void ExecuteRun(const Program& program, const Run& run, ShaderUnit& state,
                       const Uniforms& uniforms) {
    const Op* op = program.ops.data() + run.first_op;
    const Op* const end = op + run.count;
    for (; op != end; ++op) {
        ExecuteOp(*op, state, uniforms);
    }
}

} // namespace Pica::Shader::Fast

#endif // __PSVITA__
