// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/shader/generator/cg_vs_shader_gen.h"

#include <array>
#include <functional>
#include <map>
#include <set>
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include <nihstro/shader_bytecode.h>
#include "common/logging/log.h"
#ifdef __PSVITA__
#include "common/vita_diag.h"
#endif
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/pica/shader_setup.h"

namespace Pica::Shader::Generator::GXM {

std::atomic<bool> g_allow_vs_escapes{true};
std::atomic<u32> g_cg_variant{0};
std::atomic<u32> g_cg_const_bools{0};

namespace {

using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;
using VSOutputAttributes = Pica::RasterizerRegs::VSOutputAttributes;

constexpr u32 PROGRAM_END = MAX_PROGRAM_CODE_LENGTH;

/**
 * Valor del uniform booleano 'id' si se traduce con los booleanos como
 * constantes (g_cg_const_bools, 0.1.7.4). Devuelve false si no se conocen.
 */
bool KnownBool(u32 id, bool& value) {
    const u32 bools = g_cg_const_bools.load(std::memory_order_relaxed);
    if ((bools & kCgBoolsKnown) == 0 || id >= 16) {
        return false;
    }
    value = ((bools >> id) & 1u) != 0;
    return true;
}

/// Si el JMPU 'instr' salta, con los booleanos conocidos. La prueba es la del
/// interprete: salta cuando b[id] == !(num_instructions & 1).
bool KnownJmpu(const Instruction& instr, bool& taken) {
    bool value = false;
    if (!KnownBool(instr.flow_control.bool_uniform_id, value)) {
        return false;
    }
    taken = value == ((instr.flow_control.num_instructions & 1) == 0);
    return true;
}

/// Como sale del bloque de codigo analizado. Igual que en el de GLSL.
enum class ExitMethod {
    Undetermined, ///< Valor interno; solo aparece analizando un bucle de saltos.
    AlwaysReturn, ///< Todos los caminos llegan al punto de retorno.
    Conditional,  ///< Unos llegan al retorno y otros a un END.
    AlwaysEnd,    ///< Todos los caminos llegan a un END.
};

/// Un tramo de codigo al que apunta un CALL, un IF o un LOOP.
struct Subroutine {
    [[nodiscard]] std::string GetName() const {
        return fmt::format("sub_{}_{}", begin, end);
    }

    u32 begin = 0;
    u32 end = 0;
    bool end_is_free = false; ///< Ver ControlFlowAnalyzer::ExitInfo.
    bool used_jump_to_end = false; ///< Algun salto llego a `end` (0.1.5.3).
    /// Destino de un salto que SALE de este tramo saltable hacia mas adelante
    /// (0.1.5.8), o 0 si no hay. Ver ControlFlowAnalyzer::JumpFrom.
    u32 escape = 0;
    ExitMethod exit_method = ExitMethod::Undetermined;

    bool operator<(const Subroutine& rhs) const {
        return std::tie(begin, end) < std::tie(rhs.begin, rhs.end);
    }
};

/**
 * Recorre el programa y saca la lista de subrutinas.
 *
 * Es el ControlFlowAnalyzer del generador de GLSL con dos cambios:
 *
 *   1. NO USA EXCEPCIONES. Alli, rendirse es un throw que atrapa la funcion de
 *      arriba. Aqui se levanta una bandera y se deja de trabajar; el que llama
 *      pregunta con Failed(). En un emulador que corre en una consola sin
 *      depurador, un camino de error que no depende de si las excepciones estan
 *      activadas al compilar es una cosa menos que pueda sorprender.
 *
 *   2. LOS SALTOS SE RECHAZAN AQUI MISMO. Un JMPC o un JMPU obligan al
 *      generador de GLSL a montar una maquina de estados con un switch dentro
 *      de un while(true), que es control de flujo dinamico del que no se puede
 *      dar por hecho que el compilador de Cg de la consola acepte en el perfil
 *      de vertices. Al rechazarlo aqui, ninguna subrutina tiene etiquetas y el
 *      generador de mas abajo no necesita esa maquinaria: el shader se queda en
 *      el interprete, que es correcto.
 */
class ControlFlowAnalyzer {
public:
    ControlFlowAnalyzer(const ProgramCode& program_code_, u32 main_offset)
        : program_code{program_code_} {
        // El tramo principal acaba en PROGRAM_END: llegar ahi por un salto es
        // terminar el programa, igual que salir por el final (end_is_free).
        const Subroutine* main = AddSubroutine(main_offset, PROGRAM_END, true);
        if (main == nullptr || main->exit_method != ExitMethod::AlwaysEnd) {
            // Un programa que no termina siempre no se puede traducir a una
            // funcion que siempre vuelve.
            failed = true;
        }
    }

    [[nodiscard]] bool Failed() const {
        return failed;
    }

    [[nodiscard]] const char* Reason() const {
        return reason;
    }

    std::set<Subroutine> MoveSubroutines() {
        return std::move(subroutines);
    }

private:
    /**
     * end_is_free (0.1.5.2): si llegar a 'end' por un SALTO equivale a salir
     * del tramo por su final.
     *
     * Un tramo saltable creado por JMPC/JMPU NO existe en el interprete: no
     * hay pila de IF/CALL/LOOP, el interprete simplemente va del salto al
     * destino. Saltar al final de ese tramo es exactamente lo mismo que
     * terminarlo y seguir en el final, que es lo que ya hace el codigo
     * generado. En cambio los cuerpos de IF/ELSE/CALL/CALLC/CALLU/LOOP si
     * tienen pila, y el interprete solo la compara contra la instruccion que
     * sigue EN SECUENCIA: saltar a su final no la cierra, asi que ahi el salto
     * sigue rechazado (camino exacto de 0.1.5.1).
     */
    struct ExitInfo {
        ExitMethod method = ExitMethod::Undetermined;
        bool end_is_free = false;
        bool used_jump_to_end = false; ///< Algun salto llego a `end` (0.1.5.3).
        u32 escape = 0; ///< Salto que sale del tramo (0.1.5.8), ver JumpFrom.
    };

    const ProgramCode& program_code;
    std::set<Subroutine> subroutines;
    std::map<std::pair<u32, u32>, ExitInfo> exit_method_map;
    bool failed = false;
    const char* reason = "vs flujo";

    void Fail(const char* why) {
        if (!failed) {
            failed = true;
            reason = why;
        }
    }

    const Subroutine* AddSubroutine(u32 begin, u32 end, bool end_is_free) {
        if (failed) {
            return nullptr;
        }
        const auto iter = subroutines.find(Subroutine{begin, end});
        if (iter != subroutines.end()) {
            // Si el tramo ya se analizo como libre pero ahora hace falta NO
            // libre (cuerpo de IF/CALL/LOOP), el analisis viejo solo es
            // invalido si acepto un salto a `end`. Si no lo hizo, el analisis
            // es identico para los dos casos y se reutiliza (0.1.5.3).
            if (!end_is_free && iter->end_is_free &&
                (iter->used_jump_to_end || iter->escape != 0)) {
                Fail("vs salto no estructurado");
                return nullptr;
            }
            return &*iter;
        }
        Subroutine subroutine{begin, end};
        subroutine.end_is_free = end_is_free;
        subroutine.exit_method = Scan(begin, end, end_is_free);
        const ExitInfo& scanned = exit_method_map[{begin, end}];
        subroutine.used_jump_to_end = scanned.used_jump_to_end;
        subroutine.escape = scanned.escape;
        if (failed) {
            return nullptr;
        }
        if (subroutine.exit_method == ExitMethod::Undetermined) {
            // Recursion: una subrutina que se llama a si misma. Cg no tiene
            // recursion y el hardware de la PICA tampoco de verdad.
            Fail("vs recursion");
            return nullptr;
        }
        return &*subroutines.insert(std::move(subroutine)).first;
    }

    static ExitMethod ParallelExit(ExitMethod a, ExitMethod b) {
        if (a == ExitMethod::Undetermined) {
            return b;
        }
        if (b == ExitMethod::Undetermined) {
            return a;
        }
        if (a == b) {
            return a;
        }
        return ExitMethod::Conditional;
    }

    static ExitMethod SeriesExit(ExitMethod a, ExitMethod b) {
        if (a == ExitMethod::Undetermined) {
            return ExitMethod::Undetermined;
        }
        if (a == ExitMethod::AlwaysReturn) {
            return b;
        }
        if (b == ExitMethod::Undetermined || b == ExitMethod::AlwaysEnd) {
            return ExitMethod::AlwaysEnd;
        }
        return ExitMethod::Conditional;
    }

    /**
     * La cola de un tramo (lo que va detras de un salto, CALL, IF o LOOP)
     * forma parte del MISMO tramo: sus saltos al final y sus escapes son del
     * tramo. Antes (0.1.5.3) used_jump_to_end de la cola no subia al tramo,
     * y un tramo con un salto a su final en la cola podia reutilizarse como
     * cuerpo de IF/CALL, donde ese salto no vale.
     */
    void MergeTail(ExitInfo& info, u32 tail_begin, u32 end) {
        const auto it = exit_method_map.find({tail_begin, end});
        if (it == exit_method_map.end()) {
            return;
        }
        if (it->second.used_jump_to_end) {
            info.used_jump_to_end = true;
        }
        if (it->second.escape != 0) {
            if (info.escape != 0 && info.escape != it->second.escape) {
                Fail("vs salto no estructurado");
                return;
            }
            info.escape = it->second.escape;
        }
    }

    /**
     * Un salto hacia delante (0.1.5.8), real o "virtual". El trozo saltable
     * empieza en 'skip_begin' y el salto va a 'target', dentro del tramo
     * [begin, end) que se esta analizando.
     *
     *   - target < end: "si no se cumple, ejecuta [skip_begin, target)" y sigue
     *     en target (0.1.4.8).
     *   - target == end: igual, sin cola; solo en tramos de final libre
     *     (0.1.5.2).
     *   - target > end: SALE del tramo (0.1.5.8, crash.txt de 0.1.5.7:
     *     "0x130 -> 0x157, tramo [0x12a, 0x131)"). Solo en un tramo de final
     *     libre, es decir, un trozo saltable de otro salto: el interprete no
     *     tiene pila que cerrar. Se traduce como un salto al final del tramo
     *     que ademas levanta la variable de escape del tramo; quien llama al
     *     tramo, al volver, hace un salto VIRTUAL en el final del tramo hacia
     *     'target', tomado si la variable esta levantada. Ese salto virtual es
     *     otra llamada a esta misma funcion en el tramo de fuera, asi que puede
     *     volver a escapar hacia arriba. Un tramo solo admite UN destino de
     *     escape (si hubiera dos distintos, se rechaza).
     *
     * El patron medido es exactamente:
     *     JMPC c -> E      ; tramo saltable [.., E)
     *       ...
     *       JMPU b -> T    ; T > E
     *     [E, T)           ; se salta si se tomo el JMPU
     *     T: ...
     * que es "if (!c) { A; if (b) goto T; } B; T:" y queda
     *     esc = false; if (!c) { A; if (b) esc = true; else {...} } if (!esc) { B }
     */
    ExitMethod JumpFrom(u32 offset, u32 skip_begin, u32 target, u32 begin, u32 end,
                        bool end_is_free, ExitInfo& info) {
        u32 land = target;
        /**
         * DESACTIVADO en 0.1.5.9. Con los escapes, 0.1.5.8 fue de 1.8 a 0.6 FPS:
         * "vsg 0 no 142 fragmentos" y todo a software. Lo mas probable es que
         * uno de estos shaders rompiera el compilador de Cg de la consola, que
         * a partir de un error interno ya no compila nada, tampoco los de
         * fragmentos (lo mismo que en 0.1.4.7). Hasta ver el shader que lo
         * rompe (gxm_cg.cpp lo guarda ahora en ux0:/data/azahar/cg_error.txt)
         * estos saltos vuelven a la CPU, como en 0.1.5.7.
         */
        if (target > end && !g_allow_vs_escapes.load(std::memory_order_relaxed)) {
#ifdef __PSVITA__
            NoteUnstructuredJump(offset, target, begin, end);
#endif
            Fail("vs salto no estructurado");
            return ExitMethod::AlwaysReturn;
        }
        if (target > end) {
            if (!end_is_free) {
#ifdef __PSVITA__
                NoteUnstructuredJump(offset, target, begin, end);
#endif
                Fail("vs salto no estructurado");
                return ExitMethod::AlwaysReturn;
            }
            if (info.escape != 0 && info.escape != target) {
                Fail("vs salto no estructurado");
                return ExitMethod::AlwaysReturn;
            }
            info.escape = target;
            land = end;
        } else if (target == end && !end_is_free) {
#ifdef __PSVITA__
            NoteUnstructuredJump(offset, target, begin, end);
#endif
            Fail("vs salto no estructurado");
            return ExitMethod::AlwaysReturn;
        }
        ExitMethod skip_exit = ExitMethod::AlwaysReturn;
        if (skip_begin < land) {
            // El trozo saltable es un tramo creado por el salto: su final es
            // libre (ver ExitInfo).
            const Subroutine* skip = AddSubroutine(skip_begin, land, true);
            if (skip == nullptr) {
                return ExitMethod::AlwaysReturn;
            }
            skip_exit = skip->exit_method;
            if (skip->escape != 0) {
                // El trozo escapa: salto virtual en 'land' hacia skip->escape.
                const ExitMethod rest =
                    JumpFrom(land, land, skip->escape, begin, end, end_is_free, info);
                return SeriesExit(ParallelExit(skip_exit, ExitMethod::AlwaysReturn), rest);
            }
        }
        if (land == end) {
            // Sin cola. Marcar que un salto llego a `end`: si despues este tramo
            // se necesita como cuerpo de IF/CALL/LOOP, el analisis no sirve.
            info.used_jump_to_end = true;
            return ParallelExit(skip_exit, ExitMethod::AlwaysReturn);
        }
        // La cola hereda end_is_free del tramo en el que esta.
        const ExitMethod after_jump = Scan(land, end, end_is_free);
        MergeTail(info, land, end);
        return SeriesExit(ParallelExit(skip_exit, ExitMethod::AlwaysReturn), after_jump);
    }

#ifdef __PSVITA__
    /**
     * Diagnostico (0.1.4.9): que salto se rechaza exactamente. En Rubi Omega
     * casi todos los lotes se quedan en la CPU por "vs salto no estructurado"
     * y hace falta saber si son saltos hacia atras (bucles hechos con saltos),
     * saltos que salen de un IF, o saltos al final de una subrutina, porque
     * cada caso se traduce distinto. Las primeras veces, y el programa entero
     * una vez, para poder reproducirlo fuera de la consola.
     */
    void NoteUnstructuredJump(u32 offset, u32 target, u32 begin, u32 end) {
        static u32 notes = 0;
        if (notes >= 12) {
            return;
        }
        notes++;
        Common::VitaNote("vs salto",
                         fmt::format("salto en {:#05x} -> {:#05x}, tramo [{:#05x}, {:#05x}) {}",
                                     offset, target, begin, end,
                                     target <= offset ? "HACIA ATRAS" : "fuera del tramo")
                             .c_str());
        static bool dumped = false;
        if (dumped) {
            return;
        }
        dumped = true;
        u32 last = 0;
        for (u32 i = 0; i < MAX_PROGRAM_CODE_LENGTH; i++) {
            if (program_code[i] != 0) {
                last = i;
            }
        }
        last = std::min<u32>(last, 1023);
        for (u32 base = 0; base <= last; base += 8) {
            std::string line = fmt::format("c{:04x}:", base);
            for (u32 i = base; i < base + 8 && i <= last; i++) {
                line += fmt::format(" {:08x}", program_code[i]);
            }
            Common::VitaNote("vs salto", line.c_str());
        }
    }
#endif

    ExitMethod Scan(u32 begin, u32 end, bool end_is_free) {
        auto [iter, inserted] =
            exit_method_map.emplace(std::make_pair(begin, end), ExitInfo{{}, end_is_free});
        ExitInfo& info = iter->second;
        if (!inserted) {
            // Si el tramo ya se analizo como libre y ahora hace falta no libre,
            // el analisis viejo solo es invalido si acepto un salto a `end`.
            if (!end_is_free && info.end_is_free && (info.used_jump_to_end || info.escape != 0)) {
                Fail("vs salto no estructurado");
                return ExitMethod::AlwaysReturn;
            }
            return info.method;
        }

        for (u32 offset = begin; offset != end && offset != PROGRAM_END; ++offset) {
            if (failed) {
                return info.method = ExitMethod::AlwaysReturn;
            }
            const Instruction instr = {program_code[offset]};
            switch (instr.opcode.Value()) {
            case OpCode::Id::END:
                return info.method = ExitMethod::AlwaysEnd;
            case OpCode::Id::JMPC:
            case OpCode::Id::JMPU: {
                /**
                 * SALTOS HACIA DELANTE = UN "IF" DISFRAZADO (0.1.4.8).
                 *
                 * En el interprete, un salto que se cumple pone el PC en el
                 * destino y ya; no toca ninguna pila. Si el destino esta
                 * DELANTE y DENTRO del tramo que se esta recorriendo, eso es
                 * exactamente "si NO se cumple, ejecuta [salto+1, destino); y
                 * en los dos casos sigue en el destino". Se traduce asi, con el
                 * trozo saltable como una subrutina mas (lo mismo que los
                 * cuerpos de IF), sin maquina de estados ni bucles.
                 *
                 * Por que NO la maquina de estados de GLSL: se probo en 0.1.4.7
                 * (un for de 65.536 vueltas con una cadena de if) y el
                 * compilador de Cg de la consola respondio "fatal internal
                 * error" -- y a partir de ahi fallaron TODAS las compilaciones,
                 * tambien las de fragmentos: la partida entera a software.
                 *
                 * Saltar JUSTO AL FINAL del tramo (0.1.5.2): antes se exigia
                 * destino < end estricto para todos los tramos. Esa regla
                 * existe porque un cuerpo de IF/CALL/LOOP tiene PILA en el
                 * interprete y saltar a su final no la cierra. Pero un tramo
                 * saltable creado por un JMPC/JMPU no tiene pila: llegar a su
                 * final por un salto es salir del tramo, igual que terminarlo.
                 * El caso medido en 0.1.5.1 era exactamente ese (dos saltos
                 * anidados al mismo destino, el interior apuntando al final
                 * del tramo saltable del exterior).
                 *
                 * Hacia atras o fuera del tramo: sigue en el interprete.
                 */
                const u32 target = instr.flow_control.dest_offset;
                // Con los booleanos como constantes (0.1.7.4) un JMPU que no
                // salta nunca es un NOP, y uno que salta siempre deja muerto
                // [salto+1, destino): ni se analiza.
                bool taken = false;
                const bool known =
                    instr.opcode.Value() == OpCode::Id::JMPU && KnownJmpu(instr, taken);
                if (known && !taken) {
                    break;
                }
                if (target <= offset) {
#ifdef __PSVITA__
                    NoteUnstructuredJump(offset, target, begin, end);
#endif
                    Fail("vs salto no estructurado");
                    return info.method = ExitMethod::AlwaysReturn;
                }
                if (target == offset + 1) {
                    // Saltar a la siguiente: no hace nada en ninguno de los dos
                    // casos.
                    break;
                }
                // Salto seguro: un salto con el trozo saltable vacio.
                return info.method = JumpFrom(offset, known ? target : offset + 1, target, begin,
                                              end, end_is_free, info);
            }
            case OpCode::Id::CALL:
            case OpCode::Id::CALLC:
            case OpCode::Id::CALLU: {
                // CALLU con el booleano conocido (0.1.7.4): o no llama nunca
                // (NOP) o llama siempre, igual que un CALL.
                bool always = instr.opcode.Value() == OpCode::Id::CALL;
                if (instr.opcode.Value() == OpCode::Id::CALLU) {
                    bool value = false;
                    if (KnownBool(instr.flow_control.bool_uniform_id, value)) {
                        if (!value) {
                            break;
                        }
                        always = true;
                    }
                }
                const Subroutine* call =
                    AddSubroutine(instr.flow_control.dest_offset,
                                  instr.flow_control.dest_offset +
                                      instr.flow_control.num_instructions,
                                  false);
                if (call == nullptr) {
                    return info.method = ExitMethod::AlwaysReturn;
                }
                if (always) {
                    if (call->exit_method == ExitMethod::AlwaysEnd) {
                        return info.method = ExitMethod::AlwaysEnd;
                    }
                    const ExitMethod after_call = Scan(offset + 1, end, end_is_free);
                    MergeTail(info, offset + 1, end);
                    return info.method = SeriesExit(call->exit_method, after_call);
                }
                const ExitMethod after_call = Scan(offset + 1, end, end_is_free);
                MergeTail(info, offset + 1, end);
                return info.method = SeriesExit(
                           ParallelExit(call->exit_method, ExitMethod::AlwaysReturn), after_call);
            }
            case OpCode::Id::LOOP: {
                // Un cuerpo de LOOP hacia atras, o que se pasa del final del
                // tramo en el que empieza, no tiene traduccion estructurada
                // (ver los saltos). Antes de crear la subrutina del cuerpo.
                if (instr.flow_control.dest_offset < offset ||
                    instr.flow_control.dest_offset + 1 > end) {
                    Fail("vs estructura");
                    return info.method = ExitMethod::AlwaysReturn;
                }
                const Subroutine* loop =
                    AddSubroutine(offset + 1, instr.flow_control.dest_offset + 1, false);
                if (loop == nullptr) {
                    return info.method = ExitMethod::AlwaysReturn;
                }
                if (loop->exit_method == ExitMethod::AlwaysEnd) {
                    return info.method = ExitMethod::AlwaysEnd;
                }
                const ExitMethod after_loop =
                    Scan(instr.flow_control.dest_offset + 1, end, end_is_free);
                MergeTail(info, instr.flow_control.dest_offset + 1, end);
                return info.method = SeriesExit(loop->exit_method, after_loop);
            }
            case OpCode::Id::IFU:
            case OpCode::Id::IFC: {
                // Un IF cuyo else o final caen fuera del tramo (o hacia atras)
                // no tiene traduccion estructurada: ver los saltos. Se mira
                // ANTES de crear las subrutinas de los cuerpos, que con un
                // tramo al reves recorrerian el programa hasta el final.
                if (instr.flow_control.dest_offset <= offset ||
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions > end) {
                    Fail("vs estructura");
                    return info.method = ExitMethod::AlwaysReturn;
                }
                const u32 endif =
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions;
                bool value = false;
                if (instr.opcode.Value() == OpCode::Id::IFU &&
                    KnownBool(instr.flow_control.bool_uniform_id, value)) {
                    // IFU con el booleano conocido (0.1.7.4): solo el cuerpo que
                    // se ejecuta; el otro ni se analiza.
                    ExitMethod taken = ExitMethod::AlwaysReturn;
                    if (value || instr.flow_control.num_instructions != 0) {
                        const Subroutine* body =
                            value ? AddSubroutine(offset + 1, instr.flow_control.dest_offset, false)
                                  : AddSubroutine(instr.flow_control.dest_offset, endif, false);
                        if (body == nullptr) {
                            return info.method = ExitMethod::AlwaysReturn;
                        }
                        taken = body->exit_method;
                    }
                    if (taken == ExitMethod::AlwaysEnd) {
                        return info.method = ExitMethod::AlwaysEnd;
                    }
                    const ExitMethod after_if = Scan(endif, end, end_is_free);
                    MergeTail(info, endif, end);
                    return info.method = SeriesExit(taken, after_if);
                }
                const Subroutine* if_sub =
                    AddSubroutine(offset + 1, instr.flow_control.dest_offset, false);
                if (if_sub == nullptr) {
                    return info.method = ExitMethod::AlwaysReturn;
                }
                ExitMethod else_method = ExitMethod::AlwaysReturn;
                if (instr.flow_control.num_instructions != 0) {
                    const Subroutine* else_sub =
                        AddSubroutine(instr.flow_control.dest_offset,
                                      instr.flow_control.dest_offset +
                                          instr.flow_control.num_instructions,
                                      false);
                    if (else_sub == nullptr) {
                        return info.method = ExitMethod::AlwaysReturn;
                    }
                    else_method = else_sub->exit_method;
                }

                const ExitMethod both = ParallelExit(if_sub->exit_method, else_method);
                if (both == ExitMethod::AlwaysEnd) {
                    return info.method = ExitMethod::AlwaysEnd;
                }
                const ExitMethod after_call =
                    Scan(instr.flow_control.dest_offset + instr.flow_control.num_instructions, end,
                         end_is_free);
                MergeTail(info, instr.flow_control.dest_offset + instr.flow_control.num_instructions,
                          end);
                return info.method = SeriesExit(both, after_call);
            }
            default:
                break;
            }
        }
        return info.method = ExitMethod::AlwaysReturn;
    }
};

/// Pasa el selector de swizzle de nihstro a letras.
template <SwizzlePattern::Selector (SwizzlePattern::*getter)(int) const>
std::string GetSelector(const SwizzlePattern& pattern) {
    std::string out;
    for (int i = 0; i < 4; ++i) {
        switch ((pattern.*getter)(i)) {
        case SwizzlePattern::Selector::x:
            out += 'x';
            break;
        case SwizzlePattern::Selector::y:
            out += 'y';
            break;
        case SwizzlePattern::Selector::z:
            out += 'z';
            break;
        case SwizzlePattern::Selector::w:
            out += 'w';
            break;
        default:
            out += 'x';
            break;
        }
    }
    return out;
}

constexpr auto GetSelectorSrc1 = GetSelector<&SwizzlePattern::GetSelectorSrc1>;
constexpr auto GetSelectorSrc2 = GetSelector<&SwizzlePattern::GetSelectorSrc2>;
constexpr auto GetSelectorSrc3 = GetSelector<&SwizzlePattern::GetSelectorSrc3>;

using RegGetter = std::function<std::string(u32)>;

/**
 * La profundidad de bloques { } del codigo emitido y cual esta abierto en cada
 * nivel (0.3.1.0): una variable declarada en un bloque solo se ve mientras ese
 * MISMO bloque siga abierto, no en un hermano que se abra despues a la misma
 * profundidad. Ver RelCache.
 */
struct Scope {
    int depth = 0;
    std::vector<u32> open{0};
    u32 next = 1;

    Scope& operator++() {
        ++depth;
        open.push_back(next++);
        return *this;
    }
    Scope& operator--() {
        --depth;
        if (open.size() > 1) {
            open.pop_back();
        }
        return *this;
    }
    operator int() const {
        return depth;
    }
    [[nodiscard]] bool IsOpen(u32 id) const {
        return std::find(open.begin(), open.end(), id) != open.end();
    }
    [[nodiscard]] u32 Current() const {
        return open.back();
    }
};

/// Emite el cuerpo del shader con sangria, igual que el ShaderWriter de GLSL.
class Writer {
public:
    template <typename... Args>
    void AddLine(fmt::format_string<Args...> text, Args&&... args) {
        const std::string line = fmt::format(text, std::forward<Args>(args)...);
        if (!line.empty()) {
            source.append(static_cast<std::size_t>(static_cast<int>(scope)) * 4, ' ');
        }
        source += line;
        source += '\n';
    }

    void AddNewLine() {
        source += '\n';
    }

    std::string MoveResult() {
        return std::move(source);
    }

    Scope scope;

private:
    std::string source;
};

/**
 * Traduce el bytecode a Cg. Es el GLSLGenerator con la sintaxis cambiada.
 *
 * DONDE CG Y GLSL NO SE PARECEN, Y QUE SE HA HECHO EN CADA CASO:
 *
 *   - Los tipos: vec4/ivec3/bvec2 pasan a float4/int3/bool2.
 *   - Las globales MUTABLES. En GLSL una variable global normal se puede
 *     escribir; en Cg una global sin cualificar es un uniform, o sea de solo
 *     lectura. Los registros temporales, el codigo condicional y los registros
 *     de direccion se declaran 'static', que es lo que en Cg significa "global
 *     de verdad, no un parametro de fuera".
 *   - greaterThanEqual/lessThan no existen. Se usa step(), que da exactamente
 *     lo mismo: step(a, b) vale 1 cuando b >= a. SLT se escribe como 1 - step
 *     para que el caso de igualdad caiga del lado correcto.
 *   - inversesqrt se llama rsqrt.
 *   - any()/all() sobre bool2 se escriben con || y && sobre los dos
 *     componentes, que es mas simple y no depende de como trate Cg los
 *     vectores de booleanos.
 *   - Los uniforms booleanos de la PICA son una mascara de bits en GLSL. Aqui
 *     van como un array de floats que valen 0 o 1: el perfil de vertices de Cg
 *     puede no tener operaciones de bits, y comparar un float con cero las
 *     tiene todos los perfiles.
 */
class CgGenerator {
public:
    CgGenerator(const std::set<Subroutine>& subroutines_, const ProgramCode& program_code_,
                const SwizzleData& swizzle_data_, u32 main_offset_, const RegGetter& input_getter_,
                const RegGetter& output_getter_)
        : subroutines{subroutines_}, program_code{program_code_}, swizzle_data{swizzle_data_},
          main_offset{main_offset_}, input_getter{input_getter_}, output_getter{output_getter_} {
        Generate();
    }

    [[nodiscard]] bool Failed() const {
        return failed;
    }

    [[nodiscard]] const char* Reason() const {
        return reason;
    }

    std::string MoveShaderCode() {
        return shader.MoveResult();
    }

private:
    void Fail(const char* why) {
        if (!failed) {
            failed = true;
            reason = why;
        }
    }

    const Subroutine* GetSubroutine(u32 begin, u32 end) const {
        const auto iter = subroutines.find(Subroutine{begin, end});
        return iter != subroutines.end() ? &*iter : nullptr;
    }

    /// La condicion de un IF/CALL con codigo condicional.
    static std::string EvaluateCondition(Instruction::FlowControlType flow_control) {
        using Op = Instruction::FlowControlType::Op;

        const std::string result_x =
            flow_control.refx.Value() ? "conditional_code.x" : "!conditional_code.x";
        const std::string result_y =
            flow_control.refy.Value() ? "conditional_code.y" : "!conditional_code.y";

        switch (flow_control.op) {
        case Op::JustX:
            return result_x;
        case Op::JustY:
            return result_y;
        case Op::Or:
            return fmt::format("({} || {})", result_x, result_y);
        case Op::And:
            return fmt::format("({} && {})", result_x, result_y);
        default:
            // El campo tiene dos bits y los cuatro valores estan cubiertos; si
            // aun asi llega otra cosa, "no se cumple" es lo que menos dibuja.
            return "false";
        }
    }

    /**
     * LAS LECTURAS CON REGISTRO DE DIRECCION, UNA VEZ (0.3.1.0). La piel de los
     * modelos (matrices de huesos, "c[a0.x + 10]") leia cada fila con su
     * get_offset_register -- unas diez comparaciones y una lectura indexada --
     * tantas veces como instrucciones la usaban: en Pokemon Sol, seis por
     * hueso para tres filas distintas. Ahora la primera la guarda en una
     * variable y las siguientes la reutilizan mientras el registro de
     * direccion no cambie y el bloque donde se declaro siga abierto. Se olvida
     * todo al escribir el registro (MOVA), en cada bucle (el registro z cambia
     * en cada vuelta, y una vuelta vuelve a pasar por el principio) y en cada
     * subrutina (puede escribirlo). Menos trabajo por vertice en la GPU y menos
     * codigo para el compilador de la consola.
     */
    struct RelRead {
        u32 index;
        u32 component;
        u32 block;
        std::string name;
    };
    std::vector<RelRead> rel_reads;
    u32 rel_count = 0;

    void ForgetRelReads(int component = -1) {
        std::erase_if(rel_reads, [&](const RelRead& read) {
            return component < 0 || read.component == static_cast<u32>(component);
        });
    }

    std::string GetSourceRegister(const SourceRegister& source_reg,
                                  u32 address_register_index) {
        const u32 index = static_cast<u32>(source_reg.GetIndex());
        switch (source_reg.GetRegisterType()) {
        case RegisterType::Input:
            return input_getter(index);
        case RegisterType::Temporary:
            return fmt::format("reg_tmp{}", index);
        case RegisterType::FloatUniform:
            if (address_register_index != 0) {
                const u32 component = address_register_index - 1;
                for (const RelRead& read : rel_reads) {
                    if (read.index == index && read.component == component &&
                        shader.scope.IsOpen(read.block)) {
                        return read.name;
                    }
                }
                std::string name = fmt::format("rel{}", rel_count++);
                if (FloatAddress()) {
                    shader.AddLine("float4 {} = get_offset_register({}.0, address_registers.{});",
                                   name, index, "xyz"[component]);
                } else {
                    shader.AddLine("float4 {} = get_offset_register({}, address_registers.{});",
                                   name, index, "xyz"[component]);
                }
                rel_reads.push_back({index, component, shader.scope.Current(), name});
                return name;
            }
            return fmt::format("vs_f[{}]", index);
        default:
            return "float4(0.0, 0.0, 0.0, 0.0)";
        }
    }

    std::string GetDestRegister(const DestRegister& dest_reg) const {
        const u32 index = static_cast<u32>(dest_reg.GetIndex());
        switch (dest_reg.GetRegisterType()) {
        case RegisterType::Output:
            return output_getter(index);
        case RegisterType::Temporary:
            return fmt::format("reg_tmp{}", index);
        default:
            return "";
        }
    }

    /// Un uniform booleano de la PICA. Ver la nota de la cabecera de la clase.
    static std::string GetUniformBool(u32 index, bool invert_test = false) {
        return fmt::format("(vs_b[{}] {} 0.0)", index, invert_test ? "==" : "!=");
    }

    /// La variable de escape del tramo [begin, end) (0.1.5.8, ver JumpFrom).
    static std::string EscapeName(u32 begin, u32 end) {
        return fmt::format("esc_{}_{}", begin, end);
    }

    /**
     * Un salto hacia delante, real o virtual (0.1.5.8): el reflejo EXACTO de
     * ControlFlowAnalyzer::JumpFrom, que ya ha aceptado la forma. 'taken' es
     * la condicion de salto. Devuelve la instruccion donde sigue el tramo.
     *
     *   destino dentro del tramo:  if (!taken) { trozo(); }
     *   destino fuera (escape):    if (!taken) { trozo(); } else { esc_tramo = true; }
     *                              y el tramo termina ahi (el resto lo hace
     *                              quien lo llamo, con su salto virtual)
     *   trozo que escapa:          esc_trozo = false antes; y al volver, un
     *                              salto virtual con condicion esc_trozo
     */
    u32 EmitJump(u32 skip_begin, u32 target, const std::string& taken) {
        const bool escapes = target > range_end;
        const u32 land = escapes ? range_end : target;
        const std::string raise = escapes ? EscapeName(range_begin, range_end) : std::string{};
        if (skip_begin < land) {
            const Subroutine* skip = GetSubroutine(skip_begin, land);
            if (skip == nullptr) {
                Fail("vs salto");
                return land;
            }
            if (skip->escape != 0) {
                shader.AddLine("{} = false;", EscapeName(skip->begin, skip->end));
            }
            shader.AddLine("if (!({})) {{", taken);
            ++shader.scope;
            CallSubroutine(*skip);
            --shader.scope;
            if (escapes) {
                shader.AddLine("}} else {{");
                ++shader.scope;
                shader.AddLine("{} = true;", raise);
                --shader.scope;
            }
            shader.AddLine("}}");
            if (skip->escape != 0) {
                return EmitJump(land, skip->escape, EscapeName(skip->begin, skip->end));
            }
        } else if (escapes) {
            shader.AddLine("if ({}) {{", taken);
            ++shader.scope;
            shader.AddLine("{} = true;", raise);
            --shader.scope;
            shader.AddLine("}}");
        }
        return land;
    }

    /// Variantes para el compilador de la consola (0.1.7.3, ver g_cg_variant).
    static bool FloatAddress() {
        return (g_cg_variant.load(std::memory_order_relaxed) & kCgFloatAddress) != 0;
    }
    static bool Flat() {
        return (g_cg_variant.load(std::memory_order_relaxed) & kCgFlat) != 0;
    }

    void CallSubroutine(const Subroutine& subroutine) {
        ForgetRelReads();
        CallSubroutineBody(subroutine);
        ForgetRelReads();
    }

    void CallSubroutineBody(const Subroutine& subroutine) {
        if (Flat()) {
            /**
             * Subrutina EN LINEA (0.1.7.3). Todo esta dentro de exec_shader, asi
             * que un END del cuerpo ("return true;") sale del programa entero,
             * que es lo mismo que la subrutina devolviendo true y el que la
             * llama devolviendo true a su vez. Tope de tamano: ver
             * GenerateVertexShader.
             */
            if (++inline_depth > 32) {
                Fail("vs demasiado anidado");
                --inline_depth;
                return;
            }
            shader.AddLine("{{");
            ++shader.scope;
            CompileRange(subroutine.begin, subroutine.end);
            --shader.scope;
            shader.AddLine("}}");
            if (subroutine.exit_method == ExitMethod::AlwaysEnd) {
                shader.AddLine("return true;");
            }
            --inline_depth;
            return;
        }
        if (subroutine.exit_method == ExitMethod::AlwaysEnd) {
            shader.AddLine("{}();", subroutine.GetName());
            shader.AddLine("return true;");
        } else if (subroutine.exit_method == ExitMethod::Conditional) {
            shader.AddLine("if ({}()) {{ return true; }}", subroutine.GetName());
        } else {
            shader.AddLine("{}();", subroutine.GetName());
        }
    }

    /// Escribe una asignacion respetando la mascara de destino del swizzle.
    void SetDest(const SwizzlePattern& swizzle, std::string_view reg, std::string_view value,
                 u32 dest_num_components, u32 value_num_components) {
        u32 dest_mask_num_components = 0;
        std::string dest_mask_swizzle = ".";

        for (u32 i = 0; i < dest_num_components; ++i) {
            if (swizzle.DestComponentEnabled(static_cast<int>(i))) {
                dest_mask_swizzle += "xyzw"[i];
                ++dest_mask_num_components;
            }
        }

        if (reg.empty() || dest_mask_num_components == 0) {
            return;
        }

        const std::string dest =
            fmt::format("{}{}", reg, dest_num_components != 1 ? dest_mask_swizzle : "");

        std::string src{value};
        if (value_num_components == 1) {
            if (dest_mask_num_components != 1) {
                src = fmt::format("float{}({})", dest_mask_num_components, value);
            }
        } else if (value_num_components != dest_mask_num_components) {
            src = fmt::format("({}){}", value, dest_mask_swizzle);
        }

        shader.AddLine("{} = {};", dest, src);
    }

    /**
     * Traduce UNA instruccion y devuelve el desplazamiento de la siguiente.
     *
     * Devuelve PROGRAM_END cuando la instruccion termina el programa, y el
     * salto por encima del bloque cuando es un IF o un LOOP.
     */
    u32 CompileInstr(u32 offset) {
        const Instruction instr = {program_code[offset]};

        const std::size_t swizzle_offset =
            instr.opcode.Value().GetInfo().type == OpCode::Type::MultiplyAdd
                ? instr.mad.operand_desc_id
                : instr.common.operand_desc_id;
        const SwizzlePattern swizzle = {swizzle_data[swizzle_offset]};

        switch (instr.opcode.Value().GetInfo().type) {
        case OpCode::Type::Arithmetic: {
            const bool is_inverted =
                (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));

            std::string src1 = swizzle.negate_src1 ? "-" : "";
            src1 += GetSourceRegister(instr.common.GetSrc1(is_inverted),
                                      !is_inverted * instr.common.address_register_index);
            src1 += "." + GetSelectorSrc1(swizzle);

            std::string src2 = swizzle.negate_src2 ? "-" : "";
            src2 += GetSourceRegister(instr.common.GetSrc2(is_inverted),
                                      is_inverted * instr.common.address_register_index);
            src2 += "." + GetSelectorSrc2(swizzle);

            const std::string dest_reg = GetDestRegister(instr.common.dest.Value());

            switch (instr.opcode.Value().EffectiveOpCode()) {
            case OpCode::Id::ADD:
                SetDest(swizzle, dest_reg, fmt::format("{} + {}", src1, src2), 4, 4);
                break;
            case OpCode::Id::MUL:
                SetDest(swizzle, dest_reg, fmt::format("{} * {}", src1, src2), 4, 4);
                break;
            case OpCode::Id::FLR:
                SetDest(swizzle, dest_reg, fmt::format("floor({})", src1), 4, 4);
                break;
            case OpCode::Id::MAX:
                SetDest(swizzle, dest_reg, fmt::format("max({}, {})", src1, src2), 4, 4);
                break;
            case OpCode::Id::MIN:
                SetDest(swizzle, dest_reg, fmt::format("min({}, {})", src1, src2), 4, 4);
                break;
            case OpCode::Id::DP3:
                SetDest(swizzle, dest_reg, fmt::format("dot(({}).xyz, ({}).xyz)", src1, src2), 4,
                        1);
                break;
            case OpCode::Id::DP4:
                SetDest(swizzle, dest_reg, fmt::format("dot({}, {})", src1, src2), 4, 1);
                break;
            case OpCode::Id::DPH:
            case OpCode::Id::DPHI:
                // DPH pone un uno en la w del primer operando: es el producto
                // escalar de un punto homogeneo contra un plano.
                SetDest(swizzle, dest_reg,
                        fmt::format("dot(float4(({}).xyz, 1.0), {})", src1, src2), 4, 1);
                break;
            case OpCode::Id::RCP:
                // Sin multiplicacion exacta los NaN no se tratan; la guarda
                // evita el infinito, que es lo que rompia el agua en algunos
                // juegos. Es la misma solucion que toma el generador de GLSL.
                shader.AddLine("if (({}).x != 0.0)", src1);
                SetDest(swizzle, dest_reg, fmt::format("(1.0 / ({}).x)", src1), 4, 1);
                break;
            case OpCode::Id::RSQ:
                shader.AddLine("if (({}).x > 0.0)", src1);
                SetDest(swizzle, dest_reg, fmt::format("rsqrt(({}).x)", src1), 4, 1);
                break;
            case OpCode::Id::MOVA:
                // Los registros de direccion son enteros y el truncado va hacia
                // cero, igual que el ivec2() de GLSL y que el hardware.
                if (FloatAddress()) {
                    // Truncado hacia cero sin enteros (0.1.7.3, ver g_cg_variant).
                    SetDest(swizzle, "address_registers", fmt::format("ar_trunc(({}).xy)", src1),
                            2, 2);
                } else {
                    SetDest(swizzle, "address_registers", fmt::format("int2(({}).xy)", src1), 2,
                            2);
                }
                ForgetRelReads(0);
                ForgetRelReads(1);
                break;
            case OpCode::Id::MOV:
                SetDest(swizzle, dest_reg, src1, 4, 4);
                break;
            case OpCode::Id::SGE:
            case OpCode::Id::SGEI:
                // step(a, b) vale 1 donde b >= a, que es justo SGE.
                SetDest(swizzle, dest_reg, fmt::format("step({}, {})", src2, src1), 4, 4);
                break;
            case OpCode::Id::SLT:
            case OpCode::Id::SLTI:
                // Y SLT es su complemento EXACTO, incluida la igualdad: por eso
                // se escribe como uno menos el mismo step y no como step(b, a),
                // que se equivocaria cuando los dos valores son iguales.
                SetDest(swizzle, dest_reg,
                        fmt::format("(float4(1.0, 1.0, 1.0, 1.0) - step({}, {}))", src2, src1), 4,
                        4);
                break;
            case OpCode::Id::CMP: {
                using CompareOp = Instruction::Common::CompareOpType::Op;
                const auto op_string = [](CompareOp op) -> const char* {
                    switch (op) {
                    case CompareOp::Equal:
                        return "==";
                    case CompareOp::NotEqual:
                        return "!=";
                    case CompareOp::LessThan:
                        return "<";
                    case CompareOp::LessEqual:
                        return "<=";
                    case CompareOp::GreaterThan:
                        return ">";
                    case CompareOp::GreaterEqual:
                        return ">=";
                    default:
                        return nullptr;
                    }
                };
                const char* op_x = op_string(instr.common.compare_op.x.Value());
                const char* op_y = op_string(instr.common.compare_op.y.Value());
                if (op_x == nullptr || op_y == nullptr) {
                    Fail("vs comparacion");
                    break;
                }
                shader.AddLine("conditional_code.x = ({}).x {} ({}).x;", src1, op_x, src2);
                shader.AddLine("conditional_code.y = ({}).y {} ({}).y;", src1, op_y, src2);
                break;
            }
            case OpCode::Id::EX2:
                SetDest(swizzle, dest_reg, fmt::format("exp2(({}).x)", src1), 4, 1);
                break;
            case OpCode::Id::LG2:
                SetDest(swizzle, dest_reg, fmt::format("log2(({}).x)", src1), 4, 1);
                break;
            default:
                Fail("vs aritmetica");
                break;
            }
            break;
        }

        case OpCode::Type::MultiplyAdd: {
            const auto opcode = instr.opcode.Value().EffectiveOpCode();
            if (opcode != OpCode::Id::MAD && opcode != OpCode::Id::MADI) {
                Fail("vs mad");
                break;
            }
            const bool is_inverted = opcode == OpCode::Id::MADI;

            std::string src1 = swizzle.negate_src1 ? "-" : "";
            src1 += GetSourceRegister(instr.mad.GetSrc1(is_inverted), 0);
            src1 += "." + GetSelectorSrc1(swizzle);

            std::string src2 = swizzle.negate_src2 ? "-" : "";
            src2 += GetSourceRegister(instr.mad.GetSrc2(is_inverted),
                                      !is_inverted * instr.mad.address_register_index);
            src2 += "." + GetSelectorSrc2(swizzle);

            std::string src3 = swizzle.negate_src3 ? "-" : "";
            src3 += GetSourceRegister(instr.mad.GetSrc3(is_inverted),
                                      is_inverted * instr.mad.address_register_index);
            src3 += "." + GetSelectorSrc3(swizzle);

            const std::string dest_reg =
                (instr.mad.dest.Value() < 0x10)
                    ? output_getter(static_cast<u32>(instr.mad.dest.Value().GetIndex()))
                : (instr.mad.dest.Value() < 0x20)
                    ? fmt::format("reg_tmp{}", instr.mad.dest.Value().GetIndex())
                    : std::string{};

            SetDest(swizzle, dest_reg, fmt::format("{} * {} + {}", src1, src2, src3), 4, 4);
            break;
        }

        default: {
            switch (instr.opcode.Value()) {
            case OpCode::Id::END:
                shader.AddLine("return true;");
                offset = PROGRAM_END - 1;
                break;

            case OpCode::Id::CALL:
            case OpCode::Id::CALLC:
            case OpCode::Id::CALLU: {
                std::string condition;
                bool always = instr.opcode.Value() == OpCode::Id::CALL;
                if (instr.opcode.Value() == OpCode::Id::CALLC) {
                    condition = EvaluateCondition(instr.flow_control);
                } else if (instr.opcode.Value() == OpCode::Id::CALLU) {
                    // Booleano conocido (0.1.7.4): lo mismo que el analizador.
                    bool value = false;
                    if (KnownBool(instr.flow_control.bool_uniform_id, value)) {
                        if (!value) {
                            break;
                        }
                        always = true;
                    } else {
                        condition = GetUniformBool(instr.flow_control.bool_uniform_id);
                    }
                }

                if (condition.empty()) {
                    shader.AddLine("{{");
                } else {
                    shader.AddLine("if ({}) {{", condition);
                }
                ++shader.scope;

                const Subroutine* call_sub =
                    GetSubroutine(instr.flow_control.dest_offset,
                                  instr.flow_control.dest_offset +
                                      instr.flow_control.num_instructions);
                if (call_sub == nullptr) {
                    Fail("vs llamada");
                    --shader.scope;
                    shader.AddLine("}}");
                    break;
                }
                CallSubroutine(*call_sub);
                if (always && call_sub->exit_method == ExitMethod::AlwaysEnd) {
                    offset = PROGRAM_END - 1;
                }

                --shader.scope;
                shader.AddLine("}}");
                break;
            }

            case OpCode::Id::NOP:
                break;

            case OpCode::Id::JMPC:
            case OpCode::Id::JMPU: {
                /**
                 * Salto hacia delante como "if" (ver ControlFlowAnalyzer::Scan,
                 * que ya ha comprobado que el destino esta delante y dentro del
                 * tramo). Se salta cuando la condicion se cumple, asi que el
                 * trozo [salto+1, destino) se ejecuta cuando NO se cumple, y en
                 * los dos casos se sigue en el destino.
                 *
                 * JMPU: el campo num_instructions NO cuenta instrucciones, es el
                 * bit que invierte la prueba. El interprete salta cuando el
                 * uniform vale !(num_instructions & 1): con el bit a 0 salta si
                 * el uniform es verdadero, con el bit a 1 si es falso. Eso es
                 * exactamente GetUniformBool(id, invert_test = bit).
                 */
                const u32 target = instr.flow_control.dest_offset;
                // Booleano conocido (0.1.7.4): el reflejo de Scan. Si no salta
                // nunca, nada; si salta siempre, un salto con el trozo vacio.
                bool always_taken = false;
                const bool known = instr.opcode.Value() == OpCode::Id::JMPU &&
                                   KnownJmpu(instr, always_taken);
                if (known && !always_taken) {
                    break;
                }
                if (target == offset + 1) {
                    break;
                }
                if (known) {
                    offset = EmitJump(target, target, "true") - 1;
                    break;
                }
                const std::string taken =
                    instr.opcode.Value() == OpCode::Id::JMPC
                        ? EvaluateCondition(instr.flow_control)
                        : GetUniformBool(instr.flow_control.bool_uniform_id,
                                         (instr.flow_control.num_instructions & 1) != 0);
                // EmitJump devuelve donde sigue el tramo (0.1.5.8).
                offset = EmitJump(offset + 1, target, taken) - 1;
                break;
            }

            case OpCode::Id::IFC:
            case OpCode::Id::IFU: {
                bool known_value = false;
                if (instr.opcode.Value() == OpCode::Id::IFU &&
                    KnownBool(instr.flow_control.bool_uniform_id, known_value)) {
                    // Booleano conocido (0.1.7.4): solo el cuerpo que se ejecuta,
                    // sin if. El reflejo de Scan.
                    const u32 else_at = instr.flow_control.dest_offset;
                    const u32 endif_at = else_at + instr.flow_control.num_instructions;
                    const Subroutine* body = nullptr;
                    if (known_value) {
                        body = GetSubroutine(offset + 1, else_at);
                    } else if (instr.flow_control.num_instructions != 0) {
                        body = GetSubroutine(else_at, endif_at);
                    }
                    if ((known_value || instr.flow_control.num_instructions != 0) &&
                        body == nullptr) {
                        Fail("vs if");
                        break;
                    }
                    offset = endif_at - 1;
                    if (body != nullptr) {
                        shader.AddLine("{{");
                        ++shader.scope;
                        CallSubroutine(*body);
                        --shader.scope;
                        shader.AddLine("}}");
                        if (body->exit_method == ExitMethod::AlwaysEnd) {
                            offset = PROGRAM_END - 1;
                        }
                    }
                    break;
                }
                const std::string condition =
                    instr.opcode.Value() == OpCode::Id::IFC
                        ? EvaluateCondition(instr.flow_control)
                        : GetUniformBool(instr.flow_control.bool_uniform_id);

                const u32 if_offset = offset + 1;
                const u32 else_offset = instr.flow_control.dest_offset;
                const u32 endif_offset =
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions;

                const Subroutine* if_sub = GetSubroutine(if_offset, else_offset);
                if (if_sub == nullptr) {
                    Fail("vs if");
                    break;
                }

                shader.AddLine("if ({}) {{", condition);
                ++shader.scope;
                CallSubroutine(*if_sub);
                offset = else_offset - 1;

                if (instr.flow_control.num_instructions != 0) {
                    const Subroutine* else_sub = GetSubroutine(else_offset, endif_offset);
                    if (else_sub == nullptr) {
                        Fail("vs else");
                        --shader.scope;
                        shader.AddLine("}}");
                        break;
                    }
                    --shader.scope;
                    shader.AddLine("}} else {{");
                    ++shader.scope;
                    CallSubroutine(*else_sub);
                    offset = endif_offset - 1;

                    if (if_sub->exit_method == ExitMethod::AlwaysEnd &&
                        else_sub->exit_method == ExitMethod::AlwaysEnd) {
                        offset = PROGRAM_END - 1;
                    }
                }

                --shader.scope;
                shader.AddLine("}}");
                break;
            }

            case OpCode::Id::LOOP: {
                /**
                 * El bucle de la PICA: cuenta de vueltas, valor inicial del
                 * registro de direccion z y su incremento, los tres en un
                 * uniform entero. Se escribe como un for corriente; el numero
                 * de vueltas es dinamico (sale del uniform), que es control de
                 * flujo que la USSE si sabe hacer.
                 */
                const u32 id = instr.flow_control.int_uniform_id.Value();
                const std::string loop_var = fmt::format("loop{}", offset);
                ForgetRelReads();
                if (FloatAddress()) {
                    shader.AddLine("address_registers.z = vs_i[{}].y;", id);
                    shader.AddLine("for (int {0} = 0; {0} <= (int)vs_i[{1}].x; "
                                   "address_registers.z += vs_i[{1}].z, ++{0}) {{",
                                   loop_var, id);
                } else {
                    shader.AddLine("address_registers.z = (int)vs_i[{}].y;", id);
                    shader.AddLine("for (int {0} = 0; {0} <= (int)vs_i[{1}].x; "
                                   "address_registers.z += (int)vs_i[{1}].z, ++{0}) {{",
                                   loop_var, id);
                }
                ++shader.scope;

                const Subroutine* loop_sub =
                    GetSubroutine(offset + 1, instr.flow_control.dest_offset + 1);
                if (loop_sub == nullptr) {
                    Fail("vs bucle");
                    --shader.scope;
                    shader.AddLine("}}");
                    break;
                }
                CallSubroutine(*loop_sub);
                offset = instr.flow_control.dest_offset;

                --shader.scope;
                shader.AddLine("}}");
                ForgetRelReads();

                if (loop_sub->exit_method == ExitMethod::AlwaysEnd) {
                    offset = PROGRAM_END - 1;
                }
                break;
            }

            case OpCode::Id::EMIT:
            case OpCode::Id::SETEMIT:
                // Instrucciones de shader de GEOMETRIA. No deberian estar en un
                // programa de vertices, y traducirlas no tiene sentido.
                Fail("vs geometria");
                break;

            default:
                Fail("vs instruccion");
                break;
            }
            break;
        }
        }
        return offset + 1;
    }

    u32 CompileRange(u32 begin, u32 end) {
        // El tramo que se compila, para EmitJump (0.1.5.8). Las subrutinas se
        // compilan una detras de otra, no anidadas, pero se deja como estaba
        // al salir por si acaso.
        const u32 saved_begin = range_begin;
        const u32 saved_end = range_end;
        range_begin = begin;
        range_end = end;
        const u32 result = CompileRangeBody(begin, end);
        range_begin = saved_begin;
        range_end = saved_end;
        return result;
    }

    u32 CompileRangeBody(u32 begin, u32 end) {
        u32 program_counter = begin;
        while (program_counter < (begin > end ? PROGRAM_END : end)) {
            if (failed) {
                return PROGRAM_END;
            }
            program_counter = CompileInstr(program_counter);
        }
        /**
         * Una estructura (IF, LOOP, salto) que termina MAS ALLA del final del
         * tramo: el analizador ya las rechaza, pero si alguna se le escapara,
         * el codigo generado ejecutaria instrucciones de fuera del tramo. Se
         * rechaza tambien aqui; PROGRAM_END es "el programa ha terminado" y si
         * vale.
         */
        if (program_counter != PROGRAM_END && begin <= end && program_counter > end) {
            Fail("vs estructura");
            return PROGRAM_END;
        }
        return program_counter;
    }

    void Generate() {
        /**
         * Lectura de un uniform flotante con desplazamiento por registro de
         * direccion. Es la traduccion literal de la del generador de GLSL: el
         * desplazamiento fuera de [-128, 127] no cuenta, el indice se envuelve
         * a siete bits y todo lo que pase de 96 devuelve unos, que es lo que da
         * el hardware.
         */
        if (FloatAddress()) {
            /**
             * Lo mismo sin enteros ni operaciones de bits (0.1.7.3): el "& 0x7F"
             * sobre un indice en [-128, 222] es sumar o restar 128 una vez. Los
             * registros de direccion son float con valores enteros (ar_trunc),
             * asi que las cuentas son exactas.
             */
            shader.AddLine("float2 ar_trunc(float2 v) {{");
            ++shader.scope;
            shader.AddLine("return sign(v) * floor(abs(v));");
            --shader.scope;
            shader.AddLine("}}");
            shader.AddLine("float4 get_offset_register(float base_index, float offset) {{");
            ++shader.scope;
            shader.AddLine(
                "float fixed_offset = (offset >= -128.0 && offset <= 127.0) ? offset : 0.0;");
            shader.AddLine("float index = base_index + fixed_offset;");
            shader.AddLine("index = index < 0.0 ? index + 128.0 : "
                           "(index >= 128.0 ? index - 128.0 : index);");
            shader.AddLine("return index < 96.0 ? vs_f[(int)index] : float4(1.0, 1.0, 1.0, 1.0);");
            --shader.scope;
            shader.AddLine("}}");
        } else {
            shader.AddLine("float4 get_offset_register(int base_index, int offset) {{");
            ++shader.scope;
            shader.AddLine("int fixed_offset = (offset >= -128 && offset <= 127) ? offset : 0;");
            shader.AddLine("int index = (base_index + fixed_offset) & 0x7F;");
            shader.AddLine("return index < 96 ? vs_f[index] : float4(1.0, 1.0, 1.0, 1.0);");
            --shader.scope;
            shader.AddLine("}}");
        }
        shader.AddNewLine();

        // Variables de escape de los tramos saltables que salen hacia delante
        // (0.1.5.8, ver JumpFrom). 'static' por lo mismo que los registros:
        // en Cg una global sin cualificar seria un uniform de solo lectura.
        for (const auto& subroutine : subroutines) {
            if (subroutine.escape != 0) {
                shader.AddLine("static bool {} = false;",
                               EscapeName(subroutine.begin, subroutine.end));
            }
        }

        // Declaraciones de las subrutinas. Cg exige el prototipo antes del uso,
        // igual que C: una subrutina puede llamar a otra que se define despues.
        // En linea (kCgFlat) no hay funciones que declarar.
        if (!Flat()) {
            for (const auto& subroutine : subroutines) {
                shader.AddLine("bool {}();", subroutine.GetName());
            }
        }
        shader.AddNewLine();

        const Subroutine* main = GetSubroutine(main_offset, PROGRAM_END);
        if (main == nullptr) {
            Fail("vs main");
            return;
        }

        shader.AddLine("bool exec_shader() {{");
        ++shader.scope;
        CallSubroutine(*main);
        // Cg exige que toda rama devuelva algo aunque el analisis diga que no
        // se llega; sin esto, el compilador rechaza la funcion.
        shader.AddLine("return false;");
        --shader.scope;
        shader.AddLine("}}");
        shader.AddNewLine();
        if (Flat()) {
            return; // todo esta ya dentro de exec_shader
        }

        for (const auto& subroutine : subroutines) {
            shader.AddLine("bool {}() {{", subroutine.GetName());
            ++shader.scope;
            // Sin etiquetas: los saltos se rechazan en el analisis, asi que
            // aqui nunca hace falta la maquina de estados del de GLSL.
            if (CompileRange(subroutine.begin, subroutine.end) != PROGRAM_END) {
                shader.AddLine("return false;");
            } else {
                shader.AddLine("return false;");
            }
            --shader.scope;
            shader.AddLine("}}");
            shader.AddNewLine();
            if (failed) {
                return;
            }
        }
    }

    const std::set<Subroutine>& subroutines;
    const ProgramCode& program_code;
    const SwizzleData& swizzle_data;
    const u32 main_offset;
    const RegGetter& input_getter;
    const RegGetter& output_getter;

    Writer shader;
    bool failed = false;
    const char* reason = "vs";
    /// Tramo que se esta compilando (CompileRange), para EmitJump.
    u32 range_begin = 0;
    u32 range_end = PROGRAM_END;
    /// Profundidad de subrutinas en linea (kCgFlat).
    u32 inline_depth = 0;
};

} // Anonymous namespace

u32 UsedBoolUniforms(const Pica::ShaderSetup& setup) {
    // Todo el codigo, no solo el del programa: es una vez por programa y asi no
    // hay que seguir el flujo. Un bit de mas solo cuesta alguna variante de mas.
    const auto& code = setup.GetProgramCode();
    u32 mask = 0;
    for (u32 i = 0; i < MAX_PROGRAM_CODE_LENGTH; i++) {
        const Instruction instr = {code[i]};
        switch (instr.opcode.Value()) {
        case OpCode::Id::JMPU:
        case OpCode::Id::IFU:
        case OpCode::Id::CALLU:
            mask |= 1u << instr.flow_control.bool_uniform_id;
            break;
        default:
            break;
        }
    }
    return mask;
}

std::optional<std::string> GenerateVertexShader(const Pica::ShaderSetup& setup,
                                                const PicaVSConfig& config,
                                                const ExtraVSConfig& extra,
                                                const VSInputs& inputs, bool write_lighting,
                                                bool write_w, const char** out_reason) {
    const auto reject = [out_reason](const char* reason) -> std::optional<std::string> {
        if (out_reason != nullptr) {
            *out_reason = reason;
        }
        return std::nullopt;
    };

    if (extra.use_geometry_shader) {
        // El shader de geometria de la PICA sigue en la CPU siempre.
        return reject("vs con geometria");
    }
    if (extra.sanitize_mul) {
        // Ver la cabecera: la version exacta de la multiplicacion necesita
        // seleccionar por componentes con isnan y no hay traduccion a Cg que se
        // pueda dar por buena sin probarla en consola.
        return reject("vs mul exacta");
    }

    ControlFlowAnalyzer analyzer{setup.GetProgramCode(), config.state.main_offset};
    if (analyzer.Failed()) {
        return reject(analyzer.Reason());
    }
    const std::set<Subroutine> subroutines = analyzer.MoveSubroutines();
    /**
     * Los dos shaders que rompieron el compilador de la consola (cg_error.txt)
     * llevaban saltos de escape. Si la traduccion generica los necesita, no se
     * le da al compilador: el rasterizador pide la version con los booleanos
     * como constantes (0.1.7.4), donde el JMPU que escapa suele desaparecer.
     */
    if ((g_cg_const_bools.load(std::memory_order_relaxed) & kCgBoolsKnown) == 0) {
        for (const auto& subroutine : subroutines) {
            if (subroutine.escape != 0) {
                return reject("vs salto: especializar");
            }
        }
    }

    std::array<bool, 16> used_inputs{};
    const RegGetter input_getter = [&used_inputs](u32 reg) -> std::string {
        if (reg >= 16) {
            return "float4(0.0, 0.0, 0.0, 0.0)";
        }
        used_inputs[reg] = true;
        return fmt::format("vs_in_reg{}", reg);
    };

    const RegGetter output_getter = [&config](u32 reg) -> std::string {
        if (reg < 16 && config.state.output_map[reg] < config.state.num_outputs) {
            return fmt::format("vs_out_attr{}", config.state.output_map[reg]);
        }
        // Un registro de salida que el programa escribe pero que no esta
        // mapeado a ninguna salida: se tira, igual que hace el hardware.
        return "";
    };

    CgGenerator generator{subroutines,       setup.GetProgramCode(), setup.GetSwizzleData(),
                          config.state.main_offset, input_getter,    output_getter};
    if (generator.Failed()) {
        return reject(generator.Reason());
    }
    const std::string program_source = generator.MoveShaderCode();

    std::string out;
    out += "// Shader de vertices de la PICA generado por Azahar para GXM.\n";
    out += "// NO EDITAR A MANO: lo compila la consola con SceShaccCg.\n\n";

    /**
     * Los uniforms del shader de vertices de la PICA.
     *
     * Los 96 flotantes van tal cual. Los enteros y los booleanos NO van con su
     * tipo original: los enteros porque el bucle solo los usa para contar y
     * pasarlos por float no pierde nada (son de ocho bits), y los booleanos
     * porque el perfil de vertices de Cg puede no tener operaciones de bits y
     * la version de GLSL los lleva como mascara. Un array de floats que valen
     * cero o uno funciona en cualquier perfil.
     */
    out += "uniform float4 vs_f[96];\n";
    out += "uniform float4 vs_i[4];\n";
    out += "uniform float vs_b[16];\n";
    // Atributos por defecto, indexados por REGISTRO de entrada (ver VSInputs).
    out += "uniform float4 vs_default[16];\n\n";

    /**
     * Las globales del programa. 'static' porque en Cg una global sin cualificar
     * seria un uniform, o sea de solo lectura, y estas se escriben.
     */
    for (u32 i = 0; i < 16; i++) {
        out += fmt::format("static float4 reg_tmp{} = float4(0.0, 0.0, 0.0, 1.0);\n", i);
    }
    out += "static bool2 conditional_code = bool2(false, false);\n";
    // float3 con la variante sin enteros (0.1.7.3, ver g_cg_variant).
    const bool float_address =
        (g_cg_variant.load(std::memory_order_relaxed) & kCgFloatAddress) != 0;
    out += float_address ? "static float3 address_registers = float3(0.0, 0.0, 0.0);\n"
                         : "static int3 address_registers = int3(0, 0, 0);\n";
    for (u32 i = 0; i < 16; i++) {
        if (used_inputs[i]) {
            out += fmt::format("static float4 vs_in_reg{} = float4(0.0, 0.0, 0.0, 1.0);\n", i);
        }
    }
    for (u32 i = 0; i < config.state.num_outputs; i++) {
        out += fmt::format("static float4 vs_out_attr{} = float4(0.0, 0.0, 0.0, 1.0);\n", i);
    }
    out += "\n";
    out += program_source;
    out += "\n";

    /**
     * El reparto de SEMANTICAS.
     *
     * Los registros de salida de la PICA no significan nada por si mismos: una
     * tabla de registros dice que componente de que registro es la X de la
     * posicion, la U de la coordenada 0, etc. Lo que no este mapeado vale uno,
     * que es lo que devuelve el hardware.
     */
    const auto semantic_maps = config.state.gs_state.GetSemanticMaps();
    const auto semantic = [&config, &semantic_maps](VSOutputAttributes::Semantic slot_semantic) {
        const u32 slot = static_cast<u32>(slot_semantic);
        const u32 attrib = semantic_maps[slot].attribute_index;
        const u32 comp = semantic_maps[slot].component_index;
        if (attrib < config.state.gs_state.gs_output_attributes_count) {
            return fmt::format("vs_out_attr{}.{}", attrib, "xyzw"[comp]);
        }
        return std::string{"1.0"};
    };

    /**
     * La firma de salida tiene que ser LA MISMA, en tipos y en orden, que la
     * del shader de vertices fijo mas completo del rasterizador
     * (kVertexSourceLitProj): los shaders de fragmentos de esta ruta se enlazan
     * contra ESE programa (ver PipelineCache::HwvsFragmentProgram), y GXM
     * reparte los varyings segun las salidas del de vertices.
     *
     * Las entradas van SIN semantica: GXM las enlaza por nombre
     * (sceGxmProgramFindParameterByName("vs_in<registro>")), y asi no hay que
     * adivinar cuantas TEXCOORD admite el perfil de vertices.
     */
    out += "void main(";
    for (u32 i = 0; i < 16; i++) {
        if (used_inputs[i] && inputs[i].kind == VSInputSource::Array) {
            out += fmt::format("float4 vs_in{},\n          ", i);
        }
    }
    out += "out float4 gl_Position : POSITION,\n";
    out += "          out float4 out_color : COLOR0,\n";
    out += "          out float2 out_tc0 : TEXCOORD0,\n";
    out += "          out float2 out_tc1 : TEXCOORD1,\n";
    out += "          out float2 out_tc2 : TEXCOORD2";
    /**
     * LAS SALIDAS SON LAS DE LA VARIANTE FIJA CON LA QUE SE ENLAZO EL SHADER DE
     * FRAGMENTOS (0.1.4.8): kVertexSource, kVertexSourceLit, kVertexSourceProj
     * o kVertexSourceLitProj, segun write_lighting y write_w, en sus mismos
     * tipos y su mismo orden. Asi el programa de fragmentos que ya existe para
     * la ruta de la CPU sirve tal cual con este.
     *
     * En 0.1.4.6 salian siempre las ocho y el de fragmentos se enlazaba
     * aparte contra kVertexSourceLitProj: GXM lo rechazo con 0x805b0024 en
     * cuanto el de fragmentos leia MENOS varyings de los que escribe ese (en
     * NSMB2, todos los lotes sin luz), y la ruta no llegaba a usarse.
     */
    if (write_lighting) {
        out += ",\n          out float4 out_normquat : TEXCOORD3";
        out += ",\n          out float3 out_view : TEXCOORD4";
    }
    if (write_w) {
        out += ",\n          out float out_tc0w : TEXCOORD5";
    }
    out += ")\n{\n";

    /**
     * Los registros de entrada, como los deja el interprete (ver VSInputs):
     * del flujo con las componentes que falten a 0, 0, 0, 1 (VertexLoader lo
     * hace asi, y no se deja en manos de lo que GXM ponga en las que faltan);
     * del atributo por defecto; o ceros.
     */
    for (u32 i = 0; i < 16; i++) {
        if (!used_inputs[i]) {
            continue;
        }
        switch (inputs[i].kind) {
        case VSInputSource::Array: {
            out += fmt::format("    vs_in_reg{0} = vs_in{0};\n", i);
            for (u32 comp = inputs[i].components; comp < 4; comp++) {
                out += fmt::format("    vs_in_reg{}.{} = {};\n", i, "xyzw"[comp],
                                   comp == 3 ? "1.0" : "0.0");
            }
            break;
        }
        case VSInputSource::Default:
            out += fmt::format("    vs_in_reg{0} = vs_default[{0}];\n", i);
            break;
        default:
            out += fmt::format("    vs_in_reg{} = float4(0.0, 0.0, 0.0, 0.0);\n", i);
            break;
        }
    }

    // Las globales se dejan en su valor de partida a mano. Que Cg inicialice
    // una 'static' en cada invocacion es lo esperable, pero no hace falta
    // depender de ello para algo que cuesta unas pocas instrucciones.
    for (u32 i = 0; i < 16; i++) {
        out += fmt::format("    reg_tmp{} = float4(0.0, 0.0, 0.0, 1.0);\n", i);
    }
    out += "    conditional_code = bool2(false, false);\n";
    out += float_address ? "    address_registers = float3(0.0, 0.0, 0.0);\n"
                         : "    address_registers = int3(0, 0, 0);\n";
    for (u32 i = 0; i < config.state.num_outputs; i++) {
        out += fmt::format("    vs_out_attr{} = float4(0.0, 0.0, 0.0, 1.0);\n", i);
    }

    out += "\n    exec_shader();\n\n";

    /**
     * La posicion sale TAL CUAL, en coordenadas de recorte de la PICA.
     *
     * El generador de GLSL le da la vuelta a la z y se guarda un plano de
     * recorte propio, porque OpenGL tiene otro convenio de profundidad. Aqui no:
     * el camino de GXM de este port ya le entrega a la GPU las coordenadas de
     * recorte de la PICA sin tocar y deja que el viewport (que se configura con
     * los registros de rango de profundidad de la PICA) haga el resto. Ver
     * DrawBatchOnGpu en rasterizer_gxm.cpp. Cambiar aqui el convenio dejaria
     * los dos caminos dibujando distinto.
     */
    out += fmt::format("    gl_Position = float4({}, {}, {}, {});\n",
                       semantic(VSOutputAttributes::POSITION_X),
                       semantic(VSOutputAttributes::POSITION_Y),
                       semantic(VSOutputAttributes::POSITION_Z),
                       semantic(VSOutputAttributes::POSITION_W));

    // El color se recorta a uno en valor absoluto, igual que en el de GLSL: el
    // hardware deja que el shader saque valores fuera de rango y los dobla.
    out += fmt::format("    float4 vtx_color = float4({}, {}, {}, {});\n",
                       semantic(VSOutputAttributes::COLOR_R),
                       semantic(VSOutputAttributes::COLOR_G),
                       semantic(VSOutputAttributes::COLOR_B),
                       semantic(VSOutputAttributes::COLOR_A));
    out += "    out_color = min(abs(vtx_color), float4(1.0, 1.0, 1.0, 1.0));\n";

    out += fmt::format("    out_tc0 = float2({}, {});\n",
                       semantic(VSOutputAttributes::TEXCOORD0_U),
                       semantic(VSOutputAttributes::TEXCOORD0_V));
    out += fmt::format("    out_tc1 = float2({}, {});\n",
                       semantic(VSOutputAttributes::TEXCOORD1_U),
                       semantic(VSOutputAttributes::TEXCOORD1_V));
    out += fmt::format("    out_tc2 = float2({}, {});\n",
                       semantic(VSOutputAttributes::TEXCOORD2_U),
                       semantic(VSOutputAttributes::TEXCOORD2_V));

    if (write_lighting) {
        out += fmt::format("    out_normquat = float4({}, {}, {}, {});\n",
                           semantic(VSOutputAttributes::QUATERNION_X),
                           semantic(VSOutputAttributes::QUATERNION_Y),
                           semantic(VSOutputAttributes::QUATERNION_Z),
                           semantic(VSOutputAttributes::QUATERNION_W));
        out += fmt::format("    out_view = float3({}, {}, {});\n",
                           semantic(VSOutputAttributes::VIEW_X),
                           semantic(VSOutputAttributes::VIEW_Y),
                           semantic(VSOutputAttributes::VIEW_Z));
    }
    if (write_w) {
        // La W de la coordenada 0, para las texturas proyectadas (ver
        // kVertexSourceProj en rasterizer_gxm.cpp).
        out += fmt::format("    out_tc0w = {};\n", semantic(VSOutputAttributes::TEXCOORD0_W));
    }
    out += "}\n";

    // Con las subrutinas en linea (kCgFlat) un programa con muchas llamadas
    // puede crecer mucho; pasado este tope no merece la pena ni intentarlo.
    if (out.size() > 400000) {
        return reject("vs demasiado grande");
    }
    return out;
}

} // namespace Pica::Shader::Generator::GXM
