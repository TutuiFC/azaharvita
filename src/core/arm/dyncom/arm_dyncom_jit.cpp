// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// JIT del ARM11 para PS Vita. La explicacion de conjunto (que se compila, por
// que es exacto y como se comprueba) esta en arm_dyncom_jit.h.

#ifdef __PSVITA__

#include "core/arm/dyncom/arm_dyncom_jit.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include <psp2/kernel/sysmem.h>
#include "common/vita_diag.h"
#include "common/vita_vm.h"
#include "core/arm/dyncom/arm_dyncom_dec.h"
#include "core/arm/skyeye_common/armstate.h"
#include "core/arm/skyeye_common/vfp/vfp.h"
#include "core/memory.h"

namespace Core::ArmJit {

std::atomic<u32> mode{1};
std::atomic<u32> reg_cache{1};
std::atomic<u32> vfp_data{1};
std::atomic<u32> direct_link{1};
std::atomic<u32> vfp_native{1};

namespace {

// ---------------------------------------------------------------------------
// Parametros
// ---------------------------------------------------------------------------

/// Memoria de codigo. Cuando se llena se tira todo y se empieza de cero (lo
/// mismo que hace la cache de traduccion del interprete).
constexpr u32 kCodeBytes = 8u * 1024u * 1024u;
/// Un bloque se compila la SEGUNDA vez que se despacha: el codigo que solo
/// corre una vez (arranque, cargas) no merece lo que cuesta compilarlo.
constexpr u32 kCompileAfterVisits = 2;
/// Bloques mas largos no se compilan (el interprete no tiene limite, y la
/// extension tiene que coincidir con la suya).
constexpr u32 kMaxBlockInstructions = 256;
/// Comprobaciones: las primeras kFullChecks ejecuciones de cada bloque, y
/// despues una de cada kSampleEvery (potencia de dos). 0.1.9.5: de 4 a 1. Cada
/// pantalla nueva compila cientos de bloques y comprobarlos cuatro veces eran
/// 12-13 ms por fotograma ("comprobar" en crash.txt), con cero diferencias en
/// todas las partidas desde 0.1.8.x; el muestreo sigue cazando lo que dependa
/// de los datos.
constexpr u32 kFullChecks = 1;
/**
 * 0.1.7.8: de 512 a 4096. Una comprobacion cuesta del orden de 1.500 ciclos
 * (la variante, tres copias de ~400 bytes del estado, el bloque interpretado y
 * la comparacion) frente a ~30 de un bloque tipico de seis instrucciones por el
 * JIT: cada 512 eso era cerca de un 10 % del codigo generado, y cada vez corta
 * ademas la cadena de enlaces. Las primeras kFullChecks siguen siendo todas, y
 * es ahi donde han salido casi todas las DIFERENCIAS; lo que se pierde es
 * cazar antes un fallo que dependa de los datos. Se mide con "chk" y "rst".
 */
constexpr u32 kSampleEvery = 4096;
/// Escrituras que caben en el diario de una comprobacion.
constexpr u32 kJournalEntries = 96;

// Registros del anfitrion dentro del codigo generado.
enum HostReg : u32 {
    R0 = 0,
    R1 = 1,
    R2 = 2,
    R3 = 3,
    kCpu = 4,   ///< ARMul_State*
    kPages = 5, ///< tabla de punteros por pagina (PageTable::RawPointers)
    kRn = 6,    ///< operando base
    kRm = 7,    ///< segundo operando / desplazamiento
    kRs = 8,    ///< registro de desplazamiento / direccion calculada
    kRd = 9,    ///< resultado / dato
    kT0 = 10,   ///< temporal; tambien guarda los flags a traves de llamadas
    kT1 = 11,   ///< puntero de pagina
    kFlags = 12 ///< copia del APSR mientras se hace un acceso a memoria
};

// Condiciones ARM (mismos numeros que ConditionCode).
constexpr u32 kAlways = 14;
// Mas condiciones para los saltos del codigo generado (0.1.5.7).
constexpr u32 kCondEq = 0;
constexpr u32 kCondNe = 1;
constexpr u32 kCondLo = 3; ///< CC: sin acarreo (resta con prestamo, "menor sin signo")
/// log2(kSampleEvery): la prueba de muestreo en linea lo necesita.
constexpr u32 kSampleEveryBits = 12;
static_assert((1u << kSampleEveryBits) == kSampleEvery, "kSampleEvery tiene que ser 2^bits");
/**
 * Huecos de la cache de registros (0.1.5.7): r0-r3 y lr, CINCO (antes tres,
 * r0-r2). r3 solo lo usan el prologo (antes de cargar la cache) y los
 * argumentos de las llamadas (despues de volcarla); lr lo guarda el prologo
 * en la pila y solo lo pisan los BLX de las llamadas, tambien con la cache
 * volcada. En orden de preferencia.
 */
constexpr std::array<u32, 5> kCacheHosts{0, 1, 2, 3, 14};

// ---------------------------------------------------------------------------
// Emisor de instrucciones ARMv7 (A32)
// ---------------------------------------------------------------------------

class Emitter {
public:
    Emitter(u32* base_, u32 capacity_words_) : base{base_}, capacity{capacity_words_} {}

    bool Overflowed() const {
        return overflow;
    }
    u32 Position() const {
        return position;
    }
    u32* At(u32 index) const {
        return base + index;
    }

    void Emit(u32 word) {
        if (position >= capacity) {
            overflow = true;
            return;
        }
        base[position++] = word;
    }

    /// MOVW/MOVT: una constante de 32 bits sin tocar los flags.
    void Mov32(u32 rd, u32 value) {
        Emit(0xE3000000u | ((value & 0xF000u) << 4) | (rd << 12) | (value & 0x0FFFu));
        const u32 high = value >> 16;
        if (high != 0) {
            Emit(0xE3400000u | ((high & 0xF000u) << 4) | (rd << 12) | (high & 0x0FFFu));
        }
    }
    void MovReg(u32 rd, u32 rm) {
        Emit(0xE1A00000u | (rd << 12) | rm);
    }
    void LdrImm(u32 rt, u32 rn, u32 offset) {
        Emit(0xE5900000u | (rn << 16) | (rt << 12) | (offset & 0xFFFu));
    }
    void StrImm(u32 rt, u32 rn, u32 offset) {
        Emit(0xE5800000u | (rn << 16) | (rt << 12) | (offset & 0xFFFu));
    }
    /// LDR rt, [rn, rm, LSL #2]
    void LdrRegLsl2(u32 rt, u32 rn, u32 rm) {
        Emit(0xE7900100u | (rn << 16) | (rt << 12) | rm);
    }
    void LsrImm(u32 rd, u32 rm, u32 shift) {
        Emit(0xE1A00020u | (rd << 12) | (shift << 7) | rm);
    }
    void Ubfx(u32 rd, u32 rn, u32 lsb, u32 width) {
        Emit(0xE7E00050u | ((width - 1) << 16) | (rd << 12) | (lsb << 7) | rn);
    }
    void CmpImm0(u32 rn) {
        Emit(0xE3500000u | (rn << 16));
    }
    void Mrs(u32 rd) {
        Emit(0xE10F0000u | (rd << 12));
    }
    /// MSR APSR_nzcvq, rm
    void MsrFlags(u32 rm) {
        Emit(0xE128F000u | rm);
    }
    void Blx(u32 rm) {
        Emit(0xE12FFF30u | rm);
    }
    void Push(u32 mask) {
        Emit(0xE92D0000u | mask);
    }
    void Pop(u32 mask) {
        Emit(0xE8BD0000u | mask);
    }
    void AddReg(u32 rd, u32 rn, u32 rm) {
        Emit(0xE0800000u | (rn << 16) | (rd << 12) | rm);
    }
    void SubReg(u32 rd, u32 rn, u32 rm) {
        Emit(0xE0400000u | (rn << 16) | (rd << 12) | rm);
    }
    /// ORR rd, rn, rm, LSL #shift
    void OrrLsl(u32 rd, u32 rn, u32 rm, u32 shift) {
        Emit(0xE1800000u | (rn << 16) | (rd << 12) | (shift << 7) | rm);
    }
    void LslImm(u32 rd, u32 rm, u32 shift) {
        Emit(0xE1A00000u | (rd << 12) | (shift << 7) | rm);
    }

    /// Salto con condicion a una etiqueta que se resuelve despues.
    u32 BranchPlaceholder(u32 cond) {
        const u32 index = position;
        Emit((cond << 28) | 0x0A000000u);
        return index;
    }
    void PatchBranch(u32 index, u32 target_index) {
        if (overflow) {
            return;
        }
        const s32 offset = static_cast<s32>(target_index) - static_cast<s32>(index + 2);
        base[index] = (base[index] & 0xFF000000u) | (static_cast<u32>(offset) & 0x00FFFFFFu);
    }

    /// Llamada a una funcion de C por r12. Destroza r0-r3, r12 y lr (y los
    /// flags): quien llama ya ha guardado los flags en r10 antes.
    void Call(const void* function) {
        Mov32(12, static_cast<u32>(reinterpret_cast<uintptr_t>(function)));
        Blx(12);
    }

    /// Salto a una posicion ya emitida: la vuelta de un camino frio (0.2.1.0).
    void BranchTo(u32 cond, u32 target_index) {
        const s32 offset = static_cast<s32>(target_index) - static_cast<s32>(position + 2);
        Emit((cond << 28) | 0x0A000000u | (static_cast<u32>(offset) & 0x00FFFFFFu));
    }

    /// LDRD/STRD rt, rt+1, [rn, #offset], offset de 0 a 255.
    void LdrdImm(u32 rt, u32 rn, u32 offset) {
        Emit(0xE1C000D0u | (rn << 16) | (rt << 12) | ((offset & 0xF0u) << 4) | (offset & 0xFu));
    }
    void StrdImm(u32 rt, u32 rn, u32 offset) {
        Emit(0xE1C000F0u | (rn << 16) | (rt << 12) | ((offset & 0xF0u) << 4) | (offset & 0xFu));
    }

    /// El inmediato de una operacion de datos (8 bits rotados una cantidad
    /// par) que vale 'value', si lo hay.
    static bool EncodeImmediate(u32 value, u32* encoded) {
        for (u32 rotation = 0; rotation < 16; rotation++) {
            const u32 bits = std::rotl(value, static_cast<int>(rotation * 2));
            if (bits <= 0xFFu) {
                *encoded = (rotation << 8) | bits;
                return true;
            }
        }
        return false;
    }

private:
    u32* base;
    u32 capacity;
    u32 position = 0;
    bool overflow = false;
};

// ---------------------------------------------------------------------------
// Desplazamientos dentro de ARMul_State (se calculan una vez; ver Init)
// ---------------------------------------------------------------------------

struct Offsets {
    u32 reg = 0;
    u32 ext = 0; ///< ExtReg[0] (VFP singles/doubles), 0.1.5.2 4.4.
    u32 n = 0, z = 0, c = 0, v = 0, t = 0;
    u32 cpsr = 0;  ///< Cpsr, para la comprobacion de IRQ del enlazado (4.8).
    u32 nirq = 0;  ///< NirqSig, idem.
    u32 vfp = 0;   ///< VFP[0] (FPSCR y demas), 0.1.6.1.
    u32 link = 0;  ///< jit_link_budget (y jit_link_hops detras), 0.2.1.0.
    u32 cp15 = 0;  ///< CP15[0]: el TLS se lee en linea (0.2.1.0).
};
Offsets g_offsets;

u32 RegOffset(u32 guest_reg) {
    return g_offsets.reg + guest_reg * 4;
}

u32 ExtOffset(u32 vfp_reg) {
    return g_offsets.ext + vfp_reg * 4;
}

// ---------------------------------------------------------------------------
// Funciones de apoyo que llama el codigo generado
// ---------------------------------------------------------------------------

/**
 * ENLACE EN EL CODIGO GENERADO (0.1.5.7).
 *
 * Al terminar un bloque, su propio codigo intenta saltar al siguiente sin
 * volver a C++. Antes (0.1.5.2) hacia DOS llamadas a C++ por salto, cada una
 * con una busqueda en g_blocks (un unordered_map): mas caro que volver a
 * TryRun. Ahora todo se comprueba en linea (ver Compiler::EmitLink) con las
 * MISMAS condiciones que el bucle de TryRun, que son las del despacho del
 * interprete entre dos bloques:
 *   - sin interrupcion pendiente sin enmascarar (NirqSig y el bit I de Cpsr);
 *     desde 0.1.8.2 solo en el despacho y en TryRun, ver EmitLink
 *   - en modo ARM (TFlag = 0) y con el PC alineado
 *   - el destino compilado y enlazable (LinkInfo::entry != 0)
 *   - al destino no le toca comprobacion (DueForCheck con LinkInfo::runs)
 *   - el destino cabe entero en lo que queda de rodaja (budget)
 * Si algo falla, el bloque sale por su epilogo como siempre y TryRun (o el
 * despacho del interprete) sigue con el camino de toda la vida.
 *
 * budget: instrucciones que quedan de la rodaja DESPUES del bloque que se
 * esta ejecutando; lo pone TryRun y cada enlace le resta el bloque destino.
 * hops: enlaces hechos (estadistica). Lo que TryRun ve al volver:
 *   instrucciones ejecutadas de mas = budget inicial - budget final.
 *
 * Desde 0.2.1.0 los dos viven en ARMul_State (jit_link_budget y
 * jit_link_hops), a un desplazamiento pequeno de kCpu: el enlace los lee con
 * un LDRD desde el estado en vez de cargar antes la direccion de una global.
 */
/// Presupuesto retirado por StopLinksIfRescheduled: no son instrucciones
/// ejecutadas, y RunCompiled lo descuenta. Solo lo toca C++.
u32 g_link_withdrawn = 0;
/// Diagnostico (0.1.6.3): llamadas a las funciones lentas de memoria y a la
/// aritmetica VFP. Variables normales (un solo hilo emula); se publican con
/// el resto en FlushLocalStats.
u64 g_slow_calls = 0;
u64 g_vfp_calls = 0;

/**
 * TIEMPO DEL ARM POR PARTES (0.1.7.5). En 0.1.6.3 el bucle del ARM costaba
 * ~240 ns por instruccion del juego con el 90 % en el JIT: los contadores
 * decian cuantas veces pasaba cada cosa, pero no cuanto costaba. Esto reparte
 * 'arm' en codigo generado, caminos lentos de memoria, VFP por funcion,
 * comprobaciones y compilacion; lo que sobra es el interprete y el despacho.
 *
 * Lo frecuente (miles de veces por fotograma) se cronometra UNA DE CADA
 * kTimeSampleEvery y se multiplica: leer el reloj en cada llamada falsearia
 * justo lo que se quiere medir. El reloj es de microsegundos y estas llamadas
 * duran menos, pero el error de redondeo se compensa en promedio porque la
 * fase del reloj respecto a la llamada es aleatoria. Lo raro (comprobar,
 * compilar) se cronometra siempre.
 */
constexpr u32 kTimeSampleEvery = 16;
u64 g_jit_us = 0;
u64 g_jit_entries = 0;
u64 g_slow_us = 0;
u64 g_vfp_us = 0;
u64 g_check_us = 0;
u64 g_compile_us = 0;

bool TimeThisCall(u64 calls) {
    return (calls & (kTimeSampleEvery - 1)) == 0;
}

/// Suma a 'slot' el tiempo de su ambito.
struct ScopedMicros {
    explicit ScopedMicros(u64& slot_) : slot{slot_}, begin{Common::VitaMicros()} {}
    ~ScopedMicros() {
        slot += Common::VitaMicros() - begin;
    }
    u64& slot;
    u64 begin;
};

/// Como ScopedMicros, pero solo si 'timed', y multiplicado por el muestreo.
struct SampledMicros {
    SampledMicros(u64& slot_, bool timed_)
        : slot{slot_}, timed{timed_}, begin{timed_ ? Common::VitaMicros() : 0} {}
    ~SampledMicros() {
        if (timed) {
            slot += (Common::VitaMicros() - begin) * kTimeSampleEvery;
        }
    }
    u64& slot;
    bool timed;
    u64 begin;
};

/**
 * NumInstrsToExecute al empezar TryRun (0.1.5.7). El interprete mira el
 * presupuesto en CADA instruccion, asi que si algo lo cambia a mitad de
 * rodaja (PrepareReschedule lo pone a 0) para en seco. El JIT no puede parar
 * a mitad de un bloque, pero si puede no empezar el siguiente: los accesos
 * lentos (los unicos que llegan a codigo arbitrario del emulador, como
 * hardware) miran si cambio y, si es asi, dejan el presupuesto de enlace a
 * cero; el bucle de TryRun hace la misma pregunta.
 */
u64 g_run_target = 0;

void StopLinksIfRescheduled(ARMul_State* cpu) {
    if (cpu->NumInstrsToExecute != g_run_target) {
        g_link_withdrawn += cpu->jit_link_budget;
        cpu->jit_link_budget = 0;
    }
}

enum AccessKind : u32 {
    kWord = 0,
    kHalf = 1,
    kByte = 2,
    kSignedHalf = 3,
    kSignedByte = 4,
};

u32 SlowReadRaw(ARMul_State* cpu, u32 address, u32 kind) {
    switch (kind) {
    case kWord:
        return cpu->ReadMemory32(address);
    case kHalf:
        return cpu->ReadMemory16(address);
    case kByte:
        return cpu->ReadMemory8(address);
    case kSignedHalf:
        return static_cast<u32>(static_cast<s32>(static_cast<s16>(cpu->ReadMemory16(address))));
    default:
        return static_cast<u32>(static_cast<s32>(static_cast<s8>(cpu->ReadMemory8(address))));
    }
}

u32 SlowRead(ARMul_State* cpu, u32 address, u32 kind) {
    const SampledMicros timer{g_slow_us, TimeThisCall(++g_slow_calls)};
    const u32 value = SlowReadRaw(cpu, address, kind);
    StopLinksIfRescheduled(cpu);
    return value;
}

void SlowWrite(ARMul_State* cpu, u32 address, u32 value, u32 kind) {
    const SampledMicros timer{g_slow_us, TimeThisCall(++g_slow_calls)};
    switch (kind) {
    case kWord:
        cpu->WriteMemory32(address, value);
        break;
    case kHalf:
        cpu->WriteMemory16(address, static_cast<u16>(value));
        break;
    default:
        cpu->WriteMemory8(address, static_cast<u8>(value));
        break;
    }
    StopLinksIfRescheduled(cpu);
}

/**
 * LDREX y STREX (y sus variantes de byte y media palabra), 0.2.1.0: lo mismo
 * que LDREX_INST/STREX_INST del interprete, con el mismo monitor exclusivo del
 * estado. Estaban en todas las rutinas de bloqueo del sistema del 3DS y
 * mandaban sus bloques enteros al interprete ("rech ldrex" en crash.txt).
 */
u32 JitLoadExclusive(ARMul_State* cpu, u32 address, u32 kind) {
    const SampledMicros timer{g_slow_us, TimeThisCall(++g_slow_calls)};
    cpu->SetExclusiveMemoryAddress(address);
    u32 value;
    switch (kind) {
    case kByte:
        value = cpu->ReadMemory8(address);
        break;
    case kHalf:
        value = cpu->ReadMemory16(address);
        break;
    default:
        value = cpu->ReadMemory32(address);
        break;
    }
    StopLinksIfRescheduled(cpu);
    return value;
}

/// 0 si guarda, 1 si el monitor ya no es de esa direccion (como STREX).
u32 JitStoreExclusive(ARMul_State* cpu, u32 address, u32 value, u32 kind) {
    const SampledMicros timer{g_slow_us, TimeThisCall(++g_slow_calls)};
    if (!cpu->IsExclusiveMemoryAccess(address)) {
        return 1;
    }
    cpu->UnsetExclusiveMemoryAddress();
    switch (kind) {
    case kByte:
        cpu->WriteMemory8(address, static_cast<u8>(value));
        break;
    case kHalf:
        cpu->WriteMemory16(address, static_cast<u16>(value));
        break;
    default:
        cpu->WriteMemory32(address, value);
        break;
    }
    StopLinksIfRescheduled(cpu);
    return 0;
}

/**
 * LDM/STM sin bit S, con las mismas cuentas que el interprete (LdnStM(...) y
 * LDM_INST/STM_INST en arm_dyncom_interpreter.cpp): la direccion sale del base
 * ANTES de escribirlo, el base se escribe ANTES de los accesos (asi que en un
 * LDM que carga el propio base gana lo cargado), STM guarda el valor ORIGINAL
 * del base si esta en la lista, y el PC se guarda como PC+8. Un LDM que carga
 * el PC pasa a Thumb si el bit 0 viene puesto.
 */
template <typename Read, typename Write>
void BlockTransfer(ARMul_State* cpu, u32 inst, Read&& read, Write&& write) {
    const u32 rn = (inst >> 16) & 0xF;
    const u32 list = inst & 0xFFFF;
    const u32 count = static_cast<u32>(std::popcount(list));
    const u32 base = cpu->Reg[rn];
    const bool pre = ((inst >> 24) & 1) != 0;
    const bool up = ((inst >> 23) & 1) != 0;
    const bool writeback = ((inst >> 21) & 1) != 0;
    const bool load = ((inst >> 20) & 1) != 0;
    u32 address;
    u32 new_base;
    if (up) {
        address = pre ? base + 4 : base;
        new_base = base + count * 4;
    } else {
        address = pre ? base - count * 4 : base - count * 4 + 4;
        new_base = base - count * 4;
    }
    if (writeback) {
        cpu->Reg[rn] = new_base;
    }
    if (load) {
        for (u32 i = 0; i < 16; i++) {
            if (((list >> i) & 1) == 0) {
                continue;
            }
            u32 value = read(address);
            if (i == 15) {
                cpu->TFlag = value & 1;
                value &= 0xFFFFFFFEu;
            }
            cpu->Reg[i] = value;
            address += 4;
        }
    } else {
        for (u32 i = 0; i < 15; i++) {
            if (((list >> i) & 1) == 0) {
                continue;
            }
            write(address, i == rn ? base : cpu->Reg[i]);
            address += 4;
        }
        if ((list >> 15) & 1) {
            write(address, cpu->Reg[15] + 8);
        }
    }
}

void HelperBlockTransfer(ARMul_State* cpu, u32 inst) {
    const SampledMicros timer{g_slow_us, TimeThisCall(++g_slow_calls)};
    BlockTransfer(
        cpu, inst, [cpu](u32 address) { return cpu->ReadMemory32(address); },
        [cpu](u32 address, u32 value) { cpu->WriteMemory32(address, value); });
    StopLinksIfRescheduled(cpu);
}

// ---------------------------------------------------------------------------
// Comprobacion contra el interprete
// ---------------------------------------------------------------------------

struct JournalEntry {
    u32 address;
    u32 kind;
    u32 value;
};

/// Estado de la variante de comprobacion mientras corre (un solo hilo emula).
struct CheckRun {
    ARMul_State* cpu = nullptr;
    u8* const* pages = nullptr;
    bool aborted = false;
    u32 journal_size = 0;
    std::array<JournalEntry, kJournalEntries> journal{};
};
CheckRun g_check;

u32 KindSize(u32 kind) {
    return kind == kWord ? 4 : (kind == kHalf || kind == kSignedHalf) ? 2 : 1;
}

bool Overlaps(u32 a, u32 a_size, u32 b, u32 b_size) {
    return a < b + b_size && b < a + a_size;
}

/**
 * Lectura en la variante de comprobacion: SOLO de RAM (pagina con puntero),
 * que no tiene efectos secundarios. Cualquier otra cosa -- registros de
 * hardware, memoria especial -- abandona la comprobacion: leerla dos veces
 * (aqui y en el interprete) podria cambiar el resultado. Tampoco se puede leer
 * algo que el propio bloque ha escrito antes (esta en el diario, no en la
 * memoria): tambien se abandona, es raro.
 */
u32 CheckRead(u32 address, u32 kind) {
    CheckRun& run = g_check;
    if (run.aborted) {
        return 0;
    }
    const u32 size = KindSize(kind);
    for (u32 i = 0; i < run.journal_size; i++) {
        if (Overlaps(address, size, run.journal[i].address, KindSize(run.journal[i].kind))) {
            run.aborted = true;
            return 0;
        }
    }
    const u8* page = run.pages[address >> 12];
    if (page == nullptr) {
        run.aborted = true;
        return 0;
    }
    const u8* data = page + (address & 0xFFF);
    switch (kind) {
    case kWord: {
        u32 value;
        std::memcpy(&value, data, 4);
        return value;
    }
    case kHalf: {
        u16 value;
        std::memcpy(&value, data, 2);
        return value;
    }
    case kByte:
        return *data;
    case kSignedHalf: {
        s16 value;
        std::memcpy(&value, data, 2);
        return static_cast<u32>(static_cast<s32>(value));
    }
    default:
        return static_cast<u32>(static_cast<s32>(static_cast<s8>(*data)));
    }
}

void CheckWrite(u32 address, u32 value, u32 kind) {
    CheckRun& run = g_check;
    if (run.aborted) {
        return;
    }
    if (run.pages[address >> 12] == nullptr || run.journal_size == kJournalEntries) {
        run.aborted = true;
        return;
    }
    run.journal[run.journal_size++] = {address, kind, value};
}

void CheckBlockTransfer(ARMul_State* cpu, u32 inst) {
    BlockTransfer(
        cpu, inst, [](u32 address) { return CheckRead(address, kWord); },
        [](u32 address, u32 value) { CheckWrite(address, value, kWord); });
}

// ---------------------------------------------------------------------------
// VFP en el JIT, fase 2 (0.1.6.1)
// ---------------------------------------------------------------------------

/**
 * INDICES DEL DECODIFICADOR DEL INTERPRETE (arm_instruction[] en
 * arm_dyncom_dec.cpp). El JIT clasifica las VFP con el MISMO
 * DecodeARMInstruction que usa el interprete para traducir, asi que acepta
 * exactamente las palabras que el interprete trata con cada funcion. Los de
 * VFP3 (vmov(i), vmov(r), vcvt(bff)) el decodificador ni los mira: tampoco
 * llegan aqui.
 */
enum VfpDecodeIndex : int {
    kVfpVmla = 0,   ///< ... hasta kVfpVdiv: todas por vfp_*_cpdo
    kVfpVdiv = 8,
    kVfpVabs = 11,  ///< vabs, vneg, vsqrt, vcmp, vcmp2, vcvt(bds): cpdo
    kVfpVcvtBds = 16,
    kVfpVcvtBfi = 18, ///< cpdo
    kVfpVmrs = 22,
    kVfpVpush = 27,
    kVfpVstm = 28,
    kVfpVpop = 29,
    kVfpVldm = 31,
};

bool IsVfpCpdo(int index) {
    return (index >= kVfpVmla && index <= kVfpVdiv) ||
           (index >= kVfpVabs && index <= kVfpVcvtBds) || index == kVfpVcvtBfi;
}

/**
 * ARITMETICA VFP: LAS MISMAS FUNCIONES QUE EL INTERPRETE.
 *
 * Las 17 instrucciones de datos del VFP (VMLA ... VDIV, VABS, VNEG, VSQRT,
 * VCMP, VCVT) hacen en el interprete exactamente esto (vfpinstr.cpp): simple
 * o doble segun el bit 8, vfp_*_cpdo con el FPSCR del juego, y despues
 * vfp_raise_exceptions. Aqui se llama a lo mismo, en el mismo orden, asi que
 * el resultado es el del interprete bit a bit, redondeos, NaN y flags del
 * FPSCR incluidos. No es mas rapido POR INSTRUCCION; lo que gana es que el
 * bloque que la contiene ya no se queda entero en el interprete (hasta
 * 0.1.6.0 era el primer motivo de rechazo, "rech vfp").
 *
 * Solo tocan ExtReg y VFP[]: nada de memoria, asi que la variante de
 * comprobacion usa la misma funcion (StartCheck guarda y repone los dos).
 */
void HelperVfpCdp(ARMul_State* cpu, u32 inst) {
    const SampledMicros timer{g_vfp_us, TimeThisCall(++g_vfp_calls)};
    const u32 fpscr = cpu->VFP[VFP_FPSCR];
    const u32 ret = ((inst >> 8) & 1) != 0 ? vfp_double_cpdo(cpu, inst, fpscr)
                                           : vfp_single_cpdo(cpu, inst, fpscr);
    vfp_raise_exceptions(cpu, ret, inst, cpu->VFP[VFP_FPSCR]);
}

/**
 * VPUSH, VPOP, VSTM y VLDM, copiados de VPUSH_INST/VPOP_INST/VSTM_INST/
 * VLDM_INST: mismos indices de registro (d + i para singles, (d + i) * 2 y
 * + 1 para doubles), mismo orden de accesos, y la escritura del base DESPUES
 * de los accesos. Little-endian siempre: Analyze no compila en big-endian.
 * El base con el PC (solo VSTM/VLDM) es Reg[15] + 8, y Reg[15] lo deja
 * escrito el codigo generado antes de llamar.
 */
template <typename Read, typename Write>
void VfpTransfer(ARMul_State* cpu, u32 inst, int index, Read&& read, Write&& write) {
    const bool single = ((inst >> 8) & 1) == 0;
    const u32 d = single ? (((inst >> 12) & 0xF) << 1) | ((inst >> 22) & 1)
                         : ((inst >> 12) & 0xF) | (((inst >> 22) & 1) << 4);
    const u32 imm32 = (inst & 0xFF) << 2;
    const u32 regs = single ? (inst & 0xFF) : ((inst >> 1) & 0x7F);
    const u32 n = (inst >> 16) & 0xF;
    const bool add = ((inst >> 23) & 1) != 0;
    const bool wback = ((inst >> 21) & 1) != 0;
    const bool load = index == kVfpVpop || index == kVfpVldm;

    u32 address;
    if (index == kVfpVpush) {
        address = cpu->Reg[13] - imm32;
    } else if (index == kVfpVpop) {
        address = cpu->Reg[13];
    } else {
        address = cpu->Reg[n];
        if (n == 15) {
            address += 8;
        }
        if (!add) {
            address -= imm32;
        }
    }
    for (u32 i = 0; i < regs; i++) {
        if (single) {
            if (load) {
                cpu->ExtReg[d + i] = read(address);
            } else {
                write(address, cpu->ExtReg[d + i]);
            }
            address += 4;
        } else {
            if (load) {
                const u32 word1 = read(address + 0);
                const u32 word2 = read(address + 4);
                cpu->ExtReg[(d + i) * 2 + 0] = word1;
                cpu->ExtReg[(d + i) * 2 + 1] = word2;
            } else {
                const u32 word1 = cpu->ExtReg[(d + i) * 2 + 0];
                const u32 word2 = cpu->ExtReg[(d + i) * 2 + 1];
                write(address + 0, word1);
                write(address + 4, word2);
            }
            address += 8;
        }
    }
    if (index == kVfpVpush) {
        cpu->Reg[13] -= imm32;
    } else if (index == kVfpVpop) {
        cpu->Reg[13] += imm32;
    } else if (wback) {
        cpu->Reg[n] = add ? cpu->Reg[n] + imm32 : cpu->Reg[n] - imm32;
    }
}

/**
 * 'index' es el del decodificador del interprete, calculado AL COMPILAR el
 * bloque (0.1.8.5). Antes se llamaba a DecodeARMInstruction aqui, en cada
 * VPUSH/VPOP/VLDM/VSTM ejecutado: una busqueda lineal por toda la tabla de
 * instrucciones del ARM11, miles de ciclos. crash.txt de 0.1.8.4: "lento" 17 ms
 * por vblank con ~5.600 llamadas, unos 3 us cada una.
 */
void HelperVfpTransfer(ARMul_State* cpu, u32 inst, int index) {
    const SampledMicros timer{g_slow_us, TimeThisCall(++g_slow_calls)};
    VfpTransfer(
        cpu, inst, index, [cpu](u32 address) { return cpu->ReadMemory32(address); },
        [cpu](u32 address, u32 value) { cpu->WriteMemory32(address, value); });
    StopLinksIfRescheduled(cpu);
}

void CheckVfpTransfer(ARMul_State* cpu, u32 inst, int index) {
    VfpTransfer(
        cpu, inst, index, [](u32 address) { return CheckRead(address, kWord); },
        [](u32 address, u32 value) { CheckWrite(address, value, kWord); });
}

// Firmas de las funciones que llama el codigo generado. Los argumentos van en
// r0-r3 segun AAPCS.
u32 CheckReadThunk(u32 address, u32 kind) {
    return CheckRead(address, kind);
}
void CheckWriteThunk(u32 address, u32 value, u32 kind) {
    CheckWrite(address, value, kind);
}

// ---------------------------------------------------------------------------
// Bloques
// ---------------------------------------------------------------------------

using BlockFn = void (*)(ARMul_State* cpu, u8* const* pages);

enum class BlockState : u8 {
    New,
    Compiled,
    Rejected,
    Blacklisted,
};

/**
 * Lo que lee el CODIGO GENERADO de un bloque destino al enlazar (0.1.5.7).
 *
 * Estructura simple y aparte, con desplazamientos fijos (kLink*), porque
 * Block tiene un std::vector y no tiene una disposicion garantizada. El
 * codigo generado guarda punteros a esta estructura (en un literal, o los
 * saca de la tabla rapida) y lee de ella:
 *   runs  = ejecuciones, como Block::runs de antes (DueForCheck la mira)
 *   count = instrucciones del bloque (presupuesto de la rodaja)
 *   entry = direccion absoluta del punto de entrada de un enlace, o 0 si el
 *           bloque no se puede enlazar (sin compilar, rechazado, en la lista
 *           negra). Es lo UNICO que hay que tocar para desenlazar un bloque.
 * Los punteros a elementos de un unordered_map no cambian al crecer, y los
 * bloques solo se borran todos a la vez (ResetAll), junto con todo el codigo.
 */
struct LinkInfo {
    u32 runs = 0;
    u32 count = 0;
    u32 entry = 0;
};
constexpr u32 kLinkRuns = 0;
constexpr u32 kLinkCount = 4;
constexpr u32 kLinkEntry = 8;
static_assert(offsetof(LinkInfo, runs) == kLinkRuns && offsetof(LinkInfo, count) == kLinkCount &&
                  offsetof(LinkInfo, entry) == kLinkEntry,
              "el codigo generado lee LinkInfo con estos desplazamientos");

struct Block {
    LinkInfo link;
    u32 pc = 0;
    u32 visits = 0;
    BlockState state = BlockState::New;
    u32 reject = kRejectNone; ///< por que se rechazo (RejectReason)
    u32 reject_word = 0;      ///< la instruccion que no se supo compilar (0.1.9.6)
    BlockFn code = nullptr;
    BlockFn check_code = nullptr;
    /// Donde entra un enlace (despues del prologo). Se publica en link.entry
    /// cuando el bloque ha pasado sus kFullChecks comprobaciones (0.1.8.1).
    u32 chain_entry = 0;
    std::vector<u32> words; ///< Las instrucciones, para crash.txt si difiere.
    /**
     * Cache de registros (0.1.5.2, 4.3): guest -> host (r0-r3 o lr desde
     * 0.1.5.7, ver kCacheHosts) o -1.
     * Analyze lo rellena (a -1 si el interruptor esta apagado); EmitBlock se
     * lo pasa al Compiler. Al apagar el interruptor, Reset() tira los bloques
     * y los siguientes Analyze dejan todo a -1: camino exacto de 0.1.5.1.
     */
    std::array<s8, 16> cache_map{
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    bool uses_cache = false;
};

SceUID g_code_block = -1;
u32* g_code = nullptr;
u32 g_code_used_words = 0;
/// Bloques generados (las dos variantes) desde el ultimo vaciado.
u32 g_code_blocks = 0;
bool g_ready = false;
bool g_init_tried = false;

std::unordered_map<u32, Block> g_blocks;

/// Atajo delante del mapa, como block_cache del interprete.
constexpr u32 kFastSlots = 4096;
constexpr u32 kFastSlotBits = 12;
static_assert((1u << kFastSlotBits) == kFastSlots, "el enlace indirecto indexa con kFastSlotBits");
struct FastSlot {
    u32 pc = 0xFFFFFFFFu;
    Block* block = nullptr;
};
std::array<FastSlot, kFastSlots> g_fast{};
static_assert(sizeof(FastSlot) == 8 && offsetof(FastSlot, pc) == 0 && offsetof(FastSlot, block) == 4,
              "el enlace indirecto lee g_fast como pares de palabras (pc, block)");
/// Desplazamiento de Block::link dentro de Block (0 en la practica; se mide en
/// Init porque Block no tiene disposicion estandar).
u32 g_link_in_block = 0;

/**
 * Contadores. Los suma el hilo de emulacion en variables normales (una
 * operacion atomica por bloque costaria mas que lo medido) y los publica al
 * final de cada rodaja en los atomicos, que es lo que lee el overlay con
 * TakeStats. Las instrucciones totales llegan tambien al final de la rodaja
 * (AbandonPendingCheck): asi las dos cuentas del porcentaje salen del mismo
 * sitio y en el mismo momento.
 */
struct LocalStats {
    u64 instructions = 0;
    u64 slices = 0;
    u64 links = 0;
    u64 arm_us = 0;
    u64 svc_us = 0;
    u64 jit_instructions = 0;
    u64 dispatches = 0;
    u64 jit_dispatches = 0;
    std::array<u64, kRejectCount> rejects{};
};
LocalStats g_local;

/// Se pidio vaciar (memoria de codigo llena): se hace en el siguiente TryRun,
/// que es un punto donde nadie tiene un Block* en la mano.
bool g_flush_requested = false;


struct PublishedStats {
    std::atomic<u64> instructions{0};
    std::atomic<u64> slices{0};
    std::atomic<u64> links{0};
    std::atomic<u64> slow_calls{0};
    std::atomic<u64> vfp_calls{0};
    std::atomic<u64> arm_us{0};
    std::atomic<u64> svc_us{0};
    std::atomic<u64> jit_instructions{0};
    std::atomic<u64> dispatches{0};
    std::atomic<u64> jit_dispatches{0};
    std::array<std::atomic<u64>, kRejectCount> rejects{};
    std::atomic<u64> jit_us{0};
    std::atomic<u64> slow_us{0};
    std::atomic<u64> vfp_us{0};
    std::atomic<u64> check_us{0};
    std::atomic<u64> compile_us{0};
};
PublishedStats g_published;

void FlushLocalStats() {
    g_published.slices.fetch_add(g_local.slices, std::memory_order_relaxed);
    g_published.links.fetch_add(g_local.links, std::memory_order_relaxed);
    g_published.slow_calls.fetch_add(g_slow_calls, std::memory_order_relaxed);
    g_published.vfp_calls.fetch_add(g_vfp_calls, std::memory_order_relaxed);
    g_slow_calls = 0;
    g_vfp_calls = 0;
    g_published.jit_us.fetch_add(g_jit_us, std::memory_order_relaxed);
    g_published.slow_us.fetch_add(g_slow_us, std::memory_order_relaxed);
    g_published.vfp_us.fetch_add(g_vfp_us, std::memory_order_relaxed);
    g_published.check_us.fetch_add(g_check_us, std::memory_order_relaxed);
    g_published.compile_us.fetch_add(g_compile_us, std::memory_order_relaxed);
    g_jit_us = 0;
    g_slow_us = 0;
    g_vfp_us = 0;
    g_check_us = 0;
    g_compile_us = 0;
    g_published.instructions.fetch_add(g_local.instructions, std::memory_order_relaxed);
    g_published.arm_us.fetch_add(g_local.arm_us, std::memory_order_relaxed);
    g_published.svc_us.fetch_add(g_local.svc_us, std::memory_order_relaxed);
    g_published.jit_instructions.fetch_add(g_local.jit_instructions, std::memory_order_relaxed);
    g_published.dispatches.fetch_add(g_local.dispatches, std::memory_order_relaxed);
    g_published.jit_dispatches.fetch_add(g_local.jit_dispatches, std::memory_order_relaxed);
    for (u32 i = 0; i < kRejectCount; i++) {
        g_published.rejects[i].fetch_add(g_local.rejects[i], std::memory_order_relaxed);
    }
    g_local = LocalStats{};
}

bool Init(ARMul_State* cpu) {
    g_init_tried = true;
    const auto base = reinterpret_cast<const u8*>(cpu);
    g_offsets.reg = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->Reg[0]) - base);
    g_offsets.ext = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->ExtReg[0]) - base);
    g_offsets.n = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->NFlag) - base);
    g_offsets.z = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->ZFlag) - base);
    g_offsets.c = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->CFlag) - base);
    g_offsets.v = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->VFlag) - base);
    g_offsets.t = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->TFlag) - base);
    g_offsets.cpsr = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->Cpsr) - base);
    g_offsets.nirq = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->NirqSig) - base);
    g_offsets.vfp = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->VFP[0]) - base);
    g_offsets.link = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->jit_link_budget) - base);
    g_offsets.cp15 = static_cast<u32>(reinterpret_cast<const u8*>(&cpu->CP15[0]) - base);
    // El enlace lee budget y hops con un LDRD: alineados a 8, seguidos y a
    // menos de 256 bytes del principio del estado.
    if (g_offsets.link > 248 ||
        (reinterpret_cast<uintptr_t>(&cpu->jit_link_budget) & 7u) != 0 ||
        reinterpret_cast<const u8*>(&cpu->jit_link_hops) !=
            reinterpret_cast<const u8*>(&cpu->jit_link_budget) + 4) {
        Common::VitaNote("jit arm", fmt::format("contexto de enlace en {}: JIT apagado",
                                                g_offsets.link)
                                        .c_str());
        return false;
    }
    {
        // Donde cae Block::link dentro de Block (ver g_link_in_block).
        const Block probe;
        g_link_in_block = static_cast<u32>(reinterpret_cast<const u8*>(&probe.link) -
                                           reinterpret_cast<const u8*>(&probe));
    }
    // Todos tienen que caber en el inmediato de 12 bits de LDR/STR.
    const u32 highest = std::max({g_offsets.reg + 60, g_offsets.ext + 252, g_offsets.n,
                                  g_offsets.z, g_offsets.c, g_offsets.v, g_offsets.t,
                                  g_offsets.cpsr, g_offsets.nirq,
                                  g_offsets.vfp + VFP_SYSTEM_REGISTER_COUNT * 4 - 4,
                                  g_offsets.cp15 + CP15_REGISTER_COUNT * 4 - 4});
    if (highest > 4092) {
        Common::VitaNote("jit arm", fmt::format("desplazamientos de ARMul_State fuera de rango "
                                                "({}): JIT apagado",
                                                highest)
                                        .c_str());
        return false;
    }
    g_code_block = sceKernelAllocMemBlockForVM("azahar_arm_jit", kCodeBytes);
    if (g_code_block < 0) {
        Common::VitaNote("jit arm", fmt::format("sin memoria ejecutable ({:#x}): JIT apagado",
                                                static_cast<u32>(g_code_block))
                                        .c_str());
        return false;
    }
    void* base_address = nullptr;
    if (sceKernelGetMemBlockBase(g_code_block, &base_address) < 0 || base_address == nullptr) {
        Common::VitaNote("jit arm", "bloque de codigo sin direccion: JIT apagado");
        return false;
    }
    g_code = static_cast<u32*>(base_address);
    g_code_used_words = 0;
    g_ready = true;
    Common::VitaNote("jit arm", "listo (8 MB de codigo)");
    return true;
}

// ---------------------------------------------------------------------------
// Decodificacion: que instrucciones sabe compilar
// ---------------------------------------------------------------------------

enum class Kind {
    Unsupported,
    DataProcessing,
    Multiply,
    MultiplyLong,
    LoadStore,
    LoadStoreExtra,
    LoadStoreDual, ///< LDRD/STRD (0.1.4.9)
    BlockTransfer,
    Branch,
    BranchExchange,
    Nop,        ///< NOP/YIELD (0.1.4.9)
    Clz,        ///< (0.1.4.9)
    NativeReg,  ///< SXT*/UXT*/REV*: se emiten tal cual con los registros cambiados
    Cp15Read,   ///< MRC p15 (0.1.4.9)
    Cp15Write,  ///< MCR p15 (0.1.4.9)
    VfpLoadStore, ///< VLDR/VSTR (0.1.5.2, 4.4 fase 1)
    VfpMove,      ///< VMOV ARM<->VFP single (0.1.5.2, 4.4 fase 1)
    VfpCdp,       ///< VFP aritmetica por vfp_*_cpdo (0.1.6.1)
    VfpMrs,       ///< VMRS de FPSCR (0.1.6.1)
    VfpTransfer,  ///< VPUSH, VPOP, VSTM, VLDM (0.1.6.1)
    Clrex,        ///< CLREX (0.1.9.8)
    BranchLinkExchangeImm, ///< BLX inmediato: llamada a Thumb (0.1.9.8)
    Exclusive,             ///< LDREX/STREX de palabra, byte y media palabra (0.2.1.0)
};

struct Decoded {
    Kind kind = Kind::Unsupported;
    bool terminator = false;
    u32 reject = kRejectOther; ///< por que no, si kind == Unsupported
};

u32 Bits(u32 value, u32 low, u32 high) {
    return (value >> low) & ((1u << (high - low + 1)) - 1);
}

Decoded Reject(u32 reason) {
    Decoded out;
    out.reject = reason;
    return out;
}

Decoded Supported(Kind kind, bool terminator = false) {
    Decoded out;
    out.kind = kind;
    out.terminator = terminator;
    out.reject = kRejectNone;
    return out;
}

/**
 * Operaciones de datos que el interprete trata como SALTO cuando escriben el
 * PC (las que marcan INDIRECT_BRANCH en arm_dyncom_trans.cpp). AND, por
 * ejemplo, no lo marca: con Rd = PC su bloque no terminaria ahi, y la
 * extension dejaria de coincidir. Esas se quedan en el interprete.
 */
bool DataProcessingBranchesOnPc(u32 opcode) {
    switch (opcode) {
    case 1:  // EOR
    case 2:  // SUB
    case 3:  // RSB
    case 4:  // ADD
    case 5:  // ADC
    case 6:  // SBC
    case 7:  // RSC
    case 12: // ORR
    case 13: // MOV
    case 14: // BIC
    case 15: // MVN
        return true;
    default:
        return false;
    }
}

Decoded Classify(u32 inst) {
    const u32 cond = inst >> 28;
    /**
     * PLD (0.1.9.6): una pista de precarga, y el interprete no hace nada con
     * ella (PLD_INST solo avanza el PC). Esta en los bucles de memcpy y memset,
     * y rechazarla mandaba esos bucles enteros al interprete. Mismo patron que
     * la entrada "pld" de arm_dyncom_dec.cpp: inmediato y por registro.
     */
    if ((inst & 0xFD70F000u) == 0xF550F000u) {
        return Supported(Kind::Nop);
    }
    /**
     * CLREX y BLX inmediato (0.1.9.8): los dos primeros de "jit otro" en
     * crash.txt de 0.1.9.7 en los cinco juegos probados. CLREX esta en las
     * rutinas de bloqueo del sistema (miles de despachos por intervalo) y BLX
     * es la llamada de codigo ARM a una funcion Thumb; los dos mandaban el
     * bloque entero al interprete. Mismas cuentas que CLREX_INST y BLX_INST.
     */
    if (inst == 0xF57FF01Fu) {
        return Supported(Kind::Clrex);
    }
    if ((inst & 0xFE000000u) == 0xFA000000u) {
        return Supported(Kind::BranchLinkExchangeImm, true);
    }
    if (cond == 0xF) {
        return Reject(kRejectOther); // espacio incondicional (BLX inmediato, PLD...)
    }
    const u32 op1 = Bits(inst, 25, 27);

    // BX / BLX por registro: cond 0001 0010 1111 1111 1111 00L1 Rm
    if ((inst & 0x0FFFFFD0u) == 0x012FFF10u) {
        if (Bits(inst, 0, 3) == 15 && Bits(inst, 5, 5) == 1) {
            return Reject(kRejectPcWrite); // BLX pc: impredecible
        }
        return Supported(Kind::BranchExchange, true);
    }
    // NOP y YIELD: el interprete solo avanza el PC.
    if ((inst & 0x0FFFFFFEu) == 0x0320F000u) {
        return Supported(Kind::Nop);
    }
    // CLZ: cond 0001 0110 1111 Rd 1111 0001 Rm
    if ((inst & 0x0FFF0FF0u) == 0x016F0F10u) {
        if (Bits(inst, 12, 15) == 15 || Bits(inst, 0, 3) == 15) {
            return Reject(kRejectPcWrite);
        }
        return Supported(Kind::Clz);
    }

    if (op1 == 0 || op1 == 1) {
        const bool immediate = op1 == 1;
        if (!immediate && Bits(inst, 4, 4) == 1 && Bits(inst, 7, 7) == 1) {
            // Multiplicaciones y cargas/guardados "extra".
            const u32 sh = Bits(inst, 5, 6);
            if (sh == 0) {
                if (Bits(inst, 23, 27) == 0x00) {
                    // MUL / MLA
                    const u32 rd = Bits(inst, 16, 19);
                    const u32 rn = Bits(inst, 12, 15);
                    const u32 rs = Bits(inst, 8, 11);
                    const u32 rm = Bits(inst, 0, 3);
                    if (Bits(inst, 22, 22) != 0 || rd == 15 || rs == 15 || rm == 15 ||
                        (Bits(inst, 21, 21) == 1 && rn == 15)) {
                        return Reject(kRejectOther);
                    }
                    return Supported(Kind::Multiply);
                }
                if (Bits(inst, 23, 27) == 0x01) {
                    // UMULL / UMLAL / SMULL / SMLAL
                    const u32 hi = Bits(inst, 16, 19);
                    const u32 lo = Bits(inst, 12, 15);
                    if (hi == 15 || lo == 15 || hi == lo || Bits(inst, 8, 11) == 15 ||
                        Bits(inst, 0, 3) == 15) {
                        return Reject(kRejectOther);
                    }
                    return Supported(Kind::MultiplyLong);
                }
                /**
                 * LDREX/STREX, B y H (0.2.1.0). Los de 64 bits (D) y SWP se
                 * quedan en el interprete. Fuera tambien el PC en cualquier
                 * campo y, en STREX, el registro de estado igual a la direccion
                 * o al dato (la arquitectura no dice que pasa).
                 */
                const u32 op = Bits(inst, 20, 27);
                if (op == 0x18 || op == 0x19 || op == 0x1C || op == 0x1D || op == 0x1E ||
                    op == 0x1F) {
                    const u32 rn = Bits(inst, 16, 19);
                    const u32 rd = Bits(inst, 12, 15);
                    const u32 rt = Bits(inst, 0, 3);
                    const bool load = (op & 1) != 0;
                    if (rn == 15 || rd == 15 || (!load && (rt == 15 || rd == rn || rd == rt))) {
                        return Reject(kRejectExclusive);
                    }
                    return Supported(Kind::Exclusive);
                }
                return Reject(kRejectExclusive);
            }
            const bool load = Bits(inst, 20, 20) == 1;
            const bool pre = Bits(inst, 24, 24) == 1;
            const bool wb = Bits(inst, 21, 21) == 1;
            const u32 rn = Bits(inst, 16, 19);
            const u32 rd = Bits(inst, 12, 15);
            const bool imm_offset = Bits(inst, 22, 22) == 1;
            if (!pre && wb) {
                return Reject(kRejectOther);
            }
            if (!imm_offset && Bits(inst, 0, 3) == 15) {
                return Reject(kRejectOther);
            }
            if (!load && sh != 1) {
                /**
                 * LDRD (sh = 2) y STRD (sh = 3). Rt par y distinto de 14 (si
                 * no, la arquitectura no dice que pasa); con escritura del base,
                 * el base no puede ser ninguno de los dos; con desplazamiento
                 * por registro, en LDRD tampoco el registro de desplazamiento.
                 */
                if ((rd & 1) != 0 || rd == 14) {
                    return Reject(kRejectOther);
                }
                if ((!pre || wb) && (rn == 15 || rn == rd || rn == rd + 1)) {
                    return Reject(kRejectOther);
                }
                if (sh == 2 && !imm_offset &&
                    (Bits(inst, 0, 3) == rd || Bits(inst, 0, 3) == rd + 1)) {
                    return Reject(kRejectOther);
                }
                return Supported(Kind::LoadStoreDual);
            }
            // LDRH/STRH/LDRSB/LDRSH
            if (rd == 15 || ((!pre || wb) && (rn == 15 || rn == rd))) {
                return Reject(kRejectPcWrite);
            }
            return Supported(Kind::LoadStoreExtra);
        }
        const u32 opcode = Bits(inst, 21, 24);
        const bool s = Bits(inst, 20, 20) == 1;
        if (opcode >= 8 && opcode <= 11 && !s) {
            return Reject(kRejectMisc); // MRS, MSR, CPS y compania
        }
        const u32 rd = Bits(inst, 12, 15);
        const bool writes_rd = !(opcode >= 8 && opcode <= 11);
        if (!immediate && Bits(inst, 4, 4) == 1) {
            // Desplazamiento por registro: con el PC en medio es impredecible.
            if (Bits(inst, 16, 19) == 15 || Bits(inst, 0, 3) == 15 || Bits(inst, 8, 11) == 15 ||
                (writes_rd && rd == 15)) {
                return Reject(kRejectPcWrite);
            }
        }
        if (writes_rd && rd == 15) {
            /**
             * Escribe el PC (0.1.4.9): "mov pc, lr", "add pc, pc, rX, lsl #2"
             * (tablas de saltos)... Fin de bloque, como en el interprete, que
             * deja el resultado en Reg[15] sin mas. Con el bit S cambia el modo
             * del procesador: eso no, y tampoco las operaciones que el
             * interprete no marca como salto.
             */
            if (s || !DataProcessingBranchesOnPc(opcode)) {
                return Reject(kRejectPcWrite);
            }
            return Supported(Kind::DataProcessing, true);
        }
        return Supported(Kind::DataProcessing);
    }

    if (op1 == 2 || op1 == 3) {
        if (op1 == 3 && Bits(inst, 4, 4) == 1) {
            /**
             * Instrucciones "media". Las que se emiten tal cual (0.1.4.9):
             *   SXTAB/SXTAH/UXTAB/UXTAH y sus formas sin acumular (Rn = 1111):
             *     cond 0110 1x1x Rn Rd rot 00 0111 Rm
             *   REV, REV16, REVSH:
             *     cond 0110 1011 1111 Rd 1111 0011/1011 Rm y 0110 1111 ... 1011
             * Mismo significado que en el interprete (UXTB_INST, REV_INST...).
             */
            const u32 op = Bits(inst, 20, 27);
            const u32 low = Bits(inst, 4, 7);
            const u32 rd = Bits(inst, 12, 15);
            const u32 rm = Bits(inst, 0, 3);
            if (rd == 15 || rm == 15) {
                return Reject(kRejectMedia);
            }
            const bool extend = (op == 0x6A || op == 0x6B || op == 0x6E || op == 0x6F) &&
                                low == 0x7 && Bits(inst, 8, 9) == 0;
            const bool rev = Bits(inst, 16, 19) == 0xF && Bits(inst, 8, 11) == 0xF &&
                             ((op == 0x6B && (low == 0x3 || low == 0xB)) ||
                              (op == 0x6F && low == 0xB));
            if (extend || rev) {
                return Supported(Kind::NativeReg);
            }
            return Reject(kRejectMedia);
        }
        const bool pre = Bits(inst, 24, 24) == 1;
        const bool byte = Bits(inst, 22, 22) == 1;
        const bool wb = Bits(inst, 21, 21) == 1;
        const bool load = Bits(inst, 20, 20) == 1;
        const u32 rn = Bits(inst, 16, 19);
        const u32 rd = Bits(inst, 12, 15);
        if (!pre && wb) {
            return Reject(kRejectOther); // LDRT/STRT
        }
        if ((!pre || wb) && (rn == 15 || rn == rd)) {
            return Reject(kRejectOther);
        }
        if (op1 == 3 && Bits(inst, 0, 3) == 15) {
            return Reject(kRejectOther);
        }
        if (load && rd == 15) {
            // "ldr pc, [...]" (0.1.4.9): fin de bloque. Solo la de palabra; el
            // interprete pasa a Thumb si el bit 0 viene puesto.
            if (byte) {
                return Reject(kRejectPcWrite);
            }
            return Supported(Kind::LoadStore, true);
        }
        return Supported(Kind::LoadStore);
    }

    if (op1 == 4) {
        // LDM/STM
        if (Bits(inst, 22, 22) == 1) {
            return Reject(kRejectMisc);
        }
        if (Bits(inst, 16, 19) == 15 || Bits(inst, 0, 15) == 0) {
            return Reject(kRejectOther);
        }
        return Supported(Kind::BlockTransfer,
                         Bits(inst, 20, 20) == 1 && Bits(inst, 15, 15) == 1);
    }

    if (op1 == 5) {
        return Supported(Kind::Branch, true);
    }

    if (op1 == 7) {
        if (Bits(inst, 24, 24) == 1) {
            return Reject(kRejectSvc);
        }
        const u32 coprocessor = Bits(inst, 8, 11);
        if (coprocessor == 10 || coprocessor == 11) {
            // VFP. Fase 1 (0.1.5.2): solo MOVIMIENTO de datos, bit a bit.
            // VMOV core<->VFP single: cond 1110 000o Vn Rt 1010 N001 0000
            //   bits 27-24 = 1110, bits 23-21 = 000 (no es CDP: VADD tiene
            //   bits 21-20 = 11), bits 6-4 = 001, bit 7 = N libre, bit 20 = o.
            // VMRS/VMSR tienen bits 23-21 = 111 y por eso NO entran.
            // La aritmetica sigue rechazada: depende del FPSCR del juego.
            // 0.1.5.5: solo coprocesador 10 (bit 8 = 0). Con el 11 es
            // "VMOV Dd[x], Rt", que indexa ExtReg de otra forma (d*2 + x) y
            // aqui se trataba como single: registro equivocado.
            if (vfp_data.load(std::memory_order_relaxed) != 0 && coprocessor == 10 &&
                Bits(inst, 21, 23) == 0 && Bits(inst, 4, 6) == 1 && Bits(inst, 0, 3) == 0) {
                if (Bits(inst, 12, 15) == 15) {
                    return Reject(kRejectPcWrite);
                }
                return Supported(Kind::VfpMove);
            }
            // Fase 2 (0.1.6.1): aritmetica VFP y VMRS de FPSCR, clasificadas
            // con el mismo decodificador que el interprete (ver
            // VfpDecodeIndex). Solo con el interruptor de VFP encendido.
            if (vfp_data.load(std::memory_order_relaxed) != 0) {
                int index = -1;
                if (DecodeARMInstruction(inst, &index) == ARMDecodeStatus::SUCCESS) {
                    if (IsVfpCpdo(index)) {
                        return Supported(Kind::VfpCdp);
                    }
                    // VMRS: solo FPSCR (reg 1). Los demas registros de sistema
                    // dependen del modo privilegiado: al interprete.
                    if (index == kVfpVmrs && Bits(inst, 16, 19) == 1) {
                        return Supported(Kind::VfpMrs);
                    }
                }
            }
            return Reject(kRejectVfp);
        }
        // MRC/MCR del coprocesador 15 (0.1.4.9): cond 1110 opc1 L CRn Rt 1111 opc2 1 CRm
        if (Bits(inst, 4, 4) == 1 && coprocessor == 15 && Bits(inst, 12, 15) != 15) {
            return Supported(Bits(inst, 20, 20) == 1 ? Kind::Cp15Read : Kind::Cp15Write);
        }
        return Reject(kRejectCoproc);
    }

    if (op1 == 6) {
        const u32 coprocessor = Bits(inst, 8, 11);
        if (coprocessor == 10 || coprocessor == 11) {
            // VLDR/VSTR (0.1.5.2, 4.4 fase 1) si el interruptor esta encendido.
            // bits 27-24 = 1101 para single (L=bit20: 1=VLDR, 0=VSTR), bit 8
            // = 0 cuando es single (1010 en bits 11-8), 1 cuando es double.
            // bits 4-7 no son MCRR/MRRC (0100/0101) en VLDR/VSTR.
            /**
             * ARREGLO 0.1.5.5. VLDR/VSTR son "1101 U D 0 L": P = 1 y W = 0.
             * Antes solo se miraba P (bit 24), y VPUSH / VSTMDB sp! y VLDMDB
             * con escritura del base (P = 1, W = 1, por ejemplo ed2d8b02 =
             * vpush {d8}) se compilaban como un VSTR: sin bajar la pila y
             * guardando solo un registro. crash.txt de 0.1.5.4: DIFERENCIA
             * en r13 (8 bytes) en tres bloques y despues error fatal del
             * juego. Con W = 1 es un VLDM/VSTM: al interprete.
             */
            if (vfp_data.load(std::memory_order_relaxed) != 0 &&
                Bits(inst, 24, 24) == 1 && Bits(inst, 21, 21) == 0) {
                return Supported(Kind::VfpLoadStore);
            }
            // Fase 2 (0.1.6.1): VPUSH, VPOP, VSTM y VLDM por una funcion que
            // copia las del interprete (VfpTransfer). Solo si los registros que
            // toca caben en ExtReg: si no, el interprete (que indexaria fuera).
            if (vfp_data.load(std::memory_order_relaxed) != 0) {
                int index = -1;
                if (DecodeARMInstruction(inst, &index) == ARMDecodeStatus::SUCCESS &&
                    (index == kVfpVpush || index == kVfpVstm || index == kVfpVpop ||
                     index == kVfpVldm)) {
                    const bool single = Bits(inst, 8, 8) == 0;
                    const u32 d = single ? ((Bits(inst, 12, 15) << 1) | Bits(inst, 22, 22))
                                         : (Bits(inst, 12, 15) | (Bits(inst, 22, 22) << 4));
                    const u32 regs = single ? Bits(inst, 0, 7) : Bits(inst, 1, 7);
                    const bool fits = single ? d + regs <= 64 : d + regs <= 32;
                    // Escritura del base con el PC: no se traduce.
                    const bool pc_wback = (index == kVfpVstm || index == kVfpVldm) &&
                                          Bits(inst, 21, 21) == 1 && Bits(inst, 16, 19) == 15;
                    if (fits && !pc_wback) {
                        return Supported(Kind::VfpTransfer);
                    }
                }
            }
            return Reject(kRejectVfp); // VLDM/VSTM y lo demas
        }
        return Reject(kRejectCoproc);
    }

    return Reject(kRejectOther);
}

// ---------------------------------------------------------------------------
// Generacion
// ---------------------------------------------------------------------------

// Coprocesador 15, con los mismos campos que usa el interprete (CRn, opc1, CRm,
// opc2).
u32 JitCp15Read(ARMul_State* cpu, u32 inst) {
    return cpu->ReadCP15Register(Bits(inst, 16, 19), Bits(inst, 21, 23), Bits(inst, 0, 3),
                                 Bits(inst, 5, 7));
}
void JitCp15Write(ARMul_State* cpu, u32 inst, u32 value) {
    cpu->WriteCP15Register(value, Bits(inst, 16, 19), Bits(inst, 21, 23), Bits(inst, 0, 3),
                           Bits(inst, 5, 7));
}
void CheckAbortThunk() {
    g_check.aborted = true;
}
void JitClrex(ARMul_State* cpu) {
    cpu->UnsetExclusiveMemoryAddress();
}

class Compiler {
public:
    Compiler(Emitter& emitter_, bool check_mode_)
        : e{emitter_}, check_mode{check_mode_} {
        cache_map.fill(-1);
    }

    /// Cache de registros del bloque (0.1.5.2, 4.3). Ver SetCacheMap.
    void SetCacheMap(const std::array<s8, 16>& map) {
        cache_map = map;
        cache_dirty = false;
        for (s8 host : cache_map) {
            if (host >= 0) {
                cache_dirty = true;
                break;
            }
        }
    }

    void Prologue() {
        // r4-r12 y lr: diez registros, 40 bytes, la pila sigue alineada a 8.
        e.Push(0x5FF0u);
        e.MovReg(kCpu, R0);
        e.MovReg(kPages, R1);
        // Flags del juego -> APSR.
        e.LdrImm(R0, kCpu, g_offsets.n);
        e.LdrImm(R1, kCpu, g_offsets.z);
        e.LdrImm(R2, kCpu, g_offsets.c);
        e.LdrImm(R3, kCpu, g_offsets.v);
        e.LslImm(R0, R0, 31);
        e.OrrLsl(R0, R0, R1, 30);
        e.OrrLsl(R0, R0, R2, 29);
        e.OrrLsl(R0, R0, R3, 28);
        e.MsrFlags(R0);
        // Hasta aqui el prologo de 0.1.5.1 completo. El enlazado directo (4.8)
        // salta a ChainEntryPosition(): pila, kCpu, kPages y flags ya puestos
        // por el bloque origen; solo falta BodyEnter (la cache de este bloque).
    }

    /// Donde empieza el cuerpo tras el prologo: el destino de un enlace (4.8).
    [[nodiscard]] u32 ChainEntryPosition() const {
        return e.Position();
    }

    /// Primeras instrucciones del cuerpo: cache de registros (4.3).
    void BodyEnter() {
        LoadCachedRegs();
    }

    void Epilogue() {
        for (const u32 branch : to_epilogue) {
            e.PatchBranch(branch, e.Position());
        }
        // Vuelca la cache ANTES de usar r0-r1 para los flags.
        FlushCachedRegs();
        // Aqui entran los enlaces que fallan (0.2.1.0): ya volcaron la cache, y
        // r0-r3 tienen lo que dejo el enlace, asi que no se puede volcar otra vez.
        exit_noflush = e.Position();
        // APSR -> flags del juego.
        e.Mrs(R0);
        e.LsrImm(R1, R0, 31);
        e.StrImm(R1, kCpu, g_offsets.n);
        e.Ubfx(R1, R0, 30, 1);
        e.StrImm(R1, kCpu, g_offsets.z);
        e.Ubfx(R1, R0, 29, 1);
        e.StrImm(R1, kCpu, g_offsets.c);
        e.Ubfx(R1, R0, 28, 1);
        e.StrImm(R1, kCpu, g_offsets.v);
        e.Pop(0x9FF0u); // r4-r12 y pc
    }

    /// Los caminos frios, detras del epilogo (ver ColdStub).
    void EmitColdStubs() {
        for (std::size_t i = 0; i < cold_stubs.size(); i++) {
            ColdStub stub = std::move(cold_stubs[i]);
            for (const u32 branch : stub.branches) {
                e.PatchBranch(branch, e.Position());
            }
            cache_written = stub.written;
            stub.body();
        }
        cold_stubs.clear();
    }

    /// Una instruccion. 'pc' es su direccion en el juego.
    void Instruction(u32 inst, u32 pc, const Decoded& decoded) {
        const u32 cond = inst >> 28;
        if (decoded.terminator) {
            // BLX inmediato no tiene condicion: su campo vale 0xF.
            Terminator(inst, pc, decoded,
                       decoded.kind == Kind::BranchLinkExchangeImm ? kAlways : cond);
            return;
        }
        // Sin cuerpo no hay nada que saltar (y PLD lleva el campo de
        // condicion a 0xF, que no es una condicion).
        if (decoded.kind == Kind::Nop) {
            return;
        }
        // CLREX tambien lleva 0xF ahi: siempre se ejecuta.
        if (decoded.kind == Kind::Clrex) {
            FlushCachedRegs();
            e.Mrs(kFlags);
            e.MovReg(kT0, kFlags);
            e.MovReg(R0, kCpu);
            e.Call(reinterpret_cast<const void*>(&JitClrex));
            e.MsrFlags(kT0);
            ReloadCachedRegs();
            return;
        }
        // Condicion: se salta el cuerpo entero con la condicion contraria,
        // evaluada UNA vez con los flags de entrada (si el cuerpo pone flags,
        // no pueden afectar a si el cuerpo se ejecuta).
        u32 skip = 0;
        if (cond != kAlways) {
            skip = e.BranchPlaceholder(cond ^ 1u);
        }
        switch (decoded.kind) {
        case Kind::DataProcessing:
            DataProcessing(inst, pc);
            break;
        case Kind::Multiply:
            Multiply(inst);
            break;
        case Kind::MultiplyLong:
            MultiplyLong(inst);
            break;
        case Kind::LoadStore:
            LoadStore(inst, pc);
            break;
        case Kind::LoadStoreExtra:
            LoadStoreExtra(inst, pc);
            break;
        case Kind::BlockTransfer:
            BlockTransferCall(inst, pc);
            break;
        case Kind::LoadStoreDual:
            LoadStoreDual(inst, pc);
            break;
        case Kind::Nop:
            break;
        case Kind::Exclusive:
            Exclusive(inst, pc);
            break;
        case Kind::Clz:
            LoadGuest(kRm, Bits(inst, 0, 3), pc);
            e.Emit(0xE16F0F10u | (kRd << 12) | kRm); // CLZ kRd, kRm
            StoreGuest(kRd, Bits(inst, 12, 15));
            break;
        case Kind::NativeReg:
            NativeReg(inst, pc);
            break;
        case Kind::Cp15Read:
        case Kind::Cp15Write:
            Cp15(inst, pc, decoded.kind == Kind::Cp15Read);
            break;
        case Kind::VfpLoadStore:
            VfpLoadStore(inst, pc);
            break;
        case Kind::VfpMove:
            VfpMove(inst, pc);
            break;
        case Kind::VfpCdp:
            VfpCdp(inst);
            break;
        case Kind::VfpMrs:
            VfpMrs(inst);
            break;
        case Kind::VfpTransfer:
            VfpTransferCall(inst, pc);
            break;
        default:
            break;
        }
        if (cond != kAlways) {
            e.PatchBranch(skip, e.Position());
        }
    }

    /// Final de un bloque que no acaba en salto (fin de pagina).
    void FallThrough(u32 next_pc) {
        // El bloque siguiente de la pagina de al lado: destino fijo (0.1.5.7).
        ExitDirect(next_pc);
    }

private:
    Emitter& e;
    bool check_mode;
    /// guest -> registro del anfitrion (r0-r3 o lr) o -1 sin cachear. Ver
    /// SetCacheMap y kCacheHosts.
    std::array<s8, 16> cache_map{};
    /// Hay algun registro cacheado (atajo para saltarse los vuelcos).
    bool cache_dirty = false;
    /**
     * Registros cacheados que el codigo ya emitido ESCRIBE en su hueco
     * (0.1.8.1). Solo esos hace falta volcar: un registro cacheado que el
     * bloque solo lee vale lo mismo en su hueco que en cpu->Reg, y volcarlo
     * era un STR de mas en cada salida y en cada enlace (cientos de miles por
     * fotograma). Se marca al emitir, y como dentro de un bloque solo hay
     * saltos hacia delante, el conjunto marcado al emitir un vuelco cubre
     * todo lo que puede haber escrito cualquier camino que llegue a el. Lo que
     * escriben las funciones de C (LDM, VLDM...) va directo a cpu->Reg y se
     * recarga despues: no hace falta marcarlo.
     */
    std::array<bool, 16> cache_written{};
    /**
     * Los flags del juego (N, Z, C, V) se leen despues de la instruccion que
     * se esta compilando? (0.1.5.7, ver ComputeFlagsLiveAfter.) Si no, un
     * acceso a memoria no necesita guardarlos y reponerlos alrededor de su
     * comparacion. true por defecto: sin informacion, se guardan.
     */
    bool flags_live_after = true;
    /**
     * CAMINOS FRIOS FUERA DE LINEA (0.2.1.0). Lo que casi nunca corre -- el
     * acceso lento a memoria, la llamada de LDM/STM o VPUSH/VPOP cuando no hay
     * pagina directa, el enlace que falla, la aritmetica VFP con vectores --
     * iba en medio del bloque, con un salto por encima en cada acceso. Con
     * ~25.000 bloques por fotograma repartidos por megas de codigo, la cache
     * de instrucciones de 32 KB del Cortex-A9 se llenaba de lineas que no se
     * ejecutan. Ahora va todo detras del epilogo: el camino caliente queda
     * seguido, mas corto y sin el salto. Cada trozo guarda los registros
     * cacheados escritos HASTA su punto (cache_written), que es lo que vuelca.
     * Un trozo no puede crear otros.
     */
    struct ColdStub {
        std::vector<u32> branches;
        std::array<bool, 16> written;
        std::function<void()> body;
    };
    std::vector<ColdStub> cold_stubs;
    /// Saltos al epilogo (con vuelco) y donde empieza la salida sin vuelco.
    std::vector<u32> to_epilogue;
    u32 exit_noflush = 0;

    void DeferCold(std::vector<u32> branches, std::function<void()> body) {
        DeferCold(std::move(branches), cache_written, std::move(body));
    }
    void DeferCold(std::vector<u32> branches, const std::array<bool, 16>& written,
                   std::function<void()> body) {
        cold_stubs.push_back(ColdStub{std::move(branches), written, std::move(body)});
    }

public:
    void SetFlagsLiveAfter(bool live) {
        flags_live_after = live;
    }

private:
    /// Carga a r0-r2 los registros que este bloque cachea.
    void LoadCachedRegs() {
        for (u32 g = 0; g < 16; g++) {
            const s8 host = cache_map[g];
            if (host >= 0) {
                e.LdrImm(static_cast<u32>(host), kCpu, RegOffset(g));
            }
        }
    }

    /**
     * Vuelca r0-r2 a ARMul_State::Reg. Hace falta ANTES de:
     *   - cualquier Call (AAPCS destruye r0-r3, y ademas los helpers leen
     *     cpu->Reg: BlockTransfer, Cp15, SlowWrite...);
     *   - el epilogo (usa r0-r1 para los flags);
     *   - FallThrough y los terminadores que escriben Reg[15] por su cuenta.
     * No hace falta en el camino RAPIDO de MemoryAccess: ese no toca r0-r3
     * ni cpu->Reg (solo kT0/kT1/kFlags y la pagina).
     */
    void FlushCachedRegs() {
        if (!cache_dirty) {
            return;
        }
        for (u32 g = 0; g < 16; g++) {
            const s8 host = cache_map[g];
            if (host >= 0 && cache_written[g]) {
                e.StrImm(static_cast<u32>(host), kCpu, RegOffset(g));
            }
        }
    }

    /// Recarga r0-r2 despues de un helper que pudo escribir cpu->Reg (LDM...).
    void ReloadCachedRegs() {
        if (!cache_dirty) {
            return;
        }
        LoadCachedRegs();
    }

    /// El host donde vive 'guest' si esta cacheado; -1 si no.
    s8 CacheHost(u32 guest) const {
        return guest < 16 ? cache_map[guest] : static_cast<s8>(-1);
    }

    /// Carga un registro del juego; el PC se lee como direccion + 8.
    void LoadGuest(u32 host, u32 guest, u32 pc) {
        if (guest == 15) {
            e.Mov32(host, pc + 8);
            return;
        }
        const s8 cached = CacheHost(guest);
        if (cached >= 0) {
            if (static_cast<u32>(cached) != host) {
                e.MovReg(host, static_cast<u32>(cached));
            }
            return;
        }
        e.LdrImm(host, kCpu, RegOffset(guest));
    }

    void StoreGuest(u32 host, u32 guest) {
        if (guest == 15) {
            // El PC SIEMPRE se escribe a memoria: los encadenados, el
            // interprete y CompletePendingCheck lo leen de ahi. Ademas
            // FlushCachedRegs no cubre el PC porque no se cachea.
            e.StrImm(host, kCpu, RegOffset(guest));
            return;
        }
        const s8 cached = CacheHost(guest);
        if (cached >= 0) {
            if (static_cast<u32>(cached) != host) {
                e.MovReg(static_cast<u32>(cached), host);
            }
            cache_written[guest] = true;
            // NO se escribe a memoria aqui: el vuelco lo hace FlushCachedRegs
            // al salir del bloque o antes de un helper. Si 'host' ya ES el
            // registro cacheado (caso de DataProcessing con Rd cacheado), no
            // hace falta ni el mov.
            return;
        }
        e.StrImm(host, kCpu, RegOffset(guest));
    }

    void DataProcessing(u32 inst, u32 pc) {
        const u32 opcode = Bits(inst, 21, 24);
        const bool immediate = Bits(inst, 25, 25) == 1;
        const u32 rn = Bits(inst, 16, 19);
        const u32 rd = Bits(inst, 12, 15);
        const bool uses_rn = opcode != 13 && opcode != 15; // MOV, MVN
        const bool writes_rd = !(opcode >= 8 && opcode <= 11);
        /**
         * 0.1.5.7: los registros cacheados se usan DIRECTAMENTE en la
         * instruccion, sin copiarlos antes a kRn/kRm/kRs ni el resultado a su
         * hueco despues. Con Rn, Rm y Rd cacheados, "add r0, r1, r2" del juego
         * es UNA instruccion del anfitrion. Los que no estan cacheados van por
         * los temporales de siempre (y el PC, que nunca se cachea, tambien).
         */
        const u32 rn_host = uses_rn ? OperandHost(kRn, rn, pc) : 0u;
        const s8 rd_cached = writes_rd ? CacheHost(rd) : static_cast<s8>(-1);
        const u32 rd_host = rd_cached >= 0 ? static_cast<u32>(rd_cached) : kRd;
        /**
         * Los campos que la instruccion no usa van a CERO, no al registro:
         * MOV/MVN no tienen Rn y CMP/CMN/TST/TEQ no tienen Rd, y en los dos
         * casos la arquitectura exige ceros ("should be zero"). Con otro valor
         * la instruccion es IMPREDECIBLE en el procesador de la Vita.
         */
        u32 host = (inst & 0x0FFFFFFFu) | (kAlways << 28);
        host = (host & ~(0xFu << 16)) | (rn_host << 16);
        host = (host & ~(0xFu << 12)) | ((writes_rd ? rd_host : 0u) << 12);
        if (!immediate) {
            const u32 rm_host = OperandHost(kRm, Bits(inst, 0, 3), pc);
            host = (host & ~0xFu) | rm_host;
            if (Bits(inst, 4, 4) == 1) {
                const u32 rs_host = OperandHost(kRs, Bits(inst, 8, 11), pc);
                host = (host & ~(0xFu << 8)) | (rs_host << 8);
            }
        }
        // La misma instruccion, con los registros cambiados: los flags (y el
        // acarreo que leen ADC/SBC/RSC) son los del juego, que estan en APSR.
        e.Emit(host);
        if (writes_rd && rd_cached >= 0) {
            cache_written[rd] = true;
        }
        if (writes_rd && rd_cached < 0) {
            StoreGuest(kRd, rd);
        }
    }

    /// El registro del anfitrion donde esta el valor de 'guest' para leerlo:
    /// su hueco de la cache si lo tiene; si no, se carga en 'temp' (el PC,
    /// como direccion + 8). Ver DataProcessing (0.1.5.7).
    u32 OperandHost(u32 temp, u32 guest, u32 pc) {
        const s8 cached = CacheHost(guest);
        if (cached >= 0) {
            return static_cast<u32>(cached);
        }
        LoadGuest(temp, guest, pc);
        return temp;
    }

    void Multiply(u32 inst) {
        const u32 rd = Bits(inst, 16, 19);
        const u32 rn = Bits(inst, 12, 15);
        const bool accumulate = Bits(inst, 21, 21) == 1;
        LoadGuest(kRm, Bits(inst, 0, 3), 0);
        LoadGuest(kRs, Bits(inst, 8, 11), 0);
        if (accumulate) {
            LoadGuest(kRn, rn, 0);
        }
        // MUL no tiene Rn: su campo tiene que ir a cero (ver DataProcessing).
        u32 host = (inst & 0x0FFFFFFFu) | (kAlways << 28);
        host = (host & ~(0xFu << 16)) | (kRd << 16);
        host = (host & ~(0xFu << 12)) | ((accumulate ? kRn : 0u) << 12);
        host = (host & ~(0xFu << 8)) | (kRs << 8);
        host = (host & ~0xFu) | kRm;
        e.Emit(host);
        StoreGuest(kRd, rd);
    }

    void MultiplyLong(u32 inst) {
        const u32 hi = Bits(inst, 16, 19);
        const u32 lo = Bits(inst, 12, 15);
        const bool accumulate = Bits(inst, 21, 21) == 1;
        LoadGuest(kRm, Bits(inst, 0, 3), 0);
        LoadGuest(kRs, Bits(inst, 8, 11), 0);
        if (accumulate) {
            LoadGuest(kRd, lo, 0);
            LoadGuest(kT0, hi, 0);
        }
        u32 host = (inst & 0x0FFFFFFFu) | (kAlways << 28);
        host = (host & ~(0xFu << 16)) | (kT0 << 16);
        host = (host & ~(0xFu << 12)) | (kRd << 12);
        host = (host & ~(0xFu << 8)) | (kRs << 8);
        host = (host & ~0xFu) | kRm;
        e.Emit(host);
        StoreGuest(kRd, lo);
        StoreGuest(kT0, hi);
    }

    /**
     * Un acceso a memoria con la direccion en 'address'. 'data' es el registro
     * del anfitrion con el valor a guardar o el que recibe el cargado; desde
     * 0.2.1.0 puede ser el hueco de la cache del registro del juego (sin pasar
     * por kRd y un MOV).
     *
     * Camino normal: el mismo que MemorySystem::Read32Fast/Write32Fast --
     * puntero = paginas[direccion >> 12]; si no es nulo, acceso directo con el
     * desplazamiento de la pagina; si es nulo, la funcion de siempre del
     * interprete (ReadMemory32 & co.), que sabe de hardware y de memoria
     * especial, en un camino frio (ColdStub). Los flags del juego estan en
     * APSR y la comparacion con cero los pisaria: se guardan en r12 y se
     * reponen al final.
     *
     * Variante de comprobacion: siempre por CheckRead/CheckWrite (diario).
     */
    void MemoryAccess(u32 address, bool store, u32 kind, u32 pc, u32 data = kRd) {
        /**
         * 0.1.5.7: los flags solo se guardan y reponen si alguien los va a leer
         * despues (flags_live_after, ver ComputeFlagsLiveAfter). En el caso
         * tipico "ldr r0, [..]; cmp r0, #0; bne" el CMP los sobrescribe todos
         * antes de que nadie los lea, y el MRS y el MSR sobran. La variante de
         * comprobacion hace LO MISMO, para que un error en el analisis se vea
         * como DIFERENCIA en vez de quedar escondido.
         */
        const bool keep_flags = flags_live_after;
        if (keep_flags) {
            e.Mrs(kFlags);
        }
        if (check_mode) {
            SaveAccessOperands(address, store, data);
            FlushCachedRegs();
            if (keep_flags) {
                e.MovReg(kT0, kFlags);
            }
            e.MovReg(R0, kT1);
            if (store) {
                e.MovReg(R1, kRd);
                e.Mov32(R2, kind);
                e.Call(reinterpret_cast<const void*>(&CheckWriteThunk));
            } else {
                e.Mov32(R1, kind);
                e.Call(reinterpret_cast<const void*>(&CheckReadThunk));
                e.MovReg(kRd, R0);
            }
            if (keep_flags) {
                e.MsrFlags(kT0);
            }
            ReloadCachedRegs();
            if (!store && data != kRd) {
                e.MovReg(data, kRd);
            }
            return;
        }
        e.LsrImm(kT0, address, 12);
        e.LdrRegLsl2(kT1, kPages, kT0);
        e.CmpImm0(kT1);
        const u32 to_slow = e.BranchPlaceholder(kCondEq);
        e.Ubfx(kT0, address, 0, 12);
        // [kT1, kT0]
        switch (kind) {
        case kWord:
            e.Emit((store ? 0xE7800000u : 0xE7900000u) | (kT1 << 16) | (data << 12) | kT0);
            break;
        case kByte:
            e.Emit((store ? 0xE7C00000u : 0xE7D00000u) | (kT1 << 16) | (data << 12) | kT0);
            break;
        case kHalf:
            e.Emit((store ? 0xE18000B0u : 0xE19000B0u) | (kT1 << 16) | (data << 12) | kT0);
            break;
        case kSignedHalf:
            e.Emit(0xE19000F0u | (kT1 << 16) | (data << 12) | kT0);
            break;
        case kSignedByte:
            e.Emit(0xE19000D0u | (kT1 << 16) | (data << 12) | kT0);
            break;
        }
        const u32 back = e.Position();
        if (keep_flags) {
            e.MsrFlags(kFlags);
        }
        DeferCold({to_slow}, [this, address, store, kind, pc, data, keep_flags, back] {
            // r12 no sobrevive a la llamada: los flags van a r10. r0-r3 y lr se
            // destruyen, asi que la cache se vuelca antes y se recarga despues
            // (SlowRead/SlowWrite leen y escriben cpu->Reg).
            SaveAccessOperands(address, store, data);
            FlushCachedRegs();
            if (keep_flags) {
                e.MovReg(kT0, kFlags);
            }
            e.Mov32(R0, pc);
            e.StrImm(R0, kCpu, RegOffset(15));
            e.MovReg(R0, kCpu);
            e.MovReg(R1, kT1);
            if (store) {
                e.MovReg(R2, kRd);
                e.Mov32(R3, kind);
                e.Call(reinterpret_cast<const void*>(&SlowWrite));
            } else {
                e.Mov32(R2, kind);
                e.Call(reinterpret_cast<const void*>(&SlowRead));
                e.MovReg(kRd, R0);
            }
            if (keep_flags) {
                e.MovReg(kFlags, kT0);
            }
            ReloadCachedRegs();
            if (!store && data != kRd) {
                e.MovReg(data, kRd);
            }
            e.BranchTo(kAlways, back);
        });
    }

    /// La direccion a kT1 y el valor a guardar a kRd, ANTES del vuelco y de
    /// los argumentos: los dos pueden estar en huecos de la cache (r0-r3, lr).
    void SaveAccessOperands(u32 address, bool store, u32 data) {
        if (address != kT1) {
            e.MovReg(kT1, address);
        }
        if (store && data != kRd) {
            e.MovReg(kRd, data);
        }
    }

    /// Carga en el registro del juego 'rd' (no el PC): directa a su hueco si
    /// esta cacheado (0.2.1.0).
    void LoadInto(u32 address, u32 kind, u32 pc, u32 rd) {
        const s8 cached = CacheHost(rd);
        if (cached >= 0) {
            MemoryAccess(address, false, kind, pc, static_cast<u32>(cached));
            cache_written[rd] = true;
            return;
        }
        MemoryAccess(address, false, kind, pc);
        e.StrImm(kRd, kCpu, RegOffset(rd));
    }

    /// rd = rn + value (o - value) sin tocar flags: el inmediato va en el
    /// propio ADD/SUB si cabe; si no, la constante por kRm (rn no puede ser kRm).
    void AddImmediate(u32 rd, u32 rn, u32 value, bool up) {
        u32 encoded = 0;
        if (Emitter::EncodeImmediate(value, &encoded)) {
            e.Emit((up ? 0xE2800000u : 0xE2400000u) | (rn << 16) | (rd << 12) | encoded);
            return;
        }
        e.Mov32(kRm, value);
        if (up) {
            e.AddReg(rd, rn, kRm);
        } else {
            e.SubReg(rd, rn, kRm);
        }
    }

    /// rd = rn + value para los desplazamientos de LDM/STM y VPUSH/VPOP
    /// (multiplos de 4 de hasta 1020: siempre caben en el inmediato, asi que
    /// no se toca ningun temporal; kFlags puede llevar los flags).
    void AddSmall(u32 rd, u32 rn, s32 value) {
        if (value == 0) {
            if (rd != rn) {
                e.MovReg(rd, rn);
            }
            return;
        }
        AddImmediate(rd, rn, static_cast<u32>(value < 0 ? -value : value), value > 0);
    }

    /**
     * LDR/STR/LDRB/STRB. Mismo orden que el interprete: la direccion y la
     * escritura del base van ANTES del acceso (get_addr), y en un guardado del
     * PC se guarda PC+8.
     *
     * 0.2.1.0: el desplazamiento inmediato va en el propio ADD/SUB si cabe
     * (antes MOVW + ADD), el de registro con su desplazamiento en el mismo ADD
     * (antes MOV + ADD), el base cacheado se usa sin copiarlo y el dato va de y
     * a su hueco de la cache. Con escritura del base, el base se copia a kRn:
     * su hueco recibe el valor nuevo antes del acceso y el post-indice necesita
     * el original.
     */
    void LoadStore(u32 inst, u32 pc) {
        const bool reg_offset = Bits(inst, 25, 25) == 1;
        const bool pre = Bits(inst, 24, 24) == 1;
        const bool up = Bits(inst, 23, 23) == 1;
        const bool byte = Bits(inst, 22, 22) == 1;
        const bool wb = Bits(inst, 21, 21) == 1;
        const bool load = Bits(inst, 20, 20) == 1;
        const u32 rn = Bits(inst, 16, 19);
        const u32 rd = Bits(inst, 12, 15);
        const bool writeback = !pre || wb;

        u32 address = kRs;
        if (rn == 15 && !reg_offset && !writeback) {
            // Un literal: la direccion es una constante.
            const u32 offset = inst & 0xFFFu;
            e.Mov32(kRs, up ? pc + 8 + offset : pc + 8 - offset);
        } else {
            u32 base = kRn;
            if (writeback) {
                LoadGuest(kRn, rn, pc);
            } else {
                base = OperandHost(kRn, rn, pc);
            }
            u32 computed = base;
            if (reg_offset) {
                // ADD/SUB kRs, base, rm, <desplazamiento>: los bits 5-11 son los
                // mismos que en LDR. Con RRX lee el acarreo del juego (APSR).
                const u32 rm_host = OperandHost(kRm, Bits(inst, 0, 3), pc);
                e.Emit((up ? 0xE0800000u : 0xE0400000u) | (base << 16) | (kRs << 12) |
                       (inst & 0xFE0u) | rm_host);
                computed = kRs;
            } else if ((inst & 0xFFFu) != 0) {
                AddImmediate(kRs, base, inst & 0xFFFu, up);
                computed = kRs;
            }
            if (writeback) {
                StoreGuest(computed, rn);
            }
            address = pre ? computed : base;
        }
        const u32 kind = byte ? kByte : kWord;
        if (load && rd == 15) {
            // LDR al PC (fin de bloque): como LDR_INST con Rd = 15, pasa a Thumb
            // si el bit 0 viene puesto y deja el PC sin ese bit.
            // kT0 en vez de R2: R2 es un hueco de la cache de registros.
            MemoryAccess(address, false, kind, pc);
            e.Ubfx(kT0, kRd, 0, 1);
            e.StrImm(kT0, kCpu, g_offsets.t);
            e.Emit(0xE3C00001u | (kRd << 16) | (kRd << 12)); // BIC kRd, kRd, #1
            StoreGuest(kRd, 15);
        } else if (load) {
            LoadInto(address, kind, pc, rd);
        } else {
            // Despues de escribir el base; el PC se guarda como PC+8.
            const u32 value = OperandHost(kRd, rd, pc);
            MemoryAccess(address, true, kind, pc, value);
        }
    }

    /// LDRH/STRH/LDRSB/LDRSH, como LoadStore (0.2.1.0): el inmediato de 8 bits
    /// siempre cabe en el ADD/SUB.
    void LoadStoreExtra(u32 inst, u32 pc) {
        const bool pre = Bits(inst, 24, 24) == 1;
        const bool up = Bits(inst, 23, 23) == 1;
        const bool imm_offset = Bits(inst, 22, 22) == 1;
        const bool wb = Bits(inst, 21, 21) == 1;
        const bool load = Bits(inst, 20, 20) == 1;
        const u32 rn = Bits(inst, 16, 19);
        const u32 rd = Bits(inst, 12, 15);
        const u32 sh = Bits(inst, 5, 6);
        const bool writeback = !pre || wb;
        const u32 imm = (Bits(inst, 8, 11) << 4) | Bits(inst, 0, 3);

        u32 address = kRs;
        if (rn == 15 && imm_offset && !writeback) {
            e.Mov32(kRs, up ? pc + 8 + imm : pc + 8 - imm);
        } else {
            u32 base = kRn;
            if (writeback) {
                LoadGuest(kRn, rn, pc);
            } else {
                base = OperandHost(kRn, rn, pc);
            }
            u32 computed = base;
            if (!imm_offset) {
                const u32 rm_host = OperandHost(kRm, Bits(inst, 0, 3), pc);
                e.Emit((up ? 0xE0800000u : 0xE0400000u) | (base << 16) | (kRs << 12) | rm_host);
                computed = kRs;
            } else if (imm != 0) {
                AddImmediate(kRs, base, imm, up);
                computed = kRs;
            }
            if (writeback) {
                StoreGuest(computed, rn);
            }
            address = pre ? computed : base;
        }
        const u32 kind = sh == 1 ? kHalf : sh == 2 ? kSignedByte : kSignedHalf;
        if (load) {
            LoadInto(address, kind, pc, rd);
        } else {
            const u32 value = OperandHost(kRd, rd, pc);
            MemoryAccess(address, true, kHalf, pc, value);
        }
    }

    /**
     * LDRD/STRD (0.1.4.9): como LDRD_INST/STRD_INST, DOS accesos de 32 bits
     * separados (el 3DS no los hace de una vez), primero a la direccion y
     * despues a la direccion + 4, con el base escrito antes, como en get_addr.
     */
    void LoadStoreDual(u32 inst, u32 pc) {
        const bool pre = Bits(inst, 24, 24) == 1;
        const bool up = Bits(inst, 23, 23) == 1;
        const bool imm_offset = Bits(inst, 22, 22) == 1;
        const bool wb = Bits(inst, 21, 21) == 1;
        const bool load = Bits(inst, 5, 6) == 2;
        const u32 rn = Bits(inst, 16, 19);
        const u32 rd = Bits(inst, 12, 15);

        LoadGuest(kRn, rn, pc);
        if (imm_offset) {
            e.Mov32(kRm, (Bits(inst, 8, 11) << 4) | Bits(inst, 0, 3));
        } else {
            LoadGuest(kRm, Bits(inst, 0, 3), pc);
        }
        if (up) {
            e.AddReg(kRs, kRn, kRm);
        } else {
            e.SubReg(kRs, kRn, kRm);
        }
        if (!pre || wb) {
            StoreGuest(kRs, rn);
        }
        const u32 address = pre ? kRs : kRn;
        // ADD kRs, address, #4: la segunda direccion (sin tocar flags). El base
        // ya esta escrito, asi que kRs se puede reutilizar.
        const u32 second = 0xE2800004u | (address << 16) | (kRs << 12);
        if (load) {
            MemoryAccess(address, false, kWord, pc);
            StoreGuest(kRd, rd);
            e.Emit(second);
            MemoryAccess(kRs, false, kWord, pc);
            StoreGuest(kRd, rd + 1);
        } else {
            LoadGuest(kRd, rd, pc);
            MemoryAccess(address, true, kWord, pc);
            e.Emit(second);
            LoadGuest(kRd, rd + 1, pc);
            MemoryAccess(kRs, true, kWord, pc);
        }
    }

    /**
     * SXTAB, UXTH, REV y compania (0.1.4.9): la misma instruccion con los registros
     * cambiados. Rn = 1111 es la forma SIN acumular (SXTB, UXTH...) y en REV es
     * obligatorio: se deja tal cual.
     */
    void NativeReg(u32 inst, u32 pc) {
        const u32 rn = Bits(inst, 16, 19);
        LoadGuest(kRm, Bits(inst, 0, 3), pc);
        u32 host = (inst & 0x0FFFFFFFu) | (kAlways << 28);
        host = (host & ~(0xFu << 12)) | (kRd << 12);
        host = (host & ~0xFu) | kRm;
        if (rn != 15) {
            LoadGuest(kRn, rn, pc);
            host = (host & ~(0xFu << 16)) | (kRn << 16);
        }
        e.Emit(host);
        StoreGuest(kRd, Bits(inst, 12, 15));
    }

    /**
     * LDREX/STREX por JitLoadExclusive/JitStoreExclusive (0.2.1.0). Leen y
     * escriben el monitor exclusivo y la memoria como el interprete. La
     * variante de comprobacion no puede tocar ninguno de los dos: se abandona,
     * como con las escrituras de CP15.
     */
    void Exclusive(u32 inst, u32 pc) {
        const u32 op = Bits(inst, 20, 27);
        const bool load = (op & 1) != 0;
        const u32 kind = (op & 0x6) == 0x4 ? kByte : (op & 0x6) == 0x6 ? kHalf : kWord;
        // A temporales que la llamada conserva ANTES de usar r0-r3: pueden
        // venir de huecos de la cache.
        LoadGuest(kRn, Bits(inst, 16, 19), pc);
        if (!load) {
            LoadGuest(kRm, Bits(inst, 0, 3), pc);
        }
        FlushCachedRegs();
        e.Mrs(kFlags);
        e.MovReg(kT0, kFlags);
        if (check_mode) {
            e.Call(reinterpret_cast<const void*>(&CheckAbortThunk));
        } else {
            e.Mov32(R0, pc);
            e.StrImm(R0, kCpu, RegOffset(15));
            e.MovReg(R0, kCpu);
            e.MovReg(R1, kRn);
            if (load) {
                e.Mov32(R2, kind);
                e.Call(reinterpret_cast<const void*>(&JitLoadExclusive));
            } else {
                e.MovReg(R2, kRm);
                e.Mov32(R3, kind);
                e.Call(reinterpret_cast<const void*>(&JitStoreExclusive));
            }
        }
        // kRd y no un hueco: ReloadCachedRegs lo pisaria.
        e.MovReg(kRd, R0);
        e.MsrFlags(kT0);
        ReloadCachedRegs();
        StoreGuest(kRd, Bits(inst, 12, 15));
    }

    /**
     * MRC/MCR del coprocesador 15 (0.1.4.9): las mismas funciones que el
     * interprete (ReadCP15Register / WriteCP15Register). La lectura del TLS
     * ("mrc p15, 0, rX, c13, c0, 3") esta por todas partes en el codigo del 3DS.
     * Una escritura puede tener efectos (barreras, caches) que no se pueden
     * anotar en el diario: en la variante de comprobacion, se abandona.
     */
    void Cp15(u32 inst, u32 pc, bool read) {
        /**
         * EL TLS EN LINEA (0.2.1.0). "mrc p15, 0, rX, c13, c0, 3" esta en casi
         * cada funcion del 3DS, y era un vuelco de la cache, una llamada a C y
         * una recarga. Los registros de hilo (c13, c0, 2 y 3) los devuelve
         * ReadCP15Register tal cual de CP15[], sin mirar el modo: un LDR.
         */
        const u32 opc2 = Bits(inst, 5, 7);
        if (read && Bits(inst, 16, 19) == 13 && Bits(inst, 21, 23) == 0 && Bits(inst, 0, 3) == 0 &&
            (opc2 == 2 || opc2 == 3)) {
            const u32 index = opc2 == 2 ? CP15_THREAD_UPRW : CP15_THREAD_URO;
            const u32 rt = Bits(inst, 12, 15);
            const s8 cached = CacheHost(rt);
            if (cached >= 0) {
                e.LdrImm(static_cast<u32>(cached), kCpu, g_offsets.cp15 + index * 4);
                cache_written[rt] = true;
            } else {
                e.LdrImm(kRd, kCpu, g_offsets.cp15 + index * 4);
                e.StrImm(kRd, kCpu, RegOffset(rt));
            }
            return;
        }
        FlushCachedRegs();
        e.Mrs(kFlags);
        e.MovReg(kT0, kFlags);
        if (read) {
            e.MovReg(R0, kCpu);
            e.Mov32(R1, inst);
            e.Call(reinterpret_cast<const void*>(&JitCp15Read));
            // kRd guarda el resultado: StoreGuest despues de ReloadCachedRegs,
            // porque ReloadCachedRegs recarga R0-R2 desde memoria y pisaria el
            // resultado del MRC si estuviera en un hueco de la cache.
            e.MovReg(kRd, R0);
        } else if (check_mode) {
            e.Call(reinterpret_cast<const void*>(&CheckAbortThunk));
        } else {
            LoadGuest(R2, Bits(inst, 12, 15), pc);
            e.MovReg(R0, kCpu);
            e.Mov32(R1, inst);
            e.Call(reinterpret_cast<const void*>(&JitCp15Write));
        }
        e.MsrFlags(kT0);
        ReloadCachedRegs();
        if (read) {
            StoreGuest(kRd, Bits(inst, 12, 15));
        }
    }

    /**
     * VMOV entre un registro ARM y un single VFP (0.1.5.2, 4.4 fase 1).
     * Mismo bit a bit que VMOVBRS/VMOVBRC del interprete: ExtReg[n] <-> Reg[t].
     * No toca memoria ni flags; solo mueve una palabra. Con la cache de
     * registros encendida, StoreGuest/LoadGuest ya resuelven el lado ARM.
     */
    void VfpMove(u32 inst, u32 pc) {
        const bool to_arm = Bits(inst, 20, 20) == 1;
        const u32 rt = Bits(inst, 12, 15);
        // Mismo indice que VMOVBRS: n = Vn<<1 | bit7 (bit 7 = N). ARREGLO
        // 0.1.5.5: antes se ignoraba el bit 7 y s1, s3, s5... iban al single
        // par de al lado.
        const u32 n = (Bits(inst, 16, 19) << 1) | Bits(inst, 7, 7);
        if (to_arm) {
            e.LdrImm(kRd, kCpu, ExtOffset(n));
            StoreGuest(kRd, rt);
        } else {
            LoadGuest(kRd, rt, pc);
            e.StrImm(kRd, kCpu, ExtOffset(n));
        }
    }

    /**
     * VLDR/VSTR single o double (0.1.5.2, 4.4 fase 1). La direccion se
     * calcula igual que en LoadStore (base + o - inmediato, PC = pc+8) y el
     * acceso va por MemoryAccess, que en la variante de comprobacion journaliza
     * y en la normal usa la pagina rapida. El valor VFP va por ExtReg.
     * Double = dos accesos de 32 bits, little-endian (el 3DS no es BE).
     */
    void VfpLoadStore(u32 inst, u32 pc) {
        const bool load = Bits(inst, 20, 20) == 1;
        const bool add = Bits(inst, 23, 23) == 1;
        const bool single = Bits(inst, 8, 8) == 0;
        const u32 imm32 = Bits(inst, 0, 7) << 2;
        const u32 rn = Bits(inst, 16, 19);
        const u32 vd_field = Bits(inst, 12, 15);
        const u32 bit22 = Bits(inst, 22, 22);
        // Indice igual que en VSTR_INST/VLDR_INST del interprete.
        const u32 d = single ? ((vd_field << 1) | bit22) : (vd_field | (bit22 << 4));

        // 0.2.1.0: el inmediato (8 bits por 4) siempre cabe en el ADD/SUB, y con
        // el PC de base (las constantes en coma flotante) es una constante.
        if (rn == 15) {
            e.Mov32(kRs, add ? pc + 8 + imm32 : pc + 8 - imm32);
        } else {
            AddImmediate(kRs, OperandHost(kRn, rn, pc), imm32, add);
        }
        // kRs = direccion base+/-imm. Para double, la segunda palabra es +4.
        if (single) {
            if (load) {
                MemoryAccess(kRs, false, kWord, pc);
                e.StrImm(kRd, kCpu, ExtOffset(d));
            } else {
                e.LdrImm(kRd, kCpu, ExtOffset(d));
                MemoryAccess(kRs, true, kWord, pc);
            }
            return;
        }
        // Double: dos accesos LE, word1 en addr+0 y word2 en addr+4.
        // ARREGLO 0.1.5.5: el double dN vive en ExtReg[N*2] y ExtReg[N*2+1]
        // (VLDR_INST/VSTR_INST del interprete), no en ExtReg[N] y [N+1].
        const u32 next_word = 0xE2800004u | (kRs << 16) | (kRs << 12); // ADD kRs, kRs, #4
        if (load) {
            MemoryAccess(kRs, false, kWord, pc);
            e.StrImm(kRd, kCpu, ExtOffset(d * 2));
            e.Emit(next_word);
            MemoryAccess(kRs, false, kWord, pc);
            e.StrImm(kRd, kCpu, ExtOffset(d * 2 + 1));
        } else {
            e.LdrImm(kRd, kCpu, ExtOffset(d * 2));
            MemoryAccess(kRs, true, kWord, pc);
            e.Emit(next_word);
            e.LdrImm(kRd, kCpu, ExtOffset(d * 2 + 1));
            MemoryAccess(kRs, true, kWord, pc);
        }
    }

    /**
     * VFP aritmetica (0.1.6.1): una llamada a HelperVfpCdp, que hace lo mismo
     * que el interprete. La funcion no lee cpu->Reg, pero la llamada destroza
     * r0-r3 y lr (huecos de la cache) y los flags: se vuelca y se recarga la
     * cache y los flags del juego van por r10, como en Cp15.
     */
    void VfpCdp(u32 inst) {
        /**
         * VFP NATIVO (0.1.7.0). Medido en 0.1.6.3: 18.500 operaciones VFP por
         * fotograma por HelperVfpCdp, que es la coma flotante EMULADA en
         * software del interprete (cientos de ciclos cada una): la mayor parte
         * de lo que quedaba de "arm". Aqui la hace el VFP de la Vita, con el
         * FPSCR DEL JUEGO cargado en el procesador durante la instruccion
         * (redondeo, FZ y DN los del juego) y leido de vuelta despues (flags
         * de VCMP y acumulados). Las operaciones basicas de IEEE (suma, resta,
         * producto, division, raiz, conversiones) son exactas por definicion,
         * y VMLA/VMLS del VFPv3 no son fusionadas, igual que en el ARM11.
         *
         * Se queda en la funcion del interprete:
         *   - VCVT entre simple y doble (vcvt(bds)): poco frecuente;
         *   - en tiempo de ejecucion, si el FPSCR del juego pide VECTORES
         *     cortos (LEN o STRIDE distintos de 0): el Cortex-A9 no los tiene.
         * La comprobacion contra el interprete sigue: si el hardware diera otro
         * resultado, el bloque vuelve al interprete (ver CompletePendingCheck).
         */
        if (!VfpNative(inst)) {
            VfpCdpHelper(inst);
            return;
        }
        // Las nativas llegan por VfpCdpRun desde EmitBlock; aqui solo si no.
        VfpCdpRun(&inst, 1, flags_live_after, false);
    }

public:
    /// La aritmetica VFP de 'inst' va por el VFP de la Vita (ver VfpCdp).
    static bool VfpNative(u32 inst) {
        int index = -1;
        DecodeARMInstruction(inst, &index);
        return vfp_native.load(std::memory_order_relaxed) != 0 && index != kVfpVcvtBds;
    }

    /**
     * ARITMETICA VFP SEGUIDA CON UN SOLO CAMBIO DE FPSCR (0.2.1.0). Cada
     * operacion guardaba el FPSCR del emulador, ponia el del juego, lo leia de
     * vuelta y reponia el del emulador: cuatro accesos al FPSCR por cada VMUL
     * o VMLA, y en el Cortex-A9 cada uno espera a que el VFP vacie su tuberia.
     * Las seguidas (las cuentas de matrices y vectores del 3D) comparten uno al
     * principio y otro al final: entre ellas no hay nada que lea el FPSCR del
     * juego en memoria ni codigo de C que espere el del emulador, y los
     * acumulados y los flags de VCMP quedan igual que una a una. LEN y STRIDE
     * se miran una vez: solo los cambia un VMSR, que no se compila. Con
     * vectores, todas por la funcion del interprete, en un camino frio.
     * 'conditions': las condiciones de cada una se emiten aqui (falso cuando
     * Instruction ya puso la de una sola).
     */
    void VfpCdpRun(const u32* insts, u32 count, bool keep_flags, bool conditions) {
        const u32 fpscr_offset = g_offsets.vfp + VFP_FPSCR * 4;
        // Los flags del juego: el TST los pisa.
        if (keep_flags) {
            e.Mrs(kT0);
        }
        e.LdrImm(kRd, kCpu, fpscr_offset);
        // TST kRd, #0x370000 (LEN y STRIDE): con vectores, a la funcion.
        e.Emit(0xE3100000u | (kRd << 16) | (8u << 8) | 0x37u);
        const u32 to_slow = e.BranchPlaceholder(kCondNe);
        if (keep_flags) {
            e.MsrFlags(kT0);
        }
        // kT1 = &ExtReg[0]
        e.Mov32(kT1, g_offsets.ext);
        e.AddReg(kT1, kCpu, kT1);
        e.Emit(0xEEF10A10u | (kRn << 12)); // VMRS kRn, FPSCR (el del emulador)
        e.Emit(0xEEE10A10u | (kRd << 12)); // VMSR FPSCR, kRd (el del juego)
        for (u32 k = 0; k < count; k++) {
            const u32 cond = insts[k] >> 28;
            const bool conditional = conditions && cond != kAlways;
            const u32 skip = conditional ? e.BranchPlaceholder(cond ^ 1u) : 0;
            VfpCdpBody(insts[k]);
            if (conditional) {
                e.PatchBranch(skip, e.Position());
            }
        }
        e.Emit(0xEEF10A10u | (kRd << 12)); // VMRS kRd, FPSCR (el del juego, con flags)
        e.StrImm(kRd, kCpu, fpscr_offset);
        e.Emit(0xEEE10A10u | (kRn << 12)); // VMSR FPSCR, kRn (vuelve el del emulador)
        const u32 back = e.Position();
        // kT0 es de los que la llamada conserva; los flags se reponen despues
        // de cada una para la condicion de la siguiente.
        std::vector<u32> ops(insts, insts + count);
        DeferCold({to_slow}, [this, ops = std::move(ops), keep_flags, conditions, back] {
            if (keep_flags) {
                e.MsrFlags(kT0);
            }
            for (const u32 inst : ops) {
                const u32 cond = inst >> 28;
                const bool conditional = conditions && cond != kAlways;
                const u32 skip = conditional ? e.BranchPlaceholder(cond ^ 1u) : 0;
                FlushCachedRegs();
                e.MovReg(R0, kCpu);
                e.Mov32(R1, inst);
                e.Call(reinterpret_cast<const void*>(&HelperVfpCdp));
                ReloadCachedRegs();
                if (keep_flags) {
                    e.MsrFlags(kT0);
                }
                if (conditional) {
                    e.PatchBranch(skip, e.Position());
                }
            }
            e.BranchTo(kAlways, back);
        });
    }

private:
    /// Operandos, operacion y resultado de una aritmetica VFP, con el FPSCR del
    /// juego ya puesto y kT1 = &ExtReg[0] (ver VfpCdpRun).
    void VfpCdpBody(u32 inst) {
        int index = -1;
        DecodeARMInstruction(inst, &index);
        const bool sz = Bits(inst, 8, 8) == 1;
        bool d_read = false, d_write = true, d_double = sz, uses_n = false, uses_m = true;
        bool m_double = sz;
        if (index <= 3) { // vmla, vmls, vnmla, vnmls: acumulan en d
            d_read = true;
            uses_n = true;
        } else if (index <= kVfpVdiv) { // vnmul, vmul, vadd, vsub, vdiv
            uses_n = true;
        } else if (index == 14) { // vcmp d, m: solo FPSCR
            d_read = true;
            d_write = false;
        } else if (index == 15) { // vcmp2 d, #0
            d_read = true;
            d_write = false;
            uses_m = false;
        } else if (index == kVfpVcvtBfi) {
            // Entero siempre en un registro simple. Bit 18: 0 = de entero a
            // coma flotante (d con tamano sz, m simple); 1 = a entero (d
            // simple, m con tamano sz).
            if (Bits(inst, 18, 18) == 0) {
                m_double = false;
            } else {
                d_double = false;
            }
        }
        // Operandos: d -> s0/d0, n -> s2/d1, m -> s4/d2 (campo Vx 0, 1, 2).
        const auto guest_offset = [](u32 field, u32 bit, bool is_double) {
            return is_double ? (field | (bit << 4)) * 8 : ((field << 1) | bit) * 4;
        };
        const u32 d_off = guest_offset(Bits(inst, 12, 15), Bits(inst, 22, 22), d_double);
        const u32 n_off = guest_offset(Bits(inst, 16, 19), Bits(inst, 7, 7), sz);
        const u32 m_off = guest_offset(Bits(inst, 0, 3), Bits(inst, 5, 5), m_double);
        const auto vldr = [&](u32 host_field, u32 offset, bool is_double) {
            e.Emit((is_double ? 0xED900B00u : 0xED900A00u) | (kT1 << 16) | (host_field << 12) |
                   (offset / 4));
        };
        if (d_read) {
            vldr(0, d_off, d_double);
        }
        if (uses_n) {
            vldr(1, n_off, sz);
        }
        if (uses_m) {
            vldr(2, m_off, m_double);
        }
        // La misma instruccion con los registros cambiados.
        u32 host = (inst & 0x0FFFFFFFu) | (kAlways << 28);
        host &= ~((0xFu << 12) | (1u << 22)); // Vd = 0, D = 0
        if (uses_n) {
            host = (host & ~((0xFu << 16) | (1u << 7))) | (1u << 16); // Vn = 1, N = 0
        }
        if (uses_m) {
            host = (host & ~(0xFu | (1u << 5))) | 2u; // Vm = 2, M = 0
        }
        e.Emit(host);
        if (d_write) {
            e.Emit((d_double ? 0xED800B00u : 0xED800A00u) | (kT1 << 16) | (d_off / 4));
        }
    }

    /// La aritmetica VFP por la funcion del interprete (lo de 0.1.6.1).
    void VfpCdpHelper(u32 inst) {
        FlushCachedRegs();
        e.Mrs(kT0);
        e.MovReg(R0, kCpu);
        e.Mov32(R1, inst);
        e.Call(reinterpret_cast<const void*>(&HelperVfpCdp));
        e.MsrFlags(kT0);
        ReloadCachedRegs();
    }

    /**
     * VMRS de FPSCR (0.1.6.1), como VMRS_INST con reg == 1:
     *   Rt == 15: N, Z, C y V del juego salen de los bits 31-28 del FPSCR. Aqui
     *             los flags del juego SON los del APSR, asi que se escriben ahi
     *             (Q a cero: el juego no la ve, el epilogo solo guarda NZCV).
     *   Rt != 15: Rt = FPSCR.
     */
    void VfpMrs(u32 inst) {
        const u32 rt = Bits(inst, 12, 15);
        const u32 fpscr_offset = g_offsets.vfp + VFP_FPSCR * 4;
        if (rt == 15) {
            e.LdrImm(kT0, kCpu, fpscr_offset);
            // AND kT0, kT0, #0xF0000000 (0xF0 rotado 8: rot = 4)
            e.Emit(0xE2000000u | (kT0 << 16) | (kT0 << 12) | 0x4F0u);
            e.MsrFlags(kT0);
        } else {
            e.LdrImm(kRd, kCpu, fpscr_offset);
            StoreGuest(kRd, rt);
        }
    }

    /**
     * VPUSH/VPOP/VSTM/VLDM (0.1.6.1): una llamada a VfpTransfer, igual que
     * LDM/STM con BlockTransfer. Lee y escribe cpu->Reg (el base), asi que la
     * cache se vuelca antes y se recarga despues. Reg[15] = pc antes, porque
     * VSTM/VLDM con base PC lo leen de ahi (mas 8), como en el interprete.
     * Lo normal va en linea (VfpTransferFast).
     */
    void VfpTransferCall(u32 inst, u32 pc) {
        int index = -1;
        DecodeARMInstruction(inst, &index); // una vez, al compilar (0.1.8.5)
        if (!check_mode && VfpTransferInlinable(inst, index)) {
            VfpTransferFast(inst, pc, index);
            return;
        }
        FlushCachedRegs();
        e.Mrs(kFlags);
        e.MovReg(kT0, kFlags);
        e.Mov32(R0, pc);
        e.StrImm(R0, kCpu, RegOffset(15));
        e.MovReg(R0, kCpu);
        e.Mov32(R1, inst);
        e.Mov32(R2, static_cast<u32>(index));
        e.Call(check_mode ? reinterpret_cast<const void*>(&CheckVfpTransfer)
                          : reinterpret_cast<const void*>(&HelperVfpTransfer));
        e.MsrFlags(kT0);
        ReloadCachedRegs();
    }

    /// El base no es el PC (ese lee Reg[15] + 8) y hay algo que copiar.
    static bool VfpTransferInlinable(u32 inst, int index) {
        const bool stack = index == kVfpVpush || index == kVfpVpop;
        const u32 rn = stack ? 13 : Bits(inst, 16, 19);
        const bool single = Bits(inst, 8, 8) == 0;
        const u32 regs = single ? Bits(inst, 0, 7) : Bits(inst, 1, 7);
        return rn != 15 && regs != 0;
    }

    /**
     * kT1 = puntero directo a 'words' palabras seguidas desde 'start', o un
     * salto a 'slow' si no se puede: desalineada, en dos paginas o sin
     * puntero de pagina (hardware). Pisa kRm y los flags.
     */
    void DirectSpan(u32 start, u32 words, std::vector<u32>& slow) {
        e.Emit(0xE3100003u | (start << 16)); // TST start, #3
        slow.push_back(e.BranchPlaceholder(kCondNe));
        if (words > 1) {
            AddSmall(kRm, start, 4 * static_cast<s32>(words) - 4);   // ultima palabra
            e.Emit(0xE0200000u | (kRm << 16) | (kRm << 12) | start); // EOR kRm, kRm, start
            e.Emit(0xE1B00620u | (kRm << 12) | kRm);                 // LSRS kRm, kRm, #12
            slow.push_back(e.BranchPlaceholder(kCondNe));            // dos paginas
        }
        e.LsrImm(kRm, start, 12);
        e.LdrRegLsl2(kT1, kPages, kRm);
        e.CmpImm0(kT1);
        slow.push_back(e.BranchPlaceholder(kCondEq)); // sin puntero
        e.Ubfx(kRm, start, 0, 12);
        e.AddReg(kT1, kT1, kRm);
    }

    /**
     * VPUSH/VPOP/VSTM/VLDM EN LINEA (0.1.8.7). crash.txt de 0.1.8.6 en el 3D
     * de Zafiro Alfa: "lento" 8 ms por vblank con ~6.500 llamadas, casi todas
     * estas (los d8-d15 que guarda y recupera cada funcion con coma flotante).
     *
     * Camino rapido con EXACTAMENTE las cuentas de VfpTransfer: la primera
     * direccion y el base nuevo salen del mismo inmediato (imm32, que en
     * FSTMX/FLDMX lleva una palabra de mas que no se copia), y las palabras van
     * seguidas en ExtReg (s(d+i), o d(d+i) en ExtReg[2(d+i)] y [2(d+i)+1]).
     * Solo si el tramo entero esta alineado a 4 y cae en UNA pagina con
     * puntero; si no, la funcion de siempre, en un camino frio. La escritura
     * del base, despues de los accesos, como alli. Desde 0.2.1.0 sin volcar
     * la cache: el base se lee de su hueco (los accesos no tocan registros ARM).
     */
    void VfpTransferFast(u32 inst, u32 pc, int index) {
        const bool push = index == kVfpVpush;
        const bool pop = index == kVfpVpop;
        const bool single = Bits(inst, 8, 8) == 0;
        const u32 d = single ? ((Bits(inst, 12, 15) << 1) | Bits(inst, 22, 22))
                             : (Bits(inst, 12, 15) | (Bits(inst, 22, 22) << 4));
        const s32 imm32 = static_cast<s32>(Bits(inst, 0, 7) << 2);
        const u32 regs = single ? Bits(inst, 0, 7) : Bits(inst, 1, 7);
        const u32 words = single ? regs : regs * 2;
        const u32 first_ext = single ? d : d * 2;
        const u32 rn = (push || pop) ? 13 : Bits(inst, 16, 19);
        const bool add = Bits(inst, 23, 23) == 1;
        const bool load = pop || index == kVfpVldm;
        const bool writeback = push || pop || Bits(inst, 21, 21) == 1;
        const s32 first = (push || (!pop && !add)) ? -imm32 : 0;
        const s32 delta = (push || (!pop && !add)) ? -imm32 : imm32;
        const bool keep_flags = flags_live_after;
        const std::array<bool, 16> written_before = cache_written;

        if (keep_flags) {
            e.Mrs(kFlags);
        }
        const u32 base = OperandHost(kRn, rn, pc);
        const u32 start = first != 0 ? kRs : base;
        AddSmall(start, base, first);
        std::vector<u32> slow;
        DirectSpan(start, words, slow);
        for (u32 k = 0; k < words; k++) {
            if (load) {
                e.LdrImm(kRd, kT1, 4 * k);
                e.StrImm(kRd, kCpu, ExtOffset(first_ext + k));
            } else {
                e.LdrImm(kRd, kCpu, ExtOffset(first_ext + k));
                e.StrImm(kRd, kT1, 4 * k);
            }
        }
        if (writeback) {
            AddSmall(kRm, base, delta);
            StoreGuest(kRm, rn);
        }
        const u32 back = e.Position();
        if (keep_flags) {
            e.MsrFlags(kFlags);
        }
        DeferCold(std::move(slow), written_before, [this, inst, pc, index, keep_flags, back] {
            FlushCachedRegs();
            if (keep_flags) {
                e.MovReg(kT0, kFlags);
            }
            e.Mov32(R0, pc);
            e.StrImm(R0, kCpu, RegOffset(15));
            e.MovReg(R0, kCpu);
            e.Mov32(R1, inst);
            e.Mov32(R2, static_cast<u32>(index));
            e.Call(reinterpret_cast<const void*>(&HelperVfpTransfer));
            if (keep_flags) {
                e.MovReg(kFlags, kT0);
            }
            ReloadCachedRegs();
            e.BranchTo(kAlways, back);
        });
    }

    /// LDM/STM: en linea (BlockTransferFast); la variante de comprobacion, por
    /// la misma logica que el interprete con el diario. El PC del juego se deja
    /// escrito antes (STM con el PC lo guarda como PC+8).
    void BlockTransferCall(u32 inst, u32 pc) {
        if (!check_mode) {
            BlockTransferFast(inst, pc);
            return;
        }
        FlushCachedRegs();
        e.Mrs(kFlags);
        e.MovReg(kT0, kFlags);
        e.Mov32(R0, pc);
        e.StrImm(R0, kCpu, RegOffset(15));
        e.MovReg(R0, kCpu);
        e.Mov32(R1, inst);
        e.Call(reinterpret_cast<const void*>(&CheckBlockTransfer));
        e.MsrFlags(kT0);
        ReloadCachedRegs();
    }

    /**
     * LDM/STM EN LINEA (0.1.7.0). Medido en 0.1.6.3: ~15.000 llamadas lentas
     * por fotograma, casi todas LDM/STM (los push/pop de cada funcion).
     *
     * Camino rapido, con EXACTAMENTE la cuenta de BlockTransfer: si el tramo
     * de memoria entero esta alineado a 4 y cae en UNA pagina con puntero (RAM
     * normal, sin efectos), se copian las palabras directamente, en el mismo
     * orden que el interprete: la escritura del base ANTES de los accesos
     * (en un LDM que carga el propio base gana lo cargado), STM guarda el base
     * ORIGINAL si esta en la lista y el PC como PC+8, y un LDM que carga el PC
     * pasa a Thumb si el bit 0 viene puesto. Cualquier otra cosa (dos paginas,
     * hardware, desalineado) va a la funcion de siempre, en un camino frio.
     *
     * 0.2.1.0: SIN volcar ni recargar la cache. Los registros cacheados se
     * guardan y se cargan directamente desde y hacia su hueco; antes cada
     * push/pop volcaba los cinco huecos, los releia de cpu->Reg y los
     * recargaba al acabar. El camino frio vuelca lo escrito ANTES de la
     * instruccion: se entra en el antes de tocar nada.
     */
    void BlockTransferFast(u32 inst, u32 pc) {
        const u32 rn = Bits(inst, 16, 19);
        const u32 list = inst & 0xFFFFu;
        const u32 count = static_cast<u32>(std::popcount(list));
        const bool pre = Bits(inst, 24, 24) == 1;
        const bool up = Bits(inst, 23, 23) == 1;
        const bool writeback = Bits(inst, 21, 21) == 1;
        const bool load = Bits(inst, 20, 20) == 1;
        // Desde el base hasta la primera palabra, y el base nuevo.
        const s32 first = up ? (pre ? 4 : 0) : (pre ? -4 * s32(count) : -4 * s32(count) + 4);
        const s32 delta = up ? 4 * s32(count) : -4 * s32(count);
        const bool keep_flags = flags_live_after;
        const std::array<bool, 16> written_before = cache_written;

        if (keep_flags) {
            e.Mrs(kFlags);
        }
        // El base ORIGINAL. Con escritura del base, en kRn: su hueco recibe el
        // nuevo antes de los accesos, y STM guarda el original.
        u32 base = kRn;
        if (writeback) {
            LoadGuest(kRn, rn, pc);
        } else {
            base = OperandHost(kRn, rn, pc);
        }
        const u32 start = first != 0 ? kRs : base;
        AddSmall(start, base, first);
        std::vector<u32> slow;
        DirectSpan(start, count, slow);
        if (writeback) {
            AddSmall(kRm, base, delta);
            StoreGuest(kRm, rn);
        }
        u32 slot = 0;
        for (u32 i = 0; i < 16; i++) {
            if (((list >> i) & 1) == 0) {
                continue;
            }
            const u32 offset = slot * 4;
            slot++;
            if (load) {
                if (i == 15) {
                    e.LdrImm(kRd, kT1, offset);
                    e.Ubfx(kRm, kRd, 0, 1);
                    e.StrImm(kRm, kCpu, g_offsets.t);
                    e.Emit(0xE3C00001u | (kRd << 16) | (kRd << 12)); // BIC kRd, kRd, #1
                    e.StrImm(kRd, kCpu, RegOffset(15));
                    continue;
                }
                const s8 cached = CacheHost(i);
                if (cached >= 0) {
                    e.LdrImm(static_cast<u32>(cached), kT1, offset);
                    cache_written[i] = true;
                } else {
                    e.LdrImm(kRd, kT1, offset);
                    e.StrImm(kRd, kCpu, RegOffset(i));
                }
                continue;
            }
            u32 value = base; // el base ORIGINAL si esta en la lista
            if (i == 15) {
                e.Mov32(kRd, pc + 8);
                value = kRd;
            } else if (i != rn) {
                value = OperandHost(kRd, i, pc);
            }
            e.StrImm(value, kT1, offset);
        }
        const u32 back = e.Position();
        if (keep_flags) {
            e.MsrFlags(kFlags);
        }
        DeferCold(std::move(slow), written_before, [this, inst, pc, keep_flags, back] {
            FlushCachedRegs();
            if (keep_flags) {
                e.MovReg(kT0, kFlags);
            }
            e.Mov32(R0, pc);
            e.StrImm(R0, kCpu, RegOffset(15));
            e.MovReg(R0, kCpu);
            e.Mov32(R1, inst);
            e.Call(reinterpret_cast<const void*>(&HelperBlockTransfer));
            if (keep_flags) {
                e.MovReg(kFlags, kT0);
            }
            ReloadCachedRegs();
            e.BranchTo(kAlways, back);
        });
    }

    /// Con enlaces en el codigo generado (nunca en la variante de comprobacion).
    bool Linking() const {
        return !check_mode && direct_link.load(std::memory_order_relaxed) != 0;
    }

    /**
     * Salida a un destino FIJO (B, BL, el camino no tomado de un salto
     * condicional, el final de pagina). Con enlace, Reg[15] solo se escribe si
     * el enlace falla (0.2.1.0): el bloque destino no lo lee -- los caminos
     * lentos y las llamadas que lo necesitan escriben el suyo, y cada salida
     * deja el de su destino --, y eran tres instrucciones en cada salida.
     */
    void ExitDirect(u32 target_pc) {
        if (Linking()) {
            EmitLink(LinkFor(target_pc), target_pc);
            return;
        }
        e.Mov32(kT0, target_pc);
        e.StrImm(kT0, kCpu, RegOffset(15));
        to_epilogue.push_back(e.BranchPlaceholder(kAlways));
    }

    /// Salida con Reg[15] (y TFlag) ya escritos (BX, LDM con el PC...).
    void ExitIndirect() {
        if (Linking()) {
            EmitLink(nullptr, 0);
            return;
        }
        to_epilogue.push_back(e.BranchPlaceholder(kAlways));
    }

    /**
     * ENLACE AL SIGUIENTE BLOQUE, EN LINEA (0.1.5.7). Ver LinkContext.
     *
     * 'target' es el LinkInfo del destino si el destino es FIJO (va como
     * constante en el codigo) y 'target_pc' su direccion. Con nullptr el salto
     * es INDIRECTO (BX, LDM con el PC, "mov pc, lr", "ldr pc, [...]"): se lee
     * Reg[15] y se busca en g_fast, la tabla rapida de TryRun, con su misma
     * clave; si el hueco es de otro PC, no se enlaza.
     *
     * Las condiciones son EXACTAMENTE las del bucle de TryRun (y las del
     * despacho del interprete). Si alguna falla, un camino frio repone los
     * flags, escribe el PC del destino fijo y sale sin volver a volcar la
     * cache (exit_noflush). Los flags del juego se guardan en kT0 (r10) porque
     * aqui se usan CMP/SUBS; r0-r3 y r6-r9 estan libres (la cache se vuelca
     * al empezar).
     */
    void EmitLink(const LinkInfo* target, u32 target_pc) {
        std::vector<u32> fails;
        FlushCachedRegs();
        e.Mrs(kT0);
        /**
         * SIN LA PRUEBA DE INTERRUPCION (0.1.8.2). Aqui se miraba lo mismo que el
         * despacho del interprete (!NirqSig && !(Cpsr & 0x80)): cinco
         * instrucciones con dos cargas en cada enlace. NirqSig solo se escribe
         * al crear el estado (NirqSig = HIGH, armstate.cpp) -- el kernel del 3DS
         * es HLE y no hay IRQ del ARM --, el despacho ya lo comprueba antes de
         * entrar en TryRun, y nada de lo que corre dentro del codigo generado
         * (ni las funciones lentas) toca NirqSig ni el bit I (MSR y CPS van al
         * interprete). O sea que dentro de una cadena de enlaces no puede
         * cambiar: la prueba nunca fallaba.
         */
        if (target == nullptr) {
            // Indirecto: tiene que seguir en ARM y con el PC alineado (el
            // despacho alinearia; aqui simplemente no se enlaza).
            e.LdrImm(R1, kCpu, g_offsets.t);
            e.CmpImm0(R1);
            fails.push_back(e.BranchPlaceholder(kCondNe));
            e.LdrImm(R1, kCpu, RegOffset(15));
            e.Emit(0xE3100003u | (R1 << 16)); // TST r1, #3
            fails.push_back(e.BranchPlaceholder(kCondNe));
            // r3 = &g_fast[(pc >> 2) & (kFastSlots - 1)]
            e.Ubfx(R2, R1, 2, kFastSlotBits);
            e.Mov32(R3, static_cast<u32>(reinterpret_cast<uintptr_t>(g_fast.data())));
            e.Emit(0xE0800000u | (R3 << 16) | (R3 << 12) | (3u << 7) | R2); // ADD r3, r3, r2, LSL #3
            e.LdrImm(R2, R3, 0);                                           // slot.pc
            e.Emit(0xE1500000u | (R2 << 16) | R1);                         // CMP r2, r1
            fails.push_back(e.BranchPlaceholder(kCondNe));
            e.LdrImm(R0, R3, 4); // slot.block
            if (g_link_in_block != 0) {
                e.Mov32(R2, g_link_in_block);
                e.AddReg(R0, R0, R2);
            }
        } else {
            // Directo: el LinkInfo del destino es fijo (los elementos de
            // g_blocks no se mueven): MOVW/MOVT, sin literal ni salto (0.1.8.1).
            e.Mov32(R0, static_cast<u32>(reinterpret_cast<uintptr_t>(target)));
        }
        /**
         * r0 = LinkInfo* del destino. Enlazable? (0.1.8.1) entry ya implica que
         * el bloque ha pasado sus kFullChecks comprobaciones (TryRun no lo
         * publica antes, ver PublishLinkEntry), asi que aqui no se mira ni se
         * incrementa 'runs'. Las comprobaciones por muestreo siguen en las
         * entradas por TryRun.
         */
        e.LdrImm(R1, R0, kLinkEntry);
        e.CmpImm0(R1);
        fails.push_back(e.BranchPlaceholder(kCondEq));
        // Presupuesto y enlaces de una vez: LDRD kRn (budget), kRm (hops).
        e.LdrdImm(kRn, kCpu, g_offsets.link);
        e.LdrImm(R2, R0, kLinkCount);
        e.Emit(0xE0500000u | (kRn << 16) | (kRn << 12) | R2); // SUBS kRn, kRn, r2
        fails.push_back(e.BranchPlaceholder(kCondLo));
        e.Emit(0xE2800001u | (kRm << 16) | (kRm << 12)); // ADD kRm, kRm, #1
        e.StrdImm(kRn, kCpu, g_offsets.link);
        e.MsrFlags(kT0);
        e.Emit(0xE12FFF10u | R1); // BX r1 -> entrada del destino
        const bool store_pc = target != nullptr;
        DeferCold(std::move(fails), [this, store_pc, target_pc] {
            e.MsrFlags(kT0);
            if (store_pc) {
                e.Mov32(kRd, target_pc);
                e.StrImm(kRd, kCpu, RegOffset(15));
            }
            e.BranchTo(kAlways, exit_noflush);
        });
    }

    /// El LinkInfo del bloque de 'pc', creando su entrada si hace falta (un
    /// bloque New no se enlaza hasta que se compile y se le ponga entry).
    static const LinkInfo* LinkFor(u32 pc) {
        Block& target = g_blocks[pc];
        target.pc = pc;
        return &target.link;
    }

    /// Una constante a un registro del juego, directa a su hueco si esta
    /// cacheado (0.2.1.0: el LR de BL).
    void StoreGuestConst(u32 guest, u32 value) {
        const s8 cached = CacheHost(guest);
        if (cached >= 0) {
            e.Mov32(static_cast<u32>(cached), value);
            cache_written[guest] = true;
            return;
        }
        e.Mov32(kT0, value);
        e.StrImm(kT0, kCpu, RegOffset(guest));
    }

    /**
     * Instrucciones que terminan el bloque. Dejan en Reg[15] (y en TFlag, si
     * cambian de modo) lo mismo que el interprete: el destino si la condicion
     * se cumple y la instruccion siguiente si no. Cada camino acaba en una
     * salida (ExitDirect o ExitIndirect), que intenta enlazar.
     */
    void Terminator(u32 inst, u32 pc, const Decoded& decoded, u32 cond) {
        // kT0/kT1 en vez de R0-R2: R0-R3 y lr son los huecos de la cache de
        // registros. Un Mov32 ahi pisaria un valor cacheado y el Epilogue
        // (que si vuelca) escribiria basura sobre los registros del juego.
        u32 skip = 0;
        if (cond != kAlways) {
            skip = e.BranchPlaceholder(cond ^ 1u);
        }
        switch (decoded.kind) {
        case Kind::Branch: {
            const u32 offset24 = inst & 0x00FFFFFFu;
            const s32 offset = static_cast<s32>(offset24 << 8) >> 6; // extiende signo, x4
            const u32 target = pc + 8 + static_cast<u32>(offset);
            if (Bits(inst, 24, 24) == 1) {
                StoreGuestConst(14, pc + 4);
            }
            ExitDirect(target);
            break;
        }
        case Kind::BranchLinkExchangeImm: {
            // Como BLX_INST: LR = pc + 4 (desde ARM, sin bit 0), Thumb, y el
            // destino con el bit H como media palabra.
            const u32 offset24 = inst & 0x00FFFFFFu;
            const s32 offset = static_cast<s32>(offset24 << 8) >> 6;
            const u32 target = pc + 8 + static_cast<u32>(offset) + (Bits(inst, 24, 24) << 1);
            StoreGuestConst(14, pc + 4);
            e.Mov32(kT1, 1);
            e.StrImm(kT1, kCpu, g_offsets.t);
            e.Mov32(kT0, target);
            StoreGuest(kT0, 15);
            ExitIndirect();
            break;
        }
        case Kind::BranchExchange: {
            const u32 rm = Bits(inst, 0, 3);
            const bool link = Bits(inst, 5, 5) == 1;
            LoadGuest(kT0, rm, pc);
            if (link) {
                e.Mov32(kT1, pc + 4);
                StoreGuest(kT1, 14);
            }
            // TFlag = destino & 1; PC = destino & ~1 (igual que BX_INST).
            e.Ubfx(kT1, kT0, 0, 1);
            e.StrImm(kT1, kCpu, g_offsets.t);
            e.Emit(0xE3C00001u | (kT0 << 16) | (kT0 << 12)); // BIC kT0, kT0, #1
            StoreGuest(kT0, 15);
            ExitIndirect();
            break;
        }
        case Kind::BlockTransfer:
            // LDM con el PC: pone Reg[15] y TFlag.
            BlockTransferCall(inst, pc);
            ExitIndirect();
            break;
        case Kind::DataProcessing:
            // "mov pc, lr" y compania: el resultado va a Reg[15] (Rd = 15), sin
            // flags (el bit S esta excluido) y sin cambio de modo, igual que
            // MOV_INST y demas en el interprete.
            DataProcessing(inst, pc);
            ExitIndirect();
            break;
        case Kind::LoadStore:
            // "ldr pc, [...]": LoadStore pone Reg[15] y TFlag.
            LoadStore(inst, pc);
            ExitIndirect();
            break;
        default:
            to_epilogue.push_back(e.BranchPlaceholder(kAlways));
            break;
        }
        if (cond != kAlways) {
            // El camino no tomado: PC = siguiente.
            e.PatchBranch(skip, e.Position());
            ExitDirect(pc + 4);
        }
    }
};

// ---------------------------------------------------------------------------
// Analisis, emision y cache de bloques
// ---------------------------------------------------------------------------

struct PendingCheck {
    bool active = false;
    ARMul_State* cpu = nullptr;
    Block* block = nullptr;
    std::array<u32, 16> regs{};
    /// Registros VFP (0.1.5.5): el JIT ya los escribe (VLDR, VMOV). Sin
    /// compararlos, un indice equivocado pasaba la comprobacion sin ruido.
    std::array<u32, 64> ext{};
    /// Registros de sistema VFP (FPSCR...), 0.1.6.1: la aritmetica VFP los toca.
    std::array<u32, VFP_SYSTEM_REGISTER_COUNT> vfp{};
    u32 n = 0, z = 0, c = 0, v = 0, t = 0;
    u32 journal_size = 0;
    std::array<JournalEntry, kJournalEntries> journal{};
};
PendingCheck g_pending;

/// Tramos [inicio, fin) invalidados desde la ultima vez (ver InvalidateRange).
std::vector<std::pair<u32, u32>> g_invalid_ranges;

void ResetAll() {
    g_invalid_ranges.clear();
    g_blocks.clear();
    g_fast.fill(FastSlot{});
    g_code_used_words = 0;
    g_code_blocks = 0;
    g_pending.active = false;
    g_flush_requested = false;
    g_link_withdrawn = 0;
}

/**
 * Las instrucciones del bloque que empieza en block.pc, con la MISMA extension
 * que el del interprete (InterpreterTranslateBlock): hasta la primera que
 * termina bloque o hasta el final de la pagina. False si alguna no se sabe
 * compilar: entonces el bloque entero es del interprete.
 *
 * Si la cache de registros esta encendida (0.1.5.2, 4.3), aqui se decide
 * QUE registros se cachean: se cuentan las lecturas de cada registro del
 * juego (excluido el PC, que es especial) y se quedan los tres mas leidos en
 * r0-r2. Los que solo se escriben o se leen una vez no merecen hueco. Con el
 * interruptor apagado el mapa se queda entero a -1 y el codigo generado es el
 * de 0.1.5.1 palabra por palabra.
 */
bool Analyze(ARMul_State* cpu, Block& block) {
    if ((cpu->Cpsr & (1u << 9)) != 0) {
        block.reject = kRejectOther;
        return false; // modo big-endian: no lo usa ningun juego, no se compila
    }
    block.words.clear();
    block.cache_map.fill(-1);
    block.uses_cache = false;
    u32 pc = block.pc;
    bool ended = false;
    for (u32 i = 0; i < kMaxBlockInstructions; i++) {
        // La misma lectura que usa el interprete para traducir.
        const u32 inst = cpu->memory.Read32(pc & 0xFFFFFFFCu);
        const Decoded decoded = Classify(inst);
        if (decoded.kind == Kind::Unsupported) {
            block.words.clear();
            block.reject = decoded.reject;
            block.reject_word = inst;
            return false;
        }
        block.words.push_back(inst);
        if (decoded.terminator) {
            ended = true;
            break;
        }
        pc += 4;
        if ((pc & 0xFFF) == 0) {
            ended = true; // fin de pagina, como END_OF_PAGE en el interprete
            break;
        }
    }
    if (!ended) {
        block.words.clear();
        block.reject = kRejectLong;
        return false;
    }
    if (reg_cache.load(std::memory_order_relaxed) == 0) {
        return true; // interruptor apagado: cache_map ya esta a -1
    }
    /**
     * Que registros del juego se cachean (0.1.5.7: todos los usos, no solo las
     * lecturas de ALU, y cinco huecos en vez de tres).
     *
     * Se cuentan las veces que el codigo generado cargaria o guardaria cada
     * registro (LoadGuest/StoreGuest): operandos y destinos de ALU y
     * multiplicaciones, y base, dato y desplazamiento de los accesos a
     * memoria. Cachear cuesta una carga al entrar y un guardado al salir (y un
     * vuelco y una recarga alrededor de cada llamada lenta), asi que solo se
     * cachean los que se usan al menos dos veces. El PC no se cachea nunca.
     */
    std::array<u32, 16> uses{};
    const auto use = [&uses](u32 reg) {
        if (reg < 15) {
            uses[reg]++;
        }
    };
    for (const u32 inst : block.words) {
        const Decoded decoded = Classify(inst);
        switch (decoded.kind) {
        case Kind::DataProcessing: {
            const u32 opcode = Bits(inst, 21, 24);
            if (opcode != 13 && opcode != 15) {
                use(Bits(inst, 16, 19)); // Rn
            }
            if (!(opcode >= 8 && opcode <= 11)) {
                use(Bits(inst, 12, 15)); // Rd
            }
            if (Bits(inst, 25, 25) == 0) {
                use(Bits(inst, 0, 3)); // Rm
                if (Bits(inst, 4, 4) == 1) {
                    use(Bits(inst, 8, 11)); // Rs
                }
            }
            break;
        }
        case Kind::Multiply:
            use(Bits(inst, 0, 3));
            use(Bits(inst, 8, 11));
            use(Bits(inst, 16, 19));
            if (Bits(inst, 21, 21) == 1) {
                use(Bits(inst, 12, 15));
            }
            break;
        case Kind::MultiplyLong:
            use(Bits(inst, 0, 3));
            use(Bits(inst, 8, 11));
            use(Bits(inst, 12, 15));
            use(Bits(inst, 16, 19));
            break;
        case Kind::NativeReg:
            use(Bits(inst, 0, 3));
            use(Bits(inst, 12, 15));
            if (Bits(inst, 16, 19) != 15) {
                use(Bits(inst, 16, 19));
            }
            break;
        case Kind::Clz:
            use(Bits(inst, 0, 3));
            use(Bits(inst, 12, 15));
            break;
        case Kind::LoadStore:
            use(Bits(inst, 16, 19));
            use(Bits(inst, 12, 15));
            if (Bits(inst, 25, 25) == 1) {
                use(Bits(inst, 0, 3));
            }
            break;
        case Kind::LoadStoreExtra:
        case Kind::LoadStoreDual:
            use(Bits(inst, 16, 19));
            use(Bits(inst, 12, 15));
            if (decoded.kind == Kind::LoadStoreDual) {
                use(Bits(inst, 12, 15) + 1);
            }
            if (Bits(inst, 22, 22) == 0) {
                use(Bits(inst, 0, 3));
            }
            break;
        case Kind::VfpLoadStore:
            use(Bits(inst, 16, 19));
            break;
        case Kind::Exclusive:
            use(Bits(inst, 16, 19));
            use(Bits(inst, 12, 15));
            if ((Bits(inst, 20, 27) & 1) == 0) {
                use(Bits(inst, 0, 3));
            }
            break;
        case Kind::VfpMove:
            use(Bits(inst, 12, 15));
            break;
        default:
            break;
        }
    }
    u32 placed = 0;
    for (u32 pass = 0; pass < kCacheHosts.size(); pass++) {
        u32 best = 16;
        u32 best_count = 1; // al menos dos usos: uno solo no compensa
        for (u32 g = 0; g < 15; g++) {
            if (block.cache_map[g] >= 0) {
                continue;
            }
            if (uses[g] > best_count) {
                best_count = uses[g];
                best = g;
            }
        }
        if (best < 15) {
            block.cache_map[best] = static_cast<s8>(kCacheHosts[placed]);
            placed++;
        }
    }
    block.uses_cache = placed > 0;
    return true;
}

/**
 * VIDA DE LOS FLAGS (0.1.5.7): para cada instruccion del bloque, si alguien
 * puede leer los flags del juego (N, Z, C, V) DESPUES de ella.
 *
 * Se recorre el bloque hacia atras. Al final del bloque los flags estan
 * vivos (salen a cpu y los ve el bloque siguiente). Hacia atras:
 *   - una instruccion que LEE flags los hace vivos: cualquier condicion que
 *     no sea AL, ADC/SBC/RSC (leen el acarreo) y un operando RRX (en ALU o
 *     como desplazamiento de LDR/STR);
 *   - una instruccion que los ESCRIBE TODOS, sin condicion y sin leerlos,
 *     los mata: las aritmeticas con S (ADD, SUB, RSB, CMP, CMN). Las logicas
 *     con S (MOV, AND, TST...) no: dejan V (y a veces C) como estaban, asi
 *     que lo que habia antes puede seguir viendose;
 *   - las demas no cambian nada.
 * Es conservador: ante la duda, vivos. Y lo usa tambien la variante de
 * comprobacion, asi que un fallo aqui saldria como DIFERENCIA.
 */
std::vector<bool> ComputeFlagsLiveAfter(const std::vector<u32>& words) {
    std::vector<bool> live_after(words.size(), true);
    bool live = true;
    for (std::size_t i = words.size(); i-- > 0;) {
        live_after[i] = live;
        const u32 inst = words[i];
        bool reads = (inst >> 28) != kAlways;
        bool writes_all = false;
        switch (Classify(inst).kind) {
        case Kind::DataProcessing: {
            const u32 opcode = Bits(inst, 21, 24);
            const bool s = Bits(inst, 20, 20) == 1;
            if (opcode >= 5 && opcode <= 7) {
                reads = true; // ADC, SBC, RSC
            }
            // RRX: registro desplazado por inmediato, tipo ROR, cantidad 0.
            if (Bits(inst, 25, 25) == 0 && Bits(inst, 4, 4) == 0 && Bits(inst, 5, 6) == 3 &&
                Bits(inst, 7, 11) == 0) {
                reads = true;
            }
            writes_all = s && (opcode == 2 || opcode == 3 || opcode == 4 || opcode == 10 ||
                               opcode == 11);
            break;
        }
        case Kind::LoadStore:
            if (Bits(inst, 25, 25) == 1 && Bits(inst, 5, 6) == 3 && Bits(inst, 7, 11) == 0) {
                reads = true; // desplazamiento RRX
            }
            break;
        default:
            break;
        }
        if (reads) {
            live = true;
        } else if (writes_all) {
            live = false;
        }
    }
    return live_after;
}

/// Genera el codigo del bloque (normal o de comprobacion). nullptr si no cabe:
/// entonces se pide vaciar y el bloque se queda en el interprete esta vez.
/// block.link.entry se rellena en el camino normal (0.1.5.7); en comprobacion
/// no hace falta (esos bloques no encadenan).
BlockFn EmitBlock(Block& block, bool check_mode) {
    const u32 capacity = kCodeBytes / 4 - g_code_used_words;
    const std::lock_guard vm_lock{Common::vita_vm_domain_mutex};
    const int open_rc = sceKernelOpenVMDomain();
    if (open_rc < 0) {
        static bool noted = false;
        if (!noted) {
            noted = true;
            Common::VitaNote("jit arm",
                             fmt::format("no se puede abrir el dominio VM ({:#x})",
                                         static_cast<u32>(open_rc))
                                 .c_str());
        }
        return nullptr;
    }
    u32* const start = g_code + g_code_used_words;
    Emitter e{start, capacity};
    Compiler compiler{e, check_mode};
    compiler.SetCacheMap(block.cache_map);
    compiler.Prologue();
    const u32 chain_entry = compiler.ChainEntryPosition();
    compiler.BodyEnter();
    u32 pc = block.pc;
    bool terminated = false;
    // Vida de los flags del juego (0.1.5.7): ver ComputeFlagsLiveAfter.
    const std::vector<bool> flags_live = ComputeFlagsLiveAfter(block.words);
    for (std::size_t i = 0; i < block.words.size(); i++) {
        const u32 inst = block.words[i];
        const Decoded decoded = Classify(inst);
        // La aritmetica VFP seguida, con un solo cambio de FPSCR (0.2.1.0).
        if (decoded.kind == Kind::VfpCdp && Compiler::VfpNative(inst)) {
            std::size_t end = i + 1;
            while (end < block.words.size() && Classify(block.words[end]).kind == Kind::VfpCdp &&
                   Compiler::VfpNative(block.words[end])) {
                end++;
            }
            // Los flags se guardan si se leen despues o si alguna lleva condicion.
            bool keep_flags = flags_live[end - 1];
            for (std::size_t k = i; k < end; k++) {
                keep_flags = keep_flags || (block.words[k] >> 28) != kAlways;
            }
            const u32 count = static_cast<u32>(end - i);
            compiler.VfpCdpRun(&block.words[i], count, keep_flags, true);
            terminated = false;
            pc += 4 * count;
            i = end - 1;
            continue;
        }
        compiler.SetFlagsLiveAfter(flags_live[i]);
        compiler.Instruction(inst, pc, decoded);
        terminated = decoded.terminator;
        pc += 4;
    }
    compiler.SetFlagsLiveAfter(true);
    if (!terminated) {
        compiler.FallThrough(pc);
    }
    compiler.Epilogue();
    compiler.EmitColdStubs();
    if (e.Overflowed()) {
        sceKernelCloseVMDomain();
        g_flush_requested = true;
        return nullptr;
    }
    // Sin esto la cache de instrucciones puede tener lo que hubiera antes en
    // esa memoria y se ejecuta basura.
    sceKernelSyncVMDomain(g_code_block, start, e.Position() * 4);
    sceKernelCloseVMDomain();
    g_code_used_words += e.Position();
    g_code_blocks++;
    if (!check_mode) {
        // Punto de entrada de los enlaces (0.1.5.7): pila, kCpu, kPages y
        // flags ya los pone el bloque origen; aqui empieza BodyEnter. NO se
        // publica todavia en link.entry: eso lo hace TryRun cuando el bloque
        // ha pasado sus comprobaciones iniciales (0.1.8.1).
        block.chain_entry = static_cast<u32>(reinterpret_cast<uintptr_t>(start + chain_entry));
    }
    return reinterpret_cast<BlockFn>(start);
}

void NoteMismatch(const Block& block, const std::string& what) {
    Common::VitaNote("jit arm",
                     fmt::format("DIFERENCIA en el bloque {:#010x} ({} instrucciones): {} -- el "
                                 "bloque vuelve al interprete",
                                 block.pc, block.words.size(), what)
                         .c_str());
    for (std::size_t base = 0; base < block.words.size() && base < 32; base += 8) {
        std::string line = fmt::format("{:#010x}:", block.pc + static_cast<u32>(base) * 4);
        for (std::size_t i = base; i < block.words.size() && i < base + 8; i++) {
            line += fmt::format(" {:08x}", block.words[i]);
        }
        Common::VitaNote("jit arm", line.c_str());
    }
}

/**
 * Comprobacion (ver la cabecera): variante sin escrituras, estado restaurado y
 * el bloque se deja al interprete; la comparacion se hace en el siguiente
 * despacho, en CompletePendingCheck.
 */
void StartCheck(ARMul_State* cpu, Block& block, u8* const* pages) {
    if (block.check_code == nullptr) {
        block.check_code = EmitBlock(block, true);
        if (block.check_code == nullptr) {
            return;
        }
    }
    const std::array<u32, 16> regs = cpu->Reg;
    // La variante de comprobacion ESCRIBE ExtReg (VLDR, VMOV): hay que
    // restaurarlo como los registros, o el interprete partiria de lo que dejo
    // el JIT (0.1.5.5).
    const std::array<u32, 64> ext = cpu->ExtReg;
    const std::array<u32, VFP_SYSTEM_REGISTER_COUNT> vfp = cpu->VFP;
    const u32 n = cpu->NFlag, z = cpu->ZFlag, c = cpu->CFlag, v = cpu->VFlag, t = cpu->TFlag;

    g_check.cpu = cpu;
    g_check.pages = pages;
    g_check.aborted = false;
    g_check.journal_size = 0;
    block.check_code(cpu, pages);

    if (!g_check.aborted) {
        g_pending.active = true;
        g_pending.cpu = cpu;
        g_pending.block = &block;
        g_pending.regs = cpu->Reg;
        g_pending.ext = cpu->ExtReg;
        g_pending.vfp = cpu->VFP;
        g_pending.n = cpu->NFlag;
        g_pending.z = cpu->ZFlag;
        g_pending.c = cpu->CFlag;
        g_pending.v = cpu->VFlag;
        g_pending.t = cpu->TFlag;
        g_pending.journal_size = g_check.journal_size;
        g_pending.journal = g_check.journal;
        Common::FrameStats::jit_checks.fetch_add(1, std::memory_order_relaxed);
    }

    cpu->Reg = regs;
    cpu->ExtReg = ext;
    cpu->VFP = vfp;
    cpu->NFlag = n;
    cpu->ZFlag = z;
    cpu->CFlag = c;
    cpu->VFlag = v;
    cpu->TFlag = t;
}

} // Anonymous namespace

void CountRejectWord(u32 word);

namespace {

/**
 * El bloque compilado para 'pc', compilandolo si toca, o nullptr si ese
 * despacho es del interprete (y entonces anota por que). Cuenta el despacho.
 */
Block* Acquire(ARMul_State* cpu, u32 pc) {
    g_local.dispatches++;
    FastSlot& slot = g_fast[(pc >> 2) & (kFastSlots - 1)];
    Block* block;
    if (slot.pc == pc) [[likely]] {
        block = slot.block;
    } else {
        // Los punteros a elementos de un unordered_map sobreviven al rehash.
        block = &g_blocks[pc];
        block->pc = pc;
        slot.pc = pc;
        slot.block = block;
    }

    switch (block->state) {
    case BlockState::Compiled:
        return block;
    case BlockState::Rejected:
        g_local.rejects[block->reject]++;
        if (block->reject == kRejectOther) {
            CountRejectWord(block->reject_word);
        }
        return nullptr;
    case BlockState::Blacklisted:
        g_local.rejects[kRejectOther]++;
        return nullptr;
    case BlockState::New:
        break;
    }
    if (++block->visits < kCompileAfterVisits) {
        return nullptr;
    }
    const ScopedMicros timer{g_compile_us};
    if (!Analyze(cpu, *block)) {
        block->state = BlockState::Rejected;
        g_local.rejects[block->reject]++;
        Common::FrameStats::jit_rejected.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    block->link.count = static_cast<u32>(block->words.size());
    block->code = EmitBlock(*block, false);
    if (block->code == nullptr) {
        // Sin sitio: se vacia en la siguiente llamada y se vuelve a intentar.
        block->visits = 0;
        return nullptr;
    }
    block->state = BlockState::Compiled;
    Common::FrameStats::jit_blocks.fetch_add(1, std::memory_order_relaxed);
    return block;
}

/// El bloque de 'pc' si ya esta compilado; nullptr si no. Solo actualiza la
/// tabla rapida (0.1.5.7: el enlace indirecto del codigo generado busca ahi,
/// asi que conviene que tenga los bloques que de verdad se usan).
Block* Peek(u32 pc) {
    FastSlot& slot = g_fast[(pc >> 2) & (kFastSlots - 1)];
    Block* block = nullptr;
    if (slot.pc == pc) [[likely]] {
        block = slot.block;
    } else {
        const auto it = g_blocks.find(pc);
        if (it == g_blocks.end()) {
            return nullptr;
        }
        block = &it->second;
        if (block->state == BlockState::Compiled) {
            slot.pc = pc;
            slot.block = block;
        }
    }
    return block->state == BlockState::Compiled ? block : nullptr;
}

/// Toca comprobar esta ejecucion del bloque contra el interprete. La misma
/// prueba la hace en linea el enlace del codigo generado (EmitLink).
bool DueForCheck(const Block& block) {
    return block.link.runs < kFullChecks || (block.link.runs & (kSampleEvery - 1)) == 0;
}

/**
 * Ejecuta el bloque (y los que enlace su codigo) con 'chain_budget'
 * instrucciones disponibles para los enlazados. Devuelve las instrucciones
 * ejecutadas en TOTAL: las del bloque mas las de los enlazados, que son
 * exactamente lo que el enlace le ha ido restando al presupuesto.
 */
u64 RunCompiled(ARMul_State* cpu, Block& block, u8* const* pages, u64 chain_budget) {
    block.link.runs++;
    const u32 budget = chain_budget > 0xFFFFFFFFu ? 0xFFFFFFFFu : static_cast<u32>(chain_budget);
    cpu->jit_link_budget = budget;
    cpu->jit_link_hops = 0;
    g_link_withdrawn = 0;
    block.code(cpu, pages);
    const u32 hops = cpu->jit_link_hops;
    const u64 linked = budget - cpu->jit_link_budget - g_link_withdrawn;
    g_local.dispatches += hops;
    g_local.links += hops;
    g_local.jit_dispatches += 1 + hops;
    g_local.jit_instructions += block.link.count + linked;
    return block.link.count + linked;
}

} // Anonymous namespace

u32 TryRun(ARMul_State* cpu, u64 budget_left) {
    if (mode.load(std::memory_order_relaxed) == 0) {
        return 0;
    }
    if (!g_ready) {
        if (g_init_tried || !Init(cpu)) {
            return 0;
        }
    }
    if (g_flush_requested) {
        const ScopedMicros timer{g_compile_us};
        ResetAll();
    }
    if (cpu->TFlag != 0) {
        g_local.dispatches++;
        g_local.rejects[kRejectThumb]++;
        return 0; // Thumb: el interprete (de momento)
    }
    Memory::PageTable* table = cpu->memory.fast_page_table;
    if (table == nullptr) {
        return 0;
    }
    Block* block = Acquire(cpu, cpu->Reg[15]);
    if (block == nullptr) {
        return 0;
    }
    // Para StopLinksIfRescheduled (0.1.5.7).
    g_run_target = cpu->NumInstrsToExecute;
    // Temporizacion exacta: solo si el bloque cabe entero en la rodaja.
    if (block->link.count > budget_left) {
        return 0;
    }
    u8* const* pages = table->RawPointers();
    if (DueForCheck(*block)) {
        const ScopedMicros timer{g_check_us};
        block->link.runs++;
        StartCheck(cpu, *block, pages);
        return 0; // el interprete ejecuta el bloque de verdad
    }
    /**
     * Ya no le toca comprobacion, asi que ha pasado las kFullChecks iniciales
     * (la ultima se cerro en el despacho, antes de llegar aqui, y si hubiera
     * fallado el bloque estaria en la lista negra y Acquire no lo daria): ya
     * puede ser destino de un enlace del codigo generado (0.1.8.1).
     */
    if (block->link.entry == 0) {
        block->link.entry = block->chain_entry;
    }
    /**
     * Una entrada de cada kTimeSampleEvery otra vez (0.2.1.0). En 0.1.8.1 se
     * cronometraban todas para quitar ruido al reparto entre "jit" y "resto",
     * pero son ~1.000 por fotograma y cada una leia el reloj dos veces, que en
     * la Vita es una llamada al kernel. Con el intervalo de 10 s del registro
     * salen miles de muestras: el ruido se promedia.
     */
    const SampledMicros timer{g_jit_us, TimeThisCall(++g_jit_entries)};
    // Los enlazados dentro del codigo generado (EmitLink) ven lo que queda de
    // la rodaja DESPUES de este bloque, igual que el bucle de abajo.
    u64 done = RunCompiled(cpu, *block, pages, budget_left - block->link.count);

    /**
     * ENCADENADO (0.1.4.9). Si el bloque siguiente tambien esta compilado, se
     * ejecuta aqui mismo en vez de volver al despacho del interprete, que para
     * bloques de unas pocas instrucciones era una parte grande del coste.
     *
     * Hace EXACTAMENTE lo que haria el despacho entre dos bloques, en el mismo
     * orden: salir si hay una interrupcion pendiente y no estan enmascaradas
     * (la comprobacion de NirqSig del principio de DISPATCH), alinear el PC
     * (que en modo ARM es quitarle los dos bits de abajo), no seguir en Thumb,
     * y no empezar un bloque que no quepa en lo que queda de rodaja. Si toca
     * comprobar el bloque siguiente, se vuelve al despacho, que lo hara por el
     * camino de siempre. El resultado es el mismo que sin encadenar.
     *
     * Desde 0.1.5.7 la mayoria de estos saltos los hace el propio codigo
     * generado (EmitLink); aqui llegan los que el no puede: destino que no
     * estaba en la tabla rapida, PC sin alinear, etc.
     */
    while (done < budget_left) {
        // El interprete habria parado si alguien replanifico (0.1.5.7).
        if (cpu->NumInstrsToExecute != g_run_target) {
            break;
        }
        if (!cpu->NirqSig && !(cpu->Cpsr & 0x80)) {
            break;
        }
        if (cpu->TFlag != 0 || g_flush_requested) {
            break;
        }
        cpu->Reg[15] &= 0xFFFFFFFCu;
        // Solo se encadena a un bloque YA compilado, y sin tocar nada: si no lo
        // esta, el despacho de verdad lo contara, lo compilara o lo mandara al
        // interprete, como siempre.
        Block* next = Peek(cpu->Reg[15]);
        if (next == nullptr || next->link.count > budget_left - done || DueForCheck(*next)) {
            break;
        }
        /**
         * Aqui tambien se publica la entrada (0.1.8.3). Un bloque al que solo
         * se llega por este bucle -- el destino de un enlace que fallo porque
         * su entrada aun no estaba publicada -- nunca pasaba por arriba de
         * TryRun, asi que nunca se publicaba y ese enlace fallaba para siempre:
         * crash.txt de 0.1.8.2, 7.525 enlaces de 19.417 despachos, contra 15.823
         * de 18.245 en 0.1.8.0 en el mismo punto. No le toca comprobacion, asi
         * que ya paso (y se cerro en el despacho) la ultima de las iniciales.
         */
        if (next->link.entry == 0) {
            next->link.entry = next->chain_entry;
        }
        g_local.dispatches++;
        done += RunCompiled(cpu, *next, pages, budget_left - done - next->link.count);
    }
    return static_cast<u32>(done);
}

void CompletePendingCheck(ARMul_State* cpu) {
    if (!g_pending.active) {
        return;
    }
    const ScopedMicros timer{g_check_us};
    g_pending.active = false;
    if (g_pending.cpu != cpu || g_pending.block == nullptr) {
        return;
    }
    Block& block = *g_pending.block;
    std::string what;
    for (u32 i = 0; i < 16 && what.empty(); i++) {
        if (cpu->Reg[i] != g_pending.regs[i]) {
            what = fmt::format("r{} jit {:#010x} interprete {:#010x}", i, g_pending.regs[i],
                               cpu->Reg[i]);
        }
    }
    if (what.empty() && (cpu->NFlag != g_pending.n || cpu->ZFlag != g_pending.z ||
                         cpu->CFlag != g_pending.c || cpu->VFlag != g_pending.v)) {
        what = fmt::format("flags jit {}{}{}{} interprete {}{}{}{}", g_pending.n, g_pending.z,
                           g_pending.c, g_pending.v, cpu->NFlag, cpu->ZFlag, cpu->CFlag,
                           cpu->VFlag);
    }
    if (what.empty() && cpu->TFlag != g_pending.t) {
        what = fmt::format("thumb jit {} interprete {}", g_pending.t, cpu->TFlag);
    }
    for (u32 i = 0; i < 64 && what.empty(); i++) {
        if (cpu->ExtReg[i] != g_pending.ext[i]) {
            what = fmt::format("vfp ExtReg[{}] jit {:#010x} interprete {:#010x}", i,
                               g_pending.ext[i], cpu->ExtReg[i]);
        }
    }
    for (u32 i = 0; i < VFP_SYSTEM_REGISTER_COUNT && what.empty(); i++) {
        /**
         * En el FPSCR no cuentan los bits ACUMULADOS de excepcion (IOC, DZC,
         * OFC, UFC, IXC e IDC: 0x9F). Son "alguna vez paso", los juegos no los
         * leen, y el VFP de la Vita y la emulacion software del interprete no
         * siempre los ponen igual (0.1.6.3: un bloque vuelto al interprete
         * solo por IDC). Todo lo demas -- NZCV de VCMP, redondeo, FZ, DN --
         * se sigue comparando (0.1.7.0).
         */
        const u32 ignore = i == VFP_FPSCR ? 0x9Fu : 0u;
        if ((cpu->VFP[i] & ~ignore) != (g_pending.vfp[i] & ~ignore)) {
            what = fmt::format("vfp VFP[{}] jit {:#010x} interprete {:#010x}", i,
                               g_pending.vfp[i], cpu->VFP[i]);
        }
    }
    if (what.empty() && cpu->memory.fast_page_table != nullptr) {
        // Cada escritura del diario que no pise una posterior: la memoria tiene
        // que tener ahora (despues del interprete) el mismo valor.
        u8* const* pages = cpu->memory.fast_page_table->RawPointers();
        for (u32 i = 0; i < g_pending.journal_size && what.empty(); i++) {
            const JournalEntry& entry = g_pending.journal[i];
            const u32 size = KindSize(entry.kind);
            bool overwritten = false;
            for (u32 j = i + 1; j < g_pending.journal_size; j++) {
                if (Overlaps(entry.address, size, g_pending.journal[j].address,
                             KindSize(g_pending.journal[j].kind))) {
                    overwritten = true;
                    break;
                }
            }
            if (overwritten) {
                continue;
            }
            const u8* page = pages[entry.address >> 12];
            if (page == nullptr) {
                continue;
            }
            u32 actual = 0;
            std::memcpy(&actual, page + (entry.address & 0xFFF), size);
            const u32 mask = size == 4 ? 0xFFFFFFFFu : size == 2 ? 0xFFFFu : 0xFFu;
            if ((actual & mask) != (entry.value & mask)) {
                what = fmt::format("memoria {:#010x} jit {:#x} interprete {:#x}", entry.address,
                                   entry.value & mask, actual & mask);
            }
        }
    }
    if (!what.empty()) {
        block.state = BlockState::Blacklisted;
        // 0.1.5.7: y deja de ser enlazable desde el codigo generado.
        block.link.entry = 0;
        Common::FrameStats::jit_mismatches.fetch_add(1, std::memory_order_relaxed);
        NoteMismatch(block, what);
    }
}

void AbandonPendingCheck(u32 slice_instructions) {
    g_pending.active = false;
    g_local.instructions += slice_instructions;
    // 0.1.6.3: publicar cada 32 rodajas, no en cada una: son unas veinte
    // operaciones atomicas, y el overlay solo lo lee una vez por segundo.
    static u32 slices_since_flush = 0;
    if (++slices_since_flush >= 32) {
        slices_since_flush = 0;
        FlushLocalStats();
    }
}

namespace {
/**
 * Las instrucciones que mas bloques mandan al interprete como "otro" (0.1.9.6),
 * contadas por despacho: sin esto "rech otro 7%" no dice que falta por
 * traducir. 16 huecos; cuando se llenan, las nuevas no entran.
 */
struct RejectWord {
    u32 word = 0;
    u32 count = 0;
};
std::array<RejectWord, 16> g_reject_words{};
} // Anonymous namespace

void CountRejectWord(u32 word) {
    for (auto& entry : g_reject_words) {
        if (entry.count == 0) {
            entry.word = word;
            entry.count = 1;
            return;
        }
        if (entry.word == word) {
            entry.count++;
            return;
        }
    }
}

std::string TakeRejectWords() {
    auto sorted = g_reject_words;
    std::sort(sorted.begin(), sorted.end(),
              [](const RejectWord& a, const RejectWord& b) { return a.count > b.count; });
    std::string text;
    for (std::size_t i = 0; i < 6 && sorted[i].count != 0; i++) {
        text += fmt::format(" {:08x}x{}", sorted[i].word, sorted[i].count);
    }
    g_reject_words = {};
    return text.empty() ? std::string(" -") : text;
}

const char* RejectName(u32 reason) {
    switch (reason) {
    case kRejectThumb:
        return "thumb";
    case kRejectVfp:
        return "vfp";
    case kRejectCoproc:
        return "coproc";
    case kRejectSvc:
        return "svc";
    case kRejectExclusive:
        return "ldrex";
    case kRejectMedia:
        return "media";
    case kRejectMisc:
        return "estado";
    case kRejectPcWrite:
        return "pc";
    case kRejectLong:
        return "largo";
    case kRejectNone:
        return "-";
    default:
        return "otro";
    }
}

void AddSliceTime(u64 microseconds) {
    g_local.arm_us += microseconds;
    g_local.slices++;
}

void AddSvcTime(u64 microseconds) {
    g_local.svc_us += microseconds;
}

void CountInstructions(u32 instructions) {
    g_local.instructions += instructions;
}

void CodeUsage(u32& bytes, u32& blocks) {
    bytes = g_code_used_words * 4;
    blocks = g_code_blocks;
}

void TakeStats(Stats& out) {
    out.instructions = g_published.instructions.exchange(0, std::memory_order_relaxed);
    out.arm_us = g_published.arm_us.exchange(0, std::memory_order_relaxed);
    out.svc_us = g_published.svc_us.exchange(0, std::memory_order_relaxed);
    out.jit_instructions = g_published.jit_instructions.exchange(0, std::memory_order_relaxed);
    out.dispatches = g_published.dispatches.exchange(0, std::memory_order_relaxed);
    out.jit_dispatches = g_published.jit_dispatches.exchange(0, std::memory_order_relaxed);
    out.slices = g_published.slices.exchange(0, std::memory_order_relaxed);
    out.links = g_published.links.exchange(0, std::memory_order_relaxed);
    out.slow_calls = g_published.slow_calls.exchange(0, std::memory_order_relaxed);
    out.vfp_calls = g_published.vfp_calls.exchange(0, std::memory_order_relaxed);
    out.jit_us = g_published.jit_us.exchange(0, std::memory_order_relaxed);
    out.slow_us = g_published.slow_us.exchange(0, std::memory_order_relaxed);
    out.vfp_us = g_published.vfp_us.exchange(0, std::memory_order_relaxed);
    out.check_us = g_published.check_us.exchange(0, std::memory_order_relaxed);
    out.compile_us = g_published.compile_us.exchange(0, std::memory_order_relaxed);
    for (u32 i = 0; i < kRejectCount; i++) {
        out.rejects[i] = g_published.rejects[i].exchange(0, std::memory_order_relaxed);
    }
}

void Reset() {
    if (!g_ready) {
        return;
    }
    ResetAll();
}

void InvalidateRange(u32 start, u32 size) {
    if (!g_ready || size == 0) {
        return;
    }
    g_invalid_ranges.emplace_back(start, start + size);
}

void ApplyInvalidations() {
    if (!g_ready || g_invalid_ranges.empty()) {
        g_invalid_ranges.clear();
        return;
    }
    // Miles de palabras sueltas, casi siempre seguidas: se juntan en tramos.
    std::sort(g_invalid_ranges.begin(), g_invalid_ranges.end());
    std::vector<std::pair<u32, u32>> merged;
    for (const auto& range : g_invalid_ranges) {
        if (!merged.empty() && range.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    g_invalid_ranges.clear();
    for (auto& [pc, block] : g_blocks) {
        // Un bloque no pasa del final de su pagina. Sin palabras (rechazado),
        // se cuenta hasta ahi: rehacer de mas no rompe nada.
        const u32 words = block.words.empty() ? kMaxBlockInstructions
                                               : static_cast<u32>(block.words.size());
        const u32 end = std::min(pc + words * 4, (pc & ~0xFFFu) + 0x1000u);
        const auto next = std::upper_bound(
            merged.begin(), merged.end(), pc,
            [](u32 value, const std::pair<u32, u32>& range) { return value < range.first; });
        bool hit = next != merged.end() && next->first < end;
        if (!hit && next != merged.begin()) {
            hit = std::prev(next)->second > pc;
        }
        if (!hit) {
            continue;
        }
        if (g_pending.active && g_pending.block == &block) {
            g_pending.active = false;
        }
        // El Block se queda (los enlaces y g_fast lo apuntan); vuelve a nuevo.
        block.state = BlockState::New;
        block.visits = 0;
        block.code = nullptr;
        block.check_code = nullptr;
        block.chain_entry = 0;
        block.link.entry = 0;
        block.link.runs = 0;
        block.reject = kRejectNone;
        block.words.clear();
    }
}

} // namespace Core::ArmJit

#endif // __PSVITA__
