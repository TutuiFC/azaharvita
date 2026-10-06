// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

/**
 * JIT DEL ARM11 PARA PS VITA (0.1.4.8): de codigo ARM del 3DS a codigo ARM de
 * la Vita, HIBRIDO con el interprete dyncom.
 *
 * POR QUE. La CPU emulada costaba ~109 ms de cada vblank de 283 en Pokemon
 * Rubi Omega, y no bajo en ninguna version: es el techo de todo lo demas.
 * dynarmic (el JIT de Azahar) solo genera x86-64 y ARM64; la Vita es ARMv7.
 *
 * LA IDEA QUE LO HACE ABORDABLE: el 3DS y la Vita son los dos ARM. Una
 * instruccion de procesamiento de datos del juego se emite con LA MISMA
 * codificacion, cambiando solo los numeros de registro, y los flags del juego
 * viven en los flags del procesador durante el bloque: acarreos, desbordes,
 * desplazamientos y condiciones salen identicos porque los calcula el mismo
 * hardware con las mismas reglas.
 *
 * QUE SE COMPILA. BLOQUES CON LA MISMA EXTENSION QUE LOS DEL INTERPRETE: desde
 * el PC de despacho hasta la primera instruccion "de salto" (la que termina el
 * bloque del interprete) o hasta el final de la pagina de 4 KB. Si ALGUNA
 * instruccion del bloque no esta soportada, el bloque entero se queda en el
 * interprete. Asi la cuenta de instrucciones (que es la temporizacion del
 * juego) es identica, y la comprobacion puede comparar el mismo tramo.
 *
 * Soportado en esta primera version, solo en modo ARM (Thumb y VFP se quedan
 * en el interprete):
 *   - procesamiento de datos (las 16 operaciones, con y sin flags, inmediato,
 *     desplazamiento por inmediato y por registro), sin escribir el PC
 *   - MUL, MLA, UMULL, UMLAL, SMULL, SMLAL
 *   - LDR, STR, LDRB, STRB, LDRH, STRH, LDRSB, LDRSH con todos los modos de
 *     direccion (pre/post indice, escritura del base, desplazamiento inmediato
 *     o por registro desplazado)
 *   - LDM y STM sin el bit S
 *   - como final de bloque: B, BL, BX, BLX por registro y LDM con el PC
 *
 * LA TEMPORIZACION ES EXACTA: un bloque solo se ejecuta con el JIT si cabe
 * entero en lo que le queda a la rodaja; si no, lo ejecuta el interprete, que
 * para en la instruccion justa.
 *
 * LA AUTOCOMPROBACION (la red de seguridad, porque no se puede probar en un
 * PC). Las primeras ejecuciones de cada bloque, y despues una de cada
 * kSampleEvery, se hacen asi:
 *   1. se guarda el estado del ARM;
 *   2. se ejecuta una VARIANTE del bloque que NO escribe en memoria: apunta las
 *      escrituras en un diario (y si tiene que tocar memoria que no es RAM,
 *      como registros de hardware, abandona la comprobacion);
 *   3. se restaura el estado y el INTERPRETE ejecuta el bloque de verdad;
 *   4. en el siguiente despacho se comparan registros, flags, modo Thumb y
 *      cada byte que el bloque escribio.
 * Si algo difiere: crash.txt con la direccion, las instrucciones y lo que
 * difiere, y ese bloque no vuelve a usar el JIT nunca. La partida sigue con
 * el resultado del interprete, que es la referencia.
 *
 * 0.1.5.7, TRES OPTIMIZACIONES DEL CODIGO GENERADO (el detalle, en el .cpp):
 *   1. ENLACE EN LINEA (EmitDirectLink). Cada salida de un bloque intenta saltar al
 *      siguiente sin volver a C++, con las mismas condiciones que el despacho
 *      (interrupcion, Thumb, alineacion, presupuesto, comprobacion). Salidas
 *      con destino fijo (B, BL, camino no tomado, fin de pagina) por un
 *      literal; indirectas (BX, retornos con LDM/LDR/MOV al PC) por la tabla
 *      rapida. Interruptor: jit_enlace_dir.
 *   2. CACHE DE REGISTROS MAS UTIL: cinco huecos (r0-r3, lr) en vez de tres,
 *      contando tambien los usos en accesos a memoria, y las operaciones de
 *      datos trabajan DIRECTAMENTE sobre los huecos. Interruptor:
 *      jit_cache_reg.
 *   3. VIDA DE LOS FLAGS: un acceso a memoria no guarda ni repone los flags
 *      si ninguna instruccion posterior del bloque los lee antes de que una
 *      aritmetica con S los sobrescriba. La variante de comprobacion usa el
 *      mismo analisis, asi que un error saldria como DIFERENCIA.
 */

#ifdef __PSVITA__

#include <array>
#include <string>
#include <atomic>
#include "common/common_types.h"

struct ARMul_State;

namespace Core::ArmJit {

/// 0 = apagado (todo al interprete), 1 = encendido. Se cambia desde el frontend.
extern std::atomic<u32> mode;

/**
 * Cache de registros del juego DENTRO del bloque (0.1.5.2, punto 4.3).
 * 0 = el camino de 0.1.5.1 (carga y guarda en ARMul_State::Reg en CADA
 * instruccion), 1 = los registros mas usados del bloque se quedan en
 * registros del host y solo se vuelcan a memoria al salir o antes de cualquier
 * llamada a helper que lea o escriba cpu->Reg. Interruptor del menu de
 * ajustes; al cambiarlo se vacia la cache de bloques para recompilar.
 */
extern std::atomic<u32> reg_cache;

/**
 * VFP de MOVIMIENTO de datos en el JIT (0.1.5.2, punto 4.4 fase 1):
 * VLDR/VSTR, VMOV entre ARM y VFP y entre singles. 0 = el camino de 0.1.5.1
 * (todo VFP al interprete), 1 = se emite. La aritmetica VADD/VMUL... NO entra
 * aqui (fase 2): el interprete la emula en software segun el FPSCR del juego
 * y el hardware de la Vita no tiene por que dar el mismo bit. Interruptor del
 * menu de ajustes; al cambiarlo se vacia la cache de bloques.
 */
extern std::atomic<u32> vfp_data;

/**
 * Enlazado directo entre bloques en el codigo generado (0.1.5.2, 4.8).
 * 0 = el camino de 0.1.5.1 (cada bloque vuelve a C++/TryRun y ahi se
 * encadena), 1 = si el destino de un B/BL fijo ya esta compilado (o se
 * compila despues y se parchea), el final del bloque salta directo al cuerpo
 * del destino con las mismas comprobaciones de TryRun (IRQ, Thumb, rodaja,
 * comprobacion pendiente, flush). Interruptor del menu; al cambiarlo se
 * vacia la cache de bloques.
 */
extern std::atomic<u32> direct_link;

/// Los enlaces directos se parchean para saltar sin comprobaciones cuando el
/// destino esta publicado (0.2.2.9). 0 = como antes. Se lee al compilar y al
/// publicar cada bloque.
extern std::atomic<u32> direct_link_patch;

/// Superbloques (0.2.3.1): los saltos condicionales hacia delante no acaban
/// el bloque (ver DecodeArmBlock). 0 = un bloque por salto, como antes. Se lee
/// al decodificar cada bloque; al cambiarlo se vacia la cache de bloques.
extern std::atomic<u32> superblocks;

/// Registros fijos (0.3.0.0): r0-r3 del juego en r0-r3 del anfitrion durante
/// las cadenas de bloques (ver REGISTROS FIJOS). 0 = la cache de cada bloque,
/// como antes. Solo con la cache de registros encendida; al cambiarlo se
/// vacia la cache de bloques.
extern std::atomic<u32> global_regs;

/**
 * Aritmetica VFP en el hardware de la Vita (0.1.7.0), con el FPSCR del juego
 * cargado durante cada instruccion. 0 = por las funciones del interprete
 * (0.1.6.1). Va unido al interruptor "VFP datos JIT" del menu.
 */
extern std::atomic<u32> vfp_native;

/**
 * Bloques Thumb en el JIT (0.2.1.7). 0 = al interprete, como hasta 0.2.1.6.
 * Clave "jit_thumb=" de ajustes.txt.
 */
extern std::atomic<u32> thumb;

/**
 * Llamar en el DESPACHO del interprete, con el PC ya alineado. Si hay codigo
 * para este bloque y cabe en el presupuesto, lo ejecuta y devuelve cuantas
 * instrucciones del juego ha ejecutado; con 0, el interprete sigue como
 * siempre con este bloque.
 */
u32 TryRun(ARMul_State* cpu, u64 budget_left);

/// Al principio del despacho: cierra la comprobacion pendiente, si la hay.
void CompletePendingCheck(ARMul_State* cpu);

/// Al salir del bucle del interprete: una comprobacion a medias no vale. Se
/// le pasan las instrucciones de la rodaja (JIT + interprete), para que el
/// porcentaje del JIT salga de dos cuentas hechas en el mismo sitio.
void AbandonPendingCheck(u32 slice_instructions);

/**
 * Por que se queda un bloque en el interprete (0.1.4.9). Se cuenta por
 * DESPACHO, no por bloque distinto: lo que importa es cuantas veces se ejecuta
 * codigo que el JIT no cubre, y de que tipo, para saber que traducir despues.
 */
enum RejectReason : u32 {
    kRejectNone = 0,
    kRejectThumb,     ///< el procesador estaba en modo Thumb
    kRejectVfp,       ///< coma flotante (coprocesadores 10 y 11)
    kRejectCoproc,    ///< otro coprocesador
    kRejectSvc,       ///< llamada al sistema
    kRejectExclusive, ///< LDREX/STREX y compania
    kRejectMedia,     ///< instrucciones "media" que aun no se traducen
    kRejectMisc,      ///< MRS/MSR, CPS y demas de estado
    kRejectPcWrite,   ///< escritura del PC que no se traduce
    kRejectLong,      ///< bloque de mas de 256 instrucciones
    kRejectOther,
    kRejectCount,
};

const char* RejectName(u32 reason);

/// Las instrucciones que mas despachos mandan al interprete como "otro", para
/// crash.txt (0.1.9.6). Las pone a cero.
std::string TakeRejectWords();

/// Estadisticas desde la ultima llamada (las pone a cero). Las acumula el hilo
/// de emulacion y se publican al final de cada rodaja; esto lo lee el overlay.
struct Stats {
    u64 instructions = 0;     ///< instrucciones del juego (JIT + interprete)
    u64 jit_instructions = 0; ///< de ellas, por codigo generado
    u64 dispatches = 0;       ///< bloques despachados
    u64 jit_dispatches = 0;   ///< de ellos, por codigo generado
    std::array<u64, kRejectCount> rejects{};
    u64 arm_us = 0;           ///< tiempo real dentro del bucle del ARM (JIT + interprete)
    u64 svc_us = 0;           ///< de ese tiempo, el de las llamadas al sistema
    u64 slices = 0;           ///< rodajas del ARM (0.1.6.3)
    u64 links = 0;            ///< saltos entre bloques hechos por el codigo generado
    u64 slow_calls = 0;       ///< accesos a memoria por el camino lento (y LDM/STM)
    u64 vfp_calls = 0;        ///< aritmetica VFP por las funciones del interprete
    /// Reparto de arm_us (0.1.7.5), en microsegundos. jit_us, slow_us y vfp_us
    /// son estimaciones por muestreo; slow_us y vfp_us van DENTRO de jit_us.
    /// arm_us - svc_us - jit_us - check_us - compile_us = interprete y despacho.
    u64 jit_us = 0;     ///< codigo generado (con los enlazados y sus llamadas)
    u64 slow_us = 0;    ///< caminos lentos de memoria, LDM/STM y VLDM/VSTM
    u64 vfp_us = 0;     ///< aritmetica VFP por las funciones del interprete
    u64 check_us = 0;   ///< comprobaciones contra el interprete (sin su ejecucion)
    u64 compile_us = 0; ///< analizar, generar codigo y vaciar la cache
};
void TakeStats(Stats& out);

/// Codigo generado desde el ultimo vaciado: bytes y bloques (0.2.1.0, para
/// ver en crash.txt cuanto ocupa un bloque).
void CodeUsage(u32& bytes, u32& blocks);

/// Tiempo de una rodaja del ARM (lo mide ARM_DynCom::ExecuteInstructions).
void AddSliceTime(u64 microseconds);
/// Tiempo dentro de las llamadas al sistema de esa rodaja (SWI_INST): se resta.
void AddSvcTime(u64 microseconds);
/// Instrucciones ejecutadas antes de una llamada al sistema (el interprete
/// pone su cuenta a cero ahi).
void CountInstructions(u32 instructions);

/**
 * CONTADORES DEL PROCESADOR (0.2.3.6), ver PmuSliceBegin en el .cpp. Las
 * rodajas del ARM los encienden y apagan, y las llamadas al sistema los
 * paran mientras duran.
 */
void PmuSliceBegin();
void PmuSliceEnd(u32 instructions);
void PmuSvcBegin();
void PmuSvcEnd();
constexpr u32 kPmuCounters = 6;
struct PmuStats {
    /// Dos juegos de eventos (ver kPmuEvents): [juego * kPmuCounters + contador].
    std::array<u64, 2 * kPmuCounters> counts{};
    /// Instrucciones del juego mientras contaba cada juego.
    std::array<u64, 2> instructions{};
};
/// Lo contado desde la ultima llamada (lo pone a cero).
void TakePmu(PmuStats& out);

/// Tira todo el codigo generado. La llama ResetTransCache: cuando el
/// interprete tira sus traducciones, el JIT tira las suyas por lo mismo.
void Reset();

/**
 * INVALIDACION POR TRAMOS (0.2.0.5). Al cargar un modulo .cro, LDR:RO invalida
 * las palabras que reubica (miles) y el tramo del modulo; hasta ahora cada
 * tanda tiraba TODO el JIT, y Pokemon Sol recompilaba ~200.000 bloques al
 * pasar al 3D ("compilar" 500 ms por fotograma en crash.txt de 0.2.0.4).
 * InvalidateRange apunta el tramo; ApplyInvalidations, antes de volver a
 * ejecutar codigo del juego, devuelve a "nuevo" solo los bloques que lo tocan.
 */
void InvalidateRange(u32 start, u32 size);
void ApplyInvalidations();

} // namespace Core::ArmJit

#endif // __PSVITA__
