// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Diagnostico de fallos fatales en PS Vita.
//
// Azahar usa ASSERT/UNREACHABLE por todas partes. En PC el macro escribe en el
// registro y llama a Crash() (__builtin_trap), que bajo un depurador para la
// ejecucion y deja mirar el estado. En la Vita no hay depurador: __builtin_trap
// genera una instruccion indefinida y el kernel mata el proceso al instante,
// con el agravante de que el macro llama antes a Common::Log::Stop(), que para
// el hilo que vuelca el registro a disco. Resultado: la consola dice "se ha
// cerrado la aplicacion", azahar_log.txt se queda cortado a media linea y no
// queda ni rastro de que assert salto ni donde.
//
// Aqui todo se escribe antes de morir, con sceIo directo: sin hilo de registro,
// sin reservar memoria y sin depender del runtime de C++.
//
// Nota sobre por que VitaAssertFail es noexcept y no lanza: lanzar obligaba al
// compilador a generar tablas de desenrollado en cada una de las miles de
// funciones que contienen un ASSERT. Eso sumo unas 7.700 relocalizaciones al
// binario y lo paso de 2^21, justo el punto en el que vita-elf-create se cae
// sin decir nada. Terminar aqui mismo sale mas barato y el manejador de
// terminate del frontend sigue pintando el mensaje en pantalla.

#pragma once

#ifdef __PSVITA__

#include <array>
#include <atomic>

namespace Common {

/// Microsegundos de reloj de pared desde que arranco el proceso.
///
/// Envuelve sceKernelGetProcessTimeWide para que video_core pueda cronometrar
/// sin arrastrar cabeceras de psp2 a codigo que es comun a todas las
/// plataformas. Es una lectura de contador, no una llamada al kernel cara.
unsigned long long VitaMicros();

/**
 * Reparto del tiempo de CADA FOTOGRAMA, medido donde de verdad ocurre.
 *
 * POR QUE HACE FALTA, teniendo ya el 'cpu/gpu/svc/swap' de PerfStats.
 *
 * 1. PerfStats mete en 'swap' dos cosas que no se parecen en nada: convertir el
 *    framebuffer del 3DS (PrepareRenderTarget, trabajo de CPU en el hilo que
 *    emula el ARM11) y presentarlo (subir la textura y dibujar, trabajo del
 *    chip grafico). Al sustituir vita2d por un backend GXM, lo primero no
 *    cambia y lo segundo si; sin separarlos no hay forma de decir si el cambio
 *    sirvio.
 *
 * 2. Dentro de presentar hay que separar 'dibujar' de 'intercambiar buffers'.
 *    vita2d_swap_buffers ESPERA a que la GPU termine y al barrido de pantalla.
 *    Ese tiempo no es coste: es la consola parada. Sumado al resto haria
 *    parecer cara una presentacion que en realidad estaba esperando.
 *
 * 3. Todo lo de PerfStats son PORCENTAJES del fotograma. Un porcentaje no sirve
 *    para comparar dos versiones: si el fotograma entero baja de 2000 ms a
 *    1000, todos los porcentajes pueden salir identicos. El plan pide medir
 *    "antes -> despues" sobre las mismas ROMs, y para eso hacen falta
 *    milisegundos absolutos. De ahi que esto acumule tiempo, no proporciones.
 *
 * Son microsegundos acumulados desde el ultimo Reset(), que hace el overlay una
 * vez por segundo. Dos lecturas de reloj por fotograma y apartado -- unas doce
 * en total --, no por pixel: la medida no se estorba a si misma.
 */
namespace FrameStats {

/// Framebuffer del 3DS -> ScreenInfo. CPU pura, en el hilo de emulacion.
inline std::atomic<unsigned long long> convert_us{0};
/// Subir los pixeles a memoria de la GPU (memcpy a la textura).
inline std::atomic<unsigned long long> upload_us{0};
/// Encolar el dibujado de las dos pantallas y del overlay.
inline std::atomic<unsigned long long> draw_us{0};
/// Intercambio de buffers: esperar a la GPU y al barrido. Casi todo espera.
inline std::atomic<unsigned long long> swap_us{0};
/**
 * El bucle de vertices entero de PicaCore::LoadVertices: cargar atributos,
 * ejecutar el shader de vertices en el interprete y entregar cada vertice al
 * ensamblador de primitivas (que por dentro llama a AddTriangle).
 *
 * Es UNA lectura de reloj por llamada de dibujado -- unas trescientas por
 * fotograma --, no por vertice: medir por vertice costaria mas que lo medido.
 */
inline std::atomic<unsigned long long> vertices_us{0};
/// Cuantos vertices han pasado de verdad por el interprete (los que fallan el
/// cache). Con vertices_us da el coste por vertice, y dice ademas si el cache
/// se esta usando: en dibujado no indexado no se usa y esto iguala al total.
inline std::atomic<unsigned int> vertices_shaded{0};
/// Nuestro camino de GPU: DrawTriangles, o sea empaquetar el lote, poner el
/// estado de GXM y encolar el dibujado. Una lectura por lote.
inline std::atomic<unsigned long long> batch_us{0};
/**
 * Solo la fase de SOMBREADO de los vertices (cargar atributos + interprete),
 * medida desde el hilo de emulacion: es tiempo de pared, no la suma de lo que
 * trabaja cada nucleo. vertices_us - shade_us es lo que cuesta entregar los
 * vertices ya sombreados (ensamblar, AddTriangle y lo que caiga a software).
 * Desde 0.1.0.42, que reparte el sombreado entre tres nucleos.
 */
inline std::atomic<unsigned long long> shade_us{0};
/**
 * Tiempo parado en sceGxmFinish al cerrar escenas de nuestro rasterizador de
 * GXM. Es la espera a la GPU: si esto se come 'lote', lo que sobra son cierres
 * de escena, no trabajo de CPU.
 */
inline std::atomic<unsigned long long> finish_us{0};
/// Texturas decodificadas en el cache de GXM (fallos) y lo que ha costado.
inline std::atomic<unsigned int> texture_decodes{0};
inline std::atomic<unsigned long long> texture_decode_us{0};
/**
 * Por que se decodifica (0.1.0.43). reuses = texturas avisadas como posibles
 * cambiadas cuyo hash salio igual (decodificacion AHORRADA); changed = las que
 * si habian cambiado; evictions = expulsadas por falta de sitio (LRU). Si 'tx'
 * sigue alto con 'changed' bajo, el problema es de capacidad, no de avisos.
 */
inline std::atomic<unsigned int> texture_reuses{0};
inline std::atomic<unsigned int> texture_changed{0};
inline std::atomic<unsigned int> texture_evictions{0};
/**
 * Ruta rapida del interprete de shaders (0.1.0.44, ver
 * shader_interpreter_fast.h). programs = programas pre-decodificados desde el
 * arranque; checks = vertices comprobados contra el interprete puro en el
 * intervalo; mismatches = programas que dieron una diferencia y volvieron al
 * interprete (desde el arranque: tiene que ser 0).
 */
inline std::atomic<unsigned int> fast_programs{0};
inline std::atomic<unsigned int> fast_checks{0};
inline std::atomic<unsigned int> fast_mismatches{0};
/**
 * Sombreado por dentro (0.1.0.45). busy = suma del tiempo de trabajo de los
 * tres nucleos (comparado con 3 x shade_us dice si el reparto aprovecha los
 * nucleos o se queda esperando); fast_ops / slow_instrs = instrucciones del
 * shader ejecutadas por la ruta rapida y por el interprete.
 */
inline std::atomic<unsigned long long> shade_busy_us{0};
/**
 * Desglose de UN vertice de cada 32 (0.1.7.2), en microsegundos: leer los
 * atributos (LoadVertex + LoadInput), ejecutar el shader, y guardar y
 * convertir la salida. Dice si hay que atacar el shader o lo de alrededor.
 * No se pone a cero con Reset(): lo lee y lo pone a cero el overlay.
 */
inline std::atomic<unsigned long long> vtx_samples{0};
inline std::atomic<unsigned long long> vtx_load_us{0};
inline std::atomic<unsigned long long> vtx_run_us{0};
inline std::atomic<unsigned long long> vtx_out_us{0};
/// Dibujar el propio overlay (dentro de draw_us). Ver vita_window.cpp.
inline std::atomic<unsigned long long> overlay_us{0};
/**
 * JIT del ARM11 (0.1.4.8, ver arm_dyncom_jit.h). Todo lo cuenta el hilo de
 * emulacion, que es el unico que ejecuta el ARM:
 *   guest_instrs   instrucciones del juego ejecutadas en el intervalo (JIT +
 *                  interprete): con 'cpu' da el coste por instruccion.
 *   jit_instrs     de esas, las que ha ejecutado codigo generado.
 *   jit_blocks     bloques compilados desde el arranque.
 *   jit_rejected   bloques que el JIT no sabe compilar (se quedan en el
 *                  interprete), desde el arranque.
 *   jit_checks     bloques comprobados contra el interprete en el intervalo.
 *   jit_mismatches bloques que dieron una diferencia (desde el arranque:
 *                  tiene que ser 0; crash.txt dice cual).
 */
inline std::atomic<unsigned long long> guest_instrs{0};
inline std::atomic<unsigned long long> jit_instrs{0};
inline std::atomic<unsigned int> jit_blocks{0};
inline std::atomic<unsigned int> jit_rejected{0};
inline std::atomic<unsigned int> jit_checks{0};
inline std::atomic<unsigned int> jit_mismatches{0};
inline std::atomic<unsigned long long> shade_fast_ops{0};
inline std::atomic<unsigned long long> shade_slow_instrs{0};
/**
 * Que OPCODES caen a la ruta lenta (0.1.5.2, punto 4.2a). Solo medir: por
 * cuanto se queda "rap" al 80% no dice que instruccion es la culpable. Hueco
 * por valor de opcode de la PICA (hasta 0x3F; los pseudo arriba no llegan aqui).
 */
constexpr unsigned kSlowOpcodeSlots = 64;
inline std::array<std::atomic<unsigned long long>, kSlowOpcodeSlots> shade_slow_opcodes{};
/// Escrituras de registro PICA y su tiempo (0.1.5.2, punto 4.7): cuanto de
/// "resto de gx" es decodificar escrituras de registro frente a rellenos y
/// transferencias (esos dos ya tienen su tiempo en VideoCore::GxStats).
inline std::atomic<unsigned long long> pica_reg_writes{0};
inline std::atomic<unsigned long long> pica_reg_write_ns{0};
/// Presentaciones contadas en el intervalo, para poder dividir.
inline std::atomic<unsigned int> frames{0};

inline void Add(std::atomic<unsigned long long>& slot, unsigned long long begin) {
    slot.fetch_add(VitaMicros() - begin, std::memory_order_relaxed);
}

inline void Reset() {
    convert_us.store(0, std::memory_order_relaxed);
    upload_us.store(0, std::memory_order_relaxed);
    draw_us.store(0, std::memory_order_relaxed);
    swap_us.store(0, std::memory_order_relaxed);
    vertices_us.store(0, std::memory_order_relaxed);
    vertices_shaded.store(0, std::memory_order_relaxed);
    batch_us.store(0, std::memory_order_relaxed);
    shade_us.store(0, std::memory_order_relaxed);
    finish_us.store(0, std::memory_order_relaxed);
    texture_decode_us.store(0, std::memory_order_relaxed);
    shade_busy_us.store(0, std::memory_order_relaxed);
    overlay_us.store(0, std::memory_order_relaxed);
    // Los contadores de "desde el arranque" (fast_programs, jit_*, etc) NO se
    // tocan aqui. shade_slow_opcodes SI: es un desglose del intervalo, como
    // shade_slow_instrs, y el overlay lo anota en crash.txt una vez por segundo.
    shade_fast_ops.store(0, std::memory_order_relaxed);
    shade_slow_instrs.store(0, std::memory_order_relaxed);
    for (auto& slot : shade_slow_opcodes) {
        slot.store(0, std::memory_order_relaxed);
    }
    pica_reg_writes.store(0, std::memory_order_relaxed);
    pica_reg_write_ns.store(0, std::memory_order_relaxed);
    frames.store(0, std::memory_order_relaxed);
}

} // namespace FrameStats

/// Anota una linea en ux0:/data/azahar/crash.txt. Seguro en cualquier hilo y en
/// cualquier momento, incluido antes de main o con el heap agotado.
void VitaNote(const char* title, const char* detail);

/// Deja constancia de un assert fallido y termina el proceso.
///
/// El texto del assert no viaja por aqui: lo escribe el propio LOG_CRITICAL del
/// macro, que en la Vita vuelca los mensajes criticos a crash.txt de forma
/// sincrona (ver FmtLogMessageImpl). Aqui solo hace falta fichero y linea, y
/// asi el macro no crece ni una instruccion mas de lo que ya ocupaba.
[[noreturn]] void VitaAssertFail(const char* file, int line) noexcept;

/// Ultimo fallo fatal anotado, o cadena vacia. Lo usa el manejador de terminate
/// del frontend para pintar algo util en vez de "terminate directo".
const char* LastFatalMessage();

/**
 * Ata el hilo que llama a uno de los tres nucleos de usuario de la Vita.
 *
 * La Vita tiene cuatro nucleos Cortex-A9, de los cuales tres estan disponibles
 * para la aplicacion (el cuarto lo reserva el sistema). Si no se dice nada, los
 * hilos se reparten segun le parezca al planificador, y no hay garantia de que
 * acaben en nucleos distintos: un pool de hilos "en paralelo" puede terminar
 * turnandose en el mismo nucleo, con lo que no solo no gana nada sino que
 * ademas paga los cambios de contexto.
 *
 * @param index Numero de hilo. Se reparte en round-robin entre los 3 nucleos.
 * @param role  Texto corto para la anotacion en crash.txt ("rasterizador", etc).
 */
void VitaPinThreadToUserCore(unsigned int index, const char* role);

/**
 * Reloj REAL de la CPU en MHz, leido de vuelta despues de pedir el overclock.
 *
 * scePowerSetArmClockFrequency(444) puede ser rechazada en una compilacion
 * "safe" y devolver error, y nadie comprobaba el resultado. Toda la sesion se
 * han calculado ciclos por pixel suponiendo 444 MHz sin saber si es cierto: si
 * la consola lo dejo en 333, esas cuentas van un 33% infladas.
 *
 * Lo escribe main.cpp una vez al arrancar; lo lee el overlay. Un int escrito
 * antes de que arranque nada mas y leido despues no necesita atomico.
 */
extern int g_arm_clock_mhz;

/**
 * Pone la unidad de coma flotante en modo rapido, en el hilo que la llame.
 *
 * Activa dos bits del FPSCR:
 *   FZ (flush-to-zero): los denormales se tratan como cero.
 *   DN (default NaN):   cualquier NaN se convierte en el NaN por defecto.
 *
 * POR QUE. En el Cortex-A9, con FZ apagado una operacion que recibe o produce
 * un denormal no se resuelve en el camino rapido del hardware: se sale a codigo
 * de soporte y cuesta cientos de ciclos en vez de cuatro. Medido en consola:
 * ~7.500 ciclos por pixel sombreado para un camino de codigo de unas 1.000
 * instrucciones, o sea 7,5 ciclos por instruccion cuando este nucleo hace 1-1,5.
 * No se esta ejecutando demasiado: se esta esperando.
 *
 * Y denormales hay de sobra donde mirar: 1/w para geometria lejana, coordenadas
 * de textura cerca de cero, productos de baricentricas pequenas. f24 tiene menos
 * rango de exponente que float, asi que sus productos se cuelan en el rango
 * denormal de float32 con facilidad.
 *
 * La precision perdida es irrelevante aqui: la PICA200 YA descarta los
 * denormales en hardware (lo dice el propio comentario de Float::FromRaw en
 * pica_types.h), asi que tratarlos como cero se parece mas al 3DS real que no
 * hacerlo.
 *
 * El FPSCR es por hilo y se guarda en cada cambio de contexto, asi que hay que
 * llamarlo en CADA hilo: el principal y los tres del rasterizador.
 */
void VitaEnableFastFloatMode();

/**
 * True si la FPU ha visto algun denormal desde que arranco el proceso.
 *
 * Lee el bit IDC del FPSCR (denormal de entrada), que es acumulativo: una vez
 * que se pone, se queda. Sirve para confirmar o descartar la hipotesis de
 * arriba con un dato en vez de con una corazonada. Se consulta desde el overlay.
 */
bool VitaSawDenormals();

} // namespace Common

#endif // __PSVITA__
