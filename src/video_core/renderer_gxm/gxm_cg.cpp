// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/gxm_cg.h"

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <malloc.h>
#include <mutex>
#include <pthread.h>
#include <set>
#include <string>
#include <unordered_set>
#include <fmt/format.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include "common/common_types.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/vita_diag.h"

/// Tamano del heap de libc, definido en citra_vita/main.cpp (AZAHAR_HEAP_MB).
extern unsigned int _newlib_heap_size_user;

namespace Gxm {

namespace {

constexpr const char* kShacccgPaths[] = {
    "ur0:/data/libshacccg.suprx",
    "ux0:/data/libshacccg.suprx",
};

SceUID g_module = -1;
/// Compilaciones hechas con el modulo cargado ahora (0.1.8.5, ver StoreBadSource).
u32 g_compiles_since_load = 0;
SceShaccCgSourceFile g_source{};
SceShaccCgCallbackList g_callbacks{};
char g_status[64] = "sin iniciar";
/// El compilador ha devuelto un "internal error" alguna vez. Ver CgPoisoned.
/// Atomico desde 0.1.9.6: lo escribe el hilo de compilacion.
std::atomic<bool> g_poisoned{false};
/**
 * Todo lo del compilador (el modulo, su memoria, la cache, las listas) pasa
 * por aqui (0.1.9.6): desde que hay un hilo de compilacion, CompileCg puede
 * llamarse desde dos hilos. Recursivo porque CompileCg llama a EnsureCgReady
 * y a ReleaseCgOutput.
 */
std::recursive_mutex g_cg_mutex;
unsigned long long g_last_attempt_us = 0;

/// Un intento por segundo como mucho cuando falla: sin esto, cada lote cuyo
/// shader no se pueda compilar paga una carga de modulo fallida, y eso fue una
/// caida de 450 ms a 13.734 ms de fotograma.
constexpr unsigned long long kRetryIntervalUs = 1000000;

SceShaccCgSourceFile* OpenSource(const char*, const SceShaccCgSourceLocation*,
                                 const SceShaccCgCompileOptions*, const char**) {
    return &g_source;
}

/**
 * RECUPERARSE DE UN "fatal internal error" (0.1.5.9).
 *
 * Medido en 0.1.5.8: un shader de vertices dio "fatal internal error", el
 * siguiente de fragmentos dio lo mismo (el compilador se queda roto para
 * siempre), todo el 3D paso a software (0.6 FPS) y poco despues la emulacion
 * murio con bad_alloc: el compilador se rompe a medias y deja SIN DEVOLVER la
 * memoria que habia pedido.
 *
 * Arreglo: se apunta cada bloque que el compilador pide DURANTE una
 * compilacion. Si la compilacion acaba en error interno, se descarga el modulo
 * (su estado global roto se va con el), se devuelve la memoria que dejo
 * perdida y se vuelve a cargar limpio en la compilacion siguiente. Los shaders
 * de fragmentos siguen funcionando; los de vertices traducidos no se vuelven a
 * mandar en esta sesion (CgPoisoned), porque el que lo rompio volveria a salir.
 *
 * La memoria SOLO se devuelve si la descarga sale bien: si el modulo siguiera
 * cargado, su estado podria apuntar a esos bloques.
 */
bool g_tracking = false;
std::atomic<u32> g_generation{0};

/**
 * TODA LA MEMORIA VIVA DEL COMPILADOR (0.1.8.1).
 *
 * Hasta 0.1.8.0 solo se apuntaba lo que el compilador pedia DURANTE una
 * compilacion, y se olvidaba al acabar bien. crash.txt de 0.1.8.0: una
 * compilacion lleva el heap de 171 a 189 MB, y sceShaccCgReleaseCompiler no
 * devuelve NADA (189,7 MB despues). Esos 18 MB se quedan en el estado del
 * modulo; con la siguiente compilacion grande el heap no da y el compilador
 * muere por dentro (volcado de 0.1.8.0: PC dentro de libshacccg).
 *
 * Ahora se apunta todo lo que pide mientras esta cargado, y se puede
 * DESCARGAR el modulo. Desde 0.1.8.5 solo se mide (picos y lo retenido, ver
 * NoteCompileMemory): liberar lo apuntado tras descargar resulto peligroso.
 */
std::unordered_set<void*> g_compile_allocs;
std::size_t g_live_bytes = 0;
/// Lo maximo que ha tenido vivo durante la compilacion en curso.
std::size_t g_peak_bytes = 0;

/**
 * PRESUPUESTO DE UNA COMPILACION (0.1.8.0, rebajado en 0.1.8.1).
 *
 * Volcado de 0.1.7.9: el heap se acababa DENTRO de sceShaccCgCompileProgram y
 * el bad_alloc del insert de aqui no podia atravesar libshacccg (es C):
 * std::unexpected y abort(). CgAlloc no lanza nunca. El tope ya no es la via
 * normal de salir de un apuro (un NULL a mitad de compilacion probablemente lo
 * tumba igual): lo que evita el apuro es no EMPEZAR a compilar sin memoria de
 * sobra (kMinFreeToCompile). El tope solo deja kEmulatorReserve para que la
 * emulacion no muera por un bad_alloc suyo si el compilador se desboca.
 */
constexpr std::size_t kEmulatorReserve = 8u * 1024u * 1024u;
std::size_t g_compile_bytes = 0;
std::size_t g_compile_budget = static_cast<std::size_t>(-1);
bool g_compile_over_budget = false;

void* CgAlloc(unsigned int size) {
    if (g_tracking && g_compile_bytes + size > g_compile_budget) {
        g_compile_over_budget = true;
        return nullptr;
    }
    void* pointer = std::malloc(size);
    if (pointer == nullptr) {
        return nullptr;
    }
    try {
        g_compile_allocs.insert(pointer);
    } catch (...) {
        std::free(pointer);
        g_compile_over_budget = true;
        return nullptr;
    }
    const std::size_t usable = malloc_usable_size(pointer);
    g_live_bytes += usable;
    if (g_live_bytes > g_peak_bytes) {
        g_peak_bytes = g_live_bytes;
    }
    if (g_tracking) {
        g_compile_bytes += usable;
    }
    return pointer;
}

void CgFree(void* pointer) {
    if (pointer == nullptr) {
        return;
    }
    if (g_compile_allocs.erase(pointer) != 0) {
        const std::size_t size = malloc_usable_size(pointer);
        g_live_bytes = g_live_bytes > size ? g_live_bytes - size : 0;
        if (g_tracking) {
            g_compile_bytes = g_compile_bytes > size ? g_compile_bytes - size : 0;
        }
    }
    std::free(pointer);
}

/**
 * CACHE DE SHADERS COMPILADOS EN LA TARJETA (0.1.5.8).
 *
 * POR QUE. libshacccg compila en la propia consola y es lento: cada estado
 * nuevo de la GPU del 3DS (combinacion de TEV, luces, formato de vertices...)
 * es un shader nuevo, y al pasar de una pantalla 2D a una escena 3D aparecen
 * muchos de golpe. El usuario lo vio como una carga "realmente lenta" entre el
 * menu 2D y el 3D de Pokemon Zafiro Alfa.
 *
 * COMO. El binario GXP que devuelve el compilador se guarda en
 * ux0:/data/azahar/shadercache/<hash>_<perfil>.gxp, y la proxima vez que se
 * pida el MISMO codigo fuente se lee de ahi en vez de compilar. La primera vez
 * sigue costando lo mismo; las siguientes (esa escena otra vez, o la partida
 * de manana) casi nada.
 *
 * SEGURIDAD. La clave es el codigo fuente ENTERO: la cabecera guarda dos
 * hashes distintos de 64 bits (CityHash y FNV-1a), su longitud, el perfil y
 * una version del formato. Un fichero que no cuadre en todo se ignora y se
 * compila como siempre (y se reescribe). Si el fichero no se puede leer o
 * escribir, tambien se compila como siempre: la cache nunca es necesaria.
 */
constexpr char kCacheDir[] = "ux0:/data/azahar/shadercache";
constexpr u32 kCacheMagic = 0x58475A41u; // "AZGX"
constexpr u32 kCacheVersion = 1;

struct CacheHeader {
    u32 magic;
    u32 version;
    u32 profile;
    u32 source_size;
    u64 hash_city;
    u64 hash_fnv;
    u32 program_size;
    u32 reserved;
};
static_assert(sizeof(CacheHeader) == 40, "cabecera de la cache de shaders");

/// Salidas que vienen de la cache: se liberan con free, no con shacccg.
std::set<const SceShaccCgCompileOutput*> g_cached_outputs;
/// Protege g_cached_outputs (0.1.9.6): leer de la cache de la tarjeta no
/// necesita el compilador, asi que no espera a la compilacion en curso.
std::mutex g_outputs_mutex;
/// El hilo de compilacion: su etapa no se apunta para el vigilante, que mira
/// solo el hilo de emulacion (0.1.9.8).
pthread_t g_worker_thread{};
bool g_worker_running = false;
bool g_cache_dir_ready = false;
u32 g_cache_notes = 0;

u64 Fnv1a64(const char* data, std::size_t size) {
    u64 hash = 0xCBF29CE484222325ull;
    for (std::size_t i = 0; i < size; i++) {
        hash ^= static_cast<u8>(data[i]);
        hash *= 0x100000001B3ull;
    }
    return hash;
}

std::string CachePath(u64 hash_city, SceShaccCgTargetProfile profile) {
    return fmt::format("{}/{:016x}_{}.gxp", kCacheDir, hash_city, static_cast<u32>(profile));
}

void NoteCache(const std::string& text) {
    // Solo las primeras: crash.txt no es sitio para una linea por shader.
    if (g_cache_notes < 6) {
        g_cache_notes++;
        Common::VitaNote("gxm cache", text.c_str());
    }
}

const SceShaccCgCompileOutput* LoadCached(const CacheHeader& want, const std::string& path) {
    const SceUID fd = sceIoOpen(path.c_str(), SCE_O_RDONLY, 0);
    if (fd < 0) {
        return nullptr;
    }
    CacheHeader have{};
    bool ok = sceIoRead(fd, &have, sizeof(have)) == static_cast<int>(sizeof(have)) &&
              have.magic == want.magic && have.version == want.version &&
              have.profile == want.profile && have.source_size == want.source_size &&
              have.hash_city == want.hash_city && have.hash_fnv == want.hash_fnv &&
              have.program_size > 0 && have.program_size < 4u * 1024u * 1024u;
    void* memory = nullptr;
    if (ok) {
        // La salida y el programa en el mismo bloque: el programa detras de la
        // estructura (16 bytes), asi que queda alineado igual que malloc.
        memory = std::malloc(sizeof(SceShaccCgCompileOutput) + have.program_size);
        ok = memory != nullptr;
    }
    if (ok) {
        u8* program = static_cast<u8*>(memory) + sizeof(SceShaccCgCompileOutput);
        ok = sceIoRead(fd, program, have.program_size) == static_cast<int>(have.program_size);
    }
    sceIoClose(fd);
    if (!ok) {
        std::free(memory);
        return nullptr;
    }
    auto* output = static_cast<SceShaccCgCompileOutput*>(memory);
    output->programData = static_cast<u8*>(memory) + sizeof(SceShaccCgCompileOutput);
    output->programSize = have.program_size;
    output->diagnosticCount = 0;
    output->diagnostics = nullptr;
    {
        const std::lock_guard lock{g_outputs_mutex};
        g_cached_outputs.insert(output);
    }
    return output;
}

/**
 * Copia en un bloque nuestro, con el mismo formato que LoadCached, de una
 * salida del compilador (0.1.7.9). Asi la salida de libshacccg se puede
 * destruir enseguida sin que
 * el programa registrado en el parcheador apunte a memoria suya.
 */
const SceShaccCgCompileOutput* CopyOutput(const SceShaccCgCompileOutput& source) {
    void* memory = std::malloc(sizeof(SceShaccCgCompileOutput) + source.programSize);
    if (memory == nullptr) {
        return nullptr;
    }
    auto* output = static_cast<SceShaccCgCompileOutput*>(memory);
    u8* program = static_cast<u8*>(memory) + sizeof(SceShaccCgCompileOutput);
    std::memcpy(program, source.programData, source.programSize);
    output->programData = program;
    output->programSize = source.programSize;
    output->diagnosticCount = 0;
    output->diagnostics = nullptr;
    {
        const std::lock_guard lock{g_outputs_mutex};
        g_cached_outputs.insert(output);
    }
    return output;
}

std::size_t HeapInUse() {
    return mallinfo().uordblks;
}

std::size_t FreeHeap() {
    const std::size_t total = _newlib_heap_size_user;
    const std::size_t used = HeapInUse();
    return total > used ? total - used : 0;
}

/**
 * Libre minimo para EMPEZAR una compilacion (0.1.8.5). Medido en 0.1.8.4 con
 * el shader de piel de Zafiro Alfa especializado: picos de 1,6 a 10,8 MB por
 * compilacion, y el compilador retiene menos de 0,6 MB. Los umbrales de
 * 0.1.8.3 (85 MB para los de vertices) dejaban sin compilar shaders que cabian
 * de sobra en cuanto el heap pasaba de 178 MB en uso.
 */
constexpr std::size_t kMinFreeToCompile = 24u * 1024u * 1024u;
constexpr std::size_t kMinFreeToCompileVs = 32u * 1024u * 1024u;

/**
 * LO QUE GASTA CADA COMPILACION, A crash.txt (0.1.7.9). Las ocho primeras.
 *
 * 0.1.8.5: SIN sceShaccCgReleaseCompiler ni descargar el compilador despues de
 * compilar. Las dos cosas se metieron en 0.1.7.9 y 0.1.8.3 contra un supuesto
 * agotamiento de memoria, y las dos estaban en la secuencia de los dos crashes
 * siguientes: 0.1.8.0 murio DENTRO del compilador justo despues de un
 * ReleaseCompiler, y 0.1.8.4 murio dentro de malloc, con el heap corrupto,
 * tras cuatro descargas en las que se liberaba toda la memoria que el modulo
 * habia pedido (si algo suyo la seguia usando, la corrompia). Y lo medido en
 * 0.1.8.4 dice que no hacian falta: picos de 1,6-10,8 MB y menos de 0,6 MB
 * retenidos. Se vuelve a como compilaba de 0.1.5.9 a 0.1.7.8.
 */
u32 g_mem_notes = 0;

void NoteCompileMemory(const char* name, std::size_t before, std::size_t live_before) {
    if (g_mem_notes >= 8) {
        return;
    }
    g_mem_notes++;
    const std::size_t after = HeapInUse();
    const std::size_t peak = g_peak_bytes > live_before ? g_peak_bytes - live_before : 0;
    Common::VitaNote("gxm mem",
                     fmt::format("{}: heap en uso {} KB, despues {} KB; pico de la compilacion "
                                 "{} KB, el compilador retiene {} KB",
                                 name, before / 1024, after / 1024, peak / 1024,
                                 g_live_bytes / 1024)
                         .c_str());
}

void StoreCached(CacheHeader header, const std::string& path,
                 const SceShaccCgCompileOutput& output) {
    if (!g_cache_dir_ready) {
        sceIoMkdir(kCacheDir, 0777);
        g_cache_dir_ready = true;
    }
    header.program_size = output.programSize;
    const SceUID fd = sceIoOpen(path.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
    if (fd < 0) {
        return;
    }
    const bool ok =
        sceIoWrite(fd, &header, sizeof(header)) == static_cast<int>(sizeof(header)) &&
        sceIoWrite(fd, output.programData, output.programSize) ==
            static_cast<int>(output.programSize);
    sceIoClose(fd);
    if (!ok) {
        // A medias no sirve: que la proxima vez se compile.
        sceIoRemove(path.c_str());
    }
}

/**
 * EL SHADER QUE ROMPE EL COMPILADOR, A LA TARJETA (0.1.5.9).
 *
 * En 0.1.5.8 un shader de vertices traducido dejo el compilador de Cg sin
 * compilar nada mas (todo a software, 0.6 FPS), y en crash.txt solo cabe el
 * primer mensaje, no el shader. Sin el codigo fuente no hay forma de saber
 * que construccion le sienta mal. Se guardan el primer fallo y el primer
 * "internal error" (pueden ser distintos) enteros, con sus mensajes.
 */
/// Cuantos se guardan por sesion (0.1.7.3): con las variantes, un mismo shader
/// puede fallar de varias formas, y cada una dice algo distinto.
u32 g_dumped_failure = 0;
u32 g_dumped_internal = 0;

void DumpFailedSource(const char* name, const char* source, const SceShaccCgCompileOutput* output,
                      bool internal) {
    if ((internal ? g_dumped_internal : g_dumped_failure) >= 4) {
        return;
    }
    (internal ? g_dumped_internal : g_dumped_failure)++;
    const SceUID fd = sceIoOpen("ux0:/data/azahar/cg_error.txt",
                                SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
    if (fd < 0) {
        return;
    }
    std::string text = fmt::format("===== {} ({}) =====\n", name,
                                   internal ? "ERROR INTERNO" : "fallo de compilacion");
    if (output != nullptr) {
        for (int i = 0; i < output->diagnosticCount && i < 8; i++) {
            const auto& diag = output->diagnostics[i];
            text += fmt::format("linea {}: {}\n",
                                diag.location != nullptr ? diag.location->lineNumber : 0,
                                diag.message != nullptr ? diag.message : "(sin mensaje)");
        }
    }
    text += "----- codigo fuente -----\n";
    text += source;
    text += "\n\n";
    sceIoWrite(fd, text.data(), static_cast<SceSize>(text.size()));
    sceIoClose(fd);
    Common::VitaNote("gxm shader", "codigo del shader que fallo en ux0:/data/azahar/cg_error.txt");
}

/**
 * LISTA NEGRA DE CODIGOS QUE ROMPEN EL COMPILADOR (0.1.7.3).
 *
 * Antes de 0.1.7.3 el primer "internal error" dejaba los shaders de vertices
 * traducidos apagados TODA la sesion (g_poisoned), y el que lo rompio salia
 * otra vez en la siguiente partida: en la cinematica de Rubi Omega eso eran
 * 134 de 142 lotes sombreados en la CPU, ~84 ms de ~190 por vblank. Ahora el
 * generador tiene variantes (rasterizer_gxm.cpp prueba otra si una falla), asi
 * que lo que se aparta es solo ESE codigo fuente: sus dos hashes van a la
 * tarjeta y no se le vuelve a dar al compilador, ni en esta sesion ni en las
 * siguientes. El resto de shaders sigue compilando con normalidad.
 */
/**
 * 0.1.8.5: fichero NUEVO (cg_roto2.bin). En el viejo habia 24 codigos, y la nota
 * de 0.1.5.8 de aqui arriba dice por que muchos podian ser inocentes: tras un
 * error interno el compilador se queda roto y los siguientes fallan igual. Un
 * shader de fragmentos apuntado aqui se salta para siempre y TODOS sus lotes
 * se rasterizan por software (0.1.8.4: "gxm skip: compilar shader" y despues
 * 465 ms por vblank en los vertices). Ahora solo se apunta lo que falla con el
 * compilador recien cargado; lo demas solo se salta en esta sesion.
 */
constexpr char kBadListPath[] = "ux0:/data/azahar/shadercache/cg_roto2.bin";
struct BadSource {
    u64 hash_city;
    u64 hash_fnv;
    bool operator==(const BadSource& other) const {
        return hash_city == other.hash_city && hash_fnv == other.hash_fnv;
    }
};
struct BadSourceHash {
    std::size_t operator()(const BadSource& bad) const {
        return static_cast<std::size_t>(bad.hash_city ^ (bad.hash_fnv >> 7));
    }
};
std::unordered_set<BadSource, BadSourceHash> g_bad_sources;
bool g_bad_loaded = false;

void LoadBadSources() {
    if (g_bad_loaded) {
        return;
    }
    g_bad_loaded = true;
    const SceUID fd = sceIoOpen(kBadListPath, SCE_O_RDONLY, 0);
    if (fd < 0) {
        return;
    }
    BadSource bad{};
    // Tope por si el fichero estuviera mal: 4096 entradas son 64 KB.
    for (u32 i = 0; i < 4096; i++) {
        if (sceIoRead(fd, &bad, sizeof(bad)) != static_cast<int>(sizeof(bad))) {
            break;
        }
        g_bad_sources.insert(bad);
    }
    sceIoClose(fd);
    if (!g_bad_sources.empty()) {
        Common::VitaNote("gxm shader", fmt::format("{} shaders que rompen el compilador se "
                                                   "saltan (lista de la tarjeta)",
                                                   g_bad_sources.size())
                                           .c_str());
    }
}

void StoreBadSource(const BadSource& bad) {
    if (!g_bad_sources.insert(bad).second) {
        return;
    }
    if (!g_cache_dir_ready) {
        sceIoMkdir(kCacheDir, 0777);
        g_cache_dir_ready = true;
    }
    const SceUID fd =
        sceIoOpen(kBadListPath, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
    if (fd < 0) {
        return;
    }
    sceIoWrite(fd, &bad, sizeof(bad));
    sceIoClose(fd);
}

/// Ver g_tracking: descarga el compilador roto y devuelve lo que perdio.
void RecoverFromInternalError() {
    if (g_module < 0) {
        return;
    }
    /**
     * Dos topes (0.1.8.8). Con 0.1.8.7 se llego al unico que habia (8) en el 3D
     * de Zafiro Alfa y el compilador se quedaba cargado y ROTO: tampoco
     * compilaban los shaders de fragmentos y casi todo se rasterizaba por
     * software ("vs compilador roto", ts 5436 contra tg 66). Ahora, a partir de
     * kVsRecoveries recargas, se dejan de traducir shaders de vertices (son
     * los que lo rompen), pero se sigue recargando para que los de fragmentos
     * compilen. kMaxRecoveries acota la memoria que no se devuelve (ver abajo):
     * de ahi en adelante si se queda roto.
     */
    constexpr u32 kVsRecoveries = 6;
    constexpr u32 kMaxRecoveries = 16;
    if (g_generation + 1 >= kVsRecoveries && !g_poisoned) {
        g_poisoned = true;
        Common::VitaNote("gxm shader", "demasiados errores internos: no se traducen mas "
                                       "shaders de vertices en esta sesion");
    }
    if (g_generation >= kMaxRecoveries) {
        g_compile_allocs.clear();
        g_live_bytes = 0;
        return;
    }
    int status = 0;
    const int rc = sceKernelStopUnloadModule(g_module, 0, nullptr, 0, nullptr, &status);
    if (rc < 0) {
        Common::VitaNote("gxm shader", fmt::format("no se pudo descargar el compilador roto "
                                                   "({:#x}): sigue sin compilar",
                                                   static_cast<u32>(rc))
                                           .c_str());
        g_compile_allocs.clear(); // no se puede devolver: el modulo sigue vivo
        g_live_bytes = 0;
        g_poisoned = true;
        return;
    }
    /**
     * 0.1.8.5: la memoria del modulo NO se libera, solo se olvida. Liberarla
     * (0.1.5.9-0.1.8.4) da por hecho que tras la descarga nada la usa, y el
     * crash de 0.1.8.4 (malloc con el heap corrupto despues de varias
     * descargas) dice que no es seguro. Se pierde el pico de una compilacion
     * fallida (unos MB), y esto solo pasa con errores internos.
     */
    const std::size_t lost = g_live_bytes;
    g_compile_allocs.clear();
    g_live_bytes = 0;
    g_module = -1;
    g_last_attempt_us = 0; // recargar ya en la siguiente compilacion
    g_compiles_since_load = 0;
    g_generation++;
    std::strcpy(g_status, "recargando");
    Common::VitaNote("gxm shader",
                     fmt::format("compilador descargado tras el error interno ({} KB sin "
                                 "devolver); se recarga limpio",
                                 lost / 1024)
                         .c_str());
}

} // Anonymous namespace

u32 CgGeneration() {
    return g_generation;
}

bool EnsureCgReady() {
    const std::lock_guard lock{g_cg_mutex};
    if (g_module >= 0) {
        return true;
    }
    const unsigned long long now_us = Common::VitaMicros();
    if (g_last_attempt_us != 0 && now_us - g_last_attempt_us < kRetryIntervalUs) {
        return false;
    }
    g_last_attempt_us = now_us;
    for (const char* path : kShacccgPaths) {
        g_module = sceKernelLoadStartModule(path, 0, nullptr, 0, nullptr, nullptr);
        if (g_module >= 0) {
            LOG_INFO(Render, "GXM: libshacccg cargado desde {}", path);
            break;
        }
        LOG_ERROR(Render, "GXM: cargar {} fallo con {:#x}", path, static_cast<u32>(g_module));
    }
    if (g_module < 0) {
        const auto written = fmt::format_to_n(g_status, sizeof(g_status) - 1,
                                              "sin libshacccg ({:#x})",
                                              static_cast<u32>(g_module));
        *written.out = 0;
        return false;
    }
    sceShaccCgSetDefaultAllocator(&CgAlloc, &CgFree);
    g_compiles_since_load = 0;
    sceShaccCgInitializeCallbackList(&g_callbacks, SCE_SHACCCG_TRIVIAL);
    g_callbacks.openFile = &OpenSource;
    std::strcpy(g_status, "listo");
    return true;
}

const char* CgStatus() {
    return g_status;
}

bool CgPoisoned() {
    return g_poisoned;
}

bool CgHeapLow() {
    const std::size_t total = _newlib_heap_size_user;
    const std::size_t reserve = static_cast<std::size_t>(kCgHeapReserveMB) * 1024u * 1024u;
    return HeapInUse() + reserve > total;
}

const SceShaccCgCompileOutput* LoadCgCache(SceShaccCgTargetProfile profile, const char* name,
                                           const char* source) {
    const std::size_t source_size = std::strlen(source);
    CacheHeader key{};
    key.magic = kCacheMagic;
    key.version = kCacheVersion;
    key.profile = static_cast<u32>(profile);
    key.source_size = static_cast<u32>(source_size);
    key.hash_city = Common::ComputeHash64(source, source_size);
    key.hash_fnv = Fnv1a64(source, source_size);
    const unsigned long long begin_us = Common::VitaMicros();
    const SceShaccCgCompileOutput* cached = LoadCached(key, CachePath(key.hash_city, profile));
    if (cached != nullptr) {
        NoteCache(fmt::format("{} leido de la cache en {} us", name,
                              Common::VitaMicros() - begin_us));
    }
    return cached;
}

const SceShaccCgCompileOutput* CompileCg(SceShaccCgTargetProfile profile, const char* name,
                                         const char* source) {
    // Cache en la tarjeta (0.1.5.8): si este codigo fuente ya se compilo
    // alguna vez, el binario se lee de disco. Ver CacheHeader. Sin el cerrojo
    // del compilador (0.1.9.6): no hay que esperar a que acabe otra compilacion.
    const std::size_t source_size = std::strlen(source);
    CacheHeader key{};
    key.magic = kCacheMagic;
    key.version = kCacheVersion;
    key.profile = static_cast<u32>(profile);
    key.source_size = static_cast<u32>(source_size);
    key.hash_city = Common::ComputeHash64(source, source_size);
    key.hash_fnv = Fnv1a64(source, source_size);
    const std::string cache_path = CachePath(key.hash_city, profile);
    {
        const unsigned long long begin_us = Common::VitaMicros();
        if (const SceShaccCgCompileOutput* cached = LoadCached(key, cache_path)) {
            NoteCache(fmt::format("{} leido de la cache en {} us", name,
                                  Common::VitaMicros() - begin_us));
            return cached;
        }
    }
    const std::lock_guard lock{g_cg_mutex};
    LoadBadSources();
    if (g_bad_sources.count(BadSource{key.hash_city, key.hash_fnv}) != 0) {
        return nullptr; // ya rompio el compilador una vez: ni se intenta
    }
    /**
     * NO EMPEZAR SIN MEMORIA DE SOBRA (0.1.8.1). Una compilacion puede pedir
     * decenas de MB por el camino, y quedarse sin ellos a mitad tumba el
     * compilador por dentro. Si no hay kMinFreeToCompile libres, primero se
     * recupera lo que retenga el compilador; si ni asi, este shader no se
     * compila ahora (el lote se dibuja por el otro camino) y se reintentara
     * cuando lo pida otro lote.
     */
    const bool vertex = profile == SCE_SHACCCG_PROFILE_VP;
    const std::size_t min_free = vertex ? kMinFreeToCompileVs : kMinFreeToCompile;
    if (FreeHeap() < min_free) {
        static u32 skip_notes = 0;
        if (skip_notes < 4) {
            skip_notes++;
            Common::VitaNote("gxm mem", fmt::format("{}: sin compilar, solo quedan {} KB de heap",
                                                    name, FreeHeap() / 1024)
                                            .c_str());
        }
        return nullptr;
    }
    if (!EnsureCgReady()) {
        return nullptr;
    }
    const unsigned long long compile_begin_us = Common::VitaMicros();
    g_source.fileName = name;
    g_source.text = source;
    g_source.size = static_cast<SceUInt32>(std::strlen(source));

    SceShaccCgCompileOptions options{};
    sceShaccCgInitializeCompileOptions(&options);
    options.mainSourceFile = name;
    options.targetProfile = profile;
    options.entryFunctionName = "main";
    /**
     * Los de vertices, con poca optimizacion (0.1.9.6). Son los programas de
     * la PICA traducidos: miles de lineas que el compilador tardaba de varios
     * segundos a mas de medio minuto en optimizar, con picos de hasta 27 MB y
     * hasta 20 MB retenidos despues. En la GPU de la Vita los vertices de un
     * juego de 3DS no cuestan casi nada; lo caro era compilarlos. Los de
     * fragmentos (se ejecutan por pixel) siguen con la optimizacion de serie.
     */
    if (profile == SCE_SHACCCG_PROFILE_VP) {
        options.optimizationLevel = 1;
    }

    const std::size_t heap_before = HeapInUse();
    g_compile_budget = FreeHeap() - kEmulatorReserve;
    g_compile_bytes = 0;
    g_compile_over_budget = false;
    g_peak_bytes = g_live_bytes;
    const std::size_t live_before = g_live_bytes;
    g_compiles_since_load++;
    const bool fresh_compiler = g_compiles_since_load == 1;
    g_tracking = true;
    // Que se compila y de que tamano, a crash.txt (0.1.9.1): en 0.1.9.0 el
    // juego se quedaba congelado dentro de una compilacion sin saber cual.
    static u32 start_notes = 0;
    if (start_notes < 24) {
        start_notes++;
        Common::VitaNote("gxm compila", fmt::format("{}: {} bytes de codigo, heap libre {} KB, O{}",
                                                    name, g_source.size, FreeHeap() / 1024,
                                                    options.optimizationLevel)
                                            .c_str());
    }
    const SceShaccCgCompileOutput* output = nullptr;
    if (g_worker_running && pthread_equal(pthread_self(), g_worker_thread)) {
        output = sceShaccCgCompileProgram(&options, &g_callbacks, 0);
    } else {
        const Common::ScopedVitaStage stage{"compilando shader"};
        output = sceShaccCgCompileProgram(&options, &g_callbacks, 0);
    }
    g_tracking = false;
    g_compile_budget = static_cast<std::size_t>(-1);
    if (g_compile_over_budget) {
        static u32 budget_notes = 0;
        if (budget_notes < 4) {
            budget_notes++;
            Common::VitaNote("gxm mem", fmt::format("{}: el compilador agoto su presupuesto "
                                                    "({} KB); la compilacion se corta",
                                                    name, g_compile_bytes / 1024)
                                            .c_str());
        }
    }
    if (output == nullptr || output->programData == nullptr) {
        /**
         * EL MENSAJE DEL COMPILADOR VA A crash.txt, no solo al registro.
         *
         * Antes aqui solo se anotaba el NOMBRE del fichero ("azahar_gxm_f.cg")
         * y los diagnosticos iban a LOG_ERROR, que desde la consola no se ve.
         * O sea que un shader que no compilaba dejaba escrito que no compilaba
         * y ni una pista de por que, y sin esa pista no hay forma de arreglarlo
         * sin un depurador. Se anota el primero, que es el que dice donde esta
         * el error; los demas suelen ser consecuencia suya.
         */
        char note[192];
        int noted = 0;
        bool internal = false;
        if (output != nullptr) {
            for (int i = 0; i < output->diagnosticCount; i++) {
                if (output->diagnostics[i].message != nullptr &&
                    std::strstr(output->diagnostics[i].message, "internal error") != nullptr) {
                    internal = true;
                }
            }
        }
        DumpFailedSource(name, source, output, internal);
        if (output != nullptr) {
            for (int i = 0; i < output->diagnosticCount && i < 4; i++) {
                const char* text = output->diagnostics[i].message != nullptr
                                       ? output->diagnostics[i].message
                                       : "(sin mensaje)";
                LOG_ERROR(Render, "GXM {}: {}", name, text);
                /**
                 * Los DOS primeros van a crash.txt, no solo el primero.
                 *
                 * Cada arreglo de un shader generado cuesta una prueba en
                 * consola, y si hay dos errores independientes verlos de uno en
                 * uno cuesta dos vueltas. Del tercero en adelante ya suelen ser
                 * consecuencia de los anteriores y solo ensucian el fichero.
                 */
                if (noted < 2 && output->diagnostics[i].message != nullptr) {
                    const auto written =
                        fmt::format_to_n(note, sizeof(note) - 1, "{}: {}", name, text);
                    *written.out = 0;
                    Common::VitaNote("gxm shader", note);
                    noted++;
                }
            }
        } else {
            LOG_ERROR(Render, "GXM: el compilador no devolvio nada para {}", name);
        }
        if (noted == 0) {
            const auto written =
                fmt::format_to_n(note, sizeof(note) - 1, "{}: sin diagnostico", name);
            *written.out = 0;
            Common::VitaNote("gxm shader", note);
        }
        if (output != nullptr) {
            sceShaccCgDestroyCompileOutput(output);
        }
        if (internal) {
            // Ese codigo no se vuelve a mandar (ni en otra sesion) y el
            // compilador se recarga limpio para el siguiente.
            if (fresh_compiler) {
                StoreBadSource(BadSource{key.hash_city, key.hash_fnv});
            } else {
                g_bad_sources.insert(BadSource{key.hash_city, key.hash_fnv});
            }
            RecoverFromInternalError();
        } else if (g_compile_over_budget) {
            // Sin memoria a mitad de compilacion su estado puede quedar a
            // medias: se descarga y se recarga limpio, como tras un error
            // interno, pero el codigo NO va a la lista negra (no lo rompio el
            // shader, sino la falta de heap en ese momento).
            RecoverFromInternalError();
        } else {
            // Un error normal (de sintaxis) deja el compilador sano: solo se
            // suelta su memoria. Con el error interno no se le llama: el
            // modulo esta roto y RecoverFromInternalError ya lo ha descargado.
            NoteCompileMemory(name, heap_before, live_before);
        }
        return nullptr;
    }
    // Lo que tarda compilar (para saber cuanto ahorra la cache) y a la cache.
    NoteCache(fmt::format("{} compilado en {} ms", name,
                          (Common::VitaMicros() - compile_begin_us) / 1000));
    StoreCached(key, cache_path, *output);
    // 0.1.7.9: la salida pasa a un bloque nuestro y el compilador suelta todo
    // lo suyo. Si no hay memoria ni para la copia, se devuelve la original
    // (como antes de 0.1.7.9) y el compilador se queda como estaba.
    const SceShaccCgCompileOutput* copy = CopyOutput(*output);
    if (copy == nullptr) {
        return output;
    }
    sceShaccCgDestroyCompileOutput(output);
    NoteCompileMemory(name, heap_before, live_before);
    /**
     * TOPE DE LO QUE RETIENE EL COMPILADOR (0.1.9.1). Desde 0.1.8.5 no se le
     * pide que suelte nada (ver NoteCompileMemory), y con los shaders de piel
     * de Zafiro Alfa retenia 8 MB mas por compilacion: 8, 16, 24... y bad_alloc.
     * Pasado el tope no se traducen mas shaders de vertices (son los grandes);
     * los de fragmentos, que son los que hacen falta para dibujar, si.
     */
    constexpr std::size_t kMaxRetained = 12u * 1024u * 1024u;
    if (g_live_bytes > kMaxRetained && !g_poisoned) {
        g_poisoned = true;
        Common::VitaNote("gxm mem", fmt::format("el compilador retiene {} KB: no se traducen mas "
                                                "shaders de vertices en esta sesion",
                                                g_live_bytes / 1024)
                                        .c_str());
    }
    return copy;
}

void ReleaseCgOutput(const SceShaccCgCompileOutput* output) {
    if (output == nullptr) {
        return;
    }
    // Las que vinieron de la cache son un malloc nuestro (0.1.5.8).
    {
        const std::lock_guard lock{g_outputs_mutex};
        const auto it = g_cached_outputs.find(output);
        if (it != g_cached_outputs.end()) {
            g_cached_outputs.erase(it);
            std::free(const_cast<SceShaccCgCompileOutput*>(output));
            return;
        }
    }
    const std::lock_guard lock{g_cg_mutex};
    sceShaccCgDestroyCompileOutput(output);
}

namespace {

std::mutex g_jobs_mutex;
std::condition_variable g_jobs_ready;
std::deque<std::shared_ptr<CgJob>> g_jobs;
bool g_worker_started = false;

void* CgWorkerMain(void*) {
    // En los nucleos 1 y 2, con los otros ayudantes: el 0 es el de la emulacion.
    Common::VitaPinThreadToUserCore(1, "compilador de shaders");
    while (true) {
        std::shared_ptr<CgJob> job;
        {
            std::unique_lock lock{g_jobs_mutex};
            g_jobs_ready.wait(lock, [] { return !g_jobs.empty(); });
            job = std::move(g_jobs.front());
            g_jobs.pop_front();
        }
        for (std::size_t i = 0; i < job->sources.size(); i++) {
            // Los de vertices dejan de pedirse con el compilador roto; los de
            // fragmentos no (ver RecoverFromInternalError).
            if (job->profile == SCE_SHACCCG_PROFILE_VP && CgPoisoned()) {
                break;
            }
            const SceShaccCgCompileOutput* output =
                CompileCg(job->profile, job->name, job->sources[i].c_str());
            if (output != nullptr) {
                job->output = output;
                job->used_variant = job->variants[i];
                break;
            }
        }
        job->done.store(true, std::memory_order_release);
    }
    return nullptr;
}

} // Anonymous namespace

void CgSubmit(std::shared_ptr<CgJob> job, bool urgent) {
    std::lock_guard lock{g_jobs_mutex};
    if (!g_worker_started) {
        g_worker_started = true;
        /**
         * Pila de 1 MB: el compilador corria hasta ahora en el hilo principal
         * y no se sabe cuanta pila necesita con un shader grande. Mejor que
         * sobre (sale del presupuesto de la aplicacion, que tiene ~80 MB libres
         * fuera del heap).
         */
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 1024 * 1024);
        if (pthread_create(&g_worker_thread, &attr, &CgWorkerMain, nullptr) == 0) {
            g_worker_running = true;
            pthread_detach(g_worker_thread);
        } else {
            g_worker_started = false;
        }
        pthread_attr_destroy(&attr);
        if (!g_worker_started) {
            // Sin hilo: se compila aqui mismo, como antes.
            job->done.store(false, std::memory_order_relaxed);
            for (std::size_t i = 0; i < job->sources.size(); i++) {
                const SceShaccCgCompileOutput* output =
                    CompileCg(job->profile, job->name, job->sources[i].c_str());
                if (output != nullptr) {
                    job->output = output;
                    job->used_variant = job->variants[i];
                    break;
                }
            }
            job->done.store(true, std::memory_order_release);
            return;
        }
        Common::VitaNote("gxm compila", "hilo de compilacion en segundo plano listo");
    }
    if (urgent) {
        g_jobs.push_front(std::move(job));
    } else {
        g_jobs.push_back(std::move(job));
    }
    g_jobs_ready.notify_one();
}

} // namespace Gxm
