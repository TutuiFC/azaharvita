// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/gxm_cg.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <malloc.h>
#include <mutex>
#include <pthread.h>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/io/dirent.h>
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include "common/common_types.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/vita_diag.h"
#include "video_core/shader/generator/cg_vs_shader_gen.h"

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

/**
 * MEMORIA A CERO Y LIBERACION DIFERIDA (0.2.3.7). Las caidas del compilador
 * con volcado, todas, de 0.2.0.x a 0.2.3.5 (los 17 que hay en ux0:data), son
 * en el MISMO sitio: SceShaccCg+0x65ea8, "ldr r0, [r0, #4]" de una funcion
 * que lee el tipo de un nodo, llamada justo despues de un "si el nodo no es
 * NULL" (SceShaccCg+0x7dc26). El nodo vale 0x18, 1 o 0x72646461 ("addr",
 * texto de un shader): un campo que nunca se escribio, o un bloque ya
 * liberado y reutilizado. Con el reservador de Sony en un proceso recien
 * arrancado esa memoria sale de paginas nuevas, a cero, y el campo vale NULL;
 * con el heap del emulador, lleno de restos, vale basura. Por eso cada bloque
 * sale a cero, y lo liberado espera en una cola (hasta kQuarantineBytes)
 * antes de volver al heap, para que un acceso tardio lea lo que habia.
 */
// 2 MB desde 0.3.0.1 (eran 8): en el 3D de Zafiro Alfa el heap se quedo sin
// sitio ("bad_alloc" con el compilador cayendose) y esto eran 6 MB de mas.
constexpr std::size_t kQuarantineBytes = 2u * 1024u * 1024u;
/// Reservas de la compilacion en curso: cuantas y cuanto (ver CompileCg).
u32 g_alloc_calls = 0;
u64 g_alloc_total = 0;
/// Las que salieron de g_small_free, sin pasar por newlib.
u32 g_alloc_reused = 0;

/**
 * BLOQUES PEQUENOS SIN PASAR POR NEWLIB (0.3.2.6). Un shader de vertices de
 * Pokemon Sol son ~700.000 reservas y otras tantas liberaciones (memalign, que
 * por dentro es un malloc y hasta dos free), y el malloc de newlib tiene UN
 * cerrojo para todo el proceso, sin herencia de prioridad: el compilador es el
 * hilo de menos prioridad del nucleo 2, y si le quitan la CPU con el cerrojo
 * cogido, el hilo de emulacion y el de la GPU esperan a que vuelva. Durante
 * una compilacion, lo que sale de la cuarentena va a una lista por tamano (de
 * 16 en 16 bytes, hasta kSmallBytes, como mucho kSmallFreeBytes en total) y
 * las reservas de ese tamano salen de ahi. Al acabar la compilacion se
 * devuelve todo a newlib. Todo esto, como el resto, con g_cg_mutex cogido.
 */
// 1 KB y 2 MB desde 0.3.2.9: con 512 bytes y 1 MB solo salian de la lista el
// 33-40 % de las reservas ("sin newlib" en crash.txt de 0.3.2.6).
constexpr std::size_t kSmallBytes = 1024;
constexpr std::size_t kSmallFreeBytes = 2u * 1024u * 1024u;
std::array<void*, kSmallBytes / 16> g_small_free{};
std::size_t g_small_free_bytes = 0;

/// Un bloque que ya no usa nadie: a su lista si cabe, si no a newlib.
void ReleaseBlock(void* pointer, std::size_t usable) {
    const std::size_t slot = usable / 16;
    if (g_tracking && slot >= 1 && slot <= g_small_free.size() &&
        g_small_free_bytes + usable <= kSmallFreeBytes) {
        // La lista va dentro de los propios bloques (al menos 16 bytes).
        *static_cast<void**>(pointer) = g_small_free[slot - 1];
        g_small_free[slot - 1] = pointer;
        g_small_free_bytes += usable;
        return;
    }
    std::free(pointer);
}

/// Devuelve a newlib todo lo de g_small_free.
void FlushSmallFree() {
    for (void*& head : g_small_free) {
        while (head != nullptr) {
            void* const next = *static_cast<void**>(head);
            std::free(head);
            head = next;
        }
    }
    g_small_free_bytes = 0;
}

/**
 * LA CUENTA SIN UN CONJUNTO DE PUNTEROS, Y LA COLA EN UN ANILLO (0.3.1.4). Cada
 * reserva del compilador entraba en un unordered_set (un malloc de nodo, y
 * rehacer la tabla al crecer) y cada liberacion en una deque: el doble de
 * llamadas a malloc que las del propio compilador, que son millones por shader
 * de vertices grande (1,7 millones y 35 s de CPU uno de 12 KB de Pokemon Sol en
 * crash.txt de 0.3.1.2). El conjunto solo servia para no descontar bloques de
 * otro modulo, y el modulo descargado no vuelve a liberar nada: la cuenta se
 * lleva con lo que dice malloc_usable_size. El anillo guarda lo mismo que la
 * deque, con tope de entradas ademas del de bytes.
 */
constexpr std::size_t kQuarantineSlots = 32768;
std::vector<std::pair<void*, std::size_t>> g_quarantine;
std::size_t g_quarantine_head = 0;
std::size_t g_quarantine_count = 0;
std::size_t g_quarantine_bytes = 0;

void* CgAlloc(unsigned int size) {
    /**
     * Sin NULL por presupuesto (0.2.0.8). libshacccg no comprueba lo que le
     * devuelve el reservador: un NULL a mitad de compilacion lo tumbaba con un
     * acceso a una direccion casi nula (volcados de 0.2.0.5 a 0.2.0.7, Pokemon
     * Sol al pasar al 3D). Pasado el presupuesto solo se apunta; si la memoria
     * se acaba de verdad, el NULL es inevitable.
     */
    if (g_tracking && g_compile_bytes + size > g_compile_budget) {
        g_compile_over_budget = true;
    }
    /**
     * Alineado y redondeado a 16 (0.2.0.7). malloc de newlib da 8, y el
     * compilador de Sony se caia SIEMPRE en el mismo punto, justo despues de
     * pedir memoria aqui (volcados de 0.2.0.1, 0.2.0.5 y 0.2.0.6: acceso a
     * memoria invalido dentro de libshacccg con la pila pasando por CgAlloc),
     * lo que encaja con accesos NEON de 16 bytes que exigen esa alineacion.
     */
    const std::size_t rounded = (size + 15u) & ~15u;
    // Un bloque de la lista 'slot' tiene al menos slot * 16 bytes utiles.
    const std::size_t slot = rounded / 16;
    void* pointer = nullptr;
    if (slot >= 1 && slot <= g_small_free.size() && g_small_free[slot - 1] != nullptr) {
        pointer = g_small_free[slot - 1];
        g_small_free[slot - 1] = *static_cast<void**>(pointer);
        g_small_free_bytes -= malloc_usable_size(pointer);
        g_alloc_reused++;
    } else {
        pointer = memalign(16, rounded);
        if (pointer == nullptr) {
            return nullptr;
        }
    }
    std::memset(pointer, 0, rounded);
    if (g_tracking) {
        g_alloc_calls++;
        g_alloc_total += rounded;
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
    const std::size_t size = malloc_usable_size(pointer);
    g_live_bytes = g_live_bytes > size ? g_live_bytes - size : 0;
    if (g_tracking) {
        g_compile_bytes = g_compile_bytes > size ? g_compile_bytes - size : 0;
    }
    if (g_quarantine.empty()) {
        try {
            g_quarantine.resize(kQuarantineSlots);
        } catch (...) {
            std::free(pointer);
            return;
        }
    }
    const auto free_oldest = [] {
        auto& oldest = g_quarantine[g_quarantine_head];
        ReleaseBlock(oldest.first, oldest.second);
        g_quarantine_bytes -= oldest.second;
        oldest = {};
        g_quarantine_head = (g_quarantine_head + 1) % kQuarantineSlots;
        g_quarantine_count--;
    };
    if (g_quarantine_count == kQuarantineSlots) {
        free_oldest();
    }
    g_quarantine[(g_quarantine_head + g_quarantine_count) % kQuarantineSlots] = {pointer, size};
    g_quarantine_count++;
    g_quarantine_bytes += size;
    while (g_quarantine_bytes > kQuarantineBytes && g_quarantine_count > 1) {
        free_oldest();
    }
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
/**
 * Topes de las notas de compilacion, por juego (0.3.1.4): eran de toda la
 * sesion, y en crash.txt de 0.3.1.3 Kirby gasto las sesenta de tiempos antes
 * de que Pokemon Sol, cuarto juego de la sesion, compilara nada. Los pone a
 * cero PreloadCgCache, al arrancar cada juego.
 */
std::atomic<u32> g_start_notes{0};
std::atomic<u32> g_time_notes{0};

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

/// Ficheros de la cache precargados en memoria (PreloadCgCache), por ruta, y
/// la lista de los que ha usado el juego actual. g_preload_mutex protege los
/// tres: los usan el hilo de la GPU y el de compilacion.
std::mutex g_preload_mutex;
std::unordered_map<std::string, std::vector<u8>> g_preloaded;
std::unordered_set<std::string> g_game_list;
std::string g_game_list_path;
/**
 * LO QUE HAY EN LA CARPETA DE LA CACHE, EN MEMORIA (0.3.1.5). Lo que no esta
 * precargado se buscaba con sceIoOpen, y casi siempre no esta (es un shader
 * nuevo): en una carpeta de mas de mil ficheros y con esta tarjeta, decenas de
 * ms por intento, y cada programa nuevo prueba dos. crash.txt de 0.3.1.4 en
 * Pokemon Sol: "vs nuevos 72 (8692 ms, traducir 2719 ms)", en el hilo de la
 * GPU. PreloadCgCache lista la carpeta una vez por juego (tambien con
 * g_preload_mutex) y lo que no aparece ahi no se abre; StoreCached apunta lo
 * que escribe.
 */
std::unordered_set<std::string> g_card_files;
bool g_card_index_ready = false;

/// Ver AddVsIndex. Tambien con g_preload_mutex.
constexpr u32 kVsIndexMagic = 0x49565A41u; // "AZVI"
struct VsIndexHeader {
    u32 magic;
    u32 version;
    char tag[40];
};
struct VsIndexRecord {
    u64 key;
    u64 hash_city;
    u64 hash_fnv;
    u32 source_size;
    u32 variant;
};
static_assert(sizeof(VsIndexRecord) == 32, "registro del indice de vertices");
std::unordered_map<u64, VsIndexRecord> g_vs_index;
std::string g_vs_index_path;
/// El fichero ya tiene la cabecera de esta compilacion del traductor.
bool g_vs_index_valid = false;
/// Abierto para anadir al final desde el primer registro de la partida.
SceUID g_vs_index_fd = -1;

VsIndexHeader CurrentVsIndexHeader() {
    VsIndexHeader header{};
    header.magic = kVsIndexMagic;
    header.version = 1;
    std::strncpy(header.tag, Pica::Shader::Generator::GXM::VsGeneratorTag(),
                 sizeof(header.tag) - 1);
    return header;
}

/// El indice del juego que arranca (desde PreloadCgCache).
void LoadVsIndex(u64 program_id) {
    const std::string path = fmt::format("{}/vs_{:016x}.idx", kCacheDir, program_id);
    std::unordered_map<u64, VsIndexRecord> records;
    bool valid = false;
    const SceUID fd = sceIoOpen(path.c_str(), SCE_O_RDONLY, 0);
    if (fd >= 0) {
        const VsIndexHeader want = CurrentVsIndexHeader();
        VsIndexHeader have{};
        if (sceIoRead(fd, &have, sizeof(have)) == static_cast<int>(sizeof(have)) &&
            std::memcmp(&have, &want, sizeof(have)) == 0) {
            valid = true;
            VsIndexRecord record{};
            // Uno posterior con la misma clave sustituye al anterior.
            while (sceIoRead(fd, &record, sizeof(record)) == static_cast<int>(sizeof(record))) {
                records[record.key] = record;
            }
        }
        sceIoClose(fd);
    }
    const std::size_t count = records.size();
    {
        const std::lock_guard lock{g_preload_mutex};
        if (g_vs_index_fd >= 0) {
            sceIoClose(g_vs_index_fd);
            g_vs_index_fd = -1;
        }
        g_vs_index = std::move(records);
        g_vs_index_path = path;
        g_vs_index_valid = valid;
    }
    Common::VitaNote("gxm cache",
                     fmt::format("indice de vertices: {} programas{}", count,
                                 valid ? "" : " (de otra compilacion del traductor: de cero)")
                         .c_str());
}

/// Apunta el fichero en la lista del juego, una vez.
void RememberForGame(const std::string& path) {
    const std::lock_guard lock{g_preload_mutex};
    if (g_game_list_path.empty() || !g_game_list.insert(path).second) {
        return;
    }
    const SceUID fd =
        sceIoOpen(g_game_list_path.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
    if (fd < 0) {
        return;
    }
    const std::string line = path + "\n";
    sceIoWrite(fd, line.data(), static_cast<SceSize>(line.size()));
    sceIoClose(fd);
}

void NoteCache(const std::string& text) {
    // Solo las primeras: crash.txt no es sitio para una linea por shader.
    if (g_cache_notes < 6) {
        g_cache_notes++;
        Common::VitaNote("gxm cache", text.c_str());
    }
}

/// Lo mismo que LoadCached, desde una copia del fichero en memoria.
const SceShaccCgCompileOutput* FromPreloaded(const CacheHeader& want,
                                             const std::vector<u8>& file) {
    CacheHeader have{};
    if (file.size() < sizeof(have)) {
        return nullptr;
    }
    std::memcpy(&have, file.data(), sizeof(have));
    if (have.magic != want.magic || have.version != want.version ||
        have.profile != want.profile || have.source_size != want.source_size ||
        have.hash_city != want.hash_city || have.hash_fnv != want.hash_fnv ||
        have.program_size == 0 || file.size() != sizeof(have) + have.program_size) {
        return nullptr;
    }
    void* memory = std::malloc(sizeof(SceShaccCgCompileOutput) + have.program_size);
    if (memory == nullptr) {
        return nullptr;
    }
    u8* program = static_cast<u8*>(memory) + sizeof(SceShaccCgCompileOutput);
    std::memcpy(program, file.data() + sizeof(have), have.program_size);
    auto* output = static_cast<SceShaccCgCompileOutput*>(memory);
    output->programData = program;
    output->programSize = have.program_size;
    output->diagnosticCount = 0;
    output->diagnostics = nullptr;
    {
        const std::lock_guard lock{g_outputs_mutex};
        g_cached_outputs.insert(output);
    }
    return output;
}

const SceShaccCgCompileOutput* LoadCachedFromCard(const CacheHeader& want,
                                                  const std::string& path);

const SceShaccCgCompileOutput* LoadCached(const CacheHeader& want, const std::string& path) {
    {
        const std::lock_guard lock{g_preload_mutex};
        const auto it = g_preloaded.find(path);
        if (it != g_preloaded.end()) {
            if (const SceShaccCgCompileOutput* output = FromPreloaded(want, it->second)) {
                return output;
            }
        }
        if (g_card_index_ready && g_card_files.count(path) == 0) {
            return nullptr;
        }
    }
    const SceShaccCgCompileOutput* output = LoadCachedFromCard(want, path);
    if (output != nullptr) {
        RememberForGame(path);
    }
    return output;
}

const SceShaccCgCompileOutput* LoadCachedFromCard(const CacheHeader& want,
                                                  const std::string& path) {
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
        return;
    }
    {
        const std::lock_guard lock{g_preload_mutex};
        g_card_files.insert(path);
    }
    RememberForGame(path);
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
/**
 * EL SHADER QUE SE ESTA COMPILANDO, EN LA TARJETA (0.2.0.6). libshacccg no solo
 * devuelve "internal error": con algun shader se cae entero (acceso a memoria
 * invalido dentro del modulo de Sony, volcados de 0.2.0.1 y 0.2.0.5 al pasar
 * al 3D de Pokemon Sol), y eso se lleva el proceso sin que se pueda apuntar
 * nada despues. Se escribe aqui antes de compilar y se borra al volver: si al
 * arrancar sigue ahi, la sesion anterior murio compilandolo, y va a la lista
 * negra. Ese shader se dibuja por el otro camino, pero el juego ya no se cae.
 */
constexpr char kCompilingPath[] = "ux0:/data/azahar/shadercache/cg_compilando.bin";
/// El codigo del que se compila, para estudiar el que lo tumbe (0.2.0.8).
constexpr char kCompilingSourcePath[] = "ux0:/data/azahar/shadercache/cg_compilando.cg";
/**
 * Los que ya tumbaron el compilador una vez con optimizacion (0.2.0.8): se
 * reintentan sin ella (menos memoria y otro camino dentro del compilador). Si
 * tambien asi se caen, a la lista negra.
 */
constexpr char kNoOptListPath[] = "ux0:/data/azahar/shadercache/cg_sin_optimizar.bin";
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
std::unordered_set<BadSource, BadSourceHash> g_no_opt_sources;
bool g_bad_loaded = false;
/**
 * UNA SEGUNDA OPORTUNIDAD A LOS DE FRAGMENTOS (0.3.2.1). crash.txt de 0.3.2.0,
 * Pokemon X: "fs: compilar shader" en los 1000 rechazos de cada ventana y
 * ningun "gxm compila" de fragmentos en toda la sesion: su shader estaba en la
 * lista negra y CompileCg lo devolvia nulo sin decir nada. El 44 % de sus
 * triangulos, por el rasterizador de software, y el juego a 4,5 fps esperando
 * a la GPU el 85 % del tiempo. Un error interno con el compilador recien
 * cargado (lo que llevaba ahi a los de fragmentos) se da en un nivel de
 * optimizacion: sin optimizar suele compilar. Cada uno de fragmentos de la
 * lista se reintenta UNA vez en O0, apuntado en kRetriedPath ANTES de compilar
 * para que, si colgara el compilador, no se repita en otra sesion. No se
 * reintenta el que consta que tumbo el proceso (kDiedPath, o su cg_murio_*.cg).
 */
constexpr char kRetriedPath[] = "ux0:/data/azahar/shadercache/cg_reintentado.bin";
constexpr char kDiedPath[] = "ux0:/data/azahar/shadercache/cg_murio.bin";
std::unordered_set<BadSource, BadSourceHash> g_retried_sources;
std::unordered_set<BadSource, BadSourceHash> g_died_sources;
/**
 * SOSPECHOSO (0.2.3.5). Sin volcado no se culpa al shader que se estaba
 * compilando (0.2.1.6), pero entonces uno que tumba el compilador sin dejar
 * volcado -- Pokemon Sol, al entrar en combate en 0.2.3.3: la sesion acaba
 * compilando uno de 12.699 bytes y no hay psp2core -- tumbaba el juego en
 * cada combate. Ahora queda apuntado aqui; si la siguiente sesion vuelve a
 * acabar compilando ESE MISMO, a la lista negra (a la CPU). Si se llega a
 * compilar bien, deja de ser sospechoso.
 */
constexpr char kSuspectPath[] = "ux0:/data/azahar/shadercache/cg_sospechoso.bin";
BadSource g_suspect{};
bool g_has_suspect = false;

/// Lo que se apunta en kCompilingPath antes de compilar.
struct CompilingMarker {
    BadSource source;
    u32 profile;
    u32 optimization;
    u32 source_size;
    u32 free_kb;
};

/**
 * LA MARCA CON EL FICHERO ABIERTO (0.3.1.8). Antes de cada compilacion se
 * abria, truncaba y cerraba la marca (y la fuente), y al acabar se borraba:
 * operaciones de metadatos en ux0 que con la tarjeta ocupada costaban 0,5-2 s
 * POR SHADER ("tarjeta antes" en crash.txt; un 12-33 % de cada compilacion en
 * Super Mario 3D Land). Ahora el fichero se queda abierto y se reescriben sus
 * 24 bytes en su sitio; "borrar" es escribir una marca vacia, que
 * LoadBadSources ignora. La fuente solo se guarda para los de vertices y los
 * grandes, que son los que han colgado el compilador alguna vez.
 */
SceUID g_marker_fd = -1;
constexpr u32 kSaveSourceMinBytes = 4096;

void WriteCompilingMarker(const CompilingMarker& marker) {
    if (g_marker_fd < 0) {
        g_marker_fd = sceIoOpen(kCompilingPath, SCE_O_WRONLY | SCE_O_CREAT, 0777);
        if (g_marker_fd < 0) {
            return;
        }
    }
    sceIoLseek(g_marker_fd, 0, SCE_SEEK_SET);
    sceIoWrite(g_marker_fd, &marker, sizeof(marker));
}

void ClearCompilingMarker() {
    if (g_marker_fd < 0) {
        return;
    }
    const CompilingMarker empty{};
    sceIoLseek(g_marker_fd, 0, SCE_SEEK_SET);
    sceIoWrite(g_marker_fd, &empty, sizeof(empty));
}

void AppendRecord(const char* path, const BadSource& record) {
    const SceUID fd = sceIoOpen(path, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
    if (fd < 0) {
        return;
    }
    sceIoWrite(fd, &record, sizeof(record));
    sceIoClose(fd);
}

void ReadRecords(const char* path, std::unordered_set<BadSource, BadSourceHash>& out) {
    const SceUID fd = sceIoOpen(path, SCE_O_RDONLY, 0);
    if (fd < 0) {
        return;
    }
    BadSource record{};
    // Tope por si el fichero estuviera mal: 4096 entradas son 64 KB.
    for (u32 i = 0; i < 4096; i++) {
        if (sceIoRead(fd, &record, sizeof(record)) != static_cast<int>(sizeof(record))) {
            break;
        }
        out.insert(record);
    }
    sceIoClose(fd);
}

void StoreBadSource(const BadSource& bad);

/// Un SceDateTime en un numero que se puede comparar (al segundo).
u64 DateKey(const SceDateTime& t) {
    return (((((static_cast<u64>(t.year) * 13 + t.month) * 32 + t.day) * 24 + t.hour) * 60 +
             t.minute) *
                60 +
            t.second);
}

/**
 * MURIO DE VERDAD COMPILANDO? (0.2.1.6). La marca de kCompilingPath se queda
 * igual si la sesion acaba por cualquier otra cosa a mitad de una compilacion:
 * el usuario cierra el juego (hay shaders de 25-30 s), o se cuelga la GPU. Ni
 * una sola de las nueve muertes apuntadas entre 0.2.0.8 y 0.2.1.5 tiene volcado
 * de la consola detras, y sin embargo sus shaders fueron a la lista negra (a la
 * CPU para siempre); uno se reintento en O0 y ese binario colgo la GPU. Cuando
 * el compilador se cae de verdad, el sistema escribe un psp2core-* en ux0:data
 * (los de 0.2.0.x estan ahi). Solo cuenta si es posterior a la marca y no es de
 * la GPU. Si no se puede leer la carpeta, se culpa como antes.
 */
bool CompilerCrashDumpSince(const SceIoStat& marker) {
    const SceUID dir = sceIoDopen("ux0:data");
    if (dir < 0) {
        return true;
    }
    const u64 since = DateKey(marker.st_mtime);
    bool found = false;
    SceIoDirent entry{};
    while (sceIoDread(dir, &entry) > 0) {
        if (std::strncmp(entry.d_name, "psp2core-", 9) == 0 &&
            std::strstr(entry.d_name, "GPUCRASH") == nullptr &&
            DateKey(entry.d_stat.st_mtime) >= since) {
            found = true;
            break;
        }
    }
    sceIoDclose(dir);
    return found;
}

void LoadBadSources() {
    if (g_bad_loaded) {
        return;
    }
    g_bad_loaded = true;
    ReadRecords(kBadListPath, g_bad_sources);
    ReadRecords(kNoOptListPath, g_no_opt_sources);
    ReadRecords(kRetriedPath, g_retried_sources);
    ReadRecords(kDiedPath, g_died_sources);
    SceIoStat marker_stat{};
    const bool marker_dated = sceIoGetstat(kCompilingPath, &marker_stat) >= 0;
    SceUID crashed_fd = sceIoOpen(kCompilingPath, SCE_O_RDONLY, 0);
    if (crashed_fd >= 0) {
        CompilingMarker probe{};
        const int got = sceIoRead(crashed_fd, &probe, sizeof(probe));
        if (got == static_cast<int>(sizeof(probe)) && probe.source == BadSource{} &&
            probe.source_size == 0) {
            // Marca vacia (ver WriteCompilingMarker): no murio compilando.
            sceIoClose(crashed_fd);
            crashed_fd = -1;
        } else {
            sceIoLseek(crashed_fd, 0, SCE_SEEK_SET);
        }
    }
    {
        const SceUID suspect_fd = sceIoOpen(kSuspectPath, SCE_O_RDONLY, 0);
        if (suspect_fd >= 0) {
            g_has_suspect = sceIoRead(suspect_fd, &g_suspect, sizeof(g_suspect)) ==
                            static_cast<int>(sizeof(g_suspect));
            sceIoClose(suspect_fd);
        }
    }
    if (crashed_fd >= 0 && marker_dated && !CompilerCrashDumpSince(marker_stat)) {
        CompilingMarker interrupted{{}, SCE_SHACCCG_PROFILE_VP, 1, 0, 0};
        sceIoRead(crashed_fd, &interrupted, sizeof(interrupted));
        sceIoClose(crashed_fd);
        sceIoRemove(kCompilingPath);
        if (g_has_suspect && g_suspect == interrupted.source) {
            StoreBadSource(interrupted.source);
            if (g_died_sources.insert(interrupted.source).second) {
                AppendRecord(kDiedPath, interrupted.source);
            }
            sceIoRemove(kSuspectPath);
            g_has_suspect = false;
            sceIoRename(kCompilingSourcePath,
                        fmt::format("{}/cg_murio_{:016x}.cg", kCacheDir,
                                    interrupted.source.hash_city)
                            .c_str());
            Common::VitaNote("gxm shader",
                             fmt::format("la sesion anterior acabo compilando un shader de {} "
                                         "bytes por SEGUNDA vez seguida, sin volcado: a la "
                                         "lista negra (sus lotes, por la CPU)",
                                         interrupted.source_size)
                                 .c_str());
        } else {
            sceIoRemove(kCompilingSourcePath);
            const SceUID suspect_fd =
                sceIoOpen(kSuspectPath, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
            if (suspect_fd >= 0) {
                sceIoWrite(suspect_fd, &interrupted.source, sizeof(interrupted.source));
                sceIoClose(suspect_fd);
            }
            g_suspect = interrupted.source;
            g_has_suspect = true;
            Common::VitaNote("gxm shader",
                             fmt::format("la sesion anterior acabo compilando un shader de {} "
                                         "bytes, sin volcado de la consola: se compilara otra "
                                         "vez; si vuelve a pasar con el, a la lista negra",
                                         interrupted.source_size)
                                 .c_str());
        }
    } else if (crashed_fd >= 0) {
        // El de 0.2.0.6 y 0.2.0.7 solo traia el hash: se toma como uno de
        // vertices optimizado, que es lo que se caia.
        CompilingMarker crashed{{}, SCE_SHACCCG_PROFILE_VP, 1, 0, 0};
        const int read = sceIoRead(crashed_fd, &crashed, sizeof(crashed));
        const bool read_ok =
            read == static_cast<int>(sizeof(crashed)) || read == static_cast<int>(sizeof(BadSource));
        sceIoClose(crashed_fd);
        sceIoRemove(kCompilingPath);
        if (read_ok) {
            // Los de vertices, nunca sin optimizar (0.2.1.4): en 0.2.1.3 uno de
            // Pokemon Sol tardo 106 s en O0 y su programa colgo la GPU de la
            // Vita ("GPU crash" del sistema). Van a la CPU.
            const bool retry = crashed.optimization != 0 &&
                               crashed.profile != SCE_SHACCCG_PROFILE_VP &&
                               g_no_opt_sources.insert(crashed.source).second;
            if (retry) {
                AppendRecord(kNoOptListPath, crashed.source);
            } else {
                StoreBadSource(crashed.source);
            }
            if (g_died_sources.insert(crashed.source).second) {
                AppendRecord(kDiedPath, crashed.source);
            }
            // El codigo, guardado con su hash para estudiarlo desde el PC.
            sceIoRename(kCompilingSourcePath,
                        fmt::format("{}/cg_murio_{:016x}.cg", kCacheDir, crashed.source.hash_city)
                            .c_str());
            Common::VitaNote(
                "gxm shader",
                fmt::format("el compilador murio en la sesion anterior con un shader de {} "
                            "({} bytes, O{}, heap libre {} KB): {}",
                            crashed.profile == SCE_SHACCCG_PROFILE_VP ? "vertices" : "fragmentos",
                            crashed.source_size, crashed.optimization, crashed.free_kb,
                            retry ? "se reintenta sin optimizar" : "a la lista negra")
                    .c_str());
        }
    }
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

/// Ver g_retried_sources. Con g_cg_mutex cogido.
bool FragmentRetryAllowed(const BadSource& source) {
    if (g_retried_sources.count(source) != 0 || g_died_sources.count(source) != 0) {
        return false;
    }
    SceIoStat stat{};
    const std::string died_source =
        fmt::format("{}/cg_murio_{:016x}.cg", kCacheDir, source.hash_city);
    return sceIoGetstat(died_source.c_str(), &stat) < 0;
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
        g_live_bytes = 0; // no se puede devolver: el modulo sigue vivo
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

/**
 * Los de vertices compilados en O0 por 0.2.0.8-0.2.1.3 (los de la lista de sin
 * optimizar) se quedaron en la cache de la tarjeta, y uno de ellos cuelga la
 * GPU: se borran y van a la lista negra antes de precargar nada (0.2.1.4).
 * Despues de la primera vez no queda fichero que borrar y no hace nada.
 */
void PurgeUnoptimizedVertexPrograms() {
    std::unordered_set<BadSource, BadSourceHash> no_opt;
    ReadRecords(kNoOptListPath, no_opt);
    u32 purged = 0;
    const std::lock_guard lock{g_cg_mutex};
    for (const BadSource& source : no_opt) {
        if (sceIoRemove(CachePath(source.hash_city, SCE_SHACCCG_PROFILE_VP).c_str()) >= 0) {
            StoreBadSource(source);
            purged++;
        }
    }
    if (purged != 0) {
        Common::VitaNote("gxm cache",
                         fmt::format("{} shaders de vertices sin optimizar borrados de la cache "
                                     "(colgaban la GPU): a la CPU",
                                     purged)
                             .c_str());
    }
}

u32 PreloadCgCache(u64 program_id, const std::function<void(u32, u32)>& progress) {
    PurgeUnoptimizedVertexPrograms();
    g_start_notes.store(0, std::memory_order_relaxed);
    g_time_notes.store(0, std::memory_order_relaxed);
    g_cache_notes = 0;
    g_mem_notes = 0;
    // Nada de lo anterior sirve para otro juego.
    std::vector<std::string> paths;
    {
        const std::lock_guard lock{g_preload_mutex};
        g_preloaded.clear();
        g_game_list.clear();
        g_game_list_path = fmt::format("{}/juego_{:016x}.lst", kCacheDir, program_id);
    }
    sceIoMkdir(kCacheDir, 0777);
    const SceUID list_fd = sceIoOpen(g_game_list_path.c_str(), SCE_O_RDONLY, 0);
    if (list_fd >= 0) {
        std::string text;
        char chunk[4096];
        int read = 0;
        while ((read = sceIoRead(list_fd, chunk, sizeof(chunk))) > 0) {
            text.append(chunk, static_cast<std::size_t>(read));
        }
        sceIoClose(list_fd);
        std::size_t start = 0;
        while (start < text.size()) {
            std::size_t end = text.find('\n', start);
            if (end == std::string::npos) {
                end = text.size();
            }
            if (end > start) {
                paths.emplace_back(text.substr(start, end - start));
            }
            start = end + 1;
        }
    }
    /**
     * Tope de memoria: los programas de un juego pesan de unos KB a decenas de
     * KB cada uno, pero la lista crece con cada partida. Lo que no quepa se
     * lee de la tarjeta como antes.
     */
    constexpr std::size_t kMaxPreloadBytes = 24u * 1024u * 1024u;
    std::size_t total_bytes = 0;
    u32 loaded = 0;
    const u32 total = static_cast<u32>(paths.size());
    for (u32 i = 0; i < total; i++) {
        if (progress && (i % 8) == 0) {
            progress(i, total);
        }
        const std::string& path = paths[i];
        const SceUID fd = sceIoOpen(path.c_str(), SCE_O_RDONLY, 0);
        if (fd < 0) {
            continue;
        }
        const SceOff size = sceIoLseek(fd, 0, SCE_SEEK_END);
        sceIoLseek(fd, 0, SCE_SEEK_SET);
        bool ok = size > static_cast<SceOff>(sizeof(CacheHeader)) &&
                  size < 4 * 1024 * 1024 &&
                  total_bytes + static_cast<std::size_t>(size) <= kMaxPreloadBytes;
        std::vector<u8> file;
        if (ok) {
            file.resize(static_cast<std::size_t>(size));
            ok = sceIoRead(fd, file.data(), static_cast<SceSize>(file.size())) ==
                 static_cast<int>(file.size());
        }
        sceIoClose(fd);
        if (!ok) {
            continue;
        }
        total_bytes += file.size();
        loaded++;
        const std::lock_guard lock{g_preload_mutex};
        g_game_list.insert(path);
        g_preloaded.emplace(path, std::move(file));
    }
    if (progress) {
        progress(total, total);
    }
    Common::VitaNote("gxm cache", fmt::format("precarga: {} de {} shaders, {} KB", loaded, total,
                                              total_bytes / 1024)
                                      .c_str());
    LoadVsIndex(program_id);
    {
        const unsigned long long list_begin = Common::VitaMicros();
        std::unordered_set<std::string> files;
        const SceUID dir = sceIoDopen(kCacheDir);
        if (dir >= 0) {
            SceIoDirent entry{};
            while (sceIoDread(dir, &entry) > 0) {
                files.insert(fmt::format("{}/{}", kCacheDir, entry.d_name));
            }
            sceIoDclose(dir);
        }
        const std::size_t count = files.size();
        {
            const std::lock_guard lock{g_preload_mutex};
            g_card_files = std::move(files);
            g_card_index_ready = dir >= 0;
        }
        Common::VitaNote("gxm cache", fmt::format("carpeta: {} ficheros en {} ms", count,
                                                  (Common::VitaMicros() - list_begin) / 1000)
                                          .c_str());
    }
    return loaded;
}

const SceShaccCgCompileOutput* LoadCgCache(SceShaccCgTargetProfile profile, const char* name,
                                           const char* source, bool from_card) {
    const std::size_t source_size = std::strlen(source);
    CacheHeader key{};
    key.magic = kCacheMagic;
    key.version = kCacheVersion;
    key.profile = static_cast<u32>(profile);
    key.source_size = static_cast<u32>(source_size);
    key.hash_city = Common::ComputeHash64(source, source_size);
    key.hash_fnv = Fnv1a64(source, source_size);
    const unsigned long long begin_us = Common::VitaMicros();
    const std::string path = CachePath(key.hash_city, profile);
    const SceShaccCgCompileOutput* cached = nullptr;
    if (from_card) {
        cached = LoadCached(key, path);
    } else {
        const std::lock_guard lock{g_preload_mutex};
        const auto it = g_preloaded.find(path);
        if (it != g_preloaded.end()) {
            cached = FromPreloaded(key, it->second);
        }
    }
    if (cached != nullptr) {
        NoteCache(fmt::format("{} leido de la cache en {} us", name,
                              Common::VitaMicros() - begin_us));
    }
    return cached;
}

const SceShaccCgCompileOutput* LoadCgCacheIndexed(u64 key, u32* variant) {
    VsIndexRecord record{};
    {
        const std::lock_guard lock{g_preload_mutex};
        const auto it = g_vs_index.find(key);
        if (it == g_vs_index.end()) {
            return nullptr;
        }
        record = it->second;
    }
    CacheHeader want{};
    want.magic = kCacheMagic;
    want.version = kCacheVersion;
    want.profile = static_cast<u32>(SCE_SHACCCG_PROFILE_VP);
    want.source_size = record.source_size;
    want.hash_city = record.hash_city;
    want.hash_fnv = record.hash_fnv;
    const SceShaccCgCompileOutput* output =
        LoadCached(want, CachePath(record.hash_city, SCE_SHACCCG_PROFILE_VP));
    if (output != nullptr) {
        *variant = record.variant;
    }
    return output;
}

void AddVsIndex(u64 key, const std::string& source, u32 variant) {
    VsIndexRecord record{};
    record.key = key;
    record.hash_city = Common::ComputeHash64(source.data(), source.size());
    record.hash_fnv = Fnv1a64(source.data(), source.size());
    record.source_size = static_cast<u32>(source.size());
    record.variant = variant;
    const std::lock_guard lock{g_preload_mutex};
    if (g_vs_index_path.empty()) {
        return;
    }
    const auto it = g_vs_index.find(key);
    if (it != g_vs_index.end() && std::memcmp(&it->second, &record, sizeof(record)) == 0) {
        return;
    }
    g_vs_index[key] = record;
    if (g_vs_index_fd < 0) {
        // Con otra cabecera (u otra compilacion del traductor) se empieza de cero.
        g_vs_index_fd = sceIoOpen(g_vs_index_path.c_str(),
                                  SCE_O_WRONLY | SCE_O_CREAT |
                                      (g_vs_index_valid ? SCE_O_APPEND : SCE_O_TRUNC),
                                  0777);
        if (g_vs_index_fd < 0) {
            return;
        }
        if (!g_vs_index_valid) {
            const VsIndexHeader header = CurrentVsIndexHeader();
            if (sceIoWrite(g_vs_index_fd, &header, sizeof(header)) !=
                static_cast<int>(sizeof(header))) {
                sceIoClose(g_vs_index_fd);
                g_vs_index_fd = -1;
                return;
            }
            g_vs_index_valid = true;
        }
    }
    sceIoWrite(g_vs_index_fd, &record, sizeof(record));
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
    const BadSource this_source{key.hash_city, key.hash_fnv};
    bool blacklist_retry = false;
    if (g_bad_sources.count(this_source) != 0) {
        if (profile == SCE_SHACCCG_PROFILE_VP || !FragmentRetryAllowed(this_source)) {
            static u32 blacklisted_notes = 0;
            if (blacklisted_notes < 8) {
                blacklisted_notes++;
                Common::VitaNote("gxm shader",
                                 fmt::format("{}: en la lista negra, no se compila ({} bytes)",
                                             name, source_size)
                                     .c_str());
            }
            return nullptr;
        }
        blacklist_retry = true;
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
    if (blacklist_retry) {
        // Apuntado justo antes de compilar: si no se llega a intentar (sin
        // heap, sin compilador), la oportunidad sigue ahi.
        g_retried_sources.insert(this_source);
        AppendRecord(kRetriedPath, this_source);
        Common::VitaNote("gxm shader",
                         fmt::format("{}: en la lista negra; se reintenta una vez sin optimizar "
                                     "({} bytes)",
                                     name, source_size)
                             .c_str());
    }
    const unsigned long long compile_begin_us = Common::VitaMicros();
    /**
     * CPU DE VERDAD FRENTE A RELOJ (0.3.0.1). Los de vertices grandes tardan
     * 70-120 s en Zafiro Alfa con el compilador ya solo en su nucleo
     * (0.2.3.8): o le siguen quitando la CPU, o el compilador es asi de lento
     * con ellos. runClocks del hilo dice cuanto ha corrido de verdad, y las
     * reservas, si se le va el tiempo en pedir memoria.
     */
    SceKernelThreadInfo info_before{};
    info_before.size = sizeof(info_before);
    const bool have_info = sceKernelGetThreadInfo(sceKernelGetThreadId(), &info_before) >= 0;
    g_alloc_calls = 0;
    g_alloc_total = 0;
    g_alloc_reused = 0;
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
    // Los de vertices nunca en O0 (0.2.1.6), aunque esten en la lista de antes:
    // uno de ellos colgo la GPU en 0.2.1.3.
    if (profile != SCE_SHACCCG_PROFILE_VP &&
        (blacklist_retry || g_no_opt_sources.count(this_source) != 0)) {
        options.optimizationLevel = 0;
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
    // Hasta 200 (eran 24): con menos, el que tumba el compilador no salia.
    if (g_start_notes.fetch_add(1, std::memory_order_relaxed) < 200) {
        Common::VitaNote("gxm compila", fmt::format("{}: {} bytes de codigo, heap libre {} KB, O{}",
                                                    name, g_source.size, FreeHeap() / 1024,
                                                    options.optimizationLevel)
                                            .c_str());
    }
    const SceShaccCgCompileOutput* output = nullptr;
    // Lo que tarda en la tarjeta lo de antes de compilar (0.3.1.4): la nota y
    // las dos marcas, en el tiempo de reloj de "compilado en".
    const unsigned long long marker_us = Common::VitaMicros() - compile_begin_us;
    const unsigned long long marker_begin_us = Common::VitaMicros();
    {
        const CompilingMarker compiling{BadSource{key.hash_city, key.hash_fnv},
                                        static_cast<u32>(profile),
                                        static_cast<u32>(options.optimizationLevel),
                                        static_cast<u32>(source_size),
                                        static_cast<u32>(FreeHeap() / 1024)};
        if (profile == SCE_SHACCCG_PROFILE_VP || source_size >= kSaveSourceMinBytes) {
            const SceUID source_fd =
                sceIoOpen(kCompilingSourcePath, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
            if (source_fd >= 0) {
                sceIoWrite(source_fd, source, static_cast<SceSize>(source_size));
                sceIoClose(source_fd);
            }
        }
        WriteCompilingMarker(compiling);
    }
    const unsigned long long card_us = marker_us + (Common::VitaMicros() - marker_begin_us);
    if (g_worker_running && pthread_equal(pthread_self(), g_worker_thread)) {
        output = sceShaccCgCompileProgram(&options, &g_callbacks, 0);
    } else {
        const Common::ScopedVitaStage stage{"compilando shader"};
        output = sceShaccCgCompileProgram(&options, &g_callbacks, 0);
    }
    g_tracking = false;
    FlushSmallFree();
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
        ClearCompilingMarker();
        if (internal) {
            /**
             * Ese codigo no se vuelve a mandar (ni en otra sesion) y el
             * compilador se recarga limpio para el siguiente. Uno de fragmentos
             * optimizado, en cambio, se reintenta sin optimizar (0.3.2.1): el
             * error interno es del optimizador, y en la lista negra todos sus
             * lotes iban por software para siempre.
             */
            if (profile != SCE_SHACCCG_PROFILE_VP && options.optimizationLevel != 0) {
                if (g_no_opt_sources.insert(this_source).second && fresh_compiler) {
                    AppendRecord(kNoOptListPath, this_source);
                }
            } else if (fresh_compiler) {
                StoreBadSource(this_source);
            } else {
                g_bad_sources.insert(this_source);
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
    const unsigned long long compile_ms = (Common::VitaMicros() - compile_begin_us) / 1000;
    NoteCache(fmt::format("{} compilado en {} ms", name, compile_ms));
    {
        // Aparte de las notas de la cache, que se gastan al arrancar (0.2.1.1):
        // es lo que tarda en aparecer un modelo con los shaders asincronos.
        if (g_time_notes.fetch_add(1, std::memory_order_relaxed) < 60) {
            SceKernelThreadInfo info_after{};
            info_after.size = sizeof(info_after);
            std::string cpu = "-";
            if (have_info && sceKernelGetThreadInfo(sceKernelGetThreadId(), &info_after) >= 0) {
                cpu = fmt::format("{} ms, nucleo {}, {} expulsiones",
                                  (info_after.runClocks - info_before.runClocks) / 1000,
                                  info_after.currentCpuId,
                                  info_after.threadPreemptCount - info_before.threadPreemptCount);
            }
            Common::VitaNote("gxm compila",
                             fmt::format("{}: compilado en {} ms (CPU {}; {} reservas, {} sin "
                                         "newlib, {} MB pedidos; tarjeta antes {} ms)",
                                         name, compile_ms, cpu, g_alloc_calls, g_alloc_reused,
                                         g_alloc_total >> 20, card_us / 1000)
                                 .c_str());
        }
    }
    StoreCached(key, cache_path, *output);
    if (g_has_suspect && g_suspect == BadSource{key.hash_city, key.hash_fnv}) {
        sceIoRemove(kSuspectPath);
        g_has_suspect = false;
    }
    // 0.1.7.9: la salida pasa a un bloque nuestro y el compilador suelta todo
    // lo suyo. Si no hay memoria ni para la copia, se devuelve la original
    // (como antes de 0.1.7.9) y el compilador se queda como estaba.
    const SceShaccCgCompileOutput* copy = CopyOutput(*output);
    if (copy == nullptr) {
        ClearCompilingMarker();
        return output;
    }
    sceShaccCgDestroyCompileOutput(output);
    ClearCompilingMarker();
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
std::atomic<SceUID> g_worker_uid{-1};
std::atomic<unsigned long long> g_worker_busy_since{0};
std::atomic<u32> g_jobs_queued{0};
std::atomic<bool> g_worker_boosted{false};
/// El trabajo en marcha es grande (CgJob::heavy): ese no se sube (0.2.2.7).
std::atomic<bool> g_worker_heavy{false};
/// Atado al nucleo 2 (0.2.3.8): no se sube nunca.
bool g_worker_pinned = false;
/// Justo por encima de los ayudantes y del hilo de la GPU (159).
constexpr int kCgBoostedPriority = Common::kVitaPriorityHelper - 1;

void* CgWorkerMain(void*) {
    /**
     * EL NUCLEO 2 PARA EL (0.2.3.8). Suelto en cualquier nucleo (0.2.0.5) la
     * consola no lo movia al que quedaba libre: en un combate de Pokemon Sol
     * (crash.txt de 0.2.3.7) un shader de vertices de 18 KB tardo 45 s y otros
     * hasta 2 minutos, con el nucleo 0 parado el 77 % del tiempo, y mientras
     * tanto sus lotes por la CPU a 1-3 FPS. Los ayudantes del sombreado de
     * vertices ya no usan el 2 (ver PicaCore::ParallelShading); el hilo de
     * emulacion esta en el 0 y el de la GPU en el 1.
     */
    {
        const int rc = sceKernelChangeThreadCpuAffinityMask(sceKernelGetThreadId(),
                                                            SCE_KERNEL_CPU_MASK_USER_2);
        Common::VitaNote("afinidad", rc >= 0 ? "compilador de shaders: nucleo 2"
                                             : "compilador de shaders: no se pudo atar");
        g_worker_pinned = rc >= 0;
    }
    // La mas baja (0.2.0.2): compilar un shader son segundos de CPU sin soltar
    // el nucleo, y comparte nucleo con ayudantes a los que espera el hilo de
    // la GPU en cada lote. Ver kVitaPriorityBackground.
    Common::VitaSetThreadPriority(Common::kVitaPriorityBackground, "compilador de shaders");
    g_worker_uid.store(sceKernelGetThreadId(), std::memory_order_release);
    while (true) {
        std::shared_ptr<CgJob> job;
        {
            std::unique_lock lock{g_jobs_mutex};
            g_jobs_ready.wait(lock, [] { return !g_jobs.empty(); });
            /**
             * LOS GRANDES, LOS ULTIMOS (0.2.2.3). Uno de 16-23 KB pasa minutos
             * aqui y el hilo es uno solo: por orden de llegada, todo lo que
             * viniera detras esperaba esos minutos, los de fragmentos tambien
             * (sin ellos el lote no se dibuja). Primero cualquiera que no sea
             * grande; los grandes, cuando no queda otra cosa.
             */
            auto pick = std::find_if(g_jobs.begin(), g_jobs.end(),
                                     [](const std::shared_ptr<CgJob>& queued) {
                                         return !queued->heavy;
                                     });
            if (pick == g_jobs.end()) {
                pick = g_jobs.begin();
            }
            job = std::move(*pick);
            g_jobs.erase(pick);
            g_jobs_queued.store(static_cast<u32>(g_jobs.size()), std::memory_order_relaxed);
        }
        g_worker_heavy.store(job->heavy, std::memory_order_relaxed);
        g_worker_busy_since.store(Common::VitaMicros(), std::memory_order_relaxed);
        job->started_us.store(Common::VitaMicros(), std::memory_order_relaxed);
        for (std::size_t i = 0; i < job->sources.size(); i++) {
            // Los de vertices dejan de pedirse con el compilador roto; los de
            // fragmentos no (ver RecoverFromInternalError).
            if (job->profile == SCE_SHACCCG_PROFILE_VP && CgPoisoned()) {
                break;
            }
            const char* name = i < job->names.size() ? job->names[i] : job->name;
            const SceShaccCgCompileOutput* output =
                CompileCg(job->profile, name, job->sources[i].c_str());
            if (output != nullptr) {
                job->output = output;
                job->used_variant = job->variants[i];
                break;
            }
        }
        g_worker_busy_since.store(0, std::memory_order_relaxed);
        job->done.store(true, std::memory_order_release);
        if (g_worker_boosted.exchange(false, std::memory_order_acq_rel)) {
            sceKernelChangeThreadPriority(sceKernelGetThreadId(),
                                          Common::kVitaPriorityBackground);
        }
    }
    return nullptr;
}

} // Anonymous namespace

/**
 * EL COMPILADOR SE QUEDABA SIN CPU (0.2.1.3). crash.txt de Pokemon Sol en
 * 0.2.1.2: los shaders de vertices del 3D tardaban 30-60 s cada uno (uno que
 * pillo la CPU libre, 2,6 s), porque mientras no estan, sus lotes van por la
 * CPU y los ayudantes que los sombrean (159) ocupan los nucleos al 97 %, por
 * encima del compilador (191). Cuanto mas tardaba, mas tiempo a 1 FPS. Solo
 * cuando un lote lleva esperando de mas se le sube: la consola se para un
 * momento en vez de ir a 1 FPS minutos, y el resultado queda en la cache.
 */
void CgMarkStarved() {
    /**
     * Uno grande no se sube (0.2.2.7): son minutos, y subido se llevaba un
     * nucleo entero por delante de la GPU y de los ayudantes (New Super Mario
     * Bros. 2 en 0.2.2.6: 600 ms por fotograma mientras compilaba). Sus lotes
     * van por la CPU como antes de 0.2.2.3; compila con lo que sobre.
     *
     * Y desde 0.2.3.8 no se sube ninguno: subido por encima del hilo de
     * emulacion (160) lo dejaba parado mientras compilaba ("congelado: 6 s sin
     * avanzar, en 'esperando a la gpu'" en un combate de Pokemon Sol). Con el
     * nucleo 2 para el solo (ver CgWorkerMain) no le hace falta.
     */
    if (g_worker_boosted.load(std::memory_order_relaxed) ||
        g_worker_heavy.load(std::memory_order_relaxed) || g_worker_pinned) {
        return;
    }
    const SceUID thread = g_worker_uid.load(std::memory_order_acquire);
    if (thread < 0 || g_worker_boosted.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    sceKernelChangeThreadPriority(thread, kCgBoostedPriority);
    static bool noted = false;
    if (!noted) {
        noted = true;
        Common::VitaNote("gxm compila", "lotes esperando: compilador por encima de los ayudantes");
    }
}

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
    job->submitted_us = Common::VitaMicros();
    if (urgent) {
        g_jobs.push_front(std::move(job));
    } else {
        g_jobs.push_back(std::move(job));
    }
    g_jobs_queued.store(static_cast<u32>(g_jobs.size()), std::memory_order_relaxed);
    g_jobs_ready.notify_one();
}

CgActivity GetCgActivity() {
    return CgActivity{g_worker_busy_since.load(std::memory_order_relaxed),
                      g_jobs_queued.load(std::memory_order_relaxed)};
}

} // namespace Gxm
