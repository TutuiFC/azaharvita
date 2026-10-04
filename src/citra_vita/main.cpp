// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
//
// Punto de entrada de Azahar en PS Vita.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <pthread.h>
#include <string>
#include <system_error>
#include <thread>
#include <typeinfo>
#include <vector>

#include <psp2/ctrl.h>
#include <psp2/io/dirent.h>
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <malloc.h>
#include <psp2/power.h>
#include <vita2d.h>

#include <cryptopp/integer.h>

#include "citra_vita/vita_input.h"
#include "citra_vita/vita_jit_probe.h"
#include "citra_vita/vita_version.h"
#include "citra_vita/vita_window.h"
#include "common/file_util.h"
#include "common/logging/backend.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/vita_diag.h"
#include "core/arm/dyncom/arm_dyncom_jit.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "core/core.h"
#include "core/frontend/applets/default_applets.h"
#include "core/frontend/image_interface.h"
#include "core/hle/service/cfg/cfg.h"
#include "core/hle/service/service.h"
#include "core/loader/loader.h"
#include "video_core/gpu.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_gxm/gxm_cg.h"
#include "video_core/renderer_gxm/rasterizer_gxm.h"
#include "video_core/shader/generator/cg_vs_shader_gen.h"
#include "video_core/renderer_software/sw_rasterizer.h"

#ifndef AZAHAR_HEAP_MB
#define AZAHAR_HEAP_MB 192
#endif

extern "C" {
/**
 * Tamano del heap de la aplicacion.
 *
 * La newlib del VitaSDK reserva 128 MB por defecto, que ni siquiera dan para
 * la FCRAM del 3DS (128 MB justos). Sumando VRAM (6 MB), la RAM extra del
 * New 3DS (4 MB), el DSP (512 KB) y las tablas de paginas (unos 5 MB por
 * proceso: 2^20 entradas de puntero mas sus atributos), el minimo ronda los
 * 150 MB, y la cache del rasterizador crece por encima.
 *
 * libc reserva este bloque de una pieza ANTES de main(): si no cabe en el
 * presupuesto de la aplicacion, no arranca y no hay forma de avisar por
 * pantalla. Por eso el valor por defecto se queda dentro del presupuesto
 * normal de una app de Vita (~256 MB) en vez de estirarse hasta el maximo.
 * Se ajusta con -DAZAHAR_HEAP_MB=N al configurar.
 */
unsigned int _newlib_heap_size_user = AZAHAR_HEAP_MB * 1024u * 1024u;

/**
 * Tamano de pila de cada hilo creado con pthread.
 *
 * La pila de un hilo de Vita sale del presupuesto de memoria de la aplicacion,
 * no del heap de libc, y ese presupuesto va muy justo: al cargar un juego solo
 * quedan unos 25 MB. Si la reserva falla, pthread_create devuelve un error que
 * libstdc++ convierte en una excepcion, y como salta dentro de un hilo se lleva
 * el proceso por delante.
 *
 * 256 KB es de sobra: midiendo los volcados, el hilo que mas gastaba usaba
 * 9 KB. Lo que importa es que sea un valor conocido y contenido en vez del que
 * traiga la biblioteca por defecto.
 */
unsigned int _pthread_stack_default_user = 256 * 1024;
}

namespace {

/// Version del port, visible en el menu y en la pantalla de arranque. Se define
/// en vita_version.h, que es el unico sitio donde se escribe (el overlay del
/// juego usa la misma constante).
using VitaFrontend::kVersion;

constexpr char kUserDir[] = "ux0:/data/azahar/";
constexpr char kRomDir[] = "ux0:/data/azahar/roms";

constexpr unsigned int kColorText = 0xFFFFFFFF;
constexpr unsigned int kColorDim = 0xFF909090;
constexpr unsigned int kColorAccent = 0xFF4FC3F7; // ARGB -> naranja azahar en BGR
constexpr unsigned int kColorBackground = 0xFF201810;

struct RomEntry {
    std::string name;
    std::string path;
};

bool HasRomExtension(const std::string& name) {
    static const char* kExtensions[] = {".3ds", ".3dsx", ".cci", ".cxi", ".app", ".cia", ".elf"};
    for (const char* ext : kExtensions) {
        const std::size_t ext_len = std::char_traits<char>::length(ext);
        if (name.size() <= ext_len) {
            continue;
        }
        if (std::equal(ext, ext + ext_len, name.end() - ext_len,
                       [](char a, char b) { return a == std::tolower(b); })) {
            return true;
        }
    }
    return false;
}

std::vector<RomEntry> ScanRoms() {
    std::vector<RomEntry> roms;

    const SceUID dir = sceIoDopen(kRomDir);
    if (dir < 0) {
        return roms;
    }

    SceIoDirent entry{};
    while (sceIoDread(dir, &entry) > 0) {
        if (SCE_S_ISDIR(entry.d_stat.st_mode)) {
            continue;
        }
        const std::string name = entry.d_name;
        if (!HasRomExtension(name)) {
            continue;
        }
        roms.push_back({name, std::string(kRomDir) + "/" + name});
    }
    sceIoDclose(dir);

    std::sort(roms.begin(), roms.end(),
              [](const RomEntry& a, const RomEntry& b) { return a.name < b.name; });
    return roms;
}

void CreateDirectories() {
    sceIoMkdir("ux0:/data", 0777);
    sceIoMkdir("ux0:/data/azahar", 0777);
    sceIoMkdir(kRomDir, 0777);
}

/// Ajustes adaptados a lo que la Vita puede dar.
void ConfigureSettings() {
    // Motor grafico por defecto: GXM. Presenta con el chip de la consola los
    // fotogramas que sigue rasterizando el camino de software; si el
    // presentador no puede arrancar (falta libshacccg.suprx, falla la
    // compilacion de los shaders, no hay memoria de GPU), degrada solo al
    // camino de vita2d sin tocar la emulacion. En el menu, SELECT alterna entre
    // GXM y el renderer de software entero para comparar.
    Settings::values.graphics_api = Settings::GraphicsAPI::GXM;

    // Sin JIT: dynarmic solo tiene backends de x86-64 y ARM64, y la Vita es
    // ARMv7. Se usa el interprete dyncom.
    Settings::values.use_cpu_jit = false;
    Settings::values.use_shader_jit = false;
    // Shader de vertices en la GPU (0.1.4.6): PicaCore le pregunta a
    // RasterizerGXM::AccelerateDrawBatch antes de ejecutar el interprete. Lo que
    // no se pueda acelerar sigue por la CPU lote a lote.
    Settings::values.use_hw_shader = true;
    Settings::values.use_disk_shader_cache = false;
    Settings::values.async_shader_compilation = false;

    // Modo New 3DS: se decide al compilar (-DAZAHAR_NEW_3DS=ON), no aqui.
    //
    // Dejo de ser "imposible" al activar el modo de memoria ampliada: la FCRAM
    // del New 3DS son 256 MB frente a 128, y el presupuesto pasa de ~256 a
    // ~365 MB. Lo que sigue haciendo falta es un heap de >= 288 MB, y eso ata el
    // arranque a que ATTRIBUTE2=12 funcione. El CMakeLists de esta carpeta tiene
    // la cuenta completa y las dos comprobaciones que impiden encenderlo a medias.
#ifdef AZAHAR_NEW_3DS
    Settings::values.is_new_3ds = true;
#else
    Settings::values.is_new_3ds = false;
#endif

    Settings::values.resolution_factor = 1;
    Settings::values.layout_option = Settings::LayoutOption::Default;

    // Sin 3D estereoscopico: la Vita tiene una sola pantalla y el frontend solo
    // dibuja el ojo izquierdo. Algunos juegos siguen dibujando el ojo derecho
    // aunque el deslizador 3D este a cero, y todo ese trabajo se tira.
    // RightEyeDisabler bloquea esos envios a la GPU emulada antes de que lleguen
    // a rasterizarse.
    Settings::values.disable_right_eye_render = true;

    // Sin limite de velocidad: el emulador no va a llegar al 100% de todas
    // formas, y limitarlo solo anadiria esperas.
    Settings::values.frame_limit = 0;

    Settings::values.audio_emulation = Settings::AudioEmulation::HLE;

    // Audio desactivado de momento. El sink de Vita levanta su propio hilo, y
    // cada hilo consume presupuesto de memoria de la aplicacion, que va al
    // limite. Emulando a pocos fotogramas por segundo el sonido no se entiende
    // igualmente, asi que primero que vaya la imagen. Para recuperarlo basta
    // con poner SinkType::Vita aqui.
    /**
     * SONIDO ENCENDIDO (0.1.4.9, lo pidio el usuario).
     *
     * Hasta aqui la salida era Null: el DSP se emulaba igual (el juego lo
     * necesita para avanzar) pero nadie escuchaba. VitaSink (vita_sink.cpp)
     * abre el puerto principal de audio a 48 kHz y un hilo propio que pide
     * bloques de ~21 ms y los entrega a sceAudioOutOutput, que marca el ritmo.
     *
     * CON EL ESTIRADO DE AUDIO. A la velocidad actual (un 5-10 % de la real)
     * el emulador produce una decima parte del sonido que el altavoz consume.
     * Con el estirado (SoundTouch, en el hilo de audio, no en el de emulacion)
     * se oye continuo pero muy lento y grave; sin el, se oye a su tono pero a
     * trozos, con silencios. SELECT + IZQUIERDA cambia entre los dos en
     * marcha (vita_window.cpp).
     */
    Settings::values.output_type = AudioCore::SinkType::Vita;
    Settings::values.enable_audio_stretching = true;

    Settings::values.use_virtual_sd = true;

    // Operaciones de fichero en el mismo hilo, no en paralelo.
    //
    // HLERequestContext::RunAsync usa std::async(std::launch::async, ...), que
    // levanta UN HILO NUEVO en cada llamada, y el servicio de ficheros lo usa
    // en cada operacion. Al cargar un juego se hacen cientos, y en la Vita la
    // pila de cada hilo sale del presupuesto del proceso, del que quedan unos
    // 25 MB: se agotan, std::async no puede crear mas y lanza system_error.
    // Como salta dentro de un hilo del emulador, nadie la captura y se lleva el
    // proceso por delante.
    //
    // Con esto RunAsync ejecuta en linea, sin crear hilos. Va mas lento, pero
    // en una consola que ya emula por interprete eso es lo de menos.
    Settings::values.deterministic_async_operations = true;
    Settings::values.async_fs_operations = false;

    // El registro vuelve a pasar por la cola del hilo de logging. En modo
    // sincrono cada linea es una escritura a la tarjeta de memoria, y Citra
    // registra constantemente mientras emula: eso frena el emulador hasta
    // hacerlo inservible. Los eventos criticos se anotan aparte en crash.txt,
    // que se escribe con la API del sistema y no depende de esta cola.
    Settings::values.instant_debug_log = false;

    // Service::Init consulta lle_modules.at(nombre) para cada uno de los 41
    // modulos de servicio del 3DS, y .at() lanza out_of_range si la clave no
    // existe. Los frontends de escritorio rellenan el mapa al leer su fichero
    // de configuracion; aqui no hay tal fichero, asi que hay que poblarlo a
    // mano o cargar un juego falla antes de empezar.
    //
    // Todos a false = emulacion por alto nivel (HLE). Usar LLE exigiria los
    // modulos de sistema volcados de una consola real.
    for (const auto& service_module : Service::service_module_map) {
        Settings::values.lle_modules.emplace(service_module.name, false);
    }

    // La Vita tiene camaras, pero no estan portadas. "blank" es la camara nula
    // que Azahar ya trae; sin ponerla, el nombre llega vacio y el servicio deja
    // un error por cada camara aunque acabe usando la misma nula.
    for (auto& name : Settings::values.camera_name) {
        name = "blank";
    }
    for (auto& config : Settings::values.camera_config) {
        config.clear();
    }
    Settings::values.camera_flip = {};
}

// ---------------------------------------------------------------------------
// Ajustes del usuario (0.1.5.1)
// ---------------------------------------------------------------------------

/**
 * VENTANA DE AJUSTES. Se abre manteniendo START dos segundos en la lista de
 * juegos (START a secas no hace nada en el menu, y exigir que se mantenga
 * evita abrirla sin querer). Tres ajustes:
 *
 *   - Volumen: Settings::values.volume, que DspInterface::OutputFrame aplica a
 *     cada bloque de salida (con curva cubica, como el deslizador de
 *     escritorio). Vale al momento, tambien en partida.
 *   - Idioma del sistema emulado: el bloque de idioma del config del 3DS, que
 *     es lo que leen los juegos para elegir idioma. Se aplica al cargar el
 *     siguiente juego (ver Service::CFG::g_vita_system_language).
 *   - Resolucion 1x o 0.5x: FrameSkip::half_resolution. En 0.5x el
 *     rasterizador por software sombrea un pixel de cada bloque de 2x2 y lo
 *     copia a los otros tres. Los lotes que dibuja la GPU de la Vita (GXM) no
 *     cambian: ya van a la resolucion nativa del 3DS y no son el cuello de
 *     botella. Vale al momento, y en partida lo cambia tambien L+R.
 *
 * Se guardan en ux0:/data/azahar/ajustes.txt al cerrar la ventana y se leen
 * al arrancar, para no tener que repetirlos cada vez.
 */
void WriteCrashLog(const char* title, const char* detail); // definida mas abajo

constexpr char kSettingsFile[] = "ux0:/data/azahar/ajustes.txt";
constexpr SceUInt64 kSettingsHoldUs = 2'000'000;

struct LanguageOption {
    int value; ///< Service::CFG::SystemLanguage, o -1 = no tocar
    const char* name;
};

constexpr LanguageOption kLanguages[] = {
    {-1, "Automatico (el de la consola emulada)"},
    {Service::CFG::LANGUAGE_ES, "Espanol"},
    {Service::CFG::LANGUAGE_EN, "Ingles"},
    {Service::CFG::LANGUAGE_FR, "Frances"},
    {Service::CFG::LANGUAGE_DE, "Aleman"},
    {Service::CFG::LANGUAGE_IT, "Italiano"},
    {Service::CFG::LANGUAGE_PT, "Portugues"},
    {Service::CFG::LANGUAGE_NL, "Neerlandes"},
    {Service::CFG::LANGUAGE_RU, "Ruso"},
    {Service::CFG::LANGUAGE_JP, "Japones"},
    {Service::CFG::LANGUAGE_ZH, "Chino simplificado"},
    {Service::CFG::LANGUAGE_TW, "Chino tradicional"},
    {Service::CFG::LANGUAGE_KO, "Coreano"},
};
constexpr int kLanguageCount = static_cast<int>(sizeof(kLanguages) / sizeof(kLanguages[0]));

struct UserSettings {
    int volume_percent = 100;
    int language = -1;
    bool half_resolution = false;
    /// 4.3: cache de registros del JIT. Encendida por defecto (0.1.5.2).
    bool jit_reg_cache = true;
    /// VFP en el JIT (datos 0.1.5.2, aritmetica 0.1.6.1, nativa 0.1.7.0). ON.
    bool jit_vfp_data = true;
    /// 4.5: diferir sceGxmFinish hasta WriteBack/Reload. ON por defecto.
    bool gxm_no_finish = true;
    /// 4.6: presentar desde color_buffer de GXM sin WriteBack+subida. ON.
    bool gxm_present_direct = true;
    /// 4.8: enlazado directo entre bloques en el codigo generado. ON.
    bool jit_direct_link = true;
    /// 0.1.6.0: saltos VS con escape a la GPU (ver g_allow_vs_escapes). ON
    /// desde 0.1.7.3: el shader que rompia el compilador ya no deja la GPU sin
    /// shaders (se recarga, se prueba otra variante y se aparta el codigo que
    /// lo rompe). Sin esto, el shader de piel de Rubi Omega va a la CPU.
    bool vs_escapes = true;
    /// 0.1.6.1: sombrear cada indice una vez por lote. ON por defecto.
    bool full_vertex_dedup = true;
    /// 0.1.7.1: sonido. Apagado = salida nula (se aplica al arrancar el juego).
    bool sound = true;
    /// 0.1.7.1: disposicion de las pantallas (VitaFrontend::g_screen_layout).
    int screen_layout = 0;
    /// 0.1.7.1: no volver a presentar una imagen que no ha cambiado.
    bool skip_repeated = true;
    /// Programas de vertices especializados por booleanos (0.1.7.4). OFF en
    /// 0.1.8.1 (agotaban la memoria del compilador al entrar al 3D de Zafiro
    /// Alfa), y la partida bajo a 2.5 FPS con los vertices de piel en la CPU.
    /// ON otra vez en 0.1.8.3, con el compilador descargado tras cada shader de
    /// vertices y sin compilar si no hay 85 MB libres (ver gxm_cg.cpp). Clave
    /// nueva: el ajustes.txt de 0.1.8.2 trae vs_especializar=0.
    bool vs_specialize = true;
    /// 0.1.8.6: copia de pantalla en la GPU (RasterizerGXM::transfer_on_gpu).
    bool gpu_screen_copy = true;
    /// 0.2.0.0: la GPU emulada en otro nucleo (VideoCore::GPU::async_enabled).
    /// OFF hasta probarlo en consola con varios juegos.
    bool gpu_thread = false;
    /// 0.2.0.3: sin esperar al refresco al presentar (VitaFrontend::g_unlimited_speed).
    bool unlimited_speed = false;
    /// Resolucion de la GPU en mitades: 1 = 0.5x, 2 = 1x, 4 = 2x
    /// (RasterizerGXM::resolution_scale).
    int gpu_scale = 2;
};
UserSettings g_user;

constexpr const char* kLayoutNames[VitaFrontend::kScreenLayoutCount] = {
    "Normal (1:1)", "Superior grande + inferior", "Lado a lado", "Solo superior x2",
    "Solo superior, pantalla completa"};

int LanguageIndex(int value) {
    for (int i = 0; i < kLanguageCount; i++) {
        if (kLanguages[i].value == value) {
            return i;
        }
    }
    return 0;
}

void ApplyUserSettings() {
    Settings::values.volume = static_cast<float>(g_user.volume_percent) / 100.0f;
    Service::CFG::g_vita_system_language = g_user.language;
    SwRenderer::FrameSkip::half_resolution.store(g_user.half_resolution,
                                                 std::memory_order_relaxed);
    // 4.3 (0.1.5.2): al cambiar el interruptor hay que tirar los bloques ya
    // compilados, o los viejos seguirian con el mapa de cache del momento de
    // compilarlos y el interruptor no apagaria nada.
    const u32 want_cache = g_user.jit_reg_cache ? 1u : 0u;
    if (Core::ArmJit::reg_cache.exchange(want_cache, std::memory_order_relaxed) != want_cache) {
        Core::ArmJit::Reset();
    }
    // 4.4 (0.1.5.2): igual con VFP datos: recompilar, o los bloques viejos
    // seguirian rechazando o aceptando VFP segun el momento de compilarlos.
    const u32 want_vfp = g_user.jit_vfp_data ? 1u : 0u;
    if (Core::ArmJit::vfp_data.exchange(want_vfp, std::memory_order_relaxed) != want_vfp) {
        Core::ArmJit::Reset();
    }
    // 4.5 (0.1.5.2): no hace falta recompilar nada; es solo cuando se espera.
    Gxm::RasterizerGXM::no_finish_wait.store(g_user.gxm_no_finish ? 1u : 0u,
                                             std::memory_order_relaxed);
    // 4.6 (0.1.5.2): solo cambia de donde lee la textura el presentador.
    Gxm::RasterizerGXM::present_direct.store(g_user.gxm_present_direct ? 1u : 0u,
                                             std::memory_order_relaxed);
    // 4.8 (0.1.5.2): al cambiar hay que tirar los bloques: los viejos llevan
    // (o no) el enlace ya parcheado en su codigo.
    const u32 want_link = g_user.jit_direct_link ? 1u : 0u;
    if (Core::ArmJit::direct_link.exchange(want_link, std::memory_order_relaxed) != want_link) {
        Core::ArmJit::Reset();
    }
    // 0.1.6.0: solo cambia que shaders se traducen a partir de ahora.
    Pica::Shader::Generator::GXM::g_allow_vs_escapes.store(g_user.vs_escapes,
                                                            std::memory_order_relaxed);
    Pica::g_full_vertex_dedup.store(g_user.full_vertex_dedup, std::memory_order_relaxed);
    /**
     * Sonido (0.1.7.1). Apagado: salida nula y sin estirado. El DSP del 3DS se
     * sigue emulando (los juegos esperan a que termine sus bloques), pero se
     * ahorra el estirado de SoundTouch y el envio a la Vita. Se aplica al
     * arrancar el siguiente juego, que es cuando se crea la salida de audio.
     */
    Settings::values.output_type =
        g_user.sound ? AudioCore::SinkType::Vita : AudioCore::SinkType::Null;
    Settings::values.enable_audio_stretching = g_user.sound;
    VitaFrontend::g_screen_layout.store(
        std::clamp(g_user.screen_layout, 0, VitaFrontend::kScreenLayoutCount - 1),
        std::memory_order_relaxed);
    VitaFrontend::g_skip_repeated_frames.store(g_user.skip_repeated, std::memory_order_relaxed);
    Gxm::RasterizerGXM::specialize_vs.store(g_user.vs_specialize ? 1u : 0u,
                                            std::memory_order_relaxed);
    Gxm::RasterizerGXM::transfer_on_gpu.store(g_user.gpu_screen_copy ? 1u : 0u,
                                              std::memory_order_relaxed);
    VideoCore::GPU::async_enabled.store(g_user.gpu_thread, std::memory_order_relaxed);
    VitaFrontend::g_unlimited_speed.store(g_user.unlimited_speed, std::memory_order_relaxed);
    Gxm::RasterizerGXM::resolution_scale.store(static_cast<u32>(g_user.gpu_scale),
                                               std::memory_order_relaxed);
}

/// Lee ajustes.txt. Si no existe o una linea no se entiende, se queda el valor
/// por defecto de ese ajuste: un fichero roto no puede impedir arrancar.
void LoadUserSettings() {
    const SceUID fd = sceIoOpen(kSettingsFile, SCE_O_RDONLY, 0);
    if (fd < 0) {
        return;
    }
    char buffer[512] = {};
    const int read = sceIoRead(fd, buffer, sizeof(buffer) - 1);
    sceIoClose(fd);
    if (read <= 0) {
        return;
    }
    buffer[read] = '\0';
    const auto read_int = [&buffer](const char* key, int& out) {
        const char* p = std::strstr(buffer, key);
        int value = 0;
        if (p != nullptr && std::sscanf(p + std::strlen(key), "%d", &value) == 1) {
            out = value;
            return true;
        }
        return false;
    };
    const auto read_bool = [&read_int](const char* key, bool& out) {
        int value = 0;
        if (read_int(key, value)) {
            out = value != 0;
        }
    };
    int value = 0;
    if (read_int("volumen=", value)) {
        g_user.volume_percent = std::clamp(value, 0, 100);
    }
    if (read_int("idioma=", value)) {
        g_user.language = kLanguages[LanguageIndex(value)].value;
    }
    read_bool("resolucion_media=", g_user.half_resolution);
    read_bool("jit_cache_reg=", g_user.jit_reg_cache);
    read_bool("jit_vfp_datos=", g_user.jit_vfp_data);
    read_bool("gxm_sin_espera=", g_user.gxm_no_finish);
    read_bool("gxm_present_dir=", g_user.gxm_present_direct);
    read_bool("jit_enlace_dir=", g_user.jit_direct_link);
    // Clave nueva en 0.1.7.3: un ajustes.txt viejo trae vs_saltos=0 de cuando
    // el valor por defecto era OFF, y eso dejaria apagada la mayor ganancia.
    read_bool("vs_saltos2=", g_user.vs_escapes);
    read_bool("cache_vertices=", g_user.full_vertex_dedup);
    read_bool("sonido=", g_user.sound);
    if (read_int("pantallas=", value)) {
        g_user.screen_layout = std::clamp(value, 0, VitaFrontend::kScreenLayoutCount - 1);
    }
    read_bool("omitir_repetidas=", g_user.skip_repeated);
    read_bool("vs_especializar2=", g_user.vs_specialize);
    read_bool("copia_gpu=", g_user.gpu_screen_copy);
    read_bool("gpu_hilo=", g_user.gpu_thread);
    read_bool("sin_limite=", g_user.unlimited_speed);
    // Clave nueva en 0.2.0.4, en mitades (la de 0.2.0.3 iba en enteros).
    if (read_int("resolucion_gpu2=", value)) {
        g_user.gpu_scale = value == 1 || value == 4 ? value : 2;
    }
}

/// El contenido de ajustes.txt, una clave por linea.
int FormatUserSettings(char* buffer, std::size_t size) {
    return std::snprintf(
        buffer, size,
        "volumen=%d\nidioma=%d\nresolucion_media=%d\njit_cache_reg=%d\n"
        "jit_vfp_datos=%d\ngxm_sin_espera=%d\ngxm_present_dir=%d\n"
        "jit_enlace_dir=%d\nvs_saltos2=%d\ncache_vertices=%d\nsonido=%d\npantallas=%d\n"
        "omitir_repetidas=%d\nvs_especializar2=%d\ncopia_gpu=%d\ngpu_hilo=%d\nsin_limite=%d\n"
        "resolucion_gpu2=%d\n",
        g_user.volume_percent, g_user.language, g_user.half_resolution ? 1 : 0,
        g_user.jit_reg_cache ? 1 : 0, g_user.jit_vfp_data ? 1 : 0, g_user.gxm_no_finish ? 1 : 0,
        g_user.gxm_present_direct ? 1 : 0, g_user.jit_direct_link ? 1 : 0,
        g_user.vs_escapes ? 1 : 0, g_user.full_vertex_dedup ? 1 : 0, g_user.sound ? 1 : 0,
        g_user.screen_layout, g_user.skip_repeated ? 1 : 0, g_user.vs_specialize ? 1 : 0,
        g_user.gpu_screen_copy ? 1 : 0, g_user.gpu_thread ? 1 : 0,
        g_user.unlimited_speed ? 1 : 0, g_user.gpu_scale);
}

/// Los ajustes con los que se juega, a crash.txt en una linea (0.1.8.7): sin
/// esto no se sabe si una partida lenta llevaba algo apagado en ajustes.txt.
void NoteUserSettings() {
    char buffer[512];
    const int length = FormatUserSettings(buffer, sizeof(buffer));
    for (int i = 0; i < length && i < static_cast<int>(sizeof(buffer)) - 1; i++) {
        if (buffer[i] == '\n') {
            buffer[i] = ' ';
        }
    }
    WriteCrashLog("ajustes", buffer);
}

void SaveUserSettings() {
    char buffer[512];
    const int length = FormatUserSettings(buffer, sizeof(buffer));
    const SceUID fd = sceIoOpen(kSettingsFile, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
    if (fd < 0) {
        WriteCrashLog("ajustes", "no se pudo guardar ajustes.txt");
        return;
    }
    sceIoWrite(fd, buffer, static_cast<SceSize>(length));
    sceIoClose(fd);
}

constexpr int kSettingsRows = 18;

void DrawSettings(vita2d_pgf* font, int row) {
    constexpr int kFirstY = 100;
    // 17 desde la fila 18 (0.2.0.3): con mas la ultima pisaba la ayuda.
    constexpr int kRowStep = 17;
    constexpr int kValueX = 330;

    vita2d_start_drawing();
    vita2d_clear_screen();
    vita2d_pgf_draw_text(font, 40, 45, kColorAccent, 1.4f, "Ajustes");
    vita2d_pgf_draw_text(font, 40, 72, kColorDim, 0.85f,
                         "Se guardan en ux0:/data/azahar/ajustes.txt. Las cifras de FPS son "
                         "aproximadas y dependen del juego.");

    const char* labels[kSettingsRows] = {"Volumen",
                                         "Idioma del sistema",
                                         "Resolucion (software)",
                                         "Cache registros JIT",
                                         "Coma flotante en JIT",
                                         "Sin espera a la GPU",
                                         "Presentar directo",
                                         "Enlace directo JIT",
                                         "Vertices con saltos a GPU",
                                         "Cache de vertices completa",
                                         "Sonido",
                                         "Pantallas",
                                         "Omitir imagenes repetidas",
                                         "Vertices especializados a GPU",
                                         "Copia y borrado en GPU",
                                         "GPU en otro nucleo",
                                         "Limite de velocidad",
                                         "Resolucion GPU"};
    char volume_text[16];
    std::snprintf(volume_text, sizeof(volume_text), "%d %%", g_user.volume_percent);
    const char* values[kSettingsRows] = {
        volume_text,
        kLanguages[LanguageIndex(g_user.language)].name,
        g_user.half_resolution ? "0.5x" : "1x",
        g_user.jit_reg_cache ? "ON" : "OFF",
        g_user.jit_vfp_data ? "ON" : "OFF",
        g_user.gxm_no_finish ? "ON" : "OFF",
        g_user.gxm_present_direct ? "ON" : "OFF",
        g_user.jit_direct_link ? "ON" : "OFF",
        g_user.vs_escapes ? "ON" : "OFF",
        g_user.full_vertex_dedup ? "ON" : "OFF",
        g_user.sound ? "ON" : "OFF",
        kLayoutNames[std::clamp(g_user.screen_layout, 0, VitaFrontend::kScreenLayoutCount - 1)],
        g_user.skip_repeated ? "ON" : "OFF",
        g_user.vs_specialize ? "ON" : "OFF",
        g_user.gpu_screen_copy ? "ON" : "OFF",
        g_user.gpu_thread ? "ON" : "OFF",
        g_user.unlimited_speed ? "Sin limite" : "100% (refresco)",
        g_user.gpu_scale == 4 ? "2x (experimental)" : g_user.gpu_scale == 1 ? "0.5x" : "1x"};

    for (int i = 0; i < kSettingsRows; i++) {
        const int y = kFirstY + i * kRowStep;
        const bool selected = i == row;
        if (selected) {
            vita2d_draw_rectangle(30, static_cast<float>(y - 18), 900, 23, 0xFF402810);
        }
        const unsigned int color = selected ? kColorAccent : kColorText;
        vita2d_pgf_draw_text(font, 44, y, color, 0.95f, labels[i]);
        vita2d_pgf_draw_textf(font, kValueX, y, color, 0.95f, selected ? "<  %s  >" : "   %s",
                              values[i]);
    }

    // Barra del volumen, a la derecha de su fila.
    const float bar_width = 250.0f;
    vita2d_draw_rectangle(kValueX + 200, kFirstY - 8, bar_width, 6, 0xFF404040);
    vita2d_draw_rectangle(kValueX + 200, kFirstY - 8,
                          bar_width * static_cast<float>(g_user.volume_percent) / 100.0f, 6,
                          kColorAccent);

    /**
     * Que hace cada ajuste: que ganas (FPS, mas o menos) y que pierdes. Las
     * cifras son estimaciones de lo medido en Pokemon Rubi Omega; en otros
     * juegos cambian.
     */
    static const char* const help[kSettingsRows][3] = {
        {"Volumen del juego. Se aplica al momento.", "No cambia los FPS.", ""},
        {"El idioma que ve el juego. Se aplica al arrancar el juego.", "No cambia los FPS.",
         "Si el juego no trae ese idioma, usa el primero de su region."},
        {"0.5x: la CPU dibuja 1 de cada 4 pixeles y copia el resto.",
         "Mas FPS SOLO en juegos que dibujan por software (no los 3D normales).",
         "Contra: imagen mas pixelada. En partida, L+R cambia lo mismo."},
        {"ON: los registros del juego se quedan en el procesador de la Vita.",
         "Ganancia: CPU del juego ~10-20% mas rapida (varios FPS en juegos pesados).",
         "Contra: ninguno conocido. Apagar solo si un juego se cuelga."},
        {"ON: la coma flotante del juego va por el hardware de la Vita.",
         "Ganancia: la mayor del JIT en juegos 3D (del orden de +10-25% de FPS).",
         "Contra: si un juego calcula cosas raras (fisicas, animaciones), apagar."},
        {"ON: no para la CPU a esperar a la GPU al cerrar cada escena.",
         "Ganancia: ~3-7% de FPS en 3D.", "Contra: si la imagen se congela o parpadea, apagar."},
        {"ON: la imagen dibujada por la GPU se muestra sin copiarla dos veces.",
         "Ganancia: ~2-5% de FPS cuando el juego lo permite.",
         "Contra: si la imagen sale girada o con colores raros, apagar."},
        {"ON: la CPU emulada salta de bloque en bloque sin volver al emulador.",
         "Ganancia: ~5-15% de FPS.",
         "Contra: ninguno conocido. Apagar si hay cuelgues sin explicacion."},
        {"ON: los shaders de vertices con saltos complicados van a la GPU.",
         "Ganancia: la mayor en 3D (+30% o mas si el compilador los acepta).",
         "Contra: la 1a vez tarda mas en cargar la escena. Si el 3D sale roto, apagar."},
        {"ON: cada vertice repetido se calcula una sola vez por dibujo.",
         "Ganancia: 0-15% en 3D, segun el juego.",
         "Contra: si un modelo 3D sale con picos o deformado, apagar."},
        {"OFF: sin sonido. Se ahorra el estirado y el envio del audio.",
         "Ganancia: ~2-5% de FPS. Se aplica al arrancar el juego.", "Contra: no hay sonido."},
        {"Como se colocan las pantallas. La escala la hace la GPU de la Vita.",
         "No cambia los FPS. Con solo la superior no se ve la tactil ni se puede tocar.",
         "Se aplica al momento, tambien en partida."},
        {"ON: si el juego no ha dibujado nada nuevo, no se vuelve a mostrar.",
         "Ganancia: ~5-10% de FPS en juegos a 30 FPS o menos.",
         "Contra: si un juego deja la imagen congelada, apagar."},
        {"ON: los shaders con saltos se compilan una vez por combinacion de opciones.",
         "Ganancia: la mayor en 3D (Zafiro Alfa: de ~2.5 FPS con OFF a varias veces mas).",
         "Contra: la 1a vez que sale cada uno, la escena se para unos segundos. Si crashea, apagar."},
        {"ON: copia de pantalla y borrados en la GPU, sin bajar a la CPU y volver a subir.",
         "Ganancia: el 2D de Zafiro Alfa paso de 11 a 22 FPS; en 3D, varios ms por fotograma.",
         "Contra: si ves imagenes viejas, colores raros o parpadeos, apagar."},
        {"ON: la GPU emulada y la imagen van en el nucleo 1; el juego, solo en el 0.",
         "Ganancia: lo que tarde la GPU por fotograma (calculo: +20-50% donde pesa). Al arrancar juego.",
         "Contra: nuevo. Si un juego se cuelga, va a saltos o se ve raro, apagar."},
        {"Sin limite: no espera al refresco de la pantalla; puede pasar del 100% de velocidad.",
         "Ganancia: menus y escenas ligeras van mas rapido que en la consola.",
         "Contra: el juego puede ir acelerado y el sonido estirado. Se aplica al momento."},
        {"0.5x: la GPU dibuja la mitad por eje (menos nitido). 2x: el doble (mas nitido).",
         "0.5x alivia la GPU de la Vita; 2x cuesta GPU y memoria de video. Al arrancar el juego.",
         "Contra: experimental. Lo que vaya por software sigue a 1x. Si va lento o raro, 1x."},
    };
    vita2d_pgf_draw_text(font, 40, 414, kColorDim, 0.9f, help[row][0]);
    vita2d_pgf_draw_text(font, 40, 438, kColorAccent, 0.9f, help[row][1]);
    vita2d_pgf_draw_text(font, 40, 462, kColorDim, 0.9f, help[row][2]);

    vita2d_pgf_draw_text(font, 40, 504, kColorDim, 0.9f,
                         "ARRIBA/ABAJO: elegir   IZQ/DER: cambiar   CIRCULO o START: guardar y volver");
    vita2d_pgf_draw_text(font, 40, 528, kColorDim, 0.85f, kVersion);
    vita2d_end_drawing();
    vita2d_swap_buffers();
}

/// La ventana de ajustes. Vuelve al cerrarla, con los ajustes aplicados y
/// guardados.
void RunSettingsMenu(vita2d_pgf* font) {
    // L+R en partida cambia la resolucion sin pasar por aqui: partir del valor
    // real, no del ultimo que se guardo.
    g_user.half_resolution =
        SwRenderer::FrameSkip::half_resolution.load(std::memory_order_relaxed);
    g_user.jit_reg_cache = Core::ArmJit::reg_cache.load(std::memory_order_relaxed) != 0;
    g_user.jit_vfp_data = Core::ArmJit::vfp_data.load(std::memory_order_relaxed) != 0;
    g_user.gxm_no_finish = Gxm::RasterizerGXM::no_finish_wait.load(std::memory_order_relaxed) != 0;
    g_user.gxm_present_direct =
        Gxm::RasterizerGXM::present_direct.load(std::memory_order_relaxed) != 0;
    g_user.jit_direct_link = Core::ArmJit::direct_link.load(std::memory_order_relaxed) != 0;
    g_user.vs_escapes =
        Pica::Shader::Generator::GXM::g_allow_vs_escapes.load(std::memory_order_relaxed);
    g_user.full_vertex_dedup = Pica::g_full_vertex_dedup.load(std::memory_order_relaxed);
    g_user.vs_specialize =
        Gxm::RasterizerGXM::specialize_vs.load(std::memory_order_relaxed) != 0;
    g_user.gpu_screen_copy =
        Gxm::RasterizerGXM::transfer_on_gpu.load(std::memory_order_relaxed) != 0;
    g_user.gpu_thread = VideoCore::GPU::async_enabled.load(std::memory_order_relaxed);
    g_user.unlimited_speed = VitaFrontend::g_unlimited_speed.load(std::memory_order_relaxed);
    g_user.gpu_scale =
        static_cast<int>(Gxm::RasterizerGXM::resolution_scale.load(std::memory_order_relaxed));

    int row = 0;
    // Todo lo que ya este pulsado al entrar (el START que la abrio) cuenta como
    // mantenido, no como pulsacion nueva: si no, se cerraria en el acto.
    unsigned int previous_buttons = ~0u;
    while (true) {
        SceCtrlData pad{};
        sceCtrlPeekBufferPositive(0, &pad, 1);
        const unsigned int pressed = pad.buttons & ~previous_buttons;
        previous_buttons = pad.buttons;

        if (pressed & (SCE_CTRL_CIRCLE | SCE_CTRL_START)) {
            break;
        }
        if (pressed & SCE_CTRL_DOWN) {
            row = (row + 1) % kSettingsRows;
        }
        if (pressed & SCE_CTRL_UP) {
            row = (row + kSettingsRows - 1) % kSettingsRows;
        }
        const int step = (pressed & SCE_CTRL_RIGHT) ? 1 : (pressed & SCE_CTRL_LEFT) ? -1 : 0;
        if (step != 0) {
            switch (row) {
            case 0:
                g_user.volume_percent = std::clamp(g_user.volume_percent + step * 10, 0, 100);
                break;
            case 1: {
                const int index =
                    (LanguageIndex(g_user.language) + step + kLanguageCount) % kLanguageCount;
                g_user.language = kLanguages[index].value;
                break;
            }
            case 2:
                g_user.half_resolution = !g_user.half_resolution;
                break;
            case 3:
                g_user.jit_reg_cache = !g_user.jit_reg_cache;
                break;
            case 4:
                g_user.jit_vfp_data = !g_user.jit_vfp_data;
                break;
            case 5:
                g_user.gxm_no_finish = !g_user.gxm_no_finish;
                break;
            case 6:
                g_user.gxm_present_direct = !g_user.gxm_present_direct;
                break;
            case 7:
                g_user.jit_direct_link = !g_user.jit_direct_link;
                break;
            case 8:
                g_user.vs_escapes = !g_user.vs_escapes;
                break;
            case 9:
                g_user.full_vertex_dedup = !g_user.full_vertex_dedup;
                break;
            case 10:
                g_user.sound = !g_user.sound;
                break;
            case 11:
                g_user.screen_layout =
                    (g_user.screen_layout + step + VitaFrontend::kScreenLayoutCount) %
                    VitaFrontend::kScreenLayoutCount;
                break;
            case 12:
                g_user.skip_repeated = !g_user.skip_repeated;
                break;
            case 13:
                g_user.vs_specialize = !g_user.vs_specialize;
                break;
            case 14:
                g_user.gpu_screen_copy = !g_user.gpu_screen_copy;
                break;
            case 15:
                g_user.gpu_thread = !g_user.gpu_thread;
                break;
            case 16:
                g_user.unlimited_speed = !g_user.unlimited_speed;
                break;
            default:
                // 0.5x -> 1x -> 2x con DERECHA, al reves con IZQUIERDA.
                g_user.gpu_scale = step > 0 ? (g_user.gpu_scale == 1 ? 2 : g_user.gpu_scale == 2 ? 4 : 1)
                                            : (g_user.gpu_scale == 4 ? 2 : g_user.gpu_scale == 2 ? 1 : 4);
                break;
            }
            ApplyUserSettings();
        }
        DrawSettings(font, row);
    }
    ApplyUserSettings();
    SaveUserSettings();
    NoteUserSettings();
}

/// Dibuja una pantalla de texto simple centrada (menu y mensajes).
/// 'settings_hold' va de 0 a 1 mientras se mantiene START para abrir los ajustes.
void DrawMenu(vita2d_pgf* font, const std::vector<RomEntry>& roms, int selected, int scroll,
              float settings_hold) {
    constexpr int kVisibleRows = 14;
    constexpr int kRowHeight = 28;
    constexpr int kListTop = 120;

    vita2d_start_drawing();
    vita2d_clear_screen();

    vita2d_pgf_draw_text(font, 40, 50, kColorAccent, 1.4f, "Azahar  -  PS Vita");
    vita2d_pgf_draw_text(font, 40, 524, kColorDim, 0.85f, kVersion);
    vita2d_pgf_draw_text(font, 40, 80, kColorDim, 0.9f,
                         "Emulador de Nintendo 3DS  |  JIT ARM + renderizado GXM");

    if (roms.empty()) {
        vita2d_pgf_draw_text(font, 40, kListTop + 20, kColorText, 1.0f,
                             "No hay ningun juego en ux0:/data/azahar/roms");
        vita2d_pgf_draw_text(font, 40, kListTop + 55, kColorDim, 0.9f,
                             "Copia ahi ficheros .3ds / .cci / .cxi / .app / .3dsx y vuelve.");
        vita2d_pgf_draw_text(font, 40, 500, kColorDim, 0.9f, "CIRCULO: salir");
    } else {
        for (int row = 0; row < kVisibleRows; row++) {
            const int index = scroll + row;
            if (index >= static_cast<int>(roms.size())) {
                break;
            }
            const bool is_selected = index == selected;
            const int y = kListTop + row * kRowHeight;

            if (is_selected) {
                vita2d_draw_rectangle(30, static_cast<float>(y - 20), 900, kRowHeight, 0xFF402810);
            }
            vita2d_pgf_draw_text(font, 44, y, is_selected ? kColorAccent : kColorText, 1.0f,
                                 roms[index].name.c_str());
        }

        vita2d_pgf_draw_textf(font, 40, 500, kColorDim, 0.9f,
                              "CRUZ: jugar   CIRCULO: salir   (%d/%d)", selected + 1,
                              static_cast<int>(roms.size()));

        // El motor grafico se elige aqui y se aplica al siguiente juego: cada
        // partida construye su renderer desde cero en RunGame, asi que no hace
        // falta recrear nada en caliente (y el menu se dibuja a 60 fps, fuera
        // del bucle de emulacion).
        if (Settings::values.graphics_api.GetValue() == Settings::GraphicsAPI::GXM) {
            vita2d_pgf_draw_text(font, 40, 470, kColorDim, 0.9f,
                                 "SELECT: motor grafico  (GXM)");
        } else {
            vita2d_pgf_draw_text(font, 40, 470, kColorDim, 0.9f,
                                 "SELECT: motor grafico  (software)");
        }
    }

    // Ajustes: la pista siempre, y mientras se mantiene START, la barra que se
    // llena en los dos segundos que hacen falta.
    vita2d_pgf_draw_text(font, 40, 440, kColorDim, 0.9f,
                         "Manten START 2 s: ajustes (volumen, idioma, resolucion)");
    if (settings_hold > 0.0f) {
        vita2d_draw_rectangle(560, 430, 300, 10, 0xFF404040);
        vita2d_draw_rectangle(560, 430, 300.0f * std::min(settings_hold, 1.0f), 10, kColorAccent);
    }

    vita2d_end_drawing();
    vita2d_swap_buffers();
}

/// "Cargando shaders" con barra de progreso, antes de empezar el juego (0.2.0.4).
void DrawShaderLoading(vita2d_pgf* font, u32 done, u32 total) {
    vita2d_start_drawing();
    vita2d_clear_screen();
    vita2d_pgf_draw_text(font, 40, 200, kColorAccent, 1.3f, "Cargando shaders...");
    constexpr float kBarX = 40.0f;
    constexpr float kBarY = 240.0f;
    constexpr float kBarWidth = 880.0f;
    vita2d_draw_rectangle(kBarX, kBarY, kBarWidth, 18.0f, 0xFF404040);
    if (total > 0) {
        vita2d_draw_rectangle(kBarX, kBarY,
                              kBarWidth * static_cast<float>(done) / static_cast<float>(total),
                              18.0f, kColorAccent);
        vita2d_pgf_draw_textf(font, 40, 300, kColorText, 1.0f, "%u / %u shaders", done, total);
    } else {
        vita2d_pgf_draw_text(font, 40, 300, kColorDim, 1.0f,
                             "No hay shaders guardados de este juego todavia: se iran");
        vita2d_pgf_draw_text(font, 40, 330, kColorDim, 1.0f,
                             "guardando al jugar y la proxima vez cargaran aqui.");
    }
    vita2d_end_drawing();
    vita2d_swap_buffers();
}

void DrawMessage(vita2d_pgf* font, const char* line1, const char* line2) {
    vita2d_start_drawing();
    vita2d_clear_screen();
    vita2d_pgf_draw_text(font, 40, 240, kColorText, 1.2f, line1);
    if (line2 != nullptr) {
        vita2d_pgf_draw_text(font, 40, 280, kColorDim, 1.0f, line2);
    }
    vita2d_end_drawing();
    vita2d_swap_buffers();
}

const char* LoadErrorMessage(Core::System::ResultStatus status) {
    switch (status) {
    case Core::System::ResultStatus::ErrorGetLoader:
        return "No hay un cargador para este fichero.";
    case Core::System::ResultStatus::ErrorLoader:
        return "No se ha podido cargar el juego.";
    case Core::System::ResultStatus::ErrorLoader_ErrorEncrypted:
        return "El juego esta cifrado: hay que descifrarlo antes de usarlo.";
    case Core::System::ResultStatus::ErrorLoader_ErrorInvalidFormat:
        return "Formato de fichero no soportado.";
    case Core::System::ResultStatus::ErrorLoader_ErrorGbaTitle:
        return "Es un titulo de Consola Virtual de GBA, no soportado.";
    case Core::System::ResultStatus::ErrorSystemFiles:
        return "Faltan ficheros de sistema del 3DS.";
    case Core::System::ResultStatus::ErrorNotInitialized:
        return "El nucleo de CPU no se ha inicializado.";
    case Core::System::ResultStatus::ErrorSystemMode:
        // OJO: este no significa lo que dice. Es el cajon de sastre de
        // Core::System::Load: LoadKernelMemoryMode solo falla si no consigue
        // ABRIR Y LEER el ROM, y todo error que no sea cifrado / formato
        // invalido / titulo de GBA acaba aqui. El generico ResultStatus::Error
        // se devuelve en mas de quince sitios distintos de ncch_container.cpp.
        //
        // El texto lo dice ahora, y ademas la pantalla de error enseña el
        // tamaño del fichero: un ROM de mas de 2 GB era exactamente este
        // sintoma antes de pasar la lectura de ficheros a sceIo.
        return "No se ha podido leer el ROM (cabecera ilegible, claves, o fichero corrupto).";
    default:
        return "Error desconocido al cargar el juego.";
    }
}

/// Fuente global: la necesita el manejador de terminate, que no recibe argumentos.
vita2d_pgf* g_font = nullptr;

/// Identificador del hilo principal, para distinguirlo en el manejador.
SceUID g_main_thread_id = 0;

/// Ultimo error capturado, para poder enseñarlo sin bloquear al que lo lanzo.
std::string g_last_error;

constexpr char kCrashLog[] = "ux0:/data/azahar/crash.txt";

/// Escribe un mensaje en ux0:/data/azahar/crash.txt usando solo la API del
/// sistema: sirve incluso antes de que exista pantalla o runtime de C++.
void WriteCrashLog(const char* title, const char* detail) {
    sceIoMkdir("ux0:/data", 0777);
    sceIoMkdir("ux0:/data/azahar", 0777);
    const char* game_log = Common::VitaGameLog();
    for (const char* file : {static_cast<const char*>(kCrashLog), game_log}) {
        if (file[0] == '\0') {
            continue;
        }
        const SceUID fd = sceIoOpen(file, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
        if (fd < 0) {
            continue;
        }
        sceIoWrite(fd, title, std::strlen(title));
        sceIoWrite(fd, ": ", 2);
        if (detail != nullptr) {
            sceIoWrite(fd, detail, std::strlen(detail));
        }
        sceIoWrite(fd, "\n", 1);
        sceIoClose(fd);
    }
}

/**
 * El crash.txt de este juego (0.1.9.7): ux0:/data/azahar/crash_<nombre del
 * fichero de la ROM>.txt, con lo que no sea letra o numero cambiado por '_'.
 * Empieza de cero en cada partida con la version y los ajustes; el de la
 * partida anterior de ese juego queda como crash_<nombre>_anterior.txt.
 */
void StartGameLog(const std::string& rom_path) {
    std::string name = rom_path.substr(rom_path.find_last_of("/:") + 1);
    const std::size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) {
        name.resize(dot);
    }
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            c = '_';
        }
    }
    if (name.size() > 60) {
        name.resize(60);
    }
    const std::string base = "ux0:/data/azahar/crash_" + name;
    const std::string path = base + ".txt";
    const std::string previous = base + "_anterior.txt";
    Common::SetVitaGameLog(nullptr);
    sceIoRemove(previous.c_str());
    sceIoRename(path.c_str(), previous.c_str());
    Common::SetVitaGameLog(path.c_str());
    WriteCrashLog("version", kVersion);
    NoteUserSettings();
}

/// Extrae tipo y mensaje de la excepcion que esta provocando el terminate.
/// El tipo importa tanto como el mensaje: "Unknown error" a secas no dice nada,
/// pero sabiendo que viene de un std::system_error ya se sabe donde mirar.
std::string CurrentExceptionText() {
    if (!std::current_exception()) {
        // Sin excepcion en vuelo el terminate viene de un ASSERT de Azahar, que
        // deja apuntado el fichero y la linea antes de parar. El mensaje del
        // assert ya esta en crash.txt, escrito por el propio LOG_CRITICAL.
        const char* fatal = Common::LastFatalMessage();
        if (fatal != nullptr && fatal[0] != '\0') {
            return std::string("ASSERT en ") + fatal + " (mensaje completo en crash.txt)";
        }
        return "sin excepcion activa (terminate directo)";
    }
    try {
        std::rethrow_exception(std::current_exception());
    } catch (const std::system_error& e) {
        return std::string("system_error [") + std::to_string(e.code().value()) + " " +
               e.code().category().name() + "] " + e.what();
    } catch (const std::bad_alloc& e) {
        return std::string("bad_alloc (sin memoria): ") + e.what();
    } catch (const std::exception& e) {
        return std::string("[") + typeid(e).name() + "] " + e.what();
    } catch (...) {
        return "excepcion de tipo no estandar";
    }
}

/**
 * Manejador de terminate para ANTES de main().
 *
 * Los inicializadores estaticos (Citra tiene casi 200) corren antes que main,
 * asi que una excepcion ahi no la ve ni el try/catch ni el manejador que se
 * instala dentro. Y sin vita2d todavia no hay pantalla donde escribir, asi que
 * el mensaje va a un fichero.
 */
[[noreturn]] void OnTerminateEarly() {
    WriteCrashLog("Excepcion antes de main", CurrentExceptionText().c_str());
    sceKernelExitProcess(1);
    __builtin_unreachable();
}

/// Se instala en cuanto arranca el proceso, antes que los demas constructores.
/// Deja constancia de que llego hasta aqui, para poder distinguir "fallo antes
/// de arrancar nada" de "fallo mas adelante".
__attribute__((constructor(101))) void InstallEarlyTerminateHandler() {
    WriteCrashLog("inicio", "proceso arrancado, inicializadores estaticos por delante");
    std::set_terminate(OnTerminateEarly);
}

} // namespace

extern "C" {

/**
 * Implementacion propia de pthread_cancel.
 *
 * libstdc++ decide si el programa tiene hilos comprobando si resuelve una
 * referencia DEBIL a pthread_cancel (ver gthr-default.h). Sin ella,
 * __gthread_active_p() devuelve 0 y cualquier std::thread lanza
 * "Enable multithreading to use std::thread".
 *
 * Forzar la del VitaSDK con -Wl,-u arrastraba tambien pte_throw y
 * pte_osThreadCancel, la maquinaria de cancelacion de pthread-embedded, que
 * tiene su propio estado global. Definiendola aqui el simbolo resuelve igual,
 * el enlazador ya no saca ese objeto del archivo, y no entra nada mas.
 *
 * Azahar nunca cancela hilos, asi que esta funcion no llega a ejecutarse.
 */
#ifdef AZAHAR_ENABLE_THREADS
int pthread_cancel(pthread_t thread) {
    (void)thread;
    return ENOSYS;
}
#endif

/**
 * Guardas propias para los "static" locales de funcion.
 *
 * C++ protege la inicializacion de un static local frente a varios hilos
 * llamando a __cxa_guard_acquire/release. La implementacion de libstdc++ se
 * apoya en su maquinaria de espera atomica (bits/atomic_wait.h), que en esta
 * consola falla: con los hilos activos, las ~4600 guardas del binario tumbaban
 * la aplicacion antes incluso de llegar a main().
 *
 * La salida facil era compilar con -fno-threadsafe-statics, que quita las
 * guardas del todo. Arranca, pero deja las inicializaciones sin proteger: dos
 * hilos pueden inicializar el mismo objeto a la vez y corromper memoria.
 *
 * Aqui se sustituyen por una implementacion propia con un mutex normal y
 * corriente. Es mas lenta que la de libstdc++, pero la inicializacion de un
 * static ocurre una sola vez, asi que da igual.
 *
 * El mutex es recursivo a proposito: la inicializacion de un static puede
 * disparar la de otro, y hay que poder anidarlas sin bloquearse.
 */
static pthread_mutex_t g_static_guard_mutex;
static pthread_cond_t g_static_guard_cond;

__attribute__((constructor(102))) static void InitStaticGuardMutex() {
    pthread_mutex_init(&g_static_guard_mutex, nullptr);
    pthread_cond_init(&g_static_guard_cond, nullptr);
}

/**
 * Interceptores de las funciones de hilos.
 *
 * libstdc++ convierte el valor que devuelven en un std::system_error sin
 * mirarlo. Estamos viendo codigos como 0x8BBDC1CC, que cambian entre
 * ejecuciones y caen en rango de direcciones: no son errores, es memoria sin
 * inicializar. Estos envoltorios (activados con -Wl,--wrap) anotan quien
 * devuelve que, para saber cual de todas es.
 */
/// Escribe un hexadecimal de 8 digitos en buf y devuelve cuantos caracteres uso.
std::size_t AppendHex32(char* buf, unsigned int value) {
    buf[0] = '0';
    buf[1] = 'x';
    std::size_t n = 2;
    for (int shift = 28; shift >= 0; shift -= 4) {
        const unsigned int digit = (value >> shift) & 0xF;
        buf[n++] = static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
    }
    return n;
}

void LogThreadError(const char* what, int rc) {
    // Formateo a mano: la printf reducida de la Vita no es de fiar con %X.
    char buf[80];
    std::size_t n = 0;
    while (what[n] != '\0' && n < 40) {
        buf[n] = what[n];
        n++;
    }
    buf[n++] = ' ';
    n += AppendHex32(buf + n, static_cast<unsigned int>(rc));
    buf[n] = '\0';
    WriteCrashLog("hilos", buf);
}

/**
 * Interceptor de la funcion con la que libstdc++ lanza std::system_error.
 *
 * Ninguna de las funciones de hilos interceptadas devolvia error, asi que la
 * excepcion sale de otro sitio. Envolviendo directamente el punto por el que
 * pasan TODAS (std::thread, std::mutex, std::call_once, condition_variable...)
 * y anotando la direccion de retorno, esa direccion se traduce luego con
 * addr2line al fichero y la linea exactos de quien la lanza.
 */
[[noreturn]] void __real__ZSt20__throw_system_errori(int code);
[[noreturn]] void __wrap__ZSt20__throw_system_errori(int code) {
    char buf[80];
    std::size_t n = 0;
    const char* label = "system_error codigo ";
    while (label[n] != '\0') {
        buf[n] = label[n];
        n++;
    }
    n += AppendHex32(buf + n, static_cast<unsigned int>(code));
    const char* from = " lanzado desde ";
    for (std::size_t i = 0; from[i] != '\0'; i++) {
        buf[n++] = from[i];
    }
    n += AppendHex32(buf + n,
                     static_cast<unsigned int>(reinterpret_cast<std::uintptr_t>(
                         __builtin_return_address(0))));
    const char* tid = " hilo ";
    for (std::size_t i = 0; tid[i] != '\0'; i++) {
        buf[n++] = tid[i];
    }
    n += AppendHex32(buf + n, static_cast<unsigned int>(sceKernelGetThreadId()));
    buf[n] = '\0';
    WriteCrashLog("throw", buf);

    // __builtin_return_address da basura aqui (el compilador convierte esta
    // funcion en un salto directo y se pierde el marco). Se recorre la pila a
    // mano buscando direcciones que caigan dentro del codigo: esas son los
    // retornos de la cadena de llamadas, y traducidas dan la traza real.
    {
        const std::uintptr_t* stack =
            reinterpret_cast<const std::uintptr_t*>(__builtin_frame_address(0));
        char trace[96];
        std::size_t t = 0;
        std::size_t found = 0;
        for (std::size_t i = 0; i < 96 && found < 6; i++) {
            const std::uintptr_t value = stack[i];
            if (value >= 0x81000040u && value <= 0x812d0000u) {
                t += AppendHex32(trace + t, static_cast<unsigned int>(value));
                trace[t++] = ' ';
                found++;
            }
        }
        trace[t] = '\0';
        WriteCrashLog("traza", trace);
    }

    __real__ZSt20__throw_system_errori(code);
}

int __real_pthread_create(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*),
                          void* arg) {
    const int rc = __real_pthread_create(thread, attr, start, arg);
    if (rc != 0) {
        LogThreadError("pthread_create", rc);
    }
    return rc;
}

int __real_pthread_mutex_lock(pthread_mutex_t*);
int __wrap_pthread_mutex_lock(pthread_mutex_t* mutex) {
    const int rc = __real_pthread_mutex_lock(mutex);
    if (rc != 0) {
        LogThreadError("pthread_mutex_lock", rc);
    }
    return rc;
}

int __real_pthread_join(pthread_t, void**);
int __wrap_pthread_join(pthread_t thread, void** value) {
    const int rc = __real_pthread_join(thread, value);
    if (rc != 0) {
        LogThreadError("pthread_join", rc);
    }
    return rc;
}

int __real_pthread_detach(pthread_t);
int __wrap_pthread_detach(pthread_t thread) {
    const int rc = __real_pthread_detach(thread);
    if (rc != 0) {
        LogThreadError("pthread_detach", rc);
    }
    return rc;
}

// En el ABI de ARM la guarda es una palabra de 4 bytes: el bit 0 del primero
// indica "ya inicializado". Los demas quedan libres, y aqui se usa el segundo
// para marcar "hay un hilo inicializandolo ahora mismo".
int __cxa_guard_acquire(int* guard) {
    unsigned char* bytes = reinterpret_cast<unsigned char*>(guard);

    pthread_mutex_lock(&g_static_guard_mutex);
    while (true) {
        // Lectura con semantica de ADQUISICION, emparejada con la liberacion
        // de __cxa_guard_release: asegura ver el objeto completo, no solo el
        // flag. Coger el mutex ya ordena, pero dejarlo explicito evita que un
        // cambio futuro lo rompa sin darse cuenta.
        if ((__atomic_load_n(&bytes[0], __ATOMIC_ACQUIRE) & 1) != 0) {
            pthread_mutex_unlock(&g_static_guard_mutex);
            return 0; // ya inicializado, el llamante no hace nada
        }
        if (bytes[1] == 0) {
            // Nadie lo esta haciendo: marcarlo y SOLTAR el mutex.
            //
            // Soltarlo aqui es lo importante. La version anterior devolvia con
            // el mutex cogido durante toda la inicializacion, y como es uno
            // solo para las 4641 guardas del binario, bastaba con que un
            // inicializador esperase a otro hilo para bloquear el programa
            // entero: ese otro hilo no podia ni empezar un static distinto.
            bytes[1] = 1;
            pthread_mutex_unlock(&g_static_guard_mutex);
            return 1;
        }
        // Otro hilo esta con esta misma guarda: esperar a que termine.
        pthread_cond_wait(&g_static_guard_cond, &g_static_guard_mutex);
    }
}

void __cxa_guard_release(int* guard) {
    unsigned char* bytes = reinterpret_cast<unsigned char*>(guard);

    pthread_mutex_lock(&g_static_guard_mutex);
    bytes[1] = 0;

    // El flag se escribe con semantica de LIBERACION, y esto es imprescindible
    // en ARM.
    //
    // Cuando otro hilo pasa por un static ya inicializado, GCC no llama a esta
    // funcion: comprueba el bit en linea (ldr + dmb + tst) y usa el objeto
    // directamente. Esa barrera suya solo sirve si al escribir el flag se
    // garantiza que las escrituras del objeto ya son visibles.
    //
    // Coger el mutex es una barrera de ADQUISICION, no de liberacion, asi que
    // no ordenaba nada respecto a la construccion del objeto, que ocurrio antes
    // de entrar aqui. En x86 da igual porque su modelo de memoria es fuerte; en
    // ARM el otro hilo podia ver el flag puesto y el objeto a medio escribir,
    // incluido su puntero de tabla virtual: la siguiente llamada virtual
    // saltaba a una direccion invalida.
    __atomic_store_n(&bytes[0], static_cast<unsigned char>(1), __ATOMIC_RELEASE);

    pthread_mutex_unlock(&g_static_guard_mutex);
    pthread_cond_broadcast(&g_static_guard_cond);
}

void __cxa_guard_abort(int* guard) {
    // El inicializador lanzo: dejar la guarda libre para que otro reintente.
    volatile unsigned char* bytes = reinterpret_cast<volatile unsigned char*>(guard);

    pthread_mutex_lock(&g_static_guard_mutex);
    bytes[1] = 0;
    pthread_mutex_unlock(&g_static_guard_mutex);
    pthread_cond_broadcast(&g_static_guard_cond);
}

} // extern "C"

namespace {

/// Espera a que el usuario pulse CIRCULO.
void WaitForCircle() {
    SceCtrlData pad{};
    // Primero esperar a que suelte, para no comerse la pulsacion anterior.
    do {
        sceCtrlPeekBufferPositive(0, &pad, 1);
    } while (pad.buttons & SCE_CTRL_CIRCLE);
    do {
        sceCtrlPeekBufferPositive(0, &pad, 1);
    } while (!(pad.buttons & SCE_CTRL_CIRCLE));
}

/// Memoria de usuario libre en KB, o -1 si no se puede consultar.
int FreeMemoryKB() {
    SceKernelFreeMemorySizeInfo info{};
    info.size = sizeof(info);
    if (sceKernelGetFreeMemorySize(&info) < 0) {
        return -1;
    }
    return info.size_user / 1024;
}

/**
 * CDRAM libre, en KB.
 *
 * Son 128 MB de memoria dedicada al chip grafico que NO salen del presupuesto
 * de RAM de usuario, y hoy solo la toca vita2d para sus buffers de pantalla. El
 * backend GXM va a vivir ahi, asi que conviene ver el numero desde el principio:
 * es el unico sitio donde se puede gastar memoria sin quitarsela al heap.
 */
int FreeCdramKB() {
    SceKernelFreeMemorySizeInfo info{};
    info.size = sizeof(info);
    if (sceKernelGetFreeMemorySize(&info) < 0) {
        return -1;
    }
    return info.size_cdram / 1024;
}

/**
 * Presupuesto total de RAM de usuario, en KB, deducido.
 *
 * No se puede leer directamente: sceKernelGetFreeMemorySize dice lo que QUEDA,
 * y para cuando corre main() el heap de newlib ya esta reservado entero (lo
 * hace libc antes de entrar). Sumandoselo de vuelta sale el presupuesto con el
 * que arranco el proceso, menos el binario y las reservas del sistema -- que
 * son unos pocos MB y no cambian la conclusion.
 *
 * Llamarlo PRONTO importa: cada hilo y cada textura que se cree despues salen
 * de aqui, y entonces el numero ya no es el presupuesto sino lo que sobra.
 */
int TotalUserMemoryKB() {
    const int free_kb = FreeMemoryKB();
    if (free_kb < 0) {
        return -1;
    }
    return free_kb + static_cast<int>(AZAHAR_HEAP_MB) * 1024;
}

/**
 * Si la consola ha concedido de verdad el modo de memoria ampliada.
 *
 * No hay ninguna llamada que responda "ATTRIBUTE2=12 ha sido aceptado": lo
 * unico observable es cuanta memoria hay. Una aplicacion normal se mueve en
 * ~256 MB de presupuesto y el modo ampliado lo sube a ~365, asi que el total
 * deducido cae claramente a un lado u otro de 300 MB y nunca cerca.
 *
 * Es una deduccion, no una lectura, y por eso lo que se pinta en pantalla es la
 * cifra ADEMAS del SI/no: si algun dia el umbral se queda mal puesto, el numero
 * sigue diciendo la verdad.
 */
bool ExtendedMemoryGranted() {
    const int total_kb = TotalUserMemoryKB();
    return total_kb > 300 * 1024;
}

/// Muestra un error a pantalla completa con el estado de memoria y espera.
void DrawFatalError(vita2d_pgf* font, const char* title, const std::string& detail) {
    vita2d_start_drawing();
    vita2d_clear_screen();
    vita2d_pgf_draw_text(font, 40, 80, 0xFF4040FF, 1.3f, title);

    // El texto puede ser largo: se parte en lineas de ~70 caracteres.
    int y = 130;
    for (std::size_t i = 0; i < detail.size() && y < 400; i += 70) {
        vita2d_pgf_draw_text(font, 40, y, kColorText, 0.95f, detail.substr(i, 70).c_str());
        y += 26;
    }

    vita2d_pgf_draw_textf(font, 40, 440, kColorDim, 0.95f, "Memoria de usuario libre: %d KB",
                          FreeMemoryKB());

    // El heap de libc es cosa aparte de la memoria libre de la consola: lo
    // reserva newlib entero antes de main(), asi que la linea de arriba puede
    // decir que sobra memoria mientras el heap esta a tope. Justo eso paso con
    // el error "Sin memoria" al cargar una ROM: se persiguio el presupuesto de
    // la aplicacion y de la disposicion del binario durante dos versiones
    // cuando lo que faltaba era heap. Con estas cifras se ve de un vistazo.
    const struct mallinfo info = mallinfo();
    vita2d_pgf_draw_textf(font, 40, 464, kColorDim, 0.95f,
                          "Heap: %u MB totales, %d KB en uso, %d KB libres dentro",
                          static_cast<unsigned int>(AZAHAR_HEAP_MB),
                          static_cast<int>(info.uordblks / 1024),
                          static_cast<int>(info.fordblks / 1024));
    vita2d_pgf_draw_text(font, 40, 500, kColorDim, 0.95f, "CIRCULO: volver al menu");
    vita2d_end_drawing();
    vita2d_swap_buffers();

    WaitForCircle();
}

/// Carga y ejecuta un juego hasta que el usuario sale o falla.
void RunGame(vita2d_pgf* font, const std::string& path) {
    DrawMessage(font, "Cargando...", path.c_str());
    LOG_INFO(Frontend, "Cargando {} (memoria libre: {} KB)", path, FreeMemoryKB());

    // Las rutas que use el emulador quedan anotadas: si algo peta recorriendo
    // directorios, saber cual estaba mirando es media respuesta.
    StartGameLog(path);
    WriteCrashLog("carga", path.c_str());
    WriteCrashLog("ruta user", FileUtil::GetUserPath(FileUtil::UserPath::UserDir).c_str());
    WriteCrashLog("ruta sdmc", FileUtil::GetUserPath(FileUtil::UserPath::SDMCDir).c_str());
    WriteCrashLog("ruta nand", FileUtil::GetUserPath(FileUtil::UserPath::NANDDir).c_str());

    auto& system = Core::System::GetInstance();

    WriteCrashLog("carga", "creando ventana de emulacion");
    auto window = std::make_unique<VitaFrontend::EmuWindow_Vita>();
    window->SetStatsFont(font);

    WriteCrashLog("carga", "llamando a System::Load");

    // Azahar esta escrito para PC, donde una reserva de memoria no falla nunca:
    // lanza bad_alloc y nadie la captura. Sin este try, en la Vita eso acaba en
    // un abort() silencioso y la consola solo dice "la aplicacion ha fallado".
    Core::System::ResultStatus load_result;
    try {
        load_result = system.Load(*window, path);
    } catch (const std::bad_alloc&) {
        LOG_ERROR(Frontend, "Sin memoria al cargar {}", path);
        DrawFatalError(font, "Sin memoria",
                       "No hay memoria suficiente para cargar este juego. Prueba con uno mas "
                       "pequeno, o recompila ajustando AZAHAR_HEAP_MB.");
        if (system.IsPoweredOn()) {
            system.Shutdown();
        }
        return;
    } catch (const std::exception& e) {
        LOG_ERROR(Frontend, "Excepcion al cargar {}: {}", path, e.what());
        DrawFatalError(font, "Error al cargar", e.what());
        if (system.IsPoweredOn()) {
            system.Shutdown();
        }
        return;
    }
    if (load_result != Core::System::ResultStatus::Success) {
        LOG_ERROR(Frontend, "Fallo al cargar {}: {}", path, static_cast<int>(load_result));
        DrawFatalError(font, LoadErrorMessage(load_result), system.GetStatusDetails());
        if (system.IsPoweredOn()) {
            system.Shutdown();
        }
        return;
    }

    LOG_INFO(Frontend, "Juego cargado (memoria libre: {} KB)", FreeMemoryKB());
    if (Settings::GetWorkingGraphicsAPI() == Settings::GraphicsAPI::GXM) {
        // Los shaders que este juego uso en partidas anteriores, a memoria antes
        // de empezar (ver PreloadCgCache). Aun no hay emulacion: vita2d es solo
        // de este hilo.
        u64 program_id = 0;
        system.GetAppLoader().ReadProgramId(program_id);
        DrawShaderLoading(font, 0, 0);
        const u32 loaded = Gxm::PreloadCgCache(
            program_id, [font](u32 done, u32 total) { DrawShaderLoading(font, done, total); });
        if (loaded == 0) {
            // Primera partida de este juego: que se lea el aviso.
            sceKernelDelayThread(800 * 1000);
        }
    }
    WriteCrashLog("carga", "System::Load OK, entrando al bucle de emulacion");
    system.RegisterCoreLoopThreadId();

    // Mientras el juego no haya pintado su primer fotograma, la pantalla se
    // queda con el "Cargando..." y no hay forma de saber si avanza o esta
    // colgado. Este contador se dibuja hasta que llegue ese primer fotograma.
    const SceUInt64 start_us = sceKernelGetProcessTimeWide();

    // Contador de 32 bits a proposito: la printf reducida de la Vita no
    // implementa los especificadores de 64 bits (%llu). Pasarle uno descuadra
    // los argumentos, lee basura y acaba saltando a una direccion invalida.
    // Es el mismo fallo que tenia sscanf con %zd.
    unsigned int loops = 0;
    /**
     * VIGILANTE DE CONGELACIONES (0.1.9.0). En 0.1.8.9, al pulsar A en el
     * titulo de Zafiro Alfa el emulador se quedaba parado sin crashear y sin
     * escribir nada mas en crash.txt. Este hilo mira si el bucle avanza; si
     * pasan 6 segundos (y otra vez a los 30) sin una vuelta, anota en que paso
     * esta el hilo de emulacion (Common::vita_stage) y los PC de los ARM11.
     */
    std::atomic<unsigned int> heartbeat{0};
    std::atomic<bool> watchdog_stop{false};
    std::thread watchdog([&heartbeat, &watchdog_stop, &system] {
        unsigned int last = ~0u;
        unsigned int still = 0;
        while (!watchdog_stop.load(std::memory_order_relaxed)) {
            sceKernelDelayThread(1000 * 1000);
            const unsigned int now = heartbeat.load(std::memory_order_relaxed);
            if (now != last) {
                last = now;
                still = 0;
                continue;
            }
            still++;
            if (still == 6 || still == 30) {
                const std::string text = fmt::format(
                    "{} s sin avanzar, en '{}' (ciclo {}), pc0 {:#010x} pc1 {:#010x}, hilo de "
                    "la gpu en '{}'",
                    still, Common::vita_stage.load(std::memory_order_relaxed), now,
                    system.GetCore(0).GetPC(), system.GetCore(1).GetPC(),
                    Common::vita_gpu_stage.load(std::memory_order_relaxed));
                WriteCrashLog("congelado", text.c_str());
            }
        }
    });

    bool user_quit = false;

    try {
        while (!window->ShouldExit() && !user_quit) {
            // La ventana solo mira los botones cuando el juego termina un
            // fotograma. Si no llega ninguno, no habria forma de salir, asi que
            // la combinacion se comprueba tambien aqui.
            if ((loops % 16) == 0) {
                SceCtrlData pad{};
                sceCtrlPeekBufferPositive(0, &pad, 1);
                if ((pad.buttons & SCE_CTRL_START) && (pad.buttons & SCE_CTRL_SELECT)) {
                    user_quit = true;
                }
            }

            const bool loading_screen = window->FramesPresented() == 0 && (loops % 64) == 0;
            if (loading_screen) {
                // vita2d desde este hilo solo con el de la GPU parado: con la GPU
                // en otro nucleo, la primera presentacion puede estar en marcha.
                system.GPU().Sync();
            }
            if (loading_screen && window->FramesPresented() == 0) {
                const unsigned int seconds =
                    static_cast<unsigned int>((sceKernelGetProcessTimeWide() - start_us) / 1000000);
                vita2d_start_drawing();
                vita2d_clear_screen();
                vita2d_pgf_draw_text(font, 40, 80, kColorAccent, 1.3f, "Emulando...");
                vita2d_pgf_draw_text(font, 40, 130, kColorText, 0.95f,
                                     "El juego aun no ha dibujado su primer fotograma.");
                vita2d_pgf_draw_textf(font, 40, 170, kColorDim, 0.95f,
                                      "Tiempo: %u s    Ciclos de emulacion: %u", seconds, loops);
                vita2d_pgf_draw_textf(font, 40, 210, kColorDim, 0.95f,
                                      "Memoria libre: %d KB", FreeMemoryKB());

                // Si algun hilo murio por una excepcion, se avisa aqui sin
                // tapar el contador: interesa mas saber si la emulacion avanza.
                if (!g_last_error.empty()) {
                    vita2d_pgf_draw_text(font, 40, 280, 0xFF4040FF, 0.9f,
                                         "Un hilo murio por una excepcion:");
                    vita2d_pgf_draw_text(font, 40, 308, kColorDim, 0.85f,
                                         g_last_error.substr(0, 74).c_str());
                }

                vita2d_pgf_draw_text(font, 40, 500, kColorDim, 0.9f,
                                     "START + SELECT: volver al menu");
                vita2d_end_drawing();
                vita2d_swap_buffers();
            }
            // Rastro del avance en crash.txt: si el proceso muere de golpe, la
            // ultima marca dice cuantos ciclos llego a dar y con cuanta memoria.
            //
            // Los primeros 64 ciclos se anotan uno a uno porque es justo donde
            // se ha estado muriendo; a partir de ahi solo las potencias de dos,
            // para no castigar la tarjeta de memoria con escrituras constantes.
            if (loops < 64 || (loops & (loops - 1)) == 0) {
                char mark[96];
                std::size_t m = 0;
                const char* label = "ciclo ";
                while (label[m] != '\0') {
                    mark[m] = label[m];
                    m++;
                }
                m += AppendHex32(mark + m, loops);
                const char* mem = " memoria KB ";
                for (std::size_t i = 0; mem[i] != '\0'; i++) {
                    mark[m++] = mem[i];
                }
                m += AppendHex32(mark + m, static_cast<unsigned int>(FreeMemoryKB()));

                // El contador de ciclos dice cuanto avanza, pero no donde. El PC
                // de los dos nucleos ARM11 si: una direccion que no se mueve
                // entre marcas es un juego colgado, y una que salta a un sitio
                // absurdo es codigo emulado corriendo por donde no debe.
                const char* pcs = " pc0 ";
                for (std::size_t i = 0; pcs[i] != '\0'; i++) {
                    mark[m++] = pcs[i];
                }
                m += AppendHex32(mark + m, system.GetCore(0).GetPC());
                const char* pc1 = " pc1 ";
                for (std::size_t i = 0; pc1[i] != '\0'; i++) {
                    mark[m++] = pc1[i];
                }
                m += AppendHex32(mark + m, system.GetCore(1).GetPC());
                mark[m] = '\0';
                WriteCrashLog("avance", mark);
            }
            loops++;
            heartbeat.store(loops, std::memory_order_relaxed);

            Common::vita_stage.store("emulando", std::memory_order_relaxed);
            const Core::System::ResultStatus result = system.RunLoop();
            if (result != Core::System::ResultStatus::Success &&
                result != Core::System::ResultStatus::ShutdownRequested) {
                LOG_ERROR(Frontend, "RunLoop devolvio {}: {}", static_cast<int>(result),
                          system.GetStatusDetails());
                WriteCrashLog("RunLoop error", system.GetStatusDetails().c_str());
                system.GPU().StopThread();
                DrawFatalError(font, "Error durante la emulacion", system.GetStatusDetails());
                break;
            }
            if (result == Core::System::ResultStatus::ShutdownRequested) {
                break;
            }
        }
    } catch (const std::bad_alloc&) {
        // Anotado tambien en crash.txt: la pantalla se puede perder si el
        // usuario cierra antes de leerla, el fichero no.
        WriteCrashLog("emulacion", "bad_alloc: sin memoria durante la emulacion");
        LOG_ERROR(Frontend, "Sin memoria durante la emulacion");
        system.GPU().StopThread();
        DrawFatalError(font, "Sin memoria durante la emulacion",
                       "El juego cargo pero se agoto la memoria al ejecutarlo.");
    } catch (const std::exception& e) {
        std::string detail = std::string("[") + typeid(e).name() + "] " + e.what();
        WriteCrashLog("emulacion", detail.c_str());
        LOG_ERROR(Frontend, "Excepcion durante la emulacion: {}", e.what());
        system.GPU().StopThread();
        DrawFatalError(font, "Error durante la emulacion", detail);
    } catch (...) {
        WriteCrashLog("emulacion", "excepcion de tipo no estandar");
        system.GPU().StopThread();
        DrawFatalError(font, "Error durante la emulacion",
                       "Excepcion de un tipo no estandar.");
    }
    watchdog_stop.store(true, std::memory_order_relaxed);
    watchdog.join();

    // Resumen de la cache de traduccion al cerrar la partida (salga como salga:
    // el usuario cerro, o algo de lo de arriba lo corto). Sirve para decidir si
    // agrandar TRANS_CACHE_MB compensa: "capacidad" es la unica cifra que tiene
    // que ver con el tamano del buffer, "invalidacion" pasaria igual con un
    // buffer infinito. Ver el comentario de GetTransCacheFlushCounts.
    {
        std::size_t flushes_capacity = 0;
        std::size_t flushes_invalidation = 0;
        GetTransCacheFlushCounts(flushes_capacity, flushes_invalidation);
        char summary[96] = "vaciados por capacidad ";
        std::size_t n = std::strlen(summary);
        n += AppendHex32(summary + n, static_cast<unsigned int>(flushes_capacity));
        const char* sep = " por invalidacion ";
        for (std::size_t i = 0; sep[i] != '\0'; i++) {
            summary[n++] = sep[i];
        }
        n += AppendHex32(summary + n, static_cast<unsigned int>(flushes_invalidation));
        summary[n] = '\0';
        WriteCrashLog("cache de traduccion", summary);
    }

    system.Shutdown();
    // El menu se dibuja con la espera al refresco, aunque la partida fuera sin limite.
    vita2d_set_vblank_wait(1);
}

/**
 * Ultimo recurso ante una excepcion no capturada.
 *
 * El try/catch de main() solo cubre el hilo principal: una excepcion lanzada en
 * cualquier otro hilo (el de audio, el de logging, los del emulador) llama a
 * std::terminate directamente y el proceso muere sin que main se entere.
 *
 * Este manejador recupera la excepcion en curso y la pinta en pantalla, venga
 * del hilo que venga.
 */
[[noreturn]] void OnTerminate() {
    const std::string detail = CurrentExceptionText();
    WriteCrashLog("Excepcion no capturada", detail.c_str());
    g_last_error = detail;

    // Se probo a matar solo el hilo que lanza, dejando viva la emulacion. No
    // sirve: System::Load se queda esperando a ese hilo y la carga no termina
    // nunca, asi que la consola parece colgada y encima no se aprende nada.
    // Mejor cerrar y quedarse con el registro, que es lo que informa.
    (void)g_main_thread_id;

    // Ojo: este manejador puede correr en CUALQUIER hilo. Si se queda esperando
    // a que el usuario pulse un boton, el resto de hilos siguen vivos y pintando
    // encima, asi que el mensaje no se ve y la consola parece colgada -- que es
    // exactamente lo que pasaba. Se muestra unos segundos y se sale solo.
    if (g_font != nullptr) {
        for (int frame = 0; frame < 60 * 8; frame++) {
            vita2d_start_drawing();
            vita2d_clear_screen();
            vita2d_pgf_draw_text(g_font, 40, 80, 0xFF4040FF, 1.3f, "Excepcion no capturada");
            int y = 130;
            for (std::size_t i = 0; i < detail.size() && y < 420; i += 70) {
                vita2d_pgf_draw_text(g_font, 40, y, kColorText, 0.95f,
                                     detail.substr(i, 70).c_str());
                y += 26;
            }
            vita2d_pgf_draw_textf(g_font, 40, 460, kColorDim, 0.95f,
                                  "Memoria de usuario libre: %d KB", FreeMemoryKB());
            vita2d_pgf_draw_textf(g_font, 40, 500, kColorDim, 0.95f,
                                  "Anotado en ux0:/data/azahar/crash.txt  -  cerrando en %d s",
                                  8 - frame / 60);
            vita2d_end_drawing();
            vita2d_swap_buffers();
        }
    }

    sceKernelExitProcess(1);
    __builtin_unreachable();
}

/// Pinta en que paso del arranque estamos. Si la app muere, lo ultimo que se
/// vea en pantalla dice exactamente donde.
void DrawStep(vita2d_pgf* font, const char* step) {
    WriteCrashLog("paso", step);
    vita2d_start_drawing();
    vita2d_clear_screen();
    vita2d_pgf_draw_text(font, 40, 60, kColorAccent, 1.3f, "Azahar  -  PS Vita");
    vita2d_pgf_draw_text(font, 40, 90, kColorDim, 0.85f, kVersion);
    vita2d_pgf_draw_text(font, 40, 140, kColorText, 1.0f, "Iniciando...");
    vita2d_pgf_draw_text(font, 40, 180, kColorDim, 1.0f, step);

    // El presupuesto de memoria, a la vista desde el primer fotograma.
    //
    // Es el numero que decide todo lo demas -- si el heap puede subir, si el
    // modo New 3DS es viable, cuanto sitio queda para el backend GXM -- y hasta
    // ahora solo aparecia en la pantalla de error fatal, o sea justo cuando ya
    // no sirve para decidir nada. Aqui se ve siempre, se puede fotografiar y no
    // hace falta reproducir ningun fallo para leerlo.
    const bool extended = ExtendedMemoryGranted();
    vita2d_pgf_draw_textf(font, 40, 232, extended ? 0xFF80FF80 : 0xFF60C0FF, 0.85f,
                          "mem usuario %d MB totales, %d MB libres  (modo ampliado %s)",
                          TotalUserMemoryKB() / 1024, FreeMemoryKB() / 1024,
                          extended ? "SI" : "no");
    vita2d_pgf_draw_textf(font, 40, 256, kColorDim, 0.85f, "heap %u MB    CDRAM libre %d MB",
                          static_cast<unsigned int>(AZAHAR_HEAP_MB), FreeCdramKB() / 1024);

    vita2d_end_drawing();
    vita2d_swap_buffers();
}

/// Cuerpo real del programa. main() lo envuelve para poder enseñar por pantalla
/// cualquier excepcion en vez de morir con un abort() mudo.
void Run(vita2d_pgf* font) {
    DrawStep(font, "1/7 creando carpetas");
    CreateDirectories();
    FileUtil::SetUserPath(kUserDir);

    DrawStep(font, "2/7 iniciando registro de eventos");
    Common::Log::Initialize();
    Common::Log::SetGlobalFilter(Common::Log::Filter(Common::Log::Level::Info));
    Common::Log::Start();
    // La version va en la PRIMERA linea del log a proposito: con varios ciclos
    // de compilar, instalar y mirar, la duda "¿pero que build es esta?" se
    // resuelve abriendo el fichero, sin tener que cruzar fechas.
    LOG_INFO(Frontend, "Azahar para PS Vita iniciando ({})", kVersion);

    DrawStep(font, "3/7 aplicando ajustes");
    ConfigureSettings();
    // Encima, lo que el usuario eligio en la ventana de ajustes la ultima vez.
    LoadUserSettings();
    ApplyUserSettings();
    NoteUserSettings();

    DrawStep(font, "4/7 configurando mandos");
    VitaFrontend::Input::Init();
    VitaFrontend::Input::ApplyDefaultMapping();

    DrawStep(font, "5/7 preparando el nucleo");

    // CryptoPP inicializa sus tablas de punteros a funcion de forma perezosa,
    // la primera vez que se construye un Integer. Ese camino tiene una carrera
    // que en ARM si se manifiesta (ver integer.cpp): si dos hilos entran a la
    // vez, uno puede llamar a un puntero aun sin escribir y saltar a basura.
    //
    // Construir uno aqui, con un solo hilo en marcha, deja las tablas listas
    // antes de que el emulador arranque los suyos.
    {
        const CryptoPP::Integer warm_up(1);
        (void)warm_up;
    }

    auto& system = Core::System::GetInstance();

    DrawStep(font, "6/7 registrando applets");
    Frontend::RegisterDefaultApplets(system);
    system.RegisterImageInterface(std::make_shared<Frontend::ImageInterface>());

    DrawStep(font, "7/7 buscando juegos");
    std::vector<RomEntry> roms = ScanRoms();
    int selected = 0;
    int scroll = 0;
    bool running = true;

    // Estado anterior de los botones: sin esto un solo toque de la cruceta
    // recorreria la lista entera, porque el menu se dibuja a 60 fps.
    unsigned int previous_buttons = 0;
    // Cuando empezo a mantenerse START (0 = no se esta manteniendo). Solo
    // cuenta una pulsacion NUEVA: el START que viene mantenido de salir de un
    // juego (START+SELECT) o de cerrar los ajustes no abre nada.
    SceUInt64 start_hold_begin = 0;

    while (running) {
        SceCtrlData pad{};
        sceCtrlPeekBufferPositive(0, &pad, 1);
        const unsigned int pressed = pad.buttons & ~previous_buttons;
        previous_buttons = pad.buttons;

        float settings_hold = 0.0f;
        const bool start_alone =
            (pad.buttons & SCE_CTRL_START) != 0 && (pad.buttons & SCE_CTRL_SELECT) == 0;
        if (!start_alone) {
            start_hold_begin = 0;
        } else {
            const SceUInt64 now = sceKernelGetProcessTimeWide();
            if (pressed & SCE_CTRL_START) {
                start_hold_begin = now;
            }
            if (start_hold_begin != 0) {
                const SceUInt64 held = now - start_hold_begin;
                settings_hold = static_cast<float>(held) / static_cast<float>(kSettingsHoldUs);
                if (held >= kSettingsHoldUs) {
                    RunSettingsMenu(font);
                    start_hold_begin = 0;
                    settings_hold = 0.0f;
                    previous_buttons = ~0u;
                    continue;
                }
            }
        }

        if (pressed & SCE_CTRL_DOWN) {
            selected = std::min(selected + 1, static_cast<int>(roms.size()) - 1);
        }
        if (pressed & SCE_CTRL_UP) {
            selected = std::max(selected - 1, 0);
        }
        if (pressed & SCE_CTRL_CIRCLE) {
            running = false;
        }
        if (pressed & SCE_CTRL_SELECT) {
            // Alterna el motor grafico del siguiente juego. El renderer se
            // construye en System::Load, con estos ajustes ya puestos.
            const bool gxm =
                Settings::values.graphics_api.GetValue() == Settings::GraphicsAPI::GXM;
            Settings::values.graphics_api =
                gxm ? Settings::GraphicsAPI::Software : Settings::GraphicsAPI::GXM;
        }
        if ((pressed & SCE_CTRL_CROSS) && !roms.empty()) {
            RunGame(font, roms[selected].path);
            // Al volver del juego hay que reiniciar el estado de botones para
            // que la combinacion de salida no se lea aqui como una pulsacion.
            previous_buttons = ~0u;
            vita2d_set_clear_color(kColorBackground);
        }

        selected = std::clamp(selected, 0, std::max(0, static_cast<int>(roms.size()) - 1));
        if (selected < scroll) {
            scroll = selected;
        } else if (selected >= scroll + 14) {
            scroll = selected - 13;
        }

        DrawMenu(font, roms, selected, scroll, settings_hold);
    }

    VitaFrontend::Input::Shutdown();
    Common::Log::Stop();
}

} // Anonymous namespace

int main(int argc, char** argv) {
    // El Cortex-A9 de la Vita va a 333 MHz por defecto. Emular un ARM11 con un
    // interprete es lo mas caro que puede hacer esta consola, asi que se sube
    // todo al maximo que permite el sistema.
    //
    // En una compilacion "safe" (la de por defecto) estas llamadas pueden ser
    // rechazadas y devolver error. No pasa nada: se sigue a la velocidad
    // normal, solo que mas lento. Arrancar importa mas que el overclock.
    // Marcas de grano fino: si el proceso muere, la ultima linea de crash.txt
    // dice exactamente hasta donde llego.
    // La VERSION, lo primero de todo.
    //
    // Sin esto un crash.txt no se puede atribuir a una compilacion concreta, y
    // un volcado de memoria no sirve de nada: las direcciones solo casan con el
    // azahar.elf de SU build, y ese fichero se sobrescribe en cada compilacion.
    // Ha pasado dos veces: llego un volcado, las direcciones resolvieron a
    // funciones sin ninguna relacion entre si, y la unica conclusion posible fue
    // "este ELF no es el de esa build".
    //
    // build.sh guarda ahora una copia del ELF por version. Con esta linea se
    // sabe cual hay que usar.
    //
    // UN crash.txt POR SESION (0.1.7.3). El fichero crecia sin fin (6 MB con
    // sesiones desde 0.1.4), y el que llegaba para analizar era a menudo el de
    // una version vieja sin que se notara. Ahora cada arranque empieza uno
    // nuevo y el de la sesion anterior queda en crash_anterior.txt.
    sceIoMkdir("ux0:/data", 0777);
    sceIoMkdir("ux0:/data/azahar", 0777);
    sceIoRemove("ux0:/data/azahar/crash_anterior.txt");
    sceIoRename(kCrashLog, "ux0:/data/azahar/crash_anterior.txt");
    // Lo mismo con los shaders que no compilan: solo los de esta sesion.
    sceIoRemove("ux0:/data/azahar/cg_error_anterior.txt");
    sceIoRename("ux0:/data/azahar/cg_error.txt", "ux0:/data/azahar/cg_error_anterior.txt");
    WriteCrashLog("version", kVersion);
    WriteCrashLog("main", "entrando (inicializadores estaticos superados)");

    scePowerSetArmClockFrequency(444);
    scePowerSetBusClockFrequency(222);
    scePowerSetGpuClockFrequency(222);
    scePowerSetGpuXbarClockFrequency(166);

    // Leer de vuelta lo que la consola ha concedido de verdad.
    //
    // Estas llamadas pueden ser rechazadas y nadie miraba el resultado. Todas
    // las cuentas de "ciclos por pixel" de la sesion daban 444 MHz por hecho
    // sin ninguna prueba de que lo fuera. Si sale 333, esas cuentas iban un 33%
    // infladas y ademas hay un tercio de rendimiento ahi parado.
    // FPU en modo rapido ANTES de emular nada. Ver VitaEnableFastFloatMode.
    Common::VitaEnableFastFloatMode();

    Common::g_arm_clock_mhz = scePowerGetArmClockFrequency();
    {
        char buffer[96];
        const auto result = fmt::format_to_n(
            buffer, sizeof(buffer) - 1, "arm {} MHz, bus {} MHz, gpu {} MHz (pedidos 444/222/222)",
            Common::g_arm_clock_mhz, scePowerGetBusClockFrequency(),
            scePowerGetGpuClockFrequency());
        *result.out = 0;
        WriteCrashLog("relojes", buffer);
    }

    // El presupuesto de memoria, anotado ANTES de reservar nada.
    //
    // Va a crash.txt ademas de a la pantalla porque el escenario que importa es
    // justo el que no se puede fotografiar: si la aplicacion muere despues, esta
    // linea dice con cuanta memoria habia arrancado. Y si ATTRIBUTE2=12 dejara
    // de concederse en una actualizacion de firmware, aqui se veria de inmediato
    // en vez de aparecer como un fallo de reserva sin relacion aparente.
    {
        char buffer[128];
        const auto result =
            fmt::format_to_n(buffer, sizeof(buffer) - 1,
                             "usuario {} MB totales / {} MB libres, heap {} MB, CDRAM {} MB, "
                             "modo ampliado {}",
                             TotalUserMemoryKB() / 1024, FreeMemoryKB() / 1024,
                             static_cast<unsigned int>(AZAHAR_HEAP_MB), FreeCdramKB() / 1024,
                             ExtendedMemoryGranted() ? "SI" : "no");
        *result.out = 0;
        WriteCrashLog("memoria", buffer);
    }

    // El hilo principal es el que emula la CPU del 3DS, y esa es la parte de
    // todo esto que mas machaca la cache: la cache de traduccion son 24 MB por
    // los que va saltando. Atarlo a un nucleo fijo evita que el planificador lo
    // mueva y tire a la basura lo que ya tenia caliente en L1/L2.
    //
    // No compite con los hilos del rasterizador aunque comparta nucleo con uno
    // de ellos: mientras se rasteriza, este hilo esta bloqueado esperandolos.
    Common::VitaPinThreadToUserCore(0, "emulacion");

    // Antes de nada util: averiguar si esta consola permite JIT. El resultado va
    // a crash.txt bajo "jit" y no afecta a nada mas -- solo informa. Ver
    // vita_jit_probe.cpp para por que hace falta saberlo antes de escribir un
    // recompilador y no despues.
    VitaFrontend::ProbeJitSupport();

    vita2d_init();
    WriteCrashLog("main", "vita2d_init hecho");

    vita2d_set_clear_color(kColorBackground);
    vita2d_pgf* font = vita2d_load_default_pgf();
    WriteCrashLog("main", "fuente cargada");
    g_font = font;
    g_main_thread_id = sceKernelGetThreadId();

    // Cubre las excepciones de cualquier hilo; el try de abajo solo cubre este.
    std::set_terminate(OnTerminate);

    // Una excepcion que escape hasta aqui acaba en abort(), y la consola solo
    // dice "la aplicacion ha fallado" sin mas pistas. Capturarla y pintarla en
    // pantalla convierte un crash mudo en un diagnostico legible.
    try {
        Run(font);
    } catch (const std::exception& e) {
        DrawFatalError(font, "Error fatal al iniciar", e.what());
    } catch (...) {
        DrawFatalError(font, "Error fatal al iniciar", "Excepcion de tipo desconocido.");
    }

    vita2d_free_pgf(font);
    vita2d_fini();
    sceKernelExitProcess(0);
    return 0;
}
