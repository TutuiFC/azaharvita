// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
//
// Punto de entrada de Azahar en PS Vita.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <pthread.h>
#include <string>
#include <system_error>
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
#include "citra_vita/vita_window.h"
#include "common/file_util.h"
#include "common/logging/backend.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/vita_diag.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "core/core.h"
#include "core/frontend/applets/default_applets.h"
#include "core/frontend/image_interface.h"
#include "core/hle/service/service.h"

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

/// Version del port, visible en el menu y en la pantalla de arranque.
constexpr char kVersion[] = "version 0.0.3.9";

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
    Settings::values.graphics_api = Settings::GraphicsAPI::Software;

    // Sin JIT: dynarmic solo tiene backends de x86-64 y ARM64, y la Vita es
    // ARMv7. Se usa el interprete dyncom.
    Settings::values.use_cpu_jit = false;
    Settings::values.use_shader_jit = false;
    Settings::values.use_hw_shader = false;
    Settings::values.use_disk_shader_cache = false;
    Settings::values.async_shader_compilation = false;

    // Modo 3DS original: el New 3DS pide otros 128 MB de FCRAM que no caben en
    // el presupuesto de memoria de una aplicacion de Vita.
    Settings::values.is_new_3ds = false;

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
    Settings::values.output_type = AudioCore::SinkType::Null;
    // El estirado de audio intenta casar el sonido con la velocidad real de
    // emulacion. Yendo muy por debajo del 100% eso destroza el sonido y ademas
    // cuesta CPU, asi que se deja crudo.
    Settings::values.enable_audio_stretching = false;

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

/// Dibuja una pantalla de texto simple centrada (menu y mensajes).
void DrawMenu(vita2d_pgf* font, const std::vector<RomEntry>& roms, int selected, int scroll) {
    constexpr int kVisibleRows = 14;
    constexpr int kRowHeight = 28;
    constexpr int kListTop = 120;

    vita2d_start_drawing();
    vita2d_clear_screen();

    vita2d_pgf_draw_text(font, 40, 50, kColorAccent, 1.4f, "Azahar  -  PS Vita");
    vita2d_pgf_draw_text(font, 40, 524, kColorDim, 0.85f, kVersion);
    vita2d_pgf_draw_text(font, 40, 80, kColorDim, 0.9f,
                         "Emulador de Nintendo 3DS  |  interprete + renderizado por software");

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
    const SceUID fd = sceIoOpen(kCrashLog, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
    if (fd < 0) {
        return;
    }
    sceIoWrite(fd, title, std::strlen(title));
    sceIoWrite(fd, ": ", 2);
    if (detail != nullptr) {
        sceIoWrite(fd, detail, std::strlen(detail));
    }
    sceIoWrite(fd, "\n", 1);
    sceIoClose(fd);
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

            if (window->FramesPresented() == 0 && (loops % 64) == 0) {
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

            const Core::System::ResultStatus result = system.RunLoop();
            if (result != Core::System::ResultStatus::Success &&
                result != Core::System::ResultStatus::ShutdownRequested) {
                LOG_ERROR(Frontend, "RunLoop devolvio {}: {}", static_cast<int>(result),
                          system.GetStatusDetails());
                WriteCrashLog("RunLoop error", system.GetStatusDetails().c_str());
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
        DrawFatalError(font, "Sin memoria durante la emulacion",
                       "El juego cargo pero se agoto la memoria al ejecutarlo.");
    } catch (const std::exception& e) {
        std::string detail = std::string("[") + typeid(e).name() + "] " + e.what();
        WriteCrashLog("emulacion", detail.c_str());
        LOG_ERROR(Frontend, "Excepcion durante la emulacion: {}", e.what());
        DrawFatalError(font, "Error durante la emulacion", detail);
    } catch (...) {
        WriteCrashLog("emulacion", "excepcion de tipo no estandar");
        DrawFatalError(font, "Error durante la emulacion",
                       "Excepcion de un tipo no estandar.");
    }

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
    LOG_INFO(Frontend, "Azahar para PS Vita iniciando");

    DrawStep(font, "3/7 aplicando ajustes");
    ConfigureSettings();

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

    while (running) {
        SceCtrlData pad{};
        sceCtrlPeekBufferPositive(0, &pad, 1);
        const unsigned int pressed = pad.buttons & ~previous_buttons;
        previous_buttons = pad.buttons;

        if (pressed & SCE_CTRL_DOWN) {
            selected = std::min(selected + 1, static_cast<int>(roms.size()) - 1);
        }
        if (pressed & SCE_CTRL_UP) {
            selected = std::max(selected - 1, 0);
        }
        if (pressed & SCE_CTRL_CIRCLE) {
            running = false;
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

        DrawMenu(font, roms, selected, scroll);
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
