// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include "common/vita_diag.h"

#ifdef __PSVITA__

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <fmt/format.h>
#include "common/vita_threads.h"
#include <exception>
#include <mutex>
#include <pthread.h>
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/cpu.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>

namespace Common {

namespace {
constexpr char kCrashLog[] = "ux0:/data/azahar/crash.txt";
/// El crash.txt del juego en curso (0.1.9.7), o vacio. Ver SetVitaGameLog.
char g_game_log[160] = {};

/// "fichero.cpp:1234" del ultimo fallo fatal. Estatico a proposito: no reserva
/// memoria, que cuando esto se llena puede que no quede.
char g_last_fatal[256] = {};

/// Escribe un entero sin signo en decimal. No se usa snprintf a proposito: la
/// printf reducida de newlib reserva memoria, y esto tiene que funcionar
/// tambien cuando el fallo es precisamente que no queda memoria.
std::size_t AppendUInt(char* out, unsigned int value) {
    char tmp[12];
    std::size_t n = 0;
    do {
        tmp[n++] = static_cast<char>('0' + (value % 10));
        value /= 10;
    } while (value != 0);
    for (std::size_t i = 0; i < n; i++) {
        out[i] = tmp[n - 1 - i];
    }
    return n;
}

/// Se queda con el nombre del fichero: la ruta completa se come el ancho de la
/// pantalla y no aporta nada.
const char* BaseName(const char* path) {
    const char* base = path;
    for (const char* p = path; *p != 0; p++) {
        if (*p == '/' || *p == '\\') {
            base = p + 1;
        }
    }
    return base;
}

/// Las carpetas de crash.txt, creadas una vez (eran dos sceIoMkdir por nota).
std::atomic<bool> g_log_dirs_ready{false};

/**
 * LAS NOTAS, DESDE UN HILO APARTE (0.3.1.5). crash.txt de 0.3.1.4: "notas: 64
 * en 2,7-3,9 s, la peor 100-450 ms". Cada nota abria, escribia y cerraba dos
 * ficheros en la tarjeta desde el hilo que la pedia: el de la GPU escribe
 * trece cada 10 s (el overlay), con el juego esperandole, y el de compilacion
 * tres por shader. Ahora, con una partida en marcha (SetVitaGameLog), la linea
 * se copia a g_ring y NoteWriterMain las escribe todas juntas, con una sola
 * apertura por fichero, cada kNoteFlushUs. Antes de la primera partida, o si
 * el anillo esta lleno o el cerrojo no llega enseguida, se escribe aqui mismo
 * como antes. Una caida pierde, como mucho, lo de ese ultimo intervalo.
 */
constexpr std::size_t kRingBytes = 64 * 1024;
constexpr unsigned int kNoteFlushUs = 50 * 1000;
char g_ring[kRingBytes];
std::size_t g_ring_used = 0;
std::atomic_flag g_ring_lock = ATOMIC_FLAG_INIT;
std::atomic<bool> g_writer_running{false};
/// Quien escribe en los ficheros: el hilo y SetVitaGameLog, que vacia el
/// anillo en el crash.txt de la partida que se cierra antes de cambiarlo.
std::mutex g_write_mutex;
char g_batch[kRingBytes];

/// Cada kNoteReport lineas escritas por el hilo, una con lo que ha costado.
constexpr unsigned int kNoteReport = 256;
unsigned int g_report_lines = 0;
unsigned int g_report_writes = 0;
unsigned long long g_report_us = 0;
unsigned long long g_report_max_us = 0;

void WriteLogLine(const char* line, std::size_t n);

bool LockRing(unsigned int max_spins) {
    for (unsigned int spins = 0; g_ring_lock.test_and_set(std::memory_order_acquire); spins++) {
        if (spins >= max_spins) {
            return false;
        }
    }
    return true;
}

bool TryEnqueue(const char* line, std::size_t n) {
    if (!g_writer_running.load(std::memory_order_acquire) || !LockRing(20000)) {
        return false;
    }
    const bool fits = g_ring_used + n <= kRingBytes;
    if (fits) {
        std::memcpy(g_ring + g_ring_used, line, n);
        g_ring_used += n;
    }
    g_ring_lock.clear(std::memory_order_release);
    return fits;
}

/// Escribe lo encolado. Con g_write_mutex cogido.
void DrainRing() {
    // Durmiendo entre intentos: quien tiene el cerrojo puede ser un hilo de
    // menos prioridad en este mismo nucleo, y girando no le dejaria acabar.
    while (!LockRing(1000)) {
        sceKernelDelayThread(100);
    }
    const std::size_t n = g_ring_used;
    std::memcpy(g_batch, g_ring, n);
    g_ring_used = 0;
    g_ring_lock.clear(std::memory_order_release);
    if (n == 0) {
        return;
    }
    const unsigned long long begin = VitaMicros();
    WriteLogLine(g_batch, n);
    const unsigned long long spent = VitaMicros() - begin;
    g_report_writes++;
    g_report_us += spent;
    g_report_max_us = std::max(g_report_max_us, spent);
    g_report_lines += static_cast<unsigned int>(std::count(g_batch, g_batch + n, '\n'));
    if (g_report_lines < kNoteReport) {
        return;
    }
    // Sin snprintf, como el resto de este fichero.
    char report[128] = "notas: ";
    std::size_t r = std::strlen(report);
    const auto text = [&](const char* piece) {
        const std::size_t len = std::strlen(piece);
        std::memcpy(report + r, piece, len);
        r += len;
    };
    r += AppendUInt(report + r, g_report_lines);
    text(" lineas en ");
    r += AppendUInt(report + r, g_report_writes);
    text(" escrituras, ");
    r += AppendUInt(report + r, static_cast<unsigned int>(g_report_us / 1000));
    text(" ms de tarjeta, la peor ");
    r += AppendUInt(report + r, static_cast<unsigned int>(g_report_max_us / 1000));
    text(" ms\n");
    WriteLogLine(report, r);
    g_report_lines = 0;
    g_report_writes = 0;
    g_report_us = 0;
    g_report_max_us = 0;
}

void* NoteWriterMain(void*) {
    // Por encima del compilador de shaders, que comparte nucleo y pasa
    // minutos sin soltarlo, y por debajo de todo lo demas.
    sceKernelChangeThreadPriority(sceKernelGetThreadId(), kVitaPriorityBackground - 1);
    while (true) {
        sceKernelDelayThread(kNoteFlushUs);
        const std::lock_guard lock{g_write_mutex};
        DrainRing();
    }
    return nullptr;
}

void StartNoteWriter() {
    static bool started = false;
    if (started) {
        return;
    }
    started = true;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 64 * 1024);
    pthread_t thread;
    if (pthread_create(&thread, &attr, &NoteWriterMain, nullptr) == 0) {
        pthread_detach(thread);
        g_writer_running.store(true, std::memory_order_release);
    }
    pthread_attr_destroy(&attr);
}
} // Anonymous namespace

void VitaNote(const char* title, const char* detail) {
    // La linea entera se monta en un buffer y se escribe de UNA sola vez.
    //
    // Antes esto hacia cuatro sceIoWrite seguidos (titulo, ": ", detalle, salto
    // de linea). Con un solo hilo daba igual, pero desde que los hilos del
    // rasterizador anotan su afinidad al arrancar hay varios escribiendo a la
    // vez, y cuatro escrituras sueltas por hilo se entrelazan: justo el
    // diagnostico que hay que leer saldria hecho un revoltijo. Con una sola
    // escritura por linea eso deja de pasar en la practica.
    //
    // Sin mutex a proposito: esto se llama tambien desde manejadores de fallo
    // fatal, y un candado ahi puede acabar en bloqueo si el hilo que muere ya
    // lo tenia cogido.
    char line[768];
    std::size_t n = 0;

    const auto append = [&](const char* text, std::size_t max_copy) {
        if (text == nullptr) {
            return;
        }
        const std::size_t len = std::strlen(text);
        const std::size_t room = sizeof(line) - n - 2; // deja sitio al "\n"
        const std::size_t copy = std::min(std::min(len, max_copy), room);
        std::memcpy(line + n, text, copy);
        n += copy;
    };

    if (title != nullptr) {
        append(title, 200);
        line[n++] = ':';
        line[n++] = ' ';
    }
    append(detail, sizeof(line));
    line[n++] = '\n';
    if (!TryEnqueue(line, n)) {
        WriteLogLine(line, n);
    }
}

namespace {
/**
 * LOS FICHEROS SE QUEDAN ABIERTOS (0.3.1.8). Cada tanda abria y cerraba
 * crash.txt y el del juego: "notas: 256 lineas en 80 escrituras, 28699 ms de
 * tarjeta" en Pokemon Sol (~360 ms por tanda), tarjeta que no tenian la RomFS
 * ni la cache de shaders. Abiertos solo con una partida en marcha (el hilo
 * escritor ya funcionando): al arrancar main renombra crash.txt, y antes de
 * renombrar el del juego SetVitaGameLog cierra el suyo.
 */
std::atomic<int> g_crash_fd{-1};
std::atomic<int> g_game_fd{-1};

void WriteTo(std::atomic<int>& cached, const char* path, const char* line, std::size_t n) {
    if (!g_writer_running.load(std::memory_order_acquire)) {
        const SceUID fd = sceIoOpen(path, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
        if (fd >= 0) {
            sceIoWrite(fd, line, n);
            sceIoClose(fd);
        }
        return;
    }
    int fd = cached.load(std::memory_order_acquire);
    if (fd < 0) {
        fd = sceIoOpen(path, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
        if (fd < 0) {
            return;
        }
        int expected = -1;
        if (!cached.compare_exchange_strong(expected, fd, std::memory_order_acq_rel)) {
            sceIoClose(fd);
            fd = expected;
        }
    }
    sceIoWrite(fd, line, n);
}

void WriteLogLine(const char* line, std::size_t n) {
    if (!g_log_dirs_ready.exchange(true, std::memory_order_relaxed)) {
        sceIoMkdir("ux0:/data", 0777);
        sceIoMkdir("ux0:/data/azahar", 0777);
    }
    WriteTo(g_crash_fd, kCrashLog, line, n);
    if (g_game_log[0] != '\0') {
        WriteTo(g_game_fd, g_game_log, line, n);
    }
}
} // Anonymous namespace

void SetVitaGameLog(const char* path) {
    {
        // Lo encolado es de la partida que se cierra: a su fichero.
        const std::lock_guard lock{g_write_mutex};
        DrainRing();
        if (const int fd = g_game_fd.exchange(-1, std::memory_order_acq_rel); fd >= 0) {
            sceIoClose(fd);
        }
        if (path == nullptr) {
            g_game_log[0] = '\0';
            return;
        }
        std::strncpy(g_game_log, path, sizeof(g_game_log) - 1);
        g_game_log[sizeof(g_game_log) - 1] = '\0';
    }
    StartNoteWriter();
}

const char* VitaGameLog() {
    return g_game_log;
}

const char* LastFatalMessage() {
    return g_last_fatal;
}

void VitaAssertFail(const char* file, int line) noexcept {
    std::size_t n = 0;

    if (file != nullptr) {
        const char* base = BaseName(file);
        const std::size_t len = std::strlen(base);
        if (len < sizeof(g_last_fatal) - 16) {
            std::memcpy(g_last_fatal, base, len);
            n = len;
        }
    }
    g_last_fatal[n++] = ':';
    n += AppendUInt(g_last_fatal + n, static_cast<unsigned int>(line));
    g_last_fatal[n] = 0;

    VitaNote("ASSERT", g_last_fatal);
    {
        const std::lock_guard lock{g_write_mutex};
        DrainRing();
    }

    // El texto del assert ya esta unas lineas mas arriba en el mismo fichero:
    // lo escribio LOG_CRITICAL al pasar por FmtLogMessageImpl. Aqui solo queda
    // parar. std::terminate en vez de sceKernelExitProcess para que el
    // manejador del frontend llegue a pintar el aviso en pantalla.
    std::terminate();
}

namespace {
/**
 * LOS HILOS CON NOMBRE (0.3.2.0). Para medir cuanto CPU usa de verdad cada uno
 * (runClocks de sceKernelGetThreadInfo) y cuanto de cada nucleo queda libre
 * (idleClock de sceKernelGetSystemInfo). Con eso se sabe si el hilo de
 * emulacion pierde su nucleo frente a otro, cosa que los contadores de
 * tiempo de pared no distinguen de "trabajo". Se apuntan al pasar por
 * VitaPinThreadToUserCore o VitaSetThreadPriority, que son los hilos que
 * importan; el hueco de uno que ya no existe se reutiliza.
 */
constexpr std::size_t kMaxNamedThreads = 24;
struct NamedThread {
    std::atomic<int> tid{-1};
    char name[28]{};
    unsigned long long last_run = 0;
};
std::array<NamedThread, kMaxNamedThreads> g_named_threads;

bool ThreadAlive(int tid) {
    SceKernelThreadInfo info{};
    info.size = sizeof(info);
    return tid >= 0 && sceKernelGetThreadInfo(tid, &info) >= 0;
}

void RegisterThread(const char* role) {
    const int self = sceKernelGetThreadId();
    for (auto& slot : g_named_threads) {
        if (slot.tid.load(std::memory_order_acquire) == self) {
            return;
        }
    }
    for (auto& slot : g_named_threads) {
        int seen = slot.tid.load(std::memory_order_acquire);
        if (seen == -2 || (seen >= 0 && ThreadAlive(seen))) {
            continue;
        }
        // -2: reservado mientras se copia el nombre.
        if (!slot.tid.compare_exchange_strong(seen, -2, std::memory_order_acq_rel)) {
            continue;
        }
        std::size_t n = 0;
        for (; role != nullptr && role[n] != 0 && n < sizeof(slot.name) - 1; n++) {
            slot.name[n] = role[n] == ' ' ? '_' : role[n];
        }
        slot.name[n] = 0;
        slot.last_run = 0;
        slot.tid.store(self, std::memory_order_release);
        return;
    }
}
} // Anonymous namespace

std::string VitaThreadSummary() {
    static unsigned long long last_wall = 0;
    static std::array<unsigned long long, 3> last_idle{};
    static std::array<unsigned int, 3> last_switches{};
    const unsigned long long now = VitaMicros();
    const unsigned long long wall = last_wall != 0 ? now - last_wall : 0;
    last_wall = now;
    SceKernelSystemInfo system{};
    system.size = sizeof(system);
    const bool have_system = sceKernelGetSystemInfo(&system) >= 0;
    std::string text = "nucleos ocupados";
    for (int core = 0; core < 3; core++) {
        const unsigned long long idle = have_system ? system.cpuInfo[core].idleClock : 0;
        const unsigned int switches = have_system ? system.cpuInfo[core].threadSwitchCount : 0;
        if (wall > 0 && have_system) {
            const double free_percent =
                std::min(100.0, static_cast<double>(idle - last_idle[core]) * 100.0 /
                                    static_cast<double>(wall));
            text += fmt::format(" {} {:.0f}% ({} cambios/s)", core, 100.0 - free_percent,
                                static_cast<unsigned long long>(switches - last_switches[core]) *
                                    1000000ull / wall);
        }
        last_idle[core] = idle;
        last_switches[core] = switches;
    }
    if (wall == 0) {
        text += " -";
    }
    text += " | cpu de cada hilo:";
    for (auto& slot : g_named_threads) {
        const int tid = slot.tid.load(std::memory_order_acquire);
        if (tid < 0) {
            continue;
        }
        SceKernelThreadInfo info{};
        info.size = sizeof(info);
        if (sceKernelGetThreadInfo(tid, &info) < 0) {
            continue;
        }
        const unsigned long long run = info.runClocks;
        if (wall > 0 && slot.last_run != 0 && run >= slot.last_run) {
            text += fmt::format(" {} {:.0f}% n{} p{}", slot.name,
                                static_cast<double>(run - slot.last_run) * 100.0 /
                                    static_cast<double>(wall),
                                info.lastExecutedCpuId, info.currentPriority);
        }
        slot.last_run = run;
    }
    return text;
}

void VitaSetThreadPriority(int priority, const char* role) {
    RegisterThread(role);
    const int before = sceKernelGetThreadCurrentPriority();
    const int rc = sceKernelChangeThreadPriority(sceKernelGetThreadId(), priority);
    char buffer[128];
    std::size_t n = 0;
    const auto append = [&](const char* text) {
        for (std::size_t i = 0; text[i] != 0 && n < 96; i++) {
            buffer[n++] = text[i];
        }
    };
    append(role != nullptr ? role : "?");
    append(" ");
    n += AppendUInt(buffer + n, static_cast<unsigned int>(before));
    append(" -> ");
    n += AppendUInt(buffer + n, static_cast<unsigned int>(priority));
    append(rc >= 0 ? ", ok" : ", ERROR");
    buffer[n] = 0;
    VitaNote("prioridad", buffer);
}

void VitaPinThreadToUserCore(unsigned int index, const char* role) {
    // Tres nucleos de usuario: 0x10000, 0x20000, 0x40000 (psp2/kernel/cpu.h).
    // El cuarto, SCE_KERNEL_CPU_MASK_SYSTEM, lo reserva el sistema y no se toca.
    RegisterThread(role);
    static const int kUserCoreMasks[3] = {
        SCE_KERNEL_CPU_MASK_USER_0,
        SCE_KERNEL_CPU_MASK_USER_1,
        SCE_KERNEL_CPU_MASK_USER_2,
    };
    const unsigned int core = index % 3;
    const int mask = kUserCoreMasks[core];

    const SceUID thread_id = sceKernelGetThreadId();
    const int rc = sceKernelChangeThreadCpuAffinityMask(thread_id, mask);

    // Comprobacion de que ha servido de algo: sceKernelGetCpuId() devuelve el
    // nucleo en el que corre AHORA el hilo, pero el cambio de afinidad no mueve
    // el hilo al instante -- solo condiciona la siguiente vez que el
    // planificador lo coloque. Sin esta pausa se leeria el nucleo viejo y la
    // anotacion enganaria. Es una sola vez al arrancar cada hilo.
    sceKernelDelayThread(2000);
    const int actual_core = sceKernelGetCpuId();

    char buffer[160];
    std::size_t n = 0;
    const char* prefix = "hilo ";
    for (std::size_t i = 0; prefix[i] != 0; i++) {
        buffer[n++] = prefix[i];
    }
    if (role != nullptr) {
        const std::size_t len = std::strlen(role);
        if (len < 64) {
            std::memcpy(buffer + n, role, len);
            n += len;
        }
    }
    buffer[n++] = ' ';
    n += AppendUInt(buffer + n, index);

    const char* wanted = " -> nucleo pedido ";
    for (std::size_t i = 0; wanted[i] != 0; i++) {
        buffer[n++] = wanted[i];
    }
    n += AppendUInt(buffer + n, core);

    const char* got = ", real ";
    for (std::size_t i = 0; got[i] != 0; i++) {
        buffer[n++] = got[i];
    }
    n += AppendUInt(buffer + n, static_cast<unsigned int>(actual_core));

    // Solo un rc NEGATIVO es un error. Si sale bien, la llamada devuelve la
    // mascara ANTERIOR: 0 en un hilo recien creado, pero 0x70000 ("cualquier
    // nucleo de usuario") en el hilo principal. Hasta 0.1.0.43 esto comparaba
    // con 0 y anotaba ese 458752 (0x70000) como "ERROR" en el hilo de
    // emulacion, cuando la afinidad SI se habia aplicado ("real 0").
    const char* result = rc >= 0 ? ", ok" : ", ERROR rc ";
    for (std::size_t i = 0; result[i] != 0; i++) {
        buffer[n++] = result[i];
    }
    if (rc < 0) {
        n += AppendUInt(buffer + n, static_cast<unsigned int>(rc));
    }
    buffer[n] = 0;

    VitaNote("afinidad", buffer);
}

unsigned long long VitaMicros() {
    return static_cast<unsigned long long>(sceKernelGetProcessTimeWide());
}

} // namespace Common

#endif // __PSVITA__

namespace Common {

int g_arm_clock_mhz = 0;

namespace {
/// Ultimo FPSCR visto con bits acumulativos de excepcion. Ver VitaSawDenormals.
std::atomic<unsigned int> g_fpscr_seen{0};

inline unsigned int ReadFpscr() {
    unsigned int value = 0;
#if defined(__ARM_ARCH) && defined(__ARM_FP)
    __asm__ volatile("vmrs %0, fpscr" : "=r"(value));
#endif
    return value;
}

inline void WriteFpscr(unsigned int value) {
#if defined(__ARM_ARCH) && defined(__ARM_FP)
    __asm__ volatile("vmsr fpscr, %0" : : "r"(value));
#endif
}
} // Anonymous namespace

void VitaEnableFastFloatMode() {
    // Bit 24 = FZ (flush-to-zero), bit 25 = DN (default NaN).
    constexpr unsigned int kFlushToZero = 1u << 24;
    constexpr unsigned int kDefaultNaN = 1u << 25;

    const unsigned int before = ReadFpscr();
    // Se guarda lo que habia ANTES de tocar nada: los bits acumulativos de
    // excepcion cuentan lo ocurrido hasta este momento, y si se leyera despues
    // de activar FZ ya no habria denormales que ver.
    g_fpscr_seen.fetch_or(before, std::memory_order_relaxed);
    WriteFpscr(before | kFlushToZero | kDefaultNaN);
}

bool VitaSawDenormals() {
    // Bit 7 = IDC (Input Denormal Cumulative). Acumulativo: una vez puesto, se
    // queda hasta que alguien lo borre.
    constexpr unsigned int kInputDenormal = 1u << 7;
    const unsigned int current = ReadFpscr();
    const unsigned int all = g_fpscr_seen.fetch_or(current, std::memory_order_relaxed) | current;
    return (all & kInputDenormal) != 0;
}

} // namespace Common
