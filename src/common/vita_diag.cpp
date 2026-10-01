// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include "common/vita_diag.h"

#ifdef __PSVITA__

#include <algorithm>
#include <cstring>
#include <exception>
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/cpu.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>

namespace Common {

namespace {
constexpr char kCrashLog[] = "ux0:/data/azahar/crash.txt";

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

    sceIoMkdir("ux0:/data", 0777);
    sceIoMkdir("ux0:/data/azahar", 0777);
    const SceUID fd = sceIoOpen(kCrashLog, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
    if (fd < 0) {
        return;
    }
    sceIoWrite(fd, line, n);
    sceIoClose(fd);
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

    // El texto del assert ya esta unas lineas mas arriba en el mismo fichero:
    // lo escribio LOG_CRITICAL al pasar por FmtLogMessageImpl. Aqui solo queda
    // parar. std::terminate en vez de sceKernelExitProcess para que el
    // manejador del frontend llegue a pintar el aviso en pantalla.
    std::terminate();
}

void VitaPinThreadToUserCore(unsigned int index, const char* role) {
    // Tres nucleos de usuario: 0x10000, 0x20000, 0x40000 (psp2/kernel/cpu.h).
    // El cuarto, SCE_KERNEL_CPU_MASK_SYSTEM, lo reserva el sistema y no se toca.
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
