// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
//
// Sonda: comprueba si esta consola nos deja generar y ejecutar codigo en
// tiempo de ejecucion.
//
// Un recompilador (JIT) necesita memoria que se pueda ESCRIBIR y luego
// EJECUTAR. En PS Vita eso no se consigue con malloc: hace falta la familia de
// funciones "VM domain" de sceKernelSysmem, y el kernel solo se las concede a
// las aplicaciones con permisos extendidos -- las firmadas como "unsafe", y
// ademas con "Enable unsafe homebrew" activado en los ajustes de HENkaku.
//
// Azahar se firma como SAFE ahora mismo (ver AZAHAR_UNSAFE en el CMakeLists del
// frontend), asi que lo mas probable es que esto falle. Pero "lo mas probable"
// no es una respuesta: escribir un recompilador son meses de trabajo, y no
// tiene ningun sentido empezarlos sin saber con certeza si el resultado podra
// llegar a ejecutarse en esta consola concreta.
//
// La sonda hace el recorrido completo -- reservar, abrir el dominio, escribir
// una funcion ARM de dos instrucciones, sincronizar caches y LLAMARLA -- y
// anota cada paso en crash.txt. Si la ultima linea dice que devolvio 42, el
// camino del JIT esta abierto. Si falla, dice exactamente en que paso y con que
// codigo de error.

#include "citra_vita/vita_jit_probe.h"

#ifdef __PSVITA__

#include <cstring>
#include <psp2/kernel/sysmem.h>
#include "common/common_types.h"
#include "common/vita_diag.h"

namespace VitaFrontend {

namespace {

/// Escribe "paso: detalle (rc 0x...)" en crash.txt. Formateo a mano por la misma
/// razon que en el resto del port: la printf reducida de esta libc no es de
/// fiar, y esto tiene que funcionar pase lo que pase.
void NoteStep(const char* step, int rc) {
    char buffer[128];
    std::size_t n = 0;

    const std::size_t len = std::strlen(step);
    std::memcpy(buffer, step, len);
    n += len;

    const char* tag = " rc 0x";
    for (std::size_t i = 0; tag[i] != 0; i++) {
        buffer[n++] = tag[i];
    }
    for (int shift = 28; shift >= 0; shift -= 4) {
        const unsigned int digit = (static_cast<unsigned int>(rc) >> shift) & 0xF;
        buffer[n++] = static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
    }
    buffer[n] = 0;

    Common::VitaNote("jit", buffer);
}

/// Funcion ARM de prueba: "mov r0, #42" seguido de "bx lr".
///
/// Las dos codificaciones estan verificadas: 0xe12fff1e (BX) aparece en los
/// rastros del propio interprete traduciendo codigo real del juego, y
/// 0xe3a0002a es MOV r0,#42 en ARM (cond=AL, opcode MOV, Rd=r0, inmediato 42).
///
/// Se emite en ARM y no en Thumb a proposito: la direccion queda con el bit 0 a
/// cero, y al llamarla desde el codigo Thumb del emulador el BLX cambia de modo
/// solo. Es justo lo que tendria que hacer un JIT de verdad.
constexpr u32 kTestCode[] = {
    0xe3a0002a, // mov r0, #42
    0xe12fff1e, // bx lr
};

} // Anonymous namespace

void ProbeJitSupport() {
    // 1 MB: suficiente para la prueba y muy por debajo del maximo de 16 MB que
    // admite un bloque VM.
    constexpr SceSize kProbeSize = 1024 * 1024;

    const SceUID block = sceKernelAllocMemBlockForVM("azahar_jit_probe", kProbeSize);
    if (block < 0) {
        NoteStep("FALLO al reservar memoria ejecutable (hace falta unsafe)", block);
        Common::VitaNote("jit", "resultado: NO se puede hacer JIT en esta consola");
        return;
    }
    NoteStep("memoria ejecutable reservada", block);

    void* base = nullptr;
    const int base_rc = sceKernelGetMemBlockBase(block, &base);
    if (base_rc < 0 || base == nullptr) {
        NoteStep("FALLO al obtener la direccion del bloque", base_rc);
        return;
    }

    // Abre el dominio: a partir de aqui los bloques VM son ejecutables.
    const int open_rc = sceKernelOpenVMDomain();
    if (open_rc < 0) {
        NoteStep("FALLO al abrir el dominio VM", open_rc);
        Common::VitaNote("jit", "resultado: NO se puede hacer JIT en esta consola");
        return;
    }

    std::memcpy(base, kTestCode, sizeof(kTestCode));

    // Sin esto el procesador podria ejecutar lo que hubiera antes en esa
    // direccion: los datos recien escritos estan en la cache de datos, y la
    // cache de instrucciones no se entera sola.
    const int sync_rc = sceKernelSyncVMDomain(block, base, sizeof(kTestCode));
    if (sync_rc < 0) {
        NoteStep("FALLO al sincronizar caches", sync_rc);
    }

    // El momento de la verdad.
    using TestFn = int (*)();
    TestFn fn = nullptr;
    std::memcpy(&fn, &base, sizeof(fn));
    const int result = fn();

    sceKernelCloseVMDomain();

    if (result == 42) {
        Common::VitaNote("jit", "resultado: SI se puede hacer JIT (codigo generado ejecutado OK)");
    } else {
        NoteStep("el codigo generado se ejecuto pero devolvio un valor inesperado", result);
    }
}

} // namespace VitaFrontend

#endif // __PSVITA__
