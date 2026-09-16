// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <psp2/gxm.h>
#include <psp2/kernel/sysmem.h>
#include "common/common_types.h"

namespace Gxm {

/**
 * De donde sale la memoria que va a leer el chip grafico.
 *
 * La Vita tiene DOS sitios donde poner cosas para la GPU, y la diferencia no es
 * un detalle:
 *
 *   CDRAM   128 MB soldados al lado del chip grafico. NO salen del presupuesto
 *           de RAM de usuario de la aplicacion -- son memoria aparte, y hoy este
 *           emulador no usa ni un byte de ella. La GPU lee de aqui a mucha mas
 *           velocidad que de la RAM principal. A cambio, la CPU escribe MAS
 *           LENTO: va por el bus del chip grafico, sin cache.
 *
 *   Host    la RAM normal (LPDDR2) reservada SIN CACHEAR. La CPU escribe a
 *           velocidad casi normal y no hay que volcar ninguna cache antes de
 *           que mire la GPU. A cambio sale del presupuesto de la aplicacion,
 *           que es justo lo que va escaso.
 *
 * O sea que no hay una que sea mejor: depende de quien escriba mas, la CPU o la
 * GPU. Un framebuffer que la CPU rellena entero cada fotograma y la GPU lee una
 * vez tira hacia Host; una textura que se sube una vez y se lee miles de veces
 * tira hacia CDRAM. Por eso esto es un parametro y no una decision cableada, y
 * por eso el frontend deja alternarlo en caliente: la respuesta sale del
 * cronometro del overlay, no de razonarlo.
 */
enum class Pool {
    Cdram,
    Host,
};

/// Cuanto hay reservado en cada pool, en bytes. Para el overlay y crash.txt: sin
/// esto, "no arranca" no distingue entre presupuesto agotado y cualquier otra
/// cosa. Ver la seccion de presupuesto del CMakeLists raiz.
namespace Budget {
inline std::atomic<u32> cdram_bytes{0};
inline std::atomic<u32> host_bytes{0};
} // namespace Budget

/**
 * Un bloque de memoria reservado y mapeado para que la GPU pueda leerlo.
 *
 * Dos pasos, y los dos hacen falta: sceKernelAllocMemBlock da la memoria y
 * sceGxmMapMemory la mete en la tabla de paginas del chip grafico. Sin lo
 * segundo la GPU lee basura -- no falla, no avisa: dibuja mal.
 *
 * Es move-only a proposito. Un doble sceGxmUnmapMemory sobre la misma direccion
 * deja la tabla de paginas de la GPU inconsistente y el fallo aparece mucho
 * despues, en un dibujado que no tiene nada que ver.
 */
class Allocation {
public:
    Allocation() = default;
    ~Allocation();

    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;

    /// Libera el bloque. Seguro sobre un objeto vacio.
    void Reset();

    [[nodiscard]] bool Valid() const noexcept {
        return data != nullptr;
    }
    [[nodiscard]] void* Data() const noexcept {
        return data;
    }
    [[nodiscard]] u32 Size() const noexcept {
        return size;
    }
    [[nodiscard]] Pool Where() const noexcept {
        return pool;
    }

private:
    friend Allocation Allocate(Pool, u32, SceGxmMemoryAttribFlags);

    SceUID uid = -1;
    void* data = nullptr;
    u32 size = 0;
    Pool pool = Pool::Host;
};

/**
 * Reserva y mapea un bloque.
 *
 * El tamano se redondea al grano que exige cada tipo de bloque: 256 KB en CDRAM
 * y 4 KB en la RAM sin cachear. Pedir un tamano que no sea multiplo de su grano
 * NO se redondea solo: sceKernelAllocMemBlock devuelve error y no dice cual.
 *
 * Devuelve un objeto vacio si no hay sitio. Quien llame TIENE que comprobarlo:
 * quedarse sin memoria de GPU es el fallo mas probable de todo esto, y aqui una
 * dereferencia a nullptr no da excepcion, mata el proceso.
 */
[[nodiscard]] Allocation Allocate(Pool pool, u32 size,
                                  SceGxmMemoryAttribFlags attr = SCE_GXM_MEMORY_ATTRIB_READ);

/// Nombre corto del pool, para pintarlo en el overlay.
[[nodiscard]] const char* PoolName(Pool pool);

} // namespace Gxm
