// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <utility>
#include "common/logging/log.h"
#include "common/vita_diag.h"
#include "video_core/renderer_gxm/gxm_memory.h"

namespace Gxm {

namespace {

/// Grano de reserva de cada tipo de bloque. No se redondea solo: pedir un
/// tamano que no sea multiplo de esto hace fallar sceKernelAllocMemBlock.
constexpr u32 kCdramGrain = 256 * 1024;
constexpr u32 kHostGrain = 4 * 1024;

constexpr u32 AlignUp(u32 value, u32 grain) {
    return (value + grain - 1) / grain * grain;
}

} // Anonymous namespace

Allocation::~Allocation() {
    Reset();
}

Allocation::Allocation(Allocation&& other) noexcept
    : uid{other.uid}, data{other.data}, size{other.size}, pool{other.pool} {
    other.uid = -1;
    other.data = nullptr;
    other.size = 0;
}

Allocation& Allocation::operator=(Allocation&& other) noexcept {
    if (this != &other) {
        Reset();
        uid = other.uid;
        data = other.data;
        size = other.size;
        pool = other.pool;
        other.uid = -1;
        other.data = nullptr;
        other.size = 0;
    }
    return *this;
}

void Allocation::Reset() {
    if (data == nullptr) {
        return;
    }

    // Desmapear ANTES de liberar. Al reves, la tabla de paginas de la GPU se
    // queda apuntando a memoria que ya es de otro, y el estropicio aparece en un
    // dibujado posterior sin ninguna relacion aparente.
    sceGxmUnmapMemory(data);
    sceKernelFreeMemBlock(uid);

    auto& counter = pool == Pool::Cdram ? Budget::cdram_bytes : Budget::host_bytes;
    counter.fetch_sub(size, std::memory_order_relaxed);

    uid = -1;
    data = nullptr;
    size = 0;
}

Allocation Allocate(Pool pool, u32 size, SceGxmMemoryAttribFlags attr) {
    Allocation result;
    if (size == 0) {
        return result;
    }

    const bool cdram = pool == Pool::Cdram;
    const u32 aligned = AlignUp(size, cdram ? kCdramGrain : kHostGrain);
    const SceKernelMemBlockType type = cdram ? SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW
                                             : SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE;

    const SceUID uid = sceKernelAllocMemBlock(cdram ? "azahar_gxm_cdram" : "azahar_gxm_host", type,
                                              static_cast<SceSize>(aligned), nullptr);
    if (uid < 0) {
        LOG_ERROR(Render, "GXM: sin memoria en el pool {} para {} bytes (codigo {:#x})",
                  PoolName(pool), aligned, static_cast<u32>(uid));
        return result;
    }

    void* base = nullptr;
    if (sceKernelGetMemBlockBase(uid, &base) < 0 || base == nullptr) {
        sceKernelFreeMemBlock(uid);
        LOG_ERROR(Render, "GXM: el bloque reservado no tiene direccion base");
        return result;
    }

    // El paso que se olvida. Sin esto la memoria existe y la CPU la escribe sin
    // problema, pero la GPU no la tiene en su tabla de paginas: no falla, no
    // avisa, simplemente dibuja basura.
    const int mapped = sceGxmMapMemory(base, static_cast<SceSize>(aligned), attr);
    if (mapped < 0) {
        sceKernelFreeMemBlock(uid);
        LOG_ERROR(Render, "GXM: sceGxmMapMemory fallo con {:#x}", static_cast<u32>(mapped));
        return result;
    }

    result.uid = uid;
    result.data = base;
    result.size = aligned;
    result.pool = pool;

    auto& counter = cdram ? Budget::cdram_bytes : Budget::host_bytes;
    counter.fetch_add(aligned, std::memory_order_relaxed);
    return result;
}

const char* PoolName(Pool pool) {
    return pool == Pool::Cdram ? "cdram" : "lpddr";
}

} // namespace Gxm
