// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <psp2/gxm.h>
#include "common/common_types.h"
#include "video_core/renderer_gxm/gxm_memory.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct RegsInternal;
}

namespace Gxm {

/**
 * Texturas de la PICA listas para la GPU.
 *
 * La PICA guarda sus texturas con su propio entrelazado (mosaicos de 8x8) y en
 * una docena de formatos, algunos comprimidos (ETC1). GXM tiene sus formatos,
 * pero con otro entrelazado, asi que la via segura es decodificar una vez al
 * formato que entiende todo el mundo (RGBA8 lineal) y quedarse con la copia:
 * el decodificado lo hace el mismo codigo que usa el rasterizador de software
 * (Pica::Texture), asi que la imagen no puede divergir.
 *
 * El coste de decodificar se paga cuando la textura cambia, no por texel
 * muestreado, y una entrada por unidad de textura es suficiente en la
 * practica: los juegos suelen dejar la textura puesta mientras dibujan el lote.
 * Cuando el juego escribe en la memoria de una textura (FlushRegion /
 * InvalidateRegion), la entrada que la solape se tira y se redecodifica.
 *
 * Lo que no se soporta todavia devuelve nullptr y el lote entero cae al
 * rasterizador de software: bordes con color, mipmaps, texturas que no sean
 * 2D y unidades deshabilitadas.
 */
class TextureCache {
public:
    ~TextureCache();

    TextureCache() = default;
    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;

    /// Textura de la unidad (0-2) para el estado actual, o nullptr.
    [[nodiscard]] const SceGxmTexture* Get(u32 unit, const Pica::RegsInternal& regs,
                                          Memory::MemorySystem& memory);

    /// Tira las texturas que solapen el rango.
    void InvalidateRange(PAddr addr, u32 size);

    /// Tira todas.
    void Clear();

private:
    struct Entry {
        bool valid = false;
        u64 key = 0;
        PAddr address = 0;
        u32 span = 0;
        SceGxmTexture texture{};
        Allocation buffer;
    };

    std::array<Entry, 3> entries;
};

} // namespace Gxm
