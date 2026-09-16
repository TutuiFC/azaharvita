// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/common_types.h"

namespace VideoCore {

/**
 * Los 64 valores de Morton de un mosaico de 8x8, en una tabla.
 *
 * MortonInterleave hacia dos cargas de tablas de 8 entradas mas una suma (y un
 * AND por coordenada). Esto se llama varias veces por pixel -- muestreo de
 * textura y cada lectura/escritura del framebuffer -- asi que cambiarlo por una
 * sola carga y una mascara sale a cuenta. El valor es identico.
 */
inline constexpr std::array<u32, 64> kMortonValues = [] {
    constexpr u32 xlut[] = {0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15};
    constexpr u32 ylut[] = {0x00, 0x02, 0x08, 0x0a, 0x20, 0x22, 0x28, 0x2a};
    std::array<u32, 64> table{};
    for (u32 y = 0; y < 8; y++) {
        for (u32 x = 0; x < 8; x++) {
            table[y * 8 + x] = xlut[x] + ylut[y];
        }
    }
    return table;
}();

// 8x8 Z-Order coordinate from 2D coordinates
static constexpr u32 MortonInterleave(u32 x, u32 y) {
    return kMortonValues[(y & 7) * 8 + (x & 7)];
}

/**
 * Calculates the offset of the position of the pixel in Morton order
 */
static inline u32 GetMortonOffset(u32 x, u32 y, u32 bytes_per_pixel) {
    // Images are split into 8x8 tiles. Each tile is composed of four 4x4 subtiles each
    // of which is composed of four 2x2 subtiles each of which is composed of four texels.
    // Each structure is embedded into the next-bigger one in a diagonal pattern, e.g.
    // texels are laid out in a 2x2 subtile like this:
    // 2 3
    // 0 1
    //
    // The full 8x8 tile has the texels arranged like this:
    //
    // 42 43 46 47 58 59 62 63
    // 40 41 44 45 56 57 60 61
    // 34 35 38 39 50 51 54 55
    // 32 33 36 37 48 49 52 53
    // 10 11 14 15 26 27 30 31
    // 08 09 12 13 24 25 28 29
    // 02 03 06 07 18 19 22 23
    // 00 01 04 05 16 17 20 21
    //
    // This pattern is what's called Z-order curve, or Morton order.

    const unsigned int block_height = 8;
    const unsigned int coarse_x = x & ~7;

    u32 i = VideoCore::MortonInterleave(x, y);

    const unsigned int offset = coarse_x * block_height;

    return (i + offset) * bytes_per_pixel;
}

} // namespace VideoCore
