// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cstring>
#include "common/bit_field.h"
#include "common/color.h"
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/texture/etc1.h"

namespace Pica::Texture {

namespace {

constexpr std::array<std::array<u8, 2>, 8> etc1_modifier_table = {{
    {2, 8},
    {5, 17},
    {9, 29},
    {13, 42},
    {18, 60},
    {24, 80},
    {33, 106},
    {47, 183},
}};

union ETC1Tile {
    u64 raw;

    // Each of these two is a collection of 16 bits (one per lookup value)
    BitField<0, 16, u64> table_subindexes;
    BitField<16, 16, u64> negation_flags;

    unsigned GetTableSubIndex(unsigned index) const {
        return (table_subindexes >> index) & 1;
    }

    bool GetNegationFlag(unsigned index) const {
        return ((negation_flags >> index) & 1) == 1;
    }

    BitField<32, 1, u64> flip;
    BitField<33, 1, u64> differential_mode;

    BitField<34, 3, u64> table_index_2;
    BitField<37, 3, u64> table_index_1;

    union {
        // delta value + base value
        BitField<40, 3, s64> db;
        BitField<43, 5, u64> b;

        BitField<48, 3, s64> dg;
        BitField<51, 5, u64> g;

        BitField<56, 3, s64> dr;
        BitField<59, 5, u64> r;
    } differential;

    union {
        BitField<40, 4, u64> b2;
        BitField<44, 4, u64> b1;

        BitField<48, 4, u64> g2;
        BitField<52, 4, u64> g1;

        BitField<56, 4, u64> r2;
        BitField<60, 4, u64> r1;
    } separate;

    const Common::Vec3<u8> GetRGB(unsigned int x, unsigned int y) const {
        int texel = 4 * x + y;

        if (flip)
            std::swap(x, y);

        // Lookup base value
        Common::Vec3<int> ret;
        if (differential_mode) {
            ret.r() = static_cast<int>(differential.r);
            ret.g() = static_cast<int>(differential.g);
            ret.b() = static_cast<int>(differential.b);
            if (x >= 2) {
                ret.r() += static_cast<int>(differential.dr);
                ret.g() += static_cast<int>(differential.dg);
                ret.b() += static_cast<int>(differential.db);
            }
            ret.r() = Common::Color::Convert5To8(ret.r());
            ret.g() = Common::Color::Convert5To8(ret.g());
            ret.b() = Common::Color::Convert5To8(ret.b());
        } else {
            if (x < 2) {
                ret.r() = Common::Color::Convert4To8(static_cast<u8>(separate.r1));
                ret.g() = Common::Color::Convert4To8(static_cast<u8>(separate.g1));
                ret.b() = Common::Color::Convert4To8(static_cast<u8>(separate.b1));
            } else {
                ret.r() = Common::Color::Convert4To8(static_cast<u8>(separate.r2));
                ret.g() = Common::Color::Convert4To8(static_cast<u8>(separate.g2));
                ret.b() = Common::Color::Convert4To8(static_cast<u8>(separate.b2));
            }
        }

        // Add modifier
        unsigned table_index =
            static_cast<int>((x < 2) ? table_index_1.Value() : table_index_2.Value());

        int modifier = etc1_modifier_table[table_index][GetTableSubIndex(texel)];
        if (GetNegationFlag(texel))
            modifier *= -1;

        ret.r() = std::clamp(ret.r() + modifier, 0, 255);
        ret.g() = std::clamp(ret.g() + modifier, 0, 255);
        ret.b() = std::clamp(ret.b() + modifier, 0, 255);

        return ret.Cast<u8>();
    }
};

} // anonymous namespace

Common::Vec3<u8> SampleETC1Subtile(u64 value, unsigned int x, unsigned int y) {
    ETC1Tile tile{value};
    return tile.GetRGB(x, y);
}

void DecodeETC1Block(u64 value, u32* out_rgba) {
    /**
     * POR BLOQUE, NO POR TEXEL (0.2.1.1). GetRGB rehacia para cada uno de los
     * 16 texeles el color base de su mitad, la tabla y el recorte, con campos
     * de 64 bits que en ARMv7 cuestan varias instrucciones cada uno. Un bloque
     * solo tiene OCHO colores posibles (dos mitades por cuatro modificadores):
     * se calculan una vez y cada texel elige el suyo con sus dos bits. Mismas
     * cuentas que GetRGB, comprobadas bloque a bloque contra ella (tambien los
     * desbordes del modo diferencial, que pasan por Convert5To8 truncados a
     * ocho bits igual que alli).
     */
    const u32 low = static_cast<u32>(value);
    const u32 high = static_cast<u32>(value >> 32);
    const auto signed3 = [](u32 bits) { return static_cast<s32>(bits << 29) >> 29; };
    std::array<std::array<int, 3>, 2> base;
    if ((high & 2) != 0) {
        const int r = static_cast<int>((high >> 27) & 0x1F);
        const int g = static_cast<int>((high >> 19) & 0x1F);
        const int b = static_cast<int>((high >> 11) & 0x1F);
        base[0] = {Common::Color::Convert5To8(static_cast<u8>(r)),
                   Common::Color::Convert5To8(static_cast<u8>(g)),
                   Common::Color::Convert5To8(static_cast<u8>(b))};
        base[1] = {Common::Color::Convert5To8(static_cast<u8>(r + signed3(high >> 24))),
                   Common::Color::Convert5To8(static_cast<u8>(g + signed3(high >> 16))),
                   Common::Color::Convert5To8(static_cast<u8>(b + signed3(high >> 8)))};
    } else {
        base[0] = {Common::Color::Convert4To8(static_cast<u8>(high >> 28)),
                   Common::Color::Convert4To8(static_cast<u8>((high >> 20) & 0xF)),
                   Common::Color::Convert4To8(static_cast<u8>((high >> 12) & 0xF))};
        base[1] = {Common::Color::Convert4To8(static_cast<u8>((high >> 24) & 0xF)),
                   Common::Color::Convert4To8(static_cast<u8>((high >> 16) & 0xF)),
                   Common::Color::Convert4To8(static_cast<u8>((high >> 8) & 0xF))};
    }
    const std::array<u32, 2> tables{(high >> 5) & 7, (high >> 2) & 7};
    u32 palette[2][4];
    for (u32 half = 0; half < 2; half++) {
        for (u32 index = 0; index < 4; index++) {
            int modifier = etc1_modifier_table[tables[half]][index & 1];
            if ((index & 2) != 0) {
                modifier = -modifier;
            }
            const u32 r = static_cast<u32>(std::clamp(base[half][0] + modifier, 0, 255));
            const u32 g = static_cast<u32>(std::clamp(base[half][1] + modifier, 0, 255));
            const u32 b = static_cast<u32>(std::clamp(base[half][2] + modifier, 0, 255));
            palette[half][index] = r | (g << 8) | (b << 16) | 0xFF000000u;
        }
    }
    // La mitad: columnas 0-1 / 2-3, o filas si el bloque va volteado. Los dos
    // bits del texel van por columnas (4x + y), como en GetRGB.
    const bool flip = (high & 1) != 0;
    for (u32 y = 0; y < 4; y++) {
        for (u32 x = 0; x < 4; x++) {
            const u32 texel = 4 * x + y;
            const u32 index = ((low >> texel) & 1) | (((low >> (16 + texel)) & 1) << 1);
            out_rgba[x + y * 4] = palette[flip ? (y >> 1) : (x >> 1)][index];
        }
    }
}

u32 Etc1BlockCache::GetTexel(u64 block, u32 index) {
    const u32 slot = Hash(block);

    /**
     * El bloque cero es un bloque VALIDO (sale negro), asi que 'tag == 0' no
     * puede significar "entrada vacia": una entrada recien creada, con tag a
     * cero, devolveria basura al consultarla con el bloque cero. La solucion
     * es no cachear nunca el bloque cero: se decodifica, se devuelve el
     * resultado y no se escribe en la tabla. Un bloque cero es raro (una
     * textura negra entera), asi que perder su cache no se nota.
     */
    if (block != 0 && tags[slot] == block) [[likely]] {
        return pixels[slot][index];
    }

    DecodeETC1Block(block, scratch.data());
    if (block != 0) {
        tags[slot] = block;
        std::memcpy(pixels[slot].data(), scratch.data(), sizeof(scratch));
    }
    return scratch[index];
}

} // namespace Pica::Texture
