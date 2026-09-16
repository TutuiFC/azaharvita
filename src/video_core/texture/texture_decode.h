// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_texturing.h"

namespace Pica::Texture {

class Etc1BlockCache;

/// Returns the byte size of a 8*8 tile of the specified texture format.
size_t CalculateTileSize(TexturingRegs::TextureFormat format);

struct TextureInfo {
    PAddr physical_address;
    u32 width;
    u32 height;
    ptrdiff_t stride;
    TexturingRegs::TextureFormat format;
    bool is_shadow_source;

    /**
     * Bytes que ocupa un mosaico de 8x8 en este formato.
     *
     * Lo necesita LookupTexture para saltar de un mosaico al siguiente, y lo
     * llamaba POR TEXEL a traves de CalculateTileSize, que es un switch de
     * cinco ramas. El formato no cambia mientras la unidad de textura este
     * configurada, asi que se resuelve una vez aqui.
     */
    std::size_t tile_size;

    static TextureInfo FromPicaRegister(const TexturingRegs::TextureConfig& config,
                                        const TexturingRegs::TextureFormat& format);

    /// Calculates stride from format and width, assuming that the entire texture is contiguous.
    void SetDefaultStride() {
        tile_size = CalculateTileSize(format);
        stride = static_cast<ptrdiff_t>(tile_size) * (width / 8);
    }
};

/**
 * Looks up a texel from a single 8x8 texture tile.
 *
 * @param source Pointer to the beginning of the tile.
 * @param x, y In-tile coordinates to read from. Must be < 8.
 * @param info TextureInfo describing the texture format.
 * @param disable_alpha Used for debugging. Sets the result alpha to 255 and either discards the
 *                      real alpha or inserts it in an otherwise unused channel.
 * @param etc1_cache Cache de bloques ETC1 del hilo que llama (ver LookupTexture).
 */
Common::Vec4<u8> LookupTexelInTile(const u8* source, unsigned int x, unsigned int y,
                                   const TextureInfo& info, bool disable_alpha,
                                   Etc1BlockCache* etc1_cache = nullptr);

/**
 * Lookup texel located at the given coordinates and return an RGBA vector of its color.
 * @param source Source pointer to read data from
 * @param x,y Texture coordinates to read from
 * @param info TextureInfo object describing the texture setup
 * @param disable_alpha This is used for debug widgets which use this method to display textures
 * without providing a good way to visualize alpha by themselves. If true, this will return 255 for
 * the alpha component, and either drop the information entirely or store it in an "unused" color
 * channel.
 * @param etc1_cache Cache de bloques ETC1 del hilo que llama. Si es null (widgets de depuracion)
 * se decodifica el bloque en cada texel, como antes. El rasterizador siempre pasa uno.
 * @todo Eventually we should get rid of the disable_alpha parameter.
 *
 * El cuerpo va aqui, en linea, y no en el .cpp: son seis operaciones y una
 * llamada, y separarlo en otra unidad obligaba a un salto mas por unidad de
 * textura y por pixel (hasta tres por pixel) solo para calcular el mosaico.
 */
inline Common::Vec4<u8> LookupTexture(const u8* source, unsigned int x, unsigned int y,
                                      const TextureInfo& info, bool disable_alpha = false,
                                      Etc1BlockCache* etc1_cache = nullptr) {
    // Coordinate in tiles
    const unsigned int coarse_x = x / 8;
    const unsigned int coarse_y = y / 8;

    // Coordinate inside the tile
    const unsigned int fine_x = x % 8;
    const unsigned int fine_y = y % 8;

    // tile_size viene precalculado en TextureInfo: antes esto llamaba a
    // CalculateTileSize (un switch de cinco ramas) en CADA texel, para un valor
    // que solo depende del formato de la unidad de textura.
    const u8* line = source + coarse_y * info.stride;
    const u8* tile = line + coarse_x * info.tile_size;
    return LookupTexelInTile(tile, fine_x, fine_y, info, disable_alpha, etc1_cache);
}

} // namespace Pica::Texture
