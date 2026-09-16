// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <psp2/gxm.h>
#include "common/common_types.h"
// Pica::PixelFormat se define aqui; rasterizer_cache/pixel_format.h solo lo
// declara por delante y no basta para nombrar sus valores.
#include "video_core/pica/regs_external.h"

namespace Gxm {

/**
 * Formato de GXM equivalente al del framebuffer del 3DS.
 *
 * La GPU de la Vita entiende los cinco de forma nativa, asi que no hay que
 * convertir nada en la CPU: se le da el buffer crudo y ella lo interpreta al
 * muestrear la textura.
 *
 * EL ORDEN DE CANALES, DEDUCIDO Y NO ADIVINADO.
 *
 * El primer intento uso las variantes BGR/ABGR y los carteles amarillos salieron
 * azules: rojo y azul intercambiados.
 *
 * La regla sale del unico caso que ya se sabia bueno: el buffer RGBA8 que
 * escribia el codigo anterior guardaba los bytes en orden R,G,B y se subia como
 * A8B8G8R8. O sea que el nombre de GXM lista los canales del bit MAS
 * significativo al menos, y la memoria en little-endian va justo al reves.
 *
 * Aplicando eso a como lee Citra cada formato (ver Common::Color):
 *
 *   RGB565   pixel>>11 = R, >>5 = G, &0x1F = B   -> R G B de mas a menos
 *   RGB5A1   >>11 R, >>6 G, >>1 B, &1 A          -> R G B A
 *   RGBA4    >>12 R, >>8 G, >>4 B, &0xF A        -> R G B A
 *   RGB8     bytes[2]=R, [1]=G, [0]=B            -> R G B
 *   RGBA8    bytes[3]=R, [2]=G, [1]=B, [0]=A     -> R G B A
 *
 * De ahi salen todos los sufijos _RGB / _RGBA.
 *
 * Esto vivia dentro de citra_vita/vita_window.cpp, donde solo lo veia el camino
 * de vita2d. El backend GXM necesita exactamente la misma tabla, y tener dos
 * copias de una correspondencia que costo un ciclo de prueba en consola
 * averiguar es pedir que se desincronicen.
 */
[[nodiscard]] inline SceGxmTextureFormat FormatFor(Pica::PixelFormat format) {
    switch (format) {
    case Pica::PixelFormat::RGBA8:
        return SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA;
    case Pica::PixelFormat::RGB8:
        return SCE_GXM_TEXTURE_FORMAT_U8U8U8_RGB;
    case Pica::PixelFormat::RGB5A1:
        return SCE_GXM_TEXTURE_FORMAT_U5U5U5U1_RGBA;
    case Pica::PixelFormat::RGB565:
        return SCE_GXM_TEXTURE_FORMAT_U5U6U5_RGB;
    case Pica::PixelFormat::RGBA4:
        return SCE_GXM_TEXTURE_FORMAT_U4U4U4U4_RGBA;
    }
    return SCE_GXM_TEXTURE_FORMAT_U5U6U5_RGB;
}

/// Bytes por pixel del formato del 3DS.
[[nodiscard]] inline u32 BytesPerPixelFor(Pica::PixelFormat format) {
    switch (format) {
    case Pica::PixelFormat::RGBA8:
        return 4;
    case Pica::PixelFormat::RGB8:
        return 3;
    default:
        return 2;
    }
}

} // namespace Gxm
