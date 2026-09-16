// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <span>

#include "common/common_types.h"
#include "common/logging/log.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_texturing.h"

namespace SwRenderer {

/**
 * Ajusta una coordenada de textura al modo de wrap. EN LINEA en la cabecera.
 *
 * El bucle de pixeles lo llama dos veces por unidad de textura, asi que con el
 * cuerpo en el .cpp cada coordenada costaba ademas una llamada real. El camino
 * rapido (coordenada ya dentro de la textura, que es casi todas) son dos
 * comparaciones, y meterlo en el bucle deja que el compilador lo pliegue.
 */
inline int GetWrappedTexCoord(Pica::TexturingRegs::TextureConfig::WrapMode mode, s32 val, u32 size,
                              u32 size_mask = 0, bool size_pow2 = false) {
    using TextureConfig = Pica::TexturingRegs::TextureConfig;

    // Con tamano potencia de dos, el modulo es un AND. Ver TextureUnitCache.
    const auto wrap_modulo = [size, size_mask, size_pow2](u32 value) -> s32 {
        return size_pow2 ? static_cast<s32>(value & size_mask)
                         : static_cast<s32>(value % size);
    };

    /**
     * Camino rapido: una coordenada que ya cae dentro de la textura no la
     * cambia NINGUN modo de wrap.
     *
     *   Repeat y sus variantes: val % size == val, y el modulo es lo caro.
     *   ClampToEdge / ClampToEdge2: se queda igual.
     *   MirroredRepeat: val < size, asi que tampoco le da la vuelta.
     *   ClampToBorder / ClampToBorder2: el borde ya se decidio antes de
     *   llamar aqui; si se llega, es porque la coordenada es valida.
     *
     * Y el modulo es MUY caro: 'val % size' con 'size' conocido solo en
     * ejecucion es una llamada a __aeabi_uidivmod, decenas de ciclos, y esto
     * corre dos veces por pixel y por unidad de textura. En una textura con
     * Repeat -- fondos y suelos con patron, lo normal -- practicamente todos
     * los pixeles caen dentro, asi que el caso general pasa a ser una
     * comparacion.
     */
    if (val >= 0 && static_cast<u32>(val) < size) [[likely]] {
        return val;
    }

    switch (mode) {
    case TextureConfig::ClampToEdge2:
        // For negative coordinate, ClampToEdge2 behaves the same as Repeat
        if (val < 0) {
            return wrap_modulo(static_cast<u32>(val));
        }
        [[fallthrough]];
    case TextureConfig::ClampToEdge:
        val = std::max(val, 0);
        val = std::min(val, static_cast<s32>(size) - 1);
        return val;
    case TextureConfig::ClampToBorder:
        return val;
    case TextureConfig::ClampToBorder2:
    // For ClampToBorder2, the case of positive coordinate beyond the texture size is already
    // handled outside. Here we only handle the negative coordinate in the same way as Repeat.
    case TextureConfig::Repeat2:
    case TextureConfig::Repeat3:
    case TextureConfig::Repeat:
        return wrap_modulo(static_cast<u32>(val));
    case TextureConfig::MirroredRepeat: {
        // El modulo es sobre 2*size, que tambien es potencia de dos si lo es
        // size: la mascara equivalente es (2*size - 1).
        u32 coord = size_pow2 ? (static_cast<u32>(val) & ((size << 1) - 1))
                              : (static_cast<u32>(val) % (2 * size));
        if (coord >= size) {
            coord = 2 * size - 1 - coord;
        }
        return static_cast<s32>(coord);
    }
    default:
        LOG_ERROR(HW_GPU, "Unknown texture coordinate wrapping mode {:x}", (int)mode);
        UNIMPLEMENTED();
        return 0;
    }
};

Common::Vec3<u8> GetColorModifier(Pica::TexturingRegs::TevStageConfig::ColorModifier factor,
                                  const Common::Vec4<u8>& values);

u8 GetAlphaModifier(Pica::TexturingRegs::TevStageConfig::AlphaModifier factor,
                    const Common::Vec4<u8>& values);

Common::Vec3<u8> ColorCombine(Pica::TexturingRegs::TevStageConfig::Operation op,
                              std::span<const Common::Vec3<u8>, 3> input);

u8 AlphaCombine(Pica::TexturingRegs::TevStageConfig::Operation op, const std::array<u8, 3>& input);

} // namespace SwRenderer
