// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/color.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/utils.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct FramebufferRegs;
}

namespace SwRenderer {

// Los cuerpos en linea de abajo usan los enums de formato de la PICA; el alias
// evita calificar cada uno (el .cpp ya hacia este mismo using).
using Pica::FramebufferRegs;

class Framebuffer {
public:
    explicit Framebuffer(Memory::MemorySystem& memory, const Pica::FramebufferRegs& framebuffer);
    ~Framebuffer();

    /// Updates the framebuffer addresses from the PICA registers.
    void Bind();

    /**
     * Accesores de color y profundidad, EN LINEA en la cabecera.
     *
     * Los seis corren varias veces por pixel cubierto (lectura del destino,
     * prueba de profundidad, escritura) y su cuerpo es una decodificacion
     * corta. Con el cuerpo en el .cpp, cada acceso costaba ademas una llamada
     * real: en un rasterizador que ya va justo, cuatro o cinco llamadas por
     * pixel son una parte medible del presupuesto. Aqui el compilador los mete
     * en el bucle de pixeles y pliega los switches de formato.
     */
    void DrawPixel(u32 x, u32 y, const Common::Vec4<u8>& color) const {
        // Similarly to textures, the render framebuffer is laid out from bottom to top, too.
        // NOTE: The framebuffer height register contains the actual FB height minus one.
        y = cached_height - y;

        const u32 coarse_y = y & ~7;
        const u32 dst_offset =
            VideoCore::GetMortonOffset(x, y, cached_color_bpp) + coarse_y * cached_color_stride;
        u8* dst_pixel = color_buffer + dst_offset;

        switch (cached_color_format) {
        case FramebufferRegs::ColorFormat::RGBA8:
            Common::Color::EncodeRGBA8(color, dst_pixel);
            break;
        case FramebufferRegs::ColorFormat::RGB8:
            Common::Color::EncodeRGB8(color, dst_pixel);
            break;
        case FramebufferRegs::ColorFormat::RGB5A1:
            Common::Color::EncodeRGB5A1(color, dst_pixel);
            break;
        case FramebufferRegs::ColorFormat::RGB565:
            Common::Color::EncodeRGB565(color, dst_pixel);
            break;
        case FramebufferRegs::ColorFormat::RGBA4:
            Common::Color::EncodeRGBA4(color, dst_pixel);
            break;
        default:
            LOG_CRITICAL(Render_Software, "Unknown framebuffer color format {:x}",
                         static_cast<u32>(cached_color_format));
            UNIMPLEMENTED();
        }
    }

    /// Returns the current color at the specified coordinates.
    [[nodiscard]] Common::Vec4<u8> GetPixel(u32 x, u32 y) const {
        y = cached_height - y;

        const u32 coarse_y = y & ~7;
        const u32 src_offset =
            VideoCore::GetMortonOffset(x, y, cached_color_bpp) + coarse_y * cached_color_stride;
        const u8* src_pixel = color_buffer + src_offset;

        switch (cached_color_format) {
        case FramebufferRegs::ColorFormat::RGBA8:
            return Common::Color::DecodeRGBA8(src_pixel);
        case FramebufferRegs::ColorFormat::RGB8:
            return Common::Color::DecodeRGB8(src_pixel);
        case FramebufferRegs::ColorFormat::RGB5A1:
            return Common::Color::DecodeRGB5A1(src_pixel);
        case FramebufferRegs::ColorFormat::RGB565:
            return Common::Color::DecodeRGB565(src_pixel);
        case FramebufferRegs::ColorFormat::RGBA4:
            return Common::Color::DecodeRGBA4(src_pixel);
        default:
            LOG_CRITICAL(Render_Software, "Unknown framebuffer color format {:x}",
                         static_cast<u32>(cached_color_format));
            UNIMPLEMENTED();
        }

        return {0, 0, 0, 0};
    }

    /// Returns the depth value at the specified coordinates.
    [[nodiscard]] u32 GetDepth(u32 x, u32 y) const {
        y = cached_height - y;

        const u32 coarse_y = y & ~7;
        const u32 src_offset =
            VideoCore::GetMortonOffset(x, y, cached_depth_bpp) + coarse_y * cached_depth_stride;
        const u8* src_pixel = depth_buffer + src_offset;

        switch (cached_depth_format) {
        case FramebufferRegs::DepthFormat::D16:
            return Common::Color::DecodeD16(src_pixel);
        case FramebufferRegs::DepthFormat::D24:
            return Common::Color::DecodeD24(src_pixel);
        case FramebufferRegs::DepthFormat::D24S8:
            return Common::Color::DecodeD24S8(src_pixel).x;
        default:
            LOG_CRITICAL(HW_GPU, "Unimplemented depth format {}",
                         static_cast<u32>(cached_depth_format));
            UNIMPLEMENTED();
            return 0;
        }
    }

    /// Returns the stencil value at the specified coordinates.
    [[nodiscard]] u8 GetStencil(u32 x, u32 y) const {
        y = cached_height - y;

        const u32 coarse_y = y & ~7;
        const u32 src_offset =
            VideoCore::GetMortonOffset(x, y, cached_depth_bpp) + coarse_y * cached_depth_stride;
        const u8* src_pixel = depth_buffer + src_offset;

        switch (cached_depth_format) {
        case FramebufferRegs::DepthFormat::D24S8:
            return Common::Color::DecodeD24S8(src_pixel).y;
        default:
            LOG_WARNING(
                HW_GPU,
                "GetStencil called for function which doesn't have a stencil component (format {})",
                static_cast<u32>(cached_depth_format));
            return 0;
        }
    }

    /// Stores the provided depth value at the specified coordinates.
    void SetDepth(u32 x, u32 y, u32 value) const {
        y = cached_height - y;

        const u32 coarse_y = y & ~7;
        const u32 dst_offset =
            VideoCore::GetMortonOffset(x, y, cached_depth_bpp) + coarse_y * cached_depth_stride;
        u8* dst_pixel = depth_buffer + dst_offset;

        switch (cached_depth_format) {
        case FramebufferRegs::DepthFormat::D16:
            Common::Color::EncodeD16(value, dst_pixel);
            break;
        case FramebufferRegs::DepthFormat::D24:
            Common::Color::EncodeD24(value, dst_pixel);
            break;
        case FramebufferRegs::DepthFormat::D24S8:
            Common::Color::EncodeD24X8(value, dst_pixel);
            break;
        default:
            LOG_CRITICAL(HW_GPU, "Unimplemented depth format {}",
                         static_cast<u32>(cached_depth_format));
            UNIMPLEMENTED();
            break;
        }
    }

    /// Stores the provided stencil value at the specified coordinates.
    void SetStencil(u32 x, u32 y, u8 value) const {
        y = cached_height - y;

        const u32 coarse_y = y & ~7;
        const u32 dst_offset =
            VideoCore::GetMortonOffset(x, y, cached_depth_bpp) + coarse_y * cached_depth_stride;
        u8* dst_pixel = depth_buffer + dst_offset;

        switch (cached_depth_format) {
        case FramebufferRegs::DepthFormat::D16:
        case FramebufferRegs::DepthFormat::D24:
            // Nothing to do
            break;
        case FramebufferRegs::DepthFormat::D24S8:
            Common::Color::EncodeX24S8(value, dst_pixel);
            break;
        default:
            LOG_CRITICAL(HW_GPU, "Unimplemented depth format {}",
                         static_cast<u32>(cached_depth_format));
            UNIMPLEMENTED();
            break;
        }
    }

    /// Draws a pixel to the shadow buffer.
    void DrawShadowMapPixel(u32 x, u32 y, u32 depth, u8 stencil) const;

    /**
     * Escala de cuantizacion de profundidad ((1 << bits) - 1), resuelta en
     * Bind.
     *
     * Las pruebas de profundidad la recalculaban POR PIXEL a partir del
     * formato: numero de bits, desplazamiento y conversion a float. El formato
     * no cambia mientras el framebuffer este ligado, que es justo lo que
     * significa Bind.
     */
    [[nodiscard]] float DepthScale() const {
        return cached_depth_scale;
    }

private:
    Memory::MemorySystem& memory;
    const Pica::FramebufferRegs& regs;
    PAddr color_addr;
    u8* color_buffer{};
    PAddr depth_addr;
    u8* depth_buffer{};

    // Formato y dimensiones resueltos en Bind(), no en cada pixel. Ver el
    // comentario de Framebuffer::Bind.
    Pica::FramebufferRegs::ColorFormat cached_color_format{};
    Pica::FramebufferRegs::DepthFormat cached_depth_format{};
    u32 cached_color_bpp{};
    u32 cached_depth_bpp{};
    u32 cached_width{};
    u32 cached_height{};
    u32 cached_color_stride{};
    u32 cached_depth_stride{};
    float cached_depth_scale{1.0f};
};

u8 PerformStencilAction(Pica::FramebufferRegs::StencilAction action, u8 old_stencil, u8 ref);

Common::Vec4<u8> EvaluateBlendEquation(const Common::Vec4<u8>& src,
                                       const Common::Vec4<u8>& srcfactor,
                                       const Common::Vec4<u8>& dest,
                                       const Common::Vec4<u8>& destfactor,
                                       Pica::FramebufferRegs::BlendEquation equation);

u8 LogicOp(u8 src, u8 dest, Pica::FramebufferRegs::LogicOp op);

} // namespace SwRenderer
