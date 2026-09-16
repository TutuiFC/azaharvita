// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#if defined(__ARM_NEON) && defined(__PSVITA__)
#include <arm_neon.h>
#endif
#include "common/color.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica_types.h"
#include "video_core/renderer_software/sw_framebuffer.h"
#include "video_core/utils.h"

namespace SwRenderer {

using Pica::f16;
using Pica::FramebufferRegs;

namespace {

/// Decode/Encode for shadow map format. It is similar to D24S8 format,
/// but the depth field is in big-endian.
const Common::Vec2<u32> DecodeD24S8Shadow(const u8* bytes) {
    return {static_cast<u32>((bytes[0] << 16) | (bytes[1] << 8) | bytes[2]), bytes[3]};
}

void EncodeD24X8Shadow(u32 depth, u8* bytes) {
    bytes[2] = depth & 0xFF;
    bytes[1] = (depth >> 8) & 0xFF;
    bytes[0] = (depth >> 16) & 0xFF;
}

void EncodeX24S8Shadow(u8 stencil, u8* bytes) {
    bytes[3] = stencil;
}
} // Anonymous namespace

Framebuffer::Framebuffer(Memory::MemorySystem& memory_, const Pica::FramebufferRegs& regs_)
    : memory{memory_}, regs{regs_} {}

Framebuffer::~Framebuffer() = default;

void Framebuffer::Bind() {
    PAddr addr = regs.framebuffer.GetColorBufferPhysicalAddress();
    if (color_addr != addr) [[unlikely]] {
        color_addr = addr;
        color_buffer = memory.GetPhysicalPointer(color_addr);
    }

    addr = regs.framebuffer.GetDepthBufferPhysicalAddress();
    if (depth_addr != addr) [[unlikely]] {
        depth_addr = addr;
        depth_buffer = memory.GetPhysicalPointer(depth_addr);
    }

    // Formato y dimensiones, resueltos aqui y no en cada pixel.
    //
    // DrawPixel/GetPixel/GetDepth/SetDepth releian estos campos de los
    // registros de la PICA POR CADA PIXEL, y 'bytes por pixel' ademas es un
    // switch sobre el formato. Son valores fijos mientras el framebuffer este
    // ligado -- que es justo lo que significa Bind -- asi que recalcularlos
    // millones de veces por fotograma no aportaba nada.
    //
    // 'regs' es una referencia, ademas: el compilador no puede dar por hecho
    // que su contenido siga igual despues de cada llamada que no ve, asi que
    // volvia a cargarlos de memoria una y otra vez.
    cached_color_format = regs.framebuffer.color_format;
    cached_color_bpp =
        Pica::BytesPerPixel(static_cast<Pica::PixelFormat>(regs.framebuffer.color_format.Value()));
    cached_depth_format = regs.framebuffer.depth_format;
    cached_depth_bpp = Pica::FramebufferRegs::BytesPerDepthPixel(cached_depth_format);
    // (1 << bits) - 1 es exacto en float hasta 24 bits, el maximo de este
    // formato, asi que el producto de abajo da bit a bit lo mismo que la
    // version que se recalcula por pixel.
    cached_depth_scale =
        static_cast<float>((1u << Pica::FramebufferRegs::DepthBitsPerPixel(cached_depth_format)) -
                           1u);
    cached_width = regs.framebuffer.width;
    cached_height = regs.framebuffer.height;
    cached_color_stride = cached_width * cached_color_bpp;
    cached_depth_stride = cached_width * cached_depth_bpp;
}

void Framebuffer::DrawShadowMapPixel(u32 x, u32 y, u32 depth, u8 stencil) const {
    const auto& framebuffer = regs.framebuffer;
    const auto& shadow = regs.shadow;
    const PAddr addr = framebuffer.GetColorBufferPhysicalAddress();

    y = cached_height - y;

    const u32 coarse_y = y & ~7;
    u32 bytes_per_pixel = 4;
    u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) +
                     coarse_y * cached_depth_stride;
    u8* shadow_buffer = memory.GetPhysicalPointer(addr);
    u8* dst_pixel = shadow_buffer + dst_offset;

    const auto ref = DecodeD24S8Shadow(dst_pixel);
    const u32 ref_z = ref.x;
    const u32 ref_s = ref.y;

    if (depth >= ref_z) {
        return;
    }

    if (stencil == 0) {
        EncodeD24X8Shadow(depth, dst_pixel);
    } else {
        const f16 constant = f16::FromRaw(shadow.constant);
        const f16 linear = f16::FromRaw(shadow.linear);
        const f16 x_ = f16::FromFloat32(static_cast<float>(depth) / ref_z);
        const f16 stencil_new = f16::FromFloat32(stencil) / (constant + linear * x_);
        stencil = static_cast<u8>(std::clamp(stencil_new.ToFloat32(), 0.0f, 255.0f));

        if (stencil < ref_s) {
            EncodeX24S8Shadow(stencil, dst_pixel);
        }
    }
}

u8 PerformStencilAction(FramebufferRegs::StencilAction action, u8 old_stencil, u8 ref) {
    switch (action) {
    case FramebufferRegs::StencilAction::Keep:
        return old_stencil;
    case FramebufferRegs::StencilAction::Zero:
        return 0;
    case FramebufferRegs::StencilAction::Replace:
        return ref;
    case FramebufferRegs::StencilAction::Increment:
        // Saturated increment
        return std::min<u8>(old_stencil, 254) + 1;
    case FramebufferRegs::StencilAction::Decrement:
        // Saturated decrement
        return std::max<u8>(old_stencil, 1) - 1;
    case FramebufferRegs::StencilAction::Invert:
        return ~old_stencil;
    case FramebufferRegs::StencilAction::IncrementWrap:
        return old_stencil + 1;
    case FramebufferRegs::StencilAction::DecrementWrap:
        return old_stencil - 1;
    default:
        LOG_CRITICAL(HW_GPU, "Unknown stencil action {:x}", static_cast<int>(action));
        UNIMPLEMENTED();
        return 0;
    }
}

Common::Vec4<u8> EvaluateBlendEquation(const Common::Vec4<u8>& src,
                                       const Common::Vec4<u8>& srcfactor,
                                       const Common::Vec4<u8>& dest,
                                       const Common::Vec4<u8>& destfactor,
                                       FramebufferRegs::BlendEquation equation) {
#if defined(__ARM_NEON) && defined(__PSVITA__)
    /**
     * Camino NEON para la ecuacion Add, que es la mezcla alfa de toda la vida y
     * la que usan practicamente todos los juegos.
     *
     * Medido en el binario: PixelColor son 783 instrucciones y NI UNA vectorial,
     * pese a ser aritmetica de bytes sobre cuatro canales -- el caso de libro
     * para NEON. El compilador no la vectoriza porque el codigo escalar mezcla
     * promociones a int, division y clamp, y no reconoce el patron.
     *
     * Las otras cuatro ecuaciones (Subtract, ReverseSubtract, Min, Max) se
     * quedan en el camino escalar de abajo: son raras, y Subtract necesita
     * aritmetica con signo que complicaria esto sin ganar nada.
     *
     * La division entre 255 se hace exacta, no aproximada:
     *
     *     x / 255 == (x + 1 + (x >> 8)) >> 8
     *
     * y se comprueba en los extremos del rango que puede salir aqui, 0 y
     * 255*255*2 = 130.050:
     *     (130050 + 1 + 508) >> 8 = 510   y   130050 / 255 = 510  ✓
     *     (255 + 1 + 0) >> 8 = 1          y   255 / 255 = 1       ✓
     *     (254 + 1 + 0) >> 8 = 0          y   254 / 255 = 0       ✓
     *
     * Hace falta ensanchar a 32 bits antes de sumar: cada producto llega a
     * 65.025 y la suma de los dos se sale de 16 bits.
     */
    if (equation == FramebufferRegs::BlendEquation::Add) {
        // Los cuatro canales viajan como un u32: Common::Vec4<T> tiene x,y,z,w
        // contiguos, asi que es una carga, no cuatro extracciones con sus
        // desplazamientos y mascaras (que era lo que hacia antes, en los CUATRO
        // vectores, en cada pixel mezclado).
        static_assert(sizeof(Common::Vec4<u8>) == sizeof(u32));
        u32 src_packed;
        u32 srcfactor_packed;
        u32 dest_packed;
        u32 destfactor_packed;
        std::memcpy(&src_packed, src.AsArray(), sizeof(src_packed));
        std::memcpy(&srcfactor_packed, srcfactor.AsArray(), sizeof(srcfactor_packed));
        std::memcpy(&dest_packed, dest.AsArray(), sizeof(dest_packed));
        std::memcpy(&destfactor_packed, destfactor.AsArray(), sizeof(destfactor_packed));

        const uint8x8_t v_src = vreinterpret_u8_u32(vdup_n_u32(src_packed));
        const uint8x8_t v_srcf = vreinterpret_u8_u32(vdup_n_u32(srcfactor_packed));
        const uint8x8_t v_dst = vreinterpret_u8_u32(vdup_n_u32(dest_packed));
        const uint8x8_t v_dstf = vreinterpret_u8_u32(vdup_n_u32(destfactor_packed));

        // Cuatro productos de 8x8 -> 16 bits en una sola instruccion cada uno.
        const uint16x8_t prod_src = vmull_u8(v_src, v_srcf);
        const uint16x8_t prod_dst = vmull_u8(v_dst, v_dstf);

        // A 32 bits antes de sumar: 65.025 + 65.025 no cabe en 16.
        uint32x4_t sum = vaddl_u16(vget_low_u16(prod_src), vget_low_u16(prod_dst));

        // Division exacta entre 255.
        sum = vshrq_n_u32(vaddq_u32(vaddq_u32(sum, vdupq_n_u32(1)), vshrq_n_u32(sum, 8)), 8);

        // Estrechar con saturacion hace el clamp a 255 gratis.
        const uint8x8_t packed = vqmovn_u16(vcombine_u16(vqmovn_u32(sum), vdup_n_u16(0)));
        u32 out;
        vst1_lane_u32(&out, vreinterpret_u32_u8(packed), 0);
        Common::Vec4<u8> result;
        std::memcpy(result.AsArray(), &out, sizeof(out));
        return result;
    }
#endif

    Common::Vec4i result;

    const auto src_result = (src * srcfactor).Cast<s32>();
    const auto dst_result = (dest * destfactor).Cast<s32>();

    switch (equation) {
    case FramebufferRegs::BlendEquation::Add:
        result = (src_result + dst_result) / 255;
        break;
    case FramebufferRegs::BlendEquation::Subtract:
        result = (src_result - dst_result) / 255;
        break;
    case FramebufferRegs::BlendEquation::ReverseSubtract:
        result = (dst_result - src_result) / 255;
        break;
    case FramebufferRegs::BlendEquation::Min:
        result.r() = std::min(src_result.r(), dst_result.r()) / 255;
        result.g() = std::min(src_result.g(), dst_result.g()) / 255;
        result.b() = std::min(src_result.b(), dst_result.b()) / 255;
        result.a() = std::min(src_result.a(), dst_result.a()) / 255;
        break;
    case FramebufferRegs::BlendEquation::Max:
        result.r() = std::max(src_result.r(), dst_result.r()) / 255;
        result.g() = std::max(src_result.g(), dst_result.g()) / 255;
        result.b() = std::max(src_result.b(), dst_result.b()) / 255;
        result.a() = std::max(src_result.a(), dst_result.a()) / 255;
        break;
    default:
        LOG_CRITICAL(HW_GPU, "Unknown RGB blend equation 0x{:x}", equation);
        UNIMPLEMENTED();
    }

    return Common::Vec4<u8>(std::clamp(result.r(), 0, 255), std::clamp(result.g(), 0, 255),
                            std::clamp(result.b(), 0, 255), std::clamp(result.a(), 0, 255));
};

u8 LogicOp(u8 src, u8 dest, FramebufferRegs::LogicOp op) {
    switch (op) {
    case FramebufferRegs::LogicOp::Clear:
        return 0;
    case FramebufferRegs::LogicOp::And:
        return src & dest;
    case FramebufferRegs::LogicOp::AndReverse:
        return src & ~dest;
    case FramebufferRegs::LogicOp::Copy:
        return src;
    case FramebufferRegs::LogicOp::Set:
        return 255;
    case FramebufferRegs::LogicOp::CopyInverted:
        return ~src;
    case FramebufferRegs::LogicOp::NoOp:
        return dest;
    case FramebufferRegs::LogicOp::Invert:
        return ~dest;
    case FramebufferRegs::LogicOp::Nand:
        return ~(src & dest);
    case FramebufferRegs::LogicOp::Or:
        return src | dest;
    case FramebufferRegs::LogicOp::Nor:
        return ~(src | dest);
    case FramebufferRegs::LogicOp::Xor:
        return src ^ dest;
    case FramebufferRegs::LogicOp::Equiv:
        return ~(src ^ dest);
    case FramebufferRegs::LogicOp::AndInverted:
        return ~src & dest;
    case FramebufferRegs::LogicOp::OrReverse:
        return src | ~dest;
    case FramebufferRegs::LogicOp::OrInverted:
        return ~src | dest;
    }
    UNREACHABLE();
};

} // namespace SwRenderer
