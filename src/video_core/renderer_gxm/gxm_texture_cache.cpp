// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/gxm_texture_cache.h"

#include <utility>
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/texture/texture_decode.h"

namespace Gxm {

namespace {

using Pica::TexturingRegs;

/// Mapea el modo de repetido de la PICA al de GXM. Devuelve false si no hay
/// equivalente (los modos con borde, que necesitan el color de borde).
bool MapWrap(TexturingRegs::TextureConfig::WrapMode wrap, SceGxmTextureAddrMode* out) {
    switch (wrap) {
    case TexturingRegs::TextureConfig::ClampToEdge:
    case TexturingRegs::TextureConfig::ClampToEdge2:
        *out = SCE_GXM_TEXTURE_ADDR_CLAMP;
        return true;
    case TexturingRegs::TextureConfig::Repeat:
    case TexturingRegs::TextureConfig::Repeat2:
    case TexturingRegs::TextureConfig::Repeat3:
        *out = SCE_GXM_TEXTURE_ADDR_REPEAT;
        return true;
    case TexturingRegs::TextureConfig::MirroredRepeat:
        *out = SCE_GXM_TEXTURE_ADDR_MIRROR;
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

TextureCache::~TextureCache() = default;

void TextureCache::Clear() {
    for (auto& entry : entries) {
        entry.valid = false;
        entry.buffer = Allocation{};
    }
}

void TextureCache::InvalidateRange(PAddr addr, u32 size) {
    const PAddr end = addr + size;
    for (auto& entry : entries) {
        if (entry.valid && entry.address < end && addr < entry.address + entry.span) {
            entry.valid = false;
            entry.buffer = Allocation{};
        }
    }
}

const SceGxmTexture* TextureCache::Get(u32 unit, const Pica::RegsInternal& regs,
                                       Memory::MemorySystem& memory) {
    const auto& texturing = regs.texturing;
    const TexturingRegs::TextureConfig* config = nullptr;
    TexturingRegs::TextureFormat format{};
    bool enabled = false;
    switch (unit) {
    case 0:
        config = &texturing.texture0;
        format = texturing.texture0_format;
        enabled = texturing.main_config.texture0_enable;
        // La unidad 0 tiene tipo (2D, proyeccion, cubo, sombra). El resto no.
        if (config->type != TexturingRegs::TextureConfig::Texture2D) {
            return nullptr;
        }
        break;
    case 1:
        config = &texturing.texture1;
        format = texturing.texture1_format;
        enabled = texturing.main_config.texture1_enable;
        break;
    case 2:
        config = &texturing.texture2;
        format = texturing.texture2_format;
        enabled = texturing.main_config.texture2_enable;
        break;
    default:
        return nullptr;
    }
    if (!enabled) {
        // La PICA devuelve blanco para una unidad apagada, pero el generador
        // de shaders todavia no reproduce eso: mejor caer a software que
        // pintar otra cosa.
        return nullptr;
    }
    // Los mipmaps no estan soportados: la textura de GXM es un solo nivel.
    if (config->mip_filter != 0) {
        return nullptr;
    }

    SceGxmTextureAddrMode wrap_s{};
    SceGxmTextureAddrMode wrap_t{};
    if (!MapWrap(config->wrap_s.Value(), &wrap_s) || !MapWrap(config->wrap_t.Value(), &wrap_t)) {
        return nullptr;
    }

    const u64 key = static_cast<u64>(config->address.Value()) ^
                    (static_cast<u64>(format) << 32) ^
                    (static_cast<u64>(config->width) << 40) ^
                    (static_cast<u64>(config->height) << 51) ^
                    (static_cast<u64>(wrap_s) << 22) ^ (static_cast<u64>(wrap_t) << 25) ^
                    (static_cast<u64>(config->min_filter) << 28) ^
                    (static_cast<u64>(config->mag_filter) << 30);

    Entry& entry = entries[unit];
    if (entry.valid && entry.key == key) {
        return &entry.texture;
    }

    const Pica::Texture::TextureInfo info =
        Pica::Texture::TextureInfo::FromPicaRegister(*config, format);
    const u32 width = info.width;
    const u32 height = info.height;
    if (width == 0 || height == 0 || width > 1024 || height > 1024) {
        return nullptr;
    }
    const PAddr address = info.physical_address;
    const u8* source = memory.GetPhysicalPointer(address);
    if (source == nullptr) {
        return nullptr;
    }
    const u32 span = static_cast<u32>(info.stride) * (height / 8);
    if (span == 0) {
        return nullptr;
    }
    // El tramo entero tiene que estar mapeado y contiguo: la decodificacion
    // recorre todos los texeles, no solo los que el juego llegue a muestrear.
    const u8* last = memory.GetPhysicalPointer(address + span - 1);
    if (last != source + span - 1) {
        return nullptr;
    }

    const u32 needed = width * height * 4;
    // CDRAM primero: la CPU la escribe una vez y la GPU la lee en cada
    // dibujado, que es justo el caso que pide memoria dedicada.
    Allocation buffer = Allocate(Pool::Cdram, needed);
    if (!buffer.Valid()) {
        buffer = Allocate(Pool::Host, needed);
    }
    if (!buffer.Valid()) {
        LOG_ERROR(Render, "GXM: sin memoria para decodificar una textura de {}x{}", width, height);
        return nullptr;
    }

    auto* dest = static_cast<u8*>(buffer.Data());
    for (u32 y = 0; y < height; y++) {
        for (u32 x = 0; x < width; x++) {
            const Common::Vec4<u8> color = Pica::Texture::LookupTexture(source, x, y, info);
            u8* pixel = dest + (static_cast<std::size_t>(y) * width + x) * 4;
            const auto rgba = color.AsArray();
            pixel[0] = rgba[0];
            pixel[1] = rgba[1];
            pixel[2] = rgba[2];
            pixel[3] = rgba[3];
        }
    }

    // Los bytes van en orden R,G,B,A, que es lo que describe el nombre ABGR de
    // GXM (los nombres listan los canales del mas significativo al menos).
    if (sceGxmTextureInitLinear(&entry.texture, buffer.Data(), SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR,
                                width, height, 1) < 0) {
        return nullptr;
    }
    const SceGxmTextureFilter min_filter = config->min_filter ==
                                                   TexturingRegs::TextureConfig::TextureFilter::Linear
                                               ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                               : SCE_GXM_TEXTURE_FILTER_POINT;
    const SceGxmTextureFilter mag_filter = config->mag_filter ==
                                                   TexturingRegs::TextureConfig::TextureFilter::Linear
                                               ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                               : SCE_GXM_TEXTURE_FILTER_POINT;
    sceGxmTextureSetMinFilter(&entry.texture, min_filter);
    sceGxmTextureSetMagFilter(&entry.texture, mag_filter);
    sceGxmTextureSetUAddrMode(&entry.texture, wrap_s);
    sceGxmTextureSetVAddrMode(&entry.texture, wrap_t);

    entry.valid = true;
    entry.key = key;
    entry.address = address;
    entry.span = span;
    entry.buffer = std::move(buffer);
    return &entry.texture;
}

} // namespace Gxm
