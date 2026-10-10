// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/gxm_texture_cache.h"

#include <cstring>
#include <utility>
#include <arm_neon.h>
#include <fmt/format.h>
#include "common/color.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/vita_diag.h"
#include "core/memory.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/texture/etc1.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/utils.h"

namespace Gxm {

namespace {

using Pica::TexturingRegs;

/**
 * Mapea el modo de repetido de la PICA al de GXM.
 *
 * ClampToBorder SI se acepta desde 0.1.0.16, y no porque GXM sepa hacerlo: los
 * modos CLAMP_*_BORDER del chip usan un color de borde que la API no deja
 * escribir en ninguna parte (no existe sceGxmTextureSetBorderColor). El borde
 * lo pone el SHADER, que compara la coordenada contra [0, 1] y devuelve el
 * color del registro antes de muestrear (ver cg_fs_shader_gen.cpp). Por eso
 * aqui basta con CLAMP: para las coordenadas que llegan a muestrearse -- las de
 * dentro -- clamp, repeat y border dan exactamente lo mismo.
 *
 * ClampToBorder2 se queda fuera. Es uno de los modos raros de la PICA (borde
 * por el lado positivo, repeticion por el negativo) y la configuracion de
 * shader que comparten los tres backends solo marca el borde para
 * ClampToBorder, asi que el generador no sabria de el. Devolver false manda el
 * lote a software, que es el unico sitio donde ese modo esta implementado.
 */
bool MapWrap(TexturingRegs::TextureConfig::WrapMode wrap, SceGxmTextureAddrMode* out) {
    switch (wrap) {
    case TexturingRegs::TextureConfig::ClampToEdge:
    case TexturingRegs::TextureConfig::ClampToEdge2:
    case TexturingRegs::TextureConfig::ClampToBorder:
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

/// Grano de un bloque de CDRAM, repetido aqui solo para ESTIMAR lo que va a
/// costar una entrada antes de pedirla. El valor bueno lo da Allocation::Size()
/// una vez reservada; ver gxm_memory.cpp.
constexpr u32 kCdramGrain = 256u * 1024u;

constexpr u32 EstimateBlockBytes(u32 needed) {
    if (needed <= TextureCache::kSmallTexture) {
        return (needed + 4095u) / 4096u * 4096u;
    }
    return (needed + kCdramGrain - 1) / kCdramGrain * kCdramGrain;
}

constexpr u32 PackRGBA(const Common::Vec4<u8>& color) {
    return static_cast<u32>(color.r()) | (static_cast<u32>(color.g()) << 8) |
           (static_cast<u32>(color.b()) << 16) | (static_cast<u32>(color.a()) << 24);
}

/**
 * Decodifica UNA FILA DE MOSAICOS (8 lineas de texeles, todo el ancho) a 'band',
 * en RGBA8 con R en el byte bajo, linea 0 primero. Ver DecodeTexture.
 *
 * ETC1 y ETC1A4 van bloque a bloque: cada bloque de 4x4 se descomprime UNA vez
 * con DecodeETC1Block -- la misma rutina que usan el rasterizador de software y
 * su cache (Etc1BlockCache::GetTexel) -- y el alfa de ETC1A4 se saca con la
 * misma cuenta que LookupTexelInTile. El resto de formatos siguen pasando por
 * LookupTexelInTile texel a texel: son baratos y asi no hay un segundo
 * decodificador por formato que pueda divergir del de referencia.
 */
/**
 * UN MOSAICO DE 8x8 SIN COMPRIMIR, POR FORMATO (0.2.1.1). LookupTexelInTile
 * pasaba por su switch de formatos y calculaba la posicion Morton en cada
 * texel; aqui la posicion sale de una tabla y cada formato tiene su bucle.
 * Los valores son los mismos que alli (con disable_alpha a falso): las mismas
 * conversiones de Common::Color, empaquetadas como PackRGBA.
 */
template <u32 kBytes, typename Decode>
void DecodeTileTexels(const u8* tile, u32 width, u32* tile_out, Decode&& decode) {
    for (u32 y = 0; y < 8; y++) {
        u32* out = tile_out + y * width;
        const u32* morton = VideoCore::kMortonValues.data() + y * 8;
        for (u32 x = 0; x < 8; x++) {
            out[x] = decode(tile + morton[x] * kBytes);
        }
    }
}

/// Lo mismo con cuatro bits por texel: el texel m va en el byte m / 2, en el
/// medio byte alto si m es impar.
template <typename Decode>
void DecodeTileNibbles(const u8* tile, u32 width, u32* tile_out, Decode&& decode) {
    for (u32 y = 0; y < 8; y++) {
        u32* out = tile_out + y * width;
        const u32* morton = VideoCore::kMortonValues.data() + y * 8;
        for (u32 x = 0; x < 8; x++) {
            const u32 m = morton[x];
            const u8 byte = tile[m / 2];
            out[x] = decode(static_cast<u8>((m & 1) != 0 ? (byte >> 4) : (byte & 0xF)));
        }
    }
}

/**
 * LOS DE 16 BITS CON NEON (0.3.2.1). Como el RGBA8: cada fila son cuatro
 * parejas de texeles seguidas en Morton, que se cargan juntas; 'convert' saca
 * los cuatro canales de los ocho texeles en 16 bits (con las mismas cuentas
 * que Common::Color) y vst4 los deja intercalados como R,G,B,A.
 */
template <typename Convert>
void DecodeTile16Neon(const u8* tile, u32 width, u32* tile_out, Convert&& convert) {
    const u32* morton = VideoCore::kMortonValues.data();
    for (u32 y = 0; y < 8; y++, morton += 8) {
        u32 pairs[4];
        std::memcpy(&pairs[0], tile + morton[0] * 2, 4);
        std::memcpy(&pairs[1], tile + morton[2] * 2, 4);
        std::memcpy(&pairs[2], tile + morton[4] * 2, 4);
        std::memcpy(&pairs[3], tile + morton[6] * 2, 4);
        const uint16x8_t v = vreinterpretq_u16_u32(vld1q_u32(pairs));
        vst4_u8(reinterpret_cast<u8*>(tile_out + y * width), convert(v));
    }
}

/// (c << 3) | (c >> 2), (c << 2) | (c >> 4) y (c << 4) | c, en 16 bits y a 8.
inline uint8x8_t Expand5(uint16x8_t c) {
    return vmovn_u16(vorrq_u16(vshlq_n_u16(c, 3), vshrq_n_u16(c, 2)));
}
inline uint8x8_t Expand6(uint16x8_t c) {
    return vmovn_u16(vorrq_u16(vshlq_n_u16(c, 2), vshrq_n_u16(c, 4)));
}
inline uint8x8_t Expand4(uint16x8_t c) {
    return vmovn_u16(vorrq_u16(vshlq_n_u16(c, 4), c));
}

/// false si el formato no tiene bucle propio (lo hace LookupTexelInTile).
bool DecodeTileFast(const u8* tile, TexturingRegs::TextureFormat format, u32 width,
                    u32* tile_out) {
    using Format = TexturingRegs::TextureFormat;
    namespace Color = Common::Color;
    const auto gray = [](u8 i, u8 a) {
        return static_cast<u32>(i) * 0x010101u | (static_cast<u32>(a) << 24);
    };
    switch (format) {
    case Format::RGBA8: {
        /**
         * CON NEON, DE DOS EN DOS (0.3.2.1). En Morton los texeles x y x+1 de
         * una fila son m y m+1: ocho bytes seguidos. Cada fila son cuatro
         * cargas de 64 bits y dos escrituras de 128, y vrev32 da la vuelta a
         * los bytes de cada texel: el mismo p[3] | p[2] << 8 | p[1] << 16 |
         * p[0] << 24 que el bucle de texel en texel, que eran cuatro cargas de
         * un byte por texel (New Super Mario Bros. 2: "tx" 60-110 ms por
         * fotograma en sus texturas que cambian).
         */
        const u32* morton = VideoCore::kMortonValues.data();
        for (u32 y = 0; y < 8; y++, morton += 8) {
            const uint32x2_t p0 = vld1_u32(reinterpret_cast<const u32*>(tile + morton[0] * 4));
            const uint32x2_t p1 = vld1_u32(reinterpret_cast<const u32*>(tile + morton[2] * 4));
            const uint32x2_t p2 = vld1_u32(reinterpret_cast<const u32*>(tile + morton[4] * 4));
            const uint32x2_t p3 = vld1_u32(reinterpret_cast<const u32*>(tile + morton[6] * 4));
            u8* out = reinterpret_cast<u8*>(tile_out + y * width);
            vst1q_u8(out, vrev32q_u8(vreinterpretq_u8_u32(vcombine_u32(p0, p1))));
            vst1q_u8(out + 16, vrev32q_u8(vreinterpretq_u8_u32(vcombine_u32(p2, p3))));
        }
        return true;
    }
    case Format::RGB8:
        DecodeTileTexels<3>(tile, width, tile_out, [](const u8* p) {
            return static_cast<u32>(p[2]) | (static_cast<u32>(p[1]) << 8) |
                   (static_cast<u32>(p[0]) << 16) | 0xFF000000u;
        });
        return true;
    case Format::RGB5A1:
        DecodeTile16Neon(tile, width, tile_out, [](uint16x8_t v) {
            const uint16x8_t mask5 = vdupq_n_u16(0x1F);
            uint8x8x4_t out;
            out.val[0] = Expand5(vshrq_n_u16(v, 11));
            out.val[1] = Expand5(vandq_u16(vshrq_n_u16(v, 6), mask5));
            out.val[2] = Expand5(vandq_u16(vshrq_n_u16(v, 1), mask5));
            out.val[3] = vmovn_u16(vmulq_n_u16(vandq_u16(v, vdupq_n_u16(1)), 255));
            return out;
        });
        return true;
    case Format::RGB565:
        DecodeTile16Neon(tile, width, tile_out, [](uint16x8_t v) {
            uint8x8x4_t out;
            out.val[0] = Expand5(vshrq_n_u16(v, 11));
            out.val[1] = Expand6(vandq_u16(vshrq_n_u16(v, 5), vdupq_n_u16(0x3F)));
            out.val[2] = Expand5(vandq_u16(v, vdupq_n_u16(0x1F)));
            out.val[3] = vdup_n_u8(255);
            return out;
        });
        return true;
    case Format::RGBA4:
        DecodeTile16Neon(tile, width, tile_out, [](uint16x8_t v) {
            const uint16x8_t mask4 = vdupq_n_u16(0xF);
            uint8x8x4_t out;
            out.val[0] = Expand4(vshrq_n_u16(v, 12));
            out.val[1] = Expand4(vandq_u16(vshrq_n_u16(v, 8), mask4));
            out.val[2] = Expand4(vandq_u16(vshrq_n_u16(v, 4), mask4));
            out.val[3] = Expand4(vandq_u16(v, mask4));
            return out;
        });
        return true;
    case Format::IA8:
        DecodeTileTexels<2>(tile, width, tile_out, [&gray](const u8* p) { return gray(p[1], p[0]); });
        return true;
    case Format::RG8:
        DecodeTileTexels<2>(tile, width, tile_out, [](const u8* p) {
            return static_cast<u32>(p[1]) | (static_cast<u32>(p[0]) << 8) | 0xFF000000u;
        });
        return true;
    case Format::I8:
        DecodeTileTexels<1>(tile, width, tile_out, [&gray](const u8* p) { return gray(*p, 255); });
        return true;
    case Format::A8:
        DecodeTileTexels<1>(tile, width, tile_out,
                            [](const u8* p) { return static_cast<u32>(*p) << 24; });
        return true;
    case Format::IA4:
        DecodeTileTexels<1>(tile, width, tile_out, [&gray](const u8* p) {
            return gray(Color::Convert4To8(static_cast<u8>((*p & 0xF0) >> 4)),
                        Color::Convert4To8(static_cast<u8>(*p & 0xF)));
        });
        return true;
    case Format::I4:
        DecodeTileNibbles(tile, width, tile_out,
                          [&gray](u8 nibble) { return gray(Color::Convert4To8(nibble), 255); });
        return true;
    case Format::A4:
        DecodeTileNibbles(tile, width, tile_out, [](u8 nibble) {
            return static_cast<u32>(Color::Convert4To8(nibble)) << 24;
        });
        return true;
    default:
        return false;
    }
}

void DecodeTileRow(const u8* row_source, const Pica::Texture::TextureInfo& info, u32 width,
                   u32* band) {
    const bool etc1 = info.format == TexturingRegs::TextureFormat::ETC1;
    const bool etc1a4 = info.format == TexturingRegs::TextureFormat::ETC1A4;
    for (u32 tile_x = 0; tile_x < width / 8; tile_x++) {
        const u8* tile = row_source + tile_x * info.tile_size;
        u32* tile_out = band + tile_x * 8;
        if (etc1 || etc1a4) {
            const std::size_t subtile_size = etc1a4 ? 16 : 8;
            // Subbloques en el orden de LookupTexelInTile: indice = sx + 2*sy.
            for (u32 sub = 0; sub < 4; sub++) {
                const u32 sub_x = (sub & 1) * 4;
                const u32 sub_y = (sub >> 1) * 4;
                const u8* ptr = tile + sub * subtile_size;
                u64 packed_alpha = 0;
                if (etc1a4) {
                    std::memcpy(&packed_alpha, ptr, sizeof(u64));
                    ptr += sizeof(u64);
                }
                u64 block;
                std::memcpy(&block, ptr, sizeof(u64));
                u32 texels[16];
                Pica::Texture::DecodeETC1Block(block, texels);
                for (u32 y = 0; y < 4; y++) {
                    u32* out = tile_out + (sub_y + y) * width + sub_x;
                    for (u32 x = 0; x < 4; x++) {
                        u32 texel = texels[x + y * 4];
                        if (etc1a4) {
                            const u8 alpha = Common::Color::Convert4To8(
                                static_cast<u8>((packed_alpha >> (4 * (x * 4 + y))) & 0xF));
                            texel = (texel & 0x00FFFFFFu) | (static_cast<u32>(alpha) << 24);
                        }
                        out[x] = texel;
                    }
                }
            }
        } else if (!DecodeTileFast(tile, info.format, width, tile_out)) {
            for (u32 y = 0; y < 8; y++) {
                u32* out = tile_out + y * width;
                for (u32 x = 0; x < 8; x++) {
                    out[x] = PackRGBA(Pica::Texture::LookupTexelInTile(tile, x, y, info, false));
                }
            }
        }
    }
}

} // Anonymous namespace

std::array<std::atomic<u32>, TextureCache::kRejectCount> TextureCache::rejects{};
std::atomic<u32> TextureCache::partial_decodes{0};
std::atomic<u32> TextureCache::fill_skips{0};

TextureCache::TextureCache() = default;
TextureCache::~TextureCache() = default;

void TextureCache::Retire(Entry& entry) {
    if (!entry.valid) {
        return;
    }
    bytes_used -= entry.buffer.Size();
    index.erase(entry.key);
    entry.valid = false;
    entry.key = 0;
    entry.address = 0;
    entry.span = 0;
    entry.stale = false;
    entry.dirty_begin = 0;
    entry.dirty_end = 0;
    entry.fill_tag = 0;
    entry.pending_fill_tag = 0;
    entry.band_hashes.clear();
    shadow_bytes -= static_cast<u32>(entry.shadow.size() * sizeof(u32));
    std::vector<u32>().swap(entry.shadow);
    if (entry.buffer.Valid()) {
        // A la lista de espera, NO al suelo: puede haber un dibujado apuntado
        // que todavia no ha leido estos pixeles. Ver ReleaseRetired.
        retired.push_back(std::move(entry.buffer));
    }
    entry.buffer = Allocation{};
}

void TextureCache::ReleaseRetired() {
    // Destruir los Allocation es lo que desmapea y libera. El vector se queda
    // con su capacidad para no volver a pedir memoria en cada escena.
    retired.clear();
    sealed.clear();
    spare.clear();
    spare_bytes = 0;
}

void TextureCache::SealRetired(u32 fence) {
    for (auto& buffer : retired) {
        sealed.emplace_back(fence, std::move(buffer));
    }
    retired.clear();
}

void TextureCache::ReleaseUpTo(u32 completed_fence) {
    std::erase_if(sealed, [this, completed_fence](auto& item) {
        if (static_cast<s32>(completed_fence - item.first) < 0) {
            return false;
        }
        KeepSpare(std::move(item.second));
        return true;
    });
}

void TextureCache::KeepSpare(Allocation&& buffer) {
    if (buffer.Size() < kSpareMinBytes || buffer.Size() > kMaxSpareBytes) {
        return;
    }
    while (!spare.empty() &&
           (spare.size() >= kMaxSpares || spare_bytes + buffer.Size() > kMaxSpareBytes)) {
        spare_bytes -= spare.front().Size();
        spare.erase(spare.begin());
    }
    spare_bytes += buffer.Size();
    spare.push_back(std::move(buffer));
}

Allocation TextureCache::TakeBuffer(u32 needed) {
    const Pool first = needed <= kSmallTexture ? Pool::Host : Pool::Cdram;
    for (auto it = spare.begin(); it != spare.end(); ++it) {
        // El mismo pool y sin desperdiciar mas de un grano de CDRAM.
        if (it->Where() == first && it->Size() >= needed && it->Size() - needed < kSpareMinBytes) {
            Allocation buffer = std::move(*it);
            spare_bytes -= buffer.Size();
            spare.erase(it);
            return buffer;
        }
    }
    Allocation buffer = Allocate(first, needed);
    if (!buffer.Valid() && !spare.empty()) {
        // Lo guardado puede ser justo lo que falta.
        spare.clear();
        spare_bytes = 0;
        buffer = Allocate(first, needed);
    }
    if (!buffer.Valid()) {
        buffer = Allocate(first == Pool::Host ? Pool::Cdram : Pool::Host, needed);
    }
    return buffer;
}

void TextureCache::Clear() {
    for (auto& entry : entries) {
        Retire(entry);
    }
}

const SceGxmTexture* TextureCache::ConstantTexture(bool opaque) {
    if (!constant_texels.Valid()) {
        if (constant_failed) {
            return nullptr;
        }
        // 8x8 y no 1x1: el paso de una textura lineal va en multiplos de 8
        // texeles. La transparente en los primeros 256 bytes, la negra detras.
        Allocation texels = Allocate(Pool::Host, 4096);
        if (!texels.Valid()) {
            constant_failed = true;
            return nullptr;
        }
        auto* words = static_cast<u32*>(texels.Data());
        for (u32 i = 0; i < 64; i++) {
            words[i] = 0;
            words[64 + i] = 0xFF000000u;
        }
        for (u32 i = 0; i < 2; i++) {
            SceGxmTexture& texture = constant_textures[i];
            if (sceGxmTextureInitLinear(&texture, words + 64 * i,
                                        SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR, 8, 8, 1) < 0) {
                constant_failed = true;
                return nullptr;
            }
            sceGxmTextureSetMinFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
            sceGxmTextureSetMagFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
            sceGxmTextureSetMipFilter(&texture, SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
            sceGxmTextureSetUAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_REPEAT);
            sceGxmTextureSetVAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_REPEAT);
        }
        constant_texels = std::move(texels);
    }
    return &constant_textures[opaque ? 1 : 0];
}

u32 TextureCache::InvalidateRange(PAddr addr, u32 size) {
    const PAddr end = addr + size;
    u32 marked = 0;
    /**
     * SE MARCAN, NO SE TIRAN (0.1.0.43).
     *
     * Este aviso llega cuando el juego vacia o invalida la cache de datos sobre
     * un tramo -- que es como avisa de que PUEDE haber escrito ahi --, y hasta
     * 0.1.0.42 tiraba cualquier textura que lo pisara. Pokemon, como muchos
     * juegos, vacia tramos grandes en cada fotograma aunque sus texturas no
     * cambien: en la cinematica de Rubi Omega salian ~10 texturas decodificadas
     * POR FOTOGRAMA, 67 ms de los 416, siempre para producir la misma imagen.
     *
     * Ahora la entrada queda "sospechosa" y Get compara el hash de los bytes
     * actuales con el de los que se decodificaron. Iguales: la imagen
     * decodificada es la misma, porque la decodificacion es una funcion pura de
     * esos bytes (y del formato y tamano, que van en la clave). Distintos: se
     * decodifica como antes. Lo que cambia es el coste de comprobarlo -- un
     * hash que corre a la velocidad de la memoria -- frente a decodificar.
     *
     * La memoria de GPU de la entrada no se toca aqui: un dibujado pendiente de
     * la escena abierta puede estar leyendola, y si la textura resulta haber
     * cambiado se retira por el camino de siempre (Retire), que la suelta solo
     * cuando la GPU ha terminado.
     */
    for (auto& entry : entries) {
        if (entry.valid && entry.address < end && addr < entry.address + entry.span) {
            marked += MarkStale(entry, addr, end) ? 1u : 0u;
            entry.fill_tag = 0;
            entry.pending_fill_tag = 0;
        }
    }
    return marked;
}

u32 TextureCache::InvalidateFill(PAddr addr, u32 size, u32 texel, u32 bpp) {
    const PAddr end = addr + size;
    const u32 mask = bpp >= 4 ? 0xFFFFFFFFu : (1u << (bpp * 8u)) - 1u;
    u32 marked = 0;
    for (auto& entry : entries) {
        if (!entry.valid || entry.address >= end || addr >= entry.address + entry.span) {
            continue;
        }
        const bool covered = entry.address >= addr && entry.address + entry.span <= end;
        // Los bytes del patron y su fase al principio de la entrada.
        const u64 tag = covered ? (1ull << 63) | (static_cast<u64>(texel & mask) << 8) |
                                      (static_cast<u64>(bpp) << 4) | ((entry.address - addr) % bpp)
                                : 0;
        if (covered && !entry.stale && entry.fill_tag == tag) {
            fill_skips.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        marked += MarkStale(entry, addr, end) ? 1u : 0u;
        entry.fill_tag = 0;
        entry.pending_fill_tag = tag;
    }
    return marked;
}

bool TextureCache::MarkStale(Entry& entry, PAddr addr, PAddr end) {
    const u32 lo = addr > entry.address ? addr - entry.address : 0;
    const u32 hi = std::min<u32>(end - entry.address, entry.span);
    if (entry.stale) {
        entry.dirty_begin = std::min(entry.dirty_begin, lo);
        entry.dirty_end = std::max(entry.dirty_end, hi);
        return false;
    }
    entry.dirty_begin = lo;
    entry.dirty_end = hi;
    entry.stale = true;
    return true;
}

TextureCache::Entry& TextureCache::MakeRoom(u32 needed) {
    // 1. Un hueco libre, si queda alguno.
    Entry* slot = nullptr;
    for (auto& entry : entries) {
        if (!entry.valid) {
            slot = &entry;
            break;
        }
    }
    // 2. Si no, la menos usada recientemente. El reloj es logico (sube en cada
    //    consulta), asi que el last_use mas bajo es el que lleva mas tiempo sin
    //    que nadie lo pida.
    if (slot == nullptr) {
        slot = &entries[0];
        for (auto& entry : entries) {
            if (entry.last_use < slot->last_use) {
                slot = &entry;
            }
        }
        Common::FrameStats::texture_evictions.fetch_add(1, std::memory_order_relaxed);
        Retire(*slot);
    }

    // 3. Y sitio en el presupuesto. Puede hacer falta expulsar varias: una
    //    textura grande vale por muchas pequenas. La entrada que acabamos de
    //    dejar libre ya no es valida, asi que este bucle no la vuelve a coger.
    const u32 estimate = EstimateBlockBytes(needed);
    while (bytes_used + estimate > kMemoryBudget) {
        Entry* oldest = nullptr;
        for (auto& entry : entries) {
            if (!entry.valid) {
                continue;
            }
            if (oldest == nullptr || entry.last_use < oldest->last_use) {
                oldest = &entry;
            }
        }
        if (oldest == nullptr) {
            // No queda nada que expulsar y aun asi no cabe: la textura es mas
            // grande que el presupuesto entero. Se intenta igualmente; si la
            // reserva falla, el lote cae a software, que es lo correcto.
            break;
        }
        Common::FrameStats::texture_evictions.fetch_add(1, std::memory_order_relaxed);
        Retire(*oldest);
    }
    return *slot;
}

bool TextureCache::ApplyUnitSampler(u32 unit, const Pica::RegsInternal& regs,
                                    SceGxmTexture& texture) {
    const auto& texturing = regs.texturing;
    const TexturingRegs::TextureConfig* config = nullptr;
    switch (unit) {
    case 0:
        config = &texturing.texture0;
        break;
    case 1:
        config = &texturing.texture1;
        break;
    case 2:
        config = &texturing.texture2;
        break;
    default:
        return false;
    }
    SceGxmTextureAddrMode wrap_s{};
    SceGxmTextureAddrMode wrap_t{};
    if (!MapWrap(config->wrap_s.Value(), &wrap_s) || !MapWrap(config->wrap_t.Value(), &wrap_t)) {
        return false;
    }
    const bool min_linear =
        config->min_filter == TexturingRegs::TextureConfig::TextureFilter::Linear;
    const bool mag_linear =
        config->mag_filter == TexturingRegs::TextureConfig::TextureFilter::Linear;
    sceGxmTextureSetMinFilter(&texture, min_linear ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                                   : SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMagFilter(&texture, mag_linear ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                                   : SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMipFilter(&texture, SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
    sceGxmTextureSetMipmapCount(&texture, 1);
    sceGxmTextureSetUAddrMode(&texture, wrap_s);
    sceGxmTextureSetVAddrMode(&texture, wrap_t);
    return true;
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
    /**
     * LAS UNIDADES SIN TEXTURA, EN LA GPU (0.3.1.4). El rasterizador de
     * software (TextureColor) deja (0,0,0,0) en una unidad apagada y en la 0
     * con tipo "Disabled" (su array de colores arranca a cero y esas ramas no
     * lo tocan), y pone (0,0,0,255) en una con la direccion a cero. El shader
     * de fragmentos declara el sampler y muestrea igual, y hasta ahora el lote
     * entero iba a software, con la superficie bajada a la memoria y vuelta a
     * subir ("estado" y "lote soft" en crash.txt). Una textura con todos sus
     * texeles de ese color da lo mismo en cualquier coordenada y con cualquier
     * filtro.
     */
    if (!enabled || config->address == 0 ||
        (unit == 0 && config->type == TexturingRegs::TextureConfig::Disabled)) {
        // En el orden de TextureColor: la direccion se mira antes que el tipo.
        const SceGxmTexture* constant = ConstantTexture(enabled && config->address == 0);
        if (constant != nullptr) {
            rejects[kServedDisabled].fetch_add(1, std::memory_order_relaxed);
        }
        return constant;
    }
    // La unidad 0 tiene tipo (2D, proyeccion, cubo, sombra). El resto no.
    // Proyectada es una textura 2D normal: lo que cambia es la coordenada
    // (dividida por w en el shader), no la imagen. Cubo y sombra no.
    if (unit == 0 && config->type != TexturingRegs::TextureConfig::Texture2D &&
        config->type != TexturingRegs::TextureConfig::Projection2D) {
        rejects[kRejectType].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    /**
     * LOS MIPMAPS YA NO RECHAZAN, Y QUEDARSE EN EL NIVEL 0 ES LO CORRECTO.
     *
     * Aqui habia un rechazo por config->mip_filter distinto de cero. Parecia
     * prudente y costaba lotes enteros para nada: el rasterizador de software
     * de Azahar NO IMPLEMENTA mipmaps. Su TextureInfo no tiene nivel, su
     * LookupTexture lee siempre de la direccion base, y en la linea donde le
     * tocaria filtrar hay escrito un "TODO: Apply the min and mag filters". La
     * referencia de correccion muestrea el nivel 0 y ya esta.
     *
     * Construir aqui una cadena de mipmaps de verdad -- reducir el nivel 0, o
     * leer los niveles que el juego haya dejado en memoria y montarlos con
     * sceGxmTextureSetMipmapCount -- SEPARARIA el camino de GPU del de
     * software en vez de acercarlo: lo lejano saldria suavizado en uno y con
     * dientes en el otro, y no hay forma de compararlos desde la consola. Asi
     * que un solo nivel, y el filtro de mipmap apagado explicitamente mas
     * abajo.
     */

    SceGxmTextureAddrMode wrap_s{};
    SceGxmTextureAddrMode wrap_t{};
    if (!MapWrap(config->wrap_s.Value(), &wrap_s) || !MapWrap(config->wrap_t.Value(), &wrap_t)) {
        rejects[kRejectWrap].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    const bool min_linear =
        config->min_filter == TexturingRegs::TextureConfig::TextureFilter::Linear;
    const bool mag_linear =
        config->mag_filter == TexturingRegs::TextureConfig::TextureFilter::Linear;

    /**
     * La clave, en campos de bits que NO se pisan.
     *
     * La version anterior mezclaba los campos con XOR y desplazamientos
     * solapados (el formato en el bit 32, el repetido en el 22, el filtro en el
     * 28...), asi que dos texturas distintas podian dar la misma clave y
     * devolver la imagen equivocada. Con tres entradas fijas casi nunca pasaba;
     * con cuarenta y ocho compartidas seria cuestion de tiempo. Aqui cada cosa
     * tiene su tramo y no sobra ni un bit:
     *
     *   0..27  direccion (el registro, 28 bits; la fisica es esta por ocho)
     *   28..31 formato       32..42 ancho        43..53 alto
     *   54..56 repetido S    57..59 repetido T   60 filtro min   61 filtro mag
     */
    const u64 key = (static_cast<u64>(config->address.Value()) & 0xFFFFFFFull) |
                    (static_cast<u64>(format) & 0xFull) << 28 |
                    (static_cast<u64>(config->width) & 0x7FFull) << 32 |
                    (static_cast<u64>(config->height) & 0x7FFull) << 43 |
                    (static_cast<u64>(wrap_s) & 0x7ull) << 54 |
                    (static_cast<u64>(wrap_t) & 0x7ull) << 57 |
                    (static_cast<u64>(min_linear) << 60) | (static_cast<u64>(mag_linear) << 61);

    clock++;
    bool want_shadow = false;
    if (const auto found = index.find(key); found != index.end()) {
        Entry& entry = entries[found->second];
        if (entry.stale) {
            // Sospechosa (ver InvalidateRange): se revalida por contenido. La
            // direccion y el tramo son los de la clave, asi que son los mismos
            // bytes de los que salio source_hash.
            const u8* bytes = memory.GetPhysicalPointer(entry.address);
            const u8* last =
                bytes != nullptr ? memory.GetPhysicalPointer(entry.address + entry.span - 1)
                                 : nullptr;
            bool same = false;
            bool partial = false;
            changed_bands.clear();
            if (bytes != nullptr && last == bytes + entry.span - 1) {
                const unsigned long long rehash_begin = Common::VitaMicros();
                if (!entry.band_hashes.empty()) {
                    // Solo las filas que pisa el tramo avisado.
                    const u32 bands = static_cast<u32>(entry.band_hashes.size());
                    const u32 first = std::min(entry.dirty_begin / entry.stride, bands);
                    const u32 end_band =
                        std::min((entry.dirty_end + entry.stride - 1) / entry.stride, bands);
                    for (u32 b = first; b < end_band; b++) {
                        const u64 hash = Common::ComputeHash64(
                            bytes + static_cast<std::size_t>(b) * entry.stride, entry.stride);
                        if (hash != entry.band_hashes[b]) {
                            changed_bands.push_back(b);
                        }
                    }
                    Common::FrameStats::texture_rehash_bytes.fetch_add(
                        (end_band - first) * entry.stride, std::memory_order_relaxed);
                    same = changed_bands.empty();
                    partial = !same && !entry.shadow.empty();
                } else {
                    same = Common::ComputeHash64(bytes, entry.span) == entry.source_hash;
                    Common::FrameStats::texture_rehash_bytes.fetch_add(
                        entry.span, std::memory_order_relaxed);
                }
                Common::FrameStats::Add(Common::FrameStats::texture_rehash_us, rehash_begin);
            }
            if (same) {
                entry.stale = false;
                Common::FrameStats::texture_reuses.fetch_add(1, std::memory_order_relaxed);
            } else {
                Common::FrameStats::texture_changed.fetch_add(1, std::memory_order_relaxed);
                const Pica::Texture::TextureInfo info =
                    Pica::Texture::TextureInfo::FromPicaRegister(*config, format);
                if (!partial || !RedecodeBands(entry, bytes, info, changed_bands, min_linear,
                                               mag_linear, wrap_s, wrap_t)) {
                    // Ha cambiado y no se puede por filas: se retira y se
                    // decodifica entera abajo, ya con copia en RAM.
                    want_shadow = true;
                    Retire(entry);
                }
            }
            // Revalidada: si la marco un relleno que la cubria entera, sus
            // bytes son ahora ese patron.
            if (entry.valid && !entry.stale) {
                entry.fill_tag = entry.pending_fill_tag;
            }
            entry.pending_fill_tag = 0;
        }
        if (entry.valid) {
            entry.last_use = clock;
            return &entry.texture;
        }
    }

    const Pica::Texture::TextureInfo info =
        Pica::Texture::TextureInfo::FromPicaRegister(*config, format);
    const u32 width = info.width;
    const u32 height = info.height;
    if (width == 0 || height == 0 || width > 1024 || height > 1024) {
        rejects[kRejectSize].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    const PAddr address = info.physical_address;
    const u8* source = memory.GetPhysicalPointer(address);
    if (source == nullptr) {
        rejects[kRejectMemory].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    // Filas de mosaicos redondeando hacia ARRIBA. Con height/8, un alto que no
    // fuera multiplo de ocho dejaba la ultima fila fuera del tramo comprobado y
    // la decodificacion la leia igual.
    const u32 span = static_cast<u32>(info.stride) * ((height + 7) / 8);
    if (span == 0) {
        rejects[kRejectSize].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    // El tramo entero tiene que estar mapeado y contiguo: la decodificacion
    // recorre todos los texeles, no solo los que el juego llegue a muestrear.
    const u8* last = memory.GetPhysicalPointer(address + span - 1);
    if (last != source + span - 1) {
        rejects[kRejectMemory].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    const u32 needed = width * height * 4;
    Entry& entry = MakeRoom(needed);

    // CDRAM primero: la CPU la escribe una vez y la GPU la lee en cada
    // dibujado, que es justo el caso que pide memoria dedicada. Las pequenas,
    // a memoria normal (ver kSmallTexture).
    Allocation buffer = TakeBuffer(needed);
    if (!buffer.Valid()) {
        LOG_ERROR(Render, "GXM: sin memoria para decodificar una textura de {}x{}", width, height);
        rejects[kRejectGpuMemory].fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    /**
     * LA FILA SE ESCRIBE VOLTEADA, Y NO ES UN CAPRICHO.
     *
     * Las texturas de la PICA se guardan de ABAJO ARRIBA. El rasterizador de
     * software lo compensa al muestrear: antes de leer el texel hace
     *
     *     t = height - 1 - GetWrappedTexCoord(...)
     *
     * (sw_rasterizer.cpp, con el comentario "Textures are laid out from bottom
     * to top, hence we invert the t coordinate"). Aqui se decodificaba la fila
     * y en la fila y, sin voltear, asi que la GPU muestreaba la imagen al reves
     * que el software: todo lo texturado salia espejado de arriba abajo. En una
     * caja o un fondo casi no se nota; en un texto se ve a la primera, y asi es
     * como aparecio -- el marco del dialogo bien puesto y las letras del reves.
     *
     * Se voltea al DECODIFICAR y no en el shader a proposito: se paga una vez
     * por textura en vez de una por pixel muestreado, y deja la coordenada v
     * tal cual llega del vertice, que es una cosa menos que cuadrar.
     */
    /**
     * Un texel = UNA escritura de 32 bits, con la cache de bloques ETC1.
     *
     * Antes eran cuatro escrituras de un byte por texel en memoria de video,
     * que la CPU ve sin cache (combinando escrituras): cuatro transacciones
     * donde basta una. Y sin cache ETC1, cada texel volvia a descomprimir su
     * bloque de 4x4 entero. Los bytes que acaban en memoria son los mismos:
     * R en el byte bajo, A en el alto, que en little-endian es R,G,B,A.
     */
    if (!etc1_cache) {
        etc1_cache = std::make_unique<Pica::Texture::Etc1BlockCache>();
    }
    const unsigned long long decode_begin = Common::VitaMicros();
    auto* dest = static_cast<u32*>(buffer.Data());
    entry.band_hashes.clear();
    const bool tiled_rows = width % 8 == 0 && height % 8 == 0;
    if (tiled_rows && want_shadow && needed <= kMaxShadowTexture &&
        shadow_bytes + needed <= kMaxShadowBytes) {
        entry.shadow.resize(static_cast<std::size_t>(width) * height);
        shadow_bytes += needed;
    }
    if (tiled_rows) {
        /**
         * EL CAMINO NORMAL: por filas de mosaicos, con una banda en RAM.
         *
         * Se decodifica una fila de mosaicos entera (8 lineas) en 'band', que
         * vive en memoria con cache, y despues se copian sus 8 lineas a la
         * textura -- memoria de video, que la CPU escribe sin cache -- con un
         * memcpy por linea. Asi la memoria de video solo recibe escrituras en
         * orden, que es lo que su combinacion de escrituras hace bien, en vez
         * de saltos de mosaico en mosaico. Los texeles son los mismos que por
         * el camino de abajo; solo cambia el orden en que se calculan.
         */
        band.resize(static_cast<std::size_t>(width) * 8);
        entry.band_hashes.resize(height / 8);
        for (u32 tile_y = 0; tile_y < height / 8; tile_y++) {
            const u8* row_source = source + tile_y * info.stride;
            entry.band_hashes[tile_y] = Common::ComputeHash64(row_source, info.stride);
            DecodeTileRow(row_source, info, width, band.data());
            for (u32 line = 0; line < 8; line++) {
                // Volteo vertical: ver el comentario de arriba.
                const u32 dest_y = height - 1 - (tile_y * 8 + line);
                std::memcpy(dest + static_cast<std::size_t>(dest_y) * width,
                            band.data() + static_cast<std::size_t>(line) * width,
                            static_cast<std::size_t>(width) * 4);
                if (!entry.shadow.empty()) {
                    std::memcpy(entry.shadow.data() + static_cast<std::size_t>(dest_y) * width,
                                band.data() + static_cast<std::size_t>(line) * width,
                                static_cast<std::size_t>(width) * 4);
                }
            }
        }
    } else {
        // Tamanos que no son multiplo del mosaico: texel a texel, como antes.
        for (u32 y = 0; y < height; y++) {
            const u32 dest_y = height - 1 - y;
            u32* row = dest + static_cast<std::size_t>(dest_y) * width;
            for (u32 x = 0; x < width; x++) {
                row[x] = PackRGBA(
                    Pica::Texture::LookupTexture(source, x, y, info, false, etc1_cache.get()));
            }
        }
    }
    Common::FrameStats::texture_decodes.fetch_add(1, std::memory_order_relaxed);
    Common::FrameStats::Add(Common::FrameStats::texture_decode_us, decode_begin);
    {
        // Los que cuestan, con su formato (0.3.2.1): de New Super Mario Bros. 2
        // solo se sabia que eran ~100 ms cada uno.
        const unsigned long long spent = Common::VitaMicros() - decode_begin;
        static u32 slow_notes = 0;
        if (spent > 5000 && slow_notes < 12) {
            slow_notes++;
            Common::VitaNote("gxm textura",
                             fmt::format("decodificada entera en {} ms: {}x{} formato {} en "
                                         "{:#010x}{}",
                                         spent / 1000, width, height, static_cast<u32>(format),
                                         address, want_shadow ? " (con copia en RAM)" : "")
                                 .c_str());
        }
    }

    entry.buffer = std::move(buffer);
    SetupTexture(entry, width, height, min_linear, mag_linear, wrap_s, wrap_t);
    if (!entry.buffer.Valid()) {
        shadow_bytes -= static_cast<u32>(entry.shadow.size() * sizeof(u32));
        std::vector<u32>().swap(entry.shadow);
        entry.band_hashes.clear();
        return nullptr;
    }

    entry.valid = true;
    entry.key = key;
    entry.address = address;
    entry.span = span;
    entry.stride = static_cast<u32>(info.stride);
    entry.last_use = clock;
    // Con hashes por fila no hace falta el del tramo entero (0.3.1.8).
    entry.source_hash = entry.band_hashes.empty() ? Common::ComputeHash64(source, span) : 0;
    entry.stale = false;
    bytes_used += entry.buffer.Size();
    index[key] = static_cast<u32>(&entry - entries.data());
    return &entry.texture;
}

void TextureCache::SetupTexture(Entry& entry, u32 width, u32 height, bool min_linear,
                                bool mag_linear, SceGxmTextureAddrMode wrap_s,
                                SceGxmTextureAddrMode wrap_t) {
    // Los bytes van en orden R,G,B,A, que es lo que describe el nombre ABGR de
    // GXM (los nombres listan los canales del mas significativo al menos).
    if (sceGxmTextureInitLinear(&entry.texture, entry.buffer.Data(),
                                SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR, width, height, 1) < 0) {
        retired.push_back(std::move(entry.buffer));
        entry.buffer = Allocation{};
        return;
    }
    sceGxmTextureSetMinFilter(&entry.texture, min_linear ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                                         : SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMagFilter(&entry.texture, mag_linear ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                                         : SCE_GXM_TEXTURE_FILTER_POINT);
    // Un solo nivel, y que quede dicho: una entrada reutilizada podria traer el
    // filtro de mipmap encendido de su vida anterior, y entonces el chip iria a
    // buscar niveles que no existen.
    sceGxmTextureSetMipFilter(&entry.texture, SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
    sceGxmTextureSetMipmapCount(&entry.texture, 1);
    sceGxmTextureSetUAddrMode(&entry.texture, wrap_s);
    sceGxmTextureSetVAddrMode(&entry.texture, wrap_t);
}

/**
 * SOLO LAS FILAS QUE CAMBIAN (0.3.1.8). New Super Mario Bros. 2 escribe unos
 * cientos de bytes de una textura de ~1 MB varias veces por segundo, y cada vez
 * se decodificaba entera: ~105 ms, 6 por ventana, "tx" 60-110 ms por fotograma.
 * Ahora las filas cambiadas se decodifican sobre la copia en RAM y la imagen
 * entera pasa de ahi a un bloque nuevo con un memcpy secuencial (~1-2 ms). El
 * bloque viejo va a la lista de espera, como en Retire: un dibujado de la
 * escena abierta puede estar leyendolo.
 */
bool TextureCache::RedecodeBands(Entry& entry, const u8* source,
                                 const Pica::Texture::TextureInfo& info,
                                 const std::vector<u32>& changed, bool min_linear,
                                 bool mag_linear, SceGxmTextureAddrMode wrap_s,
                                 SceGxmTextureAddrMode wrap_t) {
    const u32 width = info.width;
    const u32 height = info.height;
    const u32 needed = width * height * 4;
    if (entry.shadow.size() * sizeof(u32) != needed || entry.buffer.Size() < needed) {
        return false;
    }
    Allocation buffer = TakeBuffer(needed);
    if (!buffer.Valid()) {
        return false;
    }
    if (!etc1_cache) {
        etc1_cache = std::make_unique<Pica::Texture::Etc1BlockCache>();
    }
    const unsigned long long decode_begin = Common::VitaMicros();
    band.resize(static_cast<std::size_t>(width) * 8);
    for (const u32 tile_y : changed) {
        const u8* row_source = source + static_cast<std::size_t>(tile_y) * info.stride;
        entry.band_hashes[tile_y] = Common::ComputeHash64(row_source, info.stride);
        DecodeTileRow(row_source, info, width, band.data());
        for (u32 line = 0; line < 8; line++) {
            const u32 dest_y = height - 1 - (tile_y * 8 + line);
            std::memcpy(entry.shadow.data() + static_cast<std::size_t>(dest_y) * width,
                        band.data() + static_cast<std::size_t>(line) * width,
                        static_cast<std::size_t>(width) * 4);
        }
    }
    std::memcpy(buffer.Data(), entry.shadow.data(), needed);
    Common::FrameStats::texture_decodes.fetch_add(1, std::memory_order_relaxed);
    partial_decodes.fetch_add(1, std::memory_order_relaxed);
    Common::FrameStats::Add(Common::FrameStats::texture_decode_us, decode_begin);

    const u32 new_size = buffer.Size();
    bytes_used -= entry.buffer.Size();
    retired.push_back(std::move(entry.buffer));
    entry.buffer = std::move(buffer);
    SetupTexture(entry, width, height, min_linear, mag_linear, wrap_s, wrap_t);
    if (!entry.buffer.Valid()) {
        Retire(entry);
        return true;
    }
    bytes_used += new_size;
    entry.stale = false;
    entry.dirty_begin = 0;
    entry.dirty_end = 0;
    return true;
}

} // namespace Gxm
