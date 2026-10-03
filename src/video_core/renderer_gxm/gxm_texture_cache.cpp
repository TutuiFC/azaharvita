// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/gxm_texture_cache.h"

#include <cstring>
#include <utility>
#include "common/color.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/vita_diag.h"
#include "core/memory.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/texture/etc1.h"
#include "video_core/texture/texture_decode.h"

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
        } else {
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
}

void TextureCache::SealRetired(u32 fence) {
    for (auto& buffer : retired) {
        sealed.emplace_back(fence, std::move(buffer));
    }
    retired.clear();
}

void TextureCache::ReleaseUpTo(u32 completed_fence) {
    std::erase_if(sealed, [completed_fence](const auto& item) {
        return static_cast<s32>(completed_fence - item.first) >= 0;
    });
}

void TextureCache::Clear() {
    for (auto& entry : entries) {
        Retire(entry);
    }
}

void TextureCache::InvalidateRange(PAddr addr, u32 size) {
    const PAddr end = addr + size;
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
            entry.stale = true;
        }
    }
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
        // Proyectada es una textura 2D normal: lo que cambia es la coordenada
        // (dividida por w en el shader), no la imagen. Cubo y sombra no.
        if (config->type != TexturingRegs::TextureConfig::Texture2D &&
            config->type != TexturingRegs::TextureConfig::Projection2D) {
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
        // Una unidad apagada vale (0,0,0,0) en el rasterizador de software: su
        // array de colores arranca a cero y la rama de la unidad apagada no lo
        // toca. El generador de shaders todavia no reproduce eso -- declara el
        // sampler y muestrea igual -- asi que mejor caer a software que pintar
        // otra cosa. Arreglarlo pide que la configuracion del shader lleve que
        // unidades estan encendidas, y hoy solo lleva la 0.
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
            if (bytes != nullptr && last == bytes + entry.span - 1 &&
                Common::ComputeHash64(bytes, entry.span) == entry.source_hash) {
                entry.stale = false;
                Common::FrameStats::texture_reuses.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Ha cambiado de verdad: se retira y se decodifica abajo.
                Common::FrameStats::texture_changed.fetch_add(1, std::memory_order_relaxed);
                Retire(entry);
            }
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
        return nullptr;
    }
    const PAddr address = info.physical_address;
    const u8* source = memory.GetPhysicalPointer(address);
    if (source == nullptr) {
        return nullptr;
    }
    // Filas de mosaicos redondeando hacia ARRIBA. Con height/8, un alto que no
    // fuera multiplo de ocho dejaba la ultima fila fuera del tramo comprobado y
    // la decodificacion la leia igual.
    const u32 span = static_cast<u32>(info.stride) * ((height + 7) / 8);
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
    Entry& entry = MakeRoom(needed);

    // CDRAM primero: la CPU la escribe una vez y la GPU la lee en cada
    // dibujado, que es justo el caso que pide memoria dedicada. Las pequenas,
    // a memoria normal (ver kSmallTexture).
    const Pool first = needed <= kSmallTexture ? Pool::Host : Pool::Cdram;
    Allocation buffer = Allocate(first, needed);
    if (!buffer.Valid()) {
        buffer = Allocate(first == Pool::Host ? Pool::Cdram : Pool::Host, needed);
    }
    if (!buffer.Valid()) {
        LOG_ERROR(Render, "GXM: sin memoria para decodificar una textura de {}x{}", width, height);
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
    if (width % 8 == 0 && height % 8 == 0) {
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
        for (u32 tile_y = 0; tile_y < height / 8; tile_y++) {
            DecodeTileRow(source + tile_y * info.stride, info, width, band.data());
            for (u32 line = 0; line < 8; line++) {
                // Volteo vertical: ver el comentario de arriba.
                const u32 dest_y = height - 1 - (tile_y * 8 + line);
                std::memcpy(dest + static_cast<std::size_t>(dest_y) * width,
                            band.data() + static_cast<std::size_t>(line) * width,
                            static_cast<std::size_t>(width) * 4);
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

    // Los bytes van en orden R,G,B,A, que es lo que describe el nombre ABGR de
    // GXM (los nombres listan los canales del mas significativo al menos).
    if (sceGxmTextureInitLinear(&entry.texture, buffer.Data(), SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR,
                                width, height, 1) < 0) {
        return nullptr;
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

    entry.valid = true;
    entry.key = key;
    entry.address = address;
    entry.span = span;
    entry.last_use = clock;
    entry.source_hash = Common::ComputeHash64(source, span);
    entry.stale = false;
    bytes_used += buffer.Size();
    entry.buffer = std::move(buffer);
    index[key] = static_cast<u32>(&entry - entries.data());
    return &entry.texture;
}

} // namespace Gxm
