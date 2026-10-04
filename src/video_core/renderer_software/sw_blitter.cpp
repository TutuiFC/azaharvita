// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"
#include "common/color.h"
#include "common/vector_math.h"
#include "core/memory.h"
#include "video_core/pica/regs_external.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/utils.h"
#ifdef __PSVITA__
#include <fmt/format.h>
#include "common/vita_diag.h"
#endif

namespace SwRenderer {

static Common::Vec4<u8> DecodePixel(Pica::PixelFormat input_format, const u8* src_pixel) {
    switch (input_format) {
    case Pica::PixelFormat::RGBA8:
        return Common::Color::DecodeRGBA8(src_pixel);
    case Pica::PixelFormat::RGB8:
        return Common::Color::DecodeRGB8(src_pixel);
    case Pica::PixelFormat::RGB565:
        return Common::Color::DecodeRGB565(src_pixel);
    case Pica::PixelFormat::RGB5A1:
        return Common::Color::DecodeRGB5A1(src_pixel);
    case Pica::PixelFormat::RGBA4:
        return Common::Color::DecodeRGBA4(src_pixel);
    default:
        LOG_ERROR(HW_GPU, "Unknown source framebuffer format {:x}", input_format);
        return {0, 0, 0, 0};
    }
}

SwBlitter::SwBlitter(Memory::MemorySystem& memory_, VideoCore::RasterizerInterface* rasterizer_)
    : memory{memory_}, rasterizer{rasterizer_} {}

SwBlitter::~SwBlitter() = default;

void SwBlitter::TextureCopy(const Pica::DisplayTransferConfig& config) {
    const PAddr src_addr = config.GetPhysicalInputAddress();
    const PAddr dst_addr = config.GetPhysicalOutputAddress();

    // TODO: do hwtest with invalid addresses
    if (!memory.IsValidPhysicalAddress(src_addr)) {
        LOG_CRITICAL(HW_GPU, "invalid input address {:#010X}", src_addr);
        return;
    }

    if (!memory.IsValidPhysicalAddress(dst_addr)) {
        LOG_CRITICAL(HW_GPU, "invalid output address {:#010X}", dst_addr);
        return;
    }

    u8* src_pointer = memory.GetPhysicalPointer(src_addr);
    u8* dst_pointer = memory.GetPhysicalPointer(dst_addr);

    u32 remaining_size = Common::AlignDown(config.texture_copy.size, 16);
    if (remaining_size == 0) {
        LOG_CRITICAL(HW_GPU, "zero size. Real hardware freezes on this.");
        return;
    }

    u32 input_gap = config.texture_copy.input_gap * 16;
    u32 output_gap = config.texture_copy.output_gap * 16;

    // Zero gap means contiguous input/output even if width = 0. To avoid infinite loop below, width
    // is assigned with the total size if gap = 0.
    u32 input_width = input_gap == 0 ? remaining_size : config.texture_copy.input_width * 16;
    u32 output_width = output_gap == 0 ? remaining_size : config.texture_copy.output_width * 16;

    if (input_width == 0) {
        LOG_CRITICAL(HW_GPU, "zero input width. Real hardware freezes on this.");
        return;
    }

    if (output_width == 0) {
        LOG_CRITICAL(HW_GPU, "zero output width. Real hardware freezes on this.");
        return;
    }

    const std::size_t contiguous_input_size =
        config.texture_copy.size / input_width * (input_width + input_gap);
    rasterizer->FlushRegion(config.GetPhysicalInputAddress(),
                            static_cast<u32>(contiguous_input_size));

    const std::size_t contiguous_output_size =
        config.texture_copy.size / output_width * (output_width + output_gap);

    // Only need to flush output if it has a gap
    if (output_gap != 0) {
        rasterizer->FlushAndInvalidateRegion(dst_addr, static_cast<u32>(contiguous_output_size));
    } else {
        rasterizer->InvalidateRegion(dst_addr, static_cast<u32>(contiguous_output_size));
    }

    u32 remaining_input = input_width;
    u32 remaining_output = output_width;
    while (remaining_size > 0) {
        u32 copy_size = std::min({remaining_input, remaining_output, remaining_size});

        std::memcpy(dst_pointer, src_pointer, copy_size);
        src_pointer += copy_size;
        dst_pointer += copy_size;

        remaining_input -= copy_size;
        remaining_output -= copy_size;
        remaining_size -= copy_size;

        if (remaining_input == 0) {
            remaining_input = input_width;
            src_pointer += input_gap;
        }
        if (remaining_output == 0) {
            remaining_output = output_width;
            dst_pointer += output_gap;
        }
    }
}

namespace {
/**
 * Mosaico -> lineal con el MISMO formato, que es la transferencia de cada
 * fotograma: del framebuffer donde se dibuja al que barre la pantalla.
 *
 * El bucle general de mas abajo decodifica y vuelve a codificar CADA pixel, y
 * ademas recalcula por pixel cosas que son constantes en toda la transferencia:
 * los bytes por pixel de los dos formatos (un switch cada uno), cual de las
 * cuatro combinaciones de disposicion toca, y el desplazamiento Morton. Cuando
 * los dos formatos coinciden, todo ese ir y venir produce exactamente los
 * mismos bytes que mover la memoria de sitio.
 *
 * Y se puede mover de dos en dos. En el orden Morton de 8x8, fijada la fila,
 * los texels de x par e impar caen seguidos (la fila de abajo del mosaico es
 * 00 01 04 05 16 17 20 21), y en el lado lineal x y x+1 tambien lo son. Aqui
 * los dos lados tienen el mismo tamano de pixel, asi que la pareja es contigua
 * en ambos y vale para 2, 3 y 4 bytes.
 *
 * Medido en consola: estas transferencias eran el 45% del tiempo de GPU
 * emulada, unos 51 ms de un fotograma de 143.
 */
template <u32 kBpp>
void UntileSameFormat(const u8* src, u8* dst, u32 output_width, u32 output_height,
                      u32 input_width, bool flip_vertically) {
    const u32 full_tiles = output_width & ~7u;
    for (u32 y = 0; y < output_height; ++y) {
        const u32 output_y = flip_vertically ? (output_height - y - 1) : y;
        u8* const dst_row = dst + static_cast<std::size_t>(output_y) * output_width * kBpp;
        const u8* const tile_row =
            src + static_cast<std::size_t>(y & ~7u) * input_width * kBpp;

        u32 pair_offset[4];
        for (u32 k = 0; k < 4; k++) {
            pair_offset[k] = VideoCore::GetMortonOffset(k * 2, y, kBpp);
        }

        for (u32 tile_x = 0; tile_x < full_tiles; tile_x += 8) {
            // GetMortonOffset ya mete el salto de mosaico en su termino
            // coarse_x, asi que aqui solo se suma el del mosaico en curso.
            const u8* const tile_base =
                tile_row + static_cast<std::size_t>(tile_x) * 8 * kBpp;
            u8* const linear_base = dst_row + static_cast<std::size_t>(tile_x) * kBpp;
            for (u32 k = 0; k < 4; k++) {
                std::memcpy(linear_base + k * 2 * kBpp, tile_base + pair_offset[k], kBpp * 2);
            }
        }
        for (u32 x = full_tiles; x < output_width; ++x) {
            std::memcpy(dst_row + static_cast<std::size_t>(x) * kBpp,
                        tile_row + VideoCore::GetMortonOffset(x, y, kBpp), kBpp);
        }
    }
}
/**
 * Mosaico RGBA8 -> lineal RGB8, sin escalado: la copia de pantalla de Pokemon.
 *
 * Pokemon (y muchos otros) dibuja en RGBA8 y su buffer de pantalla es RGB8
 * (crash.txt: "fmt 0" en el de dibujo, "formato 1" en el de pantalla). Con
 * formatos distintos no se aplicaba UntileSameFormat y caia al bucle general,
 * que por cada pixel recalcula el Morton y pasa por dos switch de formato.
 *
 * La conversion es solo mover bytes: RGBA8 guarda A,B,G,R (DecodeRGBA8 lee
 * r = byte 3, g = 2, b = 1) y EncodeRGB8 escribe b, g, r en los bytes 0, 1, 2.
 * O sea, el pixel de salida son los bytes 1, 2 y 3 del de entrada, en orden:
 * los mismos bytes que produce el bucle general.
 */
void UntileRGBA8ToRGB8(const u8* src, u8* dst, u32 output_width, u32 output_height,
                       u32 input_width, bool flip_vertically) {
    const u32 full_tiles = output_width & ~7u;
    for (u32 y = 0; y < output_height; ++y) {
        const u32 output_y = flip_vertically ? (output_height - y - 1) : y;
        u8* const dst_row = dst + static_cast<std::size_t>(output_y) * output_width * 3;
        const u8* const tile_row = src + static_cast<std::size_t>(y & ~7u) * input_width * 4;

        u32 pair_offset[4];
        for (u32 k = 0; k < 4; k++) {
            pair_offset[k] = VideoCore::GetMortonOffset(k * 2, y, 4);
        }

        for (u32 tile_x = 0; tile_x < full_tiles; tile_x += 8) {
            const u8* const tile_base = tile_row + static_cast<std::size_t>(tile_x) * 8 * 4;
            u8* out = dst_row + static_cast<std::size_t>(tile_x) * 3;
            for (u32 k = 0; k < 4; k++) {
                // La pareja x, x+1 esta seguida en el mosaico (ver arriba).
                const u8* in = tile_base + pair_offset[k];
                out[0] = in[1];
                out[1] = in[2];
                out[2] = in[3];
                out[3] = in[5];
                out[4] = in[6];
                out[5] = in[7];
                out += 6;
            }
        }
        for (u32 x = full_tiles; x < output_width; ++x) {
            const u8* in = tile_row + VideoCore::GetMortonOffset(x, y, 4);
            u8* out = dst_row + static_cast<std::size_t>(x) * 3;
            out[0] = in[1];
            out[1] = in[2];
            out[2] = in[3];
        }
    }
}

#ifdef __PSVITA__
/// Cada combinacion distinta de transferencia, una vez, en crash.txt: con eso
/// se sabe que caminos rapidos faltan sin tener que adivinar. Hasta ocho.
void NoteTransferConfig(const Pica::DisplayTransferConfig& config, u32 output_width,
                        u32 output_height) {
    static u32 seen[8] = {};
    static u32 seen_count = 0;
    const u32 signature = static_cast<u32>(config.input_format.Value()) |
                          static_cast<u32>(config.output_format.Value()) << 4 |
                          static_cast<u32>(config.scaling.Value()) << 8 |
                          (config.input_linear ? 1u : 0u) << 12 |
                          (config.dont_swizzle ? 1u : 0u) << 13 |
                          (config.flip_vertically ? 1u : 0u) << 14 | (output_width << 16);
    for (u32 i = 0; i < seen_count; i++) {
        if (seen[i] == signature) {
            return;
        }
    }
    if (seen_count == 8) {
        return;
    }
    seen[seen_count++] = signature;
    Common::VitaNote("transferencia",
                     fmt::format("fmt {}->{} escala {} lineal {} sin_swz {} flip {} {}x{} -> "
                                 "{}x{}",
                                 static_cast<u32>(config.input_format.Value()),
                                 static_cast<u32>(config.output_format.Value()),
                                 static_cast<u32>(config.scaling.Value()),
                                 static_cast<u32>(config.input_linear.Value()),
                                 static_cast<u32>(config.dont_swizzle.Value()),
                                 static_cast<u32>(config.flip_vertically.Value()),
                                 static_cast<u32>(config.input_width.Value()),
                                 static_cast<u32>(config.input_height.Value()), output_width,
                                 output_height)
                         .c_str());
}
#endif
} // Anonymous namespace

void SwBlitter::DisplayTransfer(const Pica::DisplayTransferConfig& config) {
    const PAddr src_addr = config.GetPhysicalInputAddress();
    PAddr dst_addr = config.GetPhysicalOutputAddress();

    // TODO: do hwtest with these cases
    if (!memory.IsValidPhysicalAddress(src_addr)) {
        LOG_CRITICAL(HW_GPU, "invalid input address {:#010X}", src_addr);
        return;
    }

    if (!memory.IsValidPhysicalAddress(dst_addr)) {
        LOG_CRITICAL(HW_GPU, "invalid output address {:#010X}", dst_addr);
        return;
    }

    if (config.input_width == 0) {
        LOG_CRITICAL(HW_GPU, "zero input width");
        return;
    }

    if (config.input_height == 0) {
        LOG_CRITICAL(HW_GPU, "zero input height");
        return;
    }

    if (config.output_width == 0) {
        LOG_CRITICAL(HW_GPU, "zero output width");
        return;
    }

    if (config.output_height == 0) {
        LOG_CRITICAL(HW_GPU, "zero output height");
        return;
    }

    // Using flip_vertically alongside crop_input_lines produces skewed output on hardware.
    // We have to emulate this because some games rely on this behaviour to render correctly.
    if (config.flip_vertically && config.crop_input_lines) {
        dst_addr += (config.input_width - config.output_width) * (config.output_height - 1) *
                    BytesPerPixel(config.output_format);
    }

    u8* src_pointer = memory.GetPhysicalPointer(src_addr);
    u8* dst_pointer = memory.GetPhysicalPointer(dst_addr);

    if (config.scaling > config.ScaleXY) {
        LOG_CRITICAL(HW_GPU, "Unimplemented display transfer scaling mode {}",
                     config.scaling.Value());
        UNIMPLEMENTED();
        return;
    }

    if (config.input_linear && config.scaling != config.NoScale) {
        LOG_CRITICAL(HW_GPU, "Scaling is only implemented on tiled input");
        UNIMPLEMENTED();
        return;
    }

    const u32 horizontal_scale = config.scaling != config.NoScale ? 1 : 0;
    const u32 vertical_scale = config.scaling == config.ScaleXY ? 1 : 0;

    const u32 output_width = config.output_width >> horizontal_scale;
    const u32 output_height = config.output_height >> vertical_scale;

    const u32 input_size =
        config.input_width * config.input_height * BytesPerPixel(config.input_format);
    const u32 output_size = output_width * output_height * BytesPerPixel(config.output_format);

    rasterizer->FlushRegion(config.GetPhysicalInputAddress(), input_size);
    rasterizer->InvalidateRegion(config.GetPhysicalOutputAddress(), output_size);
#ifdef __PSVITA__
    NoteTransferConfig(config, output_width, output_height);
#endif

    /**
     * Caminos rapidos: transferencias que no transforman nada.
     *
     * El bucle de abajo decodifica y vuelve a codificar pixel a pixel, con el
     * calculo de Morton incluido. Cuando origen y destino comparten formato y
     * disposicion, esa ida y vuelta produce exactamente los mismos bytes que
     * una copia, y los DisplayTransfer lineales (copias de framebuffer y de
     * buffers de pantalla) son de los mas comunes.
     *
     * Los dos casos de abajo son EQUIVALENTES byte a byte al bucle -- mismo
     * formato, misma disposicion por fila, sin escalado -- y solo sustituyen
     * el trabajo por memcpy.
     */
    if (config.scaling == config.NoScale &&
        config.input_format.Value() == config.output_format.Value()) {
        const u32 bytes_per_pixel = BytesPerPixel(config.output_format);

        if (config.input_linear && config.dont_swizzle) {
            // Lineal -> lineal: una copia de filas. Con flip, el origen de cada
            // fila de salida es la fila reflejada.
            const u32 row_bytes = output_width * bytes_per_pixel;
            const u32 src_stride = config.input_width * bytes_per_pixel;
            u8* dst_row = dst_pointer;
            for (u32 y = 0; y < output_height; ++y) {
                const u32 src_y = config.flip_vertically ? (output_height - y - 1) : y;
                std::memcpy(dst_row, src_pointer + src_y * src_stride, row_bytes);
                dst_row += row_bytes;
            }
            return;
        }

        if (!config.input_linear && config.dont_swizzle && !config.flip_vertically &&
            config.input_width == output_width && (output_width % 8) == 0 &&
            (output_height % 8) == 0) {
            // Mosaico -> mosaico, mismas dimensiones: la formula de Morton es
            // identica en los dos lados, asi que el mapeo es 1:1 y se puede
            // copiar el bloque entero. Las dimensiones multiplo de 8 garantizan
            // que no hay bytes de relleno que el bucle no tocaria.
            std::memcpy(dst_pointer, src_pointer,
                        static_cast<std::size_t>(output_size));
            return;
        }

        /**
         * Mosaico -> lineal: deshacer el entrelazado y nada mas.
         *
         * Es la transferencia que hace cada juego en cada fotograma para
         * llevar lo dibujado al buffer que barre la pantalla. Con los dos
         * formatos iguales no hay nada que convertir, solo recolocar, y eso lo
         * hace UntileSameFormat de dos pixeles en dos. El bucle general de
         * abajo llega al mismo resultado decodificando y recodificando pixel a
         * pixel, que es de donde salia el 45% del tiempo de GPU emulada.
         */
        if (!config.input_linear && !config.dont_swizzle) {
            switch (bytes_per_pixel) {
            case 2:
                UntileSameFormat<2>(src_pointer, dst_pointer, output_width, output_height,
                                    config.input_width, config.flip_vertically != 0);
                return;
            case 3:
                UntileSameFormat<3>(src_pointer, dst_pointer, output_width, output_height,
                                    config.input_width, config.flip_vertically != 0);
                return;
            case 4:
                UntileSameFormat<4>(src_pointer, dst_pointer, output_width, output_height,
                                    config.input_width, config.flip_vertically != 0);
                return;
            default:
                // Formato de tamano raro: al bucle general, que lo sabe tratar.
                break;
            }
        }
    }

    if (config.scaling == config.NoScale && !config.input_linear && !config.dont_swizzle &&
        config.input_format.Value() == Pica::PixelFormat::RGBA8 &&
        config.output_format.Value() == Pica::PixelFormat::RGB8) {
        UntileRGBA8ToRGB8(src_pointer, dst_pointer, output_width, output_height,
                          config.input_width, config.flip_vertically != 0);
        return;
    }

    /**
     * Los bytes por pixel, UNA vez y no por pixel.
     *
     * Estaban dentro del bucle interno, y BytesPerPixel es un switch sobre el
     * formato: dos switches por pixel para leer dos valores que no cambian en
     * toda la transferencia. Sacarlos deja ademas que el compilador mantenga
     * los strides en registros.
     */
    const u32 dst_bytes_per_pixel = BytesPerPixel(config.output_format);
    const u32 src_bytes_per_pixel = BytesPerPixel(config.input_format);

    for (u32 y = 0; y < output_height; ++y) {
        for (u32 x = 0; x < output_width; ++x) {
            Common::Vec4<u8> src_color;

            // Calculate the [x,y] position of the input image
            // based on the current output position and the scale
            const u32 input_x = x << horizontal_scale;
            const u32 input_y = y << vertical_scale;

            u32 output_y;
            if (config.flip_vertically) {
                // Flip the y value of the output data,
                // we do this after calculating the [x,y] position of the input image
                // to account for the scaling options.
                output_y = output_height - y - 1;
            } else {
                output_y = y;
            }

            u32 src_offset;
            u32 dst_offset;

            if (config.input_linear) {
                if (!config.dont_swizzle) {
                    // Interpret the input as linear and the output as tiled
                    u32 coarse_y = output_y & ~7;
                    u32 stride = output_width * dst_bytes_per_pixel;

                    src_offset = (input_x + input_y * config.input_width) * src_bytes_per_pixel;
                    dst_offset = VideoCore::GetMortonOffset(x, output_y, dst_bytes_per_pixel) +
                                 coarse_y * stride;
                } else {
                    // Both input and output are linear
                    src_offset = (input_x + input_y * config.input_width) * src_bytes_per_pixel;
                    dst_offset = (x + output_y * output_width) * dst_bytes_per_pixel;
                }
            } else {
                if (!config.dont_swizzle) {
                    // Interpret the input as tiled and the output as linear
                    const u32 coarse_y = input_y & ~7;
                    const u32 stride = config.input_width * src_bytes_per_pixel;

                    src_offset = VideoCore::GetMortonOffset(input_x, input_y, src_bytes_per_pixel) +
                                 coarse_y * stride;
                    dst_offset = (x + output_y * output_width) * dst_bytes_per_pixel;
                } else {
                    // Both input and output are tiled
                    const u32 out_coarse_y = output_y & ~7;
                    const u32 out_stride = output_width * dst_bytes_per_pixel;

                    const u32 in_coarse_y = input_y & ~7;
                    const u32 in_stride = config.input_width * src_bytes_per_pixel;

                    src_offset = VideoCore::GetMortonOffset(input_x, input_y, src_bytes_per_pixel) +
                                 in_coarse_y * in_stride;
                    dst_offset = VideoCore::GetMortonOffset(x, output_y, dst_bytes_per_pixel) +
                                 out_coarse_y * out_stride;
                }
            }

            const u8* src_pixel = src_pointer + src_offset;
            src_color = DecodePixel(config.input_format, src_pixel);
            if (config.scaling == config.ScaleX) {
                const auto pixel =
                    DecodePixel(config.input_format, src_pixel + src_bytes_per_pixel);
                src_color = ((src_color + pixel) / 2).Cast<u8>();
            } else if (config.scaling == config.ScaleXY) {
                const auto pixel1 =
                    DecodePixel(config.input_format, src_pixel + 1 * src_bytes_per_pixel);
                const auto pixel2 =
                    DecodePixel(config.input_format, src_pixel + 2 * src_bytes_per_pixel);
                const auto pixel3 =
                    DecodePixel(config.input_format, src_pixel + 3 * src_bytes_per_pixel);
                src_color = (((src_color + pixel1) + (pixel2 + pixel3)) / 4).Cast<u8>();
            }

            u8* dst_pixel = dst_pointer + dst_offset;
            switch (config.output_format) {
            case Pica::PixelFormat::RGBA8:
                Common::Color::EncodeRGBA8(src_color, dst_pixel);
                break;
            case Pica::PixelFormat::RGB8:
                Common::Color::EncodeRGB8(src_color, dst_pixel);
                break;
            case Pica::PixelFormat::RGB565:
                Common::Color::EncodeRGB565(src_color, dst_pixel);
                break;
            case Pica::PixelFormat::RGB5A1:
                Common::Color::EncodeRGB5A1(src_color, dst_pixel);
                break;
            case Pica::PixelFormat::RGBA4:
                Common::Color::EncodeRGBA4(src_color, dst_pixel);
                break;
            default:
                LOG_ERROR(HW_GPU, "Unknown destination framebuffer format {:x}",
                          static_cast<u32>(config.output_format.Value()));
                break;
            }
        }
    }
}

void SwBlitter::MemoryFill(const Pica::MemoryFillConfig& config) {
    const PAddr start_addr = config.GetStartAddress();
    const PAddr end_addr = config.GetEndAddress();

    // TODO: do hwtest with these cases
    if (!memory.IsValidPhysicalAddress(start_addr)) {
        LOG_CRITICAL(HW_GPU, "invalid start address {:#010X}", start_addr);
        return;
    }

    if (!memory.IsValidPhysicalAddress(end_addr)) {
        LOG_CRITICAL(HW_GPU, "invalid end address {:#010X}", end_addr);
        return;
    }

    if (end_addr <= start_addr) {
        LOG_CRITICAL(HW_GPU, "invalid memory range from {:#010X} to {:#010X}", start_addr,
                     end_addr);
        return;
    }

    u8* start = memory.GetPhysicalPointer(start_addr);
    u8* end = memory.GetPhysicalPointer(end_addr);

    rasterizer->InvalidateRegion(start_addr, end_addr - start_addr);

    if (config.fill_24bit) {
        // Fill with 24-bit values
        for (u8* ptr = start; ptr < end; ptr += 3) {
            ptr[0] = config.value_24bit_r;
            ptr[1] = config.value_24bit_g;
            ptr[2] = config.value_24bit_b;
        }
    } else if (config.fill_32bit) {
        // Fill with 32-bit values
        if (end > start) {
            const u32 value = config.value_32bit;
            const std::size_t len = (end - start) / sizeof(u32);
            for (std::size_t i = 0; i < len; ++i) {
                std::memcpy(&start[i * sizeof(u32)], &value, sizeof(u32));
            }
        }
    } else {
        // Fill with 16-bit values
        const u16 value_16bit = config.value_16bit.Value();
        for (u8* ptr = start; ptr < end; ptr += sizeof(u16)) {
            std::memcpy(ptr, &value_16bit, sizeof(u16));
        }
    }
}

} // namespace SwRenderer
