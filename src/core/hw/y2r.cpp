// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "common/assert.h"
#include "common/color.h"
#include "common/common_types.h"
#include "common/microprofile.h"
#include "common/vector_math.h"
#include "core/core.h"
#include "core/hle/service/cam/y2r_u.h"
#include "core/hw/y2r.h"
#include "core/memory.h"

namespace HW::Y2R {

using namespace Service::Y2R;

static const std::size_t MAX_TILES = 1024 / 8;
static const std::size_t TILE_SIZE = 8 * 8;
using ImageTile = std::array<u32, TILE_SIZE>;

/// Converts a image strip from the source YUV format into individual 8x8 RGB32 tiles.
template <InputFormat input_format>
static void ConvertYUVToRGB(const u8* input_Y, const u8* input_U, const u8* input_V,
                            ImageTile output[], unsigned int width, unsigned int height,
                            const CoefficientSet& coefficients) {

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            s32 Y;
            s32 U;
            s32 V;
            if constexpr (input_format == InputFormat::YUV422_Indiv8 ||
                          input_format == InputFormat::YUV422_Indiv16) {
                Y = input_Y[y * width + x];
                U = input_U[(y * width + x) / 2];
                V = input_V[(y * width + x) / 2];
            } else if constexpr (input_format == InputFormat::YUV420_Indiv8 ||
                                 input_format == InputFormat::YUV420_Indiv16) {
                Y = input_Y[y * width + x];
                U = input_U[((y / 2) * width + x) / 2];
                V = input_V[((y / 2) * width + x) / 2];
            } else if constexpr (input_format == InputFormat::YUYV422_Interleaved) {
                Y = input_Y[(y * width + x) * 2];
                U = input_Y[(y * width + (x / 2) * 2) * 2 + 1];
                V = input_Y[(y * width + (x / 2) * 2) * 2 + 3];
            } else {
                UNREACHABLE_MSG("Unknown Y2R input format {}", input_format);
                return;
            }

            // This conversion process is bit-exact with hardware, as far as could be tested.
            auto& c = coefficients;
            s32 cY = c[0] * Y;

            s32 r = cY + c[1] * V;
            s32 g = cY - c[2] * V - c[3] * U;
            s32 b = cY + c[4] * U;

            const s32 rounding_offset = 0x18;
            r = (r >> 3) + c[5] + rounding_offset;
            g = (g >> 3) + c[6] + rounding_offset;
            b = (b >> 3) + c[7] + rounding_offset;

            unsigned int tile = x / 8;
            unsigned int tile_x = x % 8;
            u32* out = &output[tile][y * 8 + tile_x];
            *out = ((u32)std::clamp(r >> 5, 0, 0xFF) << 24) |
                   ((u32)std::clamp(g >> 5, 0, 0xFF) << 16) |
                   ((u32)std::clamp(b >> 5, 0, 0xFF) << 8);
        }
    }
}

#if defined(__ARM_NEON)
/**
 * Y2R CON NEON (0.3.3.0). Yo-kai Watch convierte unas 10 imagenes por segundo
 * de 13,7 ms cada una (y2r:u::StartConversion en crash.txt de 0.3.2.6, el 14 %
 * del hilo de emulacion) e Inazuma Eleven GO 8,5 de 9,7 ms: ~60 ciclos por
 * pixel en cinco pasadas escalares. Aqui ocho pixeles de una vez con las mismas
 * cuentas: productos en 32 bits (vmull/vmlal/vmlsl), los dos desplazamientos
 * de antes y la saturacion de vqshrun + vqmovn, que es el clamp a 0-255 de
 * arriba. Comprobado en el PC contra el codigo escalar con 8,8 millones de
 * pixeles (los seis juegos de coeficientes estandar y coeficientes al azar).
 */
static inline void ConvertEight(uint8x8_t y8, uint8x8_t u8x, uint8x8_t v8x,
                                const CoefficientSet& c, u32* out) {
    const int16x8_t y = vreinterpretq_s16_u16(vmovl_u8(y8));
    const int16x8_t u = vreinterpretq_s16_u16(vmovl_u8(u8x));
    const int16x8_t v = vreinterpretq_s16_u16(vmovl_u8(v8x));
    const int32x4_t kr = vdupq_n_s32(c[5] + 0x18);
    const int32x4_t kg = vdupq_n_s32(c[6] + 0x18);
    const int32x4_t kb = vdupq_n_s32(c[7] + 0x18);
    const auto half = [&](int16x4_t yh, int16x4_t uh, int16x4_t vh, uint16x4_t& r16,
                          uint16x4_t& g16, uint16x4_t& b16) {
        const int32x4_t cy = vmull_n_s16(yh, c[0]);
        const int32x4_t r = vmlal_n_s16(cy, vh, c[1]);
        const int32x4_t g = vmlsl_n_s16(vmlsl_n_s16(cy, vh, c[2]), uh, c[3]);
        const int32x4_t b = vmlal_n_s16(cy, uh, c[4]);
        r16 = vqshrun_n_s32(vaddq_s32(vshrq_n_s32(r, 3), kr), 5);
        g16 = vqshrun_n_s32(vaddq_s32(vshrq_n_s32(g, 3), kg), 5);
        b16 = vqshrun_n_s32(vaddq_s32(vshrq_n_s32(b, 3), kb), 5);
    };
    uint16x4_t r_lo, g_lo, b_lo, r_hi, g_hi, b_hi;
    half(vget_low_s16(y), vget_low_s16(u), vget_low_s16(v), r_lo, g_lo, b_lo);
    half(vget_high_s16(y), vget_high_s16(u), vget_high_s16(v), r_hi, g_hi, b_hi);
    // Bytes 0, B, G, R: en little-endian, R << 24 | G << 16 | B << 8.
    uint8x8x4_t pixels;
    pixels.val[0] = vdup_n_u8(0);
    pixels.val[1] = vqmovn_u16(vcombine_u16(b_lo, b_hi));
    pixels.val[2] = vqmovn_u16(vcombine_u16(g_lo, g_hi));
    pixels.val[3] = vqmovn_u16(vcombine_u16(r_lo, r_hi));
    vst4_u8(reinterpret_cast<u8*>(out), pixels);
}

/// Cuatro bytes de croma, cada uno en dos carriles seguidos (un valor por par
/// de pixeles).
static inline uint8x8_t LoadChromaPairs(const u8* in) {
    u32 word;
    std::memcpy(&word, in, sizeof(word));
    const uint8x8_t v = vreinterpret_u8_u32(vdup_n_u32(word));
    return vzip_u8(v, v).val[0];
}

/// Lo mismo que ConvertYUVToRGB, de ocho en ocho (el ancho siempre es multiplo
/// de 8).
template <InputFormat input_format>
static void ConvertYUVToRGBNeon(const u8* input_Y, const u8* input_U, const u8* input_V,
                                ImageTile output[], unsigned int width, unsigned int height,
                                const CoefficientSet& coefficients) {
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; x += 8) {
            uint8x8_t y8;
            uint8x8_t u8x;
            uint8x8_t v8x;
            if constexpr (input_format == InputFormat::YUYV422_Interleaved) {
                // Y0 U0 Y1 V0 Y2 U1 Y3 V1...: pares los Y, impares U y V alternados.
                const uint8x8x2_t yc = vld2_u8(input_Y + (y * width + x) * 2);
                y8 = yc.val[0];
                const uint8x8x2_t uv = vuzp_u8(yc.val[1], yc.val[1]);
                u8x = vzip_u8(uv.val[0], uv.val[0]).val[0];
                v8x = vzip_u8(uv.val[1], uv.val[1]).val[0];
            } else if constexpr (input_format == InputFormat::YUV422_Indiv8 ||
                                 input_format == InputFormat::YUV422_Indiv16) {
                y8 = vld1_u8(input_Y + y * width + x);
                u8x = LoadChromaPairs(input_U + (y * width + x) / 2);
                v8x = LoadChromaPairs(input_V + (y * width + x) / 2);
            } else {
                y8 = vld1_u8(input_Y + y * width + x);
                u8x = LoadChromaPairs(input_U + ((y / 2) * width + x) / 2);
                v8x = LoadChromaPairs(input_V + ((y / 2) * width + x) / 2);
            }
            ConvertEight(y8, u8x, v8x, coefficients, &output[x / 8][y * 8]);
        }
    }
}
#endif

template <InputFormat input_format>
static void ConvertYUV(const u8* input_Y, const u8* input_U, const u8* input_V, ImageTile output[],
                       unsigned int width, unsigned int height,
                       const CoefficientSet& coefficients) {
#if defined(__ARM_NEON)
    ConvertYUVToRGBNeon<input_format>(input_Y, input_U, input_V, output, width, height,
                                      coefficients);
#else
    ConvertYUVToRGB<input_format>(input_Y, input_U, input_V, output, width, height, coefficients);
#endif
}

/// Simulates an incoming CDMA transfer. The N parameter is used to automatically convert 16-bit
/// formats to 8-bit.
template <std::size_t N>
static void ReceiveData(Memory::MemorySystem& memory, u8* output, ConversionBuffer& buf,
                        std::size_t amount_of_data) {
    const u8* input = memory.GetPointer(buf.address);

    std::size_t output_unit = buf.transfer_unit / N;
    ASSERT(amount_of_data % output_unit == 0);

    while (amount_of_data > 0) {
        if constexpr (N == 1) {
            std::memcpy(output, input, output_unit);
        } else {
            for (std::size_t i = 0; i < output_unit; ++i) {
                output[i] = input[i * N];
            }
        }

        output += output_unit;
        input += buf.transfer_unit + buf.gap;

        buf.address += buf.transfer_unit + buf.gap;
        buf.image_size -= buf.transfer_unit;
        amount_of_data -= output_unit;
    }
}

#if defined(__ARM_NEON)
/// 'count' pixeles del formato intermedio al de salida, ocho a ocho con NEON y el
/// resto como SendData. Comprobado en el PC contra Encode* para los cuatro.
template <OutputFormat output_format>
static void EncodePixels(const u32* input, u8* output, int count, u8 alpha) {
    int i = 0;
    if constexpr (output_format == OutputFormat::RGBA8) {
        const uint32x4_t a = vdupq_n_u32(alpha);
        for (; i + 4 <= count; i += 4) {
            vst1q_u8(output + i * 4, vreinterpretq_u8_u32(vorrq_u32(vld1q_u32(input + i), a)));
        }
    } else {
        for (; i + 8 <= count; i += 8) {
            // Carriles 0, B, G, R (ver ConvertEight).
            const uint8x8x4_t px = vld4_u8(reinterpret_cast<const u8*>(input + i));
            if constexpr (output_format == OutputFormat::RGB8) {
                uint8x8x3_t bgr;
                bgr.val[0] = px.val[1];
                bgr.val[1] = px.val[2];
                bgr.val[2] = px.val[3];
                vst3_u8(output + i * 3, bgr);
            } else if constexpr (output_format == OutputFormat::RGB565) {
                uint16x8_t x = vshll_n_u8(px.val[3], 8);
                x = vsriq_n_u16(x, vshll_n_u8(px.val[2], 8), 5);
                x = vsriq_n_u16(x, vshll_n_u8(px.val[1], 8), 11);
                vst1q_u8(output + i * 2, vreinterpretq_u8_u16(x));
            } else {
                uint16x8_t x = vshll_n_u8(px.val[3], 8);
                x = vsriq_n_u16(x, vshll_n_u8(px.val[2], 8), 5);
                x = vsriq_n_u16(x, vshll_n_u8(px.val[1], 8), 10);
                x = vsriq_n_u16(x, vshll_n_u8(vdup_n_u8(alpha), 8), 15);
                vst1q_u8(output + i * 2, vreinterpretq_u8_u16(x));
            }
        }
    }
    for (; i < count; ++i) {
        const u32 color = input[i];
        const Common::Vec4<u8> col_vec{(u8)(color >> 24), (u8)(color >> 16), (u8)(color >> 8),
                                       alpha};
        if constexpr (output_format == OutputFormat::RGBA8) {
            Common::Color::EncodeRGBA8(col_vec, output + i * 4);
        } else if constexpr (output_format == OutputFormat::RGB8) {
            Common::Color::EncodeRGB8(col_vec, output + i * 3);
        } else if constexpr (output_format == OutputFormat::RGB5A1) {
            Common::Color::EncodeRGB5A1(col_vec, output + i * 2);
        } else {
            Common::Color::EncodeRGB565(col_vec, output + i * 2);
        }
    }
}
#endif

/// Convert intermediate RGB32 format to the final output format while simulating an outgoing CDMA
/// transfer.
template <OutputFormat output_format>
static void SendData(Memory::MemorySystem& memory, const u32* input, ConversionBuffer& buf,
                     int amount_of_data, u8 alpha) {

    u8* output = memory.GetPointer(buf.address);
#if defined(__ARM_NEON)
    /**
     * Con unidades de transferencia de pixeles enteros que reparten la franja
     * justa, el bucle de abajo escribe unit_pixels pixeles por unidad y salta
     * el hueco: lo mismo, de ocho en ocho. Si no, como siempre.
     */
    constexpr int bpp = output_format == OutputFormat::RGBA8  ? 4
                        : output_format == OutputFormat::RGB8 ? 3
                                                              : 2;
    const int unit_pixels = buf.transfer_unit / bpp;
    if (buf.transfer_unit % bpp == 0 && unit_pixels > 0 && amount_of_data % unit_pixels == 0) {
        while (amount_of_data > 0) {
            EncodePixels<output_format>(input, output, unit_pixels, alpha);
            input += unit_pixels;
            output += buf.transfer_unit + buf.gap;
            buf.address += buf.transfer_unit + buf.gap;
            buf.image_size -= buf.transfer_unit;
            amount_of_data -= unit_pixels;
        }
        return;
    }
#endif

    while (amount_of_data > 0) {
        u8* unit_end = output + buf.transfer_unit;
        while (output < unit_end) {
            u32 color = *input++;
            Common::Vec4<u8> col_vec{(u8)(color >> 24), (u8)(color >> 16), (u8)(color >> 8), alpha};

            if constexpr (output_format == OutputFormat::RGBA8) {
                Common::Color::EncodeRGBA8(col_vec, output);
                output += 4;
            } else if constexpr (output_format == OutputFormat::RGB8) {
                Common::Color::EncodeRGB8(col_vec, output);
                output += 3;
            } else if constexpr (output_format == OutputFormat::RGB5A1) {
                Common::Color::EncodeRGB5A1(col_vec, output);
                output += 2;
            } else if constexpr (output_format == OutputFormat::RGB565) {
                Common::Color::EncodeRGB565(col_vec, output);
                output += 2;
            } else {
                UNREACHABLE_MSG("Unknown Y2R output format {}", output_format);
            }

            amount_of_data -= 1;
        }

        output += buf.gap;
        buf.address += buf.transfer_unit + buf.gap;
        buf.image_size -= buf.transfer_unit;
    }
}

static const u8 linear_lut[TILE_SIZE] = {
    // clang-format off
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
    // clang-format on
};

static const u8 morton_lut[TILE_SIZE] = {
    // clang-format off
     0,  1,  4,  5, 16, 17, 20, 21,
     2,  3,  6,  7, 18, 19, 22, 23,
     8,  9, 12, 13, 24, 25, 28, 29,
    10, 11, 14, 15, 26, 27, 30, 31,
    32, 33, 36, 37, 48, 49, 52, 53,
    34, 35, 38, 39, 50, 51, 54, 55,
    40, 41, 44, 45, 56, 57, 60, 61,
    42, 43, 46, 47, 58, 59, 62, 63,
    // clang-format on
};

static void RotateTile0(const ImageTile& input, ImageTile& output, int height,
                        const u8 out_map[64]) {
    for (int i = 0; i < height * 8; ++i) {
        output[out_map[i]] = input[i];
    }
}

static void RotateTile90(const ImageTile& input, ImageTile& output, int height,
                         const u8 out_map[64]) {
    int out_i = 0;
    for (int x = 0; x < 8; ++x) {
        for (int y = height - 1; y >= 0; --y) {
            output[out_map[out_i++]] = input[y * 8 + x];
        }
    }
}

static void RotateTile180(const ImageTile& input, ImageTile& output, int height,
                          const u8 out_map[64]) {
    int out_i = 0;
    for (int i = height * 8 - 1; i >= 0; --i) {
        output[out_map[out_i++]] = input[i];
    }
}

static void RotateTile270(const ImageTile& input, ImageTile& output, int height,
                          const u8 out_map[64]) {
    int out_i = 0;
    for (int x = 8 - 1; x >= 0; --x) {
        for (int y = 0; y < height; ++y) {
            output[out_map[out_i++]] = input[y * 8 + x];
        }
    }
}

static void WriteTileToOutput(u32* output, const ImageTile& tile, int height, int line_stride) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < 8; ++x) {
            output[y * line_stride + x] = tile[y * 8 + x];
        }
    }
}

MICROPROFILE_DEFINE(Y2R_PerformConversion, "Y2R", "PerformConversion", MP_RGB(185, 66, 245));

/**
 * Performs a Y2R colorspace conversion.
 *
 * The Y2R hardware implements hardware-accelerated YUV to RGB colorspace conversions. It is most
 * commonly used for video playback or to display camera input to the screen.
 *
 * The conversion process is quite configurable, and can be divided in distinct steps. From
 * observation, it appears that the hardware buffers a single 8-pixel tall strip of image data
 * internally and converts it in one go before writing to the output and loading the next strip.
 *
 * The steps taken to convert one strip of image data are:
 *
 * - The hardware receives data via CDMA (http://3dbrew.org/wiki/Corelink_DMA_Engines), which is
 *   presumably stored in one or more internal buffers. This process can be done in several separate
 *   transfers, as long as they don't exceed the size of the internal image buffer. This allows
 *   flexibility in input strides.
 * - The input data is decoded into a YUV tuple. Several formats are suported, see the `InputFormat`
 *   enum.
 * - The YUV tuple is converted, using fixed point calculations, to RGB. This step can be configured
 *   using a set of coefficients to support different colorspace standards. See `CoefficientSet`.
 * - The strip can be optionally rotated 90, 180 or 270 degrees. Since each strip is processed
 *   independently, this notably rotates each *strip*, not the entire image. This means that for 90
 *   or 270 degree rotations, the output will be in terms of several 8 x height images, and for any
 *   non-zero rotation the strips will have to be re-arranged so that the parts of the image will
 *   not be shuffled together. This limitation makes this a feature of somewhat dubious utility. 90
 *   or 270 degree rotations in images with non-even height don't seem to work properly.
 * - The data is converted to the output RGB format. See the `OutputFormat` enum.
 * - The data can be output either linearly line-by-line or in the swizzled 8x8 tile format used by
 *   the PICA. This is decided by the `BlockAlignment` enum. If 8x8 alignment is used, then the
 *   image must have a height divisible by 8. The image width must always be divisible by 8.
 * - The final data is then CDMAed out to main memory and the next image strip is processed. This
 *   offers the same flexibility as the input stage.
 *
 * In this implementation, to avoid the combinatorial explosion of parameter combinations, common
 * intermediate formats are used and where possible tables or parameters are used instead of
 * diverging code paths to keep the amount of branches in check. Some steps are also merged to
 * increase efficiency.
 *
 * Output for all valid settings combinations matches hardware, however output in some edge-cases
 * differs:
 *
 * - `Block8x8` alignment with non-mod8 height produces different garbage patterns on the last
 *   strip, especially when combined with rotation.
 * - Hardware, when using `Linear` alignment with a non-even height and 90 or 270 degree rotation
 *   produces misaligned output on the last strip. This implmentation produces output with the
 *   correct "expected" alignment.
 *
 * Hardware behaves strangely (doesn't fire the completion interrupt, for example) in these cases,
 * so they are believed to be invalid configurations anyway.
 */
void PerformConversion(Memory::MemorySystem& memory, ConversionConfiguration cvt) {
    MICROPROFILE_SCOPE(Y2R_PerformConversion);

    ASSERT(cvt.input_line_width % 8 == 0);
    ASSERT(cvt.block_alignment != BlockAlignment::Block8x8 || cvt.input_lines % 8 == 0);
    // Tiles per row
    std::size_t num_tiles = cvt.input_line_width / 8;
    ASSERT(num_tiles <= MAX_TILES);

    // Buffer used as a CDMA source/target.
    std::unique_ptr<u8[]> data_buffer(new u8[cvt.input_line_width * 8 * 4]);
    // Intermediate storage for decoded 8x8 image tiles. Always stored as RGB32.
    std::unique_ptr<ImageTile[]> tiles(new ImageTile[num_tiles]);
    ImageTile tmp_tile;

    // LUT used to remap writes to a tile. Used to allow linear or swizzled output without
    // requiring two different code paths.
    const u8* tile_remap = nullptr;
    switch (cvt.block_alignment) {
    case BlockAlignment::Linear:
        tile_remap = linear_lut;
        break;
    case BlockAlignment::Block8x8:
        tile_remap = morton_lut;
        break;
    }

    for (unsigned int y = 0; y < cvt.input_lines; y += 8) {
        unsigned int row_height = std::min(cvt.input_lines - y, 8u);

        // Total size in pixels of incoming data required for this strip.
        const std::size_t row_data_size = row_height * cvt.input_line_width;

        u8* input_Y = data_buffer.get();
        u8* input_U = input_Y + 8 * cvt.input_line_width;
        u8* input_V = input_U + 8 * cvt.input_line_width / 2;

        switch (cvt.input_format) {
        case InputFormat::YUV422_Indiv8:
            ReceiveData<1>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<1>(memory, input_U, cvt.src_U, row_data_size / 2);
            ReceiveData<1>(memory, input_V, cvt.src_V, row_data_size / 2);
            ConvertYUV<InputFormat::YUV422_Indiv8>(input_Y, input_U, input_V, tiles.get(),
                                                   cvt.input_line_width, row_height,
                                                   cvt.coefficients);
            break;
        case InputFormat::YUV420_Indiv8:
            ReceiveData<1>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<1>(memory, input_U, cvt.src_U, row_data_size / 4);
            ReceiveData<1>(memory, input_V, cvt.src_V, row_data_size / 4);
            ConvertYUV<InputFormat::YUV420_Indiv8>(input_Y, input_U, input_V, tiles.get(),
                                                   cvt.input_line_width, row_height,
                                                   cvt.coefficients);
            break;
        case InputFormat::YUV422_Indiv16:
            ReceiveData<2>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<2>(memory, input_U, cvt.src_U, row_data_size / 2);
            ReceiveData<2>(memory, input_V, cvt.src_V, row_data_size / 2);
            ConvertYUV<InputFormat::YUV422_Indiv16>(input_Y, input_U, input_V, tiles.get(),
                                                    cvt.input_line_width, row_height,
                                                    cvt.coefficients);
            break;
        case InputFormat::YUV420_Indiv16:
            ReceiveData<2>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<2>(memory, input_U, cvt.src_U, row_data_size / 4);
            ReceiveData<2>(memory, input_V, cvt.src_V, row_data_size / 4);
            ConvertYUV<InputFormat::YUV420_Indiv16>(input_Y, input_U, input_V, tiles.get(),
                                                    cvt.input_line_width, row_height,
                                                    cvt.coefficients);
            break;
        case InputFormat::YUYV422_Interleaved:
            input_U = nullptr;
            input_V = nullptr;
            ReceiveData<1>(memory, input_Y, cvt.src_YUYV, row_data_size * 2);
            ConvertYUV<InputFormat::YUYV422_Interleaved>(input_Y, input_U, input_V, tiles.get(),
                                                         cvt.input_line_width, row_height,
                                                         cvt.coefficients);
            break;
        default:
            UNREACHABLE_MSG("Unknown Y2R input format {}", cvt.input_format);
            return;
        }

        u32* output_buffer = reinterpret_cast<u32*>(data_buffer.get());

        // Sin rotacion y en lineal, RotateTile0 con linear_lut es una copia
        // tal cual: la fila del mosaico va directa a la salida.
        const bool direct =
            cvt.rotation == Rotation::None && cvt.block_alignment == BlockAlignment::Linear;
        for (std::size_t i = 0; direct && i < num_tiles; ++i) {
            WriteTileToOutput(output_buffer, tiles[i], row_height, cvt.input_line_width);
            output_buffer += 8;
        }
        for (std::size_t i = 0; !direct && i < num_tiles; ++i) {
            int image_strip_width = 0;
            int output_stride = 0;

            switch (cvt.rotation) {
            case Rotation::None:
                RotateTile0(tiles[i], tmp_tile, row_height, tile_remap);
                image_strip_width = cvt.input_line_width;
                output_stride = 8;
                break;
            case Rotation::Clockwise_90:
                RotateTile90(tiles[i], tmp_tile, row_height, tile_remap);
                image_strip_width = 8;
                output_stride = 8 * row_height;
                break;
            case Rotation::Clockwise_180:
                // For 180 and 270 degree rotations we also invert the order of tiles in the strip,
                // since the rotates are done individually on each tile.
                RotateTile180(tiles[num_tiles - i - 1], tmp_tile, row_height, tile_remap);
                image_strip_width = cvt.input_line_width;
                output_stride = 8;
                break;
            case Rotation::Clockwise_270:
                RotateTile270(tiles[num_tiles - i - 1], tmp_tile, row_height, tile_remap);
                image_strip_width = 8;
                output_stride = 8 * row_height;
                break;
            }

            switch (cvt.block_alignment) {
            case BlockAlignment::Linear:
                WriteTileToOutput(output_buffer, tmp_tile, row_height, image_strip_width);
                output_buffer += output_stride;
                break;
            case BlockAlignment::Block8x8:
                WriteTileToOutput(output_buffer, tmp_tile, 8, 8);
                output_buffer += TILE_SIZE;
                break;
            }
        }

        switch (cvt.output_format) {
        case OutputFormat::RGBA8:
            SendData<OutputFormat::RGBA8>(memory, reinterpret_cast<u32*>(data_buffer.get()),
                                          cvt.dst, static_cast<int>(row_data_size),
                                          static_cast<u8>(cvt.alpha));
            break;
        case OutputFormat::RGB8:
            SendData<OutputFormat::RGB8>(memory, reinterpret_cast<u32*>(data_buffer.get()), cvt.dst,
                                         static_cast<int>(row_data_size),
                                         static_cast<u8>(cvt.alpha));
            break;
        case OutputFormat::RGB5A1:
            SendData<OutputFormat::RGB5A1>(memory, reinterpret_cast<u32*>(data_buffer.get()),
                                           cvt.dst, static_cast<int>(row_data_size),
                                           static_cast<u8>(cvt.alpha));
            break;
        case OutputFormat::RGB565:
            SendData<OutputFormat::RGB565>(memory, reinterpret_cast<u32*>(data_buffer.get()),
                                           cvt.dst, static_cast<int>(row_data_size),
                                           static_cast<u8>(cvt.alpha));
            break;
        default:
            UNREACHABLE_MSG("Unknown Y2R output format {}", cvt.output_format);
            return;
        }
    }
}
} // namespace HW::Y2R
