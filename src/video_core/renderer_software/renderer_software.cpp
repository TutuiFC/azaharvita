// Copyright 2023-2025 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include "common/color.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "video_core/gpu.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_software/renderer_software.h"
#include "video_core/renderer_software/sw_rasterizer.h"

#ifdef __PSVITA__
#include "common/vita_diag.h"
#endif

namespace SwRenderer {

#ifdef __PSVITA__
namespace {
/**
 * Anota en crash.txt como esta configurado el framebuffer, solo cuando cambia.
 *
 * El registro normal se vuelca desde otro hilo y con buffer, asi que si el
 * proceso muere de golpe se pierde justo lo ultimo. Esto se escribe con sceIo
 * directo: si la consola vuelve a cerrarse aqui, la ultima linea dice con que
 * parametros lo hizo. Son cuatro o cinco lineas por arranque, no un chorro:
 * estos registros se escriben una vez y ya.
 */
struct FramebufferSignature {
    PAddr address;
    u32 stride;
    u32 height;
    u32 format;
    u32 color_fill;
    bool seen;
};

FramebufferSignature last_signature[3]{};

void NoteFramebuffer(int i, PAddr address, u32 stride, u32 height, u32 format, u32 color_fill) {
    FramebufferSignature& last = last_signature[i];
    if (last.seen && last.address == address && last.stride == stride && last.height == height &&
        last.format == format && last.color_fill == color_fill) {
        return;
    }
    last = {address, stride, height, format, color_fill, true};

    char buffer[160];
    const auto result = fmt::format_to_n(
        buffer, sizeof(buffer) - 1,
        "pantalla {} addr {:#010x} stride {} alto {} formato {} relleno {}", i, address, stride,
        height, format, color_fill);
    *result.out = 0;
    Common::VitaNote("fb", buffer);
}
} // Anonymous namespace
#endif

RendererSoftware::RendererSoftware(Core::System& system, Pica::PicaCore& pica_,
                                   Frontend::EmuWindow& window)
    : VideoCore::RendererBase{system, window, nullptr}, memory{system.Memory()}, pica{pica_},
      rasterizer{memory, pica} {}

RendererSoftware::~RendererSoftware() = default;

void RendererSoftware::SwapBuffers() {
    const bool rendered = FrameSkip::ShouldRender();

    // Solo cuentan los fotogramas DIBUJADOS: los contadores del rasterizador
    // (triangulos, pixeles) tampoco se acumulan en los saltados, asi que
    // dividir por el total falsearia las medias del overlay.
    if (rendered) {
        RasterizerStats::frames.fetch_add(1, std::memory_order_relaxed);
    }

    system.perf_stats->StartSwap();
    if (rendered) {
        // En un fotograma saltado no se convierte el framebuffer del 3DS a
        // screen_infos: son ~170.000 pixeles con su switch de formato, y ademas
        // eso corre en el hilo principal, el mismo que emula la CPU. Al no
        // tocarlo, el frontend vuelve a presentar la ultima imagen completa.
        PrepareRenderTarget();
    }

    // EndFrame se llama SIEMPRE, tambien en los saltados: de ahi cuelga
    // PollEvents, que lee los mandos y el tactil. Saltarselo dejaria los
    // controles respondiendo a una fraccion de la velocidad.
    //
    // Va DENTRO de la medida del swap: presenta por vita2d (subida de texturas,
    // dibujo por GPU y swap de buffers), y eso puede ser una parte notable del
    // fotograma que hasta ahora no salia en ninguna linea del overlay.
    EndFrame();
    system.perf_stats->EndSwap();

    FrameSkip::AdvanceFrame();
}

void RendererSoftware::PrepareRenderTarget() {
    const auto& regs_lcd = pica.regs_lcd;
    for (u32 i = 0; i < 3; i++) {
        const u32 fb_id = i == 2 ? 1 : 0;

#ifdef __PSVITA__
        // i == 1 es la pantalla superior del ojo DERECHO. El frontend de Vita
        // solo dibuja TopLeft (0) y Bottom (2) -- no hay 3D estereoscopico en
        // esta consola --, asi que convertir esta es trabajo tirado: unos 96.000
        // pixeles por fotograma, cada uno con su switch de formato y su copia.
        //
        // Y no es trabajo de los hilos del rasterizador: LoadFBToScreenInfo
        // corre en el hilo principal, el mismo que emula la CPU del 3DS y que va
        // al 100%. O sea que esto se le quita justo al nucleo que va saturado.
        if (i == 1) {
            continue;
        }
#endif

        const auto color_fill = fb_id == 0 ? regs_lcd.color_fill_top : regs_lcd.color_fill_bottom;
        LoadFBToScreenInfo(i, color_fill);
    }
}

/**
 * Deja la pantalla en negro con el tamano que le toca.
 *
 * Se usa cuando los registros del LCD todavia no describen un framebuffer
 * valido. Hay que dejar el buffer con un tamano coherente igualmente: el
 * frontend lee estos campos para subir la imagen a la pantalla de la Vita.
 *
 * Las medidas son las del 3DS con la imagen girada, que es como esta en
 * memoria: 240 de ancho por 400 (pantalla superior) o 320 (inferior) de alto.
 */
static void BlankScreenInfo(ScreenInfo& info, bool is_bottom) {
    info.width = 240;
    info.height = is_bottom ? 320 : 400;
#ifdef __PSVITA__
    // Sin imagen valida: el frontend pinta negro y no sube ninguna textura.
    info.valid = false;
    info.fill_enabled = false;
    info.pixels.clear();
#else
    info.pixels.assign(static_cast<std::size_t>(info.width) * info.height * 4, 0);
#endif
}

void RendererSoftware::LoadFBToScreenInfo(int i, const Pica::ColorFill& color_fill) {
    const u32 fb_id = i == 2 ? 1 : 0;
    const auto& framebuffer = pica.regs.framebuffer_config[fb_id];
    auto& info = screen_infos[i];
    const bool is_bottom = i == 2;

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0 ? framebuffer.address_left1 : framebuffer.address_left2;

#ifdef __PSVITA__
    NoteFramebuffer(i, framebuffer_addr, framebuffer.stride, framebuffer.height,
                    static_cast<u32>(framebuffer.color_format.Value()), color_fill.is_enabled);
#endif

    // color_format son 3 bits del registro, o sea que puede valer 5, 6 o 7
    // mientras el juego todavia no lo ha escrito. BytesPerPixel responde a eso
    // con UNREACHABLE(), que aborta el emulador.
    const Pica::PixelFormat color_format = framebuffer.color_format;
    if (color_format > Pica::PixelFormat::RGBA4) {
        BlankScreenInfo(info, is_bottom);
        return;
    }

    const s32 bpp = static_cast<s32>(Pica::BytesPerPixel(color_format));
    if (bpp <= 0 || framebuffer.stride < static_cast<u32>(bpp)) {
        BlankScreenInfo(info, is_bottom);
        return;
    }

    const u32 pixel_stride = framebuffer.stride / bpp;
    const u32 height = framebuffer.height;

    // Las pantallas del 3DS son 400x240 y 320x240; en memoria van giradas, asi
    // que ni el ancho ni el alto pasan de 400. Un registro sin inicializar dice
    // 65535, y con eso el resize de mas abajo pide gigabytes.
    constexpr u32 kMaxDimension = 1024;
    if (pixel_stride == 0 || pixel_stride > kMaxDimension || height == 0 ||
        height > kMaxDimension) {
        BlankScreenInfo(info, is_bottom);
        return;
    }

    // Con el relleno de color activo el color sale del registro y no se toca la
    // memoria del invitado, asi que ahi no hace falta framebuffer valido.
    const u8* framebuffer_data = nullptr;
    if (!color_fill.is_enabled) {
        // Aqui estaba el fallo que cerraba la consola sin decir nada.
        //
        // GetPhysicalPointer devuelve nullptr cuando la direccion fisica no
        // esta mapeada, y el codigo original la desreferenciaba tal cual. En
        // escritorio no se nota porque este renderizador es un camino de
        // depuracion que casi nadie usa -- todos van por OpenGL o Vulkan --,
        // pero en la Vita es el unico que hay. El juego llega a este punto con
        // el relleno de color ya apagado y la direccion todavia sin mapear, y
        // la lectura mata el proceso al instante: sin excepcion, sin assert y
        // sin que al registro le de tiempo a llegar al disco.
        //
        // No basta con mirar el primer byte: la region tiene que llegar entera
        // hasta el ultimo pixel que se va a leer, y ser contigua.
        const u32 span = height * framebuffer.stride;
        const u8* first_byte = memory.GetPhysicalPointer(framebuffer_addr);
        const u8* last_byte = memory.GetPhysicalPointer(framebuffer_addr + span - 1);
        if (first_byte == nullptr || last_byte != first_byte + span - 1) {
            LOG_WARNING(Render_Software,
                        "Framebuffer {} sin mapear: addr={:#010x} stride={} alto={} formato={} "
                        "({} bytes)",
                        i, framebuffer_addr, framebuffer.stride, height,
                        static_cast<u32>(color_format), span);
            BlankScreenInfo(info, is_bottom);
            return;
        }
        framebuffer_data = first_byte;
    }

#ifdef __PSVITA__
    /**
     * Camino de Vita: copiar los bytes tal cual y que la GPU haga el resto.
     *
     * Lo que se evitaba aqui era un bucle de ~172.800 iteraciones que decodifica
     * el formato del 3DS a RGBA8 Y ADEMAS transpone la imagen -- el framebuffer
     * del 3DS esta girado, 240 de ancho por 400 de alto. Las dos cosas las hace
     * la GPU de la consola gratis: entiende RGB565, RGB5A1, RGBA4, RGB8 y RGBA8
     * de forma nativa, y girar es un parametro del dibujado.
     *
     * Aqui solo queda mover bytes. Y cuando el stride del juego coincide con el
     * ancho -- que es lo habitual -- ni eso: es una sola copia de bloque.
     */
    info.height = height;
    info.width = pixel_stride;
    info.format = color_format;
    info.stride = framebuffer.stride;
    info.valid = true;

    if (color_fill.is_enabled) {
        // Relleno de color: no hay framebuffer que copiar. El frontend pinta un
        // rectangulo liso; no se sube textura ninguna.
        info.valid = false;
        info.fill_enabled = true;
        info.fill_r = static_cast<u8>(color_fill.color_r);
        info.fill_g = static_cast<u8>(color_fill.color_g);
        info.fill_b = static_cast<u8>(color_fill.color_b);
        info.pixels.clear();
        return;
    }
    info.fill_enabled = false;

    {
        const std::size_t total = static_cast<std::size_t>(framebuffer.stride) * height;
        info.pixels.resize(total);
        std::memcpy(info.pixels.data(), framebuffer_data, total);
    }
    return;
#endif

    info.height = height;
    info.width = pixel_stride;
    info.pixels.resize(static_cast<std::size_t>(info.width) * info.height * 4);

    /**
     * Bucle especializado por formato, no un switch por pixel.
     *
     * Aqui se recorren las dos pantallas del 3DS enteras -- unos 172.800
     * pixeles -- y en cada uno se leia el formato de los registros y se
     * despachaba un switch de cinco ramas, mas la comprobacion del relleno de
     * color. Las dos cosas son constantes durante todo el bucle.
     *
     * Y esto corre en el HILO PRINCIPAL, el mismo que emula la CPU del 3DS,
     * mientras los tres hilos del rasterizador estan parados esperando. Cuesta
     * alrededor del 9% del fotograma, que es de las mayores partidas que
     * quedan fuera del rasterizado.
     *
     * Resolviendo el formato una vez fuera, el compilador puede meter el
     * decodificador en linea y dejar el bucle interno en lectura, conversion y
     * escritura.
     */
    const auto convert_all = [&](auto decode) {
        for (u32 y = 0; y < info.height; y++) {
            for (u32 x = 0; x < info.width; x++) {
                // El -1 no estaba: con "pixel_stride - x" la x=0 cae en el
                // primer pixel de la fila SIGUIENTE, asi que la imagen salia
                // corrida una columna y la ultima fila leia un pixel mas alla
                // del framebuffer.
                const u8* pixel =
                    framebuffer_data + (y * pixel_stride + pixel_stride - 1 - x) * bpp;
                const Common::Vec4<u8> color = decode(pixel);
                u8* dest = info.pixels.data() + (x * info.height + y) * 4;
                std::memcpy(dest, color.AsArray(), sizeof(color));
            }
        }
    };

    if (color_fill.is_enabled) {
        // Color plano: ni se lee el framebuffer del juego. Se escribe el mismo
        // pixel en todo el buffer.
        const Common::Vec4<u8> fill{color_fill.color_r, color_fill.color_g, color_fill.color_b,
                                    255};
        u8* dest = info.pixels.data();
        const std::size_t count = static_cast<std::size_t>(info.width) * info.height;
        for (std::size_t i = 0; i < count; i++) {
            std::memcpy(dest + i * 4, fill.AsArray(), 4);
        }
        return;
    }

    switch (color_format) {
    case Pica::PixelFormat::RGBA8:
        convert_all([](const u8* p) { return Common::Color::DecodeRGBA8(p); });
        break;
    case Pica::PixelFormat::RGB8:
        convert_all([](const u8* p) { return Common::Color::DecodeRGB8(p); });
        break;
    case Pica::PixelFormat::RGB565:
        convert_all([](const u8* p) { return Common::Color::DecodeRGB565(p); });
        break;
    case Pica::PixelFormat::RGB5A1:
        convert_all([](const u8* p) { return Common::Color::DecodeRGB5A1(p); });
        break;
    case Pica::PixelFormat::RGBA4:
        convert_all([](const u8* p) { return Common::Color::DecodeRGBA4(p); });
        break;
    }
}

} // namespace SwRenderer
