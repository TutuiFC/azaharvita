// Copyright 2023-2025 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_rasterizer.h"

namespace Core {
class System;
}

namespace SwRenderer {

struct ScreenInfo {
    u32 width;
    u32 height;
    std::vector<u8> pixels;

#ifdef __PSVITA__
    /**
     * Formato en el que estan 'pixels', para que lo convierta la GPU.
     *
     * En escritorio este buffer va siempre en RGBA8 porque LoadFBToScreenInfo
     * decodifica pixel a pixel. En la Vita eso es tirar trabajo: son ~172.800
     * pixeles por fotograma, los hace el HILO PRINCIPAL -- el mismo que emula la
     * CPU del 3DS -- mientras los tres del rasterizador estan parados, y la GPU
     * de la consola entiende de forma nativa TODOS los formatos del 3DS.
     *
     * Asi que aqui se copian los bytes crudos y se anota su formato; convertir
     * y girar lo hace el chip grafico al dibujar, que es para lo que esta.
     */
    Pica::PixelFormat format{Pica::PixelFormat::RGBA8};

    /// Bytes por fila del buffer de arriba. Puede no ser width * bytes por
    /// pixel: el framebuffer del 3DS lleva su propio stride.
    u32 stride{};

    /// False cuando no hay imagen valida y hay que pintar un color plano.
    bool valid{false};

    /// Relleno de color: el juego pide un color liso, sin framebuffer. Se usa
    /// mucho para fundidos. Con la conversion en GPU no hay textura que subir,
    /// asi que el frontend pinta un rectangulo y ya. Sin esto la pantalla
    /// saldria NEGRA en vez del color pedido.
    bool fill_enabled{false};
    u8 fill_r{}, fill_g{}, fill_b{};
#endif
};

class RendererSoftware : public VideoCore::RendererBase {
public:
    explicit RendererSoftware(Core::System& system, Pica::PicaCore& pica,
                              Frontend::EmuWindow& window);
    ~RendererSoftware() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override {
        return &rasterizer;
    }

    [[nodiscard]] const ScreenInfo& Screen(VideoCore::ScreenId id) const noexcept {
        return screen_infos[static_cast<u32>(id)];
    }

    void SwapBuffers() override;
    void TryPresent(int timeout_ms, bool is_secondary) override {}

private:
    void PrepareRenderTarget();
    void LoadFBToScreenInfo(int i, const Pica::ColorFill& color_fill);

private:
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;
    RasterizerSoftware rasterizer;
    std::array<ScreenInfo, 3> screen_infos{};
};

} // namespace SwRenderer
