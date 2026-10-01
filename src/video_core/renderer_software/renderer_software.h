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

    /**
     * Puntero DIRECTO al framebuffer dentro de la memoria del invitado.
     *
     * En la Vita este ScreenInfo ya no lleva los pixeles copiados: lleva a donde
     * estan. 'pixels' se queda vacio y no se usa.
     *
     * POR QUE. El camino de presentacion hacia DOS copias del framebuffer por
     * fotograma y por pantalla: LoadFBToScreenInfo copiaba de la memoria del
     * 3DS a este vector, y justo despues el presentador copiaba de este vector a
     * la memoria del chip grafico. La primera copia no aportaba nada -- nadie
     * modifica los bytes entre una y otra -- y costaba ~230 KB por pantalla y
     * fotograma movidos por el HILO PRINCIPAL, que es el que emula el ARM11 y el
     * que va saturado.
     *
     * Ahora se apunta y el presentador copia una sola vez, del invitado
     * directamente a la memoria de la GPU.
     *
     * POR QUE EL PUNTERO SE PUEDE GUARDAR. Apunta al almacen de la MemorySystem
     * (FCRAM o VRAM del invitado), que vive tanto como el emulador y no se mueve.
     * LoadFBToScreenInfo ya comprueba ANTES de asignarlo que el tramo entero esta
     * mapeado y es contiguo hasta el ultimo pixel. Y entre que se asigna y que se
     * consume no corre ni la CPU del invitado ni el rasterizador: las dos cosas
     * pasan dentro del mismo SwapBuffers, en el mismo hilo.
     *
     * nullptr cuando no hay imagen (framebuffer sin mapear o relleno de color).
     */
    const u8* source{nullptr};

    /// Bytes utiles a partir de 'source' (stride * alto). Sirve para que el
    /// consumidor no tenga que recalcularlo y pueda comprobar lo que copia.
    std::size_t source_size{0};

    /**
     * Direccion FISICA del framebuffer (0.1.5.2, 4.6).
     *
     * 'source' es un puntero a memoria del invitado; para preguntar al
     * rasterizador GXM si hay una superficie dibujada en la GPU sobre esa
     * misma direccion hace falta la PAddr. Cero cuando no hay imagen.
     */
    u32 source_address{0};
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
