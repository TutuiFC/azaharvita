// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/renderer_base.h"
#include "video_core/renderer_gxm/rasterizer_gxm.h"
#include "video_core/renderer_software/renderer_software.h"

namespace Gxm {

/**
 * Renderer de PS Vita contra la API nativa de la consola (sceGxm).
 *
 * FASE 2. El objeto tiene dos mitades y solo una trabaja:
 *
 *   - El rasterizado sigue siendo el de software, entero. RendererGXM contiene
 *     un RendererSoftware y le reenvia SwapBuffers(); su rasterizador (el que
 *     la PICA ata) es RasterizerGXM, que a su vez reenvia al de software. Cero
 *     duplicacion de logica y cero cambio de comportamiento.
 *   - La presentacion la hace el frontend con GXM (Gxm::ScreenPresenter), que
 *     lee los ScreenInfo que expone Screen() -- los mismos que produce el
 *     camino de software.
 *
 * POR QUE NO BASTA CON UNA LINEA EN EmuWindow_Vita. Porque la eleccion es del
 * usuario y tiene que poder volver al renderer de software en el menu, y porque
 * a partir de la Fase 3 este objeto crece (rasterizador GXM de verdad, cache de
 * texturas, pipelines) mientras el de software se queda intacto como referencia
 * de correccion. Un if en el frontend no daria sitio a eso.
 *
 * POR QUE CONTIENE AL OTRO RENDERER EN VEZ DE COMPARTIR CODIGO. El camino de
 * software tiene decidido, en RendererSoftware::SwapBuffers, el salto de
 * fotogramas, el orden de los fotogramas saltados y la llamada a EndFrame. Son
 * decisiones ya probadas en consola; reescribirlas aqui para "limpiar" seria
 * duplicarlas y arriesgar las dos. Se reutiliza tal cual.
 */
class RendererGXM final : public VideoCore::RendererBase {
public:
    RendererGXM(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window);
    ~RendererGXM() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override {
        return &rasterizer;
    }

    /**
     * Las pantallas ya convertidas del ultimo fotograma, en el mismo formato
     * que produce el renderer de software.
     *
     * El frontend las presenta con GXM; tener la misma firma que
     * RendererSoftware::Screen permite que la ventana trate a los dos igual y
     * que la comparacion entre backends sea mirar la misma imagen.
     */
    [[nodiscard]] const SwRenderer::ScreenInfo& Screen(VideoCore::ScreenId id) const noexcept {
        return software.Screen(id);
    }

    void SwapBuffers() override;

    void TryPresent(int timeout_ms, bool is_secondary) override {}

private:
    /// El camino de software completo: rasterizado, caches, salto de fotogramas
    /// y el EndFrame que acaba llamando al frontend. Ademas es la referencia de
    /// correccion a la que cae el rasterizador GXM con lo que aun no soporta.
    SwRenderer::RendererSoftware software;

    /// El rasterizador de la GPU: acumula los triangulos que salen del pipeline
    /// de geometria y los dibuja con sceGxmDraw. Ver rasterizer_gxm.h.
    RasterizerGXM rasterizer;
};

} // namespace Gxm
