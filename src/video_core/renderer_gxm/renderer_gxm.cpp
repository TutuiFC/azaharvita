// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/renderer_gxm.h"

#include "core/core.h"

namespace Gxm {

RendererGXM::RendererGXM(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window)
    : VideoCore::RendererBase{system, window, nullptr}, software{system, pica, window},
      rasterizer{*software.Rasterizer(), system.Memory(), pica} {}

RendererGXM::~RendererGXM() = default;

void RendererGXM::SwapBuffers() {
    // El rasterizador GXM puede tener una escena abierta con lo que ha dibujado
    // este fotograma. Hay que cerrarla y volcarla a memoria del invitado ANTES
    // de que PrepareRenderTarget (y el frontend) lean los framebuffers: si no,
    // se presentaria la imagen anterior. Las superficies que el presentador lee
    // directamente por una copia de pantalla en la GPU no se vuelcan (0.1.8.6).
    rasterizer.FlushForPresent();

    // El fotograma lo produce el camino de software completo (salto de
    // fotogramas incluido). La diferencia con el renderer de software esta
    // fuera, en como lo presenta EmuWindow_Vita: por GXM si el chip esta
    // disponible y por vita2d si no. Ver Gxm::ScreenPresenter.
    software.SwapBuffers();
}

} // namespace Gxm
