// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/renderer_gxm.h"

namespace Gxm {

RendererGXM::RendererGXM(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window)
    : VideoCore::RendererBase{system, window, nullptr}, software{system, pica, window},
      rasterizer{*software.Rasterizer()} {}

RendererGXM::~RendererGXM() = default;

void RendererGXM::SwapBuffers() {
    // El fotograma lo produce el camino de software completo (salto de
    // fotogramas incluido). La diferencia con el renderer de software esta
    // fuera, en como lo presenta EmuWindow_Vita: por GXM si el chip esta
    // disponible y por vita2d si no. Ver Gxm::ScreenPresenter.
    software.SwapBuffers();
}

} // namespace Gxm
