// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/rasterizer_interface.h"

namespace Gxm {

/**
 * Rasterizador del backend GXM.
 *
 * FASE 2: este objeto todavia no rasteriza nada. Reenvia entera su interfaz al
 * rasterizador por software que vive dentro de RendererGXM, para que la PICA se
 * siga emulando exactamente igual que hasta ahora mientras cambia el camino de
 * presentacion. Es lo que el plan llamaba "DrawTriangles puede delegar de
 * momento".
 *
 * POR QUE EXISTE EN VEZ DE DEVOLVER DIRECTAMENTE EL DE SOFTWARE.
 *
 * Porque la PICA ata su rasterizador UNA vez, al arrancar el emulador
 * (PicaCore::BindRasterizer), y a partir de ahi todo pasa por ese puntero. Si el
 * backend GXM devolviese el de software, la Fase 3 tendria que cambiar por
 * dentro el rasterizador de software -- que es la referencia de correccion y no
 * se toca -- o reconstruir la PICA entera. Con esta capa, la Fase 3 sustituye
 * metodo a metodo (DrawTriangles primero, luego los aceleradores, ver abajo)
 * sin que nada de fuera se entere.
 *
 * QUE DELEGA HOY Y QUE NO.
 *
 * Todo lo obligatorio (triangulos, vaciados y invalidaciones de cache) se
 * reenvia tal cual. Los aceleradores -- AccelerateDisplayTransfer, AccelerateFill
 * y AccelerateDrawBatch -- NO se sobrescriben a proposito: se quedan con el
 * "false" de la interfaz, que es lo que manda el trabajo al blitter de software.
 * Cuando la Fase 3 sepa hacerlos en el chip, se anaden aqui uno a uno y el
 * camino de software se queda como red de seguridad.
 *
 * SetAccurateMul SI se reenvia, y por eso en la interfaz es virtual: sin eso,
 * una llamada a traves de un RasterizerInterface* iria a la version base y
 * activaria la multiplicacion exacta en ESTE objeto en vez de en el que de
 * verdad sombrea, dejando al rasterizador de software con el valor contrario al
 * que pide el juego. Ver rasterizer_interface.h.
 */
class RasterizerGXM final : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerGXM(VideoCore::RasterizerInterface& delegate_) : delegate{delegate_} {}

    void AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                     const Pica::OutputVertex& v2) override {
        delegate.AddTriangle(v0, v1, v2);
    }

    void DrawTriangles() override {
        delegate.DrawTriangles();
    }

    void FlushAll() override {
        delegate.FlushAll();
    }

    void FlushRegion(PAddr addr, u32 size) override {
        delegate.FlushRegion(addr, size);
    }

    void InvalidateRegion(PAddr addr, u32 size) override {
        delegate.InvalidateRegion(addr, size);
    }

    void FlushAndInvalidateRegion(PAddr addr, u32 size) override {
        delegate.FlushAndInvalidateRegion(addr, size);
    }

    void ClearAll(bool flush) override {
        delegate.ClearAll(flush);
    }

    void SetAccurateMul(bool accurate_mul) override {
        delegate.SetAccurateMul(accurate_mul);
    }

    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override {
        delegate.LoadDefaultDiskResources(stop_loading, callback);
    }

    void SwitchDiskResources(u64 title_id) override {
        delegate.SwitchDiskResources(title_id);
    }

private:
    VideoCore::RasterizerInterface& delegate;
};

} // namespace Gxm
