// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <psp2/gxm.h>
#include "common/common_types.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_gxm/gxm_memory.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
class PicaCore;
struct RegsInternal;
} // namespace Pica

namespace Gxm {

class TextureCache;

/**
 * Rasterizador del backend GXM.
 *
 * FASE 3, PRIMER TRAMO. La PICA sigue ejecutando su shader de vertices en la
 * CPU (el interprete de siempre); lo que se lleva a la GPU es el RASTERIZADO Y
 * EL SOMBREADO DE FRAGMENTOS: los triangulos que salen del pipeline de
 * geometria se acumulan, se suben a un buffer de vertices y se dibujan con
 * sceGxmDraw usando el shader de fragmentos generado en Cg para la
 * configuracion TEV vigente.
 *
 * COMO ENCAJA SIN TOCAR EL CAMINO DE SOFTWARE. DrawArrays, cuando no puede
 * acelerar el lote entero, ejecuta el shader de vertices por software y llama
 * a AddTriangle por triangulo; este rasterizador NO dibuja en ese momento, solo
 * apunta el triangulo. En DrawTriangles decide: si la configuracion esta
 * soportada, sube el lote a GXM y lo dibuja de una vez; si no, lo reenvia al
 * rasterizador de software (que rasteriza al vuelo), que sigue siendo la
 * referencia de correccion. Antes de dejar que el software toque memoria o
 * estado, la escena GXM se cierra y se vuelca.
 *
 * QUE NO SOPORTA TODAVIA (cae a software, por tanto correcto pero sin ganancia):
 * iluminacion por fragmento, texturas procedurales, shadow maps, scissor,
 * W-buffering, prueba de plantilla y logic op. El motivo de cada rechazo va a
 * crash.txt: es lo unico que se ve desde la consola.
 *
 * LO QUE NO COMPARTE CON EL INVITADO: la profundidad. El buffer del 3DS va en
 * tiles y el nuestro en el formato interno de GXM, asi que un lote que dependa
 * de la profundidad que escribio el camino de software sale mal. El color si
 * viaja en los dos sentidos (ver CopyTiledGuest en el .cpp).
 */
class RasterizerGXM final : public VideoCore::RasterizerInterface {
public:
    RasterizerGXM(VideoCore::RasterizerInterface& software_, Memory::MemorySystem& memory_,
                  Pica::PicaCore& pica_);
    ~RasterizerGXM() override;

    void AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                     const Pica::OutputVertex& v2) override;
    void DrawTriangles() override;

    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;

    void SetAccurateMul(bool accurate_mul) override {
        software.SetAccurateMul(accurate_mul);
    }

    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override {
        software.LoadDefaultDiskResources(stop_loading, callback);
    }

    void SwitchDiskResources(u64 title_id) override {
        software.SwitchDiskResources(title_id);
    }

    /**
     * Cierra la escena abierta y vuelca a memoria del invitado todo lo sucio.
     * Lo llama RendererGXM::SwapBuffers antes de que el renderer de software
     * lea los framebuffers, y cualquier operacion que vaya a tocar la memoria
     * o a mandar trabajo por otro camino.
     */
    void FlushPending();

    /// Triangulos dibujados por la GPU en el intervalo, para el overlay.
    static std::atomic<u32> gpu_triangles;
    /// Triangulos que han caido al rasterizador de software.
    static std::atomic<u32> software_triangles;

    static void ResetCounters() {
        gpu_triangles.store(0, std::memory_order_relaxed);
        software_triangles.store(0, std::memory_order_relaxed);
    }

private:
    struct Surface;
    struct PipelineCache;

    /// Pesquisa un pipeline (y compila el shader si es la primera vez).
    /// Devuelve nullptr si la configuracion no esta soportada.
    const PipelineCache* GetPipeline();

    /// Sube el lote y lo dibuja. Devuelve false si no se ha podido.
    bool DrawBatchOnGpu();

    /// La superficie de dibujado que pide el estado actual, creada si hace
    /// falta. nullptr si no se puede (formato no soportado, sin memoria).
    Surface* CurrentSurface();

    /// Cierra la escena GXM si hay una abierta y espera a que el chip termine.
    void EndScene();

    /// Vuelca la superficie al framebuffer del invitado (de nuestro orden
    /// lineal al de tiles de la PICA). No hace nada si no esta sucia.
    void WriteBack(Surface& surface);

    /// Al reves: trae a la superficie lo que el invitado tenga en ese
    /// framebuffer. Es lo que hace que dibujar encima de lo anterior funcione.
    void Reload(Surface& surface);

    /// Anota UNA vez por motivo el rechazo de un lote (va a crash.txt). Sin
    /// esto, "tg 0" no dice si el problema es el scissor, el shader o el
    /// framebuffer, y se depura a ciegas.
    void NoteSkip(u32 index, const char* reason);

    bool EnsureInitialized();
    void Release();

    VideoCore::RasterizerInterface& software;
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;

    bool initialized = false;
    bool available = false;
    const char* status = "sin inicializar";
    SceGxmContext* context = nullptr;
    SceGxmShaderPatcher* patcher = nullptr;

    std::unique_ptr<PipelineCache> pipelines;
    std::unique_ptr<TextureCache> textures;
    std::vector<std::unique_ptr<Surface>> surfaces;
    Surface* open_surface = nullptr;

    /// Vertices del lote en curso, en el formato de la CPU del invitado.
    std::vector<Pica::OutputVertex> batch;
    /// Decision de soporte para el lote en curso: se toma al primer triangulo.
    bool batch_decided = false;
    bool batch_on_gpu = false;
    /// Motivos de rechazo ya anotados (scissor, wbuffer, shader, framebuffer,
    /// textura/pipeline, plantilla).
    bool skip_noted[6] = {};
    /**
     * Notas del camino de superficie y de dibujado, una por sitio. En orden:
     * formato de color, dimensiones, tramo sin mapear, sin memoria, superficie
     * de color, superficie de profundidad, render target, tope de superficies,
     * comienzo de escena, dibujado, vertices sin memoria e indices sin memoria.
     *
     * Cada sitio tiene la suya A PROPOSITO: con un unico flag, el primer
     * rechazo tapaba a los demas y cada prueba en consola solo derribaba un
     * muro. Ver NoteOnce en el .cpp.
     */
    bool fb_noted[12] = {};

    /// Buffer de vertices mapeado para la GPU (crece cuando hace falta) y su
    /// buffer de indices secuenciales.
    Allocation vertex_buffer;
    Allocation index_buffer;
    /// Bytes ya repartidos del buffer de vertices en la escena en curso. Vuelve
    /// a cero al cerrarla: hasta entonces la GPU sigue leyendo lo de antes.
    u32 vertex_used = 0;
    u32 index_capacity = 0;
};

} // namespace Gxm
