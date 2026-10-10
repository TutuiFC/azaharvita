// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>
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
 * texturas procedurales, shadow maps (tanto el pase como el muestreo),
 * W-buffering y logic op. El motivo de cada rechazo va a crash.txt: es lo unico
 * que se ve desde la consola.
 *
 * LA ILUMINACION POR FRAGMENTO TAMPOCO CAE (desde 0.1.0.17), que era el rechazo
 * que mas juegos en 3D mandaba a software. El modelo entero de la PICA se
 * genera en Cg (cg_fs_shader_gen.cpp): ocho luces, difusa, especular 0 y 1, las
 * seis tablas de busqueda, fresnel y bump mapping. Las tablas viajan en una
 * textura de 256x24 (UpdateLightingLut) y los parametros de las luces en
 * uniforms. Un lote con luz usa el formato de vertice largo, que lleva ademas
 * el cuaternion de normal y el vector de vista.
 *
 * EL SCISSOR YA NO CAE (desde 0.1.0.16). Los dos modos de la PICA se traducen
 * al recorte de region de GXM antes de dibujar el lote: Include a
 * CLIP_OUTSIDE y Exclude a CLIP_INSIDE, con la caja volteada en Y porque la
 * PICA cuenta su Y de pantalla desde abajo. La traduccion entera, con el porque
 * de cada decision, esta en MakeRegionClip (rasterizer_gxm.cpp).
 *
 * LA PLANTILLA TAMPOCO CAE (desde 0.1.0.16). La funcion, el valor de
 * referencia, las dos mascaras y las tres acciones van a
 * sceGxmSetFrontStencilFunc / sceGxmSetBackStencilFunc, con las ocho acciones
 * de la PICA traducidas una a una (MapStencilOp). Solo cuenta con D24S8, que es
 * el unico formato de profundidad del 3DS que guarda plantilla, igual que hace
 * el rasterizador de software.
 *
 * LO QUE NO COMPARTE CON EL INVITADO: la profundidad y la plantilla. El buffer
 * del 3DS va en tiles y el nuestro en el formato interno de GXM, asi que un
 * lote que dependa de la profundidad o de la plantilla que escribio el camino
 * de software sale mal. El color si viaja en los dos sentidos (ver
 * CopyTiledGuest en el .cpp).
 *
 * Lo que si se sigue desde 0.1.0.16 es el BORRADO: cuando el invitado rellena
 * su buffer de profundidad, la superficie se marca y la siguiente escena la
 * rehace con el valor que el invitado ha dejado (ClearDepthIfNeeded). Antes ese
 * relleno no se miraba -- solo se comparaba el tramo del color -- y nuestra
 * profundidad se quedaba con la del fotograma anterior.
 */
class RasterizerGXM final : public VideoCore::RasterizerInterface {
public:
    RasterizerGXM(VideoCore::RasterizerInterface& software_, Memory::MemorySystem& memory_,
                  Pica::PicaCore& pica_);
    ~RasterizerGXM() override;

    void AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                     const Pica::OutputVertex& v2) override;
    void DrawTriangles() override;

    /**
     * SHADER DE VERTICES EN LA GPU (0.1.4.6, la tarea 5 del plan).
     *
     * PicaCore lo pregunta ANTES de ejecutar el interprete: si devuelve true, el
     * lote entero se ha dibujado aqui y la CPU no sombrea ni un vertice. Por eso
     * esto lo valida TODO antes de tocar nada -- topologia, pipeline de
     * fragmentos, superficie, shader de vertices traducido y compilado, programa
     * de fragmentos enlazado, memoria -- y en cualquier duda devuelve false, que
     * deja el lote en el camino de siempre (interprete + DrawBatchOnGpu). Ver la
     * implementacion.
     */
    bool AccelerateDrawBatch(bool is_indexed) override;

    /**
     * COPIA DE PANTALLA EN LA GPU (0.1.8.7). Cada fotograma el juego dibuja en
     * un framebuffer en tiles y lo copia (DisplayTransfer) al de pantalla. Por
     * el camino de software eso es: esperar a la GPU, volcar la superficie a la
     * memoria del invitado, deshacer los tiles, convertir de formato y, al
     * presentar, volver a subirlo todo a la GPU (crash.txt de 0.1.8.6 en el
     * titulo de Zafiro Alfa: 73% de "gx", unos 40 ms por fotograma). Aqui la
     * copia la hace la propia GPU, con un quad, a un buffer suyo para esa
     * pantalla, y el presentador dibuja desde ese buffer. La CPU no espera ni
     * copia nada, y la superficie de origen se puede volver a usar enseguida:
     * la GPU hace las escenas en orden.
     *
     * Solo para las pantallas (las direcciones que el juego ha puesto en la
     * configuracion de los framebuffers): el resto de copias, que si las puede
     * leer la CPU, siguen por software. La memoria del invitado de la pantalla
     * se escribe solo si algo la pide (FlushRegion y compania).
     *
     * 0.1.8.6 hacia lo mismo SIN copiar, apuntando a la superficie de origen,
     * y por eso tenia que hacer la copia de verdad en cuanto el juego volvia a
     * dibujar en ella con la pantalla a la vista: casi siempre.
     */
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;

    /**
     * RELLENOS DE COLOR EN LA GPU (0.1.9.6). Casi todos los juegos borran su
     * superficie de dibujo con un relleno de memoria en cada fotograma. Por
     * software eso era escribir el relleno en la memoria del invitado y, al
     * dibujar despues, ESPERAR a la GPU y recargar la superficie entera desde
     * esa memoria (512 KB en tiles por fotograma en Zafiro Alfa). Si el relleno
     * cubre justo el color de una superficie nuestra, solo se apunta: la
     * siguiente escena empieza pintando un quad de ese color.
     */
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;
    /// Solo apunta (0.2.2.1): las copias de textura siguen por software.
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    /// Rellenos hechos asi (overlay y crash.txt).
    static std::atomic<u32> gpu_fills;
    /**
     * POR QUE UNA SUPERFICIE DEJA DE VALER (0.2.2.1). En Inazuma Eleven GO la
     * copia de la imagen 3D a la pantalla se rechazaba por "superficie por
     * recargar" en cada fotograma, y eso es volcarla esperando a toda la GPU.
     * Cuantas veces se marca para recargar y por que camino, los rellenos que
     * no pudieron ir a la GPU y las copias de textura del juego, para
     * crash.txt (TakeSurfaceSummary).
     */
    static std::array<std::atomic<u32>, 3> reload_causes;
    static std::atomic<u32> soft_fills;
    static std::atomic<u32> soft_fill_kb;
    static std::atomic<u32> lazy_fill_kb;
    static std::atomic<u32> texture_copies;
    static std::atomic<u32> texture_copy_kb;
    static std::atomic<u32> texture_copies_on_gpu;
    [[nodiscard]] static std::string TakeSurfaceSummary();

    /// Coste de AccelerateDrawBatch por fases, muestreado (0.1.9.6). La linea
    /// para crash.txt, y se pone a cero.
    static constexpr u32 kBatchPhases = 6;
    static std::array<unsigned long long, kBatchPhases> batch_phase_us;
    static u32 batch_phase_samples;
    /// Dentro de "estado" (0.2.1.0): abrir la escena y las texturas, que es
    /// donde puede esperar a la GPU o volver a mirar una textura entera.
    static unsigned long long state_scene_us;
    static unsigned long long state_texture_us;
    [[nodiscard]] static std::string TakeBatchProfile();
    /// Veces que un lote por software tuvo que bajar antes lo de la GPU.
    static std::atomic<u32> software_syncs;
    /// Lotes no dibujados porque su shader de fragmentos se compilaba (0.1.9.8).
    static std::atomic<u32> skipped_batches;

    /// Interruptor del menu ("Copia de pantalla en GPU"). Encendido.
    static std::atomic<u32> transfer_on_gpu;
    /**
     * Cada cuantas presentaciones se vuelcan las superficies que nadie ha
     * leido todavia (0.3.1.8, ver FlushForPresent). 1 = en todas, como hasta
     * 0.3.1.7. Clave "volcado_cada=" de ajustes.txt.
     */
    static std::atomic<u32> present_writeback_every;
    /// Copias de textura que son una superficie entera, como copia de mosaico a
    /// mosaico en la GPU (0.3.1.8, ver AccelerateTextureCopy). Apagado desde
    /// 0.3.1.9. Clave "copia_textura_gpu=" de ajustes.txt.
    static std::atomic<u32> texture_copy_gpu;
    /**
     * Resolucion de dibujado en la GPU EN MITADES (ajuste "Resolucion GPU"):
     * 1 = 0.5x (0.2.0.4), 2 = 1x, 4 = 2x (0.2.0.3). Las superficies tienen
     * width * escala / 2 pixeles por eje; la memoria del invitado sigue a 1x
     * (se remuestrea al volcar y al recargar), asi que el juego no nota nada.
     * Se lee al crear cada superficie.
     */
    static std::atomic<u32> resolution_scale;
    /// Copias hechas asi y copias que hubo que hacer de verdad despues (overlay).
    static std::atomic<u32> gpu_transfers;
    static std::atomic<u32> transfer_materialized;
    /// Copias de mosaico a mosaico que se quedan en la GPU de textura (0.2.2.4),
    /// y volcados de su origen porque el juego tambien lo usaba de textura.
    static std::atomic<u32> gpu_texture_transfers;
    static std::atomic<u32> texture_source_writebacks;

    /**
     * Lo que hace SwapBuffers antes de presentar: como FlushPending, pero las
     * superficies cuyo contenido ya se llevo una copia de pantalla en la GPU no
     * se vuelcan (siguen sucias: si algo pide su memoria, se vuelcan entonces).
     */
    void FlushForPresent();

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

    /**
     * Modos de ABLACION del camino de GPU, para bisecar desde la consola.
     *
     * Cada modo obliga a caer al rasterizador de software a los lotes que usen
     * UNA de las cosas que este backend acelera. Como el camino de software es
     * la referencia de correccion, el modo que hace desaparecer un fallo
     * grafico senala cual de las traducciones esta mal, sin tener que compilar
     * una version por hipotesis: desde aqui no se ve la pantalla, y cada prueba
     * cuesta un ciclo entero de compilar, instalar y mirar.
     *
     * kNoGpu no es lo mismo que apagar GXM con SELECT: ahi se cambia el
     * renderizador ENTERO, presentacion incluida. Esto deja el presentador de
     * GXM puesto y solo quita el rasterizado, que es lo que separa "dibuja mal"
     * de "presenta mal".
     */
    struct Ablation {
        static std::atomic<u32> mode;

        static constexpr u32 kNormal = 0;
        static constexpr u32 kNoLighting = 1; ///< la iluminacion por fragmento, a software
        static constexpr u32 kNoScissor = 2;  ///< los lotes con scissor, a software
        static constexpr u32 kNoStencil = 3;  ///< los lotes con plantilla, a software
        static constexpr u32 kNoTexture = 4;  ///< los lotes con texturas, a software
        static constexpr u32 kNoGpu = 5;      ///< todo a software (el presentador sigue en GXM)
        /// El shader de vertices vuelve a la CPU (interprete) y la GPU solo
        /// rasteriza, como hasta 0.1.4.5: para comparar las dos rutas.
        static constexpr u32 kCpuVertexShader = 6;
        static constexpr u32 kMax = 6;

        static const char* Name(u32 value);
    };

    /// Triangulos dibujados por la GPU en el intervalo, para el overlay.
    static std::atomic<u32> gpu_triangles;
    /// Triangulos que han caido al rasterizador de software.
    static std::atomic<u32> software_triangles;
    /**
     * LOTES dibujados en la GPU, que es distinto de triangulos y hace falta
     * para saber donde va el tiempo.
     *
     * Hay un coste FIJO por lote -- resolver la configuracion de fragmentos,
     * buscar el pipeline, poner una veintena de estados de GXM, reservar los
     * uniforms -- y otro por triangulo. Con "tg 3084" a secas no se puede
     * distinguir un fotograma de treinta lotes gordos de uno de mil lotes de
     * tres triangulos, y lo que hay que optimizar en cada caso es lo contrario.
     */
    static std::atomic<u32> gpu_batches;
    /**
     * Escenas cerradas y volcados de framebuffer en el intervalo.
     *
     * Cerrar una escena hace sceGxmFinish, que PARA la CPU hasta que el chip
     * termina; volcar convierte el framebuffer entero entre el formato en
     * mosaicos del invitado y el lineal nuestro. Los dos son caros y los dos
     * pueden ocurrir VARIAS VECES por fotograma sin que se note en ninguna
     * medida: el tiempo aparece repartido dentro de "gx" sin decir de donde
     * sale. Con estos dos numeros se ve de un vistazo si el fotograma esta
     * troceado en escenas o si el framebuffer va y viene mas de la cuenta.
     */
    static std::atomic<u32> gpu_scenes;
    static std::atomic<u32> gpu_writebacks;
    /**
     * De esas escenas, cuantas se cerraron por un motivo que NO es volcar el
     * framebuffer: buffer de vertices lleno (ll) y tablas de luz sin version
     * libre (lu). Deberian ser cero casi siempre desde 0.1.0.42; si no lo son,
     * el overlay lo dice sin tener que adivinar.
     */
    static std::atomic<u32> scene_close_full;
    static std::atomic<u32> scene_close_lut;
    /// Lotes dibujados con el shader de vertices en la GPU (AccelerateDrawBatch).
    static std::atomic<u32> hw_vs_batches;
    /// Lotes que no pudieron, y el ultimo motivo (texto literal). Ver HwVsReject.
    static std::atomic<u32> hw_vs_rejects;
    static std::atomic<const char*> hw_vs_last_reject;
    /**
     * Cuantas veces cada motivo (0.2.1.1): el ultimo solo no decia que lotes
     * van a la CPU en una escena 3D. Lo escribe el hilo de la GPU; los mas
     * frecuentes del intervalo los saca TakeRejectSummary para crash.txt.
     */
    struct RejectCount {
        std::atomic<const char*> reason{nullptr};
        std::atomic<u32> count{0};
    };
    static std::array<RejectCount, 12> reject_counts;
    [[nodiscard]] static std::string TakeRejectSummary();

    /**
     * LAS COPIAS QUE NO VAN A LA GPU Y LOS VOLCADOS, CONTADOS (0.3.1.4). Cada
     * motivo de rechazo de AccelerateDisplayTransfer salia UNA vez en
     * crash.txt, asi que no se sabia si se repetia en cada fotograma; y
     * "volcado" en las esperas era un total sin decir quien lo pedia. Por
     * intervalo, con TakeCopySummary.
     */
    static std::array<RejectCount, 12> transfer_rejects;
    /// La ultima copia rechazada por cada motivo de transfer_rejects (0.3.1.5):
    /// la primera, la de la nota unica, suele ser del arranque y no la de cada
    /// fotograma.
    static std::array<std::array<char, 96>, 12> transfer_reject_detail;
    /// La ultima superficie volcada al presentar (0.3.1.5), por lo mismo.
    static std::array<char, 64> present_writeback_detail;
    enum WritebackSite : u32 {
        kWritebackTextureCopy,
        kWritebackPending,
        kWritebackPresent,
        kWritebackSoftwareBatch,
        kWritebackFlush,
        kWritebackFlushInvalidate,
        kWritebackSites,
    };
    static std::array<std::atomic<u32>, kWritebackSites> writeback_sites;
    [[nodiscard]] static std::string TakeCopySummary();

    /**
     * QUIEN ESPERA A LA GPU (0.2.1.9). "espera a la gpu" de crash.txt era un
     * total: 9-11 ms por fotograma en el 3D de Pokemon Sol, y quitar la espera
     * de la presentacion (0.2.1.8) apenas lo movio. Ahora cada espera se
     * apunta en su motivo, y TakeGpuWaitSummary saca los que pesan, en ms por
     * fotograma ('frames' fotogramas desde la ultima vez).
     */
    enum class GpuWait : u32 {
        Scene,
        WriteBack,
        Reload,
        Present,
        Copy,
        Clear,
        Lut,
        Frame,
        VertexRing,
        Fence,
        Count,
    };
    static std::array<std::atomic<unsigned long long>, static_cast<u32>(GpuWait::Count)>
        gpu_wait_us;
    [[nodiscard]] static std::string TakeGpuWaitSummary(double frames);

    /**
     * 4.5 (0.1.5.2): NO esperar a sceGxmFinish al cerrar cada escena.
     * 1 (por defecto, riesgo medio con interruptor) = EndScene cierra con
     * sceGxmEndScene y deja la espera pendiente; WaitGpu() la paga SOLO
     * cuando alguien vaya a leer la superficie (WriteBack/Reload) o a soltar
     * memoria que la GPU aun usa (ReleaseRetired). 0 = el camino de
     * 0.1.5.1 (Finish al cerrar siempre). Se guarda en ajustes.txt.
     */
    static std::atomic<u32> no_finish_wait;

    /**
     * 4.6 (0.1.5.2): presentar DIRECTAMENTE desde el color_buffer de GXM.
     * 1 (por defecto, riesgo medio con interruptor) = si hay una Surface cuya
     * direccion coincide con la del framebuffer de la pantalla y no esta
     * sucia por otro camino (needs_reload), el presentador usa su
     * color_buffer como textura sin la ida y vuelta WriteBack + subida.
     * 0 = el camino de 0.1.5.1 (siempre por el memoria del invitado).
     */
    static std::atomic<u32> present_direct;

    /**
     * Programas de vertices ESPECIALIZADOS con los booleanos del lote (0.1.7.4).
     * En 0.1.8.1 se apago: en Zafiro Alfa la entrada al 3D pedia varios de
     * golpe, cada uno una compilacion enorme del shader de piel, y el
     * compilador de la consola se quedaba sin memoria y tumbaba la partida
     * (0.1.7.8 a 0.1.8.0). Apagado, esos lotes sombrean en la CPU y el juego
     * bajo a 2.5 FPS (crash.txt de 0.1.8.2: 'no 174 vs salto: especializar').
     * ENCENDIDO otra vez en 0.1.8.3, con la memoria del compilador bajo control
     * (gxm_cg.cpp: kMinFreeToCompileVs). Se guarda en ajustes.txt.
     */
    static std::atomic<u32> specialize_vs;
    /**
     * El anillo de vertices en memoria normal (0.3.1.2, ajuste vertices_ram).
     * Cada lote copia ahi sus vertices con la CPU ("datos", 40-55 us por lote
     * en crash.txt), y la CPU escribe en CDRAM por el bus del chip grafico,
     * sin cache (ver Pool); en memoria normal sin cachear, casi a velocidad
     * normal. Si no hay sitio, CDRAM como antes. 0 = CDRAM primero.
     */
    static std::atomic<u32> vertex_ring_host;
    /// Texturas que pasan a sospechosas, por quien avisa (ver TakeBatchProfile).
    static std::array<std::atomic<u32>, 5> texture_marks;

    /**
     * SHADERS DE VERTICES ASINCRONOS (0.2.1.1). Mientras el shader de vertices
     * de un lote se compila (segundos en esta consola, y en una escena 3D nueva
     * son decenas), el lote se salta, como ya pasaba con los de fragmentos. Lo
     * de antes era sombrearlo en la CPU: en Pokemon Sol, ~25 us por vertice,
     * 3 FPS hasta que acababan de compilar. A cambio, lo que use ese shader
     * aparece un poco despues. Se guarda en ajustes.txt.
     */
    static std::atomic<u32> async_vs;

    /**
     * Lo que el presentador necesita para dibujar una pantalla desde la
     * superficie de la GPU sin copiar. data == nullptr: no hay (usa el
     * camino normal).
     */
    struct DirectPresent {
        const u8* data = nullptr;
        u32 width = 0;
        u32 height = 0;
        u32 stride_bytes = 0;
        /// Formato GXM de la superficie de color (no el del invitado).
        u32 gxm_texture_format = 0;
        /// La de la superficie, en mitades (resolution_scale).
        u32 scale = 2;
    };

    /**
     * Pregunta si hay superficie dibujada para esa direccion fisica (4.6).
     * Paga antes la espera pendiente de la GPU si la hay (4.5): la textura
     * muestrea memoria que el chip podria seguir escribiendo. Llamar desde el
     * hilo que presenta. data == nullptr si no hay o si el interruptor esta
     * apagado.
     *
     * Estatica con instancia unica: el presentador no tiene (ni debe tener)
     * un puntero al rasterizador, y solo hay uno por proceso.
     */
    [[nodiscard]] static DirectPresent QueryDirectPresent(u32 guest_address);

    /// Espera a que la GPU haya terminado la presentacion ANTERIOR (0.2.2.0):
    /// para quien vaya a reescribir con la CPU algo que esa presentacion lee
    /// (las texturas de pantalla que se suben). Ver FlushForPresent.
    static void WaitPreviousPresentation();

    static void ResetCounters() {
        gpu_triangles.store(0, std::memory_order_relaxed);
        software_triangles.store(0, std::memory_order_relaxed);
        gpu_batches.store(0, std::memory_order_relaxed);
        gpu_scenes.store(0, std::memory_order_relaxed);
        gpu_writebacks.store(0, std::memory_order_relaxed);
        scene_close_full.store(0, std::memory_order_relaxed);
        scene_close_lut.store(0, std::memory_order_relaxed);
        hw_vs_batches.store(0, std::memory_order_relaxed);
    }

private:
    struct Surface;
    struct PipelineCache;

    /// Una pantalla copiada por la GPU (ver AccelerateDisplayTransfer).
    struct ScreenCopy;
    std::vector<std::unique_ptr<ScreenCopy>> screen_copies;
    /// El quad que copia: los programas del blit del presentador.
    struct BlitProgram;
    std::unique_ptr<BlitProgram> blit;
    /// El blit se intento crear y fallo: no se reintenta.
    bool blit_failed = false;
    /// Cuenta de copias hechas, para saber cual lleva mas sin usarse.
    u32 copy_clock = 0;
    bool EnsureBlitProgram();
    /// La copia de esa pantalla, con su render target, creada o rehecha si
    /// hace falta. nullptr si no hay memoria.
    ScreenCopy* GetScreenCopy(PAddr dst, u32 width, u32 height, u32 gxm_color_format, u32 bpp,
                              u32 scale, bool tiled);
    /// El quad de la superficie a la copia, en una escena suya.
    bool AccelerateFillOnGpu(const Pica::MemoryFillConfig& config);
    bool FillGuestMemory(const Pica::MemoryFillConfig& config);
    /**
     * RELLENOS DE PROFUNDIDAD SIN ESCRIBIR (0.2.2.7). El borrado de la
     * profundidad lo hace la GPU (depth_needs_clear, que solo lee el primer
     * pixel del invitado); el resto del patron se escribe en la memoria del
     * invitado solo si algo va a leerla (ApplyLazyFills).
     */
    struct LazyFill {
        PAddr start = 0;
        u32 bytes = 0;
        u32 texel = 0;
        u32 bpp = 0;
    };
    std::vector<LazyFill> lazy_fills;
    void ApplyLazyFills(PAddr addr, u32 size);
    /// El relleno por software que esta invalidando su tramo (bpp 0 si
    /// ninguno): sus texturas van por TextureCache::InvalidateFill (0.3.2.6).
    u32 invalidating_fill_texel = 0;
    u32 invalidating_fill_bpp = 0;
    bool BlitToCopy(ScreenCopy& copy, Surface& source, u32 first_row, bool flip,
                    u32 in_width, u32 in_height);
    /// Rellenos pendientes (ver AccelerateFill): el quad del color dentro de la
    /// escena abierta, una escena solo para el, o a mano con la CPU.
    void DrawClearQuad(Surface& surface);
    void FlushClear(Surface& surface);
    void ApplyClearOnCpu(Surface& surface);
    /// Texeles de 1x1 con el color de cada relleno, en anillo.
    Allocation clear_texels;
    /// La imagen a 1x de una superficie escalada, al volcarla o recargarla.
    std::vector<u8> scale_scratch;
    u32 clear_texel_next = 0;
    /// La direccion es una de las pantallas que el juego ha configurado.
    /// Con 'size' y 'row_bytes', tambien si una pantalla empieza dentro de
    /// [addr, addr + size) en una fila entera (0.2.2.5).
    bool IsDisplayFramebuffer(PAddr addr, u32 size = 0, u32 row_bytes = 0);
    /// Las ultimas direcciones vistas en la configuracion de las pantallas:
    /// un juego puede escribir siempre en la misma ranura y alternar la
    /// direccion, y la copia llega antes de que la pantalla nueva se configure.
    /// Con las del ojo derecho desde 0.3.1.4 (ver IsDisplayFramebuffer).
    std::array<PAddr, 16> display_addresses{};
    /// Cuando se vio cada una en la configuracion (VitaMicros), 0.3.1.5.
    std::array<unsigned long long, 16> display_seen_us{};
    u32 display_address_next = 0;
    /// Escribe en la memoria del invitado lo que solo esta en la copia.
    void MaterializeCopy(ScreenCopy& copy);
    void MaterializeCopies(PAddr addr, u32 size);
    void MaterializeAllCopies();
    /// El invitado ha escrito en esa pantalla por otro camino: copia olvidada.
    void DropCopies(PAddr addr, u32 size);
    /**
     * TEXTURAS QUE SE QUEDAN EN LA GPU (0.2.2.4). Una copia de mosaico a
     * mosaico (dont_swizzle) desde una superficie nuestra se hace con el blit y
     * no se escribe en el invitado: si el juego la usa de textura con la misma
     * direccion, medidas y formato, se muestrea su buffer tal cual. Lo que la
     * lea de otra forma (otra textura que la pise, un lote por software, la
     * CPU, una superficie encima) la escribe antes en el invitado.
     */
    const SceGxmTexture* TextureFromCopy(u32 unit);
    /// Las que pisen las texturas de las unidades encendidas, al invitado.
    void MaterializeTextureCopies();
    /// Las que pisen el tramo, al invitado, y ya no valen de textura.
    void DropTiledCopies(PAddr addr, u32 size);
    std::array<SceGxmTexture, 3> copy_textures{};

    /// Pesquisa un pipeline (y compila el shader si es la primera vez).
    /// Devuelve nullptr si la configuracion no esta soportada.
    const PipelineCache* GetPipeline();

    /// Sube el lote y lo dibuja. Devuelve false si no se ha podido.
    bool DrawBatchOnGpu();
    /// Antes de un lote por software en el framebuffer actual (ver el .cpp).
    void PrepareSoftwareBatch();

    /// Estado de dibujado comun a los dos caminos de GPU. Ver el .cpp.
    template <typename Entry>
    bool SetupDrawState(Surface* surface, const Entry* pipeline,
                        const SceGxmVertexProgram* vertex_program,
                        const SceGxmFragmentProgram* fragment_program);

    /// Hueco de 'bytes' (alineado a 16) en el anillo de vertices. Espera solo
    /// si la GPU puede estar leyendo ese tramo (ver el .cpp). nullptr si no
    /// hay memoria.
    u8* ReserveVertexSpace(u32 bytes);

    /// La ablacion activa manda este estado a la CPU (ver Ablation).
    bool AblatedByMode() const;

    /// Shaders de vertices traducidos de la PICA y sus programas enlazados.
    struct HwShaderCache;
    std::unique_ptr<HwShaderCache> hw_shaders;

    /// La superficie de dibujado que pide el estado actual, creada si hace
    /// falta. nullptr si no se puede (formato no soportado, sin memoria).
    Surface* CurrentSurface();

    /// Cierra la escena GXM si hay una abierta. Con no_finish_wait=0 ademas
    /// espera a que el chip termine (camino de 0.1.5.1); con =1 solo cierra
    /// y deja gpu_pending para WaitGpu() (0.1.5.2, 4.5).
    void EndScene();

    /**
     * Paga la espera pendiente de la GPU si la hay (4.5). La llama quien vaya
     * a LEER una superficie o a LIBERAR memoria que el chip podria estar
     * usando. Con no_finish_wait=0 no hace nada (ya se espero en EndScene).
     * Anota el tiempo en finish_us, igual que antes.
     */
    void WaitGpu(GpuWait why);

    /**
     * Deja las tablas de busqueda de la iluminacion listas para la GPU.
     *
     * Las 24 tablas de 256 entradas de la PICA viven en UNA textura de 256x24
     * con dos floats por texel (el valor y la pendiente hasta la entrada
     * siguiente). Se rehacen solo las filas que la PICA haya marcado sucias.
     * Devuelve false si no hay memoria, y entonces el lote cae a software.
     */
    bool UpdateLightingLut();

    /// Vuelca la superficie al framebuffer del invitado (de nuestro orden
    /// lineal al de tiles de la PICA). No hace nada si no esta sucia.
    /// 'site' es quien lo pide, para crash.txt (ver writeback_sites).
    void WriteBack(Surface& surface, u32 site);

    /// Al reves: trae a la superficie lo que el invitado tenga en ese
    /// framebuffer. Es lo que hace que dibujar encima de lo anterior funcione.
    void Reload(Surface& surface);

    /**
     * Deja la profundidad y la plantilla de la superficie como el invitado ha
     * dejado las suyas, si hace falta (surface.depth_needs_clear).
     *
     * Se llama JUSTO ANTES de abrir la escena y no en otro sitio: el borrado lo
     * hace GXM al cargar los tiles, con los valores de fondo de la superficie y
     * la carga forzada apagada, y eso solo se lee en sceGxmBeginScene.
     */
    void ClearDepthIfNeeded(Surface& surface);

    /// Anota UNA vez por motivo el rechazo de un lote (va a crash.txt). Sin
    /// esto, "tg 0" no dice si el problema es el scissor, el shader o el
    /// framebuffer, y se depura a ciegas.
    void NoteSkip(u32 index, const char* reason);

    bool EnsureInitialized();
    void Release();

    VideoCore::RasterizerInterface& software;
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;

    /// Instancia unica para QueryDirectPresent (4.6). Ver el .cpp.
    static RasterizerGXM* s_instance;

    bool initialized = false;
    bool available = false;
    const char* status = "sin inicializar";
    SceGxmContext* context = nullptr;
    SceGxmShaderPatcher* patcher = nullptr;
    /// El parcheador propio (0.1.9.3) y sus tres bloques: buffer, USSE de
    /// vertices y USSE de fragmentos. Ver CreateShaderPatcher.
    SceGxmShaderPatcher* own_patcher = nullptr;
    std::array<SceUID, 3> patcher_blocks{-1, -1, -1};
    std::array<void*, 3> patcher_memory{};
    bool CreateShaderPatcher();
    void DestroyShaderPatcher();

    std::unique_ptr<PipelineCache> pipelines;
    /// Lo que decidio el lote anterior, mientras la PICA no toque los registros
    /// que lo deciden (0.2.1.0). Ver BatchMemo en el .cpp.
    struct BatchMemo;
    std::unique_ptr<BatchMemo> batch_memo;
    std::unique_ptr<TextureCache> textures;
    std::vector<std::unique_ptr<Surface>> surfaces;
    Surface* open_surface = nullptr;
    /// Hay un sceGxmEndScene sin su sceGxmFinish (4.5). Ver WaitGpu.
    bool gpu_pending = false;

    /**
     * VALLAS DE LA GPU (0.1.9.4). Cada escena del rasterizador, al cerrarse,
     * pide a la GPU que escriba su numero en la region de notificaciones de
     * GXM cuando acabe sus fragmentos. Las escenas acaban en orden, asi que el
     * valor escrito es la ultima terminada: con eso cada recurso espera solo a
     * la escena que lo leyo, y no a toda la GPU como sceGxmFinish.
     */
    volatile unsigned int* fence_address = nullptr;
    u32 fence_sent = 0;
    /// La primera valla del fotograma en curso (ver FlushForPresent).
    u32 frame_first_fence = 0;
    /// Las primeras vallas de los dos fotogramas anteriores (0.2.2.0): [0] el
    /// ultimo, [1] el de antes. 0 = no hubo escena en ese fotograma.
    std::array<u32, 2> present_fences{};
    u32 presents_seen = 0;
    /// La del fotograma que se esta presentando: acabada, la presentacion
    /// anterior tambien (WaitPreviousPresentation).
    u32 present_fence_now = 0;
    [[nodiscard]] bool FenceDone(u32 fence) const;
    void WaitFence(u32 fence, GpuWait why);

    /// Vertices del lote en curso, en el formato de la CPU del invitado.
    std::vector<Pica::OutputVertex> batch;
    /// Decision de soporte para el lote en curso: se toma al primer triangulo.
    bool batch_decided = false;
    bool batch_on_gpu = false;
    /// El lote en curso se salta: su shader de fragmentos se compila (0.1.9.8).
    bool batch_skip = false;
    /**
     * Motivos de rechazo ya anotados, uno por hueco y una sola vez cada uno. En
     * orden: 0 libre (era el scissor, que ya no rechaza), 1 wbuffer, 2 shader,
     * 3 framebuffer, 4 textura/pipeline y 5 plantilla.
     *
     * El hueco 0 se deja vacio a proposito en vez de recolocar los demas: los
     * numeros no salen a ninguna parte (NoteSkip escribe el texto, no el
     * indice), y renumerar solo invita a que dos sitios acaben compartiendo
     * hueco y uno tape al otro, que es justo lo que este array evita.
     */
    bool skip_noted[6] = {};

    /**
     * Motivos de rechazo ya escritos, por TEXTO y no por hueco.
     *
     * Los motivos son literales de cadena (del generador de shaders o de aqui),
     * asi que guardar el puntero basta para no repetirlos; se comparan por
     * contenido de todas formas, que es lo que no puede equivocarse. Ver
     * NoteSkip: con un solo hueco por familia, el primer motivo tapaba a los
     * demas y la causa real de un fallo podia no llegar nunca a crash.txt.
     */
    static constexpr u32 kMaxSkipReasons = 16;
    const char* skip_reasons[kMaxSkipReasons] = {};
    u32 skip_reason_count = 0;
    /**
     * Notas del camino de superficie y de dibujado, una por sitio. En orden:
     * formato de color, dimensiones, tramo sin mapear, sin memoria, superficie
     * de color, superficie de profundidad, render target, tope de superficies,
     * comienzo de escena, dibujado, vertices sin memoria, indices sin memoria y
     * tablas de iluminacion (12; el 13 queda libre).
     *
     * Cada sitio tiene la suya A PROPOSITO: con un unico flag, el primer
     * rechazo tapaba a los demas y cada prueba en consola solo derribaba un
     * muro. Ver NoteOnce en el .cpp.
     */
    bool fb_noted[14] = {};

    /**
     * Las tablas de busqueda de la iluminacion, en una textura de 256x24 con
     * dos floats por texel. Se crea la primera vez que un shader las pide.
     *
     * Hay kLutVersions COPIAS seguidas en el mismo bloque, cada una con su
     * descriptor: cuando el juego cambia una tabla y la escena abierta ya tiene
     * dibujados apuntados que leen la version actual, se pasa a la siguiente en
     * vez de cerrar la escena. Ver UpdateLightingLut.
     *
     * 32 desde 0.3.2.5 (eran 8). New Super Mario Bros. 2 cambia de tablas
     * mas de 8 veces por escena: cada 8 se cerraba la escena y el siguiente
     * cambio esperaba a que la GPU la terminara entera ("esperas gpu: tablas
     * de luz" 7-14 ms por fotograma; 4 ms en Pokemon Sol). Son 1,5 MB.
     */
    static constexpr u32 kLutVersions = 32;
    static constexpr u32 kLutTables = 24;
    /// La valla de la ultima escena que leyo cada version (0.1.9.4).
    std::array<u32, kLutVersions> lut_fence{};
    /// Generacion de cada tabla en lighting_lut_shadow, y la que tiene escrita
    /// cada version (0.3.2.5): al estrenar una version solo se copian las
    /// tablas que no tiene al dia, no las 24 (48 KB).
    std::array<u32, kLutTables> lighting_lut_gen{};
    std::array<std::array<u32, kLutTables>, kLutVersions> lighting_lut_version_gen{};
    u32 lighting_lut_gen_counter = 0;
    Allocation lighting_lut_buffer;
    /// Las 24 tablas ya convertidas, en memoria normal (0.2.1.1): ver
    /// UpdateLightingLut.
    std::vector<f32> lighting_lut_shadow;
    std::array<SceGxmTexture, kLutVersions> lighting_lut_textures{};
    /// La version que se ata a los dibujados a partir de ahora.
    u32 lighting_lut_version = 0;
    /// Versiones ANTERIORES a la actual que la escena abierta todavia lee.
    u32 lighting_lut_retired_in_scene = 0;
    bool lighting_lut_ready = false;
    /**
     * La escena abierta ya le ha atado la textura de tablas a algun dibujado.
     *
     * Sirve para no cerrar la escena cuando no hace falta: rehacer las tablas
     * solo obliga a esperar a la GPU si hay un dibujado apuntado que todavia
     * tiene que leerlas. Sin esto, un juego que cambie de material a menudo
     * pagaba un sceGxmFinish por cada cambio aunque la escena en curso no
     * tuviera nada iluminado.
     */
    bool lighting_lut_in_scene = false;

    /// Buffer de vertices mapeado para la GPU (crece cuando hace falta) y su
    /// buffer de indices secuenciales.
    Allocation vertex_buffer;
    Allocation index_buffer;
    /// Bytes ya repartidos del buffer de vertices. Desde 0.1.9.4 es un anillo:
    /// no vuelve a cero al cerrar la escena, solo al llegar al final.
    u32 vertex_used = 0;
    /// El anillo, en kVertexSegments tramos (0.2.1.1): la valla de la ultima
    /// escena que leyo cada uno, y los que tienen datos de la escena abierta.
    static constexpr u32 kVertexBufferBytes = 8u * 1024u * 1024u;
    static constexpr u32 kVertexSegments = 8;
    std::array<u32, kVertexSegments> vertex_segment_fence{};
    u32 vertex_segments_pending = 0;
    u32 index_capacity = 0;
    /// El lote en curso es de los que se cronometran por fases.
    bool profile_state = false;
};

} // namespace Gxm
