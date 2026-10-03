// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
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

    /// Interruptor del menu ("Copia de pantalla en GPU"). Encendido.
    static std::atomic<u32> transfer_on_gpu;
    /// Copias hechas asi y copias que hubo que hacer de verdad despues (overlay).
    static std::atomic<u32> gpu_transfers;
    static std::atomic<u32> transfer_materialized;

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
    ScreenCopy* GetScreenCopy(PAddr dst, u32 width, u32 height, u32 gxm_color_format);
    /// El quad de la superficie a la copia, en una escena suya.
    bool BlitToCopy(ScreenCopy& copy, const Surface& source, u32 first_row, bool flip);
    /// La direccion es una de las pantallas que el juego ha configurado.
    bool IsDisplayFramebuffer(PAddr addr);
    /// Las ultimas direcciones vistas en la configuracion de las pantallas:
    /// un juego puede escribir siempre en la misma ranura y alternar la
    /// direccion, y la copia llega antes de que la pantalla nueva se configure.
    std::array<PAddr, 8> display_addresses{};
    u32 display_address_next = 0;
    /// Escribe en la memoria del invitado lo que solo esta en la copia.
    void MaterializeCopy(ScreenCopy& copy);
    void MaterializeCopies(PAddr addr, u32 size);
    void MaterializeAllCopies();
    /// El invitado ha escrito en esa pantalla por otro camino: copia olvidada.
    void DropCopies(PAddr addr, u32 size);

    /// Pesquisa un pipeline (y compila el shader si es la primera vez).
    /// Devuelve nullptr si la configuracion no esta soportada.
    const PipelineCache* GetPipeline();

    /// Sube el lote y lo dibuja. Devuelve false si no se ha podido.
    bool DrawBatchOnGpu();

    /// Estado de dibujado comun a los dos caminos de GPU. Ver el .cpp.
    template <typename Entry>
    bool SetupDrawState(Surface* surface, const Entry* pipeline,
                        const SceGxmVertexProgram* vertex_program,
                        const SceGxmFragmentProgram* fragment_program);

    /// Hueco de 'bytes' (alineado a 16) en el buffer de vertices de la escena.
    /// Si no cabe, cierra la escena y empieza de cero. nullptr si no cabe ni
    /// en un buffer vacio o no hay memoria.
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
    void WaitGpu();

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
    void WriteBack(Surface& surface);

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
    [[nodiscard]] bool FenceDone(u32 fence) const;
    void WaitFence(u32 fence);

    /// Vertices del lote en curso, en el formato de la CPU del invitado.
    std::vector<Pica::OutputVertex> batch;
    /// Decision de soporte para el lote en curso: se toma al primer triangulo.
    bool batch_decided = false;
    bool batch_on_gpu = false;
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
     */
    static constexpr u32 kLutVersions = 8;
    /// La valla de la ultima escena que leyo cada version (0.1.9.4).
    std::array<u32, kLutVersions> lut_fence{};
    Allocation lighting_lut_buffer;
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
    /// no vuelve a cero al cerrar la escena, solo al llegar al final, y ahi se
    /// espera a la GPU entera (una vez cada 4 MB de vertices).
    u32 vertex_used = 0;
    u32 index_capacity = 0;
};

} // namespace Gxm
