// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <psp2/gxm.h>
#include <psp2/shacccg.h>
#include "common/common_types.h"
#include "video_core/renderer_gxm/gxm_memory.h"
#include "video_core/renderer_gxm/gxm_pica_format.h"
#include "video_core/renderer_software/renderer_software.h"

namespace Gxm {

/// Los dos shaders del quad texturizado del presentador. La copia de pantalla
/// del rasterizador (0.1.8.7) usa los mismos: con el mismo texto, la cache de
/// shaders de la tarjeta los sirve sin compilar.
extern const char kBlitVertexSource[];
extern const char kBlitFragmentSource[];

/**
 * Presenta las dos pantallas del 3DS con el chip grafico de la consola.
 *
 * QUE HACE Y QUE NO. Coge los ScreenInfo que produce el renderizador de
 * software (ya en el formato nativo del 3DS), los sube a memoria que la GPU
 * puede leer y dibuja dos quads texturizados con sceGxmDraw. No toca el
 * rasterizado: la PICA se sigue emulando en la CPU en esta fase. Lo que cambia
 * con respecto al camino de vita2d es quien dibuja el resultado y con que
 * texturas.
 *
 * POR QUE USA EL CONTEXTO DE VITA2D Y NO UNO PROPIO.
 *
 * GXM admite un solo contexto por proceso. El que existe es el de vita2d, que
 * ademas es quien abre la escena (sceGxmBeginScene), la cierra y encola el
 * fotograma. Crear otro contexto no es posible, y hacerlo "en paralelo" tampoco
 * aportaria nada: lo que hace falta es meter los dos quads DENTRO de la misma
 * escena en la que vita2d dibuja el overlay, y eso es exactamente lo que
 * permiten su contexto y su patcher. vita2d reimprime sus programas y su stream
 * en cada dibujado, asi que los cambios de estado propios no le afectan.
 *
 * DE DONDE SALEN LOS SHADERS. No hay compilador Cg offline en el VitaSDK, asi
 * que los dos programas (vertices y fragmentos) se compilan EN LA CONSOLA con
 * SceShaccCg, que vive en libshacccg.suprx -- el fichero que el usuario extrae
 * de su propia consola, como en cualquier homebrew que use shaders. Si no esta,
 * o si la compilacion falla, Ready() se queda en false, Status() dice por que y
 * el frontend presenta con vita2d exactamente como antes. Nada de esto es
 * obligatorio para arrancar.
 *
 * POR QUE SE PUEDE DIBUJAR SIN TOCAR EL RESTO DEL ESTADO. Lo que se cambia
 * (cull, prueba de profundidad) se deja en el estado que necesita el dibujado
 * 2D: sin culling, sin escritura de profundidad y con la prueba siempre
 * superada. Es, con muy pocas dudas, el estado con el que ya trabajaba vita2d;
 * a diferencia de la mezcla, que va dentro del programa de fragmentos, estos
 * son estados del contexto y no se pueden leer hacia atras, asi que se dejan en
 * el valor que no puede romper a nadie.
 */
class ScreenPresenter {
public:
    /// Rectangulo en pixeles de la pantalla de la Vita (960x544).
    struct Rect {
        int left;
        int top;
        int width;
        int height;
    };

    /**
     * Las medidas de la pantalla fisica hacen falta para pasar de pixeles a
     * coordenadas de recorte, y el contexto y el patcher son los de vita2d
     * (GXM solo admite un contexto, ver arriba). Los tres los conoce el
     * frontend, que los pasa aqui para que video_core no dependa de vita2d.
     *
     * El contexto y el patcher NO son de este objeto: no se liberan al
     * destruirlo.
     */
    ScreenPresenter(float screen_width, float screen_height, SceGxmContext* gxm_context,
                    SceGxmShaderPatcher* shader_patcher);
    ~ScreenPresenter();

    ScreenPresenter(const ScreenPresenter&) = delete;
    ScreenPresenter& operator=(const ScreenPresenter&) = delete;

    /**
     * Dibuja las dos pantallas. En la primera llamada intenta inicializarse
     * (cargar el compilador, compilar los shaders, reservar memoria); si eso
     * falla, las siguientes llamadas no hacen nada y Ready() dice que no.
     *
     * Tiene que llamarse CON la escena de vita2d ya abierta: los dibujados se
     * encolan en esa misma escena.
     */
    void Draw(const SwRenderer::ScreenInfo& top, const SwRenderer::ScreenInfo& bottom, Rect top_rect,
              Rect bottom_rect);

    /**
     * Intenta inicializarse una sola vez: carga libshacccg, compila los dos
     * shaders y reserva la memoria. Idempotente. Draw() la llama solo; existe
     * para que el frontend pueda saber si hay GXM ANTES de decidir por donde
     * presenta el fotograma (Ready() solo es valido despues de intentarlo).
     */
    void Initialize();

    /// False mientras no se haya podido inicializar. La primera llamada a
    /// Draw() lo decide.
    [[nodiscard]] bool Ready() const noexcept {
        return ready;
    }

    /// Estado corto para el overlay y crash.txt: "gxm", "vita2d (sin
    /// libshacccg)", "vita2d (error de shader)"...
    [[nodiscard]] const char* Status() const noexcept {
        return status;
    }

private:
    /// Una pantalla subida a memoria de GPU, con la descripcion de su textura.
    struct Screen {
        SceGxmTexture texture{};
        Allocation buffer;
        u32 width = 0;
        u32 height = 0;
        u32 stride = 0;
        Pica::PixelFormat format = Pica::PixelFormat::RGBA8;
        /**
         * La textura apunta AL color_buffer de una Surface de GXM (4.6), no a
         * 'buffer'. En ese caso 'buffer' puede estar vacio y no se copia nada:
         * hay que rehacer la textura cuando se vuelva al camino normal o cuando
         * cambien las medidas.
         */
        bool points_at_gxm_surface = false;
    };

    /// Que hay que dibujar en una pantalla este fotograma.
    enum class Source {
        Nothing, ///< Sin imagen valida: no se dibuja nada (se ve el fondo).
        Texture, ///< Fotograma del juego.
        Fill,    ///< Color plano pedido por el juego (fundidos).
    };

    bool Init();
    void Release();

    /// Compila un shader con SceShaccCg. Devuelve el resultado (que hay que
    /// conservar: el GXP vive dentro) o nullptr, ya con el diagnostico anotado.
    const SceShaccCgCompileOutput* CompileShader(SceShaccCgTargetProfile profile, const char* name,
                                                 const char* source);

    /// Sube la pantalla si hace falta y dice de donde hay que dibujarla.
    Source PrepareScreen(Screen& screen, const SwRenderer::ScreenInfo& info);

    /// Escribe los cuatro vertices del quad (orden de triangle strip: arriba
    /// izquierda, arriba derecha, abajo izquierda, abajo derecha).
    void WriteQuad(float* out, Rect rect, float u0, float v0, float u1, float v1) const;

    [[nodiscard]] float NdcX(int pixel_x) const;
    [[nodiscard]] float NdcY(int pixel_y) const;

    float screen_width;
    float screen_height;

    SceGxmContext* context = nullptr;
    SceGxmShaderPatcher* patcher = nullptr;

    bool attempted = false;
    bool ready = false;
    const char* status = "sin inicializar";

    /// Modulo de libshacccg, para descargarlo al terminar.
    SceUID module_id = -1;
    SceShaccCgCallbackList shader_callbacks{};

    SceGxmShaderPatcherId vertex_program_id{};
    SceGxmShaderPatcherId fragment_program_id{};
    bool vertex_program_registered = false;
    bool fragment_program_registered = false;
    SceGxmVertexProgram* vertex_program = nullptr;
    SceGxmFragmentProgram* fragment_program = nullptr;
    /// Indice del recurso de textura del programa de fragmentos. Se pregunta al
    /// GXP en vez de dar por hecho que es 0: si algun dia el shader declara mas
    /// de una, esto sigue siendo correcto.
    unsigned int texture_unit = 0;

    /// Copias compiladas del GXP. No se pueden liberar mientras el patcher
    /// tenga los programas registrados.
    const SceShaccCgCompileOutput* vertex_output = nullptr;
    const SceShaccCgCompileOutput* fragment_output = nullptr;

    /// Vertices y indices, en memoria mapeada para la GPU.
    Allocation vertices;
    Allocation indices;

    Screen screens[2];
    Allocation fill_texel;
    SceGxmTexture fill_texture{};
};

} // namespace Gxm
