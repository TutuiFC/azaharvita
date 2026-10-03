// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <memory>
#include <psp2/kernel/processmgr.h>
#include <vita2d.h>
#include "core/frontend/emu_window.h"
#include "video_core/renderer_software/renderer_software.h"

namespace Gxm {
class ScreenPresenter;
}

namespace VitaFrontend {

/// Pantalla de la PS Vita.
constexpr u32 kVitaScreenWidth = 960;
constexpr u32 kVitaScreenHeight = 544;

/**
 * Disposicion de las pantallas (0.1.7.1, menu de ajustes "Pantallas"):
 *   0 = normal, 1:1 (superior 400x240 arriba, inferior 320x240 debajo)
 *   1 = superior grande (x1.6) a la izquierda, inferior 1:1 a la derecha
 *   2 = lado a lado (superior x1.2, inferior x1.5)
 *   3 = solo la superior, a x2 (800x480)
 *   4 = solo la superior, a pantalla completa (x2.27)
 * Escalar no cuesta CPU: lo hace la GPU de la Vita al dibujar.
 */
inline std::atomic<int> g_screen_layout{0};
constexpr int kScreenLayoutCount = 5;

/// No volver a subir ni dibujar una imagen que no ha cambiado (0.1.7.1).
inline std::atomic<bool> g_skip_repeated_frames{true};

/**
 * Ventana de emulacion para PS Vita.
 *
 * Las dos pantallas del 3DS se dibujan a escala 1:1 (400x240 arriba, 320x240
 * abajo, 480 px de alto en total) centradas en los 960x544 de la Vita. Escalar
 * costaria un recorrido extra por cada pixel en una CPU que ya va justisima
 * rasterizando por software, y a cambio no se gana nada: a 1:1 la imagen es
 * exacta y cabe de sobra.
 */
class EmuWindow_Vita : public Frontend::EmuWindow {
public:
    EmuWindow_Vita();
    ~EmuWindow_Vita() override;

    void PollEvents() override;

    /// El renderizador software no necesita contexto grafico.
    void MakeCurrent() override {}
    void DoneCurrent() override {}

    /// True cuando el usuario ha pedido salir de la emulacion.
    bool ShouldExit() const {
        return should_exit;
    }

    /// Fotogramas presentados desde que arranco el juego. Sirve para distinguir
    /// "va lento pero avanza" de "esta colgado".
    u64 FramesPresented() const {
        return frames_presented;
    }

    /// Fuente para el overlay de FPS/velocidad. Sin ella, PresentScreens() no
    /// dibuja nada de estadisticas: el arranque llama a esto en cuanto carga la
    /// fuente, pero conviene no asumir que siempre habra una.
    void SetStatsFont(vita2d_pgf* font) {
        stats_font = font;
    }

private:
    void BuildLayout();
    /// Rectangulos de las dos pantallas para una disposicion (ver
    /// g_screen_layout). Ancho 0 en la inferior = no se ve.
    struct ScreenRects {
        int top_x, top_y, top_w, top_h;
        int bottom_x, bottom_y, bottom_w, bottom_h;
    };
    static ScreenRects RectsFor(int layout);
    int layout_mode = 0;
    /// Lo que se presento la ultima vez (ver g_skip_repeated_frames).
    bool has_presented = false;
    unsigned long long last_frame_work = 0;
    u32 last_top_address = 0;
    u32 last_bottom_address = 0;
    const u8* last_top_source = nullptr;
    const u8* last_bottom_source = nullptr;
    bool last_top_fill = false;
    bool last_bottom_fill = false;
    u32 last_fill_colors = 0;
    void PresentScreens();
    void UpdateTouch();
    void DrawStatsOverlay();

    /// Alias corto para no arrastrar el include completo del renderizador.
    using ScreenInfoRef = SwRenderer::ScreenInfo;

    /// Sube el framebuffer del 3DS CRUDO, en su formato nativo. La conversion
    /// la hace la GPU al muestrear. Ver el comentario de ScreenInfo::format.
    static void UploadScreenNative(vita2d_texture*& texture, const ScreenInfoRef& info);

    /// Dibuja esa textura girada 90 grados, que es como esta en memoria.
    static void DrawRotatedScreen(vita2d_texture* texture, const ScreenInfoRef& info, u32 left,
                                  u32 top, u32 draw_w, u32 draw_h);
    /// SELECT + L/R ajusta el salto de fotogramas en caliente.
    void UpdateFrameSkipControl(unsigned int buttons);

    void OnMinimalClientAreaChangeRequest(std::pair<u32, u32> minimal_size) override {}

    vita2d_texture* top_texture = nullptr;
    vita2d_texture* bottom_texture = nullptr;

    /**
     * Presentacion con el chip grafico (sceGxm).
     *
     * Se crea solo cuando la API activa es GXM y se inicializa en su primera
     * presentacion. Si no puede (falta libshacccg.suprx, falla la compilacion
     * de los shaders, no hay memoria), se queda apagado y PresentScreens sigue
     * por vita2d exactamente como antes: es una degradacion silenciosa a
     * proposito. El resultado se ve en la linea "pres" del overlay.
     */
    std::unique_ptr<Gxm::ScreenPresenter> gxm_presenter;
    /// Linea "pres ..." del overlay, ya montada ("pres gxm", "pres vita2d (sin
    /// libshacccg)"). Se rehace en cada fotograma porque el presentador puede
    /// apagarse a mitad de partida si se queda sin memoria de GPU.
    char presenter_line[48] = "pres vita2d (software)";
    /// True cuando el ultimo fotograma lo ha dibujado GXM. Solo cambia el color
    /// de esa linea.
    bool presenter_uses_gxm = false;

    bool should_exit = false;
    bool is_touching = false;
    /// Evita que mantener SELECT+gatillo recorra todo el rango de golpe.
    bool frameskip_combo_held = false;
    /// Igual para L+R, que enciende y apaga la media resolucion.
    bool halfres_combo_held = false;
    /// SELECT + DERECHA: JIT del ARM11 encendido/apagado.
    bool jit_combo_held = false;
    /// SELECT + IZQUIERDA: estirado de audio. Empieza encendido (main.cpp).
    bool stretch_combo_held = false;
    /// SELECT + TRIANGULO: oculta/muestra el overlay de estadisticas (0.1.5.2,
    /// 4.10). ovl cuesta ~2.3 ms por fotograma; con la pantalla limpia se
    /// juega y se mide el coste real de tenerlo puesto.
    bool overlay_combo_held = false;
    /// False = el overlay no se dibuja (SELECT+TRIANGULO lo enciende otra vez).
    bool stats_overlay_visible = true;
    bool audio_stretching = true;
    /// Estadisticas del JIT para el overlay (ver DrawStatsOverlay).
    bool stats_jit_on = false;
    double stats_jit_percent = 0.0;
    double stats_guest_mips = 0.0;
    double stats_arm_ms = 0.0;
    /// Por fotograma (0.1.6.3): rodajas del ARM, despachos (JIT e interprete),
    /// enlaces hechos por el codigo generado, accesos lentos a memoria y
    /// aritmetica VFP. Para saber EN QUE se van los ms de "arm".
    double stats_arm_slices = 0.0;
    double stats_arm_dispatches = 0.0;
    double stats_arm_links = 0.0;
    double stats_arm_slow = 0.0;
    double stats_arm_vfp = 0.0;
    double stats_arm_interp_k = 0.0;
    /// Y CUANTO cuesta cada cosa (0.1.7.5), en ms por fotograma: codigo
    /// generado (con lento y vfp dentro), caminos lentos, VFP por funcion,
    /// comprobaciones, compilacion, y el resto (interprete y despacho).
    double stats_arm_jit_ms = 0.0;
    double stats_arm_slow_ms = 0.0;
    double stats_arm_vfp_ms = 0.0;
    double stats_arm_check_ms = 0.0;
    double stats_arm_compile_ms = 0.0;
    double stats_arm_rest_ms = 0.0;
    unsigned int stats_jit_top_reject = 0;
    double stats_jit_top_reject_percent = 0.0;
    unsigned int stats_jit_blocks = 0;
    unsigned int stats_jit_rejected = 0;
    unsigned int stats_jit_checks = 0;
    unsigned int stats_jit_mismatches = 0;
    u64 frames_presented = 0;

    // Overlay de FPS y velocidad de emulacion.
    //
    // PerfStats calcula estos numeros pero nadie los pedia: los frontends de
    // escritorio los piden con su propio temporizador para la barra de titulo,
    // y aqui no habia ninguno. Se piden una vez por segundo (no en cada
    // fotograma: son promedios sobre el intervalo, pedirlos mas seguido solo
    // da ruido) y se guarda el ultimo valor para dibujarlo aunque la peticion
    // toque en un fotograma distinto al del dibujado.
    vita2d_pgf* stats_font = nullptr;
    double stats_game_fps = 0.0;
    /**
     * FPS MEDIO DE LOS ULTIMOS 5 SEGUNDOS (0.1.6.2). A ~2 FPS, en cada ventana
     * de ~1 s caben 1, 2 o 3 fotogramas enteros y el numero saltaba entre 1.4
     * y 2.9 sin que cambiara nada. Aqui se suman fotogramas y tiempo de las
     * ultimas kFpsWindows ventanas y se divide: una media estable.
     */
    static constexpr int kFpsWindows = 5;
    double fps_frames[kFpsWindows] = {};
    double fps_seconds[kFpsWindows] = {};
    int fps_slot = 0;
    SceUInt64 fps_last_us = 0;
    double stats_speed_percent = 0.0;
    // Reparto del tiempo de cada fotograma, en porcentaje. PerfStats ya lo
    // calculaba y nadie lo miraba: sin esto no hay forma de saber si el tiempo
    // se va emulando la CPU, rasterizando o en llamadas al sistema, y optimizar
    // se convierte en adivinar.
    double stats_cpu_percent = 0.0;
    double stats_gpu_percent = 0.0;
    double stats_svc_percent = 0.0;
    // Por que es lento el rasterizador: triangulos por fotograma, cuantas veces
    // se pinta cada pixel de la pantalla (overdraw) y cuantos pixeles se prueban
    // por cada uno que acaba pintandose. Ver RasterizerStats en sw_rasterizer.h.
    double stats_triangles_per_frame = 0.0;
    double stats_overdraw = 0.0;
    double stats_test_ratio = 0.0;
    // Cobertura de geometria y porcentaje que mata el rechazo temprano por
    // profundidad. Separan "se sombrean pixeles que se tiran" de "cada pixel
    // cuesta demasiado", que se arreglan de formas distintas.
    // Reparto interno del "gpu %": listas de comandos (rasterizado), rellenos de
    // memoria y transferencias de pantalla. Suman 100 entre ellos.
    // Pixeles que pasan por iluminacion, y triangulos que reparten entre hilos.
    /// Nanosegundos de reloj de pared por pixel cubierto, medido directamente.
    double stats_ns_per_pixel = 0.0;
    /// Media de etapas TEV activas por triangulo (1 a 6).
    double stats_tev_stages = 0.0;
    /// Porcentaje de texeles en formato ETC1. Ver RasterizerStats::texels_etc1.
    double stats_etc1_percent = 0.0;
    double stats_lit_percent = 0.0;
    double stats_split_percent = 0.0;
    double stats_gx_cmdlist = 0.0;
    double stats_gx_fill = 0.0;
    double stats_gx_transfer = 0.0;
    double stats_covered = 0.0;
    double stats_zkill_percent = 0.0;
    /// De los pixeles cubiertos, cuantos mueren en la prueba de alfa y cuantos
    /// en la de profundidad final. Decide si toca adelantar pruebas o abaratar
    /// el sombreado.
    double stats_alpha_fail_percent = 0.0;
    double stats_depth_fail_percent = 0.0;
    /// Porcentaje de triangulos en los que el rechazo temprano esta DISPONIBLE.
    /// Distingue "no mata nada" de "ni se intenta" (galga con D24S8).
    double stats_early_z_percent = 0.0;
    /// Desglose del tiempo del rasterizador por triangulo (ver tri_*_us en
    /// sw_rasterizer.h): preparacion, tramo paralelo y trabajo en un solo
    /// hilo. Deciden si el problema es el coste por pixel o el reparto.
    double stats_setup_percent = 0.0;
    double stats_wait_percent = 0.0;
    double stats_single_percent = 0.0;
    /// Ocupacion de los nucleos dentro del tramo paralelo: que porcentaje del
    /// tiempo de los tres se gasta de verdad sombreando. Ver band_busy_us.
    double stats_occupancy_percent = 0.0;
    /// Que parte del tiempo de las listas de comandos ("gx tri") se va dentro
    /// de ProcessTriangle. El resto es sombreado de vertices, recorte y
    /// decodificacion de comandos: trabajo que ninguna optimizacion del bucle
    /// de pixeles toca, y que pone el techo a lo que se puede ganar ahi.
    double stats_raster_share_percent = 0.0;
    /// Porcentaje del fotograma en presentar (swap de vita2d incluido).
    double stats_swap_percent = 0.0;

    /// Triangulos del intervalo que ha rasterizado la GPU y cuantos han caido
    /// al rasterizador de software. Es la medida que dice si el camino nuevo
    /// esta cubriendo la escena o solo un trozo.
    unsigned int stats_tri_gpu = 0;
    unsigned int stats_tri_sw = 0;
    /// Lotes dibujados en la GPU. Con los triangulos da el tamano medio de lote,
    /// que es lo que dice si el coste va por lote o por triangulo.
    unsigned int stats_gpu_batches = 0;
    /// De esos, los del shader de vertices en la GPU (0.1.4.6).
    unsigned int stats_hw_vs_batches = 0;
    /// Los que no pudieron y el ultimo motivo (texto literal, ver HwVsReject).
    unsigned int stats_hw_vs_rejects = 0;
    const char* stats_hw_vs_reason = "-";
    /// Reparto de "cmdlist": el bucle de vertices (carga + interprete + entrega
    /// al ensamblador) y nuestro camino de lotes de GPU. Lo que no sume ninguno
    /// de los dos es decodificacion de comandos.
    double stats_vertices_ms = 0.0;
    double stats_batch_ms = 0.0;
    /// Vertices que han pasado de verdad por el interprete.
    unsigned int stats_vertices_shaded = 0;
    /**
     * Linea "sh" (0.1.0.42): el sombreado repartido entre nucleos, la espera a
     * la GPU al cerrar escenas, las texturas decodificadas (cuantas y cuanto
     * tiempo) y las escenas cerradas por buffer de vertices lleno (ll) o por
     * tablas de luz (lu). Ver Common::FrameStats y RasterizerGXM.
     */
    double stats_shade_ms = 0.0;
    double stats_finish_ms = 0.0;
    double stats_texdecode_ms = 0.0;
    unsigned int stats_texdecodes = 0;
    /// Vaciados de la cache de traduccion del ARM en el ultimo intervalo, y los
    /// totales en la lectura anterior para poder restar. Ver el calculo.
    std::size_t stats_flushes_capacity = 0;
    std::size_t stats_flushes_invalidation = 0;
    std::size_t last_flushes_capacity = 0;
    std::size_t last_flushes_invalidation = 0;
    /// Y por que (0.1.0.43): reutilizadas por hash, cambiadas de verdad,
    /// expulsadas por sitio. Ver Common::FrameStats::texture_reuses.
    unsigned int stats_texreuses = 0;
    unsigned int stats_texchanged = 0;
    unsigned int stats_texevictions = 0;
    /// Ruta rapida del interprete: programas, comprobaciones en el intervalo y
    /// diferencias desde el arranque. Ver Common::FrameStats::fast_programs.
    unsigned int stats_fast_programs = 0;
    unsigned int stats_fast_checks = 0;
    unsigned int stats_fast_mismatches = 0;
    /// Sombreado por dentro (0.1.0.45): ocupacion de los tres nucleos (%),
    /// instrucciones del shader por vertice y % por la ruta rapida.
    double stats_shade_occupancy = 0.0;
    /// Lo que cuesta dibujar el overlay, dentro de 'dib'.
    double stats_overlay_ms = 0.0;
    /// DSP HLE por fotograma (0.1.7.7): mezcla dentro de 'cpu', AAC dentro de 'svc'.
    double stats_dsp_ms = 0.0;
    double stats_instrs_per_vertex = 0.0;
    double stats_fast_percent = 0.0;
    unsigned int stats_close_full = 0;
    unsigned int stats_close_lut = 0;
    /// Escenas cerradas y volcados de framebuffer por intervalo. Ver la nota de
    /// RasterizerGXM::gpu_scenes: los dos son caros y no salian en ninguna linea.
    unsigned int stats_gpu_scenes = 0;
    unsigned int stats_gpu_writebacks = 0;

    /**
     * El mismo reparto, pero en MILISEGUNDOS POR FOTOGRAMA.
     *
     * Todo lo de arriba son porcentajes, y un porcentaje no sirve para comparar
     * dos versiones: si el fotograma entero pasa de 2000 ms a 1000, el reparto
     * puede salir identico y parecer que no ha cambiado nada. El plan de trabajo
     * pide reportar cada fase como "antes -> despues" sobre unas ROMs fijas, y
     * eso son milisegundos absolutos o no es nada.
     *
     * 'frame' es el total (tiempo de pared entre vblanks del invitado); el resto
     * son trozos suyos. cpu/gx/svc vienen de PerfStats; conv/sub/dib/esp de
     * Common::FrameStats, medidos en el sitio exacto donde ocurren.
     */
    double stats_frame_ms = 0.0;
    double stats_cpu_ms = 0.0;
    double stats_gx_ms = 0.0;
    double stats_svc_ms = 0.0;
    /// Framebuffer del 3DS -> ScreenInfo, en el hilo que emula el ARM11.
    double stats_convert_ms = 0.0;
    /// Subida de los pixeles a memoria de la GPU.
    double stats_upload_ms = 0.0;
    /// Encolar el dibujado de las pantallas y el overlay.
    double stats_draw_ms = 0.0;
    /// Espera al intercambio de buffers: consola parada, no trabajo.
    double stats_swapwait_ms = 0.0;
    /// Reparto de unidades de textura por formato, ya formateado ("fmt a8 3 ...").
    char stats_format_line[56] = "fmt -";
    SceUInt64 stats_next_update_us = 0;
};

} // namespace VitaFrontend
