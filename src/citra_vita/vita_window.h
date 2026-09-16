// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <psp2/kernel/processmgr.h>
#include <vita2d.h>
#include "core/frontend/emu_window.h"
#include "video_core/renderer_software/renderer_software.h"

namespace VitaFrontend {

/// Pantalla de la PS Vita.
constexpr u32 kVitaScreenWidth = 960;
constexpr u32 kVitaScreenHeight = 544;

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

    /// Copia un buffer de ScreenInfo a una textura de vita2d.
    static void UploadScreen(vita2d_texture* texture, const std::vector<u8>& pixels, u32 src_width,
                             u32 src_height);

    void OnMinimalClientAreaChangeRequest(std::pair<u32, u32> minimal_size) override {}

    vita2d_texture* top_texture = nullptr;
    vita2d_texture* bottom_texture = nullptr;

    bool should_exit = false;
    bool is_touching = false;
    /// Evita que mantener SELECT+gatillo recorra todo el rango de golpe.
    bool frameskip_combo_held = false;
    /// Igual para L+R, que enciende y apaga la media resolucion.
    bool halfres_combo_held = false;
    /// Y para SELECT+ARRIBA, que recorre los modos de ablacion (diagnostico).
    bool ablation_combo_held = false;
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
