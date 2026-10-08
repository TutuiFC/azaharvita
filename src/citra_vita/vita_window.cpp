// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <psp2/ctrl.h>
#include <psp2/touch.h>
#include <fmt/format.h>
#include <string>
#include "citra_vita/vita_version.h"
#include "citra_vita/vita_window.h"
#include "common/settings.h"
#include "audio_core/dsp_interface.h"
#include "audio_core/hle/dsp_stats.h"
#include "core/3ds.h"
#include "core/arm/dyncom/arm_dyncom_jit.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "common/vita_diag.h"
#include "common/vita_threads.h"
#include "core/core.h"
#include "core/vita_loop_profile.h"
#include "video_core/gpu.h"
#include "video_core/renderer_gxm/gxm_cg.h"
#include "video_core/renderer_gxm/gxm_pica_format.h"
#include "video_core/renderer_gxm/gxm_presenter.h"
#include "video_core/renderer_gxm/rasterizer_gxm.h"
#include "video_core/renderer_gxm/renderer_gxm.h"
#include "video_core/renderer_software/renderer_software.h"
#include "video_core/renderer_software/sw_rasterizer.h"
#include "video_core/shader/shader_neon_jit.h"

namespace VitaFrontend {

namespace {
// Posicion de cada pantalla del 3DS dentro de los 960x544 de la Vita, a 1:1.
constexpr u32 kTopWidth = static_cast<u32>(Core::kScreenTopWidth);        // 400
constexpr u32 kTopHeight = static_cast<u32>(Core::kScreenTopHeight);      // 240
constexpr u32 kBottomWidth = static_cast<u32>(Core::kScreenBottomWidth);  // 320
constexpr u32 kBottomHeight = static_cast<u32>(Core::kScreenBottomHeight);// 240

constexpr u32 kTotalHeight = kTopHeight + kBottomHeight;                  // 480
constexpr u32 kMarginY = (kVitaScreenHeight - kTotalHeight) / 2;          // 32

constexpr u32 kTopLeft = (kVitaScreenWidth - kTopWidth) / 2;              // 280
constexpr u32 kTopTop = kMarginY;                                        // 32
constexpr u32 kBottomLeft = (kVitaScreenWidth - kBottomWidth) / 2;        // 320
constexpr u32 kBottomTop = kMarginY + kTopHeight;                         // 272

// El panel tactil frontal de la Vita reporta en su propio sistema de
// coordenadas, mas fino que los pixeles de pantalla.
constexpr int kTouchMaxX = 1919;
constexpr int kTouchMaxY = 1087;

/// Monta una linea de texto de tamano fijo: 'prefix' y luego 'source'. Se hace
/// a mano y no con snprintf porque esta libc no garantiza que la variante que
/// enlaza vita2d entienda todos los formatos, y esto es una copia y ya.
void BuildLine(char* out, std::size_t capacity, const char* prefix, const char* source) {
    std::size_t n = 0;
    for (std::size_t i = 0; prefix[i] != '\0' && n + 1 < capacity; i++) {
        out[n++] = prefix[i];
    }
    for (std::size_t i = 0; source[i] != '\0' && n + 1 < capacity; i++) {
        out[n++] = source[i];
    }
    out[n] = '\0';
}
} // Anonymous namespace

EmuWindow_Vita::EmuWindow_Vita() {
    sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG_WIDE);
    sceTouchSetSamplingState(SCE_TOUCH_PORT_FRONT, SCE_TOUCH_SAMPLING_STATE_START);

    // Las dos texturas de vita2d se crean de forma perezosa, en
    // UploadScreenNative, cuando el fotograma se presenta por ese camino. Con
    // la presentacion por GXM no se usan, y crearlas aqui serian ~1 MB de CDRAM
    // ocupados para nada (vita2d las reserva en CDRAM por defecto).
    layout_mode = g_screen_layout.load(std::memory_order_relaxed);
    BuildLayout();
}

EmuWindow_Vita::~EmuWindow_Vita() {
    if (top_texture != nullptr) {
        vita2d_free_texture(top_texture);
    }
    if (bottom_texture != nullptr) {
        vita2d_free_texture(bottom_texture);
    }
    if (overlay_texture != nullptr) {
        vita2d_free_texture(overlay_texture);
    }
}

EmuWindow_Vita::ScreenRects EmuWindow_Vita::RectsFor(int layout) {
    switch (layout) {
    case 1: // superior x1.6 a la izquierda, inferior 1:1 a la derecha
        return {0, 80, 640, 384, 640, 152, 320, 240};
    case 2: // lado a lado: superior x1.2, inferior x1.5
        return {0, 128, 480, 288, 480, 92, 480, 360};
    case 3: // solo superior, x2
        return {80, 32, 800, 480, 0, 0, 0, 0};
    case 4: // solo superior, pantalla completa (x2.27, misma proporcion)
        return {27, 0, 906, 544, 0, 0, 0, 0};
    default: // normal, 1:1
        return {static_cast<int>(kTopLeft), static_cast<int>(kTopTop),
                static_cast<int>(kTopWidth), static_cast<int>(kTopHeight),
                static_cast<int>(kBottomLeft), static_cast<int>(kBottomTop),
                static_cast<int>(kBottomWidth), static_cast<int>(kBottomHeight)};
    }
}

void EmuWindow_Vita::BuildLayout() {
    // La disposicion del menu de ajustes (0.1.7.1). El tactil usa estos mismos
    // rectangulos, asi que sigue a la pantalla inferior donde este.
    const ScreenRects rects = RectsFor(layout_mode);
    Layout::FramebufferLayout layout{};
    layout.width = kVitaScreenWidth;
    layout.height = kVitaScreenHeight;
    layout.top_screen_enabled = true;
    layout.bottom_screen_enabled = rects.bottom_w > 0;
    layout.is_rotated = true;
    layout.top_screen = Common::Rectangle<u32>{
        static_cast<u32>(rects.top_x), static_cast<u32>(rects.top_y),
        static_cast<u32>(rects.top_x + rects.top_w), static_cast<u32>(rects.top_y + rects.top_h)};
    layout.bottom_screen = Common::Rectangle<u32>{
        static_cast<u32>(rects.bottom_x), static_cast<u32>(rects.bottom_y),
        static_cast<u32>(rects.bottom_x + rects.bottom_w),
        static_cast<u32>(rects.bottom_y + rects.bottom_h)};

    NotifyFramebufferLayoutChanged(layout);
}

void EmuWindow_Vita::PollEvents() {
    // El renderizador software no llama a SwapBuffers(): los frontends de
    // escritorio tienen un hilo aparte que va leyendo screen_infos. Aqui no lo
    // hay, asi que se presenta desde PollEvents(), a la que se llama desde
    // EndFrame() justo despues de que PrepareRenderTarget haya rellenado los
    // buffers de ambas pantallas.
    PresentScreens();
    UpdateTouch();

    SceCtrlData pad{};
    sceCtrlPeekBufferPositive(0, &pad, 1);

    // START + SELECT a la vez sale de la emulacion y vuelve al menu de ROMs.
    // Hacen falta las dos porque cada una por separado es un boton del 3DS.
    if ((pad.buttons & SCE_CTRL_START) && (pad.buttons & SCE_CTRL_SELECT)) {
        should_exit = true;
    }

    UpdateFrameSkipControl(pad.buttons);
}

void EmuWindow_Vita::UpdateFrameSkipControl(unsigned int buttons) {
    // SELECT + L baja el salto de fotogramas, SELECT + R lo sube.
    //
    // Se ajusta en caliente a proposito: cada recompilacion son ~15 minutos, y
    // el valor bueno depende del juego y hasta de la escena. Asi se busca el
    // punto probando, no recompilando.
    //
    // La combinacion lleva SELECT porque L y R por separado son botones del
    // 3DS. SELECT tambien lo es, pero SELECT+gatillo no es una pulsacion que
    // haga ningun juego.
    const bool select = (buttons & SCE_CTRL_SELECT) != 0;
    const bool l = (buttons & SCE_CTRL_LTRIGGER) != 0;
    const bool r = (buttons & SCE_CTRL_RTRIGGER) != 0;

    // L+R a la vez (sin SELECT) cambia entre resolucion 1x y 0.5x (0.1.5.1: antes era
    // solo media resolucion vertical). Lo mismo que el menu de ajustes.
    // Es un interruptor, no un ajuste: se pulsa una vez para encender y otra
    // para apagar. No choca con el ajuste de salto de fotogramas, que pide
    // SELECT ademas de uno solo de los dos gatillos.
    const bool half_combo = l && r && !select;
    if (half_combo) {
        if (!halfres_combo_held) {
            halfres_combo_held = true;
            const bool current =
                SwRenderer::FrameSkip::half_resolution.load(std::memory_order_relaxed);
            SwRenderer::FrameSkip::half_resolution.store(!current, std::memory_order_relaxed);
            Common::VitaNote("tecla", current ? "L+R: resolucion 1x" : "L+R: resolucion 0.5x");
        }
        return;
    }
    halfres_combo_held = false;

    /**
     * SELECT + ARRIBA y SELECT + ABAJO ya no cambian la ablacion (0.1.8.7). Eran
     * modos de diagnostico que rompen la imagen o mandan lotes a software a
     * proposito, y en la partida de 0.1.8.6 se activo uno sin querer: el 3D de
     * Zafiro Alfa cayo a 1 FPS con "gxm skip: ablacion" en crash.txt.
     */

    // SELECT + DERECHA enciende y apaga el JIT del ARM11 (0.1.4.8), para
    // comparar en la misma escena con y sin el. Ver arm_dyncom_jit.h.
    const bool jit_toggle = select && ((buttons & SCE_CTRL_RIGHT) != 0);
    if (jit_toggle) {
        if (!jit_combo_held) {
            jit_combo_held = true;
            const u32 current = Core::ArmJit::mode.load(std::memory_order_relaxed);
            Core::ArmJit::mode.store(current != 0 ? 0 : 1, std::memory_order_relaxed);
            Common::VitaNote("tecla", current != 0 ? "SELECT+DERECHA: JIT apagado"
                                                   : "SELECT+DERECHA: JIT encendido");
        }
        return;
    }
    jit_combo_held = false;

    // SELECT + IZQUIERDA: estirado de audio encendido/apagado (0.1.4.9). Ver
    // el comentario del sonido en main.cpp.
    const bool stretch_toggle = select && ((buttons & SCE_CTRL_LEFT) != 0);
    if (stretch_toggle) {
        if (!stretch_combo_held) {
            stretch_combo_held = true;
            auto& system = Core::System::GetInstance();
            if (system.IsPoweredOn()) {
                audio_stretching = !audio_stretching;
                system.DSP().EnableStretching(audio_stretching);
                Common::VitaNote("tecla", audio_stretching ? "SELECT+IZQUIERDA: sonido estirado"
                                                           : "SELECT+IZQUIERDA: sonido directo");
            }
        }
        return;
    }
    stretch_combo_held = false;

    // SELECT + TRIANGULO oculta o muestra el overlay de estadisticas (0.1.5.2,
    // 4.10). ovl mide ~2.3 ms por fotograma solo de pintar texto: apagado se
    // recupera ese tiempo y la pantalla queda limpia para jugar. El propio
    // overlay dice ovl ON/OFF cuando esta visible, y SELECT+TRIANGULO otra
    // vez lo vuelve a poner. TRIANGULO es el X del 3DS; con SELECT delante no
    // lo pisa ningun juego (mismo criterio que el resto de combos).
    const bool overlay_toggle = select && ((buttons & SCE_CTRL_TRIANGLE) != 0);
    if (overlay_toggle) {
        if (!overlay_combo_held) {
            overlay_combo_held = true;
            // Apagado -> compacto -> completo -> apagado (0.2.2.2).
            if (!stats_overlay_visible) {
                stats_overlay_visible = true;
                stats_overlay_full = false;
            } else if (!stats_overlay_full) {
                stats_overlay_full = true;
            } else {
                stats_overlay_visible = false;
            }
            stats_next_update_us = 0;
            Common::VitaNote("tecla", !stats_overlay_visible ? "SELECT+TRIANGULO: overlay OFF"
                                      : stats_overlay_full   ? "SELECT+TRIANGULO: overlay completo"
                                                             : "SELECT+TRIANGULO: overlay compacto");
        }
        return;
    }
    overlay_combo_held = false;

    const bool down = select && l;
    const bool up = select && r;

    // Solo al pulsar, no mientras se mantiene: si no, un toque recorreria todo
    // el rango en un par de fotogramas.
    if (!down && !up) {
        frameskip_combo_held = false;
        return;
    }
    if (frameskip_combo_held) {
        return;
    }
    frameskip_combo_held = true;

    u32 value = SwRenderer::FrameSkip::interval.load(std::memory_order_relaxed);
    if (up && value < SwRenderer::FrameSkip::kMaxInterval) {
        value++;
    } else if (down && value > SwRenderer::FrameSkip::kMinInterval) {
        value--;
    }
    SwRenderer::FrameSkip::interval.store(value, std::memory_order_relaxed);
    Common::VitaNote("tecla", fmt::format("SELECT+{}: salto de fotogramas {}", up ? "R" : "L",
                                          value)
                                  .c_str());
}

void EmuWindow_Vita::UpdateTouch() {
    SceTouchData touch{};
    if (sceTouchPeek(SCE_TOUCH_PORT_FRONT, &touch, 1) < 0 || touch.reportNum == 0) {
        if (is_touching) {
            is_touching = false;
            TouchReleased();
        }
        return;
    }

    // Coordenadas del panel tactil -> pixeles de la pantalla de la Vita.
    const u32 screen_x =
        static_cast<u32>(static_cast<int>(touch.report[0].x) * static_cast<int>(kVitaScreenWidth) /
                         (kTouchMaxX + 1));
    const u32 screen_y =
        static_cast<u32>(static_cast<int>(touch.report[0].y) * static_cast<int>(kVitaScreenHeight) /
                         (kTouchMaxY + 1));

    // TouchPressed ya descarta lo que cae fuera de la pantalla inferior, asi
    // que basta con pasarle las coordenadas del framebuffer.
    if (is_touching) {
        TouchMoved(screen_x, screen_y);
    } else if (TouchPressed(screen_x, screen_y)) {
        is_touching = true;
    }
}

// Aqui habia un UploadScreen(texture, pixels, ancho, alto) que no llamaba nadie:
// era el camino viejo, de cuando ScreenInfo traia los pixeles ya convertidos a
// RGBA8. Desde que la conversion la hace la GPU, ese formato fijo ya no se
// cumple, y desde que el framebuffer no se copia (ver ScreenInfo::source) el
// vector que recibia esta siempre vacio. Se quita en vez de dejarlo: una funcion
// muerta que ademas documenta una suposicion que ya es falsa solo sirve para
// que alguien la use dentro de seis meses.

// La tabla de formatos GXM y el numero de bytes por pixel de cada formato del
// 3DS viven en video_core/renderer_gxm/gxm_pica_format.h: los necesitan tanto
// el camino de vita2d de este fichero como el presentador GXM, y tenerlos
// duplicados fue exactamente el origen de un ciclo de prueba en consola
// (rojo y azul intercambiados). Ver alli el porque de cada sufijo _RGB/_RGBA.

void EmuWindow_Vita::UploadScreenNative(vita2d_texture*& texture, const ScreenInfoRef& info) {
    // 'source' apunta al framebuffer del invitado, no a una copia: ver
    // ScreenInfo::source en renderer_software.h.
    if (!info.valid || info.source == nullptr) {
        return;
    }

    // La textura se recrea solo cuando cambian tamano o formato. Los juegos los
    // cambian muy de vez en cuando (al entrar y salir de menus, sobre todo), asi
    // que en la practica esto ocurre un punado de veces por partida.
    //
    // La tabla de formatos es la misma que usa el presentador GXM
    // (renderer_gxm/gxm_pica_format.h): estaba duplicada aqui, y dos copias de
    // una correspondencia que costo un ciclo de prueba en consola averiguar es
    // pedir que se desincronicen.
    const SceGxmTextureFormat want = Gxm::FormatFor(info.format);
    if (texture == nullptr || vita2d_texture_get_width(texture) != info.width ||
        vita2d_texture_get_height(texture) != info.height ||
        vita2d_texture_get_format(texture) != want) {
        if (texture != nullptr) {
            vita2d_free_texture(texture);
        }
        texture = vita2d_create_empty_texture_format(info.width, info.height, want);
        if (texture == nullptr) {
            return;
        }
    }

    u8* dest = static_cast<u8*>(vita2d_texture_get_datap(texture));
    const u32 dest_stride = vita2d_texture_get_stride(texture);
    const u32 src_stride = info.stride;
    const u32 row_bytes = info.width * Gxm::BytesPerPixelFor(info.format);

    if (dest_stride == src_stride) {
        // Caso normal: una sola copia de bloque, sin tocar un solo pixel. Se
        // acota con source_size porque el origen es ahora memoria del invitado.
        std::memcpy(dest, info.source,
                    std::min<std::size_t>(static_cast<std::size_t>(src_stride) * info.height,
                                          info.source_size));
        return;
    }
    for (u32 y = 0; y < info.height; y++) {
        std::memcpy(dest + static_cast<std::size_t>(y) * dest_stride,
                    info.source + static_cast<std::size_t>(y) * src_stride, row_bytes);
    }
}

void EmuWindow_Vita::DrawRotatedScreen(vita2d_texture* texture, const ScreenInfoRef& info,
                                       u32 left, u32 top, u32 draw_w, u32 draw_h) {
    if (info.fill_enabled) {
        // Color plano: el juego pide un color liso, sin framebuffer. Se pinta un
        // rectangulo y ya. Sin esto la pantalla saldria NEGRA, y los juegos usan
        // esto para los fundidos.
        const unsigned int color = 0xFF000000u | (static_cast<unsigned int>(info.fill_b) << 16) |
                                   (static_cast<unsigned int>(info.fill_g) << 8) |
                                   static_cast<unsigned int>(info.fill_r);
        vita2d_draw_rectangle(static_cast<float>(left), static_cast<float>(top),
                              static_cast<float>(draw_w), static_cast<float>(draw_h), color);
        return;
    }

    if (!info.valid || texture == nullptr) {
        return;
    }

    /**
     * Giro de 90 grados en sentido antihorario.
     *
     * No es una suposicion: sale de la transformacion que hacia el bucle de CPU
     * que esto sustituye. Alli el destino se escribia como
     *
     *     destino[x][y] = origen[y][ancho - 1 - x]
     *
     * y esa es exactamente la definicion de rotar 90 grados en sentido
     * antihorario una imagen de 'alto' filas por 'ancho' columnas.
     *
     * vita2d rota alrededor del centro de la textura, asi que hay que colocar
     * ese centro donde toca: el centro del rectangulo de destino.
     */
    constexpr float kHalfPi = 1.57079632679f;
    const float center_x = static_cast<float>(left) + static_cast<float>(draw_w) * 0.5f;
    const float center_y = static_cast<float>(top) + static_cast<float>(draw_h) * 0.5f;

    // Escalado al rectangulo pedido (0.1.7.1, disposiciones de pantalla). La
    // textura esta girada: su ancho (240) acaba siendo el ALTO en pantalla.
    const float x_scale = static_cast<float>(draw_h) / static_cast<float>(info.width);
    const float y_scale = static_cast<float>(draw_w) / static_cast<float>(info.height);
    vita2d_draw_texture_scale_rotate_hotspot(texture, center_x, center_y, x_scale, y_scale,
                                             -kHalfPi, static_cast<float>(info.width) * 0.5f,
                                             static_cast<float>(info.height) * 0.5f);
}

void EmuWindow_Vita::PresentScreens() {
    auto& system = Core::System::GetInstance();
    if (!system.IsPoweredOn()) {
        return;
    }

    auto& renderer = system.GPU().Renderer();

    /**
     * Los dos renderers exponen Screen() con la misma firma a proposito (ver
     * renderer_gxm.h), asi que aqui se elige el tipo por la API activa, que es
     * la misma que uso CreateRenderer para construir el objeto. No se usa
     * dynamic_cast: con RTTI el binario engorda y el tipo ya se sabe.
     */
    const bool gxm_api = Settings::GetWorkingGraphicsAPI() == Settings::GraphicsAPI::GXM;

    // El framebuffer del 3DS esta GIRADO en memoria: 240 de ancho por 400 (o
    // 320) de alto. Se sube tal cual, en su formato nativo, y se presenta
    // rotado.
    const SwRenderer::ScreenInfo* top;
    const SwRenderer::ScreenInfo* bottom;
    if (gxm_api) {
        auto& gxm_renderer = static_cast<Gxm::RendererGXM&>(renderer);
        top = &gxm_renderer.Screen(VideoCore::ScreenId::TopLeft);
        bottom = &gxm_renderer.Screen(VideoCore::ScreenId::Bottom);
    } else {
        auto& sw_renderer = static_cast<SwRenderer::RendererSoftware&>(renderer);
        top = &sw_renderer.Screen(VideoCore::ScreenId::TopLeft);
        bottom = &sw_renderer.Screen(VideoCore::ScreenId::Bottom);
    }

    // Disposicion de pantallas del menu de ajustes (0.1.7.1).
    const int wanted_layout = g_screen_layout.load(std::memory_order_relaxed);
    if (wanted_layout != layout_mode) {
        layout_mode = wanted_layout;
        BuildLayout();
        has_presented = false;
    }

    /**
     * NO REPETIR UNA IMAGEN QUE NO HA CAMBIADO (0.1.7.1).
     *
     * Presentar (subir las dos pantallas a la GPU y dibujarlas) costaba ~19 ms
     * de cada vblank en la cinematica de Rubi Omega (sub 8 + dib 11), y el juego
     * solo produce una imagen nueva cada DOS vblanks: la mitad de las veces se
     * subia y se dibujaba exactamente lo mismo. Si desde la ultima presentacion
     * la GPU emulada no ha hecho nada que pueda cambiar la imagen (ni listas de
     * comandos, ni rellenos, ni transferencias, ni DMA; ver
     * GxStats::frame_work) y los framebuffers son los mismos, no se presenta:
     * la Vita sigue mostrando el ultimo, que es identico.
     *
     * Lo unico que no se ve asi es un juego que dibuje en el framebuffer con la
     * CPU sin pasar por la GPU. Para eso esta el interruptor del menu.
     */
    if (g_skip_repeated_frames.load(std::memory_order_relaxed)) {
        const unsigned long long work =
            VideoCore::GxStats::frame_work.load(std::memory_order_relaxed);
        const u32 fill_colors =
            (static_cast<u32>(top->fill_r) << 24) | (static_cast<u32>(top->fill_g) << 16) |
            (static_cast<u32>(bottom->fill_r) << 8) | static_cast<u32>(bottom->fill_b);
        const bool same = has_presented && work == last_frame_work &&
                          top->source_address == last_top_address &&
                          bottom->source_address == last_bottom_address &&
                          top->source == last_top_source && bottom->source == last_bottom_source &&
                          top->fill_enabled == last_top_fill &&
                          bottom->fill_enabled == last_bottom_fill &&
                          fill_colors == last_fill_colors;
        last_frame_work = work;
        last_top_address = top->source_address;
        last_bottom_address = bottom->source_address;
        last_top_source = top->source;
        last_bottom_source = bottom->source;
        last_top_fill = top->fill_enabled;
        last_bottom_fill = bottom->fill_enabled;
        last_fill_colors = fill_colors;
        if (same) {
            return;
        }
    }
    has_presented = true;
    const ScreenRects rects = RectsFor(layout_mode);

    /**
     * El presentador GXM se crea la primera vez que se presenta con la API GXM
     * y decide su propio destino al inicializarse: si no puede (falta
     * libshacccg.suprx, falla la compilacion de shaders, no hay memoria), Ready()
     * se queda en false y este fotograma -- y los siguientes -- van por vita2d
     * exactamente como antes. Es una degradacion a proposito: el backend nuevo
     * no puede impedir jugar.
     *
     * El contexto y el patcher son los de vita2d: GXM solo admite un contexto
     * por proceso y es el que abre la escena donde se dibujan los quads.
     */
    bool use_gxm = false;
    if (gxm_api) {
        if (gxm_presenter == nullptr) {
            gxm_presenter = std::make_unique<Gxm::ScreenPresenter>(
                static_cast<float>(kVitaScreenWidth), static_cast<float>(kVitaScreenHeight),
                vita2d_get_context(), vita2d_get_shader_patcher());
        }
        gxm_presenter->Initialize();
        use_gxm = gxm_presenter->Ready();
    }

    if (!use_gxm) {
        // Camino de siempre: copiar los bytes crudos a la textura de vita2d.
        // Con GXM la copia la hace el presentador y se cronometra alli. La
        // presentacion anterior puede seguir leyendo estas texturas (0.2.2.0).
        if (gxm_api) {
            Gxm::RasterizerGXM::WaitPreviousPresentation();
        }
        const unsigned long long upload_begin = Common::VitaMicros();
        UploadScreenNative(top_texture, *top);
        UploadScreenNative(bottom_texture, *bottom);
        Common::FrameStats::Add(Common::FrameStats::upload_us, upload_begin);
    }

    // Los tres tramos de presentar, cronometrados por separado. Ver el bloque de
    // Common::FrameStats para el razonamiento; en corto: subir pixeles, encolar
    // dibujos y esperar al barrido son tres cosas con costes muy distintos, y
    // sumadas no dicen nada sobre cual hay que arreglar.
    const unsigned long long draw_begin = Common::VitaMicros();
    const Common::ScopedVitaStage stage{"presentar"};
    /**
     * EL OVERLAY, EN UNA TEXTURA (0.1.9.6). Pintar sus ~30 lineas de texto
     * costaba ~2,5 ms en CADA fotograma ("ovl"), y las cifras solo cambian una
     * vez por segundo. Ahora se pintan en una textura cuando cambian, fuera de
     * la escena de la pantalla, y cada fotograma la pone encima con un quad.
     * Sin textura (sin memoria de video), como antes: directo cada fotograma.
     */
    /**
     * EL POOL DE VITA2D, SIN VACIARLO CADA FOTOGRAMA (0.2.2.0). vita2d_start_
     * drawing() lo pone a cero en cada llamada (desensamblado de libvita2d:
     * escribe 0 en su desplazamiento), y con eso los vertices de esta
     * presentacion pisaban los de la anterior aunque la GPU no la hubiera
     * dibujado aun: por eso FlushForPresent esperaba a la GPU en cada
     * fotograma. Ahora se empieza con vita2d_start_drawing_advanced(NULL, 0),
     * que es lo mismo sin vaciarlo, y se vacia aqui solo cuando pasa de la
     * mitad, despues de esperar a que la GPU acabe todo (una vez cada muchos
     * fotogramas: cada uno gasta unos pocos KB).
     */
    if (vita2d_pool_capacity == 0) {
        vita2d_wait_rendering_done();
        vita2d_pool_reset();
        vita2d_pool_capacity = vita2d_pool_free_space();
    } else if (vita2d_pool_free_space() < vita2d_pool_capacity / 2) {
        vita2d_wait_rendering_done();
        vita2d_pool_reset();
    }

    bool overlay_from_texture = false;
    {
        const unsigned long long overlay_begin = Common::VitaMicros();
        if (stats_overlay_visible && overlay_texture == nullptr) {
            overlay_texture = vita2d_create_empty_texture_rendertarget(
                kVitaScreenWidth, kVitaScreenHeight, SCE_GXM_TEXTURE_FORMAT_A8B8G8R8);
            stats_next_update_us = 0;
        }
        if (stats_overlay_visible && overlay_texture != nullptr) {
            overlay_from_texture = true;
            if (sceKernelGetProcessTimeWide() >= stats_next_update_us) {
                const unsigned int clear_color = vita2d_get_clear_color();
                vita2d_start_drawing_advanced(overlay_texture, 0);
                vita2d_set_clear_color(RGBA8(0, 0, 0, 0));
                vita2d_clear_screen();
                DrawStatsOverlay();
                vita2d_end_drawing();
                vita2d_set_clear_color(clear_color);
            }
        } else if (!stats_overlay_visible) {
            // Solo las cifras y las notas de crash.txt: no pinta nada.
            DrawStatsOverlay();
        }
        Common::FrameStats::Add(Common::FrameStats::overlay_us, overlay_begin);
    }
    vita2d_start_drawing_advanced(nullptr, 0);
    vita2d_clear_screen();
    if (use_gxm) {
        gxm_presenter->Draw(*top, *bottom, {rects.top_x, rects.top_y, rects.top_w, rects.top_h},
                            {rects.bottom_x, rects.bottom_y, rects.bottom_w, rects.bottom_h});
    } else {
        DrawRotatedScreen(top_texture, *top, static_cast<u32>(rects.top_x),
                          static_cast<u32>(rects.top_y), static_cast<u32>(rects.top_w),
                          static_cast<u32>(rects.top_h));
        if (rects.bottom_w > 0) {
            DrawRotatedScreen(bottom_texture, *bottom, static_cast<u32>(rects.bottom_x),
                              static_cast<u32>(rects.bottom_y), static_cast<u32>(rects.bottom_w),
                              static_cast<u32>(rects.bottom_h));
        }
    }

    // Estado para el overlay. El presentador puede apagarse a mitad de partida
    // (sin memoria de GPU), asi que la linea se rehace en cada fotograma.
    presenter_uses_gxm = use_gxm;
    BuildLine(presenter_line, sizeof(presenter_line), "pres ",
              gxm_api ? gxm_presenter->Status() : "vita2d (software)");

    {
        // El overlay, medido aparte (0.1.0.45): son ~30 lineas de texto por
        // fotograma, y ese coste solo existe mientras se mira. Sin separarlo,
        // 'dib' mezclaba lo que cuesta presentar el juego con lo que cuesta
        // medirlo.
        const unsigned long long overlay_begin = Common::VitaMicros();
        if (overlay_from_texture) {
            vita2d_draw_texture(overlay_texture, 0.0f, 0.0f);
        } else if (stats_overlay_visible) {
            DrawStatsOverlay();
        }
        /**
         * AVISO DE COMPILACION (0.2.1.6). Hay shaders de vertices de Pokemon
         * Sol que tardan 25-30 s en compilarse, y mientras tanto el juego va a
         * 1 FPS: parece colgado y se cierra, y la compilacion se pierde (antes
         * ademas iba a la lista negra). Solo si lleva mas de un segundo.
         */
        const Gxm::CgActivity cg = Gxm::GetCgActivity();
        const unsigned long long now_us = Common::VitaMicros();
        if (stats_font != nullptr && cg.busy_since_us != 0 && now_us > cg.busy_since_us &&
            now_us - cg.busy_since_us > 1000000ull) {
            const std::string notice = fmt::format(
                "Compilando shaders ({} en cola, {} s): no cierres el juego", cg.queued,
                (now_us - cg.busy_since_us) / 1000000ull);
            vita2d_draw_rectangle(0.0f, static_cast<float>(kVitaScreenHeight) - 24.0f, 470.0f,
                                  24.0f, RGBA8(0, 0, 0, 170));
            vita2d_pgf_draw_text(stats_font, 8, static_cast<int>(kVitaScreenHeight) - 7,
                                 RGBA8(255, 210, 90, 255), 0.85f, notice.c_str());
        }
        Common::FrameStats::Add(Common::FrameStats::overlay_us, overlay_begin);
    }
    vita2d_end_drawing();
    Common::FrameStats::Add(Common::FrameStats::draw_us, draw_begin);

    // vita2d_swap_buffers espera a que la GPU acabe y al barrido de pantalla.
    // Casi todo lo que se mida aqui es la consola PARADA, no trabajo: por eso va
    // en su propio contador y el overlay lo pinta aparte.
    const int vblank_wait = g_unlimited_speed.load(std::memory_order_relaxed) ? 0 : 1;
    if (vblank_wait != applied_vblank_wait) {
        applied_vblank_wait = vblank_wait;
        vita2d_set_vblank_wait(vblank_wait);
    }
    const unsigned long long swap_begin = Common::VitaMicros();
    Common::VitaStageSlot().store("presentar: swap", std::memory_order_relaxed);
    vita2d_swap_buffers();
    Common::FrameStats::Add(Common::FrameStats::swap_us, swap_begin);

    Common::FrameStats::frames.fetch_add(1, std::memory_order_relaxed);
    frames_presented++;
}

namespace {
/// Igual que AppendUInt de vita_diag.cpp: escribe un entero decimal a mano.
/// Se repite aqui en vez de compartirlo porque esto es la unica otra parte del
/// codigo que formatea un numero fuera de crash.txt, y no merece la pena un
/// fichero compartido para una funcion de cuatro lineas.
std::size_t AppendUInt(char* out, unsigned int value) {
    char tmp[12];
    std::size_t n = 0;
    do {
        tmp[n++] = static_cast<char>('0' + (value % 10));
        value /= 10;
    } while (value != 0);
    for (std::size_t i = 0; i < n; i++) {
        out[i] = tmp[n - 1 - i];
    }
    return n;
}

/// Redondea a la decima mas cercana y escribe "N.D", trabajando siempre en
/// enteros. No se usa vita2d_pgf_draw_textf con "%f" a proposito: esta libc
/// tiene enlazada _svfiprintf_r, la variante SIN coma flotante, junto a la que
/// si la soporta, y no hay forma de comprobar desde aqui cual usa vita2d por
/// dentro. Esta funcion se llama en CADA fotograma dibujado, justo en el
/// camino que costo dias estabilizar: no vale la pena arriesgarlo por un
/// printf mas comodo cuando escribir los digitos a mano cuesta cuatro lineas.
void AppendOneDecimal(char* out, std::size_t& n, double value) {
    if (!(value > 0.0)) { // tambien descarta NaN
        value = 0.0;
    }
    if (value > 999.0) {
        value = 999.0; // no deberia pasar, pero sin esto tmp podria desbordar
    }
    const unsigned int tenths = static_cast<unsigned int>(value * 10.0 + 0.5);
    n += AppendUInt(out + n, tenths / 10);
    out[n++] = '.';
    out[n++] = static_cast<char>('0' + (tenths % 10));
}

/**
 * Igual que la anterior, pero para milisegundos por fotograma.
 *
 * Existe aparte porque AppendOneDecimal acota a 999, y aqui eso no vale: en esta
 * consola un fotograma pasa de 2000 ms con facilidad, que es justo el numero que
 * interesa medir. Se acota en 99999,9 (siete caracteres como mucho), y quien la
 * use tiene que dejar ese hueco en su buffer.
 *
 * No se subio el tope de la otra funcion porque sus llamadas son porcentajes con
 * buffers ajustados a tres digitos -- line_gx[40] se desbordaria por un byte.
 */
void AppendMillis(char* out, std::size_t& n, double value) {
    if (!(value > 0.0)) { // tambien descarta NaN
        value = 0.0;
    }
    if (value > 99999.0) {
        value = 99999.0;
    }
    const unsigned int tenths = static_cast<unsigned int>(value * 10.0 + 0.5);
    n += AppendUInt(out + n, tenths / 10);
    out[n++] = '.';
    out[n++] = static_cast<char>('0' + (tenths % 10));
}
// El cartel de version del overlay (kOverlayBuild) ya no se escribe aqui: viene
// de citra_vita/vita_version.h, que es el unico sitio donde vive el numero.
// Estaba duplicado a mano y se habia quedado desfasado respecto al de main.cpp
// -- que es exactamente el fallo que ese cartel existe para evitar.
} // Anonymous namespace

void EmuWindow_Vita::DrawStatsOverlay() {
    if (stats_font == nullptr) {
        return;
    }

    // La pantalla superior del 3DS ocupa x:280..680; la inferior, x:320..640.
    // La franja x:0..280 queda libre en toda la altura de los 544 px de la
    // Vita, asi que el overlay va ahi: no tapa nada del juego sea cual sea el
    // contenido de cualquiera de las dos pantallas.
    constexpr float kOverlayX = 6.0f;

    // GetAndResetPerfStats() promedia sobre el intervalo transcurrido desde la
    // ultima llamada. Pedirlo en cada fotograma dibujado (que en esta consola
    // puede ser una vez por segundo o menos) daria un numero valido igualmente,
    // pero limitar a 1 vez por segundo real evita que, si algun dia esto va mas
    // rapido, el overlay parpadee con cada muestra individual en vez de mostrar
    // una media legible.
    const SceUInt64 now_us = sceKernelGetProcessTimeWide();
    if (now_us >= stats_next_update_us) {
        const auto stats = Core::System::GetInstance().GetAndResetPerfStats();
        {
            // Media de 5 s: ver fps_frames en vita_window.h.
            const double seconds =
                fps_last_us != 0 ? static_cast<double>(now_us - fps_last_us) / 1e6 : 0.0;
            fps_last_us = now_us;
            if (seconds > 0.0) {
                fps_frames[fps_slot] = stats.game_fps * seconds;
                fps_seconds[fps_slot] = seconds;
                fps_slot = (fps_slot + 1) % kFpsWindows;
            }
            double frames = 0.0;
            double total = 0.0;
            for (int i = 0; i < kFpsWindows; i++) {
                frames += fps_frames[i];
                total += fps_seconds[i];
            }
            stats_game_fps = total > 0.0 ? frames / total : stats.game_fps;
        }
        {
            const double interval_us =
                stats_gpu_last_us != 0 ? static_cast<double>(now_us - stats_gpu_last_us) : 0.0;
            stats_gpu_last_us = now_us;
            const double busy = static_cast<double>(
                VideoCore::GxStats::thread_busy_us.exchange(0, std::memory_order_relaxed));
            const double wait = static_cast<double>(
                VideoCore::GxStats::emu_wait_us.exchange(0, std::memory_order_relaxed));
            stats_gpu_busy_percent = interval_us > 0.0 ? busy / interval_us * 100.0 : 0.0;
            stats_gpu_wait_percent = interval_us > 0.0 ? wait / interval_us * 100.0 : 0.0;
            stats_gpu_irq_waits =
                VideoCore::GxStats::irq_waits.exchange(0, std::memory_order_relaxed);
            stats_gpu_syncs = VideoCore::GxStats::syncs.exchange(0, std::memory_order_relaxed);
            stats_gpu_present_waits =
                VideoCore::GxStats::present_waits.exchange(0, std::memory_order_relaxed);
            stats_gpu_queue_max =
                VideoCore::GxStats::queue_max.exchange(0, std::memory_order_relaxed);
        }
        stats_speed_percent = stats.emulation_speed * 100.0;

        // Reparto del tiempo dentro de cada fotograma. time_vblank_interval es
        // el total; el resto son trozos de ese total. "cpu" sale de
        // time_remaining, que es lo que queda despues de descontar GPU, SVC e
        // IPC: en este port eso es basicamente el interprete de ARM11.
        const double total = stats.time_vblank_interval;
        if (total > 0.0) {
            stats_cpu_percent = stats.time_remaining / total * 100.0;
            stats_gpu_percent = stats.time_gpu / total * 100.0;
            stats_svc_percent = (stats.time_hle_svc + stats.time_hle_ipc) / total * 100.0;
            stats_swap_percent = stats.time_swap / total * 100.0;
        }

        // Y lo mismo en milisegundos absolutos. PerfStats da segundos POR
        // FOTOGRAMA de sistema, asi que basta multiplicar por mil: no hay que
        // dividir por nada mas.
        stats_frame_ms = total * 1000.0;
        stats_cpu_ms = stats.time_remaining * 1000.0;
        stats_gx_ms = stats.time_gpu * 1000.0;
        stats_svc_ms = (stats.time_hle_svc + stats.time_hle_ipc) * 1000.0;

        // Los contadores del frontend, en cambio, se acumulan durante todo el
        // intervalo: hay que repartirlos entre las presentaciones que han
        // cabido. Se usa el contador propio (Common::FrameStats::frames) y no el
        // FPS de PerfStats por el mismo motivo que el resto del overlay: a menos
        // de un fotograma por segundo ese FPS es fraccionario y dividir por el
        // convierte el ruido del muestreo en cifras absurdas.
        {
            const unsigned int present_frames =
                Common::FrameStats::frames.load(std::memory_order_relaxed);
            const double n = present_frames > 0 ? static_cast<double>(present_frames) : 1.0;
            const auto per_frame_ms = [n](std::atomic<unsigned long long>& slot) {
                return static_cast<double>(slot.load(std::memory_order_relaxed)) / n / 1000.0;
            };
            stats_convert_ms = per_frame_ms(Common::FrameStats::convert_us);
            stats_upload_ms = per_frame_ms(Common::FrameStats::upload_us);
            stats_draw_ms = per_frame_ms(Common::FrameStats::draw_us);
            stats_overlay_ms = per_frame_ms(Common::FrameStats::overlay_us);
            stats_dsp_ms = per_frame_ms(Common::FrameStats::dsp_us);
            stats_aac_ms = per_frame_ms(Common::FrameStats::aac_us);
            {
                namespace DspStats = AudioCore::HLE::Stats;
                stats_dsp_decode_ms = static_cast<double>(DspStats::decode_us.exchange(
                                          0, std::memory_order_relaxed)) /
                                      n / 1000.0;
                DspStats::decoded.store(0, std::memory_order_relaxed);
                stats_dsp_worker_ms = static_cast<double>(DspStats::worker_us.exchange(
                                          0, std::memory_order_relaxed)) /
                                      n / 1000.0;
                const unsigned long long ticks = DspStats::ticks.exchange(0, std::memory_order_relaxed);
                const unsigned long long active = DspStats::active.exchange(0, std::memory_order_relaxed);
                stats_dsp_sources =
                    ticks != 0 ? static_cast<double>(active) / static_cast<double>(ticks) : 0.0;
            }
            {
                /**
                 * JIT del ARM11. Desde 0.1.4.9 todo sale de TakeStats: las dos
                 * cuentas del porcentaje las hace el propio JIT en el mismo
                 * sitio (en 0.1.4.8 salian de dos lugares distintos y daban un
                 * imposible 170 %).
                 */
                Core::ArmJit::Stats jit{};
                Core::ArmJit::TakeStats(jit);
                const double guest = static_cast<double>(jit.instructions);
                stats_guest_mips = guest / n / 1000000.0;
                stats_jit_percent =
                    guest > 0.0 ? static_cast<double>(jit.jit_instructions) / guest * 100.0 : 0.0;
                // El ARM de verdad: el bucle menos lo que se va dentro de las
                // llamadas al sistema (que incluyen toda la GPU emulada).
                stats_arm_ms = static_cast<double>(jit.arm_us > jit.svc_us ? jit.arm_us - jit.svc_us : 0) / n / 1000.0;
                stats_arm_slices = static_cast<double>(jit.slices) / n;
                stats_arm_dispatches = static_cast<double>(jit.dispatches) / n;
                stats_arm_links = static_cast<double>(jit.links) / n;
                stats_arm_slow = static_cast<double>(jit.slow_calls) / n;
                stats_arm_vfp = static_cast<double>(jit.vfp_calls) / n;
                stats_arm_interp_k =
                    static_cast<double>(jit.instructions > jit.jit_instructions
                                            ? jit.instructions - jit.jit_instructions
                                            : 0) /
                    n / 1000.0;
                {
                    const auto to_ms = [n](u64 us) { return static_cast<double>(us) / n / 1000.0; };
                    stats_arm_jit_ms = to_ms(jit.jit_us);
                    stats_arm_slow_ms = to_ms(jit.slow_us);
                    stats_arm_vfp_ms = to_ms(jit.vfp_us);
                    stats_arm_check_ms = to_ms(jit.check_us);
                    stats_arm_compile_ms = to_ms(jit.compile_us);
                    const double rest = stats_arm_ms - stats_arm_jit_ms - stats_arm_check_ms -
                                        stats_arm_compile_ms;
                    stats_arm_rest_ms = rest > 0.0 ? rest : 0.0;
                }
                {
                    // Cada 10 s a crash.txt: los datos llegan aunque no haya
                    // captura (0.1.6.3). Treinta veces desde 0.2.0.0 (eran
                    // nueve): las escenas 3D que se prueban llegan despues de
                    // las intros, y con nueve se quedaban fuera.
                    static u32 arm_ticks = 0;
                    static u32 arm_notes = 0;
                    if (++arm_ticks % 10 == 0 && arm_notes < 30) {
                        arm_notes++;
                        Common::VitaNote(
                            "arm desglose",
                            fmt::format("ms {:.1f} arm {:.1f} jit {:.0f}% Mi {:.2f} rod {:.0f} desp "
                                        "{:.0f} enl {:.0f} lent {:.0f} vfp {:.0f} int {:.1f}k "
                                        "vtx {:.1f} sh {:.1f} vsh {}",
                                        stats_frame_ms, stats_arm_ms, stats_jit_percent,
                                        stats_guest_mips, stats_arm_slices,
                                        stats_arm_dispatches, stats_arm_links, stats_arm_slow,
                                        stats_arm_vfp, stats_arm_interp_k, stats_vertices_ms,
                                        stats_shade_ms, stats_vertices_shaded)
                                .c_str());
                        Common::VitaNote(
                            "arm tiempos",
                            fmt::format("ms por fotograma: jit {:.1f} (lento {:.1f} vfp {:.1f}) "
                                        "comprobar {:.1f} compilar {:.1f} resto {:.1f} dsp {:.1f}",
                                        stats_arm_jit_ms, stats_arm_slow_ms, stats_arm_vfp_ms,
                                        stats_arm_check_ms, stats_arm_compile_ms,
                                        stats_arm_rest_ms, stats_dsp_ms)
                                .c_str());
                        {
                            // Los contadores del Cortex-A9 (0.2.3.6): ver
                            // PmuSliceBegin en arm_dyncom_jit.cpp.
                            Core::ArmJit::PmuStats pmu{};
                            Core::ArmJit::TakePmu(pmu);
                            const auto ratio = [](u64 part, u64 whole) {
                                return whole > 0 ? static_cast<double>(part) /
                                                       static_cast<double>(whole)
                                                 : 0.0;
                            };
                            const u64* a = &pmu.counts[0];
                            const u64* b = &pmu.counts[Core::ArmJit::kPmuCounters];
                            if (a[0] > 0 || b[0] > 0) {
                                Common::VitaNote(
                                    "arm pmu",
                                    fmt::format(
                                        "ciclos por instr. del juego {:.1f} | ipc {:.2f} | "
                                        "parado: codigo {:.0f}% datos {:.0f}% tlb {:.0f}% | "
                                        "saltos fallados {:.2f} por instr. || ciclos {:.1f} | "
                                        "fallos de cache por instr.: codigo {:.2f} datos {:.2f} "
                                        "| parado: tlb codigo {:.0f}% tlb datos {:.0f}% "
                                        "escritura {:.0f}%",
                                        ratio(a[0], pmu.instructions[0]), ratio(a[1], a[0]),
                                        ratio(a[2], a[0]) * 100.0, ratio(a[3], a[0]) * 100.0,
                                        ratio(a[4], a[0]) * 100.0,
                                        ratio(a[5], pmu.instructions[0]),
                                        ratio(b[0], pmu.instructions[1]),
                                        ratio(b[1], pmu.instructions[1]),
                                        ratio(b[2], pmu.instructions[1]),
                                        ratio(b[3], b[0]) * 100.0, ratio(b[4], b[0]) * 100.0,
                                        ratio(b[5], b[0]) * 100.0)
                                        .c_str());
                            }
                        }
                        // Y el desglose de un vertice (0.1.7.2), en us por vertice.
                        const double s = static_cast<double>(
                            Common::FrameStats::vtx_samples.exchange(0, std::memory_order_relaxed));
                        const double l = static_cast<double>(
                            Common::FrameStats::vtx_load_us.exchange(0, std::memory_order_relaxed));
                        const double r = static_cast<double>(
                            Common::FrameStats::vtx_run_us.exchange(0, std::memory_order_relaxed));
                        const double o = static_cast<double>(
                            Common::FrameStats::vtx_out_us.exchange(0, std::memory_order_relaxed));
                        if (s > 0.0) {
                            Common::VitaNote("vertice",
                                             fmt::format("us por vertice: leer {:.2f} shader {:.2f} "
                                                         "salida {:.2f} ({:.0f} muestras)",
                                                         l / s, r / s, o / s, s)
                                                 .c_str());
                        }
                    }
                }
                // El motivo de rechazo que mas despachos se lleva, en % de los
                // despachos del intervalo: dice que traducir despues.
                stats_jit_top_reject = 0;
                u64 top = 0;
                for (u32 i = 1; i < Core::ArmJit::kRejectCount; i++) {
                    if (jit.rejects[i] > top) {
                        top = jit.rejects[i];
                        stats_jit_top_reject = i;
                    }
                }
                stats_jit_top_reject_percent =
                    jit.dispatches > 0 ? static_cast<double>(top) /
                                             static_cast<double>(jit.dispatches) * 100.0
                                       : 0.0;
                stats_jit_blocks = Common::FrameStats::jit_blocks.load(std::memory_order_relaxed);
                stats_jit_rejected =
                    Common::FrameStats::jit_rejected.load(std::memory_order_relaxed);
                stats_jit_checks =
                    Common::FrameStats::jit_checks.exchange(0, std::memory_order_relaxed);
                stats_jit_mismatches =
                    Common::FrameStats::jit_mismatches.load(std::memory_order_relaxed);
                stats_jit_on = Core::ArmJit::mode.load(std::memory_order_relaxed) != 0;
            }
            stats_swapwait_ms = per_frame_ms(Common::FrameStats::swap_us);
            // El reparto de cmdlist. Va AQUI DENTRO y antes del Reset: fuera no
            // existe per_frame_ms, y despues del Reset los contadores ya valen
            // cero y la linea saldria siempre a 0.0.
            stats_vertices_ms = per_frame_ms(Common::FrameStats::vertices_us);
            stats_batch_ms = per_frame_ms(Common::FrameStats::batch_us);
            stats_vertices_shaded =
                Common::FrameStats::vertices_shaded.exchange(0, std::memory_order_relaxed);
            stats_shade_ms = per_frame_ms(Common::FrameStats::shade_us);
            {
                const double busy = static_cast<double>(
                    Common::FrameStats::shade_busy_us.load(std::memory_order_relaxed));
                const double wall = static_cast<double>(
                    Common::FrameStats::shade_us.load(std::memory_order_relaxed));
                stats_shade_occupancy = wall > 0.0 ? busy / (3.0 * wall) * 100.0 : 0.0;
                const double fast_ops = static_cast<double>(
                    Common::FrameStats::shade_fast_ops.exchange(0, std::memory_order_relaxed));
                const double slow = static_cast<double>(
                    Common::FrameStats::shade_slow_instrs.exchange(0, std::memory_order_relaxed));
                // Ya leido y puesto a cero justo arriba: se usa la copia.
                const double shaded = static_cast<double>(stats_vertices_shaded);
                stats_instrs_per_vertex = shaded > 0.0 ? (fast_ops + slow) / shaded : 0.0;
                stats_fast_percent =
                    (fast_ops + slow) > 0.0 ? fast_ops / (fast_ops + slow) * 100.0 : 0.0;
            }
            stats_finish_ms = per_frame_ms(Common::FrameStats::finish_us);
            stats_gpu_waits = Gxm::RasterizerGXM::TakeGpuWaitSummary(n);
            stats_texdecode_ms = per_frame_ms(Common::FrameStats::texture_decode_us);
            stats_texrehash_ms = per_frame_ms(Common::FrameStats::texture_rehash_us);
            stats_texrehash_kb = static_cast<double>(Common::FrameStats::texture_rehash_bytes.load(
                                     std::memory_order_relaxed)) /
                                 n / 1024.0;
            stats_texdecodes =
                Common::FrameStats::texture_decodes.exchange(0, std::memory_order_relaxed);
            stats_texreuses =
                Common::FrameStats::texture_reuses.exchange(0, std::memory_order_relaxed);
            stats_texchanged =
                Common::FrameStats::texture_changed.exchange(0, std::memory_order_relaxed);
            stats_texevictions =
                Common::FrameStats::texture_evictions.exchange(0, std::memory_order_relaxed);
            stats_fast_programs =
                Common::FrameStats::fast_programs.load(std::memory_order_relaxed);
            stats_fast_checks =
                Common::FrameStats::fast_checks.exchange(0, std::memory_order_relaxed);
            stats_fast_mismatches =
                Common::FrameStats::fast_mismatches.load(std::memory_order_relaxed);
            /**
             * 0.1.5.2 (4.2a y 4.7): anota en crash.txt, una vez por segundo,
             * las 5 instrucciones de shader que mas veces han caido a la ruta
             * lenta y el reparto de escrituras de registro PICA / fills /
             * transfers. Solo medir: sin esto, "rap 80%" no dice que instruccion
             * meter a la ruta rapida ni que parte del "resto de gx" es decodificar
             * registros frente a rellenos y transferencias.
             *
             * 0.1.5.8: solo las CINCO primeras veces. Cada nota es abrir,
             * escribir y cerrar crash.txt en la tarjeta de memoria, en el hilo
             * que presenta: una por segundo subio "ovl" de 2 a 17 ms por
             * fotograma en 0.1.5.7 (202 notas en una sesion).
             */
            // 0.1.8.3: cada 10 s y seis veces, no los cinco primeros segundos:
            // esos son el menu y la carga, y el shader que importa (el de piel
            // en el 3D) no salia nunca (crash.txt de 0.1.8.2).
            static u32 gx_ticks = 0;
            static u32 gx_notes = 0;
            if (++gx_ticks % 10 == 0 && gx_notes < 6) {
                gx_notes++;
                struct {
                    u32 op;
                    unsigned long long count;
                } top[5] = {};
                for (u32 i = 0; i < Common::FrameStats::kSlowOpcodeSlots; i++) {
                    const unsigned long long c =
                        Common::FrameStats::shade_slow_opcodes[i].exchange(
                            0, std::memory_order_relaxed);
                    if (c == 0) {
                        continue;
                    }
                    int slot = -1;
                    for (int j = 0; j < 5; j++) {
                        if (c > top[j].count) {
                            slot = j;
                            break;
                        }
                    }
                    if (slot < 0) {
                        continue;
                    }
                    for (int j = 4; j > slot; j--) {
                        top[j] = top[j - 1];
                    }
                    top[slot].op = i;
                    top[slot].count = c;
                }
                if (top[0].count > 0) {
                    std::string line = fmt::format(
                        "vs lenta top5: {:02x}={:} {:02x}={:} {:02x}={:} {:02x}={:} {:02x}={:}",
                        top[0].op, top[0].count, top[1].op, top[1].count, top[2].op,
                        top[2].count, top[3].op, top[3].count, top[4].op, top[4].count);
                    Common::VitaNote("vs lenta", line.c_str());
                }
                const unsigned long long reg_n =
                    Common::FrameStats::pica_reg_writes.exchange(0, std::memory_order_relaxed);
                const unsigned long long reg_ns =
                    Common::FrameStats::pica_reg_write_ns.exchange(0, std::memory_order_relaxed);
                const unsigned long long fills =
                    VideoCore::GxStats::fill_count.exchange(0, std::memory_order_relaxed);
                const unsigned long long transfers =
                    VideoCore::GxStats::transfer_count.exchange(0, std::memory_order_relaxed);
                const unsigned long long cmdlists =
                    VideoCore::GxStats::cmdlist_count.exchange(0, std::memory_order_relaxed);
                if (reg_n > 0 || fills > 0 || transfers > 0 || cmdlists > 0) {
                    Common::VitaNote(
                        "gx 4.7",
                        fmt::format("regw n={} t_us={:.1} fill={} tran={} cmdl={}", reg_n,
                                    static_cast<double>(reg_ns) / 1000.0, fills, transfers,
                                    cmdlists)
                            .c_str());
                }
            }
            Common::FrameStats::Reset();
        }

        // Los contadores del rasterizador de la GPU y los del software se
        // vacian aqui, una vez por intervalo, para que la linea diga la
        // proporcion del ultimo segundo.
        stats_tri_gpu = Gxm::RasterizerGXM::gpu_triangles.exchange(0, std::memory_order_relaxed);
        stats_gpu_batches = Gxm::RasterizerGXM::gpu_batches.exchange(0, std::memory_order_relaxed);
        stats_hw_vs_batches =
            Gxm::RasterizerGXM::hw_vs_batches.exchange(0, std::memory_order_relaxed);
        stats_hw_vs_rejects =
            Gxm::RasterizerGXM::hw_vs_rejects.exchange(0, std::memory_order_relaxed);
        stats_hw_vs_reason =
            Gxm::RasterizerGXM::hw_vs_last_reject.load(std::memory_order_relaxed);
        stats_gpu_scenes = Gxm::RasterizerGXM::gpu_scenes.exchange(0, std::memory_order_relaxed);
        stats_gpu_writebacks =
            Gxm::RasterizerGXM::gpu_writebacks.exchange(0, std::memory_order_relaxed);
        stats_close_full =
            Gxm::RasterizerGXM::scene_close_full.exchange(0, std::memory_order_relaxed);
        stats_close_lut =
            Gxm::RasterizerGXM::scene_close_lut.exchange(0, std::memory_order_relaxed);
        stats_tri_sw =
            Gxm::RasterizerGXM::software_triangles.exchange(0, std::memory_order_relaxed);

        // Reparto del tiempo del rasterizador por triangulo: cuanto se va en
        // preparar cada triangulo, cuanto en el tramo paralelo y cuanto
        // rasterizando el triangulo entero en el propio hilo (los pequenos).
        // Ver tri_*_us en sw_rasterizer.h. Ojo con las unidades: estos son
        // MICROsegundos y los de GxStats NANOsegundos.
        double tri_total_us = 0.0;
        {
            const u64 setup_us =
                SwRenderer::RasterizerStats::tri_setup_us.load(std::memory_order_relaxed);
            const u64 wait_us =
                SwRenderer::RasterizerStats::tri_wait_us.load(std::memory_order_relaxed);
            const u64 single_us =
                SwRenderer::RasterizerStats::tri_single_us.load(std::memory_order_relaxed);
            const u64 busy_us =
                SwRenderer::RasterizerStats::band_busy_us.load(std::memory_order_relaxed);
            const u32 bands =
                SwRenderer::RasterizerStats::raster_bands.load(std::memory_order_relaxed);
            const u64 total_us = setup_us + wait_us + single_us;
            tri_total_us = static_cast<double>(total_us);
            if (total_us > 0) {
                stats_setup_percent =
                    static_cast<double>(setup_us) / static_cast<double>(total_us) * 100.0;
                stats_wait_percent =
                    static_cast<double>(wait_us) / static_cast<double>(total_us) * 100.0;
                stats_single_percent =
                    static_cast<double>(single_us) / static_cast<double>(total_us) * 100.0;
            }

            /**
             * OCUPACION: que porcentaje del tiempo de los tres nucleos se gasta
             * de verdad dentro de una banda, sombreando pixeles.
             *
             * 'esp' dice cuanto dura el tramo paralelo, pero no cuanto de ese
             * tramo es trabajo y cuanto es esperar a que el planificador
             * despierte a los hilos. Sin este numero, los ciclos por pixel se
             * estaban deduciendo suponiendo que los tres nucleos van llenos.
             *
             *   ocu cerca de 100% -> los nucleos estan llenos y lo caro es el
             *                        pixel: toca abaratar el bucle.
             *   ocu bastante por debajo -> se esta pagando sincronizacion y hay
             *                        que arreglar el reparto, no la aritmetica.
             */
            if (wait_us > 0 && bands > 0) {
                stats_occupancy_percent = static_cast<double>(busy_us) /
                                          (static_cast<double>(wait_us) *
                                           static_cast<double>(bands)) *
                                          100.0;
            } else {
                stats_occupancy_percent = 0.0;
            }
        }

        // Contadores del rasterizador, normalizados por fotograma para que el
        // numero signifique lo mismo aunque cambie la velocidad.
        const u32 tri = SwRenderer::RasterizerStats::triangles.load(std::memory_order_relaxed);
        const u32 tested =
            SwRenderer::RasterizerStats::pixels_tested.load(std::memory_order_relaxed);
        const u32 drawn = SwRenderer::RasterizerStats::pixels_drawn.load(std::memory_order_relaxed);
        const u32 covered =
            SwRenderer::RasterizerStats::pixels_covered.load(std::memory_order_relaxed);
        const u32 zkill = SwRenderer::RasterizerStats::pixels_zkill.load(std::memory_order_relaxed);
        const u32 alpha_fail =
            SwRenderer::RasterizerStats::pixels_alpha_fail.load(std::memory_order_relaxed);
        const u32 depth_fail =
            SwRenderer::RasterizerStats::pixels_depth_fail.load(std::memory_order_relaxed);
        const u32 lit = SwRenderer::RasterizerStats::pixels_lit.load(std::memory_order_relaxed);
        const u32 t_split = SwRenderer::RasterizerStats::tri_split.load(std::memory_order_relaxed);
        const u32 t_single = SwRenderer::RasterizerStats::tri_single.load(std::memory_order_relaxed);
        const u32 tx_total =
            SwRenderer::RasterizerStats::tex_units_total.load(std::memory_order_relaxed);
        const u32 tx_etc1 =
            SwRenderer::RasterizerStats::tex_units_etc1.load(std::memory_order_relaxed);
        const u32 tev_used =
            SwRenderer::RasterizerStats::tev_stages_used.load(std::memory_order_relaxed);
        SwRenderer::RasterizerStats::Reset();

        // Que porcentaje de pixeles cubiertos pasa por la iluminacion, y que
        // porcentaje de triangulos se reparte de verdad entre los tres hilos.
        // Los dos salen de la misma cuenta que no cuadraba: ~7.600 ciclos por
        // pixel. Uno de los dos deberia explicarla.
        // Media de etapas TEV que hacen algo por triangulo, de 1 a 6. Si sale
        // cerca de 6, saltarse las inertes no aporta nada en este juego.
        stats_tev_stages = tri > 0 ? static_cast<double>(tev_used) / static_cast<double>(tri) : 0.0;
        // Porcentaje de texeles que vienen comprimidos en ETC1.
        stats_etc1_percent = tx_total > 0 ? static_cast<double>(tx_etc1) /
                                                static_cast<double>(tx_total) * 100.0
                                          : 0.0;

        stats_lit_percent =
            covered > 0 ? static_cast<double>(lit) / static_cast<double>(covered) * 100.0 : 0.0;
        const u32 tri_total = t_split + t_single;
        stats_split_percent =
            tri_total > 0 ? static_cast<double>(t_split) / static_cast<double>(tri_total) * 100.0
                          : 0.0;

        // Los contadores se acumulan durante el intervalo, no durante un
        // fotograma: hay que dividir por los fotogramas que han cabido dentro.
        //
        // Se usa el contador propio del rasterizador, no el FPS de PerfStats: a
        // 0,4 FPS ese FPS es fraccionario y dividir por el convertia el ruido
        // del muestreo en cifras absurdas (el overdraw saltaba de 12x a 66x).
        const u32 frame_count = SwRenderer::RasterizerStats::frames.load(std::memory_order_relaxed);
        const double frames_in_interval = frame_count > 0 ? static_cast<double>(frame_count) : 1.0;
        stats_triangles_per_frame = static_cast<double>(tri) / frames_in_interval;

        // Overdraw: veces que se pinta cada pixel de las dos pantallas del 3DS
        // (400x240 + 320x240 = 172.800) en un fotograma. 1.0 seria pintar cada
        // pixel exactamente una vez.
        constexpr double kScreenPixels = 400.0 * 240.0 + 320.0 * 240.0;
        stats_overdraw = static_cast<double>(drawn) / frames_in_interval / kScreenPixels;

        // Pixeles probados por cada uno pintado. El rasterizador recorre la caja
        // que envuelve al triangulo, asi que un valor alto significa que se esta
        // gastando el tiempo en pixeles que quedan fuera.
        stats_test_ratio = drawn > 0 ? static_cast<double>(tested) / static_cast<double>(drawn) : 0.0;

        // Reparto de los pixeles que SI caen dentro del triangulo. Estos dos son
        // los que dicen que optimizacion toca:
        //
        //   cov: veces que se cubre cada pixel de pantalla con geometria. Si es
        //        mucho mayor que 'draw', se esta sombreando para tirarlo.
        //   zk:  de esos cubiertos, que porcentaje mata el rechazo temprano por
        //        profundidad. Mide directamente lo que ahorra esa optimizacion:
        //        si sale 0%, en esta escena no sirve de nada y hay que atacar el
        //        coste por pixel; si sale alto, ya se esta ahorrando ese trabajo.
        stats_covered = static_cast<double>(covered) / frames_in_interval / kScreenPixels;
        stats_zkill_percent =
            covered > 0 ? static_cast<double>(zkill) / static_cast<double>(covered) * 100.0 : 0.0;
        stats_alpha_fail_percent = covered > 0 ? static_cast<double>(alpha_fail) /
                                                     static_cast<double>(covered) * 100.0
                                               : 0.0;
        stats_depth_fail_percent = covered > 0 ? static_cast<double>(depth_fail) /
                                                     static_cast<double>(covered) * 100.0
                                               : 0.0;
        {
            const u32 early_z_tri =
                SwRenderer::RasterizerStats::tri_early_z.load(std::memory_order_relaxed);
            stats_early_z_percent =
                tri > 0 ? static_cast<double>(early_z_tri) / static_cast<double>(tri) * 100.0 : 0.0;
        }

        // Reparto de unidades de textura por formato. Se formatea aqui, que es
        // cuando los contadores estan frescos: RasterizerStats::Reset() los pone
        // a cero justo despues.
        {
            static const char* const kFormatNames[14] = {"rgba8", "rgb8", "b5a1", "b565",
                                                         "rgba4", "ia8",  "rg8",  "i8",
                                                         "a8",    "ia4",  "i4",   "a4",
                                                         "etc1",  "ea4"};
            std::size_t nf = 0;
            const char* head = "fmt";
            while (head[nf] != '\0') {
                stats_format_line[nf] = head[nf];
                nf++;
            }
            for (u32 i = 0; i < SwRenderer::RasterizerStats::tex_format_units.size(); i++) {
                const u32 count =
                    SwRenderer::RasterizerStats::tex_format_units[i].load(std::memory_order_relaxed);
                if (count == 0) {
                    continue;
                }
                const char* name = kFormatNames[i];
                const std::size_t name_len = std::strlen(name);
                if (nf + name_len + 8 >= sizeof(stats_format_line)) {
                    break;
                }
                stats_format_line[nf++] = ' ';
                for (std::size_t j = 0; j < name_len; j++) {
                    stats_format_line[nf++] = name[j];
                }
                stats_format_line[nf++] = ' ';
                nf += AppendUInt(stats_format_line + nf, count);
            }
            stats_format_line[nf] = '\0';
        }

        // Reparto del "gpu %" por tipo de comando GX. El porcentaje de PerfStats
        // los mete todos en el mismo saco, y son tres trabajos distintos con
        // arreglos distintos. Ver VideoCore::GxStats.
        const double cmdlist = static_cast<double>(
            VideoCore::GxStats::cmdlist_ns.load(std::memory_order_relaxed));
        const double fill =
            static_cast<double>(VideoCore::GxStats::fill_ns.load(std::memory_order_relaxed));
        const double transfer = static_cast<double>(
            VideoCore::GxStats::transfer_ns.load(std::memory_order_relaxed));
        const double dma =
            static_cast<double>(VideoCore::GxStats::dma_ns.load(std::memory_order_relaxed));
        VideoCore::GxStats::Reset();

        // Se expresan sobre el total de GX, no sobre el fotograma: asi suman 100
        // entre ellos y se ve de un vistazo cual manda dentro del 'gpu %'.
        // Nanosegundos de reloj de pared por pixel cubierto.
        //
        // Es la medida que llevo toda la sesion deduciendo a mano a partir de
        // FPS, cov y una frecuencia supuesta -- y equivocandome. Esto lo mide
        // directamente: tiempo real dentro de las listas de comandos dividido
        // entre los pixeles que de verdad se sombrearon. Sin suposiciones.
        stats_ns_per_pixel = covered > 0 ? cmdlist / static_cast<double>(covered) : 0.0;

        /**
         * Que parte de "gx tri" es de verdad ProcessTriangle.
         *
         * El desglose 'set/esp/un' reparte el 100% del tiempo del rasterizador
         * entre sus tres trozos, pero nunca se ha comprobado que ese 100% sea
         * el 100% de las listas de comandos. Ahi dentro hay ademas el sombreado
         * de vertices (interpretado), el recorte contra los seis planos y la
         * decodificacion de comandos: si eso fuese la mitad, abaratar el bucle
         * de pixeles al doble solo daria un 25% de fotograma.
         *
         * tri_*_us viene en MICROsegundos (sceKernelGetProcessTimeWide) y
         * cmdlist en NANOsegundos (steady_clock): de ahi el x1000.
         */
        stats_raster_share_percent =
            cmdlist > 0.0 ? tri_total_us * 1000.0 / cmdlist * 100.0 : 0.0;

        const double gx_total = cmdlist + fill + transfer + dma;
        if (gx_total > 0.0) {
            stats_gx_cmdlist = cmdlist / gx_total * 100.0;
            stats_gx_fill = fill / gx_total * 100.0;
            stats_gx_transfer = transfer / gx_total * 100.0;
        } else {
            stats_gx_cmdlist = 0.0;
            stats_gx_fill = 0.0;
            stats_gx_transfer = 0.0;
        }

        /**
         * Vaciados de la cache de traduccion EN ESTE INTERVALO (0.1.0.43).
         *
         * Antes se mostraba el total desde el arranque, acotado a 9999, y en
         * Rubi Omega salia 9999 fijo: no habia forma de saber si los vaciados
         * venian de la carga (ldr_ro parchea los CRO palabra a palabra y cada
         * parche vacia la cache entera) o si seguian ocurriendo en cada
         * fotograma, que es lo que importaria para el tiempo de 'cpu'.
         */
        {
            std::size_t by_capacity = 0;
            std::size_t by_invalidation = 0;
            GetTransCacheFlushCounts(by_capacity, by_invalidation);
            stats_flushes_capacity = by_capacity - last_flushes_capacity;
            stats_flushes_invalidation = by_invalidation - last_flushes_invalidation;
            last_flushes_capacity = by_capacity;
            last_flushes_invalidation = by_invalidation;
        }
        {
            /**
             * EL OVERLAY EN crash.txt (0.1.8.1): cada 10 s las lineas que
             * dicen donde se va el fotograma. Sin esto cada prueba en consola
             * necesitaba una captura de pantalla, y crash.txt es lo unico que
             * llega siempre. Treinta veces (5 minutos) desde 0.2.0.0: con
             * seis solo salia el primer minuto, que suele ser la intro.
             * Por ventana desde 0.2.1.8: como static valian para toda la
             * sesion, y el segundo juego sin reiniciar no anotaba nada.
             */
            if (++frame_ticks % 10 == 0 && frame_notes < 30) {
                frame_notes++;
                Common::VitaNote(
                    "fotograma",
                    fmt::format("fps {:.1f} vel {:.0f}% | ms {:.1f} cpu {:.1f} gx {:.1f} svc {:.1f} "
                                "vtx {:.1f} lote {:.1f} dsp {:.1f} (aac {:.1f} decodificar {:.1f}, "
                                "{:.1f} fuentes, otro nucleo {:.1f}) | conv {:.1f} sub {:.1f} "
                                "dib {:.1f} ovl {:.1f} esp {:.1f} | sh {:.1f} fin {:.1f} tx {:.1f}",
                                stats_game_fps, stats_speed_percent, stats_frame_ms, stats_cpu_ms,
                                stats_gx_ms, stats_svc_ms, stats_vertices_ms, stats_batch_ms,
                                stats_dsp_ms, stats_aac_ms, stats_dsp_decode_ms, stats_dsp_sources,
                                stats_dsp_worker_ms, stats_convert_ms, stats_upload_ms, stats_draw_ms,
                                stats_overlay_ms, stats_swapwait_ms, stats_shade_ms,
                                stats_finish_ms, stats_texdecode_ms)
                        .c_str());
                Common::VitaNote(
                    "gpu",
                    fmt::format("tri/fot {:.0f} | tg {} ts {} lot {} vsg {} no {} ({}) vsh {} "
                                "esc {} vol {} | gx tri {:.0f}% rell {:.0f}% tran {:.0f}% | "
                                "par {:.0f} ins {:.0f} rap {:.0f}% | copia gpu {} de verdad {} | relleno gpu {} "
                                "sinc soft {} | saltados por compilar {}",
                                stats_triangles_per_frame, stats_tri_gpu, stats_tri_sw,
                                stats_gpu_batches, stats_hw_vs_batches, stats_hw_vs_rejects,
                                Gxm::RasterizerGXM::TakeRejectSummary(),
                                stats_vertices_shaded, stats_gpu_scenes, stats_gpu_writebacks,
                                stats_gx_cmdlist, stats_gx_fill, stats_gx_transfer,
                                stats_shade_occupancy, stats_instrs_per_vertex, stats_fast_percent,
                                Gxm::RasterizerGXM::gpu_transfers.exchange(
                                    0, std::memory_order_relaxed),
                                Gxm::RasterizerGXM::transfer_materialized.exchange(
                                    0, std::memory_order_relaxed),
                                Gxm::RasterizerGXM::gpu_fills.exchange(
                                    0, std::memory_order_relaxed),
                                Gxm::RasterizerGXM::software_syncs.exchange(
                                    0, std::memory_order_relaxed),
                                Gxm::RasterizerGXM::skipped_batches.exchange(
                                    0, std::memory_order_relaxed))
                        .c_str());
                // Lo que cuesta cargar modelos nuevos (0.2.1.1): texturas
                // decodificadas o revisadas, y las esperas enteras a la GPU.
                Common::VitaNote(
                    "texturas",
                    fmt::format("decodificadas {} ({:.1f} ms/fot) | revisadas iguales {} "
                                "cambiadas {} expulsadas {} (hash {:.1f} ms y {:.0f} KB por fot) "
                                "| escenas cerradas por vertices {} por tablas de luz {} | "
                                "espera a la gpu {:.1f} ms/fot",
                                stats_texdecodes, stats_texdecode_ms, stats_texreuses,
                                stats_texchanged, stats_texevictions, stats_texrehash_ms,
                                stats_texrehash_kb, stats_close_full, stats_close_lut,
                                stats_finish_ms)
                        .c_str());
                Common::VitaNote(
                    "gpu hilo",
                    fmt::format("{} | ocupado {:.0f}% ({:.1f} ms/fot) | juego esperando {:.0f}% "
                                "({:.1f} ms/fot) | esperas irq {} sinc {} pres {} | cola max {}",
                                VideoCore::GPU::async_enabled.load(std::memory_order_relaxed)
                                    ? "ON"
                                    : "off",
                                stats_gpu_busy_percent,
                                stats_gpu_busy_percent * stats_frame_ms / 100.0,
                                stats_gpu_wait_percent,
                                stats_gpu_wait_percent * stats_frame_ms / 100.0,
                                stats_gpu_irq_waits, stats_gpu_syncs, stats_gpu_present_waits,
                                stats_gpu_queue_max)
                        .c_str());
                Common::VitaNote("lote fases", Gxm::RasterizerGXM::TakeBatchProfile().c_str());
                Common::VitaNote("esperas gpu", stats_gpu_waits.c_str());
                Common::VitaNote("superficies",
                                 Gxm::RasterizerGXM::TakeSurfaceSummary().c_str());
                Common::VitaNote("copias", Gxm::RasterizerGXM::TakeCopySummary().c_str());
                Common::VitaNote("jit otro", Core::ArmJit::TakeRejectWords().c_str());
                Common::VitaNote("bucle", Core::TakeLoopProfile().c_str());
                Common::VitaNote("eventos", Core::TakeTimingProfile().c_str());
                Common::VitaNote("hilos", Common::VitaThreadSummary().c_str());
                Common::VitaNote("vs flujo", Pica::Shader::Fast::TakeFlowSummary().c_str());
                u32 code_bytes = 0;
                u32 code_blocks = 0;
                Core::ArmJit::CodeUsage(code_bytes, code_blocks);
                Common::VitaNote(
                    "jit",
                    fmt::format("{} {:.0f}% Mi {:.2f} arm {:.1f} | bloques {} rechazados {} "
                                "comprobados {} ya comprobados {} diferencias {} | rech {} "
                                "{:.0f}% | codigo {} KB {} B/bloque",
                                stats_jit_on ? "ON" : "off", stats_jit_percent, stats_guest_mips,
                                stats_arm_ms, stats_jit_blocks, stats_jit_rejected,
                                stats_jit_checks,
                                Common::FrameStats::jit_reverified.exchange(
                                    0, std::memory_order_relaxed),
                                stats_jit_mismatches,
                                Core::ArmJit::RejectName(stats_jit_top_reject),
                                stats_jit_top_reject_percent, code_bytes / 1024,
                                code_blocks != 0 ? code_bytes / code_blocks : 0)
                        .c_str());
            }
        }
        /**
         * El completo cada 3 s (0.3.1.2): sus ~2.000 letras cuestan ~150 ms de
         * vita2d en cada actualizacion (entre 5 y 20 ms por fotograma de media
         * con una por segundo). El compacto y las notas de crash.txt, cada
         * segundo como siempre.
         */
        stats_next_update_us =
            now_us + (stats_overlay_visible && stats_overlay_full ? 3'000'000 : 1'000'000);
    }

    // 4.10 (0.1.5.2): SELECT+TRIANGULO apaga solo el DIBUJADO del overlay
    // (ver UpdateFrameSkipControl). Los contadores de arriba SIGUEN
    // actualizandose y anotando en crash.txt: si se saltara el bloque de
    // arriba, los atomics se acumularian sin Reset y no saldrian ni las notas
    // de 4.2a/4.7. Lo unico que se ahorra aqui es el coste de 'ovl' (~2.3 ms).
    if (!stats_overlay_visible) {
        return;
    }

    char line1[20] = "FPS(5s) ";
    std::size_t n1 = 8;
    AppendOneDecimal(line1, n1, stats_game_fps);
    line1[n1] = '\0';

    char line2[16] = "vel ";
    std::size_t n2 = 4;
    AppendOneDecimal(line2, n2, stats_speed_percent);
    line2[n2++] = '%';
    line2[n2] = '\0';

    /**
     * OVERLAY COMPACTO (0.2.2.2). Las ~30 lineas son unos 2.000 caracteres, y
     * vita2d dibuja cada uno con su propia llamada: cada actualizacion (una
     * por segundo) costaba ~150 ms en el hilo que presenta, entre 3 y 20 ms
     * por fotograma de media en crash.txt ("ovl"), un tiron cada segundo y
     * FPS de menos justo mientras se miraban los FPS. Por defecto, una linea;
     * todo lo demas ya va a crash.txt cada 10 s.
     */
    if (!stats_overlay_full) {
        char compact[96];
        std::size_t c = 0;
        const auto put = [&](const char* text) {
            while (*text != '\0' && c < sizeof(compact) - 8) {
                compact[c++] = *text++;
            }
        };
        put(line1);
        put("  ");
        put(line2);
        put("  ms ");
        AppendMillis(compact, c, stats_frame_ms);
        compact[c] = '\0';
        vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 20, 0xFF60C0FF, 0.8f,
                             compact);
        return;
    }

    // Cuentas de vaciado de la cache de traduccion: para decidir si merece la
    // pena agrandar TRANS_CACHE_SIZE. "cap" es la unica que tiene que ver con
    // el tamano del buffer; "inv" (cambios de proceso, principalmente) pasaria
    // igual con un buffer infinito. Son enteros pequenos en la practica, pero
    // se acotan a 9999 por si acaso: el buffer es de sobra para eso, no para un
    // desbordamiento real.
    const std::size_t flushes_capacity = stats_flushes_capacity;
    const std::size_t flushes_invalidation = stats_flushes_invalidation;

    char line3[32] = "cache cap ";
    std::size_t n3 = 10;
    n3 += AppendUInt(line3 + n3,
                     static_cast<unsigned int>(std::min<std::size_t>(flushes_capacity, 9999)));
    const char* inv_label = " inv ";
    for (std::size_t i = 0; inv_label[i] != '\0'; i++) {
        line3[n3++] = inv_label[i];
    }
    n3 += AppendUInt(line3 + n3,
                     static_cast<unsigned int>(std::min<std::size_t>(flushes_invalidation, 9999)));
    line3[n3] = '\0';

    // Reparto del tiempo. Una linea por apartado, para poder leerlo de un
    // vistazo sin tener que mandar ficheros.
    const auto build_percent_line = [](char* out, const char* label, double value) {
        std::size_t n = 0;
        while (label[n] != '\0') {
            out[n] = label[n];
            n++;
        }
        AppendOneDecimal(out, n, value);
        out[n++] = '%';
        out[n] = '\0';
    };

    char line_cpu[24];
    build_percent_line(line_cpu, "cpu ", stats_cpu_percent);
    char line_gpu[24];
    build_percent_line(line_gpu, "gpu ", stats_gpu_percent);
    char line_svc[48];
    build_percent_line(line_svc, "svc ", stats_svc_percent);
    // El swap (presentacion por vita2d incluida) no se veia en ninguna linea y
    // puede ser un pellizco serio del fotograma.
    {
        std::size_t nsv = std::strlen(line_svc);
        const char* swap_label = "  swap ";
        for (std::size_t i = 0; swap_label[i] != '\0'; i++) {
            line_svc[nsv++] = swap_label[i];
        }
        AppendOneDecimal(line_svc, nsv, stats_swap_percent);
        line_svc[nsv++] = '%';
        line_svc[nsv] = '\0';
    }

    // La version en pantalla evita la duda de "seguro que instalaste la nueva?".
    // Detras, ovl ON/OFF: como el overlay puede apagarse entero (SELECT+TRIANGULO),
    // hay que poder saber SIN overlay si esta encendido -- si no, al ver una
    // captura limpia no se distingue "lo apago yo" de "no lo pinta".
    char version_line[48];
    {
        std::size_t vn = 0;
        const char* vb = kOverlayBuild;
        for (std::size_t i = 0; vb[i] != '\0' && vn + 2 < sizeof(version_line); i++) {
            version_line[vn++] = vb[i];
        }
        const char* ovl_tag = "  ovl ON";
        for (std::size_t i = 0; ovl_tag[i] != '\0' && vn + 1 < sizeof(version_line); i++) {
            version_line[vn++] = ovl_tag[i];
        }
        // g1 = GPU en el nucleo 1: % ocupada / % del tiempo que el juego la espera.
        if (VideoCore::GPU::async_enabled.load(std::memory_order_relaxed) &&
            vn + 16 < sizeof(version_line)) {
            const char* gpu_tag = "  g1 ";
            for (std::size_t i = 0; gpu_tag[i] != '\0'; i++) {
                version_line[vn++] = gpu_tag[i];
            }
            vn += AppendUInt(version_line + vn,
                             static_cast<unsigned int>(std::min(stats_gpu_busy_percent, 999.0)));
            version_line[vn++] = '/';
            vn += AppendUInt(version_line + vn,
                             static_cast<unsigned int>(std::min(stats_gpu_wait_percent, 999.0)));
        }
        version_line[vn] = '\0';
    }
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 20, 0xFF60C0FF, 0.8f, version_line);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 38, 0xFFA0A0A0, 0.8f, line1);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 56, 0xFFA0A0A0, 0.8f, line2);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 80, 0xFFA0A0A0, 0.8f, line_cpu);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 98, 0xFFA0A0A0, 0.8f, line_gpu);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 116, 0xFFA0A0A0, 0.8f, line_svc);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 140, 0xFFA0A0A0, 0.8f, line3);

    // Diagnostico del rasterizador: por que cuesta el 95% del fotograma.
    char line_tri[24];
    build_percent_line(line_tri, "tri ", stats_triangles_per_frame);
    line_tri[std::strlen(line_tri) - 1] = '\0'; // no es un porcentaje
    char line_over[24];
    build_percent_line(line_over, "draw ", stats_overdraw);
    line_over[std::strlen(line_over) - 1] = 'x';
    char line_test[24];
    build_percent_line(line_test, "test ", stats_test_ratio);
    line_test[std::strlen(line_test) - 1] = 'x';
    char line_cov[24];
    build_percent_line(line_cov, "cov ", stats_covered);
    line_cov[std::strlen(line_cov) - 1] = 'x';
    char line_zk[32];
    build_percent_line(line_zk, "zk ", stats_zkill_percent); // este si es porcentaje
    // 'ez' = en que porcentaje de triangulos el rechazo temprano esta
    // disponible. Con zk 0%, si ez tambien es 0 la culpa es de la galga (lo
    // desactiva), no de la escena; y entonces extenderlo tiene recorrido.
    {
        std::size_t nzk = std::strlen(line_zk);
        const char* ez_label = " ez ";
        for (std::size_t i = 0; ez_label[i] != '\0'; i++) {
            line_zk[nzk++] = ez_label[i];
        }
        nzk += AppendUInt(line_zk + nzk, static_cast<unsigned int>(stats_early_z_percent));
        line_zk[nzk++] = '%';
        line_zk[nzk] = '\0';
    }

    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 164, 0xFFFFC060, 0.8f, line_tri);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 182, 0xFFFFC060, 0.8f, line_over);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 200, 0xFFFFC060, 0.8f, line_test);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 218, 0xFFFFC060, 0.8f, line_cov);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 236, 0xFFFFC060, 0.8f, line_zk);

    // Salto de fotogramas, con el recordatorio de como se cambia. Va aparte y
    // en otro color porque no es una medida: es un ajuste que se toca en vivo.
    // Dentro del 'gpu %': que parte es rasterizado (listas de comandos), que
    // parte rellenos de memoria y que parte transferencias de pantalla. Sin
    // esto, "gpu 94%" no dice cual de los tres hay que arreglar.
    char line_gx[40] = "gx tri ";
    std::size_t ng = 7;
    AppendOneDecimal(line_gx, ng, stats_gx_cmdlist);
    const char* gx_fill_label = " rell ";
    for (std::size_t i = 0; gx_fill_label[i] != '\0'; i++) {
        line_gx[ng++] = gx_fill_label[i];
    }
    AppendOneDecimal(line_gx, ng, stats_gx_fill);
    const char* gx_tr_label = " tran ";
    for (std::size_t i = 0; gx_tr_label[i] != '\0'; i++) {
        line_gx[ng++] = gx_tr_label[i];
    }
    AppendOneDecimal(line_gx, ng, stats_gx_transfer);
    line_gx[ng] = '\0';
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 284, 0xFF60FFFF, 0.8f, line_gx);

    // Las dos hipotesis para los ~7.600 ciclos por pixel, en una linea:
    // 'luz' = % de pixeles que pasan por iluminacion por fragmento.
    // 'hilos' = % de triangulos que se reparten entre los tres nucleos.
    // El reloj real de la CPU y el coste medido por pixel. Los dos existen
    // porque llevaba toda la sesion deduciendo ciclos por pixel a partir de una
    // frecuencia que nunca comprobe y de cifras derivadas de otras cifras.
    char line_hw[192] = "cpu ";
    std::size_t nh = 4;
    nh += AppendUInt(line_hw + nh, static_cast<unsigned int>(Common::g_arm_clock_mhz));
    // 0.1.4.9: fuera 'ns/px' y 'denorm' de esta linea. Eran medidas del
    // rasterizador de software, que ya casi no dibuja nada (salian a 0), y la
    // linea hace falta para el JIT.
    line_hw[nh++] = 'M';
    line_hw[nh++] = 'H';
    line_hw[nh++] = 'z';
    /**
     * El JIT del ARM11 (0.1.4.8), en la misma linea:
     *   jit ON/off   estado (SELECT + DERECHA lo cambia)
     *   N%           instrucciones del juego que ha ejecutado el JIT
     *   Mi X         millones de instrucciones del juego por vblank (JIT +
     *                interprete): con 'cpu' da el coste por instruccion
     *   b N x N      bloques compilados / rechazados (desde el arranque)
     *   c N          bloques comprobados contra el interprete (por segundo)
     *   d N          DIFERENCIAS (desde el arranque): tiene que ser 0; si no,
     *                crash.txt dice que bloque y que registro
     */
    {
        const auto append_text = [&](const char* text) {
            for (std::size_t i = 0; text[i] != '\0'; i++) {
                line_hw[nh++] = text[i];
            }
        };
        append_text(stats_jit_on ? "  jit ON " : "  jit off ");
        nh += AppendUInt(line_hw + nh, static_cast<unsigned int>(stats_jit_percent + 0.5));
        append_text("% Mi ");
        AppendOneDecimal(line_hw, nh, stats_guest_mips);
        // arm = ms reales del bucle del ARM por vblank (dentro de 'cpu').
        append_text(" arm ");
        AppendOneDecimal(line_hw, nh, stats_arm_ms);
        append_text(" b");
        nh += AppendUInt(line_hw + nh, stats_jit_blocks);
        append_text(" x");
        nh += AppendUInt(line_hw + nh, stats_jit_rejected);
        append_text(" c");
        nh += AppendUInt(line_hw + nh, stats_jit_checks);
        append_text(" d");
        nh += AppendUInt(line_hw + nh, stats_jit_mismatches);
        // rech = lo que mas se queda en el interprete, y en que % de despachos.
        append_text(" rech ");
        append_text(Core::ArmJit::RejectName(stats_jit_top_reject));
        line_hw[nh++] = ' ';
        nh += AppendUInt(line_hw + nh,
                         static_cast<unsigned int>(stats_jit_top_reject_percent + 0.5));
        line_hw[nh++] = '%';
    }
    line_hw[nh] = '\0';
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 332, 0xFF80FF80, 0.8f, line_hw);

    char line_why[64] = "luz ";
    std::size_t nw = 4;
    AppendOneDecimal(line_why, nw, stats_lit_percent);
    const char* why_label = "%  tev ";
    for (std::size_t i = 0; why_label[i] != '\0'; i++) {
        line_why[nw++] = why_label[i];
    }
    AppendOneDecimal(line_why, nw, stats_tev_stages);
    line_why[nw++] = 0x2F; line_why[nw++] = 0x36; // "/6": etapas, no porcentaje
    const char* etc_label = "  etc1 ";
    for (std::size_t i = 0; etc_label[i] != 0; i++) {
        line_why[nw++] = etc_label[i];
    }
    AppendOneDecimal(line_why, nw, stats_etc1_percent);
    line_why[nw++] = 0x25;
    line_why[nw] = '\0';
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 308, 0xFFFF80FF, 0.8f, line_why);

    char line_skip[32] = "salto 1/";
    std::size_t ns = 8;
    ns += AppendUInt(line_skip + ns,
                     SwRenderer::FrameSkip::interval.load(std::memory_order_relaxed));
    const char* hint = "  SEL+L/R";
    for (std::size_t i = 0; hint[i] != '\0'; i++) {
        line_skip[ns++] = hint[i];
    }
    line_skip[ns] = '\0';
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 260, 0xFF80FF80, 0.8f, line_skip);

    // Marcador de la resolucion: verde en 1x, naranja en 0.5x. Es una perdida
    // de calidad y conviene que se note que el modo esta puesto, no que pase
    // desapercibido.
    const bool half_on = SwRenderer::FrameSkip::half_resolution.load(std::memory_order_relaxed);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 356,
                         half_on ? 0xFF40C0FF : 0xFF80FF80, 0.8f,
                         half_on ? "res 0.5x  L+R" : "res 1x  L+R");
    {
        /**
         * Desglose del ARM por fotograma (0.1.6.3), a la derecha de "res":
         *   rod = rodajas del ARM         desp = bloques despachados
         *   enl = enlaces en el codigo    lent = accesos lentos a memoria
         *   vfp = aritmetica VFP           int = miles de instrucciones por el
         *                                        interprete
         */
        char line_arm[96];
        std::snprintf(line_arm, sizeof(line_arm), "rod %.0f desp %.0f enl %.0f lent %.0f vfp %.0f int %.1fk",
                      stats_arm_slices, stats_arm_dispatches, stats_arm_links, stats_arm_slow,
                      stats_arm_vfp, stats_arm_interp_k);
        vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX) + 110, 356, 0xFF80FF80,
                             0.8f, line_arm);
    }

    // De los pixeles cubiertos, los que mueren en la prueba de alfa y los que
    // mueren en la de profundidad final. Decision que desbloquean: si manda
    // 'df', se esta sombreando geometria tapada y compensa adelantar la prueba
    // de profundidad; si manda 'af', no hay nada que adelantar (el alfa sale
    // del propio sombreado) y toca abaratar el bucle de pixel.
    char line_fail[40] = "af ";
    std::size_t nfail = 3;
    AppendOneDecimal(line_fail, nfail, stats_alpha_fail_percent);
    line_fail[nfail++] = '%';
    const char* df_label = "  df ";
    for (std::size_t i = 0; df_label[i] != '\0'; i++) {
        line_fail[nfail++] = df_label[i];
    }
    AppendOneDecimal(line_fail, nfail, stats_depth_fail_percent);
    line_fail[nfail++] = '%';
    line_fail[nfail] = '\0';

    // Formatos de textura que se usan de verdad, por unidades y triangulo.
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 380, 0xFFFFA0A0, 0.8f, line_fail);
    {
        /**
         * El reparto de 'arm' en ms por fotograma (0.1.7.5):
         *   jit = codigo generado; lento y vfp son la parte de jit que se va en
         *         los caminos lentos de memoria y en la VFP por funcion
         *   chk = comprobaciones contra el interprete   cmp = compilar
         *   rst = lo que queda: interprete y despacho
         */
        char line_armt[96];
        std::snprintf(line_armt, sizeof(line_armt),
                      "jit %.1f (lento %.1f vfp %.1f) chk %.1f cmp %.1f rst %.1f",
                      stats_arm_jit_ms, stats_arm_slow_ms, stats_arm_vfp_ms, stats_arm_check_ms,
                      stats_arm_compile_ms, stats_arm_rest_ms);
        vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX) + 130, 380, 0xFF80FF80,
                             0.8f, line_armt);
    }
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 398, 0xFFFFA0A0, 0.8f,
                         stats_format_line);

    // Desglose del rasterizador por triangulo:
    //   hil = % de triangulos repartidos entre los tres hilos
    //   set = % del tiempo en preparar el triangulo
    //   esp = % esperando a los hilos (raster en paralelo)
    //   un  = % rasterizando en el propio hilo (triangulos pequenos)
    const auto append_labeled_percent = [](char* out, std::size_t& n, const char* label,
                                           double value) {
        for (std::size_t i = 0; label[i] != '\0'; i++) {
            out[n++] = label[i];
        }
        AppendOneDecimal(out, n, value);
        out[n++] = '%';
    };

    char line_threads[64];
    std::size_t nth = 0;
    append_labeled_percent(line_threads, nth, "hil ", stats_split_percent);
    append_labeled_percent(line_threads, nth, " set ", stats_setup_percent);
    append_labeled_percent(line_threads, nth, " esp ", stats_wait_percent);
    append_labeled_percent(line_threads, nth, " un ", stats_single_percent);
    line_threads[nth] = '\0';
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 416, 0xFF80D0FF, 0.8f,
                         line_threads);

    // Las dos medidas que faltaban para dejar de suponer:
    //   ocu = ocupacion de los tres nucleos dentro del tramo paralelo. Separa
    //         "sombrear pixeles" de "esperar al planificador".
    //   ras = que parte de las listas de comandos es ProcessTriangle. Lo que
    //         quede fuera (vertices, recorte, decodificacion) es el techo de lo
    //         que puede dar cualquier optimizacion del bucle de pixeles.
    char line_occ[176];
    std::size_t noc = 0;
    append_labeled_percent(line_occ, noc, "ocu ", stats_occupancy_percent);
    append_labeled_percent(line_occ, noc, "  ras ", stats_raster_share_percent);
    {
        // tg = triangulos del intervalo en la GPU; ts = los que han caido al
        // software. Si ts manda, lo que falta no es optimizar: es cobertura.
        const char* gpu_label = "  tg ";
        for (std::size_t i = 0; gpu_label[i] != '\0'; i++) {
            line_occ[noc++] = gpu_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_tri_gpu);
        const char* sw_label = " ts ";
        for (std::size_t i = 0; sw_label[i] != '\0'; i++) {
            line_occ[noc++] = sw_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_tri_sw);
        // lot = lotes en la GPU. tg/lot es el tamano medio de lote: con lotes
        // pequenos manda el coste fijo por lote y con lotes grandes el de por
        // triangulo, y lo que hay que optimizar es lo contrario en cada caso.
        const char* batch_label = " lot ";
        for (std::size_t i = 0; batch_label[i] != '\0'; i++) {
            line_occ[noc++] = batch_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_gpu_batches);
        // vsg = de esos lotes, cuantos con el shader de vertices en la GPU
        // (0.1.4.6). Si es 0 con 'lot' alto, crash.txt dice por que.
        const char* hwvs_label = " vsg ";
        for (std::size_t i = 0; hwvs_label[i] != '\0'; i++) {
            line_occ[noc++] = hwvs_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_hw_vs_batches);
        // no = lotes que el shader de vertices en la GPU no pudo coger, y el
        // ultimo motivo (0.1.4.9): lo que hay que arreglar para subir 'vsg'.
        const char* hwvs_no_label = " no ";
        for (std::size_t i = 0; hwvs_no_label[i] != '\0'; i++) {
            line_occ[noc++] = hwvs_no_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_hw_vs_rejects);
        line_occ[noc++] = ' ';
        for (std::size_t i = 0; stats_hw_vs_reason[i] != '\0' && i < 28; i++) {
            line_occ[noc++] = stats_hw_vs_reason[i];
        }
        // vsh = vertices que han pasado por el interprete. Si iguala a tg*3 es
        // que el cache de vertices no sirve de nada aqui (dibujado no indexado).
        const char* shaded_label = " vsh ";
        for (std::size_t i = 0; shaded_label[i] != '\0'; i++) {
            line_occ[noc++] = shaded_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_vertices_shaded);
        // esc = escenas cerradas (cada una para la GPU); vol = volcados del
        // framebuffer (cada uno convierte la imagen entera). Los dos deberian
        // ser un punado por fotograma; si son decenas, ahi esta el tiempo.
        const char* scene_label = " esc ";
        for (std::size_t i = 0; scene_label[i] != '\0'; i++) {
            line_occ[noc++] = scene_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_gpu_scenes);
        const char* wb_label = " vol ";
        for (std::size_t i = 0; wb_label[i] != '\0'; i++) {
            line_occ[noc++] = wb_label[i];
        }
        noc += AppendUInt(line_occ + noc, stats_gpu_writebacks);
    }
    line_occ[noc] = '\0';
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 452, 0xFF80D0FF, 0.8f, line_occ);

    /**
     * EL REPARTO DEL FOTOGRAMA EN MILISEGUNDOS. La linea base del proyecto.
     *
     * Todo lo demas del overlay son porcentajes, y con porcentajes no se puede
     * decir si una version va mejor que otra: se reparten sobre el total, asi
     * que pueden salir identicos mientras el fotograma se parte por la mitad.
     * Estas dos lineas son las que se anotan antes de tocar nada y contra las
     * que se compara despues.
     *
     *   ms   = fotograma completo (tiempo de pared entre vblanks del invitado)
     *   cpu  = interprete del ARM11 y todo lo que no es GX, SVC ni presentar
     *   gx   = listas de comandos, rellenos y transferencias (rasterizado sw)
     *   svc  = llamadas al sistema del invitado e IPC
     *
     *   conv = framebuffer del 3DS -> ScreenInfo (CPU, hilo de emulacion)
     *   sub  = subir los pixeles a memoria de la GPU
     *   dib  = encolar el dibujado de las dos pantallas y este overlay
     *   esp  = esperar al intercambio de buffers -> consola PARADA, no coste.
     *          Si 'esp' es grande, sobra tiempo de GPU y el problema esta
     *          entero en la CPU; si crece al cambiar de backend grafico, es que
     *          ahora se espera mas al barrido, no que se trabaje mas.
     */
    char line_ms[96];
    {
        std::size_t n = 0;
        const char* label = "ms ";
        for (std::size_t i = 0; label[i] != '\0'; i++) {
            line_ms[n++] = label[i];
        }
        AppendMillis(line_ms, n, stats_frame_ms);
        const auto append = [&](const char* text, double value) {
            for (std::size_t i = 0; text[i] != '\0'; i++) {
                line_ms[n++] = text[i];
            }
            AppendMillis(line_ms, n, value);
        };
        append(" cpu ", stats_cpu_ms);
        append(" gx ", stats_gx_ms);
        append(" svc ", stats_svc_ms);
        // El reparto de cmdlist: 'vtx' es el bucle de vertices entero y 'lote'
        // nuestro camino de GPU. Lo que sobre de gx es decodificar comandos.
        append(" vtx ", stats_vertices_ms);
        append(" lote ", stats_batch_ms);
        // dsp = sonido emulado (0.1.7.7); no se suma: va dentro de cpu y svc.
        append(" dsp ", stats_dsp_ms);
        line_ms[n] = '\0';
    }

    char line_pres[128];
    {
        std::size_t n = 0;
        const char* label = "conv ";
        for (std::size_t i = 0; label[i] != '\0'; i++) {
            line_pres[n++] = label[i];
        }
        AppendMillis(line_pres, n, stats_convert_ms);
        const auto append = [&](const char* text, double value) {
            for (std::size_t i = 0; text[i] != '\0'; i++) {
                line_pres[n++] = text[i];
            }
            AppendMillis(line_pres, n, value);
        };
        append(" sub ", stats_upload_ms);
        append(" dib ", stats_draw_ms);
        // ovl = la parte de 'dib' que es este overlay (0.1.0.45).
        append(" ovl ", stats_overlay_ms);
        append(" esp ", stats_swapwait_ms);
        // par = ocupacion de los tres nucleos al sombrear (100 = los tres
        // trabajando todo el rato); ins = instrucciones de shader por vertice;
        // rap = % de ellas por la ruta rapida (0.1.0.45).
        const auto append_whole = [&](const char* text, double value) {
            for (std::size_t i = 0; text[i] != '\0'; i++) {
                line_pres[n++] = text[i];
            }
            n += AppendUInt(line_pres + n, static_cast<unsigned int>(value + 0.5));
        };
        append_whole(" par ", stats_shade_occupancy);
        append_whole(" ins ", stats_instrs_per_vertex);
        append_whole(" rap ", stats_fast_percent);
        // snd = sonido con estirado (est) o directo (dir). SELECT + IZQUIERDA.
        const char* snd_label = audio_stretching ? " snd est" : " snd dir";
        for (std::size_t i = 0; snd_label[i] != '\0'; i++) {
            line_pres[n++] = snd_label[i];
        }
        line_pres[n] = '\0';
    }

    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 470, 0xFF60FFC0, 0.8f, line_ms);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 488, 0xFF60FFC0, 0.8f, line_pres);

    // Con que se ha presentado el fotograma: verde cuando lo ha dibujado el
    // chip con GXM, gris cuando es el camino de vita2d. Sin esto, "¿y esto va
    // por el backend nuevo?" solo se responde mirando la fecha del VPK.
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 506,
                         presenter_uses_gxm ? 0xFF60FFC0 : 0xFFA0A0A0, 0.8f, presenter_line);

    /**
     * sh  = sombrear vertices (tiempo de pared, ya repartido en tres nucleos)
     * fin = parado esperando a la GPU al cerrar escenas (sceGxmFinish)
     * tx  = texturas decodificadas en el intervalo y los ms por fotograma
     * ll / lu = escenas cerradas por buffer de vertices lleno / tablas de luz
     *
     * Es la linea que dice si 0.1.0.42 ha hecho lo que se esperaba: 'sh'
     * deberia quedar en torno a un tercio de lo que era 'vtx', y ll/lu en cero.
     */
    char line_sh[96];
    {
        std::size_t n = 0;
        const char* label = "sh ";
        for (std::size_t i = 0; label[i] != '\0'; i++) {
            line_sh[n++] = label[i];
        }
        AppendMillis(line_sh, n, stats_shade_ms);
        const auto append_text = [&](const char* text) {
            for (std::size_t i = 0; text[i] != '\0'; i++) {
                line_sh[n++] = text[i];
            }
        };
        append_text(" fin ");
        AppendMillis(line_sh, n, stats_finish_ms);
        append_text(" tx ");
        n += AppendUInt(line_sh + n, stats_texdecodes);
        line_sh[n++] = '/';
        AppendMillis(line_sh, n, stats_texdecode_ms);
        // r = reutilizadas por hash, c = cambiadas, e = expulsadas por sitio.
        append_text(" r");
        n += AppendUInt(line_sh + n, stats_texreuses);
        append_text(" c");
        n += AppendUInt(line_sh + n, stats_texchanged);
        append_text(" e");
        n += AppendUInt(line_sh + n, stats_texevictions);
        append_text(" ll ");
        n += AppendUInt(line_sh + n, stats_close_full);
        append_text(" lu ");
        n += AppendUInt(line_sh + n, stats_close_lut);
        // vs = programas en la ruta rapida / comprobados en el intervalo /
        // DIFERENCIAS (tiene que ser 0; si no, crash.txt dice donde).
        append_text(" vs ");
        n += AppendUInt(line_sh + n, stats_fast_programs);
        line_sh[n++] = '/';
        n += AppendUInt(line_sh + n, stats_fast_checks);
        line_sh[n++] = '/';
        n += AppendUInt(line_sh + n, stats_fast_mismatches);
        line_sh[n] = '\0';
    }
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 524, 0xFF60FFC0, 0.8f, line_sh);
}

} // namespace VitaFrontend
