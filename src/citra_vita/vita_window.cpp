// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#include <psp2/ctrl.h>
#include <psp2/touch.h>
#include "citra_vita/vita_version.h"
#include "citra_vita/vita_window.h"
#include "common/settings.h"
#include "core/3ds.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "common/vita_diag.h"
#include "core/core.h"
#include "video_core/gpu.h"
#include "video_core/renderer_gxm/gxm_pica_format.h"
#include "video_core/renderer_gxm/gxm_presenter.h"
#include "video_core/renderer_gxm/renderer_gxm.h"
#include "video_core/renderer_software/renderer_software.h"
#include "video_core/renderer_software/sw_rasterizer.h"

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
    BuildLayout();
}

EmuWindow_Vita::~EmuWindow_Vita() {
    if (top_texture != nullptr) {
        vita2d_free_texture(top_texture);
    }
    if (bottom_texture != nullptr) {
        vita2d_free_texture(bottom_texture);
    }
}

void EmuWindow_Vita::BuildLayout() {
    Layout::FramebufferLayout layout{};
    layout.width = kVitaScreenWidth;
    layout.height = kVitaScreenHeight;
    layout.top_screen_enabled = true;
    layout.bottom_screen_enabled = true;
    layout.is_rotated = true;
    layout.top_screen = Common::Rectangle<u32>{kTopLeft, kTopTop, kTopLeft + kTopWidth,
                                               kTopTop + kTopHeight};
    layout.bottom_screen = Common::Rectangle<u32>{kBottomLeft, kBottomTop,
                                                  kBottomLeft + kBottomWidth,
                                                  kBottomTop + kBottomHeight};

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

    // L+R a la vez (sin SELECT) enciende y apaga la media resolucion vertical.
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
        }
        return;
    }
    halfres_combo_held = false;

    // SELECT + ARRIBA recorre los modos de ablacion, que saltan partes del
    // bucle de pixeles para medir su coste con el FPS de la escena real. Es un
    // modo de diagnostico que rompe la imagen a proposito.
    const bool ablate = select && ((buttons & SCE_CTRL_UP) != 0);
    if (ablate) {
        if (!ablation_combo_held) {
            ablation_combo_held = true;
            u32 value = SwRenderer::Ablation::mode.load(std::memory_order_relaxed);
            value = (value >= SwRenderer::Ablation::kMax) ? 0 : value + 1;
            SwRenderer::Ablation::mode.store(value, std::memory_order_relaxed);
        }
        return;
    }
    ablation_combo_held = false;

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

void EmuWindow_Vita::UploadScreen(vita2d_texture* texture, const std::vector<u8>& pixels,
                                  u32 src_width, u32 src_height) {
    if (texture == nullptr || pixels.empty()) {
        return;
    }

    u8* dest = static_cast<u8*>(vita2d_texture_get_datap(texture));
    const u32 dest_stride = vita2d_texture_get_stride(texture);
    const u32 src_stride = src_width * 4;

    if (pixels.size() < static_cast<std::size_t>(src_stride) * src_height) {
        return;
    }

    // ScreenInfo guarda RGBA byte a byte, igual que SCE_GXM_TEXTURE_FORMAT_A8B8G8R8
    // en little-endian, asi que no hace falta convertir pixel a pixel: basta
    // copiar fila a fila respetando el stride de la textura.
    if (dest_stride == src_stride) {
        std::memcpy(dest, pixels.data(), static_cast<std::size_t>(src_stride) * src_height);
        return;
    }
    for (u32 y = 0; y < src_height; y++) {
        std::memcpy(dest + static_cast<std::size_t>(y) * dest_stride,
                    pixels.data() + static_cast<std::size_t>(y) * src_stride, src_stride);
    }
}

// La tabla de formatos GXM y el numero de bytes por pixel de cada formato del
// 3DS viven en video_core/renderer_gxm/gxm_pica_format.h: los necesitan tanto
// el camino de vita2d de este fichero como el presentador GXM, y tenerlos
// duplicados fue exactamente el origen de un ciclo de prueba en consola
// (rojo y azul intercambiados). Ver alli el porque de cada sufijo _RGB/_RGBA.

void EmuWindow_Vita::UploadScreenNative(vita2d_texture*& texture, const ScreenInfoRef& info) {
    if (!info.valid || info.pixels.empty()) {
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
        // Caso normal: una sola copia de bloque, sin tocar un solo pixel.
        std::memcpy(dest, info.pixels.data(), static_cast<std::size_t>(src_stride) * info.height);
        return;
    }
    for (u32 y = 0; y < info.height; y++) {
        std::memcpy(dest + static_cast<std::size_t>(y) * dest_stride,
                    info.pixels.data() + static_cast<std::size_t>(y) * src_stride, row_bytes);
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

    vita2d_draw_texture_rotate_hotspot(texture, center_x, center_y, -kHalfPi,
                                       static_cast<float>(info.width) * 0.5f,
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
        // Con GXM la copia la hace el presentador y se cronometra alli.
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
    vita2d_start_drawing();
    vita2d_clear_screen();
    if (use_gxm) {
        gxm_presenter->Draw(*top, *bottom, {kTopLeft, kTopTop, kTopWidth, kTopHeight},
                            {kBottomLeft, kBottomTop, kBottomWidth, kBottomHeight});
    } else {
        DrawRotatedScreen(top_texture, *top, kTopLeft, kTopTop, kTopWidth, kTopHeight);
        DrawRotatedScreen(bottom_texture, *bottom, kBottomLeft, kBottomTop, kBottomWidth,
                          kBottomHeight);
    }

    // Estado para el overlay. El presentador puede apagarse a mitad de partida
    // (sin memoria de GPU), asi que la linea se rehace en cada fotograma.
    presenter_uses_gxm = use_gxm;
    BuildLine(presenter_line, sizeof(presenter_line), "pres ",
              gxm_api ? gxm_presenter->Status() : "vita2d (software)");

    DrawStatsOverlay();
    vita2d_end_drawing();
    Common::FrameStats::Add(Common::FrameStats::draw_us, draw_begin);

    // vita2d_swap_buffers espera a que la GPU acabe y al barrido de pantalla.
    // Casi todo lo que se mida aqui es la consola PARADA, no trabajo: por eso va
    // en su propio contador y el overlay lo pinta aparte.
    const unsigned long long swap_begin = Common::VitaMicros();
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
        stats_game_fps = stats.game_fps;
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
            stats_swapwait_ms = per_frame_ms(Common::FrameStats::swap_us);
            Common::FrameStats::Reset();
        }

        // Los contadores del rasterizador de la GPU y los del software se
        // vacian aqui, una vez por intervalo, para que la linea diga la
        // proporcion del ultimo segundo.
        stats_tri_gpu = Gxm::RasterizerGXM::gpu_triangles.exchange(0, std::memory_order_relaxed);
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

        stats_next_update_us = now_us + 1'000'000;
    }

    char line1[16] = "FPS ";
    std::size_t n1 = 4;
    AppendOneDecimal(line1, n1, stats_game_fps);
    line1[n1] = '\0';

    char line2[16] = "vel ";
    std::size_t n2 = 4;
    AppendOneDecimal(line2, n2, stats_speed_percent);
    line2[n2++] = '%';
    line2[n2] = '\0';

    // Cuentas de vaciado de la cache de traduccion: para decidir si merece la
    // pena agrandar TRANS_CACHE_SIZE. "cap" es la unica que tiene que ver con
    // el tamano del buffer; "inv" (cambios de proceso, principalmente) pasaria
    // igual con un buffer infinito. Son enteros pequenos en la practica, pero
    // se acotan a 9999 por si acaso: el buffer es de sobra para eso, no para un
    // desbordamiento real.
    std::size_t flushes_capacity = 0;
    std::size_t flushes_invalidation = 0;
    GetTransCacheFlushCounts(flushes_capacity, flushes_invalidation);

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
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 20, 0xFF60C0FF, 0.8f, kOverlayBuild);
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
    char line_hw[64] = "cpu ";
    std::size_t nh = 4;
    nh += AppendUInt(line_hw + nh, static_cast<unsigned int>(Common::g_arm_clock_mhz));
    const char* hw_label = "MHz  ns/px ";
    // ...y al final, si la FPU llego a ver denormales antes de activar FZ.
    for (std::size_t i = 0; hw_label[i] != '\0'; i++) {
        line_hw[nh++] = hw_label[i];
    }
    nh += AppendUInt(line_hw + nh,
                     static_cast<unsigned int>(std::min(stats_ns_per_pixel, 999999.0)));
    // 'denorm si' = la FPU vio denormales ANTES de que se activara flush-to-zero.
    // Es la confirmacion de la hipotesis: con FZ apagado, cada uno de esos se
    // sale a codigo de soporte y cuesta cientos de ciclos.
    const char* denorm_label = Common::VitaSawDenormals() ? "  denorm SI" : "  denorm no";
    for (std::size_t i = 0; denorm_label[i] != '\0'; i++) {
        line_hw[nh++] = denorm_label[i];
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

    // Marcador de media resolucion. Verde cuando esta apagada, naranja cuando
    // esta encendida: es una perdida de calidad y conviene que se note que el
    // modo esta puesto, no que pase desapercibido.
    const bool half_on = SwRenderer::FrameSkip::half_resolution.load(std::memory_order_relaxed);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 356,
                         half_on ? 0xFF40C0FF : 0xFF80FF80, 0.8f,
                         half_on ? "media res SI  L+R" : "media res no  L+R");

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
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 398, 0xFFFFA0A0, 0.8f,
                         stats_format_line);

    // Desglose del rasterizador por triangulo:
    //   hil = % de triangulos repartidos entre los tres hilos
    //   set = % del tiempo en preparar el triangulo
    //   esp = % esperando a los hilos (raster en paralelo)
    //   un  = % rasterizando en el propio hilo (triangulos pequenos)
    // Y debajo, el modo de ablacion vigente (SELECT+ARRIBA lo cambia).
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
    char line_occ[64];
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
    char line_ms[64];
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
        line_ms[n] = '\0';
    }

    char line_pres[64];
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
        append(" esp ", stats_swapwait_ms);
        line_pres[n] = '\0';
    }

    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 470, 0xFF60FFC0, 0.8f, line_ms);
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 488, 0xFF60FFC0, 0.8f, line_pres);

    // Con que se ha presentado el fotograma: verde cuando lo ha dibujado el
    // chip con GXM, gris cuando es el camino de vita2d. Sin esto, "¿y esto va
    // por el backend nuevo?" solo se responde mirando la fecha del VPK.
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 506,
                         presenter_uses_gxm ? 0xFF60FFC0 : 0xFFA0A0A0, 0.8f, presenter_line);

    const u32 ablation_mode = SwRenderer::Ablation::mode.load(std::memory_order_relaxed);
    char line_abl[40] = "abl ";
    std::size_t nab = 4;
    nab += AppendUInt(line_abl + nab, ablation_mode);
    line_abl[nab++] = ' ';
    const char* ablation_name = SwRenderer::Ablation::Name(ablation_mode);
    for (std::size_t i = 0; ablation_name[i] != '\0' && nab < sizeof(line_abl) - 1; i++) {
        line_abl[nab++] = ablation_name[i];
    }
    line_abl[nab] = '\0';
    vita2d_pgf_draw_text(stats_font, static_cast<int>(kOverlayX), 434,
                         ablation_mode == SwRenderer::Ablation::kNormal ? 0xFF80D0FF : 0xFF40C0FF,
                         0.8f, line_abl);
}

} // namespace VitaFrontend
