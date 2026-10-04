// Copyright 2015-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <bit>
#include <cstring>
#if defined(__ARM_NEON) && defined(__PSVITA__)
#include <arm_neon.h>
#endif
#include <boost/container/static_vector.hpp>
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/quaternion.h"
#include "common/vector_math.h"
#include "core/memory.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_software/sw_framebuffer.h"
#include "video_core/renderer_software/sw_lighting.h"
#include "video_core/renderer_software/sw_proctex.h"
#include "video_core/renderer_software/sw_rasterizer.h"
#include "video_core/renderer_software/sw_texturing.h"
#include "video_core/texture/texture_decode.h"

#ifdef __PSVITA__
#include <psp2/kernel/processmgr.h>
#endif

namespace SwRenderer {

using Pica::f24;
using Pica::FramebufferRegs;
using Pica::RasterizerRegs;
using Pica::TexturingRegs;
using Pica::Texture::LookupTexture;
using Pica::Texture::TextureInfo;

// Certain games render 2D elements very close to clip plane 0 resulting in very tiny
// negative/positive z values when computing with f32 precision,
// causing some vertices to get erroneously clipped. To workaround this problem,
// we can use a very small epsilon value for clip plane comparison.
constexpr f32 EPSILON_Z = 0.00000001f;

/**
 * Pasa una componente de color de f24 (0..1) a byte (0..255).
 *
 * Antes esto era round(valor * 255), y ese round() es el de la libreria de C:
 * DOBLE precision y fuera de linea. Comprobado en los simbolos del objeto, que
 * dejaba una referencia externa a 'round'. Son cuatro llamadas por pixel, y el
 * dano no es solo el coste de la rutina: una llamada que el compilador no ve le
 * obliga a volcar a la pila los registros en vuelo del bucle mas caliente del
 * emulador, cuatro veces por pixel.
 *
 * Con valores dentro de 0..1 el resultado es identico: para x positivo,
 * truncar x + 0,5 es lo mismo que redondear. Y ademas se acota, que el original
 * no lo hacia: un color fuera de rango daba un cast a u8 con valor fuera de
 * rango, que es comportamiento indefinido.
 */
/**
 * Marca que etapas TEV hacen algo de verdad. Un bit por etapa, 1 = activa.
 *
 * Una etapa es inerte cuando copia el color anterior sin tocarlo, que es como
 * los juegos dejan configuradas las etapas que no usan. La condicion tiene que
 * cumplirse ENTERA, tanto en color como en alfa:
 *
 *   operacion = Replace      (la salida es el primer operando, sin combinar)
 *   fuente 1  = Previous     (ese operando es la salida de la etapa anterior)
 *   modificador = identidad  (ni invertir ni tomar el alfa como color)
 *   multiplicador = 1        (sin escalado)
 *
 * Si todo eso se cumple, la etapa devuelve exactamente lo que recibio y se
 * puede saltar su calculo sin cambiar un solo pixel.
 *
 * La etapa 0 se marca siempre activa: ahi 'Previous' no se refiere a la salida
 * anterior -- no la hay -- sino a color_source3, asi que la equivalencia no
 * vale. Ademas es la que casi siempre hace el trabajo de verdad.
 *
 * Se calcula una vez por triangulo: los registros TEV no cambian dentro de uno.
 */
u32 ComputeActiveTevStages(
    std::span<const Pica::TexturingRegs::TevStageConfig, 6> tev_stages) {
    using Config = Pica::TexturingRegs::TevStageConfig;
    u32 mask = 1u; // la etapa 0 siempre cuenta

    for (u32 i = 1; i < tev_stages.size(); i++) {
        const auto& s = tev_stages[i];
        const bool color_passthrough = s.color_op == Config::Operation::Replace &&
                                       s.color_source1 == Config::Source::Previous &&
                                       s.color_modifier1 == Config::ColorModifier::SourceColor &&
                                       s.GetColorMultiplier() == 1;
        const bool alpha_passthrough = s.alpha_op == Config::Operation::Replace &&
                                       s.alpha_source1 == Config::Source::Previous &&
                                       s.alpha_modifier1 == Config::AlphaModifier::SourceAlpha &&
                                       s.GetAlphaMultiplier() == 1;
        if (!(color_passthrough && alpha_passthrough)) {
            mask |= 1u << i;
        }
    }
    return mask;
}

inline u8 RoundColorComponent(f24 value) {
    const f32 scaled = value.ToFloat32() * 255.0f + 0.5f;
    if (!(scaled > 0.0f)) { // con NaN se va a 0, igual que el cast de antes
        return 0;
    }
    if (scaled >= 255.0f) {
        return 255;
    }
    return static_cast<u8>(scaled);
}

/**
 * Multiplica una coordenada de textura por la dimension de la textura,
 * reproduciendo EXACTAMENTE la semantica de f24::operator* pero sin vmrs.
 *
 * f24::operator* hace, ademas de la multiplicacion:
 *
 *     if (isnan(result) && !isnan(a) && !isnan(b)) result = 0;   // inf*0
 *
 * y en ARM esa comprobacion se traduce en leer las banderas de la FPU con un
 * 'vmrs': una instruccion de latencia alta que ademas SERIALIZA la tuberia de
 * coma flotante (espera a que terminen todas las operaciones en vuelo). El
 * bucle de pixeles hace esto hasta seis veces por pixel.
 *
 * Aqui la deteccion de NaN se hace por bits, que es exactamente lo mismo para
 * IEEE-754 y no toca las banderas: en el camino normal son tres operaciones de
 * enteros y una rama que nunca se toma. El caso raro solo se paga cuando pasa.
 */
inline f32 ScaleTexCoord(f24 coord, f24 size) {
    const f32 c = coord.ToFloat32();
    const f32 s = size.ToFloat32();
    const f32 product = c * s;

    constexpr u32 kExponentMask = 0x7F800000u;
    constexpr u32 kMantissaMask = 0x007FFFFFu;
    const u32 bits = std::bit_cast<u32>(product);
    if ((bits & kExponentMask) == kExponentMask && (bits & kMantissaMask) != 0) {
        const u32 c_bits = std::bit_cast<u32>(c);
        const u32 s_bits = std::bit_cast<u32>(s);
        const bool c_nan = (c_bits & kExponentMask) == kExponentMask && (c_bits & kMantissaMask) != 0;
        const bool s_nan = (s_bits & kExponentMask) == kExponentMask && (s_bits & kMantissaMask) != 0;
        if (!c_nan && !s_nan) {
            return 0.0f;
        }
    }
    return product;
}

struct Vertex : Pica::OutputVertex {
    Vertex(const OutputVertex& v) : OutputVertex(v) {}

    /// Attributes used to store intermediate results position after perspective divide.
    Common::Vec3<f24> screenpos;

    /**
     * Linear interpolation
     * factor: 0=this, 1=vtx
     * Note: This function cannot be called after perspective divide.
     **/
    void Lerp(f24 factor, const Vertex& vtx) {
        pos = pos * factor + vtx.pos * (f24::One() - factor);
        quat = quat * factor + vtx.quat * (f24::One() - factor);
        color = color * factor + vtx.color * (f24::One() - factor);
        tc0 = tc0 * factor + vtx.tc0 * (f24::One() - factor);
        tc1 = tc1 * factor + vtx.tc1 * (f24::One() - factor);
        tc0_w = tc0_w * factor + vtx.tc0_w * (f24::One() - factor);
        view = view * factor + vtx.view * (f24::One() - factor);
        tc2 = tc2 * factor + vtx.tc2 * (f24::One() - factor);
    }

    /**
     * Linear interpolation
     * factor: 0=v0, 1=v1
     * Note: This function cannot be called after perspective divide.
     **/
    static Vertex Lerp(f24 factor, const Vertex& v0, const Vertex& v1) {
        Vertex ret = v0;
        ret.Lerp(factor, v1);
        return ret;
    }
};

namespace {

MICROPROFILE_DEFINE(GPU_Rasterization, "GPU", "Rasterization", MP_RGB(50, 50, 240));

struct ClippingEdge {
public:
    constexpr ClippingEdge(Common::Vec4<f24> coeffs,
                           Common::Vec4<f24> bias = Common::Vec4<f24>(f24::Zero(), f24::Zero(),
                                                                      f24::Zero(), f24::Zero()))
        : pos(f24::Zero()), coeffs(coeffs), bias(bias) {}

    bool IsInside(const Vertex& vertex) const {
        return Common::Dot(vertex.pos + bias, coeffs) >= f24::FromFloat32(-EPSILON_Z);
    }

    bool IsOutSide(const Vertex& vertex) const {
        return !IsInside(vertex);
    }

    Vertex GetIntersection(const Vertex& v0, const Vertex& v1) const {
        const f24 dp = Common::Dot(v0.pos + bias, coeffs);
        const f24 dp_prev = Common::Dot(v1.pos + bias, coeffs);
        const f24 factor = dp_prev / (dp_prev - dp);
        return Vertex::Lerp(factor, v0, v1);
    }

private:
    [[maybe_unused]] f24 pos;
    Common::Vec4<f24> coeffs;
    Common::Vec4<f24> bias;
};

} // Anonymous namespace

namespace {
/// En cuantas BANDAS se parte un triangulo grande, que es lo mismo que cuantos
/// nucleos lo rasterizan a la vez. El pool tiene una menos: la banda que falta
/// la hace el propio hilo que llama.
std::size_t GetRasterizerThreadCount() {
#ifdef __PSVITA__
    // Fijo a 3, que son los nucleos de usuario de la Vita (el cuarto lo reserva
    // el sistema). No se usa hardware_concurrency() porque en esta libc no hay
    // garantia de que devuelva algo util: si devolviera 0 quedarian 2 hilos
    // -- desaprovechando un nucleo entero -- y si devolviera 4 quedarian cuatro
    // hilos peleandose por tres nucleos, que para trabajo de calculo puro es
    // peor que tres (mas cambios de contexto sin ganar nada).
    return 3;
#else
    return std::max(std::thread::hardware_concurrency(), 2U);
#endif
}
} // Anonymous namespace

RasterizerSoftware::RasterizerSoftware(Memory::MemorySystem& memory_, Pica::PicaCore& pica_)
    : memory{memory_}, pica{pica_}, regs{pica.regs.internal},
      num_sw_threads{GetRasterizerThreadCount()},
      /**
       * UN HILO MENOS que bandas, y empezando por el nucleo 1.
       *
       * El hilo de emulacion esta atado al nucleo 0 (ver main.cpp) y, mientras
       * se rasterizaba un triangulo grande, se quedaba dormido en la barrera:
       * un nucleo entero parado esperando, y ademas dos viajes al planificador
       * (dormirlo y despertarlo) en el camino critico de cada triangulo, unas
       * mil veces por fotograma.
       *
       * Ahora ese hilo rasteriza la banda 0 el mismo y solo espera por las
       * otras dos. El reparto sigue siendo de tres bandas sobre tres nucleos,
       * pero con dos encolados en vez de tres y sin dormir a nadie en el caso
       * normal (ver WaitForRequestsSpin).
       *
       * De ahi el desplazamiento de nucleo: los hilos del pool van a los
       * nucleos 1 y 2. Si empezaran en el 0 compartirian nucleo con quien les
       * da trabajo, que es justo lo que se quiere evitar.
       *
       * El tercer argumento fabrica la cache de bloques ETC1 de cada hilo: el
       * pool la construye DENTRO del hilo y se la pasa como estado en cada
       * tarea. Ver el comentario de sw_workers en la cabecera.
       */
      sw_workers{num_sw_threads - 1, "SwRenderer workers",
                 [](std::size_t) { return Pica::Texture::Etc1BlockCache{}; }, true, 1},
      fb{memory, regs.framebuffer} {
    // Divisor de la ocupacion del overlay: ver RasterizerStats::band_busy_us.
    RasterizerStats::raster_bands.store(static_cast<u32>(num_sw_threads),
                                        std::memory_order_relaxed);
}

void RasterizerSoftware::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                     const Pica::OutputVertex& v2) {
    // Fotograma saltado: no se dibuja nada. Se corta aqui, en la entrada, para
    // ahorrar tambien el recorte contra los seis planos y el calculo de
    // coordenadas de pantalla, no solo el relleno de pixeles. Ver FrameSkip.
    if (!FrameSkip::ShouldRender()) {
        return;
    }

    /**
     * Clipping a planar n-gon against a plane will remove at least 1 vertex and introduces 2 at
     * the new edge (or less in degenerate cases). As such, we can say that each clipping plane
     * introduces at most 1 new vertex to the polygon. Since we start with a triangle and have a
     * fixed 6 clipping planes, the maximum number of vertices of the clipped polygon is 3 + 6 = 9.
     **/
    static constexpr std::size_t MAX_VERTICES = 9;

    boost::container::static_vector<Vertex, MAX_VERTICES> buffer_a = {v0, v1, v2};
    boost::container::static_vector<Vertex, MAX_VERTICES> buffer_b;

    FlipQuaternionIfOpposite(buffer_a[1].quat, buffer_a[0].quat);
    FlipQuaternionIfOpposite(buffer_a[2].quat, buffer_a[0].quat);

    auto* output_list = &buffer_a;
    auto* input_list = &buffer_b;

    // NOTE: We clip against a w=epsilon plane to guarantee that the output has a positive w value.
    // TODO: Not sure if this is a valid approach. Also should probably instead use the smallest
    //       epsilon possible within f24 accuracy.
    static constexpr f24 EPSILON = f24::FromFloat32(0.00001f);
    static constexpr f24 f0 = f24::Zero();
    static constexpr f24 f1 = f24::One();
    static constexpr std::array<ClippingEdge, 7> clipping_edges = {{
        {Common::MakeVec(-f1, f0, f0, f1)},                                        // x = +w
        {Common::MakeVec(f1, f0, f0, f1)},                                         // x = -w
        {Common::MakeVec(f0, -f1, f0, f1)},                                        // y = +w
        {Common::MakeVec(f0, f1, f0, f1)},                                         // y = -w
        {Common::MakeVec(f0, f0, -f1, f0)},                                        // z =  0
        {Common::MakeVec(f0, f0, f1, f1)},                                         // z = -w
        {Common::MakeVec(f0, f0, f0, f1), Common::Vec4<f24>(f0, f0, f0, EPSILON)}, // w = EPSILON
    }};

    // Simple implementation of the Sutherland-Hodgman clipping algorithm.
    // TODO: Make this less inefficient (currently lots of useless buffering overhead happens here)
    const auto clip = [&](const ClippingEdge& edge) {
        std::swap(input_list, output_list);
        output_list->clear();

        const Vertex* reference_vertex = &input_list->back();
        for (const auto& vertex : *input_list) {
            // NOTE: This algorithm changes vertex order in some cases!
            if (edge.IsInside(vertex)) {
                if (edge.IsOutSide(*reference_vertex)) {
                    output_list->push_back(edge.GetIntersection(vertex, *reference_vertex));
                }
                output_list->push_back(vertex);
            } else if (edge.IsInside(*reference_vertex)) {
                output_list->push_back(edge.GetIntersection(vertex, *reference_vertex));
            }
            reference_vertex = &vertex;
        }
    };

    for (const ClippingEdge& edge : clipping_edges) {
        clip(edge);
        if (output_list->size() < 3) {
            return;
        }
    }

    if (regs.rasterizer.clip_enable) {
        const ClippingEdge custom_edge{regs.rasterizer.GetClipCoef()};
        clip(custom_edge);
        if (output_list->size() < 3) {
            return;
        }
    }

    MakeScreenCoords((*output_list)[0]);
    MakeScreenCoords((*output_list)[1]);

    for (std::size_t i = 0; i < output_list->size() - 2; i++) {
        Vertex& vtx0 = (*output_list)[0];
        Vertex& vtx1 = (*output_list)[i + 1];
        Vertex& vtx2 = (*output_list)[i + 2];

        MakeScreenCoords(vtx2);

        LOG_TRACE(
            Render_Software,
            "Triangle {}/{} at position ({:.3}, {:.3}, {:.3}, {:.3f}), "
            "({:.3}, {:.3}, {:.3}, {:.3}), ({:.3}, {:.3}, {:.3}, {:.3}) and "
            "screen position ({:.2}, {:.2}, {:.2}), ({:.2}, {:.2}, {:.2}), ({:.2}, {:.2}, {:.2})",
            i + 1, output_list->size() - 2, vtx0.pos.x.ToFloat32(), vtx0.pos.y.ToFloat32(),
            vtx0.pos.z.ToFloat32(), vtx0.pos.w.ToFloat32(), vtx1.pos.x.ToFloat32(),
            vtx1.pos.y.ToFloat32(), vtx1.pos.z.ToFloat32(), vtx1.pos.w.ToFloat32(),
            vtx2.pos.x.ToFloat32(), vtx2.pos.y.ToFloat32(), vtx2.pos.z.ToFloat32(),
            vtx2.pos.w.ToFloat32(), vtx0.screenpos.x.ToFloat32(), vtx0.screenpos.y.ToFloat32(),
            vtx0.screenpos.z.ToFloat32(), vtx1.screenpos.x.ToFloat32(),
            vtx1.screenpos.y.ToFloat32(), vtx1.screenpos.z.ToFloat32(),
            vtx2.screenpos.x.ToFloat32(), vtx2.screenpos.y.ToFloat32(),
            vtx2.screenpos.z.ToFloat32());

        ProcessTriangle(vtx0, vtx1, vtx2);
    }
}

void RasterizerSoftware::MakeScreenCoords(Vertex& vtx) {
    Viewport viewport{};
    viewport.halfsize_x = f24::FromRaw(regs.rasterizer.viewport_size_x);
    viewport.halfsize_y = f24::FromRaw(regs.rasterizer.viewport_size_y);
    viewport.offset_x = f24::FromFloat32(static_cast<f32>(regs.rasterizer.viewport_corner.x));
    viewport.offset_y = f24::FromFloat32(static_cast<f32>(regs.rasterizer.viewport_corner.y));

    f24 inv_w = f24::One() / vtx.pos.w;
    vtx.pos.w = inv_w;
    vtx.quat *= inv_w;
    vtx.color *= inv_w;
    vtx.tc0 *= inv_w;
    vtx.tc1 *= inv_w;
    vtx.tc0_w *= inv_w;
    vtx.view *= inv_w;
    vtx.tc2 *= inv_w;

    vtx.screenpos[0] = (vtx.pos.x * inv_w + f24::One()) * viewport.halfsize_x + viewport.offset_x;
    vtx.screenpos[1] = (vtx.pos.y * inv_w + f24::One()) * viewport.halfsize_y + viewport.offset_y;
    vtx.screenpos[2] = vtx.pos.z * inv_w;
}

void RasterizerSoftware::ProcessTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                                         bool reversed) {
    MICROPROFILE_SCOPE(GPU_Rasterization);

#ifdef __PSVITA__
    // Instrumentacion del reparto por triangulo: ver tri_setup_us/tri_wait_us/
    // tri_single_us en la cabecera. Son tres lecturas de reloj por triangulo
    // (~1000 por fotograma), muy por debajo del ruido.
    const SceUInt64 profile_t0 = sceKernelGetProcessTimeWide();
#endif

    // Modo de ablacion vigente (ver Ablation). Una lectura por triangulo.
    const u32 ablation_mode = Ablation::mode.load(std::memory_order_relaxed);

    // Vertex positions in rasterizer coordinates
    static auto screen_to_rasterizer_coords = [](const Common::Vec3<f24>& vec) {
        return Common::Vec3{Fix12P4::FromFloat24(vec.x), Fix12P4::FromFloat24(vec.y),
                            Fix12P4::FromFloat24(vec.z)};
    };

    const std::array<Common::Vec3<Fix12P4>, 3> vtxpos = {
        screen_to_rasterizer_coords(v0.screenpos),
        screen_to_rasterizer_coords(v1.screenpos),
        screen_to_rasterizer_coords(v2.screenpos),
    };

    if (regs.rasterizer.cull_mode == RasterizerRegs::CullMode::KeepAll ||
        regs.rasterizer.cull_mode == RasterizerRegs::CullMode::KeepAll2) {
        // Make sure we always end up with a triangle wound counter-clockwise
        if (!reversed && SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), vtxpos[2].xy()) <= 0) {
            ProcessTriangle(v0, v2, v1, true);
            return;
        }
    } else {
        if (!reversed && regs.rasterizer.cull_mode == RasterizerRegs::CullMode::KeepClockWise) {
            // Reverse vertex order and use the CCW code path.
            ProcessTriangle(v0, v2, v1, true);
            return;
        }
        // Cull away triangles which are wound clockwise.
        if (SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), vtxpos[2].xy()) <= 0) {
            return;
        }
    }

    u16 min_x = std::min({vtxpos[0].x, vtxpos[1].x, vtxpos[2].x});
    u16 min_y = std::min({vtxpos[0].y, vtxpos[1].y, vtxpos[2].y});
    u16 max_x = std::max({vtxpos[0].x, vtxpos[1].x, vtxpos[2].x});
    u16 max_y = std::max({vtxpos[0].y, vtxpos[1].y, vtxpos[2].y});

    // Convert the scissor box coordinates to 12.4 fixed point
    const u16 scissor_x1 = static_cast<u16>(regs.rasterizer.scissor_test.x1 << 4);
    const u16 scissor_y1 = static_cast<u16>(regs.rasterizer.scissor_test.y1 << 4);
    // x2,y2 have +1 added to cover the entire sub-pixel area
    const u16 scissor_x2 = static_cast<u16>((regs.rasterizer.scissor_test.x2 + 1) << 4);
    const u16 scissor_y2 = static_cast<u16>((regs.rasterizer.scissor_test.y2 + 1) << 4);

    if (regs.rasterizer.scissor_test.mode == RasterizerRegs::ScissorMode::Include) {
        // Calculate the new bounds
        min_x = std::max(min_x, scissor_x1);
        min_y = std::max(min_y, scissor_y1);
        max_x = std::min(max_x, scissor_x2);
        max_y = std::min(max_y, scissor_y2);
    }

    min_x &= Fix12P4::IntMask();
    min_y &= Fix12P4::IntMask();
    max_x = ((max_x + Fix12P4::FracMask()) & Fix12P4::IntMask());
    max_y = ((max_y + Fix12P4::FracMask()) & Fix12P4::IntMask());

    const int bias0 =
        IsRightSideOrFlatBottomEdge(vtxpos[0].xy(), vtxpos[1].xy(), vtxpos[2].xy()) ? -1 : 0;
    const int bias1 =
        IsRightSideOrFlatBottomEdge(vtxpos[1].xy(), vtxpos[2].xy(), vtxpos[0].xy()) ? -1 : 0;
    const int bias2 =
        IsRightSideOrFlatBottomEdge(vtxpos[2].xy(), vtxpos[0].xy(), vtxpos[1].xy()) ? -1 : 0;

    const auto w_inverse = Common::MakeVec(v0.pos.w, v1.pos.w, v2.pos.w);

    // Pendiente en x de las tres funciones de borde.
    //
    // SignedArea(a, b, p) es lineal en p: avanzar un pixel en x suma siempre lo
    // mismo, sea cual sea la fila. Antes se llamaba tres veces POR PIXEL de la
    // caja envolvente -- tambien para los que se acaban descartando --, y cada
    // llamada es un salto fuera de linea a otra unidad de compilacion que hace
    // un producto vectorial de tres componentes para quedarse solo con la z.
    //
    // El overlay daba 'test' 2,1x: por cada pixel que se pinta se prueban dos,
    // asi que la mitad de esas llamadas era trabajo tirado. Sacando la pendiente
    // una vez por triangulo, el interior del bucle pasa de tres productos
    // vectoriales a tres sumas de enteros.
    //
    // La pendiente se obtiene restando dos evaluaciones de la propia SignedArea
    // en vez de reescribir la formula a mano: es aritmetica entera exacta y
    // lineal, asi que el resultado es identico pixel a pixel, y no hay ocasion
    // de equivocarse al desmontar el punto fijo 12.4.
    //
    // Resolucion 0.5x (0.1.5.1): se sombrea un pixel de cada DOS en x (y una
    // linea de cada dos en y, mas abajo), asi que la pendiente se toma a DOS
    // pixeles y la primera columna se alinea a una columna PAR absoluta, por
    // lo mismo que las filas: que todos los triangulos usen la misma rejilla y
    // no se vean costuras entre vecinos. Ver FrameSkip::half_resolution.
    const bool half_res = FrameSkip::half_resolution.load(std::memory_order_relaxed);
    const u16 x_pitch = static_cast<u16>(half_res ? 0x20 : 0x10);
    const u16 probe_x = static_cast<u16>(
        (half_res && (((min_x >> 4) & 1) != 0)) ? min_x + 0x18 : min_x + 8);
    const u16 probe_y = static_cast<u16>(min_y + 8);
    const u16 probe_x_next = static_cast<u16>(probe_x + x_pitch);
    const s32 dw0_dx = SignedArea(vtxpos[1].xy(), vtxpos[2].xy(), {probe_x_next, probe_y}) -
                       SignedArea(vtxpos[1].xy(), vtxpos[2].xy(), {probe_x, probe_y});
    const s32 dw1_dx = SignedArea(vtxpos[2].xy(), vtxpos[0].xy(), {probe_x_next, probe_y}) -
                       SignedArea(vtxpos[2].xy(), vtxpos[0].xy(), {probe_x, probe_y});
    const s32 dw2_dx = SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), {probe_x_next, probe_y}) -
                       SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), {probe_x, probe_y});

    // Rango de profundidad del viewport: son dos registros fijos durante todo el
    // triangulo, pero se leian y convertian de f24 a float en CADA pixel.
    const float depth_scale = f24::FromRaw(regs.rasterizer.viewport_depth_range).ToFloat32();
    const float depth_offset =
        f24::FromRaw(regs.rasterizer.viewport_depth_near_plane).ToFloat32();

    // Registros que no cambian dentro del triangulo, leidos una sola vez.
    //
    // 'regs' es una referencia, asi que el compilador no puede dar por hecho que
    // su contenido siga igual despues de cada llamada que no puede ver (sampleo
    // de texturas, TEV, escrituras al framebuffer): tiene que volver a cargar
    // cada campo desde memoria una y otra vez dentro del bucle mas caliente.
    const auto scissor_mode = regs.rasterizer.scissor_test.mode;
    const bool w_buffering =
        regs.rasterizer.depthmap_enable == Pica::RasterizerRegs::DepthBuffering::WBuffering;
    const auto frag_op_mode = regs.framebuffer.output_merger.fragment_operation_mode;
    const bool allow_color_write = regs.framebuffer.framebuffer.allow_color_write != 0;
    const bool lighting_enabled = !regs.lighting.disable;
    const u32 fb_height = regs.framebuffer.framebuffer.height;

    // Media resolucion vertical: se sombrea una linea de cada dos y se copia a
    // la de abajo. Ver FrameSkip::half_resolution.
    //
    // El arranque se alinea a una fila PAR absoluta para que las parejas
    // (linea pintada, linea copiada) encajen igual en todos los triangulos. Si
    // dependiera de min_y, dos triangulos vecinos partirian las parejas de
    // forma distinta y se veria una costura entre ellos.
    const u16 y_pitch = static_cast<u16>(half_res ? 0x20 : 0x10);
    const u16 y_origin =
        (half_res && (((min_y >> 4) & 1) != 0)) ? static_cast<u16>(min_y + 0x10) : min_y;

    /**
     * Escritura de un pixel sombreado. En 0.5x rellena su bloque de 2x2: el
     * mismo color a la derecha, abajo y en diagonal.
     *
     * Abajo, el limite es 'height' y no 'height - 1' porque el registro guarda
     * el alto real menos uno (ver DrawPixel). A la derecha, el ancho real del
     * framebuffer y, con la tijera en modo Include, su borde derecho: la
     * columna copiada no debe pintar fuera de la zona que el juego permite.
     * (La fila copiada no mira la tijera, igual que antes de 0.1.5.1.)
     */
    const u32 fb_width = regs.framebuffer.framebuffer.width;
    const u32 x_limit = scissor_mode == RasterizerRegs::ScissorMode::Include
                            ? std::min<u32>(fb_width, static_cast<u32>(scissor_x2 >> 4))
                            : fb_width;
    const auto draw_scaled = [&](u32 px, u32 py, const Common::Vec4<u8>& color) {
        fb.DrawPixel(px, py, color);
        if (!half_res) {
            return;
        }
        const bool right = px + 1 < x_limit;
        if (right) {
            fb.DrawPixel(px + 1, py, color);
        }
        if (py + 1 <= fb_height) {
            fb.DrawPixel(px, py + 1, color);
            if (right) {
                fb.DrawPixel(px + 1, py + 1, color);
            }
        }
    };

    /**
     * Cuando se puede descartar un pixel tapado ANTES de sombrearlo.
     *
     * El orden original prueba la profundidad al final: primero muestrea hasta
     * tres texturas, calcula la iluminacion por fragmento y pasa las seis etapas
     * TEV, y solo entonces mira si el pixel estaba tapado. Todo lo que se gasto
     * en un pixel que no se ve, tirado.
     *
     * Adelantar la comparacion es equivalente exacto solo con la galga apagada.
     * Con la galga activa, fallar la profundidad ejecuta action_depth_fail y
     * escribe, asi que saltarse el pixel cambiaria el resultado. En modo Shadow
     * ni siquiera se pasa por DoDepthStencilTest, de ahi la tercera condicion.
     *
     * La prueba de alfa no estorba: solo se salta el pixel cuando la profundidad
     * FALLA, y en ese caso ambos ordenes lo descartan sin tocar nada.
     */
    // Estado de mezcla del triangulo. Ver BlendState.
    BlendState blend;
    {
        const auto& output_merger = regs.framebuffer.output_merger;
        const auto& params = output_merger.alpha_blending;
        const bool rgb_standard =
            params.factor_source_rgb == FramebufferRegs::BlendFactor::SourceAlpha &&
            params.factor_dest_rgb == FramebufferRegs::BlendFactor::OneMinusSourceAlpha &&
            params.blend_equation_rgb == FramebufferRegs::BlendEquation::Add;
        const bool alpha_standard =
            params.factor_source_a == FramebufferRegs::BlendFactor::SourceAlpha &&
            params.factor_dest_a == FramebufferRegs::BlendFactor::OneMinusSourceAlpha &&
            params.blend_equation_a == FramebufferRegs::BlendEquation::Add;
        blend.standard_src_alpha =
            output_merger.alphablend_enable && rgb_standard && alpha_standard;
        blend.all_channels_enabled = output_merger.red_enable && output_merger.green_enable &&
                                     output_merger.blue_enable && output_merger.alpha_enable;
    }

    // Estado del output merger y de textura que tampoco cambia dentro del
    // triangulo. El bucle llamaba a DoAlphaTest/WriteFog/DoDepthStencilTest en
    // cada pixel aunque fuesen a salir por la primera rama; con estas banderas
    // ni se llama.
    const auto& output_merger_state = regs.framebuffer.output_merger;
    const bool fog_enabled = regs.texturing.fog_mode == TexturingRegs::FogMode::Fog;
    const bool alpha_test_enabled = output_merger_state.alpha_test.enable;
    // Con 'Always' la prueba de alfa no puede matar nada: adelantarla no
    // ahorraria trabajo, solo anadiria la pasada de alfa.
    const bool alpha_test_can_fail =
        alpha_test_enabled &&
        output_merger_state.alpha_test.func != FramebufferRegs::CompareFunc::Always;
    const bool stencil_action_enable =
        output_merger_state.stencil_test.enable &&
        regs.framebuffer.framebuffer.depth_format == FramebufferRegs::DepthFormat::D24S8;
    const bool depth_write_enabled =
        regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
        output_merger_state.depth_write_enable;
    // Sin prueba de profundidad, o con una prueba que siempre pasa, y sin galga
    // que actue ni escritura de profundidad, DoDepthStencilTest no escribe nada:
    // la lectura de profundidad no tiene efectos, asi que ni se llama.
    const bool depth_test_trivial =
        !output_merger_state.depth_test_enable ||
        output_merger_state.depth_test_func == FramebufferRegs::CompareFunc::Always;
    const bool depth_stencil_noop =
        depth_test_trivial && !stencil_action_enable && !depth_write_enabled;

    /**
     * Cuando se puede descartar un pixel tapado ANTES de sombrearlo.
     *
     * El orden sin esto es: muestrear texturas, iluminar, pasar el TEV y solo
     * entonces mirar si el pixel estaba tapado. Todo lo gastado en un pixel que
     * no se ve, tirado.
     *
     * Adelantar las comparaciones es equivalente exacto cuando las acciones de
     * fallo no escriben nada (Keep): el pixel muere igual y ninguna de las dos
     * pruebas deja rastro. Con acciones que escriben, en cambio, saltarse el
     * pixel se comeria ese efecto, asi que ahi no se adelanta.
     *
     * El rechazo lee galga y profundidad, y la prueba final REUTILIZA esos
     * valores (ver EarlyDepthPasses/DoDepthStencilTest): cuando no mata ningun
     * pixel sale casi gratis, no cuesta una segunda lectura por pixel.
     */
    const bool stencil_actions_keep =
        !stencil_action_enable ||
        (output_merger_state.stencil_test.action_stencil_fail == FramebufferRegs::StencilAction::Keep &&
         output_merger_state.stencil_test.action_depth_fail == FramebufferRegs::StencilAction::Keep);
    // Con 'Always' el rechazo temprano no puede matar nada: la lectura de
    // profundidad que hace seria trabajo regalado, asi que se apaga.
    const bool early_z_safe = frag_op_mode == FramebufferRegs::FragmentOperationMode::Default &&
                              output_merger_state.depth_test_enable &&
                              output_merger_state.depth_test_func !=
                                  FramebufferRegs::CompareFunc::Always &&
                              stencil_actions_keep;

    /**
     * La profundidad solo se calcula si alguien la va a leer: el rechazo
     * temprano, la niebla, la prueba final o el modo sombra.
     *
     * En juegos 2D con la profundidad apagada (en NSMB2 el rechazo temprano
     * nunca esta disponible y no hay fallos de profundidad) esto se ahorra por
     * pixel una division, un clamp -- que son dos vmrs, y cada uno serializa la
     * tuberia de coma flotante -- y el resto de la cuenta del z/w.
     */
    const bool needs_depth =
        early_z_safe || fog_enabled || !depth_stencil_noop ||
        frag_op_mode == FramebufferRegs::FragmentOperationMode::Shadow;

    if (early_z_safe) {
        RasterizerStats::tri_early_z.fetch_add(1, std::memory_order_relaxed);
    }

    const auto textures = regs.texturing.GetTextures();
    const auto tev_stages = regs.texturing.GetTevStages();

    // Que etapas TEV hacen algo. Constante en todo el triangulo: los registros
    // no cambian dentro de uno. Ver ComputeActiveTevStages.
    const u32 active_tev_stages = ComputeActiveTevStages(tev_stages);
    RasterizerStats::tev_stages_used.fetch_add(
        static_cast<u32>(__builtin_popcount(active_tev_stages)), std::memory_order_relaxed);

    // Y la configuracion TEV entera, resuelta a un programa plano. Ver TevProgram.
    const TevProgram tev_program = CompileTevProgram(tev_stages, active_tev_stages);

    fb.Bind();

    // Enter rasterization loop, starting at the center of the topleft bounding box corner.
    // TODO: Not sure if looping through x first might be faster
    {
        RasterizerStats::triangles.fetch_add(1, std::memory_order_relaxed);

        // Se resuelve UNA vez por triangulo lo que antes se recalculaba en cada
        // pixel y por cada unidad de textura: el puntero fisico de la textura y
        // su TextureInfo. Con millones de pixeles sombreados por fotograma, eran
        // millones de llamadas fuera de linea para obtener siempre lo mismo.
        std::array<TextureUnitCache, 3> tex_cache{};

        /**
         * Que coordenadas de textura hacen falta de verdad.
         *
         * El bucle interpolaba SIEMPRE las seis: uv[0], uv[1] y uv[2]. Si el
         * juego solo usa la unidad 0 -- que es lo normal en interfaces y en
         * mucha geometria -- cuatro de esas seis interpolaciones se calculan
         * para nada, y cada una son tres productos y dos sumas en coma flotante.
         *
         * Se resuelve por triangulo: los registros de textura no cambian
         * dentro de uno.
         */
        // Cuatro entradas, no tres, aunque solo se consulten las tres primeras.
        //
        // texture3_coordinates es un campo de DOS bits: puede valer 3. Con un
        // array de tres, marcar esa posicion escribe fuera y corrompe la pila
        // -- crash al abrir el juego, que es exactamente lo que provoco la
        // primera version de este cambio.
        //
        // El codigo original de Citra tiene el mismo desfase leyendo uv[3] mas
        // abajo, pero leer fuera es mucho menos destructivo que escribir, y por
        // eso nunca dio la cara. Aqui basta con reservar la cuarta posicion.
        bool need_uv[4] = {false, false, false, false};
        for (u32 i = 0; i < 3; i++) {
            // Solo las unidades que el TEV (o la iluminacion) leen de verdad:
            // muestrear una textura que nadie consulta era trabajo entero
            // tirado, y con el se va tambien su interpolacion de coordenadas.
            // Ver TevProgram::uses_texture.
            if (tev_program.uses_texture[i] && textures[i].enabled &&
                textures[i].config.address != 0) {
                // La unidad 2 puede leer las coordenadas de la 1 (ver
                // texture2_use_coord1 en TextureColor), asi que hay que marcar
                // la que de verdad se va a muestrear.
                const u32 coord =
                    (i == 2 && regs.texturing.main_config.texture2_use_coord1) ? 1 : i;
                need_uv[coord] = true;
            }
        }
        // La textura procedural toma sus coordenadas de la unidad que diga el
        // registro, aunque esa unidad de textura este apagada.
        if (regs.texturing.main_config.texture3_enable) {
            need_uv[regs.texturing.main_config.texture3_coordinates & 3u] = true;
        }

        // tc0_w solo lo lee la unidad 0, y solo si su tipo no es el normal:
        // proyectiva divide por el, y cubica y sombra lo usan como tercera
        // coordenada. Con Texture2D -- el caso habitual -- se interpolaba un
        // atributo por pixel que nadie miraba.
        const bool need_tc0_w =
            tev_program.uses_texture[0] && textures[0].enabled &&
            textures[0].config.type != TexturingRegs::TextureConfig::Texture2D &&
            textures[0].config.type != TexturingRegs::TextureConfig::Disabled;

        for (u32 i = 0; i < 3; i++) {
            const auto& texture = textures[i];
            if (!tev_program.uses_texture[i] || !texture.enabled ||
                texture.config.address == 0) {
                continue;
            }
            tex_cache[i].base_address = texture.config.GetPhysicalAddress();
            tex_cache[i].data = memory.GetPhysicalPointer(tex_cache[i].base_address);
            tex_cache[i].info = TextureInfo::FromPicaRegister(texture.config, texture.format);
            tex_cache[i].width = f24::FromFloat32(static_cast<f32>(texture.config.width));
            tex_cache[i].height = f24::FromFloat32(static_cast<f32>(texture.config.height));

            // Config de muestreo, resuelta aqui para que el bucle no la relea de
            // los registros en cada pixel. Ver TextureUnitCache.
            tex_cache[i].address = tex_cache[i].base_address;
            tex_cache[i].coord_index =
                static_cast<u8>((i == 2 && regs.texturing.main_config.texture2_use_coord1) ? 1 : i);
            tex_cache[i].width_int = texture.config.width;
            tex_cache[i].height_int = texture.config.height;
            // Potencia de dos: el wrap puede usar un AND en vez de la division.
            const u32 unit_width = tex_cache[i].width_int;
            const u32 unit_height = tex_cache[i].height_int;
            tex_cache[i].wrap_s_pow2 = unit_width != 0 && (unit_width & (unit_width - 1)) == 0;
            tex_cache[i].wrap_t_pow2 = unit_height != 0 && (unit_height & (unit_height - 1)) == 0;
            tex_cache[i].width_mask = tex_cache[i].wrap_s_pow2 ? (unit_width - 1) : 0;
            tex_cache[i].height_mask = tex_cache[i].wrap_t_pow2 ? (unit_height - 1) : 0;
            tex_cache[i].type = static_cast<u8>(texture.config.type.Value());
            tex_cache[i].is_shadow =
                texture.config.type == TexturingRegs::TextureConfig::Shadow2D ||
                texture.config.type == TexturingRegs::TextureConfig::ShadowCube;
            tex_cache[i].wrap_s = static_cast<u8>(texture.config.wrap_s.Value());
            tex_cache[i].wrap_t = static_cast<u8>(texture.config.wrap_t.Value());
            tex_cache[i].border_mode_s =
                texture.config.wrap_s == TexturingRegs::TextureConfig::ClampToBorder
                    ? u8{1}
                    : (texture.config.wrap_s == TexturingRegs::TextureConfig::ClampToBorder2 ? u8{2}
                                                                                            : u8{0});
            tex_cache[i].border_mode_t =
                texture.config.wrap_t == TexturingRegs::TextureConfig::ClampToBorder
                    ? u8{1}
                    : (texture.config.wrap_t == TexturingRegs::TextureConfig::ClampToBorder2 ? u8{2}
                                                                                            : u8{0});
            const auto& border_rgba = texture.config.border_color;
            tex_cache[i].border_color =
                Common::MakeVec(border_rgba.r.Value(), border_rgba.g.Value(), border_rgba.b.Value(),
                                border_rgba.a.Value())
                    .Cast<u8>();

            // Formato de la unidad, contado POR TRIANGULO.
            //
            // La primera version contaba por texel con variables thread_local, y
            // fue un desastre: este toolchain no tiene TLS nativo, asi que cada
            // acceso se convertia en una llamada a __emutls_get_address -- que
            // busca en una clave de pthread y toma un mutex interno. Tres veces
            // por pixel y desde tres hilos. De ahi salian los tres
            // 'pthread_mutex_lock EINVAL' del crash.
            //
            // Contar por triangulo responde igual de bien a la pregunta que
            // interesa (¿usa este juego ETC1?) y cuesta dos atomicos por unidad
            // activa en vez de tres llamadas por pixel.
            RasterizerStats::tex_units_total.fetch_add(1, std::memory_order_relaxed);
            if (texture.format == Pica::TexturingRegs::TextureFormat::ETC1 ||
                texture.format == Pica::TexturingRegs::TextureFormat::ETC1A4) {
                RasterizerStats::tex_units_etc1.fetch_add(1, std::memory_order_relaxed);
            }
            // Formato concreto, para saber que rama del switch de
            // LookupTexelInTile es la que de verdad corre. Ver tex_format_units.
            const u32 format_index = static_cast<u32>(texture.format);
            if (format_index < RasterizerStats::tex_format_units.size()) {
                RasterizerStats::tex_format_units[format_index].fetch_add(
                    1, std::memory_order_relaxed);
            }
        }

        const auto process_scanline = [&](u16 y, Pica::Texture::Etc1BlockCache& etc1_cache,
                                          RasterizerStats::ScanCounters& counters) {
            /// Galga y profundidad ya leidas por el rechazo temprano, para que
            /// la prueba final no las vuelva a leer. Solo valen con early_z_safe.
            u32 early_ref_z = 0;
            u8 early_stencil = 0;

            // Valor de las funciones de borde en el primer pixel de la fila.
            // A partir de aqui solo se le suma la pendiente. Ver dw0_dx arriba.
            s32 w0_row = bias0 + SignedArea(vtxpos[1].xy(), vtxpos[2].xy(), {probe_x, y});
            s32 w1_row = bias1 + SignedArea(vtxpos[2].xy(), vtxpos[0].xy(), {probe_x, y});
            s32 w2_row = bias2 + SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), {probe_x, y});

            for (u16 x = probe_x; x < max_x;
                 x += x_pitch, w0_row += dw0_dx, w1_row += dw1_dx, w2_row += dw2_dx) {
                counters.tested++;
                // Do not process the pixel if it's inside the scissor box and the scissor mode is
                // set to Exclude.
                if (scissor_mode == RasterizerRegs::ScissorMode::Exclude) {
                    if (x >= scissor_x1 && x < scissor_x2 && y >= scissor_y1 && y < scissor_y2) {
                        continue;
                    }
                }

                // Coordenadas baricentricas, ya calculadas por incrementos.
                const s32 w0 = w0_row;
                const s32 w1 = w1_row;
                const s32 w2 = w2_row;

                // If current pixel is not covered by the current primitive
                if (w0 < 0 || w1 < 0 || w2 < 0) {
                    continue;
                }

                counters.covered++;
                const s32 wsum = w0 + w1 + w2;

                /**
                 * Interpolacion en float plano en vez de f24.
                 *
                 * f24 guarda un float normal y su aritmetica es la del hardware,
                 * asi que el resultado es el mismo. Lo que cambia es que
                 * f24::operator* comprueba isnan en CADA multiplicacion -- para
                 * emular que la PICA devuelve 0 en vez de NaN al multiplicar por
                 * infinito -- y en ARM esa comprobacion obliga a pasar las
                 * banderas de la FPU al procesador con un 'vmrs'. Esa
                 * instruccion tiene latencia larga y ademas SERIALIZA la tuberia
                 * de coma flotante: espera a que terminen todas las operaciones
                 * en vuelo.
                 *
                 * Contados en el binario: 156 'vmrs' dentro del bucle de linea,
                 * el 13% de sus instrucciones. Aqui se quitan 44 por pixel
                 * (11 atributos x 4 productos cada uno).
                 *
                 * El caso que la comprobacion cubre no puede darse en esta
                 * cuenta concreta: las baricentricas son enteros convertidos a
                 * float y los atributos de los vertices son finitos, asi que no
                 * hay ningun infinito por cero que emular. Donde si puede
                 * pasar -- el sombreado, las texturas -- se sigue usando f24.
                 */
                const f32 bw0 = static_cast<f32>(w0);
                const f32 bw1 = static_cast<f32>(w1);
                const f32 bw2 = static_cast<f32>(w2);
                const f32 w_inverse_dot = w_inverse.x.ToFloat32() * bw0 +
                                          w_inverse.y.ToFloat32() * bw1 +
                                          w_inverse.z.ToFloat32() * bw2;
                const f32 interpolated_w_inverse_f = 1.0f / w_inverse_dot;

                // Solo si alguien lee la profundidad. Ver needs_depth.
                float depth = 0.0f;
                if (needs_depth) {
                    // interpolated_z = z / w
                    const float interpolated_z_over_w =
                        (v0.screenpos[2].ToFloat32() * w0 + v1.screenpos[2].ToFloat32() * w1 +
                         v2.screenpos[2].ToFloat32() * w2) /
                        wsum;

                    // Not fully accurate. About 3 bits in precision are missing.
                    // Z-Buffer (z / w * scale + offset)
                    depth = interpolated_z_over_w * depth_scale + depth_offset;

                    // Potentially switch to W-Buffer
                    if (w_buffering) {
                        // W-Buffer (z * scale + w * offset = (z / w * scale + offset) * w)
                        depth *= interpolated_w_inverse_f * wsum;
                    }

                    // Clamp the result
                    depth = std::clamp(depth, 0.0f, 1.0f);
                }

                // Rechazo temprano: si el pixel esta tapado, no hay por que
                // muestrear texturas, iluminar y pasar las seis etapas TEV solo
                // para acabar tirandolo. Ver early_z_safe.
                if (early_z_safe && !EarlyDepthPasses(x, y, depth, early_ref_z, early_stencil)) {
                    counters.zkill++;
                    continue;
                }

                /**
                 * Perspective correct attribute interpolation:
                 * Attribute values cannot be calculated by simple linear interpolation since
                 * they are not linear in screen space. For example, when interpolating a
                 * texture coordinate across two vertices, something simple like
                 *     u = (u0*w0 + u1*w1)/(w0+w1)
                 * will not work. However, the attribute value divided by the
                 * clipspace w-coordinate (u/w) and and the inverse w-coordinate (1/w) are linear
                 * in screenspace. Hence, we can linearly interpolate these two independently and
                 * calculate the interpolated attribute by dividing the results.
                 * I.e.
                 *     u_over_w   = ((u0/v0.pos.w)*w0 + (u1/v1.pos.w)*w1)/(w0+w1)
                 *     one_over_w = (( 1/v0.pos.w)*w0 + ( 1/v1.pos.w)*w1)/(w0+w1)
                 *     u = u_over_w / one_over_w
                 *
                 * The generalization to three vertices is straightforward in baricentric
                 *coordinates.
                 **/
                const auto get_interpolated_attribute = [&](f24 attr0, f24 attr1, f24 attr2) {
                    const f32 over_w = attr0.ToFloat32() * bw0 + attr1.ToFloat32() * bw1 +
                                       attr2.ToFloat32() * bw2;
                    return f24::FromFloat32(over_w * interpolated_w_inverse_f);
                };

                // Atributos interpolados. En modo de ablacion se saltan y se
                // usan valores fijos (ver Ablation).
                Common::Vec4<u8> primary_color = {0, 0, 0, 0};
                std::array<Common::Vec2<f24>, 3> uv{};
                f24 tc0_w = f24::Zero();

                if (ablation_mode == Ablation::kNoInterpolation) {
                    primary_color = {255, 255, 255, 255};
                } else {
                    // El color del vertice solo se interpola si alguna etapa
                    // activa lo lee como fuente (PrimaryColor). Si no, son cuatro
                    // interpolaciones con sus redondeos por pixel para un valor
                    // que nadie mira.
                    if (tev_program.uses_primary_color) {
#if defined(__ARM_NEON) && defined(__PSVITA__)
                        // Los cuatro canales de una vez: tres vmla para la
                        // interpolacion y el redondeo con vmax/vmin/vcvt.
                        // Ademas del ahorro de instrucciones, quita los OCHO
                        // vmrs por pixel (dos por canal) que hacia
                        // RoundColorComponent: cada vmrs espera a que la
                        // tuberia de coma flotante se vacie.
                        //
                        // El redondeo es identico al escalar: vmax con un NaN
                        // devuelve el otro operando (NaN -> 0, como el
                        // original), +inf acaba en 255 por el vmin, y vcvt
                        // trunca hacia cero.
                        alignas(16) float c0[4] = {v0.color.r().ToFloat32(), v0.color.g().ToFloat32(),
                                                   v0.color.b().ToFloat32(),
                                                   v0.color.a().ToFloat32()};
                        alignas(16) float c1[4] = {v1.color.r().ToFloat32(), v1.color.g().ToFloat32(),
                                                   v1.color.b().ToFloat32(),
                                                   v1.color.a().ToFloat32()};
                        alignas(16) float c2[4] = {v2.color.r().ToFloat32(), v2.color.g().ToFloat32(),
                                                   v2.color.b().ToFloat32(),
                                                   v2.color.a().ToFloat32()};
                        float32x4_t over_w = vmulq_n_f32(vld1q_f32(c0), bw0);
                        over_w = vmlaq_n_f32(over_w, vld1q_f32(c1), bw1);
                        over_w = vmlaq_n_f32(over_w, vld1q_f32(c2), bw2);
                        over_w = vmulq_n_f32(over_w, interpolated_w_inverse_f);
                        float32x4_t scaled = vmlaq_n_f32(vdupq_n_f32(0.5f), over_w, 255.0f);
                        scaled = vmaxq_f32(scaled, vdupq_n_f32(0.0f));
                        scaled = vminq_f32(scaled, vdupq_n_f32(255.0f));
                        const uint16x4_t narrow16 = vqmovn_u32(vcvtq_u32_f32(scaled));
                        const uint8x8_t narrow8 =
                            vqmovn_u16(vcombine_u16(narrow16, vdup_n_u16(0)));
                        alignas(8) u8 rgba[8];
                        vst1_u8(rgba, narrow8);
                        primary_color = {rgba[0], rgba[1], rgba[2], rgba[3]};
#else
                        primary_color = {
                            RoundColorComponent(get_interpolated_attribute(v0.color.r(), v1.color.r(),
                                                                           v2.color.r())),
                            RoundColorComponent(get_interpolated_attribute(v0.color.g(), v1.color.g(),
                                                                           v2.color.g())),
                            RoundColorComponent(get_interpolated_attribute(v0.color.b(), v1.color.b(),
                                                                           v2.color.b())),
                            RoundColorComponent(get_interpolated_attribute(v0.color.a(), v1.color.a(),
                                                                           v2.color.a())),
                        };
#endif
                    }

                    // A cero, no sin inicializar.
                    //
                    // Desde que solo se interpolan las coordenadas que alguien
                    // va a leer (ver need_uv), las demas se quedarian con lo que
                    // hubiera en la pila. Si ese analisis fallara en algun caso
                    // raro, esa basura se usaria como coordenada de textura: s y
                    // t enormes y lectura fuera de la textura. Poner a cero
                    // cuesta seis escrituras por pixel -- menos del 0,2% -- y
                    // cierra esa puerta.
                    // Solo las coordenadas que alguna unidad va a leer. Ver need_uv.
                    if (need_uv[0]) {
                        uv[0].u() = get_interpolated_attribute(v0.tc0.u(), v1.tc0.u(), v2.tc0.u());
                        uv[0].v() = get_interpolated_attribute(v0.tc0.v(), v1.tc0.v(), v2.tc0.v());
                    }
                    if (need_uv[1]) {
                        uv[1].u() = get_interpolated_attribute(v0.tc1.u(), v1.tc1.u(), v2.tc1.u());
                        uv[1].v() = get_interpolated_attribute(v0.tc1.v(), v1.tc1.v(), v2.tc1.v());
                    }
                    if (need_uv[2]) {
                        uv[2].u() = get_interpolated_attribute(v0.tc2.u(), v1.tc2.u(), v2.tc2.u());
                        uv[2].v() = get_interpolated_attribute(v0.tc2.v(), v1.tc2.v(), v2.tc2.v());
                    }

                    // Sample bound texture units.
                    tc0_w = need_tc0_w ? get_interpolated_attribute(v0.tc0_w, v1.tc0_w, v2.tc0_w)
                                       : f24::Zero();
                }

                std::array<Common::Vec4<u8>, 4> texture_color{};
                if (ablation_mode != Ablation::kNoTexture) {
                    texture_color = TextureColor(uv, textures, tc0_w, tex_cache, etc1_cache,
                                                 tev_program.uses_texture);
                }

                Common::Vec4<u8> primary_fragment_color = {0, 0, 0, 0};
                Common::Vec4<u8> secondary_fragment_color = {0, 0, 0, 0};

                if (lighting_enabled) {
                    counters.lit++;
                    const auto normquat =
                        Common::Quaternion<f32>{
                            {get_interpolated_attribute(v0.quat.x, v1.quat.x, v2.quat.x)
                                 .ToFloat32(),
                             get_interpolated_attribute(v0.quat.y, v1.quat.y, v2.quat.y)
                                 .ToFloat32(),
                             get_interpolated_attribute(v0.quat.z, v1.quat.z, v2.quat.z)
                                 .ToFloat32()},
                            get_interpolated_attribute(v0.quat.w, v1.quat.w, v2.quat.w).ToFloat32(),
                        }
                            .Normalized();

                    const Common::Vec3f view{
                        get_interpolated_attribute(v0.view.x, v1.view.x, v2.view.x).ToFloat32(),
                        get_interpolated_attribute(v0.view.y, v1.view.y, v2.view.y).ToFloat32(),
                        get_interpolated_attribute(v0.view.z, v1.view.z, v2.view.z).ToFloat32(),
                    };
                    std::tie(primary_fragment_color, secondary_fragment_color) =
                        ComputeFragmentsColors(regs.lighting, pica.lighting, normquat, view,
                                               texture_color);
                }

                // Write the TEV stages.
                bool alpha_tested = false;
                Common::Vec4<u8> combiner_output;
                if (ablation_mode == Ablation::kNoTev) {
                    combiner_output = texture_color[0];
                } else if (tev_program.alpha_pure && alpha_test_can_fail &&
                           frag_op_mode != FramebufferRegs::FragmentOperationMode::Shadow) {
                    // Sin una prueba de alfa que pueda fallar no hay nada que
                    // adelantar, y partir el TEV en dos pasadas solo añadiria
                    // trabajo.
                    /**
                     * Dos pasadas: primero la cadena de alfa, que es
                     * independiente, y la prueba de alfa; si el pixel muere,
                     * no se calcula la cadena de color ni nada de lo que viene
                     * detras (niebla, profundidad, mezcla, escritura).
                     *
                     * Medido en NSMB2: el 26% de los pixeles cubiertos muere en
                     * la prueba de alfa, despues de sombrearse entero. Esto es
                     * lo que evita ese trabajo.
                     */
                    const std::array<u8, 6> stage_alpha =
                        ComputeTevAlpha(texture_color, tev_program, primary_color,
                                        primary_fragment_color, secondary_fragment_color);
                    // En modo "sin merger" la prueba tampoco se adelanta: el
                    // modo mide el TEV con el output merger apagado entero.
                    alpha_tested = alpha_test_enabled && ablation_mode != Ablation::kNoMerger;
                    if (alpha_tested && !DoAlphaTest(stage_alpha[5])) {
                        counters.alpha_fail++;
                        continue;
                    }
                    combiner_output =
                        WriteTevConfig(texture_color, tev_program, primary_color,
                                       primary_fragment_color, secondary_fragment_color,
                                       &stage_alpha);
                } else {
                    combiner_output = WriteTevConfig(texture_color, tev_program, primary_color,
                                                     primary_fragment_color,
                                                     secondary_fragment_color);
                }

                if (frag_op_mode == FramebufferRegs::FragmentOperationMode::Shadow) {
                    const u32 depth_int = static_cast<u32>(depth * 0xFFFFFF);
                    // Use green color as the shadow intensity
                    const u8 stencil = combiner_output.y;
                    fb.DrawShadowMapPixel(x >> 4, y >> 4, depth_int, stencil);
                    // Skip the normal output merger pipeline if it is in shadow mode
                    continue;
                }

                // El output merger entero, saltado en modo de ablacion.
                if (ablation_mode != Ablation::kNoMerger) {
                    // La prueba de alfa ya se hizo en el camino de dos pasadas
                    // (alpha_tested); aqui solo corre si no se adelanto.
                    // Does alpha testing happen before or after stencil?
                    if (!alpha_tested && alpha_test_enabled && !DoAlphaTest(combiner_output.a())) {
                        counters.alpha_fail++;
                        continue;
                    }
                    if (fog_enabled) {
                        WriteFog(depth, combiner_output);
                    }
                    if (!depth_stencil_noop &&
                        !DoDepthStencilTest(x, y, depth, early_z_safe ? &early_ref_z : nullptr,
                                            early_z_safe ? &early_stencil : nullptr)) {
                        counters.depth_fail++;
                        continue;
                    }
                }
                if (allow_color_write) {
                    const u32 px = x >> 4;
                    const u32 py = y >> 4;
                    // Modo de ablacion "sin escritura": se calcula todo igual,
                    // pero el framebuffer no se toca.
                    const bool no_write = ablation_mode == Ablation::kNoWrite;

                    /**
                     * Mezcla alfa estandar: atajos para los dos extremos.
                     *
                     * Con factores (SourceAlpha, OneMinusSourceAlpha) y ecuacion
                     * Add -- que es como mezcla casi todo el 2D -- el resultado
                     * es exactamente:
                     *
                     *     src*alpha + dst*(255-alpha)
                     *     ---------------------------  =  src*alpha/255 + dst*(1-alpha/255)
                     *                255
                     *
                     * Con alpha = 255 el segundo termino es cero y el resultado
                     * es el color que entra: no hace falta leer el framebuffer,
                     * ni calcular factores, ni mezclar. Con alpha = 0 el
                     * resultado es el destino tal cual, asi que escribir seria
                     * dejar el mismo valor.
                     *
                     * Los dos casos son la mayoria de los pixeles de arte 2D
                     * (interior y exterior de los sprites): saltarselos ahorra,
                     * por pixel, una lectura del framebuffer y todo el camino de
                     * mezcla.
                     */
                    if (ablation_mode == Ablation::kNoMerger) {
                        if (!no_write) {
                            draw_scaled(px, py, combiner_output);
                        }
                        counters.drawn++;
                    } else if (blend.standard_src_alpha && blend.all_channels_enabled &&
                               combiner_output.a() == 255) {
                        if (!no_write) {
                            draw_scaled(px, py, combiner_output);
                        }
                        counters.drawn++;
                    } else if (blend.standard_src_alpha && blend.all_channels_enabled &&
                               combiner_output.a() == 0) {
                        // Resultado == destino: escribir no cambiaria nada.
                        counters.drawn++;
                    } else {
                        const auto result = PixelColor(x, y, combiner_output, blend);
                        if (!no_write) {
                            draw_scaled(px, py, result);
                        }
                        counters.drawn++;
                    }
                }
            }

        };

        // Reparto del triangulo entre hilos: por BANDAS, no por linea.
        //
        // Antes esto encolaba una tarea por cada linea del triangulo. Cada
        // encolado cuesta una reserva de heap (UniqueFunction hace make_unique,
        // no tiene buffer interno), un mutex y una senal de condicion; y al
        // final del triangulo, una barrera. Para un triangulo tipico de unos
        // pocos pixeles de alto -- que es la inmensa mayoria en cualquier juego
        // 3D -- ese coste de sincronizacion se come el trabajo real varias veces.
        //
        // Se medio: 90% del tiempo de fotograma aqui dentro, con los tres
        // nucleos al 100% y ~6.600 ciclos por pixel. Un rasterizador software
        // lento anda por 100-500. La diferencia era casi toda sincronizacion.
        //
        // Ahora: los triangulos pequenos se dibujan en el mismo hilo, sin tocar
        // el pool; los grandes se reparten en tantas bandas como hilos haya, o
        // sea 3 encolados por triangulo en vez de uno por linea.
        const u32 scanlines = (max_y > min_y) ? static_cast<u32>((max_y - min_y) >> 4) : 0u;
        const u32 width_px = (max_x > min_x) ? static_cast<u32>((max_x - min_x) >> 4) : 0u;

        // Umbral para NO repartir. Muy bajo a proposito.
        //
        // El primer intento uso 4096 pixeles (64x64) y fue un error caro: la
        // mayoria de triangulos reales miden menos que eso, asi que se fueron
        // todos al camino de un solo hilo. El resultado fue exactamente 3 veces
        // mas lento -- justo los 3 hilos que dejaron de usarse -- y el reparto
        // de tiempo lo delataba: la CPU bajo al 1% y la "GPU" subio al 98%,
        // porque el hilo principal se habia quedado rasterizando el solo.
        //
        // El reparto por bandas ya cuesta solo 3 encolados por triangulo (antes
        // era uno por linea), asi que casi siempre compensa. Solo se evita en
        // triangulos tan finos que no se pueden partir en tres de forma util, o
        // tan pequenos que el trabajo cabe en el coste de un encolado.
        // Medido en consola: con 4 y 256 solo se repartia el 14,2% de los
        // triangulos. El 86% se rasterizaba en un nucleo con los otros dos
        // parados. El contador 'hilos' del overlay da esa cifra.
        //
        // Los umbrales estaban puestos a ojo, con la idea de que encolar sale
        // caro para triangulos pequenos. Con el coste por pixel ya medido
        // (~2.500-3.000 ciclos, que es mucho) esa cuenta cambia: encolar tres
        // bandas y esperar la barrera son del orden de 15.000 ciclos, o sea
        // que a partir de unas decenas de pixeles ya compensa repartir.
        //
        // 3 lineas es el minimo con sentido: son tres hilos, una linea cada
        // uno. Por debajo de eso sobran bandas.
        //
        // CUIDADO al tocar esto en la direccion contraria: en la 0.2 se probo
        // con 4096 y mando practicamente todos los triangulos al camino de un
        // hilo. El emulador fue exactamente 3 veces mas lento -- justo los tres
        // nucleos que dejaron de usarse. Si 'hilos' baja, el rendimiento cae.

        constexpr u32 kMinScanlinesToSplit = 3;
        constexpr u32 kMinPixelsToSplit = 64;

#ifdef __PSVITA__
        const SceUInt64 profile_work = sceKernelGetProcessTimeWide();
        RasterizerStats::tri_setup_us.fetch_add(profile_work - profile_t0,
                                                std::memory_order_relaxed);
#endif

        if (scanlines == 0) {
            // Nada que dibujar.
        } else if (scanlines < kMinScanlinesToSplit ||
                   scanlines * width_px < kMinPixelsToSplit || num_sw_threads <= 1) {
            RasterizerStats::tri_single.fetch_add(1, std::memory_order_relaxed);
            RasterizerStats::ScanCounters counters{};
            for (u16 y = y_origin + 8; y < max_y; y += y_pitch) {
                process_scanline(y, caller_etc1_cache, counters);
            }
            counters.Flush();
#ifdef __PSVITA__
            RasterizerStats::tri_single_us.fetch_add(
                sceKernelGetProcessTimeWide() - profile_work, std::memory_order_relaxed);
#endif
        } else {
            RasterizerStats::tri_split.fetch_add(1, std::memory_order_relaxed);
            // Lineas INTERCALADAS, no bandas contiguas.
            //
            // Antes esto partia el triangulo en tres bandas horizontales de la
            // misma ALTURA. Y un triangulo no es un rectangulo: para uno con la
            // base abajo y el vertice arriba, las areas de esas tres bandas van
            // como 1 : 3 : 5. La banda de abajo se lleva el 56% de los pixeles
            // y las otras dos terminan pronto y se quedan esperando en la
            // barrera.
            //
            // O sea que con tres hilos el paralelismo efectivo era ~1,8x, no 3x.
            // Eso explica ademas por que bajar el umbral de reparto (de 14% a
            // 97% de triangulos repartidos) no dio nada: el problema no era
            // cuantos triangulos se repartian, era COMO.
            //
            // Intercalando, cada hilo coge una linea de cada tres. El reparto
            // queda equilibrado sea cual sea la forma del triangulo, porque
            // lineas contiguas tienen anchuras parecidas.
            //
            // El coste es que los tres hilos escriben dentro del mismo mosaico
            // de 8 lineas del framebuffer y comparten lineas de cache. Hecha la
            // cuenta, es despreciable: un mosaico son 256 bytes (8 lineas de
            // cache) y se comparte una vez por cada 8 filas, frente a miles de
            // ciclos de sombreado por fila. Se pierde eso y se gana medio nucleo.
            const u32 bands = static_cast<u32>(num_sw_threads);
            const u16 y_first = static_cast<u16>(y_origin + 8);

            /**
             * Reparto por FILAS DE MOSAICO, no por lineas sueltas.
             *
             * El framebuffer esta organizado en mosaicos de 8x8 pixeles: un
             * mosaico RGBA8 son 256 bytes, o sea OCHO lineas de cache. Al
             * recorrer el triangulo linea a linea se toca un mosaico, se usa
             * UNA de sus ocho filas y se pasa al siguiente. Para cuando toca la
             * fila de abajo ya se ha recorrido todo el ancho del triangulo y el
             * mosaico se ha desalojado.
             *
             * Y el conjunto no cabe: color 400x240 en RGB565 son 192 KB y la
             * profundidad D24S8 otros 384 KB, contra 512 KB de L2 compartida
             * entre los tres nucleos. O sea que se recarga constantemente.
             *
             * Dando a cada hilo bloques de 8 lineas consecutivas, cada traida
             * de 256 bytes se aprovecha ocho veces en vez de una.
             *
             * El reparto sigue siendo alterno -- bloque 0 al hilo 0, bloque 1 al
             * hilo 1... -- para no perder el equilibrio de carga que se gano al
             * dejar las bandas contiguas: un triangulo no es un rectangulo y
             * tres bandas grandes reparten 1:3:5 de trabajo.
             *
             * Con triangulos bajos no hay bloques que repartir, asi que por
             * debajo de kMinLinesForTileOrder se vuelve al reparto por lineas
             * alternas, que equilibra mejor cuando hay pocas filas.
             */
            constexpr u32 kTileLines = 8;
            constexpr u32 kMinLinesForTileOrder = 24;

            /**
             * La banda 0 la rasteriza ESTE hilo, no el pool.
             *
             * Antes se encolaban las tres bandas y el hilo que llama se dormia
             * en la barrera. Eso dejaba su nucleo -- el 0, donde esta atado el
             * hilo de emulacion -- ocupado por un hilo del pool, y metia en el
             * camino critico de cada triangulo dos viajes al planificador:
             * dormir a este hilo y volver a despertarlo. Mil veces por fotograma.
             *
             * Haciendo aqui una de las bandas queda el mismo reparto de tres
             * bandas sobre tres nucleos, pero con un encolado menos, un
             * despertar menos y, sobre todo, sin dormir a nadie: cuando este
             * hilo termina su banda a los otros dos les queda muy poco, asi que
             * la barrera se resuelve girando en vacio (WaitForRequestsSpin).
             *
             * El reparto de lineas es exactamente el de antes -- la banda 0
             * recorre las mismas que recorreria el hilo 0 del pool -- asi que no
             * cambia ningun pixel: las bandas tocan lineas distintas y nunca se
             * pisan entre ellas.
             */
            const auto run_band = [&](u16 y_begin, u16 y_block_height, u16 y_stride,
                                      Pica::Texture::Etc1BlockCache& etc1_cache) {
#ifdef __PSVITA__
                const SceUInt64 band_t0 = sceKernelGetProcessTimeWide();
#endif
                RasterizerStats::ScanCounters counters{};
                if (y_block_height != 0) {
                    // Reparto por bloques de 8 lineas (ver arriba).
                    for (u16 block = y_begin; block < max_y; block += y_stride) {
                        const u16 block_end = std::min<u16>(block + y_block_height, max_y);
                        for (u16 y = block; y < block_end; y += y_pitch) {
                            process_scanline(y, etc1_cache, counters);
                        }
                    }
                } else {
                    // Reparto por lineas alternas.
                    for (u16 y = y_begin; y < max_y; y += y_stride) {
                        process_scanline(y, etc1_cache, counters);
                    }
                }
                counters.Flush();
#ifdef __PSVITA__
                // Ocupacion real de los nucleos. Ver RasterizerStats::band_busy_us.
                RasterizerStats::band_busy_us.fetch_add(
                    sceKernelGetProcessTimeWide() - band_t0, std::memory_order_relaxed);
#endif
            };

            const bool tile_order = scanlines >= kMinLinesForTileOrder;
            const u16 band_height = tile_order ? static_cast<u16>(kTileLines * y_pitch) : u16{0};
            const u16 band_stride = tile_order ? static_cast<u16>(band_height * bands)
                                               : static_cast<u16>(y_pitch * bands);
            const u16 band_offset = tile_order ? band_height : y_pitch;

            // Las bandas 1..N-1 a los nucleos 1 y 2. Se encolan ANTES de ponerse
            // a trabajar, para que arranquen cuanto antes.
            u32 queued = 0;
            for (u32 band = 1; band < bands; band++) {
                const u16 y_begin = static_cast<u16>(y_first + band * band_offset);
                if (y_begin >= max_y) {
                    break;
                }
                // Se captura por referencia igual que antes: la barrera de abajo
                // garantiza que nada de lo referenciado se destruye antes de que
                // las tareas terminen.
                sw_workers.QueueWork([&run_band, y_begin, band_height,
                                      band_stride](Pica::Texture::Etc1BlockCache* etc1_cache) {
                    run_band(y_begin, band_height, band_stride, *etc1_cache);
                });
                queued++;
            }

            // Y la banda 0 aqui mismo, en el nucleo 0.
            run_band(y_first, band_height, band_stride, caller_etc1_cache);

            if (queued != 0) {
                /**
                 * Vueltas de espera activa antes de pasar a dormir.
                 *
                 * Con las bandas equilibradas, lo que queda cuando este hilo
                 * acaba la suya son microsegundos: unos miles de vueltas cubren
                 * el caso normal sin tocar el planificador. Si una banda sale
                 * mucho mas cara que las otras se agota la cuenta y se duerme,
                 * que para esperas largas es lo correcto.
                 *
                 * Girar en vacio aqui no le quita tiempo a nadie: este hilo esta
                 * atado al nucleo 0 y los del pool a los nucleos 1 y 2.
                 */
                constexpr u32 kBarrierSpinRounds = 4096;
                sw_workers.WaitForRequestsSpin(kBarrierSpinRounds);
            }
#ifdef __PSVITA__
            RasterizerStats::tri_wait_us.fetch_add(sceKernelGetProcessTimeWide() - profile_work,
                                                   std::memory_order_relaxed);
#endif
        }
    }
}

std::array<Common::Vec4<u8>, 4> RasterizerSoftware::TextureColor(
    std::span<const Common::Vec2<f24>, 3> uv,
    std::span<const Pica::TexturingRegs::FullTextureConfig, 3> textures, f24 tc0_w,
    const std::array<TextureUnitCache, 3>& tex_cache,
    Pica::Texture::Etc1BlockCache& etc1_cache, const bool (&used_units)[4]) const {
    std::array<Common::Vec4<u8>, 4> texture_color{};
    for (u32 i = 0; i < 3; ++i) {
        const auto& texture = textures[i];
        // Nadie lee esta unidad (ni el TEV ni la iluminacion): no se muestrea.
        // Ademas de ahorrar el trabajo, evita tocar una cache que a proposito
        // no se relleno (ver uses_texture en ProcessTriangle).
        if (!used_units[i]) [[unlikely]] {
            continue;
        }
        if (!texture.enabled) [[unlikely]] {
            continue;
        }
        if (texture.config.address == 0) [[unlikely]] {
            texture_color[i] = {0, 0, 0, 255};
            continue;
        }

        // Config de la unidad resuelta por triangulo. Ver TextureUnitCache.
        const auto& unit = tex_cache[i];

        const s32 coordinate_i = unit.coord_index;
        f24 u = uv[coordinate_i].u();
        f24 v = uv[coordinate_i].v();

        // Only unit 0 respects the texturing type (according to 3DBrew)
        PAddr texture_address = unit.address;
        f24 shadow_z;
        if (i == 0) {
            switch (static_cast<TexturingRegs::TextureConfig::TextureType>(unit.type)) {
            case TexturingRegs::TextureConfig::Texture2D:
                break;
            case TexturingRegs::TextureConfig::ShadowCube:
            case TexturingRegs::TextureConfig::TextureCube: {
                std::tie(u, v, shadow_z, texture_address) =
                    ConvertCubeCoord(u, v, tc0_w, regs.texturing);
                break;
            }
            case TexturingRegs::TextureConfig::Projection2D: {
                u /= tc0_w;
                v /= tc0_w;
                break;
            }
            case TexturingRegs::TextureConfig::Shadow2D: {
                if (!regs.texturing.shadow.orthographic) {
                    u /= tc0_w;
                    v /= tc0_w;
                }
                shadow_z = f24::FromFloat32(std::abs(tc0_w.ToFloat32()));
                break;
            }
            case TexturingRegs::TextureConfig::Disabled:
                continue; // skip this unit and continue to the next unit
            default:
                LOG_ERROR(HW_GPU, "Unhandled texture type {:x}", (int)unit.type);
                UNIMPLEMENTED();
                break;
            }
        }

        // Ancho y alto precalculados por triangulo (ver TextureUnitCache).
        s32 s = static_cast<s32>(ScaleTexCoord(u, unit.width));
        s32 t = static_cast<s32>(ScaleTexCoord(v, unit.height));

        const s32 width_int = static_cast<s32>(unit.width_int);
        const s32 height_int = static_cast<s32>(unit.height_int);

        bool use_border_s = false;
        bool use_border_t = false;

        if (unit.border_mode_s == 1) {
            use_border_s = s < 0 || s >= width_int;
        } else if (unit.border_mode_s == 2) {
            use_border_s = s >= width_int;
        }

        if (unit.border_mode_t == 1) {
            use_border_t = t < 0 || t >= height_int;
        } else if (unit.border_mode_t == 2) {
            use_border_t = t >= height_int;
        }

        if (use_border_s || use_border_t) {
            texture_color[i] = unit.border_color;
        } else {
            // Textures are laid out from bottom to top, hence we invert the t coordinate.
            // NOTE: This may not be the right place for the inversion.
            // TODO: Check if this applies to ETC textures, too.
            const auto wrap_s = static_cast<TexturingRegs::TextureConfig::WrapMode>(unit.wrap_s);
            const auto wrap_t = static_cast<TexturingRegs::TextureConfig::WrapMode>(unit.wrap_t);
            s = GetWrappedTexCoord(wrap_s, s, unit.width_int, unit.width_mask, unit.wrap_s_pow2);
            t = height_int - 1 -
                GetWrappedTexCoord(wrap_t, t, unit.height_int, unit.height_mask,
                                   unit.wrap_t_pow2);

            // Precalculado por triangulo (ver TextureUnitCache). 'info' solo
            // depende de los registros, asi que siempre vale; el puntero depende
            // de la direccion, que las texturas de cubo cambian por pixel -- de
            // ahi la comprobacion.
            const u8* texture_data = (texture_address == unit.base_address)
                                         ? unit.data
                                         : memory.GetPhysicalPointerThreadSafe(texture_address);
            const auto& info = unit.info;

            // TODO: Apply the min and mag filters to the texture
            texture_color[i] = LookupTexture(texture_data, s, t, info, false, &etc1_cache);
        }

        if (i == 0 && unit.is_shadow) {

            s32 z_int = static_cast<s32>(std::min(shadow_z.ToFloat32(), 1.0f) * 0xFFFFFF);
            z_int -= regs.texturing.shadow.bias << 1;
            const auto& color = texture_color[i];
            const s32 z_ref = (color.w << 16) | (color.z << 8) | color.y;
            u8 density;
            if (z_ref >= z_int) {
                density = color.x;
            } else {
                density = 0;
            }
            texture_color[i] = {density, density, density, density};
        }
    }

    // Sample procedural texture
    if (regs.texturing.main_config.texture3_enable) {
        const auto& proctex_uv = uv[regs.texturing.main_config.texture3_coordinates];
        texture_color[3] = ProcTex(proctex_uv.u().ToFloat32(), proctex_uv.v().ToFloat32(),
                                   regs.texturing, pica.proctex);
    }

    return texture_color;
}

Common::Vec4<u8> RasterizerSoftware::PixelColor(u16 x, u16 y, Common::Vec4<u8> combiner_output,
                                                const BlendState& blend) const {
    const auto& output_merger = regs.framebuffer.output_merger;

    /**
     * Sin mezcla de alfa y con LogicOp Copy, el resultado es exactamente el
     * color que entra -- y con los cuatro canales habilitados, ni siquiera se
     * mira el destino. Leerlo era una decodificacion completa desde el
     * framebuffer (Morton + switch de formato) por pixel, tirada a la basura.
     *
     * Es el caso de la geometria opaca sobre fondo: no cuesta nada
     * comprobarlo y ahorra la lectura en escenas enteras.
     */
    if (!output_merger.alphablend_enable &&
        output_merger.logic_op == FramebufferRegs::LogicOp::Copy && output_merger.red_enable &&
        output_merger.green_enable && output_merger.blue_enable && output_merger.alpha_enable) {
        return combiner_output;
    }

    const auto dest = fb.GetPixel(x >> 4, y >> 4);

    /**
     * Camino directo de la mezcla estandar.
     *
     * Factores (SourceAlpha, OneMinusSourceAlpha) en los cuatro canales y
     * ecuacion Add: los factores no dependen de los registros (son el alfa del
     * fragmento), asi que se construyen directamente y se evitan los ocho
     * despachos de lookup_factor, la carga de blend_const y la comprobacion de
     * ecuaciones distintas. El calculo lo hace el mismo camino NEON de
     * EvaluateBlendEquation que usaba el camino generico.
     */
    if (blend.standard_src_alpha) {
        const u8 alpha = combiner_output.a();
        const u8 inv_alpha = static_cast<u8>(255 - alpha);
        const auto srcfactor = Common::MakeVec<u8>(alpha, alpha, alpha, alpha);
        const auto dstfactor = Common::MakeVec<u8>(inv_alpha, inv_alpha, inv_alpha, inv_alpha);
        const auto blend_output = EvaluateBlendEquation(combiner_output, srcfactor, dest, dstfactor,
                                                        FramebufferRegs::BlendEquation::Add);
        return {static_cast<u8>(output_merger.red_enable ? blend_output.r() : dest.r()),
                static_cast<u8>(output_merger.green_enable ? blend_output.g() : dest.g()),
                static_cast<u8>(output_merger.blue_enable ? blend_output.b() : dest.b()),
                static_cast<u8>(output_merger.alpha_enable ? blend_output.a() : dest.a())};
    }

    Common::Vec4<u8> blend_output = combiner_output;

    if (output_merger.alphablend_enable) {
        const auto params = output_merger.alpha_blending;

        // Fuera del lambda: estaba DENTRO, asi que sus cuatro lecturas de
        // registro y la construccion del vector se repetian en cada una de las
        // ocho llamadas por pixel, para un valor que no cambia.
        const Common::Vec4<u8> blend_const =
            Common::MakeVec(output_merger.blend_const.r.Value(),
                            output_merger.blend_const.g.Value(),
                            output_merger.blend_const.b.Value(),
                            output_merger.blend_const.a.Value())
                .Cast<u8>();

        const auto lookup_factor = [&](u32 channel, FramebufferRegs::BlendFactor factor) -> u8 {
            DEBUG_ASSERT(channel < 4);

            switch (factor) {
            case FramebufferRegs::BlendFactor::Zero:
                return 0;
            case FramebufferRegs::BlendFactor::One:
                return 255;
            case FramebufferRegs::BlendFactor::SourceColor:
                return combiner_output[channel];
            case FramebufferRegs::BlendFactor::OneMinusSourceColor:
                return 255 - combiner_output[channel];
            case FramebufferRegs::BlendFactor::DestColor:
                return dest[channel];
            case FramebufferRegs::BlendFactor::OneMinusDestColor:
                return 255 - dest[channel];
            case FramebufferRegs::BlendFactor::SourceAlpha:
                return combiner_output.a();
            case FramebufferRegs::BlendFactor::OneMinusSourceAlpha:
                return 255 - combiner_output.a();
            case FramebufferRegs::BlendFactor::DestAlpha:
                return dest.a();
            case FramebufferRegs::BlendFactor::OneMinusDestAlpha:
                return 255 - dest.a();
            case FramebufferRegs::BlendFactor::ConstantColor:
                return blend_const[channel];
            case FramebufferRegs::BlendFactor::OneMinusConstantColor:
                return 255 - blend_const[channel];
            case FramebufferRegs::BlendFactor::ConstantAlpha:
                return blend_const.a();
            case FramebufferRegs::BlendFactor::OneMinusConstantAlpha:
                return 255 - blend_const.a();
            case FramebufferRegs::BlendFactor::SourceAlphaSaturate:
                // Returns 1.0 for the alpha channel
                if (channel == 3) {
                    return 255;
                }
                return std::min(combiner_output.a(), static_cast<u8>(255 - dest.a()));
            default:
                LOG_CRITICAL(HW_GPU, "Unknown blend factor {:x}", factor);
                UNIMPLEMENTED();
                break;
            }
            return combiner_output[channel];
        };

        /**
         * Un switch por FACTOR, no por canal.
         *
         * lookup_factor se llamaba ocho veces por pixel -- cuatro canales por
         * origen y destino -- y cada llamada es un switch de dieciseis ramas
         * sobre un valor que viene de un registro. Pero factores distintos solo
         * hay cuatro: los tres canales de color comparten el mismo, y el alfa
         * tiene el suyo.
         *
         * Resolviendo el switch una vez y aplicandolo a los tres canales, los
         * ocho despachos se quedan en cuatro y desaparecen seis lecturas
         * repetidas de blend_const.
         */
        const auto rgb_factor = [&](FramebufferRegs::BlendFactor factor) -> Common::Vec3<u8> {
            return Common::MakeVec(lookup_factor(0, factor), lookup_factor(1, factor),
                                   lookup_factor(2, factor));
        };

        const Common::Vec3<u8> src_rgb = rgb_factor(params.factor_source_rgb);
        const Common::Vec3<u8> dst_rgb = rgb_factor(params.factor_dest_rgb);
        const auto srcfactor =
            Common::MakeVec(src_rgb.r(), src_rgb.g(), src_rgb.b(),
                            lookup_factor(3, params.factor_source_a));
        const auto dstfactor =
            Common::MakeVec(dst_rgb.r(), dst_rgb.g(), dst_rgb.b(),
                            lookup_factor(3, params.factor_dest_a));

        blend_output = EvaluateBlendEquation(combiner_output, srcfactor, dest, dstfactor,
                                             params.blend_equation_rgb);
        // La segunda llamada calcula los cuatro canales para quedarse solo con
        // el alfa. Cuando las dos ecuaciones coinciden -- que es lo normal --
        // el resultado ya esta en blend_output y la llamada sobra entera.
        if (params.blend_equation_a != params.blend_equation_rgb) {
            blend_output.a() = EvaluateBlendEquation(combiner_output, srcfactor, dest, dstfactor,
                                                     params.blend_equation_a)
                                   .a();
        }
    } else {
        blend_output =
            Common::MakeVec(LogicOp(combiner_output.r(), dest.r(), output_merger.logic_op),
                            LogicOp(combiner_output.g(), dest.g(), output_merger.logic_op),
                            LogicOp(combiner_output.b(), dest.b(), output_merger.logic_op),
                            LogicOp(combiner_output.a(), dest.a(), output_merger.logic_op));
    }

    const Common::Vec4<u8> result = {
        output_merger.red_enable ? blend_output.r() : dest.r(),
        output_merger.green_enable ? blend_output.g() : dest.g(),
        output_merger.blue_enable ? blend_output.b() : dest.b(),
        output_merger.alpha_enable ? blend_output.a() : dest.a(),
    };

    return result;
}

namespace {

using TevStageConfig = Pica::TexturingRegs::TevStageConfig;
using TevSource = Pica::TexturingRegs::TevStageConfig::Source;
using TevOperation = Pica::TexturingRegs::TevStageConfig::Operation;

/// Cuantos operandos usa de verdad cada operacion. Los que sobran ni se cargan
/// de la tabla de fuentes ni pasan por su modificador.
u8 NumColorOperands(TevOperation op) {
    switch (op) {
    case TevOperation::Replace:
        return 1;
    case TevOperation::Lerp:
    case TevOperation::MultiplyThenAdd:
    case TevOperation::AddThenMultiply:
        return 3;
    default:
        return 2;
    }
}

u8 NumAlphaOperands(TevOperation op) {
    switch (op) {
    case TevOperation::Replace:
        return 1;
    case TevOperation::Lerp:
    case TevOperation::MultiplyThenAdd:
    case TevOperation::AddThenMultiply:
        return 3;
    default:
        return 2;
    }
}

/**
 * Cadena de COLOR del TEV sobre valores EMPAQUETADOS (RGBA en un u32).
 *
 * Antes, cada modificador devolvia un Vec3<u8> que el combinador volvia a
 * desenvolver canal a canal: tres operaciones escalares por modificador y
 * quince por combinador, con sus lecturas y escrituras de bytes. Con el valor
 * empaquetado, un modificador es un `vsub`/`vdup_lane` y un combinador
 * `Modulate` son cinco instrucciones NEON para los cuatro canales.
 *
 * La ALFA no aparece aqui: su cadena es de un solo byte por etapa y ya es
 * barata. Y las operaciones raras (Lerp, AddSigned, Dot3, ...) caen al camino
 * escalar de sw_texturing, que es exacto por construccion.
 */
#if defined(__ARM_NEON) && defined(__PSVITA__)

inline uint8x8_t LoadPacked(const Common::Vec4<u8>& value) {
    u32 packed;
    std::memcpy(&packed, value.AsArray(), sizeof(packed));
    return vreinterpret_u8_u32(vdup_n_u32(packed));
}

inline u32 StorePacked(uint8x8_t value) {
    u32 packed;
    vst1_lane_u32(&packed, vreinterpret_u32_u8(value), 0);
    return packed;
}

inline u32 PackRgb(Common::Vec3<u8> value) {
    return static_cast<u32>(value.r()) | (static_cast<u32>(value.g()) << 8) |
           (static_cast<u32>(value.b()) << 16);
}

inline Common::Vec3<u8> UnpackRgb(u32 packed) {
    return {static_cast<u8>(packed & 0xFF), static_cast<u8>((packed >> 8) & 0xFF),
            static_cast<u8>((packed >> 16) & 0xFF)};
}

/// Modificador de color de la PICA sobre RGBA empaquetado.
inline u32 ApplyColorModifierPacked(u8 modifier, const Common::Vec4<u8>& values) {
    using ColorModifier = Pica::TexturingRegs::TevStageConfig::ColorModifier;
    const uint8x8_t v = LoadPacked(values);

    switch (static_cast<ColorModifier>(modifier)) {
    case ColorModifier::SourceColor:
        return StorePacked(v);
    case ColorModifier::OneMinusSourceColor:
        return StorePacked(vsub_u8(vdup_n_u8(255), v));
    case ColorModifier::SourceAlpha:
        return StorePacked(vdup_lane_u8(v, 3));
    case ColorModifier::OneMinusSourceAlpha:
        return StorePacked(vsub_u8(vdup_n_u8(255), vdup_lane_u8(v, 3)));
    case ColorModifier::SourceRed:
        return StorePacked(vdup_lane_u8(v, 0));
    case ColorModifier::OneMinusSourceRed:
        return StorePacked(vsub_u8(vdup_n_u8(255), vdup_lane_u8(v, 0)));
    case ColorModifier::SourceGreen:
        return StorePacked(vdup_lane_u8(v, 1));
    case ColorModifier::OneMinusSourceGreen:
        return StorePacked(vsub_u8(vdup_n_u8(255), vdup_lane_u8(v, 1)));
    case ColorModifier::SourceBlue:
        return StorePacked(vdup_lane_u8(v, 2));
    case ColorModifier::OneMinusSourceBlue:
        return StorePacked(vsub_u8(vdup_n_u8(255), vdup_lane_u8(v, 2)));
    }
    // Modificador fuera de los diez validos: que decida el camino escalar, que
    // registra el error igual que antes.
    return PackRgb(GetColorModifier(static_cast<ColorModifier>(modifier), values));
}

/// Multiplicador de color (1, 2 o 4) sobre RGBA empaquetado: las sumas
/// saturadas hacen el `min(255, c * m)` gratis.
inline u32 ApplyColorMultiplierPacked(u32 packed, u8 multiplier) {
    if (multiplier == 1) {
        return packed;
    }
    uint8x8_t v = vreinterpret_u8_u32(vdup_n_u32(packed));
    v = vqadd_u8(v, v);
    if (multiplier == 4) {
        v = vqadd_u8(v, v);
    }
    return StorePacked(v);
}

/// Combinador de color para las operaciones comunes, sobre RGB empaquetado.
/// Devuelve true si la operacion esta cubierta; si no, el llamante usa el
/// camino escalar.
inline bool CombineColorPacked(u8 op, u32 a, u32 b, u32 c, u32& out) {
    using Operation = Pica::TexturingRegs::TevStageConfig::Operation;

    switch (static_cast<Operation>(op)) {
    case Operation::Replace:
        out = a;
        return true;
    case Operation::Modulate: {
        const uint16x8_t prod = vmull_u8(vreinterpret_u8_u32(vdup_n_u32(a)),
                                        vreinterpret_u8_u32(vdup_n_u32(b)));
        // Division exacta entre 255 en 16 bits: (x + 1 + (x >> 8)) >> 8, con
        // x <= 65025, asi que no desborda.
        const uint16x8_t shifted =
            vaddq_u16(vaddq_u16(prod, vdupq_n_u16(1)), vshrq_n_u16(prod, 8));
        out = StorePacked(vshrn_n_u16(shifted, 8));
        return true;
    }
    case Operation::Add:
        // vqadd satura en 255: es exactamente min(255, a + b).
        out = StorePacked(vqadd_u8(vreinterpret_u8_u32(vdup_n_u32(a)),
                                  vreinterpret_u8_u32(vdup_n_u32(b))));
        return true;
    case Operation::Subtract:
        // vqsub satura en 0: es exactamente max(0, a - b).
        out = StorePacked(vqsub_u8(vreinterpret_u8_u32(vdup_n_u32(a)),
                                  vreinterpret_u8_u32(vdup_n_u32(b))));
        return true;
    default:
        (void)c;
        return false;
    }
}

#else

inline u32 PackRgb(Common::Vec3<u8> value) {
    return static_cast<u32>(value.r()) | (static_cast<u32>(value.g()) << 8) |
           (static_cast<u32>(value.b()) << 16);
}

inline Common::Vec3<u8> UnpackRgb(u32 packed) {
    return {static_cast<u8>(packed & 0xFF), static_cast<u8>((packed >> 8) & 0xFF),
            static_cast<u8>((packed >> 16) & 0xFF)};
}

inline u32 ApplyColorModifierPacked(
    u8 modifier, const Common::Vec4<u8>& values) {
    using ColorModifier = Pica::TexturingRegs::TevStageConfig::ColorModifier;
    return PackRgb(GetColorModifier(static_cast<ColorModifier>(modifier), values));
}

inline u32 ApplyColorMultiplierPacked(u32 packed, u8 multiplier) {
    if (multiplier == 1) {
        return packed;
    }
    const auto value = UnpackRgb(packed);
    return PackRgb({static_cast<u8>(std::min<unsigned>(255, value.r() * multiplier)),
                    static_cast<u8>(std::min<unsigned>(255, value.g() * multiplier)),
                    static_cast<u8>(std::min<unsigned>(255, value.b() * multiplier))});
}

inline bool CombineColorPacked(u8 op, u32 a, u32 b, u32 c, u32& out) {
    using Operation = Pica::TexturingRegs::TevStageConfig::Operation;
    const std::array<Common::Vec3<u8>, 3> input = {UnpackRgb(a), UnpackRgb(b), UnpackRgb(c)};
    out = PackRgb(ColorCombine(static_cast<Operation>(op), input));
    return true;
}

#endif

} // Anonymous namespace

TevProgram RasterizerSoftware::CompileTevProgram(
    std::span<const Pica::TexturingRegs::TevStageConfig, 6> tev_stages,
    u32 active_tev_stages) const {
    TevProgram program;
    bool alpha_pure = true;
    u8 last_alpha_stage = 0;
    u8 last_active_stage = 0;

    const auto& buffer_color = regs.texturing.tev_combiner_buffer_color;
    program.buffer_color = Common::MakeVec(buffer_color.r.Value(), buffer_color.g.Value(),
                                           buffer_color.b.Value(), buffer_color.a.Value())
                               .Cast<u8>();

    for (u32 i = 0; i < tev_stages.size(); i++) {
        const auto& stage = tev_stages[i];
        auto& out = program.stages[i];

        out.active = (active_tev_stages & (1u << i)) != 0;
        if (out.active) {
            last_active_stage = static_cast<u8>(i);
        }

        out.color_op = static_cast<u8>(stage.color_op.Value());
        out.alpha_op = static_cast<u8>(stage.alpha_op.Value());

        out.color_modifier[0] = static_cast<u8>(stage.color_modifier1.Value());
        out.color_modifier[1] = static_cast<u8>(stage.color_modifier2.Value());
        out.color_modifier[2] = static_cast<u8>(stage.color_modifier3.Value());
        out.alpha_modifier[0] = static_cast<u8>(stage.alpha_modifier1.Value());
        out.alpha_modifier[1] = static_cast<u8>(stage.alpha_modifier2.Value());
        out.alpha_modifier[2] = static_cast<u8>(stage.alpha_modifier3.Value());

        // Remapeo de la etapa 0: ahi 'Previous' no es la salida de la etapa
        // anterior -- no la hay -- sino color_source3. El bucle original lo
        // comprobaba por operando y por pixel.
        const auto remap_color_source = [&](TevSource source) -> u8 {
            if (i == 0 && source == TevSource::Previous) {
                return static_cast<u8>(stage.color_source3.Value());
            }
            return static_cast<u8>(source);
        };
        out.color_source[0] = remap_color_source(stage.color_source1);
        out.color_source[1] = remap_color_source(stage.color_source2);
        out.color_source[2] = static_cast<u8>(stage.color_source3.Value());
        out.alpha_source[0] = static_cast<u8>(stage.alpha_source1.Value());
        out.alpha_source[1] = static_cast<u8>(stage.alpha_source2.Value());
        out.alpha_source[2] = static_cast<u8>(stage.alpha_source3.Value());

        out.color_sources = NumColorOperands(stage.color_op);
        out.alpha_sources = NumAlphaOperands(stage.alpha_op);
        out.color_multiplier = static_cast<u8>(stage.GetColorMultiplier());
        out.alpha_multiplier = static_cast<u8>(stage.GetAlphaMultiplier());

        // Dot3_RGBA: el resultado del combinador de color va tambien al alfa.
        out.alpha_from_color = stage.color_op == TevOperation::Dot3_RGBA;

        // Etapa que deja el color (o el alfa) exactamente como estaba. Las
        // condiciones son las mismas que las de ComputeActiveTevStages, pero
        // aplicadas por separado: una etapa puede tener el color activo y el
        // alfa de paso, y ahi se ahorra todo el combinador de alfa.
        //
        // OJO en la etapa 0: ahi 'Previous' esta remapeado a color_source3 (ver
        // arriba), asi que "Replace Previous" NO copia la salida anterior -- no
        // la hay -- sino color_source3, y saltarse el calculo cambiaria el
        // pixel. ComputeActiveTevStages ya fuerza la etapa 0 activa por este
        // mismo motivo; aqui la condicion de paso tiene que excluirla tambien.
        out.color_passthrough =
            i != 0 && stage.color_op == TevOperation::Replace &&
            stage.color_source1 == TevSource::Previous &&
            stage.color_modifier1 == TevStageConfig::ColorModifier::SourceColor &&
            out.color_multiplier == 1;
        out.alpha_passthrough =
            !out.alpha_from_color && stage.alpha_op == TevOperation::Replace &&
            stage.alpha_source1 == TevSource::Previous &&
            stage.alpha_modifier1 == TevStageConfig::AlphaModifier::SourceAlpha &&
            out.alpha_multiplier == 1;

        // Fuentes variables que la etapa lee de verdad, y que recursos globales
        // (unidades de textura, color del vertice) usa el programa entero. Solo
        // cuentan las etapas ACTIVAS: una etapa inerte no lee nada.
        const auto note_sources = [&out, &program](const u8* sources, u8 count) {
            for (u8 j = 0; j < count; j++) {
                const u8 source = sources[j];
                out.uses_previous |= source == static_cast<u8>(TevSource::Previous);
                out.uses_previous_buffer |= source == static_cast<u8>(TevSource::PreviousBuffer);
                out.uses_constant |= source == static_cast<u8>(TevSource::Constant);
                if (source == static_cast<u8>(TevSource::PrimaryColor)) {
                    program.uses_primary_color = true;
                }
                // Texture0..Texture3 son valores consecutivos del enum.
                if (source >= static_cast<u8>(TevSource::Texture0) &&
                    source <= static_cast<u8>(TevSource::Texture3)) {
                    program.uses_texture[source - static_cast<u8>(TevSource::Texture0)] = true;
                }
            }
        };
        if (out.active) {
            note_sources(out.color_source, out.color_sources);
            if (!out.alpha_from_color) {
                note_sources(out.alpha_source, out.alpha_sources);
            }

            // Hasta que etapa hay que llegar para calcular el alfa (ver
            // TevProgram::last_alpha_stage).
            if (!out.alpha_passthrough) {
                last_alpha_stage = i;
            }

            // Dependencias de la cadena de alfa para el calculo adelantado
            // (ver TevProgram::alpha_pure).
            if (out.alpha_from_color) {
                alpha_pure = false;
            } else {
                for (u8 j = 0; j < out.alpha_sources; j++) {
                    const u8 source = out.alpha_source[j];
                    const u8 modifier = out.alpha_modifier[j];
                    const bool from_previous =
                        source == static_cast<u8>(TevSource::Previous) ||
                        source == static_cast<u8>(TevSource::PreviousBuffer);
                    const bool modifier_reads_alpha_only =
                        modifier ==
                            static_cast<u8>(TevStageConfig::AlphaModifier::SourceAlpha) ||
                        modifier ==
                            static_cast<u8>(TevStageConfig::AlphaModifier::OneMinusSourceAlpha);
                    if (from_previous && !modifier_reads_alpha_only) {
                        alpha_pure = false;
                    }
                }
            }
        }

        out.updates_buffer_color =
            regs.texturing.tev_combiner_buffer_input.TevStageUpdatesCombinerBufferColor(i);
        out.updates_buffer_alpha =
            regs.texturing.tev_combiner_buffer_input.TevStageUpdatesCombinerBufferAlpha(i);

        out.constant = Common::MakeVec(stage.const_r.Value(), stage.const_g.Value(),
                                       stage.const_b.Value(), stage.const_a.Value())
                           .Cast<u8>();
    }

    program.alpha_pure = alpha_pure;
    program.last_alpha_stage = last_alpha_stage;
    program.last_active_stage = last_active_stage;

    // La iluminacion por fragmento lee texture_color[] por su cuenta (los
    // selectores de sombra y bump). Si esta activa, esas unidades tambien hay
    // que muestrearlas aunque el TEV no las nombre.
    if (regs.lighting.disable == 0) {
        if (regs.lighting.config0.enable_shadow) {
            program.uses_texture[regs.lighting.config0.shadow_selector.Value() & 3u] = true;
        }
        if (regs.lighting.config0.bump_mode != Pica::LightingRegs::LightingBumpMode::None) {
            program.uses_texture[regs.lighting.config0.bump_selector.Value() & 3u] = true;
        }
    }

    return program;
}

std::array<u8, 6> RasterizerSoftware::ComputeTevAlpha(
    std::span<const Common::Vec4<u8>, 4> texture_color, const TevProgram& program,
    Common::Vec4<u8> primary_color, Common::Vec4<u8> primary_fragment_color,
    Common::Vec4<u8> secondary_fragment_color) const {
    std::array<Common::Vec4<u8>, 16> source_table{};
    source_table[static_cast<u32>(TevSource::PrimaryColor)] = primary_color;
    source_table[static_cast<u32>(TevSource::PrimaryFragmentColor)] = primary_fragment_color;
    source_table[static_cast<u32>(TevSource::SecondaryFragmentColor)] = secondary_fragment_color;
    source_table[static_cast<u32>(TevSource::Texture0)] = texture_color[0];
    source_table[static_cast<u32>(TevSource::Texture1)] = texture_color[1];
    source_table[static_cast<u32>(TevSource::Texture2)] = texture_color[2];
    source_table[static_cast<u32>(TevSource::Texture3)] = texture_color[3];

    std::array<u8, 6> stage_alpha{};
    Common::Vec4<u8> combiner_output = {0, 0, 0, 0};
    Common::Vec4<u8> combiner_buffer = {0, 0, 0, 0};
    Common::Vec4<u8> next_combiner_buffer = program.buffer_color;

    // Solo hasta la ultima etapa que escribe alfa (ver last_alpha_stage): las
    // siguientes no la cambian, asi que no hay que recorrerlas.
    const u32 stages_to_run = static_cast<u32>(program.last_alpha_stage) + 1;
    for (u32 i = 0; i < stages_to_run; i++) {
        const auto& stage = program.stages[i];

        if (stage.active) {
            // Fuentes variables que la etapa pueda leer. De las de 'Previous' y
            // 'PreviousBuffer' la cadena de alfa solo usa el canal alfa
            // (garantizado por TevProgram::alpha_pure), y combiner_output[3] lo
            // lleva actualizado etapa a etapa.
            if (stage.uses_previous_buffer) {
                source_table[static_cast<u32>(TevSource::PreviousBuffer)] = combiner_buffer;
            }
            if (stage.uses_previous) {
                source_table[static_cast<u32>(TevSource::Previous)] = combiner_output;
            }
            if (stage.uses_constant) {
                source_table[static_cast<u32>(TevSource::Constant)] = stage.constant;
            }

            if (!stage.alpha_passthrough) {
                std::array<u8, 3> alpha_result{};
                alpha_result[0] = GetAlphaModifier(
                    static_cast<TevStageConfig::AlphaModifier>(stage.alpha_modifier[0]),
                    source_table[stage.alpha_source[0]]);
                if (stage.alpha_sources > 1) {
                    alpha_result[1] = GetAlphaModifier(
                        static_cast<TevStageConfig::AlphaModifier>(stage.alpha_modifier[1]),
                        source_table[stage.alpha_source[1]]);
                }
                if (stage.alpha_sources > 2) {
                    alpha_result[2] = GetAlphaModifier(
                        static_cast<TevStageConfig::AlphaModifier>(stage.alpha_modifier[2]),
                        source_table[stage.alpha_source[2]]);
                }
                const u8 alpha_output =
                    AlphaCombine(static_cast<TevOperation>(stage.alpha_op), alpha_result);
                combiner_output[3] =
                    std::min<unsigned>(255, alpha_output * stage.alpha_multiplier);
            }
        }

        combiner_buffer = next_combiner_buffer;
        if (stage.updates_buffer_color) {
            next_combiner_buffer.r() = combiner_output.r();
            next_combiner_buffer.g() = combiner_output.g();
            next_combiner_buffer.b() = combiner_output.b();
        }
        if (stage.updates_buffer_alpha) {
            next_combiner_buffer.a() = combiner_output.a();
        }
        stage_alpha[i] = combiner_output.a();
    }

    // Las etapas restantes no escriben alfa: el valor se queda donde quedo.
    for (u32 i = stages_to_run; i < stage_alpha.size(); i++) {
        stage_alpha[i] = combiner_output.a();
    }

    return stage_alpha;
}

Common::Vec4<u8> RasterizerSoftware::WriteTevConfig(
    std::span<const Common::Vec4<u8>, 4> texture_color, const TevProgram& program,
    Common::Vec4<u8> primary_color, Common::Vec4<u8> primary_fragment_color,
    Common::Vec4<u8> secondary_fragment_color, const std::array<u8, 6>* stage_alpha) {
    /**
     * Texture environment - consists of 6 stages of color and alpha combining.
     * Color combiners take three input color values from some source (e.g. interpolated
     * vertex color, texture color, previous stage, etc), perform some very simple
     * operations on each of them (e.g. inversion) and then calculate the output color
     * with some basic arithmetic. Alpha combiners can be configured separately but work
     * analogously.
     *
     * La configuracion de cada etapa -- operaciones, modificadores, fuentes,
     * multiplicadores y constantes -- llega ya resuelta en 'program'. Aqui solo
     * queda ejecutarla. Ver TevProgram.
     **/
    Common::Vec4<u8> combiner_output = {0, 0, 0, 0};
    Common::Vec4<u8> combiner_buffer = {0, 0, 0, 0};
    Common::Vec4<u8> next_combiner_buffer = program.buffer_color;

    /**
     * Tabla de fuentes, indexada por el propio valor del enum.
     *
     * De las diez fuentes, siete son fijas durante todo el pixel: el color
     * primario, los dos de iluminacion y las cuatro texturas. Solo cambian de
     * etapa a etapa 'Previous' (la salida anterior), 'PreviousBuffer' y la
     * constante. Se rellenan las siete fijas una vez por pixel, y las variables
     * solo cuando la etapa las usa (uses_previous/...).
     *
     * Los indices llegan ya remapeados y en 0-15 desde CompileTevProgram, asi
     * que una tabla de 16 entradas lo cubre entero sin comprobar rangos.
     */
    std::array<Common::Vec4<u8>, 16> source_table{};
    source_table[static_cast<u32>(TevSource::PrimaryColor)] = primary_color;
    source_table[static_cast<u32>(TevSource::PrimaryFragmentColor)] = primary_fragment_color;
    source_table[static_cast<u32>(TevSource::SecondaryFragmentColor)] = secondary_fragment_color;
    source_table[static_cast<u32>(TevSource::Texture0)] = texture_color[0];
    source_table[static_cast<u32>(TevSource::Texture1)] = texture_color[1];
    source_table[static_cast<u32>(TevSource::Texture2)] = texture_color[2];
    source_table[static_cast<u32>(TevSource::Texture3)] = texture_color[3];

    // Solo hasta la ultima etapa activa: las siguientes no calculan nada y su
    // rotacion del buffer no la lee nadie (ver TevProgram::last_active_stage).
    const u32 stages_to_run = static_cast<u32>(program.last_active_stage) + 1;
    for (u32 i = 0; i < stages_to_run; i++) {
        const auto& stage = program.stages[i];

        if (stage.active) [[likely]] {
            // Fuentes variables: solo las que esta etapa lee de verdad.
            if (stage.uses_previous_buffer) {
                source_table[static_cast<u32>(TevSource::PreviousBuffer)] = combiner_buffer;
            }
            if (stage.uses_previous) {
                source_table[static_cast<u32>(TevSource::Previous)] = combiner_output;
            }
            if (stage.uses_constant) {
                source_table[static_cast<u32>(TevSource::Constant)] = stage.constant;
            }

            u32 color_output = 0;
            if (!stage.color_passthrough) {
                // Cadena de color empaquetada (RGBA en un u32): ver las notas de
                // ApplyColorModifierPacked/CombineColorPacked. Las operaciones no
                // cubiertas caen al camino escalar exacto.
                const u32 color0 = ApplyColorModifierPacked(
                    stage.color_modifier[0], source_table[stage.color_source[0]]);
                u32 color1 = 0;
                u32 color2 = 0;
                if (stage.color_sources > 1) {
                    color1 = ApplyColorModifierPacked(stage.color_modifier[1],
                                                      source_table[stage.color_source[1]]);
                }
                if (stage.color_sources > 2) {
                    color2 = ApplyColorModifierPacked(stage.color_modifier[2],
                                                      source_table[stage.color_source[2]]);
                }

                /**
                 * NOTE: Not sure if the alpha combiner might use the color
                 * output of the previous stage as input. Hence, we currently
                 * don't directly write the result to combiner_output.rgb(), but
                 * instead store it in a temporary variable until alpha
                 * combining has been done.  (El combinador de alfa lee de la
                 * tabla, que ya tiene copiada la salida anterior, asi que el
                 * orden se conserva igual.)
                 **/
                if (!CombineColorPacked(stage.color_op, color0, color1, color2, color_output)) {
                    const std::array<Common::Vec3<u8>, 3> color_result = {
                        UnpackRgb(color0), UnpackRgb(color1), UnpackRgb(color2)};
                    color_output = PackRgb(
                        ColorCombine(static_cast<TevOperation>(stage.color_op), color_result));
                }
                color_output = ApplyColorMultiplierPacked(color_output, stage.color_multiplier);
                combiner_output[0] = static_cast<u8>(color_output & 0xFF);
                combiner_output[1] = static_cast<u8>((color_output >> 8) & 0xFF);
                combiner_output[2] = static_cast<u8>((color_output >> 16) & 0xFF);
            }

            if (stage_alpha != nullptr) {
                // El alfa ya viene calculado de ComputeTevAlpha. Se copia a
                // combiner_output para que los modificadores de color que lean
                // el canal alfa de 'Previous'/'PreviousBuffer' vean exactamente
                // lo mismo que verian en la pasada unica.
                combiner_output[3] = (*stage_alpha)[i];
            } else if (stage.alpha_from_color) {
                // El resultado del Dot3_RGBA (el rojo del color combinado) va
                // tambien al alfa. En ese caso la operacion no es Replace, asi
                // que color_output ya esta calculado.
                combiner_output[3] = static_cast<u8>(std::min<unsigned>(
                    255, static_cast<unsigned>(color_output & 0xFF) * stage.alpha_multiplier));
            } else if (!stage.alpha_passthrough) {
                std::array<u8, 3> alpha_result{};
                alpha_result[0] = GetAlphaModifier(
                    static_cast<TevStageConfig::AlphaModifier>(stage.alpha_modifier[0]),
                    source_table[stage.alpha_source[0]]);
                if (stage.alpha_sources > 1) {
                    alpha_result[1] = GetAlphaModifier(
                        static_cast<TevStageConfig::AlphaModifier>(stage.alpha_modifier[1]),
                        source_table[stage.alpha_source[1]]);
                }
                if (stage.alpha_sources > 2) {
                    alpha_result[2] = GetAlphaModifier(
                        static_cast<TevStageConfig::AlphaModifier>(stage.alpha_modifier[2]),
                        source_table[stage.alpha_source[2]]);
                }
                const u8 alpha_output =
                    AlphaCombine(static_cast<TevOperation>(stage.alpha_op), alpha_result);
                combiner_output[3] =
                    std::min<unsigned>(255, alpha_output * stage.alpha_multiplier);
            }
        }

        // La rotacion del buffer del combinador corre en TODAS las etapas,
        // tambien en las inertes: es contabilidad que las siguientes necesitan
        // (ver el comentario de TevProgram). Los flags vienen precalculados.
        combiner_buffer = next_combiner_buffer;

        if (stage.updates_buffer_color) {
            next_combiner_buffer.r() = combiner_output.r();
            next_combiner_buffer.g() = combiner_output.g();
            next_combiner_buffer.b() = combiner_output.b();
        }

        if (stage.updates_buffer_alpha) {
            next_combiner_buffer.a() = combiner_output.a();
        }
    }

    return combiner_output;
}

void RasterizerSoftware::WriteFog(float depth, Common::Vec4<u8>& combiner_output) const {
    /**
     * Apply fog combiner. Not fully accurate. We'd have to know what data type is used to
     * store the depth etc. Using float for now until we know more about Pica datatypes.
     **/
    if (regs.texturing.fog_mode == TexturingRegs::FogMode::Fog) {
        const Common::Vec3<u8> fog_color =
            Common::MakeVec(regs.texturing.fog_color.r.Value(), regs.texturing.fog_color.g.Value(),
                            regs.texturing.fog_color.b.Value())
                .Cast<u8>();

        float fog_index;
        if (regs.texturing.fog_flip) {
            fog_index = (1.0f - depth) * 128.0f;
        } else {
            fog_index = depth * 128.0f;
        }

        // Generate clamped fog factor from LUT for given fog index
        const f32 fog_i = std::clamp(floorf(fog_index), 0.0f, 127.0f);
        const f32 fog_f = fog_index - fog_i;
        const auto& fog_lut_entry = pica.fog.lut[static_cast<u32>(fog_i)];
        f32 fog_factor = fog_lut_entry.ToFloat() + fog_lut_entry.DiffToFloat() * fog_f;
        fog_factor = std::clamp(fog_factor, 0.0f, 1.0f);
        for (u32 i = 0; i < 3; i++) {
            combiner_output[i] = static_cast<u8>(fog_factor * combiner_output[i] +
                                                 (1.0f - fog_factor) * fog_color[i]);
        }
    }
}

bool RasterizerSoftware::DoAlphaTest(u8 alpha) const {
    const auto& output_merger = regs.framebuffer.output_merger;
    if (!output_merger.alpha_test.enable) {
        return true;
    }
    switch (output_merger.alpha_test.func) {
    case FramebufferRegs::CompareFunc::Never:
        return false;
    case FramebufferRegs::CompareFunc::Always:
        return true;
    case FramebufferRegs::CompareFunc::Equal:
        return alpha == output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::NotEqual:
        return alpha != output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::LessThan:
        return alpha < output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::LessThanOrEqual:
        return alpha <= output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::GreaterThan:
        return alpha > output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::GreaterThanOrEqual:
        return alpha >= output_merger.alpha_test.ref;
    default:
        LOG_CRITICAL(Render_Software, "Unknown alpha test condition {}",
                     output_merger.alpha_test.func.Value());
        return false;
    }
}

namespace {

/// Comparacion de la PICA, en un solo sitio: 'lhs' es el valor que llega (la
/// profundidad calculada o la referencia de galga) y 'rhs' el que ya estaba en
/// el buffer. Las dos pruebas (galga y profundidad) compartian ocho ramas cada
/// una, escritas dos veces.
bool CompareFuncPasses(Pica::FramebufferRegs::CompareFunc func, u32 lhs, u32 rhs) {
    using CompareFunc = Pica::FramebufferRegs::CompareFunc;
    switch (func) {
    case CompareFunc::Never:
        return false;
    case CompareFunc::Always:
        return true;
    case CompareFunc::Equal:
        return lhs == rhs;
    case CompareFunc::NotEqual:
        return lhs != rhs;
    case CompareFunc::LessThan:
        return lhs < rhs;
    case CompareFunc::LessThanOrEqual:
        return lhs <= rhs;
    case CompareFunc::GreaterThan:
        return lhs > rhs;
    case CompareFunc::GreaterThanOrEqual:
        return lhs >= rhs;
    }
    return false;
}

} // Anonymous namespace

bool RasterizerSoftware::EarlyDepthPasses(u16 x, u16 y, float depth, u32& ref_z, u8& stencil) const {
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& output_merger = regs.framebuffer.output_merger;

    // La galga va primero, en el mismo orden que DoDepthStencilTest. Solo se
    // llega aqui cuando sus acciones de fallo no escriben (Keep), asi que
    // rechazar el pixel sin ejecutarlas es exactamente lo que haria el camino
    // normal.
    if (output_merger.stencil_test.enable &&
        framebuffer.depth_format == FramebufferRegs::DepthFormat::D24S8) {
        stencil = fb.GetStencil(x >> 4, y >> 4);
        const u8 dest = stencil & output_merger.stencil_test.input_mask;
        const u8 ref =
            output_merger.stencil_test.reference_value & output_merger.stencil_test.input_mask;
        if (!CompareFuncPasses(output_merger.stencil_test.func, ref, dest)) {
            return false;
        }
    }

    // La escala de cuantizacion de la profundidad es fija para el framebuffer
    // (formato resuelto en Bind), asi que se lee de la cache en vez de
    // recomputar el numero de bits y el desplazamiento por pixel.
    const u32 z = static_cast<u32>(depth * fb.DepthScale());
    ref_z = fb.GetDepth(x >> 4, y >> 4);
    return CompareFuncPasses(output_merger.depth_test_func, z, ref_z);
}

bool RasterizerSoftware::DoDepthStencilTest(u16 x, u16 y, float depth, const u32* known_ref_z,
                                            const u8* known_stencil) const {
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto stencil_test = regs.framebuffer.output_merger.stencil_test;
    u8 old_stencil = 0;

    const auto update_stencil = [&](Pica::FramebufferRegs::StencilAction action) {
        const u8 new_stencil =
            PerformStencilAction(action, old_stencil, stencil_test.reference_value);
        if (framebuffer.allow_depth_stencil_write != 0) {
            const u8 stencil =
                (new_stencil & stencil_test.write_mask) | (old_stencil & ~stencil_test.write_mask);
            fb.SetStencil(x >> 4, y >> 4, stencil);
        }
    };

    const bool stencil_action_enable =
        regs.framebuffer.output_merger.stencil_test.enable &&
        regs.framebuffer.framebuffer.depth_format == FramebufferRegs::DepthFormat::D24S8;

    if (stencil_action_enable) {
        // Si el rechazo temprano ya leyo la galga, se reutiliza.
        old_stencil = known_stencil != nullptr ? *known_stencil : fb.GetStencil(x >> 4, y >> 4);
        const u8 dest = old_stencil & stencil_test.input_mask;
        const u8 ref = stencil_test.reference_value & stencil_test.input_mask;
        if (!CompareFuncPasses(stencil_test.func, ref, dest)) {
            update_stencil(stencil_test.action_stencil_fail);
            return false;
        }
    }

    const u32 z = static_cast<u32>(depth * fb.DepthScale());

    const auto& output_merger = regs.framebuffer.output_merger;
    if (output_merger.depth_test_enable) {
        // Si el rechazo temprano ya leyo esta misma profundidad, se reutiliza.
        const u32 ref_z = known_ref_z != nullptr ? *known_ref_z : fb.GetDepth(x >> 4, y >> 4);
        if (!CompareFuncPasses(output_merger.depth_test_func, z, ref_z)) {
            if (stencil_action_enable) {
                update_stencil(stencil_test.action_depth_fail);
            }
            return false;
        }
    }
    if (framebuffer.allow_depth_stencil_write != 0 && output_merger.depth_write_enable) {
        fb.SetDepth(x >> 4, y >> 4, z);
    }
    // The stencil depth_pass action is executed even if depth testing is disabled
    if (stencil_action_enable) {
        update_stencil(stencil_test.action_depth_pass);
    }

    return true;
}

} // namespace SwRenderer
