// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/rasterizer_gxm.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <fmt/format.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/shacccg.h>
#include <vita2d.h>
#include "common/logging/log.h"
#include "common/vita_diag.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica/regs_lighting.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/pica_types.h"
#include "video_core/utils.h"
#include "video_core/renderer_gxm/gxm_cg.h"
#include "video_core/renderer_gxm/gxm_presenter.h"
#include "video_core/renderer_gxm/gxm_texture_cache.h"
#include "video_core/renderer_software/sw_rasterizer.h"
#include "video_core/shader/generator/cg_fs_shader_gen.h"
#include "video_core/shader/generator/cg_vs_shader_gen.h"
#include "video_core/shader/generator/shader_gen.h"
#include "common/alignment.h"
#include "common/hash.h"

namespace Gxm {

std::atomic<u32> RasterizerGXM::gpu_triangles{0};
std::atomic<u32> RasterizerGXM::software_triangles{0};
std::atomic<u32> RasterizerGXM::gpu_batches{0};
std::atomic<u32> RasterizerGXM::gpu_scenes{0};
std::atomic<u32> RasterizerGXM::gpu_writebacks{0};
std::atomic<u32> RasterizerGXM::scene_close_full{0};
std::atomic<u32> RasterizerGXM::scene_close_lut{0};
std::atomic<u32> RasterizerGXM::hw_vs_batches{0};
std::atomic<u32> RasterizerGXM::hw_vs_rejects{0};
std::atomic<const char*> RasterizerGXM::hw_vs_last_reject{"-"};
std::array<RasterizerGXM::RejectCount, 12> RasterizerGXM::reject_counts{};
std::atomic<u32> RasterizerGXM::Ablation::mode{0};
std::atomic<u32> RasterizerGXM::no_finish_wait{1};
std::atomic<u32> RasterizerGXM::present_direct{1};
std::atomic<u32> RasterizerGXM::specialize_vs{1};
std::atomic<u32> RasterizerGXM::async_vs{1};
std::atomic<u32> RasterizerGXM::transfer_on_gpu{1};
std::atomic<u32> RasterizerGXM::resolution_scale{2};
std::atomic<u32> RasterizerGXM::gpu_transfers{0};
std::atomic<u32> RasterizerGXM::transfer_materialized{0};
std::atomic<u32> RasterizerGXM::gpu_fills{0};
std::atomic<u32> RasterizerGXM::software_syncs{0};
std::atomic<u32> RasterizerGXM::skipped_batches{0};
RasterizerGXM* RasterizerGXM::s_instance = nullptr;

const char* RasterizerGXM::Ablation::Name(u32 value) {
    switch (value) {
    case kNoLighting:
        return "sin luz";
    case kNoScissor:
        return "sin scissor";
    case kNoStencil:
        return "sin plantilla";
    case kNoTexture:
        return "sin tex";
    case kNoGpu:
        return "sin gpu";
    case kCpuVertexShader:
        return "vs en cpu";
    default:
        return "normal";
    }
}

namespace {

using Pica::FramebufferRegs;
using Pica::RasterizerRegs;

constexpr const char* kShacccgPaths[] = {
    "ur0:/data/libshacccg.suprx",
    "ux0:/data/libshacccg.suprx",
};

/// Modulo del compilador, una vez por proceso.
SceUID g_module = -1;
SceShaccCgSourceFile g_source{};
SceShaccCgCallbackList g_callbacks{};

SceShaccCgSourceFile* OpenSource(const char*, const SceShaccCgSourceLocation*,
                                 const SceShaccCgCompileOptions*, const char**) {
    return &g_source;
}

void* CgAlloc(unsigned int size) {
    return std::malloc(size);
}

void CgFree(void* pointer) {
    std::free(pointer);
}

bool EnsureCgModule() {
    // Delegado en el cargador compartido (renderer_gxm/gxm_cg.*): el modulo es
    // estado global del proceso y aqui, por separado, el segundo intento fallaba
    // y se reintentaba en cada lote (450 ms -> 13.734 ms de fotograma).
    return EnsureCgReady();
}

// Las llamadas a CompileCg de este fichero van directas a la compartida
// (Gxm::CompileCg, gxm_cg.h): tener aqui una copia hacia ambigua la llamada.

/**
 * Shader de vertices fijo: los vertices llegan del interprete de la PICA ya en
 * coordenadas de recorte, asi que aqui solo se colocan y se pasan al shader de
 * fragmentos. Cuando exista el descompilador de bytecode de la PICA este
 * programa lo generara cg_vs_shader_gen; hoy el trabajo de vertices lo hace el
 * camino de software, que es lo unico que existe.
 */
constexpr const char kVertexSource[] = R"(
void main(float4 position : POSITION,
          float4 color : COLOR0,
          float2 tc0 : TEXCOORD0,
          float2 tc1 : TEXCOORD1,
          float2 tc2 : TEXCOORD2,
          out float4 gl_Position : POSITION,
          out float4 out_color : COLOR0,
          out float2 out_tc0 : TEXCOORD0,
          out float2 out_tc1 : TEXCOORD1,
          out float2 out_tc2 : TEXCOORD2)
{
    gl_Position = position;
    out_color = color;
    out_tc0 = tc0;
    out_tc1 = tc1;
    out_tc2 = tc2;
}
)";

/**
 * El mismo shader de vertices, pero llevando ademas lo que pide la
 * ILUMINACION POR FRAGMENTO: el cuaternion de normal y el vector de vista.
 *
 * POR QUE DOS PROGRAMAS Y NO UNO SOLO CON SIETE FLOATS DE MAS. Un varying no se
 * paga una vez por vertice: se INTERPOLA por fragmento, y eso son ciclos de la
 * USSE en cada pixel dibujado. La inmensa mayoria de los lotes de un juego no
 * tienen iluminacion por fragmento (interfaz, sprites, fondos), y hacerles
 * pagar la interpolacion de siete floats que su shader ni mira es justo el tipo
 * de coste que no se ve en ningun sitio y se lleva el fotograma. Ademas son
 * siete conversiones de f24 a f32 mas por vertice en el bucle de empaquetado.
 *
 * El precio de tenerlos separados es un segundo programa compilado al arrancar
 * y un flag en la entrada del cache de pipelines. La configuracion de
 * fragmentos ya dice cual toca (config.lighting.enable), asi que la eleccion es
 * determinista y no hace falta llevarla por otro lado.
 *
 * LOS ATRIBUTOS LLEGAN SIN DIVIDIR POR W, que es lo que hace falta. La division
 * perspectiva de la PICA la hace MakeScreenCoords, y eso pasa DENTRO del
 * rasterizador de software; a AddTriangle llegan los vertices tal cual salen
 * del pipeline de geometria. Asi que aqui se entregan crudos y la
 * interpolacion con correccion de perspectiva la hace GXM, que es lo que hacen
 * los varyings de TEXCOORD por defecto.
 */
constexpr const char kVertexSourceLit[] = R"(
void main(float4 position : POSITION,
          float4 color : COLOR0,
          float2 tc0 : TEXCOORD0,
          float2 tc1 : TEXCOORD1,
          float2 tc2 : TEXCOORD2,
          float4 normquat : TEXCOORD3,
          float3 view : TEXCOORD4,
          out float4 gl_Position : POSITION,
          out float4 out_color : COLOR0,
          out float2 out_tc0 : TEXCOORD0,
          out float2 out_tc1 : TEXCOORD1,
          out float2 out_tc2 : TEXCOORD2,
          out float4 out_normquat : TEXCOORD3,
          out float3 out_view : TEXCOORD4)
{
    gl_Position = position;
    out_color = color;
    out_tc0 = tc0;
    out_tc1 = tc1;
    out_tc2 = tc2;
    out_normquat = normquat;
    out_view = view;
}
)";

/**
 * Las mismas dos variantes, llevando ademas la W de la coordenada de textura 0
 * (0.1.0.44), para las texturas PROYECTADAS (Projection2D): la PICA divide u y
 * v entre esa w antes de muestrear (ver TextureColor en sw_rasterizer.cpp), y
 * hasta ahora el lote entero se iba a software ("tipo de textura 0"). En la
 * cinematica de Rubi Omega eran ~2.300 triangulos por vblank.
 *
 * Son variantes APARTE, y no un float mas en las de siempre, para que los lotes
 * que ya funcionaban no cambien en nada: ni su formato de vertice, ni sus
 * varyings, ni su programa.
 */
constexpr const char kVertexSourceProj[] = R"(
void main(float4 position : POSITION,
          float4 color : COLOR0,
          float2 tc0 : TEXCOORD0,
          float2 tc1 : TEXCOORD1,
          float2 tc2 : TEXCOORD2,
          float tc0w : TEXCOORD5,
          out float4 gl_Position : POSITION,
          out float4 out_color : COLOR0,
          out float2 out_tc0 : TEXCOORD0,
          out float2 out_tc1 : TEXCOORD1,
          out float2 out_tc2 : TEXCOORD2,
          out float out_tc0w : TEXCOORD5)
{
    gl_Position = position;
    out_color = color;
    out_tc0 = tc0;
    out_tc1 = tc1;
    out_tc2 = tc2;
    out_tc0w = tc0w;
}
)";

constexpr const char kVertexSourceLitProj[] = R"(
void main(float4 position : POSITION,
          float4 color : COLOR0,
          float2 tc0 : TEXCOORD0,
          float2 tc1 : TEXCOORD1,
          float2 tc2 : TEXCOORD2,
          float4 normquat : TEXCOORD3,
          float3 view : TEXCOORD4,
          float tc0w : TEXCOORD5,
          out float4 gl_Position : POSITION,
          out float4 out_color : COLOR0,
          out float2 out_tc0 : TEXCOORD0,
          out float2 out_tc1 : TEXCOORD1,
          out float2 out_tc2 : TEXCOORD2,
          out float4 out_normquat : TEXCOORD3,
          out float3 out_view : TEXCOORD4,
          out float out_tc0w : TEXCOORD5)
{
    gl_Position = position;
    out_color = color;
    out_tc0 = tc0;
    out_tc1 = tc1;
    out_tc2 = tc2;
    out_normquat = normquat;
    out_view = view;
    out_tc0w = tc0w;
}
)";

constexpr u32 kVertexStride = 14 * sizeof(float);
/// El de arriba mas el cuaternion de normal (4) y el vector de vista (3).
constexpr u32 kVertexStrideLit = 21 * sizeof(float);

/// Tamano de vertice de cada variante: la W de la textura 0, si va, al final.
constexpr u32 VertexStride(bool lit, bool proj) {
    return (lit ? kVertexStrideLit : kVertexStride) + (proj ? sizeof(float) : 0);
}
constexpr u32 kMaxVerticesPerDraw = 60000;
/// Cuantas superficies de dibujado se mantienen a la vez. Cada una se lleva su
/// color y su profundidad en CDRAM y no los suelta: sin tope, un juego que
/// alterne framebuffers se come la memoria del chip grafico.
constexpr std::size_t kMaxSurfaces = 8;

SceGxmDepthFunc MapDepthFunc(FramebufferRegs::CompareFunc func) {
    switch (func) {
    case FramebufferRegs::CompareFunc::Never:
        return SCE_GXM_DEPTH_FUNC_NEVER;
    case FramebufferRegs::CompareFunc::Always:
        return SCE_GXM_DEPTH_FUNC_ALWAYS;
    case FramebufferRegs::CompareFunc::Equal:
        return SCE_GXM_DEPTH_FUNC_EQUAL;
    case FramebufferRegs::CompareFunc::NotEqual:
        return SCE_GXM_DEPTH_FUNC_NOT_EQUAL;
    case FramebufferRegs::CompareFunc::LessThan:
        return SCE_GXM_DEPTH_FUNC_LESS;
    case FramebufferRegs::CompareFunc::LessThanOrEqual:
        return SCE_GXM_DEPTH_FUNC_LESS_EQUAL;
    case FramebufferRegs::CompareFunc::GreaterThan:
        return SCE_GXM_DEPTH_FUNC_GREATER;
    case FramebufferRegs::CompareFunc::GreaterThanOrEqual:
        return SCE_GXM_DEPTH_FUNC_GREATER_EQUAL;
    }
    return SCE_GXM_DEPTH_FUNC_ALWAYS;
}

/**
 * La funcion de comparacion de la PLANTILLA, que NO es la misma tabla que la de
 * profundidad aunque el enum de la PICA si lo sea.
 *
 * GXM tiene dos enumerados distintos (SceGxmDepthFunc y SceGxmStencilFunc) con
 * valores distintos, asi que reusar MapDepthFunc aqui compilaria -- los dos son
 * enteros -- y escribiria basura en el registro de la plantilla.
 *
 * EL SENTIDO DE LA COMPARACION. El rasterizador de software llama a
 * CompareFuncPasses(func, ref, dest): primero el valor de REFERENCIA del
 * registro y despues lo que hay GUARDADO en el buffer. GXM lo describe igual
 * ("passes when fragment stencil value is less than the current stored value"),
 * asi que "menor que" significa lo mismo en los dos y la traduccion es directa.
 * Si estuviera al reves, todo lo que usara plantilla saldria en negativo.
 */
SceGxmStencilFunc MapStencilFunc(FramebufferRegs::CompareFunc func) {
    switch (func) {
    case FramebufferRegs::CompareFunc::Never:
        return SCE_GXM_STENCIL_FUNC_NEVER;
    case FramebufferRegs::CompareFunc::Always:
        return SCE_GXM_STENCIL_FUNC_ALWAYS;
    case FramebufferRegs::CompareFunc::Equal:
        return SCE_GXM_STENCIL_FUNC_EQUAL;
    case FramebufferRegs::CompareFunc::NotEqual:
        return SCE_GXM_STENCIL_FUNC_NOT_EQUAL;
    case FramebufferRegs::CompareFunc::LessThan:
        return SCE_GXM_STENCIL_FUNC_LESS;
    case FramebufferRegs::CompareFunc::LessThanOrEqual:
        return SCE_GXM_STENCIL_FUNC_LESS_EQUAL;
    case FramebufferRegs::CompareFunc::GreaterThan:
        return SCE_GXM_STENCIL_FUNC_GREATER;
    case FramebufferRegs::CompareFunc::GreaterThanOrEqual:
        return SCE_GXM_STENCIL_FUNC_GREATER_EQUAL;
    }
    return SCE_GXM_STENCIL_FUNC_ALWAYS;
}

/**
 * Las ocho acciones de plantilla de la PICA, que se corresponden UNA A UNA con
 * las de GXM.
 *
 * El unico par que se puede confundir es Increment/IncrementWrap (y su
 * simetrico): en la PICA, Increment SATURA -- sw_framebuffer.cpp hace
 * min(old, 254) + 1 -- y IncrementWrap da la vuelta. En GXM, INCR es el que
 * satura e INCR_WRAP el que da la vuelta ("with wrapping in the 0-255 range"),
 * asi que el par cruzado esta bien puesto. Cambiarlos por error solo se notaria
 * en los pocos pixeles que lleguen a 0 o a 255, que es la peor forma de
 * equivocarse: no se ve hasta que se ve.
 */
SceGxmStencilOp MapStencilOp(FramebufferRegs::StencilAction action) {
    switch (action) {
    case FramebufferRegs::StencilAction::Keep:
        return SCE_GXM_STENCIL_OP_KEEP;
    case FramebufferRegs::StencilAction::Zero:
        return SCE_GXM_STENCIL_OP_ZERO;
    case FramebufferRegs::StencilAction::Replace:
        return SCE_GXM_STENCIL_OP_REPLACE;
    case FramebufferRegs::StencilAction::Increment:
        return SCE_GXM_STENCIL_OP_INCR;
    case FramebufferRegs::StencilAction::Decrement:
        return SCE_GXM_STENCIL_OP_DECR;
    case FramebufferRegs::StencilAction::Invert:
        return SCE_GXM_STENCIL_OP_INVERT;
    case FramebufferRegs::StencilAction::IncrementWrap:
        return SCE_GXM_STENCIL_OP_INCR_WRAP;
    case FramebufferRegs::StencilAction::DecrementWrap:
        return SCE_GXM_STENCIL_OP_DECR_WRAP;
    }
    return SCE_GXM_STENCIL_OP_KEEP;
}

bool MapBlendFactor(FramebufferRegs::BlendFactor factor, SceGxmBlendFactor* out) {
    switch (factor) {
    case FramebufferRegs::BlendFactor::Zero:
        *out = SCE_GXM_BLEND_FACTOR_ZERO;
        return true;
    case FramebufferRegs::BlendFactor::One:
        *out = SCE_GXM_BLEND_FACTOR_ONE;
        return true;
    case FramebufferRegs::BlendFactor::SourceColor:
        *out = SCE_GXM_BLEND_FACTOR_SRC_COLOR;
        return true;
    case FramebufferRegs::BlendFactor::OneMinusSourceColor:
        *out = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        return true;
    case FramebufferRegs::BlendFactor::DestColor:
        *out = SCE_GXM_BLEND_FACTOR_DST_COLOR;
        return true;
    case FramebufferRegs::BlendFactor::OneMinusDestColor:
        *out = SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        return true;
    case FramebufferRegs::BlendFactor::SourceAlpha:
        *out = SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
        return true;
    case FramebufferRegs::BlendFactor::OneMinusSourceAlpha:
        *out = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        return true;
    case FramebufferRegs::BlendFactor::DestAlpha:
        *out = SCE_GXM_BLEND_FACTOR_DST_ALPHA;
        return true;
    case FramebufferRegs::BlendFactor::OneMinusDestAlpha:
        *out = SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        return true;
    case FramebufferRegs::BlendFactor::SourceAlphaSaturate:
        *out = SCE_GXM_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        return true;
    default:
        // Quedan los cuatro factores con color/alfa CONSTANTE: GXM no tiene
        // color de mezcla constante (ver MapCarrierFactor).
        return false;
    }
}

bool IsConstantFactor(FramebufferRegs::BlendFactor factor) {
    return factor == FramebufferRegs::BlendFactor::ConstantColor ||
           factor == FramebufferRegs::BlendFactor::OneMinusConstantColor ||
           factor == FramebufferRegs::BlendFactor::ConstantAlpha ||
           factor == FramebufferRegs::BlendFactor::OneMinusConstantAlpha;
}

bool ReadsSourceAlpha(FramebufferRegs::BlendFactor factor) {
    return factor == FramebufferRegs::BlendFactor::SourceAlpha ||
           factor == FramebufferRegs::BlendFactor::OneMinusSourceAlpha ||
           factor == FramebufferRegs::BlendFactor::SourceAlphaSaturate;
}

/**
 * FACTORES CONSTANTES LLEVADOS EN EL ALFA DE LA FUENTE (0.2.0.1).
 *
 * GXM no tiene color de mezcla constante, pero si el alfa que sale del shader
 * no hace falta para nada mas, el shader puede sacar ahi la constante y GXM la
 * usa como SRC_ALPHA. 'alpha_slot': factor del alfa, donde el color constante
 * vale por su componente alfa; en los del color seria un valor por canal, y
 * eso no cabe en un alfa. Las condiciones de que el alfa este libre las mira
 * BuildBlend. Kirby Triple Deluxe dibuja asi (color = destino, alfa = alfa del
 * destino por la constante: "eq 0/0 fac 0/1 0/12" en crash.txt), y esos lotes
 * iban a software con una sincronizacion cada uno.
 */
bool MapCarrierFactor(FramebufferRegs::BlendFactor factor, bool alpha_slot,
                      SceGxmBlendFactor* out) {
    switch (factor) {
    case FramebufferRegs::BlendFactor::ConstantAlpha:
        *out = SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
        return true;
    case FramebufferRegs::BlendFactor::OneMinusConstantAlpha:
        *out = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        return true;
    case FramebufferRegs::BlendFactor::ConstantColor:
        *out = SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
        return alpha_slot;
    case FramebufferRegs::BlendFactor::OneMinusConstantColor:
        *out = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        return alpha_slot;
    default:
        return MapBlendFactor(factor, out);
    }
}

bool MapBlendEquation(FramebufferRegs::BlendEquation equation, SceGxmBlendFunc* out) {
    switch (equation) {
    case FramebufferRegs::BlendEquation::Add:
        *out = SCE_GXM_BLEND_FUNC_ADD;
        return true;
    case FramebufferRegs::BlendEquation::Subtract:
        *out = SCE_GXM_BLEND_FUNC_SUBTRACT;
        return true;
    case FramebufferRegs::BlendEquation::ReverseSubtract:
        *out = SCE_GXM_BLEND_FUNC_REVERSE_SUBTRACT;
        return true;
    case FramebufferRegs::BlendEquation::Min:
        *out = SCE_GXM_BLEND_FUNC_MIN;
        return true;
    case FramebufferRegs::BlendEquation::Max:
        *out = SCE_GXM_BLEND_FUNC_MAX;
        return true;
    }
    return false;
}

/**
 * La caja de scissor de la PICA traducida al recorte de region de GXM.
 *
 * POR QUE SE PUEDE ACELERAR. El recorte no toca ni el shader ni los vertices:
 * es un rectangulo que el ISP aplica a los fragmentos antes de escribirlos, y
 * eso GXM lo tiene en hardware. Hasta 0.1.0.15 cualquier lote con el scissor
 * encendido se iba entero a software, y el scissor sale en casi todos los
 * juegos: recortes de interfaz, minimapas, dibujar en una porcion de pantalla.
 *
 * QUE SIGNIFICA CADA MODO, QUE EL SDK DE VITA LO CUENTA AL REVES. La cabecera
 * de vitasdk anota SCE_GXM_REGION_CLIP_OUTSIDE como "clip tiles inside the
 * region", que leido literal seria "descarta lo de DENTRO". Es un error de
 * transcripcion: el modo nombra LO QUE SE RECORTA, no lo que sobrevive. La
 * prueba esta en el propio SDK, dos declaraciones mas abajo:
 * sceGxmSetDefaultRegionClipAndViewport(ctx, ancho-1, alto-1) es el ayudante
 * que deja un render target dibujando NORMAL, y eso solo puede ser OUTSIDE con
 * el rectangulo entero; con la otra lectura ese ayudante dejaria la pantalla en
 * blanco. Vita3K, que reimplementa GXM por encima de OpenGL, traduce OUTSIDE a
 * un glScissor corriente por el mismo motivo.
 *
 *   Include (3) -> se dibuja SOLO dentro de la caja  -> CLIP_OUTSIDE
 *   Exclude (1) -> se dibuja SOLO fuera de la caja   -> CLIP_INSIDE
 *
 * EL EJE Y VA AL REVES Y NO ES UN DETALLE. La PICA mide su Y de pantalla desde
 * ABAJO (sw_framebuffer.h hace "y = cached_height - y", y cached_height es el
 * registro CRUDO, que guarda el alto menos uno), mientras que la fila 0 de un
 * render target de GXM es la de arriba. Para el pixel de pantalla p de la PICA,
 * la fila del framebuffer del invitado -- y por tanto la de nuestra superficie,
 * que CopyTiledGuest copia fila a fila sin voltear -- es (alto - 1) - p. Asi
 * que la caja [y1, y2] de la PICA es, en filas de GXM, [(alto-1) - y2,
 * (alto-1) - y1]: se voltea Y SE INTERCAMBIAN los extremos.
 *
 * LOS LIMITES SON INCLUSIVOS EN LAS DOS PARTES. En la PICA, x2/y2 son la ultima
 * columna y la ultima fila DENTRO de la caja (por eso el rasterizador de
 * software les suma uno al pasarlos a 12.4, para comparar con "<"). En GXM,
 * xMax/yMax tambien son inclusivos: es lo que delata el ayudante de arriba, al
 * que se le pasa ancho-1 y alto-1.
 */
struct RegionClip {
    SceGxmRegionClipMode mode = SCE_GXM_REGION_CLIP_NONE;
    u32 x_min = 0;
    u32 y_min = 0;
    u32 x_max = 0;
    u32 y_max = 0;
};

RegionClip MakeRegionClip(const RasterizerRegs& rasterizer, u32 width, u32 height) {
    RegionClip clip{};
    const auto mode = rasterizer.scissor_test.mode.Value();
    /**
     * El modo 2 no es ninguno de los dos y el rasterizador de software NO HACE
     * NADA con el: sus dos ramas preguntan por Include y por Exclude, y ese
     * valor no es ninguno de los dos. Aqui tampoco, entonces. Tratarlo como
     * Include -- que es lo que saldria de preguntar solo "distinto de
     * Disabled" -- seria inventarse un recorte que el hardware no hace.
     */
    if (mode != RasterizerRegs::ScissorMode::Include &&
        mode != RasterizerRegs::ScissorMode::Exclude) {
        return clip;
    }

    const u32 x1 = rasterizer.scissor_test.x1;
    const u32 y1 = rasterizer.scissor_test.y1;
    const u32 x2 = rasterizer.scissor_test.x2;
    const u32 y2 = rasterizer.scissor_test.y2;
    const u32 last_column = width - 1;
    const u32 last_row = height - 1;

    /**
     * Caja vacia, o entera fuera del framebuffer.
     *
     * Con x2 < x1 (o y2 < y1) no hay pixel que cumpla la comparacion del
     * rasterizador de software: con Include no se dibuja nada y con Exclude no
     * se quita nada. Y si la esquina de arriba a la izquierda ya cae fuera del
     * framebuffer, la parte visible de la caja es vacia, que es el mismo caso.
     *
     * Recortar el otro extremo a lo visible no cambia lo que se pinta (fuera
     * del framebuffer no se escribe de todas formas) y evita darle a GXM un
     * rectangulo mas grande que su render target, que es un parametro invalido.
     */
    if (x2 < x1 || y2 < y1 || x1 > last_column || y1 > last_row) {
        clip.mode = mode == RasterizerRegs::ScissorMode::Include ? SCE_GXM_REGION_CLIP_ALL
                                                                 : SCE_GXM_REGION_CLIP_NONE;
        return clip;
    }

    const u32 visible_x2 = std::min(x2, last_column);
    const u32 visible_y2 = std::min(y2, last_row);
    clip.mode = mode == RasterizerRegs::ScissorMode::Include ? SCE_GXM_REGION_CLIP_OUTSIDE
                                                             : SCE_GXM_REGION_CLIP_INSIDE;
    clip.x_min = x1;
    clip.x_max = visible_x2;
    clip.y_min = last_row - visible_y2;
    clip.y_max = last_row - y1;
    return clip;
}

/// Clave de pipeline: la configuracion de fragmentos mas la mezcla (que en GXM
/// va dentro del programa, no en el estado del contexto).
struct PipelineKey {
    Pica::Shader::FSConfig config;
    u32 blend_bits = 0;

    bool operator==(const PipelineKey& other) const noexcept {
        return blend_bits == other.blend_bits && config == other.config;
    }
};

struct PipelineKeyHash {
    std::size_t operator()(const PipelineKey& key) const noexcept {
        return Common::HashCombine(key.config.Hash(), static_cast<u64>(key.blend_bits));
    }
};

} // Anonymous namespace

/**
 * Una superficie de dibujado: el color y la profundidad con los que trabaja la
 * GPU para un framebuffer del invitado.
 *
 * La memoria es NUESTRA (reservada y mapeada para la GPU); el invitado solo ve
 * lo que se le vuelca. Por eso, al crearla, se copia dentro lo que ya tuviera
 * su framebuffer: hay juegos que dibujan por encima de lo anterior sin borrar,
 * y arrancar en negro cambiaria la imagen.
 *
 * COLOR LINEAL, PROFUNDIDAD EN TILES, Y NO ES UN CAPRICHO. El color se pide
 * lineal porque la CPU tiene que leerlo y escribirlo para traducirlo al orden
 * del invitado (ver CopyTiledGuest), y una superficie en tiles solo la entiende
 * el chip. La profundidad va en tiles porque es lo que GXM sabe conservar entre
 * escenas, y como no se comparte con el invitado, su formato interno da igual.
 */
struct RasterizerGXM::Surface {
    ~Surface() {
        if (render_target != nullptr) {
            sceGxmDestroyRenderTarget(render_target);
        }
        if (driver_uid >= 0) {
            sceKernelFreeMemBlock(driver_uid);
        }
    }

    /// Pisa el tramo [addr, addr + size)? Es lo que preguntan los vaciados y
    /// las invalidaciones por tramo para no tocar los demas framebuffers.
    [[nodiscard]] bool Overlaps(PAddr addr, u32 size) const noexcept {
        const PAddr end = guest_address + guest_stride * height;
        return addr < end && guest_address < addr + size;
    }

    /**
     * Lo mismo para el buffer de PROFUNDIDAD del invitado, que esta en otra
     * direccion y hasta ahora no lo miraba nadie.
     *
     * Y eso era un agujero: el 3DS borra su profundidad con un relleno de
     * memoria del propio chip grafico, que llega aqui como InvalidateRegion
     * sobre el tramo del buffer de profundidad. Como Overlaps solo conoce el
     * tramo del COLOR, ese borrado no tocaba nada y nuestra superficie seguia
     * con la profundidad del fotograma anterior. Con la prueba de profundidad
     * sola casi no se notaba (los juegos redibujan de cerca a lejos); con la
     * plantilla se nota entero, porque una plantilla que arranca con basura
     * recorta justo lo contrario de lo que el juego queria.
     */
    [[nodiscard]] bool OverlapsDepth(PAddr addr, u32 size) const noexcept {
        if (guest_depth_address == 0 || guest_depth_stride == 0) {
            return false;
        }
        const PAddr end = guest_depth_address + guest_depth_stride * height;
        return addr < end && guest_depth_address < addr + size;
    }

    PAddr guest_address = 0;
    u32 width = 0;
    u32 height = 0;
    /// Bytes por pixel en el invitado (2, 3 o 4) y en nuestra superficie de
    /// color: los mismos, salvo en RGB8 (3 alli, 4 aqui, porque no existe color
    /// surface de 24 bits).
    u32 bpp = 0;
    u32 rt_bpp = 0;
    /// Bytes de una FILA DE TILES del invitado. El rasterizador de software
    /// direcciona con width*bpp por cada 8 filas de pixeles, asi que este es el
    /// stride natural y NO el de la configuracion de pantalla.
    u32 guest_stride = 0;
    /// Stride de la superficie de color, en PIXELES.
    u32 color_stride = 0;
    /// Escala en mitades (resolution_scale al crearla): Phys() da nuestros
    /// pixeles. width/height son siempre los del invitado.
    u32 scale = 2;
    /// Formato de la superficie de color, como SCE_GXM_COLOR_FORMAT_*.
    /// Para presentar directo (4.6) hace falta saberlo: el del invitado y el
    /// nuestro no son el mismo enum aunque describan lo mismo.
    u32 gxm_color_format = 0;
    /// Hay dibujado nuestro que el invitado todavia no ha visto.
    bool dirty = false;
    /// Su contenido ya se lo ha llevado una copia de pantalla en la GPU y no
    /// ha cambiado desde (0.1.8.7): FlushForPresent no la vuelca.
    bool copied = false;
    /// Rellenos de color del invitado que todavia no se han pintado: los hace
    /// la siguiente escena (0.1.9.6, ver AccelerateFill). Cada uno es una
    /// franja de filas y los cuatro bytes del pixel tal como van en
    /// color_buffer.
    struct PendingClear {
        u32 first_row = 0;
        u32 rows = 0;
        u32 texel = 0;
    };
    bool clear_pending = false;
    u32 clear_count = 0;
    std::array<PendingClear, 4> clears{};
    /// El invitado ha cambiado este framebuffer por otro camino (un relleno de
    /// color, una transferencia): hay que volver a leerlo antes de dibujar.
    bool needs_reload = false;
    /**
     * Direccion y formato del buffer de PROFUNDIDAD del invitado.
     *
     * No se comparte memoria con el (el nuestro lo tiene GXM en un formato
     * interno en tiles que no se puede traducir); se guarda solo para dos
     * cosas: saber si un relleno del invitado lo pisa (OverlapsDepth) y poder
     * leer con que valor lo ha dejado, que es lo que se le pasa a GXM como
     * fondo al rehacerlo (ver GuestDepthClear en el .cpp).
     */
    PAddr guest_depth_address = 0;
    u32 guest_depth_stride = 0;
    u32 depth_bpp = 0;
    /**
     * Nuestra profundidad y plantilla no valen: hay que rehacerlas desde el
     * valor con el que el invitado ha dejado las suyas.
     *
     * Arranca en true porque una superficie recien creada tiene la memoria como
     * se la dio el sistema. Antes se hacia un memset a 0xFF confiando en que
     * eso fuera 1.0 en el formato interno de GXM; para la profundidad colaba,
     * pero dejaba la PLANTILLA a 255, que no es el 0 del que parte cualquier
     * juego. Ahora lo borra el propio GXM con sus valores de fondo, que es el
     * unico camino que no supone nada sobre como coloca el chip los bits.
     */
    bool depth_needs_clear = true;
    /// Se creo con profundidad de 16 bits (D16 del invitado) o de 24+8.
    bool depth16 = false;
    bool scene_open = false;
    SceGxmRenderTarget* render_target = nullptr;
    SceUID driver_uid = -1;
    SceGxmColorSurface color_surface{};
    SceGxmDepthStencilSurface depth_surface{};
    Allocation color_buffer;
    Allocation depth_buffer;
};

/// Una pantalla copiada por la GPU (0.1.8.7, ver AccelerateDisplayTransfer).
struct RasterizerGXM::ScreenCopy {
    ~ScreenCopy() {
        if (render_target != nullptr) {
            sceGxmDestroyRenderTarget(render_target);
        }
        if (driver_uid >= 0) {
            sceKernelFreeMemBlock(driver_uid);
        }
    }

    /// La pantalla en la memoria del invitado (lineal) y lo que ocupa alli.
    PAddr dst = 0;
    u32 dst_size = 0;
    u32 width = 0;
    u32 height = 0;
    /// El formato de la superficie de origen, que es el de la copia (32 bits).
    u32 gxm_color_format = 0;
    /// Stride del buffer, en pixeles, y bytes por pixel (4, o 2 en los de 16
    /// bits desde 0.1.9.8).
    u32 stride = 0;
    u32 bpp = 4;
    /// La de la superficie de origen, en mitades: la copia es pixel a pixel.
    u32 scale = 2;
    Pica::PixelFormat input_format = Pica::PixelFormat::RGBA8;
    Pica::PixelFormat output_format = Pica::PixelFormat::RGBA8;
    /// Tiene la ultima copia a esa pantalla...
    bool valid = false;
    /// ...y la memoria del invitado todavia no.
    bool pending = false;
    /// Para elegir cual se rehace si hacen falta mas de kMaxScreenCopies.
    u32 last_use = 0;
    /// La textura sobre la superficie de origen del ultimo blit.
    SceGxmTexture source_texture{};
    SceGxmRenderTarget* render_target = nullptr;
    SceUID driver_uid = -1;
    SceGxmColorSurface color_surface{};
    Allocation color_buffer;
    /// Sin profundidad: la superficie "desactivada" de GXM, sin memoria.
    SceGxmDepthStencilSurface depth_surface{};
};

/// El quad de la copia de pantalla, con los shaders del presentador.
struct RasterizerGXM::BlitProgram {
    ~BlitProgram() {
        if (patcher != nullptr) {
            if (fragment_program != nullptr) {
                sceGxmShaderPatcherReleaseFragmentProgram(patcher, fragment_program);
            }
            if (vertex_program != nullptr) {
                sceGxmShaderPatcherReleaseVertexProgram(patcher, vertex_program);
            }
            if (fragment_registered) {
                sceGxmShaderPatcherUnregisterProgram(patcher, fragment_id);
            }
            if (vertex_registered) {
                sceGxmShaderPatcherUnregisterProgram(patcher, vertex_id);
            }
        }
        ReleaseCgOutput(vertex_output);
        ReleaseCgOutput(fragment_output);
    }

    SceGxmShaderPatcher* patcher = nullptr;
    const SceShaccCgCompileOutput* vertex_output = nullptr;
    const SceShaccCgCompileOutput* fragment_output = nullptr;
    SceGxmShaderPatcherId vertex_id{};
    SceGxmShaderPatcherId fragment_id{};
    bool vertex_registered = false;
    bool fragment_registered = false;
    SceGxmVertexProgram* vertex_program = nullptr;
    SceGxmFragmentProgram* fragment_program = nullptr;
    u32 texture_unit = 0;
    /// Dos quads de cuatro vertices (recto y volteado) y sus cuatro indices.
    Allocation quad;
};

namespace {
/// Pantallas a la vez: las dos de arriba y las dos de abajo del doble buffer.
constexpr std::size_t kMaxScreenCopies = 4;
/// Vertice del quad del blit: posicion float4 y coordenada float2, como el
/// presentador.
constexpr u32 kBlitVertexStride = 6 * sizeof(float);
} // Anonymous namespace

/// Comprueba que el tramo [address, address + size) de la memoria del invitado
/// este mapeado y sea contiguo. Es la misma comprobacion que hace el
/// renderizador de software antes de leer un framebuffer: sin ella, una
/// direccion del juego a medio mapear haria que la GPU leyera memoria que no
/// existe.
bool GuestSpanMapped(Memory::MemorySystem& memory, PAddr address, u32 size) {
    if (size == 0) {
        return false;
    }
    const u8* first = memory.GetPhysicalPointer(address);
    const u8* last = memory.GetPhysicalPointer(address + size - 1);
    return first != nullptr && last == first + size - 1;
}

namespace {
/**
 * Una linea de crash.txt con valores dentro.
 *
 * Las notas son lo unico que se ve de la consola, y una nota sin numeros
 * ("rechazado") obliga a gastar una prueba entera en averiguar cual de los diez
 * motivos fue -- que es justo lo que costo el ultimo muro. El buffer es local y
 * corto a proposito: esto se llama desde sitios donde el emulador puede estar a
 * punto de irse al suelo.
 */
template <typename... Args>
void NoteFmt(const char* tag, fmt::format_string<Args...> format, Args&&... args) {
    char note[128];
    const auto written =
        fmt::format_to_n(note, sizeof(note) - 1, format, std::forward<Args>(args)...);
    *written.out = 0;
    Common::VitaNote(tag, note);
}

/// La misma nota, pero solo la primera vez. El flag lo pone quien llama para que
/// cada sitio tenga el suyo y ninguno tape a otro.
template <typename... Args>
void NoteOnce(bool& already, const char* tag, fmt::format_string<Args...> format, Args&&... args) {
    if (already) {
        return;
    }
    already = true;
    NoteFmt(tag, format, std::forward<Args>(args)...);
}

/**
 * La configuracion de DIBUJADO de la PICA en crash.txt, una vez por combinacion.
 *
 * Es el equivalente de las notas "fb" del renderizador de software, que cuentan
 * la de PANTALLA. Son dos registros distintos y confundirlos ya costo una
 * prueba de consola: lo que se mira aqui no es de donde barre la pantalla, sino
 * donde esta dibujando la PICA, que en la mayoria de los juegos es otro buffer
 * (en tiles, sin hueco entre filas) del que luego se copia a la pantalla.
 *
 * Se queda con las primeras combinaciones y calla: un juego que alterne dos
 * framebuffers llenaria el fichero de lineas identicas.
 */
void NoteDrawConfig(PAddr color, PAddr depth, u32 width, u32 height, u32 color_format,
                    u32 depth_format) {
    struct Signature {
        PAddr color;
        PAddr depth;
        u32 width;
        u32 height;
        u32 color_format;
        u32 depth_format;
    };
    constexpr u32 kMaxSignatures = 6;
    static Signature seen[kMaxSignatures]{};
    static u32 seen_count = 0;

    const Signature now{color, depth, width, height, color_format, depth_format};
    for (u32 i = 0; i < seen_count; i++) {
        if (std::memcmp(&seen[i], &now, sizeof(Signature)) == 0) {
            return;
        }
    }
    if (seen_count >= kMaxSignatures) {
        return;
    }
    seen[seen_count++] = now;
    NoteFmt("gxm dib", "addr {:#010x} {}x{} fmt {} z {:#010x} fmtz {}", color, width, height,
            color_format, depth, depth_format);
}

/**
 * El framebuffer de DIBUJADO del invitado NO es lineal.
 *
 * La PICA lo guarda en tiles de 8x8 en orden Morton, y las filas de tiles van
 * una detras de otra con width*bpp bytes por cada 8 filas de pixeles. Es
 * exactamente lo que hace DrawPixel del rasterizador de software:
 *
 *     GetMortonOffset(x, y, bpp) + (y & ~7) * width * bpp
 *
 * y por eso aqui se llama a LA MISMA funcion: si un dia cambia una, tiene que
 * cambiar la otra o las dos imagenes dejan de coincidir, que es lo unico que
 * este backend tiene para saber si acierta.
 *
 * Nuestra superficie de color es lineal, asi que entrar y salir de ella es
 * traducir entre los dos ordenes. Se hace en la CPU a proposito: GXM convierte
 * lineal <-> tiled (32x32) y lineal <-> swizzled (Morton de la imagen entera),
 * pero lo del 3DS no es ninguno de los dos (Morton de 8x8 dentro de filas de
 * tiles), asi que sceGxmTransferCopy -- que es lo que habia aqui -- volcaba la
 * imagen revuelta.
 *
 * NO hay volteo vertical. El invitado guarda la fila 0 arriba porque DrawPixel
 * ya voltea al escribir (y = height - y), y el viewport que DrawBatchOnGpu le
 * da a GXM deja la superficie en ese mismo sentido.
 */
/**
 * Los cuatro desplazamientos de PAREJA de una fila dentro del mosaico.
 *
 * En el orden Morton de 8x8, fijada la fila, los texels de x par y x impar caen
 * SEGUIDOS en memoria. Se ve en el dibujo de arriba: la fila de abajo es
 * 00 01 04 05 16 17 20 21, o sea las parejas (0,1) (4,5) (16,17) (20,21). Eso
 * vale para las ocho filas, cambiando solo el numero de partida.
 *
 * Por eso se puede copiar de dos en dos: dos texels contiguos en el invitado y
 * dos contiguos en la superficie lineal. La version anterior llamaba a
 * GetMortonOffset -- tres operaciones de entrelazado de bits -- y hacia un
 * memcpy POR TEXEL. En un framebuffer de 256x416 eso son 106.496 vueltas con
 * aritmetica de bits dentro, dos veces por fotograma (el volcado y la recarga),
 * y se median 45 ms de los 187 del fotograma.
 */
template <u32 kGuestBpp>
inline void MortonPairOffsets(u32 y, u32* out) {
    for (u32 k = 0; k < 4; k++) {
        out[k] = VideoCore::GetMortonOffset(k * 2, y, kGuestBpp);
    }
}

template <u32 kGuestBpp, bool kToGuest>
void CopyTiledGuestImpl(u8* guest, u8* linear, u32 width, u32 height, u32 linear_stride) {
    constexpr u32 kSurfaceBpp = kGuestBpp == 3 ? 4 : kGuestBpp;

    /**
     * HACIA EL INVITADO, LA FILA SE LEE ENTERA PRIMERO (0.1.0.45).
     *
     * La superficie vive en CDRAM, que la CPU ve SIN cache: cada lectura va a
     * memoria y espera. El bucle de abajo leia de 8 en 8 bytes saltando por la
     * fila, o sea una espera completa por cada pareja de pixeles -- 65.000 por
     * superficie de 256x512 --, y esto pasa en cada volcado ('vol'), dentro de
     * las transferencias. memcpy de la fila entera a un buffer normal lee en
     * rafagas largas y seguidas, que es lo unico que esa memoria hace rapido;
     * despues el reparto en mosaicos lee de memoria con cache. Los bytes que
     * llegan al invitado son los mismos: solo cambia de donde se leen.
     */
    alignas(16) u8 row_copy[kToGuest ? 1024 * kSurfaceBpp : 1];

    /**
     * RGB8 va por el camino de siempre, texel a texel.
     *
     * Es el unico formato donde los dos lados NO miden lo mismo: tres bytes en
     * el invitado y cuatro en la superficie. Las parejas dejan de ser contiguas
     * en el lado lineal y ademas hay que rellenar el alfa. Es tambien el
     * formato menos frecuente, asi que no compensa complicarlo.
     */
    if constexpr (kGuestBpp == 3) {
        for (u32 y = 0; y < height; y++) {
            u8* row = linear + static_cast<std::size_t>(y) * linear_stride;
            if constexpr (kToGuest) {
                std::memcpy(row_copy, row, static_cast<std::size_t>(width) * kSurfaceBpp);
                row = row_copy;
            }
            u8* const tile_row = guest + static_cast<std::size_t>(y & ~7u) * width * kGuestBpp;
            for (u32 x = 0; x < width; x++) {
                u8* const guest_pixel = tile_row + VideoCore::GetMortonOffset(x, y, kGuestBpp);
                u8* const linear_pixel = row + x * kSurfaceBpp;
                if constexpr (kToGuest) {
                    std::memcpy(guest_pixel, linear_pixel, kGuestBpp);
                } else {
                    std::memcpy(linear_pixel, guest_pixel, kGuestBpp);
                    // El invitado no guarda alfa en RGB8 y la superficie si
                    // tiene el byte: dejarlo a cero haria transparente lo que
                    // la mezcla lea de ahi.
                    linear_pixel[3] = 0xFF;
                }
            }
        }
        return;
    }

    // Los de 2 y 4 bytes: de dos en dos, con los desplazamientos calculados una
    // vez por fila en vez de una vez por texel.
    constexpr u32 kPairBytes = kGuestBpp * 2;
    const u32 full_tiles = width & ~7u;
    for (u32 y = 0; y < height; y++) {
        u8* row = linear + static_cast<std::size_t>(y) * linear_stride;
        if constexpr (kToGuest) {
            std::memcpy(row_copy, row, static_cast<std::size_t>(width) * kSurfaceBpp);
            row = row_copy;
        }
        u8* const tile_row = guest + static_cast<std::size_t>(y & ~7u) * width * kGuestBpp;
        u32 pair_offset[4];
        MortonPairOffsets<kGuestBpp>(y, pair_offset);

        for (u32 tile_x = 0; tile_x < full_tiles; tile_x += 8) {
            // GetMortonOffset ya incluye el salto del mosaico en su termino
            // coarse_x, asi que aqui solo hay que sumar el del mosaico actual.
            u8* const tile_base = tile_row + static_cast<std::size_t>(tile_x) * 8 * kGuestBpp;
            u8* const linear_base = row + static_cast<std::size_t>(tile_x) * kSurfaceBpp;
            for (u32 k = 0; k < 4; k++) {
                u8* const guest_pair = tile_base + pair_offset[k];
                u8* const linear_pair = linear_base + k * 2 * kSurfaceBpp;
                if constexpr (kToGuest) {
                    std::memcpy(guest_pair, linear_pair, kPairBytes);
                } else {
                    std::memcpy(linear_pair, guest_pair, kPairBytes);
                }
            }
        }

        // Resto, si el ancho no es multiplo de ocho. La PICA lo alinea siempre,
        // pero no cuesta nada y evita que un ancho raro se salga del buffer.
        for (u32 x = full_tiles; x < width; x++) {
            u8* const guest_pixel = tile_row + VideoCore::GetMortonOffset(x, y, kGuestBpp);
            u8* const linear_pixel = row + x * kSurfaceBpp;
            if constexpr (kToGuest) {
                std::memcpy(guest_pixel, linear_pixel, kGuestBpp);
            } else {
                std::memcpy(linear_pixel, guest_pixel, kGuestBpp);
            }
        }
    }
}

/// Medida en pixeles nuestros de 'guest' pixeles del invitado, con la escala
/// en mitades (ver RasterizerGXM::resolution_scale).
constexpr u32 Phys(u32 guest, u32 scale) {
    return guest * scale / 2;
}

/**
 * Remuestreo por vecino mas cercano de una imagen lineal a otra de otro
 * tamano: superficie escalada <-> imagen del invitado a 1x. Fila a fila por
 * una copia en memoria normal: la superficie esta en CDRAM sin cache, y
 * leerla pixel a pixel salteado costaria una lectura lenta por pixel.
 */
template <typename T>
void ResampleRows(const u8* src, u32 src_stride, u32 src_width, u32 src_height, u8* dst,
                  u32 dst_stride, u32 dst_width, u32 dst_height) {
    alignas(16) T row[2048];
    if (src_width > 2048 || src_width == 0 || src_height == 0) {
        return;
    }
    for (u32 y = 0; y < dst_height; y++) {
        const u32 sy = y * src_height / dst_height;
        std::memcpy(row, src + static_cast<std::size_t>(sy) * src_stride, src_width * sizeof(T));
        T* out = reinterpret_cast<T*>(dst + static_cast<std::size_t>(y) * dst_stride);
        for (u32 x = 0; x < dst_width; x++) {
            out[x] = row[x * src_width / dst_width];
        }
    }
}

void Resample(const u8* src, u32 src_stride, u32 src_width, u32 src_height, u8* dst,
              u32 dst_stride, u32 dst_width, u32 dst_height, u32 bpp) {
    if (bpp == 4) {
        ResampleRows<u32>(src, src_stride, src_width, src_height, dst, dst_stride, dst_width,
                          dst_height);
    } else if (bpp == 2) {
        ResampleRows<u16>(src, src_stride, src_width, src_height, dst, dst_stride, dst_width,
                          dst_height);
    }
}

/// Traduce entre el framebuffer del invitado (tiles de 8x8) y nuestra
/// superficie lineal, en el sentido que diga to_guest.
void CopyTiledGuest(u8* guest, u8* linear, u32 width, u32 height, u32 linear_stride, u32 bpp,
                    bool to_guest) {
    switch (bpp) {
    case 2:
        if (to_guest) {
            CopyTiledGuestImpl<2, true>(guest, linear, width, height, linear_stride);
        } else {
            CopyTiledGuestImpl<2, false>(guest, linear, width, height, linear_stride);
        }
        break;
    case 3:
        if (to_guest) {
            CopyTiledGuestImpl<3, true>(guest, linear, width, height, linear_stride);
        } else {
            CopyTiledGuestImpl<3, false>(guest, linear, width, height, linear_stride);
        }
        break;
    case 4:
        if (to_guest) {
            CopyTiledGuestImpl<4, true>(guest, linear, width, height, linear_stride);
        } else {
            CopyTiledGuestImpl<4, false>(guest, linear, width, height, linear_stride);
        }
        break;
    default:
        break;
    }
}

} // Anonymous namespace

/**
 * Cache de programas de fragmentos: uno por configuracion TEV y mezcla.
 *
 * Cada entrada compila su Cg (via SceShaccCg, en la consola) y lo registra en
 * el patcher de vita2d. Compilar cuesta, asi que las configuraciones ya vistas
 * se reutilizan; la Fase 4 hara que la compilacion no bloquee el hilo de
 * dibujado.
 */
struct RasterizerGXM::PipelineCache {
    struct Entry {
        /**
         * Esta entrada sirve, o esta guardada SOLO para no volver a intentarlo.
         * Ver la nota de Get: una configuracion que no se puede construir se
         * queda en el mapa marcada asi, con su motivo, para que el lote
         * siguiente no repita la compilacion.
         */
        bool usable = true;
        const char* reason = "shader";
        /// CgGeneration() al construirla (0.1.5.9). Ver Get.
        u32 cg_generation = 0;
        SceGxmFragmentProgram* program = nullptr;
        /// Este shader lee normales y vista, o sea que necesita el formato de
        /// vertice largo. Lo decide config.lighting.enable al compilarlo.
        bool lit = false;
        /// Y este lee la W de la textura 0 (textura proyectada): variante _proj.
        bool proj = false;
        const SceGxmProgramParameter* const_color = nullptr;
        const SceGxmProgramParameter* combiner_buffer_color = nullptr;
        const SceGxmProgramParameter* alphatest_ref = nullptr;
        /// La constante de mezcla en el alfa de salida (0.2.0.1, MapCarrierFactor).
        const SceGxmProgramParameter* blend_const_alpha = nullptr;
        const SceGxmProgramParameter* samplers[3] = {nullptr, nullptr, nullptr};
        u8 sampler_units[3] = {0, 0, 0};
        const SceGxmProgramParameter* fog_lut = nullptr;
        const SceGxmProgramParameter* fog_color = nullptr;
        const SceGxmProgramParameter* tex_border_color = nullptr;
        /// Tablas de busqueda de la iluminacion, como textura de 256x24.
        const SceGxmProgramParameter* lighting_lut = nullptr;
        u8 lighting_lut_unit = 0;
        const SceGxmProgramParameter* light_specular_0 = nullptr;
        const SceGxmProgramParameter* light_specular_1 = nullptr;
        const SceGxmProgramParameter* light_diffuse = nullptr;
        const SceGxmProgramParameter* light_ambient = nullptr;
        const SceGxmProgramParameter* light_position = nullptr;
        const SceGxmProgramParameter* light_spot_direction = nullptr;
        const SceGxmProgramParameter* light_dist_atten = nullptr;
        const SceGxmProgramParameter* lighting_global_ambient = nullptr;
        const SceShaccCgCompileOutput* output = nullptr;
        SceGxmShaderPatcherId id{};
        bool registered = false;
        /// Compilandose en el hilo de compilacion (0.1.9.8), y la mezcla con
        /// la que se creara el programa cuando acabe.
        std::shared_ptr<CgJob> job;
        SceGxmBlendInfo blend{};
    };

    /**
     * Un shader de vertices compilado con su descripcion de flujo.
     *
     * Hay dos: el corriente y el que ademas lleva el cuaternion de normal y el
     * vector de vista para la iluminacion por fragmento (ver kVertexSourceLit).
     * El de fragmentos elige el suyo por configuracion, no por adivinanza.
     */
    struct VertexFormat {
        const SceGxmProgram* gxp = nullptr;
        SceGxmVertexProgram* program = nullptr;
        SceShaccCgCompileOutput const* output = nullptr;
        SceGxmShaderPatcherId id{};
        bool registered = false;
    };

    PipelineCache(SceGxmContext* context_, SceGxmShaderPatcher* patcher_)
        : context{context_}, patcher{patcher_} {}

    ~PipelineCache() {
        for (auto& [key, entry] : entries) {
            if (entry->program != nullptr) {
                sceGxmShaderPatcherReleaseFragmentProgram(patcher, entry->program);
            }
            if (entry->registered) {
                sceGxmShaderPatcherUnregisterProgram(patcher, entry->id);
            }
            if (entry->output != nullptr) {
                ReleaseCgOutput(entry->output);
            }
        }
        ReleaseVertexFormat(plain);
        ReleaseVertexFormat(lit);
        ReleaseVertexFormat(plain_proj);
        ReleaseVertexFormat(lit_proj);
    }

    /// El formato de vertice de una combinacion (iluminacion, textura 0
    /// proyectada). El de fragmentos se enlaza contra este mismo.
    VertexFormat& FormatFor(bool with_lighting, bool projected) {
        if (projected) {
            return with_lighting ? lit_proj : plain_proj;
        }
        return with_lighting ? lit : plain;
    }

    void ReleaseVertexFormat(VertexFormat& format) {
        if (format.program != nullptr) {
            sceGxmShaderPatcherReleaseVertexProgram(patcher, format.program);
        }
        if (format.registered) {
            sceGxmShaderPatcherUnregisterProgram(patcher, format.id);
        }
        if (format.output != nullptr) {
            ReleaseCgOutput(format.output);
        }
        format = VertexFormat{};
    }

    bool Initialize() {
        // El corriente tiene que salir; el de iluminacion puede fallar y el
        // emulador sigue: los lotes con luz se van a software, que es
        // exactamente lo que hacian antes de que este existiera.
        if (!BuildVertexFormat(plain, "azahar_gxm_v.cg", kVertexSource, false)) {
            return false;
        }
        if (!BuildVertexFormat(lit, "azahar_gxm_vl.cg", kVertexSourceLit, true)) {
            LOG_WARNING(Render, "GXM: sin shader de vertices con normales; la iluminacion por "
                                "fragmento se queda en software");
            Common::VitaNote("gxm init", "sin vs de iluminacion");
            ReleaseVertexFormat(lit);
        }
        // Las dos de textura proyectada, con el mismo criterio: si no salen,
        // esos lotes siguen en software, como hasta 0.1.0.43.
        if (!BuildVertexFormat(plain_proj, "azahar_gxm_vp.cg", kVertexSourceProj, false, true)) {
            Common::VitaNote("gxm init", "sin vs de textura proyectada");
            ReleaseVertexFormat(plain_proj);
        }
        if (!BuildVertexFormat(lit_proj, "azahar_gxm_vlp.cg", kVertexSourceLitProj, true, true)) {
            Common::VitaNote("gxm init", "sin vs de textura proyectada con luz");
            ReleaseVertexFormat(lit_proj);
        }
        return true;
    }

    bool BuildVertexFormat(VertexFormat& format, const char* name, const char* source,
                           bool with_lighting, bool projected = false) {
        format.output = CompileCg(SCE_SHACCCG_PROFILE_VP, name, source);
        if (format.output == nullptr) {
            return false;
        }
        const auto* gxp = reinterpret_cast<const SceGxmProgram*>(format.output->programData);
        format.gxp = gxp;
        if (sceGxmShaderPatcherRegisterProgram(patcher, gxp, &format.id) != 0) {
            return false;
        }
        format.registered = true;

        struct AttribSpec {
            const char* name;
            u32 offset;
            u8 components;
        };
        // El orden y los desplazamientos son los del bucle de empaquetado de
        // DrawBatchOnGpu: si se toca uno hay que tocar el otro, porque no hay
        // nada que compruebe que coinciden (la GPU lee lo que se le diga).
        const AttribSpec specs[] = {
            {"position", 0, 4},
            {"color", 4 * sizeof(float), 4},
            {"tc0", 8 * sizeof(float), 2},
            {"tc1", 10 * sizeof(float), 2},
            {"tc2", 12 * sizeof(float), 2},
            {"normquat", 14 * sizeof(float), 4},
            {"view", 18 * sizeof(float), 3},
            // La W de la textura 0 va al FINAL, detras de lo que haya: ver
            // VertexStride y el empaquetado.
            {"tc0w", with_lighting ? 21 * sizeof(float) : 14 * sizeof(float), 1},
        };
        // Los cinco comunes, los dos de luz si van, y la W si va. La W no
        // puede ir en el hueco de los de luz: se copia el spec a su sitio.
        AttribSpec chosen[8];
        u32 count = 0;
        for (u32 i = 0; i < 5; i++) {
            chosen[count++] = specs[i];
        }
        if (with_lighting) {
            chosen[count++] = specs[5];
            chosen[count++] = specs[6];
        }
        if (projected) {
            chosen[count++] = specs[7];
        }
        SceGxmVertexAttribute attributes[8]{};
        for (u32 i = 0; i < count; i++) {
            const SceGxmProgramParameter* param =
                sceGxmProgramFindParameterByName(gxp, chosen[i].name);
            if (param == nullptr) {
                LOG_ERROR(Render, "GXM: el shader de vertices no declara {}", chosen[i].name);
                return false;
            }
            attributes[i].streamIndex = 0;
            attributes[i].offset = static_cast<u16>(chosen[i].offset);
            attributes[i].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
            attributes[i].componentCount = chosen[i].components;
            attributes[i].regIndex = sceGxmProgramParameterGetResourceIndex(param);
        }
        SceGxmVertexStream stream{};
        stream.stride = static_cast<u16>(VertexStride(with_lighting, projected));
        stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
        return sceGxmShaderPatcherCreateVertexProgram(patcher, format.id, attributes, count,
                                                      &stream, 1, &format.program) == 0;
    }

    /**
     * El pipeline de esta configuracion, o nullptr si no se puede acelerar.
     * El motivo sale por out_reason para que lo anote quien llama.
     *
     * LOS FALLOS SE RECUERDAN, Y ESTO ES LO MAS IMPORTANTE DE ESTA FUNCION.
     *
     * Hasta 0.1.0.17 todos los caminos de error de aqui devolvian nulo SIN
     * dejar nada en el mapa, asi que el lote siguiente con la misma
     * configuracion volvia a intentarlo entero: generar la fuente del shader y
     * llamar a sceShaccCgCompileProgram, que es un compilador de verdad y en
     * esta consola cuesta cientos de milisegundos. Mientras la iluminacion se
     * rechazaba antes de llegar aqui casi no se notaba; en cuanto los lotes con
     * luz empezaron a pedir pipeline, un solo shader que no compile pasa a
     * costar una compilacion fallida POR LOTE. Eso es medio fotograma por
     * segundo, y es justo lo que se midio en consola.
     *
     * Ahora la entrada se guarda pase lo que pase. Si no se pudo construir,
     * queda marcada como inservible con su motivo y las siguientes consultas
     * son una busqueda en una tabla hash. Es el mismo criterio que ya seguia el
     * cargador de libshacccg cuando le paso esto mismo (ver gxm_cg.cpp).
     */
    const Entry* Get(const Pica::RegsInternal& regs, const char** out_reason = nullptr) {
        const auto fail = [out_reason](const char* reason) -> const Entry* {
            if (out_reason != nullptr) {
                *out_reason = reason;
            }
            return nullptr;
        };

        /**
         * La mezcla se mira ANTES y fuera del cache a proposito: sus motivos de
         * rechazo (logic op, modo de fragmento, factores constantes) dependen de
         * registros que no entran en la clave, asi que no se pueden recordar por
         * clave. A cambio comprobarla no reserva memoria ni compila nada.
         */
        SceGxmBlendInfo blend{};
        u32 blend_bits = 0;
        bool alpha_carrier = false;
        if (!BuildBlend(regs, blend, blend_bits, alpha_carrier)) {
            return fail("mezcla");
        }

        Pica::Shader::FSConfig config{regs};
        // El perfil no describe capacidades de GXM todavia: sin perfil, el
        // generador aplica las reglas por defecto.
        config.ApplyProfile(Pica::Shader::Profile{});
        PipelineKey key{config, blend_bits};

        const auto it = entries.find(key);
        if (it != entries.end() && it->second->job != nullptr) {
            Entry& pending = *it->second;
            if (!pending.job->done.load(std::memory_order_acquire)) {
                return fail("fs compilando");
            }
            const SceShaccCgCompileOutput* output = pending.job->output;
            pending.job.reset();
            const char* reason = nullptr;
            if (!Finish(&pending, output, &reason)) {
                pending.usable = false;
                pending.reason = reason != nullptr ? reason : "shader";
            }
        }
        if (it != entries.end()) {
            /**
             * 0.1.5.9: un "no compila" de cuando el compilador estaba roto no es
             * culpa del shader. Si desde entonces se ha recargado el compilador
             * (CgGeneration cambio), se reintenta UNA vez con el nuevo. La
             * entrada fallida no tiene nada que soltar (ver Build).
             */
            const bool retry = !it->second->usable && it->second->cg_generation != CgGeneration() &&
                               std::strcmp(it->second->reason, "compilar shader") == 0;
            if (!retry) {
                return it->second->usable ? it->second.get() : fail(it->second->reason);
            }
            entries.erase(it);
        }

        auto entry = std::make_unique<Entry>();
        entry->cg_generation = CgGeneration();
        const char* reason = nullptr;
        if (!Build(entry.get(), config, blend, alpha_carrier, &reason)) {
            entry->usable = false;
            entry->reason = reason != nullptr ? reason : "shader";
        }
        if (entry->job != nullptr) {
            entries.emplace(key, std::move(entry));
            return fail("fs compilando");
        }
        const bool usable = entry->usable;
        const char* stored_reason = entry->reason;
        const Entry* result = entry.get();
        entries.emplace(key, std::move(entry));
        return usable ? result : fail(stored_reason);
    }

    /**
     * Compila el shader y prepara la entrada. Devuelve false si no se puede, y
     * en ese caso deja los campos de la entrada LIMPIOS: el destructor del
     * cache recorre todas las entradas, tambien las inservibles, y soltar dos
     * veces un programa del parcheador no perdona.
     */
    bool Build(Entry* entry, const Pica::Shader::FSConfig& config, const SceGxmBlendInfo& blend,
               bool alpha_carrier, const char** out_reason) {
        const auto fail = [out_reason](const char* reason) {
            if (out_reason != nullptr) {
                *out_reason = reason;
            }
            return false;
        };

        const char* generator_reason = nullptr;
        const auto source = Pica::Shader::Generator::GXM::GenerateFragmentShader(
            config, &generator_reason, alpha_carrier);
        if (!source.has_value()) {
            return fail(generator_reason != nullptr ? generator_reason : "shader");
        }
        /**
         * Con iluminacion hace falta el formato de vertice largo, y ese shader
         * puede no haber compilado al arrancar. Si no esta, no vale generar el
         * de fragmentos: quedaria pidiendo unos varyings que el de vertices no
         * escribe, y eso GXM lo rechaza al crear el programa (o, peor, lo
         * acepta y lee basura). Nulo aqui es el lote a software.
         */
        const bool needs_lighting = config.lighting.enable != 0;
        if (needs_lighting && lit.program == nullptr) {
            NoteOnce(noted[3], "gxm pipeline", "sin vs de iluminacion");
            return fail("sin vs de luz");
        }
        // Textura 0 proyectada: el de fragmentos lee TEXCOORD5 (la W), y solo
        // la escriben las variantes _proj. Mismo razonamiento que arriba.
        const bool needs_w =
            config.texture.texture0_type.Value() ==
            Pica::TexturingRegs::TextureConfig::Projection2D;
        if (needs_w && FormatFor(needs_lighting, true).program == nullptr) {
            return fail("sin vs de textura proyectada");
        }

        entry->lit = needs_lighting;
        entry->proj = needs_w;
        entry->blend = blend;
        /**
         * EN SEGUNDO PLANO (0.1.9.8). Compilar uno de estos en el hilo de
         * emulacion eran 1-2 s de juego parado por cada material nuevo (Kirby
         * Triple Deluxe: decenas seguidos), y si el hilo de compilacion estaba
         * con uno de vertices, ademas esperar a que acabara. Ahora va delante
         * de la cola del hilo de compilacion y, mientras tanto, los lotes con
         * esta configuracion no se dibujan (Get: "fs compilando"). De la cache
         * de la tarjeta se lee aqui mismo.
         */
        if (const SceShaccCgCompileOutput* cached =
                LoadCgCache(SCE_SHACCCG_PROFILE_FP, "azahar_gxm_f.cg", source->c_str())) {
            return Finish(entry, cached, out_reason);
        }
        auto job = std::make_shared<CgJob>();
        job->profile = SCE_SHACCCG_PROFILE_FP;
        job->name = "azahar_gxm_f.cg";
        job->sources.push_back(std::move(*source));
        job->variants.push_back(0);
        entry->job = job;
        CgSubmit(std::move(job), true);
        return true;
    }

    /// Registra el programa compilado y crea el de fragmentos con su mezcla.
    bool Finish(Entry* entry, const SceShaccCgCompileOutput* output, const char** out_reason) {
        const auto fail = [out_reason](const char* reason) {
            if (out_reason != nullptr) {
                *out_reason = reason;
            }
            return false;
        };
        const SceGxmBlendInfo& blend = entry->blend;
        entry->output = output;
        if (entry->output == nullptr) {
            // CompileCg deja en crash.txt el nombre del fichero Y el primer
            // mensaje del compilador, que es lo unico con lo que se puede
            // arreglar un shader generado sin tener la consola delante.
            return fail("compilar shader");
        }
        const auto* gxp = reinterpret_cast<const SceGxmProgram*>(entry->output->programData);
        const int register_rc = sceGxmShaderPatcherRegisterProgram(patcher, gxp, &entry->id);
        if (register_rc != 0) {
            NoteOnce(noted[4], "gxm pipeline", "registrar err {:#x}",
                     static_cast<u32>(register_rc));
            ReleaseCgOutput(entry->output);
            entry->output = nullptr;
            return fail("registrar shader");
        }
        entry->registered = true;
        const int program_rc = sceGxmShaderPatcherCreateFragmentProgram(
            patcher, entry->id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4, SCE_GXM_MULTISAMPLE_NONE,
            &blend, FormatFor(entry->lit, entry->proj).gxp, &entry->program);
        if (program_rc != 0) {
            NoteOnce(noted[5], "gxm pipeline", "programa de fragmentos err {:#x}",
                     static_cast<u32>(program_rc));
            sceGxmShaderPatcherUnregisterProgram(patcher, entry->id);
            ReleaseCgOutput(entry->output);
            entry->registered = false;
            entry->output = nullptr;
            entry->program = nullptr;
            return fail("programa de fragmentos");
        }
        entry->const_color = sceGxmProgramFindParameterByName(gxp, "const_color");
        entry->combiner_buffer_color =
            sceGxmProgramFindParameterByName(gxp, "tev_combiner_buffer_color");
        entry->alphatest_ref = sceGxmProgramFindParameterByName(gxp, "alphatest_ref");
        entry->blend_const_alpha = sceGxmProgramFindParameterByName(gxp, "blend_const_alpha");
        entry->samplers[0] = sceGxmProgramFindParameterByName(gxp, "tex0");
        entry->samplers[1] = sceGxmProgramFindParameterByName(gxp, "tex1");
        entry->samplers[2] = sceGxmProgramFindParameterByName(gxp, "tex2");
        entry->fog_lut = sceGxmProgramFindParameterByName(gxp, "fog_lut");
        entry->fog_color = sceGxmProgramFindParameterByName(gxp, "fog_color");
        entry->tex_border_color = sceGxmProgramFindParameterByName(gxp, "tex_border_color");
        entry->lighting_lut = sceGxmProgramFindParameterByName(gxp, "lighting_lut");
        if (entry->lighting_lut != nullptr) {
            entry->lighting_lut_unit =
                static_cast<u8>(sceGxmProgramParameterGetResourceIndex(entry->lighting_lut));
        }
        entry->light_specular_0 = sceGxmProgramFindParameterByName(gxp, "light_specular_0");
        entry->light_specular_1 = sceGxmProgramFindParameterByName(gxp, "light_specular_1");
        entry->light_diffuse = sceGxmProgramFindParameterByName(gxp, "light_diffuse");
        entry->light_ambient = sceGxmProgramFindParameterByName(gxp, "light_ambient");
        entry->light_position = sceGxmProgramFindParameterByName(gxp, "light_position");
        entry->light_spot_direction =
            sceGxmProgramFindParameterByName(gxp, "light_spot_direction");
        entry->light_dist_atten = sceGxmProgramFindParameterByName(gxp, "light_dist_atten");
        entry->lighting_global_ambient =
            sceGxmProgramFindParameterByName(gxp, "lighting_global_ambient");
        for (u32 i = 0; i < 3; i++) {
            entry->sampler_units[i] =
                entry->samplers[i] != nullptr
                    ? static_cast<u8>(sceGxmProgramParameterGetResourceIndex(entry->samplers[i]))
                    : 0;
        }

        return true;
    }

    bool BuildBlend(const Pica::RegsInternal& regs, SceGxmBlendInfo& blend, u32& bits,
                    bool& alpha_carrier) {
        alpha_carrier = false;
        const auto& merger = regs.framebuffer.output_merger;
        if (merger.fragment_operation_mode !=
            FramebufferRegs::FragmentOperationMode::Default) {
            // Sombra y gas: no son un dibujado normal y GXM no los reproduce.
            NoteOnce(noted[0], "gxm mezcla", "modo de fragmento {}",
                     static_cast<u32>(merger.fragment_operation_mode.Value()));
            return false;
        }
        blend.colorMask = 0;
        if (merger.red_enable) {
            blend.colorMask |= SCE_GXM_COLOR_MASK_R;
        }
        if (merger.green_enable) {
            blend.colorMask |= SCE_GXM_COLOR_MASK_G;
        }
        if (merger.blue_enable) {
            blend.colorMask |= SCE_GXM_COLOR_MASK_B;
        }
        if (merger.alpha_enable) {
            blend.colorMask |= SCE_GXM_COLOR_MASK_A;
        }
        blend.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
        blend.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
        blend.colorSrc = SCE_GXM_BLEND_FACTOR_ONE;
        blend.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
        blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
        blend.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
        /**
         * El logic op solo cuenta si la MEZCLA ESTA APAGADA.
         *
         * En la PICA son excluyentes -- el rasterizador de software hace
         * "if (alphablend_enable) mezcla; else LogicOp" --, asi que el registro
         * del logic op puede traer cualquier cosa mientras el juego dibuja con
         * mezcla, y el hardware ni lo mira. Rechazar el lote por ese valor,
         * como se hacia aqui, mandaba a software TODO lo que llevara mezcla.
         * GXM no tiene logic op, asi que cuando de verdad se usa, a software.
         */
        if (!merger.alphablend_enable &&
            merger.logic_op.Value() != FramebufferRegs::LogicOp::Copy) {
            NoteOnce(noted[2], "gxm mezcla", "logic op {}",
                     static_cast<u32>(merger.logic_op.Value()));
            return false;
        }
        if (merger.alphablend_enable) {
            SceGxmBlendFunc color_func{};
            SceGxmBlendFunc alpha_func{};
            // Los factores se mapean a temporales porque en SceGxmBlendInfo son
            // campos de 4 bits y no se les puede pasar la direccion.
            SceGxmBlendFactor color_src{};
            SceGxmBlendFactor color_dst{};
            SceGxmBlendFactor alpha_src{};
            SceGxmBlendFactor alpha_dst{};
            const auto& factors = merger.alpha_blending;
            const auto src_rgb = factors.factor_source_rgb.Value();
            const auto dst_rgb = factors.factor_dest_rgb.Value();
            const auto src_a = factors.factor_source_a.Value();
            const auto dst_a = factors.factor_dest_a.Value();
            bool mapped = false;
            if (!IsConstantFactor(src_rgb) && !IsConstantFactor(dst_rgb) &&
                !IsConstantFactor(src_a) && !IsConstantFactor(dst_a)) {
                mapped = MapBlendFactor(src_rgb, &color_src) &&
                         MapBlendFactor(dst_rgb, &color_dst) &&
                         MapBlendFactor(src_a, &alpha_src) && MapBlendFactor(dst_a, &alpha_dst);
            } else {
                // El alfa de la fuente queda libre si ningun factor lo lee y el
                // alfa resultante no depende de el: o no se escribe, o se mezcla
                // con factores (no min/max) y el de la fuente es cero.
                const auto equation_a = factors.blend_equation_a.Value();
                const bool alpha_free =
                    !merger.alpha_enable ||
                    (src_a == FramebufferRegs::BlendFactor::Zero &&
                     equation_a != FramebufferRegs::BlendEquation::Min &&
                     equation_a != FramebufferRegs::BlendEquation::Max);
                mapped = alpha_free && !ReadsSourceAlpha(src_rgb) && !ReadsSourceAlpha(dst_rgb) &&
                         !ReadsSourceAlpha(src_a) && !ReadsSourceAlpha(dst_a) &&
                         MapCarrierFactor(src_rgb, false, &color_src) &&
                         MapCarrierFactor(dst_rgb, false, &color_dst) &&
                         MapCarrierFactor(src_a, true, &alpha_src) &&
                         MapCarrierFactor(dst_a, true, &alpha_dst);
                alpha_carrier = mapped;
            }
            if (!MapBlendEquation(merger.alpha_blending.blend_equation_rgb.Value(), &color_func) ||
                !MapBlendEquation(merger.alpha_blending.blend_equation_a.Value(), &alpha_func) ||
                !mapped) {
                // Con los seis valores crudos en crash.txt se sabe CUAL de
                // ellos no tiene equivalente (los factores constantes) sin
                // gastar otra prueba en consola.
                NoteOnce(noted[1], "gxm mezcla", "eq {}/{} fac {}/{} {}/{}",
                         static_cast<u32>(merger.alpha_blending.blend_equation_rgb.Value()),
                         static_cast<u32>(merger.alpha_blending.blend_equation_a.Value()),
                         static_cast<u32>(merger.alpha_blending.factor_source_rgb.Value()),
                         static_cast<u32>(merger.alpha_blending.factor_dest_rgb.Value()),
                         static_cast<u32>(merger.alpha_blending.factor_source_a.Value()),
                         static_cast<u32>(merger.alpha_blending.factor_dest_a.Value()));
                return false;
            }
            blend.colorFunc = color_func;
            blend.alphaFunc = alpha_func;
            blend.colorSrc = color_src;
            blend.colorDst = color_dst;
            blend.alphaSrc = alpha_src;
            blend.alphaDst = alpha_dst;
        }
        // El bit 4 esta libre (la mascara ocupa 0-3): el shader con la
        // constante en el alfa es otro, aunque la mezcla de GXM coincida.
        bits = static_cast<u32>(blend.colorMask) | (alpha_carrier ? 1u << 4 : 0u) |
               (static_cast<u32>(blend.colorFunc) << 8) |
               (static_cast<u32>(blend.alphaFunc) << 12) |
               (static_cast<u32>(blend.colorSrc) << 16) |
               (static_cast<u32>(blend.colorDst) << 20) |
               (static_cast<u32>(blend.alphaSrc) << 24) |
               (static_cast<u32>(blend.alphaDst) << 28);
        return true;
    }

    SceGxmContext* context;
    /**
     * Motivos por los que este cache se ha negado a dar un pipeline, uno por
     * sitio y anotados una sola vez. En orden: modo de fragmento, factores de
     * mezcla, logic op, generador, registrar el programa y crear el programa de
     * fragmentos.
     *
     * Hasta 0.1.0.14 todos estos caminos devolvian nulo en silencio, y el lote
     * se iba a software sin dejar rastro: la superficie se creaba ("gxm fb:
     * creada") y despues no habia ni dibujado ni motivo. Eso es una prueba de
     * consola perdida.
     */
    bool noted[8] = {};
    SceGxmShaderPatcher* patcher;
    /// Sin iluminacion (14 floats por vertice) y con ella (21). El segundo
    /// puede quedarse vacio si su shader no compila; entonces los lotes con luz
    /// siguen cayendo a software.
    VertexFormat plain;
    VertexFormat lit;
    VertexFormat plain_proj;
    VertexFormat lit_proj;
    std::unordered_map<PipelineKey, std::unique_ptr<Entry>, PipelineKeyHash> entries;
};

/**
 * Shaders de vertices de la PICA traducidos a Cg (cg_vs_shader_gen) y sus
 * programas de vertices de GXM ya enlazados a una disposicion de atributos.
 *
 * Dos niveles, porque cuestan cosas distintas:
 *   - programs: la traduccion y la COMPILACION (sceShaccCgCompileProgram, que
 *     en esta consola son cientos de milisegundos). Por programa de la PICA,
 *     salidas, semanticas y de donde sale cada registro de entrada.
 *   - linked: el programa del parcheador para una disposicion concreta de
 *     flujos y atributos. Barato, pero no se puede reutilizar entre
 *     disposiciones.
 * Los fallos se recuerdan en los dos niveles: una traduccion o una compilacion
 * que falla no se reintenta en cada lote (la leccion de 0.1.0.17 con los de
 * fragmentos).
 */
struct RasterizerGXM::HwShaderCache {
    struct Program {
        const SceShaccCgCompileOutput* output = nullptr;
        const SceGxmProgram* gxp = nullptr;
        SceGxmShaderPatcherId id{};
        bool registered = false;
        bool usable = false;
        const char* reason = "vs";
        const SceGxmProgramParameter* uniform_f = nullptr;
        const SceGxmProgramParameter* uniform_i = nullptr;
        const SceGxmProgramParameter* uniform_b = nullptr;
        const SceGxmProgramParameter* uniform_default = nullptr;
        /// "vs_in<registro>", o null si el programa no lee ese registro de un
        /// flujo (lo elimina el compilador o sale de otro sitio).
        std::array<const SceGxmProgramParameter*, 16> inputs{};
        /// Compilandose en el hilo de compilacion (0.1.9.6): hasta que acabe,
        /// el lote va por la CPU.
        std::shared_ptr<CgJob> job;
    };

    /// Tope de programas traducidos. Un juego usa decenas; pasado esto se deja
    /// de traducir y los nuevos se quedan en la CPU, que es correcto.
    static constexpr std::size_t kMaxPrograms = 384;

    explicit HwShaderCache(SceGxmShaderPatcher* patcher_) : patcher{patcher_} {}

    ~HwShaderCache() {
        for (auto& [key, program] : linked) {
            if (program != nullptr) {
                sceGxmShaderPatcherReleaseVertexProgram(patcher, program);
            }
        }
        for (auto& [key, program] : programs) {
            if (program->registered) {
                sceGxmShaderPatcherUnregisterProgram(patcher, program->id);
            }
            if (program->output != nullptr) {
                ReleaseCgOutput(program->output);
            }
        }
    }

    const Program* GetProgram(u64 key, const Pica::ShaderSetup& setup,
                              const Pica::Shader::Generator::PicaVSConfig& config,
                              const Pica::Shader::Generator::GXM::VSInputs& inputs,
                              bool write_lighting, bool write_w, const char** out_reason) {
        const auto it = programs.find(key);
        if (it != programs.end()) {
            Program& found = *it->second;
            if (found.job != nullptr) {
                if (!found.job->done.load(std::memory_order_acquire)) {
                    *out_reason = "vs compilando";
                    return nullptr;
                }
                const SceShaccCgCompileOutput* output = found.job->output;
                const u32 variant = found.job->used_variant;
                found.job.reset();
                Finish(found, output, variant);
            }
            if (!found.usable) {
                *out_reason = found.reason;
                return nullptr;
            }
            return &found;
        }
        if (programs.size() >= kMaxPrograms) {
            *out_reason = "vs tope de programas";
            return nullptr;
        }
        auto program = std::make_unique<Program>();
        Build(*program, setup, config, inputs, write_lighting, write_w);
        const Program* result = program->usable ? program.get() : nullptr;
        if (program->job != nullptr) {
            program->reason = "vs compilando";
        }
        if (result == nullptr) {
            *out_reason = program->reason;
        }
        programs.emplace(key, std::move(program));
        return result;
    }

    SceGxmVertexProgram* GetLinked(u64 key, const Program& program,
                                   const SceGxmVertexAttribute* attributes, u32 attribute_count,
                                   const SceGxmVertexStream* streams, u32 stream_count) {
        const auto it = linked.find(key);
        if (it != linked.end()) {
            return it->second;
        }
        SceGxmVertexProgram* vertex_program = nullptr;
        const int rc = sceGxmShaderPatcherCreateVertexProgram(
            patcher, program.id, attributes, attribute_count, streams, stream_count,
            &vertex_program);
        if (rc != 0) {
            static u32 noted_count = 0;
            bool noted = noted_count >= 4;
            noted_count++;
            if (!noted) {
                // Cada atributo: registro, componentes (las nuestras / las del
                // parametro compilado), formato, flujo y desplazamiento.
                std::string detail;
                for (u32 i = 0; i < attribute_count; i++) {
                    const SceGxmVertexAttribute& a = attributes[i];
                    u32 declared = 0;
                    for (const auto* input : program.inputs) {
                        if (input != nullptr &&
                            sceGxmProgramParameterGetResourceIndex(input) == a.regIndex) {
                            declared = sceGxmProgramParameterGetComponentCount(input);
                        }
                    }
                    detail += fmt::format(" r{}:{}/{} f{} s{}+{}", a.regIndex, a.componentCount,
                                          declared, static_cast<u32>(a.format), a.streamIndex,
                                          a.offset);
                }
                for (u32 i = 0; i < stream_count; i++) {
                    detail += fmt::format(" paso{}", streams[i].stride);
                }
                // Y los atributos que declara el programa compilado: nombre,
                // registro, componentes y tipo.
                detail += " | prog";
                const u32 params = sceGxmProgramGetParameterCount(program.gxp);
                for (u32 i = 0; i < params; i++) {
                    const SceGxmProgramParameter* param = sceGxmProgramGetParameter(program.gxp, i);
                    if (sceGxmProgramParameterGetCategory(param) !=
                        SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE) {
                        continue;
                    }
                    detail += fmt::format(" {}:r{}:{}t{}", sceGxmProgramParameterGetName(param),
                                          sceGxmProgramParameterGetResourceIndex(param),
                                          sceGxmProgramParameterGetComponentCount(param),
                                          static_cast<u32>(sceGxmProgramParameterGetType(param)));
                }
                Common::VitaNote("gxm vs", fmt::format("crear programa de vertices err {:#x}:{}",
                                                       static_cast<u32>(rc), detail)
                                               .c_str());
            }
            vertex_program = nullptr;
        }
        // Tambien el fallo: nullptr en el mapa es "no se puede", sin reintentar.
        linked.emplace(key, vertex_program);
        return vertex_program;
    }

private:
    void Build(Program& program, const Pica::ShaderSetup& setup,
               const Pica::Shader::Generator::PicaVSConfig& config,
               const Pica::Shader::Generator::GXM::VSInputs& inputs, bool write_lighting,
               bool write_w) {
        /**
         * Si el compilador de Cg ya se ha roto una vez en esta sesion, no se le
         * da otro shader de vertices: en 0.1.4.7 un "fatal internal error" en
         * uno de estos dejo el compilador inservible para TODO lo que vino
         * despues, fragmentos incluidos. Ver CgPoisoned en gxm_cg.cpp.
         */
        if (CgPoisoned()) {
            program.reason = "vs compilador roto";
            return;
        }
        // Lo que el generador de GLSL llama "extra": aqui todo apagado. Sin
        // shader de geometria (AccelerateDrawBatch no llega aqui con uno) y sin
        // la multiplicacion exacta, que el traductor de Cg no hace (lo rechaza).
        Pica::Shader::Generator::ExtraVSConfig extra{};
        /**
         * VARIANTES PARA EL COMPILADOR DE LA CONSOLA (0.1.7.3).
         *
         * cg_error.txt: los dos shaders de piel de Rubi Omega que dan "fatal
         * internal error" en libshacccg tienen lo mismo que ningun otro:
         * enteros con operaciones de bits (el indexado de las matrices de
         * huesos) y ~50 funciones que se llaman unas a otras. Sin el
         * compilador en el PC no se puede saber cual de las dos cosas lo
         * rompe, asi que se prueba en la consola: sin enteros, sin enteros ni
         * funciones, sin funciones, y al final la traduccion de siempre. Entre
         * un error interno y el siguiente intento gxm_cg.cpp recarga el
         * compilador y apunta en la tarjeta el codigo que lo rompio, para no
         * volver a darselo en otra sesion.
         */
        using namespace Pica::Shader::Generator::GXM;
        static constexpr u32 kVariants[] = {kCgFloatAddress, kCgFloatAddress | kCgFlat, kCgFlat,
                                            0};
        const char* generator_reason = nullptr;
        std::vector<std::string> sources;
        std::vector<u32> variants;
        for (const u32 variant : kVariants) {
            g_cg_variant.store(variant, std::memory_order_relaxed);
            auto source = GenerateVertexShader(setup, config, extra, inputs, write_lighting,
                                               write_w, &generator_reason);
            if (!source.has_value()) {
                // Sin subrutinas en linea el analisis es el mismo en todas las
                // variantes: si falla, fallaria igual en las demas. En linea
                // puede fallar solo por tamano, y la siguiente sin linea vale.
                if ((variant & kCgFlat) == 0) {
                    break;
                }
                continue;
            }
            sources.push_back(std::move(*source));
            variants.push_back(variant);
        }
        g_cg_variant.store(0, std::memory_order_relaxed);
        if (sources.empty()) {
            program.reason = generator_reason != nullptr ? generator_reason : "vs traducir";
            return;
        }
        /**
         * TOPE DE TAMANO (0.1.9.2). crash.txt de 0.1.9.0 y 0.1.9.1: un shader de
         * vertices de 26 KB de Cg pasaba mas de 30 s en el compilador, con picos
         * de 27 MB y hasta 20 MB retenidos despues. Desde 0.1.9.6 se compila
         * aparte (sin congelar) y queda en la cache de la tarjeta, asi que el
         * tope sube de 16 a 32 KB: el coste se paga una vez y la siguiente
         * partida lo lee al momento. Mas grande que eso, a la CPU.
         */
        constexpr std::size_t kMaxCompileSource = 32u * 1024u;
        // De la cache de la tarjeta: es leer un fichero, aqui mismo.
        if (const SceShaccCgCompileOutput* cached =
                LoadCgCache(SCE_SHACCCG_PROFILE_VP, "azahar_gxm_vs.cg", sources.front().c_str())) {
            Finish(program, cached, variants.front());
            return;
        }
        if (sources.front().size() > kMaxCompileSource) {
            static u32 big_notes = 0;
            if (big_notes < 4) {
                big_notes++;
                NoteFmt("gxm vs", "{} bytes de Cg: no se compila en partida, a la CPU",
                        sources.front().size());
            }
            program.reason = "vs demasiado grande";
            return;
        }
        // Al hilo de compilacion (0.1.9.6); GetProgram recoge el resultado.
        auto job = std::make_shared<CgJob>();
        job->sources = std::move(sources);
        job->variants = std::move(variants);
        program.job = job;
        CgSubmit(std::move(job));
    }

    /// Registra en el parcheador lo que salio del compilador (o de la cache).
    void Finish(Program& program, const SceShaccCgCompileOutput* output, u32 used_variant) {
        program.output = output;
        if (program.output == nullptr) {
            program.reason = "vs compilar";
            return;
        }
        {
            static u32 variant_notes = 0;
            if (used_variant != 0 && variant_notes < 4) {
                variant_notes++;
                NoteFmt("gxm vs", "variante {} (1 sin enteros, 2 en linea, 3 ambas)",
                        used_variant);
            }
        }
        program.gxp = reinterpret_cast<const SceGxmProgram*>(program.output->programData);
        if (sceGxmShaderPatcherRegisterProgram(patcher, program.gxp, &program.id) != 0) {
            ReleaseCgOutput(program.output);
            program.output = nullptr;
            program.gxp = nullptr;
            program.reason = "vs registrar";
            return;
        }
        program.registered = true;
        program.uniform_f = sceGxmProgramFindParameterByName(program.gxp, "vs_f");
        program.uniform_i = sceGxmProgramFindParameterByName(program.gxp, "vs_i");
        program.uniform_b = sceGxmProgramFindParameterByName(program.gxp, "vs_b");
        program.uniform_default = sceGxmProgramFindParameterByName(program.gxp, "vs_default");
        for (u32 reg = 0; reg < 16; reg++) {
            const std::string name = fmt::format("vs_in{}", reg);
            program.inputs[reg] = sceGxmProgramFindParameterByName(program.gxp, name.c_str());
        }
        program.usable = true;
        static bool noted_first = false;
        if (!noted_first) {
            noted_first = true;
            Common::VitaNote("gxm vs", "primer shader de vertices traducido y compilado");
        }
    }

    SceGxmShaderPatcher* patcher;
    std::unordered_map<u64, std::unique_ptr<Program>> programs;
    std::unordered_map<u64, SceGxmVertexProgram*> linked;

public:
    /// Booleanos que lee cada programa (UsedBoolUniforms), por su clave
    /// generica: se miran una vez, al hacer falta el especializado (0.1.7.4).
    std::unordered_map<u64, u32> used_bools;

    /**
     * Especializados ya pedidos por cada clave generica (0.1.7.9). Un shader
     * que lea k booleanos puede pedir hasta 2^k programas, y cada uno es una
     * compilacion entera; en 0.1.7.8 la entrada al 3D de Zafiro Alfa acabo en
     * bad_alloc justo ahi. Pasado el tope, las combinaciones nuevas de ese
     * shader se quedan en la CPU.
     */
    static constexpr u32 kMaxSpecializedPerProgram = 16;
    std::unordered_map<u64, u32> specialized_count;

    /**
     * Claves genericas cuyo programa GXM no deja enlazar con ninguna
     * disposicion de atributos (0.1.9.1). Sus especializados no se compilan:
     * en Zafiro Alfa cada uno tardaba hasta 30 s con el juego congelado,
     * dejaba 8 MB retenidos en el compilador y despues no se podia usar.
     */
    std::unordered_set<u64> unlinkable;

    [[nodiscard]] bool Has(u64 key) const {
        return programs.find(key) != programs.end();
    }
};

namespace {
/**
 * Registros que NO deciden el programa de fragmentos ni el de vertices de un
 * lote (ver BatchMemo): datos que se leen otra vez en cada lote (viewport,
 * recorte, colores constantes, direcciones y tamanos de textura, niebla, luces,
 * tablas, punteros de vertices e indices, uniforms de coma flotante) y
 * registros de disparo. Comprobado contra lo que leen FSConfig, BuildBlend,
 * PicaVSConfig y la disposicion de atributos. Escribirlos no invalida nada.
 */
constexpr u16 kMemoIgnoredRegs[] = {
    // Rasterizador: cara oculta, viewport, profundidad, caja de recorte.
    0x040, 0x041, 0x042, 0x043, 0x044, 0x04D, 0x04E, 0x066, 0x067, 0x068,
    // Unidades de textura: borde, tamano, LOD, direcciones y formato.
    0x081, 0x082, 0x084, 0x085, 0x086, 0x087, 0x088, 0x089, 0x08A, 0x08E, 0x091, 0x092, 0x094,
    0x095, 0x096, 0x099, 0x09A, 0x09C, 0x09D, 0x09E,
    // Colores constantes del combinador, su buffer, la niebla y su tabla.
    0x0C3, 0x0CB, 0x0D3, 0x0DB, 0x0F3, 0x0FB, 0x0FD, 0x0E1, 0x0E6, 0x0E8, 0x0E9, 0x0EA, 0x0EB,
    0x0EC, 0x0ED, 0x0EE, 0x0EF,
    // Plantilla, disparos y buffers del framebuffer.
    0x105, 0x106, 0x110, 0x111, 0x112, 0x113, 0x114, 0x115, 0x116, 0x117, 0x11C, 0x11D, 0x11E,
    // Luz ambiente global, indice y datos de las tablas de luz.
    0x1C0, 0x1C5, 0x1C8, 0x1C9, 0x1CA, 0x1CB, 0x1CC, 0x1CD, 0x1CE, 0x1CF,
    // Vertices: base, desplazamiento de cada cargador, indices, cuantos,
    // disparos, atributo fijo, buffers de comandos, modo, topologia.
    0x200, 0x203, 0x206, 0x209, 0x20C, 0x20F, 0x212, 0x215, 0x218, 0x21B, 0x21E, 0x221, 0x224,
    0x227, 0x228, 0x22A, 0x22E, 0x22F, 0x232, 0x233, 0x234, 0x235, 0x238, 0x239, 0x23A, 0x23B,
    0x23C, 0x23D, 0x245, 0x25E, 0x25F,
    // Uniforms de coma flotante del shader de vertices.
    0x2C0, 0x2C1, 0x2C2, 0x2C3, 0x2C4, 0x2C5, 0x2C6, 0x2C7, 0x2C8,
};

constexpr std::array<u64, 12> MemoMask(u32 first, u32 end) {
    std::array<u64, 12> mask{};
    for (u32 reg = first; reg < end; reg++) {
        // Los datos de cada luz (colores, posicion, foco, atenuacion): todo
        // menos su palabra de configuracion, la novena.
        bool ignored = reg >= 0x140 && reg < 0x1C0 && (reg & 0xF) != 9;
        for (const u16 skip : kMemoIgnoredRegs) {
            ignored = ignored || skip == reg;
        }
        if (!ignored) {
            mask[reg >> 6] |= 1ull << (reg & 63);
        }
    }
    return mask;
}

/// Deciden el programa de fragmentos (y parte del de vertices: salidas y luz).
constexpr std::array<u64, 12> kFsMemoMask = MemoMask(0x040, 0x200);
/// Deciden el resto del de vertices: atributos, programa y geometria.
constexpr std::array<u64, 12> kVsMemoMask = MemoMask(0x200, 0x300);

constexpr std::array<u64, 12> RegListMask(std::initializer_list<u16> regs) {
    std::array<u64, 12> mask{};
    for (const u16 reg : regs) {
        mask[reg >> 6] |= 1ull << (reg & 63);
    }
    return mask;
}

constexpr std::array<u64, 12> RegRangeMask(u32 first, u32 end) {
    std::array<u64, 12> mask{};
    for (u32 reg = first; reg < end; reg++) {
        mask[reg >> 6] |= 1ull << (reg & 63);
    }
    return mask;
}

/// Las ocho luces y la ambiente global (uniforms de fragmentos ya convertidos).
constexpr std::array<u64, 12> kLightsMask = RegRangeMask(0x140, 0x1C1);
/// La tabla de niebla: su indice y sus datos.
constexpr std::array<u64, 12> kFogLutMask =
    RegListMask({0x0E6, 0x0E8, 0x0E9, 0x0EA, 0x0EB, 0x0EC, 0x0ED, 0x0EE, 0x0EF});
/// Los colores constantes de las seis etapas del combinador.
constexpr std::array<u64, 12> kConstColorMask =
    RegListMask({0x0C3, 0x0CB, 0x0D3, 0x0DB, 0x0F3, 0x0FB});

bool AnyDirty(const Pica::DirtyRegs& dirty, const std::array<u64, 12>& mask) {
    for (u32 i = 0; i < mask.size(); i++) {
        if ((dirty.qwords[i] & mask[i]) != 0) {
            return true;
        }
    }
    return false;
}
} // Anonymous namespace

/**
 * LO QUE DECIDIO EL LOTE ANTERIOR (0.2.1.0). Buscar el programa de fragmentos
 * (construir FSConfig, la mezcla, su hash y el mapa) y el de vertices
 * (PicaVSConfig, la clave, el programa, la disposicion de atributos y el
 * enlazado) eran ~20 de los ~50 us fijos de cada lote ("preguntas", "vs" y
 * "enlazar" en crash.txt), y casi siempre salia lo mismo que en el lote
 * anterior: entre dos dibujados el juego cambia matrices, vertices y texturas,
 * no materiales. La PICA marca cada registro que se escribe (dirty_regs); si
 * desde el lote anterior no se ha escrito ninguno de los que deciden esto
 * (kFsMemoMask, kVsMemoMask), se reutiliza tal cual. Las caches de donde salen
 * los punteros no borran entradas utiles mientras vive el rasterizador.
 */
struct RasterizerGXM::BatchMemo {
    struct Location {
        bool found = false;
        u8 loader = 0;
        u16 offset = 0;
    };
    /// De donde sale cada atributo, para convertirlo si hace falta (0.1.8.5).
    struct AttributeSource {
        u8 stream = 0;
        u8 format = 0;
        u16 offset = 0;
        /// Las que trae el invitado y las que declara el programa compilado.
        u8 components = 0;
        u8 declared = 0;
    };
    struct VertexLayout {
        std::array<Location, 12> locations{};
        Pica::Shader::Generator::GXM::VSInputs inputs{};
        std::array<u8, 16> input_attribute{};
        u64 program_key = 0;
        const HwShaderCache::Program* program = nullptr;
        std::array<SceGxmVertexAttribute, 16> gxm_attributes{};
        std::array<SceGxmVertexStream, 12> gxm_streams{};
        std::array<u8, 12> stream_of_loader{};
        std::array<u8, 12> loader_of_stream{};
        std::array<AttributeSource, 16> attribute_sources{};
        u32 attribute_count = 0;
        u32 stream_count = 0;
        u64 layout_key = 0;
        SceGxmVertexProgram* vertex_program = nullptr;
        bool converted = false;
        u32 converted_stride = 0;
    };
    const PipelineCache::Entry* pipeline = nullptr;
    bool vs_valid = false;
    VertexLayout layout;

    /**
     * Uniforms de fragmentos ya pasados a float (0.2.1.0): las ocho luces, la
     * tabla de niebla y los colores constantes del combinador. Convertirlos
     * eran cientos de divisiones en coma flotante por lote (cada luz en cada
     * lote iluminado, las 128 entradas de la niebla en cada lote con niebla);
     * ahora solo despues de que se escriban sus registros.
     */
    bool lights_dirty = true;
    bool fog_dirty = true;
    bool colors_dirty = true;
    std::array<f32, 32> specular_0{};
    std::array<f32, 32> specular_1{};
    std::array<f32, 32> diffuse{};
    std::array<f32, 32> ambient{};
    std::array<f32, 32> position{};
    std::array<f32, 32> spot_direction{};
    std::array<f32, 32> dist_atten{};
    std::array<f32, 4> global_ambient{};
    std::array<f32, 256> fog_lut{};
    std::array<f32, 24> const_colors{};
};

RasterizerGXM::RasterizerGXM(VideoCore::RasterizerInterface& software_, Memory::MemorySystem& memory_,
                             Pica::PicaCore& pica_)
    : software{software_}, memory{memory_}, pica{pica_}, batch_memo{std::make_unique<BatchMemo>()} {
    s_instance = this;
}

RasterizerGXM::~RasterizerGXM() {
    s_instance = nullptr;
    FlushPending();
    // El presentador puede seguir leyendo una copia de pantalla en la GPU.
    if (context != nullptr) {
        sceGxmFinish(context);
    }
    screen_copies.clear();
    blit.reset();
    textures.reset();
    // Antes que los de fragmentos: los dos usan el mismo parcheador.
    hw_shaders.reset();
    pipelines.reset();
    surfaces.clear();
    vertex_buffer = Allocation{};
    index_buffer = Allocation{};
    DestroyShaderPatcher();
}

namespace {
void* PatcherHostAlloc(void* /*user_data*/, SceSize size) {
    return std::malloc(size);
}
void PatcherHostFree(void* /*user_data*/, void* memory) {
    std::free(memory);
}
} // Anonymous namespace

/**
 * UN PARCHEADOR DE SHADERS PROPIO Y GRANDE (0.1.9.3).
 *
 * El de vita2d es para unos pocos programas 2D: su memoria USSE (donde vive el
 * codigo de cada programa de vertices y de fragmentos ya enlazado) es fija y
 * pequena, sin funciones para crecer. crash.txt de 0.1.9.2 en Zafiro Alfa:
 * pasado el titulo, TODO programa de vertices nuevo falla con 0x805b0023 --
 * tambien uno de dos entradas en float, asi que no eran los atributos -- y
 * despues tambien los de fragmentos ("gxm skip: programa de fragmentos", el
 * texto sin dibujar). Es el patron de quedarse sin esa memoria. Este es solo
 * del rasterizador; el presentador sigue con el de vita2d.
 */
bool RasterizerGXM::CreateShaderPatcher() {
    constexpr SceSize kBufferBytes = 1u * 1024u * 1024u;
    constexpr SceSize kVertexUsseBytes = 2u * 1024u * 1024u;
    constexpr SceSize kFragmentUsseBytes = 2u * 1024u * 1024u;
    constexpr std::array<SceSize, 3> sizes{kBufferBytes, kVertexUsseBytes, kFragmentUsseBytes};
    const auto fail = [this](const char* what, int rc) {
        NoteFmt("gxm init", "parcheador propio: {} err {:#x}; se usa el de vita2d", what,
                static_cast<u32>(rc));
        DestroyShaderPatcher();
        return false;
    };
    std::array<void*, 3> bases{};
    for (u32 i = 0; i < 3; i++) {
        const SceUID uid = sceKernelAllocMemBlock("azahar_gxm_patcher",
                                                  SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
                                                  sizes[i], nullptr);
        if (uid < 0) {
            return fail("memoria", uid);
        }
        patcher_blocks[i] = uid;
        if (sceKernelGetMemBlockBase(uid, &bases[i]) < 0) {
            return fail("base", 0);
        }
    }
    unsigned int vertex_offset = 0;
    unsigned int fragment_offset = 0;
    // patcher_memory solo guarda lo ya mapeado: es lo que se desmapea.
    int rc = sceGxmMapMemory(bases[0], sizes[0],
                             static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_RW));
    if (rc < 0) {
        return fail("mapear buffer", rc);
    }
    patcher_memory[0] = bases[0];
    rc = sceGxmMapVertexUsseMemory(bases[1], sizes[1], &vertex_offset);
    if (rc < 0) {
        return fail("mapear usse de vertices", rc);
    }
    patcher_memory[1] = bases[1];
    rc = sceGxmMapFragmentUsseMemory(bases[2], sizes[2], &fragment_offset);
    if (rc < 0) {
        return fail("mapear usse de fragmentos", rc);
    }
    patcher_memory[2] = bases[2];
    SceGxmShaderPatcherParams params{};
    params.hostAllocCallback = &PatcherHostAlloc;
    params.hostFreeCallback = &PatcherHostFree;
    params.bufferMem = patcher_memory[0];
    params.bufferMemSize = sizes[0];
    params.vertexUsseMem = patcher_memory[1];
    params.vertexUsseMemSize = sizes[1];
    params.vertexUsseOffset = vertex_offset;
    params.fragmentUsseMem = patcher_memory[2];
    params.fragmentUsseMemSize = sizes[2];
    params.fragmentUsseOffset = fragment_offset;
    rc = sceGxmShaderPatcherCreate(&params, &own_patcher);
    if (rc < 0) {
        own_patcher = nullptr;
        return fail("crear", rc);
    }
    Common::VitaNote("gxm init", "parcheador propio: 1 MB de buffer y 2+2 MB de USSE");
    return true;
}

void RasterizerGXM::DestroyShaderPatcher() {
    if (own_patcher != nullptr) {
        sceGxmShaderPatcherDestroy(own_patcher);
        own_patcher = nullptr;
    }
    if (patcher_memory[0] != nullptr) {
        sceGxmUnmapMemory(patcher_memory[0]);
    }
    if (patcher_memory[1] != nullptr) {
        sceGxmUnmapVertexUsseMemory(patcher_memory[1]);
    }
    if (patcher_memory[2] != nullptr) {
        sceGxmUnmapFragmentUsseMemory(patcher_memory[2]);
    }
    patcher_memory = {};
    for (SceUID& uid : patcher_blocks) {
        if (uid >= 0) {
            sceKernelFreeMemBlock(uid);
            uid = -1;
        }
    }
}

bool RasterizerGXM::EnsureInitialized() {
    if (available) {
        return true;
    }
    if (!EnsureCgModule()) {
        // NO se marca como definitivo: el primer intento cae en plena carga
        // del juego, donde puede fallar por memoria, y el fichero esta ahi (el
        // presentador lo carga unas lineas despues). Se reintenta en cada lote.
        status = "sin libshacccg";
        static bool noted_shacccg = false;
        if (!noted_shacccg) {
            noted_shacccg = true;
            Common::VitaNote("gxm init", "sin libshacccg (se reintenta)");
        }
        return false;
    }
    if (pipelines != nullptr) {
        available = true;
        status = "gxm";
        return true;
    }
    context = vita2d_get_context();
    patcher = CreateShaderPatcher() ? own_patcher : vita2d_get_shader_patcher();
    // Una palabra de la region de notificaciones de GXM, lejos del principio
    // (vita2d y el sistema no la usan). La GPU solo puede escribir ahi.
    fence_address = sceGxmGetNotificationRegion() + 400;
    *fence_address = 0;
    if (context == nullptr || patcher == nullptr) {
        status = "sin contexto gxm";
        Common::VitaNote("gxm init", "sin contexto");
        return false;
    }
    pipelines = std::make_unique<PipelineCache>(context, patcher);
    if (!pipelines->Initialize()) {
        status = "error de shader";
        Common::VitaNote("gxm init", "shader de vertices");
        pipelines.reset();
        return false;
    }
    textures = std::make_unique<TextureCache>();
    hw_shaders = std::make_unique<HwShaderCache>(patcher);
    available = true;
    status = "gxm";
    Common::VitaNote("gxm init", "rasterizador listo");
    LOG_INFO(Render, "GXM: rasterizador de la GPU listo (shaders compilados en runtime)");
    return true;
}

void RasterizerGXM::NoteSkip(u32 index, const char* reason) {
    /**
     * UN HUECO POR MOTIVO ERA DEMASIADO POCO: los motivos se tapaban entre si.
     *
     * Antes esto marcaba el hueco (shader, wbuffer, framebuffer...) y no volvia
     * a anotar nada de ese hueco. Pero bajo "shader" caben motivos muy
     * distintos -- iluminacion, proctex, sombras, que el shader no compile, la
     * ablacion de diagnostico -- y el PRIMERO que apareciera silenciaba a todos
     * los demas para el resto de la partida.
     *
     * Eso costo una vuelta entera: en la medida de Pokemon Rubi Omega, con el
     * 96% de los triangulos cayendo a software, lo unico que llego a crash.txt
     * fue "ablacion" -- de un momento en que se estaba trasteando con el modo
     * de diagnostico -- y el motivo de verdad (el shader no compilaba) no se
     * escribio nunca.
     *
     * Ahora se recuerda el TEXTO de cada motivo ya anotado, asi que cada causa
     * distinta sale una vez. El indice de hueco se conserva porque lo usan los
     * sitios que llaman, pero ya no decide nada.
     */
    static_cast<void>(index);
    if (reason == nullptr) {
        return;
    }
    for (u32 i = 0; i < skip_reason_count; i++) {
        if (std::strcmp(skip_reasons[i], reason) == 0) {
            return;
        }
    }
    if (skip_reason_count >= kMaxSkipReasons) {
        return;
    }
    skip_reasons[skip_reason_count++] = reason;
    Common::VitaNote("gxm skip", reason);
}

void RasterizerGXM::EndScene() {
    if (open_surface == nullptr) {
        // Vertices repartidos sin llegar a dibujarse: no los lee nadie, pero
        // cualquier valla ya enviada sirve para soltarlos.
        for (u32 segment = 0; segment < kVertexSegments; segment++) {
            if (((vertex_segments_pending >> segment) & 1u) != 0) {
                vertex_segment_fence[segment] = fence_sent;
            }
        }
        vertex_segments_pending = 0;
        // Lo retirado sin escena abierta lo pudo leer, como mucho, la ultima
        // escena enviada.
        if (textures != nullptr && textures->HasRetired()) {
            textures->SealRetired(fence_sent);
            textures->ReleaseUpTo(*fence_address);
        }
        return;
    }
    /**
     * La espera a la GPU (0.1.5.2, punto 4.5).
     *
     * El camino viejo (no_finish_wait=0) hace sceGxmFinish AQUI, al cerrar
     * cada escena: son ~14 ms de CPU parada mirando al chip. El nuevo (1,
     * por defecto) cierra con sceGxmEndScene y deja la espera PENDIENTE:
     * WaitGpu() la paga solo cuando alguien vaya a LEER la superficie
     * (WriteBack/Reload) o a LIBERAR memoria que la GPU aun pueda estar
     * leyendo (ReleaseRetired). Asi la espera se solapa con el ARM emulado
     * en vez de bloquear el hilo de emulacion.
     *
     * La notificacion de antes estaba mal: su memoria estaba mapeada solo de
     * lectura para la GPU, y sceGxmNotificationWait no volvia nunca (la
     * consola se colgaba sin dejar volcado). sceGxmFinish no tiene ese
     * problema y es idempotente: si la GPU ya ha terminado, vuelve ya.
     */
    /**
     * Devolver el recorte de region a "sin recorte" ANTES de cerrar.
     *
     * El contexto es el de vita2d y lo comparten el rasterizador, el
     * presentador y la propia vita2d; ninguno de los otros dos toca el recorte
     * (el presentador solo repone culling y profundidad, ver gxm_presenter.cpp)
     * asi que la caja del ultimo lote de un juego se quedaria puesta y
     * recortaria la presentacion: pantalla negra salvo un rectangulo. Se repone
     * aqui, con la escena todavia abierta, que es donde el estado tiene efecto.
     */
    sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_NONE, 0, 0, 0, 0);
    /**
     * Y la plantilla, por el mismo motivo que el recorte: el contexto es
     * compartido y su estado no se puede leer hacia atras. Una plantilla que se
     * quede encendida con la caja del ultimo lote de un juego le recorta la
     * imagen al presentador, que dibuja sobre OTRA superficie -- la de la
     * pantalla -- donde esos valores no significan nada.
     *
     * "Pasa siempre y no toques nada" es el unico estado que no puede estropear
     * a quien venga detras, que es el mismo criterio con el que el presentador
     * repone el culling y la profundidad (ver gxm_presenter.cpp).
     */
    sceGxmSetFrontStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                              SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
    sceGxmSetBackStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                             SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
    {
        const Common::ScopedVitaStage stage{"gxm cerrar escena"};
        fence_sent++;
        SceGxmNotification done{};
        done.address = fence_address;
        done.value = fence_sent;
        sceGxmEndScene(context, nullptr, &done);
    }
    if (frame_first_fence == 0) {
        frame_first_fence = fence_sent;
    }
    // Los tramos del anillo de vertices que ha leido esta escena (0.2.1.1).
    for (u32 segment = 0; segment < kVertexSegments; segment++) {
        if (((vertex_segments_pending >> segment) & 1u) != 0) {
            vertex_segment_fence[segment] = fence_sent;
        }
    }
    vertex_segments_pending = 0;
    // Las versiones de las tablas de luz que ha atado esta escena: la actual
    // si la ha usado, y las retiradas a mitad de escena.
    for (u32 back = lighting_lut_in_scene ? 0 : 1; back <= lighting_lut_retired_in_scene;
         back++) {
        lut_fence[(lighting_lut_version + kLutVersions - back) % kLutVersions] = fence_sent;
    }
    lighting_lut_in_scene = false;
    lighting_lut_retired_in_scene = 0;
    gpu_scenes.fetch_add(1, std::memory_order_relaxed);
    const bool defer = no_finish_wait.load(std::memory_order_relaxed) != 0;
    if (defer) {
        gpu_pending = true;
    } else {
        // Camino de 0.1.5.1: esperar aqui mismo, medido en finish_us.
        // gpu_pending tiene que estar en uno para que WaitGpu no se salte.
        gpu_pending = true;
        WaitGpu();
    }
    /**
     * Volver a encender la carga forzada.
     *
     * Si esta escena venia de un borrado (ClearDepthIfNeeded la apago para que
     * GXM rellenara los tiles con los valores de fondo), dejarla apagada
     * borraria la profundidad OTRA VEZ al empezar la siguiente escena, y como
     * cada lote abre y cierra la suya, eso es borrarla entre lote y lote: los
     * triangulos dejarian de taparse unos a otros. Se repone aqui, con la
     * escena ya cerrada. No depende de que la GPU haya terminado: es estado
     * del contexto para el proximo sceGxmBeginScene.
     */
    sceGxmDepthStencilSurfaceSetForceLoadMode(&open_surface->depth_surface,
                                              SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
    open_surface->scene_open = false;
    open_surface = nullptr;
    /**
     * Y la memoria de las texturas retiradas: se suelta cuando haya pasado la
     * valla de la escena que pudo leerla (0.1.9.4), sin esperar a nadie.
     */
    if (textures != nullptr && textures->HasRetired()) {
        textures->SealRetired(fence_sent);
        textures->ReleaseUpTo(*fence_address);
    }
}

bool RasterizerGXM::FenceDone(u32 fence) const {
    return static_cast<s32>(*fence_address - fence) >= 0;
}

void RasterizerGXM::WaitFence(u32 fence) {
    if (FenceDone(fence)) {
        return;
    }
    // Las escenas acaban en orden: esperar a todo cubre esta.
    gpu_pending = true;
    WaitGpu();
}

void RasterizerGXM::WaitGpu() {
    if (!gpu_pending) {
        return;
    }
    gpu_pending = false;
    // La espera a la GPU, medida aparte: ver Common::FrameStats::finish_us.
    // sceGxmFinish es idempotente: si la GPU ya ha terminado, vuelve ya.
    const unsigned long long finish_begin = Common::VitaMicros();
    const Common::ScopedVitaStage stage{"gxm espera a la gpu"};
    sceGxmFinish(context);
    Common::FrameStats::Add(Common::FrameStats::finish_us, finish_begin);
}

void RasterizerGXM::WriteBack(Surface& surface) {
    if (!surface.dirty) {
        return;
    }
    if (surface.clear_pending) {
        ApplyClearOnCpu(surface);
    }
    // Lee color_buffer: la GPU pudo no haber terminado si 4.5 difirio la espera.
    WaitGpu();
    surface.dirty = false;
    u8* guest = memory.GetPhysicalPointer(surface.guest_address);
    if (guest == nullptr) {
        return;
    }
    gpu_writebacks.fetch_add(1, std::memory_order_relaxed);
    u8* linear = static_cast<u8*>(surface.color_buffer.Data());
    u32 linear_stride = surface.color_stride * surface.rt_bpp;
    if (surface.scale != 2) {
        const u32 row_bytes = surface.width * surface.rt_bpp;
        scale_scratch.resize(static_cast<std::size_t>(row_bytes) * surface.height);
        Resample(linear, linear_stride, Phys(surface.width, surface.scale),
                 Phys(surface.height, surface.scale), scale_scratch.data(), row_bytes,
                 surface.width, surface.height, surface.rt_bpp);
        linear = scale_scratch.data();
        linear_stride = row_bytes;
    }
    CopyTiledGuest(guest, linear, surface.width, surface.height, linear_stride, surface.bpp,
                   true);
}

void RasterizerGXM::Reload(Surface& surface) {
    surface.copied = false;
    surface.clear_pending = false;
    surface.clear_count = 0;
    // Lo que lee del invitado tiene que estar escrito: una copia de pantalla
    // en la GPU que pise el tramo, primero a la memoria.
    MaterializeCopies(surface.guest_address, surface.guest_stride * surface.height);
    // Lee y escribe color_buffer: misma razon que WriteBack.
    WaitGpu();
    surface.needs_reload = false;
    u8* guest = memory.GetPhysicalPointer(surface.guest_address);
    if (guest == nullptr) {
        return;
    }
    if (surface.scale == 2) {
        CopyTiledGuest(guest, static_cast<u8*>(surface.color_buffer.Data()), surface.width,
                       surface.height, surface.color_stride * surface.rt_bpp, surface.bpp, false);
        return;
    }
    const u32 row_bytes = surface.width * surface.rt_bpp;
    scale_scratch.resize(static_cast<std::size_t>(row_bytes) * surface.height);
    CopyTiledGuest(guest, scale_scratch.data(), surface.width, surface.height, row_bytes,
                   surface.bpp, false);
    Resample(scale_scratch.data(), row_bytes, surface.width, surface.height,
             static_cast<u8*>(surface.color_buffer.Data()), surface.color_stride * surface.rt_bpp,
             Phys(surface.width, surface.scale), Phys(surface.height, surface.scale),
             surface.rt_bpp);
}

namespace {
/// El formato de textura de una superficie de color, para presentarla.
bool PresentTextureFormat(u32 color_format, SceGxmTextureFormat& out) {
    switch (color_format) {
    case SCE_GXM_COLOR_FORMAT_U8U8U8U8_RGBA:
        out = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA;
        return true;
    case SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB:
        out = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ARGB;
        return true;
    case SCE_GXM_COLOR_FORMAT_U5U5U5U1_RGBA:
        out = SCE_GXM_TEXTURE_FORMAT_U5U5U5U1_RGBA;
        return true;
    case SCE_GXM_COLOR_FORMAT_U5U6U5_RGB:
        out = SCE_GXM_TEXTURE_FORMAT_U5U6U5_RGB;
        return true;
    case SCE_GXM_COLOR_FORMAT_U4U4U4U4_RGBA:
        out = SCE_GXM_TEXTURE_FORMAT_U4U4U4U4_RGBA;
        return true;
    default:
        return false;
    }
}
} // Anonymous namespace

RasterizerGXM::DirectPresent RasterizerGXM::QueryDirectPresent(u32 guest_address) {
    DirectPresent out;
    if (s_instance == nullptr || guest_address == 0) {
        return out;
    }
    RasterizerGXM& self = *s_instance;
    /**
     * Una pantalla copiada por la GPU (0.1.8.7): su buffer, tal cual. Sin
     * esperar a la GPU: el presentador dibuja en una escena posterior a la del
     * blit, y la GPU las hace en orden.
     */
    for (const auto& copy : self.screen_copies) {
        if (!copy->valid || copy->dst != guest_address) {
            continue;
        }
        SceGxmTextureFormat tex_format{};
        if (!PresentTextureFormat(copy->gxm_color_format, tex_format)) {
            break;
        }
        out.data = static_cast<const u8*>(copy->color_buffer.Data());
        out.width = Phys(copy->width, copy->scale);
        out.height = Phys(copy->height, copy->scale);
        out.scale = copy->scale;
        out.stride_bytes = copy->stride * copy->bpp;
        out.gxm_texture_format = static_cast<u32>(tex_format);
        return out;
    }
    if (present_direct.load(std::memory_order_relaxed) == 0) {
        return out;
    }
    // La textura va a muestrear color_buffer: si 4.5 dejo la espera pendiente,
    // hay que pagarla aqui o el chip podria seguir escribiendo debajo.
    self.WaitGpu();
    for (const auto& surface : self.surfaces) {
        if (surface->guest_address != guest_address) {
            continue;
        }
        // needs_reload = el invitado ha cambiado el framebuffer por otro camino
        // (relleno, transferencia) desde la ultima vez: la superficie de la GPU
        // esta detras del invitado y no sirve para presentar.
        if (surface->needs_reload || !surface->color_buffer.Valid()) {
            return out;
        }
        // El formato de textura no es el mismo enum que el de color surface:
        // se traduce aqui, una vez, en vez de guardarlo en dos sitios.
        SceGxmTextureFormat tex_format = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA;
        switch (surface->gxm_color_format) {
        case SCE_GXM_COLOR_FORMAT_U8U8U8U8_RGBA:
            tex_format = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA;
            break;
        case SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB:
            tex_format = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ARGB;
            break;
        case SCE_GXM_COLOR_FORMAT_U5U5U5U1_RGBA:
            tex_format = SCE_GXM_TEXTURE_FORMAT_U5U5U5U1_RGBA;
            break;
        case SCE_GXM_COLOR_FORMAT_U5U6U5_RGB:
            tex_format = SCE_GXM_TEXTURE_FORMAT_U5U6U5_RGB;
            break;
        case SCE_GXM_COLOR_FORMAT_U4U4U4U4_RGBA:
            tex_format = SCE_GXM_TEXTURE_FORMAT_U4U4U4U4_RGBA;
            break;
        default:
            return out;
        }
        out.data = static_cast<const u8*>(surface->color_buffer.Data());
        out.width = Phys(surface->width, surface->scale);
        out.height = Phys(surface->height, surface->scale);
        out.scale = surface->scale;
        out.stride_bytes = surface->color_stride * surface->rt_bpp;
        out.gxm_texture_format = static_cast<u32>(tex_format);
        return out;
    }
    return out;
}

bool RasterizerGXM::IsDisplayFramebuffer(PAddr addr) {
    for (u32 i = 0; i < 2; i++) {
        const auto& framebuffer = pica.regs.framebuffer_config[i];
        for (const PAddr shown : {framebuffer.address_left1, framebuffer.address_left2}) {
            if (shown == 0 || std::find(display_addresses.begin(), display_addresses.end(),
                                        shown) != display_addresses.end()) {
                continue;
            }
            display_addresses[display_address_next] = shown;
            display_address_next = (display_address_next + 1) % display_addresses.size();
        }
    }
    return addr != 0 &&
           std::find(display_addresses.begin(), display_addresses.end(), addr) !=
               display_addresses.end();
}

void RasterizerGXM::DropCopies(PAddr addr, u32 size) {
    for (auto& copy : screen_copies) {
        if (copy->valid && addr < copy->dst + copy->dst_size && copy->dst < addr + size) {
            copy->valid = false;
            copy->pending = false;
        }
    }
}

/**
 * La copia de verdad a la memoria del invitado, desde el buffer de la copia.
 * La pantalla es lineal, asi que es fila a fila: los mismos bytes que el
 * invitado (CopyTiledGuest los mueve tal cual), y de RGBA8 a RGB8 los bytes 1,
 * 2 y 3, como UntileRGBA8ToRGB8. La copia sigue valiendo para presentar.
 */
void RasterizerGXM::MaterializeCopy(ScreenCopy& copy) {
    if (!copy.valid || !copy.pending) {
        return;
    }
    copy.pending = false;
    if (copy.width > 1024 || !GuestSpanMapped(memory, copy.dst, copy.dst_size)) {
        return;
    }
    // El blit tiene que haber terminado.
    WaitGpu();
    u8* const dst = memory.GetPhysicalPointer(copy.dst);
    const u8* const src = static_cast<const u8*>(copy.color_buffer.Data());
    const u32 src_stride = copy.stride * copy.bpp;
    const u32 out_bpp = Pica::BytesPerPixel(copy.output_format);
    const bool rgba_to_rgb = copy.input_format == Pica::PixelFormat::RGBA8 &&
                             copy.output_format == Pica::PixelFormat::RGB8;
    alignas(16) u8 row[2048 * 4];
    alignas(16) u8 scaled[2048 * 4];
    for (u32 y = 0; y < copy.height; y++) {
        // CDRAM se lee sin cache: la fila de una vez y despues de memoria normal.
        if (copy.scale == 2) {
            std::memcpy(row, src + static_cast<std::size_t>(y) * src_stride,
                        copy.width * copy.bpp);
        } else {
            const u32 phys_width = Phys(copy.width, copy.scale);
            std::memcpy(scaled, src + static_cast<std::size_t>(Phys(y, copy.scale)) * src_stride,
                        phys_width * copy.bpp);
            for (u32 x = 0; x < copy.width; x++) {
                std::memcpy(row + x * copy.bpp, scaled + Phys(x, copy.scale) * copy.bpp,
                            copy.bpp);
            }
        }
        u8* out = dst + static_cast<std::size_t>(y) * copy.width * out_bpp;
        if (rgba_to_rgb) {
            for (u32 x = 0; x < copy.width; x++) {
                out[x * 3 + 0] = row[x * 4 + 1];
                out[x * 3 + 1] = row[x * 4 + 2];
                out[x * 3 + 2] = row[x * 4 + 3];
            }
        } else if (out_bpp == copy.bpp) {
            std::memcpy(out, row, copy.width * out_bpp);
        } else {
            // RGB8: la superficie lleva 4 bytes por pixel y los 3 primeros son
            // los del invitado (ver CurrentSurface).
            for (u32 x = 0; x < copy.width; x++) {
                std::memcpy(out + x * 3, row + x * 4, 3);
            }
        }
    }
    transfer_materialized.fetch_add(1, std::memory_order_relaxed);
    // Lo que se hubiera quedado con la memoria vieja de esa pantalla.
    if (textures != nullptr) {
        textures->InvalidateRange(copy.dst, copy.dst_size);
    }
    software.InvalidateRegion(copy.dst, copy.dst_size);
}

void RasterizerGXM::MaterializeCopies(PAddr addr, u32 size) {
    for (auto& copy : screen_copies) {
        if (copy->pending && addr < copy->dst + copy->dst_size && copy->dst < addr + size) {
            MaterializeCopy(*copy);
        }
    }
}

void RasterizerGXM::MaterializeAllCopies() {
    for (auto& copy : screen_copies) {
        MaterializeCopy(*copy);
    }
}

bool RasterizerGXM::EnsureBlitProgram() {
    if (blit != nullptr) {
        return true;
    }
    if (blit_failed) {
        return false;
    }
    // Un solo intento: si falla aqui, fallara igual en el siguiente lote.
    blit_failed = true;
    const auto fail = [](const char* why, int rc) {
        NoteFmt("gxm copia", "blit: {} ({:#x}); las pantallas siguen por software", why,
                static_cast<u32>(rc));
        return false;
    };
    auto program = std::make_unique<BlitProgram>();
    program->patcher = patcher;
    program->vertex_output = CompileCg(SCE_SHACCCG_PROFILE_VP, "azahar_blit_v.cg", kBlitVertexSource);
    program->fragment_output =
        CompileCg(SCE_SHACCCG_PROFILE_FP, "azahar_blit_f.cg", kBlitFragmentSource);
    if (program->vertex_output == nullptr || program->fragment_output == nullptr) {
        return fail("sin shaders", 0);
    }
    const auto* vertex_gxp =
        reinterpret_cast<const SceGxmProgram*>(program->vertex_output->programData);
    const auto* fragment_gxp =
        reinterpret_cast<const SceGxmProgram*>(program->fragment_output->programData);
    int rc = sceGxmShaderPatcherRegisterProgram(patcher, vertex_gxp, &program->vertex_id);
    if (rc != 0) {
        return fail("registro de vertices", rc);
    }
    program->vertex_registered = true;
    rc = sceGxmShaderPatcherRegisterProgram(patcher, fragment_gxp, &program->fragment_id);
    if (rc != 0) {
        return fail("registro de fragmentos", rc);
    }
    program->fragment_registered = true;

    const SceGxmProgramParameter* position_param =
        sceGxmProgramFindParameterByName(vertex_gxp, "position");
    const SceGxmProgramParameter* texcoord_param =
        sceGxmProgramFindParameterByName(vertex_gxp, "texcoord");
    if (position_param == nullptr || texcoord_param == nullptr) {
        return fail("sin position/texcoord", 0);
    }
    SceGxmVertexAttribute attributes[2]{};
    attributes[0].streamIndex = 0;
    attributes[0].offset = 0;
    attributes[0].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
    attributes[0].componentCount = 4;
    attributes[0].regIndex = sceGxmProgramParameterGetResourceIndex(position_param);
    attributes[1].streamIndex = 0;
    attributes[1].offset = 4 * sizeof(float);
    attributes[1].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
    attributes[1].componentCount = 2;
    attributes[1].regIndex = sceGxmProgramParameterGetResourceIndex(texcoord_param);
    SceGxmVertexStream stream{};
    stream.stride = static_cast<u16>(kBlitVertexStride);
    stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
    rc = sceGxmShaderPatcherCreateVertexProgram(patcher, program->vertex_id, attributes, 2,
                                                &stream, 1, &program->vertex_program);
    if (rc != 0) {
        return fail("programa de vertices", rc);
    }
    // Sin mezcla: los bytes de la superficie, tal cual, alfa incluido.
    SceGxmBlendInfo blend{};
    blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
    blend.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
    blend.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
    blend.colorSrc = SCE_GXM_BLEND_FACTOR_ONE;
    blend.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
    blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
    blend.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
    rc = sceGxmShaderPatcherCreateFragmentProgram(
        patcher, program->fragment_id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
        SCE_GXM_MULTISAMPLE_NONE, &blend, vertex_gxp, &program->fragment_program);
    if (rc != 0) {
        return fail("programa de fragmentos", rc);
    }
    const SceGxmProgramParameter* sampler_param =
        sceGxmProgramFindParameterByName(fragment_gxp, "tex");
    program->texture_unit =
        sampler_param != nullptr ? sceGxmProgramParameterGetResourceIndex(sampler_param) : 0;

    /**
     * La fila 0 de la superficie (v = 0, el principio de su memoria) va a la
     * fila 0 de la copia, que es la de arriba: y = +1 con el viewport de
     * BlitToCopy, igual que en el presentador. Volteado, al reves, como hace
     * la copia por software con flip_vertically.
     */
    static constexpr float kQuads[8][6] = {
        {-1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f},
        {-1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f},
    };
    static constexpr u16 kIndices[4] = {0, 1, 2, 3};
    static_assert(sizeof(kQuads) == 8 * kBlitVertexStride);
    program->quad = Allocate(Pool::Host, sizeof(kQuads) + sizeof(kIndices));
    if (!program->quad.Valid()) {
        return fail("sin memoria", 0);
    }
    std::memcpy(program->quad.Data(), kQuads, sizeof(kQuads));
    std::memcpy(static_cast<u8*>(program->quad.Data()) + sizeof(kQuads), kIndices,
                sizeof(kIndices));
    blit = std::move(program);
    blit_failed = false;
    return true;
}

RasterizerGXM::ScreenCopy* RasterizerGXM::GetScreenCopy(PAddr dst, u32 width, u32 height,
                                                       u32 gxm_color_format, u32 bpp,
                                                       u32 scale) {
    std::unique_ptr<ScreenCopy>* slot = nullptr;
    for (auto& copy : screen_copies) {
        if (copy->dst == dst) {
            slot = &copy;
            break;
        }
    }
    if (slot != nullptr && (*slot)->width == width && (*slot)->height == height &&
        (*slot)->gxm_color_format == gxm_color_format && (*slot)->scale == scale) {
        return slot->get();
    }

    // Una nueva, con las mismas cuentas que la superficie de color de
    // CurrentSurface (lineal, stride a 8 pixeles) y sin profundidad.
    static bool noted = false;
    auto copy = std::make_unique<ScreenCopy>();
    copy->dst = dst;
    copy->width = width;
    copy->height = height;
    copy->gxm_color_format = gxm_color_format;
    copy->scale = scale;
    const u32 phys_width = Phys(width, scale);
    const u32 phys_height = Phys(height, scale);
    copy->stride = (phys_width + 7) & ~7u;
    copy->bpp = bpp;
    copy->color_buffer =
        Allocate(Pool::Cdram, copy->stride * phys_height * bpp, SCE_GXM_MEMORY_ATTRIB_RW);
    if (!copy->color_buffer.Valid()) {
        NoteOnce(noted, "gxm copia", "sin memoria para {}x{}", width, height);
        return nullptr;
    }
    int rc = sceGxmColorSurfaceInit(&copy->color_surface,
                                    static_cast<SceGxmColorFormat>(gxm_color_format),
                                    SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_COLOR_SURFACE_SCALE_NONE,
                                    SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, phys_width, phys_height,
                                    copy->stride, copy->color_buffer.Data());
    if (rc != 0) {
        NoteOnce(noted, "gxm copia", "color {}x{} err {:#x}", width, height, static_cast<u32>(rc));
        return nullptr;
    }
    rc = sceGxmDepthStencilSurfaceInitDisabled(&copy->depth_surface);
    if (rc != 0) {
        NoteOnce(noted, "gxm copia", "prof desactivada err {:#x}", static_cast<u32>(rc));
        return nullptr;
    }
    SceGxmRenderTargetParams params{};
    params.flags = 0;
    params.width = static_cast<u16>(phys_width);
    params.height = static_cast<u16>(phys_height);
    params.scenesPerFrame = 8;
    params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    params.multisampleLocations = 0;
    unsigned int driver_size = 0;
    rc = sceGxmGetRenderTargetMemSize(&params, &driver_size);
    if (rc < 0 || driver_size == 0) {
        NoteOnce(noted, "gxm copia", "rt memsize err {:#x}", static_cast<u32>(rc));
        return nullptr;
    }
    const SceSize driver_bytes = (driver_size + 0xFFFu) & ~0xFFFu;
    const SceUID driver_uid = sceKernelAllocMemBlock(
        "azahar_gxm_copy", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, driver_bytes, nullptr);
    if (driver_uid < 0) {
        NoteOnce(noted, "gxm copia", "rt memblock err {:#x}", static_cast<u32>(driver_uid));
        return nullptr;
    }
    copy->driver_uid = driver_uid;
    params.driverMemBlock = driver_uid;
    rc = sceGxmCreateRenderTarget(&params, &copy->render_target);
    if (rc != 0) {
        copy->render_target = nullptr;
        NoteOnce(noted, "gxm copia", "rt {}x{} err {:#x}", width, height, static_cast<u32>(rc));
        return nullptr;
    }

    if (slot == nullptr && screen_copies.size() < kMaxScreenCopies) {
        screen_copies.push_back(std::move(copy));
        return screen_copies.back().get();
    }
    if (slot == nullptr) {
        slot = &screen_copies.front();
        for (auto& candidate : screen_copies) {
            if (candidate->last_use < (*slot)->last_use) {
                slot = &candidate;
            }
        }
    }
    // La que se va puede tenerla todavia la GPU (un blit o el presentador), y
    // su pantalla puede no estar escrita en el invitado.
    MaterializeCopy(**slot);
    EndScene();
    const Common::ScopedVitaStage stage{"gxm copia: rehacer"};
    sceGxmFinish(context);
    gpu_pending = false;
    *slot = std::move(copy);
    return slot->get();
}

bool RasterizerGXM::BlitToCopy(ScreenCopy& copy, Surface& source, u32 first_row,
                               bool flip) {
    if (source.clear_pending) {
        FlushClear(source);
    }
    SceGxmTextureFormat tex_format{};
    if (!PresentTextureFormat(source.gxm_color_format, tex_format)) {
        return false;
    }
    SceGxmTexture& texture = copy.source_texture;
    const u32 source_stride = source.color_stride * source.rt_bpp;
    const u8* first = static_cast<const u8*>(source.color_buffer.Data()) +
                      static_cast<std::size_t>(Phys(first_row, source.scale)) * source_stride;
    if (sceGxmTextureInitLinearStrided(&texture, first, tex_format, Phys(copy.width, copy.scale),
                                       Phys(copy.height, copy.scale), source_stride) < 0) {
        return false;
    }
    sceGxmTextureSetMinFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMagFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetUAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
    sceGxmTextureSetVAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);

    // Lo dibujado en la superficie se manda antes que el blit que lo lee.
    EndScene();
    const Common::ScopedVitaStage stage{"gxm copia: blit"};
    const int rc = sceGxmBeginScene(context, 0, copy.render_target, nullptr, nullptr, nullptr,
                                    &copy.color_surface, &copy.depth_surface);
    if (rc != 0) {
        static bool noted = false;
        NoteOnce(noted, "gxm copia", "beginscene err {:#x}", static_cast<u32>(rc));
        return false;
    }
    // El estado del contexto es compartido: todo lo que el quad necesita, aqui.
    const float half_width = static_cast<float>(Phys(copy.width, copy.scale)) * 0.5f;
    const float half_height = static_cast<float>(Phys(copy.height, copy.scale)) * 0.5f;
    sceGxmSetViewport(context, half_width, half_width, half_height, -half_height, 0.5f, 0.5f);
    sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_NONE, 0, 0, 0, 0);
    sceGxmSetFrontPolygonMode(context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
    sceGxmSetBackPolygonMode(context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
    sceGxmSetCullMode(context, SCE_GXM_CULL_NONE);
    sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetBackDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetBackDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetFrontStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                              SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
    sceGxmSetBackStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                             SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
    sceGxmSetVertexProgram(context, blit->vertex_program);
    sceGxmSetFragmentProgram(context, blit->fragment_program);
    const u8* quad = static_cast<const u8*>(blit->quad.Data());
    sceGxmSetVertexStream(context, 0, quad + (flip ? 4 * kBlitVertexStride : 0));
    sceGxmSetFragmentTexture(context, blit->texture_unit, &texture);
    sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP, SCE_GXM_INDEX_FORMAT_U16,
               quad + 8 * kBlitVertexStride, 4);
    sceGxmEndScene(context, nullptr, nullptr);
    gpu_pending = true;
    return true;
}

bool RasterizerGXM::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    const PAddr src = config.GetPhysicalInputAddress();
    const PAddr dst = config.GetPhysicalOutputAddress();
    const u32 width = config.output_width;
    const u32 height = config.output_height;
    const Pica::PixelFormat in_format = config.input_format;
    const Pica::PixelFormat out_format = config.output_format;
    Surface* source = nullptr;

    /**
     * Cada motivo de rechazo, UNA vez en crash.txt y con los numeros: en
     * 0.1.8.6 la copia en la GPU no se hizo ni una vez y no habia forma de
     * saber por que.
     */
    const auto reject = [&](const char* why) {
        static const char* noted[12] = {};
        static u32 noted_count = 0;
        for (u32 i = 0; i < noted_count; i++) {
            if (noted[i] == why) {
                return false;
            }
        }
        if (noted_count < 12) {
            noted[noted_count++] = why;
            if (source != nullptr) {
                NoteFmt("gxm copia", "no ({}): {:#x}->{:#x} {}x{} fmt {}->{}, sup {}x{} bpp {} rec {}",
                        why, src, dst, width, height, static_cast<u32>(in_format),
                        static_cast<u32>(out_format), source->width, source->height,
                        source->bpp, source->needs_reload ? 1 : 0);
            } else {
                NoteFmt("gxm copia", "no ({}): {:#x}->{:#x} {}x{} de {}x{} fmt {}->{}", why, src,
                        dst, width, height, static_cast<u32>(config.input_width.Value()),
                        static_cast<u32>(config.input_height.Value()),
                        static_cast<u32>(in_format), static_cast<u32>(out_format));
            }
        }
        return false;
    };

    /**
     * Fotograma saltado (0.1.9.6): no se ha dibujado nada, la superficie solo
     * tiene el borrado. La copia a la pantalla no se hace, por ningun camino:
     * la pantalla se queda con el ultimo fotograma dibujado.
     */
    if (!SwRenderer::FrameSkip::ShouldRender() && available && IsDisplayFramebuffer(dst)) {
        return true;
    }
    if (transfer_on_gpu.load(std::memory_order_relaxed) == 0) {
        return reject("apagada");
    }
    if (!available) {
        return reject("sin gxm");
    }
    if (config.input_linear || config.dont_swizzle ||
        config.scaling != Pica::DisplayTransferConfig::NoScale) {
        return reject("modo");
    }
    // El mismo formato, o RGBA8 a RGB8. Los de 16 bits desde 0.1.9.8
    // (New Super Mario Bros. 2 dibuja en RGB565).
    if (in_format != out_format &&
        !(in_format == Pica::PixelFormat::RGBA8 && out_format == Pica::PixelFormat::RGB8)) {
        return reject("formato");
    }
    if (!IsDisplayFramebuffer(dst)) {
        return reject("no es pantalla");
    }

    /**
     * El origen puede empezar DENTRO de la superficie (0.1.8.9). Zafiro Alfa
     * dibuja en 256x512 y copia las 400 filas de abajo: el origen esta 112
     * filas despues del principio ("sin superficie" en 0.1.8.7). La memoria
     * del invitado va en filas de mosaicos de 8 y nuestra fila y es la fila y
     * del invitado (CopyTiledGuest no voltea), asi que basta con que el salto
     * sea de filas de mosaicos enteras.
     */
    u32 first_row = 0;
    for (auto& surface : surfaces) {
        const u32 row_bytes = surface->width * surface->bpp;
        if (src < surface->guest_address || row_bytes == 0 ||
            src >= surface->guest_address + row_bytes * surface->height) {
            continue;
        }
        const u32 offset = src - surface->guest_address;
        if (offset % (row_bytes * 8) != 0) {
            continue;
        }
        source = surface.get();
        first_row = offset / row_bytes;
        break;
    }
    if (source == nullptr) {
        return reject("sin superficie");
    }
    // Con la memoria del invitado por delante de la nuestra: por software.
    if (source->needs_reload || !source->color_buffer.Valid()) {
        return reject("superficie por recargar");
    }
    if (source->width != config.input_width || source->bpp != Pica::BytesPerPixel(in_format) ||
        width == 0 || height == 0 || width > source->width ||
        first_row + height > source->height || width > 1024) {
        return reject("medidas");
    }
    // Con el mismo tamano de pixel, el formato de verdad de la superficie
    // tiene que ser el de la copia (en 16 bits hay tres).
    const auto source_format = static_cast<SceGxmColorFormat>(source->gxm_color_format);
    const bool same_format =
        (in_format == Pica::PixelFormat::RGBA8 && source_format == SCE_GXM_COLOR_FORMAT_U8U8U8U8_RGBA) ||
        (in_format == Pica::PixelFormat::RGB8 && source_format == SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB) ||
        (in_format == Pica::PixelFormat::RGB565 && source_format == SCE_GXM_COLOR_FORMAT_U5U6U5_RGB) ||
        (in_format == Pica::PixelFormat::RGB5A1 &&
         source_format == SCE_GXM_COLOR_FORMAT_U5U5U5U1_RGBA) ||
        (in_format == Pica::PixelFormat::RGBA4 && source_format == SCE_GXM_COLOR_FORMAT_U4U4U4U4_RGBA);
    if (!same_format) {
        return reject("formato de la superficie");
    }
    if (!EnsureBlitProgram()) {
        return reject("sin blit");
    }
    ScreenCopy* copy =
        GetScreenCopy(dst, width, height, source->gxm_color_format, source->rt_bpp, source->scale);
    if (copy == nullptr) {
        return reject("sin copia");
    }
    const u32 dst_size = width * height * Pica::BytesPerPixel(out_format);
    // Lo que hubiera en esa pantalla deja de valer, como en la copia por
    // software (la copia anterior a ella incluida).
    InvalidateRegion(dst, dst_size);
    if (!BlitToCopy(*copy, *source, first_row, config.flip_vertically != 0)) {
        return reject("escena");
    }
    copy->dst_size = dst_size;
    copy->input_format = in_format;
    copy->output_format = out_format;
    copy->valid = true;
    copy->pending = true;
    copy->last_use = ++copy_clock;
    source->copied = true;
    gpu_transfers.fetch_add(1, std::memory_order_relaxed);
    static bool noted = false;
    NoteOnce(noted, "gxm copia", "pantalla {:#010x} {}x{} desde {:#010x}, copiada en la GPU", dst,
             width, height, src);
    return true;
}

namespace {
/**
 * El patron de 'bpp' bytes repetido en [dst, dst + bytes), como
 * SwBlitter::MemoryFill. Por bloques de 192 bytes (multiplo de 2, 3 y 4):
 * 0.1.9.8 lo hacia con un memcpy de tamano variable por pixel, que el
 * compilador no puede convertir en una escritura, y los rellenos pasaron de
 * ~1 ms a ~11 ms por fotograma (Pokemon Sol del 80% al 50% de velocidad).
 */
void FillPattern(u8* dst, u32 bytes, u32 texel, u32 bpp) {
    alignas(16) u8 block[192];
    for (u32 i = 0; i < sizeof(block); i += bpp) {
        std::memcpy(block + i, &texel, bpp);
    }
    u32 done = 0;
    for (; done + sizeof(block) <= bytes; done += sizeof(block)) {
        std::memcpy(dst + done, block, sizeof(block));
    }
    std::memcpy(dst + done, block, bytes - done);
}
} // Anonymous namespace

bool RasterizerGXM::AccelerateFill(const Pica::MemoryFillConfig& config) {
    // Mismo interruptor que la copia de pantalla: los dos son pintar en la
    // GPU lo que antes hacia la CPU en la memoria del invitado.
    if (!available || transfer_on_gpu.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    const PAddr start = config.GetStartAddress();
    const PAddr end = config.GetEndAddress();
    /**
     * Filas de mosaicos enteras de una superficie (0.1.9.6): Zafiro Alfa
     * dibuja las dos pantallas en franjas distintas de la misma superficie de
     * 256x512, y puede borrar solo la suya. Nuestra fila y es la fila y del
     * invitado (CopyTiledGuest no voltea), asi que una franja de filas del
     * invitado es una franja de filas nuestra.
     */
    Surface* target = nullptr;
    u32 first_row = 0;
    u32 rows = 0;
    for (auto& surface : surfaces) {
        const u32 row_bytes = surface->guest_stride;
        const PAddr surface_end = surface->guest_address + row_bytes * surface->height;
        if (row_bytes == 0 || start < surface->guest_address || end > surface_end ||
            end <= start) {
            continue;
        }
        const u32 offset = start - surface->guest_address;
        const u32 bytes = end - start;
        if (offset % (row_bytes * 8) != 0 || bytes % (row_bytes * 8) != 0) {
            continue;
        }
        target = surface.get();
        first_row = offset / row_bytes;
        rows = bytes / row_bytes;
        break;
    }
    if (target == nullptr) {
        return false;
    }
    const bool whole = first_row == 0 && rows == target->height;
    // Una franja deja el resto como estaba: si el invitado iba por delante de
    // nosotros en ese resto, o ya hay demasiadas franjas apuntadas, software.
    if (!whole && (target->needs_reload || target->clear_count == target->clears.size())) {
        return false;
    }
    /**
     * El patron, como lo escribe SwBlitter::MemoryFill, en los bytes de
     * nuestro color_buffer: en RGBA8 la superficie guarda los mismos cuatro
     * bytes que el invitado; en RGB8 los tres del invitado y el alfa a 0xFF
     * (como CopyTiledGuest). Otros tamanos de patron o de superficie, por
     * software.
     */
    u32 texel = 0;
    if (target->bpp == 4 && config.fill_32bit && !config.fill_24bit) {
        texel = config.value_32bit;
    } else if (target->bpp == 3 && config.fill_24bit) {
        texel = static_cast<u32>(config.value_24bit_r.Value()) |
                (static_cast<u32>(config.value_24bit_g.Value()) << 8) |
                (static_cast<u32>(config.value_24bit_b.Value()) << 16) | 0xFF000000u;
    } else if (target->bpp == 2 && !config.fill_24bit && !config.fill_32bit) {
        // 16 bits (0.1.9.8): mismos dos bytes en el invitado y en la superficie.
        texel = config.value_16bit;
    } else {
        static bool noted = false;
        NoteOnce(noted, "gxm relleno", "por software: {}x{} bpp {} con patron de {} bits",
                 target->width, target->height, target->bpp,
                 config.fill_32bit ? 32 : (config.fill_24bit ? 24 : 16));
        return false;
    }
    if (!EnsureBlitProgram()) {
        return false;
    }
    if (!clear_texels.Valid()) {
        clear_texels = Allocate(Pool::Host, 256 * 16);
        if (!clear_texels.Valid()) {
            return false;
        }
    }
    if (!GuestSpanMapped(memory, start, end - start)) {
        return false;
    }
    if (open_surface == target) {
        EndScene();
    }
    /**
     * La memoria del invitado se rellena igual que por software (una escritura
     * seguida en RAM normal, sin esperar a nadie), asi que la superficie queda
     * al dia y sin nada que volcar; lo unico que falta es pintar el color en
     * nuestro color_buffer, y eso lo hace la GPU en la siguiente escena.
     */
    u8* const guest = memory.GetPhysicalPointer(start);
    const u32 bytes = end - start;
    FillPattern(guest, bytes, texel, target->bpp);
    if (whole) {
        // Todo el color es el relleno: lo de antes ya no cuenta.
        target->clear_count = 0;
        target->needs_reload = false;
        target->dirty = false;
    }
    target->clears[target->clear_count++] = {first_row, rows, texel};
    target->clear_pending = true;
    target->copied = false;
    // Lo demas que pudiera tener esa memoria, como en el relleno por software.
    DropCopies(start, end - start);
    if (textures != nullptr) {
        textures->InvalidateRange(start, end - start);
    }
    software.InvalidateRegion(start, end - start);
    gpu_fills.fetch_add(1, std::memory_order_relaxed);
    static bool noted = false;
    NoteOnce(noted, "gxm relleno", "superficie {:#010x} {}x{}, filas {}-{}, rellenada en la GPU",
             target->guest_address, target->width, target->height, first_row,
             first_row + rows - 1);
    return true;
}

/**
 * El quad del relleno, dentro de la escena ya abierta en 'surface' y antes que
 * cualquier lote: el blit con una textura de 1x1 del color, sin mezcla y sin
 * tocar profundidad ni plantilla. El texel sale de un anillo de 256: cuando se
 * reutiliza, la GPU ya ha terminado ese fotograma hace mucho (FlushForPresent
 * espera a la primera escena de cada fotograma).
 */
void RasterizerGXM::DrawClearQuad(Surface& surface) {
    surface.clear_pending = false;
    const u32 count = surface.clear_count;
    surface.clear_count = 0;
    SceGxmTextureFormat tex_format{};
    if (!PresentTextureFormat(surface.gxm_color_format, tex_format)) {
        return;
    }
    sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_NONE, 0, 0, 0, 0);
    sceGxmSetFrontPolygonMode(context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
    sceGxmSetBackPolygonMode(context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
    sceGxmSetCullMode(context, SCE_GXM_CULL_NONE);
    sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetBackDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetBackDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetFrontStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                              SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
    sceGxmSetBackStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                             SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
    sceGxmSetVertexProgram(context, blit->vertex_program);
    sceGxmSetFragmentProgram(context, blit->fragment_program);
    const u8* quad = static_cast<const u8*>(blit->quad.Data());
    sceGxmSetVertexStream(context, 0, quad);
    const float half_width = static_cast<float>(Phys(surface.width, surface.scale)) * 0.5f;
    for (u32 i = 0; i < count; i++) {
        const Surface::PendingClear& clear = surface.clears[i];
        u8* texel = static_cast<u8*>(clear_texels.Data()) + (clear_texel_next % 256) * 16;
        clear_texel_next++;
        std::memcpy(texel, &clear.texel, sizeof(u32));
        SceGxmTexture texture{};
        if (sceGxmTextureInitLinear(&texture, texel, tex_format, 1, 1, 1) < 0) {
            continue;
        }
        sceGxmTextureSetMinFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
        sceGxmTextureSetMagFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
        // El quad (de -1 a 1) cubre justo las filas de la franja.
        const float half_rows = static_cast<float>(Phys(clear.rows, surface.scale)) * 0.5f;
        sceGxmSetViewport(context, half_width, half_width,
                          static_cast<float>(Phys(clear.first_row, surface.scale)) + half_rows,
                          -half_rows, 0.5f, 0.5f);
        sceGxmSetFragmentTexture(context, blit->texture_unit, &texture);
        sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP, SCE_GXM_INDEX_FORMAT_U16,
                   quad + 8 * kBlitVertexStride, 4);
    }
}

/// Una escena solo para pintar el relleno (cuando algo va a leer la
/// superficie en la GPU sin que nadie haya dibujado en ella).
void RasterizerGXM::FlushClear(Surface& surface) {
    EndScene();
    ClearDepthIfNeeded(surface);
    if (sceGxmBeginScene(context, 0, surface.render_target, nullptr, nullptr, nullptr,
                         &surface.color_surface, &surface.depth_surface) != 0) {
        ApplyClearOnCpu(surface);
        return;
    }
    surface.scene_open = true;
    open_surface = &surface;
    DrawClearQuad(surface);
    EndScene();
}

/// El relleno a mano, para quien vaya a leer color_buffer con la CPU.
void RasterizerGXM::ApplyClearOnCpu(Surface& surface) {
    surface.clear_pending = false;
    const u32 count = surface.clear_count;
    surface.clear_count = 0;
    if (open_surface == &surface) {
        EndScene();
    }
    WaitGpu();
    u8* const base = static_cast<u8*>(surface.color_buffer.Data());
    const u32 stride = surface.color_stride * surface.rt_bpp;
    alignas(16) u8 row[2048 * 4];
    const u32 width = std::min<u32>(Phys(surface.width, surface.scale), 2048);
    const u32 height = Phys(surface.height, surface.scale);
    for (u32 i = 0; i < count; i++) {
        const Surface::PendingClear& clear = surface.clears[i];
        for (u32 x = 0; x < width; x++) {
            std::memcpy(row + x * surface.rt_bpp, &clear.texel, surface.rt_bpp);
        }
        const u32 last = Phys(clear.first_row + clear.rows, surface.scale);
        for (u32 y = Phys(clear.first_row, surface.scale); y < last && y < height; y++) {
            std::memcpy(base + static_cast<std::size_t>(y) * stride, row, width * surface.rt_bpp);
        }
    }
}

bool RasterizerGXM::UpdateLightingLut() {
    /**
     * Una sola textura para las 24 tablas: 256 columnas (la entrada) por 24
     * filas (el indice de tabla, que es el mismo numero del enumerado
     * LightingSampler). Cada texel son dos floats: el valor y la pendiente
     * hasta la entrada siguiente, que es como guarda la PICA sus tablas y lo
     * que permite interpolar en el shader igual que hace el rasterizador de
     * software.
     */
    constexpr u32 kLutEntries = 256;
    constexpr u32 kLutCount = 24;
    constexpr u32 kLutBytes = kLutEntries * kLutCount * 2 * sizeof(f32);

    if (!lighting_lut_ready) {
        // CDRAM primero: esto lo lee la GPU en cada pixel iluminado y lo
        // escribe la CPU solo cuando el juego cambia una tabla. Van las
        // kLutVersions copias en un solo bloque (8 x 48 KB, medio mega de
        // CDRAM con su grano de 256 KB).
        lighting_lut_buffer = Allocate(Pool::Cdram, kLutBytes * kLutVersions);
        if (!lighting_lut_buffer.Valid()) {
            lighting_lut_buffer = Allocate(Pool::Host, kLutBytes * kLutVersions);
        }
        if (!lighting_lut_buffer.Valid()) {
            NoteOnce(fb_noted[12], "gxm luz", "sin memoria para las tablas");
            return false;
        }
        /**
         * El nombre del formato lista los canales del mas significativo al
         * menos, igual que el ABGR del cache de texturas, asi que _GR es el
         * orden natural: el PRIMER float del texel es .r y el segundo .g.
         * En 0.1.0.41 la cinematica de Rubi Omega ya sale iluminada en consola
         * con este orden; si alguna vez salieran valor y pendiente cambiados,
         * el arreglo es pasar a _00RG o leer .gr en el shader.
         */
        for (u32 version = 0; version < kLutVersions; version++) {
            SceGxmTexture& texture = lighting_lut_textures[version];
            u8* base = static_cast<u8*>(lighting_lut_buffer.Data()) +
                       static_cast<std::size_t>(version) * kLutBytes;
            const int rc = sceGxmTextureInitLinear(&texture, base,
                                                   SCE_GXM_TEXTURE_FORMAT_F32F32_GR, kLutEntries,
                                                   kLutCount, 1);
            if (rc < 0) {
                NoteOnce(fb_noted[12], "gxm luz", "textura de tablas err {:#x}",
                         static_cast<u32>(rc));
                lighting_lut_buffer = Allocation{};
                return false;
            }
            // Punto y sin mipmaps: la interpolacion la hace el shader con la
            // pendiente guardada, no el filtro del chip. Con filtro lineal, cada
            // lectura mezclaria ademas con la tabla de la fila de al lado.
            sceGxmTextureSetMinFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
            sceGxmTextureSetMagFilter(&texture, SCE_GXM_TEXTURE_FILTER_POINT);
            sceGxmTextureSetMipFilter(&texture, SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
            sceGxmTextureSetMipmapCount(&texture, 1);
            sceGxmTextureSetUAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
            sceGxmTextureSetVAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
        }
        lighting_lut_version = 0;
        lighting_lut_retired_in_scene = 0;
        lighting_lut_ready = true;
        lighting_lut_shadow.assign(static_cast<std::size_t>(kLutCount) * kLutEntries * 2, 0.0f);
        // Recien creada no tiene nada dentro: se suben las 24.
        pica.lighting.lut_dirty = Pica::PicaCore::Lighting::LutAllDirty;
    }

    if (pica.lighting.lut_dirty == 0) {
        return true;
    }

    /**
     * SOLO SE CONVIERTEN LAS QUE CAMBIAN (0.2.1.1). Al estrenar una version
     * hay que escribirla entera, y antes eso eran las 24 tablas convertidas
     * otra vez entrada a entrada (12.288 conversiones) aunque el juego hubiera
     * cambiado una sola, varias veces por fotograma en Pokemon. Ahora las
     * cambiadas se convierten a lighting_lut_shadow y cada version se copia
     * de ahi.
     */
    {
        u32 dirty = pica.lighting.lut_dirty;
        while (dirty != 0) {
            const u32 index = static_cast<u32>(std::countr_zero(dirty));
            dirty &= ~(1u << index);
            const auto& source = pica.lighting.luts[index];
            f32* row =
                lighting_lut_shadow.data() + static_cast<std::size_t>(index) * kLutEntries * 2;
            for (u32 i = 0; i < kLutEntries; i++) {
                row[i * 2] = source[i].ToFloat();
                row[i * 2 + 1] = source[i].DiffToFloat();
            }
        }
    }
    constexpr std::size_t kRowBytes = static_cast<std::size_t>(kLutEntries) * 2 * sizeof(f32);
    const auto version_data = [this](u32 version) {
        return static_cast<u8*>(lighting_lut_buffer.Data()) +
               static_cast<std::size_t>(version) * kLutBytes;
    };
    const auto write_row = [&](u32 version, u32 index) {
        std::memcpy(version_data(version) + index * kRowBytes,
                    reinterpret_cast<const u8*>(lighting_lut_shadow.data()) + index * kRowBytes,
                    kRowBytes);
    };
    const auto write_all = [&](u32 version) {
        std::memcpy(version_data(version), lighting_lut_shadow.data(), kLutBytes);
    };

    /**
     * La escena abierta ya lee la version actual: NO se puede reescribir.
     *
     * Hasta 0.1.0.41 esto cerraba la escena -- sceGxmFinish, esperar a que la
     * GPU acabara todo lo apuntado -- y reescribia la unica textura que habia.
     * Pokemon y compania cambian tablas al cambiar de material, varias veces
     * por fotograma, y cada una era una parada completa de la CPU.
     *
     * Ahora se pasa a la SIGUIENTE version y se escribe entera (las 24 tablas,
     * desde pica.lighting.luts, que es la copia de verdad y vive en memoria
     * normal). La anterior se queda como estaba para los dibujados que ya la
     * tienen atada. Solo si la escena ya tiene atadas TODAS las versiones se
     * cierra, como antes: el resultado es el mismo, solo que casi nunca hace
     * falta esperar.
     *
     * Por que la siguiente esta libre: se avanza siempre de una en una, asi
     * que las que lee la escena son la actual y las 'retired' anteriores. Con
     * retired + 1 < kLutVersions, la siguiente no es ninguna de ellas.
     */
    if (lighting_lut_in_scene) {
        if (lighting_lut_retired_in_scene + 1 < kLutVersions) {
            lighting_lut_retired_in_scene++;
            lighting_lut_version = (lighting_lut_version + 1) % kLutVersions;
            // Una escena ya enviada puede seguir leyendo esa version.
            WaitFence(lut_fence[lighting_lut_version]);
            write_all(lighting_lut_version);
            pica.lighting.lut_dirty = 0;
            // La version nueva todavia no la ha atado nadie.
            lighting_lut_in_scene = false;
            return true;
        }
        scene_close_lut.fetch_add(1, std::memory_order_relaxed);
        EndScene();
    }

    /**
     * La escena abierta no la lee, pero una ya enviada puede que si (0.1.9.4):
     * si su valla no ha pasado, se estrena la siguiente version (si esa esta
     * libre) en vez de esperar.
     */
    if (!FenceDone(lut_fence[lighting_lut_version])) {
        const u32 next = (lighting_lut_version + 1) % kLutVersions;
        if (FenceDone(lut_fence[next])) {
            lighting_lut_version = next;
            write_all(lighting_lut_version);
            pica.lighting.lut_dirty = 0;
            return true;
        }
        WaitFence(lut_fence[lighting_lut_version]);
    }
    // Nadie lee la version actual: basta con reescribir las tablas cambiadas.
    // Esta al dia en todo lo demas, porque cada version se escribe entera al
    // estrenarla y desde entonces solo se le aplican cambios.
    while (pica.lighting.lut_dirty != 0) {
        const u32 index = static_cast<u32>(std::countr_zero(pica.lighting.lut_dirty));
        pica.lighting.lut_dirty &= ~(1u << index);
        write_row(lighting_lut_version, index);
    }
    return true;
}

void RasterizerGXM::ClearDepthIfNeeded(Surface& surface) {
    if (!surface.depth_needs_clear) {
        return;
    }
    surface.depth_needs_clear = false;

    /**
     * De donde sale el valor: se LEE el buffer de profundidad del invitado.
     *
     * El 3DS borra el suyo con un relleno de memoria del chip grafico, que
     * escribe el mismo patron de 16, 24 o 32 bits en todo el tramo. Por eso
     * basta con mirar el primer pixel: si el buffer viene de un borrado, ese es
     * el valor de todo el buffer, y su posicion en memoria es el offset cero
     * pase lo que pase con los mosaicos (el Morton de (0,0) es 0).
     *
     * Si el invitado lo hubiera rellenado con algo que NO es uniforme -- una
     * transferencia, por ejemplo -- este valor seria solo el de una esquina. No
     * hay forma de hacerlo mejor sin saber como coloca GXM los bits dentro de
     * un tile, que no esta documentado; y aun en ese caso esto es mas cercano
     * que lo que habia antes, que era no enterarse del borrado en absoluto.
     * Queda anotado aqui porque es lo unico de esta ruta que no se puede
     * comprobar sin consola.
     */
    float depth = 1.0f;
    u8 stencil = 0;
    const u8* guest = surface.guest_depth_address != 0
                          ? memory.GetPhysicalPointer(surface.guest_depth_address)
                          : nullptr;
    if (guest != nullptr) {
        switch (surface.depth_bpp) {
        case 2: {
            // D16. El valor va en enteros de 0 a 65535.
            const u32 raw = static_cast<u32>(guest[0]) | (static_cast<u32>(guest[1]) << 8);
            depth = static_cast<float>(raw) / 65535.0f;
            break;
        }
        case 3:
        case 4: {
            // D24 y D24S8: los tres bytes bajos son la profundidad y, cuando
            // hay cuarto, es la plantilla (ver DecodeD24S8 en common/color.h).
            const u32 raw = static_cast<u32>(guest[0]) | (static_cast<u32>(guest[1]) << 8) |
                            (static_cast<u32>(guest[2]) << 16);
            depth = static_cast<float>(raw) / 16777215.0f;
            if (surface.depth_bpp == 4) {
                stencil = guest[3];
            }
            break;
        }
        default:
            break;
        }
    }

    sceGxmDepthStencilSurfaceSetBackgroundDepth(&surface.depth_surface, depth);
    sceGxmDepthStencilSurfaceSetBackgroundStencil(&surface.depth_surface, stencil);
    // Sin carga forzada, GXM arranca cada tile con los valores de fondo de
    // arriba en vez de traerlos de nuestro buffer: eso ES el borrado. EndScene
    // vuelve a encenderla para que la siguiente escena conserve lo dibujado.
    sceGxmDepthStencilSurfaceSetForceLoadMode(&surface.depth_surface,
                                              SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_DISABLED);
}

void RasterizerGXM::FlushPending() {
    MaterializeAllCopies();
    EndScene();
    for (auto& surface : surfaces) {
        WriteBack(*surface);
    }
}

void RasterizerGXM::FlushForPresent() {
    EndScene();
    /**
     * El presentador y vita2d reescriben sus vertices y la textura de la
     * pantalla en cada fotograma, y antes nadie tenia que esperar por ellos:
     * la espera entera al empezar cada escena ya lo cubria. Basta con que haya
     * acabado la primera escena de ESTE fotograma, que la GPU hace despues de
     * la presentacion del anterior (0.1.9.4).
     */
    if (frame_first_fence != 0) {
        WaitFence(frame_first_fence);
    } else if (context != nullptr) {
        gpu_pending = true;
        WaitGpu();
    }
    frame_first_fence = 0;
    for (auto& surface : surfaces) {
        if (!surface->copied) {
            WriteBack(*surface);
        }
    }
}

RasterizerGXM::Surface* RasterizerGXM::CurrentSurface() {
    const auto& config = pica.regs.internal.framebuffer.framebuffer;

    /**
     * Formatos de framebuffer soportados y por que la superficie de color no
     * siempre se llama "RGBA".
     *
     * El invitado guarda su RGBA8 en el orden A,B,G,R en memoria y su RGB8 en
     * B,G,R. El nombre del formato GXM lista los canales del bit mas alto al
     * mas bajo, asi que se elige el que deja la salida del shader en los bytes
     * que el invitado espera; con eso, la traduccion de CopyTiledGuest es mover
     * bytes y no tocar canales.
     *
     * En RGB8 la superficie es de 32 bits (no existe color surface de 24): ahi
     * lleva un byte de alfa que el invitado no guarda y que la copia rellena.
     */
    const auto guest_format = config.color_format.Value();
    SceGxmColorFormat rt_color_format{};
    u32 bpp = 0;
    u32 rt_bpp = 0;
    switch (guest_format) {
    case FramebufferRegs::ColorFormat::RGBA8:
        rt_color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_RGBA;
        bpp = rt_bpp = 4;
        break;
    case FramebufferRegs::ColorFormat::RGB8:
        // ARGB deja la memoria en B,G,R,A: los tres primeros bytes son ya el
        // RGB8 del invitado.
        rt_color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB;
        bpp = 3;
        rt_bpp = 4;
        break;
    case FramebufferRegs::ColorFormat::RGB5A1:
        // Los de 16 bits se eligen con el nombre que pone cada canal en el bit
        // que espera el invitado (R en los mas significativos).
        rt_color_format = SCE_GXM_COLOR_FORMAT_U5U5U5U1_RGBA;
        bpp = rt_bpp = 2;
        break;
    case FramebufferRegs::ColorFormat::RGB565:
        rt_color_format = SCE_GXM_COLOR_FORMAT_U5U6U5_RGB;
        bpp = rt_bpp = 2;
        break;
    case FramebufferRegs::ColorFormat::RGBA4:
        rt_color_format = SCE_GXM_COLOR_FORMAT_U4U4U4U4_RGBA;
        bpp = rt_bpp = 2;
        break;
    default:
        // Son tres bits del registro: vale 5, 6 o 7 mientras el juego no lo
        // haya escrito.
        NoteOnce(fb_noted[0], "gxm fb", "formato de color {}", static_cast<u32>(guest_format));
        return nullptr;
    }

    const PAddr address = config.GetColorBufferPhysicalAddress();
    // GetWidth()/GetHeight() y no los campos crudos: el registro guarda el ALTO
    // MENOS UNO, y leerlo a pelo dejaba fuera la ultima fila.
    const u32 width = config.GetWidth();
    const u32 height = config.GetHeight();
    if (address == 0 || width == 0 || height == 0 || width > 1024 || height > 1024) {
        NoteOnce(fb_noted[1], "gxm fb", "rechazado: addr {:#x} {}x{}", address, width, height);
        return nullptr;
    }
    /**
     * El stride del framebuffer de DIBUJADO es el natural, width*bpp.
     *
     * 0.1.0.11 lo fue a buscar a la configuracion de PANTALLA (que si trae
     * stride, y en NSMB2 vale 512) y estaba mal por dos motivos: esa
     * configuracion describe OTRO buffer -- lineal, el que barre la pantalla --
     * y el de dibujado va en tiles y sin hueco entre filas, que es como lo
     * direcciona el rasterizador de software. Del uno al otro se pasa con una
     * transferencia; no son el mismo sitio.
     */
    const u32 guest_stride = width * bpp;
    if (!GuestSpanMapped(memory, address, guest_stride * height)) {
        NoteOnce(fb_noted[2], "gxm fb", "sin mapear: addr {:#x} tramo {}", address,
                 guest_stride * height);
        return nullptr;
    }

    const PAddr depth_address = config.GetDepthBufferPhysicalAddress();
    const u32 depth_bpp = FramebufferRegs::BytesPerDepthPixel(config.depth_format.Value());
    const bool want_depth16 = config.depth_format.Value() == FramebufferRegs::DepthFormat::D16;

    for (auto& surface : surfaces) {
        if (surface->guest_address == address && surface->width == width &&
            surface->height == height && surface->bpp == bpp) {
            /**
             * Mismo color, OTRA profundidad.
             *
             * La superficie se busca por su framebuffer de color, que es el
             * unico que se comparte con el invitado, asi que un juego puede
             * volver a este color con otro buffer de profundidad debajo. Si ha
             * cambiado de direccion, lo que tenemos guardado describe otro
             * sitio: se apunta el nuevo y se rehace la profundidad, porque la
             * que hay dentro es la del buffer anterior.
             */
            if (surface->guest_depth_address != depth_address ||
                surface->depth_bpp != depth_bpp) {
                EndScene();
                surface->guest_depth_address = depth_address;
                surface->depth_bpp = depth_bpp;
                surface->guest_depth_stride = width * depth_bpp;
                surface->depth_needs_clear = true;
            }
            /**
             * Y si ademas ha cambiado de PRECISION, no vale con rehacerla: el
             * formato de la superficie de profundidad se fija al crearla y D16
             * no guarda ni los mismos bits ni plantilla. Dibujar igualmente
             * seria dar otra profundidad a la misma escena, que es el tipo de
             * "parecido" que aqui no vale, asi que el lote se va a software.
             */
            if (surface->depth16 != want_depth16) {
                NoteOnce(fb_noted[1], "gxm fb", "cambio de formato de profundidad");
                return nullptr;
            }
            if (surface->needs_reload) {
                // El invitado la ha cambiado por otro camino desde la ultima vez.
                EndScene();
                Reload(*surface);
            }
            return surface.get();
        }
    }

    if (surfaces.size() >= kMaxSurfaces) {
        /**
         * Los juegos alternan framebuffers: doble buffer, pantalla de arriba y
         * de abajo, efectos aparte. Sin tope, cada uno nuevo se queda con su
         * color y su profundidad en CDRAM y no los suelta nunca. Al llegar aqui
         * se vuelca lo sucio y se sueltan todas; la que vuelva a hacer falta se
         * recrea leyendo del invitado, que es correcto aunque cueste.
         */
        FlushPending();
        // Un blit de copia de pantalla puede seguir leyendo alguna.
        const Common::ScopedVitaStage stage{"gxm tope de superficies"};
        sceGxmFinish(context);
        gpu_pending = false;
        surfaces.clear();
        NoteOnce(fb_noted[7], "gxm fb", "tope de {} superficies", kMaxSurfaces);
    }

    // Nueva superficie. Antes hay que cerrar cualquier escena abierta: crear un
    // render target con una escena en marcha no es legal en GXM.
    EndScene();
    const bool depth16 = want_depth16;
    auto surface = std::make_unique<Surface>();
    surface->guest_address = address;
    surface->width = width;
    surface->height = height;
    surface->bpp = bpp;
    surface->rt_bpp = rt_bpp;
    surface->guest_stride = guest_stride;
    surface->guest_depth_address = depth_address;
    surface->depth_bpp = depth_bpp;
    surface->guest_depth_stride = width * depth_bpp;
    surface->depth16 = depth16;

    /**
     * Las dos alineaciones que exige GXM, que NO son la misma.
     *
     * El color va lineal y su stride se redondea a 8 pixeles. La profundidad va
     * en tiles y su stride se redondea al tile entero (SCE_GXM_TILE_SIZEX, 32),
     * igual que el alto del bloque. Redondear las dos a 8, como estaba, hacia
     * que sceGxmDepthStencilSurfaceInit devolviera error con cualquier ancho
     * que no fuera multiplo de 32 -- 240, que es justo el del 3DS -- y
     * CurrentSurface devolvia nulo en TODOS los lotes sin decir nada. Ese es el
     * muro que dejo tg a 0 en 0.1.0.11 y 0.1.0.12.
     */
    u32 scale = resolution_scale.load(std::memory_order_relaxed);
    if (scale != 1 && scale != 4) {
        scale = 2;
    }
    const u32 phys_width = Phys(width, scale);
    const u32 phys_height = Phys(height, scale);
    const u32 color_stride = (phys_width + 7) & ~7u;
    const u32 depth_stride = (phys_width + SCE_GXM_TILE_SIZEX - 1) & ~(SCE_GXM_TILE_SIZEX - 1);
    const u32 tiled_height = (phys_height + SCE_GXM_TILE_SIZEY - 1) & ~(SCE_GXM_TILE_SIZEY - 1);
    surface->scale = scale;
    surface->color_stride = color_stride;
    surface->gxm_color_format = static_cast<u32>(rt_color_format);

    // RW y no solo READ: aqui es la GPU la que ESCRIBE. Con el mapeo de lectura
    // el chip no tiene permiso sobre estas paginas, y eso no falla al reservar:
    // se ve mucho despues, dibujando.
    surface->color_buffer =
        Allocate(Pool::Cdram, color_stride * tiled_height * rt_bpp, SCE_GXM_MEMORY_ATTRIB_RW);
    surface->depth_buffer = Allocate(Pool::Cdram, depth_stride * tiled_height * (depth16 ? 2 : 4),
                                     SCE_GXM_MEMORY_ATTRIB_RW);
    if (!surface->color_buffer.Valid() || !surface->depth_buffer.Valid()) {
        LOG_ERROR(Render, "GXM: sin memoria para la superficie de dibujado");
        NoteOnce(fb_noted[3], "gxm fb", "sin memoria para {}x{}", width, height);
        return nullptr;
    }
    std::memset(surface->color_buffer.Data(), 0, surface->color_buffer.Size());
    /**
     * La profundidad NO se borra a mano, y esto es un cambio respecto a antes.
     *
     * Aqui habia un memset a 0xFF "porque eso es 1.0 en los dos formatos y con
     * cualquier orden de tiles". Lo primero es cierto; lo segundo es una
     * suposicion sobre como coloca GXM los bits dentro de un tile, que no esta
     * documentada. Y para la PLANTILLA era directamente falso: dejaba los ocho
     * bits a 255, cuando cualquier juego da por hecho que empieza a cero.
     *
     * El borrado lo hace ahora el propio GXM, con los valores de fondo de la
     * superficie y la carga forzada apagada en la primera escena (ver
     * ClearDepthIfNeeded). Es el unico camino que no supone nada del formato
     * interno, y de paso sirve para rehacerla cada vez que el invitado borre la
     * suya, que antes no se miraba.
     */

    const int color_rc = sceGxmColorSurfaceInit(
        &surface->color_surface, rt_color_format, SCE_GXM_COLOR_SURFACE_LINEAR,
        SCE_GXM_COLOR_SURFACE_SCALE_NONE, SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, phys_width,
        phys_height, color_stride, surface->color_buffer.Data());
    if (color_rc != 0) {
        NoteOnce(fb_noted[4], "gxm fb", "color {}x{} stride {} err {:#x}", width, height,
                 color_stride, static_cast<u32>(color_rc));
        return nullptr;
    }

    const SceGxmDepthStencilFormat depth_gxm_format =
        depth16 ? SCE_GXM_DEPTH_STENCIL_FORMAT_D16 : SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24;
    const int depth_rc = sceGxmDepthStencilSurfaceInit(
        &surface->depth_surface, depth_gxm_format, SCE_GXM_DEPTH_STENCIL_SURFACE_TILED,
        depth_stride, surface->depth_buffer.Data(), nullptr);
    if (depth_rc != 0) {
        NoteOnce(fb_noted[5], "gxm fb", "prof stride {} err {:#x}", depth_stride,
                 static_cast<u32>(depth_rc));
        return nullptr;
    }
    /**
     * La profundidad se conserva entre escenas y NO se comparte con el invitado.
     *
     * Forzar carga y guardado es lo que hace que un lote vea la profundidad que
     * dejo el anterior, porque cada lote abre y cierra su escena. El buffer de
     * profundidad del invitado ni se lee ni se escribe: el suyo tambien va en
     * tiles y el nuestro lo tiene GXM en un formato interno que no se puede
     * traducir. Mientras el camino GPU y el de software se repartan el mismo
     * fotograma, un lote que dependa de la profundidad que escribio el otro
     * saldra mal; es la primera deuda que deja esta fase.
     */
    sceGxmDepthStencilSurfaceSetBackgroundDepth(&surface->depth_surface, 1.0f);
    sceGxmDepthStencilSurfaceSetForceLoadMode(&surface->depth_surface,
                                              SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
    sceGxmDepthStencilSurfaceSetForceStoreMode(&surface->depth_surface,
                                               SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED);

    SceGxmRenderTargetParams params{};
    params.flags = 0;
    params.width = static_cast<u16>(phys_width);
    params.height = static_cast<u16>(phys_height);
    /**
     * Ocho, y no dieciseis: ESTE era el muro.
     *
     * La cabecera del SDK lo dice en una linea de comentario -- scenesPerFrame
     * va en el rango [1, SCE_GXM_MAX_SCENES_PER_RENDERTARGET], que son 8 -- y
     * ese limite no lo comprueba nadie al compilar porque vitasdk ni siquiera
     * define la constante. Con 16, sceGxmGetRenderTargetMemSize devuelve
     * 0x805b0003 (INVALID_VALUE), no se crea el render target y CurrentSurface
     * devuelve nulo en todos los lotes. Es lo que anota crash.txt de 0.1.0.13:
     * "gxm fb: rt memsize err 0x805b0003".
     *
     * Es una PISTA para el driver (cuantas escenas espera por fotograma en este
     * render target), no un tope duro: pasarse cuesta rendimiento, no un error.
     */
    params.scenesPerFrame = 8;
    params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    params.multisampleLocations = 0;
    // La memoria del driver la reservamos nosotros: el SDK de Vita no define
    // SCE_UID_INVALID_UID, asi que depender de un centinela era una suposicion,
    // y con una UID que GXM no reconozca el fallo es una caida.
    unsigned int driver_size = 0;
    const int size_rc = sceGxmGetRenderTargetMemSize(&params, &driver_size);
    if (size_rc < 0 || driver_size == 0) {
        NoteOnce(fb_noted[6], "gxm fb", "rt memsize err {:#x}", static_cast<u32>(size_rc));
        return nullptr;
    }
    // sceKernelAllocMemBlock no redondea solo: un tamano que no sea multiplo de
    // la pagina se rechaza y no dice por que.
    const SceSize driver_bytes = (driver_size + 0xFFFu) & ~0xFFFu;
    const SceUID driver_uid = sceKernelAllocMemBlock(
        "azahar_gxm_rt", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, driver_bytes, nullptr);
    if (driver_uid < 0) {
        NoteOnce(fb_noted[6], "gxm fb", "rt memblock {} err {:#x}", static_cast<u32>(driver_bytes),
                 static_cast<u32>(driver_uid));
        return nullptr;
    }
    void* driver_mem = nullptr;
    if (sceKernelGetMemBlockBase(driver_uid, &driver_mem) < 0) {
        sceKernelFreeMemBlock(driver_uid);
        NoteOnce(fb_noted[6], "gxm fb", "rt sin direccion base");
        return nullptr;
    }
    params.driverMemBlock = driver_uid;
    const int rt_rc = sceGxmCreateRenderTarget(&params, &surface->render_target);
    if (rt_rc != 0) {
        sceKernelFreeMemBlock(driver_uid);
        LOG_ERROR(Render, "GXM: no se pudo crear el render target");
        NoteOnce(fb_noted[6], "gxm fb", "rt {}x{} err {:#x}", width, height,
                 static_cast<u32>(rt_rc));
        return nullptr;
    }
    surface->driver_uid = driver_uid;

    // Copiar dentro lo que ya hubiera en el framebuffer del invitado: hay juegos
    // que dibujan encima de lo anterior sin borrar.
    Reload(*surface);

    Surface* result = surface.get();
    surfaces.emplace_back(std::move(surface));
    // Las primeras y calla: con el tope de superficies esto se puede repetir, y
    // crash.txt no es sitio para un chorro de lineas iguales.
    static u32 created_notes = 0;
    if (created_notes < 4) {
        created_notes++;
        NoteFmt("gxm fb", "addr {:#x} {}x{} creada", address, width, height);
    }
    return result;
}

/// Los modos de ablacion que mandan el estado actual a la CPU. Sacado de
/// AddTriangle (0.1.4.6) para que el camino del shader de vertices en la GPU
/// obedezca exactamente los mismos modos.
bool RasterizerGXM::AblatedByMode() const {
    const auto& internal = pica.regs.internal;
    const u32 ablation = Ablation::mode.load(std::memory_order_relaxed);
    const auto& texturing = internal.texturing.main_config;
    const bool uses_texture =
        texturing.texture0_enable || texturing.texture1_enable || texturing.texture2_enable;
    return ablation == Ablation::kNoGpu ||
           (ablation == Ablation::kNoLighting && internal.lighting.disable == 0) ||
           (ablation == Ablation::kNoScissor &&
            internal.rasterizer.scissor_test.mode != RasterizerRegs::ScissorMode::Disabled) ||
           (ablation == Ablation::kNoStencil &&
            internal.framebuffer.output_merger.stencil_test.enable != 0 &&
            internal.framebuffer.HasStencil()) ||
           (ablation == Ablation::kNoTexture && uses_texture);
}

void RasterizerGXM::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                const Pica::OutputVertex& v2) {
    if (!SwRenderer::FrameSkip::ShouldRender() || batch_skip) {
        return;
    }
    if (!batch_decided) {
        batch_decided = true;
        batch_on_gpu = false;
        /**
         * La configuracion de DIBUJADO en crash.txt, pase lo que pase con el
         * lote.
         *
         * Va aqui y no dentro de CurrentSurface a proposito: si el lote se
         * rechaza antes (por el shader, por el scissor...), esos registros
         * siguen siendo lo primero que hay que mirar, y desde la consola no hay
         * otra forma de verlos.
         */
        const auto& draw_config = pica.regs.internal.framebuffer.framebuffer;
        NoteDrawConfig(draw_config.GetColorBufferPhysicalAddress(),
                       draw_config.GetDepthBufferPhysicalAddress(), draw_config.GetWidth(),
                       draw_config.GetHeight(),
                       static_cast<u32>(draw_config.color_format.Value()),
                       static_cast<u32>(draw_config.depth_format.Value()));
        if (EnsureInitialized()) {
            // Lo que aun no reproduce el contexto GXM (W-buffering) o el
            // generador (iluminacion, proctex...) manda el lote al camino de
            // software. Las texturas ya no: las sirve el cache. El scissor y la
            // plantilla tampoco desde 0.1.0.16: el primero lo aplica
            // sceGxmSetRegionClip (ver MakeRegionClip) y la segunda
            // sceGxmSetFrontStencilFunc, los dos en DrawBatchOnGpu.
            /**
             * LA DECISION SE LE PIDE AL CACHE DE PIPELINES, NO AL GENERADOR.
             *
             * Aqui se llamaba a GenerateFragmentShader para saber si el lote se
             * podia acelerar, y se TIRABA el resultado: el cache lo volvia a
             * generar despues. Eso es construir la fuente entera del shader --
             * un string de varios kilobytes en cuanto hay iluminacion, con sus
             * reservas de memoria -- una vez por lote y por fotograma, solo para
             * responder si o no.
             *
             * El cache responde lo mismo y recuerda tanto los aciertos como los
             * fallos (ver PipelineCache::Get), asi que a partir del segundo lote
             * con la misma configuracion esto es una busqueda en una tabla hash.
             * Y como es el MISMO camino que usa despues DrawBatchOnGpu, ya no
             * pueden discrepar: antes eran dos comprobaciones distintas y habia
             * una nota ("el generador se echo atras") puesta justo para cazar
             * que se desincronizaran.
             */
            const bool wbuffering = pica.regs.internal.rasterizer.depthmap_enable ==
                                    RasterizerRegs::DepthBuffering::WBuffering;
            const char* reason = nullptr;
            batch_on_gpu =
                !wbuffering && pipelines->Get(pica.regs.internal, &reason) != nullptr;
            /**
             * Su shader de fragmentos se esta compilando en segundo plano
             * (0.1.9.8): el lote no se dibuja, ni por software. Es un momento,
             * y por software costaria mas que el propio lote (volcar y recargar
             * la superficie).
             */
            if (!batch_on_gpu && reason != nullptr && std::strcmp(reason, "fs compilando") == 0) {
                batch_skip = true;
                skipped_batches.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            /**
             * La ablacion, al final y por encima de todo lo demas: manda a
             * software los lotes que usen la caracteristica elegida aunque se
             * pudieran acelerar. Es un modo de diagnostico (SELECT + ABAJO);
             * ver RasterizerGXM::Ablation en la cabecera.
             */
            if (batch_on_gpu && AblatedByMode()) {
                batch_on_gpu = false;
                reason = "ablacion";
            }
            if (!batch_on_gpu) {
                // El motivo concreto (iluminacion, proctex, mezcla...) va a
                // crash.txt: sin el, "skip: shader" no distingue entre causas.
                const char* why = wbuffering             ? "wbuffer"
                                  : reason != nullptr    ? reason
                                                         : "shader";
                const u32 slot = wbuffering ? 1 : 2;
                NoteSkip(slot, why);
            }
        }
        if (batch_on_gpu && CurrentSurface() == nullptr) {
            batch_on_gpu = false;
            NoteSkip(3, "framebuffer");
        }
        if (!batch_on_gpu && available) {
            PrepareSoftwareBatch();
        }
    }

    if (!batch_on_gpu) {
        software_triangles.fetch_add(1, std::memory_order_relaxed);
        software.AddTriangle(v0, v1, v2);
        return;
    }
    batch.push_back(v0);
    batch.push_back(v1);
    batch.push_back(v2);
}

/**
 * UN LOTE POR SOFTWARE EN UN FRAMEBUFFER QUE TAMBIEN DIBUJA LA GPU (0.1.9.6).
 *
 * El rasterizador de software pinta en la memoria del invitado. Hasta ahora
 * nadie se lo decia a la superficie de la GPU: el software pintaba encima de
 * una memoria atrasada (sin lo que ya habia dibujado la GPU) y despues la GPU
 * seguia con SU copia y la copiaba a la pantalla. Lo dibujado por software
 * desaparecia: el profesor de Zafiro Alfa, con sus lotes a software, no se
 * veia. Ahora, antes del lote, lo de la GPU baja a la memoria, y la superficie
 * queda para recargarse antes del siguiente lote de GPU o de copiarla.
 */
void RasterizerGXM::PrepareSoftwareBatch() {
    const PAddr address =
        pica.regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress();
    for (auto& surface : surfaces) {
        if (surface->guest_address != address) {
            continue;
        }
        // Un relleno pendiente sin nada dibujado encima ya esta en la memoria
        // del invitado (ver AccelerateFill): solo hace falta bajar lo dibujado.
        if (surface->dirty) {
            if (open_surface == surface.get()) {
                EndScene();
            }
            WriteBack(*surface);
            software_syncs.fetch_add(1, std::memory_order_relaxed);
        }
        surface->needs_reload = true;
        surface->dirty = false;
        surface->copied = false;
    }
}

bool RasterizerGXM::DrawBatchOnGpu() {
    Surface* surface = CurrentSurface();
    if (surface == nullptr) {
        return false;
    }
    // GXM solo admite UNA escena abierta: si la que hay es de otra superficie
    // (juegos con doble buffer alternan framebuffers), se cierra antes de
    // empezar la nueva. Sin esto, el segundo sceGxmBeginScene con la primera
    // escena en marcha es una caida.
    if (open_surface != nullptr && open_surface != surface) {
        EndScene();
    }
    /**
     * Aqui se esperaba a la GPU entera antes de cada escena nueva (0.1.5.6),
     * porque EndScene devolvia el buffer de vertices a cero. Desde 0.1.9.4 el
     * buffer es un anillo y las tablas de luz y las texturas esperan a su
     * propia valla: la CPU sigue mientras la GPU dibuja lo anterior.
     */
    const u32 vertex_count = static_cast<u32>(batch.size());
    if (vertex_count == 0) {
        return true;
    }
    /**
     * El pipeline PRIMERO, antes de tocar el buffer de vertices.
     *
     * Antes se buscaba despues de empaquetar, que daba igual mientras todos los
     * vertices midieran lo mismo. Ya no: el shader decide si el lote lleva
     * normales y vista (21 floats) o no (14), y eso cambia cuanto sitio hay que
     * reservar y cuanto se escribe por vertice. Si no hay pipeline, ademas, se
     * ahorra el empaquetado entero.
     */
    const PipelineCache::Entry* pipeline = pipelines->Get(pica.regs.internal);
    if (pipeline == nullptr) {
        return false;
    }
    const u32 stride = VertexStride(pipeline->lit, pipeline->proj);
    const u32 floats_per_vertex = stride / sizeof(float);

    /**
     * Las tablas de la iluminacion, ANTES de repartir el buffer de vertices.
     *
     * Reescribirlas puede exigir cerrar la escena (la GPU podria estar leyendo
     * las de antes), y el lote tiene que dibujarse en la escena que se abra
     * despues, con sus vertices ya escritos.
     */
    if (pipeline->lighting_lut != nullptr && !UpdateLightingLut()) {
        return false;
    }
    /**
     * El buffer de vertices se REPARTE dentro de la escena, no se reescribe.
     *
     * GXM no dibuja cuando se lo pides: apunta el dibujado y lee los vertices
     * mas tarde, al procesar la escena. Escribir el lote siguiente encima del
     * anterior -- que es lo que hacia antes, siempre desde el principio del
     * buffer -- le cambia los vertices a un dibujado que todavia no ha ocurrido.
     * Por eso cada lote se queda su tramo. Desde 0.1.9.4 el contador tampoco
     * vuelve a cero al cerrar la escena: es un anillo, y solo al llegar al
     * final se espera a la GPU entera antes de volver al principio.
     */
    const u32 needed = vertex_count * stride;
    /**
     * EL BUFFER SE RESERVA GRANDE DESDE EL PRINCIPIO.
     *
     * Hasta 0.1.0.41 solo crecia hasta el tamano del lote MAS GRANDE visto, y
     * se reservaba justo eso. Como un fotograma son muchos lotes, se llenaba
     * enseguida, y llenarse es cerrar la escena -- sceGxmFinish, esperar a que
     * la GPU acabe todo lo apuntado -- y volver a cargar color y profundidad
     * de la superficie en la escena nueva. Un fotograma de la cinematica de
     * Rubi Omega son unos 13.000 triangulos iluminados, 39.000 vertices de 84
     * bytes: ~3,3 MB. Desde 0.2.1.1 el anillo es el mismo que el de
     * AccelerateDrawBatch, de 8 MB y por tramos (ver ReserveVertexSpace).
     *
     * En CDRAM porque la GPU lo lee una vez por vertice dibujado y la CPU solo
     * escribe en orden, que la memoria de video admite bien; si no hay sitio,
     * memoria normal, como antes.
     */
    u8* const vertex_slice = ReserveVertexSpace(needed);
    if (vertex_slice == nullptr) {
        NoteOnce(fb_noted[10], "gxm draw", "sin memoria para {} bytes de vertices", needed);
        return false;
    }
    if (index_capacity == 0) {
        Allocation buffer = Allocate(Pool::Host, kMaxVerticesPerDraw * sizeof(u16));
        if (!buffer.Valid()) {
            NoteOnce(fb_noted[11], "gxm draw", "sin memoria para los indices");
            return false;
        }
        index_buffer = std::move(buffer);
        index_capacity = kMaxVerticesPerDraw;
        auto* indices = static_cast<u16*>(index_buffer.Data());
        for (u32 i = 0; i < index_capacity; i++) {
            indices[i] = static_cast<u16>(i);
        }
    }

    auto* vertices = reinterpret_cast<float*>(vertex_slice);

    /**
     * EMPAQUETADO EN UNA SOLA PASADA (0.1.0.45).
     *
     * El buffer de vertices esta en CDRAM, que la CPU escribe sin cache y
     * combinando escrituras seguidas. Hasta 0.1.0.44 se escribia en hasta tres
     * pasadas -- lo comun, despues normales y vista, despues la W --, cada una
     * dejando huecos en cada vertice: la combinacion de escrituras se rompe en
     * cada hueco. Ahora cada vertice se escribe entero y en orden, con la
     * variante resuelta en tiempo de compilacion. Los valores y sus posiciones
     * son los mismos que antes (ver AttribSpec en BuildVertexFormat): 0-3
     * posicion, 4-7 color, 8-13 coordenadas, 14-20 normal y vista si hay luz,
     * y la W de la textura 0 al final si es proyectada. Todo CRUDO, sin dividir
     * por w: la correccion de perspectiva la pone GXM al interpolar.
     */
    const auto pack = [&]<bool kLit, bool kProj>() {
        float* out = vertices;
        for (u32 i = 0; i < vertex_count; i++) {
            const Pica::OutputVertex& v = batch[i];
            out[0] = v.pos.x.ToFloat32();
            out[1] = v.pos.y.ToFloat32();
            out[2] = v.pos.z.ToFloat32();
            out[3] = v.pos.w.ToFloat32();
            out[4] = v.color.x.ToFloat32();
            out[5] = v.color.y.ToFloat32();
            out[6] = v.color.z.ToFloat32();
            out[7] = v.color.w.ToFloat32();
            out[8] = v.tc0.x.ToFloat32();
            out[9] = v.tc0.y.ToFloat32();
            out[10] = v.tc1.x.ToFloat32();
            out[11] = v.tc1.y.ToFloat32();
            out[12] = v.tc2.x.ToFloat32();
            out[13] = v.tc2.y.ToFloat32();
            if constexpr (kLit) {
                out[14] = v.quat.x.ToFloat32();
                out[15] = v.quat.y.ToFloat32();
                out[16] = v.quat.z.ToFloat32();
                out[17] = v.quat.w.ToFloat32();
                out[18] = v.view.x.ToFloat32();
                out[19] = v.view.y.ToFloat32();
                out[20] = v.view.z.ToFloat32();
            }
            if constexpr (kProj) {
                out[kLit ? 21 : 14] = v.tc0_w.ToFloat32();
            }
            out += floats_per_vertex;
        }
    };
    if (pipeline->lit) {
        if (pipeline->proj) {
            pack.template operator()<true, true>();
        } else {
            pack.template operator()<true, false>();
        }
    } else {
        if (pipeline->proj) {
            pack.template operator()<false, true>();
        } else {
            pack.template operator()<false, false>();
        }
    }

    if (!SetupDrawState(surface, pipeline,
                        pipelines->FormatFor(pipeline->lit, pipeline->proj).program,
                        pipeline->program)) {
        return false;
    }
    const auto* indices = static_cast<const u16*>(index_buffer.Data());
    u32 drawn = 0;
    while (drawn < vertex_count) {
        const u32 chunk = std::min(vertex_count - drawn, kMaxVerticesPerDraw);
        // El flujo apunta al trozo que toca. Hasta 0.1.4.5 se ponia una sola vez
        // al principio, y como los indices van siempre de 0 a chunk-1, un lote
        // de mas de kMaxVerticesPerDraw vertices repetia el primer trozo en vez
        // de dibujar el resto. kMaxVerticesPerDraw es multiplo de 3, asi que
        // ningun triangulo queda partido entre dos trozos.
        sceGxmSetVertexStream(context, 0, vertex_slice + static_cast<std::size_t>(drawn) * stride);
        const int rc = sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                                  indices, chunk);
        if (rc < 0) {
            // Hasta ahora este fallo se tragaba en silencio y el lote se iba a
            // software sin dejar rastro: con el codigo se sabe si es el estado,
            // los vertices o la escena.
            NoteOnce(fb_noted[9], "gxm draw", "draw err {:#x} con {} vertices",
                     static_cast<u32>(rc), chunk);
            return false;
        }
        drawn += chunk;
    }
    surface->dirty = true;
    {
        // Nota unica: con esto, un volcado posterior sabe si la GPU llego a
        // dibujar algo, que es la duda que dejo el ultimo crash.
        static bool noted_first_draw = false;
        if (!noted_first_draw) {
            noted_first_draw = true;
            Common::VitaNote("gxm draw", "primer lote dibujado en la GPU");
        }
    }
    gpu_triangles.fetch_add(vertex_count / 3, std::memory_order_relaxed);
    gpu_batches.fetch_add(1, std::memory_order_relaxed);
    return true;
}

/**
 * El estado de dibujado COMUN a los dos caminos de GPU (0.1.4.6): el de
 * triangulos ya sombreados (DrawBatchOnGpu) y el de shader de vertices en la GPU
 * (AccelerateDrawBatch). Abre la escena si hace falta y pone viewport, recorte,
 * profundidad, plantilla, culling, los dos programas, texturas y uniforms de
 * fragmentos. Es el mismo codigo que estaba dentro de DrawBatchOnGpu, sacado
 * tal cual para que los dos caminos no puedan dibujar distinto; lo unico que
 * cambia de uno a otro son los programas, que llegan como parametro.
 *
 * False = no se puede dibujar este lote en la GPU (la escena no abre, o una
 * textura no se puede servir): quien llama lo manda por la CPU.
 */
template <typename Entry>
bool RasterizerGXM::SetupDrawState(Surface* surface, const Entry* pipeline,
                                   const SceGxmVertexProgram* vertex_program,
                                   const SceGxmFragmentProgram* fragment_program) {
    const bool timed = profile_state;
    unsigned long long mark = timed ? Common::VitaMicros() : 0;
    if (!surface->scene_open) {
        // Una escena nueva va a cambiar el color: lo que se llevo una copia de
        // pantalla ya no es lo que hay (0.1.8.7).
        surface->copied = false;
        // Si la profundidad o la plantilla estan por rehacer, se deja la
        // superficie preparada ANTES de abrir: el borrado lo hace GXM al cargar
        // los tiles y solo mira la superficie en sceGxmBeginScene.
        ClearDepthIfNeeded(*surface);
        const Common::ScopedVitaStage stage{"gxm abrir escena"};
        const int scene_rc =
            sceGxmBeginScene(context, 0, surface->render_target, nullptr, nullptr, nullptr,
                             &surface->color_surface, &surface->depth_surface);
        if (scene_rc != 0) {
            // Se anota el codigo: 0x805b0007 seria "ya hay una escena abierta"
            // (la de vita2d, si alguna vez se cruzan los dos caminos) y
            // 0x805b0003 un parametro malo de la superficie.
            NoteOnce(fb_noted[8], "gxm draw", "beginscene err {:#x}",
                     static_cast<u32>(scene_rc));
            return false;
        }
        surface->scene_open = true;
        open_surface = surface;
        if (surface->clear_pending) {
            DrawClearQuad(*surface);
        }
    }
    if (timed) {
        const unsigned long long now = Common::VitaMicros();
        state_scene_us += now - mark;
        mark = now;
    }

    // Viewport: la misma transformacion que hace el rasterizador de software
    // (posicion de recorte -> coordenadas del framebuffer del 3DS) con la
    // profundidad en el rango de la PICA, que es [0, 1] con z ya negada.
    const auto& rasterizer = pica.regs.internal.rasterizer;
    const float halfsize_x = Pica::f24::FromRaw(rasterizer.viewport_size_x).ToFloat32();
    const float halfsize_y = Pica::f24::FromRaw(rasterizer.viewport_size_y).ToFloat32();
    const float corner_x = static_cast<float>(rasterizer.viewport_corner.x);
    const float corner_y = static_cast<float>(rasterizer.viewport_corner.y);
    const float depth_scale = Pica::f24::FromRaw(rasterizer.viewport_depth_range).ToFloat32();
    const float depth_offset = Pica::f24::FromRaw(rasterizer.viewport_depth_near_plane).ToFloat32();
    /**
     * EL SIGNO DE LA ESCALA EN Y, QUE ES LO QUE TENIA LA IMAGEN BOCA ABAJO.
     *
     * Aqui iba -halfsize_y, razonando que GXM cuenta sus filas desde arriba y
     * la PICA su Y de pantalla desde abajo, asi que habia que invertir. El
     * razonamiento es correcto; lo que no se podia comprobar desde fuera de la
     * consola es CON QUE SIGNO se le pide esa inversion a sceGxmSetViewport.
     * El presentador demuestra que, con el viewport por defecto, el NDC y=+1
     * cae arriba (NdcY devuelve +1 para el pixel 0), pero no dice que valor de
     * yScale produce eso, porque el que lo pone es vita2d.
     *
     * EN 0.1.0.20 SE CAMBIO A POSITIVO Y FUE UN ERROR, ANOTADO AQUI PARA QUE NO
     * SE REPITA. El sintoma que lo motivo -- "la pantalla esta al reves" -- no
     * era la geometria: eran las TEXTURAS, que se decodificaban sin el volteo
     * vertical que la PICA les aplica al muestrear (ver el bucle de
     * decodificado en gxm_texture_cache.cpp). Con el marco del dialogo bien
     * colocado y las letras del reves, lo que estaba mal era la imagen dentro
     * del poligono, no el poligono. Cambiar el viewport "arreglaba" el sintoma
     * equivocado.
     *
     * La cuenta, que es la que manda: la PICA escribe en la fila
     * (alto - 1) - p del framebuffer del invitado (sw_framebuffer.h hace
     * "y = cached_height - y"), y con la escala negativa el centro del viewport
     * cae donde toca y las filas salen en ese mismo orden. Vuelve a negativo.
     */
    /**
     * EL DESPLAZAMIENTO EN Y SE MIDE DESDE EL ALTO DEL FRAMEBUFFER, NO DESDE EL
     * CENTRO DEL VIEWPORT.
     *
     * Aqui iba "halfsize_y + corner_y", que es el centro del viewport. Eso solo
     * acierta cuando el viewport cubre el framebuffer ENTERO, porque entonces
     * 2*(halfsize_y + corner_y) vale justo el alto. En cuanto el framebuffer es
     * mas grande -- y lo es a menudo, porque los juegos lo redondean: en NSMB2
     * el framebuffer mide 416 de alto y el viewport 400 -- la imagen sale
     * espejada respecto al eje equivocado y desplazada por la diferencia.
     *
     * La cuenta buena sale de igualar con el rasterizador de software, que
     * escribe en la fila (alto - 1) - p del framebuffer, con p la Y de pantalla
     * de la PICA:
     *
     *     p          = halfsize_y * ndc + halfsize_y + corner_y
     *     fila       = alto - p                 (continua; al truncar da alto-1-p)
     *                = alto - halfsize_y - corner_y - halfsize_y * ndc
     *
     * o sea desplazamiento = alto - halfsize_y - corner_y, y escala -halfsize_y
     * como estaba. Con el viewport a pantalla completa las dos formulas dan lo
     * mismo, que es por lo que esto no se veia en cuanto uno lo probaba de
     * cabeza.
     *
     * Y ES LO QUE DESCUADRABA EL SCISSOR. MakeRegionClip traduce la caja
     * suponiendo fila = (alto - 1) - p, que es lo correcto; el viewport ponia la
     * geometria en otro sitio. Los dos discrepaban justo en esa diferencia, asi
     * que el recorte caia unos pixeles corrido respecto a lo que recortaba: en
     * el dialogo de error de NSMB2 se comia las primeras letras de cada linea.
     */
    const float guest_height = static_cast<float>(surface->height);
    const float scale = static_cast<float>(surface->scale) * 0.5f;
    sceGxmSetViewport(context, (halfsize_x + corner_x) * scale, halfsize_x * scale,
                      (guest_height - halfsize_y - corner_y) * scale, -halfsize_y * scale,
                      depth_offset, depth_scale);
    sceGxmSetFrontPolygonMode(context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);

    /**
     * El recorte de region, SIEMPRE y explicito.
     *
     * No basta con ponerlo cuando el juego enciende el scissor: el contexto de
     * GXM es uno solo para todo el proceso (el de vita2d, ver
     * EnsureInitialized) y su estado no se puede leer hacia atras, asi que un
     * recorte puesto en un lote se queda puesto para el siguiente y para quien
     * venga detras. Por eso se escribe en cada lote -- con caja si hay scissor
     * y con CLIP_NONE si no -- y EndScene lo deja en CLIP_NONE antes de soltar
     * la escena, para que el presentador no herede el recorte de un juego.
     */
    RegionClip clip = MakeRegionClip(rasterizer, surface->width, surface->height);
    if (surface->scale != 2 && (clip.mode == SCE_GXM_REGION_CLIP_OUTSIDE ||
                                clip.mode == SCE_GXM_REGION_CLIP_INSIDE)) {
        // La caja es de pixeles del invitado, con los dos extremos dentro.
        const u32 s = surface->scale;
        clip.x_min = Phys(clip.x_min, s);
        clip.y_min = Phys(clip.y_min, s);
        clip.x_max = std::max(Phys(clip.x_max + 1, s), clip.x_min + 1) - 1;
        clip.y_max = std::max(Phys(clip.y_max + 1, s), clip.y_min + 1) - 1;
    }
    sceGxmSetRegionClip(context, clip.mode, clip.x_min, clip.y_min, clip.x_max, clip.y_max);

    const auto& merger = pica.regs.internal.framebuffer.output_merger;
    const SceGxmDepthFunc depth_func = MapDepthFunc(merger.depth_test_func.Value());
    const bool depth_write = merger.depth_write_enable &&
                             pica.regs.internal.framebuffer.framebuffer.allow_depth_stencil_write != 0;
    const SceGxmDepthWriteMode write_mode =
        depth_write ? SCE_GXM_DEPTH_WRITE_ENABLED : SCE_GXM_DEPTH_WRITE_DISABLED;
    if (!merger.depth_test_enable) {
        sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
        sceGxmSetBackDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    } else {
        sceGxmSetFrontDepthFunc(context, depth_func);
        sceGxmSetBackDepthFunc(context, depth_func);
    }
    sceGxmSetFrontDepthWriteEnable(context, write_mode);
    sceGxmSetBackDepthWriteEnable(context, write_mode);

    /**
     * La prueba de plantilla.
     *
     * CUANDO CUENTA. El rasterizador de software exige DOS cosas para mirarla
     * siquiera (sw_rasterizer.cpp, stencil_action_enable): que el registro este
     * encendido Y que el formato de profundidad sea D24S8, que es el unico del
     * 3DS que guarda plantilla. Con D16 o D24 el registro puede traer lo que
     * sea y el hardware no lo mira, asi que aqui tampoco: mirarlo mandaria a la
     * GPU una prueba que el de software no hace.
     *
     * LAS DOS MASCARAS NO SON LA MISMA. input_mask se aplica a los DOS lados de
     * la comparacion (referencia y valor guardado) y es el compareMask de GXM;
     * write_mask decide que bits se escriben y es el writeMask. Cambiarlas de
     * sitio es el error clasico y no falla: solo recorta mal.
     *
     * Y LA ESCRITURA SE PUEDE APAGAR APARTE. allow_depth_stencil_write del
     * registro de framebuffer apaga la ESCRITURA de profundidad y plantilla
     * pero no la prueba, igual que arriba con la profundidad; aqui eso es una
     * mascara de escritura a cero, no una prueba desactivada.
     *
     * LAS DOS CARAS CON EL MISMO ESTADO. La PICA no tiene plantilla por cara:
     * una sola configuracion para todos los triangulos. Se escribe en las dos
     * de GXM porque el contexto es compartido y no se puede leer si alguien
     * dejo encendido el modo de dos caras; con las dos iguales da igual.
     */
    const auto& stencil_test = merger.stencil_test;
    const bool stencil_enabled =
        stencil_test.enable != 0 && pica.regs.internal.framebuffer.HasStencil();
    if (stencil_enabled) {
        const SceGxmStencilFunc stencil_func = MapStencilFunc(stencil_test.func.Value());
        const SceGxmStencilOp on_stencil_fail =
            MapStencilOp(stencil_test.action_stencil_fail.Value());
        const SceGxmStencilOp on_depth_fail = MapStencilOp(stencil_test.action_depth_fail.Value());
        const SceGxmStencilOp on_depth_pass = MapStencilOp(stencil_test.action_depth_pass.Value());
        const u8 compare_mask = static_cast<u8>(stencil_test.input_mask);
        const u8 stencil_write_mask =
            pica.regs.internal.framebuffer.framebuffer.allow_depth_stencil_write != 0
                ? static_cast<u8>(stencil_test.write_mask)
                : 0;
        const u32 reference = stencil_test.reference_value;
        sceGxmSetFrontStencilRef(context, reference);
        sceGxmSetBackStencilRef(context, reference);
        sceGxmSetFrontStencilFunc(context, stencil_func, on_stencil_fail, on_depth_fail,
                                  on_depth_pass, compare_mask, stencil_write_mask);
        sceGxmSetBackStencilFunc(context, stencil_func, on_stencil_fail, on_depth_fail,
                                 on_depth_pass, compare_mask, stencil_write_mask);
    } else {
        // Apagada de verdad: pasa siempre y no toca el buffer. Hay que
        // escribirlo, no basta con no escribir nada, porque el estado del lote
        // anterior sigue puesto en el contexto.
        sceGxmSetFrontStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                                  SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
        sceGxmSetBackStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                                 SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0x00);
    }

    /**
     * El culling va DE LA MANO del signo de arriba, y por eso cambia con el.
     *
     * Invertir la Y de la pantalla invierte el sentido de giro de todos los
     * triangulos, asi que este cruce y el signo del viewport van SIEMPRE
     * juntos: la PICA decide en su espacio de pantalla (Y hacia abajo) y GXM en
     * coordenadas de recorte (Y hacia arriba), de ahi que los sentidos vayan
     * intercambiados. En 0.1.0.20 se cambiaron los dos por un diagnostico
     * equivocado (ver el viewport, arriba) y los dos vuelven a su sitio.
     */
    switch (rasterizer.cull_mode.Value()) {
    case RasterizerRegs::CullMode::KeepAll:
    case RasterizerRegs::CullMode::KeepAll2:
        sceGxmSetCullMode(context, SCE_GXM_CULL_NONE);
        break;
    case RasterizerRegs::CullMode::KeepClockWise:
        sceGxmSetCullMode(context, SCE_GXM_CULL_CCW);
        break;
    case RasterizerRegs::CullMode::KeepCounterClockWise:
        sceGxmSetCullMode(context, SCE_GXM_CULL_CW);
        break;
    }

    sceGxmSetVertexProgram(context, vertex_program);
    sceGxmSetFragmentProgram(context, fragment_program);

    // Texturas: por cada sampler que el shader use de verdad, su unidad. Si
    // alguna no se puede servir (formato, borde, unidad apagada...), el lote
    // entero vuelve a software desde DrawTriangles.
    if (timed) {
        mark = Common::VitaMicros();
    }
    for (u32 i = 0; i < 3; i++) {
        if (pipeline->samplers[i] == nullptr) {
            continue;
        }
        const SceGxmTexture* texture = textures->Get(i, pica.regs.internal, memory);
        if (texture == nullptr) {
            NoteSkip(4, "textura");
            return false;
        }
        sceGxmSetFragmentTexture(context, pipeline->sampler_units[i], texture);
    }
    if (timed) {
        state_texture_us += Common::VitaMicros() - mark;
    }
    if (pipeline->lighting_lut != nullptr) {
        // La version vigente: ver UpdateLightingLut.
        sceGxmSetFragmentTexture(context, pipeline->lighting_lut_unit,
                                 &lighting_lut_textures[lighting_lut_version]);
        lighting_lut_in_scene = true;
    }

    // Uniforms. Los que el compilador haya eliminado no existen como parametro
    // y se saltan; los samplers quedan para cuando exista el cache de texturas.
    void* uniform_buffer = nullptr;
    if (sceGxmReserveFragmentDefaultUniformBuffer(context, &uniform_buffer) == 0) {
        /**
         * Los ya convertidos de BatchMemo, rehechos si se han escrito sus
         * registros: desde el lote anterior (los apunto AccelerateDrawBatch) o
         * desde entonces (los dirty_regs que siguen puestos).
         */
        BatchMemo& memo = *batch_memo;
        if (pipeline->const_color != nullptr) {
            if (memo.colors_dirty || AnyDirty(pica.dirty_regs, kConstColorMask)) {
                const auto tev_stages = pica.regs.internal.texturing.GetTevStages();
                for (u32 i = 0; i < 6; i++) {
                    const u32 raw = tev_stages[i].const_color;
                    memo.const_colors[i * 4 + 0] = static_cast<f32>((raw >> 0) & 0xFF) / 255.0f;
                    memo.const_colors[i * 4 + 1] = static_cast<f32>((raw >> 8) & 0xFF) / 255.0f;
                    memo.const_colors[i * 4 + 2] = static_cast<f32>((raw >> 16) & 0xFF) / 255.0f;
                    memo.const_colors[i * 4 + 3] = static_cast<f32>((raw >> 24) & 0xFF) / 255.0f;
                }
                memo.colors_dirty = false;
            }
            sceGxmSetUniformDataF(uniform_buffer, pipeline->const_color, 0, 24,
                                  memo.const_colors.data());
        }
        if (pipeline->combiner_buffer_color != nullptr) {
            const auto raw = pica.regs.internal.texturing.tev_combiner_buffer_color;
            const f32 color[4] = {static_cast<f32>(raw.r) / 255.0f, static_cast<f32>(raw.g) / 255.0f,
                                  static_cast<f32>(raw.b) / 255.0f, static_cast<f32>(raw.a) / 255.0f};
            sceGxmSetUniformDataF(uniform_buffer, pipeline->combiner_buffer_color, 0, 4, color);
        }
        if (pipeline->alphatest_ref != nullptr) {
            const f32 reference = static_cast<f32>(merger.alpha_test.ref);
            sceGxmSetUniformDataF(uniform_buffer, pipeline->alphatest_ref, 0, 1, &reference);
        }
        if (pipeline->blend_const_alpha != nullptr) {
            const f32 constant = static_cast<f32>(merger.blend_const.a) / 255.0f;
            sceGxmSetUniformDataF(uniform_buffer, pipeline->blend_const_alpha, 0, 1, &constant);
        }
        if (pipeline->fog_lut != nullptr) {
            // 128 entradas de dos floats: valor y pendiente (misma LUT que usa
            // el rasterizador de software, leida como 16 bits sin signo).
            if (memo.fog_dirty || AnyDirty(pica.dirty_regs, kFogLutMask)) {
                for (u32 i = 0; i < 128; i++) {
                    const u32 raw = pica.fog.lut[i].raw;
                    memo.fog_lut[i * 2] = static_cast<f32>(raw & 0xFFFF) / 65535.0f;
                    memo.fog_lut[i * 2 + 1] = static_cast<f32>(raw >> 16) / 65535.0f;
                }
                memo.fog_dirty = false;
            }
            sceGxmSetUniformDataF(uniform_buffer, pipeline->fog_lut, 0, 256, memo.fog_lut.data());
        }
        if (pipeline->fog_color != nullptr) {
            const auto& fog = pica.regs.internal.texturing.fog_color;
            const f32 color[4] = {static_cast<f32>(fog.r) / 255.0f,
                                  static_cast<f32>(fog.g) / 255.0f,
                                  static_cast<f32>(fog.b) / 255.0f, 1.0f};
            sceGxmSetUniformDataF(uniform_buffer, pipeline->fog_color, 0, 3, color);
        }
        if (pipeline->tex_border_color != nullptr) {
            /**
             * El color de borde de las tres unidades.
             *
             * El registro los guarda como R,G,B,A de ocho bits en ese orden
             * desde el bit cero -- el mismo desempaquetado que hace
             * rasterizer_accelerated.cpp para los backends de escritorio, que
             * es de donde sale este codigo -- y el shader los quiere
             * normalizados. Solo existe este parametro si el generador declaro
             * el uniform, o sea si alguna unidad usada tiene ClampToBorder.
             */
            f32 borders[3 * 4];
            const auto pica_textures = pica.regs.internal.texturing.GetTextures();
            for (u32 i = 0; i < 3; i++) {
                const u32 raw = pica_textures[i].config.border_color.raw;
                borders[i * 4 + 0] = static_cast<f32>((raw >> 0) & 0xFF) / 255.0f;
                borders[i * 4 + 1] = static_cast<f32>((raw >> 8) & 0xFF) / 255.0f;
                borders[i * 4 + 2] = static_cast<f32>((raw >> 16) & 0xFF) / 255.0f;
                borders[i * 4 + 3] = static_cast<f32>((raw >> 24) & 0xFF) / 255.0f;
            }
            sceGxmSetUniformDataF(uniform_buffer, pipeline->tex_border_color, 0, 12, borders);
        }
        if (pipeline->lit) {
            /**
             * Los parametros de las ocho luces.
             *
             * Se suben LAS OCHO aunque el juego encienda una, porque el shader
             * indexa por el numero de luz y ese numero no tiene por que ser el
             * hueco que ocupa. Son unos 900 floats por lote; frente a lo que
             * cuesta un solo triangulo iluminado por software, es ruido.
             *
             * Cada luz ocupa cuatro floats aunque solo use tres: un array de
             * float3 en el buffer de uniforms de la USSE no tiene un
             * empaquetado documentado, y el shader los declara float4 por eso
             * mismo (ver WriteUniforms en cg_fs_shader_gen.cpp).
             */
            const auto& lighting_regs = pica.regs.internal.lighting;
            const auto put_color = [](const Pica::LightingRegs::LightColor& color, f32* out) {
                // Diez bits por canal, pero 255 ya es 1.0 (el hardware deja
                // pasar valores por encima de uno a proposito).
                out[0] = static_cast<f32>(color.r) / 255.0f;
                out[1] = static_cast<f32>(color.g) / 255.0f;
                out[2] = static_cast<f32>(color.b) / 255.0f;
                out[3] = 0.0f;
            };
            if (memo.lights_dirty || AnyDirty(pica.dirty_regs, kLightsMask)) {
                for (u32 i = 0; i < 8; i++) {
                    const auto& light = lighting_regs.light[i];
                    put_color(light.specular_0, memo.specular_0.data() + i * 4);
                    put_color(light.specular_1, memo.specular_1.data() + i * 4);
                    put_color(light.diffuse, memo.diffuse.data() + i * 4);
                    put_color(light.ambient, memo.ambient.data() + i * 4);
                    // La posicion son tres medios flotantes de 16 bits.
                    memo.position[i * 4 + 0] = Pica::f16::FromRaw(light.x).ToFloat32();
                    memo.position[i * 4 + 1] = Pica::f16::FromRaw(light.y).ToFloat32();
                    memo.position[i * 4 + 2] = Pica::f16::FromRaw(light.z).ToFloat32();
                    // La direccion del foco va en fijo 1.1.11 (de ahi el 2047).
                    memo.spot_direction[i * 4 + 0] = static_cast<f32>(light.spot_x) / 2047.0f;
                    memo.spot_direction[i * 4 + 1] = static_cast<f32>(light.spot_y) / 2047.0f;
                    memo.spot_direction[i * 4 + 2] = static_cast<f32>(light.spot_z) / 2047.0f;
                    // x sesgo, y escala: el mismo orden que espera el shader.
                    memo.dist_atten[i * 4 + 0] =
                        Pica::f20::FromRaw(light.dist_atten_bias).ToFloat32();
                    memo.dist_atten[i * 4 + 1] =
                        Pica::f20::FromRaw(light.dist_atten_scale).ToFloat32();
                }
                put_color(lighting_regs.global_ambient, memo.global_ambient.data());
                memo.lights_dirty = false;
            }
            const auto put = [&](const SceGxmProgramParameter* param,
                                 const std::array<f32, 32>& values) {
                if (param != nullptr) {
                    sceGxmSetUniformDataF(uniform_buffer, param, 0, 32, values.data());
                }
            };
            put(pipeline->light_specular_0, memo.specular_0);
            put(pipeline->light_specular_1, memo.specular_1);
            put(pipeline->light_diffuse, memo.diffuse);
            put(pipeline->light_ambient, memo.ambient);
            put(pipeline->light_position, memo.position);
            put(pipeline->light_spot_direction, memo.spot_direction);
            put(pipeline->light_dist_atten, memo.dist_atten);
            if (pipeline->lighting_global_ambient != nullptr) {
                sceGxmSetUniformDataF(uniform_buffer, pipeline->lighting_global_ambient, 0, 4,
                                      memo.global_ambient.data());
            }
        }
    }

    return true;
}

void RasterizerGXM::DrawTriangles() {
    batch_skip = false;
    if (!batch_on_gpu) {
        batch_decided = false;
        batch.clear();
        return;
    }
    // Una lectura de reloj por lote (unos trescientos por fotograma). Ver
    // Common::FrameStats::batch_us.
    const unsigned long long batch_begin = Common::VitaMicros();
    const bool drawn = DrawBatchOnGpu();
    if (!drawn) {
        // No se ha podido dibujar en la GPU (sin memoria, shader rechazado a
        // ultima hora): el lote entero vuelve al camino de software, que es la
        // referencia. La escena se cierra antes para que el software lea la
        // memoria ya volcada.
        FlushPending();
        PrepareSoftwareBatch();
        for (std::size_t i = 0; i + 2 < batch.size(); i += 3) {
            software.AddTriangle(batch[i], batch[i + 1], batch[i + 2]);
            software_triangles.fetch_add(1, std::memory_order_relaxed);
        }
    }
    batch.clear();
    batch_decided = false;
    batch_on_gpu = false;
    Common::FrameStats::Add(Common::FrameStats::batch_us, batch_begin);
}

void RasterizerGXM::FlushAll() {
    FlushPending();
    software.FlushAll();
}

void RasterizerGXM::FlushRegion(PAddr addr, u32 size) {
    // Alguien va a leer una pantalla copiada en la GPU: a la memoria.
    MaterializeCopies(addr, size);
    /**
     * Volcar SOLO lo que pisa el tramo, y cerrar la escena SOLO si hay algo que
     * pisar.
     *
     * Estos avisos llegan muchas veces por fotograma y casi ninguno habla de un
     * framebuffer nuestro. Cerrar la escena en todos ellos era pagar dos
     * precios: esperar a la GPU sin motivo y, sobre todo, gastar escenas --
     * cada render target se crea con un numero esperado de escenas por
     * fotograma (params.scenesPerFrame) y pasarse cuesta rendimiento. EndScene
     * dentro del bucle es idempotente: la primera superficie que pise el tramo
     * la cierra y las demas ya la encuentran cerrada.
     */
    for (auto& surface : surfaces) {
        if (!surface->Overlaps(addr, size)) {
            continue;
        }
        /**
         * Y SOLO SI HAY ALGO QUE VOLCAR.
         *
         * Aqui se cerraba la escena y despues se llamaba a WriteBack, que lo
         * primero que hace es volverse si la superficie no esta sucia. O sea
         * que cuando no habia nada que devolver al invitado -- que es el caso
         * corriente, porque estos avisos llegan muchas veces por fotograma --
         * ya se habia pagado el EndScene: un sceGxmFinish, que PARA la CPU
         * hasta que el chip grafico termina todo lo que tenia pendiente.
         *
         * Si la superficie no esta sucia, la memoria del invitado ya esta al
         * dia y no hay razon para esperar a nadie.
         */
        if (!surface->dirty) {
            continue;
        }
        EndScene();
        WriteBack(*surface);
    }
    // El cache de texturas se crea en el primer lote, y estos vaciados llegan
    // desde el arranque del juego, mucho antes: sin la comprobacion esto era
    // una desreferencia de puntero nulo (la caida de 0.1.0.2 a 0.1.0.4).
    if (textures != nullptr) {
        textures->InvalidateRange(addr, size);
    }
    software.FlushRegion(addr, size);
}

void RasterizerGXM::InvalidateRegion(PAddr addr, u32 size) {
    // El invitado escribe encima de una pantalla copiada en la GPU: se olvida.
    DropCopies(addr, size);
    /**
     * Invalidar NO es volcar: es lo contrario, y confundirlo se comia el
     * trabajo del juego.
     *
     * Este aviso llega DESPUES de que el invitado haya cambiado esa memoria por
     * otro camino (un relleno de color, una transferencia). Volcar ahi nuestra
     * superficie -- que es lo que hacia antes, porque llamaba a FlushPending --
     * le pasaba por encima al borrado que el juego acababa de hacer. Lo
     * correcto es olvidar lo nuestro y volver a leer del invitado antes del
     * siguiente lote.
     */
    for (auto& surface : surfaces) {
        const bool hits_color = surface->Overlaps(addr, size);
        // El tramo puede ser el de la PROFUNDIDAD y no el del color: es como
        // llega el borrado del buffer de profundidad entre fotograma y
        // fotograma, y hasta 0.1.0.16 no lo miraba nadie (ver OverlapsDepth).
        const bool hits_depth = surface->OverlapsDepth(addr, size);
        if (!hits_color && !hits_depth) {
            continue;
        }
        // Si la escena abierta esta dibujando justo ahi, hay que pararla: lo
        // que contenga esa superficie lo decide ahora el invitado. Tambien
        // cuando lo que cambia es la profundidad: el borrado se aplica al abrir
        // la escena siguiente, asi que con esta abierta se quedaria sin hacer.
        EndScene();
        if (hits_color) {
            surface->dirty = false;
            surface->needs_reload = true;
            surface->clear_pending = false;
            surface->clear_count = 0;
        }
        if (hits_depth) {
            surface->depth_needs_clear = true;
        }
    }
    if (textures != nullptr) {
        textures->InvalidateRange(addr, size);
    }
    software.InvalidateRegion(addr, size);
}

void RasterizerGXM::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    MaterializeCopies(addr, size);
    DropCopies(addr, size);
    // Las dos cosas y en este orden: el invitado se lleva lo que hemos dibujado
    // y despues lo que el haga ahi es lo que manda.
    for (auto& surface : surfaces) {
        const bool hits_color = surface->Overlaps(addr, size);
        const bool hits_depth = surface->OverlapsDepth(addr, size);
        if (!hits_color && !hits_depth) {
            continue;
        }
        // Igual que en FlushRegion: si no hay nada sucio que devolver y la
        // profundidad no esta en juego, no se para la GPU para nada.
        if (!surface->dirty && !hits_depth) {
            surface->needs_reload = surface->needs_reload || hits_color;
            continue;
        }
        EndScene();
        if (hits_color) {
            // Volcar solo tiene sentido para el color: la profundidad no se
            // comparte con el invitado, asi que ahi no hay nada que devolver.
            WriteBack(*surface);
            surface->needs_reload = true;
        }
        if (hits_depth) {
            surface->depth_needs_clear = true;
        }
    }
    if (textures != nullptr) {
        textures->InvalidateRange(addr, size);
    }
    software.FlushAndInvalidateRegion(addr, size);
}

void RasterizerGXM::ClearAll(bool flush) {
    FlushPending();
    if (textures != nullptr) {
        textures->Clear();
    }
    software.ClearAll(flush);
}

/**
 * Cada vez que un lote NO puede ir por el shader de vertices en la GPU, el
 * motivo queda aqui (0.1.4.9) y el overlay lo pinta junto a 'vsg'. En 0.1.4.8
 * el unico sitio donde se veia era crash.txt, y sin ese fichero no habia
 * forma de saber por que 'vsg' se quedaba en 12 de 177 lotes. Los motivos son
 * siempre literales de texto, que viven lo que el programa.
 */
static bool HwVsReject(const char* why) {
    RasterizerGXM::hw_vs_last_reject.store(why, std::memory_order_relaxed);
    RasterizerGXM::hw_vs_rejects.fetch_add(1, std::memory_order_relaxed);
    // Solo lo llama el hilo de la GPU: el hueco libre no se lo disputa nadie.
    for (auto& slot : RasterizerGXM::reject_counts) {
        const char* reason = slot.reason.load(std::memory_order_relaxed);
        if (reason == nullptr) {
            slot.reason.store(why, std::memory_order_relaxed);
            reason = why;
        }
        if (reason == why) {
            slot.count.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }
    return false;
}

u8* RasterizerGXM::ReserveVertexSpace(u32 bytes) {
    /**
     * EL ANILLO EN TRAMOS (0.2.1.1). Al llegar al final del anillo se cerraba
     * la escena y se esperaba a la GPU ENTERA antes de volver al principio
     * (sceGxmFinish): en 3D, con varios MB de vertices por fotograma, la CPU
     * se paraba a esperar a la GPU cada uno o dos fotogramas, en vez de dejar
     * que la GPU dibujara el fotograma anterior mientras la CPU prepara el
     * siguiente. Ahora son 8 MB en ocho tramos, cada uno con la valla de la
     * ultima escena que lo leyo: al entrar otra vez en un tramo solo se espera
     * a esa escena, que casi siempre ha terminado hace rato. Solo si la escena
     * abierta ya tiene datos en el tramo (ha llenado el anillo entero) se cierra
     * y se espera como antes. Se alinea a 16 porque los flujos de vertices y
     * los indices lo prefieren.
     */
    if (!vertex_buffer.Valid() || bytes > vertex_buffer.Size()) {
        // Nuevo, o mas grande para un lote enorme: antes, que nadie lea el viejo.
        EndScene();
        WaitGpu();
        const u32 size = std::max(kVertexBufferBytes, (bytes + 0xFFFFFu) & ~0xFFFFFu);
        Allocation buffer;
        if (size == kVertexBufferBytes) {
            buffer = Allocate(Pool::Cdram, size);
        }
        if (!buffer.Valid()) {
            buffer = Allocate(Pool::Host, size);
        }
        if (!buffer.Valid()) {
            return nullptr;
        }
        vertex_buffer = std::move(buffer);
        vertex_used = 0;
        vertex_segment_fence.fill(fence_sent);
        vertex_segments_pending = 0;
    }
    const u32 size = vertex_buffer.Size();
    const u32 span = std::max(bytes, 1u);
    u32 offset = (vertex_used + 15u) & ~15u;
    const bool wrapped = offset + span > size;
    if (wrapped) {
        offset = 0;
    }
    const u32 segment_bytes = size / kVertexSegments;
    const u32 first = offset / segment_bytes;
    const u32 last = (offset + span - 1) / segment_bytes;
    // El tramo donde acabo la reserva anterior: seguir en el no pisa nada.
    const u32 current =
        vertex_used == 0 || wrapped ? kVertexSegments : (vertex_used - 1) / segment_bytes;
    for (u32 segment = first; segment <= last; segment++) {
        if (segment == current) {
            continue;
        }
        if (((vertex_segments_pending >> segment) & 1u) != 0) {
            scene_close_full.fetch_add(1, std::memory_order_relaxed);
            EndScene();
            WaitGpu();
            break;
        }
        WaitFence(vertex_segment_fence[segment]);
    }
    for (u32 segment = first; segment <= last; segment++) {
        vertex_segments_pending |= 1u << segment;
    }
    vertex_used = offset + span;
    return static_cast<u8*>(vertex_buffer.Data()) + offset;
}

/**
 * EL SHADER DE VERTICES DE LA PICA, EN LA GPU (0.1.4.6).
 *
 * POR QUE. En la cinematica de Rubi Omega (0.1.4.5), de 283 ms por vblank unos
 * 130 eran trabajo POR VERTICE en la CPU: sombrear (92 ms, ya repartido en tres
 * nucleos), entregar al ensamblador (19) y empaquetar para GXM (~17). Aqui la
 * CPU solo copia los datos crudos de los vertices y los indices -- una memcpy
 * por flujo -- y el programa de la PICA, traducido a Cg (cg_vs_shader_gen), lo
 * ejecuta la GPU, que en un juego en 3D estaba casi parada.
 *
 * LA REGLA: NO HAY VUELTA ATRAS DESPUES DE DEVOLVER TRUE. PicaCore se salta el
 * interprete entero si esto devuelve true, asi que TODO lo que pueda fallar se
 * comprueba ANTES de tocar la memoria o el estado: si algo no cuadra, false, y
 * el lote va por el camino de siempre (interprete + AddTriangle), que sigue
 * siendo el de referencia. El orden de esta funcion es ese: primero todas las
 * preguntas, despues el trabajo.
 *
 * LO QUE NO ES IGUAL BIT A BIT, dicho claramente: aqui las cuentas del shader
 * las hace la GPU, no el interprete. RCP, RSQ, EX2 y LG2 salen de las unidades
 * de la GPU y no de las funciones de la libreria de C, y el 0 x infinito de la
 * PICA (que da 0) aqui da NaN. Es lo mismo que hace Azahar en escritorio con
 * los shaders por hardware activados, que es su opcion por defecto. Para
 * comparar con el interprete en la consola esta la ablacion "vs en cpu"
 * (SELECT + ABAJO), que devuelve el lote a la ruta de 0.1.4.5.
 */
bool RasterizerGXM::AccelerateDrawBatch(bool is_indexed) {
    /**
     * 'lote' cuenta la funcion ENTERA desde 0.1.9.5, preguntas y rechazos
     * incluidos: en el titulo de Zafiro Alfa (2D, ~230 lotes por fotograma)
     * "gx" eran 21 ms y "lote" solo 1,7, y no se veia donde iba el resto.
     *
     * Y POR FASES (0.1.9.6), uno de cada 8 lotes: en que se va el coste fijo
     * de cada lote (en el 2D de Zafiro Alfa, ~70 us por lote y 230 lotes por
     * fotograma). Solo cuentan los lotes que llegan a dibujarse. Desde 0.2.1.0
     * el total del lote tambien sale de esos, por ocho: leer el reloj es una
     * llamada al kernel, y eran dos por lote.
     */
    static u32 profile_tick = 0;
    const bool profile = (++profile_tick & 7u) == 0;
    struct BatchTimer {
        bool timed;
        unsigned long long begin = timed ? Common::VitaMicros() : 0;
        ~BatchTimer() {
            if (timed) {
                Common::FrameStats::batch_us.fetch_add((Common::VitaMicros() - begin) * 8,
                                                       std::memory_order_relaxed);
            }
        }
    } const batch_timer{profile};
    /**
     * SALTO DE FOTOGRAMAS EN LA GPU (0.1.9.6). Hasta ahora SELECT+L/R solo se
     * saltaba la PRESENTACION: la GPU emulada dibujaba todos los fotogramas
     * igual, y por eso no se ganaba nada. En un fotograma saltado el lote se
     * da por dibujado sin hacer nada: ni vertices en la CPU ni GPU.
     */
    if (!SwRenderer::FrameSkip::ShouldRender()) {
        return true;
    }
    std::array<unsigned long long, kBatchPhases> phase_us{};
    unsigned long long phase_mark = profile ? Common::VitaMicros() : 0;
    const auto phase = [&](u32 index) {
        if (profile) {
            const unsigned long long now = Common::VitaMicros();
            phase_us[index] += now - phase_mark;
            phase_mark = now;
        }
    };
    using Pica::PipelineRegs;
    using Pica::Shader::Generator::GXM::VSInputs;
    using Pica::Shader::Generator::GXM::VSInputSource;
    using Location = BatchMemo::Location;
    using AttributeSource = BatchMemo::AttributeSource;

    const u32 ablation = Ablation::mode.load(std::memory_order_relaxed);
    if (ablation == Ablation::kCpuVertexShader || ablation == Ablation::kNoGpu) {
        return HwVsReject("ablacion vs");
    }
    if (!EnsureInitialized() || hw_shaders == nullptr) {
        return HwVsReject("sin gxm");
    }
    auto& regs = pica.regs.internal;
    const auto& pipeline_regs = regs.pipeline;

    // ---- 1. Todas las preguntas ----

    SceGxmPrimitiveType primitive{};
    switch (pipeline_regs.triangle_topology.Value()) {
    case PipelineRegs::TriangleTopology::List:
    case PipelineRegs::TriangleTopology::Shader:
        // Sin shader de geometria (PicaCore no llega aqui con uno), "Shader"
        // ensambla igual que una lista: ver PrimitiveAssembler.
        primitive = SCE_GXM_PRIMITIVE_TRIANGLES;
        break;
    case PipelineRegs::TriangleTopology::Strip:
        primitive = SCE_GXM_PRIMITIVE_TRIANGLE_STRIP;
        break;
    case PipelineRegs::TriangleTopology::Fan:
        primitive = SCE_GXM_PRIMITIVE_TRIANGLE_FAN;
        break;
    default:
        return HwVsReject("topologia");
    }
    if (regs.rasterizer.depthmap_enable == RasterizerRegs::DepthBuffering::WBuffering) {
        return HwVsReject("wbuffer");
    }
    // Lo que no haya cambiado desde el lote anterior se reutiliza (BatchMemo).
    BatchMemo& memo = *batch_memo;
    {
        bool fs_dirty = false;
        bool vs_dirty = false;
        const auto& dirty = pica.dirty_regs.qwords;
        for (u32 i = 0; i < dirty.size(); i++) {
            fs_dirty = fs_dirty || (dirty[i] & kFsMemoMask[i]) != 0;
            vs_dirty = vs_dirty || (dirty[i] & kVsMemoMask[i]) != 0;
        }
        if (fs_dirty) {
            memo.pipeline = nullptr;
        }
        if (fs_dirty || vs_dirty) {
            memo.vs_valid = false;
        }
        memo.lights_dirty = memo.lights_dirty || AnyDirty(pica.dirty_regs, kLightsMask);
        memo.fog_dirty = memo.fog_dirty || AnyDirty(pica.dirty_regs, kFogLutMask);
        memo.colors_dirty = memo.colors_dirty || AnyDirty(pica.dirty_regs, kConstColorMask);
        pica.dirty_regs.Reset();
    }
    const char* fs_reason = nullptr;
    const PipelineCache::Entry* pipeline = memo.pipeline;
    if (pipeline == nullptr) {
        memo.vs_valid = false;
        pipeline = pipelines->Get(regs, &fs_reason);
        // Su shader de fragmentos se esta compilando: el lote se salta (0.1.9.8).
        if (pipeline == nullptr && fs_reason != nullptr &&
            std::strcmp(fs_reason, "fs compilando") == 0) {
            skipped_batches.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        memo.pipeline = pipeline;
    }
    if (pipeline == nullptr || AblatedByMode()) {
        // El motivo lo anota la ruta de la CPU al llegar a AddTriangle.
        return HwVsReject("fragmentos");
    }
    phase(0);

    const u32 num_vertices = pipeline_regs.num_vertices;
    if (num_vertices == 0) {
        return HwVsReject("sin vertices");
    }
    const auto& attributes = pipeline_regs.vertex_attributes;
    const PAddr base_address = attributes.GetPhysicalBaseAddress();

    // El tramo de vertices que se usa: de ahi se copian los datos.
    u32 min_index = 0;
    u32 max_index = 0;
    const u8* index_data = nullptr;
    const bool index_u16 = pipeline_regs.index_array.format != 0;
    if (is_indexed) {
        index_data = memory.GetPhysicalPointer(base_address + pipeline_regs.index_array.offset);
        if (index_data == nullptr) {
            return HwVsReject("indices");
        }
        min_index = 0xFFFFFFFFu;
        if (index_u16) {
            const auto* indices = reinterpret_cast<const u16*>(index_data);
            for (u32 i = 0; i < num_vertices; i++) {
                min_index = std::min<u32>(min_index, indices[i]);
                max_index = std::max<u32>(max_index, indices[i]);
            }
        } else {
            for (u32 i = 0; i < num_vertices; i++) {
                min_index = std::min<u32>(min_index, index_data[i]);
                max_index = std::max<u32>(max_index, index_data[i]);
            }
        }
    } else {
        // Sin indices se usa el buffer de indices secuenciales, que tiene
        // kMaxVerticesPerDraw entradas.
        if (num_vertices > kMaxVerticesPerDraw) {
            return HwVsReject("demasiados vertices");
        }
        min_index = pipeline_regs.vertex_offset;
        max_index = min_index + num_vertices - 1;
    }
    const u32 vertex_range = max_index - min_index + 1;

    /**
     * La disposicion de los atributos y el programa de vertices, los del lote
     * anterior si la PICA no ha tocado nada que los decida (BatchMemo).
     */
    BatchMemo::VertexLayout& layout = memo.layout;
    if (!memo.vs_valid) {
        /**
         * Donde vive cada atributo, con la misma cuenta que VertexLoader: cada
         * cargador va poniendo sus componentes seguidas, alineando cada una a su
         * tamano de elemento, y los codigos 12-15 son relleno de 4 a 16 bytes. Si
         * dos cargadores nombran el mismo atributo, gana el ultimo (el constructor
         * de VertexLoader lo pisa igual).
         */
        auto& locations = layout.locations;
        locations = {};
        for (u32 loader = 0; loader < 12; loader++) {
            const auto& config = attributes.attribute_loaders[loader];
            if (config.component_count == 0 || config.byte_count == 0) {
                continue;
            }
            u32 offset = 0;
            for (u32 component = 0; component < config.component_count && component < 12;
                 component++) {
                const u32 attribute = config.GetComponent(component);
                if (attribute < 12) {
                    offset = Common::AlignUp(offset, attributes.GetElementSizeInBytes(attribute));
                    locations[attribute] = {true, static_cast<u8>(loader),
                                            static_cast<u16>(offset)};
                    offset += attributes.GetStride(attribute);
                } else {
                    offset = Common::AlignUp(offset, 4);
                    offset += (attribute - 11) * 4;
                }
            }
        }

        /**
         * Y de donde sale cada REGISTRO de entrada: como ShaderUnit::LoadInput, el
         * atributo 'a' va al registro GetRegisterForAttribute(a) para a de 0 al
         * ultimo indice, y los posteriores pisan a los anteriores.
         */
        const u32 total_attributes = attributes.GetNumTotalAttributes();
        auto& inputs = layout.inputs;
        auto& input_attribute = layout.input_attribute;
        inputs = {};
        input_attribute = {};
        for (u32 attribute = 0; attribute <= regs.vs.max_input_attribute_index && attribute < 16;
             attribute++) {
            const u32 reg = regs.vs.GetRegisterForAttribute(attribute);
            if (reg >= 16) {
                continue;
            }
            VSInputSource source{};
            if (attribute < total_attributes && attribute < 12) {
                if (attributes.IsDefaultAttribute(attribute)) {
                    source.kind = VSInputSource::Default;
                } else if (locations[attribute].found &&
                           attributes.GetNumElements(attribute) != 0) {
                    source.kind = VSInputSource::Array;
                    source.components = static_cast<u8>(attributes.GetNumElements(attribute));
                }
            }
            inputs[reg] = source;
            input_attribute[reg] = static_cast<u8>(attribute);
        }

        // El programa traducido y compilado.
        const Pica::Shader::Generator::PicaVSConfig vs_config{regs, pica.vs_setup};
        u64& program_key = layout.program_key;
        program_key = vs_config.Hash();
        for (u32 reg = 0; reg < 16; reg++) {
            program_key = Common::HashCombine(program_key,
                                              (static_cast<u64>(inputs[reg].kind) << 8) |
                                                  inputs[reg].components);
        }
        // La variante de salidas es la del shader de fragmentos (ver
        // GenerateVertexShader): entra en la clave.
        program_key = Common::HashCombine(program_key, (pipeline->lit ? 2u : 0u) |
                                                           (pipeline->proj ? 1u : 0u));
        if (hw_shaders->unlinkable.count(program_key) != 0) {
            return HwVsReject("vs enlazar atributos");
        }
        const char* reason = nullptr;
        const HwShaderCache::Program*& program = layout.program;
        program = hw_shaders->GetProgram(program_key, pica.vs_setup, vs_config, inputs,
                                         pipeline->lit, pipeline->proj, &reason);
        /**
         * PROGRAMA ESPECIALIZADO CON LOS BOOLEANOS DEL LOTE (0.1.7.4).
         *
         * crash.txt de 0.1.6.3: el shader de piel de Rubi Omega se queda en la CPU
         * por "salto en 0x130 -> 0x157" fuera de su tramo, y ese salto es un JMPU
         * (salta segun un uniform booleano). Con los saltos de escape puestos, el
         * mismo shader rompe el compilador (cg_error.txt). Pero los booleanos no
         * cambian dentro de un lote: traducido con sus valores como constantes, el
         * JMPU es un NOP o un salto fijo, y los IFU/CALLU que no se toman
         * desaparecen con todo su codigo. El programa queda mucho mas pequeno y
         * sencillo para el compilador y para la GPU.
         *
         * Solo si la traduccion generica no sirve, para no multiplicar programas
         * que ya funcionan. La clave lleva los booleanos que el codigo lee.
         */
        if (program == nullptr && reason != nullptr &&
            specialize_vs.load(std::memory_order_relaxed) != 0 &&
            std::strcmp(reason, "vs tope de programas") != 0 &&
            std::strcmp(reason, "vs compilador roto") != 0 &&
            std::strcmp(reason, "vs compilando") != 0 &&
            std::strcmp(reason, "vs con geometria") != 0) {
            using namespace Pica::Shader::Generator::GXM;
            auto used = hw_shaders->used_bools.find(program_key);
            if (used == hw_shaders->used_bools.end()) {
                used = hw_shaders->used_bools.emplace(program_key, UsedBoolUniforms(pica.vs_setup))
                           .first;
            }
            u32 bools = 0;
            for (u32 i = 0; i < 16; i++) {
                if (((used->second >> i) & 1u) != 0 && pica.vs_setup.uniforms.b[i]) {
                    bools |= 1u << i;
                }
            }
            const u64 special_key =
                Common::HashCombine(program_key, 0xB0010000ull | (bools & 0xFFFFu));
            const char* special_reason = nullptr;
            // 0.1.7.9: un especializado NUEVO solo si queda hueco en el tope de
            // ese shader y heap de sobra. Los ya hechos se siguen usando siempre.
            bool may_build = true;
            if (!hw_shaders->Has(special_key)) {
                u32& count = hw_shaders->specialized_count[program_key];
                if (count >= HwShaderCache::kMaxSpecializedPerProgram) {
                    special_reason = "vs tope de especializados";
                    may_build = false;
                } else if (CgHeapLow()) {
                    special_reason = "vs sin heap para especializar";
                    may_build = false;
                } else {
                    count++;
                }
            }
            if (may_build) {
                g_cg_const_bools.store(kCgBoolsKnown | bools, std::memory_order_relaxed);
                program = hw_shaders->GetProgram(special_key, pica.vs_setup, vs_config, inputs,
                                                 pipeline->lit, pipeline->proj, &special_reason);
                g_cg_const_bools.store(0, std::memory_order_relaxed);
            }
            if (program != nullptr) {
                static u32 special_notes = 0;
                if (special_notes < 4) {
                    special_notes++;
                    NoteFmt("gxm vs", "especializado con booleanos {:#06x} ({}): a la GPU", bools,
                            reason);
                }
            } else if (special_reason != nullptr) {
                reason = special_reason;
            }
        }
        if (program == nullptr) {
            // Compilandose (el generico o su especializado): el lote se salta.
            if (reason != nullptr && std::strcmp(reason, "vs compilando") == 0 &&
                async_vs.load(std::memory_order_relaxed) != 0) {
                skipped_batches.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            NoteSkip(5, reason != nullptr ? reason : "vs");
            return HwVsReject(reason != nullptr ? reason : "vs");
        }
        phase(1);

        /**
         * Flujos y atributos de GXM. Un flujo por cargador que aporte algun
         * registro que el programa lee; cada atributo, en su flujo y su
         * desplazamiento, con el formato de la PICA SIN normalizar (la PICA
         * convierte el entero a flotante tal cual) y las componentes que traiga:
         * las que faltan las rellena el propio shader (ver VSInputs).
         */
        auto& gxm_attributes = layout.gxm_attributes;
        auto& gxm_streams = layout.gxm_streams;
        auto& stream_of_loader = layout.stream_of_loader;
        auto& loader_of_stream = layout.loader_of_stream;
        auto& attribute_sources = layout.attribute_sources;
        u32& attribute_count = layout.attribute_count;
        u32& stream_count = layout.stream_count;
        u64& layout_key = layout.layout_key;
        gxm_attributes = {};
        gxm_streams = {};
        stream_of_loader.fill(0xFF);
        loader_of_stream = {};
        attribute_sources = {};
        attribute_count = 0;
        stream_count = 0;
        layout_key = program_key;
        for (u32 reg = 0; reg < 16; reg++) {
            if (inputs[reg].kind != VSInputSource::Array || program->inputs[reg] == nullptr) {
                continue;
            }
            const u32 attribute = input_attribute[reg];
            const Location& location = locations[attribute];
            if (stream_of_loader[location.loader] == 0xFF) {
                stream_of_loader[location.loader] = static_cast<u8>(stream_count);
                loader_of_stream[stream_count] = location.loader;
                gxm_streams[stream_count].stride =
                    static_cast<u16>(attributes.attribute_loaders[location.loader].byte_count);
                gxm_streams[stream_count].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
                stream_count++;
            }
            attribute_sources[attribute_count] = {
                stream_of_loader[location.loader],
                static_cast<u8>(attributes.GetFormat(attribute)), location.offset};
            SceGxmVertexAttribute& out = gxm_attributes[attribute_count++];
            out.streamIndex = stream_of_loader[location.loader];
            out.offset = location.offset;
            switch (attributes.GetFormat(attribute)) {
            case PipelineRegs::VertexAttributeFormat::BYTE:
                out.format = SCE_GXM_ATTRIBUTE_FORMAT_S8;
                break;
            case PipelineRegs::VertexAttributeFormat::UBYTE:
                out.format = SCE_GXM_ATTRIBUTE_FORMAT_U8;
                break;
            case PipelineRegs::VertexAttributeFormat::SHORT:
                out.format = SCE_GXM_ATTRIBUTE_FORMAT_S16;
                break;
            default:
                out.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
                break;
            }
            /**
             * No mas componentes de las que tiene el parametro COMPILADO (0.1.8.9).
             * El compilador quita las que el shader no lee (un float4 del que solo
             * se usa .xy queda en 2), y pasarle a GXM mas de las que hay es el
             * 0x805b0023 de los shaders de piel de Zafiro Alfa: 213 lotes por
             * fotograma sombreados en la CPU.
             */
            out.componentCount = static_cast<u8>(
                std::min<u32>(inputs[reg].components,
                              sceGxmProgramParameterGetComponentCount(program->inputs[reg])));
            out.regIndex =
                static_cast<u16>(sceGxmProgramParameterGetResourceIndex(program->inputs[reg]));
            attribute_sources[attribute_count - 1].components = out.componentCount;
            attribute_sources[attribute_count - 1].declared = static_cast<u8>(
                std::clamp<u32>(sceGxmProgramParameterGetComponentCount(program->inputs[reg]),
                                out.componentCount, 4));
            layout_key = Common::HashCombine(
                layout_key, (static_cast<u64>(out.streamIndex) << 48) |
                                (static_cast<u64>(out.offset) << 32) |
                                (static_cast<u64>(out.format) << 24) |
                                (static_cast<u64>(out.componentCount) << 16) | out.regIndex);
        }
        for (u32 stream = 0; stream < stream_count; stream++) {
            layout_key = Common::HashCombine(layout_key, gxm_streams[stream].stride);
        }
        SceGxmVertexProgram*& vertex_program = layout.vertex_program;
        vertex_program = hw_shaders->GetLinked(layout_key, *program, gxm_attributes.data(),
                                               attribute_count, gxm_streams.data(), stream_count);
        /**
         * DISPOSICION CONVERTIDA (0.1.8.5). crash.txt de 0.1.8.4, Zafiro Alfa: el
         * shader de piel ya compila especializado, pero GXM rechaza enlazarlo con
         * los atributos tal como los guarda el 3DS ("crear programa de vertices err
         * 0x805b0023 (5 atributos, 1 flujos)") y 213 de 246 lotes por segundo
         * vuelven a la CPU. Esos modelos llevan indices y pesos de huesos en bytes,
         * con desplazamientos y paso que no son multiplos de 4. Si la disposicion
         * original no vale, se prueba la mas sencilla que hay: un solo flujo con
         * todos los atributos en float de 32 bits seguidos y alineados, convertidos
         * en la CPU solo para los vertices del lote. Los valores son los mismos: la
         * PICA pasa los enteros a float tal cual, sin normalizar.
         */
        bool& converted = layout.converted;
        u32& converted_stride = layout.converted_stride;
        converted = false;
        converted_stride = 0;
        if (vertex_program == nullptr) {
            u64 converted_key = Common::HashCombine(program_key, 0xF32F32F3ull);
            for (u32 i = 0; i < attribute_count; i++) {
                SceGxmVertexAttribute& out = gxm_attributes[i];
                out.streamIndex = 0;
                out.offset = static_cast<u16>(converted_stride);
                out.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
                /**
                 * Todas las componentes que declara el programa (0.1.9.2). crash.txt
                 * de 0.1.9.1: los shaders de piel declaran float4 en cada entrada y
                 * GXM no los enlaza con 2 o 3 (err 0x805b0023, tambien en float).
                 * Las que el invitado no trae van con los valores por defecto de
                 * la PICA: 0 y la w a 1.
                 */
                out.componentCount = attribute_sources[i].declared;
                converted_stride += out.componentCount * 4u;
                converted_key = Common::HashCombine(
                    converted_key, (static_cast<u64>(out.componentCount) << 16) | out.regIndex);
            }
            SceGxmVertexStream converted_stream{};
            converted_stream.stride = static_cast<u16>(converted_stride);
            converted_stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
            if (attribute_count > 0) {
                vertex_program =
                    hw_shaders->GetLinked(converted_key, *program, gxm_attributes.data(),
                                          attribute_count, &converted_stream, 1);
            }
            if (vertex_program != nullptr) {
                converted = true;
                static bool noted_converted = false;
                NoteOnce(noted_converted, "gxm vs",
                         "atributos convertidos a float: enlaza ({} atributos, paso {})",
                         attribute_count, converted_stride);
            }
        }
        if (vertex_program == nullptr) {
            hw_shaders->unlinkable.insert(program_key);
            NoteSkip(5, "vs enlazar atributos");
            return HwVsReject("vs enlazar atributos");
        }
        memo.vs_valid = true;
    } else {
        phase(1);
    }
    phase(2);
    const VSInputs& inputs = layout.inputs;
    const auto& input_attribute = layout.input_attribute;
    const HwShaderCache::Program* const program = layout.program;
    SceGxmVertexProgram* const vertex_program = layout.vertex_program;
    const auto& gxm_attributes = layout.gxm_attributes;
    const auto& loader_of_stream = layout.loader_of_stream;
    const auto& attribute_sources = layout.attribute_sources;
    const u32 attribute_count = layout.attribute_count;
    // Local: el camino convertido lo deja en uno.
    u32 stream_count = layout.stream_count;
    const bool converted = layout.converted;
    const u32 converted_stride = layout.converted_stride;
    // El MISMO programa de fragmentos que la ruta de la CPU: el shader de
    // vertices traducido escribe exactamente los varyings de la variante fija
    // contra la que se enlazo (FormatFor(lit, proj)).
    SceGxmFragmentProgram* fragment_program = pipeline->program;

    // Los datos de cada flujo tienen que estar enteros y seguidos en la
    // memoria del invitado.
    std::array<const u8*, 12> stream_source{};
    std::array<u32, 12> stream_bytes{};
    u32 total_bytes = 0;
    for (u32 stream = 0; stream < stream_count; stream++) {
        const auto& loader = attributes.attribute_loaders[loader_of_stream[stream]];
        const PAddr start = base_address + loader.data_offset + min_index * loader.byte_count;
        const u32 bytes = loader.byte_count * vertex_range;
        const u8* first = memory.GetPhysicalPointer(start);
        const u8* last = memory.GetPhysicalPointer(start + bytes - 1);
        if (first == nullptr || last != first + bytes - 1) {
            return HwVsReject("datos partidos");
        }
        stream_source[stream] = first;
        stream_bytes[stream] = bytes;
        total_bytes += (bytes + 15u) & ~15u;
    }
    if (converted) {
        total_bytes = (converted_stride * vertex_range + 15u) & ~15u;
    }
    const u32 index_bytes = is_indexed ? ((num_vertices * 2u + 15u) & ~15u) : 0u;
    total_bytes += index_bytes;

    Surface* surface = CurrentSurface();
    if (surface == nullptr) {
        return HwVsReject("superficie");
    }

    // ---- 2. El trabajo. A partir de aqui solo quedan fallos de GXM ----

    if (open_surface != nullptr && open_surface != surface) {
        EndScene();
    }
    // Sin espera a la GPU desde 0.1.9.4: ver DrawBatchOnGpu.
    if (pipeline->lighting_lut != nullptr && !UpdateLightingLut()) {
        return HwVsReject("tablas luz");
    }
    if (index_capacity == 0) {
        Allocation buffer = Allocate(Pool::Host, kMaxVerticesPerDraw * sizeof(u16));
        if (!buffer.Valid()) {
            return HwVsReject("memoria indices");
        }
        index_buffer = std::move(buffer);
        index_capacity = kMaxVerticesPerDraw;
        auto* sequential = static_cast<u16*>(index_buffer.Data());
        for (u32 i = 0; i < index_capacity; i++) {
            sequential[i] = static_cast<u16>(i);
        }
    }
    u8* space = ReserveVertexSpace(total_bytes);
    if (space == nullptr) {
        return HwVsReject("memoria vertices");
    }
    std::array<const u8*, 12> stream_data{};
    if (converted) {
        // Un float por componente, en el orden de gxm_attributes (ver arriba).
        float* out = reinterpret_cast<float*>(space);
        for (u32 v = 0; v < vertex_range; v++) {
            for (u32 i = 0; i < attribute_count; i++) {
                const AttributeSource& source = attribute_sources[i];
                const u32 stride =
                    attributes.attribute_loaders[loader_of_stream[source.stream]].byte_count;
                const u8* in = stream_source[source.stream] + v * stride + source.offset;
                const u32 components = source.components;
                for (u32 c = components; c < gxm_attributes[i].componentCount; c++) {
                    out[c] = c == 3 ? 1.0f : 0.0f;
                }
                for (u32 c = 0; c < components; c++) {
                    switch (static_cast<PipelineRegs::VertexAttributeFormat>(source.format)) {
                    case PipelineRegs::VertexAttributeFormat::BYTE:
                        *out++ = static_cast<float>(static_cast<s8>(in[c]));
                        break;
                    case PipelineRegs::VertexAttributeFormat::UBYTE:
                        *out++ = static_cast<float>(in[c]);
                        break;
                    case PipelineRegs::VertexAttributeFormat::SHORT: {
                        s16 value;
                        std::memcpy(&value, in + c * 2, sizeof(value));
                        *out++ = static_cast<float>(value);
                        break;
                    }
                    default: {
                        float value;
                        std::memcpy(&value, in + c * 4, sizeof(value));
                        *out++ = value;
                        break;
                    }
                    }
                }
                out += gxm_attributes[i].componentCount - components;
            }
        }
        stream_data[0] = space;
        space += (converted_stride * vertex_range + 15u) & ~15u;
        stream_count = 1;
    } else {
        for (u32 stream = 0; stream < stream_count; stream++) {
            std::memcpy(space, stream_source[stream], stream_bytes[stream]);
            stream_data[stream] = space;
            space += (stream_bytes[stream] + 15u) & ~15u;
        }
    }
    const u16* draw_indices = static_cast<const u16*>(index_buffer.Data());
    if (is_indexed) {
        // Rebasados a 0: los flujos empiezan en min_index.
        auto* out = reinterpret_cast<u16*>(space);
        if (index_u16) {
            const auto* in = reinterpret_cast<const u16*>(index_data);
            for (u32 i = 0; i < num_vertices; i++) {
                out[i] = static_cast<u16>(in[i] - min_index);
            }
        } else {
            for (u32 i = 0; i < num_vertices; i++) {
                out[i] = static_cast<u16>(index_data[i] - min_index);
            }
        }
        draw_indices = out;
    }

    phase(3);
    profile_state = profile;
    const bool state_ok = SetupDrawState(surface, pipeline, vertex_program, fragment_program);
    profile_state = false;
    if (!state_ok) {
        return HwVsReject("estado");
    }
    phase(4);
    for (u32 stream = 0; stream < stream_count; stream++) {
        sceGxmSetVertexStream(context, stream, stream_data[stream]);
    }

    // Los uniforms de la PICA, tal cual: los flotantes son floats de 32 bits
    // en memoria (f24 guarda un float), y los enteros y booleanos se pasan a
    // float como espera el traductor (ver GenerateVertexShader).
    void* vertex_uniforms = nullptr;
    if (sceGxmReserveVertexDefaultUniformBuffer(context, &vertex_uniforms) == 0) {
        const auto& uniforms = pica.vs_setup.uniforms;
        if (program->uniform_f != nullptr) {
            static_assert(sizeof(uniforms.f) == 96 * 4 * sizeof(float));
            sceGxmSetUniformDataF(vertex_uniforms, program->uniform_f, 0, 96 * 4,
                                  reinterpret_cast<const float*>(uniforms.f.data()));
        }
        if (program->uniform_i != nullptr) {
            float ints[16];
            for (u32 i = 0; i < 4; i++) {
                ints[i * 4 + 0] = static_cast<float>(uniforms.i[i].x);
                ints[i * 4 + 1] = static_cast<float>(uniforms.i[i].y);
                ints[i * 4 + 2] = static_cast<float>(uniforms.i[i].z);
                ints[i * 4 + 3] = static_cast<float>(uniforms.i[i].w);
            }
            sceGxmSetUniformDataF(vertex_uniforms, program->uniform_i, 0, 16, ints);
        }
        if (program->uniform_b != nullptr) {
            float bools[16];
            for (u32 i = 0; i < 16; i++) {
                bools[i] = uniforms.b[i] ? 1.0f : 0.0f;
            }
            sceGxmSetUniformDataF(vertex_uniforms, program->uniform_b, 0, 16, bools);
        }
        if (program->uniform_default != nullptr) {
            float defaults[16 * 4]{};
            for (u32 reg = 0; reg < 16; reg++) {
                if (inputs[reg].kind != VSInputSource::Default) {
                    continue;
                }
                const auto& value = pica.input_default_attributes[input_attribute[reg]];
                defaults[reg * 4 + 0] = value.x.ToFloat32();
                defaults[reg * 4 + 1] = value.y.ToFloat32();
                defaults[reg * 4 + 2] = value.z.ToFloat32();
                defaults[reg * 4 + 3] = value.w.ToFloat32();
            }
            sceGxmSetUniformDataF(vertex_uniforms, program->uniform_default, 0, 16 * 4, defaults);
        }
    }

    const int rc = sceGxmDraw(context, primitive, SCE_GXM_INDEX_FORMAT_U16, draw_indices,
                              num_vertices);
    if (rc < 0) {
        // Estado ya puesto pero nada dibujado: la ruta de la CPU lo repite.
        static bool noted_draw = false;
        NoteOnce(noted_draw, "gxm vs", "draw err {:#x} con {} vertices", static_cast<u32>(rc),
                 num_vertices);
        return HwVsReject("draw");
    }
    surface->dirty = true;
    const u32 triangles = primitive == SCE_GXM_PRIMITIVE_TRIANGLES
                              ? num_vertices / 3
                              : (num_vertices >= 3 ? num_vertices - 2 : 0);
    gpu_triangles.fetch_add(triangles, std::memory_order_relaxed);
    gpu_batches.fetch_add(1, std::memory_order_relaxed);
    hw_vs_batches.fetch_add(1, std::memory_order_relaxed);
    phase(5);
    if (profile) {
        for (u32 i = 0; i < kBatchPhases; i++) {
            batch_phase_us[i] += phase_us[i];
        }
        batch_phase_samples++;
    }
    return true;
}

std::array<unsigned long long, RasterizerGXM::kBatchPhases> RasterizerGXM::batch_phase_us{};
u32 RasterizerGXM::batch_phase_samples = 0;
unsigned long long RasterizerGXM::state_scene_us = 0;
unsigned long long RasterizerGXM::state_texture_us = 0;

std::string RasterizerGXM::TakeRejectSummary() {
    // El mismo texto puede venir de dos ficheros con punteros distintos: se
    // juntan por contenido.
    std::vector<std::pair<std::string, u32>> totals;
    for (auto& slot : reject_counts) {
        const char* reason = slot.reason.load(std::memory_order_relaxed);
        const u32 count = slot.count.exchange(0, std::memory_order_relaxed);
        if (reason == nullptr || count == 0) {
            continue;
        }
        const auto it = std::find_if(totals.begin(), totals.end(),
                                     [reason](const auto& total) { return total.first == reason; });
        if (it != totals.end()) {
            it->second += count;
        } else {
            totals.emplace_back(reason, count);
        }
    }
    if (totals.empty()) {
        return "-";
    }
    std::sort(totals.begin(), totals.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string text;
    for (std::size_t i = 0; i < totals.size() && i < 3; i++) {
        text += fmt::format("{}{} {}", i == 0 ? "" : ", ", totals[i].first, totals[i].second);
    }
    return text;
}

std::string RasterizerGXM::TakeBatchProfile() {
    const u32 samples = batch_phase_samples;
    const auto avg = [samples](unsigned long long total) {
        return samples == 0 ? 0.0 : static_cast<double>(total) / samples;
    };
    std::string text = fmt::format(
        "us por lote: preguntas {:.1f} vs {:.1f} enlazar {:.1f} datos {:.1f} estado {:.1f} "
        "(escena {:.1f} texturas {:.1f}) uniforms {:.1f} ({} muestras)",
        avg(batch_phase_us[0]), avg(batch_phase_us[1]), avg(batch_phase_us[2]),
        avg(batch_phase_us[3]), avg(batch_phase_us[4]), avg(state_scene_us),
        avg(state_texture_us), avg(batch_phase_us[5]), samples);
    batch_phase_us.fill(0);
    batch_phase_samples = 0;
    state_scene_us = 0;
    state_texture_us = 0;
    return text;
}

} // namespace Gxm
