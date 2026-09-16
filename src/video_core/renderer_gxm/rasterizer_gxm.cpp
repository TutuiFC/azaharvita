// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/rasterizer_gxm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <fmt/format.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/shacccg.h>
#include <vita2d.h>
#include "common/logging/log.h"
#include "common/vita_diag.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/pica_types.h"
#include "video_core/renderer_gxm/gxm_texture_cache.h"
#include "video_core/shader/generator/cg_fs_shader_gen.h"

namespace Gxm {

std::atomic<u32> RasterizerGXM::gpu_triangles{0};
std::atomic<u32> RasterizerGXM::software_triangles{0};

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
    if (g_module >= 0) {
        return true;
    }
    for (const char* path : kShacccgPaths) {
        g_module = sceKernelLoadStartModule(path, 0, nullptr, 0, nullptr, nullptr);
        if (g_module >= 0) {
            break;
        }
        // El codigo importa: un fallo en plena carga del juego puede ser falta
        // de memoria, no que el fichero no exista. Sin esto, todo se resume en
        // "sin libshacccg" y se depura a ciegas.
        LOG_ERROR(Render, "GXM: cargar {} fallo con {:#x}", path, static_cast<u32>(g_module));
    }
    if (g_module < 0) {
        return false;
    }
    sceShaccCgSetDefaultAllocator(&CgAlloc, &CgFree);
    sceShaccCgInitializeCallbackList(&g_callbacks, SCE_SHACCCG_TRIVIAL);
    g_callbacks.openFile = &OpenSource;
    return true;
}

const SceShaccCgCompileOutput* CompileCg(SceShaccCgTargetProfile profile, const char* name,
                                         const char* source) {
    g_source.fileName = name;
    g_source.text = source;
    g_source.size = static_cast<SceUInt32>(std::strlen(source));

    SceShaccCgCompileOptions options{};
    sceShaccCgInitializeCompileOptions(&options);
    options.mainSourceFile = name;
    options.targetProfile = profile;
    options.entryFunctionName = "main";

    const SceShaccCgCompileOutput* output = sceShaccCgCompileProgram(&options, &g_callbacks, 0);
    if (output == nullptr || output->programData == nullptr) {
        if (output != nullptr) {
            for (int i = 0; i < output->diagnosticCount && i < 4; i++) {
                LOG_ERROR(Render, "GXM {}: {}", name,
                          output->diagnostics[i].message != nullptr
                              ? output->diagnostics[i].message
                              : "(sin mensaje)");
            }
            sceShaccCgDestroyCompileOutput(output);
        } else {
            LOG_ERROR(Render, "GXM: el compilador no devolvio nada para {}", name);
        }
        Common::VitaNote("gxm shader", name);
        return nullptr;
    }
    return output;
}

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

constexpr u32 kVertexStride = 14 * sizeof(float);
constexpr u32 kMaxVerticesPerDraw = 60000;

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
    default:
        // Los factores con color/alfa constante no existen en GXM.
        return false;
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
 * el volcado. Por eso, al crearla, se copia dentro el contenido que ya tuviera
 * el framebuffer del invitado: hay juegos que dibujan por encima de lo anterior
 * sin borrar, y arrancar en negro cambiaria la imagen.
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

    PAddr guest_address = 0;
    u32 width = 0;
    u32 height = 0;
    /// Bytes por pixel del framebuffer del invitado (2, 3 o 4).
    u32 bpp = 0;
    /// Formatos de la copia (color surface <-> invitado). Con el mismo tamano
    /// son iguales y la copia es byte a byte; en RGB8 el origen es de 32 bits y
    /// el destino de 24, que describe el mismo orden de canales.
    SceGxmTransferFormat copy_src{};
    SceGxmTransferFormat copy_dst{};
    /// Stride en bytes de la superficie de color (32 o 16 bits por pixel).
    u32 rt_stride = 0;
    bool dirty = false;
    bool scene_open = false;
    SceGxmRenderTarget* render_target = nullptr;
    SceUID driver_uid = -1;
    SceGxmColorSurface color_surface{};
    SceGxmDepthStencilSurface depth_surface{};
    Allocation color_buffer;
    Allocation depth_buffer;
    Allocation notification_word;
    SceGxmNotification notification{};

    u32 stride_bytes() const {
        return width * bpp;
    }
};

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
        SceGxmFragmentProgram* program = nullptr;
        const SceGxmProgramParameter* const_color = nullptr;
        const SceGxmProgramParameter* combiner_buffer_color = nullptr;
        const SceGxmProgramParameter* alphatest_ref = nullptr;
        const SceGxmProgramParameter* samplers[3] = {nullptr, nullptr, nullptr};
        u8 sampler_units[3] = {0, 0, 0};
        const SceShaccCgCompileOutput* output = nullptr;
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
                sceShaccCgDestroyCompileOutput(entry->output);
            }
        }
        if (vertex_program != nullptr) {
            sceGxmShaderPatcherReleaseVertexProgram(patcher, vertex_program);
        }
        if (vertex_registered) {
            sceGxmShaderPatcherUnregisterProgram(patcher, vertex_id);
        }
        if (vertex_output != nullptr) {
            sceShaccCgDestroyCompileOutput(vertex_output);
        }
    }

    bool Initialize() {
        vertex_output = CompileCg(SCE_SHACCCG_PROFILE_VP, "azahar_gxm_v.cg", kVertexSource);
        if (vertex_output == nullptr) {
            return false;
        }
        const auto* gxp = reinterpret_cast<const SceGxmProgram*>(vertex_output->programData);
        vertex_gxp = gxp;
        if (sceGxmShaderPatcherRegisterProgram(patcher, gxp, &vertex_id) != 0) {
            return false;
        }
        vertex_registered = true;

        struct AttribSpec {
            const char* name;
            u32 offset;
            u8 components;
        };
        const AttribSpec specs[] = {
            {"position", 0, 4},
            {"color", 4 * sizeof(float), 4},
            {"tc0", 8 * sizeof(float), 2},
            {"tc1", 10 * sizeof(float), 2},
            {"tc2", 12 * sizeof(float), 2},
        };
        SceGxmVertexAttribute attributes[5]{};
        for (u32 i = 0; i < 5; i++) {
            const SceGxmProgramParameter* param =
                sceGxmProgramFindParameterByName(gxp, specs[i].name);
            if (param == nullptr) {
                LOG_ERROR(Render, "GXM: el shader de vertices no declara {}", specs[i].name);
                return false;
            }
            attributes[i].streamIndex = 0;
            attributes[i].offset = static_cast<u16>(specs[i].offset);
            attributes[i].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
            attributes[i].componentCount = specs[i].components;
            attributes[i].regIndex = sceGxmProgramParameterGetResourceIndex(param);
        }
        SceGxmVertexStream stream{};
        stream.stride = static_cast<u16>(kVertexStride);
        stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
        return sceGxmShaderPatcherCreateVertexProgram(patcher, vertex_id, attributes, 5, &stream,
                                                      1, &vertex_program) == 0;
    }

    /// Devuelve el pipeline o nullptr si la configuracion no esta soportada.
    const Entry* Get(const Pica::RegsInternal& regs) {
        Pica::Shader::FSConfig config{regs};
        // El perfil no describe capacidades de GXM todavia: sin perfil, el
        // generador aplica las reglas por defecto.
        config.ApplyProfile(Pica::Shader::Profile{});

        SceGxmBlendInfo blend{};
        u32 blend_bits = 0;
        if (!BuildBlend(regs, blend, blend_bits)) {
            return nullptr;
        }
        PipelineKey key{config, blend_bits};

        const auto it = entries.find(key);
        if (it != entries.end()) {
            return it->second.get();
        }

        const auto source = Pica::Shader::Generator::GXM::GenerateFragmentShader(config);
        if (!source.has_value()) {
            return nullptr;
        }
        auto entry = std::make_unique<Entry>();
        entry->output = CompileCg(SCE_SHACCCG_PROFILE_FP, "azahar_gxm_f.cg", source->c_str());
        if (entry->output == nullptr) {
            return nullptr;
        }
        const auto* gxp = reinterpret_cast<const SceGxmProgram*>(entry->output->programData);
        if (sceGxmShaderPatcherRegisterProgram(patcher, gxp, &entry->id) != 0) {
            sceShaccCgDestroyCompileOutput(entry->output);
            return nullptr;
        }
        entry->registered = true;
        if (sceGxmShaderPatcherCreateFragmentProgram(patcher, entry->id,
                                                     SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
                                                     SCE_GXM_MULTISAMPLE_NONE, &blend, vertex_gxp,
                                                     &entry->program) != 0) {
            sceGxmShaderPatcherUnregisterProgram(patcher, entry->id);
            sceShaccCgDestroyCompileOutput(entry->output);
            return nullptr;
        }
        entry->const_color = sceGxmProgramFindParameterByName(gxp, "const_color");
        entry->combiner_buffer_color =
            sceGxmProgramFindParameterByName(gxp, "tev_combiner_buffer_color");
        entry->alphatest_ref = sceGxmProgramFindParameterByName(gxp, "alphatest_ref");
        entry->samplers[0] = sceGxmProgramFindParameterByName(gxp, "tex0");
        entry->samplers[1] = sceGxmProgramFindParameterByName(gxp, "tex1");
        entry->samplers[2] = sceGxmProgramFindParameterByName(gxp, "tex2");
        for (u32 i = 0; i < 3; i++) {
            entry->sampler_units[i] =
                entry->samplers[i] != nullptr
                    ? static_cast<u8>(sceGxmProgramParameterGetResourceIndex(entry->samplers[i]))
                    : 0;
        }

        const Entry* result = entry.get();
        entries.emplace(key, std::move(entry));
        return result;
    }

    bool BuildBlend(const Pica::RegsInternal& regs, SceGxmBlendInfo& blend, u32& bits) {
        const auto& merger = regs.framebuffer.output_merger;
        if (merger.fragment_operation_mode !=
            FramebufferRegs::FragmentOperationMode::Default) {
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
        // El logic op no existe en GXM: si el juego lo enciende, a software.
        if (merger.logic_op.Value() != FramebufferRegs::LogicOp::Copy) {
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
            if (!MapBlendEquation(merger.alpha_blending.blend_equation_rgb.Value(), &color_func) ||
                !MapBlendEquation(merger.alpha_blending.blend_equation_a.Value(), &alpha_func) ||
                !MapBlendFactor(merger.alpha_blending.factor_source_rgb.Value(), &color_src) ||
                !MapBlendFactor(merger.alpha_blending.factor_dest_rgb.Value(), &color_dst) ||
                !MapBlendFactor(merger.alpha_blending.factor_source_a.Value(), &alpha_src) ||
                !MapBlendFactor(merger.alpha_blending.factor_dest_a.Value(), &alpha_dst)) {
                return false;
            }
            blend.colorFunc = color_func;
            blend.alphaFunc = alpha_func;
            blend.colorSrc = color_src;
            blend.colorDst = color_dst;
            blend.alphaSrc = alpha_src;
            blend.alphaDst = alpha_dst;
        }
        bits = static_cast<u32>(blend.colorMask) | (static_cast<u32>(blend.colorFunc) << 8) |
               (static_cast<u32>(blend.alphaFunc) << 12) |
               (static_cast<u32>(blend.colorSrc) << 16) |
               (static_cast<u32>(blend.colorDst) << 20) |
               (static_cast<u32>(blend.alphaSrc) << 24) |
               (static_cast<u32>(blend.alphaDst) << 28);
        return true;
    }

    SceGxmContext* context;
    SceGxmShaderPatcher* patcher;
    const SceGxmProgram* vertex_gxp = nullptr;
    SceGxmVertexProgram* vertex_program = nullptr;
    SceShaccCgCompileOutput const* vertex_output = nullptr;
    SceGxmShaderPatcherId vertex_id{};
    bool vertex_registered = false;
    std::unordered_map<PipelineKey, std::unique_ptr<Entry>, PipelineKeyHash> entries;
};

RasterizerGXM::RasterizerGXM(VideoCore::RasterizerInterface& software_, Memory::MemorySystem& memory_,
                             Pica::PicaCore& pica_)
    : software{software_}, memory{memory_}, pica{pica_} {}

RasterizerGXM::~RasterizerGXM() {
    FlushPending();
    textures.reset();
    pipelines.reset();
    surfaces.clear();
    vertex_buffer = Allocation{};
    index_buffer = Allocation{};
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
    patcher = vita2d_get_shader_patcher();
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
    available = true;
    status = "gxm";
    Common::VitaNote("gxm init", "rasterizador listo");
    LOG_INFO(Render, "GXM: rasterizador de la GPU listo (shaders compilados en runtime)");
    return true;
}

void RasterizerGXM::NoteSkip(u32 index, const char* reason) {
    if (index < 5 && !skip_noted[index]) {
        skip_noted[index] = true;
        Common::VitaNote("gxm skip", reason);
    }
}

void RasterizerGXM::EndScene() {
    if (open_surface == nullptr) {
        return;
    }
    // La notificacion se escribe cuando el trabajo de fragmentos ha terminado;
    // hacerla esperar aqui es lo que garantiza que el volcado no copie una
    // imagen a medias. En la Fase 4 esto se cambia por sincronizacion sin
    // bloqueo del hilo.
    u32* word = static_cast<u32*>(open_surface->notification_word.Data());
    *word = 0;
    open_surface->notification.address = word;
    open_surface->notification.value = 1;
    sceGxmEndScene(context, nullptr, &open_surface->notification);
    sceGxmNotificationWait(&open_surface->notification);
    open_surface->scene_open = false;
    open_surface = nullptr;
}

void RasterizerGXM::FlushPending() {
    EndScene();
    for (auto& surface : surfaces) {
        if (!surface->dirty) {
            continue;
        }
        surface->dirty = false;
        // Del render target al framebuffer del invitado, con el par de formatos
        // que describe el mismo orden de canales en los dos lados.
        const int rc = sceGxmTransferCopy(
            surface->width, surface->height, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE,
            surface->copy_src, SCE_GXM_TRANSFER_TILED, surface->color_buffer.Data(), 0, 0,
            static_cast<int>(surface->rt_stride), surface->copy_dst, SCE_GXM_TRANSFER_LINEAR,
            memory.GetPhysicalPointer(surface->guest_address), 0, 0,
            static_cast<int>(surface->stride_bytes()), nullptr, 0, nullptr);
        if (rc < 0) {
            LOG_ERROR(Render, "GXM: el volcado del framebuffer fallo ({:#x})", static_cast<u32>(rc));
        }
        // El volcado es asincrono: hasta que no termina, memoria del invitado
        // podria leerse a medias.
        sceGxmTransferFinish();
    }
}

RasterizerGXM::Surface* RasterizerGXM::CurrentSurface() {
    const auto& config = pica.regs.internal.framebuffer.framebuffer;
    /**
     * Formatos de framebuffer soportados y por que el color surface no siempre
     * es "RGBA".
     *
     * El invitado guarda su RGB8 en el orden B,G,R y su RGBA8 en A,B,G,R. El
     * render target es siempre de 32 bits (no hay color surface de 24), asi que
     * lo que se hace es elegir el NOMBRE del formato GXM de forma que la salida
     * estandar del shader (r,g,b,a) caiga en los bytes que espera el invitado
     * (los nombres listan los canales del byte mas significativo al menos):
     *
     *   invitado RGBA8 [A,B,G,R] <- surface U8U8U8U8_RGBA  (memoria [A,B,G,R])
     *   invitado RGB8  [B,G,R]   <- surface U8U8U8U8_ARGB  (memoria [B,G,R,A])
     *
     * y al volcar, la transferencia copia los bytes 0..2 tal cual a un
     * destino de 24 bits (BGR), que es lo que describe el mismo orden.
     */
    const auto guest_format = config.color_format.Value();
    SceGxmColorFormat rt_color_format{};
    SceGxmTransferFormat copy_src{};
    SceGxmTransferFormat copy_dst{};
    u32 bpp = 0;
    u32 rt_bpp = 0;
    switch (guest_format) {
    case FramebufferRegs::ColorFormat::RGBA8:
        rt_color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_RGBA;
        copy_src = copy_dst = SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
        bpp = rt_bpp = 4;
        break;
    case FramebufferRegs::ColorFormat::RGB8:
        // El color surface en ARGB deja la memoria en B,G,R,A: los tres
        // primeros bytes ya son el RGB8 del invitado, y el destino de 24 bits
        // (BGR) copia esos tres bytes tal cual.
        rt_color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB;
        copy_src = SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
        copy_dst = SCE_GXM_TRANSFER_FORMAT_U8U8U8_BGR;
        bpp = 3;
        rt_bpp = 4;
        break;
    case FramebufferRegs::ColorFormat::RGB5A1:
        // Los formatos de 16 bits se eligen con el nombre que pone cada canal
        // en el bit que espera el invitado (R en los mas significativos).
        rt_color_format = SCE_GXM_COLOR_FORMAT_U5U5U5U1_RGBA;
        copy_src = copy_dst = SCE_GXM_TRANSFER_FORMAT_U1U5U5U5_ABGR;
        bpp = rt_bpp = 2;
        break;
    case FramebufferRegs::ColorFormat::RGB565:
        rt_color_format = SCE_GXM_COLOR_FORMAT_U5U6U5_RGB;
        copy_src = copy_dst = SCE_GXM_TRANSFER_FORMAT_U5U6U5_BGR;
        bpp = rt_bpp = 2;
        break;
    case FramebufferRegs::ColorFormat::RGBA4:
        rt_color_format = SCE_GXM_COLOR_FORMAT_U4U4U4U4_RGBA;
        copy_src = copy_dst = SCE_GXM_TRANSFER_FORMAT_U4U4U4U4_ABGR;
        bpp = rt_bpp = 2;
        break;
    default:
        return nullptr;
    }
    const PAddr address = config.color_buffer_address * 8;
    const u32 width = config.width;
    const u32 height = config.height;
    if (width == 0 || height == 0 || width > 1024 || height > 1024 || address == 0) {
        return nullptr;
    }
    // El framebuffer del invitado tiene que estar entero mapeado y contiguo:
    // la copia inicial de contenido y los volcados lo recorren completo.
    if (!GuestSpanMapped(memory, address, width * height * bpp)) {
        return nullptr;
    }
    const bool depth16 = config.depth_format.Value() == FramebufferRegs::DepthFormat::D16;
    const PAddr depth_address = config.depth_buffer_address * 8;
    const u32 depth_stride = depth16 ? width * 2 : width * 4;
    const bool depth_mapped =
        depth_address != 0 && GuestSpanMapped(memory, depth_address, width * height * depth_stride);
    for (auto& surface : surfaces) {
        if (surface->guest_address == address && surface->width == width &&
            surface->height == height) {
            return surface.get();
        }
    }

    // Nueva superficie: reservar color y profundidad y crear el render target.
    // Antes hay que cerrar cualquier escena abierta: crear un render target con
    // una escena en marcha no es legal en GXM.
    EndScene();
    auto surface = std::make_unique<Surface>();
    surface->guest_address = address;
    surface->width = width;
    surface->height = height;
    surface->bpp = bpp;
    surface->copy_src = copy_src;
    surface->copy_dst = copy_dst;

    const u32 aligned_width = (width + 7) & ~7u;
    const u32 aligned_height = (height + 7) & ~7u;
    surface->rt_stride = aligned_width * rt_bpp;
    surface->color_buffer = Allocate(Pool::Cdram, aligned_width * aligned_height * rt_bpp);
    surface->depth_buffer =
        Allocate(Pool::Cdram, aligned_width * aligned_height * (depth16 ? 2 : 4));
    surface->notification_word = Allocate(Pool::Host, 4);
    if (!surface->color_buffer.Valid() || !surface->depth_buffer.Valid() ||
        !surface->notification_word.Valid()) {
        LOG_ERROR(Render, "GXM: sin memoria para la superficie de dibujado");
        Common::VitaNote("gxm fb", "sin memoria");
        return nullptr;
    }
    std::memset(surface->color_buffer.Data(), 0, surface->color_buffer.Size());
    std::memset(surface->depth_buffer.Data(), 0, surface->depth_buffer.Size());

    if (sceGxmColorSurfaceInit(&surface->color_surface, rt_color_format,
                               SCE_GXM_COLOR_SURFACE_TILED, SCE_GXM_COLOR_SURFACE_SCALE_NONE,
                               SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, width, height, aligned_width,
                               surface->color_buffer.Data()) != 0) {
        return nullptr;
    }
    const SceGxmDepthStencilFormat depth_format =
        depth16 ? SCE_GXM_DEPTH_STENCIL_FORMAT_D16 : SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24;
    if (sceGxmDepthStencilSurfaceInit(&surface->depth_surface, depth_format,
                                      SCE_GXM_DEPTH_STENCIL_SURFACE_TILED, aligned_width,
                                      surface->depth_buffer.Data(), nullptr) != 0) {
        return nullptr;
    }

    SceGxmRenderTargetParams params{};
    params.flags = 0;
    params.width = static_cast<u16>(width);
    params.height = static_cast<u16>(height);
    params.scenesPerFrame = 16;
    params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    params.multisampleLocations = 0;
    // La memoria del driver la reservamos nosotros: el SDK de Vita no define
    // SCE_UID_INVALID_UID, asi que depender de un centinela era una suposicion,
    // y con una UID que GXM no reconozca el fallo es una caida.
    unsigned int driver_size = 0;
    if (sceGxmGetRenderTargetMemSize(&params, &driver_size) < 0 || driver_size == 0) {
        return nullptr;
    }
    const SceUID driver_uid = sceKernelAllocMemBlock(
        "azahar_gxm_rt", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, driver_size, nullptr);
    if (driver_uid < 0) {
        return nullptr;
    }
    void* driver_mem = nullptr;
    if (sceKernelGetMemBlockBase(driver_uid, &driver_mem) < 0) {
        sceKernelFreeMemBlock(driver_uid);
        return nullptr;
    }
    params.driverMemBlock = driver_uid;
    if (sceGxmCreateRenderTarget(&params, &surface->render_target) != 0) {
        sceKernelFreeMemBlock(driver_uid);
        LOG_ERROR(Render, "GXM: no se pudo crear el render target");
        return nullptr;
    }
    surface->driver_uid = driver_uid;

    // Copiar dentro lo que ya hubiera en el framebuffer del invitado, en los
    // dos sentidos de la memoria (color y, si el juego lo usa, profundidad).
    const u8* guest = memory.GetPhysicalPointer(address);
    if (guest != nullptr) {
        sceGxmTransferCopy(width, height, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE, copy_src,
                           SCE_GXM_TRANSFER_LINEAR, guest, 0, 0, static_cast<int>(width * bpp),
                           copy_dst, SCE_GXM_TRANSFER_TILED, surface->color_buffer.Data(), 0, 0,
                           static_cast<int>(surface->rt_stride), nullptr, 0, nullptr);
    }
    if (depth_mapped) {
        const u8* guest_depth = memory.GetPhysicalPointer(depth_address);
        const SceGxmTransferFormat depth_format =
            depth16 ? SCE_GXM_TRANSFER_FORMAT_RAW16 : SCE_GXM_TRANSFER_FORMAT_RAW32;
        const PAddr depth_span = depth16 ? width * height * 2 : width * height * 4;
        sceGxmTransferCopy(width, height, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE, depth_format,
                           SCE_GXM_TRANSFER_LINEAR, guest_depth, 0, 0,
                           static_cast<int>(depth_stride), depth_format,
                           SCE_GXM_TRANSFER_TILED, surface->depth_buffer.Data(), 0, 0,
                           static_cast<int>(depth_stride), nullptr, 0, nullptr);
        (void)depth_span;
    }
    sceGxmTransferFinish();

    Surface* result = surface.get();
    surfaces.emplace_back(std::move(surface));
    {
        char note[64];
        const auto written =
            fmt::format_to_n(note, sizeof(note) - 1, "addr {:#x} {}x{} creada", address, width,
                             height);
        *written.out = 0;
        Common::VitaNote("gxm fb", note);
    }
    return result;
}

void RasterizerGXM::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                const Pica::OutputVertex& v2) {
    if (!batch_decided) {
        batch_decided = true;
        batch_on_gpu = false;
        if (EnsureInitialized()) {
            // Lo que aun no reproduce el contexto GXM (scissor, W-buffering) o
            // el generador (iluminacion, niebla, proctex...) manda el lote al
            // camino de software. Las texturas ya no: las sirve el cache.
            Pica::Shader::FSConfig config{pica.regs.internal};
            config.ApplyProfile(Pica::Shader::Profile{});
            const auto source = Pica::Shader::Generator::GXM::GenerateFragmentShader(config);
            const bool scissor = pica.regs.internal.rasterizer.scissor_test.mode !=
                                 RasterizerRegs::ScissorMode::Disabled;
            const bool wbuffering = pica.regs.internal.rasterizer.depthmap_enable ==
                                    RasterizerRegs::DepthBuffering::WBuffering;
            batch_on_gpu = source.has_value() && !scissor && !wbuffering;
            if (!batch_on_gpu) {
                NoteSkip(scissor ? 0 : (wbuffering ? 1 : 2),
                         scissor ? "scissor" : (wbuffering ? "wbuffer" : "shader"));
            }
        }
        if (batch_on_gpu && CurrentSurface() == nullptr) {
            batch_on_gpu = false;
            NoteSkip(3, "framebuffer");
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
    const u32 vertex_count = static_cast<u32>(batch.size());
    if (vertex_count == 0) {
        return true;
    }
    const u32 needed = vertex_count * kVertexStride;
    if (needed > vertex_buffer.Size()) {
        Allocation buffer = Allocate(Pool::Host, needed);
        if (!buffer.Valid()) {
            return false;
        }
        vertex_buffer = std::move(buffer);
    }
    if (vertex_count > index_capacity) {
        Allocation buffer = Allocate(Pool::Host, kMaxVerticesPerDraw * sizeof(u16));
        if (!buffer.Valid()) {
            return false;
        }
        index_buffer = std::move(buffer);
        index_capacity = kMaxVerticesPerDraw;
        auto* indices = static_cast<u16*>(index_buffer.Data());
        for (u32 i = 0; i < index_capacity; i++) {
            indices[i] = static_cast<u16>(i);
        }
    }

    auto* vertices = static_cast<float*>(vertex_buffer.Data());
    for (u32 i = 0; i < vertex_count; i++) {
        const Pica::OutputVertex& v = batch[i];
        float* out = vertices + i * 14;
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
    }

    const PipelineCache::Entry* pipeline = pipelines->Get(pica.regs.internal);
    if (pipeline == nullptr) {
        return false;
    }

    if (!surface->scene_open) {
        if (sceGxmBeginScene(context, 0, surface->render_target, nullptr, nullptr, nullptr,
                             &surface->color_surface, &surface->depth_surface) != 0) {
            return false;
        }
        surface->scene_open = true;
        open_surface = surface;
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
    sceGxmSetViewport(context, halfsize_x + corner_x, halfsize_x, halfsize_y + corner_y,
                      -halfsize_y, depth_offset, depth_scale);
    sceGxmSetFrontPolygonMode(context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);

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

    // Ojo con el culling: la PICA decide en su espacio de pantalla (y hacia
    // abajo) y GXM en coordenadas de recorte (y hacia arriba), asi que los
    // sentidos van intercambiados.
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

    sceGxmSetVertexProgram(context, pipelines->vertex_program);
    sceGxmSetFragmentProgram(context, pipeline->program);
    sceGxmSetVertexStream(context, 0, vertex_buffer.Data());

    // Texturas: por cada sampler que el shader use de verdad, su unidad. Si
    // alguna no se puede servir (formato, borde, unidad apagada...), el lote
    // entero vuelve a software desde DrawTriangles.
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

    // Uniforms. Los que el compilador haya eliminado no existen como parametro
    // y se saltan; los samplers quedan para cuando exista el cache de texturas.
    void* uniform_buffer = nullptr;
    if (sceGxmReserveFragmentDefaultUniformBuffer(context, &uniform_buffer) == 0) {
        if (pipeline->const_color != nullptr) {
            f32 colors[6 * 4];
            const auto tev_stages = pica.regs.internal.texturing.GetTevStages();
            for (u32 i = 0; i < 6; i++) {
                const u32 raw = tev_stages[i].const_color;
                colors[i * 4 + 0] = static_cast<f32>((raw >> 0) & 0xFF) / 255.0f;
                colors[i * 4 + 1] = static_cast<f32>((raw >> 8) & 0xFF) / 255.0f;
                colors[i * 4 + 2] = static_cast<f32>((raw >> 16) & 0xFF) / 255.0f;
                colors[i * 4 + 3] = static_cast<f32>((raw >> 24) & 0xFF) / 255.0f;
            }
            sceGxmSetUniformDataF(uniform_buffer, pipeline->const_color, 0, 24, colors);
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
    }

    const auto* indices = static_cast<const u16*>(index_buffer.Data());
    u32 drawn = 0;
    while (drawn < vertex_count) {
        const u32 chunk = std::min(vertex_count - drawn, kMaxVerticesPerDraw);
        const int rc = sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                                  indices, chunk);
        if (rc < 0) {
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
    return true;
}

void RasterizerGXM::DrawTriangles() {
    if (!batch_on_gpu) {
        batch_decided = false;
        batch.clear();
        return;
    }
    const bool drawn = DrawBatchOnGpu();
    if (!drawn) {
        // No se ha podido dibujar en la GPU (sin memoria, shader rechazado a
        // ultima hora): el lote entero vuelve al camino de software, que es la
        // referencia. La escena se cierra antes para que el software lea la
        // memoria ya volcada.
        FlushPending();
        for (std::size_t i = 0; i + 2 < batch.size(); i += 3) {
            software.AddTriangle(batch[i], batch[i + 1], batch[i + 2]);
            software_triangles.fetch_add(1, std::memory_order_relaxed);
        }
    }
    batch.clear();
    batch_decided = false;
    batch_on_gpu = false;
}

void RasterizerGXM::FlushAll() {
    FlushPending();
    software.FlushAll();
}

void RasterizerGXM::FlushRegion(PAddr addr, u32 size) {
    FlushPending();
    // El cache de texturas se crea en el primer lote, y estos vaciados llegan
    // desde el arranque del juego, mucho antes: sin la comprobacion esto era
    // una desreferencia de puntero nulo (la caida de 0.1.0.2 a 0.1.0.4).
    if (textures != nullptr) {
        textures->InvalidateRange(addr, size);
    }
    software.FlushRegion(addr, size);
}

void RasterizerGXM::InvalidateRegion(PAddr addr, u32 size) {
    FlushPending();
    // El cache de texturas se crea en el primer lote, y estos vaciados llegan
    // desde el arranque del juego, mucho antes: sin la comprobacion esto era
    // una desreferencia de puntero nulo (la caida de 0.1.0.2 a 0.1.0.4).
    if (textures != nullptr) {
        textures->InvalidateRange(addr, size);
    }
    software.InvalidateRegion(addr, size);
}

void RasterizerGXM::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    FlushPending();
    // El cache de texturas se crea en el primer lote, y estos vaciados llegan
    // desde el arranque del juego, mucho antes: sin la comprobacion esto era
    // una desreferencia de puntero nulo (la caida de 0.1.0.2 a 0.1.0.4).
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

} // namespace Gxm
