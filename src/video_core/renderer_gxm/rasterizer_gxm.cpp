// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/rasterizer_gxm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>
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
#include "video_core/utils.h"
#include "video_core/renderer_gxm/gxm_cg.h"
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

constexpr u32 kVertexStride = 14 * sizeof(float);
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
        // color de mezcla constante, asi que esos lotes se van a software.
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
    /// Hay dibujado nuestro que el invitado todavia no ha visto.
    bool dirty = false;
    /// El invitado ha cambiado este framebuffer por otro camino (un relleno de
    /// color, una transferencia): hay que volver a leerlo antes de dibujar.
    bool needs_reload = false;
    bool scene_open = false;
    SceGxmRenderTarget* render_target = nullptr;
    SceUID driver_uid = -1;
    SceGxmColorSurface color_surface{};
    SceGxmDepthStencilSurface depth_surface{};
    Allocation color_buffer;
    Allocation depth_buffer;
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
template <u32 kGuestBpp, bool kToGuest>
void CopyTiledGuestImpl(u8* guest, u8* linear, u32 width, u32 height, u32 linear_stride) {
    constexpr u32 kSurfaceBpp = kGuestBpp == 3 ? 4 : kGuestBpp;
    for (u32 y = 0; y < height; y++) {
        u8* const row = linear + static_cast<std::size_t>(y) * linear_stride;
        u8* const tile_row = guest + static_cast<std::size_t>(y & ~7u) * width * kGuestBpp;
        for (u32 x = 0; x < width; x++) {
            u8* const guest_pixel = tile_row + VideoCore::GetMortonOffset(x, y, kGuestBpp);
            u8* const linear_pixel = row + x * kSurfaceBpp;
            if constexpr (kToGuest) {
                std::memcpy(guest_pixel, linear_pixel, kGuestBpp);
            } else {
                std::memcpy(linear_pixel, guest_pixel, kGuestBpp);
                if constexpr (kGuestBpp == 3) {
                    // El invitado no guarda alfa en RGB8 y la superficie si
                    // tiene el byte: dejarlo a cero haria transparente lo que
                    // la mezcla lea de ahi.
                    linear_pixel[3] = 0xFF;
                }
            }
        }
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
        SceGxmFragmentProgram* program = nullptr;
        const SceGxmProgramParameter* const_color = nullptr;
        const SceGxmProgramParameter* combiner_buffer_color = nullptr;
        const SceGxmProgramParameter* alphatest_ref = nullptr;
        const SceGxmProgramParameter* samplers[3] = {nullptr, nullptr, nullptr};
        u8 sampler_units[3] = {0, 0, 0};
        const SceGxmProgramParameter* fog_lut = nullptr;
        const SceGxmProgramParameter* fog_color = nullptr;
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
            // No deberia pasar: AddTriangle ya genero esta misma configuracion
            // antes de aceptar el lote. Si pasa, es que las dos no miran lo
            // mismo, y eso conviene verlo escrito.
            NoteOnce(noted[3], "gxm pipeline", "el generador se echo atras");
            return nullptr;
        }
        auto entry = std::make_unique<Entry>();
        entry->output = CompileCg(SCE_SHACCCG_PROFILE_FP, "azahar_gxm_f.cg", source->c_str());
        if (entry->output == nullptr) {
            // CompileCg ya deja su nota ("gxm shader") con el fichero; el
            // mensaje del compilador va al registro normal.
            return nullptr;
        }
        const auto* gxp = reinterpret_cast<const SceGxmProgram*>(entry->output->programData);
        const int register_rc = sceGxmShaderPatcherRegisterProgram(patcher, gxp, &entry->id);
        if (register_rc != 0) {
            NoteOnce(noted[4], "gxm pipeline", "registrar err {:#x}",
                     static_cast<u32>(register_rc));
            sceShaccCgDestroyCompileOutput(entry->output);
            return nullptr;
        }
        entry->registered = true;
        const int program_rc = sceGxmShaderPatcherCreateFragmentProgram(
            patcher, entry->id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4, SCE_GXM_MULTISAMPLE_NONE,
            &blend, vertex_gxp, &entry->program);
        if (program_rc != 0) {
            NoteOnce(noted[5], "gxm pipeline", "programa de fragmentos err {:#x}",
                     static_cast<u32>(program_rc));
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
        entry->fog_lut = sceGxmProgramFindParameterByName(gxp, "fog_lut");
        entry->fog_color = sceGxmProgramFindParameterByName(gxp, "fog_color");
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
            if (!MapBlendEquation(merger.alpha_blending.blend_equation_rgb.Value(), &color_func) ||
                !MapBlendEquation(merger.alpha_blending.blend_equation_a.Value(), &alpha_func) ||
                !MapBlendFactor(merger.alpha_blending.factor_source_rgb.Value(), &color_src) ||
                !MapBlendFactor(merger.alpha_blending.factor_dest_rgb.Value(), &color_dst) ||
                !MapBlendFactor(merger.alpha_blending.factor_source_a.Value(), &alpha_src) ||
                !MapBlendFactor(merger.alpha_blending.factor_dest_a.Value(), &alpha_dst)) {
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
        bits = static_cast<u32>(blend.colorMask) | (static_cast<u32>(blend.colorFunc) << 8) |
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
    bool noted[6] = {};
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
    if (index < 6 && !skip_noted[index]) {
        skip_noted[index] = true;
        Common::VitaNote("gxm skip", reason);
    }
}

void RasterizerGXM::EndScene() {
    // Sin escena abierta la GPU ya no lee el buffer de vertices: el contador
    // vuelve a cero aunque no haya nada que cerrar (ver DrawBatchOnGpu).
    vertex_used = 0;
    if (open_surface == nullptr) {
        return;
    }
    /**
     * sceGxmFinish en lugar de la notificacion que habia aqui.
     *
     * Lo siguiente que pasa despues de cerrar una escena es que la CPU LEE la
     * superficie de color para volcarla al invitado, asi que hay que esperar a
     * la GPU de todas formas. La notificacion, ademas, exige memoria que la GPU
     * pueda ESCRIBIR, y la nuestra estaba mapeada solo de lectura: con eso
     * sceGxmNotificationWait no vuelve nunca y la consola se queda colgada sin
     * dejar ni un volcado. En la Fase 4 esto se cambia por sincronizacion que
     * no bloquee el hilo.
     */
    sceGxmEndScene(context, nullptr, nullptr);
    sceGxmFinish(context);
    open_surface->scene_open = false;
    open_surface = nullptr;
}

void RasterizerGXM::WriteBack(Surface& surface) {
    if (!surface.dirty) {
        return;
    }
    surface.dirty = false;
    u8* guest = memory.GetPhysicalPointer(surface.guest_address);
    if (guest == nullptr) {
        return;
    }
    CopyTiledGuest(guest, static_cast<u8*>(surface.color_buffer.Data()), surface.width,
                   surface.height, surface.color_stride * surface.rt_bpp, surface.bpp, true);
}

void RasterizerGXM::Reload(Surface& surface) {
    surface.needs_reload = false;
    u8* guest = memory.GetPhysicalPointer(surface.guest_address);
    if (guest == nullptr) {
        return;
    }
    CopyTiledGuest(guest, static_cast<u8*>(surface.color_buffer.Data()), surface.width,
                   surface.height, surface.color_stride * surface.rt_bpp, surface.bpp, false);
}

void RasterizerGXM::FlushPending() {
    EndScene();
    for (auto& surface : surfaces) {
        WriteBack(*surface);
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

    for (auto& surface : surfaces) {
        if (surface->guest_address == address && surface->width == width &&
            surface->height == height && surface->bpp == bpp) {
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
        surfaces.clear();
        NoteOnce(fb_noted[7], "gxm fb", "tope de {} superficies", kMaxSurfaces);
    }

    // Nueva superficie. Antes hay que cerrar cualquier escena abierta: crear un
    // render target con una escena en marcha no es legal en GXM.
    EndScene();
    const bool depth16 = config.depth_format.Value() == FramebufferRegs::DepthFormat::D16;
    auto surface = std::make_unique<Surface>();
    surface->guest_address = address;
    surface->width = width;
    surface->height = height;
    surface->bpp = bpp;
    surface->rt_bpp = rt_bpp;
    surface->guest_stride = guest_stride;

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
    const u32 color_stride = (width + 7) & ~7u;
    const u32 depth_stride = (width + SCE_GXM_TILE_SIZEX - 1) & ~(SCE_GXM_TILE_SIZEX - 1);
    const u32 tiled_height = (height + SCE_GXM_TILE_SIZEY - 1) & ~(SCE_GXM_TILE_SIZEY - 1);
    surface->color_stride = color_stride;

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
    // 0xFF en toda la profundidad es 1.0 (lo mas lejos) en los dos formatos y
    // con cualquier orden de tiles: es el "limpio" del que parte la primera
    // escena.
    std::memset(surface->depth_buffer.Data(), 0xFF, surface->depth_buffer.Size());

    const int color_rc = sceGxmColorSurfaceInit(
        &surface->color_surface, rt_color_format, SCE_GXM_COLOR_SURFACE_LINEAR,
        SCE_GXM_COLOR_SURFACE_SCALE_NONE, SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, width, height,
        color_stride, surface->color_buffer.Data());
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
    params.width = static_cast<u16>(width);
    params.height = static_cast<u16>(height);
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

void RasterizerGXM::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                const Pica::OutputVertex& v2) {
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
            // Lo que aun no reproduce el contexto GXM (scissor, W-buffering) o
            // el generador (iluminacion, niebla, proctex...) manda el lote al
            // camino de software. Las texturas ya no: las sirve el cache.
            Pica::Shader::FSConfig config{pica.regs.internal};
            config.ApplyProfile(Pica::Shader::Profile{});
            const char* reason = nullptr;
            const auto source =
                Pica::Shader::Generator::GXM::GenerateFragmentShader(config, &reason);
            const bool scissor = pica.regs.internal.rasterizer.scissor_test.mode !=
                                 RasterizerRegs::ScissorMode::Disabled;
            const bool wbuffering = pica.regs.internal.rasterizer.depthmap_enable ==
                                    RasterizerRegs::DepthBuffering::WBuffering;
            // La prueba de plantilla no esta mapeada al estado de GXM: si el
            // juego la enciende y el lote se fuera a la GPU, pintaria donde el
            // rasterizador de software no pinta. Solo cuenta si el formato de
            // profundidad tiene plantilla, que es lo que mira el de software.
            const bool stencil =
                pica.regs.internal.framebuffer.output_merger.stencil_test.enable != 0 &&
                pica.regs.internal.framebuffer.HasStencil();
            batch_on_gpu = source.has_value() && !scissor && !wbuffering && !stencil;
            if (!batch_on_gpu) {
                // El motivo concreto (iluminacion, proctex, ...) va a crash.txt:
                // sin el, "skip: shader" no distingue entre causas.
                const char* why = scissor      ? "scissor"
                                  : wbuffering ? "wbuffer"
                                  : stencil    ? "plantilla"
                                  : reason != nullptr ? reason
                                                      : "shader";
                const u32 slot = scissor ? 0 : wbuffering ? 1 : stencil ? 5 : 2;
                NoteSkip(slot, why);
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
    /**
     * El buffer de vertices se REPARTE dentro de la escena, no se reescribe.
     *
     * GXM no dibuja cuando se lo pides: apunta el dibujado y lee los vertices
     * mas tarde, al procesar la escena. Escribir el lote siguiente encima del
     * anterior -- que es lo que hacia antes, siempre desde el principio del
     * buffer -- le cambia los vertices a un dibujado que todavia no ha ocurrido.
     * Por eso cada lote se queda su tramo y el contador solo vuelve a cero
     * cuando la escena se cierra (EndScene), que es cuando la GPU ya ha
     * terminado de leer.
     */
    const u32 needed = vertex_count * kVertexStride;
    if (vertex_used + needed > vertex_buffer.Size()) {
        // Lleno: se cierra la escena (que espera a la GPU) y se empieza de cero.
        // Cambiar de bloque con la escena abierta seria soltarle la memoria
        // debajo.
        EndScene();
        if (needed > vertex_buffer.Size()) {
            Allocation buffer = Allocate(Pool::Host, needed);
            if (!buffer.Valid()) {
                NoteOnce(fb_noted[10], "gxm draw", "sin memoria para {} bytes de vertices",
                         needed);
                return false;
            }
            vertex_buffer = std::move(buffer);
        }
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

    u8* const vertex_slice = static_cast<u8*>(vertex_buffer.Data()) + vertex_used;
    auto* vertices = reinterpret_cast<float*>(vertex_slice);

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
    sceGxmSetVertexStream(context, 0, vertex_slice);

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
        if (pipeline->fog_lut != nullptr) {
            // 128 entradas de dos floats: valor y pendiente (misma LUT que usa
            // el rasterizador de software, leida como 16 bits sin signo).
            f32 lut[256];
            for (u32 i = 0; i < 128; i++) {
                const u32 raw = pica.fog.lut[i].raw;
                lut[i * 2] = static_cast<f32>(raw & 0xFFFF) / 65535.0f;
                lut[i * 2 + 1] = static_cast<f32>(raw >> 16) / 65535.0f;
            }
            sceGxmSetUniformDataF(uniform_buffer, pipeline->fog_lut, 0, 256, lut);
        }
        if (pipeline->fog_color != nullptr) {
            const auto& fog = pica.regs.internal.texturing.fog_color;
            const f32 color[4] = {static_cast<f32>(fog.r) / 255.0f,
                                  static_cast<f32>(fog.g) / 255.0f,
                                  static_cast<f32>(fog.b) / 255.0f, 1.0f};
            sceGxmSetUniformDataF(uniform_buffer, pipeline->fog_color, 0, 3, color);
        }
    }

    const auto* indices = static_cast<const u16*>(index_buffer.Data());
    u32 drawn = 0;
    while (drawn < vertex_count) {
        const u32 chunk = std::min(vertex_count - drawn, kMaxVerticesPerDraw);
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
    vertex_used += needed;
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
        if (!surface->Overlaps(addr, size)) {
            continue;
        }
        // Si la escena abierta esta dibujando justo ahi, hay que pararla: lo
        // que contenga esa superficie lo decide ahora el invitado.
        EndScene();
        surface->dirty = false;
        surface->needs_reload = true;
    }
    if (textures != nullptr) {
        textures->InvalidateRange(addr, size);
    }
    software.InvalidateRegion(addr, size);
}

void RasterizerGXM::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    // Las dos cosas y en este orden: el invitado se lleva lo que hemos dibujado
    // y despues lo que el haga ahi es lo que manda.
    for (auto& surface : surfaces) {
        if (!surface->Overlaps(addr, size)) {
            continue;
        }
        EndScene();
        WriteBack(*surface);
        surface->needs_reload = true;
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

} // namespace Gxm
