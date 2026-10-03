// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/gxm_presenter.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <arm_neon.h>
#include <psp2/kernel/modulemgr.h>
#include "common/logging/log.h"
#include "common/vita_diag.h"
#include "video_core/renderer_gxm/gxm_cg.h"
#include "video_core/renderer_gxm/rasterizer_gxm.h"

namespace Gxm {

/**
 * Shader de vertices: solo coloca y pasa la coordenada de textura.
 *
 * Las posiciones llegan ya en coordenadas de recorte (NDC) calculadas en la
 * CPU, asi que no hace falta matriz por uniform: son cuatro vertices por
 * pantalla, y la transformada de pixel a NDC son dos restas y dos divisiones.
 * Evitar el uniform evita tambien reservarlo y rellenarlo por fotograma.
 *
 * El nombre de los parametros importa: Init busca "position" y "texcoord" en
 * el GXP compilado para ligar los atributos del stream.
 */
const char kBlitVertexSource[] = R"(
void main(float4 position : POSITION,
          float2 texcoord : TEXCOORD0,
          out float4 gl_Position : POSITION,
          out float2 gl_TexCoord : TEXCOORD0)
{
    gl_Position = position;
    gl_TexCoord = texcoord;
}
)";

/**
 * Shader de fragmentos: muestra la textura tal cual.
 *
 * La textura que se enlaza puede ser la pantalla del juego o el texel de 1x1
 * que se usa para el relleno de color, asi que este unico programa cubre los
 * dos casos. El canal alfa se respeta tal cual viene del framebuffer del 3DS,
 * que es lo que hace el rasterizador de software.
 */
const char kBlitFragmentSource[] = R"(
uniform sampler2D tex;
void main(float2 texcoord : TEXCOORD0,
          out float4 gl_FragColor : COLOR)
{
    gl_FragColor = tex2D(tex, texcoord);
}
)";

namespace {

/**
 * Rutas donde puede estar el compilador de shaders.
 *
 * ur0:/data es la ruta del procedimiento estandar (la guia que usa todo el
 * homebrew de shaders deja el fichero ahi); ux0:/data se prueba despues porque
 * hay quien lo copia a la tarjeta, y no cuesta nada.
 */
constexpr const char* kShacccgPaths[] = {
    "ur0:/data/libshacccg.suprx",
    "ux0:/data/libshacccg.suprx",
};

/**
 * Fuente que se le da al compilador cuando pide abrir un fichero.
 *
 * SceShaccCg compila desde "ficheros" y aqui el shader es una cadena de texto
 * en memoria, asi que se pone una fuente estatica y se devuelve siempre. Solo
 * hay una compilacion en vuelo a la vez (Init compila los dos shaders en
 * secuencia), asi que no hace falta nada mas.
 */
SceShaccCgSourceFile g_source{};

SceShaccCgSourceFile* OpenSource(const char* file_name, const SceShaccCgSourceLocation* included_from,
                                 const SceShaccCgCompileOptions* compile_options,
                                 const char** error_string) {
    (void)file_name;
    (void)included_from;
    (void)compile_options;
    (void)error_string;
    return &g_source;
}

/// El compilador reserva memoria con estos envoltorios. Se le da la de la
/// libc para que la contabilidad sea la del proceso.
void* ShaccCgAlloc(unsigned int size) {
    return std::malloc(size);
}

void ShaccCgFree(void* pointer) {
    std::free(pointer);
}

/// Formato de vertice del quad: posicion float4 + coordenada float2. Se
/// declaran cuatro componentes aunque z y w sean constantes para que el
/// atributo coincida exactamente con el parametro float4 del shader.
constexpr u32 kVertexStride = 6 * sizeof(float);
constexpr u32 kVerticesPerQuad = 4;
constexpr u32 kQuadCount = 2;
constexpr u32 kVertexBytes = kVerticesPerQuad * kQuadCount * kVertexStride;
constexpr u32 kIndicesPerQuad = 4;
constexpr u32 kIndexBytes = kIndicesPerQuad * kQuadCount * sizeof(u16);

} // Anonymous namespace

ScreenPresenter::ScreenPresenter(float screen_width_, float screen_height_,
                                 SceGxmContext* gxm_context, SceGxmShaderPatcher* shader_patcher)
    : screen_width{screen_width_}, screen_height{screen_height_}, context{gxm_context},
      patcher{shader_patcher} {}

ScreenPresenter::~ScreenPresenter() {
    Release();
}

bool ScreenPresenter::Init() {
    const auto fail = [this](const char* why, int rc) {
        if (rc != 0) {
            LOG_ERROR(Render, "GXM: {} (codigo {:#x})", why, static_cast<u32>(rc));
        } else {
            LOG_WARNING(Render, "GXM: {}", why);
        }
        // Suele llamarse con un status mas concreto ya puesto; si no, este.
        if (std::strcmp(status, "sin inicializar") == 0) {
            status = "vita2d (error gxm)";
        }
        // Deja todo lo reservado en el camino y descarga el compilador. El
        // presentador se queda vivo pero apagado: el frontend seguir por
        // vita2d y no se vuelve a intentar.
        Release();
        return false;
    };

    // 1. El compilador, COMPARTIDO con el rasterizador (ver gxm_cg.h): cargarlo
    //    por separado hacia que el segundo intento fallara y que el
    //    rasterizador lo reintentara en cada lote, con un coste enorme.
    if (!EnsureCgReady()) {
        status = "vita2d (sin libshacccg)";
        LOG_WARNING(Render, "GXM: no hay libshacccg.suprx; se presenta con vita2d");
        return false;
    }

    // 2. El contexto. Es el de vita2d; GXM solo admite uno por proceso y este
    //    es el que abre y cierra la escena.
    if (context == nullptr || patcher == nullptr) {
        status = "vita2d (sin contexto gxm)";
        return fail("vita2d no ha dado contexto o patcher de shaders", 0);
    }

    // 3. Los dos programas.
    vertex_output = CompileShader(SCE_SHACCCG_PROFILE_VP, "azahar_blit_v.cg", kBlitVertexSource);
    fragment_output = CompileShader(SCE_SHACCCG_PROFILE_FP, "azahar_blit_f.cg", kBlitFragmentSource);
    if (vertex_output == nullptr || fragment_output == nullptr) {
        status = "vita2d (error de shader)";
        return fail("no se han podido compilar los shaders de presentacion", 0);
    }

    const auto* vertex_gxp = reinterpret_cast<const SceGxmProgram*>(vertex_output->programData);
    const auto* fragment_gxp = reinterpret_cast<const SceGxmProgram*>(fragment_output->programData);

    int rc = sceGxmShaderPatcherRegisterProgram(patcher, vertex_gxp, &vertex_program_id);
    if (rc != 0) {
        status = "vita2d (error de shader)";
        return fail("registro del programa de vertices", rc);
    }
    vertex_program_registered = true;
    rc = sceGxmShaderPatcherRegisterProgram(patcher, fragment_gxp, &fragment_program_id);
    if (rc != 0) {
        status = "vita2d (error de shader)";
        return fail("registro del programa de fragmentos", rc);
    }
    fragment_program_registered = true;

    // Los atributos se ligan por el nombre que tengan los parametros en el
    // shader; si un dia se renombran alli, esto lo dice en vez de dibujar mal.
    const SceGxmProgramParameter* position_param =
        sceGxmProgramFindParameterByName(vertex_gxp, "position");
    const SceGxmProgramParameter* texcoord_param =
        sceGxmProgramFindParameterByName(vertex_gxp, "texcoord");
    if (position_param == nullptr || texcoord_param == nullptr) {
        status = "vita2d (error de shader)";
        return fail("el shader de vertices no declara position/texcoord", 0);
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
    stream.stride = static_cast<u16>(kVertexStride);
    stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

    rc = sceGxmShaderPatcherCreateVertexProgram(patcher, vertex_program_id, attributes, 2, &stream,
                                                1, &vertex_program);
    if (rc != 0) {
        status = "vita2d (error de shader)";
        return fail("creacion del programa de vertices", rc);
    }

    // Sin mezcla: se copia el fotograma tal cual. El alfa del juego ya viene
    // resuelto por el rasterizador de software en los pixeles del framebuffer.
    SceGxmBlendInfo blend{};
    blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
    blend.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
    blend.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
    blend.colorSrc = SCE_GXM_BLEND_FACTOR_ONE;
    blend.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
    blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
    blend.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;

    rc = sceGxmShaderPatcherCreateFragmentProgram(patcher, fragment_program_id,
                                                  SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
                                                  SCE_GXM_MULTISAMPLE_NONE, &blend, vertex_gxp,
                                                  &fragment_program);
    if (rc != 0) {
        status = "vita2d (error de shader)";
        return fail("creacion del programa de fragmentos", rc);
    }

    const SceGxmProgramParameter* sampler_param =
        sceGxmProgramFindParameterByName(fragment_gxp, "tex");
    texture_unit =
        sampler_param != nullptr ? sceGxmProgramParameterGetResourceIndex(sampler_param) : 0;

    // 4. La memoria que lee la GPU.
    vertices = Allocate(Pool::Host, kVertexBytes);
    indices = Allocate(Pool::Host, kIndexBytes);
    fill_texel = Allocate(Pool::Host, 4);
    if (!vertices.Valid() || !indices.Valid() || !fill_texel.Valid()) {
        status = "vita2d (sin memoria gxm)";
        return fail("no hay memoria para vertices/indices/textura de relleno", 0);
    }

    // Los indices no cambian nunca: dos quads consecutivos en el mismo buffer
    // de vertices. El segundo dibujado empieza en el indice 4.
    static constexpr u16 kIndexData[kIndicesPerQuad * kQuadCount] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::memcpy(indices.Data(), kIndexData, sizeof(kIndexData));

    // El texel de 1x1 que sirve para el relleno de color. Se le pone el color
    // del juego en cada fotograma que lo pida.
    std::memset(fill_texel.Data(), 0, 4);
    rc = sceGxmTextureInitLinear(&fill_texture, fill_texel.Data(),
                                 SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA, 1, 1, 1);
    if (rc < 0) {
        status = "vita2d (error de textura)";
        return fail("no se pudo crear la textura de relleno", rc);
    }
    sceGxmTextureSetMinFilter(&fill_texture, SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMagFilter(&fill_texture, SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetUAddrMode(&fill_texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
    sceGxmTextureSetVAddrMode(&fill_texture, SCE_GXM_TEXTURE_ADDR_CLAMP);

    ready = true;
    status = "gxm";
    LOG_INFO(Render, "GXM: presentacion por sceGxm lista (shaders compilados en runtime)");
    return true;
}

void ScreenPresenter::Release() {
    // Primero los programas y su registro; despues las copias compiladas, que
    // son la memoria que el patcher tiene en las manos.
    if (patcher != nullptr) {
        if (fragment_program != nullptr) {
            sceGxmShaderPatcherReleaseFragmentProgram(patcher, fragment_program);
            fragment_program = nullptr;
        }
        if (vertex_program != nullptr) {
            sceGxmShaderPatcherReleaseVertexProgram(patcher, vertex_program);
            vertex_program = nullptr;
        }
        if (vertex_program_registered) {
            sceGxmShaderPatcherUnregisterProgram(patcher, vertex_program_id);
            vertex_program_registered = false;
        }
        if (fragment_program_registered) {
            sceGxmShaderPatcherUnregisterProgram(patcher, fragment_program_id);
            fragment_program_registered = false;
        }
    }
    if (vertex_output != nullptr) {
        ReleaseCgOutput(vertex_output);
        vertex_output = nullptr;
    }
    if (fragment_output != nullptr) {
        ReleaseCgOutput(fragment_output);
        fragment_output = nullptr;
    }
    // El modulo del compilador NO se descarga aqui: es estado global del
    // proceso y el rasterizador de la GPU puede tener programas compilados con
    // el vivos. Descargarlo o soltar el compilador invalidaria esa memoria y
    // el fallo apareceria mucho despues, al dibujar. Lo que si se sueltan son
    // los programas que son de este objeto (arriba).
}

const SceShaccCgCompileOutput* ScreenPresenter::CompileShader(SceShaccCgTargetProfile profile,
                                                              const char* name, const char* source) {
    // Delegado en el compilador compartido; ver gxm_cg.h.
    return Gxm::CompileCg(profile, name, source);
}

ScreenPresenter::Source ScreenPresenter::PrepareScreen(Screen& screen,
                                                       const SwRenderer::ScreenInfo& info) {
    // Relleno de color: no hay framebuffer que copiar; el juego pide un color
    // liso (los fundidos lo usan mucho) y se dibuja con el texel de 1x1.
    if (info.fill_enabled) {
        u8* texel = static_cast<u8*>(fill_texel.Data());
        texel[0] = info.fill_r;
        texel[1] = info.fill_g;
        texel[2] = info.fill_b;
        texel[3] = 255;
        return Source::Fill;
    }

    // 'source' apunta al framebuffer dentro de la memoria del invitado y no a
    // una copia: la copia intermedia que habia se quito (ver ScreenInfo::source
    // en renderer_software.h). Aqui abajo es la unica vez que estos bytes se
    // mueven en todo el fotograma.
    if (!info.valid || info.source == nullptr || info.width == 0 || info.height == 0 ||
        info.stride == 0) {
        return Source::Nothing;
    }

    /**
     * Camino directo desde la superficie de GXM (0.1.5.2, 4.6).
     *
     * Si hay una Surface dibujada por la GPU para ESA direccion fisica y no
     * esta detras del invitado (needs_reload), la textura se apunta a su
     * color_buffer: se ahorra WriteBack (lineal -> tiles del invitado) y la
     * subida de mas abajo (invitado -> textura). Son las dos copias de la
     * imagen completa que describe el encargo. Si no hay superficie, o si el
     * interruptor esta apagado, se sigue por el camino de siempre.
     *
     * Las medidas y el formato salen de la SUPERFICIE, no del ScreenInfo: la
     * superficie puede ser de 32 bits aunque el invitado sea RGB8 (no existe
     * color surface de 24), y el stride es el de GXM, no el del juego.
     */
    {
        const auto direct = RasterizerGXM::QueryDirectPresent(info.source_address);
        if (direct.data != nullptr) {
            const u32 needed = direct.stride_bytes * direct.height;
            const bool same_place =
                screen.points_at_gxm_surface && screen.width == direct.width &&
                screen.height == direct.height && screen.stride == direct.stride_bytes;
            if (!same_place) {
                const int rc = sceGxmTextureInitLinearStrided(
                    &screen.texture, const_cast<u8*>(direct.data),
                    static_cast<SceGxmTextureFormat>(direct.gxm_texture_format), direct.width,
                    direct.height, direct.stride_bytes);
                if (rc < 0) {
                    static bool noted_direct_error = false;
                    if (!noted_direct_error) {
                        noted_direct_error = true;
                        LOG_ERROR(Render, "GXM: textura directa no valida ({:#010x}) {}x{} stride {}",
                                  static_cast<u32>(rc), direct.width, direct.height,
                                  direct.stride_bytes);
                    }
                    // Se cae al camino normal de abajo. Resetear la marca: si
                    // points_at_gxm_surface seguia en true, la textura quedaria
                    // apuntando a un buffer GXM obsoleto.
                    screen.points_at_gxm_surface = false;
                    screen.width = 0;
                    screen.height = 0;
                    screen.stride = 0;
                } else {
                    sceGxmTextureSetMinFilter(&screen.texture, SCE_GXM_TEXTURE_FILTER_POINT);
                    sceGxmTextureSetMagFilter(&screen.texture, SCE_GXM_TEXTURE_FILTER_POINT);
                    sceGxmTextureSetUAddrMode(&screen.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
                    sceGxmTextureSetVAddrMode(&screen.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
                    screen.width = direct.width;
                    screen.height = direct.height;
                    screen.stride = direct.stride_bytes;
                    screen.points_at_gxm_surface = true;
                    (void)needed;
                    return Source::Texture;
                }
            } else {
                // Mismas medidas: la textura ya apunta al buffer de GXM que toca
                // (o a otro con la misma geometria; el puntero no cambia entre
                // fotogramas para la misma direccion). Si cambio de direccion
                // con las mismas medidas, points_at_gxm_surface sigue true y el
                // puntero de la textura podria ser viejo: por eso solo se confia
                // cuando ademas la Surface sigue siendo la de esta direccion.
                // QueryDirectPresent ya ha comprobado eso; aun asi se rehace la
                // textura si el buffer de la superficie es distinto del ultimo
                // que se apunto. La forma barata: rehacer siempre que se entre
                // aqui con points y no se sepa -- para no arriesgar, se rehace.
                const int rc = sceGxmTextureInitLinearStrided(
                    &screen.texture, const_cast<u8*>(direct.data),
                    static_cast<SceGxmTextureFormat>(direct.gxm_texture_format), direct.width,
                    direct.height, direct.stride_bytes);
                if (rc >= 0) {
                    sceGxmTextureSetMinFilter(&screen.texture, SCE_GXM_TEXTURE_FILTER_POINT);
                    sceGxmTextureSetMagFilter(&screen.texture, SCE_GXM_TEXTURE_FILTER_POINT);
                    sceGxmTextureSetUAddrMode(&screen.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
                    sceGxmTextureSetVAddrMode(&screen.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
                    return Source::Texture;
                }
                // rc < 0: la textura no se pudo rehacer. Resetear la marca
                // para que el camino normal la reconstruya desde cero.
                screen.points_at_gxm_surface = false;
                screen.width = 0;
                screen.height = 0;
                screen.stride = 0;
            }
        } else if (screen.points_at_gxm_surface) {
            // Volvemos al camino normal: hay que rehacer la textura sobre
            // 'buffer' mas abajo, o quedaria apuntando a una Surface vieja.
            screen.points_at_gxm_surface = false;
            screen.width = 0;
            screen.height = 0;
            screen.stride = 0;
        }
    }

    /**
     * RGB8 SE EXPANDE A 32 BITS AL SUBIRLO, Y ES POR LO QUE HABIA PANTALLA
     * NEGRA.
     *
     * El 3DS admite framebuffers de 24 bits y GXM NO tiene ninguna textura de
     * tres bytes que sirva: SCE_GXM_TEXTURE_FORMAT_U8U8U8_RGB existe como
     * constante, pero sceGxmTextureInitLinearStrided lo rechaza con
     * SCE_GXM_ERROR_UNSUPPORTED (0x805b0009). Cuando eso pasaba, PrepareScreen
     * devolvia "nada que dibujar" y esa pantalla se quedaba en NEGRO, que es
     * justo lo que se veia arriba en las capturas.
     *
     * Medido en consola: "textura de pantalla no valida (0x805b0009) 240x400
     * stride 720 fmt 1", y 720 = 240 * 3, o sea RGB8.
     *
     * La conversion es barata y sin perdida: el invitado guarda B,G,R y se le
     * anade un byte de alfa a 255. Con eso el orden en memoria queda B,G,R,A,
     * que es lo que describe el nombre ARGB de GXM (los nombres listan los
     * canales del mas significativo al menos) -- la misma correspondencia que
     * ya usa el rasterizador para sus superficies RGB8.
     */
    const bool expand_rgb8 = info.format == Pica::PixelFormat::RGB8;
    const u32 src_bpp = BytesPerPixelFor(info.format);
    const u32 bpp = expand_rgb8 ? 4u : src_bpp;
    const u32 src_row_bytes = info.width * src_bpp;
    const u32 row_bytes = info.width * bpp;
    // El salto de fila que se le da a la textura tiene que ser multiplo de 16
    // para el hardware: se aprieta la imagen a ese salto en vez de reutilizar
    // el del juego, que puede ser cualquiera.
    const u32 tight_stride = (row_bytes + 15u) & ~15u;
    const u32 needed = tight_stride * info.height;

    const unsigned long long upload_begin = Common::VitaMicros();

    bool reallocated = false;
    if (needed > screen.buffer.Size()) {
        // CDRAM primero: son 128 MB aparte del presupuesto de la aplicacion y
        // de donde mejor lee el chip. Si no queda, LPDDR sin cachear, que sale
        // del heap pero al menos funciona.
        Allocation buffer = Allocate(Pool::Cdram, needed, SCE_GXM_MEMORY_ATTRIB_READ);
        if (!buffer.Valid()) {
            buffer = Allocate(Pool::Host, needed, SCE_GXM_MEMORY_ATTRIB_READ);
        }
        if (!buffer.Valid()) {
            status = "vita2d (sin memoria gxm)";
            LOG_ERROR(Render, "GXM: sin memoria para la pantalla ({} bytes)", needed);
            ready = false;
            return Source::Nothing;
        }
        screen.buffer = std::move(buffer);
        // La textura apunta al buffer viejo: hay que rehacerla aunque el
        // tamano y el formato no hayan cambiado.
        reallocated = true;
    }

    // Copia por filas. En el caso normal (el juego usa un salto igual al
    // apretado) es una sola copia de bloque; si no, una por fila, que sigue
    // siendo memoria a memoria sin tocar un pixel.
    // Si venimos del camino directo, 'buffer' puede estar sin asignar o con
    // medidas viejas: needed de arriba ya lo ha reasignado si hacía falta.
    u8* dest = static_cast<u8*>(screen.buffer.Data());
    const u8* src = info.source;
    if (expand_rgb8) {
        // Tres bytes a cuatro, fila a fila. Se escribe la palabra entera de una
        // vez en vez de byte a byte: son ~96.000 pixeles por pantalla y por
        // fotograma, y esto corre en el hilo que emula la CPU.
        for (u32 y = 0; y < info.height; y++) {
            const std::size_t src_offset = static_cast<std::size_t>(y) * info.stride;
            if (src_offset + src_row_bytes > info.source_size) {
                // Nunca se lee mas alla del framebuffer del invitado: 'source'
                // apunta a su memoria, no a una copia nuestra.
                break;
            }
            const u8* src_row = src + src_offset;
            u8* dst_row = dest + static_cast<std::size_t>(y) * tight_stride;
            // 0.1.7.6: 16 pixeles por vuelta con NEON (VLD3 separa B, G y R,
            // VST4 los vuelve a juntar con el alfa). Mismos bytes en el mismo
            // orden que el bucle de abajo, que se queda para la cola de la fila.
            u32 x = 0;
            const uint8x16_t opaque = vdupq_n_u8(0xFF);
            for (; x + 16 <= info.width; x += 16) {
                const uint8x16x3_t bgr = vld3q_u8(src_row + x * 3);
                const uint8x16x4_t bgra = {{bgr.val[0], bgr.val[1], bgr.val[2], opaque}};
                vst4q_u8(dst_row + x * 4, bgra);
            }
            for (; x < info.width; x++) {
                const u8* pixel = src_row + x * 3;
                const u32 word = static_cast<u32>(pixel[0]) |
                                 (static_cast<u32>(pixel[1]) << 8) |
                                 (static_cast<u32>(pixel[2]) << 16) | 0xFF000000u;
                std::memcpy(dst_row + x * 4, &word, sizeof(word));
            }
        }
    } else if (info.stride == tight_stride) {
        // Con los dos saltos iguales, 'needed' y 'source_size' valen lo mismo.
        // Se toma el menor de los dos de todas formas: ahora se lee de la
        // memoria del invitado y no de un vector propio, asi que pasarse de
        // largo aqui seria leer fuera del framebuffer del juego.
        std::memcpy(dest, src, std::min<std::size_t>(needed, info.source_size));
    } else {
        for (u32 y = 0; y < info.height; y++) {
            std::memcpy(dest + static_cast<std::size_t>(y) * tight_stride,
                        src + static_cast<std::size_t>(y) * info.stride, row_bytes);
        }
    }

    Common::FrameStats::Add(Common::FrameStats::upload_us, upload_begin);

    if (reallocated || screen.width != info.width || screen.height != info.height ||
        screen.stride != tight_stride || screen.format != info.format) {
        const SceGxmTextureFormat texture_format =
            expand_rgb8 ? SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ARGB : FormatFor(info.format);
        const int rc = sceGxmTextureInitLinearStrided(&screen.texture, dest, texture_format,
                                                      info.width, info.height, tight_stride);
        if (rc < 0) {
            // Una vez y no por fotograma: esto fallaba dos veces en cada uno,
            // y son dos escrituras al registro por fotograma para repetir
            // siempre lo mismo. El dato interesante -- el formato y las medidas
            // con las que falla -- va en la misma linea, que es lo que hace
            // falta para arreglarlo.
            static bool noted_texture_error = false;
            if (!noted_texture_error) {
                noted_texture_error = true;
                LOG_ERROR(Render,
                          "GXM: textura de pantalla no valida ({:#010x}) {}x{} stride {} fmt {}",
                          static_cast<u32>(rc), info.width, info.height, tight_stride,
                          static_cast<u32>(info.format));
            }
            return Source::Nothing;
        }
        // Sin filtrado: la correspondencia es 1:1 y cualquier interpolacion
        // emborrona. Sin repetir: las coordenadas llegan justo al borde.
        sceGxmTextureSetMinFilter(&screen.texture, SCE_GXM_TEXTURE_FILTER_POINT);
        sceGxmTextureSetMagFilter(&screen.texture, SCE_GXM_TEXTURE_FILTER_POINT);
        sceGxmTextureSetUAddrMode(&screen.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
        sceGxmTextureSetVAddrMode(&screen.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);

        screen.width = info.width;
        screen.height = info.height;
        screen.stride = tight_stride;
        screen.format = info.format;
    }

    return Source::Texture;
}

float ScreenPresenter::NdcX(int pixel_x) const {
    return static_cast<float>(pixel_x) / (screen_width * 0.5f) - 1.0f;
}

float ScreenPresenter::NdcY(int pixel_y) const {
    return 1.0f - static_cast<float>(pixel_y) / (screen_height * 0.5f);
}

void ScreenPresenter::WriteQuad(float* out, Rect rect, float u0, float v0, float u1,
                                float v1) const {
    // Orden de triangle strip: arriba izquierda, arriba derecha, abajo
    // izquierda, abajo derecha. El eje Y de las coordenadas de recorte crece
    // hacia arriba (es el criterio de OpenGL, y el que usa la matriz de vita2d
    // para dibujar en pixeles), de ahi el NdcY invertido.
    const float x0 = NdcX(rect.left);
    const float x1 = NdcX(rect.left + rect.width);
    const float y0 = NdcY(rect.top);
    const float y1 = NdcY(rect.top + rect.height);

    const float xs[kVerticesPerQuad] = {x0, x1, x0, x1};
    const float ys[kVerticesPerQuad] = {y0, y0, y1, y1};
    const float us[kVerticesPerQuad] = {u0, u0, u1, u1};
    const float vs[kVerticesPerQuad] = {v0, v1, v0, v1};

    for (u32 i = 0; i < kVerticesPerQuad; i++) {
        float* vertex = out + i * 6;
        vertex[0] = xs[i];
        vertex[1] = ys[i];
        vertex[2] = 0.0f;
        vertex[3] = 1.0f;
        vertex[4] = us[i];
        vertex[5] = vs[i];
    }
}

void ScreenPresenter::Initialize() {
    if (attempted) {
        return;
    }
    attempted = true;
    ready = Init();
    // Al fichero de diagnostico, no solo al registro: si la consola muere
    // despues por algo no relacionado, esto dice con que camino de presentacion
    // iba.
    Common::VitaNote("gxm", status);
}

void ScreenPresenter::Draw(const SwRenderer::ScreenInfo& top, const SwRenderer::ScreenInfo& bottom,
                           Rect top_rect, Rect bottom_rect) {
    Initialize();
    if (!ready) {
        return;
    }

    auto* vertex_data = static_cast<float*>(vertices.Data());
    const Source top_source = PrepareScreen(screens[0], top);
    if (!ready) {
        // PrepareScreen se apaga solo si no ha podido subir la pantalla (sin
        // memoria): a partir de aqui lo presenta todo vita2d.
        return;
    }
    const Source bottom_source = PrepareScreen(screens[1], bottom);
    if (!ready || (top_source == Source::Nothing && bottom_source == Source::Nothing)) {
        return;
    }

    /**
     * La correspondencia entre pixel de destino y texel de origen sale de como
     * esta el framebuffer del 3DS en memoria, que es 240 de ancho por 400 (o
     * 320) de alto, girado respecto a como se ve. El bucle de CPU al que esto
     * sustituye escribia, para la pantalla de arriba y con W=240:
     *
     *     destino(x, y) = origen(fila = x, columna = W - 1 - y)
     *
     * O sea que la U depende de la Y del destino y la V de la X:
     *
     *     u = (W - 1 - y + 0,5) / W        v = (x + 0,5) / H
     *
     * Evaluando en los bordes del quad (x = -0,5 y x = ancho - 0,5, etc.)
     * salen esquinas limpias: arriba izquierda (u=1, v=0), arriba derecha
     * (1, 1), abajo izquierda (0, 0) y abajo derecha (0, 1). Que la U llegue
     * justo a 1 no importa porque el direccionamiento esta en CLAMP y las
     * muestras caen en el centro de los texeles.
     */
    if (top_source == Source::Fill) {
        WriteQuad(vertex_data, top_rect, 0.5f, 0.5f, 0.5f, 0.5f);
    } else if (top_source == Source::Texture) {
        WriteQuad(vertex_data, top_rect, 1.0f, 0.0f, 0.0f, 1.0f);
    }
    if (bottom_source == Source::Fill) {
        WriteQuad(vertex_data + kVerticesPerQuad * 6, bottom_rect, 0.5f, 0.5f, 0.5f, 0.5f);
    } else if (bottom_source == Source::Texture) {
        WriteQuad(vertex_data + kVerticesPerQuad * 6, bottom_rect, 1.0f, 0.0f, 0.0f, 1.0f);
    }

    // Estado 2D. La mezcla va dentro del programa de fragmentos (creado sin
    // mezcla); lo de aqui son estados del contexto que no se pueden leer hacia
    // atras, asi que se dejan en el unico valor que no puede estropear a nadie:
    // sin culling, sin prueba de profundidad y sin escritura de profundidad.
    // vita2d reimprime sus programas y su stream en cada dibujado, de modo que
    // esto no le afecta.
    sceGxmSetCullMode(context, SCE_GXM_CULL_NONE);
    sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetBackDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetBackDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetVertexProgram(context, vertex_program);
    sceGxmSetFragmentProgram(context, fragment_program);
    sceGxmSetVertexStream(context, 0, vertices.Data());

    const auto draw_quad = [&](const Screen& screen, Source source, u32 first_index) {
        const SceGxmTexture* texture = source == Source::Fill ? &fill_texture : &screen.texture;
        sceGxmSetFragmentTexture(context, texture_unit, texture);
        const auto* index_data = static_cast<const u16*>(indices.Data()) + first_index;
        sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP, SCE_GXM_INDEX_FORMAT_U16, index_data,
                   kIndicesPerQuad);
    };

    if (top_source != Source::Nothing) {
        draw_quad(screens[0], top_source, 0);
    }
    if (bottom_source != Source::Nothing) {
        draw_quad(screens[1], bottom_source, kIndicesPerQuad);
    }
}

} // namespace Gxm
