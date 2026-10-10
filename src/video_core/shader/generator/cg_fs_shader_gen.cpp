// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/shader/generator/cg_fs_shader_gen.h"

#include <array>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica/regs_lighting.h"
#include "video_core/pica/regs_texturing.h"

namespace Pica::Shader::Generator::GXM {

namespace {

using FramebufferRegs = Pica::FramebufferRegs;
using LightingRegs = Pica::LightingRegs;
using TexturingRegs = Pica::TexturingRegs;
using Source = TexturingRegs::TevStageConfig::Source;
using ColorModifier = TexturingRegs::TevStageConfig::ColorModifier;
using AlphaModifier = TexturingRegs::TevStageConfig::AlphaModifier;
using Operation = TexturingRegs::TevStageConfig::Operation;

/**
 * El shader emula la PICA con aritmetica de coma flotante, asi que ni las
 * combinaciones "de verdad" (min/max entre canales) ni el logic op se pueden
 * dejar al hardware: se generan en la shader tal cual hace el de GLSL. Lo que
 * este backend NO puede hacer todavia se rechaza antes de generar una linea.
 */
bool IsSupported(const FSConfig& config, const char** out_reason) {
    const auto reject = [out_reason](const char* reason) {
        if (out_reason != nullptr) {
            *out_reason = reason;
        }
        return false;
    };
    /**
     * LA ILUMINACION POR FRAGMENTO YA NO SE RECHAZA (desde 0.1.0.17).
     *
     * Era, con diferencia, el motivo que mas lotes tiraba a software: casi
     * cualquier juego en 3D de la eShop enciende la iluminacion por fragmento
     * de la PICA200, asi que el camino de GPU se quedaba para interfaz y
     * sprites. Ahora se genera el modelo completo -- ocho luces, difusa,
     * especular 0 y 1, las seis tablas de busqueda, fresnel y bump mapping --
     * en WriteLighting, con glsl_fs_shader_gen.cpp como referencia de que
     * emitir y sw_lighting.cpp como referencia de que tiene que salir.
     *
     * Lo que sigue fuera es el PASE DE SOMBRAS de la iluminacion: muestrea una
     * textura de sombra (que ya esta fuera por texture0_type) y mezcla su
     * resultado en la difusa y la especular. Queda para cuando haya sombras.
     */
    if (config.lighting.enable_shadow) {
        return reject("luz con sombra");
    }
    // Texturas procedurales: necesitan noise y LUTs de la PICA.
    if (config.proctex.enable) {
        return reject("proctex");
    }
    // Pase de sombras: cuando la PICA esta escribiendo el mapa de sombras, el
    // shader no pinta color sino profundidad empaquetada. Sin atomos ni imagenes
    // en GXM todavia no se puede reproducir, asi que va a software.
    if (config.framebuffer.shadow_rendering) {
        return reject("shadow pass");
    }
    // OJO: shadow_texture_orthographic NO se rechaza. Es una bandera de la
    // configuracion de textura que solo usan los shaders que muestrean una
    // textura de sombra (via texture0_type, que si se comprueba abajo); en un
    // dibujado normal no cambia nada. Rechazarla tiraba a software lotes que no
    // tienen nada de sombras (medido en consola con NSMB2: era el unico motivo).
    // Gas está sin implementar hasta en el de GLSL; Fog si está soportada.
    if (config.texture.fog_mode == TexturingRegs::FogMode::Gas) {
        return reject("fog gas");
    }
    // Modos de textura: 2D, "apagada" y, desde 0.1.0.44, PROYECTADA (u y v
    // divididas por la w de la coordenada 0; ver WriteTextureSamples). Cubo y
    // sombra llevan otra matematica y siguen en software, cada una con su
    // propio motivo: hasta 0.1.0.43 las tres salian como "tipo de textura 0" y
    // no habia forma de saber cual era la que usaba el juego.
    const auto texture0_type = config.texture.texture0_type.Value();
    switch (texture0_type) {
    case TexturingRegs::TextureConfig::Texture2D:
    case TexturingRegs::TextureConfig::Projection2D:
    case TexturingRegs::TextureConfig::Disabled:
        break;
    case TexturingRegs::TextureConfig::TextureCube:
        return reject("textura 0 de cubo");
    case TexturingRegs::TextureConfig::ShadowCube:
        return reject("textura 0 sombra de cubo");
    case TexturingRegs::TextureConfig::Shadow2D:
        return reject("textura 0 sombra 2D");
    default:
        return reject("tipo de textura 0 desconocido");
    }
    return true;
}

/// Emite el Cg de una configuracion, o deja el motivo del rechazo en el LOG.
class FragmentWriter {
public:
    explicit FragmentWriter(const FSConfig& config_, const char** reason_,
                            bool alpha_from_blend_const_, bool color_from_blend_const_)
        : config{config_}, reason{reason_}, alpha_from_blend_const{alpha_from_blend_const_},
          color_from_blend_const{color_from_blend_const_} {}

    std::optional<std::string> Generate() {
        if (!IsSupported(config, reason)) {
            LOG_DEBUG(Render, "GXM FS: configuracion no soportada todavia; se cae a software");
            return std::nullopt;
        }

        WriteHeader();

        // Fuentes que piden las etapas: solo se declaran los samplers de las
        // unidades que de verdad se muestrean, y solo se declaran los uniforms
        // que se usan. El compilador de Cg habria eliminado los que sobren,
        // pero asi el shader es mas pequeno y el mapeo de parametros mas claro.
        ScanSources();

        if (config.framebuffer.alpha_test_func == FramebufferRegs::CompareFunc::Never) {
            // Nada puede pasar la prueba de alfa: un shader que descarta todo y
            // ya. Es el mismo atajo que toma el generador de GLSL.
            // El color se escribe antes de descartar aunque no se vaya a usar:
            // un parametro de salida que ningun camino escribe es algo que
            // algunos compiladores de Cg rechazan, y aqui no cuesta nada.
            out += "void main(float4 primary_color : COLOR0, float2 tc0 : TEXCOORD0, "
                   "float2 tc1 : TEXCOORD1, float2 tc2 : TEXCOORD2, "
                   "out float4 gl_FragColor : COLOR)\n{\n"
                   "    gl_FragColor = float4(0.0, 0.0, 0.0, 0.0);\n"
                   "    discard;\n}\n";
            return out;
        }

        WriteInterface();
        WriteUniforms();

        // gl_FragCoord (WPOS) da la profundidad de ventana, que con nuestro
        // viewport ya es exactamente la de la PICA; la necesita la niebla.
        out += "void main(float4 primary_color : COLOR0, float2 tc0 : TEXCOORD0, "
               "float2 tc1 : TEXCOORD1, float2 tc2 : TEXCOORD2, ";
        if (config.lighting.enable) {
            // Los dos varyings que solo escribe el shader de vertices largo
            // (kVertexSourceLit). Declararlos cuando no hay luz costaria
            // interpolarlos en cada pixel para nada, asi que van aqui dentro.
            out += "float4 normquat : TEXCOORD3, float3 view : TEXCOORD4, ";
        }
        if (IsProjected()) {
            // La W de la coordenada 0: solo la escriben las variantes _proj
            // del shader de vertices (kVertexSourceProj en rasterizer_gxm.cpp).
            out += "float tc0_w : TEXCOORD5, ";
        }
        out += "float4 gl_FragCoord : WPOS, "
               "out float4 gl_FragColor : COLOR)\n{\n";

        // La PICA redondea el color primario a 8 bits antes de meterlo en la
        // primera etapa TEV; sin esto el resultado no coincide con el de
        // software.
        // Las texturas se muestrean UNA VEZ cada una, aqui arriba, y las etapas
        // TEV usan el resultado. Antes cada fuente emitia su propio tex2D, asi
        // que una unidad usada en cuatro etapas se muestreaba cuatro veces; el
        // compilador de Cg probablemente lo juntaba, pero con el color de borde
        // de por medio la expresion deja de ser una sola llamada y conviene no
        // depender de eso.
        WriteTextureSamples();

        out += "    float4 rounded_primary_color = byteround(primary_color);\n";
        /**
         * Los dos colores de la ILUMINACION arrancan en NEGRO, no en el color
         * del vertice.
         *
         * Aqui ponia primary_fragment_color = rounded_primary_color, y eso no
         * es lo que hace el hardware: en sw_rasterizer.cpp los dos arrancan en
         * {0,0,0,0} y solo los escribe el bloque de iluminacion. Una etapa TEV
         * que lea PrimaryFragmentColor con la luz apagada tiene que ver negro,
         * no el color interpolado del vertice -- que ademas ya tiene su propia
         * fuente, PrimaryColor. Con la iluminacion todavia sin implementar la
         * diferencia no se podia comprobar; ahora que se implementa, se
         * corrige.
         */
        out += "    float4 primary_fragment_color = float4(0.0, 0.0, 0.0, 0.0);\n";
        out += "    float4 secondary_fragment_color = float4(0.0, 0.0, 0.0, 0.0);\n";
        if (config.lighting.enable) {
            WriteLighting();
        }
        out += "    float4 combiner_buffer = float4(0.0, 0.0, 0.0, 0.0);\n";
        out += "    float4 next_combiner_buffer = tev_combiner_buffer_color;\n";
        out += "    float4 combiner_output = float4(0.0, 0.0, 0.0, 0.0);\n";
        out += "    float3 color_results_1 = float3(0.0, 0.0, 0.0);\n";
        out += "    float3 color_results_2 = float3(0.0, 0.0, 0.0);\n";
        out += "    float3 color_results_3 = float3(0.0, 0.0, 0.0);\n";
        out += "    float alpha_results_1 = 0.0;\n";
        out += "    float alpha_results_2 = 0.0;\n";
        out += "    float alpha_results_3 = 0.0;\n";

        for (u32 index = 0; index < config.texture.tev_stages.size(); index++) {
            WriteTevStage(index);
        }

        WriteAlphaTestCondition(config.framebuffer.alpha_test_func);

        // Niebla: el factor sale de la LUT de 128 entradas (dos floats por
        // entrada: valor y pendiente) indexada por la profundidad; el color es
        // el registro de niebla. Misma matematica que el generador de GLSL.
        if (config.texture.fog_mode == TexturingRegs::FogMode::Fog) {
            if (config.texture.fog_flip) {
                out += "    float fog_index = (1.0 - clamp(gl_FragCoord.z, 0.0, 1.0)) * 128.0;\n";
            } else {
                out += "    float fog_index = clamp(gl_FragCoord.z, 0.0, 1.0) * 128.0;\n";
            }
            out += "    float fog_i = clamp(floor(fog_index), 0.0, 127.0);\n";
            out += "    float fog_f = fog_index - fog_i;\n";
            out += "    int fog_ii = int(fog_i);\n";
            out += "    float fog_factor = fog_lut[fog_ii * 2] + fog_lut[fog_ii * 2 + 1] * "
                   "fog_f;\n";
            out += "    fog_factor = clamp(fog_factor, 0.0, 1.0);\n";
            out += "    combiner_output.rgb = lerp(fog_color, combiner_output.rgb, "
                   "fog_factor);\n";
        }

        // El redondeo final: la PICA escribe siempre 8 bits por canal.
        out += "    combiner_output = byteround(combiner_output);\n";
        out += "    gl_FragColor = combiner_output;\n";
        // Despues de la prueba de alfa, que sigue usando el alfa calculado.
        if (alpha_from_blend_const) {
            out += "    gl_FragColor.a = blend_const_alpha;\n";
        }
        if (color_from_blend_const) {
            out += "    gl_FragColor.rgb = blend_const_color;\n";
        }
        out += "}\n";
        return out;
    }

private:
    void WriteHeader() {
        out += "// Shader de fragmentos de la PICA generado por Azahar para GXM.\n";
        out += "// NO EDITAR A MANO: lo compila la consola con SceShaccCg.\n\n";
        // El redondeo a 8 bits de la PICA. Se hace con floor y no con round
        // porque round de Cg no garantiza el empate hacia arriba, y los
        // valores aqui son siempre no negativos.
        out += "float4 byteround(float4 x) {\n";
        out += "    return floor(x * 255.0 + 0.5) / 255.0;\n";
        out += "}\n\n";
        if (config.lighting.enable) {
            WriteLightingHelpers();
        }
    }

    /**
     * Lo que necesita la iluminacion antes de main: el giro por cuaternion y la
     * lectura de las tablas de busqueda.
     *
     * LAS TABLAS VAN EN UNA TEXTURA, NO EN UNIFORMS. Son 24 tablas de 256
     * entradas con dos floats cada una: 12.288 floats, 48 KB. Eso no cabe en el
     * buffer de uniforms por defecto de un programa de fragmentos, y aunque
     * cupiera se reescribiria entero en cada lote. Como textura se sube una vez
     * por cambio (la PICA marca cuales ensucia) y el chip la lee como cualquier
     * otra. La fila es el indice de tabla -- el mismo numero que el enumerado
     * LightingSampler, de 0 a 23 -- y la columna la entrada.
     *
     * DOS CANALES DE 32 BITS: valor y PENDIENTE. La tabla de la PICA no guarda
     * solo el valor: guarda tambien la diferencia hasta la entrada siguiente,
     * cuantizada aparte. Por eso la interpolacion se hace a mano (valor +
     * pendiente * delta) con filtrado de punto, y no dejandosela al filtro
     * bilineal del chip: el resultado no seria el mismo que el del rasterizador
     * de software, que usa la pendiente guardada.
     *
     * EL SIGNO PARTE LA TABLA POR LA MITAD. Las entradas con indice de 0 a 1
     * usan las 256 posiciones; las de -1 a 1 usan 128 por lado y las negativas
     * viven en la segunda mitad (indice + 256), que es como las coloca la PICA.
     */
    void WriteLightingHelpers() {
        out += "uniform sampler2D lighting_lut;\n\n";
        out += "float3 quaternion_rotate(float4 q, float3 v) {\n";
        out += "    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);\n";
        out += "}\n\n";
        /**
         * OJO CON EL NOMBRE 'row': ES PALABRA RESERVADA EN Cg.
         *
         * Estas tres funciones tomaban el indice de tabla en un parametro
         * llamado 'row' y el compilador de la consola lo rechazaba con
         * "expected an identifier but found keyword 'row'". Como el error es de
         * COMPILACION, no de generacion, no se veia desde aqui: el shader salia
         * bien formado a la vista y moria en la Vita.
         *
         * Y el efecto era enorme, porque estas funciones solo aparecen cuando
         * hay iluminacion: TODO lote con luz fallaba al compilar y caia al
         * rasterizador de software. Medido en consola con Pokemon Rubi Omega:
         * 14.717 triangulos por software contra 576 por GPU, con el 94,8% de
         * los pixeles pasando por iluminacion. O sea que la iluminacion por
         * fragmento no llego a ejecutarse ni una vez desde que se escribio.
         *
         * Cg reserva unas cuantas palabras que en GLSL son nombres normales
         * (las de disposicion de matrices, entre ellas 'row'). Al anadir aqui
         * una variable nueva conviene no usar nombres de una sola palabra
         * corriente; con el prefijo 'lut_' no hay duda.
         */
        out += "float LookupLightingLUT(float lut_row, int index, float delta) {\n";
        out += "    float2 uv = float2(((float)index + 0.5) / 256.0, (lut_row + 0.5) / 24.0);\n";
        out += "    float2 lut_entry = tex2D(lighting_lut, uv).rg;\n";
        out += "    return lut_entry.x + lut_entry.y * delta;\n";
        out += "}\n\n";
        out += "float LookupLightingLUTUnsigned(float lut_row, float pos) {\n";
        out += "    int index = (int)clamp(floor(pos * 256.0), 0.0, 255.0);\n";
        out += "    float delta = pos * 256.0 - (float)index;\n";
        out += "    return LookupLightingLUT(lut_row, index, delta);\n";
        out += "}\n\n";
        out += "float LookupLightingLUTSigned(float lut_row, float pos) {\n";
        out += "    int index = (int)clamp(floor(pos * 128.0), -128.0, 127.0);\n";
        out += "    float delta = pos * 128.0 - (float)index;\n";
        out += "    if (index < 0) index += 256;\n";
        out += "    return LookupLightingLUT(lut_row, index, delta);\n";
        out += "}\n\n";
    }

    /// Ojea las etapas TEV para saber que samplers hay que declarar.
    ///
    /// Y no solo las etapas: el bump mapping lee una unidad de textura que
    /// puede no usar ninguna etapa TEV (la PICA la muestrea solo para sacar la
    /// normal). Sin marcarla aqui, el shader la usaria sin declararla y el
    /// rasterizador no le ataria ninguna textura.
    void ScanSources() {
        if (config.lighting.enable &&
            config.lighting.bump_mode != LightingRegs::LightingBumpMode::None) {
            MarkTextureUnit(config.lighting.bump_selector.Value());
        }
        for (const auto& stage : config.texture.tev_stages) {
            const TexturingRegs::TevStageConfig tev = stage;
            const std::array<Source, 3> color_sources = {tev.color_source1.Value(),
                                                         tev.color_source2.Value(),
                                                         tev.color_source3.Value()};
            const std::array<Source, 3> alpha_sources = {tev.alpha_source1.Value(),
                                                         tev.alpha_source2.Value(),
                                                         tev.alpha_source3.Value()};
            for (const Source source : color_sources) {
                MarkSource(source);
            }
            for (const Source source : alpha_sources) {
                MarkSource(source);
            }
        }
    }

    void MarkTextureUnit(u32 unit) {
        switch (unit) {
        case 0:
            uses_tex0 = true;
            break;
        case 1:
            uses_tex1 = true;
            break;
        case 2:
            uses_tex2 = true;
            break;
        default:
            break;
        }
    }

    void MarkSource(Source source) {
        switch (source) {
        case Source::Texture0:
            uses_tex0 = true;
            break;
        case Source::Texture1:
            uses_tex1 = true;
            break;
        case Source::Texture2:
            uses_tex2 = true;
            break;
        case Source::Constant:
            uses_const_color = true;
            break;
        case Source::PreviousBuffer:
            uses_combiner_buffer_color = true;
            break;
        default:
            break;
        }
    }

    void WriteInterface() {
        if (uses_tex0) {
            out += "uniform sampler2D tex0;\n";
        }
        if (uses_tex1) {
            out += "uniform sampler2D tex1;\n";
        }
        if (uses_tex2) {
            out += "uniform sampler2D tex2;\n";
        }
        out += "\n";
    }

    void WriteUniforms() {
        if (uses_const_color) {
            out += "uniform float4 const_color[6];\n";
        }
        /**
         * SIEMPRE, Y NO SOLO CUANDO UNA ETAPA TEV LO LEA. Este era el fallo que
         * tenia el camino de GPU practicamente muerto.
         *
         * Aqui se declaraba solo si uses_combiner_buffer_color, que lo enciende
         * ScanSources cuando alguna etapa usa la fuente PreviousBuffer. Pero el
         * cuerpo de main escribe SIEMPRE
         *
         *     float4 next_combiner_buffer = tev_combiner_buffer_color;
         *
         * porque el buffer del combinador arranca con ese color lo lea quien lo
         * lea. O sea que en toda configuracion que no usara PreviousBuffer --
         * que son casi todas -- el shader referenciaba un identificador que no
         * existia y el compilador de la consola lo rechazaba con "unexpected
         * undeclared identifier". El lote se iba a software, y ademas pagando
         * una compilacion fallida.
         *
         * Se declara siempre. Si de verdad no se usa, el compilador lo elimina
         * y sceGxmProgramFindParameterByName devuelve nulo, que es justo el caso
         * que el rasterizador ya sabe saltarse.
         */
        out += "uniform float4 tev_combiner_buffer_color;\n";
        if (alpha_from_blend_const) {
            out += "uniform float blend_const_alpha;\n";
        }
        if (color_from_blend_const) {
            out += "uniform float3 blend_const_color;\n";
        }
        if (config.framebuffer.alpha_test_func != FramebufferRegs::CompareFunc::Always) {
            out += "uniform float alphatest_ref;\n";
        }
        if (config.texture.fog_mode == TexturingRegs::FogMode::Fog) {
            out += "uniform float fog_lut[256];\n";
            out += "uniform float3 fog_color;\n";
        }
        if (AnyBorderUsed()) {
            // Los tres a la vez aunque solo una unidad lo use: el rasterizador
            // sube las tres de golpe y asi el indice del array es la unidad,
            // sin tabla de por medio.
            out += "uniform float4 tex_border_color[3];\n";
        }
        if (config.lighting.enable) {
            /**
             * Los parametros de las ocho luces.
             *
             * SIEMPRE LOS OCHO Y SIEMPRE float4, aunque el juego encienda una
             * sola y aunque tres de los cuatro canales sobren. Lo primero
             * porque el shader indexa por el NUMERO de luz (0 a 7), que no
             * tiene por que ser el mismo que el hueco que ocupa. Lo segundo
             * porque como se empaqueta un array de float3 en el buffer de
             * uniforms de la USSE no esta documentado -- puede ir apretado o
             * alineado a cuatro -- y equivocarse ahi no da error: desplaza los
             * valores y la escena sale iluminada al azar. Con float4 no hay
             * duda posible.
             *
             * Los que el compilador de Cg vea que no se usan desaparecen del
             * programa, y el rasterizador los busca por nombre y se salta los
             * que no encuentre, asi que declararlos todos no cuesta nada.
             */
            out += "uniform float4 light_specular_0[8];\n";
            out += "uniform float4 light_specular_1[8];\n";
            out += "uniform float4 light_diffuse[8];\n";
            out += "uniform float4 light_ambient[8];\n";
            out += "uniform float4 light_position[8];\n";
            out += "uniform float4 light_spot_direction[8];\n";
            // x = bias, y = escala. Los dos de la atenuacion por distancia.
            out += "uniform float4 light_dist_atten[8];\n";
            out += "uniform float4 lighting_global_ambient;\n";
        }
        out += "\n";
    }

    /// Esta unidad devuelve el color de borde fuera de [0, 1] en algun eje?
    [[nodiscard]] bool BorderUsed(u32 unit) const {
        const auto& border = config.texture.texture_border_color[unit];
        return border.enable_s != 0 || border.enable_t != 0;
    }

    [[nodiscard]] bool AnyBorderUsed() const {
        return (uses_tex0 && BorderUsed(0)) || (uses_tex1 && BorderUsed(1)) ||
               (uses_tex2 && BorderUsed(2));
    }

    /**
     * Muestrea cada unidad usada UNA sola vez, con el color de borde aplicado.
     *
     * DE DONDE SALE LA BANDERA. No se mira el registro de repetido: se mira
     * config.texture.texture_border_color[unidad], que es un campo que ya
     * rellena la configuracion compartida en ApplyProfile cuando el perfil no
     * declara has_custom_border_color -- y el nuestro no lo declara, porque GXM
     * no tiene donde escribir un color de borde (no existe
     * sceGxmTextureSetBorderColor). Es exactamente el mismo camino por el que
     * pasa el generador de GLSL, asi que los dos emiten lo mismo.
     *
     * POR QUE EL tex2D VA FUERA DEL CONDICIONAL. Muestrear una textura dentro
     * de una rama que no toman todos los fragmentos del grupo deja las
     * derivadas -- de las que sale el nivel de detalle -- sin definir. Aqui se
     * muestrea siempre y despues se elige, que ademas es lo que hace el chip de
     * todas formas.
     *
     * DONDE NO COINCIDE EXACTAMENTE CON EL RASTERIZADOR DE SOFTWARE. El de
     * software decide el borde con la coordenada ya pasada a TEXEL entero
     * (s < 0 || s >= ancho), asi que una coordenada apenas negativa -- que al
     * truncar hacia cero da el texel 0 -- NO le saca el borde. Aqui, como en el
     * generador de GLSL, se compara la coordenada normalizada contra [0, 1].
     * Los dos backends de escritorio llevan anos con esta version, y la
     * diferencia es como mucho un texel en el borde de una textura que el juego
     * ha pedido recortar; se deja igual que ellos a proposito, para que no haya
     * tres comportamientos distintos.
     */
    void WriteTextureSamples() {
        const auto emit = [this](u32 unit, bool used, const char* coord) {
            if (!used) {
                return;
            }
            out += fmt::format("    float4 raw_tex{0} = tex2D(tex{0}, {1});\n", unit, coord);
            const auto& border = config.texture.texture_border_color[unit];
            std::string condition;
            if (border.enable_s != 0) {
                condition += fmt::format("{0}.x < 0.0 || {0}.x > 1.0", coord);
            }
            if (border.enable_t != 0) {
                if (!condition.empty()) {
                    condition += " || ";
                }
                condition += fmt::format("{0}.y < 0.0 || {0}.y > 1.0", coord);
            }
            if (condition.empty()) {
                out += fmt::format("    float4 texcolor{0} = raw_tex{0};\n", unit);
                return;
            }
            out += fmt::format(
                "    float4 texcolor{0} = ({1}) ? tex_border_color[{0}] : raw_tex{0};\n", unit,
                condition);
        };
        if (uses_tex0 && IsProjected()) {
            // Proyeccion: u/w y v/w, como TextureColor del rasterizador de
            // software, y DESPUES todo lo demas (borde incluido) con la
            // coordenada ya dividida. Las tres interpoladas con correccion de
            // perspectiva por separado, que es lo que hace tambien el software.
            out += "    float2 tc0_proj = tc0 / tc0_w;\n";
            emit(0, uses_tex0, "tc0_proj");
        } else {
            emit(0, uses_tex0, "tc0");
        }
        emit(1, uses_tex1, "tc1");
        // La unidad 2 puede muestrear con la coordenada 1 (texture2_use_coord1),
        // como en el generador de GLSL. Hasta 0.1.9.6 se rechazaba, y en la
        // intro de Zafiro Alfa (el cielo) 128 lotes por fotograma se iban a
        // software: 2,2 FPS.
        emit(2, uses_tex2, config.texture.texture2_use_coord1 ? "tc1" : "tc2");
    }

    bool IsProjected() const {
        return config.texture.texture0_type.Value() ==
               TexturingRegs::TextureConfig::Projection2D;
    }

    std::string GetSource(Source source, u32 tev_index) {
        switch (source) {
        case Source::PrimaryColor:
            return "rounded_primary_color";
        case Source::PrimaryFragmentColor:
            return "primary_fragment_color";
        case Source::SecondaryFragmentColor:
            return "secondary_fragment_color";
        case Source::Texture0:
            return "texcolor0";
        case Source::Texture1:
            return "texcolor1";
        case Source::Texture2:
            return "texcolor2";
        case Source::PreviousBuffer:
            return "combiner_buffer";
        case Source::Constant:
            return fmt::format("const_color[{}]", tev_index);
        case Source::Previous:
            return "combiner_output";
        default:
            LOG_CRITICAL(Render, "GXM FS: fuente TEV desconocida {}", source);
            return "float4(0.0, 0.0, 0.0, 0.0)";
        }
    }

    void AppendColorModifier(ColorModifier modifier, Source source, u32 tev_index) {
        const TexturingRegs::TevStageConfig stage = config.texture.tev_stages[tev_index];
        // La etapa 0 no tiene "resultado anterior": tomar Previous ahi es tomar
        // el color inicial del buffer del combinador, que es lo que hace la
        // fuente 3 (ver el generador de GLSL, del que esto es traduccion).
        const bool force_source3 = tev_index == 0 && source == Source::Previous;
        const std::string color_source =
            GetSource(force_source3 ? stage.color_source3.Value() : source, tev_index);
        switch (modifier) {
        case ColorModifier::SourceColor:
            out += fmt::format("{}.rgb", color_source);
            break;
        case ColorModifier::OneMinusSourceColor:
            out += fmt::format("(float3(1.0, 1.0, 1.0) - {}.rgb)", color_source);
            break;
        case ColorModifier::SourceAlpha:
            out += fmt::format("{}.aaa", color_source);
            break;
        case ColorModifier::OneMinusSourceAlpha:
            out += fmt::format("(float3(1.0, 1.0, 1.0) - {}.aaa)", color_source);
            break;
        case ColorModifier::SourceRed:
            out += fmt::format("{}.rrr", color_source);
            break;
        case ColorModifier::OneMinusSourceRed:
            out += fmt::format("(float3(1.0, 1.0, 1.0) - {}.rrr)", color_source);
            break;
        case ColorModifier::SourceGreen:
            out += fmt::format("{}.ggg", color_source);
            break;
        case ColorModifier::OneMinusSourceGreen:
            out += fmt::format("(float3(1.0, 1.0, 1.0) - {}.ggg)", color_source);
            break;
        case ColorModifier::SourceBlue:
            out += fmt::format("{}.bbb", color_source);
            break;
        case ColorModifier::OneMinusSourceBlue:
            out += fmt::format("(float3(1.0, 1.0, 1.0) - {}.bbb)", color_source);
            break;
        default:
            out += "float3(0.0, 0.0, 0.0)";
            LOG_CRITICAL(Render, "GXM FS: modificador de color TEV desconocido {}", modifier);
            break;
        }
    }

    void AppendAlphaModifier(AlphaModifier modifier, Source source, u32 tev_index) {
        const TexturingRegs::TevStageConfig stage = config.texture.tev_stages[tev_index];
        const bool force_source3 = tev_index == 0 && source == Source::Previous;
        const std::string alpha_source =
            GetSource(force_source3 ? stage.alpha_source3.Value() : source, tev_index);
        switch (modifier) {
        case AlphaModifier::SourceAlpha:
            out += fmt::format("{}.a", alpha_source);
            break;
        case AlphaModifier::OneMinusSourceAlpha:
            out += fmt::format("(1.0 - {}.a)", alpha_source);
            break;
        case AlphaModifier::SourceRed:
            out += fmt::format("{}.r", alpha_source);
            break;
        case AlphaModifier::OneMinusSourceRed:
            out += fmt::format("(1.0 - {}.r)", alpha_source);
            break;
        case AlphaModifier::SourceGreen:
            out += fmt::format("{}.g", alpha_source);
            break;
        case AlphaModifier::OneMinusSourceGreen:
            out += fmt::format("(1.0 - {}.g)", alpha_source);
            break;
        case AlphaModifier::SourceBlue:
            out += fmt::format("{}.b", alpha_source);
            break;
        case AlphaModifier::OneMinusSourceBlue:
            out += fmt::format("(1.0 - {}.b)", alpha_source);
            break;
        default:
            out += "0.0";
            LOG_CRITICAL(Render, "GXM FS: modificador de alfa TEV desconocido {}", modifier);
            break;
        }
    }

    void AppendColorCombiner(Operation operation) {
        const auto get_combiner = [operation] {
            switch (operation) {
            case Operation::Replace:
                return std::string{"color_results_1"};
            case Operation::Modulate:
                return std::string{"(color_results_1 * color_results_2)"};
            case Operation::Add:
                return std::string{"(color_results_1 + color_results_2)"};
            case Operation::AddSigned:
                return std::string{"(color_results_1 + color_results_2 - float3(0.5, 0.5, 0.5))"};
            case Operation::Lerp:
                return std::string{"lerp(color_results_2, color_results_1, color_results_3)"};
            case Operation::Subtract:
                return std::string{"(color_results_1 - color_results_2)"};
            case Operation::MultiplyThenAdd:
                return std::string{"(color_results_1 * color_results_2 + color_results_3)"};
            case Operation::AddThenMultiply:
                return std::string{"(min(color_results_1 + color_results_2, "
                                   "float3(1.0, 1.0, 1.0)) * color_results_3)"};
            case Operation::Dot3_RGB:
            case Operation::Dot3_RGBA: {
                // El resultado del producto escalar se repite en los tres
                // canales (vec3(x) en GLSL). En Cg un escalar no tiene swizzle,
                // asi que se repite la expresion.
                const std::string dot =
                    "dot(color_results_1 - float3(0.5, 0.5, 0.5), color_results_2 - "
                    "float3(0.5, 0.5, 0.5)) * 4.0";
                return fmt::format("float3({0}, {0}, {0})", dot);
            }
            default:
                LOG_CRITICAL(Render, "GXM FS: combinador de color TEV desconocido {}", operation);
                return std::string{"float3(0.0, 0.0, 0.0)"};
            }
        };
        out += fmt::format("clamp({}, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0))",
                           get_combiner());
    }

    void AppendAlphaCombiner(Operation operation) {
        const auto get_combiner = [operation] {
            switch (operation) {
            case Operation::Replace:
                return std::string{"alpha_results_1"};
            case Operation::Modulate:
                return std::string{"(alpha_results_1 * alpha_results_2)"};
            case Operation::Add:
                return std::string{"(alpha_results_1 + alpha_results_2)"};
            case Operation::AddSigned:
                return std::string{"(alpha_results_1 + alpha_results_2 - 0.5)"};
            case Operation::Lerp:
                return std::string{"lerp(alpha_results_2, alpha_results_1, alpha_results_3)"};
            case Operation::Subtract:
                return std::string{"(alpha_results_1 - alpha_results_2)"};
            case Operation::MultiplyThenAdd:
                return std::string{"(alpha_results_1 * alpha_results_2 + alpha_results_3)"};
            case Operation::AddThenMultiply:
                return std::string{"(min(alpha_results_1 + alpha_results_2, 1.0) * alpha_results_3)"};
            default:
                LOG_CRITICAL(Render, "GXM FS: combinador de alfa TEV desconocido {}", operation);
                return std::string{"0.0"};
            }
        };
        out += fmt::format("clamp({}, 0.0, 1.0)", get_combiner());
    }

    void WriteAlphaTestCondition(FramebufferRegs::CompareFunc func) {
        const auto get_cond = [func]() -> std::string {
            switch (func) {
            case FramebufferRegs::CompareFunc::Never:
                return "true";
            case FramebufferRegs::CompareFunc::Always:
                return "false";
            case FramebufferRegs::CompareFunc::Equal:
            case FramebufferRegs::CompareFunc::NotEqual:
            case FramebufferRegs::CompareFunc::LessThan:
            case FramebufferRegs::CompareFunc::LessThanOrEqual:
            case FramebufferRegs::CompareFunc::GreaterThan:
            case FramebufferRegs::CompareFunc::GreaterThanOrEqual: {
                // El orden de la tabla es el del enum de la PICA a partir de
                // Equal, igual que en el generador de GLSL.
                static constexpr std::array<const char*, 6> ops = {"!=", "==", ">=",
                                                                    ">",  "<=", "<"};
                const auto index = static_cast<u32>(func) -
                                   static_cast<u32>(FramebufferRegs::CompareFunc::Equal);
                /**
                 * EL +0.5 NO SOBRA: sin el, la prueba de alfa falla por uno.
                 *
                 * El rasterizador de software compara ENTEROS de ocho bits: el
                 * alfa del combinador contra el valor de referencia. Aqui el
                 * alfa es un float que vale aproximadamente k/255, y volver a
                 * multiplicarlo por 255 no da k exacto: da 126.99998 donde
                 * tocaba 127. Como int() trunca hacia cero, ese pixel se compara
                 * como 126 y la prueba sale al reves en el borde.
                 *
                 * Sumar medio antes de truncar es redondear al entero mas
                 * cercano, que es el k original. Se nota justo donde se usa la
                 * prueba de alfa: en los bordes de texto e iconos, que es donde
                 * el alfa cae cerca del valor de referencia.
                 */
                return fmt::format("int(combiner_output.a * 255.0 + 0.5) {} int(alphatest_ref)",
                                   ops[index]);
            }
            default:
                LOG_CRITICAL(Render, "GXM FS: prueba de alfa desconocida {}", func);
                return "false";
            }
        };
        out += fmt::format("    if ({}) discard;\n", get_cond());
    }

    /**
     * El modelo de iluminacion por fragmento de la PICA200, entero.
     *
     * QUE ES ESTO Y DE DONDE SALE. La PICA no ilumina por vertice: calcula por
     * PIXEL la difusa y dos especulares de hasta ocho luces, y el resultado
     * entra en el combinador TEV como dos fuentes mas (PrimaryFragmentColor y
     * SecondaryFragmentColor). Todo lo que no es aritmetica directa sale de
     * tablas de busqueda de 256 entradas que el juego rellena: distribucion
     * especular, fresnel, reflectancia por canal, atenuacion por distancia y
     * por foco. Esta funcion es la traduccion a Cg de glsl_fs_shader_gen.cpp,
     * que hace lo mismo para OpenGL; la verdad de lo que tiene que salir esta
     * en sw_lighting.cpp, que es la version por software.
     *
     * UNA DIFERENCIA A PROPOSITO CON EL GENERADOR DE GLSL. Alli, al mirar
     * two_sided_diffuse para decidir si el indice de una tabla va en valor
     * absoluto, se indexa lights[] por el NUMERO de la luz; aqui se indexa por
     * el HUECO. lights[] se rellena por hueco (pica_fs_config.cpp:
     * lights[light_index] con los registros de light[num]), asi que la version
     * de GLSL lee el hueco equivocado en cuanto un juego enciende las luces
     * desordenadas. Con num == hueco -- el caso normal -- dan lo mismo; cuando
     * no, la buena es esta, que es la que coincide con sw_lighting.cpp.
     *
     * LO QUE SE QUEDA FUERA. El pase de sombras de la iluminacion (shadow), que
     * IsSupported rechaza antes de llegar aqui. En su sitio va un uno fijo, que
     * es lo que vale cuando no hay sombra.
     */
    void WriteLighting() {
        const auto& lighting = config.lighting;

        out += "    float4 diffuse_sum = float4(0.0, 0.0, 0.0, 1.0);\n";
        out += "    float4 specular_sum = float4(0.0, 0.0, 0.0, 1.0);\n";
        out += "    float3 light_vector = float3(0.0, 0.0, 0.0);\n";
        out += "    float light_distance = 0.0;\n";
        out += "    float3 refl_value = float3(0.0, 0.0, 0.0);\n";
        out += "    float3 spot_dir = float3(0.0, 0.0, 0.0);\n";
        out += "    float3 half_vector = float3(0.0, 0.0, 0.0);\n";
        out += "    float dot_product = 0.0;\n";
        out += "    float clamp_highlights = 1.0;\n";
        out += "    float geo_factor = 1.0;\n";

        WriteSurfaceNormal();

        // El cuaternion interpolado NO viene normalizado: interpolar cuatro
        // componentes por separado no conserva la norma.
        out += "    float4 normalized_normquat = normalize(normquat);\n";
        out += "    float3 normal = quaternion_rotate(normalized_normquat, surface_normal);\n";
        out += "    float3 tangent = quaternion_rotate(normalized_normquat, surface_tangent);\n";

        for (u32 slot = 0; slot < lighting.src_num; slot++) {
            WriteLight(slot);
        }

        out += "    diffuse_sum.rgb += lighting_global_ambient.rgb;\n";
        out += "    primary_fragment_color = clamp(diffuse_sum, float4(0.0, 0.0, 0.0, 0.0), "
               "float4(1.0, 1.0, 1.0, 1.0));\n";
        out += "    secondary_fragment_color = clamp(specular_sum, float4(0.0, 0.0, 0.0, 0.0), "
               "float4(1.0, 1.0, 1.0, 1.0));\n";
    }

    /**
     * La normal y la tangente de la superficie ANTES de girarlas al espacio del
     * ojo, que es donde entra el bump mapping.
     *
     * Sin bump son las dos constantes (Z y X). Con mapa de normales, la normal
     * sale de la textura -- que guarda el vector en [0,1] y hay que devolver a
     * [-1,1] -- y con mapa de tangentes es la tangente la que sale de ahi. El
     * recalculo de Z (renorm) solo aplica a la normal: para la tangente, la
     * cuenta de mas abajo no cambia, y asi lo tiene confirmado el hardware.
     */
    void WriteSurfaceNormal() {
        const auto& lighting = config.lighting;
        const std::string perturbation =
            fmt::format("(2.0 * texcolor{}.rgb - 1.0)", lighting.bump_selector.Value());
        switch (lighting.bump_mode) {
        case LightingRegs::LightingBumpMode::NormalMap:
            out += fmt::format("    float3 surface_normal = {};\n", perturbation);
            if (lighting.bump_renorm) {
                out += "    surface_normal.z = sqrt(max(1.0 - (surface_normal.x * "
                       "surface_normal.x + surface_normal.y * surface_normal.y), 0.0));\n";
            }
            out += "    float3 surface_tangent = float3(1.0, 0.0, 0.0);\n";
            break;
        case LightingRegs::LightingBumpMode::TangentMap:
            out += fmt::format("    float3 surface_tangent = {};\n", perturbation);
            out += "    float3 surface_normal = float3(0.0, 0.0, 1.0);\n";
            break;
        default:
            out += "    float3 surface_normal = float3(0.0, 0.0, 1.0);\n";
            out += "    float3 surface_tangent = float3(1.0, 0.0, 0.0);\n";
            break;
        }
    }

    /**
     * La expresion que lee una tabla de busqueda de iluminacion.
     *
     * El INDICE es un coseno entre dos vectores, y cual de ellos depende de la
     * configuracion de la tabla (NH, VH, NV, LN, SP, CP). CP solo existe en la
     * configuracion 7 de la PICA; en las demas el hardware da cero, asi que
     * aqui tambien.
     *
     * EL SIGNO DECIDE COMO SE DIRECCIONA. Con abs_input, el indice se lleva al
     * tramo [0, 1] -- con valor absoluto si la luz es de dos caras, y
     * recortando a cero si no -- y se lee la tabla de 256 entradas entera. Sin
     * el, el indice va en [-1, 1] y se lee con el direccionamiento con signo.
     */
    std::string GetLutValue(LightingRegs::LightingSampler sampler, u32 slot,
                            LightingRegs::LightingLutInput input, bool abs_input) {
        const auto& lighting = config.lighting;
        std::string index;
        switch (input) {
        case LightingRegs::LightingLutInput::NH:
            index = "dot(normal, normalize(half_vector))";
            break;
        case LightingRegs::LightingLutInput::VH:
            index = "dot(normalize(view), normalize(half_vector))";
            break;
        case LightingRegs::LightingLutInput::NV:
            index = "dot(normal, normalize(view))";
            break;
        case LightingRegs::LightingLutInput::LN:
            index = "dot(light_vector, normal)";
            break;
        case LightingRegs::LightingLutInput::SP:
            index = "dot(light_vector, spot_dir)";
            break;
        case LightingRegs::LightingLutInput::CP:
            if (lighting.config == LightingRegs::LightingConfig::Config7) {
                // La proyeccion del vector de medio angulo sobre el plano de la
                // normal. Va contra la normal MODIFICADA por el mapa de
                // normales aunque ya no sea la del plano tangente, y no se
                // normaliza antes del producto escalar: las dos cosas estan
                // confirmadas en hardware y por eso el resultado no es
                // realmente el coseno que sugiere el nombre.
                index = "dot(normalize(half_vector) - normal * dot(normal, "
                        "normalize(half_vector)), tangent)";
            } else {
                index = "0.0";
            }
            break;
        default:
            index = "0.0";
            break;
        }

        const auto row = static_cast<u32>(sampler);
        if (abs_input) {
            index = lighting.lights[slot].two_sided_diffuse != 0
                        ? fmt::format("abs({})", index)
                        : fmt::format("max({}, 0.0)", index);
            return fmt::format("LookupLightingLUTUnsigned({}.0, {})", row, index);
        }
        return fmt::format("LookupLightingLUTSigned({}.0, {})", row, index);
    }

    /// Una luz del hueco 'slot'. Acumula en diffuse_sum y specular_sum.
    void WriteLight(u32 slot) {
        const auto& lighting = config.lighting;
        const auto& light = lighting.lights[slot];
        // El hueco dice QUE luz es; el numero dice DONDE estan sus registros.
        const u32 num = light.num.Value();

        // Direccional: la posicion ES la direccion, ya en espacio del ojo.
        // Posicional: hay que restarle la posicion del fragmento, y 'view' es
        // justo el vector del fragmento al ojo, o sea menos esa posicion.
        if (light.directional) {
            out += fmt::format("    light_vector = light_position[{}].xyz;\n", num);
        } else {
            out += fmt::format("    light_vector = light_position[{}].xyz + view;\n", num);
        }
        out += "    light_distance = length(light_vector);\n";
        out += "    light_vector = normalize(light_vector);\n";
        out += fmt::format("    spot_dir = light_spot_direction[{}].xyz;\n", num);
        out += "    half_vector = normalize(view) + light_vector;\n";

        out += light.two_sided_diffuse
                   ? "    dot_product = abs(dot(light_vector, normal));\n"
                   : "    dot_product = max(dot(light_vector, normal), 0.0);\n";

        if (lighting.clamp_highlights) {
            // Sin difusa no hay brillo: el signo vale 0 o 1 y apaga la
            // especular entera cuando la luz esta por detras.
            out += "    clamp_highlights = sign(dot_product);\n";
        }

        std::string spot_atten = "1.0";
        if (light.spot_atten_enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::SpotlightAttenuation)) {
            const std::string value =
                GetLutValue(LightingRegs::SpotlightAttenuationSampler(num), slot,
                            lighting.lut_sp.type, lighting.lut_sp.abs_input != 0);
            spot_atten = fmt::format("({:#} * {})", lighting.lut_sp.GetScale(), value);
        }

        std::string dist_atten = "1.0";
        if (light.dist_atten_enable) {
            // x es el sesgo y y la escala (ver como las sube el rasterizador).
            const std::string index =
                fmt::format("clamp(light_dist_atten[{0}].y * light_distance + "
                            "light_dist_atten[{0}].x, 0.0, 1.0)",
                            num);
            dist_atten = fmt::format(
                "LookupLightingLUTUnsigned({}.0, {})",
                static_cast<u32>(LightingRegs::DistanceAttenuationSampler(num)), index);
        }

        if (light.geometric_factor_0 || light.geometric_factor_1) {
            out += "    geo_factor = dot(half_vector, half_vector);\n";
            out += "    geo_factor = geo_factor == 0.0 ? 0.0 : min(dot_product / geo_factor, "
                   "1.0);\n";
        }

        std::string d0_lut_value = "1.0";
        if (lighting.lut_d0.enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::Distribution0)) {
            const std::string value =
                GetLutValue(LightingRegs::LightingSampler::Distribution0, slot,
                            lighting.lut_d0.type, lighting.lut_d0.abs_input != 0);
            d0_lut_value = fmt::format("({:#} * {})", lighting.lut_d0.GetScale(), value);
        }
        std::string specular_0 =
            fmt::format("({} * light_specular_0[{}].rgb)", d0_lut_value, num);
        if (light.geometric_factor_0) {
            specular_0 = fmt::format("({} * geo_factor)", specular_0);
        }

        WriteReflectance(slot);

        std::string d1_lut_value = "1.0";
        if (lighting.lut_d1.enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::Distribution1)) {
            const std::string value =
                GetLutValue(LightingRegs::LightingSampler::Distribution1, slot,
                            lighting.lut_d1.type, lighting.lut_d1.abs_input != 0);
            d1_lut_value = fmt::format("({:#} * {})", lighting.lut_d1.GetScale(), value);
        }
        std::string specular_1 = fmt::format("({} * refl_value * light_specular_1[{}].rgb)",
                                             d1_lut_value, num);
        if (light.geometric_factor_1) {
            specular_1 = fmt::format("({} * geo_factor)", specular_1);
        }

        // Fresnel: SOLO la ultima luz del reparto lo aplica, y va al alfa.
        if (slot + 1 == lighting.src_num && lighting.lut_fr.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::Fresnel)) {
            const std::string lut = GetLutValue(LightingRegs::LightingSampler::Fresnel, slot,
                                                lighting.lut_fr.type,
                                                lighting.lut_fr.abs_input != 0);
            const std::string value =
                fmt::format("({:#} * {})", lighting.lut_fr.GetScale(), lut);
            if (lighting.enable_primary_alpha) {
                out += fmt::format("    diffuse_sum.a = {};\n", value);
            }
            if (lighting.enable_secondary_alpha) {
                out += fmt::format("    specular_sum.a = {};\n", value);
            }
        }

        out += fmt::format("    diffuse_sum.rgb += ((light_diffuse[{0}].rgb * dot_product) + "
                           "light_ambient[{0}].rgb) * {1} * {2};\n",
                           num, dist_atten, spot_atten);
        out += fmt::format("    specular_sum.rgb += ({} + {}) * clamp_highlights * {} * {};\n",
                           specular_0, specular_1, dist_atten, spot_atten);
    }

    /**
     * Las tres reflectancias de la especular 1.
     *
     * Rojo se lee de su tabla si la hay, y si no vale uno. Verde y azul se leen
     * de la suya, y si no la hay COPIAN EL ROJO -- no valen uno. Es una rareza
     * del hardware, pero es lo que hace el rasterizador de software.
     */
    void WriteReflectance(u32 slot) {
        const auto& lighting = config.lighting;
        const auto write = [&](const char* channel, const LutConfig& lut,
                               LightingRegs::LightingSampler sampler, const char* fallback) {
            if (lut.enable && LightingRegs::IsLightingSamplerSupported(lighting.config, sampler)) {
                const std::string value =
                    GetLutValue(sampler, slot, lut.type, lut.abs_input != 0);
                out += fmt::format("    refl_value.{} = ({:#} * {});\n", channel, lut.GetScale(),
                                   value);
            } else {
                out += fmt::format("    refl_value.{} = {};\n", channel, fallback);
            }
        };
        write("r", lighting.lut_rr, LightingRegs::LightingSampler::ReflectRed, "1.0");
        write("g", lighting.lut_rg, LightingRegs::LightingSampler::ReflectGreen, "refl_value.r");
        write("b", lighting.lut_rb, LightingRegs::LightingSampler::ReflectBlue, "refl_value.r");
    }

    void WriteTevStage(u32 index) {
        const TexturingRegs::TevStageConfig stage = config.texture.tev_stages[index];

        out += "    color_results_1 = ";
        AppendColorModifier(stage.color_modifier1, stage.color_source1, index);
        out += ";\n    color_results_2 = ";
        AppendColorModifier(stage.color_modifier2, stage.color_source2, index);
        out += ";\n    color_results_3 = ";
        AppendColorModifier(stage.color_modifier3, stage.color_source3, index);

        out += fmt::format(";\n    float3 color_output_{} = byteround(float4(", index);
        AppendColorCombiner(stage.color_op);
        out += ", 1.0)).rgb;\n";

        if (stage.color_op == Operation::Dot3_RGBA) {
            out += fmt::format("    float alpha_output_{0} = color_output_{0}.r;\n", index);
        } else {
            out += "    alpha_results_1 = ";
            AppendAlphaModifier(stage.alpha_modifier1, stage.alpha_source1, index);
            out += ";\n    alpha_results_2 = ";
            AppendAlphaModifier(stage.alpha_modifier2, stage.alpha_source2, index);
            out += ";\n    alpha_results_3 = ";
            AppendAlphaModifier(stage.alpha_modifier3, stage.alpha_source3, index);
            out += fmt::format(";\n    float alpha_output_{} = byteround(float4(", index);
            AppendAlphaCombiner(stage.alpha_op);
            out += ", 0.0, 0.0, 0.0)).r;\n";
        }

        out += fmt::format(
            "    combiner_output = float4(clamp(color_output_{0} * {1:.1f}, float3(0.0, 0.0, "
            "0.0), float3(1.0, 1.0, 1.0)), clamp(alpha_output_{0} * {2:.1f}, 0.0, 1.0));\n",
            index, static_cast<f32>(stage.GetColorMultiplier()),
            static_cast<f32>(stage.GetAlphaMultiplier()));

        out += "    combiner_buffer = next_combiner_buffer;\n";
        if (config.TevStageUpdatesCombinerBufferColor(index)) {
            out += "    next_combiner_buffer.rgb = combiner_output.rgb;\n";
        }
        if (config.TevStageUpdatesCombinerBufferAlpha(index)) {
            out += "    next_combiner_buffer.a = combiner_output.a;\n";
        }
    }

    const FSConfig& config;
    const char** reason = nullptr;
    bool alpha_from_blend_const = false;
    bool color_from_blend_const = false;
    std::string out;
    bool uses_tex0 = false;
    bool uses_tex1 = false;
    bool uses_tex2 = false;
    bool uses_const_color = false;
    bool uses_combiner_buffer_color = false;
};

} // Anonymous namespace

std::optional<std::string> GenerateFragmentShader(const FSConfig& config,
                                                  const char** out_reason,
                                                  bool alpha_from_blend_const,
                                                  bool color_from_blend_const) {
    FragmentWriter writer{config, out_reason, alpha_from_blend_const, color_from_blend_const};
    return writer.Generate();
}

} // namespace Pica::Shader::Generator::GXM
