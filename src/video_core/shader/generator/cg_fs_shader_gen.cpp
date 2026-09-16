// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/shader/generator/cg_fs_shader_gen.h"

#include <array>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica/regs_texturing.h"

namespace Pica::Shader::Generator::GXM {

namespace {

using FramebufferRegs = Pica::FramebufferRegs;
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
    // Iluminacion: exige los LUTs de la PICA, el vector normal y el ojo, y esta
    // fuera del primer tramo de la fase.
    if (config.lighting.enable || config.lighting.enable_shadow) {
        return reject("iluminacion");
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
    // Modos de textura: solo 2D y "apagada". Proyeccion, cubo y sombra llevan
    // otra matematica de coordenadas.
    const auto texture0_type = config.texture.texture0_type.Value();
    if (texture0_type != TexturingRegs::TextureConfig::Texture2D &&
        texture0_type != TexturingRegs::TextureConfig::Disabled) {
        return reject("tipo de textura 0");
    }
    if (config.texture.texture2_use_coord1) {
        return reject("textura 2 con coord 1");
    }
    return true;
}

/// Emite el Cg de una configuracion, o deja el motivo del rechazo en el LOG.
class FragmentWriter {
public:
    explicit FragmentWriter(const FSConfig& config_, const char** reason_)
        : config{config_}, reason{reason_} {}

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
            out += "void main(float4 primary_color : COLOR0, float2 tc0 : TEXCOORD0, "
                   "float2 tc1 : TEXCOORD1, float2 tc2 : TEXCOORD2, "
                   "out float4 gl_FragColor : COLOR)\n{\n    discard;\n}\n";
            return out;
        }

        WriteInterface();
        WriteUniforms();

        // gl_FragCoord (WPOS) da la profundidad de ventana, que con nuestro
        // viewport ya es exactamente la de la PICA; la necesita la niebla.
        out += "void main(float4 primary_color : COLOR0, float2 tc0 : TEXCOORD0, "
               "float2 tc1 : TEXCOORD1, float2 tc2 : TEXCOORD2, "
               "float4 gl_FragCoord : WPOS, "
               "out float4 gl_FragColor : COLOR)\n{\n";

        // La PICA redondea el color primario a 8 bits antes de meterlo en la
        // primera etapa TEV; sin esto el resultado no coincide con el de
        // software.
        out += "    float4 rounded_primary_color = byteround(primary_color);\n";
        out += "    float4 primary_fragment_color = rounded_primary_color;\n";
        out += "    float4 secondary_fragment_color = float4(0.0, 0.0, 0.0, 0.0);\n";
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
    }

    /// Ojea las etapas TEV para saber que samplers hay que declarar.
    void ScanSources() {
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
        if (uses_combiner_buffer_color) {
            out += "uniform float4 tev_combiner_buffer_color;\n";
        }
        if (config.framebuffer.alpha_test_func != FramebufferRegs::CompareFunc::Always) {
            out += "uniform float alphatest_ref;\n";
        }
        if (config.texture.fog_mode == TexturingRegs::FogMode::Fog) {
            out += "uniform float fog_lut[256];\n";
            out += "uniform float3 fog_color;\n";
        }
        out += "\n";
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
            return "tex2D(tex0, tc0)";
        case Source::Texture1:
            return "tex2D(tex1, tc1)";
        case Source::Texture2:
            return "tex2D(tex2, tc2)";
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
                return fmt::format("int(combiner_output.a * 255.0) {} int(alphatest_ref)",
                                   ops[index]);
            }
            default:
                LOG_CRITICAL(Render, "GXM FS: prueba de alfa desconocida {}", func);
                return "false";
            }
        };
        out += fmt::format("    if ({}) discard;\n", get_cond());
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
    std::string out;
    bool uses_tex0 = false;
    bool uses_tex1 = false;
    bool uses_tex2 = false;
    bool uses_const_color = false;
    bool uses_combiner_buffer_color = false;
};

} // Anonymous namespace

std::optional<std::string> GenerateFragmentShader(const FSConfig& config,
                                                  const char** out_reason) {
    FragmentWriter writer{config, out_reason};
    return writer.Generate();
}

} // namespace Pica::Shader::Generator::GXM
