// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <string>
#include "video_core/shader/generator/pica_fs_config.h"

namespace Pica::Shader::Generator::GXM {

/**
 * Generador del shader de fragmentos de la PICA en Cg para el backend GXM.
 *
 * Es el tercer hermano de glsl_fs_shader_gen.cpp y spv_fs_shader_gen.cpp:
 * consume el MISMO PicaFSConfig (que ya es agnostico de API) y produce fuente
 * Cg en vez de GLSL o SPIR-V. La semantica la manda el de GLSL, que es la
 * referencia: cada etapa TEV, sus fuentes, modificadores y combinadores, el
 * redondeo a 8 bits y la prueba de alfa se portan de alli tal cual.
 *
 * QUE CUBRE HOY.
 *
 *   - Las seis etapas TEV completas: fuentes (color primario, texturas 0-2,
 *     color constante, buffer del combinador y resultado anterior),
 *     modificadores de color y alfa, todos los combinadores y los
 *     multiplicadores de etapa.
 *   - Muestreo de texturas 0-2 en 2D. Las texturas las decodifica el cache de
 *     texturas del backend a RGBA8, asi que el shader muestrea siempre el
 *     mismo formato y el filtrado/repetido viaja en el objeto de textura GXM.
 *   - La prueba de alfa (discard) con su valor de referencia por uniform.
 *   - El redondeo a 8 bits de la PICA entre etapas y a la salida.
 *
 * QUE NO CUBRE TODAVIA (devuelve std::nullopt y el que llama cae al
 * rasterizador por software, que sigue siendo la verdad).
 *
 *   - Iluminacion por fragmento, niebla, texturas procedurales y shadow maps.
 *   - Scissor y W-buffering: los lleva el estado del contexto GXM (viewport y
 *     region), no el shader, y aun no estan mapeados. El scissor ademas
 *     necesita gl_FragCoord, que hay que confirmar en el Cg de GXM.
 *   - Texturas 3 y proyeccion/cubo, mipmaps y modos de borde con color.
 *
 * El contrato de la interfaz con el shader de vertices (que de momento pone el
 * backend, no un generador) es:
 *
 *   float4 primary_color : COLOR0     salida de color del VS de la PICA
 *   float2 tc0/tc1/tc2   : TEXCOORD0/1/2
 *
 * y los uniforms const_color[6], tev_combiner_buffer_color, alphatest_ref y
 * los samplers tex0-tex2.
 */
std::optional<std::string> GenerateFragmentShader(const FSConfig& config,
                                                  const char** out_reason = nullptr);

} // namespace Pica::Shader::Generator::GXM
