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
 *   - COLOR DE BORDE (ClampToBorder). GXM no tiene donde escribir un color de
 *     borde, asi que lo hace el shader: compara la coordenada contra [0, 1] y
 *     devuelve tex_border_color[unidad] en vez del texel. Es el mismo camino
 *     que toma el generador de GLSL cuando el perfil no declara
 *     has_custom_border_color, y la bandera sale del mismo sitio.
 *   - La prueba de alfa (discard) con su valor de referencia por uniform.
 *   - El redondeo a 8 bits de la PICA entre etapas y a la salida.
 *   - Niebla, con su tabla de 128 entradas por uniform.
 *   - ILUMINACION POR FRAGMENTO COMPLETA (desde 0.1.0.17): hasta ocho luces,
 *     difusa, especular 0 y 1, luces direccionales y posicionales, difusa a dos
 *     caras, factores geometricos, recorte de brillos, atenuacion por distancia
 *     y por foco, las seis tablas de busqueda (D0, D1, SP, FR, RB, RG),
 *     fresnel y bump mapping por mapa de normales o de tangentes. Las tablas se
 *     leen de una textura de 256x24 que sube el rasterizador; los parametros de
 *     las luces, de uniforms. El shader pide entonces dos varyings mas
 *     (normquat y view) que solo escribe el shader de vertices largo.
 *
 * QUE NO CUBRE TODAVIA (devuelve std::nullopt y el que llama cae al
 * rasterizador por software, que sigue siendo la verdad).
 *
 *   - Texturas procedurales y shadow maps.
 *   - El PASE DE SOMBRAS de la iluminacion (lighting.enable_shadow), que
 *     mezcla una textura de sombra en la difusa y la especular.
 *   - Scissor y W-buffering: los lleva el estado del contexto GXM (viewport y
 *     region), no el shader, y aun no estan mapeados. El scissor ademas
 *     necesita gl_FragCoord, que hay que confirmar en el Cg de GXM.
 *   - Proyeccion, cubo y sombra en la unidad 0, y el modo de repetido
 *     ClampToBorder2 (el raro: borde por el lado positivo y repeticion por el
 *     negativo), que la configuracion compartida no marca.
 *   - Unidades de textura APAGADAS. El rasterizador de software da (0,0,0,0)
 *     para una unidad apagada que alguna etapa TEV lea; aqui se declara el
 *     sampler igual, asi que el cache de texturas rechaza el lote. Para
 *     arreglarlo hace falta que la configuracion del shader lleve que unidades
 *     estan encendidas, y hoy solo lleva la 0 (via texture0_type).
 *
 * El contrato de la interfaz con el shader de vertices (que de momento pone el
 * backend, no un generador) es:
 *
 *   float4 primary_color : COLOR0     salida de color del VS de la PICA
 *   float2 tc0/tc1/tc2   : TEXCOORD0/1/2
 *
 * y los uniforms const_color[6], tev_combiner_buffer_color, alphatest_ref,
 * fog_lut[256], fog_color, tex_border_color[3] y los samplers tex0-tex2. Cada
 * uno se declara SOLO si la configuracion lo usa; el que llama los busca por
 * nombre y se salta los que no existan.
 */
/// 'alpha_from_blend_const' (0.2.0.1): el alfa de salida es la constante de
/// mezcla (uniform blend_const_alpha) y no el calculado. Lo pide el
/// rasterizador cuando lleva en el alfa de la fuente un factor de mezcla
/// constante, que GXM no tiene (ver BuildBlend en rasterizer_gxm.cpp).
std::optional<std::string> GenerateFragmentShader(const FSConfig& config,
                                                  const char** out_reason = nullptr,
                                                  bool alpha_from_blend_const = false);

} // namespace Pica::Shader::Generator::GXM
