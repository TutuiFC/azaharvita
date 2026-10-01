// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <optional>
#include <string>
#include "video_core/shader/generator/shader_gen.h"

namespace Pica {
struct ShaderSetup;
}

namespace Pica::Shader::Generator::GXM {

/**
 * Saltos que salen de su tramo, traducidos con variable de escape (0.1.5.8).
 * APAGADO por defecto desde 0.1.5.9: en 0.1.5.8 uno de esos shaders dio
 * "fatal internal error" en el compilador de Cg. Se enciende desde el menu de
 * ajustes para PROBAR: si vuelve a romper el compilador, gxm_cg.cpp se
 * recupera solo y deja el shader en ux0:/data/azahar/cg_error.txt.
 */
extern std::atomic<bool> g_allow_vs_escapes;

/**
 * Variantes de traduccion para el compilador de Cg de la consola (0.1.7.3).
 * cg_error.txt de 0.1.7.1: el shader de piel de Rubi Omega (matrices de huesos
 * con indexado de uniforms) da "fatal internal error". Lo que solo ese shader
 * tiene son enteros: la funcion get_offset_register con "& 0x7F" y las
 * conversiones int2(float2) de MOVA. Bits:
 *   kCgFloatAddress: registros de direccion y calculo del indice en float,
 *                    sin enteros ni operaciones de bits.
 *   kCgFlat:         todas las subrutinas en linea dentro de exec_shader.
 * Lo elige HwShaderCache en rasterizer_gxm.cpp, probando en orden si el
 * compilador se rompe (y se recupera entre medias, ver gxm_cg.cpp).
 */
constexpr u32 kCgFloatAddress = 1;
constexpr u32 kCgFlat = 2;
extern std::atomic<u32> g_cg_variant;

/**
 * Uniforms booleanos como CONSTANTES (0.1.7.4). Con kCgBoolsKnown puesto, los
 * bits 0-15 son los valores de vs_b en el lote que se va a dibujar: JMPU, IFU
 * y CALLU se resuelven al traducir y solo queda el camino que se toma. Lo usa
 * el rasterizador cuando la traduccion generica de un programa no compila; el
 * programa especializado va con los booleanos en su clave.
 */
constexpr u32 kCgBoolsKnown = 1u << 16;
extern std::atomic<u32> g_cg_const_bools;

/// Mascara de los uniforms booleanos que lee algun JMPU/IFU/CALLU del codigo
/// de vertices: solo esos entran en la clave del programa especializado.
[[nodiscard]] u32 UsedBoolUniforms(const Pica::ShaderSetup& setup);

/**
 * Traductor del shader de VERTICES de la PICA200 a Cg para el backend GXM.
 *
 * QUE PROBLEMA RESUELVE. Hoy el shader de vertices del juego lo ejecuta el
 * INTERPRETE en la CPU, una instruccion a la vez y un vertice a la vez
 * (pica_core.cpp, shader_engine->Run). En una consola sin JIT -- la Vita es
 * ARMv7 y dynarmic no tiene backend para ella -- ese interprete es de lo mas
 * caro que hay en el fotograma, y encima corre en el mismo nucleo que ya esta
 * saturado emulando el ARM11. Esto traduce ese mismo programa a Cg para que lo
 * ejecute la GPU, que ahora mismo esta ociosa esperando.
 *
 * DE DONDE SALE. Es el hermano de glsl_shader_decompiler.cpp, que hace
 * exactamente este trabajo para OpenGL: el mismo analisis de flujo, las mismas
 * subrutinas y las mismas instrucciones, con la sintaxis de Cg en vez de la de
 * GLSL. Donde las dos se separan esta comentado en el sitio.
 *
 * QUE NO TRADUCE, Y POR QUE ESO ES LO CORRECTO. Devuelve std::nullopt -- y el
 * que llama se queda con el interprete, que es la referencia -- cuando:
 *
 *   - El programa usa SALTOS (JMPC/JMPU) que no sean "hacia delante y dentro
 *     del tramo". Los que si lo son se traducen como un if (0.1.4.8; ver
 *     ControlFlowAnalyzer::Scan en el .cpp). La maquina de estados del de
 *     GLSL se probo en 0.1.4.7 y rompio el compilador de Cg de la consola
 *     ("fatal internal error"), asi que los demas siguen en el interprete.
 *   - El analisis de flujo se rinde (recursion, un programa que no termina
 *     siempre) o aparece una instruccion sin traduccion.
 *   - Se pide multiplicacion exacta (sanitize_mul). Esa version necesita
 *     seleccionar por componentes con isnan, que en Cg no tiene una traduccion
 *     que se pueda dar por buena sin probarla.
 *   - Hay shader de geometria: ese sigue en la CPU siempre, y esta fuera de
 *     alcance.
 *
 * SOBRE LOS ERRORES DE COMPILACION. Si el Cg que sale de aqui usa algo que el
 * compilador de la consola no acepta, la compilacion falla y el lote vuelve al
 * interprete. Es decir: el modo de fallo de esta ruta es "no acelera", no
 * "dibuja mal". Por eso se prefiere emitir y que el compilador decida antes que
 * adivinar limites del perfil desde aqui.
 *
 * EL CONTRATO CON EL SHADER DE FRAGMENTOS. La salida es la misma que la del
 * shader de vertices fijo largo del rasterizador (kVertexSourceLit):
 *
 *   float4 gl_Position : POSITION     posicion de recorte de la PICA, tal cual
 *   float4 out_color   : COLOR0
 *   float2 out_tc0/1/2 : TEXCOORD0/1/2
 *   float4 out_normquat: TEXCOORD3
 *   float3 out_view    : TEXCOORD4
 *
 * Se emiten SIEMPRE los siete, tambien cuando el shader de fragmentos que vaya
 * detras no lea normales ni vista. Cuesta interpolar dos varyings de mas en
 * esos lotes, y a cambio hay un solo programa de vertices por programa de la
 * PICA en vez de dos, y su lista de salidas es identica a la del fijo largo --
 * que es lo que permite emparejarlos sin recompilar el de fragmentos.
 *
 * ---------------------------------------------------------------------------
 * ESTADO (0.1.4.6): CONECTADO. Lo llama RasterizerGXM::AccelerateDrawBatch, que
 * resuelve lo que aqui faltaba: los cargadores de atributos de la PICA pasan a
 * flujos de vertices de GXM (uno por cargador; GXM admite 16), los formatos
 * byte/ubyte/short/float van sin normalizar, los indices de 8 bits se pasan a
 * 16, los uniforms (96 float4, 4 enteros y 16 booleanos, mas los atributos por
 * defecto en vs_default) se suben en cada lote, y el estado de dibujado se
 * comparte con DrawBatchOnGpu (SetupDrawState). De donde sale cada registro de
 * entrada lo dice VSInputs.
 *
 * LA ADVERTENCIA SIGUE EN PIE: cuando AccelerateDrawBatch devuelve true, PicaCore
 * se salta el camino de la CPU entero. Por eso esa funcion lo comprueba TODO
 * antes de tocar nada y ante cualquier duda devuelve false. Y para comparar las
 * dos rutas en la consola esta la ablacion "vs en cpu" (SELECT + ABAJO).
 */
/**
 * De donde sale cada REGISTRO DE ENTRADA del programa (v0..v15), igual que lo
 * rellena el interprete (VertexLoader::LoadVertex + ShaderUnit::LoadInput):
 *
 *   Array   de un flujo de vertices, con 'components' componentes (1-4). Las
 *           que faltan valen 0, 0, 0 y 1 (x, y, z, w), como en LoadVertex.
 *   Default el atributo por defecto (registro de la PICA), que va en el uniform
 *           vs_default[registro].
 *   None    nadie lo carga: ceros, que es lo que tiene la unidad del interprete
 *           en un registro que nunca se escribe.
 *
 * Entra en la clave del shader: el mismo programa con otra disposicion de
 * atributos es otro shader.
 */
struct VSInputSource {
    enum Kind : u8 {
        None = 0,
        Array = 1,
        Default = 2,
    };
    u8 kind = None;
    u8 components = 4;
};
using VSInputs = std::array<VSInputSource, 16>;

[[nodiscard]] std::optional<std::string> GenerateVertexShader(const Pica::ShaderSetup& setup,
                                                              const PicaVSConfig& config,
                                                              const ExtraVSConfig& extra,
                                                              const VSInputs& inputs,
                                                              bool write_lighting, bool write_w,
                                                              const char** out_reason = nullptr);

} // namespace Pica::Shader::Generator::GXM
