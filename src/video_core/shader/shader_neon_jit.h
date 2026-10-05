// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

/**
 * JIT DE SHADERS DE LA PICA200 A NEON (0.1.7.0, solo PS Vita).
 *
 * POR QUE. Con el shader de vertices en la CPU, sombrear se llevaba ~70 ms de
 * cada vblank de ~200 en la cinematica de Rubi Omega (0.1.6.3), y es la parte
 * mas grande del fotograma. La ruta rapida (shader_interpreter_fast.h) ya
 * pre-decodifica cada instruccion, pero sigue INTERPRETANDOLA: por cada
 * instruccion y cada vertice, un switch por tipo, ramas por cada fuente, un
 * swizzle con tabla aunque sea .xyzw y una mezcla al guardar.
 *
 * QUE HACE. Cada TRAMO de instrucciones aritmeticas (los mismos tramos que la
 * ruta rapida, con los mismos cortes) se convierte en UNA funcion nativa con
 * las instrucciones NEON exactas que ejecuta ExecuteOp, en el mismo orden:
 * mismo producto saneado, mismas sumas escalares en el mismo orden para los
 * DP, mismos NaN en MAX/MIN, mismo 1.0/+0 en SGE/SLT. Lo que no se traduce
 * (FLR, RCP, RSQ, EX2, LG2, MOVA, CMP y cualquier fuente con registro de
 * direccion) se hace llamando a la MISMA Fast::ExecuteOp desde el codigo
 * generado. El control de flujo sigue en el interprete.
 *
 * LA RED DE SEGURIDAD es la de la ruta rapida: InterpreterEngine::Run ejecuta
 * los primeros 256 vertices de cada programa, y despues uno de cada 512,
 * TAMBIEN con el interprete puro y compara bit a bit. Si difiere, el programa
 * entero vuelve al interprete y crash.txt dice donde.
 */

#ifdef __PSVITA__

#include <atomic>
#include <string>
#include "common/common_types.h"
#include "video_core/pica/shader_setup.h"

namespace Pica::Shader::Fast {

struct Program;

/// Encendido por defecto; 0 = la ruta rapida interpretada de 0.1.6.x.
extern std::atomic<bool> g_neon_jit;

/// Genera el codigo de todos los tramos del programa (Run::code). Si no hay
/// memoria ejecutable o no cabe, deja los tramos sin codigo: la ruta rapida
/// interpretada sigue funcionando igual.
void CompileRuns(Program& program);

/// El programa ENTERO en una funcion (0.1.7.2): tramos en linea y control de
/// flujo por FlowStep/FlowPostCheck. Si el programa tiene algo que la ruta
/// rapida no decodifica, se deja sin compilar entero (sigue por tramos).
void CompileWhole(Program& program, const ProgramCode& code);

/// Reserva ya la memoria ejecutable de este JIT (0.2.1.6). Se llama al
/// arrancar, ANTES que el JIT ARM: la consola no devuelve la que se suelta, y
/// el JIT ARM se queda con lo que pueda de lo que quede.
void ReserveCodeMemory();

/// Que instrucciones de flujo siguen yendo a FlowStep (0.2.1.3), en % del
/// total, para crash.txt; "-" si ninguna. Pone los contadores a cero.
std::string TakeFlowSummary();

} // namespace Pica::Shader::Fast

#endif // __PSVITA__
