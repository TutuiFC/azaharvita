// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <psp2/shacccg.h>
#include "common/common_types.h"

namespace Gxm {

/**
 * El compilador de shaders Cg (libshacccg.suprx), COMPARTIDO por todo el
 * backend.
 *
 * El modulo es estado global del proceso: el presentador y el rasterizador lo
 * necesitan a la vez, y cargarlo por separado hacia que el segundo intento
 * fallara (y que reintentarlo en cada lote convirtiera la caida de rendimiento
 * en una tormenta de llamadas al sistema: 13,7 s de fotograma medidos en
 * consola). Aqui se carga una vez, se reutiliza el handle y, si el primer
 * intento falla, se reintenta como mucho una vez por segundo.
 */
bool EnsureCgReady();

/// Estado corto para el overlay y crash.txt.
const char* CgStatus();

/**
 * True si el compilador de Cg ha devuelto un "internal error" en esta sesion.
 *
 * En 0.1.4.7 un shader de vertices traducido hizo que sceShaccCg respondiera
 * "fatal internal error", y a partir de ahi fallaron TODAS las compilaciones,
 * tambien las de fragmentos: la partida entera se fue a software. Si vuelve a
 * pasar, la ruta del shader de vertices en la GPU deja de pedirle shaders al
 * compilador (HwShaderCache), para no gastar lo que le quede en algo que ya
 * se sabe que lo rompe.
 */
bool CgPoisoned();

/**
 * Queda poco heap (0.1.7.9): menos de kCgHeapReserveMB libres. Lo opcional
 * que cuesta compilar -- los programas especializados por booleanos -- se deja
 * de pedir, para que una escena con muchos shaders nuevos no acabe la partida
 * con bad_alloc. El lote se queda en la CPU, que es correcto.
 */
constexpr unsigned int kCgHeapReserveMB = 32;
bool CgHeapLow();

/**
 * Cuantas veces se ha descargado y recargado el compilador tras un error
 * interno (0.1.5.9). Un shader que fallo por el compilador roto se puede
 * reintentar cuando esto cambia: el fallo no era del shader.
 */
u32 CgGeneration();

/// Compila un shader. El resultado vive hasta ReleaseCgOutput.
const SceShaccCgCompileOutput* CompileCg(SceShaccCgTargetProfile profile, const char* name,
                                         const char* source);

void ReleaseCgOutput(const SceShaccCgCompileOutput* output);

} // namespace Gxm
