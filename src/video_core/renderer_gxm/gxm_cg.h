// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <psp2/shacccg.h>

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

/// Compila un shader. El resultado vive hasta ReleaseCgOutput.
const SceShaccCgCompileOutput* CompileCg(SceShaccCgTargetProfile profile, const char* name,
                                         const char* source);

void ReleaseCgOutput(const SceShaccCgCompileOutput* output);

} // namespace Gxm
