// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#ifdef __PSVITA__

namespace VitaFrontend {

/**
 * Comprueba si esta consola permite generar y ejecutar codigo en caliente.
 *
 * Es el requisito previo de cualquier recompilador (JIT). Deja el resultado
 * anotado en crash.txt, bajo la etiqueta "jit". Corre una vez al arrancar y no
 * cambia nada del emulador: solo informa.
 *
 * Ver el comentario largo en vita_jit_probe.cpp para el porque.
 */
void ProbeJitSupport();

} // namespace VitaFrontend

#endif // __PSVITA__
