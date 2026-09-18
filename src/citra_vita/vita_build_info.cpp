// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "citra_vita/vita_version.h"

/**
 * Sello de compilacion (fecha + hash de git), generado por build.sh.
 *
 * Es un fichero de una sola linea a proposito: se regenera en CADA
 * compilacion, asi que este .cpp se recompila y el binario se reenlaza, pero
 * ningun otro fichero se ve afectado. Asi el menu, el overlay y crash.txt
 * dicen siempre que build exacto esta corriendo, aunque la version de
 * vita_version.h no se haya tocado.
 *
 * Si se compila sin build.sh (por ejemplo desde un IDE), el .inc no existe y
 * se cae a un texto que lo dice.
 */
#if __has_include("citra_vita/vita_build_info.inc")
#include "citra_vita/vita_build_info.inc"
#else
namespace VitaFrontend {
const char kBuildInfo[] = "build sin sello";
}
#endif
