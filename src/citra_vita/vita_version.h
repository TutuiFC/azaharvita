// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

namespace VitaFrontend {

/**
 * Version del port. UNICO sitio donde se escribe.
 *
 * Estaba en dos: main.cpp la pintaba en el menu y vita_window.cpp la repetia a
 * mano en el overlay del juego. Eso hace inutil justo aquello para lo que el
 * overlay lleva la version -- confirmar de un vistazo QUE build esta corriendo
 * --, porque en cuanto una de las dos se olvida, la pantalla miente.
 *
 * build.sh tambien lee este fichero para archivar una copia del ELF con la
 * version en el nombre (elf/azahar-X.Y.Z.W.elf); sin el binario exacto que
 * corrio en la consola no se puede traducir un volcado de crash. Si cambia el
 * formato de la linea de abajo, hay que cambiar el grep de build.sh.
 *
 * NUMERACION: se reinicio en 0.0.1.0 al empezar el trabajo de rendimiento. Los
 * ELF de la serie anterior (0.0.2.5 a 0.0.3.9) siguen en elf/ y no se pueden
 * pisar, asi que al subir el numero hay que saltarse los que ya existan alli.
 *
 * 0.1.0.0: el usuario verifico en consola el cierre de la Fase 2 (presentacion
 * por GXM identica al camino de software) y se abre la Fase 3 con el generador
 * de shaders Cg.
 */
constexpr char kVersion[] = "version 0.1.0.22";

/// La misma version con el nombre delante, para el overlay ("Azahar 0.1.0.22").
constexpr char kOverlayBuild[] = "Azahar 0.1.0.22";

} // namespace VitaFrontend
