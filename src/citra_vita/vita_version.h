// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string_view>

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
/*
 * NUMERACION DESDE 0.1.4.5 (peticion del usuario, 22/09/2026): cada cifra va
 * de 0 a 9 y al pasar de 9 se suma uno a la de su izquierda, como un
 * cuentakilometros. La que habria sido 0.1.0.45 es 0.1.4.5; despues de 0.1.4.9
 * viene 0.1.5.0. Las versiones anteriores (0.1.0.1 a 0.1.0.44) se quedan con
 * su nombre en elf/, y ninguna choca con las nuevas.
 */
constexpr char kVersion[] = "version 0.3.2.9";

/// La misma version con el nombre delante, para el overlay ("Azahar 0.3.2.9").
constexpr char kOverlayBuild[] = "Azahar 0.3.2.9";

/**
 * Y AHORA EL COMPILADOR VIGILA QUE LAS DOS DIGAN LO MISMO.
 *
 * Este fichero nacio para que la version viviera en UN solo sitio, porque
 * main.cpp y vita_window.cpp la repetian a mano y se desincronizaron. Dentro
 * del propio fichero se volvio a repetir, y volvio a pasar exactamente lo
 * mismo: el overlay estuvo anunciando 0.1.0.15 durante seis versiones, con una
 * captura de pantalla de por medio que costo una vuelta entera de diagnostico
 * ("¿pero que build es esa?").
 *
 * Las dos cadenas no se pueden fundir en una sola sin romper el grep de
 * build.sh, que es lo que archiva el ELF con la version en el nombre. Asi que
 * se quedan separadas y se comparan aqui: "Azahar " son 7 caracteres y
 * "version " son 8, de modo que a partir de ahi tienen que ser identicas. Si
 * alguien cambia una y olvida la otra, no compila.
 */
static_assert(std::string_view{kOverlayBuild}.substr(7) ==
                  std::string_view{kVersion}.substr(8),
              "kOverlayBuild y kVersion se han desincronizado: actualiza las dos");

} // namespace VitaFrontend
