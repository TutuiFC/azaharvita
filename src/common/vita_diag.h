// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Diagnostico de fallos fatales en PS Vita.
//
// Azahar usa ASSERT/UNREACHABLE por todas partes. En PC el macro escribe en el
// registro y llama a Crash() (__builtin_trap), que bajo un depurador para la
// ejecucion y deja mirar el estado. En la Vita no hay depurador: __builtin_trap
// genera una instruccion indefinida y el kernel mata el proceso al instante,
// con el agravante de que el macro llama antes a Common::Log::Stop(), que para
// el hilo que vuelca el registro a disco. Resultado: la consola dice "se ha
// cerrado la aplicacion", azahar_log.txt se queda cortado a media linea y no
// queda ni rastro de que assert salto ni donde.
//
// Aqui todo se escribe antes de morir, con sceIo directo: sin hilo de registro,
// sin reservar memoria y sin depender del runtime de C++.
//
// Nota sobre por que VitaAssertFail es noexcept y no lanza: lanzar obligaba al
// compilador a generar tablas de desenrollado en cada una de las miles de
// funciones que contienen un ASSERT. Eso sumo unas 7.700 relocalizaciones al
// binario y lo paso de 2^21, justo el punto en el que vita-elf-create se cae
// sin decir nada. Terminar aqui mismo sale mas barato y el manejador de
// terminate del frontend sigue pintando el mensaje en pantalla.

#pragma once

#ifdef __PSVITA__

namespace Common {

/// Anota una linea en ux0:/data/azahar/crash.txt. Seguro en cualquier hilo y en
/// cualquier momento, incluido antes de main o con el heap agotado.
void VitaNote(const char* title, const char* detail);

/// Deja constancia de un assert fallido y termina el proceso.
///
/// El texto del assert no viaja por aqui: lo escribe el propio LOG_CRITICAL del
/// macro, que en la Vita vuelca los mensajes criticos a crash.txt de forma
/// sincrona (ver FmtLogMessageImpl). Aqui solo hace falta fichero y linea, y
/// asi el macro no crece ni una instruccion mas de lo que ya ocupaba.
[[noreturn]] void VitaAssertFail(const char* file, int line) noexcept;

/// Ultimo fallo fatal anotado, o cadena vacia. Lo usa el manejador de terminate
/// del frontend para pintar algo util en vez de "terminate directo".
const char* LastFatalMessage();

/**
 * Ata el hilo que llama a uno de los tres nucleos de usuario de la Vita.
 *
 * La Vita tiene cuatro nucleos Cortex-A9, de los cuales tres estan disponibles
 * para la aplicacion (el cuarto lo reserva el sistema). Si no se dice nada, los
 * hilos se reparten segun le parezca al planificador, y no hay garantia de que
 * acaben en nucleos distintos: un pool de hilos "en paralelo" puede terminar
 * turnandose en el mismo nucleo, con lo que no solo no gana nada sino que
 * ademas paga los cambios de contexto.
 *
 * @param index Numero de hilo. Se reparte en round-robin entre los 3 nucleos.
 * @param role  Texto corto para la anotacion en crash.txt ("rasterizador", etc).
 */
void VitaPinThreadToUserCore(unsigned int index, const char* role);

/**
 * Reloj REAL de la CPU en MHz, leido de vuelta despues de pedir el overclock.
 *
 * scePowerSetArmClockFrequency(444) puede ser rechazada en una compilacion
 * "safe" y devolver error, y nadie comprobaba el resultado. Toda la sesion se
 * han calculado ciclos por pixel suponiendo 444 MHz sin saber si es cierto: si
 * la consola lo dejo en 333, esas cuentas van un 33% infladas.
 *
 * Lo escribe main.cpp una vez al arrancar; lo lee el overlay. Un int escrito
 * antes de que arranque nada mas y leido despues no necesita atomico.
 */
extern int g_arm_clock_mhz;

/**
 * Pone la unidad de coma flotante en modo rapido, en el hilo que la llame.
 *
 * Activa dos bits del FPSCR:
 *   FZ (flush-to-zero): los denormales se tratan como cero.
 *   DN (default NaN):   cualquier NaN se convierte en el NaN por defecto.
 *
 * POR QUE. En el Cortex-A9, con FZ apagado una operacion que recibe o produce
 * un denormal no se resuelve en el camino rapido del hardware: se sale a codigo
 * de soporte y cuesta cientos de ciclos en vez de cuatro. Medido en consola:
 * ~7.500 ciclos por pixel sombreado para un camino de codigo de unas 1.000
 * instrucciones, o sea 7,5 ciclos por instruccion cuando este nucleo hace 1-1,5.
 * No se esta ejecutando demasiado: se esta esperando.
 *
 * Y denormales hay de sobra donde mirar: 1/w para geometria lejana, coordenadas
 * de textura cerca de cero, productos de baricentricas pequenas. f24 tiene menos
 * rango de exponente que float, asi que sus productos se cuelan en el rango
 * denormal de float32 con facilidad.
 *
 * La precision perdida es irrelevante aqui: la PICA200 YA descarta los
 * denormales en hardware (lo dice el propio comentario de Float::FromRaw en
 * pica_types.h), asi que tratarlos como cero se parece mas al 3DS real que no
 * hacerlo.
 *
 * El FPSCR es por hilo y se guarda en cada cambio de contexto, asi que hay que
 * llamarlo en CADA hilo: el principal y los tres del rasterizador.
 */
void VitaEnableFastFloatMode();

/**
 * True si la FPU ha visto algun denormal desde que arranco el proceso.
 *
 * Lee el bit IDC del FPSCR (denormal de entrada), que es acumulativo: una vez
 * que se pone, se queda. Sirve para confirmar o descartar la hipotesis de
 * arriba con un dato en vez de con una corazonada. Se consulta desde el overlay.
 */
bool VitaSawDenormals();

} // namespace Common

#endif // __PSVITA__
