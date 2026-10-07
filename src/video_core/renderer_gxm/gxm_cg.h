// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>
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

/**
 * Solo la cache de la tarjeta, sin el compilador (0.1.9.9): nullptr si no esta
 * o no vale. Es lo unico que pide el hilo de emulacion en partida: CompileCg,
 * si el fichero de la cache no sirve, va al compilador, y con el hilo de
 * compilacion ocupado eso era esperarle (Yo-kai Watch, 0.1.9.8).
 */
const SceShaccCgCompileOutput* LoadCgCache(SceShaccCgTargetProfile profile, const char* name,
                                           const char* source, bool from_card = true);

/// Compila un shader. El resultado vive hasta ReleaseCgOutput.
const SceShaccCgCompileOutput* CompileCg(SceShaccCgTargetProfile profile, const char* name,
                                         const char* source);

void ReleaseCgOutput(const SceShaccCgCompileOutput* output);

/**
 * PRECARGA DE LA CACHE DE SHADERS (0.2.0.3). Cada shader que se lee o se
 * guarda en la cache de la tarjeta queda apuntado en una lista por juego
 * (shadercache/juego_<id>.lst). Al arrancar ese juego otra vez, esto lee a
 * memoria todos los de su lista, y durante la partida se sirven de ahi en vez
 * de abrir un fichero en la tarjeta por shader: en una transicion al 3D eran
 * decenas de lecturas de varios ms cada una en mitad del juego. 'progress'
 * recibe (leidos, total). Devuelve cuantos se precargaron.
 */
u32 PreloadCgCache(u64 program_id, const std::function<void(u32, u32)>& progress);

/**
 * COMPILACION EN SEGUNDO PLANO (0.1.9.6).
 *
 * Un shader de vertices de 11-16 KB de Cg tarda de varios segundos a mas de
 * medio minuto en el compilador de la consola, y hasta ahora eso era el juego
 * congelado (Zafiro Alfa al entrar al 3D, Pokemon Sol al presentarse Kukui).
 * Ahora un hilo aparte compila y el lote sigue por la CPU mientras tanto; el
 * resultado se recoge en el hilo de emulacion (registrar el programa en el
 * parcheador no es seguro desde otro hilo). Las fuentes son las variantes a
 * probar, en orden: la primera que compile gana. El compilador sigue siendo
 * uno solo: CompileCg lo protege con un cerrojo, asi que un shader de
 * fragmentos que haga falta en el hilo de emulacion espera, como mucho, a que
 * acabe la compilacion en curso.
 */
struct CgJob {
    /// Vertices o fragmentos (0.1.9.8: los de fragmentos tambien van aqui).
    SceShaccCgTargetProfile profile = SCE_SHACCCG_PROFILE_VP;
    const char* name = "azahar_gxm_vs.cg";
    std::vector<std::string> sources;
    std::vector<u32> variants;
    /// El nombre de cada fuente para crash.txt, si no es 'name' (0.3.1.4).
    std::vector<const char*> names;
    /// Lo escribe el hilo de compilacion ANTES de poner done.
    const SceShaccCgCompileOutput* output = nullptr;
    u32 used_variant = 0;
    std::atomic<bool> done{false};
    /// Cuando se encolo (VitaMicros), para saber cuanto lleva esperando.
    unsigned long long submitted_us = 0;
    /// Cuando lo cogio el hilo de compilacion (0 = aun en la cola), 0.3.1.3.
    std::atomic<unsigned long long> started_us{0};
    /// Minutos de compilador (0.2.2.3): se coge cuando no queda otro en la cola.
    bool heavy = false;
};

/// 'urgent' pone el trabajo delante de la cola (los de fragmentos: sin ellos
/// el lote no se dibuja).
void CgSubmit(std::shared_ptr<CgJob> job, bool urgent = false);

/// Hay lotes por la CPU esperando a un trabajo de la cola: el hilo de
/// compilacion sube por encima de los ayudantes hasta acabar el que tiene entre
/// manos (0.2.1.3). Barato si ya esta subido.
void CgMarkStarved();

/// Lo que esta haciendo el hilo de compilacion, para el aviso en pantalla
/// (0.2.1.6): desde cuando compila el trabajo actual (VitaMicros, 0 si nada) y
/// cuantos esperan detras.
struct CgActivity {
    unsigned long long busy_since_us = 0;
    u32 queued = 0;
};
CgActivity GetCgActivity();

} // namespace Gxm
