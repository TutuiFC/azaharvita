// Copyright 2013 Dolphin Emulator Project / 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstdlib>
#include "common/common_funcs.h"
#include "common/logging/backend.h"
#include "common/logging/log.h"

// For asserts we'd like to keep all the junk executed when an assert happens away from the
// important code in the function. One way of doing this is to put all the relevant code inside a
// lambda and force the compiler to not inline it.

#ifdef __PSVITA__

// En la Vita la version de PC no sirve como diagnostico: Common::Log::Stop()
// para el hilo que vuelca el registro a disco -- y se pierde justo el mensaje
// que interesa, dejando azahar_log.txt cortado a media linea -- y Crash() es
// __builtin_trap(), que el kernel traduce en matar el proceso al instante. Un
// assert quedaba asi en "la aplicacion se ha cerrado": sin fichero, sin pantalla
// y sin saber que linea de codigo lo lanzo.
//
// El unico cambio es el final del macro: se mantiene el LOG_CRITICAL (que en la
// Vita vuelca los mensajes criticos a crash.txt de forma sincrona, ver
// FmtLogMessageImpl) y se sustituyen Stop()+Crash()+exit() por una sola llamada
// que anota fichero y linea y termina. Sale mas barato que el original -- tres
// llamadas menos por cada uno de los miles de asserts -- lo cual importa: la
// version que formateaba el mensaje aqui mismo sumaba 7.700 relocalizaciones y
// pasaba el binario de 2^21, justo donde vita-elf-create se cae sin decir nada.
#include "common/vita_diag.h"

#define ASSERT(_a_)                                                                                \
    do                                                                                             \
        if (!(_a_)) [[unlikely]] {                                                                 \
            []() CITRA_NO_INLINE CITRA_NO_RETURN {                                                 \
                LOG_CRITICAL(Debug, "Assertion Failed!");                                          \
                Common::VitaAssertFail(__FILE__, __LINE__);                                        \
            }();                                                                                   \
        }                                                                                          \
    while (0)

#define ASSERT_MSG(_a_, ...)                                                                       \
    do                                                                                             \
        if (!(_a_)) [[unlikely]] {                                                                 \
            [&]() CITRA_NO_INLINE CITRA_NO_RETURN {                                                \
                LOG_CRITICAL(Debug, "Assertion Failed!\n" __VA_ARGS__);                            \
                Common::VitaAssertFail(__FILE__, __LINE__);                                        \
            }();                                                                                   \
        }                                                                                          \
    while (0)

#define UNREACHABLE()                                                                              \
    ([]() CITRA_NO_INLINE CITRA_NO_RETURN {                                                        \
        LOG_CRITICAL(Debug, "Unreachable code!");                                                  \
        Common::VitaAssertFail(__FILE__, __LINE__);                                                \
    }())

#define UNREACHABLE_MSG(...)                                                                       \
    ([&]() CITRA_NO_INLINE CITRA_NO_RETURN {                                                       \
        LOG_CRITICAL(Debug, "Unreachable code!\n" __VA_ARGS__);                                    \
        Common::VitaAssertFail(__FILE__, __LINE__);                                                \
    }())

#else

#define ASSERT(_a_)                                                                                \
    do                                                                                             \
        if (!(_a_)) [[unlikely]] {                                                                 \
            []() CITRA_NO_INLINE CITRA_NO_RETURN {                                                 \
                LOG_CRITICAL(Debug, "Assertion Failed!");                                          \
                Common::Log::Stop();                                                               \
                Crash();                                                                           \
                exit(1);                                                                           \
            }();                                                                                   \
        }                                                                                          \
    while (0)

#define ASSERT_MSG(_a_, ...)                                                                       \
    do                                                                                             \
        if (!(_a_)) [[unlikely]] {                                                                 \
            [&]() CITRA_NO_INLINE CITRA_NO_RETURN {                                                \
                LOG_CRITICAL(Debug, "Assertion Failed!\n" __VA_ARGS__);                            \
                Common::Log::Stop();                                                               \
                Crash();                                                                           \
                exit(1);                                                                           \
            }();                                                                                   \
        }                                                                                          \
    while (0)

#define UNREACHABLE()                                                                              \
    ([]() CITRA_NO_INLINE CITRA_NO_RETURN {                                                        \
        LOG_CRITICAL(Debug, "Unreachable code!");                                                  \
        Common::Log::Stop();                                                                       \
        Crash();                                                                                   \
        exit(1);                                                                                   \
    }())

#define UNREACHABLE_MSG(...)                                                                       \
    ([&]() CITRA_NO_INLINE CITRA_NO_RETURN {                                                       \
        LOG_CRITICAL(Debug, "Unreachable code!\n" __VA_ARGS__);                                    \
        Common::Log::Stop();                                                                       \
        Crash();                                                                                   \
        exit(1);                                                                                   \
    }())

#endif // __PSVITA__

#ifdef _DEBUG
#define DEBUG_ASSERT(_a_) ASSERT(_a_)
#define DEBUG_ASSERT_MSG(_a_, ...) ASSERT_MSG(_a_, __VA_ARGS__)
#else // not debug
#define DEBUG_ASSERT(_a_)
#define DEBUG_ASSERT_MSG(_a_, _desc_, ...)
#endif

#define UNIMPLEMENTED() LOG_CRITICAL(Debug, "Unimplemented code!")
#define UNIMPLEMENTED_MSG(_a_, ...) LOG_CRITICAL(Debug, _a_, __VA_ARGS__)
