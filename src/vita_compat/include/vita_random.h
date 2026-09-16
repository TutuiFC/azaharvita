// Generador de numeros aleatorios para PS Vita.
//
// La version anterior llamaba a sceKernelGetRandomNumber, el generador del
// kernel. El problema es que este binario se firma como "safe" (ver
// AZAHAR_UNSAFE en el CMakeLists del frontend), y un homebrew safe tiene el
// acceso restringido a parte de las APIs del sistema: la llamada acaba en un
// stub sin resolver y salta a una direccion invalida.
//
// Aqui se usa xoshiro128**, un generador rapido y de calidad razonable,
// sembrado con el reloj del proceso. No es criptograficamente seguro, y da
// igual: Azahar solo lo usa para rellenar claves e identificadores de consola
// simulados que nunca salen del emulador.

#pragma once

#include <psp2/kernel/processmgr.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static uint32_t vita_rng_state[4];
static int vita_rng_seeded;

static inline uint32_t VitaRngRotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static inline void VitaRngSeed(void) {
    // SplitMix32 sobre el reloj para repartir bien la semilla entre los cuatro
    // estados; sembrar los cuatro con el mismo valor da secuencias pobres.
    uint32_t seed = (uint32_t)sceKernelGetProcessTimeWide();
    seed ^= (uint32_t)(sceKernelGetProcessTimeWide() >> 32);
    if (seed == 0) {
        seed = 0x9E3779B9u;
    }
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B9u;
        uint32_t z = seed;
        z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
        z = (z ^ (z >> 13)) * 0xC2B2AE35u;
        vita_rng_state[i] = z ^ (z >> 16);
    }
    vita_rng_seeded = 1;
}

static inline uint32_t VitaRngNext(void) {
    if (!vita_rng_seeded) {
        VitaRngSeed();
    }

    const uint32_t result = VitaRngRotl(vita_rng_state[1] * 5u, 7) * 9u;
    const uint32_t t = vita_rng_state[1] << 9;

    vita_rng_state[2] ^= vita_rng_state[0];
    vita_rng_state[3] ^= vita_rng_state[1];
    vita_rng_state[1] ^= vita_rng_state[2];
    vita_rng_state[0] ^= vita_rng_state[3];
    vita_rng_state[2] ^= t;
    vita_rng_state[3] = VitaRngRotl(vita_rng_state[3], 11);

    return result;
}

/// Rellena un buffer de bytes. Devuelve 1 siempre (nunca falla).
static inline int VitaRngFill(unsigned char* buf, size_t len) {
    size_t i = 0;
    while (i + 4 <= len) {
        const uint32_t value = VitaRngNext();
        buf[i + 0] = (unsigned char)(value & 0xFF);
        buf[i + 1] = (unsigned char)((value >> 8) & 0xFF);
        buf[i + 2] = (unsigned char)((value >> 16) & 0xFF);
        buf[i + 3] = (unsigned char)((value >> 24) & 0xFF);
        i += 4;
    }
    if (i < len) {
        uint32_t value = VitaRngNext();
        while (i < len) {
            buf[i++] = (unsigned char)(value & 0xFF);
            value >>= 8;
        }
    }
    return 1;
}

#ifdef __cplusplus
}
#endif
