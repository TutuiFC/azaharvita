// Shim de <openssl/rand.h> para PS Vita.
//
// Azahar solo usa RAND_bytes() en dos sitios (generar clave e IV de AES al
// envolver un ticket durante la instalacion de un CIA). No merece la pena
// portar LibreSSL entero para eso.
//
// Antes esto llamaba al generador del kernel, pero un homebrew firmado como
// "safe" no tiene acceso garantizado a esa API y la llamada acababa saltando a
// un stub sin resolver. Ahora usa el generador propio de vita_random.h.

#pragma once

#include "vita_random.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline int RAND_bytes(unsigned char* buf, int num) {
    if (buf == 0 || num <= 0) {
        return 0; // OpenSSL: 0 = fallo
    }
    return VitaRngFill(buf, (size_t)num); // siempre 1
}

static inline int RAND_priv_bytes(unsigned char* buf, int num) {
    return RAND_bytes(buf, num);
}

#ifdef __cplusplus
}
#endif
