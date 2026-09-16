// Shim de <sys/ioctl.h> para PS Vita.
//
// La newlib del VitaSDK no trae ioctl(). enet solo lo usa para poner los
// sockets en modo no bloqueante, que aqui se traduce a fcntl(O_NONBLOCK).
// El resto de peticiones devuelven error: el multijugador no es objetivo
// de este port.

#pragma once

#include <fcntl.h>
#include <stdarg.h>

#ifndef FIONBIO
#define FIONBIO 0x5421
#endif
#ifndef FIONREAD
#define FIONREAD 0x541B
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    int* argp = va_arg(ap, int*);
    va_end(ap);

    if (request == FIONBIO) {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0) {
            return -1;
        }
        if (argp != 0 && *argp != 0) {
            flags |= O_NONBLOCK;
        } else {
            flags &= ~O_NONBLOCK;
        }
        return fcntl(fd, F_SETFL, flags);
    }

    return -1;
}

#ifdef __cplusplus
}
#endif
