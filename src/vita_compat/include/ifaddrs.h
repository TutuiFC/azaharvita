// Shim de <ifaddrs.h> para PS Vita.
//
// La newlib del VitaSDK no permite enumerar interfaces de red (en la Vita eso
// se consulta con sceNetCtl, con una API distinta). cpp-httplib incluye esta
// cabecera para poder atarse a una interfaz concreta; como en este port las
// funciones online no son objetivo, getifaddrs() devuelve "ninguna interfaz"
// y quien llame se queda con el comportamiento por defecto.

#pragma once

#include <sys/socket.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ifaddrs {
    struct ifaddrs* ifa_next;
    char* ifa_name;
    unsigned int ifa_flags;
    struct sockaddr* ifa_addr;
    struct sockaddr* ifa_netmask;
    union {
        struct sockaddr* ifu_broadaddr;
        struct sockaddr* ifu_dstaddr;
    } ifa_ifu;
    void* ifa_data;
};

#define ifa_broadaddr ifa_ifu.ifu_broadaddr
#define ifa_dstaddr   ifa_ifu.ifu_dstaddr

static inline int getifaddrs(struct ifaddrs** ifap) {
    if (ifap != 0) {
        *ifap = 0;
    }
    return -1;
}

static inline void freeifaddrs(struct ifaddrs* ifa) {
    (void)ifa;
}

#ifdef __cplusplus
}
#endif
