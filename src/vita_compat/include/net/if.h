// Shim de <net/if.h> para PS Vita.
//
// Acompana al shim de <ifaddrs.h>: cpp-httplib lo incluye por if_nametoindex()
// y las banderas IFF_*. La Vita gestiona sus interfaces con sceNetCtl, asi que
// aqui solo se ofrece lo justo para compilar; if_nametoindex() devuelve 0
// ("interfaz desconocida"), que es lo que espera quien llama cuando falla.

#pragma once

#include <sys/socket.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef IF_NAMESIZE
#define IF_NAMESIZE 16
#endif

#define IFF_UP        0x1
#define IFF_BROADCAST 0x2
#define IFF_DEBUG     0x4
#define IFF_LOOPBACK  0x8
#define IFF_RUNNING   0x40
#define IFF_MULTICAST 0x8000

struct if_nameindex {
    unsigned int if_index;
    char* if_name;
};

static inline unsigned int if_nametoindex(const char* ifname) {
    (void)ifname;
    return 0;
}

static inline char* if_indextoname(unsigned int ifindex, char* ifname) {
    (void)ifindex;
    (void)ifname;
    return 0;
}

#ifdef __cplusplus
}
#endif
