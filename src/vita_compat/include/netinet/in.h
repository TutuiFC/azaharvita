// Complemento de <netinet/in.h> para PS Vita.
//
// El netinet/in.h del VitaSDK trae las estructuras IPv6 pero no las macros
// IN6_IS_ADDR_* que define POSIX. cpp-httplib las usa para detectar
// direcciones link-local y anadirles el identificador de zona.
//
// Este fichero se antepone al del SDK en la ruta de busqueda, incluye el
// original con #include_next y completa lo que falta.

#pragma once

#include_next <netinet/in.h>

#ifndef IN6_IS_ADDR_LINKLOCAL
#define IN6_IS_ADDR_LINKLOCAL(a)                                                                   \
    ((((const unsigned char*)(a))[0] == 0xfe) && ((((const unsigned char*)(a))[1] & 0xc0) == 0x80))
#endif

#ifndef IN6_IS_ADDR_SITELOCAL
#define IN6_IS_ADDR_SITELOCAL(a)                                                                   \
    ((((const unsigned char*)(a))[0] == 0xfe) && ((((const unsigned char*)(a))[1] & 0xc0) == 0xc0))
#endif

#ifndef IN6_IS_ADDR_LOOPBACK
#define IN6_IS_ADDR_LOOPBACK(a)                                                                    \
    ((((const unsigned int*)(a))[0] == 0) && (((const unsigned int*)(a))[1] == 0) &&                \
     (((const unsigned int*)(a))[2] == 0) &&                                                       \
     (((const unsigned char*)(a))[12] == 0) && (((const unsigned char*)(a))[13] == 0) &&            \
     (((const unsigned char*)(a))[14] == 0) && (((const unsigned char*)(a))[15] == 1))
#endif

#ifndef IN6_IS_ADDR_MULTICAST
#define IN6_IS_ADDR_MULTICAST(a) (((const unsigned char*)(a))[0] == 0xff)
#endif
