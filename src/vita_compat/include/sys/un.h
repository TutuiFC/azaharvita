// Shim de <sys/un.h> para PS Vita.
//
// La Vita no tiene sockets de dominio Unix. cpp-httplib incluye esta cabecera
// para poder escuchar en un socket de fichero, algo que este port no usa: solo
// hace falta que la estructura exista para que el fichero compile.

#pragma once

#include <sys/socket.h>

#ifndef AF_UNIX
#define AF_UNIX 1
#endif
#ifndef AF_LOCAL
#define AF_LOCAL AF_UNIX
#endif

struct sockaddr_un {
    unsigned short sun_family;
    char sun_path[108];
};
