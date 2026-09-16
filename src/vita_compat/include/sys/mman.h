// Shim de <sys/mman.h> para PS Vita.
//
// La Vita no tiene memoria mapeada al estilo POSIX: la reserva se hace con
// sceKernelAllocMemBlock. cpp-httplib incluye esta cabecera para servir
// ficheros con mmap, una ruta que este port no usa.
//
// mmap() devuelve MAP_FAILED, que es el fallo que cpp-httplib ya contempla;
// quien llame cae en su camino alternativo de lectura normal.

#pragma once

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROT_NONE  0x0
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define PROT_EXEC  0x4

#define MAP_SHARED  0x01
#define MAP_PRIVATE 0x02
#define MAP_ANON    0x20
#define MAP_ANONYMOUS MAP_ANON

#define MAP_FAILED ((void*)-1)

static inline void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    (void)addr;
    (void)length;
    (void)prot;
    (void)flags;
    (void)fd;
    (void)offset;
    return MAP_FAILED;
}

static inline int munmap(void* addr, size_t length) {
    (void)addr;
    (void)length;
    return -1;
}

static inline int msync(void* addr, size_t length, int flags) {
    (void)addr;
    (void)length;
    (void)flags;
    return -1;
}

#ifdef __cplusplus
}
#endif
