// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Camino rapido de la coma flotante del invitado.
//
// EL PROBLEMA. Los ficheros vfpsingle.cpp y vfpdouble.cpp son la VFP de skyeye,
// derivada de la libreria softfloat de John Hauser: emulan la unidad de coma
// flotante del ARM11 CON ARITMETICA ENTERA. Cada VADD.F32 del juego se
// convierte en desempaquetar signo, exponente y mantisa de los dos operandos a
// una struct, operar con enteros de 32 y 64 bits, normalizar, redondear segun
// el modo del FPSCR, detectar excepciones y volver a empaquetar. Son decenas o
// cientos de instrucciones del anfitrion por cada operacion del invitado.
//
// Y el Cortex-A9 de la PS Vita TIENE VFPv3-D32 nativa. O sea que se estaba
// emulando por software una unidad de coma flotante encima de una unidad de
// coma flotante real que estaba ahi parada. Los juegos de 3DS hacen aritmetica
// de vectores y matrices en cada fotograma, asi que esto se toca constantemente.
//
// LA IDEA. En el caso comun -- que es la inmensa mayoria de las operaciones --
// la aritmetica del anfitrion da EXACTAMENTE el mismo resultado bit a bit que
// la rutina de softfloat, porque las dos implementan IEEE 754 binary32/binary64
// con el mismo modo de redondeo. Cuando se puede garantizar que ese es el caso,
// se hace la operacion nativa; cuando no, se cae al camino de softfloat de
// siempre, que sigue siendo la referencia de correccion y no se ha tocado.
//
// CUANDO SE PUEDE GARANTIZAR. Las condiciones se comprueban ANTES (sobre los
// operandos) y DESPUES (sobre el resultado):
//
//   1. El modo de redondeo del invitado es el de por defecto (a mas cercano).
//      Los otros tres los usa practicamente nadie y cambian el resultado.
//   2. Los dos operandos son NORMALES: exponente ni 0 (cero o subnormal) ni el
//      maximo (infinito o NaN). Esto descarta de golpe todos los casos raros --
//      propagacion de NaN, infinitos, subnormales, el signo del cero -- que son
//      justo donde softfloat tiene logica especial y donde es facil divergir.
//   3. El RESULTADO tambien es normal. Esto descarta desbordamiento a infinito
//      y agotamiento a subnormal, y de paso hace innecesario comprobar el bit
//      de flush-to-zero del anfitrion: si el anfitrion aplastara un subnormal a
//      cero, el resultado saldria con exponente 0 y esta comprobacion lo
//      rechazaria, mandandolo a softfloat.
//
// Con esas tres condiciones, la UNICA excepcion que puede levantar la operacion
// es la de inexacto (IXC): invalido, division por cero, desbordamiento,
// agotamiento y denormal de entrada son todos imposibles por construccion. Aun
// asi se leen todos los bits acumulativos del anfitrion en vez de suponerlo,
// que sale igual de barato y no depende de que este razonamiento sea perfecto.
//
// EL FPSCR DEL ANFITRION. Los bits acumulativos de excepcion del ARM estan en
// las mismas posiciones en el anfitrion y en el invitado -- los dos son ARM --,
// asi que no hay que traducir nada: se leen y se devuelven tal cual.
//
// Se mantiene la INVARIANTE de que al entrar en una operacion rapida los bits
// acumulativos del anfitrion estan a cero, limpiandolos despues de cada una que
// los ensucie. Asi solo hace falta UNA lectura de FPSCR por operacion (y una
// escritura cuando hubo inexacto), en vez de limpiar antes y leer despues.
//
// Lo que esta invariante no cubre es que el propio emulador haga cuentas de
// coma flotante en este mismo hilo entre dos operaciones del invitado; eso
// podria colar un IXC de mas. El FPSCR es parte del contexto de cada hilo, asi
// que los otros hilos del emulador no contaminan. Y el efecto, si ocurre, es
// que un bit PEGAJOSO que el juego casi con seguridad ya tenia puesto aparezca
// puesto: ningun juego de 3DS decide nada mirando IXC.
//
// LA FAMILIA DE MULTIPLICAR Y ACUMULAR (VMLA, VMLS, VNMLA, VNMLS). Hasta
// 0.1.8.0 se quedaba entera en softfloat, porque softfloat la hacia FUNDIDA (el
// producto sin redondear, un solo redondeo al final) y la VMLA del VFP redondea
// DOS veces. La duda de cual era la correcta se resolvio en 0.1.8.1: el manual
// de ARMv7 la define como FPAdd(D[d], FPMul(D[n], D[m])), dynarmic la hace asi,
// y el JIT (VMLA nativa) discrepaba del interprete justo en esos bloques.
// Softfloat redondea ahora el producto, y con eso el camino rapido (producto
// nativo, barrera, suma nativa) da exactamente lo mismo.

#pragma once

#include <bit>
#include "common/common_types.h"
#include "core/arm/skyeye_common/vfp/asm_vfp.h"

namespace VfpFast {

/// Solo tiene sentido con una VFP de verdad en el anfitrion. En cualquier otro
/// sitio (o compilando con -msoft-float) esto queda apagado y no se genera ni
/// una instruccion: todo pasa por softfloat, como antes.
#if defined(__arm__) && defined(__ARM_FP) && !defined(__SOFTFP__)
inline constexpr bool kAvailable = true;
#else
inline constexpr bool kAvailable = false;
#endif

/// Los seis bits acumulativos de excepcion del FPSCR.
inline constexpr u32 kCumulativeMask =
    FPSCR_IOC | FPSCR_DZC | FPSCR_OFC | FPSCR_UFC | FPSCR_IXC | FPSCR_IDC;

inline u32 ReadHostFpscr() {
#if defined(__arm__) && defined(__ARM_FP) && !defined(__SOFTFP__)
    u32 value;
    __asm__ __volatile__("vmrs %0, fpscr" : "=r"(value));
    return value;
#else
    return 0;
#endif
}

inline void WriteHostFpscr(u32 value) {
#if defined(__arm__) && defined(__ARM_FP) && !defined(__SOFTFP__)
    __asm__ __volatile__("vmsr fpscr, %0" : : "r"(value));
#else
    (void)value;
#endif
}

/**
 * Cierra una operacion rapida: comprueba que el anfitrion estaba redondeando
 * como debia, recoge los bits de excepcion que ha levantado y deja el FPSCR
 * limpio para la siguiente (ver la invariante de arriba).
 *
 * Devuelve false si el anfitrion NO estaba redondeando a mas cercano, en cuyo
 * caso el resultado que se acaba de calcular no sirve y hay que tirar por
 * softfloat.
 *
 * POR QUE EL MODO DEL ANFITRION SE COMPRUEBA AQUI Y NO UNA VEZ AL ARRANCAR.
 *
 * La version anterior lo resolvia con un "static const bool" dentro de una
 * funcion, calculado la primera vez. Parecia lo barato, y era justo lo caro: un
 * static local lleva su guarda de inicializacion, y el codigo que GCC genera
 * para leerla incluye una BARRERA DE MEMORIA (dmb ish) para el orden de
 * adquisicion. Se veia en el desensamblado, al principio de cada una de estas
 * funciones: una barrera, mas cargar la direccion de la guarda y la del
 * resultado, EN CADA OPERACION DE COMA FLOTANTE DEL JUEGO. En un Cortex-A9 una
 * dmb no es gratis, y estaba pagandose millones de veces por fotograma para
 * releer un valor que no cambia nunca.
 *
 * Metiendolo aqui sale GRATIS: el FPSCR del anfitrion hay que leerlo de todas
 * formas para saber si la operacion fue inexacta, y el modo de redondeo viene
 * en esa misma lectura. Y de paso queda mas fino que antes, porque se comprueba
 * en cada operacion en vez de una sola vez al principio: si algo cambiara el
 * modo de redondeo a mitad de partida, esto lo nota.
 */
inline bool FinishHostOp(u32* raised) {
    const u32 fpscr = ReadHostFpscr();
    if ((fpscr & FPSCR_RMODE_MASK) != FPSCR_ROUND_NEAREST) [[unlikely]] {
        return false;
    }
    *raised = fpscr & kCumulativeMask;
    if (*raised != 0) {
        WriteHostFpscr(fpscr & ~kCumulativeMask);
    }
    return true;
}

/**
 * Limpia los bits acumulativos del anfitrion sin mirar nada mas. Se usa cuando
 * una operacion rapida se descarta a mitad: hay que restablecer la invariante
 * antes de dejar paso a softfloat.
 */
inline void DiscardHostExceptions() {
    const u32 fpscr = ReadHostFpscr();
    if ((fpscr & kCumulativeMask) != 0) {
        WriteHostFpscr(fpscr & ~kCumulativeMask);
    }
}

/// El invitado tambien tiene que estar redondeando a mas cercano.
inline bool RoundingIsNearest(u32 guest_fpscr) {
    return (guest_fpscr & FPSCR_RMODE_MASK) == FPSCR_ROUND_NEAREST;
}

/// Normal = exponente entre 1 y 254. Fuera de ahi hay cero, subnormal, infinito
/// o NaN, y de todo eso se encarga softfloat.
inline bool SingleIsNormal(u32 bits) {
    return ((bits >> 23) & 0xFFu) - 1u < 254u;
}

/// Lo mismo en binary64: exponente entre 1 y 2046.
inline bool DoubleIsNormal(u64 bits) {
    return (static_cast<u32>(bits >> 52) & 0x7FFu) - 1u < 2046u;
}

inline bool SingleIsPositive(u32 bits) {
    return (bits & 0x80000000u) == 0;
}

inline bool DoubleIsPositive(u64 bits) {
    return (bits & 0x8000000000000000ull) == 0;
}

inline float BitsToSingle(u32 bits) {
    return std::bit_cast<float>(bits);
}

inline u32 SingleToBits(float value) {
    return std::bit_cast<u32>(value);
}

inline double BitsToDouble(u64 bits) {
    return std::bit_cast<double>(bits);
}

inline u64 DoubleToBits(double value) {
    return std::bit_cast<u64>(value);
}

/**
 * Condicion previa comun. El modo del ANFITRION no se mira aqui sino al cerrar
 * la operacion (ver FinishHostOp): ahi sale gratis y aqui costaria una lectura
 * de FPSCR de mas.
 */
inline bool Usable(u32 guest_fpscr) {
    return kAvailable && RoundingIsNearest(guest_fpscr);
}

} // namespace VfpFast
