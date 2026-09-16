// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/common_types.h"
#include "common/vector_math.h"

namespace Pica::Texture {

Common::Vec3<u8> SampleETC1Subtile(u64 value, unsigned int x, unsigned int y);

/**
 * Decodifica un bloque ETC1 completo (4x4 texeles) de una sola vez.
 *
 * Escribe en out_rgba los 16 texeles como RGBA8, con el mismo criterio de
 * coordenadas que SampleETC1Subtile: el texel (x, y) queda en out_rgba[x + y*4].
 * El alfa se pone a 255 (ETC1 no tiene alfa; ETC1A4 lo saca de un bloque
 * aparte).
 */
void DecodeETC1Block(u64 value, u32* out_rgba);

/**
 * Cache de bloques ETC1 decodificados, dirigida por el CONTENIDO del bloque.
 *
 * Cada texel muestreado de una textura ETC1 obligaba a descomprimir el bloque
 * entero otra vez -- muestrear la misma textura de fondo fotograma tras
 * fotograma era pagar decenas de instrucciones por texel, siempre para
 * producir el mismo resultado.
 *
 * La clave es el propio bloque de 8 bytes, no su direccion: la
 * descompresion es una funcion pura de esos ocho bytes, asi que un bloque
 * reescrito por el juego simplemente cambia de clave y NUNCA hay que
 * invalidar nada. Eso es lo que permite compartir la logica con el hilo de
 * emulacion sin cerrojos: cada hilo del rasterizador tiene SU cache (ver
 * StatefulThreadWorker en sw_rasterizer), asi que no hay estado compartido.
 *
 * Directa (un solo conjunto): si dos bloques colisionan, el ultimo desaloja
 * al otro y el siguiente acceso al viejo vuelve a decodificar. Sale mas caro
 * el sondeo con asociatividad que la decodificacion que se ahorra.
 */
class Etc1BlockCache {
public:
    static constexpr u32 kEntries = 1024;
    static_assert((kEntries & (kEntries - 1)) == 0, "kEntries debe ser potencia de dos");

    /// Texel `index` = x + y*4 del bloque `block`, decodificandolo si hace falta.
    u32 GetTexel(u64 block, u32 index);

private:
    static u32 Hash(u64 block) {
        // Mezcla de 64 a 10 bits: XOR de las dos mitades y una multiplicacion
        // por la constante de Knuth. En ARMv7 son tres o cuatro instrucciones.
        const u32 mixed = static_cast<u32>(block) ^ static_cast<u32>(block >> 32);
        return (mixed * 2654435761u) >> (32 - 10);
    }

    /// Bloque con el que se lleno cada entrada. Cero = entrada sin usar.
    std::array<u64, kEntries> tags{};
    /// Los 16 texeles RGBA8 de cada bloque cacheado.
    std::array<std::array<u32, 16>, kEntries> pixels{};
    /// Bloque recien decodificado en un fallo, para no escribir en la cache
    /// cuando el bloque es el cero (ver GetTexel en etc1.cpp).
    std::array<u32, 16> scratch{};
};

} // namespace Pica::Texture
