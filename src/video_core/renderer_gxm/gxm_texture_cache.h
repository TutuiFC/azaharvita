// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <psp2/gxm.h>
#include "common/common_types.h"
#include "video_core/renderer_gxm/gxm_memory.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct RegsInternal;
}

namespace Pica::Texture {
class Etc1BlockCache;
}

namespace Gxm {

/**
 * Texturas de la PICA listas para la GPU.
 *
 * La PICA guarda sus texturas con su propio entrelazado (mosaicos de 8x8) y en
 * una docena de formatos, algunos comprimidos (ETC1). GXM tiene sus formatos,
 * pero con otro entrelazado, asi que la via segura es decodificar una vez al
 * formato que entiende todo el mundo (RGBA8 lineal) y quedarse con la copia:
 * el decodificado lo hace el mismo codigo que usa el rasterizador de software
 * (Pica::Texture), asi que la imagen no puede divergir.
 *
 * POR QUE ESTO ES ASOCIATIVO Y ANTES NO LO ERA. Hasta 0.1.0.16 habia UNA entrada
 * por unidad de textura: tres en total, fijas. Bastaba mientras un lote usara
 * siempre la misma textura en cada unidad, pero en cuanto el juego alterna dos
 * -- el caso normal en un menu, en un tileset o en cualquier cosa con atlas --
 * cada cambio tiraba la entrada y VOLVIA A DECODIFICAR la textura entera, de
 * mosaicos o de ETC1 a RGBA8, en la CPU. Eso es el trabajo mas caro de todo el
 * camino grafico y se estaba repitiendo varias veces por fotograma para las
 * mismas dos imagenes.
 *
 * Ahora hay un conjunto de kMaxEntries entradas compartidas por las tres
 * unidades, con expulsion por menos usada recientemente (LRU) y un tope de
 * memoria. Dos unidades que usen la misma textura comparten entrada sola.
 *
 * QUE ENTRA EN LA CLAVE, Y POR QUE TAMBIEN EL REPETIDO Y EL FILTRO. La imagen
 * decodificada solo depende de (direccion, formato, ancho, alto), pero el modo
 * de repetido y los filtros viven DENTRO del objeto SceGxmTexture, no fuera. Si
 * se dejaran fuera de la clave habria que reescribirlos en el objeto en cada
 * consulta, y ese objeto puede haberselo quedado ya un dibujado anterior de la
 * misma escena que todavia no ha ejecutado la GPU. Entran en la clave: una
 * textura usada con dos repetidos distintos ocupa dos entradas. Es raro, y a
 * cambio no hay que suponer nada sobre cuando copia GXM los descriptores.
 *
 * EL TOPE DE MEMORIA NO SE MIDE EN PIXELES SINO EN BLOQUES. Un bloque de CDRAM
 * se redondea a 256 KB (ver gxm_memory.cpp), asi que una textura de 64x64 -- 16
 * KB de pixeles -- se come 256 KB igual. Por eso el presupuesto se lleva con lo
 * que ocupa el bloque de verdad, Allocation::Size(), y no con ancho*alto*4: con
 * lo segundo el cache creeria que cabe diez veces mas de lo que cabe.
 *
 * Lo que no se soporta todavia devuelve nullptr y el lote entero cae al
 * rasterizador de software: texturas que no sean 2D, unidades deshabilitadas y
 * el modo de repetido ClampToBorder2.
 */
class TextureCache {
public:
    /**
     * Cuantas texturas distintas caben a la vez.
     *
     * 48 y no tres: un fotograma de un juego normal toca del orden de unas
     * decenas de texturas, y lo que se quiere evitar es que dos que se alternan
     * se echen la una a la otra. Pasarse tampoco sale gratis -- cada entrada es
     * un bloque de memoria reservado y mapeado para la GPU, y eso son dos
     * llamadas al sistema que no son baratas -- pero el tope real lo pone el
     * presupuesto de abajo, no este numero.
     */
    static constexpr u32 kMaxEntries = 384;

    /**
     * Tope de memoria de GPU para las texturas decodificadas, en bytes.
     *
     * La CDRAM son 128 MB que hoy solo usan las superficies de dibujado (unos
     * 6 MB con las ocho llenas), asi que 32 MB aqui sobran de largo y dejan
     * sitio de sobra por si una fase posterior quiere mas. Al llegar al tope se
     * expulsa la entrada menos usada recientemente hasta que quepa la nueva.
     */
    static constexpr u32 kMemoryBudget = 64u * 1024u * 1024u;
    /**
     * 0.1.9.9: 48 entradas y 32 MB se quedaban cortas en un 3D normal. Kirby
     * Triple Deluxe usa mas de 48 texturas por fotograma: se echaban unas a
     * otras y se volvian a decodificar TODAS en cada fotograma (crash.txt de
     * 0.1.9.8: "tx" 171 ms por fotograma). Y cada bloque de CDRAM se redondea a
     * 256 KB, asi que una textura de 64x64 se comia 256 KB del presupuesto: las
     * de hasta kSmallTexture van a memoria normal, que se redondea a 4 KB.
     */
    static constexpr u32 kSmallTexture = 64u * 1024u;

    TextureCache(); // en el .cpp: unique_ptr<Etc1BlockCache> necesita el tipo completo
    ~TextureCache();

    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;

    /// Textura de la unidad (0-2) para el estado actual, o nullptr.
    [[nodiscard]] const SceGxmTexture* Get(u32 unit, const Pica::RegsInternal& regs,
                                          Memory::MemorySystem& memory);

    /// Tira las texturas que solapen el rango.
    void InvalidateRange(PAddr addr, u32 size);

    /// El filtro y el repetido de la unidad en una textura que no es de la
    /// cache (0.2.2.4, ver RasterizerGXM::TextureFromCopy). false si GXM no
    /// tiene ese repetido.
    static bool ApplyUnitSampler(u32 unit, const Pica::RegsInternal& regs, SceGxmTexture& texture);

    /// Tira todas.
    void Clear();

    /**
     * Suelta de verdad la memoria de las entradas que se tiraron mientras habia
     * una escena abierta.
     *
     * POR QUE NO SE SUELTA EN EL ACTO. GXM no dibuja cuando se lo pides: apunta
     * el dibujado y lee los pixeles de la textura mas tarde, al procesar la
     * escena. Liberar el bloque en cuanto el juego escribe en esa memoria --
     * que es cuando llega la invalidacion -- le quita el suelo a un dibujado
     * que todavia no ha ocurrido, y eso no falla en el sitio: sale mas tarde,
     * como basura en pantalla o como una caida sin relacion aparente.
     *
     * Lo llama RasterizerGXM::EndScene, que es el unico punto donde se sabe con
     * certeza que la GPU ha terminado (hace sceGxmFinish).
     */
    void ReleaseRetired();

    /// Hay memoria retirada esperando a que la GPU termine (4.5): con eso el
    /// llamador sabe si tiene que pagar sceGxmFinish antes de liberar.
    [[nodiscard]] bool HasRetired() const {
        return !retired.empty() || !sealed.empty();
    }

    /**
     * Con vallas (0.1.9.4): lo retirado hasta ahora lo puede leer, como mucho,
     * la escena de la valla 'fence'; se suelta cuando esa valla haya pasado
     * (ReleaseUpTo con la ultima valla que ha completado la GPU).
     */
    void SealRetired(u32 fence);
    void ReleaseUpTo(u32 completed_fence);

private:
    struct Entry {
        bool valid = false;
        u64 key = 0;
        PAddr address = 0;
        u32 span = 0;
        /// Marca de tiempo logica del ultimo uso, para la expulsion LRU.
        u64 last_use = 0;
        /// Hash (XXH3) de los bytes del invitado con los que se decodifico.
        u64 source_hash = 0;
        /**
         * Le ha llegado un aviso de que su memoria PUEDE haber cambiado. No se
         * tira: en el siguiente uso se compara el hash de los bytes actuales
         * con source_hash y solo se decodifica otra vez si difieren. Ver Get.
         */
        bool stale = false;
        SceGxmTexture texture{};
        Allocation buffer;
    };

    /// Manda la memoria de la entrada a la lista de espera y la deja libre.
    void Retire(Entry& entry);

    /// Hueco donde meter una entrada nueva de 'needed' bytes, expulsando por
    /// LRU lo que haga falta. Nunca devuelve nullptr (siempre hay algo que
    /// expulsar), pero si puede devolver un hueco cuya expulsion no baste.
    Entry& MakeRoom(u32 needed);

    std::array<Entry, kMaxEntries> entries;
    /// Clave -> entrada (0.1.9.9): con 192, recorrerlas en cada consulta (tres
    /// por lote, cientos de lotes) ya costaria.
    std::unordered_map<u64, u32> index;
    /// Memoria a la espera de que la GPU acabe con ella (ver ReleaseRetired).
    std::vector<Allocation> retired;
    /// Lo retirado, con la valla de la ultima escena que pudo leerlo.
    std::vector<std::pair<u32, Allocation>> sealed;
    /// Reloj logico: sube en cada consulta y es lo que ordena el LRU.
    u64 clock = 0;
    /// Bytes de bloque (no de pixeles) que suman las entradas validas.
    u32 bytes_used = 0;
    /**
     * Cache de bloques ETC1 para decodificar, la misma que usa el rasterizador
     * de software en cada hilo. Sin ella, cada texel de una textura ETC1 --
     * casi todas las de 3DS -- descomprimia su bloque entero otra vez: 16
     * veces por bloque. La cache va por contenido, asi que no se invalida
     * nunca (ver Etc1BlockCache). Se crea al primer fallo: son ~70 KB.
     */
    std::unique_ptr<Pica::Texture::Etc1BlockCache> etc1_cache;
    /// Banda de 8 lineas en RAM para decodificar por filas de mosaicos (ver
    /// Get). Se reutiliza: como mucho 1024 x 8 texeles, 32 KB.
    std::vector<u32> band;
};

} // namespace Gxm
