// Copyright 2015-2025 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <span>
#include "common/thread_worker.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/pica_types.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_software/sw_clipper.h"
#include "video_core/renderer_software/sw_framebuffer.h"
#include "video_core/texture/etc1.h"
#include "video_core/texture/texture_decode.h"

namespace Pica {
struct RegsInternal;
class PicaCore;
} // namespace Pica

namespace SwRenderer {

/**
 * Contadores del rasterizador, para saber POR QUE es lento.
 *
 * Medido: el rasterizado se lleva ~95% del tiempo de fotograma. Eso puede venir
 * de tres sitios distintos, y cada uno se arregla de otra forma:
 *
 *   - Demasiados triangulos: el juego manda mucha geometria. Poco que hacer
 *     desde el emulador.
 *   - Demasiado overdraw: se pinta el mismo pixel muchas veces por fotograma.
 *     Se ataca con pruebas de profundidad tempranas.
 *   - Cada pixel cuesta demasiado: el problema esta en el bucle de sombreado.
 *     Se ataca optimizando ese bucle.
 *
 * Hay un cuarto sospechoso que estos contadores tambien destapan: el
 * rasterizador recorre la CAJA que envuelve al triangulo y descarta los pixeles
 * que caen fuera. Si "probados" es mucho mayor que "pintados", se esta gastando
 * el tiempo en pixeles que nunca se dibujan.
 *
 * Los incrementos por pixel se acumulan en variables locales y se suman al
 * atomico una vez por linea: un atomico por pixel falsearia la propia medida.
 */
namespace RasterizerStats {
inline std::atomic<u32> triangles{0};
inline std::atomic<u32> pixels_tested{0};
inline std::atomic<u32> pixels_drawn{0};

/**
 * Pixeles que SI caen dentro del triangulo, y de esos los que mata el rechazo
 * temprano por profundidad.
 *
 * Sin separar estos dos, 'probados / pintados' mezcla tres cosas muy distintas:
 * pixeles de la caja envolvente que quedan fuera del triangulo (baratos, se
 * descartan en tres comparaciones), pixeles tapados por geometria mas cercana
 * (carisimos si se sombrean antes de mirar la profundidad) y pixeles que mueren
 * en la prueba de alfa. Cada uno se arregla de una forma distinta, asi que hace
 * falta poder distinguirlos.
 */
inline std::atomic<u32> pixels_covered{0};
inline std::atomic<u32> pixels_zkill{0};

/**
 * De los pixeles sombreados, cuantos mueren en la prueba de alfa y cuantos en
 * la de profundidad FINAL.
 *
 * Hacen falta los dos para saber que atacar: si los que mueren en profundidad
 * son muchos, el rasterizador esta sombreando pixeles tapados y compensa
 * adelantar la prueba (early-Z); si son de alfa, el color de esos pixeles hay
 * que calcularlo igual porque el alfa sale del TEV.
 */
inline std::atomic<u32> pixels_alpha_fail{0};
inline std::atomic<u32> pixels_depth_fail{0};

/**
 * Pixeles que pasan por la iluminacion por fragmento.
 *
 * Sale de una cuenta que no cuadra: ~145.000 pixeles cubiertos por segundo con
 * tres nucleos a 444 MHz son unos 7.600 ciclos por pixel, y un rasterizador
 * software lento anda por 300-800. Hay algo carisimo dentro del bucle.
 *
 * ComputeFragmentsColors es el principal sospechoso: normaliza un cuaternion,
 * y por cada luz activa hace busquedas en tablas y productos escalares. Con
 * varias luces son facilmente miles de ciclos, y se ejecuta por pixel.
 */
inline std::atomic<u32> pixels_lit{0};

/**
 * Unidades de textura activas por triangulo, y cuantas usan ETC1.
 *
 * Decide si merece la pena cachear la descompresion ETC1: descomprimir cuesta
 * unas 50 instrucciones por texel frente a unas 10 de los formatos planos, pero
 * si la escena no usa ETC1 -- los menus y las pantallas 2D suelen ir en RGBA8 o
 * RGB565 -- ese trabajo seria para nada.
 *
 * Se cuenta por TRIANGULO, no por texel. Contarlo por texel exigia variables
 * thread_local, y en este toolchain no hay TLS nativo: cada acceso acaba en
 * __emutls_get_address, que toma un mutex interno. Tres por pixel y desde tres
 * hilos a la vez fue exactamente lo que provoco un crash con
 * "pthread_mutex_lock EINVAL" en los tres.
 */
inline std::atomic<u32> tex_units_total{0};
inline std::atomic<u32> tex_units_etc1{0};

/**
 * Unidades de textura por FORMATO, en el mismo orden que el enum
 * TexturingRegs::TextureFormat (14 valores). Se cuentan una vez por unidad y
 * triangulo, asi que el coste es despreciable.
 *
 * 'etc1' ya decia que este juego no usa ETC1, pero no cual usa. Sin ese dato,
 * cualquier trabajo en LookupTexelInTile (que es un switch de 14 formatos con
 * su decodificacion) es a ciegas: esta tabla lo cierra.
 */
inline std::array<std::atomic<u32>, 14> tex_format_units{};

/// Suma de etapas TEV activas sobre todos los triangulos. Dividido entre
/// 'triangles' da la media de etapas que hace de verdad cada uno, de 1 a 6.
/// Si sale cerca de 6, saltarse las inertes no aporta nada en ese juego.
inline std::atomic<u32> tev_stages_used{0};

/**
 * Triangulos repartidos entre los tres hilos, frente a los que van en uno solo.
 *
 * Ya paso una vez: en la 0.2 un umbral demasiado alto mando todos los
 * triangulos al camino de un hilo y el emulador fue exactamente 3 veces mas
 * lento. Conviene tenerlo medido en pantalla y no volver a deducirlo.
 */
inline std::atomic<u32> tri_split{0};
inline std::atomic<u32> tri_single{0};

/**
 * Triangulos en los que el rechazo temprano por profundidad esta DISPONIBLE.
 *
 * zk = 0% es ambiguo: puede ser "no lo mata nada" o "ni siquiera se intenta"
 * (con galga y D24S8 activos, early_z_safe lo desactiva). Este contador separa
 * los dos casos: si sale 0% con af/df altos, el problema es que la galga esta
 * bloqueando el early-Z, y hay margen para extenderlo sin cambiar resultados.
 */
inline std::atomic<u32> tri_early_z{0};

/**
 * Desglose del tiempo de CADA TRIANGULO dentro del rasterizador, en
 * MICROsegundos de reloj de pared (solo el hilo que llama, que es el que manda
 * los triangulos):
 *
 *   tri_setup_us  = recortar, coordenadas de pantalla, programa TEV, cache de
 *                   texturas... todo lo que se hace antes de rasterizar.
 *   tri_wait_us   = el tramo paralelo entero: encolar las bandas de los otros
 *                   nucleos, rasterizar la propia y esperar a que acaben.
 *   tri_single_us = triangulos que se rasterizan enteros en el hilo que llama
 *                   (los pequenos), con los otros nucleos parados.
 *
 * Decide el siguiente movimiento: si 'wait' manda, el trabajo por pixel es el
 * problema; si 'single' manda, hay triangulos pequenos que paralelizar; si
 * 'setup' manda, es el coste por triangulo.
 *
 * La unidad va en el nombre a proposito: sceKernelGetProcessTimeWide devuelve
 * MICROsegundos, mientras que GxStats (steady_clock) acumula NANOsegundos.
 * Mezclarlos al dividir da un factor 1000 de error, y eso es exactamente lo que
 * hace falta evitar al comparar estos contadores con GxStats::cmdlist_ns.
 */
inline std::atomic<u64> tri_setup_us{0};
inline std::atomic<u64> tri_wait_us{0};
inline std::atomic<u64> tri_single_us{0};

/**
 * Tiempo que los nucleos pasan DENTRO de una banda, sumado sobre todos ellos.
 *
 * Es la medida que falta para saber si el problema esta en el coste por pixel o
 * en el reparto. 'esp' dice que el 98% del rasterizador se va en el tramo
 * paralelo, pero ese tramo mezcla dos cosas muy distintas: sombrear pixeles y
 * esperar a que el planificador despierte a los hilos.
 *
 * Dividiendo esto entre (raster_bands * tri_wait_us) sale la OCUPACION: que
 * porcentaje del tiempo de los tres nucleos se gasta de verdad sombreando.
 *
 *   - Cerca del 100%: los nucleos estan llenos y lo caro es el pixel. Ahi el
 *     siguiente paso es el bucle (SIMD de 4 pixeles, especializacion por
 *     formato...).
 *   - Bastante por debajo: se esta pagando sincronizacion, y lo que hay que
 *     arreglar es el reparto, no la aritmetica.
 *
 * Se han estado deduciendo ciclos por pixel suponiendo ocupacion perfecta. Esto
 * quita la suposicion.
 */
inline std::atomic<u64> band_busy_us{0};

/**
 * Bandas en las que se parte un triangulo grande, o sea nucleos que trabajan a
 * la vez. Lo publica el constructor del rasterizador; no se pone a cero entre
 * intervalos porque no es una medida, es el divisor de la ocupacion.
 */
inline std::atomic<u32> raster_bands{3};

/**
 * Contadores de una TAREA del rasterizador (una banda de lineas, o un triangulo
 * entero si va en un solo hilo).
 *
 * Antes se sumaban a los atomicos al final de CADA LINEA: siete
 * lecturas-escritura atomicas sobre las mismas lineas de cache, desde los tres
 * hilos a la vez, decenas de miles de veces por fotograma. Acumulando en la
 * pila y volcando una vez por tarea, el mismo dato con una fraccion del trafico
 * y de la contencion.
 */
struct ScanCounters {
    u32 tested{};
    u32 drawn{};
    u32 covered{};
    u32 zkill{};
    u32 alpha_fail{};
    u32 depth_fail{};
    u32 lit{};

    void Flush() const {
        pixels_tested.fetch_add(tested, std::memory_order_relaxed);
        pixels_drawn.fetch_add(drawn, std::memory_order_relaxed);
        pixels_covered.fetch_add(covered, std::memory_order_relaxed);
        pixels_zkill.fetch_add(zkill, std::memory_order_relaxed);
        pixels_alpha_fail.fetch_add(alpha_fail, std::memory_order_relaxed);
        pixels_depth_fail.fetch_add(depth_fail, std::memory_order_relaxed);
        pixels_lit.fetch_add(lit, std::memory_order_relaxed);
    }
};

/**
 * Fotogramas completados dentro del intervalo de medida.
 *
 * Se cuenta aqui en vez de deducirlo del FPS que calcula PerfStats. A 0,4 FPS,
 * en la ventana de un segundo del overlay apenas cabe medio fotograma, y dividir
 * los contadores por un FPS fraccionario convierte el ruido del muestreo en
 * numeros enormes: es lo que hacia que el overdraw saltara de 12x a 66x sin que
 * necesariamente hubiera cambiado tanto la escena.
 *
 * Contando fotogramas de verdad, la division es exacta.
 */
inline std::atomic<u32> frames{0};

inline void Reset() {
    triangles.store(0, std::memory_order_relaxed);
    pixels_tested.store(0, std::memory_order_relaxed);
    pixels_drawn.store(0, std::memory_order_relaxed);
    pixels_covered.store(0, std::memory_order_relaxed);
    pixels_zkill.store(0, std::memory_order_relaxed);
    pixels_alpha_fail.store(0, std::memory_order_relaxed);
    pixels_depth_fail.store(0, std::memory_order_relaxed);
    pixels_lit.store(0, std::memory_order_relaxed);
    tex_units_total.store(0, std::memory_order_relaxed);
    tex_units_etc1.store(0, std::memory_order_relaxed);
    for (auto& count : tex_format_units) {
        count.store(0, std::memory_order_relaxed);
    }
    tev_stages_used.store(0, std::memory_order_relaxed);
    tri_split.store(0, std::memory_order_relaxed);
    tri_single.store(0, std::memory_order_relaxed);
    tri_early_z.store(0, std::memory_order_relaxed);
    tri_setup_us.store(0, std::memory_order_relaxed);
    tri_wait_us.store(0, std::memory_order_relaxed);
    tri_single_us.store(0, std::memory_order_relaxed);
    band_busy_us.store(0, std::memory_order_relaxed);
    frames.store(0, std::memory_order_relaxed);
}
} // namespace RasterizerStats

/**
 * Salto de fotogramas: rasterizar uno de cada N.
 *
 * La CPU y la GPU del 3DS se emulan EN SERIE: el hilo principal manda los
 * triangulos y se queda esperando a que los hilos del rasterizador acaben. Con
 * el rasterizado llevandose el 94% del fotograma, cada fotograma que se ve
 * cuesta lo mismo que un fotograma entero de logica del juego.
 *
 * Saltandose el dibujado de N-1 de cada N fotogramas, esos fotogramas solo
 * cuestan la emulacion de la CPU. El juego AVANZA varias veces mas rapido
 * aunque la imagen se refresque igual de poco -- que a esta velocidad es
 * justo lo que interesa: la diferencia entre una pantalla de carga eterna y
 * ver el juego progresar.
 *
 * En los fotogramas saltados no se toca screen_infos, asi que la pantalla
 * conserva la ultima imagen completa en vez de parpadear con medio dibujo.
 */
namespace FrameSkip {
/// Fotogramas emulados por cada uno rasterizado. 1 = dibujar todos.
inline std::atomic<u32> interval{3};
inline std::atomic<u32> counter{0};
inline std::atomic<bool> render_current{true};

constexpr u32 kMinInterval = 1;
constexpr u32 kMaxInterval = 10;

/**
 * Media resolucion vertical: sombrear una linea de cada dos.
 *
 * La linea sombreada se escribe tambien en la de abajo, asi que el framebuffer
 * del juego se rellena entero y conserva su tamano y su disposicion. Eso es
 * deliberado: bajar la resolucion de verdad obligaria a tocar el framebuffer
 * que el juego reserva y lee, y hay titulos que lo leen de vuelta.
 *
 * Se ahorra la mitad del sombreado -- que es la parte cara -- y no se ahorra la
 * escritura, que se hace dos veces. Sale alrededor de un 65% mas rapido.
 *
 * La profundidad solo se escribe en las lineas sombreadas. No hace falta en las
 * copiadas: como ningun triangulo posterior va a sombrear ahi tampoco, nadie
 * llega a leer esa profundidad.
 *
 * Lo enciende y apaga el usuario con L+R. Es una perdida de calidad visible y
 * no es cosa mia decidir si compensa.
 */
inline std::atomic<bool> half_resolution{false};

inline bool ShouldRender() {
    return render_current.load(std::memory_order_relaxed);
}

/// Se llama al cerrar cada fotograma emulado; decide si toca dibujar el siguiente.
inline void AdvanceFrame() {
    const u32 n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    const u32 iv = interval.load(std::memory_order_relaxed);
    render_current.store(iv <= 1 || (n % iv) == 0, std::memory_order_relaxed);
}
} // namespace FrameSkip

/**
 * Ablacion de partes del rasterizador, para MEDIR cuanto cuesta cada una.
 *
 * El overlay dice "gpu 80%" pero no donde se va: texturas, TEV, mezcla o
 * interpolacion. Se puede deducir de los contadores, pero se ha fallado dos
 * veces seguidas adivinando (ETC1 y luz resultaron ser 0%). En vez de seguir
 * adivinando, cada modo salta una parte del bucle de pixeles y el FPS que se
 * mida en pantalla dice el coste de esa parte, sobre la escena real.
 *
 * Los modos rompen la imagen a proposito. Se recorren con SELECT+ARRIBA.
 */
namespace Ablation {
inline std::atomic<u32> mode{0};

constexpr u32 kNormal = 0;
constexpr u32 kNoTexture = 1;       ///< no se muestrean texturas
constexpr u32 kNoTev = 2;           ///< no se ejecuta el TEV
constexpr u32 kNoMerger = 3;        ///< sin prueba de alfa, niebla, profundidad ni mezcla
constexpr u32 kNoInterpolation = 4; ///< atributos fijos (sin interpolar por pixel)
constexpr u32 kNoWrite = 5;         ///< se calcula todo pero no se escribe el pixel
constexpr u32 kMax = 5;

inline const char* Name(u32 value) {
    switch (value) {
    case kNoTexture:
        return "sin tex";
    case kNoTev:
        return "sin tev";
    case kNoMerger:
        return "sin merger";
    case kNoInterpolation:
        return "sin interp";
    case kNoWrite:
        return "sin escritura";
    default:
        return "normal";
    }
}
} // namespace Ablation

struct Vertex;

/**
 * Obliga a meter en linea la cadena de sombreado de un pixel.
 *
 * Medido en el binario de la 0.0.3.8, desensamblando el bucle de linea:
 * TextureColor, ComputeTevAlpha, WriteTevConfig, PixelColor y DoAlphaTest
 * salian como LLAMADAS de verdad, aun con -O3 y con LTO. GCC no las mete en
 * linea porque miden entre 2 y 3 KB cada una, muy por encima de sus umbrales
 * automaticos; ni -O3 ni LTO suben esos umbrales, solo deciden mejor con lo que
 * ya cabe.
 *
 * Y son CINCO llamadas por pixel. Lo caro no es el salto: en ARM r0-r3 y d0-d7
 * pertenecen a quien llama, asi que en cada frontera hay que bajar a la pila y
 * recargar los flotantes vivos del bucle -- las baricentricas, el 1/w, las
 * escalas de profundidad. En ese desensamblado, 432 de las 1393 instrucciones
 * del bucle de linea eran accesos a la pila: el 31%. Encima TextureColor
 * devuelve 16 bytes por puntero oculto, o sea que texture_color[] baja a
 * memoria y vuelve tres veces por pixel en vez de quedarse en registros.
 *
 * Meterlas en linea no cambia ninguna operacion de coma flotante ni su orden
 * -- no hay -ffast-math en ningun sitio del proyecto --, asi que el pixel sale
 * byte a byte igual. Lo unico que cambia es donde viven los valores.
 *
 * El precio es tamano de codigo: el bucle crece unos 13 KB. Hay que vigilar el
 * hueco entre segmentos que necesita vita-elf-create despues de cada build
 * (ver --section-start en el CMakeLists raiz).
 *
 * Son privadas de RasterizerSoftware y solo se usan desde sw_rasterizer.cpp,
 * asi que declararlas 'inline' es seguro: no hay ninguna otra unidad de
 * compilacion que las llame y se quede sin definicion. LookupTexelInTile NO
 * puede llevar esto por lo contrario: surface_base.cpp la llama a traves de
 * LookupTexture y necesita que conserve enlace externo.
 */
#if defined(__GNUC__)
#define SW_PIXEL_INLINE inline __attribute__((always_inline))
#else
#define SW_PIXEL_INLINE inline
#endif

/**
 * Configuracion TEV del triangulo, "compilada" una vez por triangulo.
 *
 * El bucle de pixeles leia los registros TEV campo a campo -- los tres
 * operandos de color, los tres de alfa, sus modificadores y las operaciones
 * son uniones de campos de bits, asi que cada uno es una carga mas un
 * desplazamiento y una mascara -- y todo eso se repetia en CADA pixel y en
 * CADA etapa, cuatro veces por pixel de media. Son mas de veinte extracciones
 * de bits por etapa y pixel solo para volver a leer la misma configuracion.
 *
 * Aqui se resuelve todo eso una vez por triangulo: indices ya normalizados a la
 * tabla de fuentes, operaciones y modificadores como bytes, multiplicadores ya
 * aplicados, y banderas que dicen exactamente que hay que recalcular en el
 * bucle. El remapeo de la etapa 0 (color_source1/2 == Previous apunta a
 * color_source3) queda aplicado aqui, que es donde el bucle original lo
 * comprobaba con dos comparaciones por etapa y pixel.
 *
 * Se conservan las seis etapas aunque solo se usen cuatro: el buffer del
 * combinador rota en TODAS las etapas, incluidas las inertes, y saltarse esa
 * rotacion desincronizaria a las siguientes. Las inertes simplemente no
 * calculan nada.
 */
struct TevProgram {
    struct Stage {
        u8 color_op{};
        u8 alpha_op{};
        u8 color_modifier[3]{};
        u8 alpha_modifier[3]{};
        /// Indices (0-15) en la tabla de fuentes del bucle, ya remapeados.
        u8 color_source[3]{};
        u8 alpha_source[3]{};
        /// Operandos que usa de verdad la operacion (1 a 3). Los que no se usan
        /// ni se cargan ni se modifican.
        u8 color_sources{};
        u8 alpha_sources{};
        u8 color_multiplier{1};
        u8 alpha_multiplier{1};
        bool active{};
        bool color_passthrough{};
        bool alpha_passthrough{};
        /// La operacion de color es Dot3_RGBA: el alfa sale del resultado de
        /// color y el combinador de alfa ni se ejecuta.
        bool alpha_from_color{};
        /// Que fuentes variables usa la etapa. Solo se refrescan en la tabla
        /// las que hacen falta.
        bool uses_previous{};
        bool uses_previous_buffer{};
        bool uses_constant{};
        bool updates_buffer_color{};
        bool updates_buffer_alpha{};
        Common::Vec4<u8> constant{};
    };

    std::array<Stage, 6> stages{};
    /// Valor inicial del buffer del combinador (tev_combiner_buffer_color).
    Common::Vec4<u8> buffer_color{};

    /**
     * Que unidades de textura lee ALGUIEN de verdad.
     *
     * Una unidad puede estar habilitada y no usarla nadie: el TEV es el unico
     * consumidor de texture_color[] (mas los selectores de sombra y bump de la
     * iluminacion), y los juegos dejan unidades encendidas que sus etapas no
     * consultan. Muestrear una textura cuesta, por pixel, el calculo de
     * coordenadas, el wrap, la decodificacion del texel y una llamada -- y todo
     * eso se tiraba a la basura si ninguna etapa activa leia su salida.
     *
     * Se resuelve en CompileTevProgram junto con el resto del programa.
     */
    bool uses_texture[4]{};

    /// Alguna fuente activa del TEV es el color interpolado del vertice. Si no
    /// la hay, interpolar y redondear cuatro canales por pixel es trabajo tirado.
    bool uses_primary_color{};

    /**
     * La cadena de alfa se puede calcular ANTES que la de color.
     *
     * El combinador de alfa es una cadena aparte: toma sus propias fuentes
     * (texturas, color de vertice, constantes) y solo se cruza con el color por
     * 'Previous'/'PreviousBuffer', y en ese caso porque lee el canal ALFA de la
     * salida anterior -- que la propia cadena de alfa ya conoce. Cuando eso se
     * cumple, se puede calcular el alfa del pixel entero sin tocar el color,
     * pasar la prueba de alfa, y AHORRARSE la cadena de color entera (mas
     * niebla, mezcla y escritura) de los pixeles que mueren en la prueba.
     *
     * Medido: el 26% de los pixeles cubiertos en NSMB2 muere en la prueba de
     * alfa despues de sombrearse del todo. Esto es lo que evita ese trabajo.
     *
     * Se pierde cuando una etapa mezcla color y alfa (Dot3_RGBA) o cuando un
     * modificador de alfa lee rojo/verde/azul de la salida anterior.
     */
    bool alpha_pure{};

    /**
     * Ultima etapa activa que ESCRIBE alfa. La pasada de alfa solo necesita
     * llegar hasta ahi: las siguientes no la tocan.
     *
     * Importa porque en la mayoria de configuraciones el alfa se fija en la
     * etapa 0 o 1 y las demas lo dejan pasar; cortando ahi, la pasada de alfa
     * (que se hace para poder descartar pixeles antes de calcular el color)
     * cuesta una o dos etapas en vez de seis.
     */
    u8 last_alpha_stage{};

    /// Ultima etapa ACTIVA. Las siguientes no calculan nada, asi que las dos
    /// pasadas (alfa y color) pueden parar aqui: su contabilidad del buffer del
    /// combinador no la lee nadie.
    u8 last_active_stage{};
};

class RasterizerSoftware : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerSoftware(Memory::MemorySystem& memory, Pica::PicaCore& pica);

    void AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                     const Pica::OutputVertex& v2) override;
    void DrawTriangles() override {}
    void FlushAll() override {}
    void FlushRegion(PAddr addr, u32 size) override {}
    void InvalidateRegion(PAddr addr, u32 size) override {}
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override {}
    void ClearAll(bool flush) override {}

private:
    /// Computes the screen coordinates of the provided vertex.
    void MakeScreenCoords(Vertex& vtx);

    /// Processes the triangle defined by the provided vertices.
    void ProcessTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                         bool reversed = false);

    /**
     * Datos de una unidad de textura que no cambian en todo el triangulo.
     *
     * Resolver el puntero de la textura (memory.GetPhysicalPointer: busqueda de
     * region fisica + desreferencia de shared_ptr + llamada fuera de linea) y
     * construir el TextureInfo desde los registros se hacia POR PIXEL y por
     * unidad. Con 2,1 millones de pixeles sombreados por fotograma y hasta 3
     * unidades, eran millones de llamadas para recalcular siempre lo mismo.
     *
     * Se precalcula una vez por triangulo y se pasa al sombreado. No hay estado
     * compartido mutable: los tres hilos del rasterizador solo leen.
     */
    struct TextureUnitCache {
        /// Direccion con la que se resolvio 'data'. Las texturas de cubo cambian
        /// de direccion por pixel, asi que hay que comprobar que sigue valiendo.
        PAddr base_address{};
        const u8* data{};
        Pica::Texture::TextureInfo info{};
        /// Ancho y alto ya en f24. Salen de un campo de bits de los registros,
        /// asi que cada lectura por pixel era extraer bits, pasar a float y
        /// truncar la mantisa a f24, tres veces por pixel para recalcular una
        /// constante del triangulo.
        Pica::f24 width{};
        Pica::f24 height{};

        /**
         * Configuracion del muestreo resuelta por triangulo.
         *
         * El bucle releia de los registros, POR PIXEL y por unidad, el tipo de
         * textura, los modos de wrap, los modos de borde y las dimensiones --
         * todos campos de bits que no cambian dentro de un triangulo. Aqui
         * quedan como bytes y enteros listos para comparar.
         */
        PAddr address{};    ///< Direccion fisica configurada en los registros
        u32 width_int{};
        u32 height_int{};
        u8 type{};          ///< TextureConfig::TextureType
        u8 wrap_s{};        ///< TextureConfig::WrapMode
        u8 wrap_t{};
        u8 border_mode_s{}; ///< 0 = sin borde, 1 = ClampToBorder, 2 = ClampToBorder2
        u8 border_mode_t{};
        /// Indice de uv que usa esta unidad (la 2 puede leer las de la 1).
        u8 coord_index{};
        /**
         * Dimension potencia de dos, y su mascara (size - 1).
         *
         * El modulo de las coordenadas fuera de rango (Repeat, Repeat2/3,
         * MirroredRepeat y el caso negativo de ClampToEdge2) es una llamada a
         * __aeabi_uidivmod con el divisor en un registro: decenas de ciclos,
         * dos veces por unidad y por pixel. Con una textura potencia de dos
         * -- lo normal -- el mismo resultado sale de un AND.
         */
        u32 width_mask{};
        u32 height_mask{};
        bool wrap_s_pow2{};
        bool wrap_t_pow2{};
        /// La unidad 0 es de sombra (Shadow2D/ShadowCube): su texel se
        /// post-procesa con shadow_z.
        bool is_shadow{};
        Common::Vec4<u8> border_color{};
    };

    /// Returns the texture color of the currently processed pixel.
    ///
    /// 'used_units' viene del programa TEV (TevProgram::uses_texture): muestrear
    /// una unidad que nadie lee era trabajo entero tirado, y ademas su cache no
    /// esta rellena, asi que hay que saltarla aqui tambien para no muestrear con
    /// un TextureInfo a cero.
    SW_PIXEL_INLINE std::array<Common::Vec4<u8>, 4> TextureColor(
        std::span<const Common::Vec2<f24>, 3> uv,
        std::span<const Pica::TexturingRegs::FullTextureConfig, 3> textures, f24 tc0_w,
        const std::array<TextureUnitCache, 3>& tex_cache,
        Pica::Texture::Etc1BlockCache& etc1_cache, const bool (&used_units)[4]) const;

    /**
     * Estado de mezcla resuelto una vez por triangulo.
     *
     * Los registros de mezcla no cambian dentro de un triangulo, pero el bucle
     * de pixeles los releia en cada pixel para elegir factores y ecuaciones.
     * Aqui queda lo unico que hace falta para dos atajos que cubren la inmensa
     * mayoria del 2D:
     *
     *   - standard_src_alpha: mezcla (SourceAlpha, OneMinusSourceAlpha) con
     *     ecuacion Add en RGB y alfa. En el bucle se puede resolver el color
     *     resultante sin pasar por los ocho despachos de factor.
     *   - all_channels_enabled: los cuatro canales de color se escriben, que es
     *     lo que permite saltarse la escritura entera cuando el resultado seria
     *     el mismo que ya habia.
     */
    struct BlendState {
        bool standard_src_alpha = false;
        bool all_channels_enabled = false;
    };

    /// Returns the final pixel color with blending or logic ops applied.
    SW_PIXEL_INLINE Common::Vec4<u8> PixelColor(u16 x, u16 y, Common::Vec4<u8> combiner_output,
                                const BlendState& blend) const;

    /// Emulates the TEV configuration and returns the combiner output.
    ///
    /// Si 'stage_alpha' no es null, la cadena de alfa ya viene calculada de
    /// ComputeTevAlpha (ver TevProgram::alpha_pure) y aqui solo se ejecuta la
    /// de color, reutilizando esos valores.
    SW_PIXEL_INLINE Common::Vec4<u8> WriteTevConfig(std::span<const Common::Vec4<u8>, 4> texture_color,
                                    const TevProgram& program, Common::Vec4<u8> primary_color,
                                    Common::Vec4<u8> primary_fragment_color,
                                    Common::Vec4<u8> secondary_fragment_color,
                                    const std::array<u8, 6>* stage_alpha = nullptr);

    /**
     * Ejecuta SOLO la cadena de alfa del TEV y devuelve el alfa que queda tras
     * cada una de las seis etapas.
     *
     * Con eso se puede pasar la prueba de alfa antes de calcular el color (ver
     * TevProgram::alpha_pure), y los valores alimentan despues a WriteTevConfig
     * para que los modificadores de color que lean el canal alfa vean lo mismo
     * que verian en la pasada unica.
     */
    SW_PIXEL_INLINE std::array<u8, 6> ComputeTevAlpha(std::span<const Common::Vec4<u8>, 4> texture_color,
                                      const TevProgram& program,
                                      Common::Vec4<u8> primary_color,
                                      Common::Vec4<u8> primary_fragment_color,
                                      Common::Vec4<u8> secondary_fragment_color) const;

    /// Compila la configuracion TEV del triangulo. Ver TevProgram.
    TevProgram CompileTevProgram(
        std::span<const Pica::TexturingRegs::TevStageConfig, 6> tev_stages,
        u32 active_tev_stages) const;

    /// Blends fog to the combiner output if enabled.
    void WriteFog(float depth, Common::Vec4<u8>& combiner_output) const;

    /// Performs the alpha test. Returns false if the test failed.
    SW_PIXEL_INLINE bool DoAlphaTest(u8 alpha) const;

    /// Performs the depth stencil test. Returns false if the test failed.
    ///
    /// 'known_ref_z' y 'known_stencil' se pasan cuando EarlyDepthPasses ya leyo
    /// esos mismos valores del framebuffer: la prueba final se ahorra la segunda
    /// lectura. El resto (comparaciones, acciones de galga, escritura de
    /// profundidad) se comporta igual.
    bool DoDepthStencilTest(u16 x, u16 y, float depth, const u32* known_ref_z = nullptr,
                            const u8* known_stencil = nullptr) const;

    /**
     * Solo las COMPARACIONES de galga y profundidad, sin escribir nada.
     *
     * Sirve para descartar un pixel tapado antes de sombrearlo. La prueba final
     * (DoDepthStencilTest) se sigue llamando despues y es la que escribe; esta
     * solo decide si merece la pena sombrear.
     *
     * Es valido saltarse el sombreado cuando las acciones de fallo no escriben
     * nada (Keep): el pixel muere igual, con o sin sombreado. Con acciones que
     * escriben, en cambio, el rechazo temprano se comeria ese efecto. Ver
     * early_z_safe en ProcessTriangle.
     *
     * 'ref_z' y 'stencil' devuelven los valores ya leidos del framebuffer, que
     * la prueba final reutiliza: asi el rechazo temprano no cuesta lecturas
     * extra cuando no llega a matar ningun pixel.
     */
    bool EarlyDepthPasses(u16 x, u16 y, float depth, u32& ref_z, u8& stencil) const;

private:
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;
    Pica::RegsInternal& regs;
    /**
     * Nucleos que rasterizan a la vez un triangulo grande, o sea en cuantas
     * bandas se parte. NO es el tamano del pool: una de las bandas la hace el
     * propio hilo que llama, asi que el pool tiene una menos.
     */
    std::size_t num_sw_threads;

    /**
     * Pool con estado POR HILO: cada worker recibe su propia cache de bloques
     * ETC1 al arrancar (StatefulThreadWorker) y la usa en exclusiva.
     *
     * Es la alternativa a un cache compartido: no hay cerrojos ni atomizados
     * en el bucle de pixeles, y tampoco hace falta TLS (que en este toolchain
     * se emula con un mutex por acceso, ver el comentario de tex_units_total
     * en este mismo fichero). La coherencia no depende de nadie porque la
     * clave de la cache es el CONTENIDO del bloque.
     *
     * Son num_sw_threads - 1 hilos, atados a los nucleos 1 y 2. El nucleo 0 lo
     * ocupa el hilo de emulacion, que es quien manda los triangulos y ahora
     * rasteriza tambien una banda en vez de quedarse dormido esperando (ver el
     * reparto en ProcessTriangle).
     */
    Common::StatefulThreadWorker<Pica::Texture::Etc1BlockCache> sw_workers;

    /// Cache del hilo que llama. La usan tanto los triangulos que se rasterizan
    /// enteros en el (los pequenos) como su banda de los triangulos grandes.
    Pica::Texture::Etc1BlockCache caller_etc1_cache;

    Framebuffer fb;
};

} // namespace SwRenderer
