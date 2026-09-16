# Azahar para PlayStation Vita

Port del emulador de Nintendo 3DS [Azahar](https://github.com/azahar-emu/azahar)
a PS Vita real (ARMv7, Cortex-A9).

---

## Cómo instalarlo

1. Copia `azahar.vpk` a la Vita e instálalo con VitaShell.
2. Crea la carpeta `ux0:/data/azahar/roms` (la app la crea sola al arrancar).
3. Mete ahí los juegos: `.3ds`, `.cci`, `.cxi`, `.app`, `.cia`, `.3dsx`.
   **Tienen que estar descifrados**, igual que en Azahar de PC.
4. Abre Azahar desde la LiveArea y elige un juego de la lista.
5. (Opcional) Para la presentación por GXM, copia `libshacccg.suprx` (extraído
   del firmware de tu propia consola, como en cualquier homebrew que compile
   shaders) a `ur0:/data/`. Si no está, el emulador presenta con vita2d y lo
   dice en el overlay; lo demás funciona exactamente igual.

### Controles

| Vita | 3DS |
|---|---|
| Círculo / Cruz / Triángulo / Cuadrado | A / B / X / Y |
| Cruceta | Cruceta |
| L / R | L / R |
| Stick izquierdo | Circle Pad |
| Stick derecho | C-Stick |
| Start / Select | Start / Select |
| Pantalla táctil | Pantalla táctil del 3DS |
| **START + SELECT a la vez** | Salir al menú |

Los botones se mapean por posición, no por nombre: el botón de confirmar del
3DS (A) está a la derecha, igual que el círculo de la Vita.

ZL/ZR del New 3DS se quedan sin asignar: a la Vita se le acaban los botones y
robárselos a algo que los juegos sí usan sería peor.

En el menú de ROMs, **SELECT** alterna el motor gráfico entre la presentación
por GXM y el renderer de software completo, y la elección se aplica al
siguiente juego que se cargue.

---

## Qué esperar de rendimiento

**Va a ir muy lento.** No es un defecto del port, es aritmética:

- La Vita **no usa JIT**. El recompilador de Azahar (dynarmic) solo tiene
  backends para x86-64 y ARM64; la Vita es ARMv7 de 32 bits y no existe backend
  para ella. Se usa el intérprete `dyncom`, que ejecuta las instrucciones del
  ARM11 una a una: entre 10 y 50 veces más lento que un JIT. (Ejecutar el ARM11
  *nativamente* sobre el Cortex-A9 con un plugin en modo kernel es posible — el
  ARM11 es ARMv6K y el Cortex-A9 ARMv7-A —, pero es un proyecto aparte que no
  está hecho.)
- La PICA200 **se sigue rasterizando en la CPU**. Azahar necesita OpenGL 4.3 o
  Vulkan 1.1 para rasterizar por hardware, y la Vita da OpenGL ES 2.0 vía GXM.
  Lo que sí hace ya este port es **presentar con el chip gráfico** (backend
  `renderer_gxm`, con `sceGxmDraw`): los fotogramas que produce
  `renderer_software` se suben a texturas GXM y los dibuja la GPU de la consola.
  Llevar ahí también el rasterizado es la Fase 3 del plan de trabajo.
- Emular un ARM11 a 268 MHz con un intérprete sobre un Cortex-A9 a 444 MHz
  deja muy poco margen.

Cuenta con pocos fotogramas por segundo incluso en juegos 2D sencillos. Los
juegos comerciales grandes probablemente no sean jugables. Homebrew ligero y
juegos muy simples es lo que tiene alguna posibilidad.

El emulador sube el reloj al máximo permitido al arrancar (CPU 444 MHz,
bus 222 MHz, GPU 222 MHz).

---

## Qué no funciona

| | |
|---|---|
| **HTTPS** | La Vita no trae OpenSSL. Las peticiones cifradas fallan de forma limpia; el HTTP normal sí funciona. |
| **Multijugador** | enet se compila pero la Vita usa `sceNetCtl`, con otra API. |
| **Grabación de vídeo** | Necesita ffmpeg, que no existe aquí. |
| **Modo New 3DS** | Desactivado por defecto, pero ya no es imposible: con el modo de memoria ampliada pide ~290 MB de los ~365 disponibles, y por eso la opción exige `AZAHAR_HEAP_MB=288` y comprobar antes en la consola que el modo ampliado se concede. Sigue en `OFF` en el VPK de serie. |
| **Giroscopio / acelerómetro** | Se devuelve una consola quieta. Mantener despiertos los sensores cuesta CPU que no sobra. |
| **Cámara y micrófono** | Sin implementar. |

---

## Compilar

Necesitas el VitaSDK con **GCC 15.2** (el 10.3 antiguo no tiene C++20
completo y Azahar lo exige), CMake ≥ 3.25 y Ninja.

```bash
export VITASDK=/ruta/al/vitasdk
./build.sh                  # compila todo y deja azahar.vpk en la raíz
./build.sh citra_core       # compila solo un módulo
./build.sh clean            # borra build/
```

`build.sh` tiene las rutas de este equipo cableadas arriba; cámbialas si
mueves el proyecto.

### Si no arranca en la consola

Lo primero que hay que mirar es la memoria. El emulador pide un heap de
**216 MB** (`AZAHAR_HEAP_MB` en el `CMakeLists.txt` raíz), que cabe tanto en el
presupuesto normal (~256 MB) como en el ampliado (~365 MB) que pide
`ATTRIBUTE2=12` en el `param.sfo`. Si el firmware no arranca la aplicación con
ese atributo, se revierte con `-DAZAHAR_EXTENDED_MEMORY=OFF` sin tocar nada más
(ver los comentarios del `CMakeLists.txt` de `src/citra_vita`).

Para no adivinar: la pantalla de arranque pinta la memoria de usuario que hay de
verdad y si el modo ampliado se ha concedido (`mem usuario ... MB ... (modo
ampliado SI/no)`), y esa misma línea queda anotada en `crash.txt`.

El desglose de para qué hace falta: FCRAM del 3DS 128 MB + VRAM 6 MB + DSP
512 KB + tablas de páginas ~5 MB por proceso + caché del rasterizador y memoria
de trabajo.

---

## Estructura

```
azaharvita/
├── build.sh                  script de compilación
├── azahar.vpk                resultado
├── CMakeLists.txt            raíz del build para Vita
├── externals/                dependencias recortadas
│   ├── CMakeLists.txt        versión mínima para Vita
│   └── CMakeLists.upstream.txt  original, como referencia
└── src/
    ├── citra_vita/           frontend nativo (NUEVO)
    │   ├── main.cpp          arranque, menú de ROMs, bucle
    │   ├── vita_window.cpp   EmuWindow: presenta con GXM o vita2d
    │   ├── vita_version.h    versión del port (única fuente)
    │   └── vita_input.cpp    mandos y táctil
    ├── video_core/renderer_gxm/  backend GXM (NUEVO)
    │   ├── renderer_gxm.*        renderer de la API GXM (delega el rasterizado)
    │   ├── rasterizer_gxm.h      rasterizador: hoy reenvía al de software
    │   ├── gxm_presenter.*       dibuja las pantallas con sceGxmDraw
    │   ├── gxm_memory.*          memoria reservada y mapeada para la GPU
    │   └── gxm_pica_format.h     formato GXM de cada formato del 3DS
    ├── vita_compat/include/  shims de POSIX que la newlib no trae
    ├── audio_core/vita_sink.*  salida de audio por sceAudioOut (NUEVO)
    ├── common/  core/  video_core/  audio_core/  network/
    └── */CMakeLists.upstream.txt  originales, como referencia
```

### Qué se recortó de las dependencias

Fuera: `dynarmic`, `xbyak`, `oaknut` (no hay backend ARMv7), `glad`,
`glslang`, `spirv-*`, `sirit`, `vma`, `vulkan-headers` (sin GPU acelerada),
`sdl2`, `cubeb`, `openal` (audio nativo), `libressl`, `cpp-jwt`,
`discord-rpc`, `libusb`, `catch2`, `libyuv`, y todo el frontend de Qt.

---

## Cambios sobre el código original

Los ficheros originales de cada `CMakeLists.txt` que se tocó están guardados
al lado como `CMakeLists.upstream.txt`. Los parches sobre el código C++ van
marcados con `#ifdef __PSVITA__` para que se vean de un vistazo.

Los más relevantes:

- **`video_core/renderer_gxm/`**: backend nuevo, no un parche de los de
  escritorio. Implementa `RendererBase` y `RasterizerInterface` (que en esta
  fase delega en el de software) y presenta las pantallas con `sceGxmDraw` y
  texturas GXM. Los dos shaders de presentación son propios y se compilan en la
  consola con `libshacccg.suprx`; si no está, se presenta con vita2d.
- **`-fno-short-enums`** (en `CMakeLists.txt`). El ABI de ARM empaqueta los
  enums al byte; Citra da por hecho que ocupan 4 bytes porque mapea las
  estructuras de registros de la PICA200 sobre la memoria del 3DS campo a
  campo. Sin este flag todos los desplazamientos salen mal.
- **`-fno-pic`** (`POSITION_INDEPENDENT_CODE OFF`). `vita-elf-create` no sabe
  traducir las reubicaciones contra la GOT y falla con *Invalid relocation
  type 25*. cryptopp y soundtouch lo activan por su cuenta.
- **`common/atomic_ops.h`**: las operaciones atómicas de 128 bits vienen
  heredadas de yuzu (el Switch sí las tiene). El ARM11 del 3DS solo llega a
  64 bits y Azahar nunca las instancia, así que se excluyen cuando el
  compilador no ofrece `__int128`.
- **`vita_compat/include/cryptopp/osrng.h`**: cryptopp solo declara
  `AutoSeededRandomPool` si encuentra CryptoAPI o `/dev/urandom`. Este shim
  usa `#include_next` y lo aporta sobre el generador por hardware del kernel
  de la Vita, sin tocar las llamadas.
- **`video_core/pica/shader_setup.cpp`**: `vmaxvq_u32` solo existe en AArch64;
  se implementa con dos máximos por parejas, que es su equivalente en el NEON
  de ARMv7.

---

Azahar es GPLv2 o posterior; este port también. Ver `license.txt`.
