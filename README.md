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

---

## Qué esperar de rendimiento

**Va a ir muy lento.** No es un defecto del port, es aritmética:

- La Vita **no puede usar JIT**. El recompilador de Azahar (dynarmic) solo
  tiene backends para x86-64 y ARM64; la Vita es ARMv7 de 32 bits y no existe
  backend para ella. Se usa el intérprete `dyncom`, que ejecuta las
  instrucciones del ARM11 una a una: entre 10 y 50 veces más lento que un JIT.
- La Vita **no puede acelerar el gráfico**. Azahar necesita OpenGL 4.3 o
  Vulkan 1.1; la Vita da OpenGL ES 2.0 vía GXM. Se usa `renderer_software`,
  que rasteriza la PICA200 con la CPU, compitiendo por los mismos ciclos que
  el intérprete.
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
| **Modo New 3DS** | Forzado a 3DS original: los 128 MB extra de FCRAM no caben en el presupuesto de memoria de la Vita. |
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
**288 MB** (`_newlib_heap_size_user` en `src/citra_vita/main.cpp`), que solo
cabe gracias al modo de memoria ampliada que activa `ATTRIBUTE2=12` en el
`param.sfo`. Ese bloque lo reserva libc **antes** de `main()`: si no cabe, la
aplicación no llega ni a pintar nada. Si se cierra nada más abrirla, baja ese
valor (por ejemplo a 200 MB) y vuelve a compilar.

El desglose de por qué hacen falta tantos: FCRAM del 3DS 128 MB + VRAM 6 MB +
RAM extra del New 3DS 4 MB + DSP 512 KB + tablas de páginas ~5 MB por proceso
+ caché del rasterizador y memoria de trabajo.

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
    │   ├── vita_window.cpp   EmuWindow: presenta con vita2d
    │   └── vita_input.cpp    mandos y táctil
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
