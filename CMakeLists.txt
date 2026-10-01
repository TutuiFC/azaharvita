# Azahar para PlayStation Vita
#
# Build minimo y dedicado: interprete dyncom (sin JIT, imposible en ARMv7)
# + renderer_software (sin OpenGL/Vulkan) + frontend nativo GXM/vita2d.
#
# Uso:
#   cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=$VITASDK/share/vita.toolchain.cmake
#   ninja -C build

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED ENV{VITASDK} AND NOT DEFINED VITASDK)
    message(FATAL_ERROR "Define la variable de entorno VITASDK")
endif()
if(NOT DEFINED VITASDK)
    set(VITASDK "$ENV{VITASDK}")
endif()

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# vita-elf-create no admite reubicaciones contra la GOT (ver externals/).
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

project(azaharvita LANGUAGES C CXX ASM)

include("${VITASDK}/share/vita.cmake" REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_C_STANDARD 11)

# ---------------------------------------------------------------------------
# Identidad de la app Vita
# ---------------------------------------------------------------------------
set(VITA_APP_NAME "Azahar")
set(VITA_TITLEID  "AZHR00001")
set(VITA_VERSION  "03.01")

# ---------------------------------------------------------------------------
# Recorte de funcionalidad para Vita
# ---------------------------------------------------------------------------
# ARMv7 => ni x86_64 ni arm64. Esto desactiva dynarmic, xbyak, oaknut y los
# JIT de shaders automaticamente en las listas de fuentes de upstream.
set(ARCHITECTURE "arm" CACHE STRING "" FORCE)
set(VITA ON)

set(ENABLE_SOFTWARE_RENDERER ON)
set(ENABLE_OPENGL            OFF)
set(ENABLE_VULKAN            OFF)
set(ENABLE_QT                OFF)
set(ENABLE_SDL2              OFF)
set(ENABLE_WEB_SERVICE       OFF)
set(ENABLE_SCRIPTING         OFF)
set(ENABLE_GDBSTUB           OFF)
set(ENABLE_CUBEB             OFF)
set(ENABLE_OPENAL            OFF)
set(ENABLE_LIBUSB            OFF)
set(ENABLE_ROOM              OFF)
set(ENABLE_TESTS             OFF)
set(ENABLE_DISCORD_RPC       OFF)
set(ENABLE_MICROPROFILE      OFF)
set(ENABLE_LIBRETRO          OFF)
set(ENABLE_DEVELOPER_OPTIONS OFF)
set(ENABLE_BUILTIN_KEYBLOB   ON)
set(CITRA_USE_PRECOMPILED_HEADERS OFF)
set(CITRA_WARNINGS_AS_ERRORS      OFF)

# ---------------------------------------------------------------------------
# Presupuesto de memoria
# ---------------------------------------------------------------------------
# AL DIA DE HOY el presupuesto que se pide es el AMPLIADO: src/citra_vita pone
# ATTRIBUTE2=12 en el param.sfo (opcion AZAHAR_EXTENDED_MEMORY, ahora ON), que
# sube la RAM de usuario de ~256 MB a ~365 MB. Lo de abajo describe el reparto
# dentro del presupuesto NORMAL a proposito: mientras no haya una medida hecha
# en consola del presupuesto ampliado, las cuentas se siguen haciendo contra los
# 256 MB y lo que el atributo conceda de mas es colchon, no gasto planificado.
# Ver la nota "Que hacer con los ~110 MB del modo ampliado" al final del bloque.
#
# Una aplicacion de Vita dispone de unos 256 MB de RAM de usuario, y ahi tiene
# que caber TODO:
#
#   binario (text + data)          ~7 MB
#   cache de traduccion (en BSS)   AZAHAR_TRANS_CACHE_MB
#   heap de libc                   AZAHAR_HEAP_MB
#   GXM, vita2d y el sistema       ~15 MB
#
# El heap lo reserva libc de una sola pieza antes de main(), y la cache de
# traduccion va en el BSS del ejecutable. Si la suma no cabe, la consola
# rechaza la aplicacion al lanzarla sin dar ninguna pista del motivo.
#
# El reparto entre los dos NO es indiferente, y el sintoma de tenerlo mal es
# enganoso: la aplicacion arranca sin problema y luego muere al cargar la ROM
# con "Sin memoria" (es la pantalla de bad_alloc de main.cpp, no un error de la
# consola). Eso es heap corto, no presupuesto corto.
#
# La cache de traduccion estaba en 24 MB y se demostro que sobraba: el contador
# "cap" del overlay -- vaciados por falta de sitio -- salio 0 en TODAS las
# medidas hechas en consola, sin una sola excepcion. O sea que el interprete
# nunca llego a llenarla. Se baja a 8 MB y esos 16 MB se le pasan al heap, que
# es donde si faltaban.
#
# Total sin cambios (~206 MB), asi que el arranque no corre ningun riesgo nuevo:
# es el mismo presupuesto repartido mejor. Si "cap" empieza a subir en el
# overlay, es que 8 MB se han quedado cortos; vaciar la cache no rompe nada,
# solo obliga a retraducir, pero conviene saberlo.
#
# QUE HACER CON LOS ~110 MB DEL MODO AMPLIADO
#
# El heap sube de 192 a 216 MB, y ni un mega mas por ahora. El motivo es que un
# presupuesto corto NO da ningun error legible: la consola se niega a lanzar la
# aplicacion y no dice por que. Con 216 MB la suma sigue cabiendo en los 256 MB
# del presupuesto NORMAL (216 + ~15 del binario + ~16 de GXM/vita2d + ~5 de
# pilas = ~252), asi que este VPK arranca EN LOS TRES ESCENARIOS posibles del
# atributo nuevo:
#
#   - el firmware lo acepta      -> presupuesto ~365 MB, sobran ~113
#   - el firmware lo ignora      -> presupuesto ~256 MB, sobran ~4: arranca igual
#   - el firmware lo rechaza     -> no arranca, y entonces la causa es el
#                                   atributo y solo el atributo (ver abajo)
#
# Subir el heap a 288 MB -- que es lo que pedia el README, y lo que hace falta
# para el modo New 3DS -- ata el arranque a que el atributo funcione, y eso no
# se puede afirmar sin probarlo en la consola del usuario. El paso siguiente es
# arrancar este VPK, leer en la pantalla de arranque la linea "mem usuario libre
# ... (modo ampliado ...)" y, con ESE numero, decidir el reparto definitivo.
# Mientras tanto el modo ampliado es colchon medido, no memoria ya gastada.
#
# La cache de traduccion se queda en 8 MB: el contador "cap" del overlay
# (vaciados por falta de sitio) sigue siendo la unica prueba de si se queda
# corta, y hasta hoy ha salido 0 en todas las medidas. Ampliarla "porque ahora
# cabe" seria gastar presupuesto sin un numero que lo pida.
set(AZAHAR_HEAP_MB "216" CACHE STRING "Tamano del heap de la aplicacion en MB")
set(AZAHAR_TRANS_CACHE_MB "8" CACHE STRING "Cache de traduccion del interprete en MB")

add_compile_definitions(
    ENABLE_SOFTWARE_RENDERER
    ENABLE_BUILTIN_KEYBLOB
    CITRA_VITA=1
    __PSVITA__=1
    NDEBUG

    # Tiene que ser identico en todo el proyecto: trans_cache_buf se declara
    # extern con este tamano, y si dos ficheros no coinciden el array queda
    # descuadrado y la memoria se corrompe.
    AZAHAR_TRANS_CACHE_MB=${AZAHAR_TRANS_CACHE_MB}

    # E/S de ficheros sobre sceIo en vez de stdio. SIN ESTO NO ABREN LOS ROM DE
    # MAS DE 2 GB, y el codigo que lo resuelve ya estaba escrito: file_util.cpp
    # y file_util.h llevan un camino completo bajo este nombre que nadie
    # encendia, asi que se compilaba fuera entero.
    #
    # El motivo esta en la cabecera de IOFile::m_vita_fd: en esta libc off_t son
    # 4 bytes, no hay fseeko64, y stat::st_size es del mismo tipo. Un ROM de 4
    # GB justos mide 0x100000000, que truncado a 32 bits da CERO: el emulador
    # cree que el fichero esta vacio, falla al leer la cabecera y lo reporta
    # como "Failed to determine system mode", culpando a algo sin relacion.
    # SceOff y SceIoStat::st_size si son de 64 bits.
    #
    # VA AQUI Y NO EN UN TARGET, y esto no es opcional: la macro anade un campo
    # (m_vita_fd) a IOFile. Si un fichero la ve y otro no, sizeof(IOFile) no
    # coincide entre unidades de traduccion y la memoria se corrompe sin avisar,
    # igual que con AZAHAR_TRANS_CACHE_MB de arriba.
    AZAHAR_VITA_NATIVE_IO=1
)

# ---------------------------------------------------------------------------
# Flags de compilacion
# ---------------------------------------------------------------------------
# -Os de base: la Vita carga el binario entero en RAM, y la mayor parte de este
# codigo (HLE de servicios, sistemas de ficheros, arranque) no esta en ningun
# camino caliente, asi que ahi el tamano manda.
#
# Los caminos calientes se sacan de aqui uno a uno con -O3: ver las listas de
# set_source_files_properties en src/video_core y src/core. Conviene revisarlas
# cuando algo se mida lento, porque lo que no este en ellas se compila para
# ocupar poco, y -Os apenas mete funciones en linea: en un bucle que corre
# millones de veces por fotograma eso se nota mucho mas que "unos pocos ciclos".
add_compile_options(
    -Wno-psabi
    -Wno-attributes
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-unused-but-set-variable
    -Wno-sign-compare
    -Wno-narrowing
    -fno-strict-aliasing
    -ffunction-sections
    -fdata-sections

    # El ABI de ARM empaqueta los enums al menor tamano posible (-fshort-enums).
    # Citra da por hecho que un enum ocupa 4 bytes: las estructuras de registros
    # de la PICA200 y del DSP se mapean sobre la memoria del 3DS campo a campo,
    # y con enums de 1 byte todos los desplazamientos salen mal. Hay que forzar
    # enums de tamano int aunque el SDK se compilara con "small".
    -fno-short-enums

    # Mismo motivo, otro detalle del ABI: en ARM el tipo 'char' es SIN signo por
    # defecto, mientras que en x86 (donde se desarrolla Azahar) es CON signo.
    # Cualquier sitio que compare un char con un negativo o que guarde un byte
    # de datos en un char cambia de comportamiento. Se fuerza el criterio de
    # x86, que es para el que esta escrito el codigo.
    -fsigned-char
)
string(APPEND CMAKE_CXX_FLAGS " -Wno-volatile")

# Las guardas de los "static" locales se mantienen activas, pero
# __cxa_guard_acquire/release se implementan en citra_vita/main.cpp sobre un
# mutex normal: la version de libstdc++ usa su maquinaria de espera atomica,
# que en esta consola falla. Los detalles estan junto a la implementacion.
# -g1 mete solo tablas de lineas (sin informacion de variables): es lo justo
# para que addr2line traduzca una direccion de un volcado a fichero y linea.
# vita-elf-create no copia las secciones de depuracion al SELF, asi que el VPK
# que va a la consola no engorda por esto.
set(CMAKE_C_FLAGS_RELEASE   "-Os -g1" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-Os -g1" CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_link_options(-Wl,--gc-sections)

# ---------------------------------------------------------------------------
# LTO para el camino caliente
# ---------------------------------------------------------------------------
# Solo unos pocos ficheros se compilan con -flto (ver src/video_core): el bucle
# de pixeles y todo lo que llama desde dentro. El motivo esta explicado alli.
#
# Con LTO, la fase final de optimizacion ocurre al ENLAZAR, asi que los flags
# que cambian el ABI o la semantica del lenguaje hay que repetirlos aqui: si no,
# el codigo que GCC regenera al enlazar saldria con otro criterio de enums, otro
# signo para 'char' y con alias estricto activado. Eso ultimo es lo peligroso:
# este codigo lee la memoria del 3DS a traves de punteros de distinto tipo por
# todas partes, y con alias estricto GCC da por hecho que no se solapan.
#
# ar y ranlib tienen que ser los envoltorios de GCC. Los normales no saben leer
# la representacion intermedia de LTO, y la biblioteca estatica saldria con un
# indice de simbolos incompleto: el enlazador no encontraria funciones que si
# estan y fallaria con "undefined reference" en sitios inexplicables.
set(CMAKE_AR      "${VITASDK}/bin/arm-vita-eabi-gcc-ar.exe"     CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB  "${VITASDK}/bin/arm-vita-eabi-gcc-ranlib.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_NM      "${VITASDK}/bin/arm-vita-eabi-gcc-nm.exe"     CACHE FILEPATH "" FORCE)

add_link_options(
    -flto
    -O3
    -fno-strict-aliasing
    -fno-short-enums
    -fsigned-char
)

# Hueco para los metadatos que anade vita-elf-create.
#
# La herramienta escribe la informacion del modulo y las tablas de importacion
# (~3,2 KB) justo detras del segmento de solo lectura, en el espacio que queda
# hasta el segmento de datos. Ese espacio lo decide la alineacion de pagina, o
# sea que vale entre 0 y 4 KB segun cuanto ocupe el codigo: puro azar. Cuando no
# cabe, la herramienta escribe igualmente, se sale del buffer y muere por
# corrupcion de heap (0xC0000374) sin decir una palabra -- parece un fallo del
# binario y no lo es.
#
# Subir la alineacion de pagina no vale: garantiza un hueco de entre 0 y el
# grano, no un minimo, y pasado cierto punto el enlazador prefiere fusionar los
# dos segmentos en uno solo RWX -- que la herramienta rechaza con "overlapping
# sections". Lo que si es determinista es fijar donde empieza el segmento de
# datos. .init_array es su primera seccion.
#
# CUIDADO: esta direccion NO es gratis. Se probo subirla a 0x81A00000 para tener
# mas sitio y la consola empezo a dar error de memoria al lanzar la aplicacion.
#
# La suposicion era que el hueco entre los dos segmentos es espacio de
# direcciones sin mapear y no cuesta RAM. Es falsa: el cargador de la Vita
# reserva segun el rango de direcciones virtuales del modulo, asi que separar
# los segmentos 5 MB son 5 MB menos para el heap y la cache de traduccion, y el
# presupuesto (ver arriba) ya va justo. La consola no lo explica: solo se niega
# a arrancar.
#
# O sea que esto se sube lo MINIMO imprescindible, de mega en mega, y solo
# cuando build.sh avise de que el hueco baja de 4 KB.
#
# Ha habido que subirlo una vez, en la version 0.0.1.x: con 0x81500000 el hueco
# se habia quedado en 2.584 bytes (el aviso de build.sh lo canto) y
# vita-elf-create moria al convertir el ELF. Con 0x81600000 el hueco vuelve a
# ~1 MB. El coste son ~1 MB de RAM de usuario, que es el precio declarado de
# este ajuste (ver arriba): al ser de mega en mega, nunca se paga mas de un
# mega de golpe.
#
# Y antes de subirla, comprobar cuanto ha crecido de verdad el codigo: pasar
# texture_decode.cpp y etc1.cpp a -O3 parecia que pedia megabytes y resulto que
# engordo el binario 1.296 bytes.
add_link_options(-Wl,--section-start=.init_array=0x81600000)

# La Vita necesita threads POSIX reales; newlib los trae via -lpthread.
find_package(Threads QUIET)

# ---------------------------------------------------------------------------
# Helpers que los CMakeLists de upstream esperan
# ---------------------------------------------------------------------------
function(create_target_directory_groups target)
    # no-op: solo organiza carpetas en IDEs de escritorio
endfunction()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

# Shims de POSIX que la newlib de la Vita no trae (ioctl, etc.)
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/src/vita_compat/include")

add_subdirectory(externals)
add_subdirectory(src)
