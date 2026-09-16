# Dependencias externas para el port de PS Vita.
#
# Solo lo que necesita un build de interprete + software renderer.
# Fuera: dynarmic, xbyak, oaknut (no hay backend ARMv7), glad, glslang,
# spirv-*, sirit, vma, vulkan-headers (sin GPU acelerada), sdl2, cubeb,
# openal (audio nativo sceAudioOut), libressl, httplib, cpp-jwt, enet,
# discord-rpc, libusb, catch2.
#
# El original de upstream queda en CMakeLists.upstream.txt como referencia.

add_compile_options(-w)

function(target_disable_warnings target)
    target_compile_options(${target} INTERFACE -w)
endfunction()

set(EXT "${CMAKE_CURRENT_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Boost (cabeceras + serialization + iostreams)
# ---------------------------------------------------------------------------
set(BOOST_ROOT        "${EXT}/boost" CACHE STRING "")
set(Boost_INCLUDE_DIR "${EXT}/boost" CACHE STRING "")
add_library(boost INTERFACE)
target_include_directories(boost INTERFACE "${EXT}/boost")
target_disable_warnings(boost)

file(GLOB boost_serialization_SRC "${EXT}/boost/libs/serialization/src/*.cpp")
add_library(boost_serialization STATIC ${boost_serialization_SRC})
target_link_libraries(boost_serialization PUBLIC boost)

# Solo file_descriptor.cpp: lo usan FileUtil::OpenFStream y el cargador de
# claves. Se apoya en open/read/lseek/close, que la newlib de la Vita si trae.
# mapped_file.cpp se queda fuera porque necesita mmap de verdad y nada en
# Azahar lo usa por esta via.
add_library(boost_iostreams STATIC
    "${EXT}/boost/libs/iostreams/src/file_descriptor.cpp"
)
target_link_libraries(boost_iostreams PUBLIC boost)

add_library(Boost::boost         ALIAS boost)
add_library(Boost::serialization ALIAS boost_serialization)
add_library(Boost::iostreams     ALIAS boost_iostreams)

# ---------------------------------------------------------------------------
# fmt
# ---------------------------------------------------------------------------
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_TEST    OFF CACHE BOOL "" FORCE)
set(FMT_DOC     OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Crypto++ (AES/SHA para descifrado de NCCH y claves)
# ---------------------------------------------------------------------------
set(CRYPTOPP_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
set(CRYPTOPP_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
set(CRYPTOPP_INSTALL         OFF CACHE BOOL "" FORCE)
set(CRYPTOPP_DISABLE_ASM     ON  CACHE BOOL "" FORCE)
add_subdirectory(cryptopp EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# zstd
# ---------------------------------------------------------------------------
set(ZSTD_LEGACY_SUPPORT   OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS   OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED     OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC     ON  CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
add_subdirectory(zstd/build/cmake EXCLUDE_FROM_ALL)

add_library(zstd_seekable STATIC
    "${EXT}/zstd/contrib/seekable_format/zstdseek_compress.c"
    "${EXT}/zstd/contrib/seekable_format/zstdseek_decompress.c"
)
target_include_directories(zstd_seekable PUBLIC
    "${EXT}/zstd/contrib/seekable_format"
    "${EXT}/zstd/lib"
    "${EXT}/zstd/lib/common"
)
target_link_libraries(zstd_seekable PUBLIC libzstd_static)

add_library(zstd INTERFACE)
target_link_libraries(zstd INTERFACE libzstd_static zstd_seekable)
target_include_directories(zstd INTERFACE "${EXT}/zstd/lib")

# ---------------------------------------------------------------------------
# xxHash
# ---------------------------------------------------------------------------
add_library(xxhash STATIC "${EXT}/xxHash/xxhash.c")
target_include_directories(xxhash PUBLIC "${EXT}/xxHash")
add_library(xxHash::xxhash ALIAS xxhash)

# ---------------------------------------------------------------------------
# Cabeceras sueltas
# ---------------------------------------------------------------------------
add_library(dds-ktx INTERFACE)
target_include_directories(dds-ktx INTERFACE "${EXT}/dds-ktx")
target_disable_warnings(dds-ktx)

add_library(nihstro-headers INTERFACE)
target_include_directories(nihstro-headers INTERFACE "${EXT}/nihstro/include")
target_compile_options(nihstro-headers INTERFACE -Wno-invalid-specialization)
target_disable_warnings(nihstro-headers)

add_library(json-headers INTERFACE)
target_include_directories(json-headers INTERFACE "${EXT}/json")
target_disable_warnings(json-headers)

add_library(tsl_robin_map INTERFACE)
target_include_directories(tsl_robin_map INTERFACE "${EXT}/robin-map/include")
add_library(tsl::robin_map ALIAS tsl_robin_map)

# ffmpeg no existe en Vita: el codigo que lo usa se compila fuera.
add_library(library-headers INTERFACE)

# microprofile desactivado, pero las cabeceras siguen incluyendose.
add_library(microprofile INTERFACE)
target_include_directories(microprofile INTERFACE "${EXT}/microprofile")
target_compile_definitions(microprofile INTERFACE MICROPROFILE_ENABLED=0)
target_disable_warnings(microprofile)

# ---------------------------------------------------------------------------
# inih (parser de config)
# ---------------------------------------------------------------------------
add_library(inih STATIC
    "${EXT}/inih/inih/ini.c"
    "${EXT}/inih/inih/ini.h"
    "${EXT}/inih/inih/cpp/INIReader.cpp"
    "${EXT}/inih/inih/cpp/INIReader.h"
)
target_include_directories(inih INTERFACE "${EXT}/inih/inih" "${EXT}/inih/inih/cpp")

# ---------------------------------------------------------------------------
# lodepng (capturas y texturas personalizadas)
# ---------------------------------------------------------------------------
add_library(lodepng STATIC "${EXT}/lodepng/lodepng/lodepng.cpp" "${EXT}/lodepng/lodepng/lodepng.h")
target_include_directories(lodepng PUBLIC "${EXT}/lodepng/lodepng")

# ---------------------------------------------------------------------------
# Teakra (DSP del 3DS por LLE)
# ---------------------------------------------------------------------------
set(TEAKRA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(TEAKRA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(teakra EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# SoundTouch (ajuste de tono/tempo del audio)
# ---------------------------------------------------------------------------
set(INTEGER_SAMPLES OFF CACHE BOOL "" FORCE)
set(SOUNDSTRETCH    OFF CACHE BOOL "" FORCE)
set(SOUNDTOUCH_DLL  OFF CACHE BOOL "" FORCE)
add_subdirectory(soundtouch EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# faad2 (decodificador AAC para el DSP)
# ---------------------------------------------------------------------------
add_subdirectory(faad2 EXCLUDE_FROM_ALL)

# libyuv se omite: upstream solo lo usa en Android.

# ---------------------------------------------------------------------------
# getopt (lo usa el parseo de argumentos comun)
# ---------------------------------------------------------------------------
add_library(getopt STATIC "${EXT}/getopt/getopt.c" "${EXT}/getopt/getopt.h")
target_include_directories(getopt PUBLIC "${EXT}/getopt")
# Sin STATIC_GETOPT la cabecera se decanta por la rama de DLL de Windows y marca
# cada declaracion con __declspec(dllimport), que gcc no entiende: el fichero no
# llega ni a pasar de la cabecera. En la Vita todo se enlaza estatico, que es
# justo para lo que esta esa macro.
target_compile_definitions(getopt PUBLIC STATIC_GETOPT)
# Aun asi la mitad del fichero es la variante de cadenas anchas de Windows y
# llama a _wgetenv, que en newlib no existe. Aqui no lo usa nadie: solo lo
# enlazan citra_cli y citra_room, que no se compilan para Vita. Se deja definido
# por si algun dia se recuperan, pero fuera de la compilacion por defecto.
set_target_properties(getopt PROPERTIES EXCLUDE_FROM_ALL TRUE)

# ---------------------------------------------------------------------------
# Nada de codigo independiente de posicion
# ---------------------------------------------------------------------------
# Los homebrew de Vita se enlazan estaticos y vita-elf-create no sabe traducir
# las reubicaciones contra la GOT que genera -fPIC: falla con "Invalid
# relocation type 25" (R_ARM_BASE_PREL). cryptopp, soundtouch y zstd lo activan
# por su cuenta en sus propios CMakeLists, asi que hay que quitarselo despues.
foreach(_pic_target cryptopp SoundTouch libzstd_static)
    if(TARGET ${_pic_target})
        set_target_properties(${_pic_target} PROPERTIES POSITION_INDEPENDENT_CODE OFF)
    endif()
endforeach()

# ---------------------------------------------------------------------------
# open_source_archives (blobs de firmware/archivos embebidos)
# ---------------------------------------------------------------------------
add_subdirectory(open_source_archives EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# cpp-httplib (solo cabecera). Sin TLS: la Vita no trae OpenSSL y las
# funciones online no son el objetivo de este port.
# ---------------------------------------------------------------------------
add_library(httplib INTERFACE)
target_include_directories(httplib INTERFACE "${EXT}/httplib")
target_disable_warnings(httplib)

# ---------------------------------------------------------------------------
# enet (multijugador local; se enlaza pero queda inactivo en Vita)
# ---------------------------------------------------------------------------
add_subdirectory(enet EXCLUDE_FROM_ALL)
target_include_directories(enet INTERFACE "${EXT}/enet/include")
# La newlib de la Vita no define SOMAXCONN (no hay backlog de escucha real).
target_compile_definitions(enet PRIVATE SOMAXCONN=128)
