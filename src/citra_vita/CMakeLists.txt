# Frontend nativo de PS Vita y empaquetado del VPK.

add_executable(azahar
    main.cpp
    vita_input.cpp
    vita_input.h
    vita_jit_probe.cpp
    vita_jit_probe.h
    vita_window.cpp
    vita_window.h
)

# citra_core, video_core y audio_core se llaman entre si, asi que el enlazador
# tiene que recorrerlos en grupo: de otro modo un simbolo que video_core pide a
# citra_core despues de haber pasado por el ya no se resuelve.
target_link_libraries(azahar PRIVATE
    -Wl,--start-group citra_core video_core audio_core citra_common network -Wl,--end-group
    Boost::iostreams
    # main.cpp construye un CryptoPP::Integer al arrancar para inicializar sus
    # tablas con un solo hilo (ver el comentario alli).
    cryptopp
)

# vita2d y los stubs del sistema.
target_link_libraries(azahar PRIVATE
    vita2d
    SceDisplay_stub
    SceGxm_stub
    SceSysmodule_stub
    SceCtrl_stub
    SceTouch_stub
    SceAudio_stub
    SceCommonDialog_stub
    SceAppMgr_stub
    ScePower_stub
    ScePgf_stub
    SceNet_stub
    SceNetCtl_stub
    SceLibKernel_stub
    freetype
    png
    jpeg
    z
    m
    pthread
)

# libstdc++ usa pthread_cancel como centinela para decidir si el programa tiene
# hilos. Se define en main.cpp en vez de forzarlo con -Wl,-u, porque asi el
# enlazador no arrastra la maquinaria de cancelacion del VitaSDK. Los detalles
# estan junto a la definicion.
#
# Desactivarlo (-DAZAHAR_ENABLE_THREADS=OFF) deja a libstdc++ en modo de un solo
# hilo: la aplicacion arranca y el menu funciona, pero cargar un juego falla con
# "Enable multithreading to use std::thread". Sirve para aislar si un problema
# viene de tener los hilos activos.
option(AZAHAR_ENABLE_THREADS "Declarar soporte de hilos a libstdc++" ON)
if(AZAHAR_ENABLE_THREADS)
    target_compile_definitions(azahar PRIVATE AZAHAR_ENABLE_THREADS)
endif()

# Interceptar las funciones de hilos para anotar cual devuelve error. libstdc++
# convierte ese valor en un std::system_error sin mirarlo, y estamos viendo
# codigos que son memoria sin inicializar. Los envoltorios estan en main.cpp.
target_link_options(azahar PRIVATE
    -Wl,--wrap=pthread_create
    -Wl,--wrap=pthread_mutex_lock
    -Wl,--wrap=pthread_join
    -Wl,--wrap=pthread_detach
    # std::__throw_system_error(int): por aqui pasan TODAS las excepciones de
    # este tipo, vengan de std::thread, std::mutex o std::call_once.
    -Wl,--wrap=_ZSt20__throw_system_errori
)

# ATTRIBUTE2=12 activa el modo de memoria ampliada de la Vita: sube el
# presupuesto de RAM de usuario de ~256 MB a ~365 MB.
#
# Desactivado por defecto: es un atributo que no esta bien documentado y, si el
# firmware no lo acepta, rechaza la aplicacion al arrancar. Con AZAHAR_HEAP_MB
# por debajo del presupuesto normal no hace falta. Activalo solo si subes el
# heap por encima de ~220 MB y sabes que tu firmware lo admite.
option(AZAHAR_EXTENDED_MEMORY "Pedir el modo de memoria ampliada (ATTRIBUTE2=12)" OFF)

# vita_create_vpk guarda VITA_MKSFOEX_FLAGS en la cache de CMake, asi que el
# valor sobrevive entre configuraciones: hay que quitar el flag viejo a mano
# antes de decidir, o al desactivar la opcion seguiria colandose el de antes
# (y al activarla se duplicaria en cada pasada).
string(REGEX REPLACE "[ ]*-d[ ]+ATTRIBUTE2=[0-9]+" "" VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS}")
if(AZAHAR_EXTENDED_MEMORY)
    set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d ATTRIBUTE2=12")
endif()
set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS}" CACHE STRING "vita-mksfoex flags" FORCE)

target_compile_definitions(azahar PRIVATE AZAHAR_HEAP_MB=${AZAHAR_HEAP_MB})

# Un eboot "unsafe" puede subir el reloj de la CPU y salirse de su carpeta,
# pero la consola se niega a lanzarlo si el usuario no tiene activado
# "Enable unsafe homebrew" en los ajustes de HENkaku, y entonces falla con un
# error de arranque que no dice nada. Por defecto se genera "safe", que
# arranca en cualquier consola; el overclock se pierde, pero es preferible a
# no arrancar. Con -DAZAHAR_UNSAFE=ON se recupera.
option(AZAHAR_UNSAFE "Generar un eboot unsafe (permite overclock; exige activar unsafe homebrew)" OFF)

# VITA_MAKE_FSELF_FLAGS tambien es variable de cache, asi que el flag -s de una
# configuracion anterior sobrevive: hay que limpiarlo antes de decidir.
string(REGEX REPLACE "[ ]*-s([ ]|$)" " " VITA_MAKE_FSELF_FLAGS "${VITA_MAKE_FSELF_FLAGS}")
set(VITA_MAKE_FSELF_FLAGS "${VITA_MAKE_FSELF_FLAGS}" CACHE STRING "vita-make-fself flags" FORCE)

if(AZAHAR_UNSAFE)
    vita_create_self(azahar.self azahar UNSAFE)
else()
    vita_create_self(azahar.self azahar)
endif()

vita_create_vpk(azahar.vpk ${VITA_TITLEID} azahar.self
    VERSION ${VITA_VERSION}
    NAME ${VITA_APP_NAME}
    FILE ${CMAKE_CURRENT_SOURCE_DIR}/livearea/icon0.png sce_sys/icon0.png
    FILE ${CMAKE_CURRENT_SOURCE_DIR}/livearea/bg.png sce_sys/livearea/contents/bg.png
    FILE ${CMAKE_CURRENT_SOURCE_DIR}/livearea/startup.png sce_sys/livearea/contents/startup.png
    FILE ${CMAKE_CURRENT_SOURCE_DIR}/livearea/template.xml sce_sys/livearea/contents/template.xml
)
