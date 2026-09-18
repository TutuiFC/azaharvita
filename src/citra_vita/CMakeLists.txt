# Frontend nativo de PS Vita y empaquetado del VPK.

add_executable(azahar
    main.cpp
    vita_build_info.cpp
    vita_input.cpp
    vita_input.h
    vita_jit_probe.cpp
    vita_jit_probe.h
    vita_version.h
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
    # El backend GXM carga libshacccg.suprx con sceKernelLoadStartModule y
    # compila los shaders de presentacion con SceShaccCg. Ver
    # video_core/renderer_gxm/gxm_presenter.cpp.
    SceKernelModulemgr_stub
    SceShaccCg_stub
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
# ACTIVADO. Son ~110 MB gratis que estaban sin pedir, y el emulador vive o muere
# por la memoria: la FCRAM del 3DS sola ya son 128 MB, y el modo New 3DS (hoy
# imposible) pide otros 128.
#
# Si tu consola NO arranca la aplicacion con este VPK -- pantalla negra y vuelta
# a la LiveArea, sin mensaje --, es esto: el firmware ha rechazado el atributo.
# Se revierte en una linea, cambiando OFF aqui o compilando con
#
#     ./build.sh clean && cmake ... -DAZAHAR_EXTENDED_MEMORY=OFF
#
# y el VPK vuelve a ser exactamente el de antes: AZAHAR_HEAP_MB se ha dejado en
# 216, que cabe tambien en el presupuesto normal (ver el CMakeLists raiz), asi
# que revertir el atributo NO obliga a tocar nada mas.
#
# Para saber si ha surtido efecto no hace falta adivinar: la pantalla de
# arranque escribe la memoria de usuario libre y dice "modo ampliado SI/no"
# comparandola con el umbral de los 256 MB. Ese numero es el que decide si el
# heap puede subir a 288 MB y si el modo New 3DS es viable.
option(AZAHAR_EXTENDED_MEMORY "Pedir el modo de memoria ampliada (ATTRIBUTE2=12)" ON)

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

# Modo New 3DS. El README lo daba por imposible "porque no cabe en el
# presupuesto de memoria de la Vita", y con el modo ampliado esa frase deja de
# ser cierta por aritmetica:
#
#   FCRAM New 3DS            256 MB   (frente a 128 del 3DS original)
#   VRAM                       6 MB
#   RAM extra del New 3DS      4 MB
#   DSP                      0,5 MB
#   tablas de paginas          5 MB por proceso
#   cache del rasterizador
#   ------------------------------
#   heap necesario          ~290 MB, contra los ~365 del presupuesto ampliado
#
# O sea que CABE, y por eso existe esta opcion. Sigue en OFF, y no por
# precaucion generica: encenderla exige subir AZAHAR_HEAP_MB a 288 como minimo,
# y eso ata el arranque a que ATTRIBUTE2=12 funcione en esa consola concreta --
# cosa que nadie ha comprobado todavia. Ademas el New 3DS emula un ARM11 mas
# rapido y juegos mas pesados sobre un interprete que ya va al limite: la
# ganancia es compatibilidad (juegos exclusivos de New 3DS), nunca velocidad.
#
# El orden correcto es: arrancar este VPK, leer "modo ampliado SI" en la
# pantalla de arranque, y solo entonces compilar con
#
#     -DAZAHAR_HEAP_MB=288 -DAZAHAR_NEW_3DS=ON
option(AZAHAR_NEW_3DS "Emular un New 3DS (exige memoria ampliada y heap >= 288 MB)" OFF)
if(AZAHAR_NEW_3DS)
    if(AZAHAR_HEAP_MB LESS 288)
        message(FATAL_ERROR
            "AZAHAR_NEW_3DS=ON necesita AZAHAR_HEAP_MB >= 288 (ahora ${AZAHAR_HEAP_MB}). "
            "La FCRAM del New 3DS son 256 MB solo de memoria del invitado.")
    endif()
    if(NOT AZAHAR_EXTENDED_MEMORY)
        message(FATAL_ERROR
            "AZAHAR_NEW_3DS=ON necesita AZAHAR_EXTENDED_MEMORY=ON: 288 MB de heap no caben "
            "en el presupuesto normal de ~256 MB y la consola rechazaria la aplicacion.")
    endif()
    target_compile_definitions(azahar PRIVATE AZAHAR_NEW_3DS=1)
endif()

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
