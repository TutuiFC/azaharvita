# Modulos del emulador para PS Vita.
# El original de upstream queda en CMakeLists.upstream.txt.
#
# VITA_STAGE controla hasta que modulo se compila. Sirve para avanzar el port
# por fases sin que un modulo aun sin adaptar rompa la configuracion entera.
#   1 = common   2 = +core/network   3 = +video_core   4 = +audio_core
#   5 = +frontend citra_vita (genera el VPK)
set(VITA_STAGE 5 CACHE STRING "Hasta que modulo compilar (1-5)")

include_directories(.)

include(GenerateSettingKeys)

add_subdirectory(common)

if(VITA_STAGE GREATER_EQUAL 2)
    add_subdirectory(network)
    add_subdirectory(core)
endif()
if(VITA_STAGE GREATER_EQUAL 3)
    add_subdirectory(video_core)
endif()
if(VITA_STAGE GREATER_EQUAL 4)
    add_subdirectory(audio_core)
endif()
if(VITA_STAGE GREATER_EQUAL 5)
    add_subdirectory(citra_vita)
endif()
