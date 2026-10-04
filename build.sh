#!/bin/bash
# Build de Azahar para PS Vita.
#   ./build.sh            -> configura (si hace falta) y compila todo
#   ./build.sh <target>   -> compila solo ese target (citra_common, citra_core, ...)
#   ./build.sh clean      -> borra el directorio de build
set -u

# VITASDK tiene que estar definido (export VITASDK=/ruta/al/vitasdk). cmake y
# ninja se toman del PATH. Si los tienes en otro sitio, anadelos al PATH antes.
if [ -z "${VITASDK:-}" ]; then
    echo "Define VITASDK, p. ej.: export VITASDK=/usr/local/vitasdk" >&2
    exit 1
fi
export PATH="$VITASDK/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"

if [ "${1:-}" = "clean" ]; then
    rm -rf "$BUILD"
    echo "build/ borrado"
    exit 0
fi

if [ ! -f "$BUILD/build.ninja" ]; then
    cmake -S "$ROOT" -B "$BUILD" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$VITASDK/share/vita.toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        "${@:2}" || exit 1
fi

# Ninja lanzaria un trabajo por nucleo, pero algunos ficheros de Azahar son
# enormes (arm_dyncom_interpreter.cpp, sw_rasterizer.cpp) y cada cc1plus se come
# varios GB. Con 12 a la vez el sistema se queda sin memoria y el compilador
# muere sin mensaje, dejando errores que parecen de codigo y no lo son.
JOBS="${AZAHAR_JOBS:-4}"

ninja -C "$BUILD" -j "$JOBS" ${1:+"$1"}
NINJA_RC=$?

# Diagnostico del hueco para los metadatos SCE.
#
# vita-elf-create escribe ~3,2 KB de informacion de modulo detras del segmento
# de solo lectura, en el espacio libre hasta el segmento de datos. Si no cabe,
# se sale del buffer y muere por corrupcion de heap sin imprimir nada: solo un
# "FAILED: azahar.velf" que parece cualquier otra cosa. Aqui se mide y se dice.
ELF="$BUILD/src/citra_vita/azahar"
if [ -f "$ELF" ]; then
    GAP=$("$VITASDK/bin/arm-vita-eabi-readelf" -lW "$ELF" 2>/dev/null | awk '
        /^  LOAD/ { n++; start[n] = strtonum($3); end[n] = strtonum($3) + strtonum($6) }
        END { if (n >= 2) print start[2] - end[1]; else print -1 }')
    if [ -n "$GAP" ] && [ "$GAP" -ge 0 ] 2>/dev/null; then
        if [ "$GAP" -lt 4096 ]; then
            echo
            echo "AVISO: solo quedan $GAP bytes entre el segmento de codigo y el de datos."
            echo "       vita-elf-create necesita unos 3200 para la informacion del modulo."
            echo "       Si falla al convertir a ELF de Sony, es esto: sube la alineacion"
            echo "       de -Wl,-z,max-page-size en el CMakeLists raiz."
        fi
    fi
fi

[ "$NINJA_RC" -eq 0 ] || exit 1

# Deja el VPK terminado en la raiz del proyecto para no tener que bucear en build/.
VPK="$BUILD/src/citra_vita/azahar.vpk"
if [ -f "$VPK" ]; then
    cp "$VPK" "$ROOT/azahar.vpk"
    # Guardar el ELF junto al VPK: sin el binario EXACTO que corrio en la
    # consola no se pueden traducir las direcciones de un volcado de crash, y
    # cada recompilacion las desplaza.
    cp "$BUILD/src/citra_vita/azahar" "$ROOT/azahar.elf" 2>/dev/null

    # Y una copia CON LA VERSION EN EL NOMBRE, que no se pisa.
    #
    # azahar.elf se sobrescribe en cada compilacion, asi que cuando llega un
    # volcado de una version anterior ya no hay con que traducirlo: las
    # direcciones resuelven a funciones sin relacion y no hay forma de saber que
    # el ELF no era el bueno salvo por lo absurdo del resultado. Ha pasado dos
    # veces. Estas copias ocupan unos 8 MB cada una y viven en elf/.
    # La version vive en vita_version.h desde 0.0.1.0; antes estaba escrita a
    # mano en main.cpp Y en vita_window.cpp, y las dos se desincronizaron.
    VER=$(grep -oE 'kVersion\[\] = "version [^"]+"' "$ROOT/src/citra_vita/vita_version.h" \
          | sed 's/.*version //; s/"//')
    if [ -n "$VER" ]; then
        mkdir -p "$ROOT/elf"
        cp "$BUILD/src/citra_vita/azahar" "$ROOT/elf/azahar-$VER.elf" 2>/dev/null
    fi

    echo
    echo "==> $ROOT/azahar.vpk  ($(du -h "$ROOT/azahar.vpk" | cut -f1))"
    echo "==> $ROOT/azahar.elf  (para analizar volcados de este VPK)"
    [ -n "$VER" ] && echo "==> $ROOT/elf/azahar-$VER.elf  (copia archivada de la version $VER)"
fi
