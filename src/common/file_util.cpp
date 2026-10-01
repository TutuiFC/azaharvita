// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Copyright Dolphin Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#ifdef AZAHAR_VITA_NATIVE_IO
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#endif
#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <fmt/format.h>
#include "common/archives.h"
#include "common/assert.h"
#include "common/common_funcs.h"
#include "common/common_paths.h"
#include "common/error.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/scope_exit.h"
#include "common/string_util.h"

#ifdef _WIN32
#include <windows.h>
// windows.h needs to be included before other windows headers
#include <direct.h> // getcwd
#include <io.h>
#include <share.h>
#include <shellapi.h>
#include <shlobj.h> // for SHGetFolderPath
#include <tchar.h>
#include "common/string_util.h"

#ifdef _MSC_VER
// 64 bit offsets for MSVC
#define fseeko _fseeki64
#define ftello _ftelli64
#define fileno _fileno
typedef struct _stat64 file_stat_t;
#define fstat _fstat64
#elif defined(HAVE_LIBRETRO)
typedef struct _stat64 file_stat_t;
#else
typedef struct stat file_stat_t;
#endif

#else
#ifdef __APPLE__
#include <sys/param.h>
#endif
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <pwd.h>
#include <unistd.h>
typedef struct stat file_stat_t;
#endif

#if defined(__APPLE__)
// CFURL contains __attribute__ directives that gcc does not know how to parse, so we need to just
// ignore them if we're not using clang. The macro is only used to prevent linking against
// functions that don't exist on older versions of macOS, and the worst case scenario is a linker
// error, so this is perfectly safe, just inconvenient.
#ifndef __clang__
#define availability(...)
#endif
#include <CoreFoundation/CFBundle.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFURL.h>
#ifdef availability
#undef availability
#endif

#endif

#if defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
#include "common/android_utils.h"
#include "common/string_util.h"
#endif

#include <algorithm>
#include <sys/stat.h>

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

#ifdef HAVE_LIBRETRO_VFS
#define SKIP_STDIO_REDEFINES
#include <streams/file_stream.h>
#include <streams/file_stream_transforms.h>

#define FILE RFILE
#define FTELL rftell
#define FOPEN rfopen
#define FCLOSE rfclose
#define FSEEK rfseek
#define FREAD rfread
#define FWRITE rfwrite
#define FEOF rfeof
#define FERROR rferror
#define FFLUSH rfflush

#else

#define FTELL ftello
#define FOPEN fopen
#define FCLOSE std::fclose
#define FSEEK fseeko
#define FREAD std::fread
#define FWRITE std::fwrite
#define FEOF feof
#define FERROR ferror
#define FFLUSH std::fflush

#ifdef _MSC_VER
#define DUP_FD _dup
#define FDOPEN _fdopen
#define CLOSE_FD _close
#else
#define DUP_FD dup
#define FDOPEN fdopen
#define CLOSE_FD close
#endif

#endif

// This namespace has various generic functions related to files and paths.
// The code still needs a ton of cleanup.
// REMEMBER: strdup considered harmful!
namespace FileUtil {

using Common::GetLastErrorMsg;

// Remove any ending forward slashes from directory paths
// Modifies argument.
static void StripTailDirSlashes(std::string& fname) {
    if (fname.length() <= 1) {
        return;
    }

    std::size_t i = fname.length();
    while (i > 0 && fname[i - 1] == DIR_SEP_CHR) {
        --i;
    }
    fname.resize(i);
}

bool Exists(const std::string& filename) {
    std::string copy(filename);
    StripTailDirSlashes(copy);

#ifdef _WIN32
    struct _stat64 file_info;
    // Windows needs a slash to identify a driver root
    if (copy.size() != 0 && copy.back() == ':')
        copy += DIR_SEP_CHR;

    int result = _wstat64(Common::UTF8ToUTF16W(copy).c_str(), &file_info);
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    int result;
    if (AndroidUtils::CanUseRawFS()) {
        struct stat file_info;
        result = stat(AndroidUtils::TranslateFilePath(copy).c_str(), &file_info);
    } else {
        result = AndroidUtils::FileExists(filename) ? 0 : -1;
    }
#else
    struct stat file_info;
    int result = stat(copy.c_str(), &file_info);
#endif

    return (result == 0);
}

bool IsDirectory(const std::string& filename) {

    std::string copy(filename);
    StripTailDirSlashes(copy);

#ifdef _WIN32
    struct _stat64 file_info;
    // Windows needs a slash to identify a driver root
    if (copy.size() != 0 && copy.back() == ':')
        copy += DIR_SEP_CHR;

    int result = _wstat64(Common::UTF8ToUTF16W(copy).c_str(), &file_info);
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    struct stat file_info;
    int result;
    if (AndroidUtils::CanUseRawFS()) {
        result = stat(AndroidUtils::TranslateFilePath(copy).c_str(), &file_info);
    } else {
        return AndroidUtils::IsDirectory(filename);
    }
#else
    struct stat file_info;
    int result = stat(copy.c_str(), &file_info);
#endif

    if (result < 0) {
        LOG_DEBUG(Common_Filesystem, "stat failed on {}: {}", filename, GetLastErrorMsg());
        return false;
    }

    return S_ISDIR(file_info.st_mode);
}

bool Delete(const std::string& filepath) {
    LOG_TRACE(Common_Filesystem, "file {}", filepath);

    // Return true because we care about the file no
    // being there, not the actual delete.
    if (!Exists(filepath)) {
        LOG_DEBUG(Common_Filesystem, "{} does not exist", filepath);
        return true;
    }

    // We can't delete a directory
    if (IsDirectory(filepath)) {
        LOG_ERROR(Common_Filesystem, "Failed: {} is a directory", filepath);
        return false;
    }

#ifdef _WIN32
    // On windows, if we delete a file with an open handle (pending to be deleted) and
    // then try to open the same file name again, it will fail. On linux this doesn't happen
    // as the new file will be a different inode, even if it has the same file name.
    // The 3ds is linux-like, so to emulate this behaviour we need to rename the file
    // first to an unique name then mark it for deletion. This way we can open new files
    // with the same name. Once all handles of the old file are closed, the old file will be
    // finally deleted.
    static std::atomic<uint64_t> counter{0};

    const std::wstring wfilepath = Common::UTF8ToUTF16W(filepath);
    const DWORD pid = GetCurrentProcessId();
    const uint64_t id = counter++;

    std::wstring deleted_path =
        wfilepath + L".deleted." + std::to_wstring(pid) + L"." + std::to_wstring(id);

    // Rename first
    if (MoveFileExW(wfilepath.c_str(), deleted_path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        // Then mark file for deletion
        DeleteFileW(deleted_path.c_str());
        return true;
    }

    LOG_ERROR(Common_Filesystem, "Rename to deleted path failed on {}: {}", filepath,
              GetLastErrorMsg());

    return false;
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    if (AndroidUtils::CanUseRawFS()) {
        if (unlink(AndroidUtils::TranslateFilePath(filepath).c_str()) == -1) {
            LOG_ERROR(Common_Filesystem, "unlink failed on {}: {}", filepath, GetLastErrorMsg());
            return false;
        }
    } else {
        if (!AndroidUtils::DeleteDocument(filepath)) {
            LOG_ERROR(Common_Filesystem, "unlink failed on {}", filepath);
            return false;
        }
    }
#else
    if (unlink(filepath.c_str()) == -1) {
        LOG_ERROR(Common_Filesystem, "unlink failed on {}: {}", filepath, GetLastErrorMsg());
        return false;
    }
#endif

    return true;
}

bool CreateDir(const std::string& path) {
#ifdef AZAHAR_VITA_NATIVE_IO
    /**
     * Camino nativo: sceIoMkdir directo, sin pasar por la libc.
     *
     * Dos motivos, y los dos vienen de haber perseguido esto a ciegas.
     *
     * 1. LA BARRA FINAL. CreateFullPath construye la ruta nivel a nivel y a
     *    cada trozo le deja la barra ("substr(0, position + 1)"), asi que llega
     *    algo como ".../data/00000001/". sceIoMkdir la rechaza. Aqui se quita
     *    siempre antes de llamar, en vez de llamar, fallar y reintentar.
     *
     * 2. EL CODIGO DE ERROR DE VERDAD. Pasando por la libc, el error de Sony se
     *    aplasta a un errno generico: todo lo que no sea "ya existe" acaba como
     *    EINVAL ("Invalid argument"), que no dice nada. El codigo crudo si
     *    distingue entre ruta demasiado larga, dispositivo no montado, sin
     *    permiso o ruta mal formada, y desde la consola es lo unico que se ve.
     *
     * Por que importa tanto una carpeta: si no se puede crear la del GUARDADO,
     * el juego no puede abrir su archivo de guardado y se queda a medio cargar
     * o se rinde con la pantalla de error del 3DS. Es el fallo por el que no
     * arrancaban varios juegos.
     */
    {
        std::string dir = path;
        while (dir.size() > 1 && dir.back() == '/' && dir[dir.size() - 2] != ':') {
            dir.pop_back();
        }

        const int rc = sceIoMkdir(dir.c_str(), 0777);

        /**
         * RASTRO DE TODOS LOS NIVELES, con presupuesto.
         *
         * Hasta ahora solo se registraba el nivel que fallaba, y eso resulto
         * enganoso: si un nivel de MAS ARRIBA fallaba y quedaba tapado (ver
         * abajo), el primer error visible era el de mas abajo, que solo es la
         * consecuencia. Con el rastro completo se ve de un vistazo donde se
         * rompe la cadena de verdad.
         *
         * El presupuesto existe porque VitaNote y el registro escriben en la
         * tarjeta: unas pocas decenas de lineas bastan para ver la cadena de
         * carpetas de un juego, y a partir de ahi solo estorbarian.
         */
        static int trace_budget = 60;
        if (trace_budget > 0) {
            trace_budget--;
            LOG_INFO(Common_Filesystem, "mkdir {} -> {:#010x}", dir,
                     static_cast<unsigned int>(rc));
        }

        if (rc >= 0) {
            return true;
        }
        // 0x80010011 es EEXIST en los errores de Sony; ya existir no es fallo.
        if (static_cast<unsigned int>(rc) == 0x80010011u) {
            return true;
        }

        /**
         * LA COMPROBACION DE "PERO SI YA EXISTE" VA POR sceIoGetstat, NO POR
         * IsDirectory.
         *
         * IsDirectory usa el stat de esta libc, que es el mismo camino de 32
         * bits que no sabe de ficheros grandes y del que ya no nos fiamos. Si
         * contestara que si sobre una carpeta que NO existe, este metodo
         * devolveria exito en silencio, el nivel se daria por creado y el
         * siguiente fallaria sin que nada dijera cual falto de verdad. Que es
         * exactamente el sintoma que se estaba persiguiendo.
         *
         * sceIoGetstat es la version nativa y trae el bit de directorio.
         */
        SceIoStat st{};
        if (sceIoGetstat(dir.c_str(), &st) >= 0 && SCE_S_ISDIR(st.st_mode)) {
            return true;
        }

        LOG_ERROR(Common_Filesystem, "sceIoMkdir fallo en {} (largo {}): {:#010x}", dir,
                  dir.size(), static_cast<unsigned int>(rc));
        return false;
    }
#endif
    LOG_TRACE(Common_Filesystem, "directory {}", path);
#ifdef _WIN32
    if (::CreateDirectoryW(Common::UTF8ToUTF16W(path).c_str(), nullptr))
        return true;
    DWORD error = GetLastError();
    if (error == ERROR_ALREADY_EXISTS) {
        LOG_DEBUG(Common_Filesystem, "CreateDirectory failed on {}: already exists", path);
        return true;
    }
    LOG_ERROR(Common_Filesystem, "CreateDirectory failed on {}: {}", path, error);
    return false;
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    if (AndroidUtils::CanUseRawFS()) {
        if (mkdir(AndroidUtils::TranslateFilePath(path).c_str(), 0755) == 0)
            return true;

        int err = errno;

        if (err == EEXIST) {
            LOG_DEBUG(Common_Filesystem, "mkdir failed on {}: already exists", path);
            return true;
        }

        LOG_ERROR(Common_Filesystem, "mkdir failed on {}: {}", path, strerror(err));
        return false;
    } else {
        std::string directory = path;
        std::string filename = path;
        if (Common::EndsWith(path, "/")) {
            directory = GetParentPath(path);
            filename = GetParentPath(path);
        }
        directory = GetParentPath(directory);
        filename = GetFilename(filename);
        // If directory path is empty, set it to root.
        if (directory.empty()) {
            directory = "/";
        }
        if (!AndroidUtils::CreateDir(directory, filename)) {
            LOG_ERROR(Common_Filesystem, "mkdir failed on {}", path);
            return false;
        };
        return true;
    }

#else
    if (mkdir(path.c_str(), 0755) == 0)
        return true;

    int err = errno;

    if (err == EEXIST) {
        LOG_DEBUG(Common_Filesystem, "mkdir failed on {}: already exists", path);
        return true;
    }

#ifdef __PSVITA__
    /**
     * SEGUNDO INTENTO SIN LA BARRA FINAL.
     *
     * CreateFullPath crea la ruta nivel a nivel y a cada trozo le deja la barra
     * del final ("substr(0, position + 1)"), asi que a mkdir le llega siempre
     * algo como ".../data/00000001/". En un sistema de ficheros normal eso da
     * igual; sceIoMkdir, que es lo que hay debajo aqui, la rechaza con EINVAL.
     *
     * Y no es un detalle: la ruta que fallaba es la del GUARDADO del juego
     * (ux0:/data/azahar/sdmc/Nintendo 3DS/.../title/00040000/0007af00/data/
     * 00000001/). Sin ella, OpenArchive del archivo de guardado falla y el
     * juego se rinde con la pantalla de "An error has occurred" del 3DS. Es
     * decir: el juego no arrancaba por una barra.
     *
     * No se toca la ruta antes de llamar por primera vez a proposito. Si
     * algun dia sceIoMkdir aceptara la barra, el primer intento acierta y esto
     * no se ejecuta; y si falla por otra cosa, el registro de abajo dice por
     * cual con el codigo crudo, que es lo unico que se ve desde la consola.
     */
    if (path.size() > 1 && path.back() == '/' && path[path.size() - 2] != ':') {
        const std::string trimmed = path.substr(0, path.size() - 1);
        if (mkdir(trimmed.c_str(), 0755) == 0) {
            return true;
        }
        const int trimmed_err = errno;
        if (trimmed_err == EEXIST) {
            return true;
        }
        LOG_ERROR(Common_Filesystem,
                  "mkdir fallo en {} (largo {}): con barra {} ({}), sin barra {} ({})", path,
                  path.size(), strerror(err), err, strerror(trimmed_err), trimmed_err);
        return false;
    }
#endif

    LOG_ERROR(Common_Filesystem, "mkdir failed on {}: {}", path, strerror(err));
    return false;
#endif
}

bool CreateFullPath(const std::string& fullPath) {
    int panicCounter = 100;
    LOG_TRACE(Common_Filesystem, "path {}", fullPath);

    if (FileUtil::Exists(fullPath)) {
        LOG_DEBUG(Common_Filesystem, "path exists {}", fullPath);
        return true;
    }

#ifdef AZAHAR_VITA_NATIVE_IO
    /**
     * En Vita se crean TODOS los niveles, sin preguntar antes si existen.
     *
     * La version de abajo se salta un nivel cuando IsDirectory dice que ya
     * esta, y IsDirectory va por el stat de la libc, que en esta consola no es
     * de fiar (su st_size es de 32 bits y es el mismo camino que no sabe de
     * ficheros grandes). Si contesta que si sobre una carpeta que NO existe, el
     * nivel se salta y el siguiente falla con EINVAL -- crear algo dentro de un
     * padre inexistente -- sin decir cual era el nivel que faltaba de verdad.
     *
     * sceIoMkdir ya distingue "ya existe" (0x80010011) de un error real y
     * CreateDir lo trata como exito, asi que preguntar antes no aporta nada:
     * sale mas barato intentarlo siempre. Y si falla, el registro dice el
     * nivel EXACTO y con el codigo crudo de Sony, que es lo unico que se ve
     * desde la consola.
     */
    {
        std::size_t vita_pos = 0;
        while (true) {
            vita_pos = fullPath.find(DIR_SEP_CHR, vita_pos);
            if (vita_pos == fullPath.npos) {
                return true;
            }
            const std::string level = fullPath.substr(0, vita_pos + 1);
            // "ux0:/" y demas raices de dispositivo no se crean: existen o no,
            // pero mkdir sobre ellas no tiene sentido.
            if (level.size() > 1 && level[level.size() - 2] != ':') {
                if (!FileUtil::CreateDir(level)) {
                    LOG_ERROR(Common, "CreateFullPath: fallo creando el nivel {} de {}", level,
                              fullPath);
                    return false;
                }
            }
            vita_pos++;
        }
    }
#endif

    std::size_t position = 0;
    while (true) {
        std::size_t prev_pos = position;
        // Find next sub path
        position = fullPath.find(DIR_SEP_CHR, prev_pos);

#ifdef _WIN32
        if (position == fullPath.npos)
            position = fullPath.find(DIR_SEP_CHR_WIN, prev_pos);
#endif

        // we're done, yay!
        if (position == fullPath.npos)
            return true;

        // Include the '/' so the first call is CreateDir("/") rather than CreateDir("")
        std::string const subPath(fullPath.substr(0, position + 1));
        if (!FileUtil::IsDirectory(subPath) && !FileUtil::CreateDir(subPath)) {
            LOG_ERROR(Common, "CreateFullPath: directory creation failed");
            return false;
        }

        // A safety check
        panicCounter--;
        if (panicCounter <= 0) {
            LOG_ERROR(Common, "CreateFullPath: directory structure is too deep");
            return false;
        }
        position++;
    }
}

bool DeleteDir(const std::string& filename) {
    LOG_TRACE(Common_Filesystem, "directory {}", filename);

    // check if a directory
    if (!FileUtil::IsDirectory(filename)) {
        LOG_ERROR(Common_Filesystem, "Not a directory {}", filename);
        return false;
    }

#ifdef _WIN32
    if (::RemoveDirectoryW(Common::UTF8ToUTF16W(filename).c_str()))
        return true;
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    if (AndroidUtils::CanUseRawFS()) {
        if (rmdir(AndroidUtils::TranslateFilePath(filename).c_str()) == 0)
            return true;
    } else {
        if (AndroidUtils::DeleteDocument(filename))
            return true;
    }
#else
    if (rmdir(filename.c_str()) == 0)
        return true;
#endif
    LOG_ERROR(Common_Filesystem, "failed {}: {}", filename, GetLastErrorMsg());

    return false;
}

bool Rename(const std::string& srcFullPath, const std::string& destFullPath) {
    LOG_TRACE(Common_Filesystem, "{} --> {}", srcFullPath, destFullPath);
#ifdef _WIN32
    if (MoveFileExW(Common::UTF8ToUTF16W(srcFullPath).c_str(),
                    Common::UTF8ToUTF16W(destFullPath).c_str(), MOVEFILE_REPLACE_EXISTING)) {
        return true;
    }
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    if (AndroidUtils::CanUseRawFS()) {
        if (rename(AndroidUtils::TranslateFilePath(srcFullPath).c_str(),
                   AndroidUtils::TranslateFilePath(destFullPath).c_str()) == 0) {
            return true;
        }
    } else {
        if (AndroidUtils::MoveAndRenameFile(srcFullPath, destFullPath)) {
            return true;
        }
    }
#else
    if (rename(srcFullPath.c_str(), destFullPath.c_str()) == 0)
        return true;
#endif
    LOG_ERROR(Common_Filesystem, "failed {} --> {}: {}", srcFullPath, destFullPath,
              GetLastErrorMsg());
    return false;
}

bool Copy(const std::string& srcFilename, const std::string& destFilename) {
    LOG_TRACE(Common_Filesystem, "{} --> {}", srcFilename, destFilename);
#ifdef _WIN32
    if (CopyFileW(Common::UTF8ToUTF16W(srcFilename).c_str(),
                  Common::UTF8ToUTF16W(destFilename).c_str(), FALSE))
        return true;

    LOG_ERROR(Common_Filesystem, "failed {} --> {}: {}", srcFilename, destFilename,
              GetLastErrorMsg());
    return false;
#else
    auto copy_files = [](const std::string& src, const std::string& dst) -> bool {
        // Open input file
        FILE* input = FOPEN(src.c_str(), "rb");
        if (!input) {
            LOG_ERROR(Common_Filesystem, "opening input failed {} --> {}: {}", src, dst,
                      GetLastErrorMsg());
            return false;
        }
        SCOPE_EXIT({ FCLOSE(input); });

        // open output file
        FILE* output = FOPEN(dst.c_str(), "wb");
        if (!output) {
            LOG_ERROR(Common_Filesystem, "opening output failed {} --> {}: {}", src, dst,
                      GetLastErrorMsg());
            return false;
        }
        SCOPE_EXIT({ FCLOSE(output); });

        // copy loop
        std::array<char, 1024> buffer;
        while (!FEOF(input)) {
            // read input
            std::size_t rnum = FREAD(buffer.data(), sizeof(char), buffer.size(), input);
            if (rnum != buffer.size()) {
                if (FERROR(input) != 0) {
                    LOG_ERROR(Common_Filesystem, "failed reading from source, {} --> {}: {}", src,
                              dst, GetLastErrorMsg());
                    return false;
                }
            }

            // write output
            std::size_t wnum = FWRITE(buffer.data(), sizeof(char), rnum, output);
            if (wnum != rnum) {
                LOG_ERROR(Common_Filesystem, "failed writing to output, {} --> {}: {}", src, dst,
                          GetLastErrorMsg());
                return false;
            }
        }

        return true;
    };

#if defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    if (AndroidUtils::CanUseRawFS()) {
        return copy_files(AndroidUtils::TranslateFilePath(srcFilename),
                          AndroidUtils::TranslateFilePath(destFilename));
    } else {
        return AndroidUtils::CopyFile(srcFilename, std::string(GetParentPath(destFilename)),
                                      std::string(GetFilename(destFilename)));
    }
#else
    return copy_files(srcFilename, destFilename);
#endif
#endif
}

u64 GetSize(const std::string& filename) {
    if (!Exists(filename)) {
        LOG_ERROR(Common_Filesystem, "failed {}: No such file", filename);
        return 0;
    }

#ifdef AZAHAR_VITA_NATIVE_IO
    // stat no vale aqui: su st_size es de 32 bits y un ROM de 4 GB justos
    // (0x100000000) se trunca a CERO. sceIoGetstat lo devuelve de 64 bits.
    {
        SceIoStat st{};
        if (sceIoGetstat(filename.c_str(), &st) >= 0) {
            return static_cast<u64>(st.st_size);
        }
        LOG_ERROR(Common_Filesystem, "sceIoGetstat fallo en {}", filename);
        return 0;
    }
#endif

    if (IsDirectory(filename)) {
        LOG_ERROR(Common_Filesystem, "failed {}: is a directory", filename);
        return 0;
    }
#ifndef _WIN32
    struct stat buf;
#endif
#ifdef _WIN32
    struct _stat64 buf;
    if (_wstat64(Common::UTF8ToUTF16W(filename).c_str(), &buf) == 0)
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    if (AndroidUtils::CanUseRawFS()) {
        if (stat(AndroidUtils::TranslateFilePath(filename).c_str(), &buf) == 0) {
            return buf.st_size;
        }
    } else {
        u64 result = AndroidUtils::GetSize(filename);
        LOG_TRACE(Common_Filesystem, "{}: {}", filename, result);
        return result;
    }
#else
    if (stat(filename.c_str(), &buf) == 0)
#endif
    {
        LOG_TRACE(Common_Filesystem, "{}: {}", filename, buf.st_size);
        return buf.st_size;
    }

    LOG_ERROR(Common_Filesystem, "Stat failed {}: {}", filename, GetLastErrorMsg());
    return 0;
}

u64 GetSize(const int fd) {
    file_stat_t buf;
    if (fstat(fd, &buf) != 0) {
        LOG_ERROR(Common_Filesystem, "GetSize: stat failed {}: {}", fd, GetLastErrorMsg());
        return 0;
    }
    return buf.st_size;
}

u64 GetSize(FILE* f) {
    // can't use off_t here because it can be 32-bit
    u64 pos = FTELL(f);
    if (FSEEK(f, 0, SEEK_END) != 0) {
        LOG_ERROR(Common_Filesystem, "GetSize: seek failed {}: {}", fmt::ptr(f), GetLastErrorMsg());
        return 0;
    }
    u64 size = FTELL(f);
    if ((size != pos) && (FSEEK(f, pos, SEEK_SET) != 0)) {
        LOG_ERROR(Common_Filesystem, "GetSize: seek failed {}: {}", fmt::ptr(f), GetLastErrorMsg());
        return 0;
    }
    return size;
}

bool CreateEmptyFile(const std::string& filename) {
    LOG_TRACE(Common_Filesystem, "{}", filename);

    if (!FileUtil::IOFile(filename, "wb").IsOpen()) {
        LOG_ERROR(Common_Filesystem, "failed {}: {}", filename, GetLastErrorMsg());
        return false;
    }

    return true;
}

namespace {

#ifdef _WIN32

std::optional<std::vector<std::string>> ListDirectoryEntries(const std::string& directory) {
    std::vector<std::string> entries;

    WIN32_FIND_DATAW ffd;
    const std::wstring search_path = Common::UTF8ToUTF16W(directory + "\\*");

    HANDLE handle = FindFirstFileW(search_path.c_str(), &ffd);
    if (handle == INVALID_HANDLE_VALUE) {
        return std::nullopt;
    }

    do {
        entries.emplace_back(Common::UTF16ToUTF8(ffd.cFileName));
    } while (FindNextFileW(handle, &ffd) != 0);

    FindClose(handle);
    return entries;
}

#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)

std::optional<std::vector<std::string>> ListDirectoryEntries(const std::string& directory) {
    if (AndroidUtils::CanUseRawFS()) {
        std::vector<std::string> entries;

        DIR* dirp = opendir(AndroidUtils::TranslateFilePath(directory).c_str());
        if (!dirp)
            return std::nullopt;

        while (dirent* entry = readdir(dirp)) {
            entries.emplace_back(entry->d_name);
        }

        closedir(dirp);
        return entries;
    } else {
        return AndroidUtils::GetFilesName(directory);
    }
}

#else

std::optional<std::vector<std::string>> ListDirectoryEntries(const std::string& directory) {
    std::vector<std::string> entries;

    DIR* dirp = opendir(directory.c_str());
    if (!dirp)
        return std::nullopt;

    while (dirent* entry = readdir(dirp)) {
        entries.emplace_back(entry->d_name);
    }

    closedir(dirp);
    return entries;
}

#endif

} // anonymous namespace

bool ForeachDirectoryEntry(u64* num_entries_out, const std::string& directory,
                           DirectoryEntryCallable callback) {
    LOG_TRACE(Common_Filesystem, "directory {}", directory);

    const auto entries = ListDirectoryEntries(directory);
    if (!entries.has_value()) {
        return false;
    }

    u64 found_entries = 0;

    for (const std::string& virtual_name : *entries) {
        if (virtual_name == "." || virtual_name == "..")
            continue;

        u64 ret_entries = 0;
        if (!callback(&ret_entries, directory, virtual_name))
            return false;

        found_entries += ret_entries;
    }

    // num_entries_out is allowed to be specified nullptr, in which case we shouldn't try to set it
    if (num_entries_out)
        *num_entries_out = found_entries;

    return true;
}

u64 ScanDirectoryTree(const std::string& directory, FSTEntry& parent_entry, unsigned int recursion,
                      std::atomic<bool>* stop_flag) {
    const auto callback = [recursion, &parent_entry,
                           stop_flag](u64* num_entries_out, const std::string& directory,
                                      const std::string& virtual_name) -> bool {
        // Break early and return error if stop is requested
        if (stop_flag && *stop_flag) {
            return false;
        }

        FSTEntry entry;
        entry.virtualName = virtual_name;
        entry.physicalName = directory + DIR_SEP + virtual_name;

        if (IsDirectory(entry.physicalName)) {
            entry.isDirectory = true;
            // is a directory, lets go inside if we didn't recurse to often
            if (recursion > 0) {
                entry.size = ScanDirectoryTree(entry.physicalName, entry, recursion - 1);
                *num_entries_out += entry.size;
            } else {
                entry.size = 0;
            }
        } else { // is a file
            entry.isDirectory = false;
            entry.size = GetSize(entry.physicalName);
        }
        (*num_entries_out)++;

        // Push into the tree
        parent_entry.children.push_back(std::move(entry));
        return true;
    };

    u64 num_entries;
    return ForeachDirectoryEntry(&num_entries, directory, callback) ? num_entries : 0;
}

void GetAllFilesFromNestedEntries(FSTEntry& directory, std::vector<FSTEntry>& output) {
    std::vector<FSTEntry> files;
    for (auto& entry : directory.children) {
        if (entry.isDirectory) {
            GetAllFilesFromNestedEntries(entry, output);
        } else {
            output.push_back(entry);
        }
    }
}

bool DeleteDirRecursively(const std::string& directory, unsigned int recursion) {
    const auto callback = [recursion]([[maybe_unused]] u64* num_entries_out,
                                      const std::string& directory,
                                      const std::string& virtual_name) -> bool {
        std::string new_path = directory + DIR_SEP_CHR + virtual_name;

        if (IsDirectory(new_path)) {
            if (recursion == 0)
                return false;
            return DeleteDirRecursively(new_path, recursion - 1);
        }
        return Delete(new_path);
    };

    if (!ForeachDirectoryEntry(nullptr, directory, callback))
        return false;

    // Delete the outermost directory
    FileUtil::DeleteDir(directory);
    return true;
}

void CopyDir([[maybe_unused]] const std::string& source_path,
             [[maybe_unused]] const std::string& dest_path) {
    if (source_path == dest_path)
        return;

    if (!FileUtil::Exists(source_path))
        return;

    if (!FileUtil::Exists(dest_path))
        FileUtil::CreateFullPath(dest_path);

    const auto entries = ListDirectoryEntries(source_path);
    if (!entries.has_value() || (*entries).empty())
        return;

    for (const std::string& virtual_name : *entries) {
        if (virtual_name == "." || virtual_name == "..")
            continue;

        std::string source = source_path + virtual_name;
        std::string dest = dest_path + virtual_name;

        if (IsDirectory(source)) {
            source += '/';
            dest += '/';

            if (!FileUtil::Exists(dest))
                FileUtil::CreateFullPath(dest);

            CopyDir(source, dest);
        } else {
            if (!FileUtil::Exists(dest))
                FileUtil::Copy(source, dest);
        }
    }
}

std::optional<std::string> GetCurrentDir() {
// Get the current working directory (getcwd uses malloc)
#ifdef _WIN32
    wchar_t* dir = _wgetcwd(nullptr, 0);
    if (!dir) {
#else
    char* dir = getcwd(nullptr, 0);
    if (!dir) {
#endif
        LOG_ERROR(Common_Filesystem, "GetCurrentDirectory failed: {}", GetLastErrorMsg());
        return {};
    }
#ifdef _WIN32
    std::string strDir = Common::UTF16ToUTF8(dir);
#else
    std::string strDir = dir;
#endif
    free(dir);

    if (!strDir.ends_with(DIR_SEP)) {
        strDir += DIR_SEP;
    }
    return strDir;
} // namespace FileUtil

bool SetCurrentDir(const std::string& directory) {
#ifdef _WIN32
    return _wchdir(Common::UTF8ToUTF16W(directory).c_str()) == 0;
#else
    return chdir(directory.c_str()) == 0;
#endif
}

#if defined(__APPLE__)
std::optional<std::string> GetBundleDirectory() {
    // Get the main bundle for the app
    CFBundleRef bundle_ref = CFBundleGetMainBundle();
    if (!bundle_ref) {
        return {};
    }

    CFURLRef bundle_url_ref = CFBundleCopyBundleURL(bundle_ref);
    if (!bundle_url_ref) {
        return {};
    }
    SCOPE_EXIT({ CFRelease(bundle_url_ref); });

    CFStringRef bundle_path_ref = CFURLCopyFileSystemPath(bundle_url_ref, kCFURLPOSIXPathStyle);
    if (!bundle_path_ref) {
        return {};
    }
    SCOPE_EXIT({ CFRelease(bundle_path_ref); });

    char app_bundle_path[MAXPATHLEN];
    if (!CFStringGetFileSystemRepresentation(bundle_path_ref, app_bundle_path,
                                             sizeof(app_bundle_path))) {
        return {};
    }

    std::string path_str(app_bundle_path);
    if (!path_str.ends_with(DIR_SEP)) {
        path_str += DIR_SEP;
    }
    return path_str;
}
#endif

#ifdef _WIN32
const std::string& GetExeDirectory() {
    static std::string exe_path;
    if (exe_path.empty()) {
        wchar_t wchar_exe_path[2048];
        GetModuleFileNameW(nullptr, wchar_exe_path, 2048);
        exe_path = Common::UTF16ToUTF8(wchar_exe_path);
        exe_path = exe_path.substr(0, exe_path.find_last_of('\\'));
    }
    return exe_path;
}

std::string AppDataRoamingDirectory() {
    PWSTR path = nullptr;

    const HRESULT hr =
        SHGetKnownFolderPath(FOLDERID_RoamingAppData, KF_FLAG_DEFAULT, nullptr, &path);

    ASSERT_MSG(SUCCEEDED(hr) && path != nullptr, "Failed to get AppData directory: {:X}", hr);

    std::string result = Common::UTF16ToUTF8(path);
    CoTaskMemFree(path);
    return result;
}
#else
/**
 * @return The user’s home directory on POSIX systems
 */
const std::string GetHomeDirectory() {
    std::string home_path;
    if (home_path.empty()) {
#ifdef __PSVITA__
        // La Vita no tiene usuarios ni /etc/passwd. El equivalente al "home"
        // es la carpeta de datos de la aplicacion en la memoria interna.
        home_path = "ux0:/data/azahar";
#else
        const char* envvar = getenv("HOME");
        if (envvar) {
            home_path = envvar;
        } else {
            auto pw = getpwuid(getuid());
            ASSERT_MSG(pw,
                       "$HOME isn’t defined, and the current user can’t be found in /etc/passwd.");
            home_path = pw->pw_dir;
        }
#endif
    }
    return home_path;
}

/**
 * Follows the XDG Base Directory Specification to get a directory path
 * @param envvar The XDG environment variable to get the value from
 * @return The directory path
 * @sa http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
 */
[[maybe_unused]] const std::string GetUserDirectory(const std::string& envvar) {
    const char* directory = getenv(envvar.c_str());

    std::string user_dir;
    if (directory) {
        user_dir = directory;
    } else {
        std::string subdirectory;
        if (envvar == "XDG_DATA_HOME")
            subdirectory = DIR_SEP ".local" DIR_SEP "share";
        else if (envvar == "XDG_CONFIG_HOME")
            subdirectory = DIR_SEP ".config";
        else if (envvar == "XDG_CACHE_HOME")
            subdirectory = DIR_SEP ".cache";
        else
            ASSERT_MSG(false, "Unknown XDG variable {}.", envvar);
        user_dir = GetHomeDirectory() + subdirectory;
    }

    ASSERT_MSG(!user_dir.empty(), "User directory {} musn’t be empty.", envvar);
    ASSERT_MSG(user_dir[0] == '/', "User directory {} must be absolute.", envvar);

    return user_dir;
}
#endif

namespace {
std::unordered_map<UserPath, std::string> g_paths;
std::unordered_map<UserPath, std::string> g_default_paths;
} // namespace

void SetUserPath(const std::string& path) {
    std::string& user_path = g_paths[UserPath::UserDir];

    if (!path.empty() && CreateFullPath(path)) {
        LOG_INFO(Common_Filesystem, "Using {} as the user directory", path);
        user_path = path;
        g_paths.emplace(UserPath::ConfigDir, user_path + CONFIG_DIR DIR_SEP);
        g_paths.emplace(UserPath::CacheDir, user_path + CACHE_DIR DIR_SEP);
    } else {
#ifdef _WIN32
        user_path = GetExeDirectory() + DIR_SEP USERDATA_DIR DIR_SEP;
        std::string& legacy_citra_user_path = g_paths[UserPath::LegacyCitraUserDir];
        std::string& legacy_lime3ds_user_path = g_paths[UserPath::LegacyLime3DSUserDir];

        if (!FileUtil::IsDirectory(user_path)) {
            user_path = AppDataRoamingDirectory() + DIR_SEP EMU_DATA_DIR DIR_SEP;
            legacy_citra_user_path =
                AppDataRoamingDirectory() + DIR_SEP LEGACY_CITRA_DATA_DIR DIR_SEP;
            legacy_lime3ds_user_path =
                AppDataRoamingDirectory() + DIR_SEP LEGACY_LIME3DS_DATA_DIR DIR_SEP;
        } else {
            LOG_INFO(Common_Filesystem, "Using the local user directory");
        }

        g_paths.emplace(UserPath::ConfigDir, user_path + CONFIG_DIR DIR_SEP);
        g_paths.emplace(UserPath::CacheDir, user_path + CACHE_DIR DIR_SEP);
#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
        user_path = "/";
        g_paths.emplace(UserPath::ConfigDir, user_path + CONFIG_DIR DIR_SEP);
        g_paths.emplace(UserPath::CacheDir, user_path + CACHE_DIR DIR_SEP);
#else
        std::string& legacy_citra_user_path = g_paths[UserPath::LegacyCitraUserDir];
        std::string& legacy_lime3ds_user_path = g_paths[UserPath::LegacyLime3DSUserDir];
        auto current_dir = FileUtil::GetCurrentDir();
        if (current_dir.has_value() &&
            FileUtil::Exists(current_dir.value() + USERDATA_DIR DIR_SEP)) {
            user_path = current_dir.value() + USERDATA_DIR DIR_SEP;
            g_paths.emplace(UserPath::ConfigDir, user_path + CONFIG_DIR DIR_SEP);
            g_paths.emplace(UserPath::CacheDir, user_path + CACHE_DIR DIR_SEP);
        } else {
            std::string data_dir = GetUserDirectory("XDG_DATA_HOME") + DIR_SEP EMU_DATA_DIR DIR_SEP;

            std::string legacy_citra_data_dir =
                GetUserDirectory("XDG_DATA_HOME") + DIR_SEP LEGACY_CITRA_DATA_DIR DIR_SEP;
            std::string legacy_lime3ds_data_dir =
                GetUserDirectory("XDG_DATA_HOME") + DIR_SEP LEGACY_LIME3DS_DATA_DIR DIR_SEP;
            std::string config_dir =
                GetUserDirectory("XDG_CONFIG_HOME") + DIR_SEP EMU_DATA_DIR DIR_SEP;
            std::string cache_dir =
                GetUserDirectory("XDG_CACHE_HOME") + DIR_SEP EMU_DATA_DIR DIR_SEP;

            g_paths.emplace(UserPath::LegacyCitraConfigDir,
                            GetUserDirectory("XDG_CONFIG_HOME") +
                                DIR_SEP LEGACY_CITRA_DATA_DIR DIR_SEP);
            g_paths.emplace(UserPath::LegacyCitraCacheDir,
                            GetUserDirectory("XDG_CACHE_HOME") +
                                DIR_SEP LEGACY_CITRA_DATA_DIR DIR_SEP);
            g_paths.emplace(UserPath::LegacyLime3DSConfigDir,
                            GetUserDirectory("XDG_CONFIG_HOME") +
                                DIR_SEP LEGACY_LIME3DS_DATA_DIR DIR_SEP);
            g_paths.emplace(UserPath::LegacyLime3DSCacheDir,
                            GetUserDirectory("XDG_CACHE_HOME") +
                                DIR_SEP LEGACY_LIME3DS_DATA_DIR DIR_SEP);

#if defined(__APPLE__)
            // If XDG directories don't already exist from a previous setup, use standard macOS
            // paths.
            if (!FileUtil::Exists(data_dir) && !FileUtil::Exists(config_dir) &&
                !FileUtil::Exists(cache_dir)) {
                data_dir = GetHomeDirectory() + DIR_SEP EMU_APPLE_DATA_DIR DIR_SEP;
                legacy_citra_data_dir =
                    GetHomeDirectory() + DIR_SEP LEGACY_CITRA_APPLE_DATA_DIR DIR_SEP;
                legacy_lime3ds_data_dir =
                    GetHomeDirectory() + DIR_SEP LEGACY_LIME3DS_APPLE_DATA_DIR DIR_SEP;
                config_dir = data_dir + CONFIG_DIR DIR_SEP;
                cache_dir = data_dir + CACHE_DIR DIR_SEP;
            }
#endif

            user_path = data_dir;
            legacy_citra_user_path = legacy_citra_data_dir;
            legacy_lime3ds_user_path = legacy_lime3ds_data_dir;
            g_paths.emplace(UserPath::ConfigDir, config_dir);
            g_paths.emplace(UserPath::CacheDir, cache_dir);
        }
#endif
    }

    g_paths.emplace(UserPath::SDMCDir, user_path + SDMC_DIR DIR_SEP);
    g_paths.emplace(UserPath::NANDDir, user_path + NAND_DIR DIR_SEP);
    g_paths.emplace(UserPath::SysDataDir, user_path + SYSDATA_DIR DIR_SEP);
    // TODO: Put the logs in a better location for each OS
    g_paths.emplace(UserPath::LogDir, user_path + LOG_DIR DIR_SEP);
    g_paths.emplace(UserPath::CheatsDir, user_path + CHEATS_DIR DIR_SEP);
    g_paths.emplace(UserPath::DLLDir, user_path + DLL_DIR DIR_SEP);
    g_paths.emplace(UserPath::ShaderDir, user_path + SHADER_DIR DIR_SEP);
    g_paths.emplace(UserPath::DumpDir, user_path + DUMP_DIR DIR_SEP);
    g_paths.emplace(UserPath::LoadDir, user_path + LOAD_DIR DIR_SEP);
    g_paths.emplace(UserPath::StatesDir, user_path + STATES_DIR DIR_SEP);
    g_paths.emplace(UserPath::IconsDir, user_path + ICONS_DIR DIR_SEP);
    g_default_paths = g_paths;
}

std::string g_currentRomPath{};

void SetCurrentRomPath(const std::string& path) {
    g_currentRomPath = path;
}

bool StringReplace(std::string& haystack, const std::string& a, const std::string& b, bool swap) {
    const auto& needle = swap ? b : a;
    const auto& replacement = swap ? a : b;
    if (needle.empty()) {
        return false;
    }
    auto index = haystack.find(needle, 0);
    if (index == std::string::npos) {
        return false;
    }
    haystack.replace(index, needle.size(), replacement);
    return true;
}

std::string SerializePath(const std::string& input, bool is_saving) {
    auto result = input;
    StringReplace(result, "%CITRA_ROM_FILE%", g_currentRomPath, is_saving);
    StringReplace(result, "%CITRA_USER_DIR%", GetUserPath(UserPath::UserDir), is_saving);
    StringReplace(result, "%CITRA_USER_SDMC%", GetUserPath(UserPath::SDMCDir), is_saving);
    StringReplace(result, "%CITRA_USER_NAND%", GetUserPath(UserPath::NANDDir), is_saving);
    return result;
}

const std::string& GetUserPath(UserPath path) {
    // Set up all paths and files on the first run
    if (g_paths.empty())
        SetUserPath();
    return g_paths[path];
}

const std::string& GetDefaultUserPath(UserPath path) {
    // Set up all paths and files on the first run
    if (g_default_paths.empty())
        SetUserPath();
    return g_default_paths[path];
}

void UpdateUserPath(UserPath path, const std::string& filename) {
    if (filename.empty()) {
        return;
    }
    if (!FileUtil::IsDirectory(filename)) {
        LOG_ERROR(Common_Filesystem, "Path is not a directory. UserPath: {}  filename: {}", path,
                  filename);
        return;
    }
    g_paths[path] = SanitizePath(filename) + DIR_SEP;
}

std::size_t WriteStringToFile(bool text_file, const std::string& filename, std::string_view str) {
    return IOFile(filename, text_file ? "w" : "wb").WriteString(str);
}

std::size_t ReadFileToString(bool text_file, const std::string& filename, std::string& str) {
    IOFile file(filename, text_file ? "r" : "rb");

    if (!file.IsOpen())
        return 0;

    str.resize(static_cast<u32>(file.GetSize()));
    return file.ReadArray(str.data(), str.size());
}

void SplitFilename83(const std::string& filename, std::array<char, 9>& short_name,
                     std::array<char, 4>& extension) {
    const std::string forbidden_characters = ".\"/\\[]:;=, ";

    // On a FAT32 partition, 8.3 names are stored as a 11 bytes array, filled with spaces.
    short_name = {{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '\0'}};
    extension = {{' ', ' ', ' ', '\0'}};

    std::string::size_type point = filename.rfind('.');
    if (point == filename.size() - 1)
        point = filename.rfind('.', point);

    // Get short name.
    int j = 0;
    for (char letter : filename.substr(0, point)) {
        if (forbidden_characters.find(letter, 0) != std::string::npos)
            continue;
        if (j == 8) {
            // TODO(Link Mauve): also do that for filenames containing a space.
            // TODO(Link Mauve): handle multiple files having the same short name.
            short_name[6] = '~';
            short_name[7] = '1';
            break;
        }
        short_name[j++] = Common::ToUpper(letter);
    }

    // Get extension.
    if (point != std::string::npos) {
        j = 0;
        for (char letter : filename.substr(point + 1, 3))
            extension[j++] = Common::ToUpper(letter);
    }
}

std::vector<std::string> SplitPathComponents(std::string_view filename) {
    std::string copy(filename);
    std::replace(copy.begin(), copy.end(), '\\', '/');
    std::vector<std::string> out;

    std::stringstream stream(copy);
    std::string item;
    while (std::getline(stream, item, '/')) {
        out.push_back(std::move(item));
    }

    return out;
}

std::string_view GetParentPath(std::string_view path) {
    const auto name_bck_index = path.rfind('\\');
    const auto name_fwd_index = path.rfind('/');
    std::size_t name_index;

    if (name_bck_index == std::string_view::npos || name_fwd_index == std::string_view::npos) {
        name_index = std::min(name_bck_index, name_fwd_index);
    } else {
        name_index = std::max(name_bck_index, name_fwd_index);
    }

    return path.substr(0, name_index);
}

std::string_view GetPathWithoutTop(std::string_view path) {
    if (path.empty()) {
        return path;
    }

    while (path[0] == '\\' || path[0] == '/') {
        path.remove_prefix(1);
        if (path.empty()) {
            return path;
        }
    }

    const auto name_bck_index = path.find('\\');
    const auto name_fwd_index = path.find('/');
    return path.substr(std::min(name_bck_index, name_fwd_index) + 1);
}

std::string_view GetFilename(std::string_view path) {
    const auto name_index = path.find_last_of("\\/");

    if (name_index == std::string_view::npos) {
        return path;
    }

    return path.substr(name_index + 1);
}

std::string_view GetExtensionFromFilename(std::string_view name) {
    const std::size_t index = name.rfind('.');

    if (index == std::string_view::npos) {
        return {};
    }

    return name.substr(index + 1);
}

std::string_view RemoveTrailingSlash(std::string_view path) {
    if (path.empty()) {
        return path;
    }

    if (path.back() == '\\' || path.back() == '/') {
        path.remove_suffix(1);
        return path;
    }

    return path;
}

std::string SanitizePath(std::string_view path_, DirectorySeparator directory_separator) {
    std::string path(path_);
#if defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    return std::string(RemoveTrailingSlash(path));
#endif
    char type1 = directory_separator == DirectorySeparator::BackwardSlash ? '/' : '\\';
    char type2 = directory_separator == DirectorySeparator::BackwardSlash ? '\\' : '/';

    if (directory_separator == DirectorySeparator::PlatformDefault) {
#ifdef _WIN32
        type1 = '/';
        type2 = '\\';
#endif
    }

    std::replace(path.begin(), path.end(), type1, type2);

    auto start = path.begin();
#ifdef _WIN32
    // allow network paths which start with a double backslash (e.g. \\server\share)
    if (start != path.end())
        ++start;
#endif
    path.erase(std::unique(start, path.end(),
                           [type2](char c1, char c2) { return c1 == type2 && c2 == type2; }),
               path.end());
    return std::string(RemoveTrailingSlash(path));
}

IOFile::IOFile() = default;

IOFile::IOFile(const std::string& filename, const char openmode[], int flags)
    : filename(filename), openmode(openmode), flags(flags) {
    Open();
}

IOFile::~IOFile() {
    Close();
}

IOFile::IOFile(IOFile&& other) noexcept {
    Swap(other);
}

IOFile& IOFile::operator=(IOFile&& other) noexcept {
    Swap(other);
    return *this;
}

void IOFile::Swap(IOFile& other) noexcept {
    std::swap(m_file, other.m_file);
    std::swap(m_fd, other.m_fd);
#ifdef AZAHAR_VITA_NATIVE_IO
    /**
     * EL DESCRIPTOR NATIVO TAMBIEN, Y SIN ESTO NO ABRE NINGUNA ROM.
     *
     * Swap es lo que usan el constructor y la asignacion POR MOVIMIENTO, y un
     * IOFile se mueve constantemente: se devuelve por valor, se guarda en
     * contenedores, se pasa de una capa a otra. Al no intercambiar m_vita_fd,
     * el destino se quedaba con -1 -- o sea IsOpen() falso, "fichero cerrado"
     * -- mientras el origen conservaba el descriptor bueno y lo cerraba al
     * destruirse. El fichero se perdia en el primer movimiento.
     *
     * No salto a la vista al escribir el camino nativo porque el resto de
     * metodos si miran m_vita_fd; este es el unico sitio que toca el campo sin
     * nombrarlo, y estuvo compilado fuera hasta que se activo la macro.
     */
    std::swap(m_vita_fd, other.m_vita_fd);
#endif
    std::swap(m_good, other.m_good);
    std::swap(filename, other.filename);
    std::swap(openmode, other.openmode);
    std::swap(flags, other.flags);
}

#ifdef AZAHAR_VITA_NATIVE_IO
namespace {
/**
 * Traduce el modo de fopen ("rb", "w+b", "ab"...) a banderas de sceIoOpen.
 *
 * Solo importan las letras r/w/a y el '+'; la 'b' no significa nada aqui
 * porque en esta consola no hay traduccion de fin de linea.
 */
int VitaOpenFlags(const std::string& openmode) {
    const bool update = openmode.find('+') != std::string::npos;
    if (openmode.find('a') != std::string::npos) {
        return (update ? SCE_O_RDWR : SCE_O_WRONLY) | SCE_O_CREAT | SCE_O_APPEND;
    }
    if (openmode.find('w') != std::string::npos) {
        return (update ? SCE_O_RDWR : SCE_O_WRONLY) | SCE_O_CREAT | SCE_O_TRUNC;
    }
    return update ? SCE_O_RDWR : SCE_O_RDONLY;
}
} // Anonymous namespace
#endif

bool IOFile::Open() {
    Close();

#ifdef AZAHAR_VITA_NATIVE_IO
    // Camino nativo: stdio no puede con ficheros de mas de 2 GB en esta libc.
    // Ver el comentario de m_vita_fd en file_util.h.
    m_vita_fd = sceIoOpen(filename.c_str(), VitaOpenFlags(openmode), 0666);
    m_good = m_vita_fd >= 0;
    if (!m_good) {
        /**
         * "No existe" NO es un error que merezca una linea de registro.
         *
         * El emulador tantea ficheros que casi nunca estan (mods, parches,
         * guardados todavia sin crear) y espera que no esten: abrir y fallar es
         * parte del funcionamiento normal. Registrarlo como error llenaba el
         * log -- de 9 KB paso a 87 KB en una sola partida -- y escribir cuesta
         * tiempo en el hilo que emula la CPU. Los demas codigos si se anotan,
         * porque esos si son algo que va mal.
         */
        constexpr unsigned int kSceErrnoEnoent = 0x80010002u;
        if (static_cast<unsigned int>(m_vita_fd) == kSceErrnoEnoent) {
            LOG_DEBUG(Common_Filesystem, "sceIoOpen: no existe {}", filename);
        } else {
            LOG_ERROR(Common_Filesystem, "sceIoOpen fallo en {} (modo {}): {:#010x}", filename,
                      openmode, static_cast<unsigned int>(m_vita_fd));
        }
    }
    return m_good;
#endif

    // Any filename with the format fd://<file_descriptor> represents a file that
    // must be opened by duplicating the provided file_descriptor. This is used
    // on Android vanilla builds when the ROM absolute path is not known.
    if (filename.starts_with("fd://")) {

#if !defined(HAVE_LIBRETRO_VFS)
        const std::string fd_str = filename.substr(5);

        // Check that fd_str is not empty and contains only digits
        if (fd_str.empty() || !std::all_of(fd_str.begin(), fd_str.end(), ::isdigit)) {
            m_good = false;
            return false;
        }

        int fd = std::stoi(fd_str);

        int dup_fd = DUP_FD(fd);
        if (dup_fd == -1) {
            m_good = false;
            return false;
        }

        m_file = FDOPEN(dup_fd, openmode.c_str());
        if (!m_file) {
            CLOSE_FD(dup_fd);
            m_good = false;
            return false;
        }

        m_good = true;
        return true;
#else
        // TODO: Add support for libretro vfs when needed.
        m_good = false;
        return false;
#endif
    }

#ifdef _WIN32
    // Open with FILE_SHARE_READ, FILE_SHARE_WRITE and FILE_SHARE_DELETE
    // flags. This mimics linux behaviour as much as possible, which
    // the 3DS also does.

    const std::wstring wfilename = Common::UTF8ToUTF16W(filename);

    DWORD access = 0;
    DWORD creation = OPEN_EXISTING;

    if (openmode.find("r") != std::string::npos)
        access |= GENERIC_READ;
    if (openmode.find("w") != std::string::npos) {
        access |= GENERIC_WRITE;
        creation = CREATE_ALWAYS;
    }
    if (openmode.find("a") != std::string::npos) {
        access |= FILE_APPEND_DATA;
        creation = OPEN_ALWAYS;
    }
    if (openmode.find("+") != std::string::npos) {
        access |= GENERIC_READ | GENERIC_WRITE;
    }

    HANDLE h = CreateFileW(wfilename.c_str(), access,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           creation, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (h == INVALID_HANDLE_VALUE) {
        m_good = false;
        return false;
    }

    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(h), 0);
    if (fd == -1) {
        CloseHandle(h);
        m_good = false;
        return m_good;
    }

    m_file = _fdopen(fd, openmode.c_str());
    m_good = m_file != nullptr;

#elif defined(ANDROID) && !defined(HAVE_LIBRETRO_VFS)
    if (AndroidUtils::CanUseRawFS()) {
        m_file = FOPEN(AndroidUtils::TranslateFilePath(filename).c_str(), openmode.c_str());
    } else {
        // Check whether filepath is startsWith content
        AndroidUtils::AndroidOpenMode android_open_mode = AndroidUtils::ParseOpenmode(openmode);
        if (android_open_mode == AndroidUtils::AndroidOpenMode::WRITE ||
            android_open_mode == AndroidUtils::AndroidOpenMode::READ_WRITE ||
            android_open_mode == AndroidUtils::AndroidOpenMode::WRITE_APPEND ||
            android_open_mode == AndroidUtils::AndroidOpenMode::WRITE_TRUNCATE ||
            android_open_mode == AndroidUtils::AndroidOpenMode::READ_WRITE_TRUNCATE ||
            android_open_mode == AndroidUtils::AndroidOpenMode::READ_WRITE_APPEND) {
            if (!FileUtil::Exists(filename)) {
                std::string directory(GetParentPath(filename));
                std::string display_name(GetFilename(filename));
                if (!AndroidUtils::CreateFile(directory, display_name)) {
                    m_good = m_file != nullptr;
                    return m_good;
                }
            }
        }
        m_fd = AndroidUtils::OpenContentUri(filename, android_open_mode);
        if (m_fd != -1) {
            int error_num = 0;
            m_file = fdopen(m_fd, openmode.c_str());
            error_num = errno;
            if (error_num != 0 && m_file == nullptr) {
                LOG_ERROR(Common_Filesystem, "Error on file: {}, error: {}", filename,
                          strerror(error_num));
            }
        }
    }
    m_good = m_file != nullptr;
#else
    m_file = FOPEN(filename.c_str(), openmode.c_str());
    m_good = m_file != nullptr;
#endif

    return m_good;
}

bool IOFile::Close() {
#ifdef AZAHAR_VITA_NATIVE_IO
    if (m_vita_fd >= 0) {
        if (sceIoClose(m_vita_fd) < 0) {
            m_good = false;
        }
        m_vita_fd = -1;
    }
    return m_good;
#endif
    if (!IsOpen() || 0 != FCLOSE(m_file))
        m_good = false;

    m_file = nullptr;
    return m_good;
}

u64 IOFile::GetSize() const {
#ifdef AZAHAR_VITA_NATIVE_IO
    if (m_vita_fd < 0) {
        return 0;
    }
    // Ir al final y volver. sceIoLseek devuelve SceOff (64 bits), asi que un
    // fichero de 4 GB sale entero en vez de truncado a cero.
    const SceOff current = sceIoLseek(m_vita_fd, 0, SCE_SEEK_CUR);
    const SceOff end = sceIoLseek(m_vita_fd, 0, SCE_SEEK_END);
    sceIoLseek(m_vita_fd, current, SCE_SEEK_SET);
    return end > 0 ? static_cast<u64>(end) : 0;
#else
    if (IsOpen())
        return FileUtil::GetSize(m_file);

    return 0;
#endif
}

bool IOFile::Seek(s64 off, int origin) {
#ifdef AZAHAR_VITA_NATIVE_IO
    if (m_vita_fd < 0) {
        m_good = false;
        return m_good;
    }
    // Los SEEK_* de stdio coinciden en valor con los SCE_SEEK_*, pero se
    // traducen a mano para no depender de esa coincidencia.
    const int whence = (origin == SEEK_SET)   ? SCE_SEEK_SET
                       : (origin == SEEK_CUR) ? SCE_SEEK_CUR
                                              : SCE_SEEK_END;
    if (sceIoLseek(m_vita_fd, off, whence) < 0) {
        m_good = false;
    }
    return m_good;
#else
    if (!IsOpen() || 0 != FSEEK(m_file, off, origin))
        m_good = false;

    return m_good;
#endif
}

u64 IOFile::Tell() const {
#ifdef AZAHAR_VITA_NATIVE_IO
    if (m_vita_fd < 0) {
        return std::numeric_limits<u64>::max();
    }
    const SceOff pos = sceIoLseek(m_vita_fd, 0, SCE_SEEK_CUR);
    return pos >= 0 ? static_cast<u64>(pos) : std::numeric_limits<u64>::max();
#else
    if (IsOpen())
        return FTELL(m_file);

    return std::numeric_limits<u64>::max();
#endif
}

bool IOFile::Flush() {
#ifdef AZAHAR_VITA_NATIVE_IO
    // sceIo escribe sin buffer intermedio, asi que no hay nada que vaciar.
    return m_good;
#else
    if (!IsOpen() || 0 != FFLUSH(m_file))
        m_good = false;

    return m_good;
#endif
}

std::size_t IOFile::ReadImpl(void* data, std::size_t length, std::size_t elem_size) {
    if (!IsOpen()) {
        m_good = false;
        return std::numeric_limits<std::size_t>::max();
    }

    if (length == 0) {
        return 0;
    }

    DEBUG_ASSERT(data != nullptr);

#ifdef AZAHAR_VITA_NATIVE_IO
    {
        const SceSSize got =
            sceIoRead(m_vita_fd, data, static_cast<SceSize>(length * elem_size));
        if (got < 0) {
            m_good = false;
            return 0;
        }
        return static_cast<std::size_t>(got) / elem_size;
    }
#endif

    std::size_t read = FREAD(data, elem_size, length, m_file);
    if (read != length) {
        m_good = FERROR(m_file) != 0;
    }
    return read;
}

#ifdef _WIN32
static std::size_t pread(int fd, void* buf, std::size_t count, uint64_t offset) {
    long unsigned int read_bytes = 0;
    OVERLAPPED overlapped = {0};
    HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));

    overlapped.OffsetHigh = static_cast<uint32_t>(offset >> 32);
    overlapped.Offset = static_cast<uint32_t>(offset & 0xFFFF'FFFFLL);
    LARGE_INTEGER orig, dummy;
    // TODO(PabloMK7): This is not fully async, windows being messy again...
    // The file pos pointer will be undefined if ReadAt is used in multiple
    // threads. Normally not problematic, but worth remembering.
    SetFilePointerEx(file, {}, &orig, FILE_CURRENT);
    SetLastError(0);
    bool ret = ReadFile(file, buf, static_cast<uint32_t>(count), &read_bytes, &overlapped);
    DWORD last_error = GetLastError();
    SetFilePointerEx(file, orig, &dummy, FILE_BEGIN);

    if (!ret && last_error != ERROR_HANDLE_EOF) {
        errno = last_error;
        return std::numeric_limits<std::size_t>::max();
    }
    return read_bytes;
}
#else
#define pread ::pread
#endif

std::size_t IOFile::ReadAtImpl(void* data, std::size_t byte_count, std::size_t offset) {
    if (!IsOpen()) {
        m_good = false;
        return std::numeric_limits<std::size_t>::max();
    }

    if (byte_count == 0) {
        return 0;
    }

    DEBUG_ASSERT(data != nullptr);

#ifdef AZAHAR_VITA_NATIVE_IO
    {
        /**
         * sceIoPread: UNA llamada, sin mover la posicion y sin candado.
         *
         * Aqui habia guardar la posicion, saltar, leer y volver a saltar, todo
         * bajo un mutex, porque el comentario daba por hecho que sceIo no tenia
         * pread. SI LO TIENE: sceIoPread esta declarado en psp2/io/fcntl.h y
         * exportado en libSceIofilemgr_stub.a, y ademas toma el desplazamiento
         * como SceOff de 64 bits, que es justo lo que hacia falta.
         *
         * Importa mucho mas de lo que parece: este es el camino por el que se
         * lee el RomFS, o sea CASI TODO lo que un juego carga. Pasar de cuatro
         * llamadas al sistema y un candado a una sola llamada cambia el coste
         * de cargar un juego grande, que es justo donde se notaba que no
         * terminaba nunca. Y al no tocar la posicion del fichero, dos hilos
         * pueden leer a la vez sin estorbarse.
         */
        const SceSSize got =
            sceIoPread(m_vita_fd, data, static_cast<SceSize>(byte_count),
                       static_cast<SceOff>(offset));
        if (got < 0) {
            m_good = false;
            return 0;
        }
        return static_cast<std::size_t>(got);
    }
#endif

    std::size_t read;
#ifdef HAVE_LIBRETRO_VFS
    std::scoped_lock lock(m_file_pos_mutex);
    int64_t pos = filestream_tell(m_file);
    FSEEK(m_file, offset, RETRO_VFS_SEEK_POSITION_START);
    int64_t rv = FREAD(data, 1, byte_count, m_file);
    FSEEK(m_file, pos, RETRO_VFS_SEEK_POSITION_START);
    read = static_cast<std::size_t>(rv);
#else
    read = pread(fileno(m_file), data, byte_count, offset);
#endif
    if (read != byte_count) {
        m_good = FERROR(m_file) != 0;
    }

    return read;
}

std::size_t IOFile::WriteImpl(const void* data, std::size_t length, std::size_t elem_size) {
    if (!IsOpen()) {
        m_good = false;
        return std::numeric_limits<std::size_t>::max();
    }

    if (length == 0) {
        return 0;
    }

    DEBUG_ASSERT(data != nullptr);

#ifdef AZAHAR_VITA_NATIVE_IO
    {
        const SceSSize put =
            sceIoWrite(m_vita_fd, data, static_cast<SceSize>(length * elem_size));
        if (put < 0) {
            m_good = false;
            return 0;
        }
        return static_cast<std::size_t>(put) / elem_size;
    }
#endif

    std::size_t written;
#if defined(HAVE_LIBRETRO_VFS)
    written = rfwrite(data, elem_size, length, m_file) / elem_size;
#else
    written = std::fwrite(data, elem_size, length, m_file);
#endif
    if (written != length) {
        m_good = FERROR(m_file) != 0;
    }
    return written;
}

bool IOFileBase::ReadLine(std::string& line) {
    line.clear();

    char ch;
    bool read_anything = false;

    while (true) {
        const std::size_t read = ReadImpl(&ch, sizeof(ch), 1);

        if (read != sizeof(ch)) {
            return read_anything;
        }
        read_anything = true;

        if (ch == '\n') {
            return true;
        }

        // Always convert to UNIX style
        if (ch != '\r') {
            line.push_back(ch);
        }
    }
}

size_t IOFileBase::WriteLine(const std::string_view line) {
    const size_t written_line = WriteImpl(line.data(), line.size(), 1);
    if (written_line != line.size()) {
        return written_line;
    }

    char nl = '\n';
    const size_t written_nl = WriteImpl(&nl, sizeof(nl), 1);
    if (written_nl != sizeof(nl)) {
        return written_nl;
    }

    return written_line + written_nl;
}

inline bool IOFile::IsOpen() const {
#ifdef AZAHAR_VITA_NATIVE_IO
    return m_vita_fd >= 0;
#else
    return nullptr != m_file;
#endif
}

inline bool IOFile::IsGood() const {
    return m_good;
}

inline void IOFile::Clear() {
    m_good = true;

#if defined(AZAHAR_VITA_NATIVE_IO)
    // En el camino nativo no hay FILE*: m_file es SIEMPRE nulo, y
    // std::clearerr(nullptr) no es "no hacer nada", es desreferenciar un puntero
    // nulo. sceIo no guarda banderas de error por descriptor, asi que poner
    // m_good a true de arriba es todo lo que hay que limpiar.
#elif defined(HAVE_LIBRETRO_VFS)
    filestream_rewind(m_file);
#else
    std::clearerr(m_file);
#endif
}

inline const std::string& IOFile::Filename() const {
    return filename;
}

std::unique_ptr<IOFileBase> IOFile::OpenCopy() const {
    std::unique_ptr<IOFile> ret = std::make_unique<IOFile>();
    ret->filename = filename;
    ret->openmode = openmode;
    ret->flags = flags;
    ret->Open();

    return ret;
}

int IOFile::GetFd() const {
#ifdef HAVE_LIBRETRO_VFS
    if (m_file == nullptr)
        return -1;
    return fileno(filestream_get_vfs_handle(m_file)->fp);
#else
#ifdef ANDROID
    if (!AndroidUtils::CanUseRawFS()) {
        return m_fd;
    }
#endif // ANDROID
    if (m_file == nullptr)
        return -1;
    return fileno(m_file);
#endif // HAVE_LIBRETRO_VFS
}

bool IOFile::Resize(u64 size) {
#ifdef AZAHAR_VITA_NATIVE_IO
    // Sin equivalente directo en sceIo. Solo lo usan rutas de escritura de
    // guardado, que en la Vita no se ejercitan todavia; se deja avisado en vez
    // de fingir que ha funcionado.
    LOG_WARNING(Common_Filesystem, "Resize({}) no implementado en Vita", size);
    return false;
#endif
    if (!IsOpen() || 0 !=
#if defined(HAVE_LIBRETRO_VFS)
                         filestream_truncate(m_file, size)
#elif defined(_WIN32)
                         // ector: _chsize sucks, not 64-bit safe
                         // F|RES: changed to _chsize_s. i think it is 64-bit safe
                         _chsize_s(_fileno(m_file), size)
#else
                         // TODO: handle 64bit and growing
                         ftruncate(fileno(m_file), size)
#endif
    )
        m_good = false;

    return m_good;
}

template <typename T>
using boost_iostreams = boost::iostreams::stream<T>;

template <>
void OpenFStream<std::ios_base::in>(
    boost_iostreams<boost::iostreams::file_descriptor_source>& fstream,
    const std::string& filename) {
    IOFile file(filename, "r");
    if (file.GetFd() == -1)
        return;
    int fd = dup(file.GetFd());
    if (fd == -1)
        return;
    boost::iostreams::file_descriptor_source file_descriptor_source(fd,
                                                                    boost::iostreams::close_handle);
    fstream.open(file_descriptor_source);
}

template <>
void OpenFStream<std::ios_base::out>(
    boost_iostreams<boost::iostreams::file_descriptor_sink>& fstream, const std::string& filename) {
    IOFile file(filename, "w");
    if (file.GetFd() == -1)
        return;
    int fd = dup(file.GetFd());
    if (fd == -1)
        return;
    boost::iostreams::file_descriptor_sink file_descriptor_sink(fd, boost::iostreams::close_handle);
    fstream.open(file_descriptor_sink);
}
} // namespace FileUtil

SERIALIZE_EXPORT_IMPL(FileUtil::IOFile)
