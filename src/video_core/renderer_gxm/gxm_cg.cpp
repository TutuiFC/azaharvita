// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/gxm_cg.h"

#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/kernel/processmgr.h>
#include "common/logging/log.h"
#include "common/vita_diag.h"

namespace Gxm {

namespace {

constexpr const char* kShacccgPaths[] = {
    "ur0:/data/libshacccg.suprx",
    "ux0:/data/libshacccg.suprx",
};

SceUID g_module = -1;
SceShaccCgSourceFile g_source{};
SceShaccCgCallbackList g_callbacks{};
char g_status[64] = "sin iniciar";
unsigned long long g_last_attempt_us = 0;

/// Un intento por segundo como mucho cuando falla: sin esto, cada lote cuyo
/// shader no se pueda compilar paga una carga de modulo fallida, y eso fue una
/// caida de 450 ms a 13.734 ms de fotograma.
constexpr unsigned long long kRetryIntervalUs = 1000000;

SceShaccCgSourceFile* OpenSource(const char*, const SceShaccCgSourceLocation*,
                                 const SceShaccCgCompileOptions*, const char**) {
    return &g_source;
}

void* CgAlloc(unsigned int size) {
    return std::malloc(size);
}

void CgFree(void* pointer) {
    std::free(pointer);
}

} // Anonymous namespace

bool EnsureCgReady() {
    if (g_module >= 0) {
        return true;
    }
    const unsigned long long now_us = Common::VitaMicros();
    if (g_last_attempt_us != 0 && now_us - g_last_attempt_us < kRetryIntervalUs) {
        return false;
    }
    g_last_attempt_us = now_us;
    for (const char* path : kShacccgPaths) {
        g_module = sceKernelLoadStartModule(path, 0, nullptr, 0, nullptr, nullptr);
        if (g_module >= 0) {
            LOG_INFO(Render, "GXM: libshacccg cargado desde {}", path);
            break;
        }
        LOG_ERROR(Render, "GXM: cargar {} fallo con {:#x}", path, static_cast<u32>(g_module));
    }
    if (g_module < 0) {
        const auto written = fmt::format_to_n(g_status, sizeof(g_status) - 1,
                                              "sin libshacccg ({:#x})",
                                              static_cast<u32>(g_module));
        *written.out = 0;
        return false;
    }
    sceShaccCgSetDefaultAllocator(&CgAlloc, &CgFree);
    sceShaccCgInitializeCallbackList(&g_callbacks, SCE_SHACCCG_TRIVIAL);
    g_callbacks.openFile = &OpenSource;
    std::strcpy(g_status, "listo");
    return true;
}

const char* CgStatus() {
    return g_status;
}

const SceShaccCgCompileOutput* CompileCg(SceShaccCgTargetProfile profile, const char* name,
                                         const char* source) {
    if (!EnsureCgReady()) {
        return nullptr;
    }
    g_source.fileName = name;
    g_source.text = source;
    g_source.size = static_cast<SceUInt32>(std::strlen(source));

    SceShaccCgCompileOptions options{};
    sceShaccCgInitializeCompileOptions(&options);
    options.mainSourceFile = name;
    options.targetProfile = profile;
    options.entryFunctionName = "main";

    const SceShaccCgCompileOutput* output = sceShaccCgCompileProgram(&options, &g_callbacks, 0);
    if (output == nullptr || output->programData == nullptr) {
        /**
         * Nota a crash.txt solo las primeras veces, y con el PRIMER diagnostico
         * del compilador dentro.
         *
         * Antes se anotaba cada fallo sin mensaje, y ademas el cache no
         * recordaba el fallo: cada lote con esa configuracion volvia a compilar
         * y a fallar. En una sesion medida fueron 1.456 intentos. Sin el
         * mensaje, "gxm shader" tampoco dice por que falla, que es lo que hace
         * falta para arreglar el generador.
         */
        static int noted = 0;
        if (noted < 4) {
            noted++;
            const char* message = nullptr;
            if (output != nullptr && output->diagnosticCount > 0) {
                message = output->diagnostics[0].message;
            }
            char note[192];
            const auto written =
                fmt::format_to_n(note, sizeof(note) - 1, "{}: {}", name,
                                 message != nullptr ? message : "sin diagnostico");
            *written.out = 0;
            Common::VitaNote("gxm shader", note);
        }
        if (output != nullptr) {
            for (int i = 0; i < output->diagnosticCount && i < 4; i++) {
                LOG_ERROR(Render, "GXM {}: {}", name,
                          output->diagnostics[i].message != nullptr
                              ? output->diagnostics[i].message
                              : "(sin mensaje)");
            }
            sceShaccCgDestroyCompileOutput(output);
        } else {
            LOG_ERROR(Render, "GXM: el compilador no devolvio nada para {}", name);
        }
        return nullptr;
    }
    return output;
}

void ReleaseCgOutput(const SceShaccCgCompileOutput* output) {
    if (output != nullptr) {
        sceShaccCgDestroyCompileOutput(output);
    }
}

} // namespace Gxm
