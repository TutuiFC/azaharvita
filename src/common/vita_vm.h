// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#ifdef __PSVITA__

#include <mutex>

namespace Common {

/**
 * El dominio VM (sceKernelOpenVMDomain / CloseVMDomain) es UNO para todo el
 * proceso (0.2.0.5). Con la GPU en otro nucleo hay dos JIT que lo abren y lo
 * cierran desde hilos distintos -- el del ARM11 en el de emulacion y el de
 * shaders de la PICA en el de la GPU --: si uno lo cierra mientras el otro
 * escribe codigo, el segundo falla al abrirlo (0x80010058 en crash.txt de
 * 0.2.0.4) o escribe en memoria que ya no se puede escribir. Todo el que lo
 * abra, con este cerrojo cogido hasta cerrarlo.
 */
inline std::mutex vita_vm_domain_mutex;

} // namespace Common

#endif
