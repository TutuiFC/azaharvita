// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#ifdef __PSVITA__

#include <string>

namespace Core {

/// En que se va el hilo de emulacion por partes de System::RunLoop, en ms por
/// segundo de reloj desde la llamada anterior (0.3.2.0).
std::string TakeLoopProfile();

/// Los eventos de temporizacion que mas tiempo se llevan, desde la llamada
/// anterior (0.3.2.0).
std::string TakeTimingProfile();

} // namespace Core

#endif
