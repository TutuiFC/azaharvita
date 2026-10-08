// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#ifdef __PSVITA__

#include <string>

namespace Common {

/**
 * Lo ocupado de cada nucleo de usuario y el tiempo de CPU de cada hilo con
 * nombre (los que pasan por VitaPinThreadToUserCore o VitaSetThreadPriority),
 * desde la llamada anterior (0.3.2.0). Solo desde un hilo a la vez.
 */
std::string VitaThreadSummary();

} // namespace Common

#endif
