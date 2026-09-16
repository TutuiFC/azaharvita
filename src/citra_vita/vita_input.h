// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

namespace VitaFrontend::Input {

/// Registra las fabricas de dispositivos de entrada de la Vita en Azahar.
void Init();

/// Quita las fabricas registradas por Init().
void Shutdown();

/// Rellena Settings con el mapeo de botones Vita -> 3DS.
void ApplyDefaultMapping();

} // namespace VitaFrontend::Input
