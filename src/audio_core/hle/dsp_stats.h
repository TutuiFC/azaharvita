// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>

namespace AudioCore::HLE::Stats {

/**
 * EN QUE SE VA EL "dsp" DE crash.txt (0.3.1.4). En New Super Mario Bros. 2 son
 * 8-10 ms por fotograma y en las cuevas de Pokemon Sol 7-13, y la cifra era un
 * total. Lo escribe el hilo de emulacion (el DSP HLE va en su tick de audio) y
 * lo lee y pone a cero el overlay con Take.
 *   ticks          ticks de audio (uno cada 160 muestras)
 *   active         fuentes encendidas, sumadas en cada tick
 *   decode_us      decodificar buffers enteros (al empezar a sonar, al dar la
 *                  vuelta un bucle, al alargarse uno a medio sonar)
 *   decoded        muestras decodificadas por eso
 * La mezcla es el resto de Common::FrameStats::dsp_us, quitando el AAC.
 */
inline std::atomic<unsigned long long> ticks{0};
inline std::atomic<unsigned long long> active{0};
inline std::atomic<unsigned long long> decode_us{0};
inline std::atomic<unsigned long long> decoded{0};
/// Lo que tarda la mezcla en el hilo del DSP (0.3.1.5), fuera del de emulacion.
inline std::atomic<unsigned long long> worker_us{0};

} // namespace AudioCore::HLE::Stats
