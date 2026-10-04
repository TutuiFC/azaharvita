// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include <psp2/audioout.h>
#include "audio_core/vita_sink.h"
#include "common/logging/log.h"
#include "common/vita_diag.h"

namespace AudioCore {

namespace {
// La Vita acepta 48000 Hz de forma nativa. El DSP del 3DS trabaja a 32728 Hz,
// asi que Azahar remuestrea a lo que aqui se declare.
constexpr int kSampleRate = 48000;

// Muestras estereo por bloque. sceAudioOutOutput bloquea hasta que el hardware
// consume el bloque, lo que marca el ritmo del hilo de salida. 1024 muestras
// son ~21 ms: suficiente margen para que un emulador lento no corte el sonido
// a cada momento, sin meter un retardo molesto.
constexpr int kGrain = 1024;
} // Anonymous namespace

VitaSink::VitaSink(std::string device_id) {
    port = sceAudioOutOpenPort(SCE_AUDIO_OUT_PORT_TYPE_MAIN, kGrain, kSampleRate,
                               SCE_AUDIO_OUT_MODE_STEREO);
    if (port < 0) {
        LOG_ERROR(Audio_Sink, "sceAudioOutOpenPort fallo ({:#x}), sin audio", port);
        return;
    }

    int volume[2] = {SCE_AUDIO_VOLUME_0DB, SCE_AUDIO_VOLUME_0DB};
    sceAudioOutSetVolume(port, static_cast<SceAudioOutChannelFlag>(SCE_AUDIO_VOLUME_FLAG_L_CH |
                                                                  SCE_AUDIO_VOLUME_FLAG_R_CH),
                         volume);

    sample_rate = kSampleRate;
    buffer.resize(static_cast<std::size_t>(kGrain) * 2);
}

VitaSink::~VitaSink() {
    running = false;
    if (thread.joinable()) {
        thread.join();
    }
    if (port >= 0) {
        sceAudioOutReleasePort(port);
    }
}

unsigned int VitaSink::GetNativeSampleRate() const {
    return sample_rate == 0 ? kSampleRate : sample_rate;
}

void VitaSink::SetCallback(std::function<void(s16*, std::size_t)> cb) {
    callback = std::move(cb);

    if (port < 0 || running) {
        return;
    }
    running = true;
    thread = std::thread(&VitaSink::OutputThread, this);
}

void VitaSink::OutputThread() {
    // Alta (0.2.0.2): pasa casi todo el tiempo bloqueado en sceAudioOutOutput,
    // pero cuando le toca tiene 21 ms para rellenar el bloque o el sonido se
    // corta. Con la prioridad por defecto de std::thread (la mas baja) y los
    // tres nucleos ocupados, no llegaba.
    Common::VitaSetThreadPriority(Common::kVitaPriorityAudio, "audio");
    while (running) {
        std::memset(buffer.data(), 0, buffer.size() * sizeof(s16));
        if (callback) {
            callback(buffer.data(), kGrain);
        }
        // Bloquea hasta que el hardware haya consumido el bloque anterior.
        sceAudioOutOutput(port, buffer.data());
    }
}

std::vector<std::string> ListVitaSinkDevices() {
    return {std::string(auto_device_name)};
}

} // namespace AudioCore
