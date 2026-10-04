// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include "audio_core/sink.h"

namespace AudioCore {

/// Salida de audio de PS Vita sobre sceAudioOut.
class VitaSink final : public Sink {
public:
    explicit VitaSink(std::string device_id);
    ~VitaSink() override;

    unsigned int GetNativeSampleRate() const override;
    void SetCallback(std::function<void(s16*, std::size_t)> cb) override;

private:
    void OutputThread();

    int port = -1;
    unsigned int sample_rate = 0;
    std::function<void(s16*, std::size_t)> callback;
    std::vector<s16> buffer;
    /// Muestras del DSP (a native_sample_rate) aun sin convertir, y la
    /// posicion fraccionaria de la siguiente salida entre ellas.
    std::vector<s16> source;
    double source_pos = 0.0;
    std::thread thread;
    std::atomic<bool> running{false};
};

std::vector<std::string> ListVitaSinkDevices();

} // namespace AudioCore
