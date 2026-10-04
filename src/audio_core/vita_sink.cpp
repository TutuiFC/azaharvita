// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cmath>
#include <cstring>
#include <psp2/audioout.h>
#include "audio_core/audio_types.h"
#include "audio_core/vita_sink.h"
#include "common/logging/log.h"
#include "common/vita_diag.h"

namespace AudioCore {

namespace {
/**
 * El puerto principal de la Vita solo acepta 48000 Hz y el DSP del 3DS saca
 * 32728 Hz. Hasta 0.2.0.2 se declaraba 48000 como frecuencia nativa y NADIE
 * convertia: el estirado (SoundTouch) recibia muestras de 32728 Hz como si
 * fueran de 48000, y para cuadrar las cuentas estiraba el sonido un 47 % todo
 * el rato. Ese estirado constante es el "petardeo", y el tono salia agudo.
 * Ahora el DSP y el estirado trabajan a su frecuencia y la conversion a 48000
 * se hace aqui, interpolando (ver OutputThread).
 */
constexpr int kSampleRate = 48000;
constexpr double kSourceStep =
    static_cast<double>(native_sample_rate) / static_cast<double>(kSampleRate);

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
    return native_sample_rate;
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
        // Lo que falte del DSP para este bloque, de una vez: cada llamada al
        // callback pasa por el estirado, que tiene su coste fijo.
        const std::size_t have = source.size() / 2;
        const std::size_t need =
            static_cast<std::size_t>(std::floor(source_pos + kGrain * kSourceStep)) + 2;
        if (need > have) {
            source.resize(need * 2, 0);
            if (callback) {
                callback(source.data() + have * 2, need - have);
            }
        }
        for (int i = 0; i < kGrain; i++) {
            const std::size_t index = static_cast<std::size_t>(source_pos);
            const float frac = static_cast<float>(source_pos - static_cast<double>(index));
            for (int channel = 0; channel < 2; channel++) {
                const float a = source[index * 2 + channel];
                const float b = source[(index + 1) * 2 + channel];
                buffer[i * 2 + channel] = static_cast<s16>(a + (b - a) * frac);
            }
            source_pos += kSourceStep;
        }
        const std::size_t consumed = static_cast<std::size_t>(source_pos);
        source.erase(source.begin(), source.begin() + consumed * 2);
        source_pos -= static_cast<double>(consumed);
        // Bloquea hasta que el hardware haya consumido el bloque anterior.
        sceAudioOutOutput(port, buffer.data());
    }
}

std::vector<std::string> ListVitaSinkDevices() {
    return {std::string(auto_device_name)};
}

} // namespace AudioCore
