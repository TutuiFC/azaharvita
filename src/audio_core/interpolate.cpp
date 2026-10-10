// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "audio_core/interpolate.h"
#include "common/assert.h"

namespace AudioCore::AudioInterp {

// Calculations are done in fixed point with 24 fractional bits.
// (This is not verified. This was chosen for minimal error.)
constexpr u64 scale_factor = 1 << 24;
constexpr u64 scale_mask = scale_factor - 1;

/// Here we step over the input in steps of rate, until we consume all of the input.
/// Three adjacent samples are passed to fn each step.
/**
 * SOBRE UNA COPIA CONTIGUA (0.3.2.1). La entrada es una deque: meterle las dos
 * muestras de historia por delante y leer tres por muestra de salida con su
 * operator[] (cuentas de nodo en cada acceso) era casi todo el coste de una
 * fuente en la mezcla del DSP, que el hilo de emulacion espera en cada tick
 * ("publish_event" 28-64 ms por segundo en Pokemon Sol de 0.3.2.0). Ahora se
 * copian a la pila solo las que puede leer este fotograma -- la historia y
 * hasta la ultima posicion que alcanza el bucle -- y se quitan de la deque las
 * consumidas. Mismas cuentas, mismas muestras y el mismo corte si se acaba la
 * entrada: el bucle no puede pasar de needed, y si la entrada es mas corta se
 * copia entera. Con un ritmo enorme (needed fuera del tope), como antes.
 */
template <typename Function>
static bool StepOverSamplesContiguous(State& state, StereoBuffer16& input, u64 step_size,
                                      StereoFrame16& output, std::size_t& outputi, Function fn) {
    constexpr std::size_t kMaxWindow = 1024;
    using Sample = std::array<s16, 2>;
    const std::size_t remaining = output.size() - outputi;
    std::size_t count = input.size();
    if (step_size > (u64{1} << 48)) {
        return false;
    }
    if (remaining != 0) {
        const u64 last = state.fposition + static_cast<u64>(remaining - 1) * step_size;
        const u64 needed = last / scale_factor + 1;
        if (needed < count) {
            count = static_cast<std::size_t>(needed);
        }
    } else {
        count = 0;
    }
    if (count > kMaxWindow) {
        return false;
    }
    std::array<Sample, kMaxWindow + 2> window;
    window[0] = state.xn2;
    window[1] = state.xn1;
    std::copy_n(input.begin(), count, window.begin() + 2);
    const std::size_t window_size = count + 2;

    u64 fposition = state.fposition;
    std::size_t inputi = 0;
    while (outputi < output.size()) {
        inputi = static_cast<std::size_t>(fposition / scale_factor);
        if (inputi + 2 >= window_size) {
            inputi = window_size - 2;
            break;
        }
        const u64 fraction = fposition & scale_mask;
        output[outputi++] = fn(fraction, window[inputi], window[inputi + 1], window[inputi + 2]);
        fposition += step_size;
    }

    state.xn2 = window[inputi];
    state.xn1 = window[inputi + 1];
    state.fposition = fposition - inputi * scale_factor;
    input.erase(input.begin(), std::next(input.begin(), inputi));
    return true;
}

template <typename Function>
static void StepOverSamples(State& state, StereoBuffer16& input, float rate, StereoFrame16& output,
                            std::size_t& outputi, Function fn) {
    ASSERT(rate > 0);

    if (input.empty())
        return;

    if (StepOverSamplesContiguous(state, input, static_cast<u64>(rate * scale_factor), output,
                                  outputi, fn)) {
        return;
    }

    input.insert(input.begin(), {state.xn2, state.xn1});

    const u64 step_size = static_cast<u64>(rate * scale_factor);
    u64 fposition = state.fposition;
    std::size_t inputi = 0;

    while (outputi < output.size()) {
        inputi = static_cast<std::size_t>(fposition / scale_factor);

        if (inputi + 2 >= input.size()) {
            inputi = input.size() - 2;
            break;
        }

        u64 fraction = fposition & scale_mask;
        output[outputi++] = fn(fraction, input[inputi], input[inputi + 1], input[inputi + 2]);

        fposition += step_size;
    }

    state.xn2 = input[inputi];
    state.xn1 = input[inputi + 1];
    state.fposition = fposition - inputi * scale_factor;

    input.erase(input.begin(), std::next(input.begin(), inputi + 2));
}

void None(State& state, StereoBuffer16& input, float rate, StereoFrame16& output,
          std::size_t& outputi) {
    StepOverSamples(
        state, input, rate, output, outputi,
        [](u64 fraction, const auto& x0, const auto& x1, const auto& x2) { return x0; });
}

void Linear(State& state, StereoBuffer16& input, float rate, StereoFrame16& output,
            std::size_t& outputi) {
    // Note on accuracy: Some values that this produces are +/- 1 from the actual firmware.
    StepOverSamples(state, input, rate, output, outputi,
                    [](u64 fraction, const auto& x0, const auto& x1, const auto& x2) {
                        // This is a saturated subtraction. (Verified by black-box fuzzing.)
                        s64 delta0 = std::clamp<s64>(x1[0] - x0[0], -32768, 32767);
                        s64 delta1 = std::clamp<s64>(x1[1] - x0[1], -32768, 32767);

                        return std::array<s16, 2>{
                            static_cast<s16>(x0[0] + fraction * delta0 / scale_factor),
                            static_cast<s16>(x0[1] + fraction * delta1 / scale_factor),
                        };
                    });
}

} // namespace AudioCore::AudioInterp
