// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <boost/serialization/access.hpp>

#include "core/hle/service/gsp/gsp_interrupt.h"

namespace Service::GSP {
struct Command;
struct FrameBufferInfo;
} // namespace Service::GSP

namespace Core {
class System;
}

namespace Pica {
class DebugContext;
class PicaCore;
struct RegsLcd;
union ColorFill;
} // namespace Pica

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {

/**
 * Reparto del tiempo de la "GPU" por tipo de comando GX.
 *
 * Hacia falta porque el 'gpu %' de PerfStats mide GPU::Execute ENTERO, y ahi
 * dentro hay tres cosas con costes y arreglos completamente distintos:
 *
 *   - listas de comandos: acaban rasterizando triangulos. Es lo unico que se
 *     salta el salto de fotogramas.
 *   - rellenos de memoria: borrar pantallas y buffers. Puro ancho de banda.
 *   - transferencias de pantalla y copias de textura: conversion de formato
 *     pixel a pixel. Ocurren en TODOS los fotogramas, saltados incluidos.
 *
 * Sin separarlas, el 94% de "gpu" parecia rasterizado y se optimizo eso durante
 * tres versiones. Con el salto a 1/5 el rasterizado se redujo a una quinta
 * parte y el porcentaje apenas bajo: la mayor parte estaba en los otros dos.
 *
 * Se mide con dos lecturas de reloj por comando GX -- unas decenas por
 * fotograma, no por pixel --, asi que la medida no se estorba a si misma.
 */
namespace GxStats {
inline std::atomic<u64> cmdlist_ns{0};
inline std::atomic<u64> fill_ns{0};
inline std::atomic<u64> transfer_ns{0};
inline std::atomic<u64> dma_ns{0};

inline void Reset() {
    cmdlist_ns.store(0, std::memory_order_relaxed);
    fill_ns.store(0, std::memory_order_relaxed);
    transfer_ns.store(0, std::memory_order_relaxed);
    dma_ns.store(0, std::memory_order_relaxed);
}
} // namespace GxStats

/// Measured on hardware to be 2240568 timer cycles or 4481136 ARM11 cycles
constexpr u64 FRAME_TICKS = 4481136ull;

class GraphicsDebugger;
class RendererBase;
class RightEyeDisabler;

/**
 * The GPU class is the high level interface to the video_core for core services.
 */
class GPU {
public:
    explicit GPU(Core::System& system, Frontend::EmuWindow& emu_window,
                 Frontend::EmuWindow* secondary_window);
    ~GPU();

    /// Sets the function to call for signalling GSP interrupts.
    void SetInterruptHandler(Service::GSP::InterruptHandler handler);

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(PAddr addr, u32 size);

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(PAddr addr, u32 size);

    /// Flushes and invalidates all memory in the rasterizer cache and removes any leftover state.
    void ClearAll(bool flush);

    /// Executes the provided GSP command.
    void Execute(const Service::GSP::Command& command);

    /// Updates GPU display framebuffer configuration using the specified parameters.
    void SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info);

    /// Sets the LCD color fill configuration for the top and bottom screens.
    void SetColorFill(const Pica::ColorFill& fill);

    /// Reads a word from the GPU virtual address.
    u32 ReadReg(VAddr addr);

    /// Writes the provided value to the GPU virtual address.
    void WriteReg(VAddr addr, u32 data);

    /// Returns a mutable reference to the renderer.
    [[nodiscard]] VideoCore::RendererBase& Renderer();

    /// Returns a mutable reference to the PICA GPU.
    [[nodiscard]] Pica::PicaCore& PicaCore();

    /// Returns an immutable reference to the PICA GPU.
    [[nodiscard]] const Pica::PicaCore& PicaCore() const;

    /// Returns a mutable reference to the GSP command debugger.
    [[nodiscard]] GraphicsDebugger& Debugger();

    RightEyeDisabler& GetRightEyeDisabler() {
        return *right_eye_disabler;
    }

    void ApplyPerProgramSettings(u64 program_ID);

    /// Recreates the renderer (for GL context reset in libretro)
    void RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window);

    /// Releases the renderer (for GL context destroy in libretro)
    void ReleaseRenderer();

private:
    void SubmitCmdList(u32 index);

    // Interrupt index must be 0 or 1 to signal the relative PSC interrupt.
    void MemoryFill(u32 index, u32 intr_index);

    void MemoryTransfer();

    void VBlankCallback(uintptr_t user_data, s64 cycles_late);

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const u32 file_version);

    std::unique_ptr<RightEyeDisabler> right_eye_disabler;

private:
    friend class RightEyeDisabler;
    struct Impl;
    std::unique_ptr<Impl> impl;

    PAddr VirtualToPhysicalAddress(VAddr addr);
};

} // namespace VideoCore
