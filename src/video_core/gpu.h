// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <chrono>
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

namespace Kernel {
class Process;
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
/// Contadores de COMANDOS en el intervalo (0.1.5.2, 4.7): el tiempo ya estaba;
/// sin el numero de veces no se sabe si subio porque hay mas o porque cada uno
/// es mas caro. fill/transfer cuentan tambien los que entran por WriteReg
/// (registro trigger), que no pasan por Execute().
inline std::atomic<u64> fill_count{0};
inline std::atomic<u64> transfer_count{0};
inline std::atomic<u64> cmdlist_count{0};
/**
 * Cuantas operaciones que pueden cambiar la imagen ha hecho la GPU emulada
 * desde el arranque (0.1.7.1): listas de comandos, rellenos, transferencias y
 * copias DMA. No se pone a cero. Si entre dos presentaciones no ha cambiado
 * (ni la direccion de los framebuffers), la imagen es la misma y la Vita no
 * tiene que volver a subirla ni dibujarla. Ver EmuWindow_Vita::PresentScreens.
 */
inline std::atomic<u64> frame_work{0};
/**
 * GPU en otro nucleo (0.2.0.0, ver GPU::async_enabled). Por intervalo: lo que
 * ha trabajado el hilo de la GPU, lo que el de emulacion ha estado parado
 * esperandola (y cuantas veces: por interrupciones con la CPU emulada sin
 * nada que hacer, por sincronizar para leer o escribir su estado, por tener
 * ya dos presentaciones en cola), y lo mas larga que ha llegado a estar la
 * cola. Las lee y las pone a cero el overlay.
 */
inline std::atomic<u64> thread_busy_us{0};
inline std::atomic<u64> emu_wait_us{0};
inline std::atomic<u32> irq_waits{0};
inline std::atomic<u32> syncs{0};
inline std::atomic<u32> present_waits{0};
inline std::atomic<u32> queue_max{0};

inline void Reset() {
    cmdlist_ns.store(0, std::memory_order_relaxed);
    fill_ns.store(0, std::memory_order_relaxed);
    transfer_ns.store(0, std::memory_order_relaxed);
    dma_ns.store(0, std::memory_order_relaxed);
    fill_count.store(0, std::memory_order_relaxed);
    transfer_count.store(0, std::memory_order_relaxed);
    cmdlist_count.store(0, std::memory_order_relaxed);
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

    /**
     * LA GPU EMULADA EN OTRO NUCLEO (0.2.0.0, ajuste "GPU en otro nucleo").
     *
     * Sin esto, el hilo de emulacion hace las dos cosas por turnos: la CPU del
     * 3DS y, dentro de cada llamada a GSP, todo el trabajo de la GPU (listas de
     * comandos, rellenos, copias), mas la presentacion en cada VBlank. Con
     * esto, las ordenes de GSP y la presentacion van a una cola que vacia un
     * hilo en el nucleo 1, y el juego sigue mientras tanto en el 0.
     *
     * El juego no ve nada fuera de orden: las ordenes se ejecutan en el orden
     * en que llegan, y sus interrupciones (P3D, PPF, PSC, DMA) se le dan en el
     * hilo de emulacion cuando la GPU ha terminado de verdad (ver
     * DeliverInterrupts). Lo que necesite el estado de la GPU al momento (leer
     * sus registros, volcar superficies a memoria) espera antes a que la cola
     * se vacie (ver Sync).
     *
     * Se lee al crear la GPU: un cambio vale para el siguiente juego.
     */
    static inline std::atomic<bool> async_enabled{false};

    /// Da a GSP, en el hilo de emulacion, las interrupciones que el hilo de la
    /// GPU ha levantado. Lanza si ese hilo murio por una excepcion.
    void DeliverInterrupts();

    /// Hay ordenes en cola (o ejecutandose) que pueden acabar en interrupcion.
    [[nodiscard]] bool HasInterruptWork() const;

    /**
     * Para cuando la CPU emulada no tiene ningun hilo que ejecutar: espera a
     * que la GPU levante una interrupcion o se quede sin ordenes que puedan
     * levantarla. Sin esto, el tiempo emulado correria hasta el siguiente
     * evento mientras el juego espera a la GPU, y el juego veria una GPU mucho
     * mas lenta que con todo en un hilo.
     */
    void WaitForInterrupts();

    /// Espera a que el hilo de la GPU acabe todo lo encolado. No hace nada sin
    /// ese hilo ni llamada desde el.
    void Sync();

    /// Para el hilo de la GPU tirando lo que quede en la cola. Va antes de
    /// destruir nada de lo que ese hilo usa.
    void StopThread();

    /// Tiempo emulado del VBlank que se presenta: el limitador de velocidad y
    /// las estadisticas no pueden leer el reloj emulado desde otro hilo.
    [[nodiscard]] std::chrono::microseconds PresentTimeUs() const;

private:
    void SubmitCmdList(u32 index);

    // Interrupt index must be 0 or 1 to signal the relative PSC interrupt.
    void MemoryFill(u32 index, u32 intr_index);

    void MemoryTransfer();

    void VBlankCallback(uintptr_t user_data, s64 cycles_late);

    void ExecuteCommand(const Service::GSP::Command& command,
                        const std::shared_ptr<Kernel::Process>& dma_process);

    void ApplyBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info);

    struct AsyncWorker;
    std::unique_ptr<AsyncWorker> async;

    [[nodiscard]] bool Async() const;
    [[nodiscard]] bool OnGpuThread() const;
    void StartThread();
    void AsyncLoop();
    void RouteInterrupt(Service::GSP::InterruptId interrupt_id, u64 wait_delay_ns);
    void QueueCommand(const Service::GSP::Command& command);
    void QueueBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info);
    void QueuePresent();

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
