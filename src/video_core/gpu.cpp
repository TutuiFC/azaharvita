// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef __PSVITA__
#include <pthread.h>
#endif
#include "common/archives.h"
#include "common/hacks/hack_manager.h"
#include "common/microprofile.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "core/loader/loader.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/gpu.h"
#ifdef __PSVITA__
#include "common/vita_diag.h"
#endif
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

namespace VideoCore {

constexpr VAddr VADDR_LCD = 0x1ED02000;
constexpr VAddr VADDR_GPU = 0x1EF00000;

class DelayGenerator {
private:
    DelayGenerator() = default;

    // Average transfer speed based on measurements taken from real
    // hardware. 4 different modes have been taken into consideration:
    // RAM -> RAM, RAM -> VRAM, VRAM -> RAM and VRAM -> VRAM.
    // Furthermore, measurements are split into DMA transfers and tex
    // copies. For simplicity, we will assume fills are as fast as
    // texture copies.

    static constexpr double mibps_to_ns_per_byte(double mib_per_sec) {
        return 1'000'000'000.0 / (mib_per_sec * 1024.0 * 1024.0);
    }

    static constexpr std::array<std::array<double, 4>, 2> speed_mibps = {
        {{
             190.0, // DMA RAMTORAM
             310.0, // DMA RAMTOVRAM
             380.0, // DMA VRAMTORAM
             380.0, // DMA VRAMTOVRAM
         },
         {
             450.0,  // TEX RAMTORAM
             3100.0, // TEX RAMTOVRAM
             5400.0, // TEX VRAMTORAM
             5400.0, // TEX VRAMTOVRAM
         }}};

public:
    enum class CopyMode {
        RAMTORAM,
        RAMTOVRAM,
        VRAMTORAM,
        VRAMTOVRAM,
    };

    static CopyMode GetCopyMode(bool input_vram, bool output_vram) {
        if (!input_vram && !output_vram) {
            return CopyMode::RAMTORAM;
        } else if (!input_vram && output_vram) {
            return CopyMode::RAMTOVRAM;
        } else if (input_vram && !output_vram) {
            return CopyMode::VRAMTORAM;
        } else {
            return CopyMode::VRAMTOVRAM;
        }
    }

    static u64 CalculateDelayNanoseconds(CopyMode mode, bool is_textre, size_t size) {
        double base_ns_per_byte =
            mibps_to_ns_per_byte(speed_mibps[is_textre][static_cast<u32>(mode)]);

        return static_cast<u64>(size * base_ns_per_byte);
    }
};

/**
 * El hilo de la GPU (0.2.0.0). Ver GPU::async_enabled en gpu.h.
 */
struct GPU::AsyncWorker {
    enum class Type : u8 {
        Command,
        BufferSwap,
        Present,
    };
    struct Item {
        Type type = Type::Command;
        u32 screen_id = 0;
        Service::GSP::Command command{};
        Service::GSP::FrameBufferInfo info{};
        /// RequestDma copia con la tabla de paginas del proceso que la pidio,
        /// que se toma al encolar: el nucleo emulado no se lee desde este hilo.
        std::shared_ptr<Kernel::Process> process;
        std::chrono::microseconds time_us{};
    };

    /// Pila explicita, para saber que direcciones son de este hilo (ver
    /// OnGpuThread). 1 MB, como el compilador de shaders: aqui corre todo el
    /// video_core, y nunca se ha medido cuanta pila gasta en el hilo principal.
    static constexpr std::size_t kStackSize = 1024 * 1024;
    /// Presentaciones en cola como mucho. Con mas, la imagen iria cada vez mas
    /// atrasada respecto al juego: el hilo de emulacion espera.
    static constexpr u32 kMaxPresents = 2;

    std::mutex mutex;
    /// Lo espera el hilo de la GPU: hay algo en la cola (o hay que parar).
    std::condition_variable work_ready;
    /// Lo espera el de emulacion: la GPU acabo algo o levanto una interrupcion.
    std::condition_variable progress;
    std::deque<Item> queue;
    /// Ordenes de GSP en cola o ejecutandose, las unicas que levantan
    /// interrupciones. Atomico para mirarlo sin el cerrojo en cada RunLoop.
    std::atomic<u32> command_work{0};
    u32 presents = 0;
    bool executing = false;
    bool stop = false;
    bool failed = false;
    std::string error;

    /// Interrupciones levantadas en el hilo de la GPU y aun sin dar a GSP, y
    /// las que se estan dando (se intercambian para no reservar memoria).
    std::vector<std::pair<Service::GSP::InterruptId, u64>> interrupts;
    std::vector<std::pair<Service::GSP::InterruptId, u64>> delivering;
    std::atomic<bool> interrupts_ready{false};
    /// El de GSP. Solo se llama desde el hilo de emulacion.
    Service::GSP::InterruptHandler handler;

    std::atomic<std::uintptr_t> stack_lo{0};
    std::atomic<std::uintptr_t> stack_hi{0};
    std::chrono::microseconds present_time_us{};
#ifdef __PSVITA__
    pthread_t thread{};
#endif
    bool running = false;
};

namespace {
u64 NowUs() {
#ifdef __PSVITA__
    return Common::VitaMicros();
#else
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
#endif
}
} // Anonymous namespace

MICROPROFILE_DEFINE(GPU_DisplayTransfer, "GPU", "DisplayTransfer", MP_RGB(100, 100, 255));
MICROPROFILE_DEFINE(GPU_CmdlistProcessing, "GPU", "Cmdlist Processing", MP_RGB(100, 255, 100));

GPU::GPU(Core::System& system, Frontend::EmuWindow& emu_window,
         Frontend::EmuWindow* secondary_window)
    : right_eye_disabler{std::make_unique<RightEyeDisabler>(*this)},
      impl{std::make_unique<Impl>(system, emu_window, secondary_window)} {
    impl->vblank_event = impl->timing.RegisterEvent(
        "GPU::VBlankCallback",
        [this](uintptr_t user_data, s64 cycles_late) { VBlankCallback(user_data, cycles_late); });
    impl->timing.ScheduleEvent(FRAME_TICKS, impl->vblank_event);

    // Bind the rasterizer to the PICA GPU
    impl->pica.BindRasterizer(impl->rasterizer);

    if (async_enabled.load(std::memory_order_relaxed)) {
        StartThread();
    }
}

GPU::~GPU() {
    StopThread();
}

PAddr GPU::VirtualToPhysicalAddress(VAddr addr) {
    if (addr == 0) {
        return 0;
    }

    if (addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END) {
        return addr - Memory::VRAM_VADDR + Memory::VRAM_PADDR;
    }
    if (addr >= Memory::LINEAR_HEAP_VADDR && addr <= Memory::LINEAR_HEAP_VADDR_END) {
        return addr - Memory::LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    if (addr >= Memory::NEW_LINEAR_HEAP_VADDR && addr <= Memory::NEW_LINEAR_HEAP_VADDR_END) {
        return addr - Memory::NEW_LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    PAddr plg_fb_addr;
    if (addr >= Memory::PLUGIN_3GX_FB_VADDR && addr <= Memory::PLUGIN_3GX_FB_VADDR_END &&
        (plg_fb_addr = impl->system.Memory().Plugin3GXFramebufferAddress())) {
        return addr - Memory::PLUGIN_3GX_FB_VADDR + plg_fb_addr;
    }

    LOG_ERROR(HW_Memory, "Unknown virtual address @ 0x{:08X}", addr);
    return addr;
}

void GPU::SetInterruptHandler(Service::GSP::InterruptHandler handler) {
    if (async != nullptr) {
        async->handler = std::move(handler);
        Service::GSP::InterruptHandler routed = [this](Service::GSP::InterruptId interrupt_id,
                                                       u64 wait_delay_ns) {
            RouteInterrupt(interrupt_id, wait_delay_ns);
        };
        impl->signal_interrupt = routed;
        impl->pica.SetInterruptHandler(routed);
        return;
    }
    impl->signal_interrupt = handler;
    impl->pica.SetInterruptHandler(handler);
}

void GPU::FlushRegion(PAddr addr, u32 size) {
    Sync();
    impl->rasterizer->FlushRegion(addr, size);
}

void GPU::InvalidateRegion(PAddr addr, u32 size) {
    Sync();
    impl->rasterizer->InvalidateRegion(addr, size);
}

void GPU::ClearAll(bool flush) {
    Sync();
    impl->rasterizer->ClearAll(flush);
}

namespace {
/// Acumula en 'sink' lo que se tarde en el bloque. Ver VideoCore::GxStats.
class GxScopeTimer {
public:
    explicit GxScopeTimer(std::atomic<u64>& sink_)
        : sink{sink_}, start{std::chrono::steady_clock::now()} {}
    ~GxScopeTimer() {
        const auto elapsed = std::chrono::steady_clock::now() - start;
        sink.fetch_add(
            static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
            std::memory_order_relaxed);
    }

private:
    std::atomic<u64>& sink;
    std::chrono::steady_clock::time_point start;
};
} // Anonymous namespace

void GPU::Execute(const Service::GSP::Command& command) {
    if (Async()) {
        QueueCommand(command);
        return;
    }
    ExecuteCommand(command, nullptr);
}

void GPU::ExecuteCommand(const Service::GSP::Command& command,
                         const std::shared_ptr<Kernel::Process>& dma_process) {
    using Service::GSP::CommandId;
    auto& regs = impl->pica.regs;

    switch (command.id) {
    case CommandId::RequestDma: {
        const GxScopeTimer timer{GxStats::dma_ns};
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
        impl->system.Memory().RasterizerFlushVirtualRegion(
            command.dma_request.source_address, command.dma_request.size, Memory::FlushMode::Flush);
        impl->system.Memory().RasterizerFlushVirtualRegion(command.dma_request.dest_address,
                                                           command.dma_request.size,
                                                           Memory::FlushMode::Invalidate);

        // TODO(Subv): These memory accesses should not go through the application's memory mapping.
        // They should go through the GSP module's memory mapping.
        const auto process =
            dma_process != nullptr ? dma_process : impl->system.Kernel().GetCurrentProcess();
        impl->memory.CopyBlock(*process, command.dma_request.dest_address,
                               command.dma_request.source_address, command.dma_request.size);

        auto is_vram = [&](u32 addr) {
            return addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END;
        };

        u64 delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(is_vram(command.dma_request.source_address),
                                        is_vram(command.dma_request.dest_address)),
            false, command.dma_request.size);

        impl->signal_interrupt(Service::GSP::InterruptId::DMA, delay);
        break;
    }
    case CommandId::SubmitCmdList: {
        const GxScopeTimer timer{GxStats::cmdlist_ns};
        GxStats::cmdlist_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
        auto& params = command.submit_gpu_cmdlist;
        auto& cmdbuffer = regs.internal.pipeline.command_buffer;

        // Write to the command buffer GPU registers
        cmdbuffer.addr[0].Assign(VirtualToPhysicalAddress(params.address) >> 3);
        cmdbuffer.size[0].Assign(params.size >> 3);
        cmdbuffer.trigger[0] = 1;

        // Trigger processing of the command list
        SubmitCmdList(0);
        break;
    }
    case CommandId::MemoryFill: {
        const GxScopeTimer timer{GxStats::fill_ns};
        GxStats::fill_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
        auto& params = command.memory_fill;
        auto& memfill = regs.memory_fill_config;

        // Write to the memory fill GPU registers.
        // If both buffers are set GSP dispatches PSC0 only.
        const bool has_both_bufs = params.start1 != 0 && params.start2 != 0;
        if (params.start1 != 0) {
            memfill[0].address_start = VirtualToPhysicalAddress(params.start1) >> 3;
            memfill[0].address_end = VirtualToPhysicalAddress(params.end1) >> 3;
            memfill[0].value_32bit = params.value1;
            memfill[0].control = params.control1;
            MemoryFill(0, has_both_bufs ? std::numeric_limits<u32>::max() : 0);
        }
        if (params.start2 != 0) {
            memfill[1].address_start = VirtualToPhysicalAddress(params.start2) >> 3;
            memfill[1].address_end = VirtualToPhysicalAddress(params.end2) >> 3;
            memfill[1].value_32bit = params.value2;
            memfill[1].control = params.control2;
            MemoryFill(1, has_both_bufs ? 0 : 1);
        }
        break;
    }
    case CommandId::DisplayTransfer: {
        const GxScopeTimer timer{GxStats::transfer_ns};
        GxStats::transfer_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
        auto& params = command.display_transfer;
        auto& display_transfer = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        display_transfer.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        display_transfer.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        display_transfer.input_size = params.in_buffer_size;
        display_transfer.output_size = params.out_buffer_size;
        display_transfer.flags = params.flags;
        display_transfer.trigger.Assign(1);

        // Trigger the display transfer.
        MemoryTransfer();
        break;
    }
    case CommandId::TextureCopy: {
        const GxScopeTimer timer{GxStats::transfer_ns};
        GxStats::transfer_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
        auto& params = command.texture_copy;
        auto& texture_copy = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        texture_copy.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        texture_copy.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        texture_copy.texture_copy.size = params.size;
        texture_copy.texture_copy.input_size = params.in_width_gap;
        texture_copy.texture_copy.output_size = params.out_width_gap;
        texture_copy.flags = params.flags;
        texture_copy.trigger.Assign(1);

        // Trigger the texture copy.
        MemoryTransfer();
        break;
    }
    case CommandId::CacheFlush: {
        // Rasterizer flushing handled elsewhere in CPU read/write and other GPU handlers
        // Use command.cache_flush.regions to implement this handler
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Unknown command {:#08X}", command.id.Value());
    }

    // Notify debugger that a GSP command was processed.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::GSPCommandProcessed, &command);
    }
}

void GPU::SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    if (Async() && !OnGpuThread()) {
        QueueBufferSwap(screen_id, info);
        return;
    }
    ApplyBufferSwap(screen_id, info);
}

void GPU::ApplyBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    const PAddr phys_address_left = VirtualToPhysicalAddress(info.address_left);
    const PAddr phys_address_right = VirtualToPhysicalAddress(info.address_right);

    // Update framebuffer properties.
    auto& framebuffer = impl->pica.regs.framebuffer_config[screen_id];
    if (info.active_fb == 0) {
        framebuffer.address_left1 = phys_address_left;
        framebuffer.address_right1 = phys_address_right;
    } else {
        framebuffer.address_left2 = phys_address_left;
        framebuffer.address_right2 = phys_address_right;
    }

    framebuffer.stride = info.stride;
    framebuffer.format = info.format;
    framebuffer.active_fb = info.shown_fb;

    // Notify debugger about the buffer swap.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::BufferSwapped, nullptr);
    }

    if (screen_id == 0) {
        MicroProfileFlip();
        impl->system.perf_stats->EndGameFrame();
        right_eye_disabler->ReportEndFrame();
    }
}

void GPU::SetColorFill(const Pica::ColorFill& fill) {
    Sync();
    impl->pica.regs_lcd.color_fill_top = fill;
    impl->pica.regs_lcd.color_fill_bottom = fill;
}

u32 GPU::ReadReg(VAddr addr) {
    Sync();
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        return impl->pica.regs_lcd[index];
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        return impl->pica.regs.reg_array[index];
    }
    default:
        UNREACHABLE_MSG("Read from unknown GPU address {:#08X}", addr);
    }
}

void GPU::WriteReg(VAddr addr, u32 data) {
    // Una escritura puede lanzar un relleno, una copia o una lista: se ejecuta
    // aqui, con el hilo de la GPU parado, despues de todo lo encolado.
    Sync();
#ifdef __PSVITA__
    // 0.1.5.2 (4.7): cuanto del "resto de gx" es decodificar escrituras de
    // registro PICA. El timer es RAII para que tambien cuente los returns
    // tempranos de los ASSERT; el overlay lo anota en crash.txt junto al
    // numero de fills y transfers del intervalo.
    const auto regw_begin = std::chrono::steady_clock::now();
    struct RegwTimer {
        std::chrono::steady_clock::time_point begin;
        ~RegwTimer() {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            Common::FrameStats::pica_reg_writes.fetch_add(1, std::memory_order_relaxed);
            Common::FrameStats::pica_reg_write_ns.fetch_add(
                static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                std::memory_order_relaxed);
        }
    } regw_timer{regw_begin};
#endif
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        impl->pica.regs_lcd[index] = data;
        break;
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);

        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        impl->pica.regs.reg_array[index] = data;

        // Handle registers that trigger GPU actions
        switch (index) {
        case GPU_REG_INDEX(memory_fill_config[0].trigger):
            MemoryFill(0, 0);
            GxStats::fill_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
            break;
        case GPU_REG_INDEX(memory_fill_config[1].trigger):
            MemoryFill(1, 1);
            GxStats::fill_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
            break;
        case GPU_REG_INDEX(display_transfer_config.trigger):
            MemoryTransfer();
            GxStats::transfer_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[0]):
            SubmitCmdList(0);
            GxStats::cmdlist_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[1]):
            SubmitCmdList(1);
            GxStats::cmdlist_count.fetch_add(1, std::memory_order_relaxed);
        GxStats::frame_work.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
        }
        break;
    }
    default:
        UNREACHABLE_MSG("Write to unknown GPU address {:#08X}", addr);
    }
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

Pica::PicaCore& GPU::PicaCore() {
    return impl->pica;
}

const Pica::PicaCore& GPU::PicaCore() const {
    return impl->pica;
}

GraphicsDebugger& GPU::Debugger() {
    return impl->gpu_debugger;
}

void GPU::ApplyPerProgramSettings(u64 program_ID) {
    Sync();
    auto hack = Common::Hacks::hack_manager.GetHack(
        Common::Hacks::HackType::ACCURATE_MULTIPLICATION, program_ID);
    bool use_accurate_mul = Settings::values.shaders_accurate_mul.GetValue();
    if (hack) {
        switch (hack->mode) {
        case Common::Hacks::HackAllowMode::DISALLOW:
            use_accurate_mul = false;
            break;
        case Common::Hacks::HackAllowMode::FORCE:
            use_accurate_mul = true;
            break;
        case Common::Hacks::HackAllowMode::ALLOW:
        default:
            break;
        }
    }
    impl->rasterizer->SetAccurateMul(use_accurate_mul);
}

void GPU::SubmitCmdList(u32 index) {
    // Check if a command list was triggered.
    auto& config = impl->pica.regs.internal.pipeline.command_buffer;
    if (!config.trigger[index]) {
        return;
    }

    MICROPROFILE_SCOPE(GPU_CmdlistProcessing);

    // Forward command list processing to the PICA core.
    const PAddr addr = config.GetPhysicalAddress(index);
    const u32 size = config.GetSize(index);
    impl->pica.ProcessCmdList(addr, size,
                              !right_eye_disabler->ShouldAllowCmdQueueTrigger(addr, size));
    config.trigger[index] = 0;
}

void GPU::MemoryFill(u32 index, u32 intr_index) {
    // Check if a memory fill was triggered.
    auto& config = impl->pica.regs.memory_fill_config[index];
    if (!config.trigger) {
        return;
    }

    // Perform memory fill.
    if (!impl->rasterizer->AccelerateFill(config)) {
        impl->sw_blitter->MemoryFill(config);
    }

    // Treat fill as texture transfer from VRAM
    u64 delay = DelayGenerator::CalculateDelayNanoseconds(
        DelayGenerator::GetCopyMode(true, config.IsVRAM()), true,
        config.GetEndAddress() - config.GetStartAddress());

    // It seems that it won't signal interrupt if "address_start" is zero.
    // TODO: hwtest this
    if (config.GetStartAddress() != 0) {
        if (intr_index == 0) {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC0, delay);
        } else if (intr_index == 1) {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC1, delay);
        }
    }

    // Reset "trigger" flag and set the "finish" flag
    // This was confirmed to happen on hardware even if "address_start" is zero.
    config.trigger.Assign(0);
    config.finished.Assign(1);
}

void GPU::MemoryTransfer() {
    // Check if a transfer was triggered.
    auto& config = impl->pica.regs.display_transfer_config;
    if (!config.trigger.Value()) {
        return;
    }

    MICROPROFILE_SCOPE(GPU_DisplayTransfer);

    // Notify debugger about the display transfer.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::IncomingDisplayTransfer, nullptr);
    }

    u64 delay{};
    // Perform memory transfer
    if (config.is_texture_copy) {
        if (!impl->rasterizer->AccelerateTextureCopy(config)) {
            impl->sw_blitter->TextureCopy(config);
        }
        delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(config.IsInputVRAM(), config.IsOutputVRAM()), true,
            config.texture_copy.size);
    } else {
        if (right_eye_disabler->ShouldAllowDisplayTransfer(config.GetPhysicalInputAddress(),
                                                           config.input_height)) {
            if (!impl->rasterizer->AccelerateDisplayTransfer(config)) {
                impl->sw_blitter->DisplayTransfer(config);
            }
        }
        delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(config.IsInputVRAM(), config.IsOutputVRAM()), true,
            config.input_width * config.input_height * BytesPerPixel(config.input_format));
    }

    // Complete transfer.
    config.trigger.Assign(0);
    impl->signal_interrupt(Service::GSP::InterruptId::PPF, delay);
}

void GPU::VBlankCallback(std::uintptr_t user_data, s64 cycles_late) {
#ifdef __PSVITA__
    // El VBlank tendria que saltar cada ~4 rodajas de emulacion, o sea unas
    // cuatro veces antes del punto donde se muere. Si no aparece ninguna de
    // estas lineas en crash.txt, es que el evento no llega a dispararse; si
    // aparece "vblank" pero no "vblank-gpu", muere dentro del aviso a GSP.
    //
    // CON PRESUPUESTO, igual que el rastro del interprete (ver g_trace_budget en
    // arm_dyncom_interpreter.cpp). Sin el, esto se ejecutaba UNA VEZ POR
    // FOTOGRAMA durante toda la partida, y VitaNote no es barato: cada llamada
    // hace dos sceIoMkdir, un sceIoOpen con O_APPEND, un sceIoWrite y un
    // sceIoClose contra la tarjeta de memoria. Eran diez llamadas al sistema de
    // ficheros por fotograma pagadas para siempre a cambio de un diagnostico
    // que solo sirve en los primeros fotogramas -- si el evento dispara, lo
    // sabemos ya; si no dispara, se ve igual de bien en las ocho primeras.
    static int vblank_budget = 8;
    const bool note_vblank = vblank_budget > 0;
    if (note_vblank) {
        vblank_budget--;
        Common::VitaNote("vblank", "entrando");
    }
#endif

    // Signal to GSP that GPU interrupt has occurred
    impl->signal_interrupt(Service::GSP::InterruptId::PDC0, 0);
    impl->signal_interrupt(Service::GSP::InterruptId::PDC1, 0);

#ifdef __PSVITA__
    if (note_vblank) {
        Common::VitaNote("vblank-gpu", "interrupciones avisadas, presentando");
    }
#endif

    // Present renderered frame.
    if (Async()) {
        QueuePresent();
    } else {
        impl->renderer->SwapBuffers();
    }

    // Reschedule recurrent event
    impl->timing.ScheduleEvent(FRAME_TICKS - cycles_late, impl->vblank_event);
}

void GPU::RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window) {
    // Reset the renderer (this will destroy OpenGL resources)
    impl->renderer.reset();

    // Create a new renderer
    impl->renderer =
        VideoCore::CreateRenderer(emu_window, secondary_window, impl->pica, impl->system);
    impl->rasterizer = impl->renderer->Rasterizer();

    // Rebind the rasterizer to the PICA GPU
    impl->pica.BindRasterizer(impl->rasterizer);

    // Update the sw_blitter with the new rasterizer
    impl->sw_blitter = std::make_unique<SwRenderer::SwBlitter>(impl->memory, impl->rasterizer);

    // Re-apply per-game configuration and reload disk shader cache
    u64 program_id{};
    impl->system.GetAppLoader().ReadProgramId(program_id);
    ApplyPerProgramSettings(program_id);
    if (Settings::values.use_disk_shader_cache) {
        impl->renderer->Rasterizer()->LoadDefaultDiskResources(false, nullptr);
    }

    // Mark ALL GPU registers as dirty so current state gets uploaded to new renderer
    impl->pica.dirty_regs.SetAllDirty();

    // Also mark shader setups as dirty so uniforms get re-uploaded and
    // stale pointers to the old rasterizer's JIT cache are cleared.
    impl->pica.vs_setup.uniforms_dirty = true;
    impl->pica.vs_setup.cached_shader = nullptr;
    impl->pica.gs_setup.uniforms_dirty = true;
    impl->pica.gs_setup.cached_shader = nullptr;

    // Mark all cached LUT/table state in pica as dirty
    impl->pica.lighting.lut_dirty = impl->pica.lighting.LutAllDirty;
    impl->pica.fog.lut_dirty = true;
    impl->pica.proctex.table_dirty = impl->pica.proctex.TableAllDirty;
}

void GPU::ReleaseRenderer() {
    // Just reset the renderer to release OpenGL resources
    // Don't null out rasterizer pointer as it will become dangling
    impl->renderer.reset();
    impl->sw_blitter.reset();
    LOG_INFO(HW_GPU, "Renderer released for context destroy");
}

bool GPU::Async() const {
    return async != nullptr && async->running;
}

bool GPU::OnGpuThread() const {
    if (async == nullptr) {
        return false;
    }
    volatile char marker = 0;
    const auto here = reinterpret_cast<std::uintptr_t>(&marker);
    return here >= async->stack_lo.load(std::memory_order_relaxed) &&
           here < async->stack_hi.load(std::memory_order_relaxed);
}

void GPU::StartThread() {
#ifdef __PSVITA__
    async = std::make_unique<AsyncWorker>();
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, AsyncWorker::kStackSize);
    const int rc = pthread_create(
        &async->thread, &attr,
        [](void* gpu) -> void* {
            static_cast<GPU*>(gpu)->AsyncLoop();
            return nullptr;
        },
        this);
    pthread_attr_destroy(&attr);
    if (rc != 0) {
        Common::VitaNote("gpu", "no se pudo crear el hilo de la GPU: va en el de emulacion");
        async.reset();
        return;
    }
    async->running = true;
#endif
}

void GPU::StopThread() {
#ifdef __PSVITA__
    if (!Async()) {
        return;
    }
    {
        std::lock_guard lock{async->mutex};
        async->stop = true;
        async->queue.clear();
        async->command_work.store(0, std::memory_order_relaxed);
        async->presents = 0;
    }
    async->work_ready.notify_all();
    pthread_join(async->thread, nullptr);
    async->running = false;
    // Su pila ya no existe y otro hilo puede recibir esa memoria.
    async->stack_lo.store(0, std::memory_order_relaxed);
    async->stack_hi.store(0, std::memory_order_relaxed);
    Common::vita_gpu_stack_lo.store(0, std::memory_order_relaxed);
    Common::vita_gpu_stack_hi.store(0, std::memory_order_relaxed);
#endif
}

void GPU::AsyncLoop() {
    AsyncWorker& worker = *async;
    {
        // Lo que este por debajo es la pila de este hilo (crece hacia abajo).
        // Los margenes dejan fuera los bordes, que puede compartir con otra.
        volatile char marker = 0;
        const auto top = reinterpret_cast<std::uintptr_t>(&marker);
        const std::uintptr_t lo = top - AsyncWorker::kStackSize + 4096;
        const std::uintptr_t hi = top + 512;
        worker.stack_lo.store(lo, std::memory_order_relaxed);
        worker.stack_hi.store(hi, std::memory_order_relaxed);
#ifdef __PSVITA__
        Common::vita_gpu_stack_lo.store(lo, std::memory_order_relaxed);
        Common::vita_gpu_stack_hi.store(hi, std::memory_order_relaxed);
#endif
    }
#ifdef __PSVITA__
    // El FPSCR es por hilo: sin esto la GPU emulada redondearia distinto que
    // en el hilo de emulacion.
    Common::VitaEnableFastFloatMode();
    Common::VitaPinThreadToUserCore(1, "gpu");
#endif
    const auto fail = [&worker](const char* what) {
        std::lock_guard lock{worker.mutex};
        worker.failed = true;
        worker.error = what != nullptr ? what : "?";
        worker.interrupts_ready.store(true, std::memory_order_release);
    };
    while (true) {
        AsyncWorker::Item item;
        {
            std::unique_lock lock{worker.mutex};
#ifdef __PSVITA__
            Common::vita_gpu_stage.store("gpu: sin trabajo", std::memory_order_relaxed);
#endif
            worker.work_ready.wait(lock,
                                   [&worker] { return worker.stop || !worker.queue.empty(); });
            if (worker.stop) {
                break;
            }
            item = std::move(worker.queue.front());
            worker.queue.pop_front();
            worker.executing = true;
        }
        const u64 begin = NowUs();
        // Muerto el hilo, se sigue vaciando la cola sin ejecutar para que nadie
        // se quede esperando; el de emulacion lo ve en DeliverInterrupts.
        if (!worker.failed) {
            try {
                switch (item.type) {
                case AsyncWorker::Type::Command:
#ifdef __PSVITA__
                    Common::vita_gpu_stage.store("gpu: orden de GSP", std::memory_order_relaxed);
#endif
                    ExecuteCommand(item.command, item.process);
                    break;
                case AsyncWorker::Type::BufferSwap:
                    ApplyBufferSwap(item.screen_id, item.info);
                    break;
                case AsyncWorker::Type::Present:
#ifdef __PSVITA__
                    Common::vita_gpu_stage.store("gpu: presentar", std::memory_order_relaxed);
#endif
                    worker.present_time_us = item.time_us;
                    impl->renderer->SwapBuffers();
                    break;
                }
            } catch (const std::exception& e) {
                fail(e.what());
            } catch (...) {
                fail("excepcion de tipo no estandar");
            }
        }
        item.process.reset();
        GxStats::thread_busy_us.fetch_add(NowUs() - begin, std::memory_order_relaxed);
        {
            std::lock_guard lock{worker.mutex};
            worker.executing = false;
            if (item.type == AsyncWorker::Type::Command) {
                worker.command_work.fetch_sub(1, std::memory_order_relaxed);
            } else if (item.type == AsyncWorker::Type::Present) {
                worker.presents--;
            }
        }
        worker.progress.notify_all();
    }
}

void GPU::RouteInterrupt(Service::GSP::InterruptId interrupt_id, u64 wait_delay_ns) {
    if (OnGpuThread()) {
        {
            std::lock_guard lock{async->mutex};
            async->interrupts.emplace_back(interrupt_id, wait_delay_ns);
            async->interrupts_ready.store(true, std::memory_order_release);
        }
        async->progress.notify_all();
        return;
    }
    async->handler(interrupt_id, wait_delay_ns);
}

void GPU::DeliverInterrupts() {
    if (async == nullptr || !async->interrupts_ready.load(std::memory_order_acquire)) {
        return;
    }
    {
        std::lock_guard lock{async->mutex};
        if (async->failed) {
            throw std::runtime_error("hilo de la GPU: " + async->error);
        }
        async->delivering.swap(async->interrupts);
        async->interrupts_ready.store(false, std::memory_order_relaxed);
    }
    for (const auto& [interrupt_id, wait_delay_ns] : async->delivering) {
        async->handler(interrupt_id, wait_delay_ns);
    }
    async->delivering.clear();
}

bool GPU::HasInterruptWork() const {
    return Async() && async->command_work.load(std::memory_order_relaxed) != 0;
}

void GPU::WaitForInterrupts() {
    if (!Async()) {
        return;
    }
    const u64 begin = NowUs();
    {
#ifdef __PSVITA__
        const Common::ScopedVitaStage stage{"esperando a la gpu"};
#endif
        std::unique_lock lock{async->mutex};
        async->progress.wait(lock, [this] {
            return !async->interrupts.empty() || async->failed ||
                   async->command_work.load(std::memory_order_relaxed) == 0;
        });
    }
    GxStats::irq_waits.fetch_add(1, std::memory_order_relaxed);
    GxStats::emu_wait_us.fetch_add(NowUs() - begin, std::memory_order_relaxed);
}

void GPU::Sync() {
    if (!Async() || OnGpuThread()) {
        return;
    }
    std::unique_lock lock{async->mutex};
    if (async->queue.empty() && !async->executing) {
        return;
    }
    const u64 begin = NowUs();
    {
#ifdef __PSVITA__
        const Common::ScopedVitaStage stage{"esperando a la gpu (sincronizar)"};
#endif
        async->progress.wait(lock, [this] { return async->queue.empty() && !async->executing; });
    }
    GxStats::syncs.fetch_add(1, std::memory_order_relaxed);
    GxStats::emu_wait_us.fetch_add(NowUs() - begin, std::memory_order_relaxed);
}

std::chrono::microseconds GPU::PresentTimeUs() const {
    if (OnGpuThread()) {
        return async->present_time_us;
    }
    return impl->timing.GetGlobalTimeUs();
}

void GPU::QueueCommand(const Service::GSP::Command& command) {
    AsyncWorker::Item item;
    item.type = AsyncWorker::Type::Command;
    item.command = command;
    if (command.id == Service::GSP::CommandId::RequestDma) {
        item.process = impl->system.Kernel().GetCurrentProcess();
    }
    u32 depth = 0;
    {
        std::lock_guard lock{async->mutex};
        async->queue.push_back(std::move(item));
        async->command_work.fetch_add(1, std::memory_order_relaxed);
        depth = static_cast<u32>(async->queue.size());
    }
    async->work_ready.notify_one();
    u32 seen = GxStats::queue_max.load(std::memory_order_relaxed);
    while (depth > seen &&
           !GxStats::queue_max.compare_exchange_weak(seen, depth, std::memory_order_relaxed)) {
    }
}

void GPU::QueueBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    AsyncWorker::Item item;
    item.type = AsyncWorker::Type::BufferSwap;
    item.screen_id = screen_id;
    item.info = info;
    {
        std::lock_guard lock{async->mutex};
        async->queue.push_back(std::move(item));
    }
    async->work_ready.notify_one();
}

void GPU::QueuePresent() {
    AsyncWorker::Item item;
    item.type = AsyncWorker::Type::Present;
    item.time_us = impl->timing.GetGlobalTimeUs();
    {
        std::unique_lock lock{async->mutex};
        if (async->presents >= AsyncWorker::kMaxPresents) {
            const u64 begin = NowUs();
            {
#ifdef __PSVITA__
                const Common::ScopedVitaStage stage{"esperando a la gpu (presentar)"};
#endif
                async->progress.wait(
                    lock, [this] { return async->presents < AsyncWorker::kMaxPresents; });
            }
            GxStats::present_waits.fetch_add(1, std::memory_order_relaxed);
            GxStats::emu_wait_us.fetch_add(NowUs() - begin, std::memory_order_relaxed);
        }
        async->presents++;
        async->queue.push_back(std::move(item));
    }
    async->work_ready.notify_one();
}

template <class Archive>
void GPU::serialize(Archive& ar, const u32 file_version) {
    ar & impl->pica;
}

SERIALIZE_IMPL(GPU)

} // namespace VideoCore
