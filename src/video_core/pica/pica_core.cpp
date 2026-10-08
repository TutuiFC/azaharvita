// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <vector>
#include "common/arch.h"
#include "common/archives.h"
#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/pica/pica_core.h"
#include "common/vita_diag.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader/shader.h"
#ifdef __PSVITA__
#include "common/thread_worker.h"
#endif

namespace Pica {

MICROPROFILE_DEFINE(GPU_Drawing, "GPU", "Drawing", MP_RGB(50, 50, 240));

using namespace DebugUtils;

#ifdef __PSVITA__
/// Estado del sombreado repartido. Va aqui arriba, antes del destructor de
/// PicaCore, porque unique_ptr necesita el tipo completo para destruirlo. La
/// explicacion esta en ShadeVerticesParallel.
struct PicaCore::ParallelShading {
    /**
     * UN AYUDANTE, EN EL NUCLEO 0 (0.2.3.8). Eran dos, en los nucleos 1 y 2,
     * de cuando el primer tramo lo hacia el hilo de emulacion (nucleo 0). Con
     * el hilo de la GPU (nucleo 1) haciendolo, el ayudante del 1 y el se
     * turnaban en el mismo nucleo, el 0 se quedaba parado (el juego esperando
     * a la GPU: 77 % en un combate de Pokemon Sol) y el 2 lo ocupaba el otro
     * ayudante, que es donde compila el compilador de shaders: con los
     * ayudantes por encima, un shader de 18 KB tardaba 45 s, y mientras tanto
     * sus lotes iban por aqui. Ahora: el tramo 0 en el hilo de la GPU, el 1
     * en el nucleo 0, y el 2 entero para el compilador (ver CgWorkerMain).
     */
    Common::StatefulThreadWorker<> workers{1, "VertexShader workers", {}, true, 0};
    struct Job {
        u32 index;  ///< Posicion en el lote (la que recibe LoadVertex).
        u32 vertex; ///< Vertice a cargar.
    };
    std::vector<Job> jobs;
    /// Por posicion del lote: trabajo del que sale su resultado, o
    /// kFromSlot | hueco si sale de un segmento anterior (ver slot_data).
    std::vector<u32> submit;
    /**
     * Resultados YA CONVERTIDOS a OutputVertex (0.1.0.45).
     *
     * Antes se guardaba la salida cruda del shader y la conversion a
     * OutputVertex la hacia la entrega, en el nucleo 0, UNA VEZ POR POSICION
     * del lote: con dibujado indexado cada vertice se entrega unas 2,4 veces,
     * asi que eran ~30.000 conversiones por vblank en un solo nucleo. La
     * conversion depende solo de la salida del shader y de los registros del
     * rasterizador (fijos en todo el lote), asi que ahora se hace una vez por
     * vertice unico y en el nucleo que lo sombrea. Ademas ocupa 96 bytes en vez
     * de 256.
     */
    std::vector<OutputVertex> outputs;
    std::array<OutputVertex, 64> slot_data;
    /// Hueco del cache donde entro cada vertice por ultima vez (los indices son
    /// de 8 o 16 bits, asi que caben todos). Siempre < 64, empezando en 0.
    std::array<u8, 65536> slot_of{};
    /**
     * Cache COMPLETA por lote (0.1.6.1): trabajo que sombreo cada indice en el
     * lote actual, valido si stamp_of[indice] == stamp. Ver g_full_vertex_dedup.
     * El sello evita limpiar 256 KB en cada lote.
     */
    std::array<u32, 65536> job_of{};
    std::array<u32, 65536> stamp_of{};
    u32 stamp = 0;
};

/**
 * SOMBREAR CADA INDICE UNA SOLA VEZ POR LOTE (0.1.6.1).
 *
 * El cache de 64 vertices con reemplazo circular es el del hardware, pero aqui
 * no hace falta imitarlo: el resultado de sombrear un vertice depende solo de
 * ese vertice (mismo programa y mismos uniforms en todo el lote). Con el cache
 * pequeno, un indice que vuelve a salir despues de 64 vertices nuevos se
 * SOMBREA OTRA VEZ entero y da lo mismo. En la cinematica de Rubi Omega se
 * sombraban ~37.000 vertices por vblank para ~30.000 triangulos, y cada
 * vertice se entregaba ~2.4 veces (medido en 0.1.0.45): mucho trabajo repetido.
 *
 * La unica diferencia posible es la que ya asume el reparto en tres nucleos
 * (ver la cabecera de ShadeVerticesParallel): un programa que lea un registro
 * que no escribe en todos los vertices trae el valor del vertice anterior, y
 * ese "anterior" cambia. Ese programa ya es impredecible en la consola de
 * verdad. Aun asi hay interruptor en el menu de ajustes ("Cache vertices
 * completa"): apagado, vuelve el cache de 64 exacto de 0.1.6.0.
 */
std::atomic<bool> g_full_vertex_dedup{true};
#endif

// Class representing implementation details of each internal register
// The set/get pattern is used instead of a bitfield union to allow
// constexpr evaluation.
class RegImplInfo {
private:
    using NeedsSpecialHandlingBF = BitField<0, 1, u16>;
    using SupportsBatchBF = BitField<1, 1, u16>;
    using RegsUntilSpecialBF = BitField<2, 14, u16>;

    u16 raw{};

public:
    constexpr bool NeedsSpecialHandling() const {
        return NeedsSpecialHandlingBF::ExtractValue(raw) != 0;
    }

    constexpr void SetNeedsSpecialHandling() {
        raw = (raw & ~NeedsSpecialHandlingBF::mask) | NeedsSpecialHandlingBF::FormatValue(1);
    }

    constexpr bool SupportsBatch() const {
        return SupportsBatchBF::ExtractValue(raw) != 0;
    }

    constexpr void SetSupportsBatch() {
        raw = (raw & ~SupportsBatchBF::mask) | SupportsBatchBF::FormatValue(1);
    }

    constexpr u16 RegsUntilSpecial() const {
        return RegsUntilSpecialBF::ExtractValue(raw);
    }

    constexpr void SetRegsUntilSpecial(u16 value) {
        raw = (raw & ~RegsUntilSpecialBF::mask) | RegsUntilSpecialBF::FormatValue(value);
    }
};

union CommandHeader {
    u32 hex;
    BitField<0, 16, u32> cmd_id;
    BitField<16, 4, u32> parameter_mask;
    BitField<20, 8, u32> extra_data_length;
    BitField<31, 1, u32> group_commands;
};
static_assert(sizeof(CommandHeader) == sizeof(u32), "CommandHeader has incorrect size!");

PicaCore::PicaCore(Memory::MemorySystem& memory_, std::shared_ptr<DebugContext> debug_context_)
    : memory{memory_}, debug_context{std::move(debug_context_)},
      geometry_pipeline{regs.internal, gs_unit, gs_setup},
      shader_engine{CreateEngine(Settings::values.use_shader_jit.GetValue())} {
    InitializeRegs();
    dirty_regs.SetAllDirty();

    const auto submit_vertex = [this](const AttributeBuffer& buffer) {
        const auto add_triangle = [this](const OutputVertex& v0, const OutputVertex& v1,
                                         const OutputVertex& v2) {
            rasterizer->AddTriangle(v0, v1, v2);
        };
        const auto vertex = OutputVertex(regs.internal.rasterizer, buffer);
        primitive_assembler.SubmitVertex(vertex, add_triangle);
    };

    gs_unit.SetVertexHandlers(submit_vertex, [this]() { primitive_assembler.SetWinding(); });
    geometry_pipeline.SetVertexHandler(submit_vertex);

    primitive_assembler.Reconfigure(PipelineRegs::TriangleTopology::List);
}

PicaCore::~PicaCore() = default;

void PicaCore::InitializeRegs() {
    // Values initialized by GSP
    regs.internal.irq_autostop = 1;
    regs.internal.irq_mask = 0xFFFFFFF0;
    // Older versions of libctru didn't initialize this, initialize it here to avoid endless black
    // screen. Not needed on actual hardware due to previous software already having set it up
    regs.internal.irq_compare = 0x12345678;

    auto& framebuffer_top = regs.framebuffer_config[0];
    auto& framebuffer_sub = regs.framebuffer_config[1];

    // Set framebuffer defaults from nn::gx::Initialize
    framebuffer_top.address_left1 = 0x181E6000;
    framebuffer_top.address_left2 = 0x1822C800;
    framebuffer_top.address_right1 = 0x18273000;
    framebuffer_top.address_right2 = 0x182B9800;
    framebuffer_sub.address_left1 = 0x1848F000;
    framebuffer_sub.address_left2 = 0x184C7800;

    framebuffer_top.width.Assign(240);
    framebuffer_top.height.Assign(400);
    framebuffer_top.stride = 3 * 240;
    framebuffer_top.color_format.Assign(PixelFormat::RGB8);
    framebuffer_top.active_fb = 0;

    framebuffer_sub.width.Assign(240);
    framebuffer_sub.height.Assign(320);
    framebuffer_sub.stride = 3 * 240;
    framebuffer_sub.color_format.Assign(PixelFormat::RGB8);
    framebuffer_sub.active_fb = 0;

    // Tales of Abyss expects this register to have the following default values.
    auto& gs = regs.internal.gs;
    gs.max_input_attribute_index.Assign(1);
    gs.shader_mode.Assign(ShaderRegs::ShaderMode::VS);
}

void PicaCore::BindRasterizer(VideoCore::RasterizerInterface* rasterizer) {
    this->rasterizer = rasterizer;
}

void PicaCore::SetInterruptHandler(Service::GSP::InterruptHandler& signal_interrupt) {
    this->signal_interrupt = signal_interrupt;
}

static consteval std::array<RegImplInfo, RegsInternal::NUM_REGS> BuildRegImplFlagsLUT() {
    std::array<RegImplInfo, RegsInternal::NUM_REGS> table{};

    // Marks the register as needing special handling.
    const auto mark_special = [&table](u32 index) { table[index].SetNeedsSpecialHandling(); };

    // Marks the register as supporting batch writes.
    const auto mark_batch = [&table](u32 index) { table[index].SetSupportsBatch(); };

    // Single registers
    mark_special(PICA_REG_INDEX(irq_request));
    mark_special(PICA_REG_INDEX(pipeline.triangle_topology));
    mark_special(PICA_REG_INDEX(pipeline.restart_primitive));
    mark_special(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index));
    mark_special(PICA_REG_INDEX(pipeline.trigger_draw));
    mark_special(PICA_REG_INDEX(pipeline.trigger_draw_indexed));
    mark_special(PICA_REG_INDEX(gs.bool_uniforms));
    mark_special(PICA_REG_INDEX(vs.output_mask));
    mark_special(PICA_REG_INDEX(vs.bool_uniforms));

    // Array based registers
    for (u32 i = 0; i < 2; ++i) {
        mark_special(PICA_REG_INDEX(pipeline.command_buffer.trigger[0]) + i);
    }
    for (u32 i = 0; i < 3; ++i) {
        mark_special(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]) + i);
        mark_batch(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]) + i);
    }
    for (u32 i = 0; i < 4; ++i) {
        mark_special(PICA_REG_INDEX(vs.int_uniforms[0]) + i);
        mark_special(PICA_REG_INDEX(gs.int_uniforms[0]) + i);
    }
    for (u32 i = 0; i < 8; ++i) {
        mark_special(PICA_REG_INDEX(gs.uniform_setup.set_value[0]) + i);
        mark_batch(PICA_REG_INDEX(gs.uniform_setup.set_value[0]) + i);
        mark_special(PICA_REG_INDEX(vs.uniform_setup.set_value[0]) + i);
        mark_batch(PICA_REG_INDEX(vs.uniform_setup.set_value[0]) + i);
        mark_special(PICA_REG_INDEX(gs.program.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(gs.program.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(vs.program.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(vs.program.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(lighting.lut_data[0]) + i);
        mark_batch(PICA_REG_INDEX(lighting.lut_data[0]) + i);
        mark_special(PICA_REG_INDEX(texturing.fog_lut_data[0]) + i);
        mark_batch(PICA_REG_INDEX(texturing.fog_lut_data[0]) + i);
        mark_special(PICA_REG_INDEX(texturing.proctex_lut_data[0]) + i);
        mark_batch(PICA_REG_INDEX(texturing.proctex_lut_data[0]) + i);
    }

    // Build distances to next special register.
    u16 regs_since_special = std::numeric_limits<u16>::max();
    for (size_t i = RegsInternal::NUM_REGS; i-- > 0;) {
        if (table[i].NeedsSpecialHandling()) {
            regs_since_special = 0;
        }
        table[i].SetRegsUntilSpecial(regs_since_special);
        if (regs_since_special != std::numeric_limits<u16>::max()) {
            regs_since_special++;
        }
    }

    return table;
}

static constexpr std::array<RegImplInfo, RegsInternal::NUM_REGS> reg_impl_flags_lut =
    BuildRegImplFlagsLUT();

// Expand a 4-bit mask to 4-byte mask, e.g. 0b0101 -> 0x00FF00FF
static constexpr std::array<u32, 16> ExpandBitsToBytes = {
    0x00000000, 0x000000ff, 0x0000ff00, 0x0000ffff, 0x00ff0000, 0x00ff00ff, 0x00ffff00, 0x00ffffff,
    0xff000000, 0xff0000ff, 0xff00ff00, 0xff00ffff, 0xffff0000, 0xffff00ff, 0xffffff00, 0xffffffff,
};

/**
 * This is the main loop for processing GPU command lists. On Azahar, it's the most
 * CPU expensive function (excluding the inner Draw calls) due to applications submitting
 * 10-50 command lists per frame, each with hundreds of commands in them. For this reason,
 * it is important that this function is well optimized to reduce the load on the CPU.
 *
 * Each command in the list has the following properties:
 *  - Commands come in [value (32 bit), header (32 bit)] pairs, most of the time.
 *  - The register ID that the 32 bit value should be written to is stored in the header.
 *  - The mask of bits that should be written comes in the header
 *    (to be able to write individual bytes of the 4-byte register)
 *  - Commands can have an extra length N, which means that N extra words follow
 *    after the header word.
 *    - If group_command is set in the header, N sequential registers are
 *      written to starting from the ID + 1 specified in the header.
 *    - If group_command is not set, the same register is written
 *      to with the N extra words. This is used for things like shader uploads
 *      which has a single register ID.
 *
 * Regarding implementation details, we store all register values in an array,
 * as well as a dirty array to indicate which registers have changed since
 * the last draw. Some registers need special handling, as they are "trigger"
 * registers that start the draw, or store the data in the shader units.
 *
 * To be able to determine if a register is special, we use a lookup table
 * generated by BuildRegImplFlagsLUT(). This allows determining if a register
 * is special or not in O(1). This LUT also determines if a register has
 * support for batch handling (extra_data_length != 0 && group_command == 0)
 * and the amount of commands away from the next special register, useful for
 * sequential writes (extra_data_length != 0 && group_command == 1).
 *
 * As much as possible, we want to target the following optimizations:
 *  - We should prevent branches and jumps to functions if they are not needed.
 *  - We should clearly separate special command handling from normal commands
 *    that are much cheaper to handle.
 *  - Commands with extra length should be processed in batch if possible.
 *  - Vectorization should be used as much as possible.
 *
 * On the other hand, if PICA debugging is enabled we should avoid optimizations
 * that would make debugging more complicated.
 */
void PicaCore::ProcessCmdList(PAddr list, u32 size, bool ignore_list) [[hot]] {
    if (ignore_list) {
        signal_interrupt(Service::GSP::InterruptId::P3D, delay_generator.CalculateAndResetDelay());
        return;
    }

    const u8* head = memory.GetPhysicalPointer(list);
    cmd_list.Reset(list, head, size);

    bool stop_requested = false;
    bool skip_fast_path = false;
    while (cmd_list.current_index < cmd_list.length) {
        if (stop_requested) [[unlikely]] {
            break;
        }
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        // Early path that processes commands in batches of 4. If any of the commands
        // needs special handling or has extra length it stops and falls back to the
        // slower path. This pattern allows the compiler to auto-vectorize the function
        // if the current ISA allows it (that's why we process in batches of 4
        // as most SIMD operations work with 128 bit registers). MSVC is not able to
        // auto-vectorize this part with SSE4.2, due to the LUT read, instead it just
        // unrolls the loop. Other ISAs and/or compilers may be able to do it,
        // that's why it was decided to keep the structure like this.
        if (!debug_context) [[likely]] {
            if (!skip_fast_path) {
                constexpr u32 batch_size = 4;
                u32 index = cmd_list.current_index;
                u32 ids[batch_size], values[batch_size], masks[batch_size];
                u32 run = 0;

                while (run < batch_size && index + 1 < cmd_list.length) {
                    const u32 value = cmd_list.head[index];
                    const CommandHeader header{cmd_list.head[index + 1]};

                    // If extra handling is needed stop and fallback to slower path.
                    if (header.extra_data_length != 0 || header.cmd_id >= RegsInternal::NUM_REGS ||
                        reg_impl_flags_lut[header.cmd_id].NeedsSpecialHandling()) {
                        skip_fast_path = true;
                        break;
                    }

                    ids[run] = header.cmd_id;
                    values[run] = value;
                    masks[run] = header.parameter_mask;
                    ++run;
                    index += 2;
                }

                // Process the commands that we have read so far (up to 4).
                if (run > 0) {
                    delay_generator.AddCommands(run);
                    for (u32 i = 0; i < run; ++i) {
                        const u32 id = ids[i];
                        const u32 write_mask = ExpandBitsToBytes[masks[i]];
                        regs.internal.reg_array[id] =
                            (regs.internal.reg_array[id] & ~write_mask) | (values[i] & write_mask);
                        dirty_regs.Set(id);
                    }
                    cmd_list.current_index = index;

                    // Continue from the while loop in case we reached the end of the list.
                    continue;
                }
            }
        }
        // Slow path, command needs special handling.

        skip_fast_path = false;

        // Read the header and the value to write.
        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        // Write to the requested PICA register.
        WriteInternalReg(header.cmd_id, value, header.parameter_mask, stop_requested);

        // Write any extra paramters as well.
        const u32 count = header.extra_data_length;
        if (count == 0)
            continue;

        if (debug_context) [[unlikely]] {
            // Fallback to per word register writes if debugging is
            // enabled.
            for (u32 i = 0; i < count; ++i) {
                if (stop_requested) [[unlikely]] {
                    break;
                }
                const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
                const u32 extra_value = cmd_list.head[cmd_list.current_index++];
                WriteInternalReg(cmd, extra_value, header.parameter_mask, stop_requested);
            }
        } else {
            // Handle commands with extra length.
            const u32* extra = &cmd_list.head[cmd_list.current_index];
            cmd_list.current_index += count;

            if (!header.group_commands) {
                // Same register written count times in a row (program/swizzle upload, LUT, etc.).
                WriteInternalRegBatch(header.cmd_id, extra, count, header.parameter_mask,
                                      stop_requested);
            } else {
                // Sequential registers written from header.cmd_id+1 to header.cmd_id+count.
                WriteInternalRegSequential(header.cmd_id + 1, extra, count, header.parameter_mask,
                                           stop_requested);
            }
        }
    }
}

static bool any_byte_match(u32 a, u32 b) {
    return ((a & 0xFF) == (b & 0xFF)) || (((a >> 8) & 0xFF) == ((b >> 8) & 0xFF)) ||
           (((a >> 16) & 0xFF) == ((b >> 16) & 0xFF)) || (((a >> 24) & 0xFF) == ((b >> 24) & 0xFF));
}

// Handle registers which our backend support batch writes.
void PicaCore::HandleSpecialRegBatch(u32 id, const u32* values, u32 count) {
    switch (id) {
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[1]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[2]): {
        for (u32 i = 0; i < count; i++) {
            SubmitImmediate(values[i]);
        }
        break;
    }
    case PICA_REG_INDEX(gs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[7]): {
        gs_setup.WriteUniformFloatRegRange(regs.internal.gs, values, count);
        break;
    }
    case PICA_REG_INDEX(gs.program.set_word[0]):
    case PICA_REG_INDEX(gs.program.set_word[1]):
    case PICA_REG_INDEX(gs.program.set_word[2]):
    case PICA_REG_INDEX(gs.program.set_word[3]):
    case PICA_REG_INDEX(gs.program.set_word[4]):
    case PICA_REG_INDEX(gs.program.set_word[5]):
    case PICA_REG_INDEX(gs.program.set_word[6]):
    case PICA_REG_INDEX(gs.program.set_word[7]): {
        u32& offset = regs.internal.gs.program.offset;
        if (offset + count > 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {} count {}", offset, count);
        } else {
            gs_setup.UpdateProgramCodeRange(offset, values, count);
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.gs.swizzle_patterns.offset;
        if (offset + count > gs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {} count {}", offset, count);
        } else {
            gs_setup.UpdateSwizzleDataRange(offset, values, count);
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(vs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[7]): {
        const auto range = vs_setup.WriteUniformFloatRegRange(regs.internal.vs, values, count);
        if (range && !regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            for (u32 i = 0; i < range->count; ++i) {
                const u32 idx = range->first_index + i;
                gs_setup.uniforms.f[idx] = vs_setup.uniforms.f[idx];
            }
        }
        break;
    }
    case PICA_REG_INDEX(vs.program.set_word[0]):
    case PICA_REG_INDEX(vs.program.set_word[1]):
    case PICA_REG_INDEX(vs.program.set_word[2]):
    case PICA_REG_INDEX(vs.program.set_word[3]):
    case PICA_REG_INDEX(vs.program.set_word[4]):
    case PICA_REG_INDEX(vs.program.set_word[5]):
    case PICA_REG_INDEX(vs.program.set_word[6]):
    case PICA_REG_INDEX(vs.program.set_word[7]): {
        u32& offset = regs.internal.vs.program.offset;
        if (offset + count > 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {} count {}", offset, count);
        } else {
            vs_setup.UpdateProgramCodeRange(offset, values, count);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateProgramCodeRange(offset, values, count);
            }
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.vs.swizzle_patterns.offset;
        if (offset + count > vs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {} count {}", offset, count);
        } else {
            vs_setup.UpdateSwizzleDataRange(offset, values, count);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateSwizzleDataRange(offset, values, count);
            }
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        auto& lut_config = regs.internal.lighting.lut_config;

        for (u32 i = 0; i < count; i++) {
            const u32 prev =
                std::exchange(lighting
                                  .luts[lut_config.type][(lut_config.index + i) %
                                                         lighting.luts[lut_config.type].size()]
                                  .raw,
                              values[i]);
            lighting.lut_dirty |= (prev != values[i]) << lut_config.type;
        }
        lut_config.index.Assign(lut_config.index + count);
        break;
    }
    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]): {
        for (u32 i = 0; i < count; i++) {
            const u32 prev = std::exchange(
                fog.lut[(regs.internal.texturing.fog_lut_offset + i) % 128].raw, values[i]);
            fog.lut_dirty |= prev != values[i];
        }
        regs.internal.texturing.fog_lut_offset.Assign(regs.internal.texturing.fog_lut_offset +
                                                      count);
        break;
    }
    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]): {
        auto& index = regs.internal.texturing.proctex_lut_config.index;
        const auto lut_table = regs.internal.texturing.proctex_lut_config.ref_table.Value();

        for (u32 i = 0; i < count; i++) {

            const auto sync_lut = [&](auto& proctex_table) {
                const u32 prev =
                    std::exchange(proctex_table[(index + i) % proctex_table.size()].raw, values[i]);
                proctex.table_dirty |= (prev != values[i]) << u32(lut_table);
            };

            switch (lut_table) {
            case TexturingRegs::ProcTexLutTable::Noise:
                sync_lut(proctex.noise_table);
                break;
            case TexturingRegs::ProcTexLutTable::ColorMap:
                sync_lut(proctex.color_map_table);
                break;
            case TexturingRegs::ProcTexLutTable::AlphaMap:
                sync_lut(proctex.alpha_map_table);
                break;
            case TexturingRegs::ProcTexLutTable::Color:
                sync_lut(proctex.color_table);
                break;
            case TexturingRegs::ProcTexLutTable::ColorDiff:
                sync_lut(proctex.color_diff_table);
                break;
            }
        }
        index.Assign(index + count);
        break;
    }
    }
}

// Handle special registers. This function should also include the
// registers that support batch processing can be submitted
// individually.
void PicaCore::HandleSpecialReg(u32 id, u32 value, bool& stop_requested) {
    switch (id) {
    // Trigger IRQ
    case PICA_REG_INDEX(irq_request):
        // TODO(PabloMK7): This logic is not fully accurate, but close enough:
        // https://problemkaputt.de/gbatek-3ds-gpu-internal-registers-finalize-interrupt-registers.htm
        if (any_byte_match(regs.internal.reg_array[id], regs.internal.irq_compare)) [[likely]] {
            signal_interrupt(Service::GSP::InterruptId::P3D,
                             delay_generator.CalculateAndResetDelay());
            if (regs.internal.irq_autostop) [[likely]] {
                stop_requested = true;
            }
        }
        break;

    case PICA_REG_INDEX(pipeline.triangle_topology):
        primitive_assembler.Reconfigure(regs.internal.pipeline.triangle_topology);
        break;

    case PICA_REG_INDEX(pipeline.restart_primitive):
        primitive_assembler.Reset();
        break;

    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index):
        immediate.Reset();
        break;

    // Load default vertex input attributes
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[1]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[2]):
        SubmitImmediate(value);
        break;

    case PICA_REG_INDEX(pipeline.gpu_mode):
        // This register likely just enables vertex processing and doesn't need any special handling
        break;

    case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
    case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
        const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
        const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
        const u8* head = memory.GetPhysicalPointer(addr);
        cmd_list.Reset(addr, head, size);
        break;
    }

    // It seems like these trigger vertex rendering
    case PICA_REG_INDEX(pipeline.trigger_draw):
    case PICA_REG_INDEX(pipeline.trigger_draw_indexed): {
        const bool is_indexed = (id == PICA_REG_INDEX(pipeline.trigger_draw_indexed));
        DrawArrays(is_indexed);
        break;
    }

    case PICA_REG_INDEX(gs.bool_uniforms):
        gs_setup.WriteUniformBoolReg(regs.internal.gs.bool_uniforms.Value());
        break;

    case PICA_REG_INDEX(gs.int_uniforms[0]):
    case PICA_REG_INDEX(gs.int_uniforms[1]):
    case PICA_REG_INDEX(gs.int_uniforms[2]):
    case PICA_REG_INDEX(gs.int_uniforms[3]): {
        const u32 index = (id - PICA_REG_INDEX(gs.int_uniforms[0]));
        gs_setup.WriteUniformIntReg(index, regs.internal.gs.GetIntUniform(index));
        break;
    }

    case PICA_REG_INDEX(gs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[7]): {
        gs_setup.WriteUniformFloatReg(regs.internal.gs, value);
        break;
    }

    case PICA_REG_INDEX(gs.program.set_word[0]):
    case PICA_REG_INDEX(gs.program.set_word[1]):
    case PICA_REG_INDEX(gs.program.set_word[2]):
    case PICA_REG_INDEX(gs.program.set_word[3]):
    case PICA_REG_INDEX(gs.program.set_word[4]):
    case PICA_REG_INDEX(gs.program.set_word[5]):
    case PICA_REG_INDEX(gs.program.set_word[6]):
    case PICA_REG_INDEX(gs.program.set_word[7]): {
        u32& offset = regs.internal.gs.program.offset;
        if (offset >= 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {}", offset);
        } else {
            gs_setup.UpdateProgramCode(offset, value);
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.gs.swizzle_patterns.offset;
        if (offset >= gs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {}", offset);
        } else {
            gs_setup.UpdateSwizzleData(offset, value);
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.output_mask):
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            regs.internal.gs.output_mask.Assign(value);
        }
        break;

    case PICA_REG_INDEX(vs.bool_uniforms):
        vs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        }
        break;

    case PICA_REG_INDEX(vs.int_uniforms[0]):
    case PICA_REG_INDEX(vs.int_uniforms[1]):
    case PICA_REG_INDEX(vs.int_uniforms[2]):
    case PICA_REG_INDEX(vs.int_uniforms[3]): {
        const u32 index = (id - PICA_REG_INDEX(vs.int_uniforms[0]));
        vs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        }
        break;
    }

    case PICA_REG_INDEX(vs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[7]): {
        const auto index = vs_setup.WriteUniformFloatReg(regs.internal.vs, value);
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No && index) {
            gs_setup.uniforms.f[index.value()] = vs_setup.uniforms.f[index.value()];
        }
        break;
    }

    case PICA_REG_INDEX(vs.program.set_word[0]):
    case PICA_REG_INDEX(vs.program.set_word[1]):
    case PICA_REG_INDEX(vs.program.set_word[2]):
    case PICA_REG_INDEX(vs.program.set_word[3]):
    case PICA_REG_INDEX(vs.program.set_word[4]):
    case PICA_REG_INDEX(vs.program.set_word[5]):
    case PICA_REG_INDEX(vs.program.set_word[6]):
    case PICA_REG_INDEX(vs.program.set_word[7]): {
        u32& offset = regs.internal.vs.program.offset;
        if (offset >= 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {}", offset);
        } else {
            vs_setup.UpdateProgramCode(offset, value);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateProgramCode(offset, value);
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.vs.swizzle_patterns.offset;
        if (offset >= vs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {}", offset);
        } else {
            vs_setup.UpdateSwizzleData(offset, value);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateSwizzleData(offset, value);
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        auto& lut_config = regs.internal.lighting.lut_config;

        const u32 prev = std::exchange(lighting.luts[lut_config.type][lut_config.index].raw, value);
        lighting.lut_dirty |= (prev != value) << lut_config.type;
        lut_config.index.Assign(lut_config.index + 1);
        break;
    }

    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]): {
        const u32 prev =
            std::exchange(fog.lut[regs.internal.texturing.fog_lut_offset % 128].raw, value);
        fog.lut_dirty |= prev != value;
        regs.internal.texturing.fog_lut_offset.Assign(regs.internal.texturing.fog_lut_offset + 1);
        break;
    }

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]): {
        auto& index = regs.internal.texturing.proctex_lut_config.index;
        const auto lut_table = regs.internal.texturing.proctex_lut_config.ref_table.Value();

        const auto sync_lut = [&](auto& proctex_table) {
            const u32 prev = std::exchange(proctex_table[index % proctex_table.size()].raw, value);
            proctex.table_dirty |= (prev != value) << u32(lut_table);
        };

        switch (lut_table) {
        case TexturingRegs::ProcTexLutTable::Noise:
            sync_lut(proctex.noise_table);
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            sync_lut(proctex.color_map_table);
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            sync_lut(proctex.alpha_map_table);
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            sync_lut(proctex.color_table);
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            sync_lut(proctex.color_diff_table);
            break;
        }
        index.Assign(index + 1);
        break;
    }
    default:
        break;
    }
}

// Handle batch register writes
void PicaCore::WriteInternalRegBatch(u32 id, const u32* values, u32 count, u32 mask,
                                     bool& stop_requested) {
    if (id >= RegsInternal::NUM_REGS) [[unlikely]] {
        // Writes to OOB registers are no-op.
        LOG_DEBUG(HW_GPU,
                  "Commandlist tried to write to invalid register 0x{:03X} repeated 0x{:04X} times"
                  "(mask: {:X})",
                  id, count, mask);
        return;
    }

    delay_generator.AddCommands(count);
    const u32 write_mask = ExpandBitsToBytes[mask];
    // Only write the last value to the register array.
    // Batch handlers should take this in mind.
    regs.internal.reg_array[id] =
        (regs.internal.reg_array[id] & ~write_mask) | (values[count - 1] & write_mask);
    dirty_regs.Set(id);

    if (reg_impl_flags_lut[id].SupportsBatch()) {
        // If the register supports batch then call the handler.
        HandleSpecialRegBatch(id, values, count);
    } else if (reg_impl_flags_lut[id].NeedsSpecialHandling()) [[unlikely]] {
        // Unlikely as all special regs that make sense to use batch mode already
        // support batch handling.
        for (u32 i = 0; i < count && !stop_requested; ++i) {
            HandleSpecialReg(id, values[i], stop_requested);
        }
    }
}

// Handle sequential register writes.
void PicaCore::WriteInternalRegSequential(u32 id, const u32* __restrict values, u32 count, u32 mask,
                                          bool& stop_requested) {
    if (id + count > RegsInternal::NUM_REGS) [[unlikely]] {
        // Writes to OOB registers are no-op.
        LOG_DEBUG(
            HW_GPU,
            "Commandlist tried to write to invalid register range 0x{:03X}-0x{:03X} (mask: {:X})",
            id, id + count, mask);
        if (id >= RegsInternal::NUM_REGS) {
            return;
        } else {
            count = RegsInternal::NUM_REGS - id;
        }
    }
    const u32 write_mask = ExpandBitsToBytes[mask];
    u32* __restrict dst = &regs.internal.reg_array[id];
    u32 offset = 0;

    // This code is structured so that it uses the LUT to get the distance from the
    // register ID to the next special register. Then copies the range of normal registers
    // until it reaches the special register which is handled individually, and so on.
    while (offset < count) {
        if (stop_requested) [[unlikely]] {
            break;
        }
        const u32 reg = id + offset;
        // Distance to the next special register, capped by how many writes
        // are actually left in this write. If this register is special this is 0.
        const u32 batch_count =
            std::min<u32>(reg_impl_flags_lut[reg].RegsUntilSpecial(), count - offset);
        if (batch_count > 0) {
            delay_generator.AddCommands(batch_count);

            // Allows the compiler to auto-vectorize thanks to the __restrict keywords.
            // Verified in MSVC that vectorization is happening.
            u32* __restrict batch_dst = dst + offset;
            const u32* __restrict batch_src = values + offset;
            for (u32 i = 0; i < batch_count; ++i) {
                batch_dst[i] = (batch_dst[i] & ~write_mask) | (batch_src[i] & write_mask);
            }

            dirty_regs.SetRange(reg, batch_count);
            offset += batch_count;
        }
        // Whatever register stopped the batch (if we didn't reach the end)
        // needs individual handling, then resume batching after it.
        if (offset < count) {
            WriteInternalReg(id + offset, values[offset], mask, stop_requested);
            ++offset;
        }
    }
}

// Handle individual command write.
void PicaCore::WriteInternalReg(u32 id, u32 value, u32 mask, bool& stop_requested) {
    if (id >= RegsInternal::NUM_REGS) [[unlikely]] {
        // Writes to OOB registers are no-op.
        LOG_DEBUG(
            HW_GPU,
            "Commandlist tried to write to invalid register 0x{:03X} (value: {:08X}, mask: {:X})",
            id, value, mask);
        return;
    }

    delay_generator.AddCommands(1);

    // TODO: Figure out how register masking acts on e.g. vs.uniform_setup.set_value
    const u32 old_value = regs.internal.reg_array[id];
    const u32 write_mask = ExpandBitsToBytes[mask];
    regs.internal.reg_array[id] = (old_value & ~write_mask) | (value & write_mask);

    if (debug_context) [[unlikely]] {
        // Track register write.
        DebugUtils::OnPicaRegWrite(id, mask, regs.internal.reg_array[id]);
        // Track events.
        debug_context->OnEvent(DebugContext::Event::PicaCommandLoaded, &id);
    }

    if (reg_impl_flags_lut[id].NeedsSpecialHandling()) {
        HandleSpecialReg(id, value, stop_requested);
    }

    dirty_regs.Set(id);

    if (debug_context) [[unlikely]] {
        debug_context->OnEvent(DebugContext::Event::PicaCommandProcessed, &id);
    }
}

void PicaCore::SubmitImmediate(u32 value) {
    // Push to word to the queue. This returns true when a full attribute is formed.
    if (!immediate.queue.Push(value)) {
        return;
    }

    constexpr std::size_t IMMEDIATE_MODE_INDEX = 0xF;

    auto& setup = regs.internal.pipeline.vs_default_attributes_setup;
    if (setup.index > IMMEDIATE_MODE_INDEX) {
        LOG_ERROR(HW_GPU, "Invalid VS default attribute index {}", setup.index);
        return;
    }

    // Retrieve the attribute and place it in the default attribute buffer.
    const auto attribute = immediate.queue.Get();
    if (setup.index < IMMEDIATE_MODE_INDEX) {
        input_default_attributes[setup.index] = attribute;
        setup.index++;
        return;
    }

    // When index is 0xF the attribute is used for immediate mode drawing.
    immediate.input_vertex[immediate.current_attribute] = attribute;
    if (immediate.current_attribute < regs.internal.pipeline.max_input_attrib_index) {
        immediate.current_attribute++;
        return;
    }

    // We formed a vertex, flush.
    DrawImmediate();
}

void PicaCore::DrawImmediate() {
    // Compile the vertex shader.
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                               std::addressof(immediate.input_vertex));
    }

    ShaderUnit shader_unit;
    AttributeBuffer output{};

    // Invoke the vertex shader for the vertex.
    shader_unit.LoadInput(regs.internal.vs, immediate.input_vertex);
    shader_engine->Run(vs_setup, shader_unit);
    shader_unit.WriteOutput(regs.internal.vs, output);

    // Reconfigure geometry pipeline if needed.
    if (immediate.reset_geometry_pipeline) {
        geometry_pipeline.Reconfigure();
        immediate.reset_geometry_pipeline = false;
    }

    // Send to geometry pipeline.
    ASSERT(!geometry_pipeline.NeedIndexInput());
    geometry_pipeline.Setup(shader_engine.get());
    geometry_pipeline.SubmitVertex(output);

    // Flush the immediate triangle.
    rasterizer->DrawTriangles();
    immediate.current_attribute = 0;

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr);
    }
}

void PicaCore::DrawArrays(bool is_indexed) {
    MICROPROFILE_SCOPE(GPU_Drawing);

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::IncomingPrimitiveBatch, nullptr);
    }

    const bool accelerate_draw = [this] {
        // Geometry shaders cannot be accelerated due to register preservation.
        if (regs.internal.pipeline.use_gs == PipelineRegs::UseGS::Yes) {
            return false;
        }

        // TODO (wwylele): for Strip/Fan topology, if the primitive assember is not restarted
        // after this draw call, the buffered vertex from this draw should "leak" to the next
        // draw, in which case we should buffer the vertex into the software primitive assember,
        // or disable accelerate draw completely. However, there is not game found yet that does
        // this, so this is left unimplemented for now. Revisit this when an issue is found in
        // games.

        bool accelerate_draw = Settings::values.use_hw_shader && primitive_assembler.IsEmpty();
        const auto topology = primitive_assembler.GetTopology();
        if (topology == PipelineRegs::TriangleTopology::Shader ||
            topology == PipelineRegs::TriangleTopology::List) {
            accelerate_draw = accelerate_draw && (regs.internal.pipeline.num_vertices % 3) == 0;
        }
        return accelerate_draw;
    }();

    // Add vertices to the delay generator.
    delay_generator.AddVertices(regs.internal.pipeline.num_vertices,
                                regs.internal.pipeline.triangle_topology);

    // Attempt to use hardware vertex shaders if possible.
    if (accelerate_draw && rasterizer->AccelerateDrawBatch(is_indexed)) {
        return;
    }

    // We cannot accelerate the draw, so load and execute the vertex shader for each vertex.
    LoadVertices(is_indexed);

    // Draw emitted triangles.
    rasterizer->DrawTriangles();

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr);
    }
}

void PicaCore::LoadVertices(bool is_indexed) {
    // Read and validate vertex information from the loaders
    const auto& pipeline = regs.internal.pipeline;
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto loader = VertexLoader(memory, pipeline);
    regs.internal.rasterizer.ValidateSemantics();

    // Locate index buffer.
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GetPhysicalPointer(base_address + index_info.offset);
    if (index_address_8 == nullptr) {
        // Mario & Luigi: Superstar Saga sets an invalid base address
        // for the vertex attributes. Return early if that is the case.
        return;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    // Simple circular-replacement vertex cache
    //
    // POR QUE LOS IDENTIFICADORES SON u32 Y NO u16 CON UN ARRAY DE bool APARTE.
    //
    // La validez vivia en su propio array y la busqueda era
    // "if (valido[i] && vertice == ids[i])" sobre las 64 entradas, una vez POR
    // CADA VERTICE INDEXADO del lote. Consultar dos arrays distintos en la misma
    // condicion impide que GCC vectorice el bucle, asi que quedaba un barrido
    // escalar de hasta 64 vueltas con dos cargas y dos saltos en cada una.
    //
    // Con identificadores de 32 bits y un centinela, el centinela HACE DE BIT DE
    // VALIDEZ: 'vertice' sale de un indice de 8 o 16 bits, o sea que nunca puede
    // valer 0xFFFFFFFF. La busqueda queda como un unico barrido de u32 sin
    // dependencias entre vueltas, que el NEON del Cortex-A9 recorre de cuatro en
    // cuatro.
    //
    // La SEMANTICA no cambia: sigue siendo totalmente asociativa con reemplazo
    // circular, asi que acierta y falla en exactamente los mismos vertices que
    // antes. Esto solo abarata el mirar, no cambia lo que se encuentra -- que es
    // justo lo que hace falta para poder cambiarlo sin poder probar el dibujado.
    constexpr u32 VERTEX_CACHE_SIZE = 64;
    constexpr u32 VERTEX_CACHE_EMPTY = 0xFFFFFFFFu;
    static_assert((VERTEX_CACHE_SIZE & (VERTEX_CACHE_SIZE - 1)) == 0,
                  "El avance del reemplazo circular usa una mascara, no un modulo");

    std::array<u32, VERTEX_CACHE_SIZE> vertex_cache_ids;
    vertex_cache_ids.fill(VERTEX_CACHE_EMPTY);
    std::array<AttributeBuffer, VERTEX_CACHE_SIZE> vertex_cache;
    u32 vertex_cache_pos = 0;

    /**
     * Pista de hueco por indice de vertice: acelera la busqueda sin cambiarla.
     *
     * 256 entradas de un byte (el cache tiene 64 huecos, asi que cabe de
     * sobra). Se rellena con VERTEX_CACHE_SIZE, que es "sin pista" y nunca es
     * un hueco valido. Ver el uso en el bucle de abajo: es un acelerador, no
     * una fuente de verdad -- lo que decide si hay acierto sigue siendo
     * vertex_cache_ids.
     */
    constexpr u32 VERTEX_CACHE_HINTS = 256;
    static_assert(VERTEX_CACHE_SIZE <= 255, "la pista se guarda en un byte");
    std::array<u8, VERTEX_CACHE_HINTS> vertex_cache_hint;
    vertex_cache_hint.fill(static_cast<u8>(VERTEX_CACHE_SIZE));

    // Compile the vertex shader for this batch.
    ShaderUnit shader_unit;
    AttributeBuffer vs_output;
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    // Setup geometry pipeline in case we are using a geometry shader.
    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());
    ASSERT(!geometry_pipeline.NeedIndexInput() || is_indexed);

#ifdef __PSVITA__
    // El bucle entero. Ver Common::FrameStats::vertices_us: hace falta para
    // separar dentro de "cmdlist" lo que es shader de vertices de lo que es
    // nuestro camino de GPU. Una llamada de dibujado de cada ocho, por ocho
    // (0.2.1.0): en 3D son cientos por fotograma y leer el reloj es una llamada
    // al kernel.
    static u32 vertices_tick = 0;
    const bool vertices_timed = (++vertices_tick & 7u) == 0;
    const unsigned long long vertices_begin = vertices_timed ? Common::VitaMicros() : 0;
    const auto add_vertices_time = [vertices_timed, vertices_begin] {
        if (vertices_timed) {
            Common::FrameStats::vertices_us.fetch_add((Common::VitaMicros() - vertices_begin) * 8,
                                                      std::memory_order_relaxed);
        }
    };

    /**
     * Sin shader de geometria y sin depurador, el sombreado va repartido entre
     * los tres nucleos (ver ShadeVerticesParallel). Con shader de geometria se
     * queda el bucle de siguiente, que es el de siempre: ahi cada vertice
     * alimenta a un segundo programa con estado propio y el orden importa
     * dentro del propio sombreado. Con depurador tambien, porque este le avisa
     * vertice a vertice y en orden.
     */
    if (!debug_context && pipeline.use_gs == PipelineRegs::UseGS::No) {
        ShadeVerticesParallel(loader, base_address, is_indexed, index_address_8, index_address_16,
                              index_u16);
        add_vertices_time();
        return;
    }
#endif
    for (u32 index = 0; index < pipeline.num_vertices; ++index) {
        // Indexed rendering doesn't use the start offset
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);

        /**
         * El resultado del vertice se ENTREGA POR REFERENCIA, no copiandolo.
         *
         * Antes esto movia un AttributeBuffer entero -- 16 vectores de cuatro
         * f24, 256 bytes -- en los dos caminos: del cache a vs_output al
         * acertar, y de vs_output al cache al fallar. Con unos 15.000 vertices
         * por fotograma son cerca de 4 MB de copias que no hacian falta, porque
         * lo unico que se hace despues con el resultado es pasarlo a
         * SubmitVertex, que lo toma por referencia constante.
         *
         * Ahora al acertar se apunta a la entrada del cache y al fallar el
         * shader escribe DIRECTAMENTE en el hueco que le toca. El resultado es
         * el mismo byte a byte; solo se ahorra el ir y venir.
         */
        const AttributeBuffer* submit = nullptr;

        if (is_indexed) {
            if (geometry_pipeline.NeedIndexInput()) {
                geometry_pipeline.SubmitIndex(vertex);
                continue;
            }

            /**
             * Y la busqueda empieza por una PISTA en vez de por el barrido.
             *
             * El cache es completamente asociativo, asi que localizar un
             * vertice costaba recorrer las 64 entradas, y en los fallos las 64
             * enteras. Son del orden de un millon de comparaciones por
             * fotograma solo para buscar.
             *
             * hint mapea los ocho bits bajos del indice al hueco donde se
             * guardo por ultima vez. Es SOLO una pista: lo que decide sigue
             * siendo la comparacion contra vertex_cache_ids, asi que si la
             * pista esta obsoleta se cae al barrido de siempre y el resultado
             * -- y la tasa de acierto -- no cambian. Lo que cambia es que la
             * mayoria de los aciertos pasan a costar una comparacion.
             */
            const u32 hinted = vertex_cache_hint[vertex & (VERTEX_CACHE_HINTS - 1)];
            if (hinted < VERTEX_CACHE_SIZE && vertex_cache_ids[hinted] == vertex) {
                submit = std::addressof(vertex_cache[hinted]);
            } else {
                for (u32 i = 0; i < VERTEX_CACHE_SIZE; ++i) {
                    if (vertex_cache_ids[i] == vertex) {
                        submit = std::addressof(vertex_cache[i]);
                        break;
                    }
                }
            }
        }

        if (submit == nullptr) {
            // Initialize data for the current vertex
            AttributeBuffer input;
            loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);

            // Record vertex processing to the debugger.
            if (debug_context) {
                debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                                       std::addressof(input));
            }

            // Invoke the vertex shader for this vertex.
            shader_unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, shader_unit);
#ifdef __PSVITA__
            // Solo se cuentan los que de verdad pasan por el interprete: los
            // que acierta el cache no llegan aqui.
            Common::FrameStats::vertices_shaded.fetch_add(1, std::memory_order_relaxed);
#endif

            if (is_indexed) {
                // El shader escribe en el hueco del cache y de ahi se entrega.
                AttributeBuffer& slot = vertex_cache[vertex_cache_pos];
                shader_unit.WriteOutput(regs.internal.vs, slot);
                vertex_cache_ids[vertex_cache_pos] = vertex;
                vertex_cache_hint[vertex & (VERTEX_CACHE_HINTS - 1)] =
                    static_cast<u8>(vertex_cache_pos);
                submit = std::addressof(slot);
                vertex_cache_pos = (vertex_cache_pos + 1) & (VERTEX_CACHE_SIZE - 1);
            } else {
                shader_unit.WriteOutput(regs.internal.vs, vs_output);
                submit = std::addressof(vs_output);
            }
        }

        // Send to geometry pipeline
        geometry_pipeline.SubmitVertex(*submit);
    }
#ifdef __PSVITA__
    add_vertices_time();
#endif
}

#ifdef __PSVITA__
/**
 * SOMBREADO DE VERTICES EN LOS TRES NUCLEOS.
 *
 * POR QUE. En 0.1.0.41 (cinematica de Rubi Omega, 444 MHz) el fotograma medio
 * 706 ms y el shader de vertices se comia unos 370: ~19.000 vertices a unos
 * 20 us cada uno en el interprete, todos en el nucleo 0. Mientras tanto los
 * nucleos 1 y 2 -- los del rasterizador de software -- estaban parados, porque
 * desde que la iluminacion va en la GPU casi no les llega trabajo.
 *
 * Sombrear un vertice no depende de los demas: mismo programa, mismos uniforms,
 * su propia entrada. Asi que el lote se hace en tres fases:
 *
 *   1. ORDEN (serie, barato). Se recorre el lote igual que el bucle de siempre,
 *      con el MISMO cache de 64 vertices y el mismo reemplazo circular, pero
 *      sin sombrear: solo se apunta que vertices hay que sombrear ("trabajos")
 *      y, para cada posicion del lote, de que trabajo sale su resultado. Aciertos
 *      y fallos salen identicos a los del bucle serie, porque es la misma
 *      busqueda sobre el mismo estado.
 *   2. SOMBREADO (en paralelo). Los trabajos se parten en tres tramos
 *      contiguos: el nucleo 0 hace el primero y los nucleos 1 y 2 los otros.
 *   3. ENTREGA (serie, en orden). Cada posicion entrega su resultado al
 *      ensamblador de primitivas en el orden original, que es lo unico que ve
 *      el resto del pipeline.
 *
 * LO QUE CAMBIA, DICHO SIN RODEOS. El bucle serie usa UNA ShaderUnit para todo
 * el lote, asi que un registro que el programa lea sin haberlo escrito antes
 * -- un temporal, un registro de direccion, el codigo de condicion o una salida
 * que solo se escribe en una rama -- trae el valor del vertice ANTERIOR. Aqui
 * cada tramo tiene su propia unidad, y para no perder eso cada tramo que no
 * empieza el lote SOMBREA ANTES EL VERTICE QUE LE PRECEDE y tira el resultado
 * ("calentamiento"): asi entra en su primer vertice con el mismo estado que
 * tendria en serie siempre que ese estado dependa solo del vertice anterior,
 * que es el caso de cualquier programa que escriba un registro en todos los
 * vertices o en ninguno. Solo divergiria un programa que lea un registro que
 * se escribe en unos vertices si y en otros no, y ese programa ya es
 * impredecible en la consola de verdad: la PICA200 reparte los vertices entre
 * cuatro unidades de shader, cada una con su propio estado, y ademas es lo
 * que hace Azahar en escritorio con shaders por hardware (cada vertice parte
 * de cero). El primer tramo del lote no se calienta: arranca de una unidad
 * nueva, exactamente como el bucle serie.
 *
 * El modo de coma flotante es el mismo en los tres nucleos (FZ y DN, ver
 * Common::VitaEnableFastFloatMode, que el pool activa en cada hilo), asi que
 * cada vertice da el mismo resultado bit a bit se sombree donde se sombree.
 *
 * MEMORIA. Los resultados se guardan en un vector reutilizable. Para acotarlo,
 * un lote enorme se procesa en segmentos de kMaxJobs trabajos; entre segmento
 * y segmento, lo que siga vivo en el cache se copia a slot_data, porque los
 * resultados del segmento anterior se van a pisar.
 */
void PicaCore::ShadeVerticesParallel(const VertexLoader& loader, PAddr base_address,
                                     bool is_indexed, const u8* index_address_8,
                                     const u16* index_address_16, bool index_u16) {
    // Mismo tamano y misma politica que el cache del bucle serie de arriba.
    constexpr u32 kCacheSize = 64;
    constexpr u32 kEmpty = 0xFFFFFFFFu;
    constexpr u32 kFromSlot = 0x80000000u;
    /// Tope de trabajos por segmento: 1024 resultados de 256 bytes.
    constexpr u32 kMaxJobs = 1024;
    /**
     * Por debajo de esto no se reparte. Encolar y despertar un hilo cuesta del
     * orden de decenas de microsegundos, y un vertice unos 20: con menos de
     * ~24 vertices por tramo el reparto cuesta mas de lo que ahorra.
     */
    constexpr u32 kMinJobsPerPart = 24;

    if (!parallel_shading) {
        parallel_shading = std::make_unique<ParallelShading>();
    }
    auto& ps = *parallel_shading;
    const auto& pipeline = regs.internal.pipeline;
    const u32 num_vertices = pipeline.num_vertices;

    std::array<u32, kCacheSize> cache_ids;
    cache_ids.fill(kEmpty);
    // Que hay en cada hueco: un trabajo de este segmento o kFromSlot | hueco.
    std::array<u32, kCacheSize> cache_ref{};
    u32 cache_pos = 0;

    // El ultimo trabajo del segmento anterior, para calentar el primer tramo
    // del siguiente. Ver la cabecera: el lote en si arranca sin calentar.
    bool has_previous = false;
    ParallelShading::Job previous{};

    // Sombrea [begin, end) con una unidad propia. 'warmup' es el vertice que
    // precede al tramo, o null si el tramo abre el lote.
    const auto shade_range = [&](u32 begin, u32 end, const ParallelShading::Job* warmup) {
        const unsigned long long busy_begin = Common::VitaMicros();
        ShaderUnit unit;
        // El contador de muestreo de la autocomprobacion vive en la unidad, que
        // es nueva en cada tramo: empezaba en 0 y el primer vertice de CADA
        // tramo (y el calentamiento) se comprobaba con el interprete entero,
        // 2-3 por lote en vez de 1 de cada 512 (0.3.1.8).
        static std::atomic<u32> check_seed{1};
        unit.fast_check_counter =
            check_seed.fetch_add(end - begin + 1, std::memory_order_relaxed);
        AttributeBuffer input;
        AttributeBuffer result;
        if (warmup != nullptr) {
            loader.LoadVertex(base_address, warmup->index, warmup->vertex, input,
                              input_default_attributes);
            unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, unit);
            // Sin WriteOutput: el resultado ya lo produjo su tramo; aqui solo
            // interesa el estado en que deja la unidad.
        }
        u64 load_us = 0, run_us = 0, out_us = 0, samples = 0;
        for (u32 j = begin; j < end; ++j) {
            const auto& job = ps.jobs[j];
            // Uno de cada 32 se cronometra por partes (0.1.7.2, ver vtx_samples).
            const bool sample = (j & 31) == 0;
            const unsigned long long t0 = sample ? Common::VitaMicros() : 0;
            loader.LoadVertex(base_address, job.index, job.vertex, input,
                              input_default_attributes);
            unit.LoadInput(regs.internal.vs, input);
            const unsigned long long t1 = sample ? Common::VitaMicros() : 0;
            shader_engine->Run(vs_setup, unit);
            const unsigned long long t2 = sample ? Common::VitaMicros() : 0;
            unit.WriteOutput(regs.internal.vs, result);
            // La misma conversion que hace el manejador de vertices de PicaCore
            // (ver el constructor) antes de entregar al ensamblador.
            ps.outputs[j] = OutputVertex(regs.internal.rasterizer, result);
            if (sample) {
                const unsigned long long t3 = Common::VitaMicros();
                load_us += t1 - t0;
                run_us += t2 - t1;
                out_us += t3 - t2;
                samples++;
            }
        }
        if (samples != 0) {
            Common::FrameStats::vtx_samples.fetch_add(samples, std::memory_order_relaxed);
            Common::FrameStats::vtx_load_us.fetch_add(load_us, std::memory_order_relaxed);
            Common::FrameStats::vtx_run_us.fetch_add(run_us, std::memory_order_relaxed);
            Common::FrameStats::vtx_out_us.fetch_add(out_us, std::memory_order_relaxed);
        }
        // Diagnostico: instrucciones ejecutadas (por la ruta rapida y por el
        // interprete) y tiempo de trabajo de este nucleo. Ver shade_busy_us.
        Common::FrameStats::shade_fast_ops.fetch_add(unit.fast_ops, std::memory_order_relaxed);
        Common::FrameStats::shade_slow_instrs.fetch_add(unit.slow_instrs,
                                                        std::memory_order_relaxed);
        Common::FrameStats::Add(Common::FrameStats::shade_busy_us, busy_begin);
    };

    // Cache completa por lote (ver g_full_vertex_dedup): solo con indices, que
    // es donde se repiten vertices. Sin segmentos: todos los resultados del
    // lote viven a la vez (como mucho un trabajo por indice distinto).
    const bool dedup = is_indexed && g_full_vertex_dedup.load(std::memory_order_relaxed);
    if (dedup) {
        if (++ps.stamp == 0) {
            ps.stamp_of.fill(0);
            ps.stamp = 1;
        }
    }

    u32 position = 0;
    while (position < num_vertices) {
        // ---- Fase 1: orden ----
        ps.jobs.clear();
        ps.submit.clear();
        for (; position < num_vertices && (dedup || ps.jobs.size() < kMaxJobs); ++position) {
            const u32 vertex = is_indexed
                                   ? (index_u16 ? index_address_16[position]
                                                : index_address_8[position])
                                   : (position + pipeline.vertex_offset);
            if (dedup) {
                if (ps.stamp_of[vertex] == ps.stamp) {
                    ps.submit.push_back(ps.job_of[vertex]);
                    continue;
                }
                const u32 job = static_cast<u32>(ps.jobs.size());
                ps.jobs.push_back({position, vertex});
                ps.stamp_of[vertex] = ps.stamp;
                ps.job_of[vertex] = job;
                ps.submit.push_back(job);
            } else if (is_indexed) {
                /**
                 * Busqueda en O(1) con el mapa inverso vertice -> hueco.
                 *
                 * Un vertice solo entra al cache cuando no esta, asi que como
                 * mucho hay UN hueco con su id. slot_of[v] guarda el hueco donde
                 * se metio v por ultima vez; si ese hueco sigue teniendo a v, es
                 * un acierto, y si no (lo pisaron despues) v no esta en ningun
                 * otro sitio. Es la misma respuesta que el barrido de 64 del
                 * bucle serie, sin el barrido: en un fallo, que es lo que pasa
                 * con cada vertice nuevo, el barrido era siempre entero.
                 *
                 * El mapa no se limpia entre lotes a proposito: un valor viejo
                 * solo puede apuntar a un hueco que no tiene a v, y eso se lee
                 * como fallo, que es lo correcto.
                 */
                const u32 hinted = ps.slot_of[vertex];
                if (cache_ids[hinted] == vertex) {
                    ps.submit.push_back(cache_ref[hinted]);
                    continue;
                }
                const u32 job = static_cast<u32>(ps.jobs.size());
                ps.jobs.push_back({position, vertex});
                cache_ids[cache_pos] = vertex;
                cache_ref[cache_pos] = job;
                ps.slot_of[vertex] = static_cast<u8>(cache_pos);
                cache_pos = (cache_pos + 1) & (kCacheSize - 1);
                ps.submit.push_back(job);
            } else {
                const u32 job = static_cast<u32>(ps.jobs.size());
                ps.jobs.push_back({position, vertex});
                ps.submit.push_back(job);
            }
        }

        // ---- Fase 2: sombreado ----
        const unsigned long long shade_begin = Common::VitaMicros();
        const u32 jobs = static_cast<u32>(ps.jobs.size());
        if (ps.outputs.size() < jobs) {
            ps.outputs.resize(jobs);
        }
        const u32 parts = std::clamp<u32>(jobs / kMinJobsPerPart, 1, 2);
        const ParallelShading::Job* first_warmup = has_previous ? &previous : nullptr;
        if (parts == 1) {
            shade_range(0, jobs, first_warmup);
        } else {
            // El tramo 1 al ayudante, encolado ANTES de ponerse a trabajar
            // para que arranque cuanto antes.
            for (u32 part = 1; part < parts; ++part) {
                const u32 begin = jobs * part / parts;
                const u32 end = jobs * (part + 1) / parts;
                ps.workers.QueueWork([&shade_range, &ps, begin, end] {
                    shade_range(begin, end, &ps.jobs[begin - 1]);
                });
            }
            shade_range(0, jobs / parts, first_warmup);
            // Misma barrera que las bandas del rasterizador: ver la razon de
            // la espera activa en WaitForRequestsSpin.
            ps.workers.WaitForRequestsSpin(4096);
        }
        if (jobs != 0) {
            has_previous = true;
            previous = ps.jobs[jobs - 1];
        }
        Common::FrameStats::vertices_shaded.fetch_add(jobs, std::memory_order_relaxed);
        Common::FrameStats::Add(Common::FrameStats::shade_us, shade_begin);

        // ---- Fase 3: entrega, en el orden del lote ----
        //
        // Directamente al ensamblador de primitivas: es exactamente lo que hace
        // geometry_pipeline.SubmitVertex sin shader de geometria (llama al
        // manejador de vertices, que convierte y entrega; ver el constructor),
        // solo que la conversion ya esta hecha y el std::function del
        // manejador de triangulos se construye UNA vez por segmento, no una
        // por vertice.
        const PrimitiveAssembler::TriangleHandler add_triangle =
            [this](const OutputVertex& v0, const OutputVertex& v1, const OutputVertex& v2) {
                rasterizer->AddTriangle(v0, v1, v2);
            };
        for (const u32 ref : ps.submit) {
            const OutputVertex& output =
                (ref & kFromSlot) != 0 ? ps.slot_data[ref & ~kFromSlot] : ps.outputs[ref];
            primitive_assembler.SubmitVertex(output, add_triangle);
        }

        // Lo que siga en el cache tiene que sobrevivir al segmento siguiente,
        // que reutiliza 'outputs'.
        if (is_indexed && position < num_vertices) {
            for (u32 slot = 0; slot < kCacheSize; ++slot) {
                if (cache_ids[slot] != kEmpty && (cache_ref[slot] & kFromSlot) == 0) {
                    ps.slot_data[slot] = ps.outputs[cache_ref[slot]];
                    cache_ref[slot] = kFromSlot | slot;
                }
            }
        }
    }
}
#endif

PicaCore::RenderPropertiesGuess PicaCore::GuessCmdRenderProperties(PAddr list, u32 size) {
    // Initialize command list tracking.
    const u8* head = memory.GetPhysicalPointer(list);
    cmd_list.Reset(list, head, size);

    constexpr size_t max_iterations = 0x100;

    RenderPropertiesGuess find_info{};

    find_info.vp_height = regs.internal.rasterizer.viewport_size_y.Value();
    find_info.paddr = regs.internal.framebuffer.framebuffer.color_buffer_address.Value() * 8;

    auto process_write = [this, &find_info](u32 cmd_id, u32 value) {
        switch (cmd_id) {
        case PICA_REG_INDEX(rasterizer.viewport_size_y):
            find_info.vp_height = value;
            find_info.vp_heigh_found = true;
            break;
        case PICA_REG_INDEX(framebuffer.framebuffer.color_buffer_address):
            find_info.paddr = value * 8;
            find_info.paddr_found = true;
            break;
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
            const u32 index =
                static_cast<u32>(cmd_id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
            const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
            const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
            const u8* head = memory.GetPhysicalPointer(addr);
            cmd_list.Reset(addr, head, size);
            break;
        }
        default:
            break;
        }
        return find_info.vp_heigh_found && find_info.paddr_found;
    };

    size_t iterations = 0;
    while (cmd_list.current_index < cmd_list.length && iterations < max_iterations) {
        // Align read pointer to 8 bytes
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        // Read the header and the value to write.
        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        // Write to the requested PICA register.
        if (process_write(header.cmd_id, value))
            break;

        // Write any extra paramters as well.
        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];
            if (process_write(cmd, extra_value))
                break;
        }

        iterations++;
    }

    return find_info;
}

template <class Archive>
void PicaCore::CommandList::serialize(Archive& ar, const u32 file_version) {
    ar & addr;
    ar & length;
    ar & current_index;
    if (Archive::is_loading::value) {
        const u8* ptr = Core::System::GetInstance().Memory().GetPhysicalPointer(addr);
        head = reinterpret_cast<const u32*>(ptr);
    }
}

SERIALIZE_IMPL(PicaCore::CommandList)

} // namespace Pica
