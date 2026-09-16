// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#include <memory>
#include "core/arm/dyncom/arm_dyncom.h"
#include "core/arm/dyncom/arm_dyncom_interpreter.h"
#include "core/arm/dyncom/arm_dyncom_trans.h"
#include "core/arm/skyeye_common/armstate.h"
#include "core/core.h"
#include "core/core_timing.h"

namespace Core {

ARM_DynCom::ARM_DynCom(Core::System& system_, Memory::MemorySystem& memory,
                       PrivilegeMode initial_mode, u32 id,
                       std::shared_ptr<Core::Timing::Timer> timer)
    : ARM_Interface(id, timer), system(system_) {
    state = std::make_unique<ARMul_State>(system, memory, initial_mode);
    // Los nucleos comparten un unico buffer de traduccion, asi que quien lo
    // vacie tiene que poder invalidar el indice de todos. Ver
    // FlushTransCacheIfNeeded en arm_dyncom_trans.h.
    RegisterTransCacheUser(state.get());
}

ARM_DynCom::~ARM_DynCom() {
    UnregisterTransCacheUser(state.get());
}

void ARM_DynCom::Run() {
    if (break_flag) [[unlikely]] {
        return;
    }
    ExecuteInstructions(std::max<s64>(timer->GetDowncount(), 0));
}

void ARM_DynCom::Step() {
    if (break_flag) [[unlikely]] {
        return;
    }
    ExecuteInstructions(1);
}

void ARM_DynCom::ClearInstructionCache() {
    // Antes esto vaciaba el indice de ESTE nucleo y ponia el buffer compartido a
    // cero, dejando al otro nucleo con un indice que apuntaba a posiciones que
    // se iban a reutilizar: en cuanto le tocaba el turno, ejecutaba basura. Como
    // el buffer es unico, el vaciado tambien tiene que serlo.
    //
    // Se cuenta como "por invalidacion" y no "por capacidad": esto se llama en
    // cada cambio de tabla de paginas (cambio de proceso) y en cada
    // invalidacion explicita de rango, sea cual sea el tamano del buffer. Ver
    // GetTransCacheFlushCounts.
    ResetTransCacheFromInvalidation();
}

void ARM_DynCom::InvalidateCacheRange(u32, std::size_t) {
    ClearInstructionCache();
}

void ARM_DynCom::SetPageTable(const std::shared_ptr<Memory::PageTable>& page_table) {
    // Solo se tira la cache de traduccion si la tabla de paginas ha CAMBIADO de
    // verdad. Antes se tiraba siempre, y eso era devastador:
    //
    // System::RunLoop llama a KernelSystem::SetRunningCPU cuatro veces por
    // rodaja (dos nucleos x dos bucles). Cada una de esas llamadas acaba en
    // SetCurrentProcess -> SetCurrentMemoryPageTable -> aqui. Y como
    // normalmente solo hay UN proceso emulado (el juego), la tabla que llega es
    // siempre la misma... pero se vaciaba la cache igual.
    //
    // Resultado: la cache de traduccion se destruia entera unas cuatro veces
    // por rodaja, para los dos nucleos a la vez. El interprete no llegaba a
    // reutilizar practicamente nada: volvia a decodificar el mismo codigo ARM
    // una y otra vez. La cache de 24 MB, el vaciado por capacidad, los dos
    // niveles de indice... todo eso daba igual, porque nada sobrevivia a la
    // siguiente rodaja.
    //
    // Si la tabla es la misma, las traducciones siguen siendo validas por
    // definicion: traducen codigo leido a traves de ese mismo mapa de memoria.
    if (current_page_table == page_table) {
        return;
    }
    current_page_table = page_table;
    ClearInstructionCache();
}

std::shared_ptr<Memory::PageTable> ARM_DynCom::GetPageTable() const {
    return nullptr;
}

void ARM_DynCom::SetPC(u32 pc) {
    state->Reg[15] = pc;
}

u32 ARM_DynCom::GetPC() const {
    return state->Reg[15];
}

u32 ARM_DynCom::GetReg(int index) const {
    return state->Reg[index];
}

void ARM_DynCom::SetReg(int index, u32 value) {
    state->Reg[index] = value;
}

u32 ARM_DynCom::GetVFPReg(int index) const {
    return state->ExtReg[index];
}

void ARM_DynCom::SetVFPReg(int index, u32 value) {
    state->ExtReg[index] = value;
}

u32 ARM_DynCom::GetVFPSystemReg(VFPSystemRegister reg) const {
    return state->VFP[reg];
}

void ARM_DynCom::SetVFPSystemReg(VFPSystemRegister reg, u32 value) {
    state->VFP[reg] = value;
}

u32 ARM_DynCom::GetCPSR() const {
    return state->Cpsr;
}

void ARM_DynCom::SetCPSR(u32 cpsr) {
    state->Cpsr = cpsr;
}

u32 ARM_DynCom::GetCP15Register(CP15Register reg) const {
    return state->CP15[reg];
}

void ARM_DynCom::SetCP15Register(CP15Register reg, u32 value) {
    state->CP15[reg] = value;
}

void ARM_DynCom::ExecuteInstructions(u64 num_instructions) {
    // Antes de entrar al interprete, no dentro: el bucle de despacho se ejecuta
    // una vez por bloque traducido y es lo mas caliente del emulador, asi que no
    // conviene meterle ni una comparacion mas. Aqui la comprobacion sale gratis
    // -- una vez por rodaja de tiempo -- a cambio de reservar un margen mas
    // generoso, porque entre dos comprobaciones cabe una rodaja entera de
    // traducciones nuevas. Ver TRANS_CACHE_MARGIN.
    FlushTransCacheIfNeeded();

    state->NumInstrsToExecute = num_instructions;
    const u32 ticks_executed = InterpreterMainLoop(state.get());
    if (timer) {
        timer->AddTicks(ticks_executed);
    }
    state->ServeBreak();
}

void ARM_DynCom::SaveContext(ThreadContext& ctx) {
    ctx.cpu_registers = state->Reg;
    ctx.cpsr = state->Cpsr;
    ctx.fpu_registers = state->ExtReg;
    ctx.fpscr = state->VFP[VFP_FPSCR];
    ctx.fpexc = state->VFP[VFP_FPEXC];
}

void ARM_DynCom::LoadContext(const ThreadContext& ctx) {
    state->Reg = ctx.cpu_registers;
    state->Cpsr = ctx.cpsr;
    state->ExtReg = ctx.fpu_registers;
    state->VFP[VFP_FPSCR] = ctx.fpscr;
    state->VFP[VFP_FPEXC] = ctx.fpexc;
}

void ARM_DynCom::PrepareReschedule() {
    state->NumInstrsToExecute = 0;
}

} // namespace Core
