// Copyright 2015-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

/*  armdefs.h -- ARMulator common definitions:  ARM6 Instruction Emulator.
    Copyright (C) 1994 Advanced RISC Machines Ltd.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA. */

#pragma once

#include <array>
#include <unordered_map>
#include "common/common_types.h"
#include "core/arm/skyeye_common/arm_regformat.h"
#ifdef ENABLE_GDBSTUB
#include "core/gdbstub/gdbstub.h"
#endif

namespace Core {
class System;
}

// Antes solo se declaraba MemorySystem hacia delante. Se incluye entero para
// poder definir los accesos a memoria del invitado INLINE aqui abajo: si se
// quedan en armstate.cpp, cada LDR/STR emulado paga una llamada que el
// compilador no puede inlinear desde el interprete, que esta en otro fichero.
#include "core/memory.h"

// Signal levels
enum { LOW = 0, HIGH = 1, LOWHIGH = 1, HIGHLOW = 2 };

// Cache types
enum {
    NONCACHE = 0,
    DATACACHE = 1,
    INSTCACHE = 2,
};

// ARM privilege modes
enum PrivilegeMode {
    USER32MODE = 16,
    FIQ32MODE = 17,
    IRQ32MODE = 18,
    SVC32MODE = 19,
    ABORT32MODE = 23,
    UNDEF32MODE = 27,
    SYSTEM32MODE = 31
};

// ARM privilege mode register banks
enum {
    USERBANK = 0,
    FIQBANK = 1,
    IRQBANK = 2,
    SVCBANK = 3,
    ABORTBANK = 4,
    UNDEFBANK = 5,
    DUMMYBANK = 6,
    SYSTEMBANK = 7
};

// Hardware vector addresses
enum {
    ARMResetV = 0,
    ARMUndefinedInstrV = 4,
    ARMSWIV = 8,
    ARMPrefetchAbortV = 12,
    ARMDataAbortV = 16,
    ARMAddrExceptnV = 20,
    ARMIRQV = 24,
    ARMFIQV = 28,
    ARMErrorV = 32, // This is an offset, not an address!

    ARMul_ResetV = ARMResetV,
    ARMul_UndefinedInstrV = ARMUndefinedInstrV,
    ARMul_SWIV = ARMSWIV,
    ARMul_PrefetchAbortV = ARMPrefetchAbortV,
    ARMul_DataAbortV = ARMDataAbortV,
    ARMul_AddrExceptnV = ARMAddrExceptnV,
    ARMul_IRQV = ARMIRQV,
    ARMul_FIQV = ARMFIQV
};

// Coprocessor status values
enum {
    ARMul_FIRST = 0,
    ARMul_TRANSFER = 1,
    ARMul_BUSY = 2,
    ARMul_DATA = 3,
    ARMul_INTERRUPT = 4,
    ARMul_DONE = 0,
    ARMul_CANT = 1,
    ARMul_INC = 3
};

// Instruction condition codes
enum ConditionCode {
    EQ = 0,
    NE = 1,
    CS = 2,
    CC = 3,
    MI = 4,
    PL = 5,
    VS = 6,
    VC = 7,
    HI = 8,
    LS = 9,
    GE = 10,
    LT = 11,
    GT = 12,
    LE = 13,
    AL = 14,
    NV = 15,
};

// Flags for use with the APSR.
enum : u32 {
    NBIT = (1U << 31U),
    ZBIT = (1 << 30),
    CBIT = (1 << 29),
    VBIT = (1 << 28),
    QBIT = (1 << 27),
    JBIT = (1 << 24),
    EBIT = (1 << 9),
    ABIT = (1 << 8),
    IBIT = (1 << 7),
    FBIT = (1 << 6),
    TBIT = (1 << 5),

    // Masks for groups of bits in the APSR.
    MODEBITS = 0x1F,
    INTBITS = 0x1C0,
};

// Values for Emulate.
enum {
    STOP = 0,       // Stop
    CHANGEMODE = 1, // Change mode
    ONCE = 2,       // execute just one iteration
    RUN = 3         // Continuous execution
};

struct ARMul_State final {
public:
    explicit ARMul_State(Core::System& system, Memory::MemorySystem& memory,
                         PrivilegeMode initial_mode);

    void ChangePrivilegeMode(u32 new_mode);
    void Reset();

    // Reads/writes data in big/little endian format based on the
    // state of the E (endian) bit in the APSR.
    //
    // Definidos aqui (inline) y no en armstate.cpp a proposito: son el camino
    // de cada LDR/STR del juego emulado, o sea del orden de una de cada cuatro
    // instrucciones. Estando en el .cpp, el interprete pagaba una llamada sin
    // inlinear hasta aqui y otra desde aqui hasta MemorySystem, mas la
    // indireccion del pimpl. Con las dos inline, el caso normal queda en un
    // indice a un array y una copia.
    u8 ReadMemory8(u32 address) const {
        return memory.Read8Fast(address);
    }
    u16 ReadMemory16(u32 address) const {
        const u16 data = memory.Read16Fast(address);
        return InBigEndianMode() ? Common::swap16(data) : data;
    }
    u32 ReadMemory32(u32 address) const {
        const u32 data = memory.Read32Fast(address);
        return InBigEndianMode() ? Common::swap32(data) : data;
    }
    u64 ReadMemory64(u32 address) const;
    void WriteMemory8(u32 address, u8 data) {
        memory.Write8Fast(address, data);
    }
    void WriteMemory16(u32 address, u16 data) {
        memory.Write16Fast(address, InBigEndianMode() ? Common::swap16(data) : data);
    }
    void WriteMemory32(u32 address, u32 data) {
        memory.Write32Fast(address, InBigEndianMode() ? Common::swap32(data) : data);
    }
    void WriteMemory64(u32 address, u64 data);

    u32 ReadCP15Register(u32 crn, u32 opcode_1, u32 crm, u32 opcode_2) const;
    void WriteCP15Register(u32 value, u32 crn, u32 opcode_1, u32 crm, u32 opcode_2);

    // Exclusive memory access functions
    bool IsExclusiveMemoryAccess(u32 address) const {
        return exclusive_state && exclusive_tag == (address & RESERVATION_GRANULE_MASK);
    }
    void SetExclusiveMemoryAddress(u32 address) {
        exclusive_tag = address & RESERVATION_GRANULE_MASK;
        exclusive_state = true;
    }
    void UnsetExclusiveMemoryAddress() {
        exclusive_tag = 0xFFFFFFFF;
        exclusive_state = false;
    }

    // Whether or not the given CPU is in big endian mode (E bit is set)
    bool InBigEndianMode() const {
        return (Cpsr & (1 << 9)) != 0;
    }
    // Whether or not the given CPU is in a mode other than user mode.
    bool InAPrivilegedMode() const {
        return (Mode != USER32MODE);
    }
    // Whether or not the current CPU mode has a Saved Program Status Register
    bool CurrentModeHasSPSR() const {
        return Mode != SYSTEM32MODE && InAPrivilegedMode();
    }
    // Note that for the 3DS, a Thumb instruction will only ever be
    // two bytes in size. Thus we don't need to worry about ThumbEE
    // or Thumb-2 where instructions can be 4 bytes in length.
    u32 GetInstructionSize() const {
        return TFlag ? 2 : 4;
    }

#ifdef ENABLE_GDBSTUB
    void RecordBreak(GDBStub::BreakpointAddress bkpt) {
        last_bkpt = bkpt;
        last_bkpt_hit = true;
    }
#endif

    void ServeBreak();

    Core::System& system;
    Memory::MemorySystem& memory;

    std::array<u32, 16> Reg{}; // The current register file
    std::array<u32, 2> Reg_usr{};
    std::array<u32, 2> Reg_svc{};   // R13_SVC R14_SVC
    std::array<u32, 2> Reg_abort{}; // R13_ABORT R14_ABORT
    std::array<u32, 2> Reg_undef{}; // R13 UNDEF R14 UNDEF
    std::array<u32, 2> Reg_irq{};   // R13_IRQ R14_IRQ
    std::array<u32, 7> Reg_firq{};  // R8---R14 FIRQ
    std::array<u32, 7> Spsr{};      // The exception psr's
    std::array<u32, CP15_REGISTER_COUNT> CP15{};

    // FPSID, FPSCR, and FPEXC
    std::array<u32, VFP_SYSTEM_REGISTER_COUNT> VFP{};

    // VFPv2 and VFPv3-D16 has 16 doubleword registers (D0-D16 or S0-S31).
    // VFPv3-D32/ASIMD may have up to 32 doubleword registers (D0-D31),
    // and only 32 singleword registers are accessible (S0-S31).
    std::array<u32, 64> ExtReg{};

    u32 Emulate; // To start and stop emulation
    u32 Cpsr;    // The current PSR
    u32 Spsr_copy;
    u32 phys_pc;

    u32 Mode; // The current mode
    u32 Bank; // The current register bank

    u32 NFlag, ZFlag, CFlag, VFlag, IFFlags; // Dummy flags for speed
    unsigned int shifter_carry_out;

    u32 TFlag; // Thumb state

    unsigned long long NumInstrs; // The number of instructions executed
    u64 NumInstrsToExecute;

    unsigned NresetSig; // Reset the processor
    unsigned NfiqSig;
    unsigned NirqSig;

    unsigned abortSig;
    unsigned NtransSig;
    unsigned bigendSig;
    unsigned syscallSig;

    // TODO(bunnei): Move this cache to a better place - it should be per codeset (likely per
    // process for our purposes), not per ARMul_State (which tracks CPU core state).
    std::unordered_map<u32, std::size_t> instruction_cache;

    /**
     * Cache de primer nivel del indice de bloques traducidos.
     *
     * instruction_cache se consulta en CADA salto entre bloques, o sea cada
     * pocas instrucciones emuladas. Es un unordered_map: nodos sueltos repartidos
     * por el heap, asi que cada consulta es calcular el hash y seguir punteros a
     * memoria que casi nunca esta en la cache del Cortex-A9. En una consola donde
     * la emulacion de CPU se come el 100% de un nucleo, eso se nota.
     *
     * Esto es un array plano de mapeo directo delante: una indexacion y una
     * comparacion. Si acierta, nos ahorramos el mapa entero; si falla, se
     * consulta el mapa como siempre y se guarda aqui para la proxima. El mapa
     * sigue siendo la fuente de verdad -- esto es solo un atajo -- asi que un
     * fallo de esta cache nunca da un resultado incorrecto, solo mas lento.
     *
     * 8192 entradas de 8 bytes = 64 KB por nucleo. El indice usa los bits 2 en
     * adelante del PC porque los dos de abajo son siempre cero en ARM (y el bit
     * 1 en Thumb), y si no se descartan la mitad de las ranuras no se usarian.
     */
    static constexpr std::size_t BLOCK_CACHE_SIZE = 8192;
    static constexpr std::size_t BLOCK_CACHE_MASK = BLOCK_CACHE_SIZE - 1;

    struct BlockCacheEntry {
        u32 pc;
        std::size_t offset;
    };
    /// pc = INVALID_BLOCK_PC marca la ranura como vacia. Es una direccion impar,
    /// que nunca puede ser un PC valido (ARM alinea a 4, Thumb a 2).
    static constexpr u32 INVALID_BLOCK_PC = 0xFFFFFFFF;
    std::array<BlockCacheEntry, BLOCK_CACHE_SIZE> block_cache;

    void ClearBlockCache() {
        for (auto& entry : block_cache) {
            entry.pc = INVALID_BLOCK_PC;
        }
    }

private:
    void ResetMPCoreCP15Registers();

    // Defines a reservation granule of 2 words, which protects the first 2 words starting at the
    // tag. This is the smallest granule allowed by the v7 spec, and is coincidentally just large
    // enough to support LDR/STREXD.
    static const u32 RESERVATION_GRANULE_MASK = 0xFFFFFFF8;

    u32 exclusive_tag; // The address for which the local monitor is in exclusive access mode
    bool exclusive_state;

#ifdef ENABLE_GDBSTUB
    GDBStub::BreakpointAddress last_bkpt{};
    bool last_bkpt_hit = false;
#endif
};
