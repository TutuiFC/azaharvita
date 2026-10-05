// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// JIT de shaders de la PICA200 a NEON. La explicacion, en shader_neon_jit.h.

#ifdef __PSVITA__

#include "video_core/shader/shader_neon_jit.h"

#include <algorithm>
#include <cstddef>
#include <vector>
#include <nihstro/shader_bytecode.h>
#include <psp2/kernel/sysmem.h>
#include "common/vita_diag.h"
#include "common/vita_vm.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"
#include "video_core/shader/shader_interpreter_fast.h"

namespace Pica::Shader::Fast {

std::atomic<bool> g_neon_jit{true};

namespace {

using nihstro::Instruction;
using nihstro::OpCode;

/// Memoria de codigo para todos los programas de la sesion. Un programa
/// entero son unas decenas de KB. Cuando se llena, los programas nuevos se
/// quedan en la ruta rapida interpretada, que es correcta.
constexpr u32 kCodeBytes = 4u * 1024u * 1024u;
SceUID g_block = -1;
u32* g_code = nullptr;
u32 g_used = 0; // palabras
bool g_init_tried = false;

/// Constantes que el codigo carga al empezar y despues de cada llamada.
struct alignas(16) Constants {
    u32 sign[4] = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u};
    float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};
const Constants g_constants;

/// Lo que no se traduce (FLR, EX2, LG2): la misma funcion de la ruta rapida.
void CallExecuteOp(const Op* op, ShaderUnit* state, const Uniforms* uniforms) {
    ExecuteOp(*op, *state, *uniforms);
}

bool Init() {
    g_init_tried = true;
    g_block = sceKernelAllocMemBlockForVM("azahar_pica_jit", kCodeBytes);
    if (g_block < 0) {
        Common::VitaNote("vs neon", "sin memoria ejecutable: se sigue con la ruta rapida");
        return false;
    }
    void* base = nullptr;
    if (sceKernelGetMemBlockBase(g_block, &base) < 0 || base == nullptr) {
        return false;
    }
    g_code = static_cast<u32*>(base);
    return true;
}

// Registros del anfitrion. r4-r10 los conserva cualquier funcion de C.
constexpr u32 R0 = 0, R1 = 1, R2 = 2, kState = 4, kUniforms = 5, kOut = 6, kTmp = 7, kFlow = 8,
              kTable = 9, R12 = 12;
// Registros Q de NEON: a, b, c (fuentes), resultado y temporales. q11 = signo,
// q12 = unos.
constexpr u32 QA = 0, QB = 1, QC = 2, QR = 3, QT0 = 8, QT1 = 9, QT2 = 10, QSign = 11,
              QOnes = 12, QIdx = 14, QRaw = 15;

// Condiciones ARM.
constexpr u32 kEq = 0, kNe = 1, kHs = 2, kMi = 4, kLs = 9, kGe = 10, kLt = 11, kGt = 12,
              kAl = 14;

class Emitter {
public:
    Emitter(u32* base_, u32 capacity_) : base{base_}, capacity{capacity_} {}
    bool Overflowed() const {
        return overflow;
    }
    u32 Position() const {
        return position;
    }
    /// Descarta lo emitido desde 'index' (sin saltos pendientes dentro).
    void Rewind(u32 index) {
        if (index <= position) {
            position = index;
        }
    }
    u32 Address(u32 index) const {
        return static_cast<u32>(reinterpret_cast<uintptr_t>(base + index));
    }
    void Emit(u32 word) {
        if (position >= capacity) {
            overflow = true;
            return;
        }
        base[position++] = word;
    }
    void Mov32(u32 rd, u32 value, u32 cond = kAl) {
        Emit((cond << 28) | 0x03000000u | ((value & 0xF000u) << 4) | (rd << 12) |
             (value & 0x0FFFu));
        const u32 high = value >> 16;
        if (high != 0) {
            Emit((cond << 28) | 0x03400000u | ((high & 0xF000u) << 4) | (rd << 12) |
                 (high & 0x0FFFu));
        }
    }
    void MovReg(u32 rd, u32 rm) {
        Emit(0xE1A00000u | (rd << 12) | rm);
    }
    void AddReg(u32 rd, u32 rn, u32 rm) {
        Emit(0xE0800000u | (rn << 16) | (rd << 12) | rm);
    }
    void AddConst(u32 rd, u32 rn, u32 value) {
        Mov32(R12, value);
        AddReg(rd, rn, R12);
    }
    void Ldr(u32 rt, u32 rn, u32 offset) {
        Emit(0xE5900000u | (rn << 16) | (rt << 12) | (offset & 0xFFFu));
    }
    void Str(u32 rt, u32 rn, u32 offset) {
        Emit(0xE5800000u | (rn << 16) | (rt << 12) | (offset & 0xFFFu));
    }
    void Strb(u32 rt, u32 rn, u32 offset) {
        Emit(0xE5C00000u | (rn << 16) | (rt << 12) | (offset & 0xFFFu));
    }
    void Call(const void* function) {
        Mov32(R12, static_cast<u32>(reinterpret_cast<uintptr_t>(function)));
        Emit(0xE12FFF3Cu); // BLX r12
    }
    /// Salto con condicion a una posicion que se resuelve despues.
    u32 BranchPlaceholder(u32 cond) {
        const u32 index = position;
        Emit((cond << 28) | 0x0A000000u);
        return index;
    }
    void Patch(u32 index, u32 target) {
        if (overflow) {
            return;
        }
        const s32 offset = static_cast<s32>(target) - static_cast<s32>(index + 2);
        base[index] = (base[index] & 0xFF000000u) | (static_cast<u32>(offset) & 0x00FFFFFFu);
    }
    // ---- NEON, con registros Q (Qn = D(2n), D(2n+1)) ----
    static u32 QD(u32 q) {
        const u32 d = q * 2;
        return ((d >> 4) << 22) | ((d & 15) << 12);
    }
    static u32 QN(u32 q) {
        const u32 d = q * 2;
        return ((d >> 4) << 7) | ((d & 15) << 16);
    }
    static u32 QM(u32 q) {
        const u32 d = q * 2;
        return ((d >> 4) << 5) | (d & 15);
    }
    void Q3(u32 opcode, u32 qd, u32 qn, u32 qm) {
        Emit(opcode | QD(qd) | QN(qn) | QM(qm));
    }
    void Vld1(u32 qd, u32 rn) {
        Emit(0xF4200A8Fu | QD(qd) | (rn << 16));
    }
    void Vst1(u32 qd, u32 rn) {
        Emit(0xF4000A8Fu | QD(qd) | (rn << 16));
    }
    void Vtbl2(u32 dd, u32 dn, u32 dm) {
        Emit(0xF3B00900u | ((dd >> 4) << 22) | ((dd & 15) << 12) | ((dn >> 4) << 7) |
             ((dn & 15) << 16) | ((dm >> 4) << 5) | (dm & 15));
    }
    // ---- VFP escalar (registros S) ----
    static u32 SD(u32 s) {
        return ((s & 1) << 22) | ((s >> 1) << 12);
    }
    static u32 SN(u32 s) {
        return ((s & 1) << 7) | ((s >> 1) << 16);
    }
    static u32 SM(u32 s) {
        return ((s & 1) << 5) | (s >> 1);
    }
    void VaddS(u32 sd, u32 sn, u32 sm) {
        Emit(0xEE300A00u | SD(sd) | SN(sn) | SM(sm));
    }
    void VdivS(u32 sd, u32 sn, u32 sm) {
        Emit(0xEE800A00u | SD(sd) | SN(sn) | SM(sm));
    }
    void VsqrtS(u32 sd, u32 sm) {
        Emit(0xEEB10AC0u | SD(sd) | SM(sm));
    }
    /// VCVT.S32.F32 Sd, Sm (hacia cero, como static_cast<s32>)
    void VcvtS32(u32 sd, u32 sm) {
        Emit(0xEEBD0AC0u | SD(sd) | SM(sm));
    }
    /// VMOV Rt, Sn
    void VmovToCore(u32 rt, u32 sn) {
        Emit(0xEE100A10u | (rt << 12) | SN(sn));
    }
    void VcmpS(u32 sd, u32 sm) {
        Emit(0xEEB40A40u | SD(sd) | SM(sm));
    }
    void VmrsFlags() {
        Emit(0xEEF1FA10u); // VMRS APSR_nzcv, FPSCR
    }
    void VldrS(u32 sd, u32 rn, u32 offset) {
        Emit(0xED900A00u | SD(sd) | (rn << 16) | (offset / 4));
    }
    void VdupLane0(u32 qd, u32 dm) {
        Emit(0xF3B40C40u | QD(qd) | ((dm >> 4) << 5) | (dm & 15));
    }

private:
    u32* base;
    u32 capacity;
    u32 position = 0;
    bool overflow = false;
};

// Codificaciones NEON (Q = 1).
constexpr u32 kVadd = 0xF2000D40u;
constexpr u32 kVmul = 0xF3000D50u;
constexpr u32 kVceq = 0xF2000E40u;
constexpr u32 kVcgt = 0xF3200E40u;
constexpr u32 kVcge = 0xF3000E40u;
constexpr u32 kVand = 0xF2000150u;
constexpr u32 kVbic = 0xF2100150u;
constexpr u32 kVeor = 0xF3000150u;
constexpr u32 kVbsl = 0xF3100150u;

u32 SourceCount(Kind kind) {
    switch (kind) {
    case Kind::Mad:
        return 3;
    case Kind::Mov:
    case Kind::Flr:
    case Kind::Rcp:
    case Kind::Rsq:
    case Kind::Ex2:
    case Kind::Lg2:
    case Kind::Mova:
        return 1;
    default:
        return 2;
    }
}

/// FLR, EX2 y LG2 llaman a la misma funcion de la ruta rapida (usan floor,
/// exp2 y log2 de la biblioteca, que no tienen instruccion en ARMv7).
bool NeedsCall(Kind kind) {
    return kind == Kind::Flr || kind == Kind::Ex2 || kind == Kind::Lg2;
}

void EmitLoadConstants(Emitter& e) {
    e.Mov32(kTmp, static_cast<u32>(reinterpret_cast<uintptr_t>(g_constants.sign)));
    e.Vld1(QSign, kTmp);
    e.Mov32(kTmp, static_cast<u32>(reinterpret_cast<uintptr_t>(g_constants.ones)));
    e.Vld1(QOnes, kTmp);
}

/// La fuente n de op en el registro Q 'q' (igual que LoadSource).
void EmitLoadSource(Emitter& e, const Op& op, u32 n, u32 q) {
    const Src& src = op.src[n];
    if (src.type == SrcType::Uniform && src.addr_reg != 0) {
        /**
         * Uniform con registro de direccion (0.1.7.2): el mismo calculo que
         * LoadSource. Desplazamiento fuera de [-128, 127] = 0; indice
         * envuelto a 7 bits; de 96 en adelante, unos.
         */
        e.Ldr(R0, kState,
              static_cast<u32>(offsetof(ShaderUnit, address_registers)) +
                  (src.addr_reg - 1u) * 4u);
        e.Emit(0xE3700080u | (R0 << 16));             // CMN r0, #128
        e.Emit((kLt << 28) | 0x03A00000u | (R0 << 12)); // MOVLT r0, #0
        e.Emit(0xE350007Fu | (R0 << 16));             // CMP r0, #127
        e.Emit((kGt << 28) | 0x03A00000u | (R0 << 12)); // MOVGT r0, #0
        e.Emit(0xE2800000u | (R0 << 16) | (R0 << 12) | src.index); // ADD r0, r0, #index
        e.Emit(0xE2000000u | (R0 << 16) | (R0 << 12) | 0x7Fu);     // AND r0, r0, #0x7F
        e.Emit(0xE3500060u | (R0 << 16));                          // CMP r0, #96
        e.Emit((kLt << 28) | 0x00800000u | (kUniforms << 16) | (kTmp << 12) | (4u << 7) |
               R0); // ADDLT kTmp, kUniforms, r0, LSL #4
        e.Mov32(kTmp, static_cast<u32>(reinterpret_cast<uintptr_t>(g_constants.ones)), kGe);
    } else {
        const u32 base = src.type == SrcType::Uniform ? kUniforms : kState;
        e.AddConst(kTmp, base, src.offset);
    }
    if (src.identity) {
        e.Vld1(q, kTmp);
    } else {
        e.Vld1(QRaw, kTmp);
        e.Mov32(kTmp, static_cast<u32>(reinterpret_cast<uintptr_t>(op.swizzle[n].data())));
        e.Vld1(QIdx, kTmp);
        e.Vtbl2(q * 2, QRaw * 2, QIdx * 2);
        e.Vtbl2(q * 2 + 1, QRaw * 2, QIdx * 2 + 1);
    }
    if (src.negated) {
        e.Q3(kVeor, q, q, QSign);
    }
}

/// qd = SanitizedMul(qa, qb): f24::operator* en cuatro carriles.
void EmitSanitizedMul(Emitter& e, u32 qd, u32 qa, u32 qb) {
    e.Q3(kVmul, qd, qa, qb);
    e.Q3(kVceq, QT0, qd, qd);
    e.Q3(kVceq, QT1, qa, qa);
    e.Q3(kVceq, QT2, qb, qb);
    e.Q3(kVand, QT1, QT1, QT2);
    e.Q3(kVbic, QT1, QT1, QT0);
    e.Q3(kVbic, qd, qd, QT1);
}

/// Guarda QR en el destino de op (igual que Store).
void EmitStore(Emitter& e, const Op& op) {
    if (op.dest_is_output) {
        const u32 index_offset = static_cast<u32>(op.dest_index) * 16;
        if (index_offset != 0) {
            e.AddConst(kTmp, kOut, index_offset);
        } else {
            e.MovReg(kTmp, kOut);
        }
    } else {
        e.AddConst(kTmp, kState, op.dest_offset);
    }
    if (op.full_mask) {
        e.Vst1(QR, kTmp);
        return;
    }
    e.Vld1(QT0, kTmp);
    e.Mov32(R12, static_cast<u32>(reinterpret_cast<uintptr_t>(op.write_mask.data())));
    e.Vld1(QT1, R12);
    e.Q3(kVbsl, QT1, QR, QT0);
    e.Vst1(QT1, kTmp);
}

/// Una comparacion de CMP (Compare de la ruta rapida) de sa con sb, al byte
/// de conditional_code[index].
void EmitCompare(Emitter& e, u8 op, u32 sa, u32 sb, u32 index) {
    static constexpr u32 kCond[6] = {kEq, kNe, kMi, kLs, kGt, kGe};
    e.VcmpS(sa, sb);
    e.VmrsFlags();
    e.Emit(0xE3A00000u | (R0 << 12));                       // MOV r0, #0
    e.Emit((kCond[op] << 28) | 0x03A00001u | (R0 << 12));   // MOV<cc> r0, #1
    e.Strb(R0, kState, static_cast<u32>(offsetof(ShaderUnit, conditional_code)) + index);
}

void EmitOp(Emitter& e, const Op& op) {
    const u32 sources = SourceCount(op.kind);
    if (NeedsCall(op.kind)) {
        e.Mov32(R0, static_cast<u32>(reinterpret_cast<uintptr_t>(&op)));
        e.MovReg(R1, kState);
        e.MovReg(R2, kUniforms);
        e.Call(reinterpret_cast<const void*>(&CallExecuteOp));
        // La llamada destroza q0-q3 y q8-q15: las constantes, otra vez.
        EmitLoadConstants(e);
        return;
    }
    EmitLoadSource(e, op, 0, QA);
    if (sources >= 2) {
        EmitLoadSource(e, op, 1, QB);
    }
    if (sources >= 3) {
        EmitLoadSource(e, op, 2, QC);
    }
    switch (op.kind) {
    case Kind::Add:
        e.Q3(kVadd, QR, QA, QB);
        break;
    case Kind::Mul:
        EmitSanitizedMul(e, QR, QA, QB);
        break;
    case Kind::Mad:
        EmitSanitizedMul(e, QR, QA, QB);
        e.Q3(kVadd, QR, QR, QC);
        break;
    case Kind::Dp3:
    case Kind::Dp4:
    case Kind::Dph:
        if (op.kind == Kind::Dph) {
            e.Mov32(kTmp, static_cast<u32>(reinterpret_cast<uintptr_t>(g_constants.ones)));
            e.VldrS(3, kTmp, 0); // a.w = 1.0, despues de negar
        }
        EmitSanitizedMul(e, QR, QA, QB); // productos en s12..s15
        e.Mov32(kTmp, static_cast<u32>(reinterpret_cast<uintptr_t>(g_constants.zero)));
        e.VldrS(0, kTmp, 0);
        e.VaddS(0, 0, 12);
        e.VaddS(0, 0, 13);
        e.VaddS(0, 0, 14);
        if (op.kind != Kind::Dp3) {
            e.VaddS(0, 0, 15);
        }
        e.VdupLane0(QR, 0);
        break;
    case Kind::Max:
        e.Q3(kVcgt, QR, QA, QB);
        e.Q3(kVbsl, QR, QA, QB);
        break;
    case Kind::Min:
        e.Q3(kVcgt, QR, QB, QA);
        e.Q3(kVbsl, QR, QA, QB);
        break;
    case Kind::Sge:
        e.Q3(kVcge, QR, QA, QB);
        e.Q3(kVand, QR, QR, QOnes);
        break;
    case Kind::Slt:
        e.Q3(kVcgt, QR, QB, QA);
        e.Q3(kVand, QR, QR, QOnes);
        break;
    case Kind::Mov:
        e.Q3(kVand, QR, QA, QA);
        break;
    case Kind::Rcp:
    case Kind::Rsq:
        // 1.0f / x  y  1.0f / sqrt(x) con la x del carril 0, en VFP escalar,
        // como la ruta rapida (misma division y misma raiz correctamente
        // redondeadas).
        e.Mov32(kTmp, static_cast<u32>(reinterpret_cast<uintptr_t>(g_constants.ones)));
        e.VldrS(8, kTmp, 0);
        if (op.kind == Kind::Rsq) {
            e.VsqrtS(0, 0);
        }
        e.VdivS(0, 8, 0);
        e.VdupLane0(QR, 0);
        break;
    case Kind::Mova:
        // static_cast<s32>(float) = VCVT hacia cero, como la ruta rapida.
        if (op.mova_mask & 1) {
            e.VcvtS32(2, 0);
            e.VmovToCore(R0, 2);
            e.Str(R0, kState, static_cast<u32>(offsetof(ShaderUnit, address_registers)));
        }
        if (op.mova_mask & 2) {
            e.VcvtS32(3, 1);
            e.VmovToCore(R0, 3);
            e.Str(R0, kState, static_cast<u32>(offsetof(ShaderUnit, address_registers)) + 4);
        }
        return; // sin destino
    case Kind::Cmp:
        EmitCompare(e, op.cmp_x, 0, 4, 0); // a.x (s0) con b.x (s4)
        EmitCompare(e, op.cmp_y, 1, 5, 1); // a.y (s1) con b.y (s5)
        return; // sin destino
    default:
        break;
    }
    EmitStore(e, op);
}

/// Prologo comun: registros, salida del banco activo y constantes.
void EmitPrologue(Emitter& e) {
    e.MovReg(kState, R0);
    e.MovReg(kUniforms, R1);
    // kOut = salida del banco activo (EMIT, lo unico que lo cambia, no entra).
    e.AddConst(kOut, kState, static_cast<u32>(ShaderUnit::OutputOffset(0)));
    e.Emit(0xE5D00000u | (kState << 16) | (kTmp << 12) |
           static_cast<u32>(ShaderUnit::OutputBankOffset())); // LDRB kTmp, [kState, #bank]
    e.Emit(0xE3500000u | (kTmp << 16));                        // CMP kTmp, #0
    e.Mov32(R12, static_cast<u32>(ShaderUnit::OutputBankSize));
    e.Emit(0x10800000u | (kOut << 16) | (kOut << 12) | R12); // ADDNE kOut, kOut, r12
    EmitLoadConstants(e);
}

bool IsFlowInstruction(u32 word) {
    Instruction instr{};
    instr.hex = word;
    switch (instr.opcode.Value()) {
    case OpCode::Id::END:
    case OpCode::Id::JMPC:
    case OpCode::Id::JMPU:
    case OpCode::Id::CALL:
    case OpCode::Id::CALLU:
    case OpCode::Id::CALLC:
    case OpCode::Id::NOP:
    case OpCode::Id::IFU:
    case OpCode::Id::IFC:
    case OpCode::Id::LOOP:
    case OpCode::Id::BREAK:
    case OpCode::Id::BREAKC:
        return true;
    default:
        return false;
    }
}

bool EnsureMemory() {
    if (g_code == nullptr && (g_init_tried || !Init())) {
        return false;
    }
    return true;
}

} // Anonymous namespace

void CompileRuns(Program& program) {
    for (Run& run : program.runs) {
        run.code = nullptr;
    }
    if (!g_neon_jit.load(std::memory_order_relaxed) || !EnsureMemory()) {
        return;
    }
    // El dominio VM es de todo el proceso: ver Common::vita_vm_domain_mutex.
    const std::lock_guard vm_lock{Common::vita_vm_domain_mutex};
    if (sceKernelOpenVMDomain() < 0) {
        return;
    }
    for (Run& run : program.runs) {
        u32* const start = g_code + g_used;
        Emitter e{start, kCodeBytes / 4 - g_used};
        e.Emit(0xE92D47F0u); // PUSH {r4-r10, lr}
        EmitPrologue(e);
        for (u32 i = 0; i < run.count; i++) {
            EmitOp(e, program.ops[run.first_op + i]);
        }
        e.Emit(0xE8BD87F0u); // POP {r4-r10, pc}
        if (e.Overflowed()) {
            break;
        }
        sceKernelSyncVMDomain(g_block, start, e.Position() * 4);
        g_used += e.Position();
        run.code = reinterpret_cast<RunFn>(start);
    }
    sceKernelCloseVMDomain();
}

/// fast_ops += count (estadistica del overlay: ins / rap).
void EmitCountFast(Emitter& e, u32 count) {
    e.Ldr(R0, kState, static_cast<u32>(offsetof(ShaderUnit, fast_ops)));
    e.Mov32(R12, count);
    e.AddReg(R0, R0, R12);
    e.Str(R0, kState, static_cast<u32>(offsetof(ShaderUnit, fast_ops)));
}

/**
 * NOP, END, JMPU y JMPC EN NATIVO (0.2.1.3). Eran una llamada a FlowStep cada
 * uno (unos 150 ciclos con la vuelta por la tabla). Solo si la direccion de
 * detras NO es el final de ningun IF, CALL o LOOP: entonces las comprobaciones
 * de pila de despues de la instruccion no pueden hacer nada (CheckStacks solo
 * mira old_pc + 1) y el resultado es el mismo. Los saltos, solo a direcciones
 * con etiqueta (principio de tramo o instruccion de flujo). Si no, false y se
 * llama a FlowStep como antes.
 */
bool EmitNativeFlow(Emitter& e, u32 word, bool next_is_stack_address, bool target_ok,
                    std::vector<std::pair<u32, u32>>& to_label, std::vector<u32>& to_epilogue) {
    if (next_is_stack_address) {
        return false;
    }
    Instruction instr{};
    instr.hex = word;
    const u32 dest = instr.flow_control.dest_offset;
    switch (instr.opcode.Value()) {
    case OpCode::Id::NOP:
        EmitCountFast(e, 1);
        return true;
    case OpCode::Id::END:
        EmitCountFast(e, 1);
        to_epilogue.push_back(e.BranchPlaceholder(kAl));
        return true;
    case OpCode::Id::JMPU: {
        if (!target_ok) {
            return false;
        }
        EmitCountFast(e, 1);
        // Salta si b[id] == !(num & 1).
        const u32 expected = (instr.flow_control.num_instructions & 1) != 0 ? 0u : 1u;
        e.Emit(0xE5D00000u | (kUniforms << 16) | (R0 << 12) |
               static_cast<u32>(Uniforms::GetBoolUniformOffset(
                   instr.flow_control.bool_uniform_id))); // LDRB r0, [kUniforms, #b]
        e.Emit(0xE3500000u | (R0 << 16) | expected);    // CMP r0, #expected
        to_label.emplace_back(e.BranchPlaceholder(kEq), dest);
        return true;
    }
    case OpCode::Id::JMPC: {
        if (!target_ok) {
            return false;
        }
        using FlowOp = Instruction::FlowControlType::Op;
        const u32 refx = instr.flow_control.refx.Value() ? 1u : 0u;
        const u32 refy = instr.flow_control.refy.Value() ? 1u : 0u;
        const u32 cc = static_cast<u32>(offsetof(ShaderUnit, conditional_code));
        EmitCountFast(e, 1);
        e.Emit(0xE5D00000u | (kState << 16) | (R0 << 12) | cc);        // LDRB r0, cc0
        e.Emit(0xE5D00000u | (kState << 16) | (R1 << 12) | (cc + 1)); // LDRB r1, cc1
        switch (instr.flow_control.op) {
        case FlowOp::JustX:
            e.Emit(0xE3500000u | (R0 << 16) | refx); // CMP r0, #refx
            break;
        case FlowOp::JustY:
            e.Emit(0xE3500000u | (R1 << 16) | refy); // CMP r1, #refy
            break;
        case FlowOp::And:
            e.Emit(0xE3500000u | (R0 << 16) | refx);                  // CMP r0, #refx
            e.Emit((kEq << 28) | 0x03500000u | (R1 << 16) | refy);    // CMPEQ r1, #refy
            break;
        case FlowOp::Or:
            e.Emit(0xE3500000u | (R0 << 16) | refx);                  // CMP r0, #refx
            e.Emit((kNe << 28) | 0x03500000u | (R1 << 16) | refy);    // CMPNE r1, #refy
            break;
        default:
            return false;
        }
        to_label.emplace_back(e.BranchPlaceholder(kEq), dest);
        return true;
    }
    default:
        return false;
    }
}


// Registros libres en el flujo nativo: r0-r3, r10 y r12 (r10 lo guarda el PUSH
// del prologo y ningun otro codigo generado lo usa).
constexpr u32 R3 = 3, R10 = 10;

void EmitSubImm(Emitter& e, u32 rd, u32 rn, u32 imm8) {
    e.Emit(0xE2400000u | (rn << 16) | (rd << 12) | imm8);
}
void EmitAndImm(Emitter& e, u32 rd, u32 rn, u32 imm8) {
    e.Emit(0xE2000000u | (rn << 16) | (rd << 12) | imm8);
}
/// rd = kFlow + rm * 8
void EmitFlowSlot(Emitter& e, u32 rd, u32 rm) {
    e.Emit(0xE0800000u | (kFlow << 16) | (rd << 12) | (3u << 7) | rm);
}

/// Flags a EQ si se cumple la condicion de JMPC/IFC/CALLC (evaluate_condition
/// del interprete: refx == cc0, refy == cc1; O, Y, solo X o solo Y).
bool EmitCondition(Emitter& e, const Instruction& instr) {
    using FlowOp = Instruction::FlowControlType::Op;
    const u32 refx = instr.flow_control.refx.Value() ? 1u : 0u;
    const u32 refy = instr.flow_control.refy.Value() ? 1u : 0u;
    const u32 cc = static_cast<u32>(offsetof(ShaderUnit, conditional_code));
    e.Emit(0xE5D00000u | (kState << 16) | (R1 << 12) | cc);        // LDRB r1, cc0
    e.Emit(0xE5D00000u | (kState << 16) | (R2 << 12) | (cc + 1)); // LDRB r2, cc1
    switch (instr.flow_control.op) {
    case FlowOp::JustX:
        e.Emit(0xE3500000u | (R1 << 16) | refx);
        return true;
    case FlowOp::JustY:
        e.Emit(0xE3500000u | (R2 << 16) | refy);
        return true;
    case FlowOp::And:
        e.Emit(0xE3500000u | (R1 << 16) | refx);
        e.Emit((kEq << 28) | 0x03500000u | (R2 << 16) | refy);
        return true;
    case FlowOp::Or:
        e.Emit(0xE3500000u | (R1 << 16) | refx);
        e.Emit((kNe << 28) | 0x03500000u | (R2 << 16) | refy);
        return true;
    default:
        return false;
    }
}

/// Flags a EQ si b[id] vale 'expected' (0 o 1).
void EmitBoolTest(Emitter& e, u32 id, u32 expected) {
    e.Emit(0xE5D00000u | (kUniforms << 16) | (R1 << 12) |
           static_cast<u32>(Uniforms::GetBoolUniformOffset(id))); // LDRB r1, [kUniforms, #b]
    e.Emit(0xE3500000u | (R1 << 16) | expected);                  // CMP r1, #expected
}

/// push_back de FixedRingStack<T, n> con {first, second}: si esta llena, el
/// nuevo ocupa el hueco del mas antiguo y la cabeza avanza.
void EmitRingPush(Emitter& e, u32 items, u32 head, u32 count, u32 n, u32 first, u32 second) {
    e.Ldr(R2, kFlow, count);
    e.Ldr(R3, kFlow, head);
    e.Emit(0xE3500000u | (R2 << 16) | n);                                    // CMP r2, #n
    e.Emit((kNe << 28) | 0x00800000u | (R3 << 16) | (R12 << 12) | R2);      // ADDNE r12, r3, r2
    e.Emit((kNe << 28) | 0x02000000u | (R12 << 16) | (R12 << 12) | (n - 1)); // ANDNE r12, #n-1
    e.Emit((kNe << 28) | 0x02800000u | (R2 << 16) | (R2 << 12) | 1u);      // ADDNE r2, #1
    e.Emit((kNe << 28) | 0x05800000u | (kFlow << 16) | (R2 << 12) | count); // STRNE r2, count
    e.Emit((kEq << 28) | 0x01A00000u | (R12 << 12) | R3);                   // MOVEQ r12, r3
    e.Emit((kEq << 28) | 0x02800000u | (R3 << 16) | (R3 << 12) | 1u);      // ADDEQ r3, #1
    e.Emit((kEq << 28) | 0x02000000u | (R3 << 16) | (R3 << 12) | (n - 1)); // ANDEQ r3, #n-1
    e.Emit((kEq << 28) | 0x05800000u | (kFlow << 16) | (R3 << 12) | head); // STREQ r3, head
    EmitFlowSlot(e, R12, R12);
    e.Mov32(R10, first);
    e.Str(R10, R12, items);
    e.Mov32(R10, second);
    e.Str(R10, R12, items + 4);
}

/// r3 = kFlow + 8 * ((head + count - 1) & (n - 1)), con count (> 0) en r2.
void EmitRingBack(Emitter& e, u32 head, u32 n) {
    e.Ldr(R3, kFlow, head);
    e.AddReg(R3, R3, R2);
    EmitSubImm(e, R3, R3, 1);
    EmitAndImm(e, R3, R3, n - 1);
    EmitFlowSlot(e, R3, R3);
}

/**
 * CheckStacks del interprete tras la instruccion en 'old_pc', con el pc ya
 * calculado en r0 y SIN pila de LOOP (solo en programas sin LOOP ni BREAK):
 * hasta cuatro CALL que acaban en old_pc + 1 (los tres primeros vuelven, el
 * cuarto no), y despues el IF de arriba si su else es old_pc + 1.
 */
void EmitCheckStacks(Emitter& e, const FlowLayout& layout, u32 old_pc) {
    std::vector<u32> calls_done;
    e.Mov32(R1, old_pc + 1); // next_program_counter
    for (u32 i = 0; i < 4; i++) {
        e.Ldr(R2, kFlow, layout.call_count);
        e.Emit(0xE3500000u | (R2 << 16)); // CMP r2, #0
        calls_done.push_back(e.BranchPlaceholder(kEq));
        EmitRingBack(e, layout.call_head, 4);
        e.Ldr(R12, R3, layout.call_items);      // end_address
        e.Emit(0xE1500000u | (R12 << 16) | R1); // CMP r12, r1
        calls_done.push_back(e.BranchPlaceholder(kNe));
        if (i < 3) {
            e.Ldr(R0, R3, layout.call_items + 4); // return_address
            e.MovReg(R1, R0);
        }
        EmitSubImm(e, R2, R2, 1);
        e.Str(R2, kFlow, layout.call_count);
    }
    for (const u32 index : calls_done) {
        e.Patch(index, e.Position());
    }
    e.Ldr(R2, kFlow, layout.if_count);
    e.Emit(0xE3500000u | (R2 << 16)); // CMP r2, #0
    const u32 if_done = e.BranchPlaceholder(kEq);
    EmitRingBack(e, layout.if_head, 8);
    e.Ldr(R12, R3, layout.if_items); // else_address
    e.Mov32(R10, old_pc + 1);
    e.Emit(0xE1500000u | (R12 << 16) | R10); // CMP r12, r10
    const u32 if_miss = e.BranchPlaceholder(kNe);
    e.Ldr(R0, R3, layout.if_items + 4); // end_address
    EmitSubImm(e, R2, R2, 1);
    e.Str(R2, kFlow, layout.if_count);
    e.Patch(if_done, e.Position());
    e.Patch(if_miss, e.Position());
}

/**
 * IF, CALL, saltos, NOP y END en nativo, con sus pilas (0.2.1.5). En Pokemon
 * Sol el control de flujo era ~30 % del sombreado de un vertice: unas 40
 * llamadas a FlowStep/FlowPostCheck por vertice ("vs flujo" de 0.2.1.4: if
 * 20 %, call 25 %, comprobaciones tras tramo 38 %). Copia de FlowStep: deja el
 * pc siguiente en r0 y hace las comprobaciones de pila de detras si old_pc + 1
 * puede ser el final de algo. False si no sabe (y se llama a FlowStep).
 */
bool EmitFlowWithStacks(Emitter& e, const FlowLayout& layout, u32 a, u32 word,
                        bool next_is_stack_address, std::vector<u32>& to_epilogue) {
    Instruction instr{};
    instr.hex = word;
    const u32 dest = instr.flow_control.dest_offset;
    const u32 num = instr.flow_control.num_instructions;
    std::vector<u32> to_join;
    switch (instr.opcode.Value()) {
    case OpCode::Id::END:
        EmitCountFast(e, 1);
        to_epilogue.push_back(e.BranchPlaceholder(kAl));
        return true;
    case OpCode::Id::NOP:
        EmitCountFast(e, 1);
        e.Mov32(R0, a + 1);
        break;
    case OpCode::Id::JMPU:
    case OpCode::Id::JMPC:
        EmitCountFast(e, 1);
        if (instr.opcode.Value() == OpCode::Id::JMPU) {
            EmitBoolTest(e, instr.flow_control.bool_uniform_id, (num & 1) != 0 ? 0u : 1u);
        } else if (!EmitCondition(e, instr)) {
            return false;
        }
        // MOVW/MOVT no tocan los flags.
        e.Mov32(R0, a + 1);
        e.Mov32(R0, dest, kEq);
        break;
    case OpCode::Id::IFU:
    case OpCode::Id::IFC: {
        EmitCountFast(e, 1);
        if (instr.opcode.Value() == OpCode::Id::IFU) {
            EmitBoolTest(e, instr.flow_control.bool_uniform_id, 1);
        } else if (!EmitCondition(e, instr)) {
            return false;
        }
        const u32 to_false = e.BranchPlaceholder(kNe);
        EmitRingPush(e, layout.if_items, layout.if_head, layout.if_count, 8, dest, dest + num);
        e.Mov32(R0, a + 1);
        to_join.push_back(e.BranchPlaceholder(kAl));
        e.Patch(to_false, e.Position());
        e.Mov32(R0, dest);
        break;
    }
    case OpCode::Id::CALL:
    case OpCode::Id::CALLU:
    case OpCode::Id::CALLC: {
        EmitCountFast(e, 1);
        u32 to_skip = 0xFFFFFFFFu;
        if (instr.opcode.Value() == OpCode::Id::CALLU) {
            EmitBoolTest(e, instr.flow_control.bool_uniform_id, 1);
            to_skip = e.BranchPlaceholder(kNe);
        } else if (instr.opcode.Value() == OpCode::Id::CALLC) {
            if (!EmitCondition(e, instr)) {
                return false;
            }
            to_skip = e.BranchPlaceholder(kNe);
        }
        EmitRingPush(e, layout.call_items, layout.call_head, layout.call_count, 4, dest + num,
                     a + 1);
        e.Mov32(R0, dest);
        if (to_skip != 0xFFFFFFFFu) {
            to_join.push_back(e.BranchPlaceholder(kAl));
            e.Patch(to_skip, e.Position());
            e.Mov32(R0, a + 1);
        }
        break;
    }
    default:
        return false;
    }
    for (const u32 index : to_join) {
        e.Patch(index, e.Position());
    }
    if (next_is_stack_address) {
        EmitCheckStacks(e, layout, a);
    }
    return true;
}

void CompileWhole(Program& program, const ProgramCode& code) {
    program.whole = nullptr;
    if (!g_neon_jit.load(std::memory_order_relaxed) || !EnsureMemory()) {
        return;
    }
    constexpr u32 kMax = MAX_PROGRAM_CODE_LENGTH;
    constexpr u32 kLast = kMax - 1;

    // Hasta donde hay programa: la ultima palabra distinta de cero.
    u32 last = 0;
    for (u32 a = 0; a < kMax; a++) {
        if (code[a] != 0) {
            last = a;
        }
    }
    if (program.entry > last) {
        return;
    }
    // Que direcciones estan dentro de un tramo, y cuales pueden ser el final
    // de algo que haya en una pila (solo ahi hace falta comprobar tras un
    // tramo: ver FlowPostCheck).
    std::vector<bool> covered(kMax + 2, false);
    for (const Run& run : program.runs) {
        for (u32 a = run.end - run.count; a < run.end; a++) {
            covered[a] = true;
        }
    }
    std::vector<bool> stack_address(kMax + 2, false);
    // Sin LOOP ni BREAK, el flujo entero va en nativo (EmitFlowWithStacks).
    bool native_stacks = true;
    for (u32 a = 0; a <= last; a++) {
        if (covered[a]) {
            continue;
        }
        if (a != kLast && !IsFlowInstruction(code[a])) {
            // Una aritmetica que la ruta rapida no decodifica (DST, LIT...):
            // este programa no se compila entero.
            return;
        }
        Instruction instr{};
        instr.hex = code[a];
        const u32 dest = instr.flow_control.dest_offset;
        const u32 num = instr.flow_control.num_instructions;
        switch (instr.opcode.Value()) {
        case OpCode::Id::CALL:
        case OpCode::Id::CALLU:
        case OpCode::Id::CALLC:
            stack_address[std::min(dest + num, kMax + 1)] = true;
            break;
        case OpCode::Id::IFU:
        case OpCode::Id::IFC:
            stack_address[std::min(dest, kMax + 1)] = true;
            stack_address[std::min(dest + num, kMax + 1)] = true;
            break;
        case OpCode::Id::LOOP:
            stack_address[std::min(dest + 1, kMax + 1)] = true;
            native_stacks = false;
            break;
        case OpCode::Id::BREAK:
        case OpCode::Id::BREAKC:
            native_stacks = false;
            break;
        default:
            break;
        }
    }
    const FlowLayout layout = GetFlowLayout();

    const std::lock_guard vm_lock{Common::vita_vm_domain_mutex};
    if (sceKernelOpenVMDomain() < 0) {
        return;
    }
    u32* const start = g_code + g_used;
    Emitter e{start, kCodeBytes / 4 - g_used};
    std::vector<u32> label(kMax, 0xFFFFFFFFu);
    std::vector<u32> to_epilogue;
    std::vector<u32> to_trap;
    std::vector<std::pair<u32, u32>> to_label;

    e.Emit(0xE92D47F0u); // PUSH {r4-r10, lr}
    e.MovReg(kFlow, R2);
    // MOVW/MOVT kTable, tabla: dos palabras que se parchean abajo, cuando la
    // tabla ya existe.
    const u32 table_mov = e.Position();
    e.Emit(0xE320F000u); // NOP
    e.Emit(0xE320F000u); // NOP
    EmitPrologue(e);
    e.Mov32(R0, program.entry);
    e.Emit(0xE790F100u | (kTable << 16) | R0); // LDR pc, [kTable, r0, LSL #2]

    // Tras una llamada a FlowStep/FlowPostCheck (r0 = pc siguiente): si es la
    // direccion que viene justo detras, se sigue; si es el fin, al epilogo; si
    // no, por la tabla.
    const auto dispatch = [&](u32 expected) {
        EmitLoadConstants(e);
        e.Mov32(R12, expected);
        e.Emit(0xE1500000u | (R0 << 16) | R12); // CMP r0, r12
        const u32 skip = e.BranchPlaceholder(kEq);
        e.Emit(0xE3700001u | (R0 << 16)); // CMN r0, #1
        to_epilogue.push_back(e.BranchPlaceholder(kEq));
        e.Mov32(R12, kMax);
        e.Emit(0xE1500000u | (R0 << 16) | R12); // CMP r0, r12
        to_trap.push_back(e.BranchPlaceholder(kHs));
        e.Emit(0xE790F100u | (kTable << 16) | R0); // LDR pc, [kTable, r0, LSL #2]
        e.Patch(skip, e.Position());
    };
    // Lo mismo tras el flujo nativo, que deja el pc en r0, no toca NEON y no
    // termina el programa (END salta al epilogo por su cuenta).
    const auto native_dispatch = [&](u32 expected) {
        e.Mov32(R12, expected);
        e.Emit(0xE1500000u | (R0 << 16) | R12); // CMP r0, r12
        const u32 skip = e.BranchPlaceholder(kEq);
        e.Mov32(R12, kMax);
        e.Emit(0xE1500000u | (R0 << 16) | R12); // CMP r0, r12
        to_trap.push_back(e.BranchPlaceholder(kHs));
        e.Emit(0xE790F100u | (kTable << 16) | R0); // LDR pc, [kTable, r0, LSL #2]
        e.Patch(skip, e.Position());
    };

    u32 a = 0;
    while (a <= last) {
        const u16 run_id = program.run_at[a];
        if (run_id != kNoRun) {
            const Run& run = program.runs[run_id];
            label[a] = e.Position();
            for (u32 i = 0; i < run.count; i++) {
                EmitOp(e, program.ops[run.first_op + i]);
            }
            EmitCountFast(e, run.count);
            if (stack_address[run.end] && native_stacks) {
                e.Mov32(R0, run.end);
                EmitCheckStacks(e, layout, run.end - 1);
                native_dispatch(run.end);
            } else if (stack_address[run.end]) {
                e.MovReg(R0, kFlow);
                e.Mov32(R1, run.end - 1);
                e.Call(reinterpret_cast<const void*>(&FlowPostCheck));
                dispatch(run.end);
            }
            a = run.end;
            continue;
        }
        if (covered[a]) {
            a++;
            continue;
        }
        label[a] = e.Position();
        if (a != kLast && native_stacks) {
            const u32 before = e.Position();
            const std::size_t epilogue_jumps = to_epilogue.size();
            if (EmitFlowWithStacks(e, layout, a, code[a], stack_address[a + 1], to_epilogue)) {
                native_dispatch(a + 1);
                a++;
                continue;
            }
            // Condicion desconocida (no deberia pasar): se descarta y FlowStep.
            e.Rewind(before);
            to_epilogue.resize(epilogue_jumps);
        }
        if (a != kLast) {
            Instruction flow{};
            flow.hex = code[a];
            const u32 dest = flow.flow_control.dest_offset;
            const bool target_ok =
                dest <= last && (program.run_at[dest] != kNoRun || !covered[dest]);
            if (EmitNativeFlow(e, code[a], stack_address[a + 1], target_ok, to_label,
                               to_epilogue)) {
                a++;
                continue;
            }
        }
        e.MovReg(R0, kFlow);
        e.Mov32(R1, a);
        e.Call(reinterpret_cast<const void*>(&FlowStep));
        dispatch(a + 1);
        a++;
    }
    // Fuera del programa compilado: trampa (no deberia pasar).
    const u32 trap = e.Position();
    e.MovReg(R0, kFlow);
    e.Mov32(R1, 0);
    e.Call(reinterpret_cast<const void*>(&FlowTrap));
    const u32 epilogue = e.Position();
    e.Emit(0xE8BD87F0u); // POP {r4-r10, pc}

    if (e.Overflowed()) {
        sceKernelCloseVMDomain();
        return;
    }
    for (const u32 index : to_epilogue) {
        e.Patch(index, epilogue);
    }
    for (const u32 index : to_trap) {
        e.Patch(index, trap);
    }
    for (const auto& [index, dest] : to_label) {
        e.Patch(index, label[dest] != 0xFFFFFFFFu ? label[dest] : trap);
    }
    program.jump_table.assign(kMax, e.Address(trap));
    for (u32 i = 0; i < kMax; i++) {
        if (label[i] != 0xFFFFFFFFu) {
            program.jump_table[i] = e.Address(label[i]);
        }
    }
    // La direccion de la tabla en el MOVW/MOVT del principio (dos palabras).
    const u32 table = static_cast<u32>(reinterpret_cast<uintptr_t>(program.jump_table.data()));
    start[table_mov] = 0xE3000000u | ((table & 0xF000u) << 4) | (kTable << 12) |
                           (table & 0x0FFFu);
    start[table_mov + 1] = 0xE3400000u | (((table >> 16) & 0xF000u) << 4) | (kTable << 12) |
                       ((table >> 16) & 0x0FFFu);
    sceKernelSyncVMDomain(g_block, start, e.Position() * 4);
    sceKernelCloseVMDomain();
    g_used += e.Position();
    program.whole = reinterpret_cast<ProgramFn>(start);
}

} // namespace Pica::Shader::Fast

#endif // __PSVITA__
