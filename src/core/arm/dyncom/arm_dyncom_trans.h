#pragma once
#ifdef _MSC_VER
// nonstandard extension used: zero-sized array in struct/union
#pragma warning(disable : 4200)
#endif

#include <cstddef>
#include "common/common_types.h"

struct ARMul_State;
typedef unsigned int (*shtop_fp_t)(ARMul_State* cpu, unsigned int sht_oper);

enum class TransExtData {
    COND = (1 << 0),
    NON_BRANCH = (1 << 1),
    DIRECT_BRANCH = (1 << 2),
    INDIRECT_BRANCH = (1 << 3),
    CALL = (1 << 4),
    RET = (1 << 5),
    END_OF_PAGE = (1 << 6),
    THUMB = (1 << 7),
    SINGLE_STEP = (1 << 8)
};

struct arm_inst {
    unsigned int idx;
    unsigned int cond;
    TransExtData br;
    char component[0];
};

/**
 * Cuanto ocupa de verdad una instruccion traducida dentro de trans_cache_buf.
 *
 * Esta formula la usan DOS sitios que tienen que coincidir al byte:
 *
 *   - AllocBuffer (arm_dyncom_trans.cpp), que reparte el buffer al traducir.
 *   - INC_PC (arm_dyncom_interpreter.cpp), que avanza por el buffer al ejecutar.
 *
 * El redondeo a 4 hace falta porque lo que se reparte se usa como arm_inst*, y
 * el compilador da por hecho que un unsigned int esta alineado: en ARM junta
 * lecturas de campos contiguos en ldm/ldrd, y esas fallan con direccion
 * desalineada. Y hace falta redondear porque cuatro structs de operandos
 * (clrex_inst, ldc_inst, pld_inst, stc_inst) estan declaradas vacias, y en C++
 * una struct vacia mide 1, no 0.
 *
 * Que las dos formulas vivan aqui juntas no es cosmetico: si una redondea y la
 * otra no, el interprete avanza 13 donde el traductor reservo 16, se mete en
 * mitad de la siguiente instruccion, lee un idx que no existe y el salto
 * calculado (goto *InstLabel[idx]) se va a cualquier sitio -- normalmente a la
 * direccion 0, que en la Vita mata el proceso al instante y sin rastro.
 */
constexpr std::size_t AlignTransSize(std::size_t size) {
    return (size + 3u) & ~static_cast<std::size_t>(3u);
}

/// Paso entre una instruccion traducida y la siguiente, dado el tamano de su
/// struct de operandos.
constexpr std::size_t TransInstStride(std::size_t operand_size) {
    return AlignTransSize(sizeof(arm_inst) + operand_size);
}

struct generic_arm_inst {
    u32 Ra;
    u32 Rm;
    u32 Rn;
    u32 Rd;
    u8 op1;
    u8 op2;
};

struct adc_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct add_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct orr_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct and_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct eor_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct bbl_inst {
    unsigned int L;
    int signed_immed_24;
    unsigned int next_addr;
    unsigned int jmp_addr;
};

struct bx_inst {
    unsigned int Rm;
};

struct blx_inst {
    union {
        s32 signed_immed_24;
        u32 Rm;
    } val;
    unsigned int inst;
};

struct clz_inst {
    unsigned int Rm;
    unsigned int Rd;
};

struct cps_inst {
    unsigned int imod0;
    unsigned int imod1;
    unsigned int mmod;
    unsigned int A, I, F;
    unsigned int mode;
};

struct clrex_inst {};

struct cpy_inst {
    unsigned int Rm;
    unsigned int Rd;
};

struct bic_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct sub_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct tst_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct cmn_inst {
    unsigned int I;
    unsigned int Rn;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct teq_inst {
    unsigned int I;
    unsigned int Rn;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct stm_inst {
    unsigned int inst;
};

struct bkpt_inst {
    u32 imm;
};

struct stc_inst {};

struct ldc_inst {};

struct swi_inst {
    unsigned int num;
};

struct cmp_inst {
    unsigned int I;
    unsigned int Rn;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct mov_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct mvn_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct rev_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int op1;
    unsigned int op2;
};

struct rsb_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct rsc_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct sbc_inst {
    unsigned int I;
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int shifter_operand;
    shtop_fp_t shtop_func;
};

struct mul_inst {
    unsigned int S;
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rm;
};

struct smul_inst {
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rm;
    unsigned int x;
    unsigned int y;
};

struct umull_inst {
    unsigned int S;
    unsigned int RdHi;
    unsigned int RdLo;
    unsigned int Rs;
    unsigned int Rm;
};

struct smlad_inst {
    unsigned int m;
    unsigned int Rm;
    unsigned int Rd;
    unsigned int Ra;
    unsigned int Rn;
    unsigned int op1;
    unsigned int op2;
};

struct smla_inst {
    unsigned int x;
    unsigned int y;
    unsigned int Rm;
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rn;
};

struct smlalxy_inst {
    unsigned int x;
    unsigned int y;
    unsigned int RdLo;
    unsigned int RdHi;
    unsigned int Rm;
    unsigned int Rn;
};

struct ssat_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int imm5;
    unsigned int sat_imm;
    unsigned int shift_type;
};

struct umaal_inst {
    unsigned int Rn;
    unsigned int Rm;
    unsigned int RdHi;
    unsigned int RdLo;
};

struct umlal_inst {
    unsigned int S;
    unsigned int Rm;
    unsigned int Rs;
    unsigned int RdHi;
    unsigned int RdLo;
};

struct smlal_inst {
    unsigned int S;
    unsigned int Rm;
    unsigned int Rs;
    unsigned int RdHi;
    unsigned int RdLo;
};

struct smlald_inst {
    unsigned int RdLo;
    unsigned int RdHi;
    unsigned int Rm;
    unsigned int Rn;
    unsigned int swap;
    unsigned int op1;
    unsigned int op2;
};

struct mla_inst {
    unsigned int S;
    unsigned int Rn;
    unsigned int Rd;
    unsigned int Rs;
    unsigned int Rm;
};

struct mrc_inst {
    unsigned int opcode_1;
    unsigned int opcode_2;
    unsigned int cp_num;
    unsigned int crn;
    unsigned int crm;
    unsigned int Rd;
    unsigned int inst;
};

struct mcr_inst {
    unsigned int opcode_1;
    unsigned int opcode_2;
    unsigned int cp_num;
    unsigned int crn;
    unsigned int crm;
    unsigned int Rd;
    unsigned int inst;
};

struct mcrr_inst {
    unsigned int opcode_1;
    unsigned int cp_num;
    unsigned int crm;
    unsigned int rt;
    unsigned int rt2;
};

struct mrs_inst {
    unsigned int R;
    unsigned int Rd;
};

struct msr_inst {
    unsigned int field_mask;
    unsigned int R;
    unsigned int inst;
};

struct pld_inst {};

struct sxtb_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct sxtab_inst {
    unsigned int Rd;
    unsigned int Rn;
    unsigned int Rm;
    unsigned rotate;
};

struct sxtah_inst {
    unsigned int Rd;
    unsigned int Rn;
    unsigned int Rm;
    unsigned int rotate;
};

struct sxth_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct uxtab_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int rotate;
    unsigned int Rm;
};

struct uxtah_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int rotate;
    unsigned int Rm;
};

struct uxth_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct cdp_inst {
    unsigned int opcode_1;
    unsigned int CRn;
    unsigned int CRd;
    unsigned int cp_num;
    unsigned int opcode_2;
    unsigned int CRm;
    unsigned int inst;
};

struct uxtb_inst {
    unsigned int Rd;
    unsigned int Rm;
    unsigned int rotate;
};

struct swp_inst {
    unsigned int Rn;
    unsigned int Rd;
    unsigned int Rm;
};

struct setend_inst {
    unsigned int set_bigend;
};

struct b_2_thumb {
    unsigned int imm;
};
struct b_cond_thumb {
    unsigned int imm;
    unsigned int cond;
};

struct bl_1_thumb {
    unsigned int imm;
};
struct bl_2_thumb {
    unsigned int imm;
};
struct blx_1_thumb {
    unsigned int imm;
    unsigned int instr;
};

struct pkh_inst {
    unsigned int Rm;
    unsigned int Rn;
    unsigned int Rd;
    unsigned char imm;
};

// Floating point VFPv3 structures
#define VFP_INTERPRETER_STRUCT
#include "core/arm/skyeye_common/vfp/vfpinstr.cpp"
#undef VFP_INTERPRETER_STRUCT

typedef void (*get_addr_fp_t)(ARMul_State* cpu, unsigned int inst, unsigned int& virt_addr);

struct ldst_inst {
    unsigned int inst;
    get_addr_fp_t get_addr;
};

typedef arm_inst* ARM_INST_PTR;
typedef ARM_INST_PTR (*transop_fp_t)(unsigned int, int);

extern const transop_fp_t arm_instruction_trans[];
extern const std::size_t arm_instruction_trans_len;

// Cache de traduccion del interprete: un unico buffer estatico que nunca se
// recicla (cuando se llena, el ASSERT de arm_dyncom_trans.cpp aborta). En
// escritorio se reservan 125 MB y sobra memoria, pero en PS Vita ese buffer va
// al BSS del ejecutable y se suma al heap: el total pasaba de los ~256 MB de
// presupuesto de una aplicacion y la consola se negaba a lanzarla.
//
// Se ajusta con -DAZAHAR_TRANS_CACHE_MB=N al configurar. Subirlo permite
// juegos con mas codigo distinto; bajarlo deja mas sitio para el heap.
//
// Con el buffer cinco veces mas pequeno que en escritorio, llenarlo deja de ser
// una rareza y pasa a ser cuestion de tiempo: un juego real traduce codigo nuevo
// sin parar durante el arranque. Por eso se vacia solo al acercarse al limite
// (ver FlushTransCacheIfNeeded) en vez de esperar al ASSERT.
//
// TRANS_CACHE_MARGIN es el hueco reservado entre dos comprobaciones. La
// comprobacion se hace una vez por rodaja de tiempo, no por bloque, para no
// tocar el bucle de despacho del interprete, asi que el margen tiene que cubrir
// todo lo que una rodaja entera pueda llegar a traducir: unas 20.000
// instrucciones nuevas en el peor caso, con su struct de operandos cada una.
#ifdef __PSVITA__
#ifndef AZAHAR_TRANS_CACHE_MB
#define AZAHAR_TRANS_CACHE_MB 24
#endif
#define TRANS_CACHE_SIZE (AZAHAR_TRANS_CACHE_MB * 1024 * 1024)
#else
#define TRANS_CACHE_SIZE (64 * 1024 * 2000)
#endif
#define TRANS_CACHE_MARGIN (4 * 1024 * 1024)
alignas(4) extern char trans_cache_buf[TRANS_CACHE_SIZE];
extern std::size_t trans_cache_buf_top;

/// Apunta un nucleo para poder vaciarle la cache cuando se reinicie el buffer.
void RegisterTransCacheUser(ARMul_State* cpu);
void UnregisterTransCacheUser(ARMul_State* cpu);

/// Vacia el buffer y el indice de TODOS los nucleos. Ver FlushTransCacheIfNeeded
/// para por que tienen que ser todos y no solo el que llama.
void ResetTransCache();

/// Igual que ResetTransCache(), pero cuenta el vaciado como debido a una
/// invalidacion explicita (cambio de tabla de paginas, escritura en codigo,
/// etc.) y no a que el buffer se haya llenado. Usar esta desde
/// ARM_DynCom::ClearInstructionCache(); FlushTransCacheIfNeeded ya lleva su
/// propia cuenta por separado.
void ResetTransCacheFromInvalidation();

/// Lo mismo para un tramo: el JIT solo tira los bloques que lo tocan (0.2.0.5).
void ResetTransCacheFromRange(u32 start, std::size_t size);

/// Cuantas veces se ha vaciado el buffer por cada motivo, desde que arranco el
/// proceso. Sirve para decidir si agrandar TRANS_CACHE_SIZE serviria de algo:
/// solo los vaciados "por_capacidad" tienen que ver con el tamano del buffer.
/// Los "por_invalidacion" (cambios de proceso, principalmente) pasarian igual
/// con un buffer infinito.
void GetTransCacheFlushCounts(std::size_t& by_capacity, std::size_t& by_invalidation);

/**
 * Reinicia el buffer de traduccion si queda menos de TRANS_CACHE_MARGIN libre.
 *
 * Ojo con el detalle que hace falta acertar: el 3DS tiene dos nucleos ARM11 y
 * Azahar emula los dos, pero el buffer de traduccion es UNO solo y compartido.
 * Cada nucleo guarda aparte su indice (instruction_cache) de posiciones dentro
 * de ese buffer. Vaciar el indice de un nucleo y poner el buffer a cero dejaria
 * al otro con un indice apuntando a memoria que se va a sobrescribir, y en
 * cuanto le tocara el turno ejecutaria basura. Por eso se vacian los indices de
 * todos los nucleos a la vez.
 *
 * Es seguro hacerlo desde dentro del bucle del interprete porque Citra ejecuta
 * los nucleos por turnos en el mismo hilo: cuando uno esta en DISPATCH, el otro
 * no esta dentro de InterpreterMainLoop y no tiene punteros vivos al buffer.
 *
 * @return true si se ha vaciado, para que quien llame sepa que su propio indice
 *         ya no vale.
 */
bool FlushTransCacheIfNeeded();
