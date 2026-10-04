/*
    vfp/vfpsingle.c - ARM VFPv3 emulation unit - SoftFloat single instruction
    Copyright (C) 2003 Skyeye Develop Group
    for help please send mail to <skyeye-developer@lists.gro.clinux.org>

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
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

/*
 * This code is derived in part from :
 * - Android kernel
 * - John R. Housers softfloat library, which
 * carries the following notice:
 *
 * ===========================================================================
 * This C source file is part of the SoftFloat IEC/IEEE Floating-point
 * Arithmetic Package, Release 2.
 *
 * Written by John R. Hauser.  This work was made possible in part by the
 * International Computer Science Institute, located at Suite 600, 1947 Center
 * Street, Berkeley, California 94704.  Funding was partially provided by the
 * National Science Foundation under grant MIP-9311980.  The original version
 * of this code was written as part of a project to build a fixed-point vector
 * processor in collaboration with the University of California at Berkeley,
 * overseen by Profs. Nelson Morgan and John Wawrzynek.  More information
 * is available through the web page `http://HTTP.CS.Berkeley.EDU/~jhauser/
 * arithmetic/softfloat.html'.
 *
 * THIS SOFTWARE IS DISTRIBUTED AS IS, FOR FREE.  Although reasonable effort
 * has been made to avoid it, THIS SOFTWARE MAY CONTAIN FAULTS THAT WILL AT
 * TIMES RESULT IN INCORRECT BEHAVIOR.  USE OF THIS SOFTWARE IS RESTRICTED TO
 * PERSONS AND ORGANIZATIONS WHO CAN AND WILL TAKE FULL RESPONSIBILITY FOR ANY
 * AND ALL LOSSES, COSTS, OR OTHER PROBLEMS ARISING FROM ITS USE.
 *
 * Derivative works are acceptable, even for commercial purposes, so long as
 * (1) they include prominent notice that the work is derivative, and (2) they
 * include prominent notice akin to these three paragraphs for those parts of
 * this code that are retained.
 * ===========================================================================
 */

#include <algorithm>
#include "common/assert.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "core/arm/skyeye_common/vfp/asm_vfp.h"
#include "core/arm/skyeye_common/vfp/vfp.h"
#include "core/arm/skyeye_common/vfp/vfp_fastpath.h"
#include "core/arm/skyeye_common/vfp/vfp_helper.h"

/**
 * Camino rapido de las operaciones binarias de precision simple.
 *
 * El porque completo esta en vfp_fastpath.h. Aqui solo queda la mecanica: si
 * las condiciones se cumplen se hace la cuenta con la VFP del anfitrion y se
 * escribe el registro; si no, se devuelve false y el llamante sigue por el
 * camino de softfloat de siempre, que no se ha tocado.
 *
 * 'n' y 'm' son los operandos ya en bits, tal y como los tiene el invitado.
 * 'op' recibe los dos float y devuelve el resultado.
 *
 * OJO AL ORDEN de las comprobaciones: el resultado se valida DESPUES de
 * calcularlo, asi que una operacion que se acabe rechazando se ha hecho dos
 * veces (nativa y luego softfloat). Sale a cuenta igualmente porque rechazar es
 * raro, pero por eso se descartan antes por los operandos todos los casos que
 * se puede: cuanto antes se rechace, menos se tira.
 */
template <typename F>
static inline bool vfp_single_fast_binop(ARMul_State* state, int sd, s32 n, s32 m, u32 fpscr,
                                         u32* exceptions, F op) {
    if (!VfpFast::Usable(fpscr)) {
        return false;
    }
    const u32 n_bits = static_cast<u32>(n);
    const u32 m_bits = static_cast<u32>(m);
    if (!VfpFast::SingleIsNormal(n_bits) || !VfpFast::SingleIsNormal(m_bits)) {
        return false;
    }

    const float result = op(VfpFast::BitsToSingle(n_bits), VfpFast::BitsToSingle(m_bits));
    const u32 result_bits = VfpFast::SingleToBits(result);
    if (!VfpFast::SingleIsNormal(result_bits)) {
        // Desbordamiento, agotamiento, o el anfitrion ha aplastado un subnormal.
        // Se deshace lo andado: softfloat lo hara bien, incluidos los bits de
        // excepcion que toquen.
        VfpFast::DiscardHostExceptions();
        return false;
    }

    if (!VfpFast::FinishHostOp(exceptions)) {
        return false;
    }
    vfp_put_float(state, static_cast<s32>(result_bits), sd);
    return true;
}

static struct vfp_single vfp_single_default_qnan = {
    255,
    0,
    VFP_SINGLE_SIGNIFICAND_QNAN,
};

static void vfp_single_dump(const char* str, struct vfp_single* s) {
    LOG_TRACE(Core_ARM11, "{}: sign={} exponent={} significand={:08x}", str, s->sign != 0,
              s->exponent, s->significand);
}

static void vfp_single_normalise_denormal(struct vfp_single* vs) {
    int bits = 31 - fls(vs->significand);

    vfp_single_dump("normalise_denormal: in", vs);

    if (bits) {
        vs->exponent -= bits - 1;
        vs->significand <<= bits;
    }

    vfp_single_dump("normalise_denormal: out", vs);
}

/// El cuerpo de vfp_single_normaliseround sin escribir el registro: deja el
/// valor redondeado y empaquetado en *packed (0.1.8.1, ver la VMLA).
static u32 vfp_single_round_pack(struct vfp_single* vs, u32 fpscr, u32 exceptions,
                                 s32* packed) {
    u32 significand, incr, rmode;
    int exponent, shift, underflow;

    vfp_single_dump("pack: in", vs);

    /*
     * Infinities and NaNs are a special case.
     */
    if (vs->exponent == 255 && (vs->significand == 0 || exceptions))
        goto pack;

    /*
     * Special-case zero.
     */
    if (vs->significand == 0) {
        vs->exponent = 0;
        goto pack;
    }

    exponent = vs->exponent;
    significand = vs->significand;

    /*
     * Normalise first.  Note that we shift the significand up to
     * bit 31, so we have VFP_SINGLE_LOW_BITS + 1 below the least
     * significant bit.
     */
    shift = 32 - fls(significand);
    if (shift < 32 && shift) {
        exponent -= shift;
        significand <<= shift;
    }

#if 1
    vs->exponent = exponent;
    vs->significand = significand;
    vfp_single_dump("pack: normalised", vs);
#endif

    /*
     * Tiny number?
     */
    underflow = exponent < 0;
    if (underflow) {
        significand = vfp_shiftright32jamming(significand, -exponent);
        exponent = 0;
#if 1
        vs->exponent = exponent;
        vs->significand = significand;
        vfp_single_dump("pack: tiny number", vs);
#endif
        if (!(significand & ((1 << (VFP_SINGLE_LOW_BITS + 1)) - 1)))
            underflow = 0;

        int type = vfp_single_type(vs);

        if ((type & VFP_DENORMAL) && (fpscr & FPSCR_FLUSH_TO_ZERO)) {
            // Flush denormal to positive 0
            significand = 0;

            vs->sign = 0;
            vs->significand = significand;

            underflow = 0;
            exceptions |= FPSCR_UFC;
        }
    }

    /*
     * Select rounding increment.
     */
    incr = 0;
    rmode = fpscr & FPSCR_RMODE_MASK;

    if (rmode == FPSCR_ROUND_NEAREST) {
        incr = 1 << VFP_SINGLE_LOW_BITS;
        if ((significand & (1 << (VFP_SINGLE_LOW_BITS + 1))) == 0)
            incr -= 1;
    } else if (rmode == FPSCR_ROUND_TOZERO) {
        incr = 0;
    } else if ((rmode == FPSCR_ROUND_PLUSINF) ^ (vs->sign != 0))
        incr = (1 << (VFP_SINGLE_LOW_BITS + 1)) - 1;

    LOG_TRACE(Core_ARM11, "rounding increment = 0x{:08x}", incr);

    /*
     * Is our rounding going to overflow?
     */
    if ((significand + incr) < significand) {
        exponent += 1;
        significand = (significand >> 1) | (significand & 1);
        incr >>= 1;
#if 1
        vs->exponent = exponent;
        vs->significand = significand;
        vfp_single_dump("pack: overflow", vs);
#endif
    }

    /*
     * If any of the low bits (which will be shifted out of the
     * number) are non-zero, the result is inexact.
     */
    if (significand & ((1 << (VFP_SINGLE_LOW_BITS + 1)) - 1))
        exceptions |= FPSCR_IXC;

    /*
     * Do our rounding.
     */
    significand += incr;

    /*
     * Infinity?
     */
    if (exponent >= 254) {
        exceptions |= FPSCR_OFC | FPSCR_IXC;
        if (incr == 0) {
            vs->exponent = 253;
            vs->significand = 0x7fffffff;
        } else {
            vs->exponent = 255; /* infinity */
            vs->significand = 0;
        }
    } else {
        if (significand >> (VFP_SINGLE_LOW_BITS + 1) == 0)
            exponent = 0;
        if (exponent || significand > 0x80000000)
            underflow = 0;
        if (underflow)
            exceptions |= FPSCR_UFC;
        vs->exponent = exponent;
        vs->significand = significand >> 1;
    }

pack:
    vfp_single_dump("pack: final", vs);
    *packed = vfp_single_pack(vs);
    return exceptions;
}

u32 vfp_single_normaliseround(ARMul_State* state, int sd, struct vfp_single* vs, u32 fpscr,
                              u32 exceptions, const char* func) {
    s32 d;
    exceptions = vfp_single_round_pack(vs, fpscr, exceptions, &d);
    LOG_TRACE(Core_ARM11, "{}: d(s{})={:08x} exceptions={:08x}", func, sd, d, exceptions);
    vfp_put_float(state, d, sd);
    return exceptions;
}

/*
 * Propagate the NaN, setting exceptions if it is signalling.
 * 'n' is always a NaN.  'm' may be a number, NaN or infinity.
 */
static u32 vfp_propagate_nan(struct vfp_single* vsd, struct vfp_single* vsn, struct vfp_single* vsm,
                             u32 fpscr) {
    struct vfp_single* nan;
    int tn, tm = 0;

    tn = vfp_single_type(vsn);

    if (vsm)
        tm = vfp_single_type(vsm);

    if (fpscr & FPSCR_DEFAULT_NAN)
        /*
         * Default NaN mode - always returns a quiet NaN
         */
        nan = &vfp_single_default_qnan;
    else {
        /*
         * Contemporary mode - select the first signalling
         * NAN, or if neither are signalling, the first
         * quiet NAN.
         */
        if (tn == VFP_SNAN || (tm != VFP_SNAN && tn == VFP_QNAN))
            nan = vsn;
        else
            nan = vsm;
        /*
         * Make the NaN quiet.
         */
        nan->significand |= VFP_SINGLE_SIGNIFICAND_QNAN;
    }

    *vsd = *nan;

    /*
     * If one was a signalling NAN, raise invalid operation.
     */
    return tn == VFP_SNAN || tm == VFP_SNAN ? FPSCR_IOC : VFP_NAN_FLAG;
}

/*
 * Extended operations
 */
static u32 vfp_single_fabs(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    vfp_put_float(state, vfp_single_packed_abs(m), sd);
    return 0;
}

static u32 vfp_single_fcpy(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    vfp_put_float(state, m, sd);
    return 0;
}

static u32 vfp_single_fneg(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    vfp_put_float(state, vfp_single_packed_negate(m), sd);
    return 0;
}

static const u16 sqrt_oddadjust[] = {
    0x0004, 0x0022, 0x005d, 0x00b1, 0x011d, 0x019f, 0x0236, 0x02e0,
    0x039c, 0x0468, 0x0545, 0x0631, 0x072b, 0x0832, 0x0946, 0x0a67,
};

static const u16 sqrt_evenadjust[] = {
    0x0a2d, 0x08af, 0x075a, 0x0629, 0x051a, 0x0429, 0x0356, 0x029e,
    0x0200, 0x0179, 0x0109, 0x00af, 0x0068, 0x0034, 0x0012, 0x0002,
};

u32 vfp_estimate_sqrt_significand(u32 exponent, u32 significand) {
    int index;
    u32 z, a;

    if ((significand & 0xc0000000) != 0x40000000) {
        LOG_TRACE(Core_ARM11, "invalid significand");
    }

    a = significand << 1;
    index = (a >> 27) & 15;
    if (exponent & 1) {
        z = 0x4000 + (a >> 17) - sqrt_oddadjust[index];
        z = ((a / z) << 14) + (z << 15);
        a >>= 1;
    } else {
        z = 0x8000 + (a >> 17) - sqrt_evenadjust[index];
        z = a / z + z;
        z = (z >= 0x20000) ? 0xffff8000 : (z << 15);
        if (z <= a)
            return (s32)a >> 1;
    }
    {
        u64 v = (u64)a << 31;
        do_div(v, z);
        return (u32)(v + (z >> 1));
    }
}

static u32 vfp_single_fsqrt(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    struct vfp_single vsm, vsd, *vsp;
    int ret, tm;
    u32 exceptions = 0;

    /*
     * Camino rapido: VSQRT.F32 del anfitrion.
     *
     * La raiz cuadrada de IEEE 754 esta correctamente redondeada, igual que la
     * rutina de abajo, asi que con un operando normal y POSITIVO las dos dan el
     * mismo resultado bit a bit. El signo hay que mirarlo aparte de lo normal:
     * la raiz de un negativo es invalida (IOC) y la de un normal positivo nunca
     * desborda ni se agota, pero eso ultimo lo comprueba igualmente el propio
     * ayudante sobre el resultado.
     */
    if (VfpFast::Usable(fpscr)) {
        const u32 m_bits = static_cast<u32>(m);
        if (VfpFast::SingleIsNormal(m_bits) && VfpFast::SingleIsPositive(m_bits)) {
            const float root = __builtin_sqrtf(VfpFast::BitsToSingle(m_bits));
            const u32 root_bits = VfpFast::SingleToBits(root);
            u32 raised = 0;
            if (VfpFast::SingleIsNormal(root_bits) && VfpFast::FinishHostOp(&raised)) {
                vfp_put_float(state, static_cast<s32>(root_bits), sd);
                return raised;
            }
            VfpFast::DiscardHostExceptions();
        }
    }

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);
    tm = vfp_single_type(&vsm);
    if (tm & (VFP_NAN | VFP_INFINITY)) {
        vsp = &vsd;

        if (tm & VFP_NAN)
            ret = vfp_propagate_nan(vsp, &vsm, nullptr, fpscr);
        else if (vsm.sign == 0) {
        sqrt_copy:
            vsp = &vsm;
            ret = 0;
        } else {
        sqrt_invalid:
            vsp = &vfp_single_default_qnan;
            ret = FPSCR_IOC;
        }
        vfp_put_float(state, vfp_single_pack(vsp), sd);
        return ret;
    }

    /*
     * sqrt(+/- 0) == +/- 0
     */
    if (tm & VFP_ZERO)
        goto sqrt_copy;

    /*
     * Normalise a denormalised number
     */
    if (tm & VFP_DENORMAL)
        vfp_single_normalise_denormal(&vsm);

    /*
     * sqrt(<0) = invalid
     */
    if (vsm.sign)
        goto sqrt_invalid;

    vfp_single_dump("sqrt", &vsm);

    /*
     * Estimate the square root.
     */
    vsd.sign = 0;
    vsd.exponent = ((vsm.exponent - 127) >> 1) + 127;
    vsd.significand = vfp_estimate_sqrt_significand(vsm.exponent, vsm.significand) + 2;

    vfp_single_dump("sqrt estimate", &vsd);

    /*
     * And now adjust.
     */
    if ((vsd.significand & VFP_SINGLE_LOW_BITS_MASK) <= 5) {
        if (vsd.significand < 2) {
            vsd.significand = 0xffffffff;
        } else {
            u64 term;
            s64 rem;
            vsm.significand <<= static_cast<u32>((vsm.exponent & 1) == 0);
            term = (u64)vsd.significand * vsd.significand;
            rem = ((u64)vsm.significand << 32) - term;

            LOG_TRACE(Core_ARM11, "term={} rem={}", term, rem);

            while (rem < 0) {
                vsd.significand -= 1;
                rem += ((u64)vsd.significand << 1) | 1;
            }
            vsd.significand |= rem != 0;
        }
    }
    vsd.significand = vfp_shiftright32jamming(vsd.significand, 1);

    exceptions |= vfp_single_normaliseround(state, sd, &vsd, fpscr, 0, "fsqrt");

    return exceptions;
}

/*
 * Equal	:= ZC
 * Less than	:= N
 * Greater than	:= C
 * Unordered	:= CV
 */
static u32 vfp_compare(ARMul_State* state, int sd, int signal_on_qnan, s32 m, u32 fpscr) {
    s32 d;
    u32 ret = 0;

    d = vfp_get_float(state, sd);
    if (vfp_single_packed_exponent(m) == 255 && vfp_single_packed_mantissa(m)) {
        ret |= FPSCR_CFLAG | FPSCR_VFLAG;
        if (signal_on_qnan ||
            !(vfp_single_packed_mantissa(m) & (1 << (VFP_SINGLE_MANTISSA_BITS - 1))))
            /*
             * Signalling NaN, or signalling on quiet NaN
             */
            ret |= FPSCR_IOC;
    }

    if (vfp_single_packed_exponent(d) == 255 && vfp_single_packed_mantissa(d)) {
        ret |= FPSCR_CFLAG | FPSCR_VFLAG;
        if (signal_on_qnan ||
            !(vfp_single_packed_mantissa(d) & (1 << (VFP_SINGLE_MANTISSA_BITS - 1))))
            /*
             * Signalling NaN, or signalling on quiet NaN
             */
            ret |= FPSCR_IOC;
    }

    if (ret == 0) {
        if (d == m || vfp_single_packed_abs(d | m) == 0) {
            /*
             * equal
             */
            ret |= FPSCR_ZFLAG | FPSCR_CFLAG;
        } else if (vfp_single_packed_sign(d ^ m)) {
            /*
             * different signs
             */
            if (vfp_single_packed_sign(d))
                /*
                 * d is negative, so d < m
                 */
                ret |= FPSCR_NFLAG;
            else
                /*
                 * d is positive, so d > m
                 */
                ret |= FPSCR_CFLAG;
        } else if ((vfp_single_packed_sign(d) != 0) ^ (d < m)) {
            /*
             * d < m
             */
            ret |= FPSCR_NFLAG;
        } else if ((vfp_single_packed_sign(d) != 0) ^ (d > m)) {
            /*
             * d > m
             */
            ret |= FPSCR_CFLAG;
        }
    }
    return ret;
}

static u32 vfp_single_fcmp(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    return vfp_compare(state, sd, 0, m, fpscr);
}

static u32 vfp_single_fcmpe(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    return vfp_compare(state, sd, 1, m, fpscr);
}

static u32 vfp_single_fcmpz(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    return vfp_compare(state, sd, 0, 0, fpscr);
}

static u32 vfp_single_fcmpez(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    return vfp_compare(state, sd, 1, 0, fpscr);
}

static u32 vfp_single_fcvtd(ARMul_State* state, int dd, int unused, s32 m, u32 fpscr) {
    struct vfp_single vsm;
    struct vfp_double vdd;
    int tm;
    u32 exceptions = 0;

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);

    tm = vfp_single_type(&vsm);

    /*
     * If we have a signalling NaN, signal invalid operation.
     */
    if (tm == VFP_SNAN)
        exceptions |= FPSCR_IOC;

    if (tm & VFP_DENORMAL)
        vfp_single_normalise_denormal(&vsm);

    vdd.sign = vsm.sign;
    vdd.significand = (u64)vsm.significand << 32;

    /*
     * If we have an infinity or NaN, the exponent must be 2047.
     */
    if (tm & (VFP_INFINITY | VFP_NAN)) {
        vdd.exponent = 2047;
        if (tm == VFP_QNAN)
            vdd.significand |= VFP_DOUBLE_SIGNIFICAND_QNAN;
        goto pack_nan;
    } else if (tm & VFP_ZERO)
        vdd.exponent = 0;
    else
        vdd.exponent = vsm.exponent + (1023 - 127);

    return vfp_double_normaliseround(state, dd, &vdd, fpscr, exceptions, "fcvtd");

pack_nan:
    vfp_put_double(state, vfp_double_pack(&vdd), dd);
    return exceptions;
}

static u32 vfp_single_fuito(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    struct vfp_single vs;

    vs.sign = 0;
    vs.exponent = 127 + 31 - 1;
    vs.significand = (u32)m;

    return vfp_single_normaliseround(state, sd, &vs, fpscr, 0, "fuito");
}

static u32 vfp_single_fsito(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    struct vfp_single vs;

    vs.sign = (m & 0x80000000) >> 16;
    vs.exponent = 127 + 31 - 1;
    vs.significand = vs.sign ? -m : m;

    return vfp_single_normaliseround(state, sd, &vs, fpscr, 0, "fsito");
}

static u32 vfp_single_ftoui(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    struct vfp_single vsm;
    u32 d, exceptions = 0;
    int rmode = fpscr & FPSCR_RMODE_MASK;
    int tm;

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);
    vfp_single_dump("VSM", &vsm);

    /*
     * Do we have a denormalised number?
     */
    tm = vfp_single_type(&vsm);
    if (tm & VFP_DENORMAL)
        exceptions |= FPSCR_IDC;

    if (tm & VFP_NAN)
        vsm.sign = 1;

    if (vsm.exponent >= 127 + 32) {
        d = vsm.sign ? 0 : 0xffffffff;
        exceptions |= FPSCR_IOC;
    } else if (vsm.exponent >= 127) {
        int shift = 127 + 31 - vsm.exponent;
        u32 rem, incr = 0;

        /*
         * 2^0 <= m < 2^32-2^8
         */
        d = (vsm.significand << 1) >> shift;
        if (shift > 0) {
            rem = (vsm.significand << 1) << (32 - shift);
        } else {
            rem = 0;
        }

        if (rmode == FPSCR_ROUND_NEAREST) {
            incr = 0x80000000;
            if ((d & 1) == 0)
                incr -= 1;
        } else if (rmode == FPSCR_ROUND_TOZERO) {
            incr = 0;
        } else if ((rmode == FPSCR_ROUND_PLUSINF) ^ (vsm.sign != 0)) {
            incr = ~0;
        }

        if ((rem + incr) < rem) {
            if (d < 0xffffffff)
                d += 1;
            else
                exceptions |= FPSCR_IOC;
        }

        if (d && vsm.sign) {
            d = 0;
            exceptions |= FPSCR_IOC;
        } else if (rem)
            exceptions |= FPSCR_IXC;
    } else {
        d = 0;
        if (vsm.exponent | vsm.significand) {
            if (rmode == FPSCR_ROUND_NEAREST) {
                if (vsm.exponent >= 126) {
                    d = vsm.sign ? 0 : 1;
                    exceptions |= vsm.sign ? FPSCR_IOC : FPSCR_IXC;
                } else {
                    exceptions |= FPSCR_IXC;
                }
            } else if (rmode == FPSCR_ROUND_PLUSINF && vsm.sign == 0) {
                d = 1;
                exceptions |= FPSCR_IXC;
            } else if (rmode == FPSCR_ROUND_MINUSINF) {
                exceptions |= vsm.sign ? FPSCR_IOC : FPSCR_IXC;
            } else {
                exceptions |= FPSCR_IXC;
            }
        }
    }

    LOG_TRACE(Core_ARM11, "ftoui: d(s{})={:08x} exceptions={:08x}", sd, d, exceptions);

    vfp_put_float(state, d, sd);

    return exceptions;
}

static u32 vfp_single_ftouiz(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    return vfp_single_ftoui(state, sd, unused, m, (fpscr & ~FPSCR_RMODE_MASK) | FPSCR_ROUND_TOZERO);
}

static u32 vfp_single_ftosi(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    struct vfp_single vsm;
    u32 d, exceptions = 0;
    int rmode = fpscr & FPSCR_RMODE_MASK;
    int tm;

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);
    vfp_single_dump("VSM", &vsm);

    /*
     * Do we have a denormalised number?
     */
    tm = vfp_single_type(&vsm);
    if (vfp_single_type(&vsm) & VFP_DENORMAL)
        exceptions |= FPSCR_IDC;

    if (tm & VFP_NAN) {
        d = 0;
        exceptions |= FPSCR_IOC;
    } else if (vsm.exponent >= 127 + 31) {
        /*
         * m >= 2^31-2^7: invalid
         */
        d = 0x7fffffff;
        if (vsm.sign)
            d = ~d;
        exceptions |= FPSCR_IOC;
    } else if (vsm.exponent >= 127) {
        int shift = 127 + 31 - vsm.exponent;
        u32 rem, incr = 0;

        /* 2^0 <= m <= 2^31-2^7 */
        d = (vsm.significand << 1) >> shift;
        rem = (vsm.significand << 1) << (32 - shift);

        if (rmode == FPSCR_ROUND_NEAREST) {
            incr = 0x80000000;
            if ((d & 1) == 0)
                incr -= 1;
        } else if (rmode == FPSCR_ROUND_TOZERO) {
            incr = 0;
        } else if ((rmode == FPSCR_ROUND_PLUSINF) ^ (vsm.sign != 0)) {
            incr = ~0;
        }

        if ((rem + incr) < rem && d < 0xffffffff)
            d += 1;
        if (d > (0x7fffffffu + (vsm.sign != 0))) {
            d = (0x7fffffffu + (vsm.sign != 0));
            exceptions |= FPSCR_IOC;
        } else if (rem)
            exceptions |= FPSCR_IXC;

        if (vsm.sign)
            d = (~d + 1);
    } else {
        d = 0;
        if (vsm.exponent | vsm.significand) {
            exceptions |= FPSCR_IXC;
            if (rmode == FPSCR_ROUND_NEAREST) {
                if (vsm.exponent >= 126)
                    d = vsm.sign ? 0xffffffff : 1;
            } else if (rmode == FPSCR_ROUND_PLUSINF && vsm.sign == 0) {
                d = 1;
            } else if (rmode == FPSCR_ROUND_MINUSINF && vsm.sign) {
                d = 0xffffffff;
            }
        }
    }

    LOG_TRACE(Core_ARM11, "ftosi: d(s{})={:08x} exceptions={:08x}", sd, d, exceptions);

    vfp_put_float(state, (s32)d, sd);

    return exceptions;
}

static u32 vfp_single_ftosiz(ARMul_State* state, int sd, int unused, s32 m, u32 fpscr) {
    return vfp_single_ftosi(state, sd, unused, m, (fpscr & ~FPSCR_RMODE_MASK) | FPSCR_ROUND_TOZERO);
}

static struct op fops_ext[] = {
    {vfp_single_fcpy, 0},  // 0x00000000 - FEXT_FCPY
    {vfp_single_fabs, 0},  // 0x00000001 - FEXT_FABS
    {vfp_single_fneg, 0},  // 0x00000002 - FEXT_FNEG
    {vfp_single_fsqrt, 0}, // 0x00000003 - FEXT_FSQRT
    {nullptr, 0},
    {nullptr, 0},
    {nullptr, 0},
    {nullptr, 0},
    {vfp_single_fcmp, OP_SCALAR},   // 0x00000008 - FEXT_FCMP
    {vfp_single_fcmpe, OP_SCALAR},  // 0x00000009 - FEXT_FCMPE
    {vfp_single_fcmpz, OP_SCALAR},  // 0x0000000A - FEXT_FCMPZ
    {vfp_single_fcmpez, OP_SCALAR}, // 0x0000000B - FEXT_FCMPEZ
    {nullptr, 0},
    {nullptr, 0},
    {nullptr, 0},
    {vfp_single_fcvtd, OP_SCALAR | OP_DD}, // 0x0000000F - FEXT_FCVT
    {vfp_single_fuito, OP_SCALAR},         // 0x00000010 - FEXT_FUITO
    {vfp_single_fsito, OP_SCALAR},         // 0x00000011 - FEXT_FSITO
    {nullptr, 0},
    {nullptr, 0},
    {nullptr, 0},
    {nullptr, 0},
    {nullptr, 0},
    {nullptr, 0},
    {vfp_single_ftoui, OP_SCALAR},  // 0x00000018 - FEXT_FTOUI
    {vfp_single_ftouiz, OP_SCALAR}, // 0x00000019 - FEXT_FTOUIZ
    {vfp_single_ftosi, OP_SCALAR},  // 0x0000001A - FEXT_FTOSI
    {vfp_single_ftosiz, OP_SCALAR}, // 0x0000001B - FEXT_FTOSIZ
};

static u32 vfp_single_fadd_nonnumber(struct vfp_single* vsd, struct vfp_single* vsn,
                                     struct vfp_single* vsm, u32 fpscr) {
    struct vfp_single* vsp;
    u32 exceptions = 0;
    int tn, tm;

    tn = vfp_single_type(vsn);
    tm = vfp_single_type(vsm);

    if (tn & tm & VFP_INFINITY) {
        /*
         * Two infinities.  Are they different signs?
         */
        if (vsn->sign ^ vsm->sign) {
            /*
             * different signs -> invalid
             */
            exceptions |= FPSCR_IOC;
            vsp = &vfp_single_default_qnan;
        } else {
            /*
             * same signs -> valid
             */
            vsp = vsn;
        }
    } else if (tn & VFP_INFINITY && tm & VFP_NUMBER) {
        /*
         * One infinity and one number -> infinity
         */
        vsp = vsn;
    } else {
        /*
         * 'n' is a NaN of some type
         */
        return vfp_propagate_nan(vsd, vsn, vsm, fpscr);
    }
    *vsd = *vsp;
    return exceptions;
}

static u32 vfp_single_add(struct vfp_single* vsd, struct vfp_single* vsn, struct vfp_single* vsm,
                          u32 fpscr) {
    u32 exp_diff, m_sig;

    if (vsn->significand & 0x80000000 || vsm->significand & 0x80000000) {
        LOG_WARNING(Core_ARM11, "bad FP values");
        vfp_single_dump("VSN", vsn);
        vfp_single_dump("VSM", vsm);
    }

    /*
     * Ensure that 'n' is the largest magnitude number.  Note that
     * if 'n' and 'm' have equal exponents, we do not swap them.
     * This ensures that NaN propagation works correctly.
     */
    if (vsn->exponent < vsm->exponent) {
        std::swap(vsm, vsn);
    }

    /*
     * Is 'n' an infinity or a NaN?  Note that 'm' may be a number,
     * infinity or a NaN here.
     */
    if (vsn->exponent == 255)
        return vfp_single_fadd_nonnumber(vsd, vsn, vsm, fpscr);

    /*
     * We have two proper numbers, where 'vsn' is the larger magnitude.
     *
     * Copy 'n' to 'd' before doing the arithmetic.
     */
    *vsd = *vsn;

    /*
     * Align both numbers.
     */
    exp_diff = vsn->exponent - vsm->exponent;
    m_sig = vfp_shiftright32jamming(vsm->significand, exp_diff);

    /*
     * If the signs are different, we are really subtracting.
     */
    if (vsn->sign ^ vsm->sign) {
        m_sig = vsn->significand - m_sig;
        if ((s32)m_sig < 0) {
            vsd->sign = vfp_sign_negate(vsd->sign);
            m_sig = (~m_sig + 1);
        } else if (m_sig == 0) {
            vsd->sign = (fpscr & FPSCR_RMODE_MASK) == FPSCR_ROUND_MINUSINF ? 0x8000 : 0;
        }
    } else {
        m_sig = vsn->significand + m_sig;
    }
    vsd->significand = m_sig;

    return 0;
}

static u32 vfp_single_multiply(struct vfp_single* vsd, struct vfp_single* vsn,
                               struct vfp_single* vsm, u32 fpscr) {
    vfp_single_dump("VSN", vsn);
    vfp_single_dump("VSM", vsm);

    /*
     * Ensure that 'n' is the largest magnitude number.  Note that
     * if 'n' and 'm' have equal exponents, we do not swap them.
     * This ensures that NaN propagation works correctly.
     */
    if (vsn->exponent < vsm->exponent) {
        std::swap(vsm, vsn);
        LOG_TRACE(Core_ARM11, "swapping M <-> N");
    }

    vsd->sign = vsn->sign ^ vsm->sign;

    /*
     * If 'n' is an infinity or NaN, handle it.  'm' may be anything.
     */
    if (vsn->exponent == 255) {
        if (vsn->significand || (vsm->exponent == 255 && vsm->significand))
            return vfp_propagate_nan(vsd, vsn, vsm, fpscr);
        if ((vsm->exponent | vsm->significand) == 0) {
            *vsd = vfp_single_default_qnan;
            return FPSCR_IOC;
        }
        vsd->exponent = vsn->exponent;
        vsd->significand = 0;
        return 0;
    }

    /*
     * If 'm' is zero, the result is always zero.  In this case,
     * 'n' may be zero or a number, but it doesn't matter which.
     */
    if ((vsm->exponent | vsm->significand) == 0) {
        vsd->exponent = 0;
        vsd->significand = 0;
        return 0;
    }

    /*
     * We add 2 to the destination exponent for the same reason as
     * the addition case - though this time we have +1 from each
     * input operand.
     */
    vsd->exponent = vsn->exponent + vsm->exponent - 127 + 2;
    vsd->significand = vfp_hi64to32jamming((u64)vsn->significand * vsm->significand);

    vfp_single_dump("VSD", vsd);
    return 0;
}

#define NEG_MULTIPLY (1 << 0)
#define NEG_SUBTRACT (1 << 1)

/**
 * VMLA y familia por la VFP del anfitrion (0.1.8.1). Producto redondeado y
 * suma redondeada: lo mismo que el softfloat de abajo y que la VMLA nativa del
 * JIT. Las garantias son las de vfp_single_fast_binop, con el producto
 * intermedio tambien normal. La barrera de asm obliga a materializar el
 * producto redondeado: sin ella el compilador podria fundir las dos
 * operaciones en una FMA (que redondea una sola vez) si el destino la tuviera.
 */
static inline bool vfp_single_fast_mac(ARMul_State* state, int sd, s32 n, s32 m, u32 fpscr,
                                       u32 negate, u32* exceptions) {
    if (!VfpFast::Usable(fpscr)) {
        return false;
    }
    const u32 n_bits = static_cast<u32>(n);
    const u32 m_bits = static_cast<u32>(m);
    const u32 d_bits = static_cast<u32>(vfp_get_float(state, sd));
    if (!VfpFast::SingleIsNormal(n_bits) || !VfpFast::SingleIsNormal(m_bits) ||
        !VfpFast::SingleIsNormal(d_bits)) {
        return false;
    }
    float product = VfpFast::BitsToSingle(n_bits) * VfpFast::BitsToSingle(m_bits);
    __asm__("" : "+t"(product));
    if (!VfpFast::SingleIsNormal(VfpFast::SingleToBits(product))) {
        VfpFast::DiscardHostExceptions();
        return false;
    }
    if (negate & NEG_MULTIPLY) {
        product = -product;
    }
    float addend = VfpFast::BitsToSingle(d_bits);
    if (negate & NEG_SUBTRACT) {
        addend = -addend;
    }
    const float result = addend + product;
    const u32 result_bits = VfpFast::SingleToBits(result);
    if (!VfpFast::SingleIsNormal(result_bits)) {
        VfpFast::DiscardHostExceptions();
        return false;
    }
    if (!VfpFast::FinishHostOp(exceptions)) {
        return false;
    }
    vfp_put_float(state, static_cast<s32>(result_bits), sd);
    return true;
}

static u32 vfp_single_multiply_accumulate(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr,
                                          u32 negate, const char* func) {
    vfp_single vsd, vsp, vsn, vsm;
    u32 exceptions = 0;
    s32 v;

    if (vfp_single_fast_mac(state, sd, vfp_get_float(state, sn), m, fpscr, negate,
                            &exceptions)) {
        return exceptions;
    }

    v = vfp_get_float(state, sn);
    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, v);
    exceptions |= vfp_single_unpack(&vsn, v, fpscr);
    if (vsn.exponent == 0 && vsn.significand)
        vfp_single_normalise_denormal(&vsn);

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);
    if (vsm.exponent == 0 && vsm.significand)
        vfp_single_normalise_denormal(&vsm);

    exceptions |= vfp_single_multiply(&vsp, &vsn, &vsm, fpscr);

    /**
     * EL PRODUCTO SE REDONDEA ANTES DE SUMAR (0.1.8.1). Aqui se mantenia sin
     * redondear (unos 32 bits de significando con bit pegajoso) y se redondeaba
     * una sola vez al final: eso es una FMA fusionada. VMLA, VMLS, VNMLA y VNMLS
     * del VFP del ARM11 no lo son: el manual de ARMv7 las define como
     * FPAdd(D[d], FPMul(D[n], D[m])), con el producto redondeado, y asi las
     * hace dynarmic (ir.FPAdd(reg_d, ir.FPMul(reg_n, reg_m))) en el Azahar de
     * escritorio. Medido en consola: TODOS los bloques que el JIT devolvia al
     * interprete por "DIFERENCIA ... vfp" llevaban una de estas, con el JIT
     * (VMLA nativa, no fusionada) a 1-8 ULP del interprete, o 0 frente a un
     * resto de cancelacion. Los NaN e infinitos no se redondean (no hay nada
     * que redondear) y siguen el camino de siempre.
     */
    if (vsp.exponent != 255) {
        s32 product;
        exceptions = vfp_single_round_pack(&vsp, fpscr, exceptions, &product);
        exceptions |= vfp_single_unpack(&vsp, product, fpscr);
        if (vsp.exponent == 0 && vsp.significand)
            vfp_single_normalise_denormal(&vsp);
    }

    if (negate & NEG_MULTIPLY)
        vsp.sign = vfp_sign_negate(vsp.sign);

    v = vfp_get_float(state, sd);
    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sd, v);
    exceptions |= vfp_single_unpack(&vsn, v, fpscr);
    if (vsn.exponent == 0 && vsn.significand != 0)
        vfp_single_normalise_denormal(&vsn);

    if (negate & NEG_SUBTRACT)
        vsn.sign = vfp_sign_negate(vsn.sign);

    exceptions |= vfp_single_add(&vsd, &vsn, &vsp, fpscr);

    return vfp_single_normaliseround(state, sd, &vsd, fpscr, exceptions, func);
}

/*
 * Standard operations
 */

/*
 * sd = sd + (sn * sm)
 */
static u32 vfp_single_fmac(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, sd);
    return vfp_single_multiply_accumulate(state, sd, sn, m, fpscr, 0, "fmac");
}

/*
 * sd = sd - (sn * sm)
 */
static u32 vfp_single_fnmac(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    // TODO: this one has its arguments inverted, investigate.
    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sd, sn);
    return vfp_single_multiply_accumulate(state, sd, sn, m, fpscr, NEG_MULTIPLY, "fnmac");
}

/*
 * sd = -sd + (sn * sm)
 */
static u32 vfp_single_fmsc(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, sd);
    return vfp_single_multiply_accumulate(state, sd, sn, m, fpscr, NEG_SUBTRACT, "fmsc");
}

/*
 * sd = -sd - (sn * sm)
 */
static u32 vfp_single_fnmsc(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, sd);
    return vfp_single_multiply_accumulate(state, sd, sn, m, fpscr, NEG_SUBTRACT | NEG_MULTIPLY,
                                          "fnmsc");
}

/*
 * sd = sn * sm
 */
static u32 vfp_single_fmul(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    struct vfp_single vsd, vsn, vsm;
    u32 exceptions = 0;
    s32 n = vfp_get_float(state, sn);

    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, n);

    if (vfp_single_fast_binop(state, sd, n, m, fpscr, &exceptions,
                              [](float a, float b) { return a * b; })) {
        return exceptions;
    }

    exceptions |= vfp_single_unpack(&vsn, n, fpscr);
    if (vsn.exponent == 0 && vsn.significand)
        vfp_single_normalise_denormal(&vsn);

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);
    if (vsm.exponent == 0 && vsm.significand)
        vfp_single_normalise_denormal(&vsm);

    exceptions |= vfp_single_multiply(&vsd, &vsn, &vsm, fpscr);
    return vfp_single_normaliseround(state, sd, &vsd, fpscr, exceptions, "fmul");
}

/*
 * sd = -(sn * sm)
 */
static u32 vfp_single_fnmul(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    struct vfp_single vsd, vsn, vsm;
    u32 exceptions = 0;
    s32 n = vfp_get_float(state, sn);

    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, n);

    // Negar un resultado normal es exacto (solo cambia el bit de signo), asi que
    // no altera ni el redondeo ni las excepciones.
    if (vfp_single_fast_binop(state, sd, n, m, fpscr, &exceptions,
                              [](float a, float b) { return -(a * b); })) {
        return exceptions;
    }

    exceptions |= vfp_single_unpack(&vsn, n, fpscr);
    if (vsn.exponent == 0 && vsn.significand)
        vfp_single_normalise_denormal(&vsn);

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);
    if (vsm.exponent == 0 && vsm.significand)
        vfp_single_normalise_denormal(&vsm);

    exceptions |= vfp_single_multiply(&vsd, &vsn, &vsm, fpscr);
    vsd.sign = vfp_sign_negate(vsd.sign);
    return vfp_single_normaliseround(state, sd, &vsd, fpscr, exceptions, "fnmul");
}

/*
 * sd = sn + sm
 */
static u32 vfp_single_fadd(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    struct vfp_single vsd, vsn, vsm;
    u32 exceptions = 0;
    s32 n = vfp_get_float(state, sn);

    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, n);

    if (vfp_single_fast_binop(state, sd, n, m, fpscr, &exceptions,
                              [](float a, float b) { return a + b; })) {
        return exceptions;
    }

    /*
     * Unpack and normalise denormals.
     */
    exceptions |= vfp_single_unpack(&vsn, n, fpscr);
    if (vsn.exponent == 0 && vsn.significand)
        vfp_single_normalise_denormal(&vsn);

    exceptions |= vfp_single_unpack(&vsm, m, fpscr);
    if (vsm.exponent == 0 && vsm.significand)
        vfp_single_normalise_denormal(&vsm);

    exceptions |= vfp_single_add(&vsd, &vsn, &vsm, fpscr);

    return vfp_single_normaliseround(state, sd, &vsd, fpscr, exceptions, "fadd");
}

/*
 * sd = sn - sm
 */
static u32 vfp_single_fsub(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, sd);

    // El camino rapido va ANTES de todo el enredo de abajo a proposito: ese
    // enredo existe por los subnormales aplastados a cero y por el signo del
    // cero, y el camino rapido solo acepta operandos normales, donde ninguna de
    // las dos cosas puede pasar.
    {
        u32 exceptions = 0;
        const s32 n = vfp_get_float(state, sn);
        if (vfp_single_fast_binop(state, sd, n, m, fpscr, &exceptions,
                                  [](float a, float b) { return a - b; })) {
            return exceptions;
        }
    }

    /*
     * Subtraction is addition with one sign inverted. Unpack the second operand to perform FTZ if
     * necessary, we can't let fadd do this because a denormal in m might get flushed to +0 in FTZ
     * mode, and the resulting sign of 0 OP +0 differs between fadd and fsub. We do not need to do
     * this for n because +0 OP 0 is always +0 for both fadd and fsub.
     */
    struct vfp_single vsm;
    u32 exceptions = vfp_single_unpack(&vsm, m, fpscr);
    if (exceptions & FPSCR_IDC) {
        // The value was flushed to zero, re-pack it.
        m = vfp_single_pack(&vsm);
    }

    if (m != 0x7FC00000) // Only negate if m isn't NaN.
        m = vfp_single_packed_negate(m);

    return vfp_single_fadd(state, sd, sn, m, fpscr) | exceptions;
}

/*
 * sd = sn / sm
 */
static u32 vfp_single_fdiv(ARMul_State* state, int sd, int sn, s32 m, u32 fpscr) {
    struct vfp_single vsd, vsn, vsm;
    u32 exceptions = 0;
    s32 n = vfp_get_float(state, sn);
    int tm, tn;

    LOG_TRACE(Core_ARM11, "s{} = {:08x}", sn, n);

    // El divisor normal no puede ser cero, asi que aqui no hay division por cero
    // que atrapar: ese caso lo rechaza la comprobacion de operandos.
    if (vfp_single_fast_binop(state, sd, n, m, fpscr, &exceptions,
                              [](float a, float b) { return a / b; })) {
        return exceptions;
    }

    exceptions |= vfp_single_unpack(&vsn, n, fpscr);
    exceptions |= vfp_single_unpack(&vsm, m, fpscr);

    vsd.sign = vsn.sign ^ vsm.sign;

    tn = vfp_single_type(&vsn);
    tm = vfp_single_type(&vsm);

    /*
     * Is n a NAN?
     */
    if (tn & VFP_NAN)
        goto vsn_nan;

    /*
     * Is m a NAN?
     */
    if (tm & VFP_NAN)
        goto vsm_nan;

    /*
     * If n and m are infinity, the result is invalid
     * If n and m are zero, the result is invalid
     */
    if (tm & tn & (VFP_INFINITY | VFP_ZERO))
        goto invalid;

    /*
     * If n is infinity, the result is infinity
     */
    if (tn & VFP_INFINITY)
        goto infinity;

    /*
     * If m is zero, raise div0 exception
     */
    if (tm & VFP_ZERO)
        goto divzero;

    /*
     * If m is infinity, or n is zero, the result is zero
     */
    if (tm & VFP_INFINITY || tn & VFP_ZERO)
        goto zero;

    if (tn & VFP_DENORMAL)
        vfp_single_normalise_denormal(&vsn);
    if (tm & VFP_DENORMAL)
        vfp_single_normalise_denormal(&vsm);

    /*
     * Ok, we have two numbers, we can perform division.
     */
    vsd.exponent = vsn.exponent - vsm.exponent + 127 - 1;
    vsm.significand <<= 1;
    if (vsm.significand <= (2 * vsn.significand)) {
        vsn.significand >>= 1;
        vsd.exponent++;
    }
    {
        u64 significand = (u64)vsn.significand << 32;
        do_div(significand, vsm.significand);
        vsd.significand = (u32)significand;
    }
    if ((vsd.significand & 0x3f) == 0)
        vsd.significand |= ((u64)vsm.significand * vsd.significand != (u64)vsn.significand << 32);

    return vfp_single_normaliseround(state, sd, &vsd, fpscr, 0, "fdiv");

vsn_nan:
    exceptions |= vfp_propagate_nan(&vsd, &vsn, &vsm, fpscr);
pack:
    vfp_put_float(state, vfp_single_pack(&vsd), sd);
    return exceptions;

vsm_nan:
    exceptions |= vfp_propagate_nan(&vsd, &vsm, &vsn, fpscr);
    goto pack;

zero:
    vsd.exponent = 0;
    vsd.significand = 0;
    goto pack;

divzero:
    exceptions |= FPSCR_DZC;
infinity:
    vsd.exponent = 255;
    vsd.significand = 0;
    goto pack;

invalid:
    vfp_put_float(state, vfp_single_pack(&vfp_single_default_qnan), sd);
    return FPSCR_IOC;
}

static struct op fops[] = {
    {vfp_single_fmac, 0},  {vfp_single_fmsc, 0},  {vfp_single_fmul, 0},
    {vfp_single_fadd, 0},  {vfp_single_fnmac, 0}, {vfp_single_fnmsc, 0},
    {vfp_single_fnmul, 0}, {vfp_single_fsub, 0},  {vfp_single_fdiv, 0},
};

#define FREG_BANK(x) ((x) & 0x18)
#define FREG_IDX(x) ((x) & 7)

u32 vfp_single_cpdo(ARMul_State* state, u32 inst, u32 fpscr) {
    u32 op = inst & FOP_MASK;
    u32 exceptions = 0;
    unsigned int dest;
    unsigned int sn = vfp_get_sn(inst);
    unsigned int sm = vfp_get_sm(inst);
    unsigned int vecitr, veclen, vecstride;
    struct op* fop;

    vecstride = 1 + ((fpscr & FPSCR_STRIDE_MASK) == FPSCR_STRIDE_MASK);

    fop = (op == FOP_EXT) ? &fops_ext[FEXT_TO_IDX(inst)] : &fops[FOP_TO_IDX(op)];

    /*
     * fcvtsd takes a dN register number as destination, not sN.
     * Technically, if bit 0 of dd is set, this is an invalid
     * instruction.  However, we ignore this for efficiency.
     * It also only operates on scalars.
     */
    if (fop->flags & OP_DD)
        dest = vfp_get_dd(inst);
    else
        dest = vfp_get_sd(inst);

    /*
     * If destination bank is zero, vector length is always '1'.
     * ARM DDI0100F C5.1.3, C5.3.2.
     */
    if ((fop->flags & OP_SCALAR) || FREG_BANK(dest) == 0)
        veclen = 0;
    else
        veclen = fpscr & FPSCR_LENGTH_MASK;

    LOG_TRACE(Core_ARM11, "vecstride={} veclen={}", vecstride, (veclen >> FPSCR_LENGTH_BIT) + 1);

    if (!fop->fn) {
        // Igual que en vfp.cpp: Crash() mataba el proceso sin dejar rastro.
        UNREACHABLE_MSG("VFP: operacion simple {} desconocida, inst=0x{:x}@0x{:x}",
                        FEXT_TO_IDX(inst), inst, state->Reg[15]);
        goto invalid;
    }

    for (vecitr = 0; vecitr <= veclen; vecitr += 1 << FPSCR_LENGTH_BIT) {
        s32 m = vfp_get_float(state, sm);
        u32 except;
        [[maybe_unused]] char type;

        type = (fop->flags & OP_DD) ? 'd' : 's';
        if (op == FOP_EXT)
            LOG_TRACE(Core_ARM11, "itr{} ({}{}) = op[{}] (s{}={:08x})", vecitr >> FPSCR_LENGTH_BIT,
                      type, dest, sn, sm, m);
        else
            LOG_TRACE(Core_ARM11, "itr{} ({}{}) = (s{}) op[{}] (s{}={:08x})",
                      vecitr >> FPSCR_LENGTH_BIT, type, dest, sn, FOP_TO_IDX(op), sm, m);

        except = fop->fn(state, dest, sn, m, fpscr);
        LOG_TRACE(Core_ARM11, "itr{}: exceptions={:08x}", vecitr >> FPSCR_LENGTH_BIT, except);

        exceptions |= except & ~VFP_NAN_FLAG;

        /*
         * CHECK: It appears to be undefined whether we stop when
         * we encounter an exception.  We continue.
         */
        dest = FREG_BANK(dest) + ((FREG_IDX(dest) + vecstride) & 7);
        sn = FREG_BANK(sn) + ((FREG_IDX(sn) + vecstride) & 7);
        if (FREG_BANK(sm) != 0)
            sm = FREG_BANK(sm) + ((FREG_IDX(sm) + vecstride) & 7);
    }
    return exceptions;

invalid:
    return (u32)-1;
}
