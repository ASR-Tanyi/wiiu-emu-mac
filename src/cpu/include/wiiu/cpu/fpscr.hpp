// Phase 4.6 — the floating-point status and control register.
//
// FPSCR is three things at once: sticky records of what has gone wrong, enable
// bits deciding whether those conditions interrupt, and a description of the
// last result. Two of its bits are not independent state at all -- VX and FEX
// are summaries recomputed from the others -- which is the part an
// implementation is most likely to get wrong, so it lives in one function here
// rather than being open-coded per instruction.
//
// PowerPC numbers FPSCR bits from the most significant end, so the documented
// bit number appears in a comment beside each constant and nowhere else.
#pragma once

#include <wiiu/common/types.hpp>

namespace wiiu::cpu {

struct FpscrBits {
    static constexpr u32 fx = 1U << 31;     ///< bit 0  any exception became set
    static constexpr u32 fex = 1U << 30;    ///< bit 1  summary: enabled and set
    static constexpr u32 vx = 1U << 29;     ///< bit 2  summary: any invalid op
    static constexpr u32 ox = 1U << 28;     ///< bit 3  overflow
    static constexpr u32 ux = 1U << 27;     ///< bit 4  underflow
    static constexpr u32 zx = 1U << 26;     ///< bit 5  zero divide
    static constexpr u32 xx = 1U << 25;     ///< bit 6  inexact (sticky)
    static constexpr u32 vxsnan = 1U << 24; ///< bit 7  signalling NaN operand
    static constexpr u32 vxisi = 1U << 23;  ///< bit 8  infinity - infinity
    static constexpr u32 vxidi = 1U << 22;  ///< bit 9  infinity / infinity
    static constexpr u32 vxzdz = 1U << 21;  ///< bit 10 zero / zero
    static constexpr u32 vximz = 1U << 20;  ///< bit 11 infinity * zero
    static constexpr u32 vxvc = 1U << 19;   ///< bit 12 ordered compare with NaN
    static constexpr u32 fr = 1U << 18;     ///< bit 13 last result rounded up
    static constexpr u32 fi = 1U << 17;     ///< bit 14 last result inexact
    static constexpr u32 fprf = 0x0001F000; ///< bits 15-19 class and compare
    static constexpr u32 vxsoft = 1U << 10; ///< bit 21 software request
    static constexpr u32 vxsqrt = 1U << 9;  ///< bit 22 square root of negative
    static constexpr u32 vxcvi = 1U << 8;   ///< bit 23 invalid integer convert
    static constexpr u32 ve = 1U << 7;      ///< bit 24 invalid op enable
    static constexpr u32 oe = 1U << 6;      ///< bit 25 overflow enable
    static constexpr u32 ue = 1U << 5;      ///< bit 26 underflow enable
    static constexpr u32 ze = 1U << 4;      ///< bit 27 zero divide enable
    static constexpr u32 xe = 1U << 3;      ///< bit 28 inexact enable
    static constexpr u32 ni = 1U << 2;      ///< bit 29 non-IEEE mode
    static constexpr u32 rn = 0x3;          ///< bits 30-31 rounding mode

    /// Every condition VX summarises.
    static constexpr u32 invalid =
        vxsnan | vxisi | vxidi | vxzdz | vximz | vxvc | vxsoft | vxsqrt | vxcvi;

    /// Every sticky exception bit. Setting one of these that was clear is what
    /// sets FX; mtfsf and friends deliberately do not.
    static constexpr u32 exceptions = ox | ux | zx | xx | invalid;
};

/// FPSCR[FPRF] as the architecture tabulates it: the C bit followed by the
/// four condition bits FL, FG, FE, FU.
enum class FpClass : u32 {
    QuietNan = 0x11,
    NegativeInfinity = 0x09,
    NegativeNormal = 0x08,
    NegativeDenormal = 0x18,
    NegativeZero = 0x12,
    PositiveZero = 0x02,
    PositiveDenormal = 0x14,
    PositiveNormal = 0x04,
    PositiveInfinity = 0x05,
};

/// The rounding modes FPSCR[RN] selects.
enum class RoundingMode : u32 {
    Nearest = 0,
    TowardZero = 1,
    TowardPositive = 2,
    TowardNegative = 3,
};

[[nodiscard]] FpClass classify(f64 value) noexcept;

[[nodiscard]] constexpr u32 with_class(u32 fpscr, FpClass value) noexcept {
    return (fpscr & ~FpscrBits::fprf) | (static_cast<u32>(value) << 12);
}

/// Recompute the two summary bits. VX is the OR of the invalid-operation
/// conditions; FEX is the OR of every exception that is both set and enabled.
/// Neither can be written directly -- software that tries is ignored, which is
/// why every path that touches FPSCR ends here.
[[nodiscard]] constexpr u32 with_summaries(u32 fpscr) noexcept {
    u32 out = fpscr & ~(FpscrBits::vx | FpscrBits::fex);

    if ((fpscr & FpscrBits::invalid) != 0) {
        out |= FpscrBits::vx;
    }

    const bool enabled = ((out & FpscrBits::vx) != 0 && (out & FpscrBits::ve) != 0) ||
                         ((out & FpscrBits::ox) != 0 && (out & FpscrBits::oe) != 0) ||
                         ((out & FpscrBits::ux) != 0 && (out & FpscrBits::ue) != 0) ||
                         ((out & FpscrBits::zx) != 0 && (out & FpscrBits::ze) != 0) ||
                         ((out & FpscrBits::xx) != 0 && (out & FpscrBits::xe) != 0);
    if (enabled) {
        out |= FpscrBits::fex;
    }
    return out;
}

/// A signalling NaN: a NaN whose most significant mantissa bit is clear.
[[nodiscard]] bool is_signalling_nan(f64 value) noexcept;

} // namespace wiiu::cpu
