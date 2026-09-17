// Phase 4.6 / 4.7 — floating point and paired singles.
//
// Split from interpreter.cpp because the concerns barely overlap: the integer
// core is about exact bit manipulation, this is about rounding discipline.
//
// Four rules govern nearly everything here:
//
//   Single-precision instructions round once. `fadds` computes in double and
//   rounds the result to single; it does not compute in float. Doing the
//   arithmetic in float would round the operands too, which is a different
//   answer in the last bit.
//
//   The multiply-add family is genuinely fused. PowerPC rounds once, after the
//   multiply and the add, so `std::fma` is the correct implementation and
//   `a * c + b` is not.
//
//   `fres` and `frsqrte` are estimates from a table, not arithmetic. Hardware
//   guarantees only 1/256 and 1/32 relative accuracy, and guest code refines
//   the answer with Newton-Raphson steps tuned to *that* estimate. An exact
//   reciprocal here converges somewhere else.
//
//   Exceptions are recorded, not invented. The host FPU runs in the guest's
//   rounding mode and its raised flags are read back, so overflow, underflow
//   and inexact come from the same arithmetic that produced the result. Only
//   the invalid-operation *reason* is decided here, because the host reports
//   that something was invalid without saying which rule was broken.
//
// Known gap: when an exception is enabled (FPSCR[VE] and friends) the hardware
// suppresses the result write and delivers the interrupt instead. This records
// the exception and signals the interrupt, but still writes the default
// result. Cafe OS leaves the enables clear, so nothing observes the difference
// today.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/fpscr.hpp>
#include <wiiu/cpu/interpreter.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>

namespace wiiu::cpu {
namespace {

/// Round a double to single precision and back, which is what every
/// single-precision PowerPC instruction does to its result.
[[nodiscard]] f64 to_single(f64 value) noexcept {
    return static_cast<f64>(static_cast<f32>(value));
}

/// The value 2^exponent, for a signed quantisation scale.
[[nodiscard]] f64 scale_factor(s32 exponent) noexcept {
    return std::ldexp(1.0, exponent);
}

/// GQR scale fields are six-bit signed.
[[nodiscard]] s32 signed_scale(u32 raw) noexcept {
    return (raw & 0x20) != 0 ? static_cast<s32>(raw) - 64 : static_cast<s32>(raw);
}

template <typename T>
[[nodiscard]] f64 clamp_to(f64 value) noexcept {
    constexpr auto low = static_cast<f64>(std::numeric_limits<T>::min());
    constexpr auto high = static_cast<f64>(std::numeric_limits<T>::max());
    if (std::isnan(value)) {
        return 0.0;
    }
    return std::clamp(value, low, high);
}

// --------------------------------------------------------------------------
// Estimate tables
// --------------------------------------------------------------------------
//
// The 750CL computes fres and frsqrte from a 32-entry table. Each entry gives
// a base value for its interval and a decrement applied across it, so the
// estimate is piecewise linear in the mantissa. The tables are hardware data;
// the shape of the lookup -- which mantissa bits index it, and where the
// result's exponent comes from -- is the part worth reading carefully, since
// an off-by-one in either produces answers that look nearly right.

struct EstimateEntry {
    u32 base;
    u32 decrement;
};

constexpr std::array<EstimateEntry, 32> kReciprocal{{
    {0x3FFC000, 0x3E1}, {0x3C1F000, 0x3A7}, {0x3875000, 0x371}, {0x3504000, 0x340},
    {0x31C4000, 0x313}, {0x2EB1000, 0x2EA}, {0x2BC8000, 0x2C4}, {0x2907000, 0x2A0},
    {0x2669000, 0x27F}, {0x23EC000, 0x261}, {0x218D000, 0x245}, {0x1F4B000, 0x22A},
    {0x1D22000, 0x212}, {0x1B12000, 0x1FB}, {0x1917000, 0x1E5}, {0x1732000, 0x1D1},
    {0x1561000, 0x1BE}, {0x13A2000, 0x1AC}, {0x11F6000, 0x19B}, {0x105A000, 0x18B},
    {0x0ECF000, 0x17C}, {0x0D51000, 0x16E}, {0x0BE1000, 0x15B}, {0x0A7D000, 0x15B},
    {0x0926000, 0x143}, {0x07DA000, 0x143}, {0x0698000, 0x12D}, {0x0563000, 0x12D},
    {0x0439000, 0x11A}, {0x0318000, 0x11A}, {0x0201000, 0x109}, {0x00F4000, 0x0F4},
}};

// The last row's decrement is the one value here that is derived rather than
// transcribed: the reciprocal of a mantissa approaching 2.0 approaches 0.5
// exactly, so the row's interpolation has to reach zero at its end, which
// fixes the decrement at base/4096. Worth checking against the 750CL manual
// along with the rest of the table -- the accuracy test below is what would
// catch any of these being wrong, and it is deliberately tight.

/// Sixteen entries for an even exponent and sixteen for an odd one: halving an
/// odd exponent leaves a factor of sqrt(2) behind, and the second half of the
/// table carries it.
constexpr std::array<EstimateEntry, 32> kReciprocalSqrt{{
    {0x3FFA000, 0x7A4}, {0x3C29000, 0x700}, {0x38AA000, 0x670}, {0x3572000, 0x5F2},
    {0x3279000, 0x584}, {0x2FB7000, 0x524}, {0x2D26000, 0x4CC}, {0x2AC0000, 0x47E},
    {0x2881000, 0x43A}, {0x2665000, 0x3FA}, {0x2468000, 0x3C2}, {0x2287000, 0x38E},
    {0x20C1000, 0x35E}, {0x1F12000, 0x332}, {0x1D79000, 0x30A}, {0x1BF4000, 0x2E6},
    {0x1A7E800, 0x568}, {0x17CB800, 0x4F3}, {0x1552800, 0x48D}, {0x130C000, 0x435},
    {0x10F2000, 0x3E7}, {0x0EFF000, 0x3A2}, {0x0D2E000, 0x365}, {0x0B7C000, 0x32E},
    {0x09E5000, 0x2FC}, {0x0867000, 0x2D0}, {0x06FF000, 0x2A8}, {0x05AB800, 0x283},
    {0x046A000, 0x261}, {0x0339800, 0x243}, {0x0218800, 0x226}, {0x0105800, 0x20B},
}};

constexpr u64 kSignBit = 1ULL << 63;
constexpr u64 kExponentMask = 0x7FFULL << 52;
constexpr u64 kMantissaMask = (1ULL << 52) - 1;

// --------------------------------------------------------------------------
// Host FPU control
// --------------------------------------------------------------------------

[[nodiscard]] int host_rounding(u32 fpscr) noexcept {
    switch (static_cast<RoundingMode>(fpscr & FpscrBits::rn)) {
        case RoundingMode::TowardZero:
            return FE_TOWARDZERO;
        case RoundingMode::TowardPositive:
            return FE_UPWARD;
        case RoundingMode::TowardNegative:
            return FE_DOWNWARD;
        case RoundingMode::Nearest:
            break;
    }
    return FE_TONEAREST;
}

/// Runs an instruction's arithmetic in the guest's rounding mode and collects
/// what the host FPU raised doing it. Reading those flags back is what makes
/// overflow, underflow and inexact real rather than guessed.
class HostFpScope {
public:
    explicit HostFpScope(u32 fpscr) noexcept : previous_(std::fegetround()) {
        static_cast<void>(std::feclearexcept(FE_ALL_EXCEPT));
        const int wanted = host_rounding(fpscr);
        if (wanted != previous_) {
            static_cast<void>(std::fesetround(wanted));
        }
    }

    HostFpScope(const HostFpScope&) = delete;
    HostFpScope& operator=(const HostFpScope&) = delete;
    HostFpScope(HostFpScope&&) = delete;
    HostFpScope& operator=(HostFpScope&&) = delete;

    ~HostFpScope() { static_cast<void>(std::fesetround(previous_)); }

    [[nodiscard]] int raised() const noexcept { return std::fetestexcept(FE_ALL_EXCEPT); }

private:
    int previous_;
};

// --------------------------------------------------------------------------
// Invalid-operation reasons
// --------------------------------------------------------------------------

[[nodiscard]] u32 snan_bits(f64 value) noexcept {
    return is_signalling_nan(value) ? FpscrBits::vxsnan : 0U;
}

[[nodiscard]] u32 invalid_add(f64 lhs, f64 rhs) noexcept {
    const bool both_infinite = std::isinf(lhs) && std::isinf(rhs);
    return (both_infinite && std::signbit(lhs) != std::signbit(rhs)) ? FpscrBits::vxisi : 0U;
}

[[nodiscard]] u32 invalid_subtract(f64 lhs, f64 rhs) noexcept {
    const bool both_infinite = std::isinf(lhs) && std::isinf(rhs);
    return (both_infinite && std::signbit(lhs) == std::signbit(rhs)) ? FpscrBits::vxisi : 0U;
}

[[nodiscard]] u32 invalid_multiply(f64 lhs, f64 rhs) noexcept {
    const bool zero_times_infinity =
        (lhs == 0.0 && std::isinf(rhs)) || (std::isinf(lhs) && rhs == 0.0);
    return zero_times_infinity ? FpscrBits::vximz : 0U;
}

[[nodiscard]] u32 invalid_divide(f64 lhs, f64 rhs) noexcept {
    if (lhs == 0.0 && rhs == 0.0) {
        return FpscrBits::vxzdz;
    }
    if (std::isinf(lhs) && std::isinf(rhs)) {
        return FpscrBits::vxidi;
    }
    return 0U;
}

/// fmadd and friends: the product can be invalid on its own, and an infinite
/// product added to an infinity of the other sign is invalid too.
[[nodiscard]] u32 invalid_multiply_add(f64 lhs, f64 rhs, f64 addend, bool subtract) noexcept {
    if (const u32 product = invalid_multiply(lhs, rhs); product != 0) {
        return product;
    }
    if (!std::isinf(lhs) && !std::isinf(rhs)) {
        return 0U;
    }
    const f64 effective = subtract ? -addend : addend;
    if (!std::isinf(effective)) {
        return 0U;
    }
    const bool product_negative = std::signbit(lhs) != std::signbit(rhs);
    return std::signbit(effective) != product_negative ? FpscrBits::vxisi : 0U;
}

/// What an arithmetic instruction computes, separated from which instruction
/// asked for it. Naming the operation rather than inlining the expression is
/// what lets the same computation be repeated in another rounding mode, which
/// is the only way to recover FPSCR[FR] for a double-precision result.
enum class FpOp : u8 {
    Identity, ///< frB unchanged: frsp rounds it on the way out
    Add,
    Subtract,
    Multiply,
    Divide,
    MultiplyAdd,
    MultiplySubtract,
    NegateMultiplyAdd,
    NegateMultiplySubtract,
    Reciprocal,
    ReciprocalSqrt,
};

[[nodiscard]] f64 evaluate(FpOp op, f64 a, f64 b, f64 c) noexcept {
    switch (op) {
        case FpOp::Add:
            return a + b;
        case FpOp::Subtract:
            return a - b;
        case FpOp::Multiply:
            return a * c;
        case FpOp::Divide:
            return a / b;
        case FpOp::MultiplyAdd:
            return std::fma(a, c, b);
        case FpOp::MultiplySubtract:
            return std::fma(a, c, -b);
        case FpOp::NegateMultiplyAdd:
            return -std::fma(a, c, b);
        case FpOp::NegateMultiplySubtract:
            return -std::fma(a, c, -b);
        case FpOp::Reciprocal:
            return Interpreter::reciprocal_estimate(b);
        case FpOp::ReciprocalSqrt:
            return Interpreter::reciprocal_sqrt_estimate(b);
        case FpOp::Identity:
            break;
    }
    return b;
}

/// The same computation rounded toward zero. FR says whether rounding moved
/// the result away from zero; for a single-precision instruction the
/// unrounded double answers that, but for a double-precision one there is
/// nothing left to compare against, so the FPU is asked again.
[[nodiscard]] f64 evaluate_toward_zero(FpOp op, f64 a, f64 b, f64 c) noexcept {
    const int previous = std::fegetround();
    static_cast<void>(std::fesetround(FE_TOWARDZERO));
    const f64 value = evaluate(op, a, b, c);
    static_cast<void>(std::fesetround(previous));
    return value;
}

/// The estimates divide by their operand, so zero is a zero divide; a
/// reciprocal square root of a negative number is an invalid operation.
[[nodiscard]] u32 invalid_reciprocal(f64 value) noexcept {
    return value == 0.0 ? FpscrBits::zx : 0U;
}

[[nodiscard]] u32 invalid_reciprocal_sqrt(f64 value) noexcept {
    if (value == 0.0) {
        return FpscrBits::zx;
    }
    return value < 0.0 ? FpscrBits::vxsqrt : 0U;
}

} // namespace

// --------------------------------------------------------------------------
// The helpers fpscr.hpp declares
// --------------------------------------------------------------------------

bool is_signalling_nan(f64 value) noexcept {
    const u64 bits = std::bit_cast<u64>(value);
    // A NaN whose quiet bit is clear. The mantissa must be non-zero, or this
    // would be an infinity rather than a NaN at all.
    return (bits & (kExponentMask | (1ULL << 51))) == kExponentMask && (bits & kMantissaMask) != 0;
}

FpClass classify(f64 value) noexcept {
    const bool negative = std::signbit(value);
    switch (std::fpclassify(value)) {
        case FP_NAN:
            return FpClass::QuietNan;
        case FP_INFINITE:
            return negative ? FpClass::NegativeInfinity : FpClass::PositiveInfinity;
        case FP_ZERO:
            return negative ? FpClass::NegativeZero : FpClass::PositiveZero;
        case FP_SUBNORMAL:
            return negative ? FpClass::NegativeDenormal : FpClass::PositiveDenormal;
        default:
            break;
    }
    return negative ? FpClass::NegativeNormal : FpClass::PositiveNormal;
}

// --------------------------------------------------------------------------
// Estimates
// --------------------------------------------------------------------------

f64 Interpreter::reciprocal_estimate(f64 value) noexcept {
    const u64 bits = std::bit_cast<u64>(value);
    const u64 sign = bits & kSignBit;
    const u64 exponent = bits & kExponentMask;
    const u64 mantissa = bits & kMantissaMask;

    if (exponent == 0 && mantissa == 0) {
        return std::bit_cast<f64>(sign | kExponentMask); // 1/±0 is ±infinity
    }
    if (exponent == kExponentMask) {
        // 1/±infinity is ±0; a NaN comes back quietened.
        return mantissa == 0 ? std::bit_cast<f64>(sign) : value + 0.0;
    }

    // The estimate saturates rather than producing subnormals: the hardware's
    // exponent arithmetic cannot express the answer outside this range.
    if (exponent < (895ULL << 52)) {
        return std::bit_cast<f64>(sign | kExponentMask);
    }
    if (exponent >= (1149ULL << 52)) {
        return std::bit_cast<f64>(sign);
    }

    // The top eighteen mantissa bits: five choose the row, thirteen
    // interpolate across it. The base and decrement are in units of 2^26, so
    // the largest base fills the 52-bit mantissa exactly -- which is the check
    // that the split above is the right one.
    const auto index = static_cast<u32>(mantissa >> 34);
    const EstimateEntry& entry = kReciprocal[index / 8192];

    // The interpolation never takes the fraction below zero on hardware. The
    // clamp costs nothing and means a mistranscribed table entry degrades the
    // estimate instead of borrowing into the exponent and returning something
    // twice the right size.
    const u32 drop = ((entry.decrement * (index % 8192)) + 1) / 2;
    const u32 fraction = drop < entry.base ? entry.base - drop : 0;

    return std::bit_cast<f64>(sign | ((0x7FDULL << 52) - exponent) |
                              (static_cast<u64>(fraction) << 26));
}

f64 Interpreter::reciprocal_sqrt_estimate(f64 value) noexcept {
    const u64 bits = std::bit_cast<u64>(value);
    const u64 sign = bits & kSignBit;
    u64 exponent = bits & kExponentMask;
    u64 mantissa = bits & kMantissaMask;

    if (exponent == 0 && mantissa == 0) {
        return std::bit_cast<f64>(sign | kExponentMask); // 1/sqrt(±0) is ±infinity
    }
    if (exponent == kExponentMask) {
        if (mantissa == 0) {
            return sign != 0 ? std::numeric_limits<f64>::quiet_NaN() : 0.0;
        }
        return value + 0.0;
    }
    if (sign != 0) {
        return std::numeric_limits<f64>::quiet_NaN(); // square root of a negative
    }

    if (exponent == 0) {
        // Normalise a subnormal: shift the mantissa up until the implicit bit
        // appears, paying for it in the exponent, then treat it as normal.
        do {
            exponent -= 1ULL << 52;
            mantissa <<= 1;
        } while ((mantissa & (1ULL << 52)) == 0);
        mantissa &= kMantissaMask;
        exponent += 1ULL << 52;
    }

    const bool odd_exponent = ((exponent >> 52) & 1) == 0;

    const auto biased = static_cast<s64>(exponent) - (0x3FELL << 52);
    const s64 result_exponent = ((0x3FFLL << 52) - (biased / 2)) & (0x7FFLL << 52);

    const auto index = static_cast<u32>(mantissa >> 37);
    const EstimateEntry& entry = kReciprocalSqrt[(index / 2048) + (odd_exponent ? 16 : 0)];
    const u32 drop = entry.decrement * (index % 2048);
    const u32 fraction = drop < entry.base ? entry.base - drop : 0;

    return std::bit_cast<f64>(sign | static_cast<u64>(result_exponent) |
                              (static_cast<u64>(fraction) << 26));
}

// --------------------------------------------------------------------------
// Quantisation (psq_l / psq_st)
// --------------------------------------------------------------------------

f64 Interpreter::dequantize(u32 raw, QuantizeType type, s32 scale) noexcept {
    // The float format carries no scale: it is already a float, and there is
    // nothing to normalise.
    switch (type) {
        case QuantizeType::Float:
            return static_cast<f64>(std::bit_cast<f32>(raw));
        case QuantizeType::U8:
            return static_cast<f64>(static_cast<u8>(raw)) * scale_factor(-scale);
        case QuantizeType::U16:
            return static_cast<f64>(static_cast<u16>(raw)) * scale_factor(-scale);
        case QuantizeType::S8:
            return static_cast<f64>(static_cast<s8>(raw)) * scale_factor(-scale);
        case QuantizeType::S16:
            return static_cast<f64>(static_cast<s16>(raw)) * scale_factor(-scale);
    }
    return 0.0;
}

u32 Interpreter::quantize(f64 value, QuantizeType type, s32 scale) noexcept {
    switch (type) {
        case QuantizeType::Float:
            return std::bit_cast<u32>(static_cast<f32>(value));
        case QuantizeType::U8:
            return static_cast<u8>(clamp_to<u8>(value * scale_factor(scale)));
        case QuantizeType::U16:
            return static_cast<u16>(clamp_to<u16>(value * scale_factor(scale)));
        case QuantizeType::S8:
            return static_cast<u32>(
                static_cast<u8>(static_cast<s8>(clamp_to<s8>(value * scale_factor(scale)))));
        case QuantizeType::S16:
            return static_cast<u32>(
                static_cast<u16>(static_cast<s16>(clamp_to<s16>(value * scale_factor(scale)))));
    }
    return 0;
}

// --------------------------------------------------------------------------
// FPSCR bookkeeping
// --------------------------------------------------------------------------

void Interpreter::set_fp_exception(u32 bits) noexcept {
    // FX means "an exception bit went from clear to set". An instruction that
    // re-raises a condition already recorded does not set it again, which is
    // what lets software use FX to find the *first* failure.
    const u32 newly_set = bits & ~state_.fpscr;
    state_.fpscr |= bits;
    if ((newly_set & FpscrBits::exceptions) != 0) {
        state_.fpscr |= FpscrBits::fx;
    }
    state_.fpscr = with_summaries(state_.fpscr);
}

void Interpreter::record_fp_result(f64 rounded, f64 exact, int raised) noexcept {
    // FI and FR describe the last result only; unlike the exception bits they
    // are replaced each time rather than accumulated.
    state_.fpscr &= ~(FpscrBits::fi | FpscrBits::fr);

    u32 bits = 0;
    if ((raised & FE_DIVBYZERO) != 0) {
        bits |= FpscrBits::zx;
    }
    if ((raised & FE_OVERFLOW) != 0) {
        bits |= FpscrBits::ox;
    }
    if ((raised & FE_UNDERFLOW) != 0) {
        bits |= FpscrBits::ux;
    }
    if ((raised & FE_INEXACT) != 0) {
        bits |= FpscrBits::xx;
        state_.fpscr |= FpscrBits::fi;

        // FR says the rounding moved the result away from zero. That is only
        // knowable while the unrounded value is still around, which is the
        // case for a single-precision result computed in double.
        if (std::fabs(rounded) > std::fabs(exact)) {
            state_.fpscr |= FpscrBits::fr;
        }
    }

    state_.fpscr = with_class(state_.fpscr, classify(rounded));
    set_fp_exception(bits);
}

StepResult Interpreter::fp_program_check() const noexcept {
    const bool interrupts_enabled = (state_.msr & (MsrBits::fe0 | MsrBits::fe1)) != 0;
    return ((state_.fpscr & FpscrBits::fex) != 0 && interrupts_enabled) ? StepResult::FpException
                                                                        : StepResult::Ok;
}

// --------------------------------------------------------------------------
// Compare and CR1
// --------------------------------------------------------------------------

void Interpreter::update_cr1() noexcept {
    // CR1 is a copy of the top four FPSCR bits: FX, FEX, VX and OX.
    state_.cr.set_field(1, (state_.fpscr >> 28) & 0xF);
}

void Interpreter::compare_floats(u32 field, f64 lhs, f64 rhs, bool ordered) noexcept {
    u32 bits = 0;
    if (std::isnan(lhs) || std::isnan(rhs)) {
        bits = ConditionRegister::so; // unordered

        // A signalling NaN is invalid in either compare; an ordered compare is
        // invalid even against a quiet one, which is the whole difference
        // between fcmpo and fcmpu. The exception: with invalid-operation
        // interrupts enabled, an SNaN is reported as VXSNAN alone, so that a
        // handler can tell which of the two conditions it is looking at.
        u32 invalid = snan_bits(lhs) | snan_bits(rhs);
        if (ordered && (invalid == 0 || (state_.fpscr & FpscrBits::ve) == 0)) {
            invalid |= FpscrBits::vxvc;
        }
        if (invalid != 0) {
            set_fp_exception(invalid);
        }
    } else if (lhs < rhs) {
        bits = ConditionRegister::lt;
    } else if (lhs > rhs) {
        bits = ConditionRegister::gt;
    } else {
        bits = ConditionRegister::eq;
    }

    state_.cr.set_field(field, bits);

    // FPSCR[FPCC] carries the same four bits in the same order, so that a
    // later mcrfs can recover the comparison.
    state_.fpscr = (state_.fpscr & ~0xF000U) | ((bits & 0xF) << 12);
}

// --------------------------------------------------------------------------
// Loads and stores
// --------------------------------------------------------------------------

StepResult Interpreter::execute_fp_load_store(Opcode id, Instruction inst) {
    const bool indexed =
        id == Opcode::Lfsx || id == Opcode::Lfdx || id == Opcode::Stfsx || id == Opcode::Stfdx;
    const ppcaddr_t address = indexed ? effective_address_indexed(inst) : effective_address(inst);

    const bool update =
        id == Opcode::Lfsu || id == Opcode::Lfdu || id == Opcode::Stfsu || id == Opcode::Stfdu;

    switch (id) {
        // A single-precision load widens to double in the register: the
        // register file only holds doubles.
        case Opcode::Lfs:
        case Opcode::Lfsu:
        case Opcode::Lfsx: {
            u32 raw = 0;
            if (!read_guest<u32>(address, raw)) {
                return StepResult::MemoryFault;
            }
            state_.fpr[inst.frd].splat(static_cast<f64>(std::bit_cast<f32>(raw)));
            break;
        }

        case Opcode::Lfd:
        case Opcode::Lfdu:
        case Opcode::Lfdx: {
            u64 raw = 0;
            if (!read_guest<u64>(address, raw)) {
                return StepResult::MemoryFault;
            }
            state_.fpr[inst.frd].ps0 = std::bit_cast<f64>(raw);
            break;
        }

        case Opcode::Stfs:
        case Opcode::Stfsu:
        case Opcode::Stfsx:
            // A loop that stores is doing work, not waiting (P2).
            note_side_effect();
            if (!write_guest<u32>(address,
                                  std::bit_cast<u32>(static_cast<f32>(state_.fpr[inst.frs].ps0)))) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Stfd:
        case Opcode::Stfdu:
        case Opcode::Stfdx:
            note_side_effect();
            if (!write_guest<u64>(address, std::bit_cast<u64>(state_.fpr[inst.frs].ps0))) {
                return StepResult::MemoryFault;
            }
            break;

        default:
            return StepResult::Unimplemented;
    }

    if (update && inst.ra != 0) {
        state_.gpr[inst.ra] = address;
    }
    return StepResult::Ok;
}

StepResult Interpreter::execute_paired_load_store(Opcode id, Instruction inst) {
    // Any store means this is not an idle loop (P2); the paired forms are no
    // different from the integer ones in that respect.

    const u32 base = inst.ra == 0 ? 0 : state_.gpr[inst.ra];
    const ppcaddr_t address = base + static_cast<u32>(inst.ps_d.value());

    const GqrRegister& gqr = state_.gqr[inst.ps_i];
    const bool store = id == Opcode::PsqSt || id == Opcode::PsqStu;
    const bool update = id == Opcode::PsqLu || id == Opcode::PsqStu;

    const auto type = static_cast<QuantizeType>(store ? gqr.store_type() : gqr.load_type());
    const s32 scale = signed_scale(store ? gqr.store_scale() : gqr.load_scale());

    // The element width follows the quantisation type, not the register.
    const u32 width = (type == QuantizeType::Float)                              ? 4
                      : (type == QuantizeType::U16 || type == QuantizeType::S16) ? 2
                                                                                 : 1;

    const auto read_element = [&](ppcaddr_t at, u32& out) {
        switch (width) {
            case 1: {
                u8 value = 0;
                if (!read_guest<u8>(at, value)) {
                    return false;
                }
                out = value;
                return true;
            }
            case 2: {
                u16 value = 0;
                if (!read_guest<u16>(at, value)) {
                    return false;
                }
                out = value;
                return true;
            }
            default:
                return read_guest<u32>(at, out);
        }
    };

    const auto write_element = [&](ppcaddr_t at, u32 value) {
        switch (width) {
            case 1:
                return write_guest<u8>(at, static_cast<u8>(value));
            case 2:
                return write_guest<u16>(at, static_cast<u16>(value));
            default:
                return write_guest<u32>(at, value);
        }
    };

    FpuRegister& reg = state_.fpr[inst.frd];

    if (store) {
        note_side_effect();
        if (!write_element(address, quantize(reg.ps0, type, scale))) {
            return StepResult::MemoryFault;
        }
        // W selects a single-element transfer; ps1 is simply not stored.
        if (!inst.ps_w && !write_element(address + width, quantize(reg.ps1, type, scale))) {
            return StepResult::MemoryFault;
        }
    } else {
        u32 first = 0;
        if (!read_element(address, first)) {
            return StepResult::MemoryFault;
        }
        reg.ps0 = dequantize(first, type, scale);

        if (inst.ps_w) {
            // A single-element load leaves 1.0 in the other slot, so that a
            // subsequent ps_mul behaves as a scalar multiply.
            reg.ps1 = 1.0;
        } else {
            u32 second = 0;
            if (!read_element(address + width, second)) {
                return StepResult::MemoryFault;
            }
            reg.ps1 = dequantize(second, type, scale);
        }
    }

    if (update && inst.ra != 0) {
        state_.gpr[inst.ra] = address;
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// FPSCR instructions
// --------------------------------------------------------------------------

StepResult Interpreter::execute_fpscr(Opcode id, Instruction inst) {
    // FEX and VX are summaries: software cannot set them, only the conditions
    // they summarise can. Every path below therefore ends in with_summaries.
    constexpr u32 unwritable = FpscrBits::fex | FpscrBits::vx;

    switch (id) {
        case Opcode::Mffs:
            // The high word is undefined on hardware. A recognisable pattern
            // makes accidental reliance on it obvious.
            state_.fpr[inst.frd].ps0 =
                std::bit_cast<f64>(0xFFF8000000000000ULL | static_cast<u64>(state_.fpscr));
            break;

        case Opcode::Mtfsf: {
            const auto raw = static_cast<u32>(std::bit_cast<u64>(state_.fpr[inst.frb].ps0));
            u32 mask = 0;
            for (u32 field = 0; field < 8; ++field) {
                if ((inst.fm.value() & (0x80U >> field)) != 0) {
                    mask |= 0xFU << (28 - field * 4);
                }
            }
            mask &= ~unwritable;
            state_.fpscr = with_summaries((state_.fpscr & ~mask) | (raw & mask));
            break;
        }

        case Opcode::Mtfsb0:
        case Opcode::Mtfsb1: {
            const u32 bit = 1U << (31 - inst.crbd);
            if ((bit & unwritable) == 0) {
                state_.fpscr = id == Opcode::Mtfsb1 ? (state_.fpscr | bit) : (state_.fpscr & ~bit);
            }
            state_.fpscr = with_summaries(state_.fpscr);
            break;
        }

        case Opcode::Mtfsfi: {
            const u32 shift = 28 - inst.crfd * 4;
            const u32 mask = (0xFU << shift) & ~unwritable;
            state_.fpscr =
                with_summaries((state_.fpscr & ~mask) | ((inst.imm4.value() << shift) & mask));
            break;
        }

        case Opcode::Mcrfs: {
            const u32 shift = 28 - inst.crfs * 4;
            state_.cr.set_field(inst.crfd, (state_.fpscr >> shift) & 0xF);

            // Reading a field clears the exception bits in it: mcrfs is how
            // software consumes them, and leaving them set would report the
            // same failure forever. The summaries are not cleared directly --
            // they follow from whatever is left.
            const u32 cleared = (0xFU << shift) & (FpscrBits::exceptions | FpscrBits::fx);
            state_.fpscr = with_summaries(state_.fpscr & ~cleared);
            return StepResult::Ok; // mcrfs has no Rc bit
        }

        default:
            return StepResult::Unimplemented;
    }

    if (inst.rc) {
        update_cr1();
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// Arithmetic
// --------------------------------------------------------------------------

StepResult Interpreter::execute_fp(Opcode id, Instruction inst) {
    const FpuRegister& a = state_.fpr[inst.fra];
    const FpuRegister& b = state_.fpr[inst.frb];
    const FpuRegister& c = state_.fpr[inst.frc];
    FpuRegister& d = state_.fpr[inst.frd];

    // Moves and sign manipulation are bit operations: they do not round, do
    // not update the result class, and do not signal on a NaN of any kind.
    // fsel belongs with them -- it is a branchless select whose comparison is
    // >= 0.0, so a NaN compares false and picks frB, which is what makes it
    // usable for clamping.
    switch (id) {
        case Opcode::Fmr:
        case Opcode::Fneg:
        case Opcode::Fabs:
        case Opcode::Fnabs:
        case Opcode::Fsel: {
            d.ps0 = (id == Opcode::Fmr)     ? b.ps0
                    : (id == Opcode::Fneg)  ? -b.ps0
                    : (id == Opcode::Fabs)  ? std::fabs(b.ps0)
                    : (id == Opcode::Fnabs) ? -std::fabs(b.ps0)
                                            : (a.ps0 >= 0.0 ? c.ps0 : b.ps0);
            if (inst.rc) {
                update_cr1();
            }
            return StepResult::Ok;
        }

        case Opcode::Fcmpu:
            compare_floats(inst.crfd, a.ps0, b.ps0, false);
            return fp_program_check();

        case Opcode::Fcmpo:
            compare_floats(inst.crfd, a.ps0, b.ps0, true);
            return fp_program_check();

        default:
            break;
    }

    // Whether the result is rounded to single precision -- which also decides
    // whether both paired slots are written. The register file is the paired
    // one, so a single-precision scalar result lands in both slots, while a
    // double-precision one occupies ps0 alone.
    const bool single = [id] {
        switch (id) {
            case Opcode::Fadds:
            case Opcode::Fsubs:
            case Opcode::Fmuls:
            case Opcode::Fdivs:
            case Opcode::Fmadds:
            case Opcode::Fmsubs:
            case Opcode::Fnmadds:
            case Opcode::Fnmsubs:
            case Opcode::Fres:
            case Opcode::Frsp:
                return true;
            default:
                return false;
        }
    }();

    const HostFpScope scope{state_.fpscr};

    u32 invalid = 0;
    FpOp op = FpOp::Identity;

    switch (id) {
        case Opcode::Fadd:
        case Opcode::Fadds:
            op = FpOp::Add;
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | invalid_add(a.ps0, b.ps0);
            break;

        case Opcode::Fsub:
        case Opcode::Fsubs:
            op = FpOp::Subtract;
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | invalid_subtract(a.ps0, b.ps0);
            break;

        case Opcode::Fmul:
        case Opcode::Fmuls:
            op = FpOp::Multiply;
            invalid = snan_bits(a.ps0) | snan_bits(c.ps0) | invalid_multiply(a.ps0, c.ps0);
            break;

        case Opcode::Fdiv:
        case Opcode::Fdivs:
            op = FpOp::Divide;
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | invalid_divide(a.ps0, b.ps0);
            break;

        // Fused: one rounding for the whole expression, which is why these are
        // std::fma and not a*c+b.
        case Opcode::Fmadd:
        case Opcode::Fmadds:
            op = FpOp::MultiplyAdd;
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | snan_bits(c.ps0) |
                      invalid_multiply_add(a.ps0, c.ps0, b.ps0, false);
            break;

        case Opcode::Fmsub:
        case Opcode::Fmsubs:
            op = FpOp::MultiplySubtract;
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | snan_bits(c.ps0) |
                      invalid_multiply_add(a.ps0, c.ps0, b.ps0, true);
            break;

        case Opcode::Fnmadd:
        case Opcode::Fnmadds:
            op = FpOp::NegateMultiplyAdd;
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | snan_bits(c.ps0) |
                      invalid_multiply_add(a.ps0, c.ps0, b.ps0, false);
            break;

        case Opcode::Fnmsub:
        case Opcode::Fnmsubs:
            op = FpOp::NegateMultiplySubtract;
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | snan_bits(c.ps0) |
                      invalid_multiply_add(a.ps0, c.ps0, b.ps0, true);
            break;

        case Opcode::Fres:
            op = FpOp::Reciprocal;
            invalid = snan_bits(b.ps0) | invalid_reciprocal(b.ps0);
            break;

        case Opcode::Frsqrte:
            op = FpOp::ReciprocalSqrt;
            invalid = snan_bits(b.ps0) | invalid_reciprocal_sqrt(b.ps0);
            break;

        // frsp is the rounding itself: the operation is the identity and the
        // single-precision result comes from the same rounding step every
        // other single-precision instruction ends with.
        case Opcode::Frsp:
            op = FpOp::Identity;
            invalid = snan_bits(b.ps0);
            break;

        case Opcode::Fctiw:
        case Opcode::Fctiwz: {
            const f64 source = b.ps0;
            u32 bits = snan_bits(source);
            s32 result = 0;

            // FI and FR describe *this* conversion. Without clearing them
            // first they would report whatever the previous instruction did.
            state_.fpscr &= ~(FpscrBits::fi | FpscrBits::fr);

            if (std::isnan(source)) {
                result = INT32_MIN;
                bits |= FpscrBits::vxcvi;
            } else {
                // Round first, then check the range: under round-to-nearest a
                // value just below 2^31 rounds *up* to 2^31, and it is the
                // rounded value that has to fit in the register.
                const f64 converted =
                    id == Opcode::Fctiwz ? std::trunc(source) : std::nearbyint(source);

                if (converted >= 2147483648.0) {
                    result = INT32_MAX;
                    bits |= FpscrBits::vxcvi;
                } else if (converted < -2147483648.0) {
                    result = INT32_MIN;
                    bits |= FpscrBits::vxcvi;
                } else {
                    result = static_cast<s32>(converted);
                    if (converted != source) {
                        bits |= FpscrBits::xx;
                        state_.fpscr |= FpscrBits::fi;
                        if (std::fabs(converted) > std::fabs(source)) {
                            state_.fpscr |= FpscrBits::fr;
                        }
                    }
                }
            }

            // The integer lands in the low word; the high word is undefined on
            // hardware, so a recognisable pattern makes accidental reliance on
            // it obvious. FPRF is undefined after a conversion, and is left
            // alone rather than given a plausible-looking value.
            d.ps0 = std::bit_cast<f64>(0xFFF8000000000000ULL | static_cast<u32>(result));
            set_fp_exception(bits);
            if (inst.rc) {
                update_cr1();
            }
            return fp_program_check();
        }

        default:
            return StepResult::Unimplemented;
    }

    const f64 value = evaluate(op, a.ps0, b.ps0, c.ps0);
    const f64 rounded = single ? to_single(value) : value;

    // Sampled before the re-evaluation below, which runs the same arithmetic
    // again and would add its own flags to the host's.
    const int raised = scope.raised();

    // The reference FR is measured against: the unrounded value for a
    // single-precision result, and the same computation rounded toward zero
    // for a double-precision one -- but only when there was any rounding to
    // describe.
    const f64 reference = (!single && (raised & FE_INEXACT) != 0)
                              ? evaluate_toward_zero(op, a.ps0, b.ps0, c.ps0)
                              : value;

    set_fp_exception(invalid);
    record_fp_result(rounded, reference, raised);

    if (single) {
        d.splat(rounded);
    } else {
        d.ps0 = rounded;
    }

    if (inst.rc) {
        update_cr1();
    }
    return fp_program_check();
}

// --------------------------------------------------------------------------
// Paired singles
// --------------------------------------------------------------------------

StepResult Interpreter::execute_paired(Opcode id, Instruction inst) {
    const FpuRegister& a = state_.fpr[inst.fra];
    const FpuRegister& b = state_.fpr[inst.frb];
    const FpuRegister& c = state_.fpr[inst.frc];
    FpuRegister& d = state_.fpr[inst.frd];

    // Moves, selects and the merge family rearrange bits. Like their scalar
    // counterparts they neither round nor signal.
    switch (id) {
        case Opcode::PsMr:
        case Opcode::PsNeg:
        case Opcode::PsAbs:
        case Opcode::PsNabs:
        case Opcode::PsSel:
        case Opcode::PsMerge00:
        case Opcode::PsMerge01:
        case Opcode::PsMerge10:
        case Opcode::PsMerge11: {
            f64 slot0 = 0.0;
            f64 slot1 = 0.0;
            switch (id) {
                case Opcode::PsMr:
                    slot0 = b.ps0;
                    slot1 = b.ps1;
                    break;
                case Opcode::PsNeg:
                    slot0 = -b.ps0;
                    slot1 = -b.ps1;
                    break;
                case Opcode::PsAbs:
                    slot0 = std::fabs(b.ps0);
                    slot1 = std::fabs(b.ps1);
                    break;
                case Opcode::PsNabs:
                    slot0 = -std::fabs(b.ps0);
                    slot1 = -std::fabs(b.ps1);
                    break;
                case Opcode::PsSel:
                    slot0 = a.ps0 >= 0.0 ? c.ps0 : b.ps0;
                    slot1 = a.ps1 >= 0.0 ? c.ps1 : b.ps1;
                    break;
                // The merge family is the shuffle: the digits name which slot
                // of frA and frB end up in ps0 and ps1.
                case Opcode::PsMerge00:
                    slot0 = a.ps0;
                    slot1 = b.ps0;
                    break;
                case Opcode::PsMerge01:
                    slot0 = a.ps0;
                    slot1 = b.ps1;
                    break;
                case Opcode::PsMerge10:
                    slot0 = a.ps1;
                    slot1 = b.ps0;
                    break;
                default:
                    slot0 = a.ps1;
                    slot1 = b.ps1;
                    break;
            }
            d.ps0 = slot0;
            d.ps1 = slot1;
            if (inst.rc) {
                update_cr1();
            }
            return StepResult::Ok;
        }

        default:
            break;
    }

    const HostFpScope scope{state_.fpscr};

    u32 invalid = 0;
    f64 slot0 = 0.0;
    f64 slot1 = 0.0;

    // Each lane is an independent computation with its own invalid-operation
    // conditions, so FPSCR accumulates both. The result class at the end
    // describes ps0, which is the slot the scalar view of the register sees.
    switch (id) {
        case Opcode::PsAdd:
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | invalid_add(a.ps0, b.ps0) |
                      snan_bits(a.ps1) | snan_bits(b.ps1) | invalid_add(a.ps1, b.ps1);
            slot0 = a.ps0 + b.ps0;
            slot1 = a.ps1 + b.ps1;
            break;

        case Opcode::PsSub:
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | invalid_subtract(a.ps0, b.ps0) |
                      snan_bits(a.ps1) | snan_bits(b.ps1) | invalid_subtract(a.ps1, b.ps1);
            slot0 = a.ps0 - b.ps0;
            slot1 = a.ps1 - b.ps1;
            break;

        case Opcode::PsMul:
            invalid = snan_bits(a.ps0) | snan_bits(c.ps0) | invalid_multiply(a.ps0, c.ps0) |
                      snan_bits(a.ps1) | snan_bits(c.ps1) | invalid_multiply(a.ps1, c.ps1);
            slot0 = a.ps0 * c.ps0;
            slot1 = a.ps1 * c.ps1;
            break;

        case Opcode::PsDiv:
            invalid = snan_bits(a.ps0) | snan_bits(b.ps0) | invalid_divide(a.ps0, b.ps0) |
                      snan_bits(a.ps1) | snan_bits(b.ps1) | invalid_divide(a.ps1, b.ps1);
            slot0 = a.ps0 / b.ps0;
            slot1 = a.ps1 / b.ps1;
            break;

        // The s0/s1 variants broadcast one slot of frC across both lanes,
        // which is how a vector is scaled by a scalar without a shuffle.
        case Opcode::PsMuls0:
            invalid = invalid_multiply(a.ps0, c.ps0) | invalid_multiply(a.ps1, c.ps0);
            slot0 = a.ps0 * c.ps0;
            slot1 = a.ps1 * c.ps0;
            break;

        case Opcode::PsMuls1:
            invalid = invalid_multiply(a.ps0, c.ps1) | invalid_multiply(a.ps1, c.ps1);
            slot0 = a.ps0 * c.ps1;
            slot1 = a.ps1 * c.ps1;
            break;

        case Opcode::PsMadd:
            invalid = invalid_multiply_add(a.ps0, c.ps0, b.ps0, false) |
                      invalid_multiply_add(a.ps1, c.ps1, b.ps1, false);
            slot0 = std::fma(a.ps0, c.ps0, b.ps0);
            slot1 = std::fma(a.ps1, c.ps1, b.ps1);
            break;

        case Opcode::PsMsub:
            invalid = invalid_multiply_add(a.ps0, c.ps0, b.ps0, true) |
                      invalid_multiply_add(a.ps1, c.ps1, b.ps1, true);
            slot0 = std::fma(a.ps0, c.ps0, -b.ps0);
            slot1 = std::fma(a.ps1, c.ps1, -b.ps1);
            break;

        case Opcode::PsNmadd:
            invalid = invalid_multiply_add(a.ps0, c.ps0, b.ps0, false) |
                      invalid_multiply_add(a.ps1, c.ps1, b.ps1, false);
            slot0 = -std::fma(a.ps0, c.ps0, b.ps0);
            slot1 = -std::fma(a.ps1, c.ps1, b.ps1);
            break;

        case Opcode::PsNmsub:
            invalid = invalid_multiply_add(a.ps0, c.ps0, b.ps0, true) |
                      invalid_multiply_add(a.ps1, c.ps1, b.ps1, true);
            slot0 = -std::fma(a.ps0, c.ps0, -b.ps0);
            slot1 = -std::fma(a.ps1, c.ps1, -b.ps1);
            break;

        case Opcode::PsMadds0:
            invalid = invalid_multiply_add(a.ps0, c.ps0, b.ps0, false) |
                      invalid_multiply_add(a.ps1, c.ps0, b.ps1, false);
            slot0 = std::fma(a.ps0, c.ps0, b.ps0);
            slot1 = std::fma(a.ps1, c.ps0, b.ps1);
            break;

        case Opcode::PsMadds1:
            invalid = invalid_multiply_add(a.ps0, c.ps1, b.ps0, false) |
                      invalid_multiply_add(a.ps1, c.ps1, b.ps1, false);
            slot0 = std::fma(a.ps0, c.ps1, b.ps0);
            slot1 = std::fma(a.ps1, c.ps1, b.ps1);
            break;

        case Opcode::PsRes:
            invalid = snan_bits(b.ps0) | invalid_reciprocal(b.ps0) | snan_bits(b.ps1) |
                      invalid_reciprocal(b.ps1);
            slot0 = reciprocal_estimate(b.ps0);
            slot1 = reciprocal_estimate(b.ps1);
            break;

        case Opcode::PsRsqrte:
            invalid = snan_bits(b.ps0) | invalid_reciprocal_sqrt(b.ps0) | snan_bits(b.ps1) |
                      invalid_reciprocal_sqrt(b.ps1);
            slot0 = reciprocal_sqrt_estimate(b.ps0);
            slot1 = reciprocal_sqrt_estimate(b.ps1);
            break;

        default:
            return StepResult::Unimplemented;
    }

    // Every paired-single result is single precision: that is the point of the
    // instruction set.
    const f64 rounded0 = to_single(slot0);
    d.ps0 = rounded0;
    d.ps1 = to_single(slot1);

    set_fp_exception(invalid);
    record_fp_result(rounded0, slot0, scope.raised());

    if (inst.rc) {
        update_cr1();
    }
    return fp_program_check();
}

} // namespace wiiu::cpu
