// Phase 4.6 / 4.7 — floating point and paired singles.
//
// Split from interpreter.cpp because the concerns barely overlap: the integer
// core is about exact bit manipulation, this is about rounding discipline.
//
// Two rules govern nearly everything here:
//
//   Single-precision instructions round once. `fadds` computes in double and
//   rounds the result to single; it does not compute in float. Doing the
//   arithmetic in float would round the operands too, which is a different
//   answer in the last bit.
//
//   The multiply-add family is genuinely fused. PowerPC rounds once, after the
//   multiply and the add, so `std::fma` is the correct implementation and
//   `a * c + b` is not.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>

#include <bit>
#include <cmath>
#include <limits>

namespace wiiu::cpu {
namespace {

using wiiu::log::Channel;

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

} // namespace

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
// Compare and CR1
// --------------------------------------------------------------------------

void Interpreter::update_cr1() noexcept {
    // CR1 is a copy of the top four FPSCR bits. Exception bookkeeping beyond
    // this is deliberately not modelled yet; see the note in the header.
    state_.cr.set_field(1, (state_.fpscr >> 28) & 0xF);
}

void Interpreter::compare_floats(u32 field, f64 lhs, f64 rhs, bool ordered) noexcept {
    u32 bits = 0;
    if (std::isnan(lhs) || std::isnan(rhs)) {
        bits = ConditionRegister::so;
        if (ordered) {
            // An ordered compare against a NaN is an invalid operation. The
            // FPSCR bookkeeping is a gap; the CR result is correct.
            state_.fpscr |= 0x80000000U; // FX
        }
    } else if (lhs < rhs) {
        bits = ConditionRegister::lt;
    } else if (lhs > rhs) {
        bits = ConditionRegister::gt;
    } else {
        bits = ConditionRegister::eq;
    }
    state_.cr.set_field(field, bits);
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
            if (!memory_.try_read<u32>(address, raw)) {
                return StepResult::MemoryFault;
            }
            state_.fpr[inst.frd].splat(static_cast<f64>(std::bit_cast<f32>(raw)));
            break;
        }

        case Opcode::Lfd:
        case Opcode::Lfdu:
        case Opcode::Lfdx: {
            u64 raw = 0;
            if (!memory_.try_read<u64>(address, raw)) {
                return StepResult::MemoryFault;
            }
            state_.fpr[inst.frd].ps0 = std::bit_cast<f64>(raw);
            break;
        }

        case Opcode::Stfs:
        case Opcode::Stfsu:
        case Opcode::Stfsx:
            if (!memory_.try_write<u32>(
                    address, std::bit_cast<u32>(static_cast<f32>(state_.fpr[inst.frs].ps0)))) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Stfd:
        case Opcode::Stfdu:
        case Opcode::Stfdx:
            if (!memory_.try_write<u64>(address, std::bit_cast<u64>(state_.fpr[inst.frs].ps0))) {
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
                if (!memory_.try_read<u8>(at, value)) {
                    return false;
                }
                out = value;
                return true;
            }
            case 2: {
                u16 value = 0;
                if (!memory_.try_read<u16>(at, value)) {
                    return false;
                }
                out = value;
                return true;
            }
            default:
                return memory_.try_read<u32>(at, out);
        }
    };

    const auto write_element = [&](ppcaddr_t at, u32 value) {
        switch (width) {
            case 1:
                return memory_.try_write<u8>(at, static_cast<u8>(value));
            case 2:
                return memory_.try_write<u16>(at, static_cast<u16>(value));
            default:
                return memory_.try_write<u32>(at, value);
        }
    };

    FpuRegister& reg = state_.fpr[inst.frd];

    if (store) {
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
// Arithmetic
// --------------------------------------------------------------------------

StepResult Interpreter::execute_fp(Opcode id, Instruction inst) {
    const FpuRegister& a = state_.fpr[inst.fra];
    const FpuRegister& b = state_.fpr[inst.frb];
    const FpuRegister& c = state_.fpr[inst.frc];
    FpuRegister& d = state_.fpr[inst.frd];

    // Whether the result is rounded to single precision, and whether it
    // applies to both paired slots.
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
                return true;
            default:
                return false;
        }
    }();

    const auto finish = [&](f64 value) {
        d.splat(single ? to_single(value) : value);
        if (inst.rc) {
            update_cr1();
        }
        return StepResult::Ok;
    };

    switch (id) {
        case Opcode::Fadd:
        case Opcode::Fadds:
            return finish(a.ps0 + b.ps0);
        case Opcode::Fsub:
        case Opcode::Fsubs:
            return finish(a.ps0 - b.ps0);
        case Opcode::Fmul:
        case Opcode::Fmuls:
            return finish(a.ps0 * c.ps0);
        case Opcode::Fdiv:
        case Opcode::Fdivs:
            return finish(a.ps0 / b.ps0);

        // Fused: one rounding for the whole expression, which is why these
        // are std::fma and not a*c+b.
        case Opcode::Fmadd:
        case Opcode::Fmadds:
            return finish(std::fma(a.ps0, c.ps0, b.ps0));
        case Opcode::Fmsub:
        case Opcode::Fmsubs:
            return finish(std::fma(a.ps0, c.ps0, -b.ps0));
        case Opcode::Fnmadd:
        case Opcode::Fnmadds:
            return finish(-std::fma(a.ps0, c.ps0, b.ps0));
        case Opcode::Fnmsub:
        case Opcode::Fnmsubs:
            return finish(-std::fma(a.ps0, c.ps0, -b.ps0));

        // fsel is a branchless select, and the comparison is >= 0.0. A NaN
        // compares false and so picks frB, which is what makes it usable for
        // clamping.
        case Opcode::Fsel:
            return finish(a.ps0 >= 0.0 ? c.ps0 : b.ps0);

        // Estimates. See the precision note in interpreter.hpp: hardware uses
        // a lookup table and these are exact, which differs in the low bits.
        case Opcode::Fres:
            return finish(1.0 / b.ps0);
        case Opcode::Frsqrte:
            return finish(1.0 / std::sqrt(b.ps0));

        case Opcode::Fmr:
            d.ps0 = b.ps0;
            if (inst.rc) {
                update_cr1();
            }
            return StepResult::Ok;
        case Opcode::Fneg:
            return finish(-b.ps0);
        case Opcode::Fabs:
            return finish(std::fabs(b.ps0));
        case Opcode::Fnabs:
            return finish(-std::fabs(b.ps0));
        case Opcode::Frsp:
            return finish(to_single(b.ps0));

        case Opcode::Fctiw:
        case Opcode::Fctiwz: {
            const f64 value = b.ps0;
            s32 result = 0;
            if (std::isnan(value)) {
                result = INT32_MIN;
            } else if (value >= 2147483647.0) {
                result = INT32_MAX;
            } else if (value <= -2147483648.0) {
                result = INT32_MIN;
            } else {
                result = static_cast<s32>(id == Opcode::Fctiwz ? std::trunc(value)
                                                               : std::nearbyint(value));
            }
            // The integer lands in the low word; the high word is undefined on
            // hardware, so a recognisable pattern makes accidental reliance
            // on it obvious.
            d.ps0 = std::bit_cast<f64>((0xFFF8000000000000ULL) | static_cast<u32>(result));
            if (inst.rc) {
                update_cr1();
            }
            return StepResult::Ok;
        }

        case Opcode::Fcmpu:
            compare_floats(inst.crfd, a.ps0, b.ps0, false);
            return StepResult::Ok;
        case Opcode::Fcmpo:
            compare_floats(inst.crfd, a.ps0, b.ps0, true);
            return StepResult::Ok;

        case Opcode::Mffs:
            d.ps0 = std::bit_cast<f64>(static_cast<u64>(state_.fpscr));
            if (inst.rc) {
                update_cr1();
            }
            return StepResult::Ok;

        case Opcode::Mtfsf: {
            const auto raw = static_cast<u32>(std::bit_cast<u64>(b.ps0));
            u32 mask = 0;
            for (u32 field = 0; field < 8; ++field) {
                if ((inst.fm.value() & (0x80U >> field)) != 0) {
                    mask |= 0xFU << (28 - field * 4);
                }
            }
            state_.fpscr = (state_.fpscr & ~mask) | (raw & mask);
            if (inst.rc) {
                update_cr1();
            }
            return StepResult::Ok;
        }

        case Opcode::Mcrfs:
            state_.cr.set_field(inst.crfd, (state_.fpscr >> (28 - inst.crfs * 4)) & 0xF);
            return StepResult::Ok;

        default:
            return StepResult::Unimplemented;
    }
}

// --------------------------------------------------------------------------
// Paired singles
// --------------------------------------------------------------------------

StepResult Interpreter::execute_paired(Opcode id, Instruction inst) {
    const FpuRegister& a = state_.fpr[inst.fra];
    const FpuRegister& b = state_.fpr[inst.frb];
    const FpuRegister& c = state_.fpr[inst.frc];
    FpuRegister& d = state_.fpr[inst.frd];

    // Every paired-single result is single precision: that is the point of the
    // instruction set.
    const auto finish = [&](f64 slot0, f64 slot1) {
        d.ps0 = to_single(slot0);
        d.ps1 = to_single(slot1);
        if (inst.rc) {
            update_cr1();
        }
        return StepResult::Ok;
    };

    switch (id) {
        case Opcode::PsAdd:
            return finish(a.ps0 + b.ps0, a.ps1 + b.ps1);
        case Opcode::PsSub:
            return finish(a.ps0 - b.ps0, a.ps1 - b.ps1);
        case Opcode::PsMul:
            return finish(a.ps0 * c.ps0, a.ps1 * c.ps1);
        case Opcode::PsDiv:
            return finish(a.ps0 / b.ps0, a.ps1 / b.ps1);

        // The s0/s1 variants broadcast one slot of frC across both lanes,
        // which is how a vector is scaled by a scalar without a shuffle.
        case Opcode::PsMuls0:
            return finish(a.ps0 * c.ps0, a.ps1 * c.ps0);
        case Opcode::PsMuls1:
            return finish(a.ps0 * c.ps1, a.ps1 * c.ps1);

        case Opcode::PsMadd:
            return finish(std::fma(a.ps0, c.ps0, b.ps0), std::fma(a.ps1, c.ps1, b.ps1));
        case Opcode::PsMsub:
            return finish(std::fma(a.ps0, c.ps0, -b.ps0), std::fma(a.ps1, c.ps1, -b.ps1));
        case Opcode::PsNmadd:
            return finish(-std::fma(a.ps0, c.ps0, b.ps0), -std::fma(a.ps1, c.ps1, b.ps1));
        case Opcode::PsNmsub:
            return finish(-std::fma(a.ps0, c.ps0, -b.ps0), -std::fma(a.ps1, c.ps1, -b.ps1));
        case Opcode::PsMadds0:
            return finish(std::fma(a.ps0, c.ps0, b.ps0), std::fma(a.ps1, c.ps0, b.ps1));
        case Opcode::PsMadds1:
            return finish(std::fma(a.ps0, c.ps1, b.ps0), std::fma(a.ps1, c.ps1, b.ps1));

        case Opcode::PsSel:
            return finish(a.ps0 >= 0.0 ? c.ps0 : b.ps0, a.ps1 >= 0.0 ? c.ps1 : b.ps1);

        case Opcode::PsRes:
            return finish(1.0 / b.ps0, 1.0 / b.ps1);
        case Opcode::PsRsqrte:
            return finish(1.0 / std::sqrt(b.ps0), 1.0 / std::sqrt(b.ps1));

        case Opcode::PsNeg:
            return finish(-b.ps0, -b.ps1);
        case Opcode::PsAbs:
            return finish(std::fabs(b.ps0), std::fabs(b.ps1));
        case Opcode::PsNabs:
            return finish(-std::fabs(b.ps0), -std::fabs(b.ps1));

        case Opcode::PsMr:
            // A move copies the bits; it does not round.
            d.ps0 = b.ps0;
            d.ps1 = b.ps1;
            if (inst.rc) {
                update_cr1();
            }
            return StepResult::Ok;

        // The merge family is the shuffle: the digits name which slot of frA
        // and frB end up in ps0 and ps1.
        case Opcode::PsMerge00:
            return finish(a.ps0, b.ps0);
        case Opcode::PsMerge01:
            return finish(a.ps0, b.ps1);
        case Opcode::PsMerge10:
            return finish(a.ps1, b.ps0);
        case Opcode::PsMerge11:
            return finish(a.ps1, b.ps1);

        default:
            return StepResult::Unimplemented;
    }
}

} // namespace wiiu::cpu
