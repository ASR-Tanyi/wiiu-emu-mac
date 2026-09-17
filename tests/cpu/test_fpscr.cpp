// Phase 4.6 — the estimate tables and FPSCR bookkeeping.
//
// Two things are being checked here that are easy to fake and hard to notice:
//
//   The estimates must be the *hardware's* estimates. An exact reciprocal
//   passes any accuracy test and still breaks guest code, because the
//   Newton-Raphson refinement that follows fres in real software is tuned to
//   the table's error. So the tests check the bound the architecture
//   guarantees *and* that the answer is not simply 1/x.
//
//   Exception bits must come from the arithmetic, not from a guess about it.
//   The sweep below runs every table row; the exception tests drive real
//   instructions rather than poking FPSCR directly.

#include <wiiu/cpu/fpscr.hpp>
#include <wiiu/cpu/interpreter.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <limits>
#include <string>

using namespace wiiu;
using namespace wiiu::cpu;
using Catch::Approx;

namespace {

constexpr ppcaddr_t kLoadAddress = 0x02000000;

// Assembled with binutils; frD is f1, frA is f2, frB is f3, frC is f4.
constexpr u32 kEnc_fadds = 0xEC22182A;      // fadds f1,f2,f3
constexpr u32 kEnc_fsub = 0xFC221828;       // fsub  f1,f2,f3
constexpr u32 kEnc_fmul = 0xFC220132;       // fmul  f1,f2,f4
constexpr u32 kEnc_fdiv = 0xFC221824;       // fdiv  f1,f2,f3
constexpr u32 kEnc_fdivs = 0xEC221824;      // fdivs f1,f2,f3
constexpr u32 kEnc_fres = 0xEC201830;       // fres  f1,f3
constexpr u32 kEnc_fres_rc = 0xEC201831;    // fres. f1,f3
constexpr u32 kEnc_frsqrte = 0xFC201834;    // frsqrte f1,f3
constexpr u32 kEnc_fcmpo = 0xFC021840;      // fcmpo cr0,f2,f3
constexpr u32 kEnc_fcmpu = 0xFC021800;      // fcmpu cr0,f2,f3
constexpr u32 kEnc_fctiwz = 0xFC20181E;     // fctiwz f1,f3
constexpr u32 kEnc_mtfsb1_ve = 0xFF00004C;  // mtfsb1 24  (VE)
constexpr u32 kEnc_mtfsb1_fex = 0xFC20004C; // mtfsb1 1   (FEX -- a summary)
constexpr u32 kEnc_mtfsb0_fx = 0xFC00008C;  // mtfsb0 0   (FX)
constexpr u32 kEnc_mtfsfi_7_1 = 0xFF80110C; // mtfsfi 7,1 (RN = toward zero)
constexpr u32 kEnc_mcrfs_0_0 = 0xFC000080;  // mcrfs cr0,0
constexpr u32 kEnc_mtfsf = 0xFDFE158E;      // mtfsf 255,f2
constexpr u32 kEnc_ps_res = 0x10201830;     // ps_res f1,f3
constexpr u32 kEnc_ps_rsqrte = 0x10201834;  // ps_rsqrte f1,f3

const f64 kQuietNan = std::numeric_limits<f64>::quiet_NaN();
const f64 kInfinity = std::numeric_limits<f64>::infinity();

[[nodiscard]] f64 signalling_nan() noexcept {
    // A NaN with the quiet bit clear. std::numeric_limits::signaling_NaN is
    // not guaranteed to survive being passed around, so build it by hand.
    return std::bit_cast<f64>(0x7FF4000000000000ULL);
}

struct FpscrFixture {
    memory::Memory memory;
    CoreState state;
    Interpreter interpreter{state, memory};

    FpscrFixture() {
        std::string error;
        const bool ok = memory.initialize(error);
        INFO("initialize: " << error);
        REQUIRE(ok);
        state.pc = kLoadAddress;
    }

    StepResult exec(u32 word) {
        REQUIRE(memory.try_write<u32>(kLoadAddress, word));
        state.pc = kLoadAddress;
        return interpreter.step();
    }

    void set_operands(f64 a, f64 b, f64 c = 0.0) {
        state.fpr[2].splat(a);
        state.fpr[3].splat(b);
        state.fpr[4].splat(c);
        state.fpscr = 0;
    }

    [[nodiscard]] bool has(u32 bit) const { return (state.fpscr & bit) != 0; }
};

} // namespace

// ---------------------------------------------------------------------------
// Estimates
// ---------------------------------------------------------------------------

TEST_CASE("the reciprocal estimate stays inside the architected bound", "[cpu][fpu][estimate]") {
    // Every table row, across a range of exponents. The architecture promises
    // 1/256; the 750CL's table is better than that, and being well inside the
    // bound everywhere is what says the interpolation is right rather than
    // merely plausible.
    f64 worst = 0.0;
    for (int exponent = -20; exponent <= 20; ++exponent) {
        for (int step = 0; step < 64; ++step) {
            const f64 mantissa = 1.0 + static_cast<f64>(step) / 64.0;
            const f64 value = std::ldexp(mantissa, exponent);

            for (const f64 input : {value, -value}) {
                const f64 estimate = Interpreter::reciprocal_estimate(input);
                const f64 exact = 1.0 / input;
                worst = std::max(worst, std::fabs((estimate - exact) / exact));
            }
        }
    }
    INFO("worst relative error " << worst);
    CHECK(worst < 1.0 / 256.0);
    CHECK(worst > 0.0); // an exact reciprocal would be a different instruction
}

TEST_CASE("the reciprocal square root estimate stays inside its bound", "[cpu][fpu][estimate]") {
    f64 worst = 0.0;
    for (int exponent = -20; exponent <= 20; ++exponent) {
        for (int step = 0; step < 64; ++step) {
            const f64 value = std::ldexp(1.0 + static_cast<f64>(step) / 64.0, exponent);
            const f64 estimate = Interpreter::reciprocal_sqrt_estimate(value);
            const f64 exact = 1.0 / std::sqrt(value);
            worst = std::max(worst, std::fabs((estimate - exact) / exact));
        }
    }
    INFO("worst relative error " << worst);
    CHECK(worst < 1.0 / 32.0);
    CHECK(worst > 0.0);
}

TEST_CASE("the estimates are estimates, not divisions", "[cpu][fpu][estimate]") {
    // The specific value matters less than the fact that it is wrong in the
    // low bits and right in the high ones: guest code refines it.
    CHECK(Interpreter::reciprocal_estimate(3.0) != 1.0 / 3.0);
    CHECK(Interpreter::reciprocal_estimate(3.0) == Approx(1.0 / 3.0).epsilon(1e-3));

    // Even a power of two comes back slightly low: the first row of the table
    // starts just under the exact value, so the estimate is an under-estimate
    // everywhere rather than straddling the true answer. Guest code relies on
    // that when it refines the result.
    CHECK(Interpreter::reciprocal_estimate(1.0) < 1.0);
    CHECK(Interpreter::reciprocal_estimate(1.0) == Approx(1.0).epsilon(1e-3));
    CHECK(Interpreter::reciprocal_estimate(0.5) == Approx(2.0).epsilon(1e-3));
    CHECK(Interpreter::reciprocal_estimate(-4.0) == Approx(-0.25).epsilon(1e-3));
    CHECK(Interpreter::reciprocal_sqrt_estimate(1.0) == Approx(1.0).epsilon(1e-3));
    CHECK(Interpreter::reciprocal_sqrt_estimate(4.0) == Approx(0.5).epsilon(1e-3));

    // The sign is carried, not computed: a negative input has a negative
    // reciprocal, and the estimate never changes it.
    CHECK(Interpreter::reciprocal_estimate(-3.0) < 0.0);
}

TEST_CASE("the estimates handle the values arithmetic cannot", "[cpu][fpu][estimate]") {
    CHECK(Interpreter::reciprocal_estimate(0.0) == kInfinity);
    CHECK(Interpreter::reciprocal_estimate(-0.0) == -kInfinity);
    CHECK(Interpreter::reciprocal_estimate(kInfinity) == 0.0);
    CHECK(std::signbit(Interpreter::reciprocal_estimate(-kInfinity)));
    CHECK(std::isnan(Interpreter::reciprocal_estimate(kQuietNan)));

    CHECK(Interpreter::reciprocal_sqrt_estimate(0.0) == kInfinity);
    CHECK(Interpreter::reciprocal_sqrt_estimate(-0.0) == -kInfinity);
    CHECK(Interpreter::reciprocal_sqrt_estimate(kInfinity) == 0.0);
    CHECK(std::isnan(Interpreter::reciprocal_sqrt_estimate(-1.0)));
    CHECK(std::isnan(Interpreter::reciprocal_sqrt_estimate(kQuietNan)));

    // A subnormal is normalised before the lookup rather than falling into
    // the zero case, which is where a naive implementation divides by zero.
    const f64 subnormal = std::ldexp(1.0, -1060);
    const f64 estimate = Interpreter::reciprocal_sqrt_estimate(subnormal);
    CHECK(estimate == Approx(1.0 / std::sqrt(subnormal)).epsilon(1.0 / 32.0));
}

TEST_CASE_METHOD(FpscrFixture, "fres and frsqrte use the tables", "[cpu][fpu][estimate]") {
    set_operands(0.0, 3.0);
    REQUIRE(exec(kEnc_fres) == StepResult::Ok);

    // fres is single precision, so the table's answer is rounded to single and
    // written to both slots of the paired register.
    const f64 expected = static_cast<f64>(static_cast<f32>(Interpreter::reciprocal_estimate(3.0)));
    CHECK(state.fpr[1].ps0 == expected);
    CHECK(state.fpr[1].ps1 == expected);

    set_operands(0.0, 4.0);
    REQUIRE(exec(kEnc_frsqrte) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == Interpreter::reciprocal_sqrt_estimate(4.0));
    CHECK(state.fpr[1].ps0 == Approx(0.5).epsilon(1e-3));

    // frsqrte is a double-precision instruction: it leaves the second slot
    // alone rather than splatting.
    state.fpr[1].ps1 = 99.0;
    REQUIRE(exec(kEnc_frsqrte) == StepResult::Ok);
    CHECK(state.fpr[1].ps1 == 99.0);
}

TEST_CASE_METHOD(FpscrFixture, "the paired estimates work per lane", "[cpu][fpu][estimate]") {
    state.fpscr = 0;
    state.fpr[3] = {.ps0 = 2.0, .ps1 = 8.0};

    REQUIRE(exec(kEnc_ps_res) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == Approx(0.5).epsilon(1e-3));
    CHECK(state.fpr[1].ps1 == Approx(0.125).epsilon(1e-3));

    // Each lane went through the table separately rather than one answer
    // being splatted across both.
    CHECK(state.fpr[1].ps0 != state.fpr[1].ps1);

    state.fpr[3] = {.ps0 = 4.0, .ps1 = 16.0};
    REQUIRE(exec(kEnc_ps_rsqrte) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == Approx(0.5).epsilon(1e-3));
    CHECK(state.fpr[1].ps1 == Approx(0.25).epsilon(1e-3));
}

// ---------------------------------------------------------------------------
// Exception bits
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(FpscrFixture, "dividing by zero is recorded, not thrown", "[cpu][fpu][fpscr]") {
    set_operands(1.0, 0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);

    CHECK(has(FpscrBits::zx));
    CHECK(has(FpscrBits::fx));
    CHECK_FALSE(has(FpscrBits::vx));  // a zero divide is not an invalid op
    CHECK_FALSE(has(FpscrBits::fex)); // nothing is enabled
    CHECK(state.fpr[1].ps0 == kInfinity);
}

TEST_CASE_METHOD(FpscrFixture, "each invalid operation names its own reason", "[cpu][fpu][fpscr]") {
    // Zero over zero and infinity over infinity are different failures, and
    // software that inspects FPSCR can tell them apart.
    set_operands(0.0, 0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(has(FpscrBits::vxzdz));
    CHECK(has(FpscrBits::vx));
    CHECK_FALSE(has(FpscrBits::vxidi));

    set_operands(kInfinity, kInfinity);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(has(FpscrBits::vxidi));

    // Infinity minus infinity.
    set_operands(kInfinity, kInfinity);
    REQUIRE(exec(kEnc_fsub) == StepResult::Ok);
    CHECK(has(FpscrBits::vxisi));

    // Zero times infinity.
    set_operands(0.0, 0.0, kInfinity);
    REQUIRE(exec(kEnc_fmul) == StepResult::Ok);
    CHECK(has(FpscrBits::vximz));

    // A square root of a negative number.
    set_operands(0.0, -4.0);
    REQUIRE(exec(kEnc_frsqrte) == StepResult::Ok);
    CHECK(has(FpscrBits::vxsqrt));

    // A conversion that cannot be represented.
    set_operands(0.0, 1e300);
    REQUIRE(exec(kEnc_fctiwz) == StepResult::Ok);
    CHECK(has(FpscrBits::vxcvi));
}

TEST_CASE_METHOD(FpscrFixture, "a signalling NaN signals", "[cpu][fpu][fpscr]") {
    set_operands(signalling_nan(), 1.0);
    REQUIRE(exec(kEnc_fadds) == StepResult::Ok);
    CHECK(has(FpscrBits::vxsnan));
    CHECK(has(FpscrBits::vx));

    // A quiet NaN propagates without signalling: that is the difference.
    set_operands(kQuietNan, 1.0);
    REQUIRE(exec(kEnc_fadds) == StepResult::Ok);
    CHECK_FALSE(has(FpscrBits::vxsnan));
    CHECK(std::isnan(state.fpr[1].ps0));
}

TEST_CASE_METHOD(FpscrFixture, "an ordered compare with a NaN is invalid", "[cpu][fpu][fpscr]") {
    set_operands(kQuietNan, 1.0);
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK_FALSE(has(FpscrBits::vxvc)); // fcmpu tolerates a quiet NaN
    CHECK(state.cr.field(0) == ConditionRegister::so);

    set_operands(kQuietNan, 1.0);
    REQUIRE(exec(kEnc_fcmpo) == StepResult::Ok);
    CHECK(has(FpscrBits::vxvc)); // fcmpo does not

    set_operands(signalling_nan(), 1.0);
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK(has(FpscrBits::vxsnan)); // either compare signals on an SNaN
}

TEST_CASE_METHOD(FpscrFixture, "FX marks the first exception, not every one", "[cpu][fpu][fpscr]") {
    set_operands(1.0, 0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    REQUIRE(has(FpscrBits::fx));

    // Software clears FX and repeats the same failure. ZX is already set, so
    // nothing *became* set and FX stays clear -- which is how FX identifies
    // the first failure in a sequence rather than the latest.
    REQUIRE(exec(kEnc_mtfsb0_fx) == StepResult::Ok);
    REQUIRE_FALSE(has(FpscrBits::fx));

    state.fpr[2].splat(1.0);
    state.fpr[3].splat(0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(has(FpscrBits::zx));
    CHECK_FALSE(has(FpscrBits::fx));
}

TEST_CASE_METHOD(FpscrFixture, "FEX follows the enable bits", "[cpu][fpu][fpscr]") {
    // Enable zero-divide interrupts, then cause one.
    state.fpscr = FpscrBits::ze;
    state.fpr[2].splat(1.0);
    state.fpr[3].splat(0.0);

    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(has(FpscrBits::zx));
    CHECK(has(FpscrBits::fex));

    // MSR decides whether an enabled exception actually interrupts. With
    // FE0/FE1 clear -- which is how Cafe OS runs -- it does not.
    CHECK(state.pc == kLoadAddress + 4);
}

TEST_CASE_METHOD(FpscrFixture,
                 "an enabled exception interrupts when MSR asks for it",
                 "[cpu][fpu][fpscr]") {
    state.fpscr = FpscrBits::ze;
    state.msr = MsrBits::fe0;
    state.fpr[2].splat(1.0);
    state.fpr[3].splat(0.0);

    // In Report mode the interpreter surfaces it rather than vectoring, which
    // is what makes it visible in a test.
    CHECK(exec(kEnc_fdiv) == StepResult::FpException);
    CHECK(step_result_name(StepResult::FpException) == "floating-point exception");
}

TEST_CASE_METHOD(FpscrFixture, "inexact results say so", "[cpu][fpu][fpscr]") {
    // A third is not representable; FI describes this result and XX is sticky.
    set_operands(1.0, 3.0);
    REQUIRE(exec(kEnc_fdivs) == StepResult::Ok);
    CHECK(has(FpscrBits::xx));
    CHECK(has(FpscrBits::fi));

    // An exact result clears FI while leaving the sticky bit alone.
    set_operands(1.0, 2.0);
    state.fpscr = FpscrBits::xx | FpscrBits::fi;
    REQUIRE(exec(kEnc_fdivs) == StepResult::Ok);
    CHECK_FALSE(has(FpscrBits::fi));
    CHECK(has(FpscrBits::xx));
}

TEST_CASE_METHOD(FpscrFixture, "FPRF describes the result", "[cpu][fpu][fpscr]") {
    const auto result_class = [this] {
        return static_cast<FpClass>((state.fpscr & FpscrBits::fprf) >> 12);
    };

    set_operands(1.0, 2.0);
    REQUIRE(exec(kEnc_fadds) == StepResult::Ok);
    CHECK(result_class() == FpClass::PositiveNormal);

    set_operands(-1.0, -2.0);
    REQUIRE(exec(kEnc_fadds) == StepResult::Ok);
    CHECK(result_class() == FpClass::NegativeNormal);

    set_operands(1.0, -1.0);
    REQUIRE(exec(kEnc_fadds) == StepResult::Ok);
    CHECK(result_class() == FpClass::PositiveZero);

    set_operands(1.0, 0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(result_class() == FpClass::PositiveInfinity);

    set_operands(0.0, 0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(result_class() == FpClass::QuietNan);
}

TEST_CASE_METHOD(FpscrFixture, "a compare lands in FPCC as well as CR", "[cpu][fpu][fpscr]") {
    set_operands(1.0, 2.0);
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK(state.cr.field(0) == ConditionRegister::lt);
    CHECK(((state.fpscr >> 12) & 0xF) == ConditionRegister::lt);

    set_operands(2.0, 2.0);
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK(((state.fpscr >> 12) & 0xF) == ConditionRegister::eq);
}

TEST_CASE_METHOD(FpscrFixture, "the rounding mode reaches the arithmetic", "[cpu][fpu][fpscr]") {
    // Round to nearest gives one answer; round toward zero gives the one
    // below it. If FPSCR[RN] were ignored these would be identical.
    set_operands(1.0, 3.0);
    REQUIRE(exec(kEnc_fdivs) == StepResult::Ok);
    const f64 nearest = state.fpr[1].ps0;

    state.fpscr = 0;
    REQUIRE(exec(kEnc_mtfsfi_7_1) == StepResult::Ok);
    REQUIRE((state.fpscr & FpscrBits::rn) == static_cast<u32>(RoundingMode::TowardZero));

    state.fpr[2].splat(1.0);
    state.fpr[3].splat(3.0);
    REQUIRE(exec(kEnc_fdivs) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 < nearest);
}

// ---------------------------------------------------------------------------
// Moving FPSCR around
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(FpscrFixture, "the summary bits cannot be written directly", "[cpu][fpu][fpscr]") {
    state.fpscr = 0;

    // mtfsb1 on FEX is defined to do nothing: it is a summary of other bits,
    // and letting software set it would make it lie.
    REQUIRE(exec(kEnc_mtfsb1_fex) == StepResult::Ok);
    CHECK_FALSE(has(FpscrBits::fex));

    // Setting an ordinary bit works.
    REQUIRE(exec(kEnc_mtfsb1_ve) == StepResult::Ok);
    CHECK(has(FpscrBits::ve));

    // And with VE now set, an invalid operation raises FEX by itself.
    state.fpr[2].splat(0.0);
    state.fpr[3].splat(0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(has(FpscrBits::vx));
    CHECK(has(FpscrBits::fex));
}

TEST_CASE_METHOD(FpscrFixture, "mtfsf writes the fields its mask names", "[cpu][fpu][fpscr]") {
    state.fpscr = 0;
    state.fpr[2].ps0 = std::bit_cast<f64>(static_cast<u64>(FpscrBits::xe | FpscrBits::ni));

    REQUIRE(exec(kEnc_mtfsf) == StepResult::Ok);
    CHECK(has(FpscrBits::xe));
    CHECK(has(FpscrBits::ni));
}

TEST_CASE_METHOD(FpscrFixture, "mcrfs consumes the bits it reports", "[cpu][fpu][fpscr]") {
    set_operands(1.0, 0.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    REQUIRE(has(FpscrBits::fx));
    REQUIRE(has(FpscrBits::zx));

    // Field 0 is FX, FEX, VX, OX. Reading it hands those to CR0 and clears the
    // exception bits behind them -- otherwise the same failure would be
    // reported forever.
    REQUIRE(exec(kEnc_mcrfs_0_0) == StepResult::Ok);
    CHECK((state.cr.field(0) & ConditionRegister::lt) != 0); // FX was set
    CHECK_FALSE(has(FpscrBits::fx));

    // ZX lives in another field and is untouched.
    CHECK(has(FpscrBits::zx));
}

TEST_CASE_METHOD(FpscrFixture, "Rc copies the summary bits into CR1", "[cpu][fpu][fpscr]") {
    set_operands(0.0, 0.0);
    REQUIRE(exec(kEnc_fres_rc) == StepResult::Ok);

    // fres of zero is a zero divide, so FX is set and CR1 says so.
    CHECK(has(FpscrBits::zx));
    CHECK((state.cr.field(1) & ConditionRegister::lt) != 0);
    CHECK(state.cr.field(1) == ((state.fpscr >> 28) & 0xF));
}

// ---------------------------------------------------------------------------
// The summary rule on its own
// ---------------------------------------------------------------------------

TEST_CASE("VX and FEX are computed, never stored", "[cpu][fpu][fpscr]") {
    STATIC_REQUIRE(with_summaries(FpscrBits::vxsnan) == (FpscrBits::vxsnan | FpscrBits::vx));

    // Set but not enabled: no FEX.
    STATIC_REQUIRE((with_summaries(FpscrBits::ox) & FpscrBits::fex) == 0);

    // Enabled but not set: still no FEX.
    STATIC_REQUIRE((with_summaries(FpscrBits::oe) & FpscrBits::fex) == 0);

    // Both.
    STATIC_REQUIRE((with_summaries(FpscrBits::ox | FpscrBits::oe) & FpscrBits::fex) != 0);

    // A stale summary is cleared rather than believed.
    STATIC_REQUIRE(
        (with_summaries(FpscrBits::vx | FpscrBits::fex) & (FpscrBits::vx | FpscrBits::fex)) == 0);
}

// ---------------------------------------------------------------------------
// Findings from the review pass
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(FpscrFixture,
                 "FR reports the direction a double was rounded",
                 "[cpu][fpu][fpscr]") {
    // A fifth rounds away from zero, a third rounds toward it. Both are
    // inexact, and neither result carries the unrounded value, so this only
    // passes if the direction is measured rather than assumed.
    set_operands(1.0, 5.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(has(FpscrBits::fi));
    CHECK(has(FpscrBits::fr));

    set_operands(1.0, 3.0);
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(has(FpscrBits::fi));
    CHECK_FALSE(has(FpscrBits::fr));

    // An exact result leaves both clear rather than inheriting them.
    set_operands(1.0, 2.0);
    state.fpscr = FpscrBits::fi | FpscrBits::fr;
    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK_FALSE(has(FpscrBits::fi));
    CHECK_FALSE(has(FpscrBits::fr));
}

TEST_CASE_METHOD(FpscrFixture, "fctiw describes its own rounding", "[cpu][fpu][fpscr]") {
    constexpr u32 kEnc_fctiw = 0xFC20181C; // fctiw f1,f3

    // An inexact conversion sets FI, and FR says which way it went.
    set_operands(0.0, 2.5);
    REQUIRE(exec(kEnc_fctiw) == StepResult::Ok);
    CHECK(has(FpscrBits::fi));
    CHECK(has(FpscrBits::xx));

    // An exact one clears them, rather than reporting what the last
    // instruction did.
    set_operands(0.0, 2.0);
    state.fpscr = FpscrBits::fi | FpscrBits::fr;
    REQUIRE(exec(kEnc_fctiw) == StepResult::Ok);
    CHECK_FALSE(has(FpscrBits::fi));
    CHECK_FALSE(has(FpscrBits::fr));
}

TEST_CASE_METHOD(FpscrFixture,
                 "fctiw checks the rounded value, not the source",
                 "[cpu][fpu][fpscr]") {
    // 2147483647.5 fits in a double and rounds to 2^31, which does not fit in
    // the register. Checking the source instead of the result converts it to
    // the most negative integer and reports nothing wrong.
    constexpr u32 kEnc_fctiw = 0xFC20181C; // fctiw f1,f3
    set_operands(0.0, 2147483647.5);

    REQUIRE(exec(kEnc_fctiw) == StepResult::Ok);
    CHECK(static_cast<u32>(std::bit_cast<u64>(state.fpr[1].ps0) & 0xFFFFFFFF) == 0x7FFFFFFFU);
    CHECK(has(FpscrBits::vxcvi));

    // Truncation of the same value is in range and perfectly ordinary.
    set_operands(0.0, 2147483647.5);
    REQUIRE(exec(kEnc_fctiwz) == StepResult::Ok);
    CHECK(static_cast<u32>(std::bit_cast<u64>(state.fpr[1].ps0) & 0xFFFFFFFF) == 0x7FFFFFFFU);
    CHECK_FALSE(has(FpscrBits::vxcvi));
}

TEST_CASE_METHOD(FpscrFixture,
                 "fcmpo on a signalling NaN with VE set reports only VXSNAN",
                 "[cpu][fpu][fpscr]") {
    // With invalid-operation interrupts enabled the handler is told which of
    // the two conditions it is looking at; with them disabled both are
    // recorded.
    state.fpr[2].splat(signalling_nan());
    state.fpr[3].splat(1.0);
    state.fpscr = FpscrBits::ve;

    REQUIRE(exec(kEnc_fcmpo) == StepResult::Ok);
    CHECK(has(FpscrBits::vxsnan));
    CHECK_FALSE(has(FpscrBits::vxvc));

    state.fpr[2].splat(signalling_nan());
    state.fpscr = 0;
    REQUIRE(exec(kEnc_fcmpo) == StepResult::Ok);
    CHECK(has(FpscrBits::vxsnan));
    CHECK(has(FpscrBits::vxvc));
}

TEST_CASE_METHOD(FpscrFixture, "frsp fills both slots", "[cpu][fpu][fpscr]") {
    // It produces a single-precision value, and on this register file that
    // means both lanes -- otherwise a following paired operation reads
    // whatever was in the second slot before.
    constexpr u32 kEnc_frsp = 0xFC201818; // frsp f1,f3

    state.fpr[1].ps1 = 99.0;
    set_operands(0.0, 1.0 / 3.0);

    REQUIRE(exec(kEnc_frsp) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == static_cast<f64>(static_cast<f32>(1.0 / 3.0)));
    CHECK(state.fpr[1].ps1 == state.fpr[1].ps0);
    CHECK(has(FpscrBits::fi)); // a third does not survive the narrowing
}

TEST_CASE_METHOD(FpscrFixture, "mtfsf writes only the fields its mask names", "[cpu][fpu][fpscr]") {
    constexpr u32 kEnc_mtfsf_field0 = 0xFD00158E; // mtfsf 128,f2 -- field 0 only
    constexpr u32 kEnc_mtfsf_field7 = 0xFC02158E; // mtfsf 1,f2   -- field 7 only

    // Field 7 is the low four bits: XE, NI and the rounding mode.
    state.fpscr = 0;
    state.fpr[2].ps0 = std::bit_cast<f64>(static_cast<u64>(0xFFFFFFFFU));
    REQUIRE(exec(kEnc_mtfsf_field7) == StepResult::Ok);
    CHECK((state.fpscr & 0xF) == 0xFU);
    CHECK((state.fpscr & ~0xFU) == 0U); // nothing else moved

    // Field 0 is FX and the two summaries, which software may not set.
    state.fpscr = 0;
    REQUIRE(exec(kEnc_mtfsf_field0) == StepResult::Ok);
    CHECK(has(FpscrBits::fx));
    CHECK_FALSE(has(FpscrBits::fex));
    CHECK_FALSE(has(FpscrBits::vx));
}
