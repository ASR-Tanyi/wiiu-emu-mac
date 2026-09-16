#include <wiiu/cpu/disassembler.hpp>
#include <wiiu/cpu/interpreter.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <span>
#include <string>

using namespace wiiu;
using namespace wiiu::cpu;
using Catch::Approx;

namespace {

#include "fixtures.inc" // NOLINT(bugprone-suspicious-include)

constexpr ppcaddr_t kLoadAddress = 0x02000000;
constexpr ppcaddr_t kReturnSentinel = 0x02100000;
constexpr ppcaddr_t kScratch = memory::data_heap_base;

struct FpuFixture {
    memory::Memory memory;
    CoreState state;
    Interpreter interpreter{state, memory};

    FpuFixture() {
        std::string error;
        const bool ok = memory.initialize(error);
        INFO("initialize: " << error);
        REQUIRE(ok);
        state.pc = kLoadAddress;
    }

    StepResult exec(u32 word) {
        REQUIRE(memory.try_write<u32>(kLoadAddress, word));
        state.pc = kLoadAddress;
        const StepResult result = interpreter.step();
        INFO("executing " << disassemble(word, kLoadAddress));
        return result;
    }

    void load(std::span<const u32> program) {
        ppcaddr_t address = kLoadAddress;
        for (const u32 word : program) {
            REQUIRE(memory.try_write<u32>(address, word));
            address += 4;
        }
        state.pc = kLoadAddress;
        state.lr = kReturnSentinel;
    }

    StepResult run_to_return(u64 budget = 10000) {
        for (u64 i = 0; i < budget; ++i) {
            if (state.pc == kReturnSentinel) {
                return StepResult::Ok;
            }
            const StepResult result = interpreter.step();
            if (result != StepResult::Ok) {
                INFO("stopped on " << disassemble(interpreter.last_instruction().raw, state.pc));
                return result;
            }
        }
        FAIL("program did not return within budget");
        return StepResult::Ok;
    }
};

} // namespace

TEST_CASE("the FPU register holds two independent doubles", "[cpu][fpu]") {
    // Modelling this as a union over one 64-bit value would put ps1 on top of
    // half the double on a little-endian host.
    STATIC_REQUIRE(sizeof(FpuRegister) == 16);

    FpuRegister reg;
    reg.ps0 = 1.5;
    reg.ps1 = -2.5;
    CHECK(reg.ps0 == 1.5);
    CHECK(reg.ps1 == -2.5);
    CHECK(reg.as_double() == 1.5);

    reg.splat(3.25);
    CHECK(reg.ps0 == 3.25);
    CHECK(reg.ps1 == 3.25);
}

TEST_CASE_METHOD(FpuFixture, "double arithmetic", "[cpu][fpu]") {
    state.fpr[2].ps0 = 1.5;
    state.fpr[3].ps0 = 0.25;
    state.fpr[4].ps0 = 4.0;

    REQUIRE(exec(kEnc_fadd) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 1.75);

    REQUIRE(exec(kEnc_fsub) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 1.25);

    REQUIRE(exec(kEnc_fmul) == StepResult::Ok); // frA * frC
    CHECK(state.fpr[1].ps0 == 6.0);

    REQUIRE(exec(kEnc_fdiv) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 6.0);
}

TEST_CASE_METHOD(FpuFixture, "multiply-add is fused, not two roundings", "[cpu][fpu]") {
    // Chosen so the unfused and fused answers differ: the product needs more
    // bits than a double can hold, and rounding it before the add loses the
    // difference entirely.
    const f64 a = 1.0 + std::ldexp(1.0, -27); // 1 + 2^-27
    const f64 c = a;
    const f64 b = -1.0;

    state.fpr[2].ps0 = a;
    state.fpr[4].ps0 = c;
    state.fpr[3].ps0 = b;

    REQUIRE(exec(kEnc_fmadd) == StepResult::Ok);

    const f64 fused = std::fma(a, c, b);
    const f64 unfused = a * c + b;
    REQUIRE(fused != unfused); // the test is only meaningful if they differ
    CHECK(state.fpr[1].ps0 == fused);
}

TEST_CASE_METHOD(FpuFixture, "single-precision instructions round once", "[cpu][fpu]") {
    // fadds computes in double and rounds the *result* to single. Computing in
    // float instead would round the operands too, which is a different answer.
    const f64 a = 1.0;
    const f64 b = std::ldexp(1.0, -30); // far below single precision

    state.fpr[2].ps0 = a;
    state.fpr[3].ps0 = b;
    REQUIRE(exec(kEnc_fadds) == StepResult::Ok);

    CHECK(state.fpr[1].ps0 == static_cast<f64>(static_cast<f32>(a + b)));
    CHECK(state.fpr[1].ps0 == 1.0); // the tiny addend vanishes in the rounding

    // ... and the result really is representable as a single.
    CHECK(static_cast<f64>(static_cast<f32>(state.fpr[1].ps0)) == state.fpr[1].ps0);
}

TEST_CASE_METHOD(FpuFixture, "fmuls rounds its result to single", "[cpu][fpu]") {
    state.fpr[2].ps0 = 1.0 / 3.0;
    state.fpr[4].ps0 = 3.0;
    REQUIRE(exec(kEnc_fmuls) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == static_cast<f64>(static_cast<f32>((1.0 / 3.0) * 3.0)));
}

TEST_CASE_METHOD(FpuFixture, "fsel selects on >= 0 and treats NaN as negative", "[cpu][fpu]") {
    state.fpr[4].ps0 = 10.0; // frC, chosen when frA >= 0
    state.fpr[3].ps0 = 20.0; // frB, chosen otherwise

    state.fpr[2].ps0 = 0.0;
    REQUIRE(exec(kEnc_fsel) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 10.0);

    state.fpr[2].ps0 = -0.0;
    REQUIRE(exec(kEnc_fsel) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 10.0); // -0.0 >= 0.0 is true

    state.fpr[2].ps0 = -1.0;
    REQUIRE(exec(kEnc_fsel) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 20.0);

    // A NaN compares false, so it picks frB. That is what makes fsel usable
    // for clamping without a branch.
    state.fpr[2].ps0 = std::nan("");
    REQUIRE(exec(kEnc_fsel) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 20.0);
}

TEST_CASE_METHOD(FpuFixture, "sign manipulation", "[cpu][fpu]") {
    state.fpr[3].ps0 = -7.5;

    REQUIRE(exec(kEnc_fmr) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == -7.5);

    REQUIRE(exec(kEnc_fneg) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 7.5);

    REQUIRE(exec(kEnc_fabs) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 7.5);
}

TEST_CASE_METHOD(FpuFixture, "frsp rounds a double to single", "[cpu][fpu]") {
    state.fpr[3].ps0 = 1.0 / 3.0;
    REQUIRE(exec(kEnc_frsp) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == static_cast<f64>(static_cast<f32>(1.0 / 3.0)));
    CHECK(state.fpr[1].ps0 != 1.0 / 3.0);
}

TEST_CASE_METHOD(FpuFixture, "fctiwz truncates toward zero", "[cpu][fpu]") {
    const auto low_word = [this] { return static_cast<u32>(std::bit_cast<u64>(state.fpr[1].ps0)); };

    state.fpr[3].ps0 = 3.7;
    REQUIRE(exec(kEnc_fctiwz) == StepResult::Ok);
    CHECK(low_word() == 3U);

    state.fpr[3].ps0 = -3.7;
    REQUIRE(exec(kEnc_fctiwz) == StepResult::Ok);
    CHECK(static_cast<s32>(low_word()) == -3);

    // Out of range saturates rather than wrapping.
    state.fpr[3].ps0 = 1e20;
    REQUIRE(exec(kEnc_fctiwz) == StepResult::Ok);
    CHECK(static_cast<s32>(low_word()) == INT32_MAX);
}

TEST_CASE_METHOD(FpuFixture, "comparisons set the CR field", "[cpu][fpu]") {
    state.fpr[2].ps0 = 1.0;
    state.fpr[3].ps0 = 2.0;
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK(state.cr.field(0) == ConditionRegister::lt);

    state.fpr[2].ps0 = 2.0;
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK(state.cr.field(0) == ConditionRegister::eq);

    state.fpr[2].ps0 = 3.0;
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK(state.cr.field(0) == ConditionRegister::gt);

    // NaN is unordered: the summary-overflow bit, not less/greater/equal.
    state.fpr[2].ps0 = std::nan("");
    REQUIRE(exec(kEnc_fcmpu) == StepResult::Ok);
    CHECK(state.cr.field(0) == ConditionRegister::so);
}

TEST_CASE_METHOD(FpuFixture, "single loads widen and single stores narrow", "[cpu][fpu]") {
    state.gpr[4] = kScratch;

    // The register file only holds doubles, so lfs widens on the way in.
    REQUIRE(memory.try_write<u32>(kScratch, std::bit_cast<u32>(1.5F)));
    REQUIRE(exec(kEnc_lfs_0_r4) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 1.5);
    CHECK(state.fpr[1].ps1 == 1.5); // a scalar load fills both slots

    // ... and stfs narrows on the way out, losing precision as it must.
    state.fpr[1].ps0 = 1.0 / 3.0;
    REQUIRE(exec(kEnc_stfs_0_r4) == StepResult::Ok);
    CHECK(std::bit_cast<f32>(memory.read<u32>(kScratch)) == 1.0F / 3.0F);

    // A double round-trips exactly.
    state.fpr[1].ps0 = 1.0 / 3.0;
    REQUIRE(exec(kEnc_stfd_0_r4) == StepResult::Ok);
    state.fpr[1].ps0 = 0.0;
    REQUIRE(exec(kEnc_lfd_0_r4) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 1.0 / 3.0);
}

// ---------------------------------------------------------------------------
// Paired singles
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(FpuFixture, "paired arithmetic works on both slots", "[cpu][fpu][ps]") {
    state.fpr[2] = {.ps0 = 1.0, .ps1 = 2.0};
    state.fpr[3] = {.ps0 = 10.0, .ps1 = 20.0};
    state.fpr[4] = {.ps0 = 3.0, .ps1 = 4.0};

    REQUIRE(exec(kEnc_ps_add) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 11.0);
    CHECK(state.fpr[1].ps1 == 22.0);

    REQUIRE(exec(kEnc_ps_sub) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == -9.0);
    CHECK(state.fpr[1].ps1 == -18.0);

    REQUIRE(exec(kEnc_ps_mul) == StepResult::Ok); // frA * frC, lane-wise
    CHECK(state.fpr[1].ps0 == 3.0);
    CHECK(state.fpr[1].ps1 == 8.0);
}

TEST_CASE_METHOD(FpuFixture, "the s0 and s1 forms broadcast one lane", "[cpu][fpu][ps]") {
    // This is how a vector gets scaled by a scalar without a shuffle.
    state.fpr[2] = {.ps0 = 1.0, .ps1 = 2.0};
    state.fpr[4] = {.ps0 = 10.0, .ps1 = 100.0};

    REQUIRE(exec(kEnc_ps_muls0) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 10.0);
    CHECK(state.fpr[1].ps1 == 20.0); // both scaled by frC.ps0

    REQUIRE(exec(kEnc_ps_muls1) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 100.0);
    CHECK(state.fpr[1].ps1 == 200.0); // both scaled by frC.ps1
}

TEST_CASE_METHOD(FpuFixture, "paired results are single precision", "[cpu][fpu][ps]") {
    state.fpr[2] = {.ps0 = 1.0 / 3.0, .ps1 = 1.0 / 7.0};
    state.fpr[3] = {.ps0 = 0.0, .ps1 = 0.0};

    REQUIRE(exec(kEnc_ps_add) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == static_cast<f64>(1.0F / 3.0F));
    CHECK(state.fpr[1].ps1 == static_cast<f64>(1.0F / 7.0F));
}

TEST_CASE_METHOD(FpuFixture, "the merge family shuffles lanes", "[cpu][fpu][ps]") {
    state.fpr[2] = {.ps0 = 1.0, .ps1 = 2.0};
    state.fpr[3] = {.ps0 = 3.0, .ps1 = 4.0};

    REQUIRE(exec(kEnc_ps_merge00) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 1.0);
    CHECK(state.fpr[1].ps1 == 3.0);

    REQUIRE(exec(kEnc_ps_merge01) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 1.0);
    CHECK(state.fpr[1].ps1 == 4.0);

    REQUIRE(exec(kEnc_ps_merge10) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 2.0);
    CHECK(state.fpr[1].ps1 == 3.0);

    REQUIRE(exec(kEnc_ps_merge11) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 2.0);
    CHECK(state.fpr[1].ps1 == 4.0);
}

TEST_CASE_METHOD(FpuFixture, "ps_sel selects per lane", "[cpu][fpu][ps]") {
    state.fpr[2] = {.ps0 = 1.0, .ps1 = -1.0}; // one lane positive, one negative
    state.fpr[4] = {.ps0 = 10.0, .ps1 = 20.0};
    state.fpr[3] = {.ps0 = 30.0, .ps1 = 40.0};

    REQUIRE(exec(kEnc_ps_sel) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 10.0); // frA.ps0 >= 0 -> frC
    CHECK(state.fpr[1].ps1 == 40.0); // frA.ps1 <  0 -> frB
}

// ---------------------------------------------------------------------------
// Quantisation
// ---------------------------------------------------------------------------

TEST_CASE("quantisation round-trips through each format", "[cpu][fpu][quantize]") {
    // The float format ignores the scale; the integer formats apply it as a
    // signed power-of-two exponent.
    CHECK(Interpreter::dequantize(
              Interpreter::quantize(1.5, QuantizeType::Float, 5), QuantizeType::Float, 5) == 1.5);

    CHECK(Interpreter::quantize(100.0, QuantizeType::U8, 0) == 100U);
    CHECK(Interpreter::dequantize(100, QuantizeType::U8, 0) == 100.0);

    // Scale 1 means the stored integer is twice the value.
    CHECK(Interpreter::quantize(50.0, QuantizeType::U8, 1) == 100U);
    CHECK(Interpreter::dequantize(100, QuantizeType::U8, 1) == 50.0);

    // A negative scale divides instead.
    CHECK(Interpreter::quantize(200.0, QuantizeType::U8, -1) == 100U);
    CHECK(Interpreter::dequantize(100, QuantizeType::U8, -1) == 200.0);
}

TEST_CASE("quantisation clamps rather than wrapping", "[cpu][fpu][quantize]") {
    // Wrapping would turn a slightly-too-bright colour into a dark one, which
    // is the classic symptom of getting this wrong.
    CHECK(Interpreter::quantize(300.0, QuantizeType::U8, 0) == 255U);
    CHECK(Interpreter::quantize(-10.0, QuantizeType::U8, 0) == 0U);

    CHECK(static_cast<s8>(Interpreter::quantize(200.0, QuantizeType::S8, 0)) == 127);
    CHECK(static_cast<s8>(Interpreter::quantize(-200.0, QuantizeType::S8, 0)) == -128);

    CHECK(Interpreter::quantize(100000.0, QuantizeType::U16, 0) == 65535U);
}

TEST_CASE("signed formats sign-extend on the way back", "[cpu][fpu][quantize]") {
    const u32 stored = Interpreter::quantize(-5.0, QuantizeType::S8, 0);
    CHECK(Interpreter::dequantize(stored, QuantizeType::S8, 0) == -5.0);

    const u32 wide = Interpreter::quantize(-1000.0, QuantizeType::S16, 0);
    CHECK(Interpreter::dequantize(wide, QuantizeType::S16, 0) == -1000.0);
}

TEST_CASE_METHOD(FpuFixture, "psq_l loads a quantised pair", "[cpu][fpu][ps]") {
    // GQR2 left at zero: float format, no scale.
    state.gpr[4] = kScratch;
    REQUIRE(memory.try_write<u32>(kScratch, std::bit_cast<u32>(1.5F)));
    REQUIRE(memory.try_write<u32>(kScratch + 4, std::bit_cast<u32>(-2.5F)));

    REQUIRE(exec(kEnc_psq_l_gqr2) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 1.5);
    CHECK(state.fpr[1].ps1 == -2.5);
}

TEST_CASE_METHOD(FpuFixture,
                 "a single-element psq_l leaves 1.0 in the other lane",
                 "[cpu][fpu][ps]") {
    // W=1 loads one value; the 1.0 means a following ps_mul acts as a scalar
    // multiply rather than multiplying by leftover data.
    state.gpr[4] = kScratch;
    REQUIRE(memory.try_write<u32>(kScratch, std::bit_cast<u32>(7.5F)));
    state.fpr[1].ps1 = 999.0;

    REQUIRE(exec(kEnc_psq_l_w1_gqr2) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 7.5);
    CHECK(state.fpr[1].ps1 == 1.0);
}

TEST_CASE_METHOD(FpuFixture, "psq_st writes a quantised pair", "[cpu][fpu][ps]") {
    state.gpr[4] = kScratch;
    state.fpr[1] = {.ps0 = 3.5, .ps1 = -4.5};

    REQUIRE(exec(kEnc_psq_st_gqr2) == StepResult::Ok);
    CHECK(std::bit_cast<f32>(memory.read<u32>(kScratch)) == 3.5F);
    CHECK(std::bit_cast<f32>(memory.read<u32>(kScratch + 4)) == -4.5F);
}

TEST_CASE_METHOD(FpuFixture, "psq_l honours the GQR's format and scale", "[cpu][fpu][ps]") {
    // Set GQR2 to unsigned 8-bit with load scale 2, which is how a game packs
    // normalised data: the stored byte is the value times 4.
    state.gqr[2].value = (2U << 24) | (static_cast<u32>(QuantizeType::U8) << 16);

    state.gpr[4] = kScratch;
    REQUIRE(memory.try_write<u8>(kScratch, 200));
    REQUIRE(memory.try_write<u8>(kScratch + 1, 100));

    REQUIRE(exec(kEnc_psq_l_gqr2) == StepResult::Ok);
    CHECK(state.fpr[1].ps0 == 50.0);
    CHECK(state.fpr[1].ps1 == 25.0);
}

// ---------------------------------------------------------------------------
// Whole programs
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(FpuFixture, "a paired-single dot product", "[cpu][fpu][program]") {
    // Two 4-element vectors, computed with ps_mul and ps_madd the way a Wii U
    // game's maths library would.
    const std::array<f32, 4> lhs{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<f32, 4> rhs{5.0F, 6.0F, 7.0F, 8.0F};

    for (u32 i = 0; i < 4; ++i) {
        REQUIRE(memory.try_write<u32>(kScratch + i * 4, std::bit_cast<u32>(lhs[i])));
        REQUIRE(memory.try_write<u32>(kScratch + 0x100 + i * 4, std::bit_cast<u32>(rhs[i])));
    }

    load(kProgram_dot_product);
    state.gpr[3] = kScratch;
    state.gpr[4] = kScratch + 0x100;
    REQUIRE(run_to_return() == StepResult::Ok);

    // f5 holds the two partial sums; f6 is each doubled.
    CHECK(state.fpr[5].ps0 == 1.0 * 5.0 + 3.0 * 7.0);
    CHECK(state.fpr[5].ps1 == 2.0 * 6.0 + 4.0 * 8.0);
    CHECK(state.fpr[6].ps0 == 2 * (1.0 * 5.0 + 3.0 * 7.0));
}

TEST_CASE_METHOD(FpuFixture, "a float average", "[cpu][fpu][program]") {
    REQUIRE(memory.try_write<u32>(kScratch, std::bit_cast<u32>(7.0F)));
    REQUIRE(memory.try_write<u32>(kScratch + 4, std::bit_cast<u32>(4.0F)));
    REQUIRE(memory.try_write<u32>(kScratch + 8, std::bit_cast<u32>(2.0F)));

    load(kProgram_float_average);
    state.gpr[3] = kScratch;
    state.gpr[4] = kScratch + 0x100;
    REQUIRE(run_to_return() == StepResult::Ok);

    CHECK(std::bit_cast<f32>(memory.read<u32>(kScratch + 0x100)) == 5.5F);
}
