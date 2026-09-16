#include <wiiu/common/log.hpp>
#include <wiiu/cpu/disassembler.hpp>
#include <wiiu/cpu/interpreter.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <span>
#include <string>

using namespace wiiu;
using namespace wiiu::cpu;

namespace {

#include "fixtures.inc" // NOLINT(bugprone-suspicious-include)

/// Where test programs are loaded, and the address a `blr` returns to when the
/// program is done.
constexpr ppcaddr_t kLoadAddress = 0x02000000;
constexpr ppcaddr_t kReturnSentinel = 0x02100000;

struct CpuFixture {
    memory::Memory memory;
    CoreState state;
    Interpreter interpreter{state, memory};

    CpuFixture() {
        std::string error;
        const bool ok = memory.initialize(error);
        INFO("initialize: " << error);
        REQUIRE(ok);
        state.pc = kLoadAddress;
    }

    /// Assemble-free single-instruction execution: place one word, run it.
    StepResult exec(u32 word) {
        REQUIRE(memory.try_write<u32>(kLoadAddress, word));
        state.pc = kLoadAddress;
        const StepResult result = interpreter.step();
        INFO("executing " << disassemble(word, kLoadAddress) << " -> " << step_result_name(result));
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

    /// Run until the program returns to the sentinel, or something goes wrong.
    StepResult run_to_return(u64 budget = 10000) {
        for (u64 i = 0; i < budget; ++i) {
            if (state.pc == kReturnSentinel) {
                return StepResult::Ok;
            }
            const StepResult result = interpreter.step();
            if (result != StepResult::Ok) {
                INFO("stopped at " << std::to_string(state.pc) << ": "
                                   << disassemble(interpreter.last_instruction().raw, state.pc));
                return result;
            }
        }
        FAIL("program did not return within budget");
        return StepResult::Ok;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Golden vectors -- see scripts/gen_cpu_fixtures.py
// ---------------------------------------------------------------------------

TEST_CASE("rotate masks match an independent implementation", "[cpu][interp]") {
    // Including every MB > ME wrapping case, which is where a hand-rolled mask
    // usually goes wrong: it produces a plausible value rather than a crash.
    for (u32 mb = 0; mb < 32; ++mb) {
        for (u32 me = 0; me < 32; ++me) {
            INFO("mb=" << mb << " me=" << me);
            CHECK(rotate_mask(mb, me) == kRotateMasks[mb * 32 + me]);
        }
    }
}

TEST_CASE_METHOD(CpuFixture, "addc carry matches the reference", "[cpu][interp]") {
    for (const AddVector& vector : kAddVectors) {
        if (vector.carry_in != 0) {
            continue; // addc ignores carry in
        }
        INFO(std::to_string(vector.a) << " + " << std::to_string(vector.b));

        state.gpr[4] = vector.a;
        state.gpr[5] = vector.b;
        // addc r3,r4,r5
        REQUIRE(exec(kEnc_addc) == StepResult::Ok);

        CHECK(state.gpr[3] == vector.sum);
        CHECK(state.xer.carry() == (vector.carry_out != 0));
    }
}

TEST_CASE_METHOD(CpuFixture, "adde threads carry through", "[cpu][interp]") {
    for (const AddVector& vector : kAddVectors) {
        state.gpr[4] = vector.a;
        state.gpr[5] = vector.b;
        state.xer.set_carry(vector.carry_in != 0);
        // adde r3,r4,r5
        REQUIRE(exec(kEnc_adde) == StepResult::Ok);

        CHECK(state.gpr[3] == vector.sum);
        CHECK(state.xer.carry() == (vector.carry_out != 0));
    }
}

TEST_CASE_METHOD(CpuFixture, "addo sets overflow where the reference says", "[cpu][interp]") {
    for (const AddVector& vector : kAddVectors) {
        if (vector.carry_in != 0) {
            continue;
        }
        state.xer.value = 0; // clear the sticky summary bit between cases
        state.gpr[4] = vector.a;
        state.gpr[5] = vector.b;
        // addo r3,r4,r5
        REQUIRE(exec(kEnc_addo) == StepResult::Ok);

        CHECK(state.gpr[3] == vector.sum);
        CHECK(state.xer.overflow() == (vector.overflow != 0));
        // SO is sticky: once set by OV it stays until software clears XER.
        CHECK(state.xer.summary_overflow() == (vector.overflow != 0));
    }
}

TEST_CASE_METHOD(CpuFixture, "srawi result and carry match the reference", "[cpu][interp]") {
    for (const ShiftVector& vector : kShiftVectors) {
        INFO("value=" << std::to_string(vector.value) << " shift=" << vector.shift);

        state.gpr[3] = vector.value;
        // srawi r4,r3,SH
        const u32 word = kEnc_srawi_base | (vector.shift << 11);
        REQUIRE(exec(word) == StepResult::Ok);

        CHECK(state.gpr[4] == vector.result);
        CHECK(state.xer.carry() == (vector.carry != 0));
    }
}

// ---------------------------------------------------------------------------
// Semantics
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CpuFixture, "subf subtracts rA from rB, not the other way", "[cpu][interp]") {
    // The operand order reads backwards from the mnemonic, and getting it
    // wrong is a classic PowerPC mistake that produces sign-flipped results.
    state.gpr[4] = 10;
    state.gpr[5] = 3;
    REQUIRE(exec(kEnc_subf) == StepResult::Ok);  // subf r3,r4,r5
    CHECK(static_cast<s32>(state.gpr[3]) == -7); // 3 - 10
}

TEST_CASE_METHOD(CpuFixture, "rA of zero means literal zero for addi", "[cpu][interp]") {
    state.gpr[0] = 0xDEADBEEF;
    REQUIRE(exec(kEnc_li_r3_5) == StepResult::Ok); // addi r3,r0,5 == li r3,5
    CHECK(state.gpr[3] == 5U);                     // not 0xDEADBEEF + 5
}

TEST_CASE_METHOD(CpuFixture, "shifts of 32 or more clear the register", "[cpu][interp]") {
    // The shift amount is six bits, so 32 shifts everything out rather than
    // wrapping to zero the way a bare C shift would.
    state.gpr[3] = 0xFFFFFFFF;
    state.gpr[5] = 32;
    REQUIRE(exec(kEnc_slw) == StepResult::Ok); // slw r4,r3,r5
    CHECK(state.gpr[4] == 0U);

    state.gpr[5] = 32;
    REQUIRE(exec(kEnc_srw) == StepResult::Ok); // srw r4,r3,r5
    CHECK(state.gpr[4] == 0U);
}

TEST_CASE_METHOD(CpuFixture, "record bit updates CR0", "[cpu][interp]") {
    state.gpr[4] = 5;
    state.gpr[5] = 5;
    REQUIRE(exec(kEnc_subf_rc) == StepResult::Ok); // subf. r3,r4,r5
    CHECK(state.gpr[3] == 0U);
    CHECK(state.cr.field(0) == ConditionRegister::eq);

    state.gpr[4] = 10;
    state.gpr[5] = 3;
    REQUIRE(exec(kEnc_subf_rc) == StepResult::Ok);
    CHECK(state.cr.field(0) == ConditionRegister::lt);

    state.gpr[4] = 3;
    state.gpr[5] = 10;
    REQUIRE(exec(kEnc_subf_rc) == StepResult::Ok);
    CHECK(state.cr.field(0) == ConditionRegister::gt);
}

TEST_CASE_METHOD(CpuFixture, "CR0 carries the sticky summary overflow bit", "[cpu][interp]") {
    state.xer.set_overflow(true); // sets SO too
    state.gpr[4] = 1;
    state.gpr[5] = 1;
    REQUIRE(exec(kEnc_subf_rc) == StepResult::Ok); // subf. r3,r4,r5 -> 0
    CHECK((state.cr.field(0) & ConditionRegister::eq) != 0U);
    CHECK((state.cr.field(0) & ConditionRegister::so) != 0U);
}

TEST_CASE_METHOD(CpuFixture, "signed and unsigned compares differ", "[cpu][interp]") {
    state.gpr[3] = 0xFFFFFFFF; // -1 signed, 4294967295 unsigned
    state.gpr[4] = 1;

    REQUIRE(exec(kEnc_cmp) == StepResult::Ok); // cmp 0,0,r3,r4
    CHECK(state.cr.field(0) == ConditionRegister::lt);

    REQUIRE(exec(kEnc_cmpl) == StepResult::Ok); // cmpl 0,0,r3,r4
    CHECK(state.cr.field(0) == ConditionRegister::gt);
}

TEST_CASE_METHOD(CpuFixture, "divide by zero does not crash", "[cpu][interp]") {
    state.gpr[4] = 100;
    state.gpr[5] = 0;
    REQUIRE(exec(kEnc_divw) == StepResult::Ok); // divw r3,r4,r5
    CHECK(state.gpr[3] == 0U);

    // INT_MIN / -1 is the other undefined case.
    state.gpr[4] = 0x80000000;
    state.gpr[5] = 0xFFFFFFFF;
    REQUIRE(exec(kEnc_divw) == StepResult::Ok);
    CHECK(state.gpr[3] == 0U);
}

TEST_CASE_METHOD(CpuFixture, "loads and stores round-trip through memory", "[cpu][interp]") {
    state.gpr[4] = memory::data_heap_base;

    state.gpr[3] = 0x12345678;
    REQUIRE(exec(kEnc_stw_16_r4) == StepResult::Ok); // stw r3,16(r4)
    CHECK(memory.read<u32>(memory::data_heap_base + 16) == 0x12345678U);

    state.gpr[3] = 0;
    REQUIRE(exec(kEnc_lwz_16_r4) == StepResult::Ok); // lwz r3,16(r4)
    CHECK(state.gpr[3] == 0x12345678U);

    // A halfword load sees the top half, because the guest is big-endian.
    REQUIRE(exec(kEnc_lhz_16_r4) == StepResult::Ok); // lhz r3,16(r4)
    CHECK(state.gpr[3] == 0x1234U);

    REQUIRE(exec(kEnc_lbz_16_r4) == StepResult::Ok); // lbz r3,16(r4)
    CHECK(state.gpr[3] == 0x12U);
}

TEST_CASE_METHOD(CpuFixture, "algebraic loads sign-extend", "[cpu][interp]") {
    state.gpr[4] = memory::data_heap_base;
    REQUIRE(memory.try_write<u16>(memory::data_heap_base, 0xFFFE));

    REQUIRE(exec(kEnc_lha_0_r4) == StepResult::Ok); // lha r3,0(r4)
    CHECK(static_cast<s32>(state.gpr[3]) == -2);

    REQUIRE(exec(kEnc_lhz_0_r4) == StepResult::Ok); // lhz r3,0(r4)
    CHECK(state.gpr[3] == 0xFFFEU);
}

TEST_CASE_METHOD(CpuFixture, "byte-reversed accesses swap", "[cpu][interp]") {
    state.gpr[4] = memory::data_heap_base;
    state.gpr[5] = 0;
    REQUIRE(memory.try_write<u32>(memory::data_heap_base, 0x12345678));

    REQUIRE(exec(kEnc_lwbrx) == StepResult::Ok); // lwbrx r3,r4,r5
    CHECK(state.gpr[3] == 0x78563412U);
}

TEST_CASE_METHOD(CpuFixture, "update forms write back the address", "[cpu][interp]") {
    state.gpr[1] = memory::data_heap_base + 0x100;
    state.gpr[3] = 0xAABBCCDD;

    REQUIRE(exec(kEnc_stwu_neg32) == StepResult::Ok); // stwu r3,-32(r1)
    CHECK(state.gpr[1] == memory::data_heap_base + 0x100 - 32);
    CHECK(memory.read<u32>(state.gpr[1]) == 0xAABBCCDDU);
}

TEST_CASE_METHOD(CpuFixture, "an unmapped access is a memory fault, not a crash", "[cpu][interp]") {
    state.gpr[4] = 0x70000000; // the gap in the address space
    CHECK(exec(kEnc_lwz_0_r4) == StepResult::MemoryFault);
}

TEST_CASE_METHOD(CpuFixture, "lmw and stmw move register ranges", "[cpu][interp]") {
    for (u32 i = 28; i < 32; ++i) {
        state.gpr[i] = 0x1000 + i;
    }
    state.gpr[4] = memory::data_heap_base;

    REQUIRE(exec(kEnc_stmw_r28) == StepResult::Ok); // stmw r28,0(r4)
    CHECK(memory.read<u32>(memory::data_heap_base) == 0x101CU);
    CHECK(memory.read<u32>(memory::data_heap_base + 12) == 0x101FU);

    for (u32 i = 28; i < 32; ++i) {
        state.gpr[i] = 0;
    }
    REQUIRE(exec(kEnc_lmw_r28) == StepResult::Ok); // lmw r28,0(r4)
    CHECK(state.gpr[28] == 0x101CU);
    CHECK(state.gpr[31] == 0x101FU);
}

TEST_CASE_METHOD(CpuFixture, "lwarx and stwcx. pair up", "[cpu][interp]") {
    state.gpr[4] = memory::data_heap_base;
    state.gpr[5] = 0;
    REQUIRE(memory.try_write<u32>(memory::data_heap_base, 7));

    REQUIRE(exec(kEnc_lwarx) == StepResult::Ok); // lwarx r3,r4,r5
    CHECK(state.gpr[3] == 7U);
    CHECK(state.reserved);

    state.gpr[3] = 99;
    REQUIRE(exec(kEnc_stwcx) == StepResult::Ok); // stwcx. r3,r4,r5
    CHECK(memory.read<u32>(memory::data_heap_base) == 99U);
    CHECK((state.cr.field(0) & ConditionRegister::eq) != 0U); // succeeded
    CHECK_FALSE(state.reserved);

    // Without a reservation the store does nothing and CR0[EQ] stays clear.
    state.gpr[3] = 123;
    REQUIRE(exec(kEnc_stwcx) == StepResult::Ok);
    CHECK(memory.read<u32>(memory::data_heap_base) == 99U);
    CHECK((state.cr.field(0) & ConditionRegister::eq) == 0U);
}

TEST_CASE_METHOD(CpuFixture, "cache instructions drive the coherency model", "[cpu][interp]") {
    // dcbf is how a guest publishes a write to the GPU, so it has to reach the
    // Phase 3.6 listener rather than being a no-op.
    int flushes = 0;
    memory.set_cache_listener([&flushes](memory::CacheOp op, ppcaddr_t, u32) {
        if (op == memory::CacheOp::Flush) {
            ++flushes;
        }
    });

    state.gpr[4] = memory::data_heap_base;
    state.gpr[5] = 0;
    REQUIRE(exec(kEnc_dcbf) == StepResult::Ok); // dcbf r4,r5
    CHECK(flushes == 1);
}

TEST_CASE_METHOD(CpuFixture, "branches and the link register", "[cpu][interp]") {
    state.pc = kLoadAddress;
    REQUIRE(memory.try_write<u32>(kLoadAddress, kEnc_bl_0x100)); // bl +0x100
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == kLoadAddress + 0x100);
    CHECK(state.lr == kLoadAddress + 4);

    // blr returns through LR, ignoring its low two bits.
    state.lr = kLoadAddress + 0x42;
    REQUIRE(exec(kEnc_blr) == StepResult::Ok);
    CHECK(state.pc == kLoadAddress + 0x40);
}

TEST_CASE_METHOD(CpuFixture, "conditional branches honour BO and BI", "[cpu][interp]") {
    // beq +0x20 == bc 12,2,+0x20
    state.cr.set_field(0, ConditionRegister::eq);
    state.pc = kLoadAddress;
    REQUIRE(memory.try_write<u32>(kLoadAddress, kEnc_beq_0x20));
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == kLoadAddress + 0x20);

    // Not equal: falls through.
    state.cr.set_field(0, ConditionRegister::gt);
    state.pc = kLoadAddress;
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == kLoadAddress + 4);
}

TEST_CASE_METHOD(CpuFixture, "bdnz decrements CTR and branches while non-zero", "[cpu][interp]") {
    state.ctr = 3;
    state.pc = kLoadAddress;
    REQUIRE(memory.try_write<u32>(kLoadAddress, kEnc_bdnz_back4)); // bdnz -4

    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.ctr == 2U);
    CHECK(state.pc == kLoadAddress - 4);

    state.pc = kLoadAddress;
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.ctr == 1U);

    state.pc = kLoadAddress;
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.ctr == 0U);
    CHECK(state.pc == kLoadAddress + 4); // falls through at zero
}

TEST_CASE_METHOD(CpuFixture, "SPR moves reach the right registers", "[cpu][interp]") {
    state.gpr[3] = 0x1234;
    REQUIRE(exec(kEnc_mtlr_r3) == StepResult::Ok); // mtlr r3
    CHECK(state.lr == 0x1234U);

    REQUIRE(exec(kEnc_mflr_r4) == StepResult::Ok); // mflr r4
    CHECK(state.gpr[4] == 0x1234U);

    state.gpr[3] = 0x5678;
    REQUIRE(exec(kEnc_mtctr_r3) == StepResult::Ok); // mtctr r3
    CHECK(state.ctr == 0x5678U);
}

TEST_CASE_METHOD(CpuFixture, "sc stops for the HLE layer", "[cpu][interp]") {
    CHECK(exec(kEnc_sc) == StepResult::SystemCall);
    // pc has already advanced, so resuming continues after the call.
    CHECK(state.pc == kLoadAddress + 4);
}

TEST_CASE_METHOD(CpuFixture,
                 "every instruction in the table is dispatched",
                 "[cpu][interp][coverage]") {
    // Stronger than spot-checking: execute every entry in the table and
    // require that none reports Unimplemented or Illegal. An instruction the
    // decoder knows about but the interpreter silently ignores is the failure
    // mode this exists to prevent -- it looks like a wrong result, not a
    // missing feature.
    //
    // Operands are whatever the bare pattern leaves, so memory accesses land
    // at address 0 and fault; that is a legitimate outcome, just not a gap.
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    for (const OpcodeInfo& info : opcode_table()) {
        INFO("opcode " << info.name);

        state.halted = false;
        const StepResult result = exec(info.pattern);

        CHECK(result != StepResult::Unimplemented);
        CHECK(result != StepResult::IllegalInstruction);
    }
}

// ---------------------------------------------------------------------------
// Whole programs, assembled by binutils
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CpuFixture, "a loop summing 1..10", "[cpu][interp][program]") {
    load(kProgram_sum_to_ten);
    REQUIRE(run_to_return() == StepResult::Ok);
    CHECK(state.gpr[3] == 55U);
}

TEST_CASE_METHOD(CpuFixture, "a CTR countdown loop", "[cpu][interp][program]") {
    load(kProgram_countdown_ctr);
    REQUIRE(run_to_return() == StepResult::Ok);
    CHECK(state.gpr[3] == 20U);
    CHECK(state.ctr == 0U);
}

TEST_CASE_METHOD(CpuFixture, "a byte copy loop", "[cpu][interp][program]") {
    const ppcaddr_t source = memory::data_heap_base;
    const ppcaddr_t destination = memory::data_heap_base + 0x100;
    for (u32 i = 0; i < 8; ++i) {
        REQUIRE(memory.try_write<u8>(source + i, static_cast<u8>(0xA0 + i)));
    }

    load(kProgram_copy_bytes);
    state.gpr[3] = source;
    state.gpr[4] = destination;
    REQUIRE(run_to_return() == StepResult::Ok);

    for (u32 i = 0; i < 8; ++i) {
        INFO("byte " << i);
        CHECK(memory.read<u8>(destination + i) == 0xA0 + i);
    }
}

TEST_CASE_METHOD(CpuFixture, "a 64-bit add threads carry between words", "[cpu][interp][program]") {
    load(kProgram_add64);
    // 0x00000001_FFFFFFFF + 0x00000000_00000001 == 0x00000002_00000000
    state.gpr[3] = 0x00000001; // high
    state.gpr[4] = 0xFFFFFFFF; // low
    state.gpr[5] = 0x00000000;
    state.gpr[6] = 0x00000001;

    REQUIRE(run_to_return() == StepResult::Ok);
    CHECK(state.gpr[3] == 0x00000002U);
    CHECK(state.gpr[4] == 0x00000000U);
}

TEST_CASE_METHOD(CpuFixture, "a call and return with a stack frame", "[cpu][interp][program]") {
    load(kProgram_call_and_return);
    state.gpr[1] = memory::data_heap_base + 0x1000; // stack pointer
    state.gpr[3] = 100;

    REQUIRE(run_to_return() == StepResult::Ok);
    CHECK(state.gpr[3] == 142U);
    // The stack pointer is restored and LR made it back to the sentinel.
    CHECK(state.gpr[1] == memory::data_heap_base + 0x1000);
    CHECK(state.pc == kReturnSentinel);
}

TEST_CASE_METHOD(CpuFixture, "bitfield operations on a real value", "[cpu][interp][program]") {
    load(kProgram_bitfield_ops);
    state.gpr[3] = 0x89ABCDEF;

    REQUIRE(run_to_return() == StepResult::Ok);

    CHECK(state.gpr[4] == (rotate_left(0x89ABCDEF, 8) & rotate_mask(8, 20)));
    CHECK(state.gpr[5] == static_cast<u32>(static_cast<s32>(0x89ABCDEF) >> 4));
    CHECK(state.gpr[6] == 0xFFFFFFEFU); // extsb of 0xEF
    CHECK(state.gpr[7] == 0U);          // cntlzw: the top bit is set
}
