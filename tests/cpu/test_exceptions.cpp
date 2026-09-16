#include <wiiu/common/log.hpp>
#include <wiiu/cpu/exceptions.hpp>
#include <wiiu/cpu/interpreter.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::cpu;

namespace {

#include "fixtures.inc" // NOLINT(bugprone-suspicious-include)

constexpr ppcaddr_t kLoadAddress = 0x02000000;

struct ExceptionFixture {
    memory::Memory memory;
    CoreState state;
    Interpreter interpreter{state, memory};

    ExceptionFixture() {
        // The Cafe OS map has nothing at low addresses, so the exception
        // vectors are not mapped by default -- a real finding: the loader has
        // to place the vector page before Deliver mode is usable at all.
        std::vector<memory::Region> regions(memory::default_regions().begin(),
                                            memory::default_regions().end());
        regions.insert(regions.begin(),
                       memory::Region{"vectors",
                                      0x00000000,
                                      0x4000,
                                      0,
                                      memory::Backing::None,
                                      memory::read_write_execute});

        std::string error;
        const bool ok = memory.initialize(regions, error);
        INFO("initialize: " << error);
        REQUIRE(ok);
        state.pc = kLoadAddress;
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        interpreter.set_exception_mode(ExceptionMode::Deliver);
    }

    StepResult exec(u32 word) {
        REQUIRE(memory.try_write<u32>(kLoadAddress, word));
        state.pc = kLoadAddress;
        return interpreter.step();
    }
};

} // namespace

TEST_CASE("the vector base follows MSR[IP]", "[cpu][exception]") {
    STATIC_REQUIRE(exception_base(0) == 0x00000000U);
    STATIC_REQUIRE(exception_base(MsrBits::ip) == 0xFFF00000U);
}

TEST_CASE_METHOD(ExceptionFixture, "reporting is the default", "[cpu][exception]") {
    Interpreter fresh{state, memory};
    CHECK(fresh.exception_mode() == ExceptionMode::Report);
}

TEST_CASE_METHOD(ExceptionFixture,
                 "a bad load becomes a data storage exception",
                 "[cpu][exception]") {
    state.msr = MsrBits::ee | MsrBits::pr | MsrBits::dr;
    state.gpr[4] = 0x70000000; // unmapped

    REQUIRE(exec(kEnc_lwz_0_r4) == StepResult::Ok); // delivered, not reported
    CHECK(state.pc == static_cast<u32>(Exception::Dsi));
    CHECK(state.srr0 == kLoadAddress); // retryable: points at the faulting insn
    CHECK(state.dar == 0x70000000U);
    CHECK((state.dsisr & DsisrBits::not_translated) != 0U);
    CHECK((state.dsisr & DsisrBits::is_store) == 0U);

    // The handler runs with interrupts and user mode off.
    CHECK((state.msr & MsrBits::ee) == 0U);
    CHECK((state.msr & MsrBits::pr) == 0U);
}

TEST_CASE_METHOD(ExceptionFixture, "a bad store is flagged as a store", "[cpu][exception]") {
    state.gpr[4] = 0x70000000;
    REQUIRE(exec(kEnc_stw_16_r4) == StepResult::Ok);
    CHECK((state.dsisr & DsisrBits::is_store) != 0U);
}

TEST_CASE_METHOD(ExceptionFixture,
                 "a protection violation is distinguishable",
                 "[cpu][exception]") {
    // Writing read-only memory is mapped-but-not-permitted, which the handler
    // has to tell apart from nothing being mapped at all.
    state.gpr[4] = memory::shared_data_base;
    REQUIRE(exec(kEnc_stw_16_r4) == StepResult::Ok);
    CHECK((state.dsisr & DsisrBits::protection) != 0U);
    CHECK((state.dsisr & DsisrBits::not_translated) == 0U);
}

TEST_CASE_METHOD(ExceptionFixture,
                 "a failed fetch is an instruction storage exception",
                 "[cpu][exception]") {
    // Not a data exception: the address that failed was the program counter.
    state.pc = 0x70000000;
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Isi));
    CHECK(state.srr0 == 0x70000000U);
}

TEST_CASE_METHOD(ExceptionFixture,
                 "an illegal instruction is a program exception",
                 "[cpu][exception]") {
    REQUIRE(exec(0x00000000) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Program));
    CHECK((state.srr1 & ProgramCause::illegal_instruction) != 0U);
}

TEST_CASE_METHOD(ExceptionFixture, "a taken trap is a program exception", "[cpu][exception]") {
    // twi 31,r3,0 traps unconditionally when the comparison holds.
    state.gpr[3] = 0;
    const u32 twi_always = 0x0FE30000U; // twi 31,r3,0
    REQUIRE(exec(twi_always) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Program));
    CHECK((state.srr1 & ProgramCause::trap) != 0U);
}

TEST_CASE_METHOD(ExceptionFixture, "rfi returns to the saved address", "[cpu][exception]") {
    state.msr = MsrBits::ee | MsrBits::pr;
    state.gpr[4] = 0x70000000;
    REQUIRE(exec(kEnc_lwz_0_r4) == StepResult::Ok);

    const u32 saved_msr = state.srr1;
    const ppcaddr_t saved_pc = state.srr0;

    // rfi restores both, so execution resumes at the faulting instruction with
    // the original machine state.
    const u32 rfi = 0x4C000064U;
    REQUIRE(memory.try_write<u32>(state.pc, rfi));
    REQUIRE(interpreter.step() == StepResult::Ok);

    CHECK(state.pc == saved_pc);
    CHECK(state.msr == saved_msr);
}

// ---------------------------------------------------------------------------
// Asynchronous interrupts
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(ExceptionFixture, "an external interrupt waits for MSR[EE]", "[cpu][exception]") {
    REQUIRE(memory.try_write<u32>(kLoadAddress, kEnc_li_r3_5));
    state.msr = 0; // interrupts disabled
    interpreter.set_external_interrupt(true);

    // Disabled: the instruction runs and the interrupt stays pending rather
    // than being dropped.
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.gpr[3] == 5U);
    CHECK(interpreter.external_interrupt_pending());

    // Enabled: delivered before the next instruction.
    state.msr = MsrBits::ee;
    state.pc = kLoadAddress;
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::ExternalInterrupt));
    CHECK_FALSE(interpreter.external_interrupt_pending());
}

TEST_CASE_METHOD(ExceptionFixture,
                 "the decrementer fires when it goes negative",
                 "[cpu][exception]") {
    state.decrementer = 5;
    interpreter.advance_time(4);
    CHECK(state.decrementer == 1U);
    CHECK_FALSE(interpreter.decrementer_pending());

    // Stepping past zero, not landing on it, is what signals. A check for
    // "reached zero" would miss this entirely.
    interpreter.advance_time(4);
    CHECK(state.decrementer == 0xFFFFFFFDU);
    CHECK(interpreter.decrementer_pending());
}

TEST_CASE_METHOD(ExceptionFixture,
                 "a decrementer interrupt vectors once enabled",
                 "[cpu][exception]") {
    REQUIRE(memory.try_write<u32>(kLoadAddress, kEnc_li_r3_5));
    state.msr = MsrBits::ee;
    state.decrementer = 1;
    interpreter.advance_time(2);
    REQUIRE(interpreter.decrementer_pending());

    state.pc = kLoadAddress;
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Decrementer));
    CHECK_FALSE(interpreter.decrementer_pending());
}

// ---------------------------------------------------------------------------
// Timebase
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(ExceptionFixture, "the timebase advances monotonically", "[cpu][time]") {
    REQUIRE(memory.try_write<u32>(kLoadAddress, kEnc_li_r3_5));

    const u64 start = state.timebase;
    for (int i = 0; i < 10; ++i) {
        state.pc = kLoadAddress;
        REQUIRE(interpreter.step() == StepResult::Ok);
    }
    CHECK(state.timebase == start + 10);
}

TEST_CASE_METHOD(ExceptionFixture, "the tick rate is configurable", "[cpu][time]") {
    REQUIRE(memory.try_write<u32>(kLoadAddress, kEnc_li_r3_5));
    interpreter.set_ticks_per_instruction(4);

    const u64 start = state.timebase;
    state.pc = kLoadAddress;
    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.timebase == start + 4);
}

TEST_CASE("the timebase runs at a quarter of the core clock", "[cpu][time]") {
    STATIC_REQUIRE(cpu_clock_hz == 1243125000ULL);
    STATIC_REQUIRE(bus_clock_hz == cpu_clock_hz / 5);
    STATIC_REQUIRE(timebase_hz == bus_clock_hz / 4);
    STATIC_REQUIRE(timebase_hz == 62156250);
}

TEST_CASE_METHOD(ExceptionFixture, "mftb reads the running timebase", "[cpu][time]") {
    state.timebase = 0x0000000123456789ULL;

    const u32 mftb_r3 = 0x7C6C42E6U;
    REQUIRE(exec(mftb_r3) == StepResult::Ok);
    // The instruction itself ticks the clock, so allow for that.
    CHECK(state.gpr[3] == 0x23456789U);
}
