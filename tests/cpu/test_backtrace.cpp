#include <wiiu/cpu/backtrace.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace wiiu;
using namespace wiiu::cpu;

namespace {

constexpr ppcaddr_t kStackTop = memory::data_heap_base + 0x10000;

struct BacktraceFixture {
    memory::Memory memory;
    CoreState state;

    BacktraceFixture() {
        std::string error;
        REQUIRE(memory.initialize(error));
    }

    /// Build a frame at `frame` linking to `previous`, storing `return_to` as
    /// the return address of the function that owns `previous`.
    void link(ppcaddr_t frame, ppcaddr_t previous, ppcaddr_t return_to) {
        REQUIRE(memory.try_write<u32>(frame, previous));
        REQUIRE(memory.try_write<u32>(previous + 4, return_to));
    }
};

} // namespace

TEST_CASE_METHOD(BacktraceFixture,
                 "the innermost return address comes from LR",
                 "[cpu][backtrace]") {
    // A function's prologue stores LR into its caller's frame, and at the
    // point of a fault it may not have run yet -- so the register is the only
    // reliable source for the top frame.
    state.gpr[1] = kStackTop;
    state.lr = 0x02000100;
    REQUIRE(memory.try_write<u32>(kStackTop, 0)); // null back-chain

    const auto frames = backtrace(state, memory);
    REQUIRE(frames.size() == 1);
    CHECK(frames[0].return_to == 0x02000100U);
    CHECK(frames[0].frame == kStackTop);
}

TEST_CASE_METHOD(BacktraceFixture, "a chain of frames is walked", "[cpu][backtrace]") {
    const ppcaddr_t inner = kStackTop - 0x60;
    const ppcaddr_t middle = kStackTop - 0x40;
    const ppcaddr_t outer = kStackTop - 0x20;

    link(inner, middle, 0x02000200);
    link(middle, outer, 0x02000300);
    REQUIRE(memory.try_write<u32>(outer, 0)); // end of the stack

    state.gpr[1] = inner;
    state.lr = 0x02000100;

    const auto frames = backtrace(state, memory);
    REQUIRE(frames.size() == 3);
    CHECK(frames[0].return_to == 0x02000100U); // from LR
    CHECK(frames[1].return_to == 0x02000200U);
    CHECK(frames[2].return_to == 0x02000300U);
}

TEST_CASE_METHOD(BacktraceFixture,
                 "a corrupted chain stops rather than inventing frames",
                 "[cpu][backtrace]") {
    // A crash is exactly when the stack is likely to be damaged, and a
    // plausible-looking stack of nonsense is worse than a short true one.
    state.gpr[1] = kStackTop;
    state.lr = 0x02000100;

    SECTION("a back-chain that points downward") {
        REQUIRE(memory.try_write<u32>(kStackTop, kStackTop - 0x100));
        CHECK(backtrace(state, memory).size() == 1);
    }

    SECTION("a back-chain that points at itself") {
        REQUIRE(memory.try_write<u32>(kStackTop, kStackTop));
        CHECK(backtrace(state, memory).size() == 1);
    }

    SECTION("an absurd frame size") {
        REQUIRE(memory.try_write<u32>(kStackTop, kStackTop + 0x1000000));
        CHECK(backtrace(state, memory).size() == 1);
    }

    SECTION("an unaligned stack pointer") {
        state.gpr[1] = kStackTop + 1;
        CHECK(backtrace(state, memory).size() == 1);
    }

    SECTION("a back-chain into unmapped memory") {
        REQUIRE(memory.try_write<u32>(kStackTop, 0x70000000));
        CHECK(backtrace(state, memory).size() == 1);
    }
}

TEST_CASE_METHOD(BacktraceFixture, "a cyclic chain terminates", "[cpu][backtrace]") {
    // Two frames pointing at each other would loop forever without the
    // must-move-upward rule.
    const ppcaddr_t a = kStackTop - 0x40;
    const ppcaddr_t b = kStackTop - 0x20;
    REQUIRE(memory.try_write<u32>(a, b));
    REQUIRE(memory.try_write<u32>(b, a));
    REQUIRE(memory.try_write<u32>(b + 4, 0x02000200));

    state.gpr[1] = a;
    state.lr = 0x02000100;

    const auto frames = backtrace(state, memory);
    CHECK(frames.size() <= 2);
}

TEST_CASE_METHOD(BacktraceFixture, "the walk is bounded", "[cpu][backtrace]") {
    // A long but valid chain still has to stop somewhere.
    ppcaddr_t frame = memory::data_heap_base;
    for (u32 i = 0; i < 200; ++i) {
        const ppcaddr_t next = frame + 0x20;
        REQUIRE(memory.try_write<u32>(frame, next));
        REQUIRE(memory.try_write<u32>(next + 4, 0x02000000 + i * 4));
        frame = next;
    }

    state.gpr[1] = memory::data_heap_base;
    state.lr = 0x02000100;

    const auto frames = backtrace(state, memory, {.max_frames = 16});
    CHECK(frames.size() == 16);
}

TEST_CASE_METHOD(BacktraceFixture,
                 "the state dump shows code, registers and the stack",
                 "[cpu][backtrace]") {
    REQUIRE(memory.try_write<u32>(0x02000000, 0x7C0802A6)); // mflr r0
    REQUIRE(memory.try_write<u32>(0x02000004, 0x9421FFE0)); // stwu r1,-32(r1)

    state.pc = 0x02000004;
    state.gpr[1] = kStackTop;
    state.gpr[3] = 0xDEADBEEF;
    state.lr = 0x02000100;
    REQUIRE(memory.try_write<u32>(kStackTop, 0));

    const std::string dump = format_state_dump(state, memory);

    CHECK(dump.find("stopped at 0x02000004") != std::string::npos);
    CHECK(dump.find("stwu r1,-32(r1)") != std::string::npos); // disassembled
    CHECK(dump.find("->02000004") != std::string::npos);      // and marked
    CHECK(dump.find("mflr r0") != std::string::npos);         // with context
    CHECK(dump.find("0xdeadbeef") != std::string::npos);      // registers
    CHECK(dump.find("call stack") != std::string::npos);
}

TEST_CASE_METHOD(BacktraceFixture, "unmapped code is reported, not skipped", "[cpu][backtrace]") {
    // The window around a fault often runs off the end of a mapping, and
    // saying so is more useful than printing nothing.
    state.pc = memory::data_heap_base;
    state.gpr[1] = kStackTop;
    REQUIRE(memory.try_write<u32>(kStackTop, 0));

    // 0x70000000 is the gap between the data heap and the hardware area.
    const std::string window = format_code_window(memory, 0x70000000, 2, 2);
    CHECK(window.find("<unmapped>") != std::string::npos);
}
