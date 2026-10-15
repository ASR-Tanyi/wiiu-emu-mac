// Phase 8.2 — the GPU's timeline (ADR-004).
//
// What is being guarded is when things become visible. Nothing a submission
// does may show before its deadline, everything it does must show at it, and
// the same stream submitted at the same tick must write the same bytes on
// every run -- which is the whole reason the GPU runs on the guest's clock.

#include <wiiu/common/log.hpp>
#include <wiiu/latte/timeline.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;
using pm4::Opcode;
using pm4::type3_header;

namespace {

constexpr u64 kLatency = 1000;
constexpr ppcaddr_t kMem1 = memory::mem1_base;

struct TimelineFixture {
    memory::Memory memory;
    std::unique_ptr<RegisterFile> registers = std::make_unique<RegisterFile>();
    CommandProcessor processor{*registers};
    GpuTimeline timeline{processor, kLatency};

    TimelineFixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        processor.set_memory(&memory);
    }

    /// A stream that writes `value` to the context register at `offset`.
    [[nodiscard]] static std::vector<u32> set_context(u32 offset, u32 value) {
        return {type3_header(Opcode::SetContextReg, 2), offset, value};
    }

    [[nodiscard]] u32 context(u32 offset) const { return registers->read(0x28000 + (offset * 4)); }
};

} // namespace

TEST_CASE_METHOD(TimelineFixture, "nothing shows before the deadline", "[latte][timeline]") {
    const u64 timestamp = timeline.submit(set_context(0x10, 7), 5000);
    CHECK(timestamp == 1);
    CHECK(timeline.submitted() == 1);
    CHECK(timeline.retired() == 0);
    REQUIRE(timeline.next_deadline().has_value());
    CHECK(*timeline.next_deadline() == 5000 + kLatency);

    CHECK(timeline.retire_until(5000 + kLatency - 1) == 0);
    CHECK(context(0x10) == 0);
    CHECK(timeline.retired() == 0);

    CHECK(timeline.retire_until(5000 + kLatency) == 1);
    CHECK(context(0x10) == 7);
    CHECK(timeline.retired() == 1);
    CHECK(timeline.pending() == 0);
    CHECK_FALSE(timeline.next_deadline().has_value());
}

TEST_CASE_METHOD(TimelineFixture, "submissions retire in order", "[latte][timeline]") {
    // Three streams writing the same register: whatever retires last wins,
    // so the register says which order they ran in.
    static_cast<void>(timeline.submit(set_context(0x10, 1), 100));
    static_cast<void>(timeline.submit(set_context(0x10, 2), 200));

    // Submitted at an *earlier* tick than the one ahead of it -- a second core
    // a little behind. It still retires after it, not before.
    const u64 third = timeline.submit(set_context(0x10, 3), 50);
    REQUIRE(timeline.deadline_of(third).has_value());
    CHECK(*timeline.deadline_of(third) == 200 + kLatency);

    CHECK(timeline.retire_until(100 + kLatency) == 1);
    CHECK(context(0x10) == 1);
    CHECK(timeline.retire_until(1'000'000) == 2);
    CHECK(context(0x10) == 3);
    CHECK(timeline.retired() == 3);
}

TEST_CASE_METHOD(TimelineFixture, "a submission is copied when it is made", "[latte][timeline]") {
    // GX2 reuses its command buffer as soon as it has submitted it. What the
    // GPU runs is what was there at submission.
    std::vector<u32> stream = set_context(0x20, 0xAAAA);
    static_cast<void>(timeline.submit(stream, 0));
    stream[2] = 0xBBBB;

    static_cast<void>(timeline.retire_until(kLatency));
    CHECK(context(0x20) == 0xAAAA);
}

TEST_CASE_METHOD(TimelineFixture, "the GPU clock is the retirement tick", "[latte][timeline]") {
    // A MEM_WRITE of the clock, 64-bit, swapped to land big-endian.
    const std::vector<u32> stream{type3_header(Opcode::MemWrite, 4), 0x5000 | 3, 1U << 16, 0, 0};
    static_cast<void>(timeline.submit(stream, 12'345));

    // Retired late -- the caller looked long after the deadline. The value
    // written is still the deadline, not the moment someone looked: that is
    // what makes two runs agree.
    static_cast<void>(timeline.retire_until(99'999'999));
    CHECK(memory.read<u64>(kMem1 + 0x5000) == 12'345 + kLatency);
}

TEST_CASE_METHOD(TimelineFixture, "a timestamp's deadline answers a wait", "[latte][timeline]") {
    CHECK_FALSE(timeline.deadline_of(1).has_value()); // nothing submitted
    static_cast<void>(timeline.submit(set_context(0x10, 1), 0));
    static_cast<void>(timeline.submit(set_context(0x10, 2), 10));
    CHECK(*timeline.deadline_of(1) == kLatency);
    CHECK(*timeline.deadline_of(2) == 10 + kLatency);
    CHECK_FALSE(timeline.deadline_of(3).has_value());
    CHECK_FALSE(timeline.deadline_of(0).has_value());

    // After retiring, a wait for it is already satisfied: its deadline is in
    // the past.
    static_cast<void>(timeline.retire_until(10 + kLatency));
    CHECK(*timeline.deadline_of(2) <= 10 + kLatency);
}

TEST_CASE_METHOD(TimelineFixture,
                 "the same submissions write the same bytes",
                 "[latte][timeline]") {
    // The property ADR-004 exists for, checked directly: two timelines fed
    // the same streams at the same ticks, retired by callers that look at
    // different moments, leave identical memory.
    const std::vector<u32> stream{type3_header(Opcode::MemWrite, 4), 0x6000 | 3, 1U << 16, 0, 0};

    static_cast<void>(timeline.submit(stream, 777));
    static_cast<void>(timeline.retire_until(777 + kLatency));
    const u64 first = memory.read<u64>(kMem1 + 0x6000);

    TimelineFixture other;
    static_cast<void>(other.timeline.submit(stream, 777));
    static_cast<void>(other.timeline.retire_until(5'000'000));
    CHECK(other.memory.read<u64>(kMem1 + 0x6000) == first);
}
