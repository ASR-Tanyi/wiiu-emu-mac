// Phase 8.1 — GX2's command buffers, timestamps and display lists.
//
// What is being guarded is the contract a title builds its frame loop on:
// a flush hands the GPU what was written and returns at once; a timestamp
// retires when the guest's clock reaches it (ADR-004), never sooner; a thread
// that waits for one sleeps exactly until then and wakes to find it retired;
// and a display list is bytes in the title's memory, called by address.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using latte::pm4::Opcode;
using latte::pm4::type3_header;

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kAttributes = 0x10002000;
constexpr ppcaddr_t kList = 0x10003000;
constexpr ppcaddr_t kOut = 0x10004000;
constexpr ppcaddr_t kMem1 = memory::mem1_base;

struct Gx2Fixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    Gx2Fixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);

        coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
    }

    u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size() && i < 8; ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    /// A call returning an OSTime, which comes back in r3:r4.
    u64 call_long(std::string_view name) {
        static_cast<void>(call(name, {}));
        return (static_cast<u64>(state.gpr[3]) << 32) | state.gpr[4];
    }

    /// GX2WaitTimeStamp takes an OSTime: r3:r4, high half first.
    u32 wait(u64 timestamp) {
        return call("GX2WaitTimeStamp",
                    {static_cast<u32>(timestamp >> 32), static_cast<u32>(timestamp)});
    }

    void init() { call("GX2Init", {0}); }

    /// What GX2 would write for a context register: the cheapest packet whose
    /// effect is visible once it retires.
    void write_context(u32 offset, u32 value) {
        const std::vector<u32> packet{type3_header(Opcode::SetContextReg, 2), offset, value};
        REQUIRE(os.gpu.write(state.core_id, packet, state.timebase));
    }

    [[nodiscard]] u32 context(u32 offset) const {
        return os.gpu.registers().read(0x28000 + (offset * 4));
    }

    /// Move the clock and let the scheduler do what it does at a timeslice.
    void advance_to(u64 tick) {
        state.timebase = tick;
        static_cast<void>(os.scheduler.reschedule(state));
    }

    [[nodiscard]] u64 latency() const { return os.gpu.timeline().latency(); }
};

} // namespace

TEST_CASE("GX2Init's attribute list is read to its end", "[hle][gx2]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(memory::default_regions(), error));
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    const std::vector<u32> list{1, 0x10100000, 2, 0x400000, 7, 2, 8, 0x10200000, 0, 0, 1, 0xBAD};
    for (std::size_t i = 0; i < list.size(); ++i) {
        REQUIRE(memory.try_write<u32>(kAttributes + static_cast<u32>(i * 4), list[i]));
    }

    const gx2::InitAttributes attributes = gx2::read_init_attributes(memory, kAttributes);
    CHECK(attributes.command_buffer_base == 0x10100000);
    CHECK(attributes.command_buffer_pool_size == 0x400000);
    CHECK(attributes.argc == 2);
    CHECK(attributes.argv == 0x10200000);
    CHECK(attributes.profile_mode == 0);

    // wut passes null for "no attributes".
    CHECK(gx2::read_init_attributes(memory, 0).command_buffer_base == 0);
}

TEST_CASE_METHOD(Gx2Fixture, "GX2Init remembers the core it was called on", "[hle][gx2]") {
    CHECK_FALSE(os.gpu.initialized());
    state.core_id = 2;
    init();
    CHECK(os.gpu.initialized());
    CHECK(call("GX2GetMainCoreId", {}) == 2);
}

TEST_CASE_METHOD(Gx2Fixture, "a flush submits and returns at once", "[hle][gx2]") {
    init();
    write_context(0x10, 7);
    CHECK(call_long("GX2GetLastSubmittedTimeStamp") == 0);

    call("GX2Flush", {});
    CHECK(call_long("GX2GetLastSubmittedTimeStamp") == 1);
    CHECK(call_long("GX2GetRetiredTimeStamp") == 0);
    CHECK(context(0x10) == 0); // not before its deadline
    CHECK(os.gpu.pending_commands().empty());

    // The scheduler's timer list is what retires it.
    REQUIRE(os.scheduler.has_timer(gx2::gpu_timer_id));
    advance_to(10'000 + latency() - 1);
    CHECK(call_long("GX2GetRetiredTimeStamp") == 0);
    advance_to(10'000 + latency());
    CHECK(call_long("GX2GetRetiredTimeStamp") == 1);
    CHECK(context(0x10) == 7);

    // Nothing left in flight, so nothing left armed.
    CHECK_FALSE(os.scheduler.has_timer(gx2::gpu_timer_id));
}

TEST_CASE_METHOD(Gx2Fixture, "a flush pads to eight words", "[hle][gx2]") {
    std::vector<u32> stream{type3_header(Opcode::Nop, 1), 0};
    gx2::pad_command_buffer(stream);
    REQUIRE(stream.size() == 8);
    CHECK(stream[2] == latte::pm4::type2_filler);
    CHECK(stream[7] == latte::pm4::type2_filler);

    gx2::pad_command_buffer(stream);
    CHECK(stream.size() == 8); // already aligned: nothing added
}

TEST_CASE_METHOD(Gx2Fixture, "an empty flush submits nothing", "[hle][gx2]") {
    init();
    call("GX2Flush", {});
    CHECK(call_long("GX2GetLastSubmittedTimeStamp") == 0);
    CHECK_FALSE(os.scheduler.has_timer(gx2::gpu_timer_id));
}

TEST_CASE_METHOD(Gx2Fixture, "waiting for a retired timestamp returns at once", "[hle][gx2]") {
    init();
    CHECK(wait(0) == 1); // nothing submitted, nothing to wait for

    write_context(0x10, 1);
    call("GX2Flush", {});
    advance_to(state.timebase + latency());
    CHECK(wait(1) == 1);
    CHECK(os.scheduler.current()->state == ThreadState::Running);
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a thread waiting for a timestamp sleeps until it retires",
                 "[hle][gx2]") {
    init();
    write_context(0x10, 5);
    call("GX2Flush", {});
    const u64 deadline = 10'000 + latency();

    CHECK(wait(1) == 1);
    const Thread* const main = os.scheduler.find(kMain);
    REQUIRE(main != nullptr);
    CHECK(main->state == ThreadState::Sleeping);
    CHECK(main->wake_time == deadline);

    // The idle skip lands exactly there; the timer fires before the thread
    // wakes, so what it wakes to is the timestamp retired.
    CHECK(os.scheduler.idle_skip_ticks(state) == latency());
    advance_to(deadline);
    CHECK(main->state == ThreadState::Running);
    CHECK(os.gpu.timeline().retired() == 1);
    CHECK(context(0x10) == 5);
}

TEST_CASE_METHOD(Gx2Fixture, "waiting for an unsubmitted timestamp fails", "[hle][gx2]") {
    init();
    CHECK(wait(3) == 0);
    CHECK(os.scheduler.current()->state == ThreadState::Running);
}

TEST_CASE_METHOD(Gx2Fixture, "GX2DrawDone flushes and waits", "[hle][gx2]") {
    init();
    write_context(0x10, 9);
    CHECK(call("GX2DrawDone", {}) == 1);
    CHECK(call_long("GX2GetLastSubmittedTimeStamp") == 1);

    const Thread* const main = os.scheduler.find(kMain);
    CHECK(main->state == ThreadState::Sleeping);
    advance_to(10'000 + latency());
    CHECK(context(0x10) == 9);
    CHECK(main->state == ThreadState::Running);
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a display list is written big-endian into the title's memory",
                 "[hle][gx2]") {
    init();
    call("GX2BeginDisplayListEx", {kList, 0x100, 1});
    CHECK(call("GX2GetDisplayListWriteStatus", {}) == 1);
    CHECK(call("GX2GetCurrentDisplayList", {kOut, kOut + 4}) == 1);
    CHECK(memory.read<u32>(kOut) == kList);
    CHECK(memory.read<u32>(kOut + 4) == 0x100);

    write_context(0x10, 0xCAFE);
    CHECK(os.gpu.pending_commands().empty()); // the list, not the main buffer
    CHECK(memory.read<u32>(kList) == type3_header(Opcode::SetContextReg, 2));
    CHECK(memory.read<u32>(kList + 8) == 0xCAFE);

    // Ended padded to eight words.
    CHECK(call("GX2EndDisplayList", {kList}) == 32);
    CHECK(memory.read<u32>(kList + 28) == latte::pm4::type2_filler);
    CHECK(call("GX2GetDisplayListWriteStatus", {}) == 0);
    CHECK(call("GX2GetCurrentDisplayList", {kOut, kOut + 4}) == 0);
}

TEST_CASE_METHOD(Gx2Fixture, "calling a display list runs it when it retires", "[hle][gx2]") {
    init();
    call("GX2BeginDisplayListEx", {kList, 0x100, 1});
    write_context(0x10, 0x1234);
    const u32 bytes = call("GX2EndDisplayList", {kList});

    call("GX2CallDisplayList", {kList, bytes});
    const std::span<const u32> pending = os.gpu.pending_commands();
    REQUIRE(pending.size() == 4);
    CHECK(pending[0] == type3_header(Opcode::IndirectBufferPriv, 3));
    CHECK(pending[1] == 0x20000000 + (kList - 0x10000000)); // the data heap's physical base
    CHECK(pending[3] == bytes / 4);

    call("GX2Flush", {});
    advance_to(10'000 + latency());
    CHECK(context(0x10) == 0x1234);
    CHECK(os.gpu.processor().last_error().empty());
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a direct call is its own submission, after what came before",
                 "[hle][gx2]") {
    init();
    call("GX2BeginDisplayListEx", {kList, 0x100, 1});
    write_context(0x10, 2);
    const u32 bytes = call("GX2EndDisplayList", {kList});

    write_context(0x10, 1);
    call("GX2DirectCallDisplayList", {kList, bytes});
    CHECK(call_long("GX2GetLastSubmittedTimeStamp") == 2);
    CHECK(os.gpu.pending_commands().empty());

    // Both retire; the list's write lands last.
    advance_to(10'000 + latency());
    CHECK(call_long("GX2GetRetiredTimeStamp") == 2);
    CHECK(context(0x10) == 2);
}

TEST_CASE_METHOD(Gx2Fixture, "copying a display list copies its words now", "[hle][gx2]") {
    init();
    call("GX2BeginDisplayListEx", {kList, 0x100, 1});
    write_context(0x10, 0xAAAA);
    const u32 bytes = call("GX2EndDisplayList", {kList});

    call("GX2CopyDisplayList", {kList, bytes});
    REQUIRE(memory.try_write<u32>(kList + 8, 0xBBBB)); // the title reuses it

    call("GX2Flush", {});
    advance_to(10'000 + latency());
    CHECK(context(0x10) == 0xAAAA);
}

TEST_CASE_METHOD(Gx2Fixture, "a full display list drops what does not fit", "[hle][gx2]") {
    init();
    call("GX2BeginDisplayListEx", {kList, 16, 1}); // four words
    write_context(0x10, 1);                        // three
    const std::vector<u32> packet{type3_header(Opcode::SetContextReg, 2), 0x11, 2};
    CHECK_FALSE(os.gpu.write(state.core_id, packet, state.timebase));
    REQUIRE(os.gpu.display_list(state.core_id) != nullptr);
    CHECK(os.gpu.display_list(state.core_id)->overrun);

    // Ends at the end of its buffer: one word of filler is all there is room
    // for.
    CHECK(call("GX2EndDisplayList", {kList}) == 16);
}

TEST_CASE_METHOD(Gx2Fixture, "only the main core has a command buffer", "[hle][gx2]") {
    init();
    state.core_id = 0;
    const std::vector<u32> packet{type3_header(Opcode::SetContextReg, 2), 0x10, 1};
    CHECK_FALSE(os.gpu.write(state.core_id, packet, state.timebase));

    // A display list it can have.
    call("GX2BeginDisplayListEx", {kList, 0x100, 1});
    CHECK(os.gpu.write(state.core_id, packet, state.timebase));
    CHECK(call("GX2EndDisplayList", {kList}) == 32);
}

TEST_CASE_METHOD(Gx2Fixture, "event callbacks are stored and read back", "[hle][gx2]") {
    init();
    call("GX2SetEventCallback", {2, 0x02001000, 0xD00D});
    CHECK(memory.try_write<u32>(kOut, 0xFFFFFFFF));
    call("GX2GetEventCallback", {2, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == 0x02001000);
    CHECK(memory.read<u32>(kOut + 4) == 0xD00D);

    // One that does not exist reads back empty.
    call("GX2GetEventCallback", {9, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == 0);
}

TEST_CASE_METHOD(Gx2Fixture, "a reset forgets everything in flight", "[hle][gx2]") {
    init();
    write_context(0x10, 3);
    call("GX2Flush", {});
    os.reset();

    CHECK_FALSE(os.gpu.initialized());
    CHECK(os.gpu.timeline().submitted() == 0);
    CHECK(os.gpu.timeline().pending() == 0);
    CHECK_FALSE(os.scheduler.has_timer(gx2::gpu_timer_id));
    CHECK(context(0x10) == 0);
}

TEST_CASE_METHOD(Gx2Fixture,
                 "the same calls at the same ticks write the same bytes",
                 "[hle][gx2]") {
    // ADR-004's property, end to end through GX2: a timestamp written by the
    // GPU is the retirement tick, however late the scheduler looks.
    init();
    const std::vector<u32> clock_write{
        type3_header(Opcode::MemWrite, 4), 0x5000 | 3, 1U << 16, 0, 0};
    REQUIRE(os.gpu.write(state.core_id, clock_write, state.timebase));
    call("GX2Flush", {});
    advance_to(10'000 + latency() + 12'345); // looked late
    CHECK(memory.read<u64>(kMem1 + 0x5000) == 10'000 + latency());
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a full main buffer is submitted on its own, and retires",
                 "[hle][gx2]") {
    init();
    const std::vector<u32> filler(gx2::max_command_buffer_words - 2, latte::pm4::type2_filler);
    REQUIRE(os.gpu.write(state.core_id, filler, state.timebase));
    CHECK(call_long("GX2GetLastSubmittedTimeStamp") == 0);

    // Eight more words do not fit: what was there goes first, by itself.
    call("GX2BeginDisplayListEx", {kList, 0x100, 1});
    write_context(0x10, 4);
    const u32 bytes = call("GX2EndDisplayList", {kList});
    call("GX2CopyDisplayList", {kList, bytes});
    CHECK(call_long("GX2GetLastSubmittedTimeStamp") == 1);
    CHECK(os.gpu.pending_commands().size() == bytes / 4);

    // And the GPU's timer was armed for it, without waiting for a flush.
    REQUIRE(os.scheduler.has_timer(gx2::gpu_timer_id));
    advance_to(10'000 + latency());
    CHECK(call_long("GX2GetRetiredTimeStamp") == 1);
}

// --------------------------------------------------------------------------
// Vsync, swaps and flips
// --------------------------------------------------------------------------

TEST_CASE("vsync falls on fixed ticks at 59.94 Hz", "[hle][gx2]") {
    CHECK(gx2::vsync_tick(0) == 0);
    CHECK(gx2::vsync_tick(1) == 1'036'973);
    // 60000 vsyncs take exactly 1001 seconds of timebase: no drift.
    CHECK(gx2::vsync_tick(60'000) == 1001ULL * 62'156'250);

    CHECK(gx2::next_vsync_index(0) == 1);
    for (u64 index = 1; index < 200; ++index) {
        const u64 tick = gx2::vsync_tick(index);
        CHECK(gx2::next_vsync_index(tick - 1) == index);
        CHECK(gx2::next_vsync_index(tick) == index + 1);
    }
}

namespace {

/// A frame's worth of GX2: a register write, the swap, the flush.
void frame(Gx2Fixture& fixture, u32 value) {
    fixture.write_context(0x10, value);
    fixture.call("GX2SwapScanBuffers", {});
    fixture.call("GX2Flush", {});
}

} // namespace

TEST_CASE_METHOD(Gx2Fixture, "GX2Init starts vsync", "[hle][gx2]") {
    CHECK_FALSE(os.scheduler.has_timer(gx2::vsync_timer_id));
    init();
    CHECK(os.scheduler.has_timer(gx2::vsync_timer_id));

    advance_to(gx2::vsync_tick(3));
    CHECK(os.gpu.swap_status().last_vsync == gx2::vsync_tick(3));
    CHECK(os.scheduler.has_timer(gx2::vsync_timer_id)); // and keeps going

    call("GX2Shutdown", {});
    CHECK_FALSE(os.scheduler.has_timer(gx2::vsync_timer_id));
}

TEST_CASE_METHOD(Gx2Fixture, "GX2WaitForVsync sleeps to the next vsync", "[hle][gx2]") {
    init();
    call("GX2WaitForVsync", {});
    const Thread* const main = os.scheduler.find(kMain);
    REQUIRE(main != nullptr);
    CHECK(main->state == ThreadState::Sleeping);
    CHECK(main->wake_time == gx2::vsync_tick(1));

    advance_to(gx2::vsync_tick(1));
    CHECK(main->state == ThreadState::Running);
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a swap flips at the first vsync after its submission retires",
                 "[hle][gx2]") {
    init();
    frame(*this, 1);
    CHECK(os.gpu.swap_status().swaps == 1);
    CHECK(os.gpu.swap_status().flips == 0);

    advance_to(gx2::vsync_tick(1));
    CHECK(os.gpu.swap_status().flips == 1);
    CHECK(os.gpu.swap_status().last_flip == gx2::vsync_tick(1));
    CHECK(os.gpu.swap_status().pending() == 0);
}

TEST_CASE_METHOD(Gx2Fixture, "an unflushed swap does not flip", "[hle][gx2]") {
    init();
    write_context(0x10, 1);
    call("GX2SwapScanBuffers", {});
    advance_to(gx2::vsync_tick(2));
    CHECK(os.gpu.swap_status().flips == 0);

    // Flushed, it flips at the next vsync it is ready by.
    call("GX2Flush", {});
    advance_to(gx2::vsync_tick(3));
    CHECK(os.gpu.swap_status().flips == 1);
    CHECK(os.gpu.swap_status().last_flip == gx2::vsync_tick(3));
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a swap that retires after a vsync waits for the next, however late the "
                 "clock is looked at",
                 "[hle][gx2]") {
    init();
    // Something submitted early, due well before vsync 1: its timer is the
    // first thing due, and must retire only what is due by *its* deadline.
    write_context(0x11, 1);
    call("GX2Flush", {});

    // The swap's submission, due just past vsync 1.
    state.timebase = gx2::vsync_tick(1) - latency() + 10;
    frame(*this, 1);

    // Serviced long after vsync 2: both GPU deadlines and both vsyncs are all
    // due at once. The flip still lands on vsync 2, not on vsync 1.
    advance_to(gx2::vsync_tick(2) + 500'000);
    CHECK(os.gpu.swap_status().flips == 1);
    CHECK(os.gpu.swap_status().last_flip == gx2::vsync_tick(2));
}

TEST_CASE_METHOD(Gx2Fixture, "the swap interval spaces flips out", "[hle][gx2]") {
    init();
    call("GX2SetSwapInterval", {2});
    CHECK(call("GX2GetSwapInterval", {}) == 2);

    frame(*this, 1);
    frame(*this, 2);
    for (u64 index = 1; index <= 4; ++index) {
        advance_to(gx2::vsync_tick(index));
    }
    CHECK(os.gpu.swap_status().flips == 2);
    CHECK(os.gpu.swap_status().last_flip == gx2::vsync_tick(3)); // 1, then 1 + 2

    // Out of range is refused, and the interval kept.
    call("GX2SetSwapInterval", {20});
    CHECK(call("GX2GetSwapInterval", {}) == 2);
}

TEST_CASE_METHOD(Gx2Fixture, "GX2WaitForFlip waits only for a pending swap", "[hle][gx2]") {
    init();
    const Thread* const main = os.scheduler.find(kMain);
    REQUIRE(main != nullptr);

    call("GX2WaitForFlip", {}); // nothing pending
    CHECK(main->state == ThreadState::Running);

    frame(*this, 1);
    call("GX2WaitForFlip", {});
    CHECK(main->state == ThreadState::Blocked);

    advance_to(gx2::vsync_tick(1));
    CHECK(main->state == ThreadState::Running);
    CHECK(os.gpu.swap_status().flips == 1);
}

TEST_CASE_METHOD(Gx2Fixture, "GX2GetSwapStatus reports the counts and times", "[hle][gx2]") {
    init();
    frame(*this, 1);
    frame(*this, 2);
    advance_to(gx2::vsync_tick(1));

    REQUIRE(memory.fill(kOut, 24, 0xFF));
    call("GX2GetSwapStatus", {kOut, kOut + 4, kOut + 8, kOut + 16});
    CHECK(memory.read<u32>(kOut) == 2);
    CHECK(memory.read<u32>(kOut + 4) == 1);
    CHECK(memory.read<u64>(kOut + 8) == gx2::vsync_tick(1));
    CHECK(memory.read<u64>(kOut + 16) == gx2::vsync_tick(1));

    // Null pointers are skipped, not written through.
    call("GX2GetSwapStatus", {0, 0, 0, 0});
}

TEST_CASE_METHOD(Gx2Fixture,
                 "swapping too far ahead of the display waits for a flip",
                 "[hle][gx2]") {
    init();
    const Thread* const main = os.scheduler.find(kMain);
    REQUIRE(main != nullptr);

    for (u32 i = 0; i < gx2::max_pending_swaps; ++i) {
        write_context(0x10, i);
        call("GX2SwapScanBuffers", {});
        REQUIRE(main->state == ThreadState::Running);
    }

    // One more is one too many. It is flushed, so the swaps can retire, and
    // the thread waits for the first of them to flip.
    call("GX2SwapScanBuffers", {});
    CHECK(main->state == ThreadState::Blocked);
    CHECK(os.gpu.pending_commands().empty());

    advance_to(gx2::vsync_tick(1));
    CHECK(main->state == ThreadState::Running);
    CHECK(os.gpu.swap_status().pending() == gx2::max_pending_swaps);
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a swap from another core or inside a display list is not counted",
                 "[hle][gx2]") {
    init();
    call("GX2BeginDisplayListEx", {kList, 0x100, 1});
    call("GX2SwapScanBuffers", {});
    static_cast<void>(call("GX2EndDisplayList", {kList}));

    state.core_id = 0;
    call("GX2SwapScanBuffers", {});
    CHECK(os.gpu.swap_status().swaps == 0);
}

TEST_CASE_METHOD(Gx2Fixture, "vsync and flip run the title's callbacks", "[hle][gx2]") {
    constexpr ppcaddr_t kVsyncCallback = 0x02000100;
    constexpr ppcaddr_t kFlipCallback = 0x02000200;
    std::string error;
    REQUIRE(os.heaps.create_default_heaps(memory, error)); // for the callbacks' stacks
    init();
    call("GX2SetEventCallback", {2, kVsyncCallback, 0x1111});
    call("GX2SetEventCallback", {3, kFlipCallback, 0x2222});
    frame(*this, 1);
    advance_to(gx2::vsync_tick(1) - 1); // nothing yet

    const auto callbacks_at = [this](ppcaddr_t entry) {
        u32 count = 0;
        for (const Thread* thread : os.scheduler.threads()) {
            if (thread->context.pc == entry && thread->state != ThreadState::Exited) {
                CHECK(thread->context.gpr[4] == (entry == kVsyncCallback ? 0x1111U : 0x2222U));
                ++count;
            }
        }
        return count;
    };
    CHECK(callbacks_at(kVsyncCallback) == 0);

    advance_to(gx2::vsync_tick(1));
    CHECK(callbacks_at(kVsyncCallback) == 1);
    CHECK(callbacks_at(kFlipCallback) == 1);
}

// --------------------------------------------------------------------------
// The outputs
// --------------------------------------------------------------------------

namespace {

constexpr u32 kRgba8 = 0x01A;  // GX2_SURFACE_FORMAT_UNORM_R8_G8_B8_A8
constexpr u32 kSrgb8 = 0x41A;  // GX2_SURFACE_FORMAT_SRGB_R8_G8_B8_A8
constexpr u32 kRgba16 = 0x01F; // GX2_SURFACE_FORMAT_UNORM_R16_G16_B16_A16
constexpr u32 kDouble = 2;     // GX2_BUFFERING_MODE_DOUBLE

} // namespace

TEST_CASE("the TV and the GamePad have the shapes their modes name", "[hle][gx2]") {
    CHECK(gx2::tv_scan_geometry(1)->width == 640);
    CHECK(gx2::tv_scan_geometry(2)->width == 854);
    CHECK(gx2::tv_scan_geometry(3)->height == 720);
    CHECK(gx2::tv_scan_geometry(4)->height == 720);
    CHECK(gx2::tv_scan_geometry(5)->width == 1920);
    CHECK(gx2::tv_scan_geometry(5)->height == 1080);
    CHECK_FALSE(gx2::tv_scan_geometry(0).has_value()); // disabled
    CHECK_FALSE(gx2::tv_scan_geometry(6).has_value());

    CHECK(gx2::drc_scan_geometry(1)->width == 854);
    CHECK(gx2::drc_scan_geometry(1)->pitch == 864);
    CHECK_FALSE(gx2::drc_scan_geometry(0).has_value());

    CHECK(gx2::scan_buffer_count(1) == 1);
    CHECK(gx2::scan_buffer_count(3) == 3);
    CHECK(gx2::scan_buffer_count(0) == 0);
    CHECK(gx2::scan_buffer_count(4) == 0);
}

TEST_CASE_METHOD(Gx2Fixture, "GX2CalcTVSize and GX2CalcDRCSize size every frame", "[hle][gx2]") {
    REQUIRE(memory.fill(kOut, 8, 0xFF));

    // 720p, RGBA8, double-buffered: the same two frames OSScreen's TV buffer
    // holds, and the same size.
    call("GX2CalcTVSize", {3, kRgba8, kDouble, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == 0x708000);
    CHECK(memory.read<u32>(kOut + 4) == 0); // no scaling needed

    call("GX2CalcTVSize", {5, kSrgb8, 3, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == 1920U * 1080 * 4 * 3);
    call("GX2CalcTVSize", {5, kRgba16, kDouble, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == 1920U * 1080 * 8 * 2);

    call("GX2CalcDRCSize", {1, kRgba8, kDouble, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == 864U * 480 * 4 * 2);

    // Disabled, or a format with no element size: nothing to allocate.
    call("GX2CalcTVSize", {0, kRgba8, kDouble, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == 0);
    call("GX2CalcTVSize", {3, 0x031, kDouble, kOut, kOut + 4}); // BC1
    CHECK(memory.read<u32>(kOut) == 0);
}

TEST_CASE_METHOD(Gx2Fixture, "the scan buffers are what the title set", "[hle][gx2]") {
    init();
    constexpr ppcaddr_t kTvScan = kMem1 + 0x100000;
    constexpr ppcaddr_t kDrcScan = kMem1 + 0x900000;

    call("GX2SetTVBuffer", {kTvScan, 0x708000, 3, kRgba8, kDouble});
    call("GX2SetDRCBuffer", {kDrcScan, 0x32A000, 1, kRgba8, kDouble});
    call("GX2SetTVEnable", {1});
    call("GX2SetDRCScale", {854, 480});

    const gx2::ScanBuffer& tv = os.gpu.scan_buffer(gx2::ScanTarget::Tv);
    CHECK(tv.configured());
    CHECK(tv.address == kTvScan);
    CHECK(tv.size == 0x708000);
    CHECK(tv.geometry.width == 1280);
    CHECK(tv.surface_format == kRgba8);
    CHECK(tv.buffering_mode == kDouble);
    CHECK(tv.enabled);

    const gx2::ScanBuffer& drc = os.gpu.scan_buffer(gx2::ScanTarget::Drc);
    CHECK(drc.address == kDrcScan);
    CHECK(drc.geometry.pitch == 864);
    CHECK_FALSE(drc.enabled);
    CHECK(drc.scale_x == 854);
    CHECK(drc.scale_y == 480);

    call("GX2SetTVEnable", {0});
    CHECK_FALSE(tv.enabled);

    // A shutdown forgets them; so does a reset.
    call("GX2Shutdown", {});
    CHECK_FALSE(os.gpu.scan_buffer(gx2::ScanTarget::Tv).configured());
}

TEST_CASE_METHOD(Gx2Fixture, "the outputs report a 1080p TV and one GamePad", "[hle][gx2]") {
    CHECK(call("GX2GetSystemTVScanMode", {}) == 7);
    CHECK(call("GX2GetSystemTVAspectRatio", {}) == 1);
    CHECK(call("GX2GetSystemDRCMode", {}) == 1);
    CHECK(call("GX2IsVideoOutReady", {}) == 1);
}

TEST_CASE_METHOD(Gx2Fixture,
                 "a GamePad connect callback is told about its GamePad at once",
                 "[hle][gx2]") {
    constexpr ppcaddr_t kConnect = 0x02000300;
    constexpr ppcaddr_t kOther = 0x02000400;
    std::string error;
    REQUIRE(os.heaps.create_default_heaps(memory, error)); // for the callbacks' stacks

    const auto connect_calls = [this](ppcaddr_t entry) {
        std::vector<std::pair<u32, u32>> calls;
        for (const Thread* thread : os.scheduler.threads()) {
            if (thread->context.pc == entry && thread->state != ThreadState::Exited) {
                calls.emplace_back(thread->context.gpr[3], thread->context.gpr[4]);
            }
        }
        return calls;
    };

    CHECK(call("GX2SetDRCConnectCallback", {0, kConnect}) == 0);
    CHECK(os.gpu.drc_connect_callback() == kConnect);
    const auto first = connect_calls(kConnect);
    REQUIRE(first.size() == 1);
    CHECK(first[0] == std::pair<u32, u32>{0, 1}); // slot 0, attached

    // The second GamePad is never there; the old callback comes back.
    CHECK(call("GX2SetDRCConnectCallback", {1, kOther}) == kConnect);
    const auto second = connect_calls(kOther);
    REQUIRE(second.size() == 1);
    CHECK(second[0] == std::pair<u32, u32>{1, 0});

    // Clearing it calls nothing.
    CHECK(call("GX2SetDRCConnectCallback", {0, 0}) == kOther);
    CHECK(os.gpu.drc_connect_callback() == 0);
}
