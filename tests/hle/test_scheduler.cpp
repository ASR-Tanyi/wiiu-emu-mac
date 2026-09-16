#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/hle/scheduler.hpp>
#include <wiiu/loader/rpl.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;

namespace {

constexpr ppcaddr_t kEntry = 0x02000000;
constexpr ppcaddr_t kStack = memory::data_heap_base + 0x10000;

/// Handles are guest OSThread pointers; any distinct values will do here.
constexpr u32 kMain = main_thread_handle;
constexpr u32 kWorkerA = 0x10004000;
constexpr u32 kWorkerB = 0x10005000;

struct SchedulerFixture {
    Scheduler scheduler;
    cpu::CoreState state;

    SchedulerFixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        state.pc = kEntry;
        static_cast<void>(scheduler.adopt_main(state, kMain));
    }

    u32 spawn(u32 handle, s32 priority = default_priority, ppcaddr_t entry = kEntry + 0x100) {
        return scheduler.create(handle, "worker", entry, 0, 0, kStack, 0x1000, priority, 0x7);
    }
};

} // namespace

TEST_CASE_METHOD(SchedulerFixture, "a new thread does not run until resumed", "[hle][scheduler]") {
    // OSCreateThread creates; OSResumeThread starts. Starting on creation
    // would run threads a title expected to configure first.
    static_cast<void>(spawn(kWorkerA));

    const Thread* const worker = scheduler.find(kWorkerA);
    REQUIRE(worker != nullptr);
    CHECK(worker->state == ThreadState::Suspended);
    CHECK_FALSE(scheduler.reschedule(state));
    CHECK(scheduler.current_handle() == kMain);

    REQUIRE(scheduler.resume(kWorkerA));
    CHECK(worker->state == ThreadState::Ready);
}

TEST_CASE_METHOD(SchedulerFixture,
                 "a context switch saves and restores registers",
                 "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    state.gpr[10] = 0xAAAAAAAA;
    state.pc = 0x02000040;

    REQUIRE(scheduler.reschedule(state));
    CHECK(scheduler.current_handle() == kWorkerA);
    CHECK(state.pc == kEntry + 0x100); // the worker's entry
    CHECK(state.gpr[10] == 0U);        // its own registers, not main's
    CHECK(state.lr == thread_exit_sentinel);

    // Back to main: everything is where it was left.
    REQUIRE(scheduler.reschedule(state));
    CHECK(scheduler.current_handle() == kMain);
    CHECK(state.pc == 0x02000040U);
    CHECK(state.gpr[10] == 0xAAAAAAAAU);
}

TEST_CASE_METHOD(SchedulerFixture, "higher priority runs first", "[hle][scheduler]") {
    // Cafe OS priorities run 0 (highest) to 31; the number is backwards from
    // the intuition, which is exactly why it is worth a test.
    static_cast<void>(spawn(kWorkerA, 20));
    static_cast<void>(spawn(kWorkerB, 5));
    REQUIRE(scheduler.resume(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerB));

    REQUIRE(scheduler.reschedule(state));
    CHECK(scheduler.current_handle() == kWorkerB); // priority 5 beats 20
}

TEST_CASE_METHOD(SchedulerFixture, "equal priorities take turns", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA, 16));
    static_cast<void>(spawn(kWorkerB, 16));
    REQUIRE(scheduler.resume(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerB));

    std::vector<u32> order;
    for (int i = 0; i < 6; ++i) {
        static_cast<void>(scheduler.reschedule(state));
        order.push_back(scheduler.current_handle());
    }

    // Nobody is starved. "Prefer anything but the current thread" looks like
    // round-robin with two threads and quietly alternates between two of three,
    // so all three have to be seen.
    CHECK(std::ranges::find(order, kMain) != order.end());
    CHECK(std::ranges::find(order, kWorkerA) != order.end());
    CHECK(std::ranges::find(order, kWorkerB) != order.end());
}

TEST_CASE_METHOD(SchedulerFixture, "a suspended thread is skipped", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));
    REQUIRE(scheduler.suspend(kWorkerA));

    CHECK_FALSE(scheduler.reschedule(state));
    CHECK(scheduler.current_handle() == kMain);
}

TEST_CASE_METHOD(SchedulerFixture, "join blocks until the target exits", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    s32 value = 0;
    CHECK(scheduler.join(state, kWorkerA, 0, value));

    // Main is blocked, so the worker is what runs now.
    CHECK(scheduler.find(kMain)->state == ThreadState::Blocked);
    CHECK(scheduler.find(kMain)->block_reason == BlockReason::Join);
    CHECK(scheduler.current_handle() == kWorkerA);

    scheduler.exit_current(state, 42);

    CHECK(scheduler.find(kWorkerA)->state == ThreadState::Exited);
    CHECK(scheduler.find(kMain)->state == ThreadState::Running);
    CHECK(scheduler.current_handle() == kMain);
}

TEST_CASE_METHOD(SchedulerFixture, "joining a finished thread does not block", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));
    REQUIRE(scheduler.reschedule(state));
    scheduler.exit_current(state, 7);

    s32 value = 0;
    CHECK_FALSE(scheduler.join(state, kWorkerA, 0, value));
    CHECK(value == 7);
    CHECK(scheduler.current_handle() == kMain);
}

TEST_CASE_METHOD(SchedulerFixture, "the exit value reaches the joiner", "[hle][scheduler]") {
    // The scheduler has no memory access of its own, so it hands the value to
    // a callback instead.
    s32 delivered = 0;
    ppcaddr_t delivered_to = 0;
    scheduler.set_exit_callback([&](s32 value, ppcaddr_t out) {
        delivered = value;
        delivered_to = out;
    });

    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    s32 value = 0;
    REQUIRE(scheduler.join(state, kWorkerA, 0x12345678, value));
    scheduler.exit_current(state, 99);

    CHECK(delivered == 99);
    CHECK(delivered_to == 0x12345678U);
}

TEST_CASE_METHOD(SchedulerFixture, "sleeping wakes at its deadline", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    state.timebase = 1000;
    scheduler.sleep(state, 500);

    CHECK(scheduler.find(kMain)->state == ThreadState::Sleeping);
    CHECK(scheduler.current_handle() == kWorkerA);

    // Not yet.
    state.timebase = 1400;
    static_cast<void>(scheduler.reschedule(state));
    CHECK(scheduler.find(kMain)->state == ThreadState::Sleeping);

    // Now.
    state.timebase = 1500;
    static_cast<void>(scheduler.reschedule(state));
    CHECK(scheduler.find(kMain)->state != ThreadState::Sleeping);
}

TEST_CASE_METHOD(SchedulerFixture, "a blocked thread is woken by its object", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    constexpr u32 object = 0x10009000;
    scheduler.block(state, BlockReason::Mutex, object);
    CHECK(scheduler.find(kMain)->state == ThreadState::Blocked);

    CHECK_FALSE(scheduler.unblock_one(0xDEADBEEF)); // a different object
    CHECK(scheduler.unblock_one(object));
    CHECK(scheduler.find(kMain)->state == ThreadState::Ready);
}

TEST_CASE_METHOD(SchedulerFixture,
                 "the highest-priority waiter is woken first",
                 "[hle][scheduler]") {
    // Otherwise a low-priority thread that happened to arrive first holds up
    // everything behind it.
    // Main is put at the bottom so the two workers are what run.
    scheduler.find(kMain)->priority = 30;
    static_cast<void>(spawn(kWorkerA, 25));
    static_cast<void>(spawn(kWorkerB, 3));
    REQUIRE(scheduler.resume(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerB));

    constexpr u32 object = 0x10009000;

    REQUIRE(scheduler.reschedule(state)); // to B (priority 3)
    REQUIRE(scheduler.current_handle() == kWorkerB);
    scheduler.block(state, BlockReason::Event, object);

    REQUIRE(scheduler.current_handle() == kWorkerA); // next highest
    scheduler.block(state, BlockReason::Event, object);

    REQUIRE(scheduler.current_handle() == kMain);
    CHECK(scheduler.unblock_one(object));
    CHECK(scheduler.find(kWorkerB)->state == ThreadState::Ready);
    CHECK(scheduler.find(kWorkerA)->state == ThreadState::Blocked);
}

TEST_CASE_METHOD(SchedulerFixture, "unblock_all releases every waiter", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    static_cast<void>(spawn(kWorkerB));
    REQUIRE(scheduler.resume(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerB));

    constexpr u32 object = 0x10009000;
    REQUIRE(scheduler.reschedule(state));
    scheduler.block(state, BlockReason::Event, object);
    scheduler.block(state, BlockReason::Event, object);

    CHECK(scheduler.unblock_all(object) == 2);
}

TEST_CASE_METHOD(SchedulerFixture,
                 "a deadlock is detected rather than spun on",
                 "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    constexpr u32 object = 0x10009000;
    scheduler.block(state, BlockReason::Mutex, object); // main blocks
    scheduler.block(state, BlockReason::Mutex, object); // and so does the worker

    CHECK(scheduler.stalled());
    CHECK_FALSE(scheduler.all_exited());
}

TEST_CASE_METHOD(SchedulerFixture, "everything exiting is not a deadlock", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    REQUIRE(scheduler.reschedule(state));
    scheduler.exit_current(state, 0); // worker
    scheduler.exit_current(state, 0); // main

    CHECK(scheduler.all_exited());
    CHECK_FALSE(scheduler.stalled());
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

namespace {

struct ThreadedMachine {
    memory::Memory memory;
    cpu::CoreState state;
    cpu::Interpreter interpreter{state, memory};
    Registry registry;
    coreinit::Context os;
    std::vector<std::string> output;

    ThreadedMachine() {
        std::vector<memory::Region> regions(memory::default_regions().begin(),
                                            memory::default_regions().end());
        regions.insert(regions.begin() + 1,
                       memory::Region{"hle",
                                      Registry::trampoline_base,
                                      Registry::trampoline_size,
                                      0,
                                      memory::Backing::None,
                                      memory::read_write_execute});

        std::string error;
        REQUIRE(memory.initialize(regions, error));

        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        os.report_sink = [this](std::string_view text) { output.emplace_back(text); };
        coreinit::register_all(registry, os);
        REQUIRE(os.heaps.create_default_heaps(memory, error));
    }

    [[nodiscard]] RunReport run_module(std::string_view file) {
        loader::Rpl rpl;
        const auto path = std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / file;
        REQUIRE(loader::parse_file(path.string(), rpl).ok);

        LoadedModule module;
        const LoadResult loaded = load_module(rpl, memory, registry, state, module);
        INFO("load: " << loaded.error);
        REQUIRE(loaded.ok);

        static_cast<void>(os.scheduler.adopt_main(state, main_thread_handle));
        os.scheduler.set_exit_callback([this](s32 value, ppcaddr_t out) {
            static_cast<void>(memory.try_write<u32>(out, static_cast<u32>(value)));
        });

        return run(interpreter, memory, registry, {.scheduler = &os.scheduler});
    }
};

} // namespace

TEST_CASE_METHOD(ThreadedMachine,
                 "a module creates, runs and joins a thread",
                 "[hle][scheduler][program]") {
    const RunReport report = run_module("threaded.rpx");

    INFO("outcome: " << run_outcome_name(report.outcome));
    CHECK(report.outcome == RunOutcome::Returned);
    CHECK(report.context_switches >= 2); // into the worker and back

    REQUIRE(output.size() == 3);
    CHECK(output[0] == "main: starting worker");
    CHECK(output[1] == "worker: running");
    // The exit value travelled back through OSJoinThread's out pointer.
    CHECK(output[2] == "main: worker returned 99");
}

TEST_CASE_METHOD(ThreadedMachine,
                 "the single-threaded module still runs",
                 "[hle][scheduler][program]") {
    // Adding a scheduler must not change a module that never creates a thread.
    const RunReport report = run_module("homebrew.rpx");
    CHECK(report.outcome == RunOutcome::Returned);
    REQUIRE(output.size() == 2);
    CHECK(output[1] == "sum 1..10 = 55");
}
