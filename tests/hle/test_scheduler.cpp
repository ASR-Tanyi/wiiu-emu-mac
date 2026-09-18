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

// ---------------------------------------------------------------------------
// Phase 6.5 — the rest of the synchronisation primitives
// ---------------------------------------------------------------------------

namespace {

/// A scheduler with coreinit's handlers on top, so the blocking rules can be
/// exercised the way a guest reaches them: by calling the function.
struct SyncFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    static constexpr u32 kMutex = 0x10001000;
    static constexpr u32 kCond = 0x10001100;
    static constexpr u32 kFast = 0x10001200;
    static constexpr u32 kBarrier = 0x10001300;
    static constexpr u32 kWord = 0x10001400;
    static constexpr u32 kQueue = 0x10001500;
    static constexpr u32 kRing = 0x10001600;   ///< the array the title supplies
    static constexpr u32 kOutbox = 0x10001800; ///< where a receiver wants its message
    static constexpr u32 kOutboxB = 0x10001820;
    static constexpr u32 kMessage = 0x10001900; ///< the message a sender is sending

    SyncFixture() {
        std::string error;
        REQUIRE(memory.initialize(error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);

        coreinit::register_all(registry, os);
        state.pc = kEntry;

        // Cafe OS puts the main thread on core 1, the one with the 2 MiB of
        // L2. The rendezvous test cares which core it is on.
        state.core_id = 1;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
    }

    u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("coreinit", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size() && i < 8; ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    /// A worker that is runnable straight away, so that a blocking call from
    /// the main thread switches to it.
    void spawn_ready(u32 handle, u32 affinity = 0x7) {
        static_cast<void>(
            os.scheduler.create(handle, "worker", kEntry, 0, 0, kStack, 0x1000, 16, affinity));
        REQUIRE(os.scheduler.resume(handle));
    }

    [[nodiscard]] ThreadState state_of(u32 handle) {
        const Thread* const thread = os.scheduler.find(handle);
        REQUIRE(thread != nullptr);
        return thread->state;
    }

    /// Put an OSMessage at `address` and send it, returning what the call did.
    u32 send(u32 queue, u32 value, u32 flags, std::string_view function = "OSSendMessage") {
        REQUIRE(memory.try_write<u32>(kMessage, value));
        REQUIRE(memory.try_write<u32>(kMessage + 4, value ^ 0xA5A5A5A5));
        REQUIRE(memory.try_write<u32>(kMessage + 8, 0));
        REQUIRE(memory.try_write<u32>(kMessage + 12, 0));
        return call(function, {queue, kMessage, flags});
    }

    [[nodiscard]] u32 received(u32 address = kOutbox) { return memory.read<u32>(address); }
};

} // namespace

TEST_CASE_METHOD(SyncFixture,
                 "a condition variable releases its mutex while waiting",
                 "[hle][scheduler][sync]") {
    spawn_ready(kWorkerA);

    static_cast<void>(call("OSInitMutex", {kMutex}));
    static_cast<void>(call("OSInitCond", {kCond}));
    static_cast<void>(call("OSLockMutex", {kMutex}));
    REQUIRE(os.sync.mutexes[kMutex].owner == kMain);

    static_cast<void>(call("OSWaitCond", {kCond, kMutex}));

    // Waiting has to give the mutex back: whoever is going to signal the
    // condition almost certainly needs it first. A wait that kept the mutex
    // is a deadlock that looks like a hang in the title.
    CHECK(state_of(kMain) == ThreadState::Blocked);
    CHECK(os.sync.mutexes[kMutex].owner == 0U);
    CHECK(os.scheduler.current_handle() == kWorkerA);

    // The worker signals; the waiter becomes runnable again.
    static_cast<void>(call("OSSignalCond", {kCond}));
    CHECK(state_of(kMain) == ThreadState::Ready);
}

TEST_CASE_METHOD(SyncFixture, "a fast mutex is not recursive", "[hle][scheduler][sync]") {
    spawn_ready(kWorkerA);

    static_cast<void>(call("OSFastMutex_Init", {kFast}));
    static_cast<void>(call("OSFastMutex_Lock", {kFast}));
    CHECK(os.sync.fast_mutexes[kFast].owner == kMain);

    // OSMutex is recursive for its owner and OSFastMutex is not: locking one
    // twice deadlocks on hardware, and quietly succeeding here would hide a
    // title's bug until it showed up as something stranger.
    static_cast<void>(call("OSFastMutex_Lock", {kFast}));
    CHECK(state_of(kMain) == ThreadState::Blocked);
}

TEST_CASE_METHOD(SyncFixture,
                 "unlocking a fast mutex hands it to a waiter",
                 "[hle][scheduler][sync]") {
    spawn_ready(kWorkerA);

    static_cast<void>(call("OSFastMutex_Init", {kFast}));
    static_cast<void>(call("OSFastMutex_Lock", {kFast}));

    // Block the main thread so the worker becomes current, then contend.
    static_cast<void>(call("OSFastMutex_Lock", {kFast}));
    REQUIRE(os.scheduler.current_handle() == kWorkerA);

    CHECK(call("OSFastMutex_TryLock", {kFast}) == 0U); // still held
    static_cast<void>(call("OSFastMutex_Unlock", {kFast}));

    // The woken waiter *owns* it. It resumes after its own lock call -- the
    // return address is set before the handler blocks it -- so it never gets
    // another chance to take the lock, and leaving the mutex unowned here put
    // two threads inside the same critical section.
    CHECK(state_of(kMain) == ThreadState::Ready);
    CHECK(os.sync.fast_mutexes[kFast].owner == kMain);
    CHECK(call("OSFastMutex_TryLock", {kFast}) == 0U); // and nobody else can take it
}

TEST_CASE_METHOD(SyncFixture,
                 "a rendezvous waits for every core in the mask",
                 "[hle][scheduler][sync]") {
    // The main thread is on core 1; give the worker core 2, so the two of them
    // are what the barrier is waiting for.
    spawn_ready(kWorkerA, 0x4);

    static_cast<void>(call("OSInitRendezvous", {kBarrier}));

    constexpr u32 mask = 0x6; // cores 1 and 2
    static_cast<void>(call("OSWaitRendezvous", {kBarrier, mask}));
    CHECK(state_of(kMain) == ThreadState::Blocked);
    REQUIRE(os.scheduler.current_handle() == kWorkerA);

    static_cast<void>(call("OSWaitRendezvous", {kBarrier, mask}));
    CHECK(state_of(kMain) == ThreadState::Ready);

    // And it resets, because a title uses the same rendezvous every frame.
    CHECK(os.sync.rendezvous[kBarrier].arrived == 0U);
}

TEST_CASE_METHOD(SyncFixture, "the atomics read, modify and report", "[hle][scheduler][sync]") {
    REQUIRE(memory.try_write<u32>(kWord, 10));

    // OSSwapAtomic returns what was there.
    CHECK(call("OSSwapAtomic", {kWord, 42}) == 10U);
    CHECK(memory.read<u32>(kWord) == 42U);

    // Compare-and-swap only stores on a match, and says which happened.
    CHECK(call("OSCompareAndSwapAtomic", {kWord, 1, 99}) == 0U);
    CHECK(memory.read<u32>(kWord) == 42U);
    CHECK(call("OSCompareAndSwapAtomic", {kWord, 42, 99}) == 1U);
    CHECK(memory.read<u32>(kWord) == 99U);

    // The Ex form also reports what it found, so a failed attempt does not
    // need a second read.
    constexpr ppcaddr_t found = kWord + 4;
    CHECK(call("OSCompareAndSwapAtomicEx", {kWord, 1, 7, found}) == 0U);
    CHECK(memory.read<u32>(found) == 99U);

    // OSAddAtomic returns the *new* value; the bitwise ones return the old.
    REQUIRE(memory.try_write<u32>(kWord, 5));
    CHECK(call("OSAddAtomic", {kWord, 3}) == 8U);
    CHECK(memory.read<u32>(kWord) == 8U);

    CHECK(call("OSOrAtomic", {kWord, 0x1}) == 8U);
    CHECK(memory.read<u32>(kWord) == 9U);
    CHECK(call("OSAndAtomic", {kWord, 0x8}) == 9U);
    CHECK(memory.read<u32>(kWord) == 8U);
}

TEST_CASE_METHOD(SyncFixture,
                 "unlocking a mutex hands it to the waiter",
                 "[hle][scheduler][sync]") {
    // The hand-off is the part that is easy to get wrong and impossible to
    // notice: a woken thread resumes *after* its OSLockMutex, so if the
    // unlock leaves the mutex unowned, two threads are inside the same
    // critical section and nothing says so until their shared state is wrong.
    spawn_ready(kWorkerA);

    static_cast<void>(call("OSInitMutex", {kMutex}));
    static_cast<void>(call("OSLockMutex", {kMutex}));
    REQUIRE(os.sync.mutexes[kMutex].owner == kMain);

    // Contend from the main thread's point of view by locking again from the
    // worker: block main first so the worker is current.
    static_cast<void>(call("OSLockMutex", {kMutex})); // no-op, recursive for the owner
    REQUIRE(os.sync.mutexes[kMutex].recursion == 2U);

    static_cast<void>(call("OSUnlockMutex", {kMutex}));
    static_cast<void>(call("OSUnlockMutex", {kMutex}));
    CHECK(os.sync.mutexes[kMutex].owner == 0U);

    // Now a real contention: the worker takes it, main blocks, the worker
    // releases it.
    CHECK(call("OSTryLockMutex", {kMutex}) == 1U); // main takes it again
    os.scheduler.block_thread(kWorkerA, BlockReason::Mutex, kMutex);

    static_cast<void>(call("OSUnlockMutex", {kMutex}));
    CHECK(state_of(kWorkerA) == ThreadState::Ready);
    CHECK(os.sync.mutexes[kMutex].owner == kWorkerA);
    CHECK(os.sync.mutexes[kMutex].recursion == 1U);

    // And nobody else can take it while the woken thread believes it holds it.
    CHECK(call("OSTryLockMutex", {kMutex}) == 0U);
}

TEST_CASE_METHOD(SyncFixture,
                 "a signalled condition returns with the mutex held",
                 "[hle][scheduler][sync]") {
    // Two waiters: one gets the mutex, the other queues up on it. Both have
    // to be holding it when they eventually resume, or the code after
    // OSWaitCond runs unprotected.
    spawn_ready(kWorkerA);
    spawn_ready(kWorkerB);

    static_cast<void>(call("OSInitMutex", {kMutex}));
    static_cast<void>(call("OSInitCond", {kCond}));

    static_cast<void>(call("OSLockMutex", {kMutex}));
    static_cast<void>(call("OSWaitCond", {kCond, kMutex})); // main waits
    REQUIRE(os.scheduler.current_handle() == kWorkerA);

    static_cast<void>(call("OSLockMutex", {kMutex}));
    static_cast<void>(call("OSWaitCond", {kCond, kMutex})); // worker A waits
    REQUIRE(os.scheduler.current_handle() == kWorkerB);

    static_cast<void>(call("OSSignalCond", {kCond}));

    // One of them holds the mutex and is runnable; the other is waiting for
    // the mutex rather than for the condition.
    const u32 owner = os.sync.mutexes[kMutex].owner;
    CHECK((owner == kMain || owner == kWorkerA));
    CHECK(os.sync.mutexes[kMutex].recursion == 1U);

    const u32 other = owner == kMain ? kWorkerA : kMain;
    CHECK(state_of(owner) == ThreadState::Ready);
    CHECK(state_of(other) == ThreadState::Blocked);

    const Thread* const waiting = os.scheduler.find(other);
    REQUIRE(waiting != nullptr);
    CHECK(waiting->block_reason == BlockReason::Mutex);
    CHECK(waiting->blocked_on == kMutex);
}

TEST_CASE_METHOD(SyncFixture, "OSSignalEventAll releases every waiter", "[hle][scheduler][sync]") {
    constexpr u32 kEvent = 0x10001500;
    spawn_ready(kWorkerA);
    spawn_ready(kWorkerB);

    // An auto-reset event releases one waiter per signal -- which is exactly
    // why OSSignalEventAll exists, and why aliasing the two is wrong.
    static_cast<void>(call("OSInitEvent", {kEvent, 0, 0}));

    static_cast<void>(call("OSWaitEvent", {kEvent}));
    static_cast<void>(call("OSWaitEvent", {kEvent}));
    REQUIRE(state_of(kMain) == ThreadState::Blocked);
    REQUIRE(state_of(kWorkerA) == ThreadState::Blocked);

    static_cast<void>(call("OSSignalEventAll", {kEvent}));
    CHECK(state_of(kMain) == ThreadState::Ready);
    CHECK(state_of(kWorkerA) == ThreadState::Ready);
    CHECK_FALSE(os.sync.events[kEvent].signalled); // consumed by the waiters
}

TEST_CASE_METHOD(SyncFixture,
                 "threads that may run anywhere are spread across cores",
                 "[hle][scheduler][sync]") {
    // OSCreateThread with no affinity bits means "any core". Putting every
    // such thread on core 0 makes anything that coordinates between cores --
    // a rendezvous, most obviously -- wait for a core nothing is on.
    spawn_ready(kWorkerA);
    spawn_ready(kWorkerB);

    const Thread* const first = os.scheduler.find(kWorkerA);
    const Thread* const second = os.scheduler.find(kWorkerB);
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    CHECK(first->core != second->core);

    // A thread pinned to one core still lands there.
    constexpr u32 kPinned = 0x10008000;
    spawn_ready(kPinned, 0x4);
    CHECK(os.scheduler.find(kPinned)->core == 2U);
}

TEST_CASE_METHOD(SchedulerFixture, "timers fire in deadline order", "[hle][scheduler][timer]") {
    // Out of order means a periodic alarm can overtake a one-shot set before
    // it, and a title that arms two alarms to sequence something gets them
    // backwards.
    std::vector<u32> fired;
    scheduler.set_timer(1, 300, [&fired](u64) { fired.push_back(1); });
    scheduler.set_timer(2, 100, [&fired](u64) { fired.push_back(2); });
    scheduler.set_timer(3, 200, [&fired](u64) { fired.push_back(3); });

    CHECK(scheduler.next_deadline() == 100U);
    CHECK(scheduler.service_timers(250) == 2U);
    CHECK(fired == std::vector<u32>{2, 3});
    CHECK(scheduler.has_timer(1));

    CHECK(scheduler.service_timers(300) == 1U);
    CHECK(fired == std::vector<u32>{2, 3, 1});
    CHECK(scheduler.timer_count() == 0);
}

TEST_CASE_METHOD(SchedulerFixture, "only an exited thread can be forgotten", "[hle][scheduler]") {
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    // Removing a live thread would take its context with it, and the handle
    // could then be handed to something else.
    CHECK_FALSE(scheduler.remove(kWorkerA));
    CHECK(scheduler.find(kWorkerA) != nullptr);

    scheduler.exit_current(state, 0); // exits main
    CHECK(scheduler.remove(kMain));
    CHECK(scheduler.find(kMain) == nullptr);

    // ... and the scheduler no longer believes the core is running it, which
    // is what lets the handle be reused.
    CHECK(scheduler.current_handle() != kMain);
}

TEST_CASE_METHOD(SchedulerFixture,
                 "the clock does not go backwards across a switch",
                 "[hle][scheduler]") {
    // The timebase lives in CoreState because that is where the interpreter
    // counts it, but it belongs to the machine: it only ever goes forward, and
    // every deadline in the scheduler is measured against it. Restoring it
    // wholesale with the rest of a thread's context handed the core whatever
    // that thread last saw -- zero, for one that had not run yet -- so an
    // alarm or an asynchronous completion armed before the switch came due the
    // moment the clock was put back.
    static_cast<void>(spawn(kWorkerA));
    REQUIRE(scheduler.resume(kWorkerA));

    state.timebase = 5'000'000;
    REQUIRE(scheduler.reschedule(state));
    REQUIRE(scheduler.current_handle() == kWorkerA);
    CHECK(state.timebase == 5'000'000U);

    state.timebase = 6'000'000;
    REQUIRE(scheduler.reschedule(state));
    REQUIRE(scheduler.current_handle() == kMain);
    CHECK(state.timebase == 6'000'000U);
}

TEST_CASE_METHOD(SchedulerFixture,
                 "equal-priority waiters are woken in the order they waited",
                 "[hle][scheduler]") {
    // Cafe OS thread queues are FIFO within a priority level. Without a
    // tiebreaker the order fell out of std::map's, which is by handle -- and a
    // handle is wherever the title happened to allocate its OSThread. Which of
    // two threads was served first then depended on the title's allocator.
    constexpr u32 object = 0x10009000;

    // kWorkerB has the higher handle, and waits first.
    static_cast<void>(spawn(kWorkerB));
    static_cast<void>(spawn(kWorkerA));
    scheduler.block_thread(kWorkerB, BlockReason::Mutex, object);
    scheduler.block_thread(kWorkerA, BlockReason::Mutex, object);

    CHECK(scheduler.waiters(object) == std::vector<u32>{kWorkerB, kWorkerA});
    CHECK(scheduler.unblock_one(object) == kWorkerB);

    // Priority still comes first, though.
    scheduler.block_thread(kWorkerB, BlockReason::Mutex, object);
    scheduler.find(kWorkerA)->priority = 5;
    CHECK(scheduler.waiters(object) == std::vector<u32>{kWorkerA, kWorkerB});
}

TEST_CASE_METHOD(SchedulerFixture, "a reset scheduler keeps no deadlines", "[hle][scheduler]") {
    scheduler.set_timer(0x1234, 500, [](u64) {});
    REQUIRE(scheduler.has_timer(0x1234));

    scheduler.reset();
    CHECK(scheduler.timer_count() == 0);
    CHECK_FALSE(scheduler.next_deadline().has_value());
}

// ---------------------------------------------------------------------------
// Message queues (6.5)
// ---------------------------------------------------------------------------
//
// The queue a title waits on for anything the system hands it. The rule these
// are mostly about: a woken thread resumes *after* the call it blocked in, so
// whoever wakes a receiver has to have delivered the message already.

TEST_CASE_METHOD(SyncFixture, "a message queue is first in, first out", "[hle][scheduler][queue]") {
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    CHECK(send(kQueue, 0x1111, 0) == 1U);
    CHECK(send(kQueue, 0x2222, 0) == 1U);
    CHECK(send(kQueue, 0x3333, 0) == 1U);

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x1111U);
    // The other three words come with it; a title that packs a command into
    // args[0] and gets only `message` back is the bug this is watching for.
    CHECK(memory.read<u32>(kOutbox + 4) == (0x1111U ^ 0xA5A5A5A5U));

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x2222U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x3333U);

    // Empty, and the caller did not ask to wait.
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 0U);
    CHECK(state_of(kMain) == ThreadState::Running);
}

TEST_CASE_METHOD(SyncFixture, "OSPeekMessage leaves the message alone", "[hle][scheduler][queue]") {
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));
    CHECK(call("OSPeekMessage", {kQueue, kOutbox}) == 0U); // empty

    CHECK(send(kQueue, 0x4444, 0) == 1U);
    CHECK(call("OSPeekMessage", {kQueue, kOutbox}) == 1U);
    CHECK(received() == 0x4444U);

    // Still there.
    CHECK(os.sync.message_queues[kQueue].entries.size() == 1U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x4444U);
}

TEST_CASE_METHOD(SyncFixture, "a jammed message goes to the head", "[hle][scheduler][queue]") {
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    CHECK(send(kQueue, 0x1111, 0) == 1U);
    CHECK(send(kQueue, 0x2222, 0, "OSJamMessage") == 1U);
    // OS_MESSAGE_QUEUE_FLAG_HIGH_PRIORITY on an ordinary send does the same
    // thing, which is how most titles jam one.
    CHECK(send(kQueue, 0x3333, coreinit::message_flag_high_priority) == 1U);

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x3333U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x2222U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x1111U);
}

TEST_CASE_METHOD(SyncFixture, "a full queue refuses a message", "[hle][scheduler][queue]") {
    // The capacity is the title's: it sized the array, and a send that ignored
    // the size would overrun what it thinks it allocated.
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 2}));

    CHECK(send(kQueue, 0x1111, 0) == 1U);
    CHECK(send(kQueue, 0x2222, 0) == 1U);
    CHECK(send(kQueue, 0x3333, 0) == 0U);
    CHECK(state_of(kMain) == ThreadState::Running); // and it did not wait
    CHECK(os.sync.message_queues[kQueue].entries.size() == 2U);
}

TEST_CASE_METHOD(SyncFixture,
                 "a blocking send waits for room and keeps its place",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 1}));

    CHECK(send(kQueue, 0x1111, 0) == 1U);
    static_cast<void>(send(kQueue, 0x2222, coreinit::message_flag_block));

    // Full, so the sender parks -- with its message, because by the time there
    // is room this handler is long gone and nothing else knows what it wanted
    // to send.
    CHECK(state_of(kMain) == ThreadState::Blocked);
    CHECK(os.scheduler.current_handle() == kWorkerA);
    CHECK(os.sync.message_queues[kQueue].pending_sends.size() == 1U);

    // The worker takes one out, which lets the parked sender in.
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x1111U);
    CHECK(state_of(kMain) == ThreadState::Ready);
    CHECK(os.sync.message_queues[kQueue].pending_sends.empty());

    // And it went in behind what was already queued, not in front of it.
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x2222U);
}

TEST_CASE_METHOD(SyncFixture,
                 "a waiting receiver is handed the message directly",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutbox, coreinit::message_flag_block}));
    CHECK(state_of(kMain) == ThreadState::Blocked);
    CHECK(os.scheduler.current_handle() == kWorkerA);

    CHECK(send(kQueue, 0x5555, 0) == 1U);

    // The message is already where the receiver asked for it. It resumes after
    // its OSReceiveMessage and never looks at the queue again, so delivering it
    // at wake time is the only moment there is.
    CHECK(state_of(kMain) == ThreadState::Ready);
    CHECK(received() == 0x5555U);
    CHECK(memory.read<u32>(kOutbox + 4) == (0x5555U ^ 0xA5A5A5A5U));

    // And it did not also land in the ring.
    CHECK(os.sync.message_queues[kQueue].entries.empty());
    CHECK(os.sync.message_queues[kQueue].pending_receives.empty());
}

TEST_CASE_METHOD(SyncFixture,
                 "the highest-priority receiver is served first",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    spawn_ready(kWorkerB);
    os.scheduler.find(kWorkerA)->priority = 5; // higher than the main thread's 16

    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    // Main parks first, then the worker that runs next parks too; the last
    // runnable thread does the sending.
    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutbox, coreinit::message_flag_block}));
    REQUIRE(os.scheduler.current_handle() == kWorkerA);
    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutboxB, coreinit::message_flag_block}));
    REQUIRE(os.scheduler.current_handle() == kWorkerB);

    CHECK(send(kQueue, 0x6666, 0) == 1U);

    // Priority order, not arrival order: the worker waited second and is
    // served first.
    CHECK(state_of(kWorkerA) == ThreadState::Ready);
    CHECK(state_of(kMain) == ThreadState::Blocked);
    CHECK(received(kOutboxB) == 0x6666U);
    CHECK(received(kOutbox) == 0U);
}

TEST_CASE_METHOD(SyncFixture,
                 "a queue that was never initialised is refused",
                 "[hle][scheduler][queue]") {
    // Zero capacity is not an empty queue, it is a queue nothing can ever go
    // through. Treating it as empty would leave a title blocked on a send that
    // can never be satisfied, with nothing to say why.
    CHECK(send(kQueue, 0x1111, coreinit::message_flag_block) == 0U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, coreinit::message_flag_block}) == 0U);
    CHECK(call("OSPeekMessage", {kQueue, kOutbox}) == 0U);
    CHECK(state_of(kMain) == ThreadState::Running);

    // Nor does a queue sized zero or negative pretend to work.
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 0}));
    CHECK(send(kQueue, 0x1111, 0) == 0U);
}

TEST_CASE_METHOD(SyncFixture,
                 "the system can post to a queue from outside a call",
                 "[hle][scheduler][queue]") {
    // This is the path an asynchronous FS completion takes (6.9): it fires
    // from a timer, where there is no guest call to hang the delivery off.
    spawn_ready(kWorkerA);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 1}));

    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutbox, coreinit::message_flag_block}));
    REQUIRE(state_of(kMain) == ThreadState::Blocked);

    coreinit::Message message;
    message.value = 0x7777;
    message.args[0] = 0x8888;
    CHECK(coreinit::post_message(os, memory, kQueue, message));

    CHECK(state_of(kMain) == ThreadState::Ready);
    CHECK(received() == 0x7777U);
    CHECK(memory.read<u32>(kOutbox + 4) == 0x8888U);

    // With nobody waiting it queues; with no room it has nowhere to wait, so it
    // says no rather than losing the message silently.
    CHECK(coreinit::post_message(os, memory, kQueue, message));
    CHECK_FALSE(coreinit::post_message(os, memory, kQueue, message));
}

TEST_CASE_METHOD(SyncFixture,
                 "a thread that exits while parked is not delivered to",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutbox, coreinit::message_flag_block}));
    REQUIRE(os.scheduler.current_handle() == kWorkerA);

    // The main thread goes away while parked. Handles are guest OSThread
    // pointers and do get reused, so a record left behind could be handed a
    // message meant for a thread that no longer exists.
    Thread* const parked = os.scheduler.find(kMain);
    REQUIRE(parked != nullptr);
    parked->state = ThreadState::Exited;

    CHECK(send(kQueue, 0x9999, 0) == 1U);
    CHECK(received() == 0U); // nothing was written into the dead thread's outbox
    CHECK(os.sync.message_queues[kQueue].entries.size() == 1U);
    CHECK(os.sync.message_queues[kQueue].pending_receives.empty());
}

TEST_CASE_METHOD(SyncFixture,
                 "equal-priority receivers are served in the order they waited",
                 "[hle][scheduler][queue]") {
    // kWorkerB has the higher handle and parks first, so a queue that served
    // by handle would serve kWorkerA. The order two messages come back in
    // would then depend on where the title allocated its OSThreads.
    spawn_ready(kWorkerA);
    spawn_ready(kWorkerB);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    // Park kWorkerB first, then kWorkerA, leaving the main thread free to send.
    REQUIRE(os.scheduler.suspend(kWorkerA));
    REQUIRE(os.scheduler.reschedule(state));
    REQUIRE(os.scheduler.current_handle() == kWorkerB);
    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutboxB, coreinit::message_flag_block}));

    REQUIRE(os.scheduler.suspend(kMain));
    REQUIRE(os.scheduler.resume(kWorkerA));
    REQUIRE(os.scheduler.reschedule(state));
    REQUIRE(os.scheduler.current_handle() == kWorkerA);
    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutbox, coreinit::message_flag_block}));

    coreinit::Message message;
    message.value = 0xAAAA;
    CHECK(coreinit::post_message(os, memory, kQueue, message));

    CHECK(received(kOutboxB) == 0xAAAAU); // waited first
    CHECK(received(kOutbox) == 0U);
    CHECK(state_of(kWorkerB) == ThreadState::Ready);
    CHECK(state_of(kWorkerA) == ThreadState::Blocked);
}

TEST_CASE_METHOD(SyncFixture,
                 "equal-priority senders are admitted in the order they waited",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    spawn_ready(kWorkerB);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 1}));
    CHECK(send(kQueue, 0x1111, 0) == 1U); // fills it

    // Same shape as the receiver case: the higher handle parks first.
    REQUIRE(os.scheduler.suspend(kWorkerA));
    REQUIRE(os.scheduler.reschedule(state));
    REQUIRE(os.scheduler.current_handle() == kWorkerB);
    static_cast<void>(send(kQueue, 0x2222, coreinit::message_flag_block));

    REQUIRE(os.scheduler.suspend(kMain));
    REQUIRE(os.scheduler.resume(kWorkerA));
    REQUIRE(os.scheduler.reschedule(state));
    REQUIRE(os.scheduler.current_handle() == kWorkerA);
    static_cast<void>(send(kQueue, 0x3333, coreinit::message_flag_block));

    REQUIRE(os.scheduler.resume(kMain));
    REQUIRE(os.scheduler.reschedule(state));
    REQUIRE(os.scheduler.current_handle() == kMain);

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x1111U);

    // Exactly one sender gets in per message taken out, and it is the one that
    // has been waiting longest.
    CHECK(state_of(kWorkerB) == ThreadState::Ready);
    CHECK(state_of(kWorkerA) == ThreadState::Blocked);
    CHECK(os.sync.message_queues[kQueue].pending_sends.size() == 1U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x2222U);
}

TEST_CASE_METHOD(SyncFixture,
                 "the highest-priority sender is admitted first",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    spawn_ready(kWorkerB);
    os.scheduler.find(kWorkerB)->priority = 5; // higher than the others' 16
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 1}));
    CHECK(send(kQueue, 0x1111, 0) == 1U);

    // kWorkerA parks first, but kWorkerB outranks it.
    REQUIRE(os.scheduler.suspend(kWorkerB));
    REQUIRE(os.scheduler.reschedule(state));
    REQUIRE(os.scheduler.current_handle() == kWorkerA);
    static_cast<void>(send(kQueue, 0x2222, coreinit::message_flag_block));

    REQUIRE(os.scheduler.suspend(kMain));
    REQUIRE(os.scheduler.resume(kWorkerB));
    REQUIRE(os.scheduler.reschedule(state));
    REQUIRE(os.scheduler.current_handle() == kWorkerB);
    static_cast<void>(send(kQueue, 0x3333, coreinit::message_flag_block));

    REQUIRE(os.scheduler.resume(kMain));
    REQUIRE(os.scheduler.reschedule(state));
    static_cast<void>(call("OSReceiveMessage", {kQueue, kOutbox, 0}));

    CHECK(state_of(kWorkerB) == ThreadState::Ready);
    CHECK(state_of(kWorkerA) == ThreadState::Blocked);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x3333U);
}

TEST_CASE_METHOD(SyncFixture,
                 "a parked send comes back true, with everything it was sending",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 1}));
    CHECK(send(kQueue, 0x1111, 0) == 1U);

    // The sender sets its return value before it parks -- it cannot set one
    // afterwards, because by then it is not the thread running. A blocking
    // send only ever comes back having sent, so that value is TRUE.
    static_cast<void>(send(kQueue, 0x2222, coreinit::message_flag_block));
    REQUIRE(state_of(kMain) == ThreadState::Blocked);
    CHECK(os.scheduler.find(kMain)->context.gpr[3] == 1U);

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);

    // All four words were parked with it, not just the payload.
    CHECK(received() == 0x2222U);
    CHECK(memory.read<u32>(kOutbox + 4) == (0x2222U ^ 0xA5A5A5A5U));
}

TEST_CASE_METHOD(SyncFixture, "a parked jam is still a jam", "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 2}));
    CHECK(send(kQueue, 0x1111, 0) == 1U);
    CHECK(send(kQueue, 0x2222, 0) == 1U); // full

    // Waiting for room does not cost a message its place at the head.
    static_cast<void>(send(kQueue, 0x3333, coreinit::message_flag_block, "OSJamMessage"));
    REQUIRE(state_of(kMain) == ThreadState::Blocked);

    // The receiver takes what was already at the head, and the jam goes in
    // front of what is left -- not behind it.
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x1111U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x3333U);
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x2222U);
}

TEST_CASE_METHOD(SyncFixture,
                 "a sender that exits while parked is not admitted",
                 "[hle][scheduler][queue]") {
    spawn_ready(kWorkerA);
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 1}));
    CHECK(send(kQueue, 0x1111, 0) == 1U);
    static_cast<void>(send(kQueue, 0x2222, coreinit::message_flag_block));
    REQUIRE(os.scheduler.current_handle() == kWorkerA);

    Thread* const parked = os.scheduler.find(kMain);
    REQUIRE(parked != nullptr);
    parked->state = ThreadState::Exited;

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x1111U);

    // The dead thread's message is not admitted, and its record is gone: a
    // handle is a guest OSThread pointer and gets reused.
    CHECK(os.sync.message_queues[kQueue].entries.empty());
    CHECK(os.sync.message_queues[kQueue].pending_sends.empty());
}

TEST_CASE_METHOD(SyncFixture,
                 "a failed receive leaves the caller's message alone",
                 "[hle][scheduler][queue]") {
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));
    REQUIRE(memory.try_write<u32>(kOutbox, 0xFEEDFACE));

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 0U);
    CHECK(received() == 0xFEEDFACEU); // untouched, as a failed call leaves it

    CHECK(call("OSPeekMessage", {kQueue, kOutbox}) == 0U);
    CHECK(received() == 0xFEEDFACEU);
}

TEST_CASE_METHOD(SyncFixture, "re-initialising a queue empties it", "[hle][scheduler][queue]") {
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));
    CHECK(send(kQueue, 0x1111, 0) == 1U);

    // OSInitMessageQueue is what a title calls to (re)claim a queue; whatever
    // was in it belongs to the queue it used to be.
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 1}));
    CHECK(os.sync.message_queues[kQueue].entries.empty());
    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 0U);

    // ... and the new capacity is the one in force.
    CHECK(send(kQueue, 0x2222, 0) == 1U);
    CHECK(send(kQueue, 0x3333, 0) == 0U);

    // A negative count is refused the same way zero is.
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 0xFFFFFFFF}));
    CHECK(send(kQueue, 0x4444, 0) == 0U);
}

TEST_CASE_METHOD(SyncFixture, "the system can jam a message too", "[hle][scheduler][queue]") {
    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));
    CHECK(send(kQueue, 0x1111, 0) == 1U);

    coreinit::Message urgent;
    urgent.value = 0x2222;
    CHECK(coreinit::post_message(os, memory, kQueue, urgent, true));

    CHECK(call("OSReceiveMessage", {kQueue, kOutbox, 0}) == 1U);
    CHECK(received() == 0x2222U);
}
