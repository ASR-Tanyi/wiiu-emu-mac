// Phase 6.7 / 6.9 — callbacks from the OS into the title.
//
// An alarm handler and an asynchronous read completing are both the same
// problem: guest code that has to run on a stack, at a time nobody called it.
// These tests drive the whole path -- HLE call, timer, scheduler, interpreter
// -- rather than checking that a structure was filled in, because the failure
// this is guarding against is a callback that is scheduled beautifully and
// never actually runs.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/loader/vfs.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;

namespace {

constexpr ppcaddr_t kMainProgram = 0x02000000;
constexpr ppcaddr_t kCallbackProgram = 0x02001000;

/// Where the callback program keeps its results: a count, then the two
/// arguments it was given.
constexpr ppcaddr_t kScratch = 0x10002000;

// Assembled with binutils.
constexpr u32 kEnc_mflr_r31 = 0x7FE802A6;
constexpr u32 kEnc_mtctr_r12 = 0x7D8903A6;
constexpr u32 kEnc_bctrl = 0x4E800421;
constexpr u32 kEnc_mtlr_r31 = 0x7FE803A6;
constexpr u32 kEnc_blr = 0x4E800020;

[[nodiscard]] constexpr u32 lis_r12(u32 value) {
    return 0x3D800000U | (value >> 16);
}

[[nodiscard]] constexpr u32 ori_r12(u32 value) {
    return 0x618C0000U | (value & 0xFFFF);
}

[[nodiscard]] constexpr u32 li(u32 reg, u32 value) {
    return 0x38000000U | (reg << 21) | (value & 0xFFFF);
}

/// `lis reg,hi` then `ori reg,reg,lo`: a full 32-bit constant, which the
/// addresses these tests pass around all are.
[[nodiscard]] constexpr u32 lis(u32 reg, u32 value) {
    return 0x3C000000U | (reg << 21) | (value >> 16);
}

[[nodiscard]] constexpr u32 ori(u32 reg, u32 value) {
    return 0x60000000U | (reg << 21) | (reg << 16) | (value & 0xFFFF);
}

/// `stw src,offset(base)`.
[[nodiscard]] constexpr u32 stw(u32 src, u32 offset, u32 base) {
    return 0x90000000U | (src << 21) | (base << 16) | (offset & 0xFFFF);
}

/// A machine with coreinit registered, a scheduler, and room to write small
/// programs by hand.
struct CallbackFixture {
    memory::Memory memory;
    cpu::CoreState state;
    cpu::Interpreter interpreter{state, memory};
    Registry registry;
    coreinit::Context os;
    loader::VirtualFileSystem vfs;

    CallbackFixture() {
        // The trampoline area is outside every Cafe OS region, so it has to be
        // mapped explicitly, as the loader's own machine does.
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
        wiiu::log::set_all_levels(std::getenv("WIIU_TEST_LOG") != nullptr ? wiiu::log::Level::Trace
                                                                          : wiiu::log::Level::Off);

        coreinit::register_all(registry, os);
        REQUIRE(os.heaps.create_default_heaps(memory, error));
        REQUIRE(registry.install(memory, error));

        REQUIRE(vfs.mount_title_directory(
            std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title", error));
        os.filesystem.set_filesystem(&vfs);
        os.memory = &memory;

        write_callback_program();

        state.pc = kMainProgram;
        state.gpr[1] = memory::data_heap_base + 0x20000;
        state.lr = return_sentinel;
        static_cast<void>(os.scheduler.adopt_main(state, main_thread_handle));
    }

    /// Counts its calls at kScratch and records its first two arguments after
    /// it, so a test can tell not just that it ran but what it was handed.
    void write_callback_program() {
        const std::array<u32, 8> program{
            0x3CA01000, // lis   r5,0x1000
            0x60A52000, // ori   r5,r5,0x2000
            0x80C50000, // lwz   r6,0(r5)
            0x38C60001, // addi  r6,r6,1
            0x90C50000, // stw   r6,0(r5)
            0x90650004, // stw   r3,4(r5)
            0x90850008, // stw   r4,8(r5)
            kEnc_blr,
        };
        write_program(kCallbackProgram, program);
        REQUIRE(memory.fill(kScratch, 16, 0));
    }

    template <typename Container>
    void write_program(ppcaddr_t at, const Container& words) {
        ppcaddr_t address = at;
        for (const u32 word : words) {
            REQUIRE(memory.try_write<u32>(address, word));
            address += 4;
        }
    }

    /// A main program that calls one coreinit function with one argument and
    /// then returns, which is all most of these tests need it to do.
    void write_main_calling(std::string_view function, u32 argument) {
        write_main_program(function, {li(3, argument)});
    }

    /// OSSleepTicks takes an OSTime: 64 bits, so r3:r4, high half first.
    void write_main_sleeping(u32 ticks) {
        write_main_program("OSSleepTicks", {li(3, 0), li(4, ticks)});
    }

    void write_main_program(std::string_view function, std::initializer_list<u32> arguments) {
        const ppcaddr_t trampoline = registry.resolve("coreinit", function);
        std::vector<u32> program{
            kEnc_mflr_r31,
            lis_r12(trampoline),
            ori_r12(trampoline),
            kEnc_mtctr_r12,
        };
        program.insert(program.end(), arguments);
        program.insert(program.end(), {kEnc_bctrl, kEnc_mtlr_r31, li(3, 0), kEnc_blr});
        write_program(kMainProgram, program);

        std::string error;
        REQUIRE(registry.install(memory, error));
    }

    [[nodiscard]] u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("coreinit", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size() && i < 8; ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    RunReport run_it(u64 budget = 200'000) {
        return run(interpreter,
                   memory,
                   registry,
                   {.scheduler = &os.scheduler, .budget = budget, .timeslice = 200});
    }

    [[nodiscard]] u32 callback_count() { return memory.read<u32>(kScratch); }

    [[nodiscard]] u32 callback_argument(u32 index) {
        return memory.read<u32>(kScratch + 4 + index * 4);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// The mechanism
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CallbackFixture, "a guest callback runs on its own thread", "[hle][callback]") {
    const std::array<u32, 2> arguments{0xABCD, 0x1234};
    const u32 handle =
        coreinit::invoke_guest_callback(os, memory, "test", kCallbackProgram, arguments);
    REQUIRE(handle != 0);

    const Thread* const thread = os.scheduler.find(handle);
    REQUIRE(thread != nullptr);
    CHECK(thread->state == ThreadState::Ready);
    CHECK(thread->context.pc == kCallbackProgram);
    CHECK(thread->context.gpr[3] == 0xABCDU);
    CHECK(thread->stack_base != 0U);

    // The main thread has nothing to do; the callback does.
    write_main_sleeping(4000);
    const RunReport report = run_it();

    INFO("outcome: " << run_outcome_name(report.outcome));
    CHECK(callback_count() == 1U);
    CHECK(callback_argument(0) == 0xABCDU);
    CHECK(callback_argument(1) == 0x1234U);
}

TEST_CASE_METHOD(CallbackFixture, "callback stacks are recycled", "[hle][callback]") {
    // A title with a periodic alarm fires one of these every frame. One stack
    // per firing is a leak with a slow fuse, which is the kind that gets
    // diagnosed as "the emulator runs out of memory in long sessions".
    const std::array<u32, 1> arguments{1};

    const u32 first = coreinit::invoke_guest_callback(os, memory, "a", kCallbackProgram, arguments);
    REQUIRE(first != 0);
    const ppcaddr_t stack = os.scheduler.find(first)->stack_base;

    write_main_sleeping(4000);
    static_cast<void>(run_it());
    REQUIRE(callback_count() == 1U);

    const u32 second =
        coreinit::invoke_guest_callback(os, memory, "b", kCallbackProgram, arguments);
    REQUIRE(second != 0);

    // The stack *and* the handle come back together. Handing out a fresh
    // handle every time walks up through the main thread's handle in about
    // eighteen minutes of a 60 Hz alarm, and then into the addresses real
    // OSThread pointers have.
    CHECK(second == first);
    CHECK(os.scheduler.find(second)->stack_base == stack);
    CHECK(os.scheduler.threads().size() == 2); // main, and the one callback

    // The recycled thread has to be a *fresh* one. Reusing the handle while
    // the scheduler still thought that thread was running left the new one
    // with the old one's registers, and it exited without executing an
    // instruction.
    static_cast<void>(run_it());
    CHECK(callback_count() == 2U);
}

// ---------------------------------------------------------------------------
// Idle skipping (P2)
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CallbackFixture,
                 "an idle wait skips ahead instead of deadlocking",
                 "[hle][callback][idle]") {
    // Every thread asleep means no instructions run, which without the skip
    // looks exactly like a deadlock -- and a sleeping thread can never wake,
    // because nothing is advancing the clock it is waiting on.
    write_main_sleeping(20000);

    const RunReport report = run_it();
    CHECK(report.outcome == RunOutcome::Returned);
    CHECK(report.skipped_ticks >= 20000U);

    // And it cost nothing: the wait was jumped, not run.
    CHECK(report.instructions < 100U);
}

// ---------------------------------------------------------------------------
// Alarms
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CallbackFixture, "a one-shot alarm fires once", "[hle][alarm]") {
    constexpr u32 kAlarm = 0x10003000;

    static_cast<void>(call("OSCreateAlarm", {kAlarm}));
    // OSSetAlarm(alarm, ticks, callback): the tick count is 64-bit, so it
    // occupies r5:r6 and the callback lands in r7.
    static_cast<void>(call("OSSetAlarm", {kAlarm, 0, 0, 1000, kCallbackProgram}));
    CHECK(os.scheduler.has_timer(kAlarm));

    write_main_sleeping(8000);
    const RunReport report = run_it();

    INFO("outcome: " << run_outcome_name(report.outcome));
    CHECK(callback_count() == 1U);
    CHECK(callback_argument(0) == kAlarm); // OSAlarmCallback(OSAlarm*, OSContext*)
    CHECK_FALSE(os.scheduler.has_timer(kAlarm));
    CHECK(os.alarms[kAlarm].fire_count == 1U);
}

TEST_CASE_METHOD(CallbackFixture, "a periodic alarm keeps firing", "[hle][alarm]") {
    constexpr u32 kAlarm = 0x10003000;

    static_cast<void>(call("OSCreateAlarm", {kAlarm}));
    // OSSetPeriodicAlarm(alarm, start, interval, callback): two 64-bit
    // arguments, so r5:r6 and r7:r8, with the callback in r9.
    static_cast<void>(call("OSSetPeriodicAlarm", {kAlarm, 0, 0, 1000, 0, 1000, kCallbackProgram}));

    write_main_sleeping(5500);
    const RunReport report = run_it();
    INFO("outcome: " << run_outcome_name(report.outcome) << ", fired "
                     << os.alarms[kAlarm].fire_count << " times, callback ran " << callback_count()
                     << " times");

    // Deadlines at 1000, 2000, 3000, 4000, 5000 -- the sleep ends at 5500.
    // Both halves matter: an alarm firing five times is not the same as the
    // callback running five times, and a recycled thread that never ran put
    // exactly that gap between them.
    CHECK(os.alarms[kAlarm].fire_count == 5U);
    CHECK(callback_count() == 5U);
    CHECK(os.scheduler.has_timer(kAlarm)); // and it is still armed
}

TEST_CASE_METHOD(CallbackFixture, "a cancelled alarm does not fire", "[hle][alarm]") {
    constexpr u32 kAlarm = 0x10003000;

    static_cast<void>(call("OSCreateAlarm", {kAlarm}));
    static_cast<void>(call("OSSetAlarm", {kAlarm, 0, 0, 1000, kCallbackProgram}));
    CHECK(call("OSCancelAlarm", {kAlarm}) == 1U);
    CHECK_FALSE(os.scheduler.has_timer(kAlarm));

    write_main_sleeping(8000);
    static_cast<void>(run_it());
    CHECK(callback_count() == 0U);

    // Cancelling one that was never armed says so rather than pretending.
    CHECK(call("OSCancelAlarm", {kAlarm}) == 0U);
}

TEST_CASE_METHOD(CallbackFixture, "alarms carry user data and a tag", "[hle][alarm]") {
    constexpr u32 kAlarm = 0x10003000;
    constexpr u32 kOther = 0x10003100;

    static_cast<void>(call("OSCreateAlarm", {kAlarm}));
    static_cast<void>(call("OSCreateAlarm", {kOther}));
    static_cast<void>(call("OSSetAlarmUserData", {kAlarm, 0xFEEDFACE}));
    CHECK(call("OSGetAlarmUserData", {kAlarm}) == 0xFEEDFACEU);

    // OSCancelAlarms cancels by tag, which is how a title drops a whole
    // subsystem's alarms at once.
    static_cast<void>(call("OSSetAlarmTag", {kAlarm, 7}));
    static_cast<void>(call("OSSetAlarmTag", {kOther, 9}));
    static_cast<void>(call("OSSetAlarm", {kAlarm, 0, 0, 1000, kCallbackProgram}));
    static_cast<void>(call("OSSetAlarm", {kOther, 0, 0, 1000, kCallbackProgram}));

    CHECK(call("OSCancelAlarms", {7}) == 1U);
    CHECK_FALSE(os.scheduler.has_timer(kAlarm));
    CHECK(os.scheduler.has_timer(kOther));
}

TEST_CASE_METHOD(CallbackFixture, "waiting on an alarm blocks until it fires", "[hle][alarm]") {
    constexpr u32 kAlarm = 0x10003000;

    static_cast<void>(call("OSCreateAlarm", {kAlarm}));
    static_cast<void>(call("OSSetAlarm", {kAlarm, 0, 0, 1000, 0})); // no callback

    CHECK(call("OSWaitAlarm", {kAlarm}) == 1U);
    const Thread* const main = os.scheduler.find(main_thread_handle);
    REQUIRE(main != nullptr);
    CHECK(main->state == ThreadState::Blocked);

    // The alarm firing releases it, even with no callback to run.
    state.timebase = 1000;
    CHECK(os.scheduler.service_timers(state.timebase) == 1U);
    CHECK(main->state == ThreadState::Ready);

    // Waiting on an alarm that is not armed returns instead of blocking
    // forever, which is the difference between a title carrying on and one
    // that stops for no visible reason.
    CHECK(call("OSWaitAlarm", {kAlarm}) == 0U);
}

// ---------------------------------------------------------------------------
// Asynchronous FS
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CallbackFixture,
                 "an asynchronous read calls back with its status",
                 "[hle][fs][async]") {
    constexpr ppcaddr_t kAsyncData = 0x10004000;
    constexpr ppcaddr_t kPath = 0x10004100;
    constexpr ppcaddr_t kMode = 0x10004200;
    constexpr ppcaddr_t kHandleOut = 0x10004300;
    constexpr ppcaddr_t kBuffer = 0x10004400;
    constexpr u32 kClient = 0x10005000;
    constexpr u32 kBlock = 0x10006000;

    // FSAsyncData: callback, context word, message queue.
    REQUIRE(memory.try_write<u32>(kAsyncData, kCallbackProgram));
    REQUIRE(memory.try_write<u32>(kAsyncData + 4, 0xC0DE));
    REQUIRE(memory.try_write<u32>(kAsyncData + 8, 0));

    const std::string path = "/vol/content/data.bin";
    REQUIRE(memory.write_bytes(kPath, std::vector<u8>(path.begin(), path.end())));
    REQUIRE(memory.try_write<u8>(kPath + static_cast<u32>(path.size()), 0));
    REQUIRE(memory.try_write<u8>(kMode, 'r'));
    REQUIRE(memory.try_write<u8>(kMode + 1, 0));

    static_cast<void>(call("FSOpenFile", {kClient, kBlock, kPath, kMode, kHandleOut, 0xFFFFFFFF}));
    const u32 handle = memory.read<u32>(kHandleOut);
    REQUIRE(handle != 0U);

    static_cast<void>(
        call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, handle, 0, kAsyncData}));

    // The completion is scheduled, not immediate: a title that starts a read
    // and then sets a flag expects its callback to run after the call.
    CHECK(callback_count() == 0U);
    CHECK(os.scheduler.timer_count() == 1U);

    write_main_sleeping(8000);
    static_cast<void>(run_it());

    CHECK(callback_count() == 1U);
    CHECK(callback_argument(0) == kClient); // FSAsyncCallback(FSClient*, FSCmdBlock*, ...)
    CHECK(callback_argument(1) == kBlock);

    // ... and the data really arrived.
    CHECK(memory.read<u8>(kBuffer) != 0U);
}

TEST_CASE_METHOD(CallbackFixture,
                 "an asynchronous read can report through a message queue",
                 "[hle][fs][async][queue]") {
    // The other half of the API: instead of a callback, the completion is
    // posted to a queue the title waits on, and FSGetAsyncResult turns the
    // message back into the result.
    constexpr ppcaddr_t kAsyncData = 0x10004000;
    constexpr ppcaddr_t kPath = 0x10004100;
    constexpr ppcaddr_t kMode = 0x10004200;
    constexpr ppcaddr_t kHandleOut = 0x10004300;
    constexpr ppcaddr_t kBuffer = 0x10004400;
    constexpr ppcaddr_t kQueue = 0x10004500;
    constexpr ppcaddr_t kRing = 0x10004600;
    constexpr ppcaddr_t kInbox = 0x10004800;
    constexpr u32 kClient = 0x10005000;
    constexpr u32 kBlock = 0x10006000;

    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    // FSAsyncData: no callback this time, and a queue instead.
    REQUIRE(memory.try_write<u32>(kAsyncData, 0));
    REQUIRE(memory.try_write<u32>(kAsyncData + 4, 0xC0DE));
    REQUIRE(memory.try_write<u32>(kAsyncData + 8, kQueue));

    const std::string path = "/vol/content/data.bin";
    REQUIRE(memory.write_bytes(kPath, std::vector<u8>(path.begin(), path.end())));
    REQUIRE(memory.try_write<u8>(kPath + static_cast<u32>(path.size()), 0));
    REQUIRE(memory.try_write<u8>(kMode, 'r'));
    REQUIRE(memory.try_write<u8>(kMode + 1, 0));

    static_cast<void>(call("FSOpenFile", {kClient, kBlock, kPath, kMode, kHandleOut, 0xFFFFFFFF}));
    const u32 handle = memory.read<u32>(kHandleOut);
    REQUIRE(handle != 0U);

    static_cast<void>(
        call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, handle, 0, kAsyncData}));

    // Nothing has been posted yet -- the completion is a deadline away.
    CHECK(call("OSPeekMessage", {kQueue, kInbox}) == 0U);
    CHECK(os.scheduler.timer_count() == 1U);

    write_main_sleeping(8000);
    static_cast<void>(run_it());

    // No callback ran; the message arrived instead.
    CHECK(callback_count() == 0U);
    REQUIRE(call("OSReceiveMessage", {kQueue, kInbox, 0}) == 1U);

    const u32 result = call("FSGetAsyncResult", {kInbox});
    REQUIRE(result == kBlock + coreinit::fs_async_result_offset);

    // FSAsyncResult: the asyncData it was given, the message it arrived as,
    // and what the request actually did.
    CHECK(memory.read<u32>(result + 0x04) == 0xC0DEU); // userContext
    CHECK(memory.read<u32>(result + 0x08) == kQueue);  // ioMsgQueue
    CHECK(memory.read<u32>(result + 0x0C) == result);  // ioMsg.message
    CHECK(memory.read<u32>(result + 0x1C) == kClient);
    CHECK(memory.read<u32>(result + 0x20) == kBlock);
    CHECK(memory.read<u32>(result + 0x24) == 16U); // bytes read, as FSStatus

    CHECK(memory.read<u8>(kBuffer) != 0U);
    CHECK(os.busy_fs_blocks.empty());
}

TEST_CASE_METHOD(CallbackFixture,
                 "a guest thread parked on a queue is woken by the completion",
                 "[hle][fs][async][queue][idle]") {
    // The whole path, as a title would meet it: start a read, block on the
    // queue, and have the run loop skip the wait, fire the completion, post
    // the message and resume the thread with it. Every piece of this works on
    // its own above; what this is guarding is the seam between them, where a
    // thread that is woken but never handed its message simply stops.
    constexpr ppcaddr_t kAsyncData = 0x10004000;
    constexpr ppcaddr_t kPath = 0x10004100;
    constexpr ppcaddr_t kMode = 0x10004200;
    constexpr ppcaddr_t kHandleOut = 0x10004300;
    constexpr ppcaddr_t kBuffer = 0x10004400;
    constexpr ppcaddr_t kQueue = 0x10004500;
    constexpr ppcaddr_t kRing = 0x10004600;
    constexpr ppcaddr_t kInbox = 0x10004800;
    constexpr u32 kClient = 0x10005000;
    constexpr u32 kBlock = 0x10006000;

    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    REQUIRE(memory.try_write<u32>(kAsyncData, 0));
    REQUIRE(memory.try_write<u32>(kAsyncData + 4, 0xC0DE));
    REQUIRE(memory.try_write<u32>(kAsyncData + 8, kQueue));

    const std::string path = "/vol/content/data.bin";
    REQUIRE(memory.write_bytes(kPath, std::vector<u8>(path.begin(), path.end())));
    REQUIRE(memory.try_write<u8>(kPath + static_cast<u32>(path.size()), 0));
    REQUIRE(memory.try_write<u8>(kMode, 'r'));
    REQUIRE(memory.try_write<u8>(kMode + 1, 0));

    static_cast<void>(call("FSOpenFile", {kClient, kBlock, kPath, kMode, kHandleOut, 0xFFFFFFFF}));
    const u32 handle = memory.read<u32>(kHandleOut);
    REQUIRE(handle != 0U);

    static_cast<void>(
        call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, handle, 0, kAsyncData}));

    // OSReceiveMessage(&queue, &inbox, OS_MESSAGE_QUEUE_FLAG_BLOCK), then
    // record what it returned and what arrived.
    const ppcaddr_t trampoline = registry.resolve("coreinit", "OSReceiveMessage");
    const std::array<u32, 16> program{
        kEnc_mflr_r31,
        lis_r12(trampoline),
        ori_r12(trampoline),
        kEnc_mtctr_r12,
        lis(3, kQueue),
        ori(3, kQueue),
        lis(4, kInbox),
        ori(4, kInbox),
        li(5, coreinit::message_flag_block),
        kEnc_bctrl,
        lis(6, kScratch),
        ori(6, kScratch),
        stw(3, 12, 6), // what OSReceiveMessage returned
        kEnc_mtlr_r31,
        li(3, 0),
        kEnc_blr,
    };
    write_program(kMainProgram, program);

    std::string error;
    REQUIRE(registry.install(memory, error));

    const RunReport report = run_it();
    INFO("outcome: " << run_outcome_name(report.outcome));
    CHECK(report.outcome == RunOutcome::Returned);

    // It came back true, with the message -- and the wait itself was skipped
    // rather than emulated, since nothing else was runnable.
    CHECK(memory.read<u32>(kScratch + 12) == 1U);
    CHECK(memory.read<u32>(kInbox) == kBlock + coreinit::fs_async_result_offset);
    // Nearly all of the latency was skipped -- all of it bar the handful of
    // instructions that ran before the thread parked.
    CHECK(report.skipped_ticks > coreinit::fs_async_latency_ticks - 100);
    CHECK(report.instructions < 100U);
    CHECK(memory.read<u32>(kBlock + coreinit::fs_async_result_offset + 0x24) == 16U);
}

// ---------------------------------------------------------------------------
// Spin detection (P2)
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CallbackFixture,
                 "a spinning thread gives the core back early",
                 "[hle][callback][idle]") {
    // The main thread waits for a worker by polling a word -- which is what a
    // wait looks like when a title writes one by hand. Emulated faithfully it
    // costs a whole timeslice per turn, and the timeslice here is far larger
    // than the budget: without noticing the spin, this run cannot finish.
    constexpr ppcaddr_t kFlag = kScratch + 12;
    constexpr ppcaddr_t kWorkerProgram = kCallbackProgram + 0x100;
    constexpr u32 kWorker = 0x10007000;

    const std::array<u32, 7> main_program{
        0x3C801000, // lis   r4,0x1000
        0x6084200C, // ori   r4,r4,0x200C   -- the flag
        0x80640000, // lwz   r3,0(r4)
        0x2C030000, // cmpwi r3,0
        0x4182FFF8, // beq   -8             -- spin
        0x38600000, // li    r3,0
        kEnc_blr,
    };
    write_program(kMainProgram, main_program);

    const std::array<u32, 5> worker_program{
        0x3C801000, // lis   r4,0x1000
        0x6084200C, // ori   r4,r4,0x200C
        0x38600001, // li    r3,1
        0x90640000, // stw   r3,0(r4)       -- release the waiter
        kEnc_blr,
    };
    write_program(kWorkerProgram, worker_program);
    REQUIRE(memory.try_write<u32>(kFlag, 0));

    static_cast<void>(os.scheduler.create(kWorker,
                                          "worker",
                                          kWorkerProgram,
                                          0,
                                          0,
                                          memory::data_heap_base + 0x30000,
                                          0x1000,
                                          default_priority,
                                          0x7));
    REQUIRE(os.scheduler.resume(kWorker));

    const RunReport report =
        run(interpreter,
            memory,
            registry,
            {.scheduler = &os.scheduler, .budget = 100'000, .timeslice = 1'000'000});

    INFO("outcome: " << run_outcome_name(report.outcome) << " after " << report.instructions
                     << " instructions, " << report.spin_yields << " spin yields");
    CHECK(report.outcome == RunOutcome::Returned);
    CHECK(report.spin_yields >= 1U);

    // A few hundred instructions of spin before the core is handed over, not
    // the million the timeslice would have allowed.
    CHECK(report.instructions < 5000U);
    CHECK(memory.read<u32>(kFlag) == 1U);
}

TEST_CASE_METHOD(CallbackFixture,
                 "a periodic alarm keeps its phase when the clock overshoots",
                 "[hle][alarm]") {
    // The idle skip lands exactly on a deadline, but a busy run does not: the
    // clock arrives late, and re-arming from "now" would let a 60 Hz alarm
    // drift a little further behind every frame.
    constexpr u32 kAlarm = 0x10003000;

    static_cast<void>(call("OSCreateAlarm", {kAlarm}));
    static_cast<void>(call("OSSetPeriodicAlarm", {kAlarm, 0, 0, 1000, 0, 1000, kCallbackProgram}));
    REQUIRE(os.alarms[kAlarm].deadline == 1000U);

    // Three and a half periods late.
    state.timebase = 3500;
    CHECK(os.scheduler.service_timers(state.timebase) == 1U);

    // The next deadline is the next slot on the original grid, not 4500.
    CHECK(os.alarms[kAlarm].deadline == 4000U);
    CHECK(os.alarms[kAlarm].fire_count == 1U);

    // Hardware does not deliver the backlog either, so one firing is right.
    CHECK(os.scheduler.has_timer(kAlarm));
}

TEST_CASE_METHOD(CallbackFixture,
                 "a run with nothing but a timer left is a deadlock",
                 "[hle][callback][idle]") {
    // A periodic alarm with no callback keeps the clock moving forever while
    // nothing can run. Skipping to each deadline in turn runs no instructions,
    // so the budget never advances: without noticing, the run loop spins on
    // the clock and the emulator simply stops responding.
    constexpr u32 kAlarm = 0x10003000;
    static_cast<void>(call("OSCreateAlarm", {kAlarm}));
    static_cast<void>(call("OSSetPeriodicAlarm", {kAlarm, 0, 0, 1000, 0, 1000, 0}));

    // The main thread waits on an event nobody will ever signal.
    constexpr u32 kEvent = 0x10003200;
    static_cast<void>(call("OSInitEvent", {kEvent, 0, 1}));

    write_main_calling("OSWaitEvent", kEvent);
    const RunReport report = run_it();

    INFO("outcome: " << run_outcome_name(report.outcome) << " after " << report.instructions
                     << " instructions");
    CHECK(report.outcome == RunOutcome::Deadlocked);
}

TEST_CASE_METHOD(CallbackFixture,
                 "one command block carries one request at a time",
                 "[hle][fs][async]") {
    // A second request through a block that already has one in flight would
    // deliver two completions through the same structure. Hardware refuses
    // it, and so does this.
    constexpr ppcaddr_t kAsyncData = 0x10004000;
    constexpr ppcaddr_t kBuffer = 0x10004400;
    constexpr u32 kClient = 0x10005000;
    constexpr u32 kBlock = 0x10006000;

    REQUIRE(memory.try_write<u32>(kAsyncData, kCallbackProgram));
    REQUIRE(memory.try_write<u32>(kAsyncData + 4, 0));
    REQUIRE(memory.try_write<u32>(kAsyncData + 8, 0));

    CHECK(call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, 0, 0, kAsyncData}) == 0U);
    CHECK(call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, 0, 0, kAsyncData}) ==
          static_cast<u32>(FsStatus::FatalError));
    CHECK(os.scheduler.timer_count() == 1U);

    // Once the completion has been delivered the block is free again.
    write_main_sleeping(8000);
    static_cast<void>(run_it());
    CHECK(os.busy_fs_blocks.empty());
    CHECK(call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, 0, 0, kAsyncData}) == 0U);
}

TEST_CASE_METHOD(CallbackFixture,
                 "a request with both a callback and a queue uses the callback",
                 "[hle][fs][async][queue]") {
    // FSAsyncData carries both fields and a title fills in one of them. If it
    // fills in both, hardware calls back: the queue coreinit posts to on the
    // way there is its own, not the title's. Delivering to both would run the
    // callback *and* leave a message the title never asked for sitting in its
    // queue, which it would eventually receive as something else entirely.
    constexpr ppcaddr_t kAsyncData = 0x10004000;
    constexpr ppcaddr_t kPath = 0x10004100;
    constexpr ppcaddr_t kMode = 0x10004200;
    constexpr ppcaddr_t kHandleOut = 0x10004300;
    constexpr ppcaddr_t kBuffer = 0x10004400;
    constexpr ppcaddr_t kQueue = 0x10004500;
    constexpr ppcaddr_t kRing = 0x10004600;
    constexpr ppcaddr_t kInbox = 0x10004800;
    constexpr u32 kClient = 0x10005000;
    constexpr u32 kBlock = 0x10006000;

    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    REQUIRE(memory.try_write<u32>(kAsyncData, kCallbackProgram));
    REQUIRE(memory.try_write<u32>(kAsyncData + 4, 0xC0DE));
    REQUIRE(memory.try_write<u32>(kAsyncData + 8, kQueue));

    const std::string path = "/vol/content/data.bin";
    REQUIRE(memory.write_bytes(kPath, std::vector<u8>(path.begin(), path.end())));
    REQUIRE(memory.try_write<u8>(kPath + static_cast<u32>(path.size()), 0));
    REQUIRE(memory.try_write<u8>(kMode, 'r'));
    REQUIRE(memory.try_write<u8>(kMode + 1, 0));

    static_cast<void>(call("FSOpenFile", {kClient, kBlock, kPath, kMode, kHandleOut, 0xFFFFFFFF}));
    const u32 handle = memory.read<u32>(kHandleOut);
    REQUIRE(handle != 0U);

    static_cast<void>(
        call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, handle, 0, kAsyncData}));

    write_main_sleeping(8000);
    static_cast<void>(run_it());

    CHECK(callback_count() == 1U);
    CHECK(call("OSReceiveMessage", {kQueue, kInbox, 0}) == 0U); // and nothing was posted
}

TEST_CASE_METHOD(CallbackFixture,
                 "a request with nowhere to report is not left in flight",
                 "[hle][fs][async][queue]") {
    constexpr ppcaddr_t kAsyncData = 0x10004000;
    constexpr ppcaddr_t kPath = 0x10004100;
    constexpr ppcaddr_t kMode = 0x10004200;
    constexpr ppcaddr_t kHandleOut = 0x10004300;
    constexpr ppcaddr_t kBuffer = 0x10004400;
    constexpr ppcaddr_t kQueue = 0x10004500;
    constexpr ppcaddr_t kRing = 0x10004600;
    constexpr u32 kClient = 0x10005000;
    constexpr u32 kBlock = 0x10006000;

    static_cast<void>(call("OSInitMessageQueue", {kQueue, kRing, 4}));

    const std::string path = "/vol/content/data.bin";
    REQUIRE(memory.write_bytes(kPath, std::vector<u8>(path.begin(), path.end())));
    REQUIRE(memory.try_write<u8>(kPath + static_cast<u32>(path.size()), 0));
    REQUIRE(memory.try_write<u8>(kMode, 'r'));
    REQUIRE(memory.try_write<u8>(kMode + 1, 0));

    static_cast<void>(call("FSOpenFile", {kClient, kBlock, kPath, kMode, kHandleOut, 0xFFFFFFFF}));
    const u32 handle = memory.read<u32>(kHandleOut);
    REQUIRE(handle != 0U);

    SECTION("neither a callback nor a queue") {
        REQUIRE(memory.fill(kAsyncData, 12, 0));
        static_cast<void>(
            call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, handle, 0, kAsyncData}));
    }

    SECTION("a queue, but no command block to put the result in") {
        REQUIRE(memory.try_write<u32>(kAsyncData, 0));
        REQUIRE(memory.try_write<u32>(kAsyncData + 4, 0));
        REQUIRE(memory.try_write<u32>(kAsyncData + 8, kQueue));
        static_cast<void>(
            call("FSReadFileAsync", {kClient, 0, kBuffer, 1, 16, handle, 0, kAsyncData}));
    }

    // Nothing to deliver to means nothing to wait for: no deadline is armed,
    // and the command block is free again straight away rather than a
    // completion later that is never going to arrive.
    CHECK(os.scheduler.timer_count() == 0U);
    CHECK(os.busy_fs_blocks.empty());
}
