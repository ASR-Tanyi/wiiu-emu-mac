// Phase 6.11 — proc_ui.
//
// A Wii U title's main loop is `while (ProcUIProcessMessages(true) !=
// EXITING) draw();`, so everything here is about that loop: that it turns,
// that it can be stopped, and that the callbacks a title hangs off the
// foreground transitions actually run -- as guest code, on the calling thread,
// finished before the call returns. That last part is the one worth driving
// through the real interpreter rather than checking a structure was filled in,
// because a callback that is registered beautifully and never runs is how a
// title loses its GPU resources and draws nothing.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/hle/proc_ui.hpp>
#include <wiiu/hle/system_libraries.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;

namespace {

constexpr ppcaddr_t kMainProgram = 0x02000000;

/// Each callback program bumps its own counter and appends its identity to a
/// log, so a test can check not just that they ran but in what order.
constexpr ppcaddr_t kOrderLog = 0x10002000;  ///< count, then one word per call
constexpr ppcaddr_t kScratch = 0x10002100;   ///< what the main program recorded
constexpr ppcaddr_t kCallbacks = 0x02001000; ///< the callback programs

constexpr u32 kEnc_mflr_r31 = 0x7FE802A6;
constexpr u32 kEnc_mtctr_r12 = 0x7D8903A6;
constexpr u32 kEnc_bctrl = 0x4E800421;
constexpr u32 kEnc_mtlr_r31 = 0x7FE803A6;
constexpr u32 kEnc_blr = 0x4E800020;

[[nodiscard]] constexpr u32 lis(u32 reg, u32 value) {
    return 0x3C000000U | (reg << 21) | (value >> 16);
}

[[nodiscard]] constexpr u32 ori(u32 reg, u32 value) {
    return 0x60000000U | (reg << 21) | (reg << 16) | (value & 0xFFFF);
}

[[nodiscard]] constexpr u32 li(u32 reg, u32 value) {
    return 0x38000000U | (reg << 21) | (value & 0xFFFF);
}

[[nodiscard]] constexpr u32 lwz(u32 dest, u32 offset, u32 base) {
    return 0x80000000U | (dest << 21) | (base << 16) | (offset & 0xFFFF);
}

[[nodiscard]] constexpr u32 stw(u32 src, u32 offset, u32 base) {
    return 0x90000000U | (src << 21) | (base << 16) | (offset & 0xFFFF);
}

[[nodiscard]] constexpr u32 stwx(u32 src, u32 base, u32 index) {
    return 0x7C00012EU | (src << 21) | (base << 16) | (index << 11);
}

[[nodiscard]] constexpr u32 addi(u32 dest, u32 source, u32 value) {
    return 0x38000000U | (dest << 21) | (source << 16) | (value & 0xFFFF);
}

[[nodiscard]] constexpr u32 slwi_2(u32 dest, u32 source) {
    // rlwinm rDest,rSource,2,0,29 -- multiply by four for a word index.
    return 0x5400103AU | (source << 21) | (dest << 16);
}

struct ProcUiFixture {
    memory::Memory memory;
    cpu::CoreState state;
    cpu::Interpreter interpreter{state, memory};
    Registry registry;
    coreinit::Context os;
    procui::State procui;

    ProcUiFixture() {
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
        procui::register_all(registry, os, procui);
        register_system_libraries(registry);

        REQUIRE(os.heaps.create_default_heaps(memory, error));
        os.memory = &memory;

        REQUIRE(memory.fill(kOrderLog, 64, 0));
        REQUIRE(memory.fill(kScratch, 64, 0));

        state.pc = kMainProgram;
        state.gpr[1] = memory::data_heap_base + 0x20000;
        state.lr = return_sentinel;
        static_cast<void>(os.scheduler.adopt_main(state, main_thread_handle));
    }

    /// Call a registered function directly, the way a test drives one that
    /// does not need the interpreter.
    u32 call(std::string_view library, std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find(library, name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size() && i < 8; ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    u32 process_messages(bool block = false) {
        return call("proc_ui", "ProcUIProcessMessages", {block ? 1U : 0U});
    }

    template <typename Container>
    void write_program(ppcaddr_t at, const Container& words) {
        ppcaddr_t address = at;
        for (const u32 word : words) {
            REQUIRE(memory.try_write<u32>(address, word));
            address += 4;
        }
    }

    /// A guest callback that appends `id` to the order log and returns. Each
    /// one is written at `kCallbacks + index * 0x40`.
    [[nodiscard]] ppcaddr_t write_callback(u32 index, u32 id) {
        const ppcaddr_t entry = kCallbacks + (index * 0x40);
        const std::array<u32, 12> program{
            lis(5, kOrderLog),
            ori(5, kOrderLog),
            lwz(6, 0, 5),    // the count so far
            addi(7, 6, 1),   //
            stw(7, 0, 5),    // ... incremented
            slwi_2(8, 7),    // (count + 1) * 4 -- entries start one word in
            li(9, id),       //
            stwx(9, 5, 8),   // log[count + 1] = id
            stw(3, 0x30, 5), // and the parameter it was given
            li(3, 0),        //
            kEnc_blr,        //
            kEnc_blr,        // padding
        };
        write_program(entry, program);
        return entry;
    }

    /// Everything the callbacks recorded, in the order they ran.
    [[nodiscard]] std::vector<u32> order() {
        const u32 count = memory.read<u32>(kOrderLog);
        std::vector<u32> ids;
        for (u32 index = 0; index < count; ++index) {
            ids.push_back(memory.read<u32>(kOrderLog + 4 + (index * 4)));
        }
        return ids;
    }

    [[nodiscard]] u32 last_parameter() { return memory.read<u32>(kOrderLog + 0x30); }

    /// A main program that calls ProcUIProcessMessages(block) once and stores
    /// what it returned at kScratch, then returns.
    void write_main_processing(bool block) {
        const ppcaddr_t trampoline = registry.resolve("proc_ui", "ProcUIProcessMessages");
        const std::array<u32, 12> program{
            kEnc_mflr_r31,
            lis(12, trampoline),
            ori(12, trampoline),
            kEnc_mtctr_r12,
            li(3, block ? 1U : 0U),
            kEnc_bctrl,
            lis(6, kScratch),
            ori(6, kScratch),
            stw(3, 0, 6),
            kEnc_mtlr_r31,
            li(3, 0),
            kEnc_blr,
        };
        write_program(kMainProgram, program);

        std::string error;
        REQUIRE(registry.install(memory, error));
    }

    RunReport run_it(u64 budget = 200'000) {
        return run(interpreter,
                   memory,
                   registry,
                   {.scheduler = &os.scheduler, .budget = budget, .timeslice = 200});
    }
};

} // namespace

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(ProcUiFixture, "a title runs in the foreground until asked", "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));

    CHECK(call("proc_ui", "ProcUIIsRunning", {}) == 1U);
    CHECK(call("proc_ui", "ProcUIInForeground", {}) == 1U);

    // Nothing in the emulator wants the screen, so the loop just turns.
    for (int i = 0; i < 4; ++i) {
        CHECK(process_messages() == static_cast<u32>(procui::Status::InForeground));
    }
    CHECK(procui.poll_count == 4U);
}

TEST_CASE_METHOD(ProcUiFixture, "an exit request stops the loop", "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InForeground));

    REQUIRE(procui.request_exit(os, memory));
    CHECK(process_messages() == static_cast<u32>(procui::Status::Exiting));

    // And it stays stopped. A title's loop tests for Exiting; being told it is
    // back in the foreground half way out would put it back in the loop.
    CHECK(process_messages() == static_cast<u32>(procui::Status::Exiting));
    CHECK(process_messages() == static_cast<u32>(procui::Status::Exiting));
}

TEST_CASE_METHOD(ProcUiFixture, "ProcUIProcessMessages before init exits", "[hle][procui]") {
    // A title that never initialised is not going to get a coherent answer out
    // of this, and Exiting is the only answer that cannot leave it looping for
    // good waiting for something that will not come.
    CHECK(process_messages() == static_cast<u32>(procui::Status::Exiting));
    CHECK(call("proc_ui", "ProcUIIsRunning", {}) == 0U);
}

// ---------------------------------------------------------------------------
// Foreground handover
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(ProcUiFixture, "releasing the foreground takes two steps", "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InForeground));

    REQUIRE(procui.request_release(os, memory));

    // First the title is told to finish what it is drawing. It is still in the
    // foreground here -- it has a frame left to put something on screen.
    CHECK(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    CHECK(call("proc_ui", "ProcUIInForeground", {}) == 1U);

    // Until it says it is done, it keeps being told.
    CHECK(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));

    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));
    CHECK(process_messages() == static_cast<u32>(procui::Status::InBackground));
    CHECK(call("proc_ui", "ProcUIInForeground", {}) == 0U);

    // And back again.
    REQUIRE(procui.request_acquire(os, memory));
    CHECK(process_messages() == static_cast<u32>(procui::Status::InForeground));
    CHECK(call("proc_ui", "ProcUIInForeground", {}) == 1U);
}

TEST_CASE_METHOD(ProcUiFixture, "an exit request overtakes a handover", "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    REQUIRE(procui.request_release(os, memory));
    REQUIRE(procui.request_exit(os, memory));

    // Both are waiting. What is being asked for is that the title stops, so
    // there is no point walking it through a handover first.
    CHECK(process_messages() == static_cast<u32>(procui::Status::Exiting));
}

TEST_CASE_METHOD(ProcUiFixture,
                 "a backgrounded title waits rather than spinning",
                 "[hle][procui][idle]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InBackground));

    // With `block` set there is nothing to do until the system says otherwise,
    // so the thread parks on the system queue instead of coming straight back
    // round the title's loop.
    static_cast<void>(process_messages(true));
    const Thread* const main = os.scheduler.find(main_thread_handle);
    REQUIRE(main != nullptr);
    CHECK(main->state == ThreadState::Blocked);
    CHECK(main->block_reason == BlockReason::MessageQueue);

    // The request that wakes it is also the one it acts on: the message was
    // delivered while it was parked, so the next call has to pick it up from
    // there rather than from the queue it is no longer in.
    REQUIRE(procui.request_acquire(os, memory));
    CHECK(main->state == ThreadState::Ready);
    CHECK(process_messages() == static_cast<u32>(procui::Status::InForeground));
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(ProcUiFixture, "a release callback runs as guest code", "[hle][procui]") {
    // The whole point of the release callback is that the title has freed its
    // GPU resources by the time ProcUIProcessMessages returns. So this runs
    // the real thing: the interpreter executes the callback, on the calling
    // thread, and the call returns the status afterwards.
    const ppcaddr_t callback = write_callback(0, 0xA1);

    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    static_cast<void>(
        call("proc_ui",
             "ProcUIRegisterCallback",
             {static_cast<u32>(procui::CallbackType::Release), callback, 0xBEEF, 100}));

    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));

    write_main_processing(false);
    const RunReport report = run_it();

    INFO("outcome: " << run_outcome_name(report.outcome));
    CHECK(report.outcome == RunOutcome::Returned);

    // It ran, with the parameter it was registered with...
    CHECK(order() == std::vector<u32>{0xA1});
    CHECK(last_parameter() == 0xBEEFU);

    // ... and the title still got its status back, from the call it made.
    CHECK(memory.read<u32>(kScratch) == static_cast<u32>(procui::Status::InBackground));
    CHECK(call("proc_ui", "ProcUIInForeground", {}) == 0U);
}

TEST_CASE_METHOD(ProcUiFixture, "callbacks run in priority order", "[hle][procui]") {
    // Highest priority first, and within a priority the order they were
    // registered. A title that releases a texture before the heap it came from
    // depends on getting this the right way round.
    const ppcaddr_t low = write_callback(0, 0x10);
    const ppcaddr_t high = write_callback(1, 0x20);
    const ppcaddr_t first_equal = write_callback(2, 0x30);
    const ppcaddr_t second_equal = write_callback(3, 0x40);

    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    const u32 release = static_cast<u32>(procui::CallbackType::Release);
    static_cast<void>(call("proc_ui", "ProcUIRegisterCallback", {release, low, 0, 1}));
    static_cast<void>(call("proc_ui", "ProcUIRegisterCallback", {release, high, 0, 200}));
    static_cast<void>(call("proc_ui", "ProcUIRegisterCallback", {release, first_equal, 0, 50}));
    static_cast<void>(call("proc_ui", "ProcUIRegisterCallback", {release, second_equal, 0, 50}));

    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));

    write_main_processing(false);
    const RunReport report = run_it();
    INFO("outcome: " << run_outcome_name(report.outcome));

    CHECK(order() == std::vector<u32>{0x20, 0x30, 0x40, 0x10});
    CHECK(memory.read<u32>(kScratch) == static_cast<u32>(procui::Status::InBackground));
}

TEST_CASE_METHOD(ProcUiFixture, "the save callback runs before exiting", "[hle][procui]") {
    // The title's last chance to write anything out, and it runs before the
    // exit callbacks take apart what it might need.
    const ppcaddr_t save = write_callback(0, 0x51);
    const ppcaddr_t on_exit = write_callback(1, 0x52);

    static_cast<void>(call("proc_ui", "ProcUIInit", {save}));
    static_cast<void>(call("proc_ui",
                           "ProcUIRegisterCallback",
                           {static_cast<u32>(procui::CallbackType::Exit), on_exit, 0, 0}));

    REQUIRE(procui.request_exit(os, memory));

    write_main_processing(false);
    const RunReport report = run_it();
    INFO("outcome: " << run_outcome_name(report.outcome));

    CHECK(order() == std::vector<u32>{0x51, 0x52});
    CHECK(memory.read<u32>(kScratch) == static_cast<u32>(procui::Status::Exiting));
}

TEST_CASE_METHOD(ProcUiFixture, "callbacks only run on their own transition", "[hle][procui]") {
    const ppcaddr_t acquire = write_callback(0, 0x61);

    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    static_cast<void>(call("proc_ui",
                           "ProcUIRegisterCallback",
                           {static_cast<u32>(procui::CallbackType::Acquire), acquire, 0, 0}));

    // Staying in the foreground is not a transition, so nothing runs.
    write_main_processing(false);
    static_cast<void>(run_it());
    CHECK(order().empty());
    CHECK(memory.read<u32>(kScratch) == static_cast<u32>(procui::Status::InForeground));
}

TEST_CASE_METHOD(ProcUiFixture, "cleared callbacks do not run", "[hle][procui]") {
    const ppcaddr_t callback = write_callback(0, 0x71);

    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    static_cast<void>(call("proc_ui",
                           "ProcUIRegisterCallback",
                           {static_cast<u32>(procui::CallbackType::Release), callback, 0, 0}));
    static_cast<void>(call("proc_ui", "ProcUIClearCallbacks", {}));

    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));

    write_main_processing(false);
    static_cast<void>(run_it());

    CHECK(order().empty());
    CHECK(memory.read<u32>(kScratch) == static_cast<u32>(procui::Status::InBackground));
}

// ---------------------------------------------------------------------------
// The system libraries
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(ProcUiFixture, "the start-up stubs report success", "[hle][procui]") {
    // Zero is success for all of these. What matters is that a title calling
    // them at start-up gets past them, and that the call is named in the log
    // rather than showing up as an unresolved import.
    CHECK(call("nsysnet", "socket_lib_init", {}) == 0U);
    CHECK(call("nn_save", "SAVEInit", {}) == 0U);
    CHECK(call("nn_acp", "ACPInitialize", {}) == 0U);

    const Function* const function = registry.find("nn_save", "SAVEInit");
    REQUIRE(function != nullptr);
    CHECK(function->is_stub);
}

TEST_CASE_METHOD(ProcUiFixture, "sockets open and send into no network", "[hle][procui]") {
    // What WHB's UDP log does, which most homebrew starts first: a socket, an
    // option, the broadcast address in network order -- big-endian, as the
    // guest already is -- and sends, every one of them all sent.
    const u32 socket = call("nsysnet", "socket", {2, 2, 17});
    CHECK(static_cast<s32>(socket) > 0);
    CHECK(call("nsysnet", "setsockopt", {socket, 0xFFFF, 0x20, 0, 4}) == 0U);
    CHECK(call("nsysnet", "htons", {4405}) == 4405U);
    CHECK(call("nsysnet", "htonl", {0xFFFFFFFF}) == 0xFFFFFFFFU);

    constexpr ppcaddr_t text = memory::data_heap_base + 0x7000;
    constexpr ppcaddr_t address = memory::data_heap_base + 0x7100;
    const std::string line = "Begin rendering...\n";
    REQUIRE(memory.write_bytes(text, std::vector<u8>(line.begin(), line.end())));
    REQUIRE(memory.try_write<u16>(address, 2));        // AF_INET
    REQUIRE(memory.try_write<u16>(address + 2, 4405)); // WHB's log port
    REQUIRE(memory.try_write<u32>(address + 4, 0xFFFFFFFF));
    CHECK(call("nsysnet",
               "sendto",
               {socket, text, static_cast<u32>(line.size()), 0, address, 16}) == line.size());

    CHECK(call("nsysnet", "shutdown", {socket, 1}) == 0U);
    CHECK(call("nsysnet", "socketclose", {socket}) == 0U);
    // Closed is closed.
    CHECK(static_cast<s32>(call("nsysnet", "sendto", {socket, text, 4, 0, address, 16})) == -1);

    // nn_ac takes the request to connect; nothing connects.
    CHECK(call("nn_ac", "ACInitialize", {}) == 0U);
    CHECK(call("nn_ac", "ACConnectAsync", {}) == 0U);
}

// ---------------------------------------------------------------------------
// The whole loop, in guest code
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(ProcUiFixture, "a title's main loop turns and then stops", "[hle][procui]") {
    // The shape every Wii U title has:
    //
    //     while (ProcUIProcessMessages(true) != PROCUI_STATUS_EXITING) { ... }
    //
    // written out in PowerPC and run through the interpreter. The pieces are
    // covered above one at a time; what this is for is the loop itself, and
    // what it caught was not in ProcUI at all -- see the note on the timer
    // below.
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));

    const ppcaddr_t trampoline = registry.resolve("proc_ui", "ProcUIProcessMessages");
    const std::array<u32, 17> program{
        kEnc_mflr_r31,
        // loop:
        lis(12, trampoline),
        ori(12, trampoline),
        kEnc_mtctr_r12,
        li(3, 1), // block
        kEnc_bctrl,
        lis(6, kScratch),
        ori(6, kScratch),
        stw(3, 0, 6),  // the status it last saw
        lwz(7, 4, 6),  //
        addi(7, 7, 1), //
        stw(7, 4, 6),  // ... and how many times round it has been
        0x2C030003,    // cmpwi r3,3   -- PROCUI_STATUS_EXITING
        0x4082FFD0,    // bne   loop   -- back to the lis, 48 bytes up
        kEnc_mtlr_r31,
        li(3, 0),
        kEnc_blr,
    };
    write_program(kMainProgram, program);

    std::string error;
    REQUIRE(registry.install(memory, error));

    // Something asks it to stop part way through, the way a HOME button or a
    // close request will. Nothing else would ever end this loop.
    //
    // This is also what makes the test worth having. A deadline only comes due
    // when the run loop reschedules, and the run loop used to start the
    // timeslice again after *every* HLE call -- so a title like this one,
    // which calls into the OS several times a frame, never reached the end of
    // a timeslice, never rescheduled, and never saw an alarm or a deadline at
    // all. It just kept going.
    os.scheduler.set_timer(0x99, state.timebase + 20'000, [&](u64) {
        static_cast<void>(procui.request_exit(os, memory));
    });

    const RunReport report = run_it();
    INFO("outcome: " << run_outcome_name(report.outcome));
    CHECK(report.outcome == RunOutcome::Returned);

    CHECK(memory.read<u32>(kScratch) == static_cast<u32>(procui::Status::Exiting));
    CHECK(memory.read<u32>(kScratch + 4) > 1U); // it went round more than once
    CHECK(procui.poll_count == memory.read<u32>(kScratch + 4));
    CHECK(report.instructions < 100'000U); // and it stopped when it was asked to
}

// ---------------------------------------------------------------------------
// The edges an adversarial read found
// ---------------------------------------------------------------------------

namespace {

/// A second guest thread, so that parking the main one switches to something
/// rather than stalling the core.
constexpr u32 kOtherThread = 0x10003000;

} // namespace

TEST_CASE_METHOD(ProcUiFixture,
                 "a parked thread's message survives another thread's poll",
                 "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InBackground));

    REQUIRE(os.scheduler.create(kOtherThread,
                                "other",
                                kMainProgram,
                                0,
                                0,
                                memory::data_heap_base + 0x30000,
                                0x1000,
                                16,
                                0x7) != 0);
    REQUIRE(os.scheduler.resume(kOtherThread));

    static_cast<void>(process_messages(true));
    const Thread* const main = os.scheduler.find(main_thread_handle);
    REQUIRE(main != nullptr);
    REQUIRE(main->state == ThreadState::Blocked);
    REQUIRE(os.scheduler.current_handle() == kOtherThread);

    // Another thread polls while the first is parked. What the parked thread
    // is waiting for is its own: a single shared slot meant this poll consumed
    // the flag, applied an empty message, and left the real one -- handed
    // straight into that slot rather than left in the queue -- with nobody who
    // would ever look at it. The title stayed in the background for good.
    static_cast<void>(process_messages(false));

    REQUIRE(procui.request_exit(os, memory));
    REQUIRE(main->state == ThreadState::Ready);
    CHECK(process_messages() == static_cast<u32>(procui::Status::Exiting));
}

TEST_CASE_METHOD(ProcUiFixture, "two parked threads each get their own message", "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InBackground));

    REQUIRE(os.scheduler.create(kOtherThread,
                                "other",
                                kMainProgram,
                                0,
                                0,
                                memory::data_heap_base + 0x30000,
                                0x1000,
                                16,
                                0x7) != 0);
    REQUIRE(os.scheduler.resume(kOtherThread));

    static_cast<void>(process_messages(true)); // main parks
    REQUIRE(os.scheduler.current_handle() == kOtherThread);
    static_cast<void>(process_messages(true)); // and so does the other
    REQUIRE(os.scheduler.find(kOtherThread)->state == ThreadState::Blocked);

    // Two messages, two parked threads, one each. Sharing a slot meant the
    // second delivery landed on top of the first before anyone read it, and
    // the exit -- which is the one that must not be lost -- was the one that
    // went.
    REQUIRE(procui.request_exit(os, memory));
    REQUIRE(procui.request_acquire(os, memory));
    CHECK(procui.inboxes.size() == 2U);

    CHECK(process_messages() == static_cast<u32>(procui::Status::Exiting));
}

TEST_CASE_METHOD(ProcUiFixture,
                 "ProcUIDrawDoneRelease outside a handover does not latch",
                 "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InForeground));

    // A title that calls this at the end of every frame. Latching it here
    // meant the next handover skipped ReleaseForeground: the title was never
    // told to finish its frame, and its release callbacks ran underneath it
    // while it was still drawing.
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));

    REQUIRE(procui.request_release(os, memory));
    CHECK(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
}

TEST_CASE_METHOD(ProcUiFixture,
                 "a release callback may call ProcUIDrawDoneRelease",
                 "[hle][procui]") {
    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));

    // One full cycle, exactly as a title does it.
    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InBackground));

    // A release callback tearing down GX2 calls it on the way out, which is
    // ordinary. It must not arm the *next* handover.
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));

    REQUIRE(procui.request_acquire(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::InForeground));

    REQUIRE(procui.request_release(os, memory));
    CHECK(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
}

TEST_CASE_METHOD(ProcUiFixture,
                 "a callback that polls again does not strand the outer chain",
                 "[hle][procui]") {
    // Calling ProcUIProcessMessages from inside a ProcUI callback is not
    // something a title should do, but a poll buried in a shared shutdown
    // helper is an easy way to end up doing it. Starting a second chain on the
    // same thread overwrote the first one's saved return address: the outer
    // call never got back to the main loop, and the callback's own return then
    // landed on a trampoline with no chain behind it and halted the core.
    const ppcaddr_t release_cb = write_callback(0, 0xC1);
    const ppcaddr_t save_cb = write_callback(1, 0xC2);

    static_cast<void>(call("proc_ui", "ProcUIInit", {save_cb}));
    static_cast<void>(call("proc_ui",
                           "ProcUIRegisterCallback",
                           {static_cast<u32>(procui::CallbackType::Release), release_cb, 0, 0}));

    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));

    // The main loop polls; the release chain starts on this thread.
    constexpr ppcaddr_t kMainLoopResume = 0x02008000;
    state.lr = kMainLoopResume;
    state.pc = kMainLoopResume; // what Registry::dispatch does before a handler
    static_cast<void>(process_messages());
    REQUIRE(procui.chains.count(main_thread_handle) == 1U);
    REQUIRE(state.pc == release_cb);
    REQUIRE(procui.chains.at(main_thread_handle).resume_pc == kMainLoopResume);

    // Inside that callback the title polls again, and by now it has been asked
    // to stop.
    REQUIRE(procui.request_exit(os, memory));
    state.lr = release_cb + 0x20;
    state.pc = release_cb + 0x20;
    static_cast<void>(process_messages());

    CHECK(procui.chains.count(main_thread_handle) == 1U);
    CHECK(procui.chains.at(main_thread_handle).resume_pc == kMainLoopResume);

    // And when the callback does return, it goes where the main loop was
    // going, with the status that call was going to report.
    static_cast<void>(call("proc_ui", "__procui_callback_returned", {}));
    CHECK_FALSE(state.halted);
    CHECK(state.pc == kMainLoopResume);
    CHECK(procui.chains.empty());
}

TEST_CASE_METHOD(ProcUiFixture,
                 "a thread that dies mid-chain leaves nothing behind",
                 "[hle][procui]") {
    const ppcaddr_t release_cb = write_callback(0, 0xD1);

    static_cast<void>(call("proc_ui", "ProcUIInit", {0}));
    static_cast<void>(call("proc_ui",
                           "ProcUIRegisterCallback",
                           {static_cast<u32>(procui::CallbackType::Release), release_cb, 0, 0}));

    REQUIRE(os.scheduler.create(kOtherThread,
                                "other",
                                kMainProgram,
                                0,
                                0,
                                memory::data_heap_base + 0x30000,
                                0x1000,
                                16,
                                0x7) != 0);
    REQUIRE(os.scheduler.resume(kOtherThread));

    REQUIRE(procui.request_release(os, memory));
    REQUIRE(process_messages() == static_cast<u32>(procui::Status::ReleaseForeground));
    static_cast<void>(call("proc_ui", "ProcUIDrawDoneRelease", {}));

    state.pc = 0x02008000;
    state.lr = 0x02008000;
    static_cast<void>(process_messages());
    REQUIRE(procui.chains.count(main_thread_handle) == 1U);

    // The callback calls OSExitThread instead of returning, so it never
    // reaches the trampoline and its chain is never finished.
    os.scheduler.exit_current(state, 0);
    REQUIRE(os.scheduler.find(main_thread_handle)->state == ThreadState::Exited);

    // The title frees that OSThread and allocates another at the same address,
    // which is what a thread pool does. The new thread must not inherit a
    // chain it has never heard of -- it looked like it was in the middle of a
    // callback, so it never parked and spun its main loop instead.
    REQUIRE(os.scheduler.remove(main_thread_handle));
    REQUIRE(os.scheduler.create(main_thread_handle,
                                "reborn",
                                kMainProgram,
                                0,
                                0,
                                memory::data_heap_base + 0x20000,
                                0x1000,
                                16,
                                0x7) != 0);
    REQUIRE(os.scheduler.resume(main_thread_handle));
    for (int i = 0; i < 4 && os.scheduler.current_handle() != main_thread_handle; ++i) {
        static_cast<void>(os.scheduler.reschedule(state));
    }
    REQUIRE(os.scheduler.current_handle() == main_thread_handle);

    // (The return value of a parking call belongs to whichever thread runs
    // next, so what is checked is that it parked.)
    static_cast<void>(process_messages(true));
    CHECK(os.scheduler.find(main_thread_handle)->state == ThreadState::Blocked);
    CHECK(procui.chains.empty());
}
