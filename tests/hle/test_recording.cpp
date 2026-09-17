// Phase 10.4 — record and replay.
//
// The claim these tests defend is narrow and load-bearing: two runs of the
// same module, with the same recorded host values, produce the same machine
// at every point — and when they do not, the divergence is reported at the
// first place it could possibly be noticed rather than wherever it happens to
// surface.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/hle/recording.hpp>
#include <wiiu/hle/scheduler.hpp>
#include <wiiu/loader/rpl.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;

namespace {

[[nodiscard]] std::filesystem::path data_path(std::string_view name) {
    return std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / name;
}

/// A machine that can be stood up twice from the same starting point, which
/// is the whole point of the exercise.
struct Machine {
    memory::Memory memory;
    cpu::CoreState state;
    cpu::Interpreter interpreter{state, memory};
    Registry registry;
    coreinit::Context os;
    std::vector<std::string> output;

    Machine() {
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
        const bool ok = memory.initialize(regions, error);
        INFO("initialize: " << error);
        REQUIRE(ok);

        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        os.report_sink = [this](std::string_view text) { output.emplace_back(text); };
        coreinit::register_all(registry, os);

        std::string heap_error;
        REQUIRE(os.heaps.create_default_heaps(memory, heap_error));
    }

    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;

    void load(std::string_view file) {
        loader::Rpl rpl;
        const loader::ParseResult parsed = parse_file(data_path(file).string(), rpl);
        INFO("parse: " << parsed.error);
        REQUIRE(parsed.ok);

        LoadedModule module;
        const LoadResult result = load_module(rpl, memory, registry, state, module);
        INFO("load: " << result.error);
        REQUIRE(result.ok);

        static_cast<void>(os.scheduler.adopt_main(state, main_thread_handle));
        os.scheduler.set_exit_callback([this](s32 value, ppcaddr_t out) {
            static_cast<void>(memory.try_write<u32>(out, static_cast<u32>(value)));
        });
    }

    RunReport run(Recording& recording, u64 checkpoint_interval = 1000) {
        os.recording = &recording;
        if (recording.mode() == Recording::Mode::Off) {
            recording.start_record({.checkpoint_interval = checkpoint_interval});
        }
        return hle::run(
            interpreter, memory, registry, {.scheduler = &os.scheduler, .recording = &recording});
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

TEST_CASE("the state hash covers every architectural register", "[hle][recording]") {
    cpu::CoreState state;
    const u64 base = Recording::hash_state(state);

    // Each of these must move the hash. A register the hash ignores is a
    // register a divergence can hide in.
    auto changes = [&](auto&& mutate) {
        cpu::CoreState copy;
        mutate(copy);
        return Recording::hash_state(copy) != base;
    };

    CHECK(changes([](cpu::CoreState& s) { s.gpr[31] = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.fpr[7].ps1 = 1.0; }));
    CHECK(changes([](cpu::CoreState& s) { s.gqr[3].value = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.pc = 4; }));
    CHECK(changes([](cpu::CoreState& s) { s.lr = 4; }));
    CHECK(changes([](cpu::CoreState& s) { s.ctr = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.cr.value = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.xer.value = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.fpscr = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.timebase = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.decrementer = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.halted = true; }));

    // Supervisor and Espresso-specific state. None of these is read by an
    // ordinary instruction, which is exactly why leaving them out would be
    // easy -- and why a divergence could then hide in them indefinitely.
    CHECK(changes([](cpu::CoreState& s) { s.msr = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.srr0 = 4; }));
    CHECK(changes([](cpu::CoreState& s) { s.srr1 = 4; }));
    CHECK(changes([](cpu::CoreState& s) { s.sprg[2] = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.dar = 4; }));
    CHECK(changes([](cpu::CoreState& s) { s.dsisr = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.hid0 = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.hid1 = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.hid2 = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.scr = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.wpar = 1; }));

    // Translation state. Two runs that disagree about a segment register or a
    // BAT are reading different memory through the same addresses.
    CHECK(changes([](cpu::CoreState& s) { s.sr[9] = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.ibat_u[2] = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.ibat_l[2] = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.dbat_u[7] = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.dbat_l[7] = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.sdr1 = 1; }));
    CHECK(changes([](cpu::CoreState& s) { s.hid4 = 1; }));

    // A held lwarx reservation decides what the next stwcx. does, so two runs
    // that disagree about it have already diverged.
    CHECK(changes([](cpu::CoreState& s) { s.reserved = true; }));
    CHECK(changes([](cpu::CoreState& s) { s.reservation_address = 0x10000000; }));

    // Two separately constructed states must agree: CoreState is padded for
    // alignment, and hashing the padding would make identical machines differ.
    cpu::CoreState other;
    CHECK(Recording::hash_state(other) == base);
}

TEST_CASE("negative zero and zero hash differently", "[hle][recording]") {
    // Comparing floats would call them equal. The bit patterns are not, and a
    // divergence detector that cannot see the difference would miss a whole
    // class of FPU bug.
    cpu::CoreState zero;
    cpu::CoreState negative;
    negative.fpr[1].ps0 = -0.0;
    CHECK(Recording::hash_state(zero) != Recording::hash_state(negative));
}

// ---------------------------------------------------------------------------
// Host values
// ---------------------------------------------------------------------------

TEST_CASE("a recorded host value comes back on replay", "[hle][recording]") {
    u64 clock = 0;

    Recording recording;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 0});

    clock = 100;
    CHECK(recording.value(EventKind::HostTime, 0, u64{0x1234}) == 0x1234ULL);
    clock = 250;
    CHECK(recording.value(EventKind::Random, 0, u32{7}) == 7U);

    recording.finish(clock);
    REQUIRE(recording.event_count() == 2);

    // The replay produces different values and must be handed the old ones.
    clock = 0;
    recording.start_replay();
    clock = 100;
    CHECK(recording.value(EventKind::HostTime, 0, u64{0xFFFF}) == 0x1234ULL);
    clock = 250;
    CHECK(recording.value(EventKind::Random, 0, u32{99}) == 7U);
    recording.finish(clock);

    CHECK_FALSE(recording.diverged());
    CHECK(recording.remaining() == 0);
}

TEST_CASE("asking at a different instruction is a divergence", "[hle][recording]") {
    u64 clock = 0;

    Recording recording;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 0});
    clock = 100;
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{1}));
    recording.finish(150);

    clock = 0;
    recording.start_replay();
    clock = 101; // one instruction late: the run has already done something else
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{1}));

    REQUIRE(recording.diverged());
    CHECK(recording.divergence().recorded_instructions == 100);
    CHECK(recording.divergence().actual_instructions == 101);
}

TEST_CASE("asking for the wrong kind is a divergence", "[hle][recording]") {
    u64 clock = 0;

    Recording recording;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 0});
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{1}));

    recording.start_replay();
    static_cast<void>(recording.value(EventKind::Input, 0, u32{1}));
    REQUIRE(recording.diverged());
    CHECK(recording.divergence().detail.find("input") != std::string::npos);
}

TEST_CASE("a longer replay is a divergence", "[hle][recording]") {
    u64 clock = 0;

    Recording recording;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 0});
    recording.finish(500);

    recording.start_replay();
    recording.finish(600);
    REQUIRE(recording.diverged());
    CHECK(recording.divergence().recorded_instructions == 500);
    CHECK(recording.divergence().actual_instructions == 600);
}

TEST_CASE("only the first divergence is kept", "[hle][recording]") {
    u64 clock = 0;

    Recording recording;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 0});
    clock = 10;
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{1}));
    clock = 20;
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{2}));
    recording.finish(30);

    recording.start_replay();
    clock = 11;
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{1}));
    clock = 21;
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{2}));
    recording.finish(31);

    CHECK(recording.divergence().actual_instructions == 11);
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

TEST_CASE("a checkpoint catches a divergence nothing else would", "[hle][recording]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    u64 clock = 0;
    cpu::CoreState state;

    Recording recording;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 10});

    for (clock = 0; clock <= 30; ++clock) {
        state.gpr[3] = static_cast<u32>(clock);
        recording.tick(state, &memory);
    }
    recording.finish(30);
    REQUIRE(recording.event_count() == 4); // 0, 10, 20, 30

    // The same run, except one register holds something else. No host value is
    // ever asked for, so only the state hash can see it.
    recording.start_replay();
    for (clock = 0; clock <= 30; ++clock) {
        state.gpr[3] = clock == 20 ? 0xBAD : static_cast<u32>(clock);
        recording.tick(state, &memory);
    }

    REQUIRE(recording.diverged());
    CHECK(recording.divergence().actual_instructions == 20);
    CHECK(recording.divergence().detail.find("state hash") != std::string::npos);
}

TEST_CASE("a watched memory range is part of the checkpoint", "[hle][recording]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    u64 clock = 0;
    cpu::CoreState state;

    Recording recording;
    recording.bind_clock(&clock);
    recording.watch(memory::data_heap_base, 256);
    recording.start_record({.checkpoint_interval = 10});

    REQUIRE(memory.fill(memory::data_heap_base, 256, 0xAA));
    recording.tick(state, &memory);
    clock = 10;
    recording.tick(state, &memory);
    recording.finish(10);

    // Identical registers, different memory.
    clock = 0;
    recording.start_replay();
    recording.tick(state, &memory);
    clock = 10;
    REQUIRE(memory.try_write<u8>(memory::data_heap_base + 128, 0xBB));
    recording.tick(state, &memory);

    REQUIRE(recording.diverged());
    CHECK(recording.divergence().actual_instructions == 10);
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

TEST_CASE("a trace survives a round trip through a file", "[hle][recording]") {
    u64 clock = 0;

    Recording recording;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 64});
    recording.set_module("hello.rpx", 0xABCDEF0123456789ULL);

    clock = 5;
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{0x0102030405060708}));
    clock = 9;
    static_cast<void>(recording.value(EventKind::Input, 2, u32{0xDEADBEEF}));
    recording.finish(17);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "wiiu-recording-roundtrip.trace";
    std::string error;
    INFO("save: " << error);
    REQUIRE(recording.save(path, error));

    Recording reloaded;
    INFO("load: " << error);
    REQUIRE(reloaded.load(path, error));

    CHECK(reloaded.module_name() == "hello.rpx");
    CHECK(reloaded.module_hash() == 0xABCDEF0123456789ULL);
    CHECK(reloaded.options().checkpoint_interval == 64U);
    CHECK(reloaded.total_instructions() == 17U);
    REQUIRE(reloaded.event_count() == recording.event_count());

    for (std::size_t i = 0; i < reloaded.event_count(); ++i) {
        const Event& original = recording.events()[i];
        const Event& copy = reloaded.events()[i];
        CHECK(copy.kind == original.kind);
        CHECK(copy.channel == original.channel);
        CHECK(copy.sequence == original.sequence);
        CHECK(copy.instructions == original.instructions);
        CHECK(copy.payload == original.payload);
    }

    // And it still replays.
    reloaded.bind_clock(&clock);
    reloaded.start_replay();
    clock = 5;
    CHECK(reloaded.value(EventKind::HostTime, 0, u64{0}) == 0x0102030405060708ULL);

    std::filesystem::remove(path);
}

TEST_CASE("a trace from a different build is refused", "[hle][recording]") {
    Recording recording;
    recording.set_module("hello.rpx", 0x1111);

    std::string error;
    CHECK(recording.check_module("hello.rpx", 0x1111, error));
    CHECK_FALSE(recording.check_module("other.rpx", 0x1111, error));
    CHECK_FALSE(recording.check_module("hello.rpx", 0x2222, error));

    // An unknown hash on either side means "not known", not "does not match".
    CHECK(recording.check_module("hello.rpx", 0, error));
}

TEST_CASE("a file that is not a trace is refused", "[hle][recording]") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "wiiu-recording-garbage.trace";
    {
        std::ofstream out{path, std::ios::binary};
        out << "this is not a trace, it is a text file";
    }

    Recording recording;
    std::string error;
    CHECK_FALSE(recording.load(path, error));
    CHECK(error.find("not a wiiu-emu trace") != std::string::npos);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// The real thing
// ---------------------------------------------------------------------------

TEST_CASE("a real module records and replays identically", "[hle][recording]") {
    Recording trace;

    {
        Machine machine;
        machine.load("homebrew.rpx");
        const RunReport report = machine.run(trace);
        trace.bind_clock(nullptr);

        INFO("outcome: " << run_outcome_name(report.outcome));
        REQUIRE(report.outcome == RunOutcome::Returned);
        REQUIRE(report.instructions > 0);
        CHECK(trace.total_instructions() == report.instructions);
        CHECK(trace.event_count() > 0);
    }

    // A second machine, from scratch, replaying the first one's trace.
    Machine machine;
    machine.load("homebrew.rpx");
    trace.start_replay();
    const RunReport report = machine.run(trace);

    INFO(trace.format_summary());
    CHECK_FALSE(trace.diverged());
    CHECK(report.outcome == RunOutcome::Returned);
    CHECK(trace.remaining() == 0);
}

TEST_CASE("a perturbed replay is caught", "[hle][recording]") {
    Recording trace;

    {
        Machine machine;
        machine.load("homebrew.rpx");
        const RunReport report = machine.run(trace, 32);
        trace.bind_clock(nullptr);
        REQUIRE(report.outcome == RunOutcome::Returned);
    }

    // The same module, run with one register poisoned before it starts. The
    // checkpoints must notice, and the run must stop rather than carrying on
    // producing consequences of a fault it has already seen.
    Machine machine;
    machine.load("homebrew.rpx");
    machine.state.gpr[14] = 0xDEADBEEF;

    trace.start_replay();
    const RunReport report = machine.run(trace, 32);

    INFO(trace.format_summary());
    CHECK(trace.diverged());
    CHECK(report.outcome == RunOutcome::Diverged);
}

TEST_CASE("the recorded clock is what the replay sees", "[hle][recording]") {
    // OSGetTime is the one value in the emulator that genuinely differs
    // between two runs, so it is what record/replay exists for. clock.rpx
    // asks for it, turns it into a date, and prints it.
    Recording trace;
    std::vector<std::string> recorded_output;

    {
        Machine machine;
        machine.load("clock.rpx");
        const RunReport report = machine.run(trace);
        trace.bind_clock(nullptr);

        INFO("outcome: " << run_outcome_name(report.outcome));
        REQUIRE(report.outcome == RunOutcome::Returned);
        recorded_output = machine.output;
    }

    REQUIRE(recorded_output.size() == 2);
    INFO("recorded: " << recorded_output[0]);
    CHECK(recorded_output[0].rfind("date 20", 0) == 0); // this century, at least

    // Exactly one host value: the real-time clock, read once.
    std::size_t host_times = 0;
    for (const Event& event : trace.events()) {
        host_times += event.kind == EventKind::HostTime ? 1 : 0;
    }
    CHECK(host_times == 1);

    Machine machine;
    machine.load("clock.rpx");
    trace.start_replay();
    const RunReport report = machine.run(trace);

    INFO(trace.format_summary());
    CHECK_FALSE(trace.diverged());
    CHECK(report.outcome == RunOutcome::Returned);
    CHECK(machine.output == recorded_output);
}

TEST_CASE("an authored clock puts the guest on a chosen date", "[hle][recording]") {
    // The sharp version of the test above: the replay is handed a clock value
    // that is nothing like the host's, and the guest must print that date. If
    // the emulator were quietly re-reading the host clock on replay, this is
    // the test that would notice.
    Recording captured;
    {
        Machine machine;
        machine.load("clock.rpx");
        REQUIRE(machine.run(captured).outcome == RunOutcome::Returned);
        captured.bind_clock(nullptr);
    }

    const coreinit::CalendarTime chosen{.second = 5,
                                        .minute = 4,
                                        .hour = 3,
                                        .day = 2,
                                        .month = 5, // June, counting from zero
                                        .year = 2011};
    const u64 ticks = coreinit::calendar_to_ticks(chosen);

    // Same shape as the captured trace -- same kinds, same instruction counts
    // -- with the one host value replaced.
    Recording authored;
    for (const Event& event : captured.events()) {
        if (event.kind != EventKind::HostTime) {
            authored.add_event(event.kind, event.channel, event.instructions, event.payload);
            continue;
        }
        std::vector<u8> payload(sizeof(ticks));
        std::memcpy(payload.data(), &ticks, sizeof(ticks));
        authored.add_event(event.kind, event.channel, event.instructions, payload);
    }

    Machine machine;
    machine.load("clock.rpx");
    authored.start_replay();
    static_cast<void>(machine.run(authored));

    REQUIRE(machine.output.size() == 2);
    INFO(machine.output[0]);
    CHECK(machine.output[0] == "date 2011-6-2 3:4:5");
}

TEST_CASE("threads that are not running are part of the hash", "[hle][recording]") {
    // A hash of the live core says nothing about a thread that is blocked, so
    // a divergence in one would stay invisible until it was scheduled back in
    // -- long after whatever caused it.
    const auto build = [](u32 poisoned) {
        Scheduler scheduler;
        cpu::CoreState state;
        static_cast<void>(scheduler.adopt_main(state, main_thread_handle));
        static_cast<void>(scheduler.create(
            0x11000000, "worker", 0x02000100, 0, 0, 0x10800000, 0x1000, default_priority, 0x7));
        Thread* const worker = scheduler.find(0x11000000);
        REQUIRE(worker != nullptr);
        worker->context.gpr[10] = poisoned;
        return scheduler.hash(0);
    };

    CHECK(build(0) == build(0));
    CHECK(build(0) != build(1));
}

TEST_CASE("a value of the wrong width is a divergence", "[hle][recording]") {
    // Consuming the event and quietly returning the host value would leave a
    // replay non-deterministic while reporting a clean run, which is worse
    // than failing.
    u64 clock = 0;
    Recording recording;
    recording.bind_clock(&clock);

    const u32 narrow = 0x11223344;
    const std::array<u8, sizeof(narrow)> payload{static_cast<u8>(narrow),
                                                 static_cast<u8>(narrow >> 8),
                                                 static_cast<u8>(narrow >> 16),
                                                 static_cast<u8>(narrow >> 24)};
    recording.add_event(EventKind::HostTime, 0, 0, payload);
    recording.start_replay();

    CHECK(recording.value(EventKind::HostTime, 0, u64{0xFFFF}) == 0xFFFFULL);
    CHECK(recording.diverged());
    CHECK(recording.divergence().detail.find("bytes") != std::string::npos);
}

TEST_CASE("watched ranges survive a trace file", "[hle][recording]") {
    // A checkpoint hash means nothing without them: replaying a trace whose
    // ranges were dropped reports a divergence that is really a configuration
    // mismatch, at the first checkpoint, every time.
    Recording recording;
    recording.watch(memory::data_heap_base, 256);
    recording.watch(memory::mem1_base, 64);
    recording.start_record({.checkpoint_interval = 0});

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "wiiu-recording-watched.trace";
    std::string error;
    REQUIRE(recording.save(path, error));

    Recording reloaded;
    REQUIRE(reloaded.load(path, error));
    REQUIRE(reloaded.watched().size() == 2);
    CHECK(reloaded.watched()[0].base == memory::data_heap_base);
    CHECK(reloaded.watched()[0].size == 256U);
    CHECK(reloaded.watched()[1].base == memory::mem1_base);

    std::filesystem::remove(path);
}

TEST_CASE("a corrupted trace is rejected, not acted on", "[hle][recording]") {
    Recording recording;
    u64 clock = 0;
    recording.bind_clock(&clock);
    recording.start_record({.checkpoint_interval = 0});
    recording.set_module("hello.rpx", 0x1234);
    clock = 7;
    static_cast<void>(recording.value(EventKind::HostTime, 0, u64{1}));
    recording.finish(9);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "wiiu-recording-corrupt.trace";
    std::string error;
    REQUIRE(recording.save(path, error));

    std::vector<u8> bytes;
    {
        std::ifstream in{path, std::ios::binary};
        bytes.assign(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});
    }
    REQUIRE(bytes.size() > 48);

    const auto write_back = [&](const std::vector<u8>& data) {
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    };

    SECTION("an event count the file cannot hold") {
        // The count sits after magic(8) version(4) flags(4) name-length(4)
        // name(9) hash(8) interval(8) length(8) watch-count(4).
        std::vector<u8> broken = bytes;
        const std::size_t count_at = 8 + 4 + 4 + 4 + 9 + 8 + 8 + 8 + 4;
        REQUIRE(broken.size() > count_at + 8);
        for (std::size_t i = 0; i < 8; ++i) {
            broken[count_at + i] = 0xFF;
        }
        write_back(broken);

        // Reserving for 2^64 events must be an error message, not a throw and
        // not a multi-gigabyte allocation.
        Recording loaded;
        CHECK_FALSE(loaded.load(path, error));
        CHECK(error.find("cannot hold") != std::string::npos);
    }

    SECTION("a truncated file") {
        std::vector<u8> broken(bytes.begin(), bytes.begin() + 40);
        write_back(broken);

        Recording loaded;
        CHECK_FALSE(loaded.load(path, error));
    }

    SECTION("a failed load leaves the previous trace alone") {
        std::vector<u8> broken(bytes.begin(), bytes.begin() + 40);
        write_back(broken);

        Recording loaded;
        std::string ok_error;
        // Load something good first, then something broken.
        REQUIRE(recording.save(std::filesystem::temp_directory_path() / "wiiu-recording-good.trace",
                               ok_error));
        REQUIRE(loaded.load(std::filesystem::temp_directory_path() / "wiiu-recording-good.trace",
                            ok_error));
        REQUIRE(loaded.event_count() == 1);

        CHECK_FALSE(loaded.load(path, error));
        CHECK(loaded.event_count() == 1); // untouched
        CHECK(loaded.module_name() == "hello.rpx");

        std::filesystem::remove(std::filesystem::temp_directory_path() /
                                "wiiu-recording-good.trace");
    }

    std::filesystem::remove(path);
}

TEST_CASE("a fault on the last budgeted instruction keeps its outcome",
          "[hle][recording][loader]") {
    // Faulted and Halted are set after the instruction counter has been
    // bumped, so a trailing "did we run out of budget" test would relabel a
    // real fault as a budget expiry and hide it.
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    cpu::CoreState state;
    state.pc = 0x70000000; // outside every mapped region
    cpu::Interpreter interpreter{state, memory};

    Registry registry;
    const RunReport report = hle::run(interpreter, memory, registry, {.budget = 1});

    CHECK(report.instructions == 1);
    CHECK(report.outcome == RunOutcome::Faulted);
}
