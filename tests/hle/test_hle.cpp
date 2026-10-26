#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/hle/proc_ui.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/hle/system_libraries.hpp>
#include <wiiu/loader/rpl.hpp>
#include <wiiu/loader/vfs.hpp>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;

namespace {

[[nodiscard]] std::filesystem::path data_path(std::string_view name) {
    return std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / name;
}

/// A whole machine: memory, a core, the HLE registry, and somewhere for
/// OSReport output to land.
struct Machine {
    memory::Memory memory;
    cpu::CoreState state;
    cpu::Interpreter interpreter{state, memory};
    Registry registry;
    coreinit::Context os;
    loader::VirtualFileSystem vfs;
    std::vector<std::string> output;

    Machine() {
        // The trampoline area is outside every Cafe OS region, so it has to be
        // mapped explicitly -- the same gap the exception vectors have.
        std::vector<memory::Region> regions(memory::default_regions().begin(),
                                            memory::default_regions().end());
        // Writable because the registry writes the `sc` words into it as
        // imports resolve; the guest only ever branches here.
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
        INFO("heaps: " << heap_error);
        REQUIRE(os.heaps.create_default_heaps(memory, heap_error));

        std::string mount_error;
        REQUIRE(vfs.mount_title_directory(
            std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title", mount_error));
        os.filesystem.set_filesystem(&vfs);
    }

    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;

    [[nodiscard]] LoadResult load(std::string_view file, LoadedModule& module) {
        loader::Rpl rpl;
        const loader::ParseResult parsed = parse_file(data_path(file).string(), rpl);
        INFO("parse: " << parsed.error);
        REQUIRE(parsed.ok);
        return load_module(rpl, memory, registry, state, module);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Argument marshalling
// ---------------------------------------------------------------------------

TEST_CASE("arguments are read from the ABI registers", "[hle][registry]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    cpu::CoreState state;
    state.gpr[3] = 10;
    state.gpr[4] = 0xFFFFFFFF;
    state.gpr[5] = 7;
    state.fpr[1].ps0 = 2.5;

    Call call{.state = state, .memory = memory};

    CHECK(call.next_word() == 10U);
    CHECK(static_cast<s32>(call.next_word()) == -1);
    CHECK(call.next_word() == 7U);
    CHECK(call.next_double() == 2.5);
}

// The three cases that pin the parity down, each checked against what
// powerpc-linux-gnu-gcc -m32 actually emits for the corresponding prototype.
// The pair starts at an *odd* register, so a leading 64-bit argument uses
// r3:r4 with no gap and one preceding word costs a skipped register.
TEST_CASE("a 64-bit argument takes an odd-aligned register pair", "[hle][registry]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    SECTION("f(long long) uses r3:r4") {
        cpu::CoreState state;
        state.gpr[3] = 0x11223344;
        state.gpr[4] = 0x55667788;

        Call call{.state = state, .memory = memory};
        CHECK(call.next_long() == 0x1122334455667788ULL);
        CHECK(call.next_gpr == 5U);
    }

    SECTION("f(int, long long) skips r4 and uses r5:r6") {
        cpu::CoreState state;
        state.gpr[3] = 0x11;
        state.gpr[4] = 0xDEADBEEF; // the skipped register: must not be read
        state.gpr[5] = 0x22334455;
        state.gpr[6] = 0x66778899;

        Call call{.state = state, .memory = memory};
        CHECK(call.next_word() == 0x11U);
        CHECK(call.next_long() == 0x2233445566778899ULL);
    }

    SECTION("f(int, int, long long) needs no gap") {
        cpu::CoreState state;
        state.gpr[3] = 0x11;
        state.gpr[4] = 0x22;
        state.gpr[5] = 0x33445566;
        state.gpr[6] = 0x77889900;

        Call call{.state = state, .memory = memory};
        CHECK(call.next_word() == 0x11U);
        CHECK(call.next_word() == 0x22U);
        CHECK(call.next_long() == 0x3344556677889900ULL);
    }

    SECTION("f(long long, int, long long) skips r6") {
        cpu::CoreState state;
        state.gpr[3] = 0x11111111;
        state.gpr[4] = 0x22222222;
        state.gpr[5] = 0x33;
        state.gpr[6] = 0xDEADBEEF;
        state.gpr[7] = 0x44444444;
        state.gpr[8] = 0x55555555;

        Call call{.state = state, .memory = memory};
        CHECK(call.next_long() == 0x1111111122222222ULL);
        CHECK(call.next_word() == 0x33U);
        CHECK(call.next_long() == 0x4444444455555555ULL);
    }
}

// Past r10 and f8, arguments are in the caller's parameter area at 8(r1).
// Each layout is what powerpc-linux-gnu-gcc -m32 -O2 emits for the call.
TEST_CASE("arguments past the registers are read from the caller's stack", "[hle][registry]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));
    constexpr ppcaddr_t sp = 0x10100000;

    SECTION("f(int x13): the ninth to thirteenth at 8(r1) to 24(r1)") {
        cpu::CoreState state;
        for (u32 r = 3; r <= 10; ++r) {
            state.gpr[r] = r - 2;
        }
        state.gpr[1] = sp;
        for (u32 i = 0; i < 5; ++i) {
            REQUIRE(memory.try_write<u32>(sp + 8 + (i * 4), 9 + i));
        }
        Call call{.state = state, .memory = memory};
        for (u32 expected = 1; expected <= 13; ++expected) {
            CHECK(call.next_word() == expected);
        }
    }

    SECTION("f(int x9, long long, int): the long is 8-aligned, at 16(r1)") {
        cpu::CoreState state;
        state.gpr[1] = sp;
        REQUIRE(memory.try_write<u32>(sp + 8, 9));
        REQUIRE(memory.try_write<u32>(sp + 12, 0xDEADBEEF)); // the gap
        REQUIRE(memory.try_write<u64>(sp + 16, 0x1122334455667788ULL));
        REQUIRE(memory.try_write<u32>(sp + 24, 13));
        Call call{.state = state, .memory = memory};
        for (u32 i = 0; i < 8; ++i) {
            static_cast<void>(call.next_word());
        }
        CHECK(call.next_word() == 9U);
        CHECK(call.next_long() == 0x1122334455667788ULL);
        CHECK(call.next_word() == 13U);
    }

    SECTION("a long long with only r10 left goes to the stack, and so does what follows") {
        cpu::CoreState state;
        state.gpr[1] = sp;
        state.gpr[10] = 0xDEADBEEF; // skipped, and not used afterwards either
        REQUIRE(memory.try_write<u64>(sp + 8, 0x0102030405060708ULL));
        REQUIRE(memory.try_write<u32>(sp + 16, 42));
        Call call{.state = state, .memory = memory};
        for (u32 i = 0; i < 7; ++i) {
            static_cast<void>(call.next_word());
        }
        CHECK(call.next_long() == 0x0102030405060708ULL);
        CHECK(call.next_word() == 42U);
    }

    SECTION("floats past f8 take a word each; doubles eight bytes") {
        cpu::CoreState state;
        state.gpr[1] = sp;
        REQUIRE(memory.try_write<u32>(sp + 8, std::bit_cast<u32>(2.5F)));
        REQUIRE(memory.try_write<u32>(sp + 12, std::bit_cast<u32>(-1.0F)));
        REQUIRE(memory.try_write<u64>(sp + 16, std::bit_cast<u64>(0.125)));
        Call call{.state = state, .memory = memory};
        for (u32 i = 0; i < 8; ++i) {
            static_cast<void>(call.next_single());
        }
        CHECK(call.next_single() == 2.5F);
        CHECK(call.next_single() == -1.0F);
        CHECK(call.next_double() == 0.125);
    }
}

TEST_CASE("guest strings are read, and bounded", "[hle][registry]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    const ppcaddr_t at = memory::data_heap_base;
    const std::string text = "coreinit";
    REQUIRE(memory.write_bytes(at, {reinterpret_cast<const u8*>(text.data()), text.size() + 1}));

    cpu::CoreState state;
    Call call{.state = state, .memory = memory};
    CHECK(call.read_string(at) == "coreinit");

    // A string with no terminator must stop rather than run through memory.
    REQUIRE(memory.fill(at, 64, 'A'));
    CHECK(call.read_string(at, 16).size() == 16);
}

TEST_CASE("a handler is an ordinary function", "[hle][registry]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    Registry registry;
    registry.add("test", "add", +[](Call&, u32 lhs, u32 rhs) -> u32 { return lhs + rhs; });

    cpu::CoreState state;
    state.gpr[3] = 20;
    state.gpr[4] = 22;
    state.lr = 0x02000000;

    const ppcaddr_t trampoline = registry.resolve("test", "add");
    REQUIRE(trampoline != 0);
    REQUIRE(registry.dispatch(trampoline, state, memory));

    CHECK(state.gpr[3] == 42U);
    CHECK(state.pc == 0x02000000U); // returned through LR
}

TEST_CASE("an unimplemented import still resolves", "[hle][registry]") {
    // A title must load even when something it imports is missing; the gap
    // should surface when the function is called, not before.
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    Registry registry;
    const ppcaddr_t trampoline = registry.resolve("coreinit", "SomethingNew");
    REQUIRE(trampoline != 0);

    const Function* const function = registry.find_by_trampoline(trampoline);
    REQUIRE(function != nullptr);
    CHECK(function->is_stub);

    cpu::CoreState state;
    state.gpr[3] = 0xDEADBEEF;
    state.lr = 0x02000000;
    REQUIRE(registry.dispatch(trampoline, state, memory));
    CHECK(state.gpr[3] == 0U);

    const auto called = registry.unimplemented_called();
    REQUIRE(called.size() == 1);
    CHECK(called[0]->name == "SomethingNew");
    CHECK(called[0]->call_count == 1);
}

TEST_CASE("resolving twice gives the same trampoline", "[hle][registry]") {
    Registry registry;
    const ppcaddr_t first = registry.resolve("coreinit", "OSReport");
    const ppcaddr_t second = registry.resolve("coreinit", "OSReport");
    CHECK(first == second);
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

TEST_CASE("OSReport formatting handles the common conversions", "[hle][coreinit]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    cpu::CoreState state;
    state.gpr[4] = 42;
    state.gpr[5] = 0xBEEF;
    state.gpr[6] = static_cast<u32>(-7);

    Call call{.state = state, .memory = memory};
    CHECK(coreinit::format_guest(call, "n=%d hex=%x neg=%d", 4) == "n=42 hex=beef neg=-7");
}

TEST_CASE("width and length modifiers are skipped, not printed", "[hle][coreinit]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    cpu::CoreState state;
    state.gpr[4] = 5;

    Call call{.state = state, .memory = memory};
    CHECK(coreinit::format_guest(call, "[%08d]", 4) == "[5]");
    CHECK(coreinit::format_guest(call, "100%% done", 4) == "100% done");
}

TEST_CASE("a %s argument is followed into guest memory", "[hle][coreinit]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    const ppcaddr_t at = memory::data_heap_base;
    const std::string text = "world";
    REQUIRE(memory.write_bytes(at, {reinterpret_cast<const u8*>(text.data()), text.size() + 1}));

    cpu::CoreState state;
    state.gpr[4] = at;
    state.gpr[5] = 0;

    Call call{.state = state, .memory = memory};
    CHECK(coreinit::format_guest(call, "hello %s", 4) == "hello world");
    CHECK(coreinit::format_guest(call, "%s", 5) == "(null)");
}

// ---------------------------------------------------------------------------
// M2 -- loading and running a module
// ---------------------------------------------------------------------------

TEST_CASE("a module loads into guest memory", "[hle][loader]") {
    Machine machine;
    LoadedModule module;
    const LoadResult result = machine.load("homebrew.rpx", module);

    INFO("error: " << result.error);
    REQUIRE(result.ok);

    CHECK(module.name == "homebrew.rpx");
    CHECK(module.entry == 0x02000000U);
    CHECK(module.unresolved_imports.empty());

    // The first instruction is really in guest memory now.
    CHECK(machine.memory.read<u32>(0x02000000) == 0x7C0802A6U); // mflr r0
}

TEST_CASE("imports resolve to trampolines", "[hle][loader]") {
    Machine machine;
    LoadedModule module;
    REQUIRE(machine.load("homebrew.rpx", module).ok);

    REQUIRE_FALSE(module.resolved_imports.empty());
    const bool has_report = std::ranges::find(module.resolved_imports, "coreinit.OSReport") !=
                            module.resolved_imports.end();
    CHECK(has_report);

    // The call site was relocated from `bl .` to the trampoline.
    const Function* const report = machine.registry.find("coreinit", "OSReport");
    REQUIRE(report != nullptr);
    REQUIRE(report->trampoline != 0);

    const u32 call_site = machine.memory.read<u32>(0x02000014);
    const auto displacement =
        static_cast<s32>((call_site & 0x03FFFFFC) << 6) >> 6; // sign-extend 26 bits
    CHECK(0x02000014 + displacement == static_cast<s32>(report->trampoline));
}

TEST_CASE("a string address is assembled from lis and addi", "[hle][loader]") {
    Machine machine;
    LoadedModule module;
    REQUIRE(machine.load("homebrew.rpx", module).ok);

    // The two halves have to reconstruct the .rodata address exactly, which is
    // what the HA adjustment exists for.
    const u32 lis = machine.memory.read<u32>(0x0200000C);
    const u32 addi = machine.memory.read<u32>(0x02000010);

    const auto high = static_cast<u16>(lis & 0xFFFF);
    const auto low = static_cast<s16>(addi & 0xFFFF);
    const u32 address = (static_cast<u32>(high) << 16) + static_cast<u32>(low);

    CHECK(address == 0x10000000U);

    // ... and that address holds the string.
    std::string text;
    for (u32 i = 0; i < 32; ++i) {
        const u8 byte = machine.memory.read<u8>(address + i);
        if (byte == 0) {
            break;
        }
        text.push_back(static_cast<char>(byte));
    }
    CHECK(text == "hello from wiiu-emu");
}

TEST_CASE("M2: a module runs and prints through OSReport", "[hle][m2]") {
    Machine machine;
    LoadedModule module;
    REQUIRE(machine.load("homebrew.rpx", module).ok);

    const RunReport report = run(machine.interpreter, machine.memory, machine.registry);

    INFO("outcome: " << run_outcome_name(report.outcome) << " after " << report.instructions
                     << " instructions");
    CHECK(report.outcome == RunOutcome::Returned);
    CHECK(report.hle_calls == 2);

    REQUIRE(machine.output.size() == 2);
    CHECK(machine.output[0] == "hello from wiiu-emu");
    CHECK(machine.output[1] == "sum 1..10 = 55");

    // The program returned 0 and left the stack where it found it.
    CHECK(machine.state.gpr[3] == 0U);
}

TEST_CASE("the load region stays the loader's, and a data import is a pointer to call through",
          "[hle][loader]") {
    // dataimport.rpx is laid out as elf2rpl lays out a wut RPX: its import
    // table at 0xC0000000, the load region, which the program never
    // addresses; and a call through coreinit's MEMAllocFromDefaultHeap,
    // which coreinit exports as data -- a pointer to the default heap's
    // allocator.
    Machine machine;
    LoadedModule module;
    const LoadResult result = machine.load("dataimport.rpx", module);
    INFO("error: " << result.error);
    REQUIRE(result.ok);
    CHECK(result.warnings.empty());
    CHECK(module.unresolved_imports.empty());
    CHECK(std::ranges::find(module.resolved_imports, "coreinit.MEMAllocFromDefaultHeap") !=
          module.resolved_imports.end());

    // lis/lwz name a word in the HLE area holding the allocator's trampoline.
    const u32 lis = machine.memory.read<u32>(0x0200000C);
    const u32 lwz = machine.memory.read<u32>(0x02000010);
    const u32 cell = ((lis & 0xFFFFU) << 16) +
                     static_cast<u32>(static_cast<s32>(static_cast<s16>(lwz & 0xFFFF)));
    CHECK(cell >= Registry::data_export_base);
    const Function* const alloc = machine.registry.find("coreinit", "MEMAllocFromDefaultHeap");
    REQUIRE(alloc != nullptr);
    CHECK(machine.memory.read<u32>(cell) == alloc->trampoline);
    CHECK(machine.registry.resolve_data("coreinit", "MEMAllocFromDefaultHeap") == cell);

    // And the call through it allocates.
    const RunReport report = run(machine.interpreter, machine.memory, machine.registry);
    CHECK(report.outcome == RunOutcome::Returned);
    CHECK(alloc->call_count == 1);
    CHECK(machine.state.gpr[3] != 0U);

    // A data import that is no function implemented here stays unresolved.
    CHECK(machine.registry.resolve_data("coreinit", "__OSCurrentThread") == 0);
}

TEST_CASE("an RPX's __preinit_user runs before its entry point", "[hle][loader]") {
    // Cafe OS's loader calls it with pointers to the MEM1, foreground and
    // MEM2 base heap handles, and the handles it leaves there become the
    // base heaps. wut's sets up malloc there: skipping it leaves every
    // allocation failing, long after the cause.
    Machine machine;
    LoadedModule module;
    const LoadResult result = machine.load("preinit.rpx", module);
    INFO("error: " << result.error);
    REQUIRE(result.ok);
    const ppcaddr_t preinit = module.find_export("__preinit_user");
    CHECK(preinit == 0x02000010U);
    CHECK(module.find_export("main") == 0U);

    const u32 stack = machine.state.gpr[1];
    const u32 mem1 = machine.os.heaps.base(BaseHeap::Mem1);
    const u32 mem2 = machine.os.heaps.base(BaseHeap::Mem2);
    REQUIRE(machine.os.heaps.base(BaseHeap::Foreground) == 0U);

    std::string error;
    REQUIRE(
        coreinit::start_with_preinit(machine.os, machine.memory, machine.state, preinit, error));
    CHECK(machine.state.pc == preinit);

    const RunReport report = run(machine.interpreter, machine.memory, machine.registry);
    CHECK(report.outcome == RunOutcome::Returned);

    // The preinit was handed the MEM2 handle, and ran first: the entry point
    // read what it recorded.
    CHECK(machine.state.gpr[4] == mem2);
    // The entry point had the loader's stack, as if nothing had run before it.
    CHECK(machine.state.gpr[3] == stack);

    // What the preinit left is the base heaps now; what it did not change
    // stays.
    CHECK(machine.os.heaps.base(BaseHeap::Foreground) == 0x12345678U);
    CHECK(machine.os.heaps.base(BaseHeap::Mem1) == mem1);
    CHECK(machine.os.heaps.base(BaseHeap::Mem2) == mem2);
    CHECK(machine.os.startup.frame == 0U);
}

TEST_CASE("a module without __preinit_user is entered directly", "[hle][loader]") {
    Machine machine;
    LoadedModule module;
    REQUIRE(machine.load("homebrew.rpx", module).ok);
    const cpu::CoreState before = machine.state;

    std::string error;
    REQUIRE(coreinit::start_with_preinit(
        machine.os, machine.memory, machine.state, module.find_export("__preinit_user"), error));
    CHECK(machine.state.pc == before.pc);
    CHECK(machine.state.gpr[1] == before.gpr[1]);
    CHECK(machine.state.lr == before.lr);
}

TEST_CASE("a wut-built RPX loads with every relocation applied", "[hle][loader][wut]") {
    // scripts/wut-samples.sh builds wut's helloworld sample: compiled against
    // wut and newlib and made an RPX by elf2rpl, as a homebrew is. Not every
    // checkout has one; without it there is nothing to check.
    if (!std::filesystem::exists(data_path("wut/helloworld.rpx"))) {
        SKIP("tests/loader/data/wut/helloworld.rpx is not built (scripts/wut-samples.sh)");
    }
    Machine machine;
    LoadedModule module;
    const LoadResult result = machine.load("wut/helloworld.rpx", module);
    INFO("error: " << result.error);
    REQUIRE(result.ok);
    for (const std::string& warning : result.warnings) {
        UNSCOPED_INFO(warning);
    }
    CHECK(result.warnings.empty());
    CHECK(module.entry == 0x02000000U);
    CHECK(module.unresolved_imports.empty());
    // Its default-heap pointers are data imports, called through.
    for (const char* name :
         {"MEMAllocFromDefaultHeap", "MEMAllocFromDefaultHeapEx", "MEMFreeToDefaultHeap"}) {
        INFO(name);
        CHECK(std::ranges::find(module.resolved_imports, std::string{"coreinit."} + name) !=
              module.resolved_imports.end());
    }
}

TEST_CASE("wut's helloworld reaches its main loop and draws its console", "[hle][wut]") {
    // The whole of a wut title's start-up: __preinit_user's malloc heap,
    // newlib's per-thread state in a thread-specific value, the spin locks
    // around malloc, the FSA client for the SD card, WHBProc's ProcUI loop,
    // and WHBLogConsole's OSScreen frames. It prints a line, then a line a
    // second, and draws them every frame.
    if (!std::filesystem::exists(data_path("wut/helloworld.rpx"))) {
        SKIP("tests/loader/data/wut/helloworld.rpx is not built (scripts/wut-samples.sh)");
    }
    Machine machine;
    procui::State procui;
    procui::register_all(machine.registry, machine.os, procui);
    register_system_libraries(machine.registry);

    LoadedModule module;
    const LoadResult result = machine.load("wut/helloworld.rpx", module);
    INFO("error: " << result.error);
    REQUIRE(result.ok);
    std::string error;
    REQUIRE(coreinit::start_with_preinit(
        machine.os, machine.memory, machine.state, module.find_export("__preinit_user"), error));
    static_cast<void>(machine.os.scheduler.adopt_main(machine.state, main_thread_handle));

    // What the console puts on the TV, row by row.
    std::map<u32, std::string> rows;
    Function* const put = machine.registry.find("coreinit", "OSScreenPutFontEx");
    REQUIRE(put != nullptr);
    put->handler = [inner = put->handler, &rows](Call& call) {
        if (call.state.gpr[3] == 0 && call.state.gpr[6] != 0) {
            rows[call.state.gpr[5]] = call.read_string(call.state.gpr[6]);
        }
        inner(call);
    };

    // Twenty-five frames, a tenth of a second each, and the last one's
    // corner pixel: the console's background.
    constexpr u64 frames = 25;
    u64 flips = 0;
    u32 corner = 0;
    machine.os.screens.on_flip = [&](const coreinit::ScreenFlip& flip,
                                     const memory::Memory& memory) {
        if (flip.screen != coreinit::ScreenId::Tv) {
            return true;
        }
        const u32 last = (flip.geometry.height - 1) * flip.geometry.pitch * 4;
        corner = memory.read<u32>(flip.address + last);
        return ++flips < frames;
    };

    const RunReport report = run(machine.interpreter,
                                 machine.memory,
                                 machine.registry,
                                 {.scheduler = &machine.os.scheduler});
    INFO("outcome: " << run_outcome_name(report.outcome));
    CHECK(report.outcome == RunOutcome::Halted);
    CHECK(machine.os.screens.stop_requested);
    CHECK(flips == frames);
    for (const Function* function : machine.registry.unimplemented_called()) {
        UNSCOPED_INFO(function->library << "." << function->name);
    }
    CHECK(machine.registry.unimplemented_called().empty());

    CHECK(rows[0] == "Hello World!\n");
    // Then the time, once a second: two and a half seconds is three lines.
    const std::string still_here = "I'm still here.\n";
    for (u32 row = 1; row <= 3; ++row) {
        INFO("row " << row << ": " << rows[row]);
        CHECK(rows[row].ends_with(still_here));
    }
    CHECK(corner == 0x993333FFU);
}

TEST_CASE("the run loop stops on a fault rather than looping", "[hle][loader]") {
    Machine machine;
    LoadedModule module;
    REQUIRE(machine.load("homebrew.rpx", module).ok);

    // Point the entry at unmapped memory.
    machine.state.pc = 0x70000000;

    const RunReport report =
        run(machine.interpreter, machine.memory, machine.registry, {.budget = 1000});
    CHECK(report.outcome == RunOutcome::Faulted);
    CHECK(report.last_result == cpu::StepResult::MemoryFault);
}

TEST_CASE("a budget stops a runaway module", "[hle][loader]") {
    Machine machine;
    LoadedModule module;
    REQUIRE(machine.load("homebrew.rpx", module).ok);

    // `b .` at the entry point: an infinite loop.
    REQUIRE(machine.memory.try_write<u32>(module.entry, 0x48000000));

    const RunReport report =
        run(machine.interpreter, machine.memory, machine.registry, {.budget = 500});
    CHECK(report.outcome == RunOutcome::BudgetExpired);
    CHECK(report.instructions == 500);
}

TEST_CASE("a stub never replaces a real implementation", "[hle][registry]") {
    // The "known but not implemented" lists are long and hand-maintained, so a
    // name drifting into one after being implemented is a matter of time. It
    // must not quietly turn a working function back into a placeholder.
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    Registry registry;
    registry.add("test", "thing", +[](Call&) -> u32 { return 42; });
    registry.add_stub("test", "thing");

    const Function* const function = registry.find("test", "thing");
    REQUIRE(function != nullptr);
    CHECK_FALSE(function->is_stub);

    cpu::CoreState state;
    state.lr = 0x02000000;
    const ppcaddr_t trampoline = registry.resolve("test", "thing");
    REQUIRE(registry.dispatch(trampoline, state, memory));
    CHECK(state.gpr[3] == 42U);
}

TEST_CASE("a real implementation does replace a stub", "[hle][registry]") {
    // The other direction has to work: imports resolve before every library is
    // registered, so a stub is often created first.
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    Registry registry;
    const ppcaddr_t trampoline = registry.resolve("test", "later");
    REQUIRE(registry.find("test", "later")->is_stub);

    registry.add("test", "later", +[](Call&) -> u32 { return 7; });

    const Function* const function = registry.find("test", "later");
    CHECK_FALSE(function->is_stub);
    // ... and the trampoline the module was already relocated against still
    // points at it.
    CHECK(function->trampoline == trampoline);

    cpu::CoreState state;
    state.lr = 0x02000000;
    REQUIRE(registry.dispatch(trampoline, state, memory));
    CHECK(state.gpr[3] == 7U);
}

TEST_CASE("call statistics record what a title actually uses", "[hle][trace]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    Registry registry;
    registry.add("test", "often", +[](Call&) -> u32 { return 0; });
    registry.add("test", "once", +[](Call&) -> u32 { return 0; });

    cpu::CoreState state;
    state.lr = 0x02000000;

    const ppcaddr_t often = registry.resolve("test", "often");
    const ppcaddr_t once = registry.resolve("test", "once");

    for (int i = 0; i < 5; ++i) {
        REQUIRE(registry.dispatch(often, state, memory));
    }
    REQUIRE(registry.dispatch(once, state, memory));

    CHECK(registry.total_calls() == 6);

    const auto used = registry.called();
    REQUIRE(used.size() == 2);
    CHECK(used[0]->name == "often"); // most-called first
    CHECK(used[0]->call_count == 5);
    CHECK(used[1]->name == "once");
}

TEST_CASE("the recent-call ring keeps the last calls before a fault", "[hle][trace]") {
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    Registry registry;
    registry.add("test", "thing", +[](Call&) -> u32 { return 0; });

    // Off by default: a ring of every HLE call is not free, and most runs do
    // not need one.
    CHECK(registry.recent_calls().empty());

    registry.set_trace_capacity(3);

    cpu::CoreState state;
    state.lr = 0x02000000;
    const ppcaddr_t trampoline = registry.resolve("test", "thing");

    for (u32 i = 0; i < 10; ++i) {
        state.gpr[3] = i;
        REQUIRE(registry.dispatch(trampoline, state, memory));
    }

    const auto recent = registry.recent_calls();
    REQUIRE(recent.size() == 3); // the oldest fall off
    CHECK(recent[0].arguments[0] == 7U);
    CHECK(recent[2].arguments[0] == 9U);
    CHECK(recent[2].caller == 0x02000000U);
    CHECK(recent[2].function->name == "thing");
}

TEST_CASE("the summary names stubs as stubs", "[hle][trace]") {
    // An unimplemented function showing up in the call counts is the single
    // most useful line in a fault report, so it has to be marked.
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));

    Registry registry;
    cpu::CoreState state;
    state.lr = 0x02000000;

    const ppcaddr_t trampoline = registry.resolve("coreinit", "SomethingMissing");
    REQUIRE(registry.dispatch(trampoline, state, memory));

    const std::string summary = registry.format_summary();
    CHECK(summary.find("coreinit.SomethingMissing") != std::string::npos);
    CHECK(summary.find("(stub)") != std::string::npos);
}
