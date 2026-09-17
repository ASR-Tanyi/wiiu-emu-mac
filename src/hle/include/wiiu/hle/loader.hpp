// Phase 6.2 — loading a module into guest memory.
//
// Takes a parsed RPX, writes its sections where they belong, relocates them,
// resolves every import to an HLE trampoline, and hands back a core ready to
// run.
#pragma once

#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/recording.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/hle/scheduler.hpp>
#include <wiiu/loader/rpl.hpp>
#include <wiiu/memory/memory.hpp>

#include <string>
#include <vector>

namespace wiiu::hle {

struct LoadedModule {
    std::string name;
    ppcaddr_t entry = 0;
    s32 delta = 0; ///< where it landed, minus where it was linked

    /// Imports resolved, by "library.symbol".
    std::vector<std::string> resolved_imports;
    std::vector<std::string> unresolved_imports;
};

struct LoadResult {
    bool ok = false;
    std::string error;
    std::vector<std::string> warnings;
};

struct LoadOptions {
    /// Stack size and where the stack pointer starts. Real values come from
    /// cos.xml (Phase 5.5); this is the fallback for a module without one.
    u32 stack_size = 0x10000;
    ppcaddr_t stack_top = memory::data_heap_base + 0x00100000;

    /// Relocate the module to a different base. 0 loads it where it was
    /// linked.
    ppcaddr_t load_base = 0;
};

/// Map a module and prepare `state` to enter it.
[[nodiscard]] LoadResult load_module(const loader::Rpl& rpl,
                                     memory::Memory& memory,
                                     Registry& registry,
                                     cpu::CoreState& state,
                                     LoadedModule& out,
                                     const LoadOptions& options = {});

/// Why the run loop stopped.
enum class RunOutcome : u8 {
    Returned,      ///< the entry point returned to the sentinel
    Halted,        ///< OSFatal, or the core parked itself
    Faulted,       ///< the interpreter could not continue
    BudgetExpired, ///< ran out of instructions
    Deadlocked,    ///< threads remain, none can run
    Diverged,      ///< a replay stopped matching its recording (10.4)
};

[[nodiscard]] std::string_view run_outcome_name(RunOutcome outcome) noexcept;

struct RunReport {
    RunOutcome outcome = RunOutcome::Returned;
    cpu::StepResult last_result = cpu::StepResult::Ok;
    u64 instructions = 0;
    u64 hle_calls = 0;
    u64 context_switches = 0;

    /// Timebase ticks jumped rather than emulated, because every thread was
    /// waiting for one of them (**P2**). Worth reporting: a title spending
    /// most of its time here is one where the emulator is keeping up easily,
    /// and a title where it stays at zero is one where nothing ever waits.
    u64 skipped_ticks = 0;

    /// Times a thread was found spinning on a flag and gave the core back
    /// early (**P2**). Against `context_switches` this says how much of the
    /// scheduling was the emulator noticing a wait rather than a timeslice
    /// running out.
    u64 spin_yields = 0;
};

struct RunOptions {
    /// Without one, the module is a single thread and the run loop stops when
    /// its entry point returns.
    Scheduler* scheduler = nullptr;

    u64 budget = 10'000'000;

    /// Instructions between scheduler checks. A thread that never yields --
    /// spinning on a flag another thread sets -- would otherwise run forever,
    /// and the program would appear to hang for reasons that look like an
    /// emulator fault rather than a scheduling one.
    u64 timeslice = Scheduler::default_timeslice;

    /// Record host-derived values and state checkpoints, or replay them
    /// (10.4). The run loop owns the instruction counter the events are
    /// anchored to, so it is the only place that can lend it out.
    Recording* recording = nullptr;
};

/// Run until the module returns, stops, or runs out of budget.
///
/// A system call is either an HLE trampoline, which the registry handles and
/// returns from, or something else — which stops the loop, because a guest
/// making an unrecognised kernel call is a gap worth seeing.
[[nodiscard]] RunReport run(cpu::Interpreter& interpreter,
                            memory::Memory& memory,
                            Registry& registry,
                            const RunOptions& options = {});

/// Address a `blr` returns to when the entry point finishes. Outside every
/// mapped region, so the run loop can spot it without the module noticing.
inline constexpr ppcaddr_t return_sentinel = 0x01FF0000;

} // namespace wiiu::hle
