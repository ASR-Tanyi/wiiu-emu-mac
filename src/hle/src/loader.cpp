#include <wiiu/common/log.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/loader/relocations.hpp>

#include <fmt/format.h>

#include <map>

namespace wiiu::hle {
namespace {

using wiiu::log::Channel;

} // namespace

LoadResult load_module(const loader::Rpl& rpl,
                       memory::Memory& memory,
                       Registry& registry,
                       cpu::CoreState& state,
                       LoadedModule& out,
                       const LoadOptions& options) {
    LoadResult result;
    out = LoadedModule{};
    out.name = rpl.module_name.empty() ? rpl.path : rpl.module_name;

    // --- where does it go --------------------------------------------------
    const loader::Section* const text = rpl.find_section(".text");
    if (text == nullptr) {
        result.error = "module has no .text section";
        return result;
    }

    out.delta = options.load_base == 0 ? 0 : static_cast<s32>(options.load_base - text->address);
    out.entry = rpl.entry + static_cast<u32>(out.delta);

    // --- resolve imports ---------------------------------------------------
    //
    // An imported symbol's section index points at the `.fimport_<library>`
    // section it comes from; that is the only thing tying a call site to a
    // library. Each one gets a trampoline, and relocations then point call
    // sites at it.
    std::map<u32, ppcaddr_t> symbol_addresses;

    for (std::size_t i = 0; i < rpl.symbols.size(); ++i) {
        const loader::Symbol& symbol = rpl.symbols[i];

        // Unnamed symbols are not skipped. A relocation against a section --
        // which is how `lis r3,msg@ha` refers to .rodata -- targets that
        // section's symbol, and section symbols have no name. Skipping them
        // leaves every string address at zero while the module otherwise
        // appears to load fine.
        const std::string library =
            symbol.name.empty() ? std::string{} : rpl.import_library(symbol);
        if (library.empty()) {
            // A local symbol: it moves with the module.
            symbol_addresses[static_cast<u32>(i)] = symbol.value + static_cast<u32>(out.delta);
            continue;
        }

        const ppcaddr_t trampoline = registry.resolve(library, symbol.name);
        if (trampoline == 0) {
            out.unresolved_imports.push_back(fmt::format("{}.{}", library, symbol.name));
            result.warnings.push_back(fmt::format("could not resolve {}.{}", library, symbol.name));
            continue;
        }

        symbol_addresses[static_cast<u32>(i)] = trampoline;
        out.resolved_imports.push_back(fmt::format("{}.{}", library, symbol.name));
    }

    std::string error;
    if (!registry.install(memory, error)) {
        result.error = error;
        return result;
    }

    // --- relocate ----------------------------------------------------------
    // Sections are relocated before being written, so guest memory is only
    // ever touched with finished contents.
    std::map<std::string, loader::Section> relocated;

    loader::RelocationContext context;
    context.delta = out.delta;
    context.resolve = [&symbol_addresses](u32 symbol, ppcaddr_t& address) {
        const auto it = symbol_addresses.find(symbol);
        if (it == symbol_addresses.end()) {
            return false;
        }
        address = it->second;
        return true;
    };

    for (const loader::Section& section : rpl.sections) {
        if (section.type != loader::sht_rela || section.data.empty()) {
            continue;
        }

        const loader::Section* const target = rpl.find_section(section.info);
        if (target == nullptr || !target->occupies_memory()) {
            continue;
        }

        loader::Section& working = relocated.try_emplace(target->name, *target).first->second;

        const auto relocations = loader::parse_relocations(section);
        const loader::RelocationResult applied =
            loader::apply_relocations(relocations, working, context);

        result.warnings.insert(
            result.warnings.end(), applied.warnings.begin(), applied.warnings.end());

        WIIU_LOG_DEBUG(Loader,
                       "{}: applied {} of {} relocations",
                       target->name,
                       applied.applied,
                       relocations.size());
    }

    // --- write into guest memory -------------------------------------------
    for (const loader::Section& section : rpl.sections) {
        if (!section.occupies_memory() || section.size == 0) {
            continue;
        }

        const auto it = relocated.find(section.name);
        const loader::Section& source = it == relocated.end() ? section : it->second;
        const ppcaddr_t address = section.address + static_cast<u32>(out.delta);

        if (section.type == loader::sht_nobits) {
            if (!memory.fill(address, section.size, 0)) {
                result.error = fmt::format("cannot zero {} at {:#010x}", section.name, address);
                return result;
            }
            continue;
        }

        if (!memory.write_bytes(address, source.data)) {
            result.error = fmt::format("cannot write {} ({} bytes) at {:#010x}",
                                       section.name,
                                       source.data.size(),
                                       address);
            return result;
        }

        WIIU_LOG_DEBUG(
            Loader, "{:<16} -> {:#010x} ({} bytes)", section.name, address, source.data.size());
    }

    // --- prepare the core --------------------------------------------------
    state.reset();
    state.pc = out.entry;
    state.next_pc = out.entry;

    // The ABI wants the stack 8-byte aligned with a null back-chain at the
    // top, so a walk up the frames terminates instead of wandering.
    const ppcaddr_t stack_pointer = (options.stack_top - 8) & ~7U;
    if (!memory.try_write<u32>(stack_pointer, 0)) {
        result.error = "cannot set up the stack";
        return result;
    }
    state.gpr[1] = stack_pointer;

    // Returning from the entry point lands on the sentinel, which the run
    // loop watches for.
    state.lr = return_sentinel;

    WIIU_LOG_INFO(Loader,
                  "loaded {} at {:#010x} (entry {:#010x}), {} imports resolved",
                  out.name,
                  text->address + static_cast<u32>(out.delta),
                  out.entry,
                  out.resolved_imports.size());

    result.ok = true;
    return result;
}

} // namespace wiiu::hle

namespace wiiu::hle {

std::string_view run_outcome_name(RunOutcome outcome) noexcept {
    switch (outcome) {
        case RunOutcome::Returned:
            return "returned";
        case RunOutcome::Halted:
            return "halted";
        case RunOutcome::Faulted:
            return "faulted";
        case RunOutcome::BudgetExpired:
            return "budget expired";
        case RunOutcome::Deadlocked:
            return "deadlocked";
        case RunOutcome::Diverged:
            return "diverged from the recording";
    }
    return "?";
}

RunReport run(cpu::Interpreter& interpreter,
              memory::Memory& memory,
              Registry& registry,
              const RunOptions& options) {
    RunReport report;
    cpu::CoreState& state = interpreter.state();
    Scheduler* const scheduler = options.scheduler;

    // The recording anchors its events to the instruction count, and the
    // count lives here. Lending the counter rather than keeping a second one
    // removes the possibility of the two disagreeing (10.4).
    Recording* const recording =
        options.recording != nullptr && options.recording->active() ? options.recording : nullptr;

    // A guard rather than a call at the end: `report.instructions` lives on
    // this frame, so an exception escaping the loop would otherwise leave the
    // recording holding a pointer into it. Same for the scheduler the hash
    // source captures.
    struct ClockGuard {
        Recording* recording;

        ~ClockGuard() {
            if (recording != nullptr) {
                recording->bind_clock(nullptr);
                recording->set_hash_source(nullptr);
            }
        }
    } const guard{recording};

    if (recording != nullptr) {
        recording->bind_clock(&report.instructions);
        if (scheduler != nullptr) {
            recording->set_hash_source([scheduler](u64 seed) { return scheduler->hash(seed); });
        }
    }

    u64 since_switch = 0;

    // See the idle-skip block below.
    u64 idle_skips_without_progress = 0;
    constexpr u64 max_idle_skips_without_progress = 1024;

    // The budget is checked here rather than in the loop condition so that a
    // fault or a halt on the very last budgeted instruction keeps its own
    // outcome: those are set after the instruction counter has been bumped,
    // and a trailing "did we run out of budget" test would relabel them.
    while (true) {
        if (report.instructions >= options.budget) {
            report.outcome = RunOutcome::BudgetExpired;
            break;
        }

        if (recording != nullptr) {
            recording->tick(state, &memory);
            if (recording->stop_requested()) {
                report.outcome = RunOutcome::Diverged;
                break;
            }
        }

        // Nothing can run, but something is waiting for the clock: a sleeping
        // thread or an alarm. Skip straight to it (**P2**) rather than
        // emulating the wait -- which, with every thread asleep, would mean
        // running no instructions at all and calling it a deadlock.
        //
        // This is checked before the sentinels below on purpose. A thread that
        // exits while every other thread is asleep leaves the core sitting on
        // the exit sentinel with nothing to switch to; skipping the clock here
        // is what gets a thread back, and doing it afterwards would spin on
        // the sentinel instead.
        if (scheduler != nullptr) {
            if (const u64 skip = scheduler->idle_skip_ticks(state); skip > 0) {
                interpreter.advance_time(skip);
                scheduler->note_skipped(skip);
                report.skipped_ticks += skip;
                static_cast<void>(scheduler->service_timers(state.timebase));
                static_cast<void>(scheduler->reschedule(state));

                // A skip that leaves nothing runnable has achieved nothing.
                // One or two of those is ordinary -- a periodic alarm with no
                // callback, a timer whose thread is waiting on something else
                // -- but a run of them means the deadline is the only thing
                // still moving and no amount of clock will help. Without this
                // the loop spins on the clock forever: it runs no
                // instructions, so the budget never advances either.
                if (scheduler->runnable_count() == 0) {
                    if (++idle_skips_without_progress >= max_idle_skips_without_progress) {
                        report.outcome = RunOutcome::Deadlocked;
                        break;
                    }
                } else {
                    idle_skips_without_progress = 0;
                }
                continue;
            }
        }

        // A thread's entry point returning lands here.
        if (scheduler != nullptr && state.pc == thread_exit_sentinel) {
            scheduler->exit_current(state, static_cast<s32>(state.gpr[3]));
            if (scheduler->all_exited()) {
                report.outcome = RunOutcome::Returned;
                break;
            }
            continue;
        }

        if (state.pc == return_sentinel) {
            if (scheduler == nullptr) {
                report.outcome = RunOutcome::Returned;
                break;
            }
            // The main thread finished. Others may still be running: a title
            // that spawns workers and returns from main is normal.
            scheduler->exit_current(state, static_cast<s32>(state.gpr[3]));
            if (scheduler->all_exited()) {
                report.outcome = RunOutcome::Returned;
                break;
            }
            continue;
        }

        if (scheduler != nullptr && scheduler->stalled()) {
            report.outcome = RunOutcome::Deadlocked;
            break;
        }

        const cpu::StepResult result = interpreter.step();
        ++report.instructions;
        report.last_result = result;

        if (result == cpu::StepResult::Ok) {
            // A thread spinning on a flag gives the core back now rather than
            // at the end of its timeslice (**P2**). It cannot make progress by
            // itself, and whatever it is waiting for is on another thread.
            const bool spinning = scheduler != nullptr && interpreter.spinning();
            if (spinning) {
                ++report.spin_yields;
            }

            if (scheduler != nullptr && (spinning || ++since_switch >= options.timeslice)) {
                since_switch = 0;
                interpreter.clear_spin();
                static_cast<void>(scheduler->reschedule(state));
            }
            continue;
        }

        if (result == cpu::StepResult::SystemCall) {
            // The trampoline is the instruction just executed.
            const u32 before = scheduler == nullptr ? 0 : scheduler->current_handle();
            if (registry.dispatch(state.pc - 4, state, memory)) {
                ++report.hle_calls;

                // Only a call that *did* switch starts the timeslice again.
                // Resetting on every call looks harmless and is not: a title's
                // main loop calls into the OS several times a frame, so the
                // counter never reached the timeslice, `reschedule` was never
                // called, and with it went every alarm, every deadline and
                // every other thread's turn. A busy title starved everything
                // else and nothing in the run loop said so.
                if (scheduler != nullptr && scheduler->current_handle() != before) {
                    since_switch = 0;
                }
                if (recording != nullptr && recording->stop_requested()) {
                    report.outcome = RunOutcome::Diverged;
                    break;
                }
                continue;
            }
            WIIU_LOG_WARN(Coreinit, "unrecognised system call at {:#010x}", state.pc - 4);
            report.outcome = RunOutcome::Faulted;
            break;
        }

        if (result == cpu::StepResult::Halted) {
            report.outcome = RunOutcome::Halted;
            break;
        }

        WIIU_LOG_ERROR(Cpu, "stopped at {:#010x}: {}", state.pc, cpu::step_result_name(result));
        report.outcome = RunOutcome::Faulted;
        break;
    }

    if (scheduler != nullptr) {
        report.context_switches = scheduler->switch_count();
    }
    if (recording != nullptr) {
        recording->finish(report.instructions);
        if (recording->stop_requested()) {
            report.outcome = RunOutcome::Diverged;
        }
    }
    return report;
}

} // namespace wiiu::hle
