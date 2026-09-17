// Phase 4.4 / 4.5 — the interpreter.
//
// Correctness first; speed is Phase 12's problem. This implementation is also
// the reference the JIT will be validated against in lockstep (P1 / 12.7), so
// it stays in the tree permanently and is never "replaced".
//
// Integer, branch, load/store, condition register and SPR instructions are
// implemented here. Floating point arrives in Phase 4.6, paired singles in
// 4.7; until then those report Unimplemented rather than quietly doing
// nothing, so a missing instruction is never mistaken for a wrong result.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/exceptions.hpp>
#include <wiiu/cpu/instruction.hpp>
#include <wiiu/cpu/mmu.hpp>
#include <wiiu/cpu/opcodes.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/memory/memory.hpp>

#include <string>

namespace wiiu::cpu {

/// Why `step` stopped. Phase 4.10 turns the failure cases into real exception
/// vectors; for now they surface to the caller, which is what makes them
/// testable.
enum class StepResult : u8 {
    Ok,
    Halted,             ///< the core is parked
    SystemCall,         ///< sc — the HLE layer takes over (Phase 6)
    Trap,               ///< tw/twi condition met
    MemoryFault,        ///< the access could not be satisfied
    IllegalInstruction, ///< nothing in the table matches
    Privileged,         ///< a supervisor instruction attempted in user mode
    FpException,        ///< an enabled floating-point exception fired
    Unimplemented,      ///< a known instruction this phase does not cover yet
};

[[nodiscard]] std::string_view step_result_name(StepResult result) noexcept;

/// What the interpreter does with a fault.
///
/// `Report` surfaces it as a StepResult and leaves the core where it was.
/// `Deliver` does what the hardware does: save SRR0/SRR1 and jump to the
/// vector.
///
/// Report is the default on purpose. Until Cafe OS has installed handlers
/// (Phase 6), delivering an exception means jumping into zeroed memory, which
/// turns a clear "memory fault at 0x70000000" into an illegal-instruction loop
/// somewhere else. The emulator switches to Deliver once there is something to
/// deliver to.
enum class ExceptionMode : u8 {
    Report,
    Deliver,
};

class Interpreter {
public:
    Interpreter(CoreState& state, memory::Memory& memory)
        : state_(state), memory_(memory), mmu_(state, memory) {}

    /// The MMU (Phase 3.4). Exposed because the debugger inspects
    /// translations, and because programming BATs or a page table is how a
    /// test turns translation on.
    [[nodiscard]] Mmu& mmu() noexcept { return mmu_; }

    [[nodiscard]] const Mmu& mmu() const noexcept { return mmu_; }

    void set_exception_mode(ExceptionMode mode) noexcept { exception_mode_ = mode; }

    [[nodiscard]] ExceptionMode exception_mode() const noexcept { return exception_mode_; }

    /// Enter an exception handler: save SRR0/SRR1, narrow MSR, jump to the
    /// vector. Public so the HLE layer and the debugger can inject one.
    void raise(Exception exception, u32 srr1_extra = 0);

    /// Pending interrupts, delivered at the start of the next instruction if
    /// MSR[EE] allows it. External interrupts come from devices; the
    /// decrementer one is raised by the timebase model below.
    void set_external_interrupt(bool pending) noexcept { external_pending_ = pending; }

    [[nodiscard]] bool external_interrupt_pending() const noexcept { return external_pending_; }

    [[nodiscard]] bool decrementer_pending() const noexcept { return decrementer_pending_; }

    // --- Phase 4.11: time -------------------------------------------------

    /// Timebase ticks charged per instruction. Real Espresso retires up to
    /// four instructions per cycle and the timebase runs at a quarter of the
    /// core clock, so one tick per instruction is a deliberate placeholder:
    /// monotonic and roughly proportional, not cycle-accurate. Games read the
    /// timebase for frame pacing, so it has to advance sensibly; nothing here
    /// pretends to model the pipeline.
    void set_ticks_per_instruction(u64 ticks) noexcept { ticks_per_instruction_ = ticks; }

    [[nodiscard]] u64 ticks_per_instruction() const noexcept { return ticks_per_instruction_; }

    /// Advance the timebase and count the decrementer down. A decrementer
    /// transition from non-negative to negative raises its interrupt.
    void advance_time(u64 ticks) noexcept;

    // --- idle-loop detection (P2) -----------------------------------------
    //
    // A guest thread waiting for another one writes the wait as a loop: load
    // the flag, compare, branch back. Emulating that faithfully means running
    // it for the whole timeslice -- hundreds of thousands of instructions
    // that cannot change anything, because the only thing that could is the
    // thread this one is keeping off the core.
    //
    // The shape is recognisable: a short backward branch, taken over and over,
    // whose body stores nothing and calls nothing. Storing or making a system
    // call means the loop is doing work, and the count starts again.

    /// Iterations of the same short backward branch, with nothing written in
    /// between, before the loop is called a spin. Low enough to give the time
    /// back quickly, high enough that an ordinary tight loop -- a memcpy, a
    /// checksum -- is never mistaken for one. Those store, so they reset it
    /// anyway; this is the belt to that pair of braces.
    static constexpr u32 spin_threshold = 200;

    [[nodiscard]] bool spinning() const noexcept { return spin_count_ >= spin_threshold; }

    /// Forget what has been seen so far. The run loop calls this after giving
    /// the time back, and on a context switch: the next thread's loops are
    /// its own.
    void clear_spin() noexcept {
        spin_target_ = 0;
        spin_count_ = 0;
        spin_dirty_ = false;
    }

    [[nodiscard]] u32 spin_iterations() const noexcept { return spin_count_; }

    /// Execute one instruction. On anything but Ok, `pc` is left pointing at
    /// the offending instruction so it can be reported and retried.
    StepResult step();

    /// Execute up to `budget` instructions, stopping early on anything but Ok.
    /// Returns the reason it stopped and how many instructions ran.
    struct RunResult {
        StepResult result = StepResult::Ok;
        u64 executed = 0;
    };

    RunResult run(u64 budget);

    [[nodiscard]] CoreState& state() noexcept { return state_; }

    [[nodiscard]] const CoreState& state() const noexcept { return state_; }

    /// The instruction that last failed, for diagnostics.
    [[nodiscard]] Instruction last_instruction() const noexcept {
        return Instruction{.raw = last_raw_};
    }

private:
    // --- flag helpers -----------------------------------------------------

    void update_cr0(u32 result) noexcept;
    void update_cr_field(u32 field, s32 lhs, s32 rhs) noexcept;
    void update_cr_field_unsigned(u32 field, u32 lhs, u32 rhs) noexcept;

    /// Sum with carry in and out, as addc/adde/subfc/subfe all need.
    [[nodiscard]] static u64 add64(u32 lhs, u32 rhs, u32 carry_in) noexcept {
        return static_cast<u64>(lhs) + rhs + carry_in;
    }

    void set_overflow_add(u32 lhs, u32 rhs, u32 result) noexcept;

    // --- execution --------------------------------------------------------

    StepResult execute(Opcode id, Instruction inst);

    StepResult execute_integer(Opcode id, Instruction inst);
    StepResult execute_logical(Opcode id, Instruction inst);
    StepResult execute_compare(Opcode id, Instruction inst);
    StepResult execute_rotate(Opcode id, Instruction inst);
    StepResult execute_branch(Opcode id, Instruction inst);
    StepResult execute_load_store(Opcode id, Instruction inst);
    StepResult execute_condition(Opcode id, Instruction inst);
    StepResult execute_system(Opcode id, Instruction inst);

    // Phase 4.6 / 4.7, in interpreter_fpu.cpp.
    StepResult execute_fp(Opcode id, Instruction inst);
    StepResult execute_fpscr(Opcode id, Instruction inst);
    StepResult execute_fp_load_store(Opcode id, Instruction inst);
    StepResult execute_paired(Opcode id, Instruction inst);
    StepResult execute_paired_load_store(Opcode id, Instruction inst);

    void update_cr1() noexcept;
    void compare_floats(u32 field, f64 lhs, f64 rhs, bool ordered) noexcept;

    /// Set exception bits, with FX and the summary bits following from them.
    /// Every path that raises an FP condition goes through here so that FX
    /// ("something became set") cannot drift from the bits it summarises.
    void set_fp_exception(u32 bits) noexcept;

    /// Record what an instruction produced: the class of the result in FPRF,
    /// whether it was inexact, and whatever the host FPU raised while
    /// computing it. `exact` is the value before single-precision rounding,
    /// which is what makes FR (rounded away from zero) knowable.
    void record_fp_result(f64 rounded, f64 exact, int raised) noexcept;

    /// Whether an enabled floating-point exception should interrupt. Cafe OS
    /// leaves MSR[FE0]/[FE1] clear, so this is normally dormant.
    [[nodiscard]] StepResult fp_program_check() const noexcept;

public:
    /// Quantisation used by psq_l / psq_st, exposed because it is worth
    /// testing on its own: the scale is a six-bit signed exponent and the
    /// integer formats clamp rather than wrap.
    [[nodiscard]] static f64 dequantize(u32 raw, QuantizeType type, s32 scale) noexcept;
    [[nodiscard]] static u32 quantize(f64 value, QuantizeType type, s32 scale) noexcept;

    /// The estimates behind `fres` / `ps_res` and `frsqrte` / `ps_rsqrte`.
    ///
    /// These are table lookups, not divisions: the 750CL keeps 32 entries of a
    /// base and a decrement, interpolates between them, and is specified only
    /// to within 1/256 (reciprocal) and 1/32 (reciprocal square root). Code
    /// that refines the estimate with Newton-Raphson depends on getting the
    /// hardware's answer rather than a better one, so an exact reciprocal here
    /// would converge to different low bits.
    [[nodiscard]] static f64 reciprocal_estimate(f64 value) noexcept;
    [[nodiscard]] static f64 reciprocal_sqrt_estimate(f64 value) noexcept;

private:
    /// Effective address for a D-form access: (rA|0) + displacement.
    /// rA == 0 means literal zero, not the contents of r0.
    [[nodiscard]] ppcaddr_t effective_address(Instruction inst) const noexcept {
        const u32 base = inst.ra == 0 ? 0 : state_.gpr[inst.ra];
        return base + static_cast<u32>(inst.d.value());
    }

    [[nodiscard]] ppcaddr_t effective_address_indexed(Instruction inst) const noexcept {
        const u32 base = inst.ra == 0 ? 0 : state_.gpr[inst.ra];
        return base + state_.gpr[inst.rb];
    }

    /// Turn a failed step into an exception, or report it, per the mode.
    StepResult handle_fault(StepResult result, Instruction inst);

    /// The BAT register an SPR number names, or nullptr for anything else.
    /// Upper and lower alternate: 528 is IBAT0U, 529 is IBAT0L, and the 750CL's
    /// second set picks up again at 560.
    [[nodiscard]] u32* bat_register(u32 number) noexcept;

    // --- guest memory -----------------------------------------------------
    //
    // Every data access goes through the MMU first. With translation off --
    // which is where the HLE loader leaves the core -- that is one test and
    // the effective address is used unchanged, so nothing is charged to the
    // common path.

    void
    note_translation_fault(ppcaddr_t address, bool is_write, TranslationFault reason) noexcept {
        translation_fault_ = {.address = address, .reason = reason, .is_write = is_write};
    }

    /// Does this access run off the end of a 4 KiB page? Two pages that are
    /// next to each other in the guest's address space are almost never next
    /// to each other once translated, so a straddling access has to be split.
    [[nodiscard]] static constexpr bool crosses_page(ppcaddr_t address, u32 size) noexcept {
        return ((address & 0xFFF) + size) > 0x1000;
    }

    template <typename T>
    [[nodiscard]] bool read_guest(ppcaddr_t address, T& out) {
        if constexpr (sizeof(T) > 1) {
            if (crosses_page(address, sizeof(T)) && mmu_.enabled(AccessKind::Read)) {
                return read_guest_split(address, out);
            }
        }

        const Translation translation = mmu_.translate(address, AccessKind::Read);
        if (!translation.ok()) {
            note_translation_fault(address, false, translation.fault);
            out = T{};
            return false;
        }
        return memory_.try_read<T>(translation.address, out);
    }

    template <typename T>
    [[nodiscard]] bool write_guest(ppcaddr_t address, T value) {
        if constexpr (sizeof(T) > 1) {
            if (crosses_page(address, sizeof(T)) && mmu_.enabled(AccessKind::Write)) {
                return write_guest_split(address, value);
            }
        }

        const Translation translation = mmu_.translate(address, AccessKind::Write);
        if (!translation.ok()) {
            note_translation_fault(address, true, translation.fault);
            return false;
        }
        return memory_.try_write<T>(translation.address, value);
    }

    /// The slow path for an access spanning two pages: one byte at a time, in
    /// guest order, so each byte is translated and permission-checked against
    /// the page it actually belongs to.
    template <typename T>
    [[nodiscard]] bool read_guest_split(ppcaddr_t address, T& out) {
        u64 value = 0;
        for (u32 offset = 0; offset < sizeof(T); ++offset) {
            u8 byte = 0;
            if (!read_guest<u8>(address + offset, byte)) {
                // DAR holds the address the instruction asked for, not the
                // byte the split happened to stop on -- otherwise a split and
                // an unsplit access report the same fault differently.
                note_translation_fault(address, false, translation_fault_.reason);
                out = T{};
                return false;
            }
            value = (value << 8) | byte;
        }
        out = static_cast<T>(value);
        return true;
    }

    template <typename T>
    [[nodiscard]] bool write_guest_split(ppcaddr_t address, T value) {
        // Check the far page before touching the near one. A store that faults
        // half way through has already changed memory the guest will see when
        // its handler returns, which is a far nastier thing to debug than one
        // that never started.
        const Translation far = mmu_.translate(address + sizeof(T) - 1, AccessKind::Write);
        if (!far.ok()) {
            note_translation_fault(address, true, far.fault);
            return false;
        }

        const auto bits = static_cast<u64>(value);
        for (u32 offset = 0; offset < sizeof(T); ++offset) {
            const u32 shift = 8 * (static_cast<u32>(sizeof(T)) - 1 - offset);
            if (!write_guest<u8>(address + offset, static_cast<u8>((bits >> shift) & 0xFF))) {
                return false;
            }
        }
        return true;
    }

    /// MSR[PR] selects which segment key applies, so a change to it retires
    /// every cached translation.
    void msr_changed(u32 previous) noexcept {
        if (((previous ^ state_.msr) & MsrBits::pr) != 0) {
            mmu_.invalidate_all();
        }
    }

    struct TranslationFaultInfo {
        ppcaddr_t address = 0;
        TranslationFault reason = TranslationFault::None;
        bool is_write = false;

        [[nodiscard]] bool active() const noexcept { return reason != TranslationFault::None; }
    };

    CoreState& state_;
    memory::Memory& memory_;
    Mmu mmu_;
    TranslationFaultInfo translation_fault_;

    /// Note that the current loop body did something a poll loop would not.
    void note_side_effect() noexcept { spin_dirty_ = true; }

    /// Count a taken backward branch towards the spin threshold.
    void note_backward_branch(ppcaddr_t target, ppcaddr_t from) noexcept {
        // Only a short loop: a backward branch spanning a whole function is
        // an ordinary loop over data, not a thread waiting for a flag.
        constexpr u32 longest_spin = 128;
        if (target >= from || from - target > longest_spin) {
            clear_spin();
            return;
        }
        if (target != spin_target_) {
            spin_target_ = target;
            spin_count_ = 0;
            spin_dirty_ = false;
            return;
        }
        if (spin_dirty_) {
            spin_count_ = 0;
            spin_dirty_ = false;
            return;
        }
        ++spin_count_;
    }

    ExceptionMode exception_mode_ = ExceptionMode::Report;
    bool external_pending_ = false;
    bool decrementer_pending_ = false;
    u64 ticks_per_instruction_ = 1;

    ppcaddr_t spin_target_ = 0;
    u32 spin_count_ = 0;
    bool spin_dirty_ = false;

    // Stored as the raw word rather than as an Instruction: BitField deletes
    // copy assignment on purpose, and that propagates to any union holding
    // one, so an Instruction member could never be reassigned.
    u32 last_raw_ = 0;
};

/// The 32-bit mask PowerPC's rotate instructions build from MB and ME.
///
/// MB > ME is legal and wraps: the mask is the complement of the run from
/// ME+1 to MB-1. Getting this wrong produces subtly wrong results in bitfield
/// code rather than an obvious failure, so it is exposed and tested directly.
[[nodiscard]] constexpr u32 rotate_mask(u32 mb, u32 me) noexcept {
    const u32 begin = 0xFFFFFFFFU >> mb;
    const u32 end = (me >= 31) ? 0xFFFFFFFFU : ~(0xFFFFFFFFU >> (me + 1));
    return (mb <= me) ? (begin & end) : (begin | end);
}

[[nodiscard]] constexpr u32 rotate_left(u32 value, u32 amount) noexcept {
    return amount == 0 ? value : ((value << amount) | (value >> (32 - amount)));
}

} // namespace wiiu::cpu
