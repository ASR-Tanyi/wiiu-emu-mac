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
    Interpreter(CoreState& state, memory::Memory& memory) : state_(state), memory_(memory) {}

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
    StepResult execute_fp_load_store(Opcode id, Instruction inst);
    StepResult execute_paired(Opcode id, Instruction inst);
    StepResult execute_paired_load_store(Opcode id, Instruction inst);

    void update_cr1() noexcept;
    void compare_floats(u32 field, f64 lhs, f64 rhs, bool ordered) noexcept;

public:
    /// Quantisation used by psq_l / psq_st, exposed because it is worth
    /// testing on its own: the scale is a six-bit signed exponent and the
    /// integer formats clamp rather than wrap.
    ///
    /// NOTE: `fres` and `frsqrte` are implemented as exact reciprocals. Real
    /// hardware uses a lookup table with defined imprecision (1/256 and 1/32
    /// relative error), so results differ in the low bits. That is fine for
    /// almost everything and wrong for code that depends on the exact
    /// estimate; the tables are a known gap to close against hardware.
    [[nodiscard]] static f64 dequantize(u32 raw, QuantizeType type, s32 scale) noexcept;
    [[nodiscard]] static u32 quantize(f64 value, QuantizeType type, s32 scale) noexcept;

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

    CoreState& state_;
    memory::Memory& memory_;

    ExceptionMode exception_mode_ = ExceptionMode::Report;
    bool external_pending_ = false;
    bool decrementer_pending_ = false;
    u64 ticks_per_instruction_ = 1;

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
