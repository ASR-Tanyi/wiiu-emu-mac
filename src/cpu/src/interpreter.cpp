#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>

#include <bit>

namespace wiiu::cpu {
namespace {

using wiiu::log::Channel;

/// Sign-extend the low byte / halfword, for lbz-family algebraic loads.
[[nodiscard]] constexpr u32 extend8(u32 value) noexcept {
    return static_cast<u32>(static_cast<s32>(static_cast<s8>(value & 0xFF)));
}

[[nodiscard]] constexpr u32 extend16(u32 value) noexcept {
    return static_cast<u32>(static_cast<s32>(static_cast<s16>(value & 0xFFFF)));
}

} // namespace

std::string_view step_result_name(StepResult result) noexcept {
    switch (result) {
        case StepResult::Ok:
            return "ok";
        case StepResult::Halted:
            return "halted";
        case StepResult::SystemCall:
            return "system call";
        case StepResult::Trap:
            return "trap";
        case StepResult::MemoryFault:
            return "memory fault";
        case StepResult::IllegalInstruction:
            return "illegal instruction";
        case StepResult::Unimplemented:
            return "unimplemented";
    }
    return "?";
}

// --------------------------------------------------------------------------
// Flags
// --------------------------------------------------------------------------

void Interpreter::update_cr0(u32 result) noexcept {
    const auto signed_result = static_cast<s32>(result);

    u32 bits = 0;
    if (signed_result < 0) {
        bits |= ConditionRegister::lt;
    } else if (signed_result > 0) {
        bits |= ConditionRegister::gt;
    } else {
        bits |= ConditionRegister::eq;
    }
    // CR0's summary-overflow bit is a copy of XER[SO], not a fresh
    // computation: it reflects the sticky bit, so it stays set until software
    // clears XER.
    if (state_.xer.summary_overflow()) {
        bits |= ConditionRegister::so;
    }
    state_.cr.set_field(0, bits);
}

void Interpreter::update_cr_field(u32 field, s32 lhs, s32 rhs) noexcept {
    u32 bits = 0;
    if (lhs < rhs) {
        bits |= ConditionRegister::lt;
    } else if (lhs > rhs) {
        bits |= ConditionRegister::gt;
    } else {
        bits |= ConditionRegister::eq;
    }
    if (state_.xer.summary_overflow()) {
        bits |= ConditionRegister::so;
    }
    state_.cr.set_field(field, bits);
}

void Interpreter::update_cr_field_unsigned(u32 field, u32 lhs, u32 rhs) noexcept {
    u32 bits = 0;
    if (lhs < rhs) {
        bits |= ConditionRegister::lt;
    } else if (lhs > rhs) {
        bits |= ConditionRegister::gt;
    } else {
        bits |= ConditionRegister::eq;
    }
    if (state_.xer.summary_overflow()) {
        bits |= ConditionRegister::so;
    }
    state_.cr.set_field(field, bits);
}

void Interpreter::set_overflow_add(u32 lhs, u32 rhs, u32 result) noexcept {
    // Signed overflow: both operands share a sign and the result differs.
    const bool overflow = (~(lhs ^ rhs) & (lhs ^ result) & 0x80000000U) != 0;
    state_.xer.set_overflow(overflow);
}

// --------------------------------------------------------------------------
// Exceptions and time
// --------------------------------------------------------------------------

void Interpreter::raise(Exception exception, u32 srr1_extra) {
    // SRR0 holds the instruction to come back to. For a fault that is the
    // instruction that faulted, so it can be retried once the handler has
    // fixed whatever was wrong.
    state_.srr0 = state_.pc;

    // SRR1 carries the old MSR plus, for a program exception, the reason.
    state_.srr1 = (state_.msr & 0x87C0FF73U) | srr1_extra;

    state_.msr &= ~msr_exception_clear_mask;

    state_.pc = exception_base(state_.msr) + static_cast<u32>(exception);
    state_.next_pc = state_.pc;

    WIIU_LOG_DEBUG(Cpu,
                   "{} exception -> {:#010x} (srr0 {:#010x})",
                   exception_name(exception),
                   state_.pc,
                   state_.srr0);
}

void Interpreter::advance_time(u64 ticks) noexcept {
    state_.timebase += ticks;

    // The decrementer counts down at the timebase rate and signals when it
    // rolls past zero -- that is, when bit 31 goes from 0 to 1. Testing for
    // "reached zero" would miss every case where the countdown steps over it.
    const u32 before = state_.decrementer;
    const u32 after = before - static_cast<u32>(ticks);
    state_.decrementer = after;

    if ((before & 0x80000000U) == 0 && (after & 0x80000000U) != 0) {
        decrementer_pending_ = true;
    }
}

StepResult Interpreter::handle_fault(StepResult result, Instruction inst) {
    if (exception_mode_ == ExceptionMode::Report) {
        return result;
    }

    switch (result) {
        case StepResult::MemoryFault: {
            const memory::Fault& fault = memory_.last_fault();
            state_.dar = fault.address;
            state_.dsisr = (fault.unmapped ? DsisrBits::not_translated : DsisrBits::protection) |
                           (fault.is_write ? DsisrBits::is_store : 0U);
            raise(Exception::Dsi);
            return StepResult::Ok;
        }

        case StepResult::IllegalInstruction:
            raise(Exception::Program, ProgramCause::illegal_instruction);
            return StepResult::Ok;

        case StepResult::Trap:
            raise(Exception::Program, ProgramCause::trap);
            return StepResult::Ok;

        case StepResult::Unimplemented:
            // Not an architectural condition -- it means this emulator has a
            // gap. Delivering a program exception would disguise that as guest
            // misbehaviour, so it always surfaces.
            WIIU_LOG_WARN(
                Cpu, "unimplemented instruction {:#010x} at {:#010x}", inst.raw, state_.pc);
            return result;

        default:
            return result;
    }
}

// --------------------------------------------------------------------------
// Stepping
// --------------------------------------------------------------------------

StepResult Interpreter::step() {
    if (state_.halted) {
        return StepResult::Halted;
    }

    // Asynchronous interrupts are checked before the instruction, and only
    // when the guest has them enabled. A pending interrupt with MSR[EE] clear
    // stays pending rather than being lost.
    if (exception_mode_ == ExceptionMode::Deliver && (state_.msr & MsrBits::ee) != 0) {
        if (external_pending_) {
            external_pending_ = false;
            raise(Exception::ExternalInterrupt);
            return StepResult::Ok;
        }
        if (decrementer_pending_) {
            decrementer_pending_ = false;
            raise(Exception::Decrementer);
            return StepResult::Ok;
        }
    }

    u32 word = 0;
    if (!memory_.try_read<u32>(state_.pc, word)) {
        // A failed fetch is an instruction storage exception, not a data one.
        if (exception_mode_ == ExceptionMode::Deliver) {
            raise(Exception::Isi);
            return StepResult::Ok;
        }
        return StepResult::MemoryFault;
    }

    const Instruction inst{.raw = word};
    last_raw_ = word;

    const Opcode id = decode(inst);
    if (id == Opcode::Invalid) {
        return handle_fault(StepResult::IllegalInstruction, inst);
    }

    // Default to falling through to the next instruction. Branches overwrite
    // next_pc; everything else leaves it alone. Keeping it in state rather
    // than as a local is what lets Phase 4.10's exceptions redirect cleanly.
    state_.next_pc = state_.pc + 4;

    const StepResult result = execute(id, inst);
    if (result == StepResult::Ok) {
        state_.pc = state_.next_pc;
        advance_time(ticks_per_instruction_);
        return result;
    }

    return handle_fault(result, inst);
}

Interpreter::RunResult Interpreter::run(u64 budget) {
    RunResult out;
    for (; out.executed < budget; ++out.executed) {
        const StepResult result = step();
        if (result != StepResult::Ok) {
            out.result = result;
            return out;
        }
    }
    return out;
}

StepResult Interpreter::execute(Opcode id, Instruction inst) {
    switch (id) {
        case Opcode::Add:
        case Opcode::Addc:
        case Opcode::Adde:
        case Opcode::Addme:
        case Opcode::Addze:
        case Opcode::Divw:
        case Opcode::Divwu:
        case Opcode::Mulhw:
        case Opcode::Mulhwu:
        case Opcode::Mullw:
        case Opcode::Neg:
        case Opcode::Subf:
        case Opcode::Subfc:
        case Opcode::Subfe:
        case Opcode::Subfme:
        case Opcode::Subfze:
        case Opcode::Addi:
        case Opcode::Addic:
        case Opcode::AddicRc:
        case Opcode::Addis:
        case Opcode::Mulli:
        case Opcode::Subfic:
            return execute_integer(id, inst);

        case Opcode::And:
        case Opcode::Andc:
        case Opcode::Eqv:
        case Opcode::Nand:
        case Opcode::Nor:
        case Opcode::Or:
        case Opcode::Orc:
        case Opcode::Xor:
        case Opcode::Slw:
        case Opcode::Sraw:
        case Opcode::Srawi:
        case Opcode::Srw:
        case Opcode::Cntlzw:
        case Opcode::Extsb:
        case Opcode::Extsh:
        case Opcode::Andi:
        case Opcode::Andis:
        case Opcode::Ori:
        case Opcode::Oris:
        case Opcode::Xori:
        case Opcode::Xoris:
            return execute_logical(id, inst);

        case Opcode::Cmp:
        case Opcode::Cmpl:
        case Opcode::Cmpi:
        case Opcode::Cmpli:
            return execute_compare(id, inst);

        case Opcode::Rlwimi:
        case Opcode::Rlwinm:
        case Opcode::Rlwnm:
            return execute_rotate(id, inst);

        case Opcode::B:
        case Opcode::Bc:
        case Opcode::Bclr:
        case Opcode::Bcctr:
            return execute_branch(id, inst);

        case Opcode::Crand:
        case Opcode::Crandc:
        case Opcode::Creqv:
        case Opcode::Crnand:
        case Opcode::Crnor:
        case Opcode::Cror:
        case Opcode::Crorc:
        case Opcode::Crxor:
        case Opcode::Mcrf:
            return execute_condition(id, inst);

        case Opcode::Mfspr:
        case Opcode::Mtspr:
        case Opcode::Mftb:
        case Opcode::Mfcr:
        case Opcode::Mtcrf:
        case Opcode::Mfmsr:
        case Opcode::Mtmsr:
        case Opcode::Sc:
        case Opcode::Rfi:
        case Opcode::Tw:
        case Opcode::Twi:
        case Opcode::Sync:
        case Opcode::Isync:
        case Opcode::Eieio:
        case Opcode::Dcbf:
        case Opcode::Dcbi:
        case Opcode::Dcbst:
        case Opcode::Dcbt:
        case Opcode::Dcbtst:
        case Opcode::Dcbz:
        case Opcode::Icbi:
            return execute_system(id, inst);

        case Opcode::Lfs:
        case Opcode::Lfsu:
        case Opcode::Lfsx:
        case Opcode::Lfd:
        case Opcode::Lfdu:
        case Opcode::Lfdx:
        case Opcode::Stfs:
        case Opcode::Stfsu:
        case Opcode::Stfsx:
        case Opcode::Stfd:
        case Opcode::Stfdu:
        case Opcode::Stfdx:
            return execute_fp_load_store(id, inst);

        case Opcode::PsqL:
        case Opcode::PsqLu:
        case Opcode::PsqSt:
        case Opcode::PsqStu:
            return execute_paired_load_store(id, inst);

        case Opcode::Fadd:
        case Opcode::Fadds:
        case Opcode::Fsub:
        case Opcode::Fsubs:
        case Opcode::Fmul:
        case Opcode::Fmuls:
        case Opcode::Fdiv:
        case Opcode::Fdivs:
        case Opcode::Fmadd:
        case Opcode::Fmadds:
        case Opcode::Fmsub:
        case Opcode::Fmsubs:
        case Opcode::Fnmadd:
        case Opcode::Fnmadds:
        case Opcode::Fnmsub:
        case Opcode::Fnmsubs:
        case Opcode::Fsel:
        case Opcode::Fres:
        case Opcode::Frsqrte:
        case Opcode::Fmr:
        case Opcode::Fneg:
        case Opcode::Fabs:
        case Opcode::Fnabs:
        case Opcode::Frsp:
        case Opcode::Fctiw:
        case Opcode::Fctiwz:
        case Opcode::Fcmpu:
        case Opcode::Fcmpo:
        case Opcode::Mffs:
        case Opcode::Mtfsf:
        case Opcode::Mcrfs:
            return execute_fp(id, inst);

        case Opcode::PsAdd:
        case Opcode::PsSub:
        case Opcode::PsMul:
        case Opcode::PsDiv:
        case Opcode::PsMadd:
        case Opcode::PsMsub:
        case Opcode::PsNmadd:
        case Opcode::PsNmsub:
        case Opcode::PsSel:
        case Opcode::PsRes:
        case Opcode::PsRsqrte:
        case Opcode::PsMuls0:
        case Opcode::PsMuls1:
        case Opcode::PsMadds0:
        case Opcode::PsMadds1:
        case Opcode::PsMr:
        case Opcode::PsNeg:
        case Opcode::PsAbs:
        case Opcode::PsNabs:
        case Opcode::PsMerge00:
        case Opcode::PsMerge01:
        case Opcode::PsMerge10:
        case Opcode::PsMerge11:
            return execute_paired(id, inst);

        default:
            return execute_load_store(id, inst);
    }
}

// --------------------------------------------------------------------------
// Integer
// --------------------------------------------------------------------------

StepResult Interpreter::execute_integer(Opcode id, Instruction inst) {
    const u32 a = state_.gpr[inst.ra];
    const u32 b = state_.gpr[inst.rb];
    u32 result = 0;

    switch (id) {
        case Opcode::Add:
            result = a + b;
            if (inst.oe) {
                set_overflow_add(a, b, result);
            }
            break;

        case Opcode::Addc: {
            const u64 wide = add64(a, b, 0);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(a, b, result);
            }
            break;
        }

        case Opcode::Adde: {
            const u64 wide = add64(a, b, state_.xer.carry() ? 1U : 0U);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(a, b, result);
            }
            break;
        }

        case Opcode::Addme: {
            const u64 wide = add64(a, 0xFFFFFFFFU, state_.xer.carry() ? 1U : 0U);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(a, 0xFFFFFFFFU, result);
            }
            break;
        }

        case Opcode::Addze: {
            const u64 wide = add64(a, 0, state_.xer.carry() ? 1U : 0U);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(a, 0, result);
            }
            break;
        }

        // subf computes rB - rA, not rA - rB. The operand order in the
        // mnemonic is the opposite of what it reads like, and getting it
        // backwards is a classic PowerPC mistake.
        case Opcode::Subf:
            result = b - a;
            if (inst.oe) {
                set_overflow_add(~a, b, result);
            }
            break;

        case Opcode::Subfc: {
            const u64 wide = add64(~a, b, 1);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(~a, b, result);
            }
            break;
        }

        case Opcode::Subfe: {
            const u64 wide = add64(~a, b, state_.xer.carry() ? 1U : 0U);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(~a, b, result);
            }
            break;
        }

        case Opcode::Subfme: {
            const u64 wide = add64(~a, 0xFFFFFFFFU, state_.xer.carry() ? 1U : 0U);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(~a, 0xFFFFFFFFU, result);
            }
            break;
        }

        case Opcode::Subfze: {
            const u64 wide = add64(~a, 0, state_.xer.carry() ? 1U : 0U);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            if (inst.oe) {
                set_overflow_add(~a, 0, result);
            }
            break;
        }

        case Opcode::Neg:
            result = ~a + 1;
            if (inst.oe) {
                // The single overflow case: negating the most negative value.
                state_.xer.set_overflow(a == 0x80000000U);
            }
            break;

        case Opcode::Mullw:
            result = static_cast<u32>(static_cast<s64>(static_cast<s32>(a)) * static_cast<s32>(b));
            if (inst.oe) {
                const s64 wide = static_cast<s64>(static_cast<s32>(a)) * static_cast<s32>(b);
                state_.xer.set_overflow(wide != static_cast<s64>(static_cast<s32>(result)));
            }
            break;

        case Opcode::Mulhw:
            result = static_cast<u32>(
                (static_cast<s64>(static_cast<s32>(a)) * static_cast<s32>(b)) >> 32);
            break;

        case Opcode::Mulhwu:
            result = static_cast<u32>((static_cast<u64>(a) * b) >> 32);
            break;

        case Opcode::Divw: {
            const auto dividend = static_cast<s32>(a);
            const auto divisor = static_cast<s32>(b);
            // Division by zero and INT_MIN/-1 both leave rD undefined on
            // hardware and set OV. Producing a stable value keeps the JIT
            // comparison in Phase 12.7 meaningful.
            if (divisor == 0 || (dividend == INT32_MIN && divisor == -1)) {
                result = 0;
                if (inst.oe) {
                    state_.xer.set_overflow(true);
                }
            } else {
                result = static_cast<u32>(dividend / divisor);
                if (inst.oe) {
                    state_.xer.set_overflow(false);
                }
            }
            break;
        }

        case Opcode::Divwu:
            if (b == 0) {
                result = 0;
                if (inst.oe) {
                    state_.xer.set_overflow(true);
                }
            } else {
                result = a / b;
                if (inst.oe) {
                    state_.xer.set_overflow(false);
                }
            }
            break;

        // D-form. rA == 0 means literal zero for addi/addis only.
        case Opcode::Addi:
            result = (inst.ra == 0 ? 0 : a) + static_cast<u32>(inst.simm.value());
            state_.gpr[inst.rd] = result;
            return StepResult::Ok;

        case Opcode::Addis:
            result = (inst.ra == 0 ? 0 : a) + (static_cast<u32>(inst.simm.value()) << 16);
            state_.gpr[inst.rd] = result;
            return StepResult::Ok;

        case Opcode::Addic:
        case Opcode::AddicRc: {
            const u64 wide = add64(a, static_cast<u32>(inst.simm.value()), 0);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            state_.gpr[inst.rd] = result;
            if (id == Opcode::AddicRc) {
                update_cr0(result);
            }
            return StepResult::Ok;
        }

        case Opcode::Subfic: {
            const u64 wide = add64(~a, static_cast<u32>(inst.simm.value()), 1);
            result = static_cast<u32>(wide);
            state_.xer.set_carry(wide > 0xFFFFFFFFULL);
            state_.gpr[inst.rd] = result;
            return StepResult::Ok;
        }

        case Opcode::Mulli:
            result = static_cast<u32>(static_cast<s32>(a) * inst.simm.value());
            state_.gpr[inst.rd] = result;
            return StepResult::Ok;

        default:
            return StepResult::Unimplemented;
    }

    state_.gpr[inst.rd] = result;
    if (inst.rc) {
        update_cr0(result);
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// Logical and shifts
// --------------------------------------------------------------------------

StepResult Interpreter::execute_logical(Opcode id, Instruction inst) {
    const u32 s = state_.gpr[inst.rs];
    const u32 b = state_.gpr[inst.rb];
    u32 result = 0;
    bool record = inst.rc;

    switch (id) {
        case Opcode::And:
            result = s & b;
            break;
        case Opcode::Andc:
            result = s & ~b;
            break;
        case Opcode::Eqv:
            result = ~(s ^ b);
            break;
        case Opcode::Nand:
            result = ~(s & b);
            break;
        case Opcode::Nor:
            result = ~(s | b);
            break;
        case Opcode::Or:
            result = s | b;
            break;
        case Opcode::Orc:
            result = s | ~b;
            break;
        case Opcode::Xor:
            result = s ^ b;
            break;

        // The shift amount is six bits: a count of 32 or more shifts the
        // register out entirely rather than wrapping, which is why this is not
        // a bare C shift.
        case Opcode::Slw: {
            const u32 amount = b & 0x3F;
            result = amount >= 32 ? 0 : (s << amount);
            break;
        }
        case Opcode::Srw: {
            const u32 amount = b & 0x3F;
            result = amount >= 32 ? 0 : (s >> amount);
            break;
        }

        case Opcode::Sraw: {
            const u32 amount = b & 0x3F;
            const auto value = static_cast<s32>(s);
            if (amount >= 32) {
                result = value < 0 ? 0xFFFFFFFFU : 0;
                state_.xer.set_carry(value < 0);
            } else {
                result = static_cast<u32>(value >> amount);
                // Carry is set when a negative value loses any one bits.
                const u32 lost = s & ((1U << amount) - 1);
                state_.xer.set_carry(value < 0 && lost != 0);
            }
            break;
        }

        case Opcode::Srawi: {
            const u32 amount = inst.sh;
            const auto value = static_cast<s32>(s);
            result = static_cast<u32>(value >> amount);
            const u32 lost = amount == 0 ? 0 : (s & ((1U << amount) - 1));
            state_.xer.set_carry(value < 0 && lost != 0);
            break;
        }

        case Opcode::Cntlzw:
            result = static_cast<u32>(std::countl_zero(s));
            break;
        case Opcode::Extsb:
            result = extend8(s);
            break;
        case Opcode::Extsh:
            result = extend16(s);
            break;

        // D-form logical operations take an unsigned immediate and never
        // treat rA as zero.
        case Opcode::Andi:
            result = s & inst.uimm.value();
            record = true;
            break;
        case Opcode::Andis:
            result = s & (inst.uimm.value() << 16);
            record = true;
            break;
        case Opcode::Ori:
            result = s | inst.uimm.value();
            break;
        case Opcode::Oris:
            result = s | (inst.uimm.value() << 16);
            break;
        case Opcode::Xori:
            result = s ^ inst.uimm.value();
            break;
        case Opcode::Xoris:
            result = s ^ (inst.uimm.value() << 16);
            break;

        default:
            return StepResult::Unimplemented;
    }

    state_.gpr[inst.ra] = result;
    if (record) {
        update_cr0(result);
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// Compare
// --------------------------------------------------------------------------

StepResult Interpreter::execute_compare(Opcode id, Instruction inst) {
    const u32 a = state_.gpr[inst.ra];

    switch (id) {
        case Opcode::Cmp:
            update_cr_field(inst.crfd, static_cast<s32>(a), static_cast<s32>(state_.gpr[inst.rb]));
            break;
        case Opcode::Cmpl:
            update_cr_field_unsigned(inst.crfd, a, state_.gpr[inst.rb]);
            break;
        case Opcode::Cmpi:
            update_cr_field(inst.crfd, static_cast<s32>(a), inst.simm.value());
            break;
        case Opcode::Cmpli:
            update_cr_field_unsigned(inst.crfd, a, inst.uimm.value());
            break;
        default:
            return StepResult::Unimplemented;
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// Rotate
// --------------------------------------------------------------------------

StepResult Interpreter::execute_rotate(Opcode id, Instruction inst) {
    const u32 s = state_.gpr[inst.rs];
    const u32 mask = rotate_mask(inst.mb, inst.me);

    u32 result = 0;
    switch (id) {
        case Opcode::Rlwinm:
            result = rotate_left(s, inst.sh) & mask;
            break;
        case Opcode::Rlwnm:
            result = rotate_left(s, state_.gpr[inst.rb] & 0x1F) & mask;
            break;
        case Opcode::Rlwimi:
            // Insert: the bits outside the mask keep their old value.
            result = (rotate_left(s, inst.sh) & mask) | (state_.gpr[inst.ra] & ~mask);
            break;
        default:
            return StepResult::Unimplemented;
    }

    state_.gpr[inst.ra] = result;
    if (inst.rc) {
        update_cr0(result);
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// Branch
// --------------------------------------------------------------------------

StepResult Interpreter::execute_branch(Opcode id, Instruction inst) {
    const ppcaddr_t link_target = state_.pc + 4;

    if (id == Opcode::B) {
        const auto displacement = static_cast<u32>(inst.li.value() * 4);
        state_.next_pc = inst.aa ? displacement : state_.pc + displacement;
        if (inst.lk) {
            state_.lr = link_target;
        }
        return StepResult::Ok;
    }

    // BO is five bits, numbered from the most significant end.
    const u32 bo = inst.bo;
    const bool bo0 = ((bo >> 4) & 1) != 0; // ignore the condition
    const bool bo1 = ((bo >> 3) & 1) != 0; // condition value to branch on
    const bool bo2 = ((bo >> 2) & 1) != 0; // do not decrement CTR
    const bool bo3 = ((bo >> 1) & 1) != 0; // branch on CTR == 0 rather than != 0

    if (!bo2) {
        --state_.ctr;
    }

    const bool ctr_ok = bo2 || ((state_.ctr != 0) != bo3);
    const bool cond_ok = bo0 || (state_.cr.bit(inst.bi) == bo1);
    const bool taken = ctr_ok && cond_ok;

    if (taken) {
        switch (id) {
            case Opcode::Bc: {
                const auto displacement = static_cast<u32>(inst.bd.value() * 4);
                state_.next_pc = inst.aa ? displacement : state_.pc + displacement;
                break;
            }
            // The low two bits of LR and CTR are ignored, not an error.
            case Opcode::Bclr:
                state_.next_pc = state_.lr & ~3U;
                break;
            case Opcode::Bcctr:
                state_.next_pc = state_.ctr & ~3U;
                break;
            default:
                return StepResult::Unimplemented;
        }
    }

    // LR is written whether or not the branch is taken.
    if (inst.lk) {
        state_.lr = link_target;
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// Condition register
// --------------------------------------------------------------------------

StepResult Interpreter::execute_condition(Opcode id, Instruction inst) {
    if (id == Opcode::Mcrf) {
        state_.cr.set_field(inst.crfd, state_.cr.field(inst.crfs));
        return StepResult::Ok;
    }

    const bool a = state_.cr.bit(inst.crba);
    const bool b = state_.cr.bit(inst.crbb);
    bool result = false;

    switch (id) {
        case Opcode::Crand:
            result = a && b;
            break;
        case Opcode::Crandc:
            result = a && !b;
            break;
        case Opcode::Creqv:
            result = a == b;
            break;
        case Opcode::Crnand:
            result = !(a && b);
            break;
        case Opcode::Crnor:
            result = !(a || b);
            break;
        case Opcode::Cror:
            result = a || b;
            break;
        case Opcode::Crorc:
            result = a || !b;
            break;
        case Opcode::Crxor:
            result = a != b;
            break;
        default:
            return StepResult::Unimplemented;
    }

    state_.cr.set_bit(inst.crbd, result);
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// Loads and stores
// --------------------------------------------------------------------------

StepResult Interpreter::execute_load_store(Opcode id, Instruction inst) {
    const bool indexed = [id] {
        switch (id) {
            case Opcode::Lbzx:
            case Opcode::Lbzux:
            case Opcode::Lhax:
            case Opcode::Lhzx:
            case Opcode::Lhzux:
            case Opcode::Lwzx:
            case Opcode::Lwzux:
            case Opcode::Stbx:
            case Opcode::Sthx:
            case Opcode::Stwx:
            case Opcode::Stwux:
            case Opcode::Lhbrx:
            case Opcode::Lwbrx:
            case Opcode::Sthbrx:
            case Opcode::Stwbrx:
            case Opcode::Lwarx:
            case Opcode::Stwcx:
                return true;
            default:
                return false;
        }
    }();

    const ppcaddr_t address = indexed ? effective_address_indexed(inst) : effective_address(inst);

    const bool update = [id] {
        switch (id) {
            case Opcode::Lbzu:
            case Opcode::Lhau:
            case Opcode::Lhzu:
            case Opcode::Lwzu:
            case Opcode::Stbu:
            case Opcode::Sthu:
            case Opcode::Stwu:
            case Opcode::Lbzux:
            case Opcode::Lhzux:
            case Opcode::Lwzux:
            case Opcode::Stwux:
                return true;
            default:
                return false;
        }
    }();

    u32 value = 0;
    bool is_load = true;

    switch (id) {
        case Opcode::Lbz:
        case Opcode::Lbzu:
        case Opcode::Lbzx:
        case Opcode::Lbzux: {
            u8 byte = 0;
            if (!memory_.try_read<u8>(address, byte)) {
                return StepResult::MemoryFault;
            }
            value = byte;
            break;
        }

        case Opcode::Lhz:
        case Opcode::Lhzu:
        case Opcode::Lhzx:
        case Opcode::Lhzux: {
            u16 half = 0;
            if (!memory_.try_read<u16>(address, half)) {
                return StepResult::MemoryFault;
            }
            value = half;
            break;
        }

        case Opcode::Lha:
        case Opcode::Lhau:
        case Opcode::Lhax: {
            u16 half = 0;
            if (!memory_.try_read<u16>(address, half)) {
                return StepResult::MemoryFault;
            }
            value = extend16(half);
            break;
        }

        case Opcode::Lwz:
        case Opcode::Lwzu:
        case Opcode::Lwzx:
        case Opcode::Lwzux:
            if (!memory_.try_read<u32>(address, value)) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Lwarx:
            if (!memory_.try_read<u32>(address, value)) {
                return StepResult::MemoryFault;
            }
            state_.reserved = true;
            state_.reservation_address = address;
            break;

        // Byte-reversed loads: the guest is big-endian, so these read a
        // little-endian value out of guest memory.
        case Opcode::Lhbrx: {
            u16 half = 0;
            if (!memory_.try_read<u16>(address, half)) {
                return StepResult::MemoryFault;
            }
            value = swap_bytes(half);
            break;
        }
        case Opcode::Lwbrx: {
            u32 word = 0;
            if (!memory_.try_read<u32>(address, word)) {
                return StepResult::MemoryFault;
            }
            value = swap_bytes(word);
            break;
        }

        case Opcode::Lmw: {
            // Loads rD through r31 from consecutive words.
            ppcaddr_t cursor = address;
            for (u32 reg = inst.rd; reg < 32; ++reg, cursor += 4) {
                u32 word = 0;
                if (!memory_.try_read<u32>(cursor, word)) {
                    return StepResult::MemoryFault;
                }
                state_.gpr[reg] = word;
            }
            return StepResult::Ok;
        }

        case Opcode::Stmw: {
            ppcaddr_t cursor = address;
            for (u32 reg = inst.rs; reg < 32; ++reg, cursor += 4) {
                if (!memory_.try_write<u32>(cursor, state_.gpr[reg])) {
                    return StepResult::MemoryFault;
                }
            }
            return StepResult::Ok;
        }

        case Opcode::Stb:
        case Opcode::Stbu:
        case Opcode::Stbx:
            is_load = false;
            if (!memory_.try_write<u8>(address, static_cast<u8>(state_.gpr[inst.rs]))) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Sth:
        case Opcode::Sthu:
        case Opcode::Sthx:
            is_load = false;
            if (!memory_.try_write<u16>(address, static_cast<u16>(state_.gpr[inst.rs]))) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Stw:
        case Opcode::Stwu:
        case Opcode::Stwx:
        case Opcode::Stwux:
            is_load = false;
            if (!memory_.try_write<u32>(address, state_.gpr[inst.rs])) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Sthbrx:
            is_load = false;
            if (!memory_.try_write<u16>(address,
                                        swap_bytes(static_cast<u16>(state_.gpr[inst.rs])))) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Stwbrx:
            is_load = false;
            if (!memory_.try_write<u32>(address, swap_bytes(state_.gpr[inst.rs]))) {
                return StepResult::MemoryFault;
            }
            break;

        case Opcode::Stwcx: {
            is_load = false;
            // A conditional store succeeds only if this core still holds its
            // reservation. Espresso's real implementation is buggy and Cafe OS
            // works around it; this is the documented behaviour.
            const bool held = state_.reserved && state_.reservation_address == address;
            if (held && !memory_.try_write<u32>(address, state_.gpr[inst.rs])) {
                return StepResult::MemoryFault;
            }
            state_.reserved = false;

            u32 bits = held ? ConditionRegister::eq : 0;
            if (state_.xer.summary_overflow()) {
                bits |= ConditionRegister::so;
            }
            state_.cr.set_field(0, bits);
            return StepResult::Ok;
        }

        default:
            return StepResult::Unimplemented;
    }

    if (is_load) {
        state_.gpr[inst.rd] = value;
    }
    if (update && inst.ra != 0) {
        state_.gpr[inst.ra] = address;
    }
    return StepResult::Ok;
}

// --------------------------------------------------------------------------
// System, cache and special registers
// --------------------------------------------------------------------------

StepResult Interpreter::execute_system(Opcode id, Instruction inst) {
    switch (id) {
        case Opcode::Mfspr: {
            const u32 number = inst.spr_number();
            switch (static_cast<Spr>(number)) {
                case Spr::Xer:
                    state_.gpr[inst.rd] = state_.xer.value;
                    break;
                case Spr::Lr:
                    state_.gpr[inst.rd] = state_.lr;
                    break;
                case Spr::Ctr:
                    state_.gpr[inst.rd] = state_.ctr;
                    break;
                case Spr::Srr0:
                    state_.gpr[inst.rd] = state_.srr0;
                    break;
                case Spr::Srr1:
                    state_.gpr[inst.rd] = state_.srr1;
                    break;
                case Spr::Dec:
                    state_.gpr[inst.rd] = state_.decrementer;
                    break;
                case Spr::Tbl:
                    state_.gpr[inst.rd] = static_cast<u32>(state_.timebase);
                    break;
                case Spr::Tbu:
                    state_.gpr[inst.rd] = static_cast<u32>(state_.timebase >> 32);
                    break;
                case Spr::Hid0:
                    state_.gpr[inst.rd] = state_.hid0;
                    break;
                case Spr::Hid1:
                    state_.gpr[inst.rd] = state_.hid1;
                    break;
                case Spr::Hid2:
                    state_.gpr[inst.rd] = state_.hid2;
                    break;
                case Spr::Scr:
                    state_.gpr[inst.rd] = state_.scr;
                    break;
                case Spr::Wpar:
                    state_.gpr[inst.rd] = state_.wpar;
                    break;
                default:
                    if (number >= 272 && number <= 275) {
                        state_.gpr[inst.rd] = state_.sprg[number - 272];
                    } else if (number >= 912 && number <= 919) {
                        state_.gpr[inst.rd] = state_.gqr[number - 912].value;
                    } else {
                        WIIU_LOG_WARN(
                            Cpu, "mfspr from unmodelled SPR {} at {:#010x}", number, state_.pc);
                        state_.gpr[inst.rd] = 0;
                    }
                    break;
            }
            return StepResult::Ok;
        }

        case Opcode::Mtspr: {
            const u32 number = inst.spr_number();
            const u32 value = state_.gpr[inst.rs];
            switch (static_cast<Spr>(number)) {
                case Spr::Xer:
                    state_.xer.value = value;
                    break;
                case Spr::Lr:
                    state_.lr = value;
                    break;
                case Spr::Ctr:
                    state_.ctr = value;
                    break;
                case Spr::Srr0:
                    state_.srr0 = value;
                    break;
                case Spr::Srr1:
                    state_.srr1 = value;
                    break;
                case Spr::Dec:
                    state_.decrementer = value;
                    break;
                case Spr::Hid0:
                    state_.hid0 = value;
                    break;
                case Spr::Hid1:
                    state_.hid1 = value;
                    break;
                case Spr::Hid2:
                    state_.hid2 = value;
                    break;
                case Spr::Scr:
                    state_.scr = value;
                    break;
                case Spr::Wpar:
                    state_.wpar = value;
                    break;
                default:
                    if (number >= 272 && number <= 275) {
                        state_.sprg[number - 272] = value;
                    } else if (number >= 912 && number <= 919) {
                        state_.gqr[number - 912].value = value;
                    } else {
                        WIIU_LOG_WARN(
                            Cpu, "mtspr to unmodelled SPR {} at {:#010x}", number, state_.pc);
                    }
                    break;
            }
            return StepResult::Ok;
        }

        case Opcode::Mftb: {
            const u32 number = inst.spr_number();
            state_.gpr[inst.rd] = (number == 269) ? static_cast<u32>(state_.timebase >> 32)
                                                  : static_cast<u32>(state_.timebase);
            return StepResult::Ok;
        }

        case Opcode::Mfcr:
            state_.gpr[inst.rd] = state_.cr.value;
            return StepResult::Ok;

        case Opcode::Mtcrf: {
            // Each CRM bit enables one 4-bit CR field.
            u32 mask = 0;
            for (u32 field = 0; field < 8; ++field) {
                if ((inst.crm.value() & (0x80U >> field)) != 0) {
                    mask |= 0xFU << (28 - field * 4);
                }
            }
            state_.cr.value = (state_.cr.value & ~mask) | (state_.gpr[inst.rs] & mask);
            return StepResult::Ok;
        }

        case Opcode::Mfmsr:
            state_.gpr[inst.rd] = state_.msr;
            return StepResult::Ok;

        case Opcode::Mtmsr:
            state_.msr = state_.gpr[inst.rs];
            return StepResult::Ok;

        case Opcode::Rfi:
            state_.next_pc = state_.srr0 & ~3U;
            state_.msr = state_.srr1;
            return StepResult::Ok;

        case Opcode::Sc:
            // Cafe OS reaches the kernel through sc, and the HLE layer wants
            // to intercept it rather than run a guest handler, so this always
            // surfaces. In Deliver mode it also vectors, so that a guest that
            // installs its own handler still works.
            state_.pc = state_.next_pc;
            if (exception_mode_ == ExceptionMode::Deliver) {
                raise(Exception::SystemCall);
            }
            return StepResult::SystemCall;

        case Opcode::Tw:
        case Opcode::Twi: {
            const auto a = static_cast<s32>(state_.gpr[inst.ra]);
            const s32 b =
                (id == Opcode::Tw) ? static_cast<s32>(state_.gpr[inst.rb]) : inst.simm.value();
            const u32 to = inst.to;

            const bool fire = ((to & 0x10) != 0 && a < b) || ((to & 0x08) != 0 && a > b) ||
                              ((to & 0x04) != 0 && a == b) ||
                              ((to & 0x02) != 0 && static_cast<u32>(a) < static_cast<u32>(b)) ||
                              ((to & 0x01) != 0 && static_cast<u32>(a) > static_cast<u32>(b));
            return fire ? StepResult::Trap : StepResult::Ok;
        }

        // Cache maintenance drives the Phase 3.6 model, which is what makes a
        // CPU write visible to the GPU. These are not no-ops.
        case Opcode::Dcbf:
            memory_.dc_flush_range(effective_address_indexed(inst), 1);
            return StepResult::Ok;
        case Opcode::Dcbst:
            memory_.dc_flush_range(effective_address_indexed(inst), 1);
            return StepResult::Ok;
        case Opcode::Dcbi:
            memory_.dc_invalidate_range(effective_address_indexed(inst), 1);
            return StepResult::Ok;
        case Opcode::Dcbz:
            memory_.dc_zero_range(effective_address_indexed(inst), 1);
            return StepResult::Ok;
        case Opcode::Icbi:
            memory_.ic_invalidate_range(effective_address_indexed(inst), 1);
            return StepResult::Ok;

        // Prefetch hints and barriers have no architectural effect here: the
        // interpreter is sequential and the memory model is already stronger
        // than the guest's.
        case Opcode::Dcbt:
        case Opcode::Dcbtst:
        case Opcode::Sync:
        case Opcode::Isync:
        case Opcode::Eieio:
            return StepResult::Ok;

        default:
            return StepResult::Unimplemented;
    }
}

} // namespace wiiu::cpu
