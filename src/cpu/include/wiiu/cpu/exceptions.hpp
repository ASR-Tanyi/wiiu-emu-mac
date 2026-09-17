// Phase 4.10 — exception vectors.
//
// Each exception has a fixed offset from the vector base, which MSR[IP]
// selects: 0x00000000 normally, 0xFFF00000 when the exception prefix is set.
#pragma once

#include <wiiu/common/types.hpp>

#include <string_view>

namespace wiiu::cpu {

enum class Exception : u32 {
    SystemReset = 0x0100,
    MachineCheck = 0x0200,
    Dsi = 0x0300, ///< data access could not be completed
    Isi = 0x0400, ///< instruction fetch could not be completed
    ExternalInterrupt = 0x0500,
    Alignment = 0x0600,
    Program = 0x0700, ///< illegal instruction, trap, privileged
    FpUnavailable = 0x0800,
    Decrementer = 0x0900,
    SystemCall = 0x0C00,
    Trace = 0x0D00,
};

[[nodiscard]] std::string_view exception_name(Exception exception) noexcept;

/// SRR1 bits describing why a program exception happened. They sit in the
/// low half of the register, alongside the saved MSR.
struct ProgramCause {
    static constexpr u32 floating_point = 1U << 20;
    static constexpr u32 illegal_instruction = 1U << 19;
    static constexpr u32 privileged = 1U << 18;
    static constexpr u32 trap = 1U << 17;
};

/// DSISR bits set by a data storage exception.
struct DsisrBits {
    static constexpr u32 not_translated = 1U << 30; ///< nothing mapped there
    static constexpr u32 protection = 1U << 27;     ///< mapped, but not permitted
    static constexpr u32 is_store = 1U << 25;       ///< cleared for a load
};

/// An instruction storage exception reports the same conditions, but in SRR1
/// rather than DSISR, and at different bit positions. Keeping the two sets
/// apart is the point: a fetch fault and a data fault do not share a register.
struct IsiCause {
    static constexpr u32 not_translated = 1U << 30; ///< bit 1: no PTE
    static constexpr u32 no_execute = 1U << 28;     ///< bit 3: SR[N] or guarded
    static constexpr u32 protection = 1U << 27;     ///< bit 4: protection violation
};

/// MSR bits an exception clears on entry. ME and IP survive; everything to do
/// with translation, interrupts and user mode does not.
///
/// IP is the one that has to be excluded by hand: it selects the vector base,
/// so clearing it before the vector is computed sends every exception to
/// 0x00000100 regardless of where the guest asked for them.
inline constexpr u32 msr_exception_clear_mask = 0x0004EF33;

/// The MSR bits an exception saves into SRR1 and `rfi` restores from it.
/// Everything outside this mask keeps its current value on return: SRR1 also
/// carries the exception's cause bits, and copying those into MSR wholesale
/// would, among other things, put the core into power-saving mode on the way
/// back from a privileged-instruction exception.
inline constexpr u32 msr_restore_mask = 0x87C0FF73;

/// Vector base, selected by MSR[IP].
[[nodiscard]] constexpr ppcaddr_t exception_base(u32 msr) noexcept {
    return (msr & (1U << 6)) != 0 ? 0xFFF00000U : 0x00000000U;
}

} // namespace wiiu::cpu
