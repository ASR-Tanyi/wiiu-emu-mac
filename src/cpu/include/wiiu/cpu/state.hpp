// Phase 4.1 — Espresso core state.
//
// Laid out hot-fields-first and cacheline-aligned (P37): three of these live
// on three host threads, and false sharing between them would be an
// unpleasant thing to discover later.
#pragma once

#include <wiiu/common/types.hpp>

#include <array>
#include <cstddef>

namespace wiiu::cpu {

/// A floating-point register.
///
/// Gekko (and so Espresso) widened the PowerPC register file: each FPR holds
/// *two* independent doubles. `ps0` is the ordinary PowerPC double-precision
/// register — every classic FP instruction reads and writes it — and `ps1` is
/// the second slot the paired-single instructions operate on alongside it.
///
/// Modelling this as a union over one 64-bit value would be wrong: on a
/// little-endian host the two halves of a double land in the opposite order
/// from the hardware's ps0/ps1, so `ps1` would silently alias the low word of
/// the double. Two named doubles is both correct and what the paired-single
/// instructions actually need, since each slot carries full double precision
/// internally and is only rounded to single when an instruction says so.
struct FpuRegister {
    f64 ps0 = 0.0;
    f64 ps1 = 0.0;

    [[nodiscard]] constexpr f64 as_double() const noexcept { return ps0; }

    constexpr void set_double(f64 value) noexcept { ps0 = value; }

    /// Both slots to the same value, as a scalar result leaves the register.
    constexpr void splat(f64 value) noexcept {
        ps0 = value;
        ps1 = value;
    }
};

static_assert(sizeof(FpuRegister) == 16);

/// Condition register, as eight 4-bit fields. CR0 is bits 0-3 in PowerPC
/// numbering, which is the *most* significant nibble of the 32-bit value.
struct ConditionRegister {
    u32 value = 0;

    static constexpr u32 lt = 1U << 3;
    static constexpr u32 gt = 1U << 2;
    static constexpr u32 eq = 1U << 1;
    static constexpr u32 so = 1U << 0;

    /// Field 0 is the high nibble, so field n sits at shift 28 - 4n.
    [[nodiscard]] constexpr u32 field(u32 index) const noexcept {
        return (value >> (28 - index * 4)) & 0xF;
    }

    constexpr void set_field(u32 index, u32 bits) noexcept {
        const u32 shift = 28 - index * 4;
        value = (value & ~(0xFU << shift)) | ((bits & 0xF) << shift);
    }

    /// CR bit numbering runs 0-31 from the most significant end.
    [[nodiscard]] constexpr bool bit(u32 index) const noexcept {
        return ((value >> (31 - index)) & 1) != 0;
    }

    constexpr void set_bit(u32 index, bool set) noexcept {
        const u32 mask = 1U << (31 - index);
        value = set ? (value | mask) : (value & ~mask);
    }
};

/// Fixed-point exception register.
struct XerRegister {
    u32 value = 0;

    static constexpr u32 so_mask = 1U << 31; ///< summary overflow (sticky)
    static constexpr u32 ov_mask = 1U << 30; ///< overflow
    static constexpr u32 ca_mask = 1U << 29; ///< carry

    [[nodiscard]] constexpr bool carry() const noexcept { return (value & ca_mask) != 0; }

    [[nodiscard]] constexpr bool overflow() const noexcept { return (value & ov_mask) != 0; }

    [[nodiscard]] constexpr bool summary_overflow() const noexcept {
        return (value & so_mask) != 0;
    }

    constexpr void set_carry(bool set) noexcept {
        value = set ? (value | ca_mask) : (value & ~ca_mask);
    }

    /// Setting OV also sets SO, which is sticky until software clears it.
    constexpr void set_overflow(bool set) noexcept {
        if (set) {
            value |= ov_mask | so_mask;
        } else {
            value &= ~ov_mask;
        }
    }

    /// The byte count used by lswx/stswx.
    [[nodiscard]] constexpr u32 string_length() const noexcept { return value & 0x7F; }
};

/// Machine state register bits that matter to us.
struct MsrBits {
    static constexpr u32 le = 1U << 0; ///< little-endian mode
    static constexpr u32 ri = 1U << 1; ///< recoverable interrupt
    static constexpr u32 dr = 1U << 4; ///< data address translation
    static constexpr u32 ir = 1U << 5; ///< instruction address translation
    static constexpr u32 ip = 1U << 6; ///< exception prefix
    static constexpr u32 fe1 = 1U << 8;
    static constexpr u32 be = 1U << 9;  ///< branch trace
    static constexpr u32 se = 1U << 10; ///< single step
    static constexpr u32 fe0 = 1U << 11;
    static constexpr u32 me = 1U << 12; ///< machine check enable
    static constexpr u32 fp = 1U << 13; ///< floating point available
    static constexpr u32 pr = 1U << 14; ///< problem state (user mode)
    static constexpr u32 ee = 1U << 15; ///< external interrupt enable
    static constexpr u32 pow = 1U << 18;
};

/// A graphics quantisation register: the scale and type used by psq_l/psq_st.
struct GqrRegister {
    u32 value = 0;

    [[nodiscard]] constexpr u32 store_type() const noexcept { return value & 0x7; }

    [[nodiscard]] constexpr u32 store_scale() const noexcept { return (value >> 8) & 0x3F; }

    [[nodiscard]] constexpr u32 load_type() const noexcept { return (value >> 16) & 0x7; }

    [[nodiscard]] constexpr u32 load_scale() const noexcept { return (value >> 24) & 0x3F; }
};

/// Quantisation formats for psq_l / psq_st.
enum class QuantizeType : u32 {
    Float = 0,
    U8 = 4,
    U16 = 5,
    S8 = 6,
    S16 = 7,
};

/// One Espresso core.
struct alignas(64) CoreState {
    // Hot: touched by almost every instruction.
    std::array<u32, 32> gpr{};
    ppcaddr_t pc = 0;
    ppcaddr_t next_pc = 0;
    ConditionRegister cr;
    XerRegister xer;
    u32 ctr = 0;
    u32 lr = 0;

    // Floating point.
    std::array<FpuRegister, 32> fpr{};
    u32 fpscr = 0;

    // Supervisor.
    u32 msr = 0;
    u32 srr0 = 0;
    u32 srr1 = 0;
    std::array<u32, 4> sprg{};
    u32 dar = 0;
    u32 dsisr = 0;

    // Address translation (Phase 3.4). Cold: only the MMU reads these, and
    // only when MSR[IR] or MSR[DR] is set.
    //
    // The 750 architecture has four IBAT and four DBAT pairs; the 750CL (and
    // so Espresso) adds a second set of four, enabled by HID4. Both sets live
    // here and the MMU decides which are live, so a title that programs the
    // upper BATs is a configuration question rather than a missing register.
    std::array<u32, 16> sr{};    ///< segment registers, one per 256 MiB
    std::array<u32, 8> ibat_u{}; ///< IBAT0U..IBAT7U
    std::array<u32, 8> ibat_l{}; ///< IBAT0L..IBAT7L
    std::array<u32, 8> dbat_u{}; ///< DBAT0U..DBAT7U
    std::array<u32, 8> dbat_l{}; ///< DBAT0L..DBAT7L
    u32 sdr1 = 0;                ///< page table origin and size
    u32 hid4 = 0;                ///< 750CL: gates the upper four BAT pairs

    // Espresso / Gekko specifics.
    std::array<GqrRegister, 8> gqr{};
    u32 hid0 = 0;
    u32 hid1 = 0;
    u32 hid2 = 0;
    u32 scr = 0;  ///< inter-core interrupts
    u32 wpar = 0; ///< write-gather pipe

    // Timing.
    u64 timebase = 0;
    u32 decrementer = 0;

    /// Which of the three cores this is. Core 1 is the one with 2 MiB of L2;
    /// Cafe OS puts the main game thread there.
    u32 core_id = 0;

    /// Set when the core is parked — halted, or skipped forward past an idle
    /// loop (P2).
    bool halted = false;

    /// A reservation created by lwarx, cleared by stwcx. or an intervening
    /// write. Espresso's implementation is buggy in hardware and Cafe OS
    /// carries workarounds; this models the documented behaviour.
    bool reserved = false;
    ppcaddr_t reservation_address = 0;

    void reset() { *this = CoreState{}; }
};

static_assert(alignof(CoreState) == 64, "cores share cache lines otherwise");

/// The three cores, as the hardware arranges them.
inline constexpr u32 core_count = 3;

/// Espresso runs at 1243.125 MHz off a 248.625 MHz bus — the core is exactly
/// five times the bus clock. As on every PowerPC 750 derivative, the timebase
/// counts the *bus* clock divided by four, not the core clock, so it ticks at
/// 62.15625 MHz. (Cafe OS calls this `OSTimerClockSpeed`; the Wii's Broadway
/// has the same relationship at 243 MHz / 4.) Getting this wrong by a factor
/// of five makes every emulated second last a fifth as long, which shows up
/// as animations running fast rather than as anything that looks like a bug.
inline constexpr u64 cpu_clock_hz = 1243125000;
inline constexpr u64 bus_clock_hz = cpu_clock_hz / 5;
inline constexpr u64 timebase_hz = bus_clock_hz / 4;
static_assert(timebase_hz == 62156250);

/// L2 is asymmetric: core 1 has 2 MiB, the other two have 512 KiB.
[[nodiscard]] constexpr u32 l2_cache_size(u32 core_id) noexcept {
    return core_id == 1 ? 2U * 1024 * 1024 : 512U * 1024;
}

} // namespace wiiu::cpu
