// Phase 3.4 — address translation.
//
// Espresso translates an effective address one of three ways, and which one
// applies is decided per access:
//
//   translation off   MSR[IR] for fetches, MSR[DR] for data. The effective
//                     address is used as-is. This is the state the HLE loader
//                     leaves the core in, so it is the fast path and costs one
//                     test.
//   block (BAT)       a matching BAT pair maps a 128 KiB - 256 MiB block with
//                     no memory traffic at all.
//   page              segment register -> VSID, hashed page table walk in
//                     guest memory, 4 KiB pages.
//
// All three are modelled. The plan (3.4) asks for a flat fast path plus a
// slow path that honours the real translation, and that is what this is: the
// walk is real, the TLB caches it, and nothing is charged to a guest that
// never turns translation on.
//
// ---------------------------------------------------------------------------
// One thing here is an emulator convention rather than hardware.
//
// Phase 3.1 keys the whole guest address space by *virtual* address: MEM1 is
// committed at 0xF4000000, where Cafe OS shows it to the application, not at
// physical 0x00000000. A real address produced by a BAT or a PTE is therefore
// translated once more — through the region table — back into that space, so
// that a page mapped to physical 0x00001000 reads the same bytes MEM1 shows at
// 0xF4001000. `Translation` carries both: `physical` is what the hardware
// would put on the bus, `address` is where those bytes live here. A real
// address with no region behind it is passed through unchanged, so it faults
// in the memory layer exactly as an unmapped access does today.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>

namespace wiiu::cpu {

enum class AccessKind : u8 {
    Read,
    Write,
    Execute,
};

/// Why a translation failed. Data accesses turn these into DSISR bits, fetches
/// into SRR1 bits; the two registers do not use the same encoding, which is
/// why the reason is kept abstract until it reaches the interpreter.
enum class TranslationFault : u8 {
    None,
    NotTranslated, ///< no BAT and no PTE covers the address
    Protection,    ///< mapped, but this access is not permitted
    NoExecute,     ///< SR[N]: instruction fetch from a no-execute segment
    DirectStore,   ///< SR[T]: the 750 has no direct-store segments
};

struct Translation {
    ppcaddr_t address = 0;   ///< where to read/write in this emulator
    physaddr_t physical = 0; ///< what the hardware would put on the bus
    TranslationFault fault = TranslationFault::None;
    bool translated = false; ///< went through a BAT or a PTE
    bool from_bat = false;
    u8 wimg = 0;
    /// Everything the mapping permits, not just what this access asked for.
    /// The TLB caches this, so a later store to a read-only page still
    /// re-walks and faults rather than inheriting a read's success.
    memory::Access allowed = memory::Access::None;

    [[nodiscard]] constexpr bool ok() const noexcept { return fault == TranslationFault::None; }
};

/// BAT upper/lower register layout, in PowerPC bit order.
struct BatBits {
    static constexpr u32 bepi_mask = 0xFFFE0000; ///< upper bits 0-14
    static constexpr u32 brpn_mask = 0xFFFE0000; ///< lower bits 0-14
    static constexpr u32 valid_supervisor = 1U << 1;
    static constexpr u32 valid_problem = 1U << 0;

    [[nodiscard]] static constexpr u32 block_length(u32 upper) noexcept {
        return (upper >> 2) & 0x7FF; ///< bits 19-29
    }

    /// The bits of an address that fall *within* the block.
    [[nodiscard]] static constexpr u32 block_mask(u32 upper) noexcept {
        return (block_length(upper) << 17) | 0x1FFFF;
    }

    [[nodiscard]] static constexpr u32 wimg(u32 lower) noexcept { return (lower >> 3) & 0xF; }

    [[nodiscard]] static constexpr u32 protection(u32 lower) noexcept { return lower & 0x3; }
};

/// Segment register layout.
struct SegmentBits {
    static constexpr u32 direct_store = 1U << 31;   ///< T
    static constexpr u32 key_supervisor = 1U << 30; ///< Ks
    static constexpr u32 key_problem = 1U << 29;    ///< Kp
    static constexpr u32 no_execute = 1U << 28;     ///< N
    static constexpr u32 vsid_mask = 0x00FFFFFF;    ///< bits 8-31
};

/// Page table entry layout. A PTE is two big-endian words.
struct PteBits {
    static constexpr u32 valid = 1U << 31;     ///< word0 bit 0
    static constexpr u32 secondary = 1U << 6;  ///< word0 bit 25 (H)
    static constexpr u32 referenced = 1U << 8; ///< word1 bit 23 (R)
    static constexpr u32 changed = 1U << 7;    ///< word1 bit 24 (C)

    [[nodiscard]] static constexpr u32 vsid_of(u32 word0) noexcept {
        return (word0 >> 7) & 0x00FFFFFF; ///< bits 1-24
    }

    [[nodiscard]] static constexpr u32 api_of(u32 word0) noexcept {
        return word0 & 0x3F; ///< bits 26-31
    }

    [[nodiscard]] static constexpr u32 rpn_of(u32 word1) noexcept { return word1 & 0xFFFFF000; }

    [[nodiscard]] static constexpr u32 wimg_of(u32 word1) noexcept { return (word1 >> 3) & 0xF; }

    [[nodiscard]] static constexpr u32 protection_of(u32 word1) noexcept { return word1 & 0x3; }
};

/// 750CL: the second set of four BAT pairs is not live until software says so.
///
/// The enable lives in HID4. The bit position below is transcribed rather than
/// verified against hardware — nothing in Cafe OS is known to program HID4
/// here yet, so if a title turns out to use the upper BATs this is the one
/// constant to check first.
inline constexpr u32 hid4_extra_bats = 1U << 25;

class Mmu {
public:
    Mmu(CoreState& state, memory::Memory& memory) noexcept : state_(state), memory_(memory) {}

    /// Is translation on for this kind of access?
    [[nodiscard]] bool enabled(AccessKind kind) const noexcept {
        const u32 bit = (kind == AccessKind::Execute) ? MsrBits::ir : MsrBits::dr;
        return (state_.msr & bit) != 0;
    }

    /// Effective address to real address. One test when translation is off.
    [[nodiscard]] Translation translate(ppcaddr_t ea, AccessKind kind) {
        if (!enabled(kind)) {
            return Translation{.address = ea, .physical = ea};
        }
        return translate_translated(ea, kind);
    }

    /// Translate without recording referenced/changed bits or filling the TLB.
    /// For the debugger and the crash dump, which must not perturb the state
    /// they are inspecting.
    [[nodiscard]] Translation inspect(ppcaddr_t ea, AccessKind kind) const;

    // --- TLB --------------------------------------------------------------

    void invalidate(ppcaddr_t ea) noexcept; ///< tlbie: one page
    void invalidate_all() noexcept;         ///< an SR, BAT or SDR1 write

    struct Stats {
        u64 hits = 0;     ///< satisfied from the TLB
        u64 bat_hits = 0; ///< satisfied by a BAT
        u64 walks = 0;    ///< page table searches
        u64 faults = 0;   ///< translations that failed
    };

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

    void reset_stats() noexcept { stats_ = Stats{}; }

    // --- page table arithmetic, exposed because it is worth testing --------

    /// The primary hash. The secondary is its one's complement.
    [[nodiscard]] static constexpr u32 page_hash(u32 vsid, u32 page_index) noexcept {
        return (vsid ^ page_index) & 0x7FFFF;
    }

    /// Physical address of the eight-entry group a hash selects.
    [[nodiscard]] static constexpr physaddr_t pteg_address(u32 sdr1, u32 hash) noexcept {
        const u32 htaborg = (sdr1 >> 16) & 0xFFFF; ///< bits 0-15
        const u32 htabmask = sdr1 & 0x1FF;         ///< bits 23-31
        const u32 high = (htaborg & ~htabmask) | ((hash >> 10) & htabmask);
        return (high << 16) | ((hash & 0x3FF) << 6);
    }

    /// Bytes spanned by the page table SDR1 describes.
    [[nodiscard]] static constexpr u64 page_table_size(u32 sdr1) noexcept {
        return static_cast<u64>((sdr1 & 0x1FF) + 1) << 16;
    }

    /// Where a real address lands in this emulator's address space. See the
    /// note at the top of the file.
    [[nodiscard]] static ppcaddr_t space_address(const memory::Memory& memory,
                                                 physaddr_t physical) noexcept;

    /// Write a page table entry mapping one 4 KiB page, for tests and for
    /// anything that wants to run the core with translation actually on.
    /// Returns false when every slot in the group is taken.
    [[nodiscard]] static bool install_pte(memory::Memory& memory,
                                          u32 sdr1,
                                          u32 vsid,
                                          ppcaddr_t ea,
                                          physaddr_t physical,
                                          u32 protection,
                                          u32 wimg = 0,
                                          bool secondary = false);

private:
    /// A cached translation. Deliberately one 4 KiB page even when it came
    /// from a BAT: uniform entries keep the hit path to a compare and a
    /// shift, and a BAT re-match on a miss costs nothing.
    struct TlbEntry {
        u32 tag = kEmptyTag; ///< effective page number
        u32 page = 0;        ///< emulator address space page
        u32 physical = 0;    ///< real page number
        ppcaddr_t pte = 0;   ///< where the PTE lives, 0 for a BAT entry
        u8 access = 0;       ///< memory::Access bits
        u8 wimg = 0;
        bool changed = false; ///< the PTE's C bit is already set
    };

    static constexpr u32 kEmptyTag = 0xFFFFFFFF;
    static constexpr u32 kTlbEntries = 256;

    using Tlb = std::array<TlbEntry, kTlbEntries>;

    [[nodiscard]] static constexpr u32 tlb_index(ppcaddr_t ea) noexcept {
        return (ea >> 12) & (kTlbEntries - 1);
    }

    [[nodiscard]] Translation translate_translated(ppcaddr_t ea, AccessKind kind);

    /// The shared, side-effect-free part: BAT match, then page table walk.
    /// `found_pte` receives the PTE's address so the caller can set R and C.
    [[nodiscard]] Translation walk(ppcaddr_t ea, AccessKind kind, ppcaddr_t& found_pte) const;

    [[nodiscard]] Translation match_bat(ppcaddr_t ea, AccessKind kind) const;

    CoreState& state_;
    memory::Memory& memory_;

    Tlb data_tlb_{};
    Tlb instruction_tlb_{};
    Stats stats_{};
};

/// Permitted access for a PP field, given the segment key that applies.
/// Documented as a table in the architecture, and worth keeping as one.
[[nodiscard]] constexpr memory::Access protection_access(u32 pp, bool key) noexcept {
    if (key) {
        switch (pp) {
            case 0:
                return memory::Access::None;
            case 1:
            case 3:
                return memory::read_execute;
            default:
                return memory::read_write_execute;
        }
    }
    return pp == 3 ? memory::read_execute : memory::read_write_execute;
}

} // namespace wiiu::cpu
