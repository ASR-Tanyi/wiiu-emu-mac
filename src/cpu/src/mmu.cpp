#include <wiiu/common/log.hpp>
#include <wiiu/cpu/mmu.hpp>

namespace wiiu::cpu {
namespace {

using wiiu::log::Channel;

[[nodiscard]] constexpr memory::Access required_access(AccessKind kind) noexcept {
    switch (kind) {
        case AccessKind::Write:
            return memory::Access::Write;
        case AccessKind::Execute:
            return memory::Access::Execute;
        case AccessKind::Read:
            break;
    }
    return memory::Access::Read;
}

/// PTEs are two words; reading one out of unmapped memory would record a fault
/// against the *guest's* access rather than the walk, so the range is checked
/// first and a missing group is simply treated as empty.
[[nodiscard]] bool read_pte(const memory::Memory& memory, ppcaddr_t at, u32& word0, u32& word1) {
    if (!memory.range_is(at, 8, memory::Access::Read)) {
        return false;
    }
    return memory.try_read<u32>(at, word0) && memory.try_read<u32>(at + 4, word1);
}

} // namespace

ppcaddr_t Mmu::space_address(const memory::Memory& memory, physaddr_t physical) noexcept {
    for (const memory::Region& region : memory.regions()) {
        if (region.backing == memory::Backing::None) {
            continue;
        }
        const physaddr_t offset = physical - region.physical;
        if (physical >= region.physical && offset < region.size) {
            return region.base + static_cast<u32>(offset);
        }
    }
    // Nothing backs it. Handing the real address straight to the memory layer
    // means MMIO windows still reach their devices and everything else faults,
    // which is what the hardware would do with it too.
    return physical;
}

Translation Mmu::match_bat(ppcaddr_t ea, AccessKind kind) const {
    const bool instruction = kind == AccessKind::Execute;
    const std::array<u32, 8>& uppers = instruction ? state_.ibat_u : state_.dbat_u;
    const std::array<u32, 8>& lowers = instruction ? state_.ibat_l : state_.dbat_l;

    const bool problem = (state_.msr & MsrBits::pr) != 0;
    const u32 valid_bit = problem ? BatBits::valid_problem : BatBits::valid_supervisor;
    const u32 count = (state_.hid4 & hid4_extra_bats) != 0 ? 8U : 4U;

    for (u32 index = 0; index < count; ++index) {
        const u32 upper = uppers[index];
        if ((upper & valid_bit) == 0) {
            continue;
        }

        const u32 offset_mask = BatBits::block_mask(upper);
        if (((ea ^ upper) & BatBits::bepi_mask & ~offset_mask) != 0) {
            continue;
        }

        // A BAT's PP field is used directly, without a segment key: there is
        // no segment involved.
        const memory::Access allowed = protection_access(BatBits::protection(lowers[index]), true);
        const physaddr_t physical =
            (lowers[index] & BatBits::brpn_mask & ~offset_mask) | (ea & offset_mask);

        Translation result{.address = 0,
                           .physical = physical,
                           .fault = TranslationFault::None,
                           .translated = true,
                           .from_bat = true,
                           .wimg = static_cast<u8>(BatBits::wimg(lowers[index])),
                           .allowed = allowed};

        if (!has(allowed, required_access(kind))) {
            result.fault = TranslationFault::Protection;
            return result;
        }

        result.address = space_address(memory_, physical);
        return result;
    }

    return Translation{}; // no block covers it; not a fault by itself
}

Translation Mmu::walk(ppcaddr_t ea, AccessKind kind, ppcaddr_t& found_pte) const {
    found_pte = 0;

    const Translation block = match_bat(ea, kind);
    if (block.translated) {
        return block;
    }

    const u32 segment = state_.sr[ea >> 28];

    // Direct-store segments are a 601 feature the 750 dropped: the hardware
    // takes the exception rather than inventing a translation.
    if ((segment & SegmentBits::direct_store) != 0) {
        return Translation{.fault = TranslationFault::DirectStore};
    }

    if (kind == AccessKind::Execute && (segment & SegmentBits::no_execute) != 0) {
        return Translation{.fault = TranslationFault::NoExecute};
    }

    const u32 vsid = segment & SegmentBits::vsid_mask;
    const bool key = (state_.msr & MsrBits::pr) != 0 ? (segment & SegmentBits::key_problem) != 0
                                                     : (segment & SegmentBits::key_supervisor) != 0;

    const u32 page_index = (ea >> 12) & 0xFFFF;
    const u32 api = (ea >> 22) & 0x3F;
    const u32 primary = page_hash(vsid, page_index);

    for (u32 pass = 0; pass < 2; ++pass) {
        const bool secondary = pass == 1;
        const u32 hash = secondary ? (~primary & 0x7FFFF) : primary;
        const physaddr_t group = pteg_address(state_.sdr1, hash);

        for (u32 slot = 0; slot < 8; ++slot) {
            const ppcaddr_t at = space_address(memory_, group + slot * 8);

            u32 word0 = 0;
            u32 word1 = 0;
            if (!read_pte(memory_, at, word0, word1)) {
                continue;
            }
            if ((word0 & PteBits::valid) == 0 || PteBits::vsid_of(word0) != vsid ||
                PteBits::api_of(word0) != api || ((word0 & PteBits::secondary) != 0) != secondary) {
                continue;
            }

            found_pte = at;

            // PowerPC has no execute permission bit: a page you may read, you
            // may fetch from. SR[N], checked above, is the only thing that
            // says otherwise.
            const memory::Access allowed = protection_access(PteBits::protection_of(word1), key);
            const physaddr_t physical = PteBits::rpn_of(word1) | (ea & 0xFFF);

            Translation result{.address = 0,
                               .physical = physical,
                               .fault = TranslationFault::None,
                               .translated = true,
                               .from_bat = false,
                               .wimg = static_cast<u8>(PteBits::wimg_of(word1)),
                               .allowed = allowed};

            if (!has(allowed, required_access(kind))) {
                result.fault = TranslationFault::Protection;
                return result;
            }

            result.address = space_address(memory_, physical);
            return result;
        }
    }

    return Translation{.fault = TranslationFault::NotTranslated};
}

Translation Mmu::inspect(ppcaddr_t ea, AccessKind kind) const {
    if (!enabled(kind)) {
        return Translation{.address = ea, .physical = ea};
    }
    ppcaddr_t pte = 0;
    return walk(ea, kind, pte);
}

Translation Mmu::translate_translated(ppcaddr_t ea, AccessKind kind) {
    Tlb& tlb = (kind == AccessKind::Execute) ? instruction_tlb_ : data_tlb_;
    TlbEntry& entry = tlb[tlb_index(ea)];

    const u32 tag = ea >> 12;
    const u32 offset = ea & 0xFFF;

    if (entry.tag == tag && has(static_cast<memory::Access>(entry.access), required_access(kind))) {
        ++stats_.hits;

        // A store to a page whose change bit is still clear has to reach the
        // PTE: software uses C to find the pages it must write back.
        if (kind == AccessKind::Write && !entry.changed && entry.pte != 0) {
            u32 word1 = 0;
            if (memory_.try_read<u32>(entry.pte + 4, word1)) {
                static_cast<void>(memory_.try_write<u32>(
                    entry.pte + 4, word1 | PteBits::referenced | PteBits::changed));
            }
            entry.changed = true;
        }

        return Translation{.address = (entry.page << 12) | offset,
                           .physical = (entry.physical << 12) | offset,
                           .fault = TranslationFault::None,
                           .translated = true,
                           .from_bat = entry.pte == 0,
                           .wimg = entry.wimg};
    }

    ppcaddr_t pte = 0;
    const Translation result = walk(ea, kind, pte);

    if (result.from_bat) {
        ++stats_.bat_hits;
    } else {
        ++stats_.walks;
    }

    if (!result.ok()) {
        ++stats_.faults;
        return result;
    }

    bool changed = true;
    if (pte != 0) {
        // Reference and change bits are architecturally required, and guest
        // page-replacement code reads them. Setting them here rather than in
        // the walk keeps `inspect` free of side effects.
        u32 word1 = 0;
        if (memory_.try_read<u32>(pte + 4, word1)) {
            const u32 wanted =
                word1 | PteBits::referenced | (kind == AccessKind::Write ? PteBits::changed : 0U);
            if (wanted != word1) {
                static_cast<void>(memory_.try_write<u32>(pte + 4, wanted));
            }
            changed = (wanted & PteBits::changed) != 0;
        }
    }

    entry = TlbEntry{.tag = tag,
                     .page = result.address >> 12,
                     .physical = result.physical >> 12,
                     .pte = pte,
                     .access = static_cast<u8>(result.allowed),
                     .wimg = result.wimg,
                     .changed = changed};

    return result;
}

void Mmu::invalidate(ppcaddr_t ea) noexcept {
    const u32 index = tlb_index(ea);
    const u32 tag = ea >> 12;
    if (data_tlb_[index].tag == tag) {
        data_tlb_[index] = TlbEntry{};
    }
    if (instruction_tlb_[index].tag == tag) {
        instruction_tlb_[index] = TlbEntry{};
    }
}

void Mmu::invalidate_all() noexcept {
    data_tlb_.fill(TlbEntry{});
    instruction_tlb_.fill(TlbEntry{});
}

bool Mmu::install_pte(memory::Memory& memory,
                      u32 sdr1,
                      u32 vsid,
                      ppcaddr_t ea,
                      physaddr_t physical,
                      u32 protection,
                      u32 wimg,
                      bool secondary) {
    const u32 page_index = (ea >> 12) & 0xFFFF;
    const u32 primary = page_hash(vsid, page_index);
    const u32 hash = secondary ? (~primary & 0x7FFFF) : primary;
    const physaddr_t group = pteg_address(sdr1, hash);

    const u32 word0 = PteBits::valid | ((vsid & 0x00FFFFFF) << 7) |
                      (secondary ? PteBits::secondary : 0U) | ((ea >> 22) & 0x3F);
    const u32 word1 = (physical & 0xFFFFF000) | ((wimg & 0xF) << 3) | (protection & 0x3);

    for (u32 slot = 0; slot < 8; ++slot) {
        const ppcaddr_t at = space_address(memory, group + slot * 8);

        u32 existing = 0;
        if (!memory.try_read<u32>(at, existing)) {
            return false;
        }
        // Free, or already describing this page: either way it is ours.
        const bool same = (existing & PteBits::valid) != 0 && PteBits::vsid_of(existing) == vsid &&
                          PteBits::api_of(existing) == ((ea >> 22) & 0x3F) &&
                          ((existing & PteBits::secondary) != 0) == secondary;
        if ((existing & PteBits::valid) != 0 && !same) {
            continue;
        }

        return memory.try_write<u32>(at, word0) && memory.try_write<u32>(at + 4, word1);
    }

    WIIU_LOG_WARN(Mem, "page table group for {:#010x} is full", ea);
    return false;
}

} // namespace wiiu::cpu
