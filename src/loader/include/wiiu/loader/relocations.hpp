// Phase 5.4 — relocations.
//
// An RPL is linked at a nominal base but loaded wherever the loader puts it,
// so its code and data carry RELA entries describing every address that has to
// be adjusted. PowerPC's split-immediate forms make this more interesting than
// "add a delta": a 32-bit address is often assembled from two 16-bit halves in
// separate instructions, and the high half depends on the low half's sign.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/loader/rpl.hpp>

#include <functional>
#include <span>
#include <string>
#include <vector>

namespace wiiu::loader {

struct Elf32Rela {
    u32be offset; ///< address within the section being relocated
    u32be info;   ///< symbol index in the high 24 bits, type in the low 8
    s32be addend;
};

static_assert(sizeof(Elf32Rela) == 12);

/// The subset PowerPC RPLs actually use.
enum class RelocationType : u8 {
    None = 0,
    Addr32 = 1,   ///< the whole 32-bit address
    Addr16Lo = 4, ///< low half, as in `ori rX,rY,lo`
    Addr16Hi = 5, ///< high half, unadjusted
    Addr16Ha = 6, ///< high half, adjusted for the low half's sign
    Addr24 = 2,
    Addr14 = 7,
    Rel24 = 10,    ///< branch displacement, as in `bl`
    Rel14 = 11,    ///< conditional branch displacement
    DtpMod32 = 68, ///< TLS module id
    DtpRel32 = 78, ///< TLS offset within the module
    /// Green Hills' halves of a displacement, S + A - P, each written at its
    /// own halfword. elf2rpl turns each R_PPC_REL32 -- a switch's jump table
    /// in .rodata, say -- into a HI at the word and a LO at the word + 2 (its
    /// addend 2 more, so both are the word's displacement); Cafe OS's loader
    /// has no REL32 (decaf's reimplementation of it).
    GhsRel16Ha = 251,
    GhsRel16Hi = 252,
    GhsRel16Lo = 253,
};

[[nodiscard]] std::string_view relocation_name(RelocationType type) noexcept;

struct Relocation {
    ppcaddr_t offset = 0;
    u32 symbol = 0;
    RelocationType type = RelocationType::None;
    s32 addend = 0;
};

/// Read a SHT_RELA section.
[[nodiscard]] std::vector<Relocation> parse_relocations(const Section& section);

struct RelocationContext {
    /// Where the module ended up, minus where it was linked.
    s32 delta = 0;

    /// Resolved address for a symbol index, for imports. Returning false
    /// leaves the relocation unapplied and records a warning.
    std::function<bool(u32 symbol, ppcaddr_t& out)> resolve;

    /// Whether a symbol is an undefined weak one, whose address is zero. A
    /// branch to one is made a branch to itself, as Cafe OS's loader makes it
    /// (decaf): wut's crt calls `__init_wut_socket` only `if
    /// (&__init_wut_socket)`, so the branch is there and never runs, and zero
    /// is out of a relative branch's reach.
    std::function<bool(u32 symbol)> undefined_weak;
};

struct RelocationResult {
    bool ok = false;
    std::string error;
    std::vector<std::string> warnings;
    u32 applied = 0;
};

/// The adjusted high half: `(value + 0x8000) >> 16`.
///
/// The low half is sign-extended by the instruction that uses it, so when it
/// is 0x8000 or above it will be treated as negative and subtract 0x10000.
/// HA pre-compensates. Using the plain top 16 bits instead is correct exactly
/// half the time, which is the worst possible failure mode.
[[nodiscard]] constexpr u16 high_adjusted(u32 value) noexcept {
    return static_cast<u16>(((value + 0x8000U) >> 16) & 0xFFFF);
}

/// True for the relocations that patch a 16-bit immediate field.
///
/// This matters more than it looks. A `lis r3,sym@ha` relocation's offset
/// points at the *immediate halfword* -- the instruction address plus two --
/// not at the instruction. Reading a 32-bit word there straddles two
/// instructions and corrupts both, while leaving the module looking like it
/// loaded correctly.
[[nodiscard]] constexpr bool is_halfword_relocation(RelocationType type) noexcept {
    return type == RelocationType::Addr16Lo || type == RelocationType::Addr16Hi ||
           type == RelocationType::Addr16Ha || type == RelocationType::GhsRel16Ha ||
           type == RelocationType::GhsRel16Hi || type == RelocationType::GhsRel16Lo;
}

/// The 16-bit value a halfword relocation at `at` writes.
[[nodiscard]] constexpr u16
relocation_halfword(RelocationType type, ppcaddr_t target, ppcaddr_t at) noexcept {
    const ppcaddr_t relative = target - at;
    switch (type) {
        case RelocationType::Addr16Lo:
            return static_cast<u16>(target & 0xFFFF);
        case RelocationType::Addr16Hi:
            return static_cast<u16>((target >> 16) & 0xFFFF);
        case RelocationType::Addr16Ha:
            return high_adjusted(target);
        case RelocationType::GhsRel16Lo:
            return static_cast<u16>(relative & 0xFFFF);
        case RelocationType::GhsRel16Hi:
            return static_cast<u16>((relative >> 16) & 0xFFFF);
        case RelocationType::GhsRel16Ha:
            return high_adjusted(relative);
        default:
            return 0;
    }
}

/// Apply one relocation to a 32-bit word, returning the new word.
///
/// Exposed separately because the split-immediate arithmetic is the part worth
/// testing directly: HA is not simply the top 16 bits.
[[nodiscard]] bool
apply_relocation(RelocationType type, u32 word, ppcaddr_t target, ppcaddr_t at, u32& out) noexcept;

/// Relocate a section's contents in place.
[[nodiscard]] RelocationResult apply_relocations(std::span<const Relocation> relocations,
                                                 Section& section,
                                                 const RelocationContext& context);

} // namespace wiiu::loader
