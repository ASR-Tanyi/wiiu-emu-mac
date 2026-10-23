#include <wiiu/loader/relocations.hpp>

#include <fmt/format.h>

#include <cstring>

namespace wiiu::loader {

std::string_view relocation_name(RelocationType type) noexcept {
    switch (type) {
        case RelocationType::None:
            return "NONE";
        case RelocationType::Addr32:
            return "ADDR32";
        case RelocationType::Addr24:
            return "ADDR24";
        case RelocationType::Addr16Lo:
            return "ADDR16_LO";
        case RelocationType::Addr16Hi:
            return "ADDR16_HI";
        case RelocationType::Addr16Ha:
            return "ADDR16_HA";
        case RelocationType::Addr14:
            return "ADDR14";
        case RelocationType::Rel24:
            return "REL24";
        case RelocationType::Rel14:
            return "REL14";
        case RelocationType::GhsRel16Ha:
            return "GHS_REL16_HA";
        case RelocationType::GhsRel16Hi:
            return "GHS_REL16_HI";
        case RelocationType::GhsRel16Lo:
            return "GHS_REL16_LO";
        case RelocationType::DtpMod32:
            return "DTPMOD32";
        case RelocationType::DtpRel32:
            return "DTPREL32";
    }
    return "?";
}

std::vector<Relocation> parse_relocations(const Section& section) {
    std::vector<Relocation> out;
    const std::size_t count = section.data.size() / sizeof(Elf32Rela);
    out.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        Elf32Rela entry{};
        std::memcpy(&entry, section.data.data() + i * sizeof(Elf32Rela), sizeof(entry));

        Relocation relocation;
        relocation.offset = entry.offset;
        relocation.symbol = entry.info.value() >> 8;
        relocation.type = static_cast<RelocationType>(entry.info.value() & 0xFF);
        relocation.addend = entry.addend;
        out.push_back(relocation);
    }
    return out;
}

bool apply_relocation(
    RelocationType type, u32 word, ppcaddr_t target, ppcaddr_t at, u32& out) noexcept {
    switch (type) {
        case RelocationType::None:
            out = word;
            return true;

        case RelocationType::Addr32:
        case RelocationType::DtpRel32:
        case RelocationType::DtpMod32:
            out = target;
            return true;

        case RelocationType::Addr16Lo:
            out = (word & 0xFFFF0000U) | (target & 0xFFFFU);
            return true;

        case RelocationType::Addr16Hi:
            out = (word & 0xFFFF0000U) | ((target >> 16) & 0xFFFFU);
            return true;

        case RelocationType::Addr16Ha:
            out = (word & 0xFFFF0000U) | high_adjusted(target);
            return true;

        case RelocationType::Addr24:
            // An absolute branch: the target replaces the displacement field,
            // keeping the opcode and the AA/LK bits.
            out = (word & 0xFC000003U) | (target & 0x03FFFFFCU);
            return true;

        case RelocationType::Addr14:
            out = (word & 0xFFFF0003U) | (target & 0x0000FFFCU);
            return true;

        case RelocationType::Rel24: {
            const auto displacement = static_cast<s32>(target - at);
            // A 26-bit signed range with two implied zero bits. Out of range
            // means the linker needed a trampoline and did not emit one.
            if (displacement > 0x01FFFFFF || displacement < -0x02000000) {
                return false;
            }
            out = (word & 0xFC000003U) | (static_cast<u32>(displacement) & 0x03FFFFFCU);
            return true;
        }

        case RelocationType::Rel14: {
            const auto displacement = static_cast<s32>(target - at);
            if (displacement > 0x7FFF || displacement < -0x8000) {
                return false;
            }
            out = (word & 0xFFFF0003U) | (static_cast<u32>(displacement) & 0x0000FFFCU);
            return true;
        }

        case RelocationType::GhsRel16Ha:
        case RelocationType::GhsRel16Hi:
        case RelocationType::GhsRel16Lo:
            return false; // halfwords: relocation_halfword
    }
    return false;
}

RelocationResult apply_relocations(std::span<const Relocation> relocations,
                                   Section& section,
                                   const RelocationContext& context) {
    RelocationResult result;

    for (const Relocation& relocation : relocations) {
        // The offset is an address; the data is indexed from the section base.
        if (relocation.offset < section.address) {
            result.warnings.push_back(fmt::format(
                "relocation at {:#010x} is before section '{}'", relocation.offset, section.name));
            continue;
        }
        const u32 index = relocation.offset - section.address;
        const u32 width = is_halfword_relocation(relocation.type) ? 2U : 4U;
        if (index + width > section.data.size()) {
            result.warnings.push_back(
                fmt::format("relocation at {:#010x} is past the end of section '{}'",
                            relocation.offset,
                            section.name));
            continue;
        }

        // An entry with a symbol is an import and needs resolving; one without
        // just moves with the module.
        const ppcaddr_t at = relocation.offset + static_cast<u32>(context.delta);
        ppcaddr_t target = 0;
        if (relocation.symbol != 0 && context.resolve) {
            if (!context.resolve(relocation.symbol, target)) {
                result.warnings.push_back(fmt::format("unresolved symbol {} for {} at {:#010x}",
                                                      relocation.symbol,
                                                      relocation_name(relocation.type),
                                                      relocation.offset));
                continue;
            }
            const bool branch = relocation.type == RelocationType::Rel24 ||
                                relocation.type == RelocationType::Rel14;
            if (branch && target == 0 && context.undefined_weak &&
                context.undefined_weak(relocation.symbol)) {
                target = at; // to itself (decaf's loader)
            }
            target += static_cast<u32>(relocation.addend);
        } else {
            target = static_cast<u32>(static_cast<s64>(relocation.addend) + context.delta);
        }

        if (width == 2) {
            // The offset addresses the immediate field itself, so only those
            // two bytes are written; the opcode beside them is not ours to
            // touch.
            const u16be written = relocation_halfword(relocation.type, target, at);
            std::memcpy(section.data.data() + index, &written, sizeof(written));
            ++result.applied;
            continue;
        }

        u32be stored{};
        std::memcpy(&stored, section.data.data() + index, sizeof(stored));

        u32 patched = 0;
        if (!apply_relocation(relocation.type, stored.value(), target, at, patched)) {
            result.warnings.push_back(
                fmt::format("{} at {:#010x} could not be applied (target {:#010x} out of range)",
                            relocation_name(relocation.type),
                            relocation.offset,
                            target));
            continue;
        }

        const u32be written = patched;
        std::memcpy(section.data.data() + index, &written, sizeof(written));
        ++result.applied;
    }

    result.ok = true;
    return result;
}

} // namespace wiiu::loader
