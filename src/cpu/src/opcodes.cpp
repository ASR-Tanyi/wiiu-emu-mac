#include <wiiu/cpu/opcodes.hpp>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace wiiu::cpu {
namespace {

constexpr std::array<OpcodeInfo, static_cast<std::size_t>(Opcode::Count) - 1> kTable{{
#define WIIU_PPC_ENTRY(id, mnemonic, ...) {Opcode::id, mnemonic, __VA_ARGS__},
    WIIU_PPC_OPCODES(WIIU_PPC_ENTRY)
#undef WIIU_PPC_ENTRY
}};

constexpr OpcodeInfo kInvalid{Opcode::Invalid, "(invalid)", 0, 0, Form::Invalid, FlagNone};

/// Candidates grouped by primary opcode.
///
/// A flat scan of ~180 entries per instruction would be visible even in a
/// disassembler; bucketing by the six-bit primary opcode leaves at most a
/// couple of dozen candidates. Within a bucket the more specific pattern (the
/// larger mask) is tried first, so an exact encoding like `sc` wins over the
/// general form it sits inside.
class Index {
public:
    Index() {
        for (const OpcodeInfo& info : kTable) {
            const u32 primary = (info.pattern >> 26) & 0x3F;
            buckets_[primary].push_back(&info);
        }
        for (auto& bucket : buckets_) {
            std::ranges::sort(bucket, [](const OpcodeInfo* lhs, const OpcodeInfo* rhs) {
                return lhs->mask > rhs->mask;
            });
        }

        for (const OpcodeInfo& info : kTable) {
            by_id_[static_cast<std::size_t>(info.id)] = &info;
        }
        by_id_[static_cast<std::size_t>(Opcode::Invalid)] = &kInvalid;
    }

    [[nodiscard]] Opcode decode(u32 word) const noexcept {
        for (const OpcodeInfo* info : buckets_[(word >> 26) & 0x3F]) {
            if (info->matches(word)) {
                return info->id;
            }
        }
        return Opcode::Invalid;
    }

    [[nodiscard]] const OpcodeInfo& info(Opcode id) const noexcept {
        const auto index = static_cast<std::size_t>(id);
        if (index >= by_id_.size() || by_id_[index] == nullptr) {
            return kInvalid;
        }
        return *by_id_[index];
    }

private:
    std::array<std::vector<const OpcodeInfo*>, 64> buckets_{};
    std::array<const OpcodeInfo*, static_cast<std::size_t>(Opcode::Count)> by_id_{};
};

const Index& index() {
    static const Index instance;
    return instance;
}

} // namespace

std::span<const OpcodeInfo> opcode_table() noexcept {
    return kTable;
}

const OpcodeInfo& opcode_info(Opcode id) noexcept {
    return index().info(id);
}

Opcode decode(u32 word) noexcept {
    return index().decode(word);
}

std::string decorated_name(Opcode id, Instruction instruction) {
    const OpcodeInfo& info = opcode_info(id);
    std::string name{info.name};

    // Order matters and follows the architecture's own spelling: `addo.` is
    // overflow then record, `bcla` is link then absolute.
    if ((info.flags & FlagOe) != 0 && instruction.oe) {
        name += 'o';
    }
    if ((info.flags & FlagLk) != 0 && instruction.lk) {
        name += 'l';
    }
    if ((info.flags & FlagAa) != 0 && instruction.aa) {
        name += 'a';
    }
    if ((info.flags & FlagRc) != 0 && instruction.rc) {
        name += '.';
    }
    return name;
}

} // namespace wiiu::cpu
