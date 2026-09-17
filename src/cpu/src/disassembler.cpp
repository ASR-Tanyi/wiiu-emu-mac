#include <wiiu/cpu/disassembler.hpp>

#include <fmt/format.h>

#include <array>
#include <string_view>

namespace wiiu::cpu {
namespace {

struct Formatter {
    const DisassemblyOptions& options;

    [[nodiscard]] std::string gpr(u32 number) const {
        return options.register_names ? fmt::format("r{}", number) : fmt::format("{}", number);
    }

    [[nodiscard]] std::string fpr(u32 number) const {
        return options.register_names ? fmt::format("f{}", number) : fmt::format("{}", number);
    }

    [[nodiscard]] static std::string hex_or_dec(s32 value) {
        // Negative displacements read better in decimal; large positive
        // constants read better in hex. GNU as accepts both.
        return fmt::format("{}", value);
    }
};

struct SprName {
    u32 number;
    std::string_view name;
};

constexpr std::array kSprNames{
    SprName{1, "xer"},
    SprName{8, "lr"},
    SprName{9, "ctr"},
    SprName{18, "dsisr"},
    SprName{19, "dar"},
    SprName{22, "dec"},
    SprName{25, "sdr1"},
    SprName{26, "srr0"},
    SprName{27, "srr1"},
    SprName{272, "sprg0"},
    SprName{273, "sprg1"},
    SprName{274, "sprg2"},
    SprName{275, "sprg3"},
    SprName{284, "tbl"},
    SprName{285, "tbu"},
    SprName{287, "pvr"},
    SprName{912, "gqr0"},
    SprName{913, "gqr1"},
    SprName{914, "gqr2"},
    SprName{915, "gqr3"},
    SprName{916, "gqr4"},
    SprName{917, "gqr5"},
    SprName{918, "gqr6"},
    SprName{919, "gqr7"},
    SprName{920, "hid2"},
    SprName{921, "wpar"},
    SprName{922, "dmau"},
    SprName{923, "dmal"},
    SprName{947, "scr"},
    SprName{1008, "hid0"},
    SprName{1009, "hid1"},
    SprName{1011, "hid4"},
    // Block address translation, in upper/lower pairs (Phase 3.4).
    SprName{528, "ibat0u"},
    SprName{529, "ibat0l"},
    SprName{530, "ibat1u"},
    SprName{531, "ibat1l"},
    SprName{532, "ibat2u"},
    SprName{533, "ibat2l"},
    SprName{534, "ibat3u"},
    SprName{535, "ibat3l"},
    SprName{536, "dbat0u"},
    SprName{537, "dbat0l"},
    SprName{538, "dbat1u"},
    SprName{539, "dbat1l"},
    SprName{540, "dbat2u"},
    SprName{541, "dbat2l"},
    SprName{542, "dbat3u"},
    SprName{543, "dbat3l"},
    SprName{560, "ibat4u"},
    SprName{561, "ibat4l"},
    SprName{562, "ibat5u"},
    SprName{563, "ibat5l"},
    SprName{564, "ibat6u"},
    SprName{565, "ibat6l"},
    SprName{566, "ibat7u"},
    SprName{567, "ibat7l"},
    SprName{568, "dbat4u"},
    SprName{569, "dbat4l"},
    SprName{570, "dbat5u"},
    SprName{571, "dbat5l"},
    SprName{572, "dbat6u"},
    SprName{573, "dbat6l"},
    SprName{574, "dbat7u"},
    SprName{575, "dbat7l"},
};

[[nodiscard]] std::string branch_target(ppcaddr_t address,
                                        s32 displacement,
                                        bool absolute,
                                        const DisassemblyOptions& options);

/// Simplified mnemonics, as GNU prints them. Only the ones that actually turn
/// up: a disassembler that rewrites everything is harder to trust than one
/// that rewrites the handful of idioms a compiler emits constantly.
[[nodiscard]] std::string try_simplified(Opcode id,
                                         Instruction inst,
                                         ppcaddr_t address,
                                         const Formatter& fmt_ctx,
                                         const DisassemblyOptions& options) {
    switch (id) {
        case Opcode::Ori:
            // ori 0,0,0 is the canonical nop.
            if (inst.rs == 0 && inst.ra == 0 && inst.uimm == 0) {
                return "nop";
            }
            if (inst.uimm == 0) {
                return fmt::format("mr {},{}", fmt_ctx.gpr(inst.ra), fmt_ctx.gpr(inst.rs));
            }
            break;

        case Opcode::Addi:
            if (inst.ra == 0) {
                return fmt::format("li {},{}", fmt_ctx.gpr(inst.rd), inst.simm.value());
            }
            break;

        case Opcode::Addis:
            if (inst.ra == 0) {
                return fmt::format("lis {},{}", fmt_ctx.gpr(inst.rd), inst.simm.value());
            }
            break;

        case Opcode::Or:
            // or rA,rS,rS is a register move.
            if (inst.rs == inst.rb && !inst.rc) {
                return fmt::format("mr {},{}", fmt_ctx.gpr(inst.ra), fmt_ctx.gpr(inst.rs));
            }
            break;

        case Opcode::Nor:
            if (inst.rs == inst.rb && !inst.rc) {
                return fmt::format("not {},{}", fmt_ctx.gpr(inst.ra), fmt_ctx.gpr(inst.rs));
            }
            break;

        case Opcode::Bclr:
            // BO 20 is "branch always".
            if (inst.bo == 20 && inst.bi == 0) {
                return inst.lk ? "blrl" : "blr";
            }
            break;

        case Opcode::Bcctr:
            if (inst.bo == 20 && inst.bi == 0) {
                return inst.lk ? "bctrl" : "bctr";
            }
            break;

        case Opcode::Mfspr:
            switch (inst.spr_number()) {
                case 8:
                    return fmt::format("mflr {}", fmt_ctx.gpr(inst.rd));
                case 9:
                    return fmt::format("mfctr {}", fmt_ctx.gpr(inst.rd));
                case 1:
                    return fmt::format("mfxer {}", fmt_ctx.gpr(inst.rd));
                default:
                    break;
            }
            break;

        // 32-bit compares are spelled cmpw/cmplw; the L bit that selects
        // 64-bit operands does not exist on this core.
        case Opcode::Cmpi:
            if (!inst.l) {
                return inst.crfd == 0
                           ? fmt::format("cmpwi {},{}", fmt_ctx.gpr(inst.ra), inst.simm.value())
                           : fmt::format("cmpwi cr{},{},{}",
                                         inst.crfd.value(),
                                         fmt_ctx.gpr(inst.ra),
                                         inst.simm.value());
            }
            break;

        case Opcode::Cmpli:
            if (!inst.l) {
                return inst.crfd == 0
                           ? fmt::format("cmplwi {},{}", fmt_ctx.gpr(inst.ra), inst.uimm.value())
                           : fmt::format("cmplwi cr{},{},{}",
                                         inst.crfd.value(),
                                         fmt_ctx.gpr(inst.ra),
                                         inst.uimm.value());
            }
            break;

        case Opcode::Cmp:
            if (!inst.l) {
                return inst.crfd == 0
                           ? fmt::format("cmpw {},{}", fmt_ctx.gpr(inst.ra), fmt_ctx.gpr(inst.rb))
                           : fmt::format("cmpw cr{},{},{}",
                                         inst.crfd.value(),
                                         fmt_ctx.gpr(inst.ra),
                                         fmt_ctx.gpr(inst.rb));
            }
            break;

        case Opcode::Cmpl:
            if (!inst.l) {
                return inst.crfd == 0
                           ? fmt::format("cmplw {},{}", fmt_ctx.gpr(inst.ra), fmt_ctx.gpr(inst.rb))
                           : fmt::format("cmplw cr{},{},{}",
                                         inst.crfd.value(),
                                         fmt_ctx.gpr(inst.ra),
                                         fmt_ctx.gpr(inst.rb));
            }
            break;

        case Opcode::Bc: {
            // The conditional branch mnemonics. BO 12 branches when the CR bit
            // is set, BO 4 when it is clear; BI's low two bits pick which bit
            // of the field, and the rest names the field.
            static constexpr std::array<std::string_view, 4> kTrue{"blt", "bgt", "beq", "bso"};
            static constexpr std::array<std::string_view, 4> kFalse{"bge", "ble", "bne", "bns"};

            const u32 bo = inst.bo;
            if (bo != 12 && bo != 4) {
                break; // counter forms and the rest stay in their raw shape
            }

            const u32 field = inst.bi.value() / 4;
            const std::string_view name =
                (bo == 12) ? kTrue[inst.bi.value() % 4] : kFalse[inst.bi.value() % 4];

            std::string mnemonic{name};
            if (inst.lk) {
                mnemonic += 'l';
            }
            if (inst.aa) {
                mnemonic += 'a';
            }

            const std::string target =
                branch_target(address, inst.bd.value() * 4, inst.aa, options);

            return field == 0 ? fmt::format("{} {}", mnemonic, target)
                              : fmt::format("{} cr{},{}", mnemonic, field, target);
        }

        case Opcode::Mtspr:
            switch (inst.spr_number()) {
                case 8:
                    return fmt::format("mtlr {}", fmt_ctx.gpr(inst.rs));
                case 9:
                    return fmt::format("mtctr {}", fmt_ctx.gpr(inst.rs));
                case 1:
                    return fmt::format("mtxer {}", fmt_ctx.gpr(inst.rs));
                default:
                    break;
            }
            break;

        default:
            break;
    }

    static_cast<void>(address);
    return {};
}

[[nodiscard]] std::string branch_target(ppcaddr_t address,
                                        s32 displacement,
                                        bool absolute,
                                        const DisassemblyOptions& options) {
    const auto offset = static_cast<u32>(displacement);
    const ppcaddr_t target = absolute ? offset : address + offset;

    if (!options.absolute_branch_targets && !absolute) {
        return fmt::format(".{:+}", displacement);
    }
    return fmt::format("{:#x}", target);
}

} // namespace

std::string_view spr_name(u32 number) noexcept {
    for (const SprName& entry : kSprNames) {
        if (entry.number == number) {
            return entry.name;
        }
    }
    return {};
}

std::string disassemble(Instruction inst, ppcaddr_t address, const DisassemblyOptions& options) {
    const Opcode id = decode(inst);
    if (id == Opcode::Invalid) {
        return fmt::format(".long {:#010x}", inst.raw);
    }

    const Formatter formatter{options};

    if (options.simplified) {
        std::string simplified = try_simplified(id, inst, address, formatter, options);
        if (!simplified.empty()) {
            return simplified;
        }
    }

    const OpcodeInfo& info = opcode_info(id);
    const std::string name = decorated_name(id, inst);

    switch (info.form) {
        case Form::XO:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.gpr(inst.rd),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb));
        case Form::XO_AB:
            return fmt::format("{} {},{}", name, formatter.gpr(inst.rd), formatter.gpr(inst.ra));

        case Form::DArith:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.gpr(inst.rd),
                               formatter.gpr(inst.ra),
                               inst.simm.value());
        case Form::DLogic:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rs),
                               inst.uimm.value());

        case Form::DLoad:
            return fmt::format("{} {},{}({})",
                               name,
                               formatter.gpr(inst.rd),
                               inst.d.value(),
                               formatter.gpr(inst.ra));
        case Form::DStore:
            return fmt::format("{} {},{}({})",
                               name,
                               formatter.gpr(inst.rs),
                               inst.d.value(),
                               formatter.gpr(inst.ra));
        case Form::DLoadFp:
            return fmt::format("{} {},{}({})",
                               name,
                               formatter.fpr(inst.frd),
                               inst.d.value(),
                               formatter.gpr(inst.ra));
        case Form::DStoreFp:
            return fmt::format("{} {},{}({})",
                               name,
                               formatter.fpr(inst.frs),
                               inst.d.value(),
                               formatter.gpr(inst.ra));

        case Form::DCmp:
            return fmt::format("{} {},{},{},{}",
                               name,
                               inst.crfd.value(),
                               static_cast<u32>(inst.l.value()),
                               formatter.gpr(inst.ra),
                               inst.simm.value());
        case Form::DTrap:
            return fmt::format(
                "{} {},{},{}", name, inst.to.value(), formatter.gpr(inst.ra), inst.simm.value());

        case Form::XLogic:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rs),
                               formatter.gpr(inst.rb));
        case Form::XSourceA:
            return fmt::format("{} {},{}", name, formatter.gpr(inst.ra), formatter.gpr(inst.rs));
        case Form::XShift:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rs),
                               inst.sh.value());
        case Form::XCmp:
            return fmt::format("{} {},{},{},{}",
                               name,
                               inst.crfd.value(),
                               static_cast<u32>(inst.l.value()),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb));
        case Form::XLoad:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.gpr(inst.rd),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb));
        case Form::XStore:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.gpr(inst.rs),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb));
        case Form::XLoadFp:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.fpr(inst.frd),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb));
        case Form::XStoreFp:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.fpr(inst.frs),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb));
        case Form::XCache:
            return fmt::format("{} {},{}", name, formatter.gpr(inst.ra), formatter.gpr(inst.rb));
        case Form::XTrap:
            return fmt::format("{} {},{},{}",
                               name,
                               inst.to.value(),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb));
        case Form::XNone:
        case Form::Sc:
            return name;
        case Form::XMsrFrom:
            return fmt::format("{} {}", name, formatter.gpr(inst.rd));
        case Form::XMsrTo:
            return fmt::format("{} {}", name, formatter.gpr(inst.rs));

        case Form::XSegFrom:
            return fmt::format("{} {},{}", name, formatter.gpr(inst.rd), inst.sr_field.value());
        case Form::XSegTo:
            return fmt::format("{} {},{}", name, inst.sr_field.value(), formatter.gpr(inst.rs));
        case Form::XSegFromReg:
            return fmt::format("{} {},{}", name, formatter.gpr(inst.rd), formatter.gpr(inst.rb));
        case Form::XSegToReg:
            return fmt::format("{} {},{}", name, formatter.gpr(inst.rs), formatter.gpr(inst.rb));
        case Form::XTlb:
            return fmt::format("{} {}", name, formatter.gpr(inst.rb));

        case Form::XlBranch:
            return fmt::format("{} {},{}", name, inst.bo.value(), inst.bi.value());
        case Form::XlCr:
            return fmt::format(
                "{} {},{},{}", name, inst.crbd.value(), inst.crba.value(), inst.crbb.value());
        case Form::XlMcrf:
            return fmt::format("{} {},{}", name, inst.crfd.value(), inst.crfs.value());

        case Form::XfxSprFrom: {
            // An SPR is a register, so its symbolic name follows the same
            // option as r3/f3 does. Assemblers want the number unless told
            // otherwise.
            const std::string_view special =
                options.register_names ? spr_name(inst.spr_number()) : std::string_view{};
            return fmt::format("{} {},{}",
                               name,
                               formatter.gpr(inst.rd),
                               special.empty() ? fmt::format("{}", inst.spr_number())
                                               : std::string{special});
        }
        case Form::XfxSprTo: {
            const std::string_view special =
                options.register_names ? spr_name(inst.spr_number()) : std::string_view{};
            return fmt::format("{} {},{}",
                               name,
                               special.empty() ? fmt::format("{}", inst.spr_number())
                                               : std::string{special},
                               formatter.gpr(inst.rs));
        }
        case Form::XfxMftb:
            return fmt::format("{} {},{}", name, formatter.gpr(inst.rd), inst.spr_number());
        case Form::XfxMfcr:
            return fmt::format("{} {}", name, formatter.gpr(inst.rd));
        case Form::XfxMtcrf:
            return fmt::format("{} {},{}", name, inst.crm.value(), formatter.gpr(inst.rs));

        case Form::MImm:
            return fmt::format("{} {},{},{},{},{}",
                               name,
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rs),
                               inst.sh.value(),
                               inst.mb.value(),
                               inst.me.value());
        case Form::MReg:
            return fmt::format("{} {},{},{},{},{}",
                               name,
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rs),
                               formatter.gpr(inst.rb),
                               inst.mb.value(),
                               inst.me.value());

        case Form::IBranch:
            return fmt::format(
                "{} {}", name, branch_target(address, inst.li.value() * 4, inst.aa, options));
        case Form::BBranch:
            return fmt::format("{} {},{},{}",
                               name,
                               inst.bo.value(),
                               inst.bi.value(),
                               branch_target(address, inst.bd.value() * 4, inst.aa, options));

        case Form::Aabc:
            return fmt::format("{} {},{},{},{}",
                               name,
                               formatter.fpr(inst.frd),
                               formatter.fpr(inst.fra),
                               formatter.fpr(inst.frc),
                               formatter.fpr(inst.frb));
        case Form::Aab:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.fpr(inst.frd),
                               formatter.fpr(inst.fra),
                               formatter.fpr(inst.frb));
        case Form::Aac:
            return fmt::format("{} {},{},{}",
                               name,
                               formatter.fpr(inst.frd),
                               formatter.fpr(inst.fra),
                               formatter.fpr(inst.frc));
        case Form::Ab:
        case Form::XFp2:
            return fmt::format("{} {},{}", name, formatter.fpr(inst.frd), formatter.fpr(inst.frb));
        case Form::XFpCmp:
            return fmt::format("{} {},{},{}",
                               name,
                               inst.crfd.value(),
                               formatter.fpr(inst.fra),
                               formatter.fpr(inst.frb));
        case Form::XMffs:
            return fmt::format("{} {}", name, formatter.fpr(inst.frd));
        case Form::XMtfsf:
            return fmt::format("{} {},{}", name, inst.fm.value(), formatter.fpr(inst.frb));
        case Form::XMcrfs:
            return fmt::format("{} {},{}", name, inst.crfd.value(), inst.crfs.value());
        case Form::XMtfsb:
            return fmt::format("{} {}", name, inst.crbd.value());
        case Form::XMtfsfi:
            return fmt::format("{} {},{}", name, inst.crfd.value(), inst.imm4.value());

        case Form::PsQuantized:
            return fmt::format("{} {},{}({}),{},{}",
                               name,
                               formatter.fpr(inst.frd),
                               inst.ps_d.value(),
                               formatter.gpr(inst.ra),
                               static_cast<u32>(inst.ps_w.value()),
                               inst.ps_i.value());
        case Form::PsQuantizedX:
            return fmt::format("{} {},{},{},{},{}",
                               name,
                               formatter.fpr(inst.frd),
                               formatter.gpr(inst.ra),
                               formatter.gpr(inst.rb),
                               static_cast<u32>(inst.ps_w.value()),
                               inst.ps_i.value());

        case Form::Invalid:
            break;
    }

    return fmt::format(".long {:#010x}", inst.raw);
}

} // namespace wiiu::cpu
