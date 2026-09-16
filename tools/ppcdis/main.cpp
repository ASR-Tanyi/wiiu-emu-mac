// ppcdis — disassemble PowerPC words.
//
//   ppcdis 7c832378 38600001        disassemble the given words
//   ppcdis --address 0x2000000 ...  resolve branch targets from this address
//   ppcdis --raw ...                no simplified mnemonics, no r/f prefixes
//   ppcdis --corpus                 emit one synthesised encoding per table
//                                   entry, for the round-trip check
//
// The corpus mode exists so that the disassembler can be validated against a
// real assembler: every line it prints should reassemble to the word it came
// from. See scripts/ppc_roundtrip.py.

#include <wiiu/cpu/disassembler.hpp>
#include <wiiu/cpu/opcodes.hpp>

#include <fmt/format.h>

#include <charconv>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace wiiu;
using namespace wiiu::cpu;

/// Plant plausible operands in a pattern so the encoding is a legal
/// instruction rather than a bare opcode with zero fields.
[[nodiscard]] u32 synthesize(const OpcodeInfo& info) {
    Instruction inst{.raw = info.pattern};

    const auto set = [&inst](auto field, u32 value) {
        Instruction scratch{.raw = inst.raw};
        scratch.*field = value;
        inst.raw = scratch.raw;
    };
    static_cast<void>(set);

    switch (info.form) {
        case Form::XO:
        case Form::XLoad:
        case Form::XStore:
        case Form::XLogic:
            inst.rd = 3;
            inst.ra = 4;
            inst.rb = 5;
            break;
        case Form::XO_AB:
        case Form::XSourceA:
            inst.rd = 3;
            inst.ra = 4;
            break;
        case Form::DArith:
        case Form::DLogic:
        case Form::DLoad:
        case Form::DStore:
            // lmw loads rD through r31, and rA must not be one of them, so it
            // needs a base register below the destination rather than above.
            inst.rd = (info.id == Opcode::Lmw) ? 28 : 3;
            inst.ra = 4;
            inst.d = 8;
            break;
        case Form::DLoadFp:
        case Form::DStoreFp:
            inst.frd = 3;
            inst.ra = 4;
            inst.d = 8;
            break;
        case Form::DCmp:
            inst.crfd = 1;
            inst.ra = 4;
            inst.simm = 7;
            break;
        case Form::DTrap:
            inst.to = 12;
            inst.ra = 4;
            inst.simm = 7;
            break;
        case Form::XShift:
            inst.rd = 3;
            inst.ra = 4;
            inst.sh = 7;
            break;
        case Form::XCmp:
            inst.crfd = 1;
            inst.ra = 4;
            inst.rb = 5;
            break;
        case Form::XLoadFp:
        case Form::XStoreFp:
            inst.frd = 3;
            inst.ra = 4;
            inst.rb = 5;
            break;
        case Form::XCache:
            inst.ra = 4;
            inst.rb = 5;
            break;
        case Form::XTrap:
            inst.to = 12;
            inst.ra = 4;
            inst.rb = 5;
            break;
        case Form::XMsrFrom:
        case Form::XMsrTo:
        case Form::XfxMfcr:
            inst.rd = 3;
            break;
        case Form::XlBranch:
            inst.bo = 12;
            inst.bi = 2;
            break;
        case Form::XlCr:
            inst.crbd = 1;
            inst.crba = 2;
            inst.crbb = 3;
            break;
        case Form::XlMcrf:
        case Form::XMcrfs:
            inst.crfd = 1;
            inst.crfs = 2;
            break;
        case Form::XfxSprFrom:
        case Form::XfxSprTo:
            inst.rd = 3;
            // SPR 9 (CTR) with the halves swapped as the encoding wants.
            inst.spr_field = ((9U & 0x1F) << 5) | (9U >> 5);
            break;
        case Form::XfxMftb:
            inst.rd = 3;
            inst.spr_field = ((268U & 0x1F) << 5) | (268U >> 5);
            break;
        case Form::XfxMtcrf:
            inst.crm = 0x80;
            inst.rs = 3;
            break;
        case Form::MImm:
            inst.ra = 4;
            inst.rs = 3;
            inst.sh = 7;
            inst.mb = 8;
            inst.me = 20;
            break;
        case Form::MReg:
            inst.ra = 4;
            inst.rs = 3;
            inst.rb = 5;
            inst.mb = 8;
            inst.me = 20;
            break;
        case Form::IBranch:
            inst.li = 8;
            break;
        case Form::BBranch:
            inst.bo = 12;
            inst.bi = 2;
            inst.bd = 4;
            break;
        case Form::Aabc:
            inst.frd = 1;
            inst.fra = 2;
            inst.frb = 3;
            inst.frc = 4;
            break;
        case Form::Aab:
            inst.frd = 1;
            inst.fra = 2;
            inst.frb = 3;
            break;
        case Form::Aac:
            inst.frd = 1;
            inst.fra = 2;
            inst.frc = 4;
            break;
        case Form::Ab:
        case Form::XFp2:
            inst.frd = 1;
            inst.frb = 3;
            break;
        case Form::XFpCmp:
            inst.crfd = 1;
            inst.fra = 2;
            inst.frb = 3;
            break;
        case Form::XMffs:
            inst.frd = 1;
            break;
        case Form::XMtfsf:
            inst.fm = 0xFF;
            inst.frb = 3;
            break;
        case Form::PsQuantized:
            inst.frd = 1;
            inst.ra = 4;
            inst.ps_d = 8;
            inst.ps_w = false;
            inst.ps_i = 2;
            break;
        case Form::PsQuantizedX:
            inst.frd = 1;
            inst.ra = 4;
            inst.rb = 5;
            inst.ps_w = false;
            inst.ps_i = 2;
            break;
        case Form::XNone:
        case Form::Sc:
        case Form::Invalid:
            break;
    }

    return inst.raw;
}

[[nodiscard]] bool parse_hex(std::string_view text, u32& out) {
    if (text.starts_with("0x") || text.starts_with("0X")) {
        text.remove_prefix(2);
    }
    const auto* const end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, out, 16);
    return result.ec == std::errc{} && result.ptr == end;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    bool corpus = false;
    bool raw = false;
    ppcaddr_t address = 0;
    std::vector<u32> words;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--corpus") {
            corpus = true;
        } else if (arg == "--raw") {
            raw = true;
        } else if (arg == "--address" && i + 1 < args.size()) {
            if (!parse_hex(args[++i], address)) {
                fmt::print(stderr, "ppcdis: bad address\n");
                return 2;
            }
        } else {
            u32 word = 0;
            if (!parse_hex(arg, word)) {
                fmt::print(stderr, "ppcdis: '{}' is not a hex word\n", arg);
                return 2;
            }
            words.push_back(word);
        }
    }

    const DisassemblyOptions options{
        .simplified = !raw && !corpus,
        .register_names = !raw && !corpus,
        .absolute_branch_targets = true,
    };

    if (corpus) {
        for (const OpcodeInfo& info : opcode_table()) {
            const u32 word = synthesize(info);
            fmt::print("{:08x}\t{}\t{}\n", word, info.name, disassemble(word, 0, options));
        }
        return 0;
    }

    if (words.empty()) {
        fmt::print(stderr,
                   "usage: ppcdis [--raw] [--address HEX] WORD...\n"
                   "       ppcdis --corpus\n");
        return 2;
    }

    for (std::size_t i = 0; i < words.size(); ++i) {
        const ppcaddr_t at = address + static_cast<u32>(i * 4);
        fmt::print("{:08x}:  {:08x}  {}\n", at, words[i], disassemble(words[i], at, options));
    }
    return 0;
}
