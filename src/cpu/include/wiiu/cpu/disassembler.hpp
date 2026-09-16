// Phase 4.3 — disassembly.
//
// Written before the interpreter on purpose. Every CPU bug after this point is
// reported as an address and a word; without a disassembler, chasing one means
// decoding by hand.
//
// Output follows GNU assembler syntax closely enough that it reassembles to
// the same encoding, which is how it is tested.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/instruction.hpp>
#include <wiiu/cpu/opcodes.hpp>

#include <string>

namespace wiiu::cpu {

struct DisassemblyOptions {
    /// Use simplified mnemonics where one applies: `li` for `addi rD,0,n`,
    /// `blr` for `bclr 20,0`, `nop` for `ori 0,0,0`. Easier to read; turn off
    /// when the exact encoding matters.
    bool simplified = true;

    /// Prefix register numbers with 'r' and 'f'. GNU as needs -mregnames for
    /// this spelling, so the round-trip test turns it off.
    bool register_names = true;

    /// Branch targets as absolute addresses rather than displacements.
    bool absolute_branch_targets = true;
};

/// Disassemble one instruction. `address` is where the instruction lives, used
/// to resolve relative branch targets.
[[nodiscard]] std::string
disassemble(Instruction instruction, ppcaddr_t address, const DisassemblyOptions& options = {});

[[nodiscard]] inline std::string
disassemble(u32 word, ppcaddr_t address = 0, const DisassemblyOptions& options = {}) {
    return disassemble(Instruction{.raw = word}, address, options);
}

/// The SPR's conventional name, or an empty view if it has none.
[[nodiscard]] std::string_view spr_name(u32 number) noexcept;

} // namespace wiiu::cpu
