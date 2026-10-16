// Phase 8.6 — a shader's control-flow program, decoded.
//
// An R700 program starts with its control flow: 64-bit instructions, run in
// order, each either doing something itself -- an export, a jump, a loop,
// a call -- or running a *clause* elsewhere in the program: a run of ALU
// instruction groups, of texture fetches, or of vertex fetches
// (microcode.hpp has the words). The main program ends at the instruction
// with END_OF_PROGRAM set; a subroutine, and the fetch shader (which the
// vertex shader calls), at its first RETURN.
//
// This is the decompiler's first stage: it walks the control flow from the
// entry point, follows CALLs into their subroutines, and lists every clause
// the program runs and where it is -- the shape the later stages fill in.
// Clause contents are read by the next stage, shader_program.hpp. What it
// refuses, with a message that says where: running off the end of the
// program before the end, an instruction R700 does not have, a jump or a
// clause outside the program, and a subroutine that never returns. That is
// everything the later stages can rely on not having to check again.
//
// `disassemble` prints it in AMD's assembly syntax, decaf's spelling of it
// (its disassembler and latte-assembler), so a program can be read, and
// compared, as text. decaf's own example vertex shader is one of the tests.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/microcode.hpp>

#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wiiu::latte {

/// One control-flow instruction: its two words, and where it was.
struct CfInstruction {
    u32 index = 0; ///< in 64-bit words from the start of the program
    u32 word0 = 0;
    u32 word1 = 0;

    [[nodiscard]] CfInstType type() const noexcept { return CfWord1{.raw = word1}.cf_inst_type; }

    /// The forms' words. Which applies is `type()`'s to say.
    [[nodiscard]] CfWord0 normal0() const noexcept { return {.raw = word0}; }

    [[nodiscard]] CfWord1 normal1() const noexcept { return {.raw = word1}; }

    [[nodiscard]] CfAluWord0 alu0() const noexcept { return {.raw = word0}; }

    [[nodiscard]] CfAluWord1 alu1() const noexcept { return {.raw = word1}; }

    [[nodiscard]] CfExportWord0 export0() const noexcept { return {.raw = word0}; }

    [[nodiscard]] CfExportWord1 export1() const noexcept { return {.raw = word1}; }

    [[nodiscard]] bool is_alu() const noexcept {
        return type() == CfInstType::Alu || type() == CfInstType::AluExtended;
    }

    /// Whether this is the common form's `inst`.
    [[nodiscard]] bool is(CfInst inst) const noexcept {
        return type() == CfInstType::Normal && normal1().cf_inst == inst;
    }

    /// END_OF_PROGRAM, which an ALU clause does not have.
    [[nodiscard]] bool ends_program() const noexcept;
};

/// What a clause holds.
enum class ClauseKind : u8 {
    Alu,
    Tex,
    Vtx, ///< VTX or VTX_TC
};

/// A clause, as the control flow names it.
struct Clause {
    ClauseKind kind = ClauseKind::Alu;
    u32 cf_index = 0; ///< the instruction that runs it
    u32 address = 0;  ///< in 64-bit words from the start of the program
    u32 count = 0;    ///< fetches; for ALU, 64-bit slots (instructions and literals)

    [[nodiscard]] u32 offset_bytes() const noexcept { return address * 8; }

    [[nodiscard]] u32 size_bytes() const noexcept {
        return count * (kind == ClauseKind::Alu ? 8U : 16U);
    }
};

/// A subroutine a CALL names: from its first instruction to its RETURN.
struct Subroutine {
    u32 address = 0;
    std::vector<CfInstruction> instructions;
};

/// Where a program's control flow ends.
enum class ProgramShape : u8 {
    Shader,      ///< at END_OF_PROGRAM: a vertex, pixel or geometry shader
    FetchShader, ///< at its RETURN, like a subroutine
};

struct ControlFlow {
    std::vector<CfInstruction> main;
    std::vector<Subroutine> subroutines; ///< in the order they are first called
    std::vector<Clause> clauses;         ///< the main program's, then each subroutine's
};

/// Decode a program's control flow from its words -- 32-bit, in host order,
/// as read little-endian out of memory. False, with `error` saying where,
/// for a program the later stages could not trust.
[[nodiscard]] bool decode_control_flow(std::span<const u32> program,
                                       ProgramShape shape,
                                       ControlFlow& out,
                                       std::string& error);

/// One instruction in AMD's syntax, as decaf's disassembler writes it --
/// "VTX_TC: ADDR(4) CNT(2)", "EXP_DONE: POS0, R2.xyzw" -- but for two
/// things: decaf prints no address for POP_JUMP, which has one, and this
/// does; and decaf refuses the four *_ACK instructions, which this prints
/// by name alone.
[[nodiscard]] std::string disassemble(const CfInstruction& instruction);

/// What goes beneath an instruction's line -- its clause's contents, when
/// they are decoded -- given the indent the line was printed at.
using ClauseLines = std::function<void(
    std::string& out, const CfInstruction& instruction, std::string_view indent)>;

/// The whole program: one numbered line an instruction, loops indented, the
/// main program ending in END_OF_PROGRAM, then each subroutine. `after`, if
/// given, adds lines beneath each instruction (shader_program.hpp's).
[[nodiscard]] std::string disassemble(const ControlFlow& flow, const ClauseLines& after = {});

/// CF_INST's name, as AMD and decaf spell it.
[[nodiscard]] std::string_view cf_inst_name(const CfInstruction& instruction) noexcept;

} // namespace wiiu::latte
