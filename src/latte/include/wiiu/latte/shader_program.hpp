// Phase 8.6 — a shader program decoded: its control flow, and what is in
// the clauses that control flow runs.
//
// The decompiler's second stage. The first (control_flow.hpp) found every
// clause and checked it lies inside the program; this reads what each
// holds. So far that is the vertex fetch clauses -- what a fetch shader is
// made of, the program GX2 writes to read a draw's attributes (8.1) -- each
// fetch saying which attribute buffer to read, at which index and offset, in
// what format and byte order, and where in the vertex shader's registers
// the value goes. Texture fetch clauses and ALU clauses follow; until then
// they are listed, as the control flow names them, and empty.
//
// A vertex fetch is refused if it is not one R700 has: FETCH or SEMANTIC.
// Its other fields are the emitter's to make sense of.
//
// `disassemble` prints a program as decaf's disassembler does, each clause's
// instructions beneath the instruction that runs it, numbered in one count
// across the program. Its text for a fetch is decaf's exactly.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/control_flow.hpp>
#include <wiiu/latte/microcode.hpp>

#include <span>
#include <string>
#include <vector>

namespace wiiu::latte {

/// One vertex fetch: its three words (the fourth is padding).
struct VertexFetch {
    u32 word0 = 0;
    u32 word1 = 0;
    u32 word2 = 0;

    [[nodiscard]] VtxWord0 fetch0() const noexcept { return {.raw = word0}; }

    [[nodiscard]] VtxWord1 fetch1() const noexcept { return {.raw = word1}; }

    [[nodiscard]] VtxWord2 fetch2() const noexcept { return {.raw = word2}; }

    /// A semantic fetch writes the register SQ_VTX_SEMANTIC_n maps its
    /// semantic to; a plain one, the register it names.
    [[nodiscard]] bool is_semantic() const noexcept {
        return fetch0().vtx_inst == VtxInst::Semantic;
    }
};

/// A clause, and what is decoded of it.
struct DecodedClause {
    Clause clause;
    std::vector<VertexFetch> vertex_fetches; ///< a VTX clause's, in order
};

struct ShaderProgram {
    ControlFlow control_flow;

    /// `control_flow.clauses`, in that order, each with its contents.
    std::vector<DecodedClause> clauses;
};

/// Decode a program: its control flow (`decode_control_flow`), then its
/// clauses. False, with `error` saying where, for a program the later
/// stages could not trust.
[[nodiscard]] bool decode_program(std::span<const u32> program,
                                  ProgramShape shape,
                                  ShaderProgram& out,
                                  std::string& error);

/// One vertex fetch, as decaf's disassembler prints it: "SEMANTIC
/// SEM[0].xyzw, R0.x, b160 FETCH_TYPE(VERTEX_DATA) FORMAT( 32 SCALED
/// UNSIGNED 0) MEGA(8) OFFSET(0) ENDIAN_SWAP(8IN16)", the name padded to
/// fifteen.
[[nodiscard]] std::string disassemble(const VertexFetch& fetch);

/// The whole program, each clause's contents beneath the instruction that
/// runs it.
[[nodiscard]] std::string disassemble(const ShaderProgram& program);

} // namespace wiiu::latte
