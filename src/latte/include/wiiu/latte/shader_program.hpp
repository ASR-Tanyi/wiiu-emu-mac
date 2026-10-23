// Phase 8.6 — a shader program decoded: its control flow, and what is in
// the clauses that control flow runs.
//
// The decompiler's second stage. The first (control_flow.hpp) found every
// clause and checked it lies inside the program; this reads what each
// holds. So far that is the fetch clauses. A vertex fetch clause is what a
// fetch shader is made of, the program GX2 writes to read a draw's
// attributes (8.1) -- each fetch saying which attribute buffer to read, at
// which index and offset, in what format and byte order, and where in the
// vertex shader's registers the value goes. A texture fetch clause is how a
// shader samples: each fetch saying what to do (SAMPLE and its forms, a
// gradient, a size query), with which resource and sampler, from which
// register's coordinates, into which register's components. An ALU clause
// is groups of up to five instructions -- one each for the X, Y, Z and W
// units and the transcendental T, the last marked LAST -- each followed by
// the literal constants its instructions read, two to a 64-bit slot.
//
// Which unit an instruction runs on is decaf's rule, which Cemu's shares:
// one only the T unit runs goes there; otherwise its destination channel's
// unit, or T if an earlier instruction of the group has that one -- unless
// only the vector units run it, when two to one unit is a collision. The
// literal count is one more than the highest channel any instruction reads
// a literal at.
//
// An ALU group is refused if an instruction is not one decaf's table names,
// two instructions want one unit, it runs past five instructions or its
// clause, or a field holds a value decaf's disassembler aborts on (a bank
// swizzle past the unit's six or four, PRED_SEL 1).
//
// A vertex fetch is refused if it is not one R700 has: FETCH or SEMANTIC.
// A texture fetch cannot be: its five bits all name an R700 instruction, the
// first two being the vertex fetches, which a texture clause may hold in a
// vertex fetch's words. The other fields are the later stages' to make
// sense of.
//
// `disassemble` prints a program as decaf's disassembler does, each clause's
// instructions beneath the instruction that runs it, numbered in one count
// across the program. Its text for a fetch is decaf's exactly, a vertex
// fetch in a texture clause printed as a vertex fetch, as decaf does.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/control_flow.hpp>
#include <wiiu/latte/microcode.hpp>

#include <span>
#include <string>
#include <string_view>
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

/// One texture fetch: its three words (the fourth is padding).
struct TextureFetch {
    u32 word0 = 0;
    u32 word1 = 0;
    u32 word2 = 0;

    [[nodiscard]] TexWord0 tex0() const noexcept { return {.raw = word0}; }

    [[nodiscard]] TexWord1 tex1() const noexcept { return {.raw = word1}; }

    [[nodiscard]] TexWord2 tex2() const noexcept { return {.raw = word2}; }

    [[nodiscard]] TexInst inst() const noexcept { return tex0().tex_inst; }

    /// VTX_FETCH or VTX_SEMANTIC: a vertex fetch's words, in a texture
    /// clause.
    [[nodiscard]] bool is_vertex_fetch() const noexcept {
        return inst() == TexInst::VtxFetch || inst() == TexInst::VtxSemantic;
    }

    [[nodiscard]] VertexFetch as_vertex_fetch() const noexcept {
        return {.word0 = word0, .word1 = word1, .word2 = word2};
    }

    /// LOD_BIAS: seven bits, signed, three of them fractional.
    [[nodiscard]] f32 lod_bias() const noexcept;

    /// OFFSET_X, _Y and _Z: five bits each, signed, one fractional.
    [[nodiscard]] f32 offset(u32 axis) const noexcept;
};

/// The unit an ALU instruction runs on.
enum class AluUnit : u8 {
    X,
    Y,
    Z,
    W,
    T,
};

/// One ALU instruction: its two words, what it is, and its unit.
struct AluInstruction {
    u32 word0 = 0;
    u32 word1 = 0;
    AluOp op;
    AluUnit unit = AluUnit::X;

    [[nodiscard]] AluWord0 alu0() const noexcept { return {.raw = word0}; }

    [[nodiscard]] AluWord1 alu1() const noexcept { return {.raw = word1}; }

    [[nodiscard]] AluWord1Op2 op2() const noexcept { return {.raw = word1}; }

    [[nodiscard]] AluWord1Op3 op3() const noexcept { return {.raw = word1}; }

    [[nodiscard]] bool is_op3() const noexcept { return alu1().encoding != 0; }
};

/// An ALU instruction group, and the literals after it.
struct AluGroup {
    std::vector<AluInstruction> instructions;
    std::vector<u32> literals;
};

/// A clause, and what is decoded of it.
struct DecodedClause {
    Clause clause;
    std::vector<VertexFetch> vertex_fetches;   ///< a VTX clause's, in order
    std::vector<TextureFetch> texture_fetches; ///< a TEX clause's, in order
    std::vector<AluGroup> alu_groups;          ///< an ALU clause's, in order
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

/// TEX_INST's name, as decaf spells it: "SAMPLE", "SAMPLE_C_LZ".
[[nodiscard]] std::string_view tex_inst_name(TexInst inst) noexcept;

/// One texture fetch, as decaf's disassembler prints it: "SAMPLE
/// R1.xyzw, R0.xy0x, t0, s0", the name padded to fifteen, then the LOD bias,
/// the flags, the coordinates not normalised and the offsets, where set. A
/// vertex fetch's words are printed as a vertex fetch.
[[nodiscard]] std::string disassemble(const TextureFetch& fetch);

/// One ALU instruction, as decaf's disassembler prints it: "MULADD
/// R1.x, R0.x, KC0[0].x, (0x3F800000, 1)", the name -- with an OP2's output
/// modifier -- padded to fifteen, the destination ("____" for an OP2 that
/// writes nothing), the sources, and the flags. `group` is the group's
/// number in the program's count, which PV and PS name the one before;
/// `literals` the group's.
[[nodiscard]] std::string
disassemble(const AluInstruction& instruction, u32 group, std::span<const u32> literals);

/// The whole program, each clause's contents beneath the instruction that
/// runs it.
[[nodiscard]] std::string disassemble(const ShaderProgram& program);

} // namespace wiiu::latte
