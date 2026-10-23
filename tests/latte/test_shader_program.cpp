// Phase 8.6 — a program decoded: the fetch clauses and the ALU clauses.
//
// What is guarded: a fetch shader a game ships (Breath of the Wild's, from
// Cemu's notes) decoded field by field and printed as decaf prints it; each
// of a fetch's fields in decaf's text, its quirks included; the one count
// decaf numbers fetches by, across clauses; and the refusal of a fetch R700
// does not have. And texture fetch clauses: a pixel shader that samples,
// decoded and printed; a texture fetch's every field in decaf's text, its
// signed fixed-point LOD bias and offsets included; and a vertex fetch in a
// texture clause printed as one. And ALU clauses: a pixel shader that
// computes, its groups split where LAST says, each instruction on the unit
// decaf gives it -- a transcendental-only one on T, and a second to a taken
// unit there too -- its literals after it, and the whole printed as decaf
// prints it; an ALU instruction's every field in decaf's text, a literal's
// float in fmt 8's fixed notation included; and the refusal of a group that
// is not one: an instruction not in the table, two to a unit, six to a group,
// a group or its literals past the clause, and the values decaf aborts on.
// That this text is decaf's for every field is a scratch harness's to say,
// as for the control flow (see plan.md, 8.6).

#include <wiiu/latte/shader_program.hpp>

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

[[nodiscard]] ShaderProgram decoded(const std::vector<u32>& words,
                                    ProgramShape shape = ProgramShape::FetchShader) {
    ShaderProgram program;
    std::string error;
    INFO(error);
    REQUIRE(decode_program(words, shape, program, error));
    return program;
}

[[nodiscard]] std::string text(u32 word0, u32 word1, u32 word2) {
    return disassemble(VertexFetch{.word0 = word0, .word1 = word1, .word2 = word2});
}

/// The words a game's fetch shader starts with: one VTX_TC clause of one
/// fetch, the RETURN, and the fetch -- 32_32 float, from attribute buffer 0,
/// into semantic 0 as XY01, big-endian.
const std::vector<u32> botw{
    0x00000002, 0x01800000, 0x00000000, 0x8A000000, 0x1C00A001, 0x27961000, 0x000A0000, 0};

} // namespace

TEST_CASE("a fetch shader a game ships, decoded", "[latte][shader][vtx]") {
    const ShaderProgram program = decoded(botw);
    REQUIRE(program.clauses.size() == 1);
    REQUIRE(program.clauses[0].vertex_fetches.size() == 1);
    const VertexFetch& fetch = program.clauses[0].vertex_fetches[0];

    CHECK(fetch.is_semantic());
    CHECK(fetch.fetch0().fetch_type == VtxFetchType::VertexData);
    CHECK(fetch.fetch0().buffer_id == 0xA0U); // attribute buffer 0
    CHECK(fetch.fetch0().src_gpr == 0U);      // the vertex index
    CHECK(fetch.fetch0().mega_fetch_count == 7U);
    CHECK(fetch.fetch1().semantic_id == 0U);
    CHECK(fetch.fetch1().dst_sel_z == Sel::Zero);
    CHECK(fetch.fetch1().dst_sel_w == Sel::One);
    CHECK(fetch.fetch1().data_format == 0x1EU); // FMT_32_32_FLOAT
    CHECK(fetch.fetch1().num_format_all == NumFormat::Scaled);
    CHECK(fetch.fetch2().endian_swap == Endian::Swap8In32);
    CHECK(fetch.fetch2().mega_fetch);

    CHECK(disassemble(program) ==
          "00 VTX_TC: ADDR(2) CNT(1) NO_BARRIER\n"
          "  0      SEMANTIC        SEM[0].xy01, R0.x, b160 FETCH_TYPE(VERTEX_DATA) "
          "FORMAT( 30 SCALED UNSIGNED 0) MEGA(8) OFFSET(0) ENDIAN_SWAP(8IN32)\n"
          "01 RETURN\n");
}

TEST_CASE("a fetch's fields, as decaf prints them", "[latte][shader][vtx]") {
    // A plain fetch into R5, relative, from R1's W, of instance data, with
    // the resource's own format, a whole quad, every flag and the fourth
    // byte swap.
    const u32 plain0 = 0U | (1U << 5) | (1U << 7) | (0xAAU << 8) | (1U << 16) | (1U << 23) |
                       (3U << 24) | (3U << 26);
    const u32 plain1 =
        5U | (1U << 7) | (0U << 9) | (1U << 12) | (7U << 15) | (6U << 18) | (1U << 21);
    const u32 plain2 = 12U | (3U << 16) | (1U << 18) | (1U << 20);
    CHECK(text(plain0, plain1, plain2) ==
          "FETCH           R5[AL].xy_?, R1[AL].w, b170 FETCH_TYPE(INSTANCE_DATA) MINI(4) "
          "OFFSET(12) WHOLE_QUAD ENDIAN_SWAP(3) CONST_BUF_NO_STRIDE ALT_CONST");

    // Nothing written, and decaf's spacing for a number format of 3.
    const u32 masked = (7U << 9) | (7U << 12) | (7U << 15) | (7U << 18);
    CHECK(text(0, masked | (0x22U << 22) | (3U << 28) | (1U << 30) | (1U << 31), 1U << 19) ==
          "FETCH           ____, R0.x, b0 FETCH_TYPE(VERTEX_DATA) FORMAT( 343 SIGNED 1) MEGA(1) "
          "OFFSET(0)");
}

TEST_CASE("fetches are numbered across the program", "[latte][shader][vtx]") {
    // Two clauses of two fetches each, at 32 and 64 bytes, after three
    // control-flow instructions and a pad.
    std::vector<u32> words(24, 0);
    words[0] = 4;
    words[1] = (3U << 23) | (1U << 10); // VTX_TC, two fetches
    words[2] = 8;
    words[3] = (3U << 23) | (1U << 10);
    words[5] = (0x14U << 23) | (1U << 31); // RETURN
    for (u32 fetch = 0; fetch < 4; ++fetch) {
        words[8 + (4 * fetch)] = 1U | (0xA0U << 8);         // SEMANTIC, buffer 0
        words[8 + (4 * fetch) + 1] = fetch | (0x688U << 9); // semantic n, XYZW
    }

    const ShaderProgram program = decoded(words);
    REQUIRE(program.clauses.size() == 2);
    const std::string listing = disassemble(program);
    CHECK(listing.find("  0      SEMANTIC        SEM[0]") != std::string::npos);
    CHECK(listing.find("  1      SEMANTIC        SEM[1]") != std::string::npos);
    CHECK(listing.find("  2      SEMANTIC") != std::string::npos);
    CHECK(listing.find("  3      SEMANTIC") != std::string::npos);
}

TEST_CASE("a fetch R700 does not have is refused", "[latte][shader][vtx]") {
    std::vector<u32> words = botw;
    words[4] = (words[4] & ~0x1FU) | 2U; // VTX_INST 2
    ShaderProgram program;
    std::string error;
    CHECK_FALSE(decode_program(words, ProgramShape::FetchShader, program, error));
    CHECK(error.find("not an R700 vertex fetch") != std::string::npos);
}

namespace {

/// A pixel shader that samples: a TEX clause of one fetch -- SAMPLE into R1
/// from R0's XY, texture 0, sampler 0 -- and the export of R1.
const std::vector<u32> sampling{2,
                                0x80800000, // TEX: ADDR(2) CNT(1)
                                0x00008000,
                                0x94200688, // EXP_DONE: PIX0, R1.xyzw, END_OF_PROGRAM
                                0x00000010,
                                0xF00D1001,
                                0x90800000,
                                0};

} // namespace

TEST_CASE("a pixel shader that samples, decoded", "[latte][shader][tex]") {
    const ShaderProgram program = decoded(sampling, ProgramShape::Shader);
    REQUIRE(program.clauses.size() == 1);
    CHECK(program.clauses[0].clause.kind == ClauseKind::Tex);
    REQUIRE(program.clauses[0].texture_fetches.size() == 1);
    CHECK(program.clauses[0].vertex_fetches.empty());
    const TextureFetch& fetch = program.clauses[0].texture_fetches[0];

    CHECK(fetch.inst() == TexInst::Sample);
    CHECK_FALSE(fetch.is_vertex_fetch());
    CHECK(fetch.tex0().resource_id == 0U);
    CHECK(fetch.tex0().src_gpr == 0U);
    CHECK(fetch.tex1().dst_gpr == 1U);
    CHECK(fetch.tex1().dst_sel_w == Sel::W);
    CHECK(fetch.tex1().coord_type_x);
    CHECK(fetch.tex2().sampler_id == 0U);
    CHECK(fetch.tex2().src_sel_y == Sel::Y);
    CHECK(fetch.tex2().src_sel_z == Sel::Zero);
    CHECK(fetch.lod_bias() == 0.0F);

    CHECK(disassemble(program) == "00 TEX: ADDR(2) CNT(1)\n"
                                  "  0      SAMPLE          R1.xyzw, R0.xy00, t0, s0\n"
                                  "01 EXP_DONE: PIX0, R1.xyzw\n"
                                  "END_OF_PROGRAM\n");
}

TEST_CASE("a texture fetch's fields, as decaf prints them", "[latte][shader][tex]") {
    // SAMPLE_C_LZ into R2, relative, from R3, relative, with texture 5 and
    // sampler 17; a LOD bias of -1/8, X and W in texels, offsets of a half
    // and minus a half, and every flag.
    const u32 word0 =
        0x1BU | (1U << 5) | (1U << 7) | (5U << 8) | (3U << 16) | (1U << 23) | (1U << 24);
    const u32 word1 = 2U | (1U << 7) | (3U << 9) | (2U << 12) | (7U << 15) | (6U << 18) |
                      (0x7FU << 21) | (1U << 29) | (1U << 30);
    const u32 word2 =
        1U | (0x1FU << 5) | (17U << 15) | (1U << 20) | (0U << 23) | (5U << 26) | (4U << 29);
    const TextureFetch fetch{.word0 = word0, .word1 = word1, .word2 = word2};
    CHECK(fetch.lod_bias() == -0.125F);
    CHECK(fetch.offset(0) == 0.5F);
    CHECK(fetch.offset(1) == -0.5F);
    CHECK(fetch.offset(2) == 0.0F);
    CHECK(disassemble(fetch) == "SAMPLE_C_LZ     R2[AL].wz_?, R3[AL].yx10, t5, s17 LOD(-0.125) "
                                "WHOLE_QUAD BC_FRAC_MODE ALT_CONST DENORM(XW) XOFFSET(0.5) "
                                "YOFFSET(-0.5)");

    // Nothing written, every coordinate normalised, the largest bias.
    const u32 masked = (7U << 9) | (7U << 12) | (7U << 15) | (7U << 18);
    CHECK(disassemble(TextureFetch{
              .word0 = 0x11U, .word1 = masked | (0x3FU << 21) | 0xF0000000U, .word2 = 0}) ==
          "SAMPLE_L        ____, R0.xxxx, t0, s0 LOD(7.875)");

    // A vertex fetch's words, in a texture clause, are a vertex fetch.
    const u32 vtx0 = 1U | (0xA0U << 8);
    const u32 vtx1 = 0x688U << 9;
    const TextureFetch vertex{.word0 = vtx0, .word1 = vtx1, .word2 = 0};
    CHECK(vertex.is_vertex_fetch());
    CHECK(disassemble(vertex) == text(vtx0, vtx1, 0));
    CHECK(disassemble(vertex).starts_with("SEMANTIC"));

    CHECK(tex_inst_name(TexInst::GetTextureInfo) == "GET_TEXTURE_INFO");
    CHECK(tex_inst_name(TexInst::SampleCGLz) == "SAMPLE_C_G_LZ");
}

namespace {

/// An ALU instruction's first word: its first two sources, as a select and a
/// channel, and whether it is its group's last.
[[nodiscard]] constexpr u32
alu0(u32 src0, u32 chan0, u32 src1 = 0, u32 chan1 = 0, bool last = false) {
    return src0 | (chan0 << 10) | (src1 << 13) | (chan1 << 23) | (last ? 1U << 31 : 0U);
}

/// An OP2's second word: the instruction, writing `dst`.`chan`.
[[nodiscard]] constexpr u32 op2(u32 inst, u32 dst, u32 chan) {
    return (1U << 4) | (inst << 7) | (dst << 21) | (chan << 29);
}

/// An OP3's: the third source, the instruction, and the destination.
[[nodiscard]] constexpr u32 op3(u32 src2, u32 chan2, u32 inst, u32 dst, u32 chan) {
    return src2 | (chan2 << 10) | (inst << 13) | (dst << 21) | (chan << 29);
}

constexpr u32 kAdd = 0x00;
constexpr u32 kMul = 0x01;
constexpr u32 kMov = 0x19;
constexpr u32 kDot4 = 0x50;
constexpr u32 kRecipIeee = 0x66;
constexpr u32 kMulAdd = 0x10;

constexpr u32 kKc0 = alu_src::kcache0_first;
constexpr u32 kPv = alu_src::previous_vector;
constexpr u32 kPs = alu_src::previous_scalar;
constexpr u32 kLiteral = alu_src::literal;

/// A pixel shader that computes: R0 = R1 * KC0[0]; then R2.x = PV.x * 0.5
/// + 0.5, R2.y = 1 / PV.w and R2.z = 8388608.0; then R3.x = R2.x + R2.z
/// and R4.x = PS - R2.x, clamped -- the second X instruction of its group,
/// so on T -- and the export of R0.
[[nodiscard]] std::vector<u32> computing() {
    std::vector<u32> words{
        2U | (1U << 30),                      // ALU: ADDR(2), KCACHE0 locks CB0:0-15
        (9U << 18) | (8U << 26) | (1U << 31), // CNT(10)
        0x00000000,                           // EXP_DONE: PIX0, R0.xyzw,
        0x94200688,                           // END_OF_PROGRAM
    };
    const auto put = [&words](u32 word0, u32 word1) {
        words.push_back(word0);
        words.push_back(word1);
    };
    for (u32 chan = 0; chan < 4; ++chan) {
        put(alu0(1, chan, kKc0, chan, chan == 3), op2(kMul, 0, chan));
    }
    put(alu0(kPv, 0, kLiteral, 0), op3(alu_src::half, 0, kMulAdd, 2, 0));
    put(alu0(kPv, 3), op2(kRecipIeee, 2, 1));
    put(alu0(kLiteral, 1, 0, 0, true), op2(kMov, 2, 2));
    put(0x3F000000, 0x4B000000);
    put(alu0(2, 0, 2, 2), op2(kAdd, 3, 0));
    put(alu0(kPs, 0, 2, 0, true) | (1U << 25), op2(kAdd, 4, 0) | (1U << 31));
    return words;
}

} // namespace

TEST_CASE("a pixel shader that computes, decoded", "[latte][shader][alu]") {
    const ShaderProgram program = decoded(computing(), ProgramShape::Shader);
    REQUIRE(program.clauses.size() == 1);
    CHECK(program.clauses[0].clause.kind == ClauseKind::Alu);
    const std::vector<AluGroup>& groups = program.clauses[0].alu_groups;
    REQUIRE(groups.size() == 3);

    REQUIRE(groups[0].instructions.size() == 4);
    CHECK(groups[0].literals.empty());
    for (u32 chan = 0; chan < 4; ++chan) {
        const AluInstruction& mul = groups[0].instructions[chan];
        CHECK(mul.unit == static_cast<AluUnit>(chan));
        CHECK(mul.op.name == "MUL");
        CHECK_FALSE(mul.is_op3());
        CHECK(mul.alu1().dst_gpr == 0U);
        CHECK(mul.alu0().src1_sel == kKc0);
    }

    // RECIP_IEEE writes Y, but only T runs it.
    REQUIRE(groups[1].instructions.size() == 3);
    CHECK(groups[1].instructions[0].is_op3());
    CHECK(groups[1].instructions[0].op.name == "MULADD");
    CHECK(groups[1].instructions[0].unit == AluUnit::X);
    CHECK(groups[1].instructions[1].unit == AluUnit::T);
    CHECK(groups[1].instructions[1].op.transcendental_only());
    CHECK(groups[1].instructions[2].unit == AluUnit::Z);
    CHECK(groups[1].literals == std::vector<u32>{0x3F000000, 0x4B000000});

    // Two to X: the second, which T can run too, goes there.
    REQUIRE(groups[2].instructions.size() == 2);
    CHECK(groups[2].instructions[0].unit == AluUnit::X);
    CHECK(groups[2].instructions[1].unit == AluUnit::T);
    CHECK(groups[2].literals.empty());

    CHECK(disassemble(program) == "00 ALU: ADDR(2) CNT(10) KCACHE0(CB0:0-15)\n"
                                  "  0   x: MUL             R0.x, R1.x, KC0[0].x\n"
                                  "      y: MUL             R0.y, R1.y, KC0[0].y\n"
                                  "      z: MUL             R0.z, R1.z, KC0[0].z\n"
                                  "      w: MUL             R0.w, R1.w, KC0[0].w\n"
                                  "  1   x: MULADD          R2.x, PV0.x, (0x3F000000, 0.5), 0.5f\n"
                                  "      t: RECIP_IEEE      R2.y, PV0.w SCL_210\n"
                                  "      z: MOV             R2.z, (0x4B000000, 8388608)\n"
                                  "  2   x: ADD             R3.x, R2.x, R2.z\n"
                                  "      t: ADD             R4.x, PS1, -R2.x CLAMP\n"
                                  "01 EXP_DONE: PIX0, R0.xyzw\n"
                                  "END_OF_PROGRAM\n");
}

TEST_CASE("an ALU instruction's fields, as decaf prints them", "[latte][shader][alu]") {
    const std::vector<u32> literals{0x3F800000, 0xC0000000, 0x00000001, 0x7F800000};
    const auto text = [&literals](u32 word0, u32 word1, u32 group = 5) {
        AluInstruction alu{.word0 = word0, .word1 = word1, .op = {}, .unit = AluUnit::X};
        const std::optional<AluOp> op =
            alu.is_op3() ? alu_op3(alu.op3().alu_inst) : alu_op2(alu.op2().alu_inst);
        REQUIRE(op.has_value());
        alu.op = *op;
        return disassemble(alu, group, literals);
    };

    // An OP2: a relative source indexed by AR.y, negated and absolute, a
    // constant-file source, a relative destination, the output doubled and
    // clamped, a bank swizzle, and the predicate.
    const u32 word0 = alu0(7, 2, 256 + 5, 3) | (1U << 9) | (1U << 12) | (1U << 26) | (3U << 29);
    const u32 word1 = op2(kMul, 9, 1) | 1U | (1U << 5) | (2U << 18) | (1U << 28) | (1U << 31);
    CHECK(text(word0, word1) ==
          "MUL*2           R9[AR.y].y, -|R7[AR.y].z|, C5.w CLAMP VEC_120 PRED_SEL_ONE");

    // Nothing written; with UPDATE_EXEC_MASK the output modifier is not printed.
    const u32 kill = alu0(kKc0 + 32 + 3, 0, alu_src::loop_index, 0) | (2U << 29);
    const u32 kill1 = (0x2DU << 7) | (1U << 2) | (1U << 3) | (3U << 5);
    CHECK(text(kill, kill1) ==
          "KILLGT          ____, KC1[3].x, AL PRED_SEL_ZERO UPDATE_EXEC_MASK UPDATE_PRED");

    // Literals by channel; a float past 1e7 in fixed notation, as fmt 8 has it.
    CHECK(text(alu0(kLiteral, 1, kLiteral, 3), op2(kAdd, 1, 0)) ==
          "ADD             R1.x, (0xC0000000, -2), (0x7F800000, inf)");
    CHECK(text(alu0(kLiteral, 2), op2(kMov, 1, 0)) == "MOV             R1.x, (0x00000001, 1e-45)");

    // A transcendental-only instruction prints its bank swizzle even when
    // it is zero; inline constants have no channel.
    CHECK(text(alu0(alu_src::one, 0), op2(kRecipIeee, 0, 0) | (1U << 18)) ==
          "RECIP_IEEE      R0.x, 1.0f SCL_122");
    CHECK(text(alu0(alu_src::minus_one, 0), op2(kMov, 0, 0)) == "MOV             R0.x, -1");

    // An OP3: the third source negated and relative to the loop index.
    const u32 muladd0 = alu0(1, 0, alu_src::zero, 0) | (4U << 26);
    const u32 muladd1 = op3(3, 1, kMulAdd, 2, 3) | (1U << 9) | (1U << 12);
    CHECK(text(muladd0, muladd1) == "MULADD          R2.w, R1.x, 0.0f, -R3[AL].y");

    // decaf's count is unsigned: PV before the first group wraps.
    CHECK(text(alu0(kPv, 2), op2(kMov, 0, 0), 0) ==
          "MOV             R0.x, PV18446744073709551615.z");
}

TEST_CASE("an ALU group that is not one is refused", "[latte][shader][alu]") {
    const auto refused = [](std::vector<u32> slots) {
        std::vector<u32> words{2U, 0, 0x00000000, 0x94200688};
        words[1] = ((static_cast<u32>(slots.size() / 2) - 1) << 18) | (8U << 26) | (1U << 31);
        words.insert(words.end(), slots.begin(), slots.end());
        ShaderProgram program;
        std::string error;
        CHECK_FALSE(decode_program(words, ProgramShape::Shader, program, error));
        return error;
    };
    const u32 last = 1U << 31;

    CHECK(refused({last, 0x7FU << 7}).find("OP2 0x7f, not an R700 ALU instruction") !=
          std::string::npos);
    // DOT4 runs only on the vector units: a second to X is a collision.
    CHECK(refused({0, op2(kDot4, 0, 0), last, op2(kDot4, 1, 0)}) ==
          "the ALU group at slot 0 of the clause instruction 0 gives two instructions the x unit");
    // A third MUL to X, with T taken.
    CHECK(refused({0, op2(kMul, 0, 0), 0, op2(kMul, 1, 0), last, op2(kMul, 2, 0)})
              .find("the t unit") != std::string::npos);

    std::vector<u32> six;
    for (u32 chan = 0; chan < 6; ++chan) {
        six.push_back(0);
        six.push_back(op2(kMov, chan, chan & 3U));
    }
    CHECK(refused(six).find("more than five") != std::string::npos);
    CHECK(refused({0, op2(kMov, 0, 0)}).find("past the end of its clause") != std::string::npos);
    CHECK(refused({alu0(kLiteral, 0, 0, 0, true), op2(kMov, 0, 0)}).find("literals past") !=
          std::string::npos);

    CHECK(refused({last, op2(kMov, 0, 0) | (6U << 18)}).find("bank swizzle 6") !=
          std::string::npos);
    CHECK(refused({last, op2(kRecipIeee, 0, 0) | (4U << 18)})
              .find("gives RECIP_IEEE bank swizzle 4") != std::string::npos);
    CHECK(refused({last | (1U << 29), op2(kMov, 0, 0)}).find("PRED_SEL 1") != std::string::npos);
}
