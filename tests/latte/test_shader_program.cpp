// Phase 8.6 — a program decoded: the fetch clauses.
//
// What is guarded: a fetch shader a game ships (Breath of the Wild's, from
// Cemu's notes) decoded field by field and printed as decaf prints it; each
// of a fetch's fields in decaf's text, its quirks included; the one count
// decaf numbers fetches by, across clauses; and the refusal of a fetch R700
// does not have. And texture fetch clauses: a pixel shader that samples,
// decoded and printed; a texture fetch's every field in decaf's text, its
// signed fixed-point LOD bias and offsets included; and a vertex fetch in a
// texture clause printed as one. That this text is decaf's for every field
// is a scratch harness's to say, as for the control flow (see plan.md, 8.6).

#include <wiiu/latte/shader_program.hpp>

#include <catch2/catch_test_macros.hpp>

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
