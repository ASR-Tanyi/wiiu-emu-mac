// Phase 8.6 — the intermediate form.
//
// What is guarded: decaf's example triangle -- its vertex shader, a fetch
// shader for its two attributes, its pixel shader and the registers its
// shaders' headers set -- described as the emitter will read it; the
// register lookups each fact depends on (a semantic's register is R(n + 1)
// for the SQ_VTX_SEMANTIC that names it, a parameter's semantic is its
// SPI_VS_OUT_ID byte, a pixel input is matched by semantic or takes its
// default, a pixel export goes to the n-th enabled colour target); bursts;
// a SAMPLE, with the bound texture's selects; ALU clauses decoded from
// microcode into their groups -- each unit's instruction, sources (registers,
// literals, inline constants, PV and PS), modifiers, write mask, and a DOT4
// across the four vector units -- with a vertex shader's R0 readable only
// once written; and the refusal of everything not translated yet, by name.

#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_ir.hpp>
#include <wiiu/latte/shader_state.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;
namespace reg = latte::reg;

namespace {

// Control-flow words, from the field positions (microcode.hpp).
constexpr u32 barrier = 1U << 31;
constexpr u32 end_of_program = 1U << 21;
constexpr u32 xyzw = 0 | (1U << 3) | (2U << 6) | (3U << 9);
constexpr u32 call_fs = 0x13U << 23;
constexpr u32 exp_done = 0x28U << 23;
constexpr u32 exp_more = 0x27U << 23; // EXP; not `exp`, which <math.h> has

[[nodiscard]] constexpr u32 exp0(u32 base, u32 type, u32 gpr) noexcept {
    return base | (type << 13) | (gpr << 15);
}

/// decaf's example vertex shader: the fetch shader, then POS0 from R2 and
/// PARAM0 from R1.
const std::vector<u32> example_vs{0,
                                  call_fs,
                                  exp0(60, 1, 2),
                                  xyzw | exp_done | barrier,
                                  exp0(0, 2, 1),
                                  xyzw | exp_done | end_of_program};

/// And its pixel shader: PIX0 from R0.
const std::vector<u32> example_ps{exp0(0, 0, 0), xyzw | exp_done | barrier | end_of_program};

/// A semantic fetch of four floats, big-endian, from attribute buffer
/// `buffer`.
[[nodiscard]] std::array<u32, 4>
float4_fetch(u32 semantic, u32 buffer, u32 select = 0, u32 type = 0) {
    return {1U | (type << 5) | ((0xA0U + buffer) << 8) | (select << 24) | (15U << 26),
            semantic | (0x688U << 9) | (0x23U << 22) | (2U << 28),
            (2U << 16) | (1U << 19),
            0};
}

/// A fetch shader: one VTX_TC clause of `fetches`, then the RETURN.
[[nodiscard]] std::vector<u32> fetch_shader(const std::vector<std::array<u32, 4>>& fetches) {
    std::vector<u32> words{
        2, (3U << 23) | ((static_cast<u32>(fetches.size()) - 1) << 10), 0, 0x8A000000};
    for (const auto& fetch : fetches) {
        words.insert(words.end(), fetch.begin(), fetch.end());
    }
    return words;
}

[[nodiscard]] ShaderProgram program(const std::vector<u32>& words, ProgramShape shape) {
    ShaderProgram out;
    std::string error;
    INFO(error);
    REQUIRE(decode_program(words, shape, out, error));
    return out;
}

/// The registers decaf's example shaders' headers set, with the unused
/// semantics cleared as SQ_VTX_SEMANTIC_CLEAR leaves them.
struct Registers {
    std::unique_ptr<RegisterFile> file = std::make_unique<RegisterFile>();

    Registers() {
        for (u32 n = 0; n < reg::sq_vtx_semantic_count; ++n) {
            set(reg::sq_vtx_semantic_0 + (4 * n), n < 2 ? n : 0xFFFFFFFF);
        }
        for (u32 n = 0; n < spi_vs_out_id_count; ++n) {
            set(reg::spi_vs_out_id_0 + (4 * n), n == 0 ? 0xFFFFFF00 : 0xFFFFFFFF);
        }
        set(reg::spi_ps_in_control_0, 1);             // one input
        set(reg::spi_ps_input_cntl_0, 0 | (1U << 8)); // semantic 0, default 0001
        set(reg::cb_shader_mask, 0xF);
    }

    void set(u32 address, u32 value) const { REQUIRE(file->write(address, value)); }
};

[[nodiscard]] std::string
vertex(const std::vector<u32>& vs, const std::vector<u32>* fs, const Registers& registers) {
    const ShaderProgram vertex_program = program(vs, ProgramShape::Shader);
    std::optional<ShaderProgram> fetch_program;
    if (fs != nullptr) {
        fetch_program = program(*fs, ProgramShape::FetchShader);
    }
    ir::Shader shader;
    std::string error;
    INFO(error);
    REQUIRE(ir::build_vertex_shader(
        vertex_program, fetch_program ? &*fetch_program : nullptr, *registers.file, shader, error));
    return ir::describe(shader);
}

[[nodiscard]] std::string pixel(const std::vector<u32>& ps, const Registers& registers) {
    ir::Shader shader;
    std::string error;
    INFO(error);
    REQUIRE(
        ir::build_pixel_shader(program(ps, ProgramShape::Shader), *registers.file, shader, error));
    return ir::describe(shader);
}

[[nodiscard]] std::string
refused_vertex(const std::vector<u32>& vs, const std::vector<u32>* fs, const Registers& registers) {
    const ShaderProgram vertex_program = program(vs, ProgramShape::Shader);
    std::optional<ShaderProgram> fetch_program;
    if (fs != nullptr) {
        fetch_program = program(*fs, ProgramShape::FetchShader);
    }
    ir::Shader shader;
    std::string error;
    CHECK_FALSE(ir::build_vertex_shader(
        vertex_program, fetch_program ? &*fetch_program : nullptr, *registers.file, shader, error));
    return error;
}

[[nodiscard]] std::string refused_pixel(const std::vector<u32>& ps, const Registers& registers) {
    ir::Shader shader;
    std::string error;
    CHECK_FALSE(
        ir::build_pixel_shader(program(ps, ProgramShape::Shader), *registers.file, shader, error));
    return error;
}

} // namespace

TEST_CASE("decaf's example triangle, as the emitter will read it", "[latte][shader][ir]") {
    // Colour at location 0 from buffer 0, position at location 1 from buffer 1.
    const std::vector<u32> fs = fetch_shader({float4_fetch(0, 0), float4_fetch(1, 1)});
    const Registers registers;

    // SQ_VTX_SEMANTIC_0 is 0 and _1 is 1, so the colour lands in R1 and the
    // position in R2 -- which are the registers the shader exports as its
    // parameter and its position.
    CHECK(vertex(example_vs, &fs, registers) ==
          "vertex shader\n"
          "  attribute R1.xyzw <- semantic 0: buffer 0 + 0, 16 bytes as FMT 0x23 scaled, endian "
          "swap 2, per vertex\n"
          "  attribute R2.xyzw <- semantic 1: buffer 1 + 0, 16 bytes as FMT 0x23 scaled, endian "
          "swap 2, per vertex\n"
          "  fetch attributes\n"
          "  position <- R2.xyzw\n"
          "  parameter 0 (semantic 0) <- R1.xyzw\n");

    CHECK(pixel(example_ps, registers) == "pixel shader\n"
                                          "  input R0 <- parameter 0 (semantic 0), perspective\n"
                                          "  colour 0 (target 0) <- R0.xyzw\n");
}

TEST_CASE("an attribute's register is SQ_VTX_SEMANTIC's", "[latte][shader][ir]") {
    Registers registers;
    registers.set(reg::sq_vtx_semantic_0 + (4 * 5), 7); // semantic 7 into R6
    registers.set(reg::sq_vtx_semantic_0 + 4, 0xFFFFFFFF);
    // Semantic 1 is named by no register now: its fetch does nothing.
    const std::vector<u32> fs =
        fetch_shader({float4_fetch(7, 3), float4_fetch(1, 1), float4_fetch(0, 0)});
    const std::string text = vertex(example_vs, &fs, registers);
    CHECK(text.find("attribute R6.xyzw <- semantic 7: buffer 3") != std::string::npos);
    CHECK(text.find("semantic 1") == std::string::npos);
    CHECK(text.find("attribute R1.xyzw <- semantic 0") != std::string::npos);
}

TEST_CASE("an instance attribute's index", "[latte][shader][ir]") {
    const Registers registers;
    // R0.w is the instance index; R0.y and R0.z are it over the step rates.
    for (const auto& [select, expected] :
         std::vector<std::pair<u32, std::string>>{{3, "per instance\n"},
                                                  {1, "per instance / step rate 0\n"},
                                                  {2, "per instance / step rate 1\n"}}) {
        const std::vector<u32> fs = fetch_shader({float4_fetch(0, 0, select, 1)});
        CHECK(vertex(example_vs, &fs, registers).find(expected) != std::string::npos);
    }
    const std::vector<u32> from_x = fetch_shader({float4_fetch(0, 0, 0, 1)});
    CHECK(refused_vertex(example_vs, &from_x, registers).find("R0.x") != std::string::npos);
}

TEST_CASE("a pixel input is a parameter, the position or its default", "[latte][shader][ir]") {
    Registers registers;
    registers.set(reg::spi_vs_out_id_0, 0xFF0400FFU); // parameter 1 is semantic 0, 2 is 4
    registers.set(reg::spi_ps_in_control_0, 3U | (1U << 8) | (2U << 10)); // input 2 the position
    registers.set(reg::spi_ps_input_cntl_0, 4U | (1U << 10));             // semantic 4, flat
    registers.set(reg::spi_ps_input_cntl_0 + 4, 9U | (3U << 8) | (1U << 11) | (1U << 12));
    registers.set(reg::spi_ps_input_cntl_0 + 8, 0);
    CHECK(pixel(example_ps, registers) ==
          "pixel shader\n"
          "  input R0 <- parameter 2 (semantic 4), flat\n"
          "  input R1 <- default 3 (semantic 9 unmatched), linear, centroid\n"
          "  input R2 <- position, perspective\n"
          "  colour 0 (target 0) <- R0.xyzw\n");
}

TEST_CASE("pixel exports go to the enabled targets, in order", "[latte][shader][ir]") {
    Registers registers;
    registers.set(reg::cb_shader_mask, 0x0F0F); // targets 0 and 2
    // PIX0 and PIX1 from R3 and R4, as one burst; a third PIX goes nowhere.
    const std::vector<u32> ps{exp0(0, 0, 3),
                              xyzw | (1U << 17) | exp_more | barrier,
                              exp0(2, 0, 5),
                              xyzw | exp_done | barrier | end_of_program};
    CHECK(pixel(ps, registers) == "pixel shader\n"
                                  "  input R0 <- parameter 0 (semantic 0), perspective\n"
                                  "  colour 0 (target 0) <- R3.xyzw\n"
                                  "  colour 1 (target 2) <- R4.xyzw\n");
}

TEST_CASE("a parameter burst, and a parameter nothing receives", "[latte][shader][ir]") {
    Registers registers;
    registers.set(reg::spi_vs_out_id_0, 0xFF0B0A00U); // 0, 10, 11, unused
    const std::vector<u32> vs{exp0(60, 1, 1),
                              xyzw | exp_done | barrier,
                              exp0(0, 2, 2),
                              xyzw | (3U << 17) | exp_done | end_of_program};
    CHECK(vertex(vs, nullptr, registers) == "vertex shader\n"
                                            "  position <- R1.xyzw\n"
                                            "  parameter 0 (semantic 0) <- R2.xyzw\n"
                                            "  parameter 1 (semantic 10) <- R3.xyzw\n"
                                            "  parameter 2 (semantic 11) <- R4.xyzw\n");
}

TEST_CASE("what is not translated yet is refused by name", "[latte][shader][ir]") {
    const Registers registers;
    const std::vector<u32> fs = fetch_shader({float4_fetch(0, 0)});

    // An ALU clause of one group: ADD R0.x, R0.x, R0.x -- in a vertex
    // shader, R0 is the vertex and instance indices, not translated yet.
    const std::vector<u32> alu{2,
                               (8U << 26) | barrier,
                               exp0(60, 1, 0),
                               xyzw | exp_done | end_of_program,
                               1U << 31,
                               1U << 4};
    CHECK(refused_vertex(alu, nullptr, registers).find("reads R0.x before the shader writes it") !=
          std::string::npos);

    // A vertex fetch in a texture clause in a pixel shader.
    const std::vector<u32> tex{
        2, (1U << 23) | barrier, exp0(0, 0, 0), xyzw | exp_done | end_of_program, 0, 0, 0, 0};
    CHECK(refused_pixel(tex, registers).find("vertex fetches in a texture clause") !=
          std::string::npos);

    // CALL_FS with no fetch shader.
    CHECK(refused_vertex(example_vs, nullptr, registers).find("there is none") !=
          std::string::npos);

    // POS1, the point size.
    const std::vector<u32> point{exp0(61, 1, 0), xyzw | exp_done | end_of_program};
    CHECK(refused_vertex(point, nullptr, registers).find("point-size") != std::string::npos);

    // A memory export (stream out), and a relative register.
    const std::vector<u32> stream{exp0(0, 0, 0), (0x20U << 23) | end_of_program};
    CHECK(refused_vertex(stream, nullptr, registers).find("memory exports") != std::string::npos);
    const std::vector<u32> relative{exp0(60, 1, 0) | (1U << 22), xyzw | exp_done | end_of_program};
    CHECK(refused_vertex(relative, nullptr, registers).find("relative") != std::string::npos);

    // A depth export.
    const std::vector<u32> depth{exp0(61, 0, 0), xyzw | exp_done | end_of_program};
    CHECK(refused_pixel(depth, registers).find("depth exports") != std::string::npos);

    // A plain fetch in the fetch shader, and one from a resource that is not
    // an attribute buffer.
    std::vector<u32> plain = fs;
    plain[4] &= ~1U;
    CHECK(refused_vertex(example_vs, &plain, registers).find("plain fetch") != std::string::npos);
    std::vector<u32> uniform = fs;
    uniform[4] = (uniform[4] & ~(0xFFU << 8)) | (0x80U << 8);
    CHECK(refused_vertex(example_vs, &uniform, registers).find("not an attribute buffer") !=
          std::string::npos);

    // A position export in a pixel shader, a pixel export in a vertex shader.
    const std::vector<u32> wrong_way{exp0(60, 1, 0), xyzw | exp_done | end_of_program};
    CHECK(refused_pixel(wrong_way, registers).find("not a pixel export") != std::string::npos);
    CHECK(refused_vertex(example_ps, nullptr, registers).find("pixel export in a vertex") !=
          std::string::npos);
}

// --------------------------------------------------------------------------
// Texture samples
// --------------------------------------------------------------------------

namespace {

/// A SAMPLE into R1.xyzw from R0's X and Y, texture `texture` and sampler
/// `sampler`, normalised coordinates; `word0`, `word1` and `word2` are
/// ORed in.
[[nodiscard]] std::vector<u32>
sampling_ps(u32 texture = 0, u32 sampler = 0, u32 word0 = 0, u32 word1 = 0, u32 word2 = 0) {
    return {2,
            (1U << 23) | barrier, // TEX: ADDR(2) CNT(1)
            exp0(0, 0, 1),
            xyzw | exp_done | barrier | end_of_program,
            0x10U | (texture << 8) | word0,
            1U | (0x688U << 9) | 0xF0000000U | word1,
            (sampler << 15) | (0U << 20) | (1U << 23) | (4U << 26) | (4U << 29) | word2,
            0};
}

/// A texture resource at the pixel stage's unit `unit`: its shape and its
/// component selects.
void bind_texture(const Registers& registers,
                  u32 unit,
                  TexDim dim = TexDim::D2,
                  std::array<Sel, 4> selects = {Sel::X, Sel::Y, Sel::Z, Sel::W}) {
    const u32 base = resource::address(resource::ps_base + unit);
    TexResourceWord0 word0{.raw = 0};
    word0.dim = dim;
    TexResourceWord4 word4{.raw = 0};
    word4.dst_sel_x = selects[0];
    word4.dst_sel_y = selects[1];
    word4.dst_sel_z = selects[2];
    word4.dst_sel_w = selects[3];
    TexResourceWord6 word6{.raw = 0};
    word6.type = ResourceType::ValidTexture;
    registers.set(base, word0.raw);
    registers.set(base + 16, word4.raw);
    registers.set(base + 24, word6.raw);
}

} // namespace

TEST_CASE("a sample is written down with its texture's selects", "[latte][shader][ir]") {
    const Registers registers;
    // An RGB texture: alpha reads one (GX2's component map for it).
    bind_texture(registers, 3, TexDim::D2, {Sel::X, Sel::Y, Sel::Z, Sel::One});
    CHECK(pixel(sampling_ps(3, 5), registers) ==
          "pixel shader\n"
          "  input R0 <- parameter 0 (semantic 0), perspective\n"
          "  R1.xyzw <- sample texture 3 (xyz1) with sampler 5 at R0.xy\n"
          "  colour 0 (target 0) <- R1.xyzw\n");

    // The shader's own selects, and coordinates from other components.
    std::vector<u32> words = sampling_ps(3, 5);
    words[5] = 1U | (3U << 9) | (2U << 12) | (7U << 15) | (0U << 18) | 0xF0000000U;
    words[6] = (5U << 15) | (3U << 20) | (2U << 23) | (4U << 26) | (4U << 29);
    CHECK(pixel(words, registers)
              .find("R1.wz_x <- sample texture 3 (xyz1) with sampler 5 at "
                    "R0.wz") != std::string::npos);
}

TEST_CASE("what a texture clause holds that is not translated yet is refused",
          "[latte][shader][ir]") {
    const Registers registers;
    bind_texture(registers, 0);

    // Another instruction: SAMPLE_L.
    std::vector<u32> lod = sampling_ps();
    lod[4] = (lod[4] & ~0x1FU) | 0x11U;
    CHECK(refused_pixel(lod, registers).find("is SAMPLE_L; only SAMPLE") != std::string::npos);
    // Coordinates in texels.
    std::vector<u32> texels = sampling_ps();
    texels[5] &= ~(1U << 28);
    CHECK(refused_pixel(texels, registers).find("texel coordinates") != std::string::npos);
    // A LOD bias, and an offset.
    CHECK(refused_pixel(sampling_ps(0, 0, 0, 1U << 21), registers).find("LOD bias") !=
          std::string::npos);
    CHECK(refused_pixel(sampling_ps(0, 0, 0, 0, 1U), registers).find("offsets") !=
          std::string::npos);
    // Relative registers.
    CHECK(refused_pixel(sampling_ps(0, 0, 1U << 23), registers).find("relative") !=
          std::string::npos);
    // A unit with no texture, and a cube.
    CHECK(refused_pixel(sampling_ps(1), registers).find("not a valid texture") !=
          std::string::npos);
    bind_texture(registers, 2, TexDim::Cube);
    CHECK(refused_pixel(sampling_ps(2), registers).find("not 2D (SQ_TEX_DIM 3)") !=
          std::string::npos);
}

// --------------------------------------------------------------------------
// ALU groups
// --------------------------------------------------------------------------

namespace {

// ALU source selects (microcode.hpp's alu_src).
constexpr u32 src_zero = 248;
constexpr u32 src_one = 249;
constexpr u32 src_literal = 253;
constexpr u32 src_pv = 254;
constexpr u32 src_ps = 255;

struct Src {
    u32 sel = src_zero;
    u32 chan = 0;
    bool neg = false;
    bool abs = false;
    bool rel = false;
};

/// An OP2's two words, from the field positions (microcode.hpp).
struct Op2 {
    u32 inst = 0;
    u32 dst = 0;
    u32 chan = 0;
    Src a{};
    Src b{};
    bool write = true;
    u32 omod = 0;
    bool clamp = false;
    u32 pred = 0;
    bool last = false;
};

[[nodiscard]] u32 word0(const Src& a, const Src& b, u32 pred, bool last) {
    return a.sel | (u32{a.rel} << 9) | (a.chan << 10) | (u32{a.neg} << 12) | (b.sel << 13) |
           (u32{b.rel} << 22) | (b.chan << 23) | (u32{b.neg} << 25) | (pred << 29) |
           (u32{last} << 31);
}

[[nodiscard]] std::array<u32, 2> op2(const Op2& o) {
    return {word0(o.a, o.b, o.pred, o.last),
            u32{o.a.abs} | (u32{o.b.abs} << 1) | (u32{o.write} << 4) | (o.omod << 5) |
                (o.inst << 7) | (o.dst << 21) | (o.chan << 29) | (u32{o.clamp} << 31)};
}

/// An OP3's.
struct Op3 {
    u32 inst = 0;
    u32 dst = 0;
    u32 chan = 0;
    Src a{};
    Src b{};
    Src c{};
    bool clamp = false;
    bool last = false;
};

[[nodiscard]] std::array<u32, 2> op3(const Op3& o) {
    return {word0(o.a, o.b, 0, o.last),
            o.c.sel | (u32{o.c.rel} << 9) | (o.c.chan << 10) | (u32{o.c.neg} << 12) |
                (o.inst << 13) | (o.dst << 21) | (o.chan << 29) | (u32{o.clamp} << 31)};
}

constexpr u32 alu_add = 0x00;
constexpr u32 alu_mul = 0x01;
constexpr u32 alu_mov = 0x19;
constexpr u32 alu_nop = 0x1A;
constexpr u32 alu_floor = 0x14;
constexpr u32 alu_dot4 = 0x50;
constexpr u32 alu_muladd = 0x10;

/// A program of ALU clauses, each a list of 64-bit slots, run in order,
/// then one export: PIX0 (type 0) or POS0 (type 1) from `gpr`. `cf_inst`
/// is each clause's ALU CF_INST.
[[nodiscard]] std::vector<u32>
alu_program(const std::vector<std::vector<std::array<u32, 2>>>& clauses,
            u32 export_type,
            u32 gpr,
            u32 cf_inst = 8) {
    const auto cf_slots = static_cast<u32>(clauses.size() + 1);
    std::vector<u32> words;
    u32 address = cf_slots;
    for (const auto& clause : clauses) {
        words.push_back(address);
        words.push_back(((static_cast<u32>(clause.size()) - 1) << 18) | (cf_inst << 26) | barrier);
        address += static_cast<u32>(clause.size());
    }
    words.push_back(exp0(export_type == 1 ? 60 : 0, export_type, gpr));
    words.push_back(xyzw | exp_done | end_of_program);
    for (const auto& clause : clauses) {
        for (const auto& slot : clause) {
            words.insert(words.end(), slot.begin(), slot.end());
        }
    }
    return words;
}

/// The group the Metal test draws too (test_metal.cpp): R0 is the
/// interpolated colour, R1 what is exported.
[[nodiscard]] std::vector<std::array<u32, 2>> example_alu_clause() {
    return {
        // MUL R2.x, R0.y, 0.25; MOV ____, R0.y; MOV R0.y, 0.0 (on T, since Y
        // is taken) -- which the X and Y units still read as it was.
        op2({.inst = alu_mul, .dst = 2, .chan = 0, .a = {0, 1}, .b = {src_literal, 0}}),
        op2({.inst = alu_mov, .dst = 2, .chan = 1, .a = {0, 1}, .write = false}),
        op2({.inst = alu_mov, .dst = 0, .chan = 1, .a = {src_zero}, .last = true}),
        {0x3E800000, 0},
        // ADD R1.x, PV.x, PV.x, *2; MOV R1.y, R0.y; MUL R1.z, PS, PV.y;
        // MULADD R1.w, PV.y, 2.0, PS, clamped.
        op2({.inst = alu_add, .dst = 1, .chan = 0, .a = {src_pv, 0}, .b = {src_pv, 0}, .omod = 1}),
        op2({.inst = alu_mov, .dst = 1, .chan = 1, .a = {0, 1}}),
        op2({.inst = alu_mul, .dst = 1, .chan = 2, .a = {src_ps}, .b = {src_pv, 1}}),
        op3({.inst = alu_muladd,
             .dst = 1,
             .chan = 3,
             .a = {src_pv, 1},
             .b = {src_literal, 0},
             .c = {src_ps},
             .clamp = true,
             .last = true}),
        {0x40000000, 0},
        // DOT4 of R1 and (1, 1, 1, 0), written to R1.y alone.
        op2({.inst = alu_dot4, .dst = 1, .chan = 0, .a = {1, 0}, .b = {src_one}, .write = false}),
        op2({.inst = alu_dot4, .dst = 1, .chan = 1, .a = {1, 1}, .b = {src_one}}),
        op2({.inst = alu_dot4, .dst = 1, .chan = 2, .a = {1, 2}, .b = {src_one}, .write = false}),
        op2({.inst = alu_dot4,
             .dst = 1,
             .chan = 3,
             .a = {1, 3},
             .b = {src_zero},
             .write = false,
             .last = true}),
    };
}

} // namespace

TEST_CASE("an ALU clause is its groups, each unit's instruction with its sources",
          "[latte][shader][ir][alu]") {
    const Registers registers;
    CHECK(pixel(alu_program({example_alu_clause()}, 0, 1), registers) ==
          "pixel shader\n"
          "  input R0 <- parameter 0 (semantic 0), perspective\n"
          "  alu group\n"
          "    x: R2.x <- MUL(R0.y, 0x3E800000)\n"
          "    y: ____ <- MOV(R0.y)\n"
          "    t: R0.y <- MOV(0x00000000)\n"
          "  alu group\n"
          "    x: R1.x <- ADD(PV.x, PV.x) * 2\n"
          "    y: R1.y <- MOV(R0.y)\n"
          "    z: R1.z <- MUL(PS, PV.y)\n"
          "    w: R1.w <- MULADD(PV.y, 0x40000000, PS), clamped\n"
          "  alu group\n"
          "    x: ____ <- DOT4(R1.x, 0x3F800000)\n"
          "    y: R1.y <- DOT4(R1.y, 0x3F800000)\n"
          "    z: ____ <- DOT4(R1.z, 0x3F800000)\n"
          "    w: ____ <- DOT4(R1.w, 0x00000000)\n"
          "  colour 0 (target 0) <- R1.xyzw\n");
}

TEST_CASE("an ALU source's modifiers, the inline constants, and the MULADD forms",
          "[latte][shader][ir][alu]") {
    const Registers registers;
    // MOV R1.x, -|R0.z|; MOV R1.y, 0.5; MOV R1.z, 1 (the integer); MOV R1.w,
    // -1 (the integer), clamped, /2.
    const std::vector<std::array<u32, 2>> moves{
        op2({.inst = alu_mov, .dst = 1, .chan = 0, .a = {0, 2, true, true}}),
        op2({.inst = alu_mov, .dst = 1, .chan = 1, .a = {252}}),
        op2({.inst = alu_mov, .dst = 1, .chan = 2, .a = {250}}),
        op2({.inst = alu_mov,
             .dst = 1,
             .chan = 3,
             .a = {251},
             .omod = 3,
             .clamp = true,
             .last = true}),
        // MULADD_IEEE_M4 R2.x, R1.x, -R1.y, R0.w; MULADD_D2 R2.y, ...
        op3({.inst = 0x16, .dst = 2, .chan = 0, .a = {1, 0}, .b = {1, 1, true}, .c = {0, 3}}),
        op3({.inst = 0x13,
             .dst = 2,
             .chan = 1,
             .a = {1, 2},
             .b = {1, 3},
             .c = {0, 0, true},
             .last = true}),
    };
    const std::string text = pixel(alu_program({moves}, 0, 1), registers);
    CHECK(text.find("    x: R1.x <- MOV(-|R0.z|)\n"
                    "    y: R1.y <- MOV(0x3F000000)\n"
                    "    z: R1.z <- MOV(0x00000001)\n"
                    "    w: R1.w <- MOV(0xFFFFFFFF) / 2, clamped\n") != std::string::npos);
    CHECK(text.find("    x: R2.x <- MULADD_IEEE(R1.x, -R1.y, R0.w) * 4\n"
                    "    y: R2.y <- MULADD(R1.z, R1.w, -R0.x) / 2\n") != std::string::npos);
}

TEST_CASE("a vertex shader may read R0 once it has written it", "[latte][shader][ir][alu]") {
    const Registers registers;
    const std::vector<std::array<u32, 2>> clause{
        op2({.inst = alu_mov, .dst = 0, .chan = 0, .a = {src_one}, .last = true}),
        op2({.inst = alu_add, .dst = 0, .chan = 1, .a = {0, 0}, .b = {0, 0}, .last = true}),
    };
    CHECK(vertex(alu_program({clause}, 1, 0), nullptr, registers) ==
          "vertex shader\n"
          "  alu group\n"
          "    x: R0.x <- MOV(0x3F800000)\n"
          "  alu group\n"
          "    y: R0.y <- ADD(R0.x, R0.x)\n"
          "  position <- R0.xyzw\n");
    // But not a component it has not.
    const std::vector<std::array<u32, 2>> other{
        op2({.inst = alu_mov, .dst = 0, .chan = 0, .a = {src_one}, .last = true}),
        op2({.inst = alu_add, .dst = 0, .chan = 1, .a = {0, 0}, .b = {0, 3}, .last = true}),
    };
    CHECK(refused_vertex(alu_program({other}, 1, 0), nullptr, registers)
              .find("reads R0.w before the shader writes it") != std::string::npos);
    // Nor in the group that writes it: the group reads first.
    const std::vector<std::array<u32, 2>> same{
        op2({.inst = alu_mov, .dst = 0, .chan = 0, .a = {src_one}}),
        op2({.inst = alu_mov, .dst = 1, .chan = 1, .a = {0, 0}, .last = true}),
    };
    CHECK(refused_vertex(alu_program({same}, 1, 0), nullptr, registers)
              .find("reads R0.x before the shader writes it") != std::string::npos);
}

TEST_CASE("what an ALU clause holds that is not translated yet is refused",
          "[latte][shader][ir][alu]") {
    const Registers registers;
    const auto refused = [&](const std::vector<std::array<u32, 2>>& clause, u32 cf_inst = 8) {
        return refused_pixel(alu_program({clause}, 0, 1, cf_inst), registers);
    };
    const auto mov = [](Src a, bool last = true) {
        return op2({.inst = alu_mov, .dst = 1, .a = a, .last = last});
    };

    // Uniforms: KCACHE, and the constant file.
    CHECK(refused({mov({128})}).find("reads a uniform") != std::string::npos);
    CHECK(refused({mov({300})}).find("reads a uniform") != std::string::npos);
    // PV and PS where the previous group left nothing: the first group, a
    // unit that ran a NOP, a T unit that ran nothing.
    CHECK(refused({mov({src_pv})}).find("reads PV.x, which the clause's previous group") !=
          std::string::npos);
    CHECK(refused({op2({.inst = alu_nop, .chan = 1}),
                   op2({.inst = alu_mov, .dst = 1, .a = {0}, .last = true}),
                   mov({src_pv, 1})})
              .find("reads PV.y") != std::string::npos);
    CHECK(refused({mov({0}), mov({src_ps})}).find("reads PS") != std::string::npos);
    // ...nor from one clause into the next.
    CHECK(refused_pixel(alu_program({{mov({0})}, {mov({src_pv})}}, 0, 1), registers)
              .find("reads PV.x") != std::string::npos);
    // Another instruction.
    CHECK(refused({op2({.inst = alu_floor, .dst = 1, .a = {0}, .last = true})})
              .find("FLOOR is not translated yet") != std::string::npos);
    // A predicated instruction, a relative source and destination.
    CHECK(refused({op2({.inst = alu_mov, .dst = 1, .a = {0}, .pred = 2, .last = true})})
              .find("is predicated") != std::string::npos);
    CHECK(refused({mov({0, 0, false, false, true})}).find("relative register") !=
          std::string::npos);
    std::array<u32, 2> relative = mov({0});
    relative[1] |= 1U << 28;
    CHECK(refused({relative}).find("writes a relative register") != std::string::npos);
    // A DOT4 on one unit.
    CHECK(refused({op2({.inst = alu_dot4, .dst = 1, .a = {0}, .b = {0}, .last = true})})
              .find("DOT4 on fewer than the four vector units") != std::string::npos);
    // A clause that pushes first.
    CHECK(refused({mov({0})}, 9).find("push, pop or branch") != std::string::npos);
}
