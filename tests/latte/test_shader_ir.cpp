// Phase 8.6 — the intermediate form.
//
// What is guarded: decaf's example triangle -- its vertex shader, a fetch
// shader for its two attributes, its pixel shader and the registers its
// shaders' headers set -- described as the emitter will read it; the
// register lookups each fact depends on (a semantic's register is R(n + 1)
// for the SQ_VTX_SEMANTIC that names it, a parameter's semantic is its
// SPI_VS_OUT_ID byte, a pixel input is matched by semantic or takes its
// default, a pixel export goes to the n-th enabled colour target); bursts;
// and the refusal of everything not translated yet, by name.

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

    // An ALU clause, in a program with room for it.
    const std::vector<u32> alu{
        2, (8U << 26) | barrier, exp0(60, 1, 0), xyzw | exp_done | end_of_program, 0, 0};
    CHECK(refused_vertex(alu, nullptr, registers).find("ALU clauses") != std::string::npos);

    // A texture clause in a pixel shader.
    const std::vector<u32> tex{
        2, (1U << 23) | barrier, exp0(0, 0, 0), xyzw | exp_done | end_of_program, 0, 0, 0, 0};
    CHECK(refused_pixel(tex, registers).find("texture clauses") != std::string::npos);

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
