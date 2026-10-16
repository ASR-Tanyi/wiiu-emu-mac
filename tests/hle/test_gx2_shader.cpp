// Phase 8.1 — binding shaders, uniform registers and GX2Invalidate.
//
// What is guarded: each shader structure's layout (static_asserts in the
// header) and the registers binding it writes -- decaf's sequence, at the
// addresses Linux's register lists give -- run through a command processor
// and read back; the GX2R fallback for a program's address; the uniform
// registers' place in the ALU constants; and the SURFACE_SYNC GX2Invalidate
// writes, against decaf's bits and Linux's names for them.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_fetch_shader.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/shader_ir.hpp>
#include <wiiu/latte/shader_program.hpp>
#include <wiiu/latte/shader_state.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using latte::pm4::Opcode;
namespace reg = latte::reg;

namespace {

/// A register file after running `words`, checked to have run cleanly.
struct Ran {
    std::unique_ptr<latte::RegisterFile> registers = std::make_unique<latte::RegisterFile>();
    latte::CommandProcessor processor{*registers};

    Ran() = default;

    explicit Ran(std::span<const u32> words) { run(words); }

    void run(std::span<const u32> words) {
        REQUIRE(processor.execute(words));
        CHECK(processor.stats().errors == 0);
        CHECK(processor.stats().unknown == 0);
    }

    [[nodiscard]] u32 operator[](u32 address) const { return registers->read(address); }
};

[[nodiscard]] gx2::VertexShader sample_vertex_shader() {
    gx2::VertexShader shader{};
    shader.sq_pgm_resources_vs = 0x0000020C; // 12 GPRs, 2 stack entries
    shader.vgt_primitiveid_en = 1;
    shader.spi_vs_out_config = 0x4;
    shader.num_spi_vs_out_id = 3;
    shader.spi_vs_out_id[0] = 0x03020100;
    shader.spi_vs_out_id[2] = 0x0B0A0908;
    shader.pa_cl_vs_out_cntl = 0x100;
    shader.sq_vtx_semantic_clear = 0xFFFFFFFC; // keep 0 and 1
    shader.num_sq_vtx_semantic = 2;
    shader.sq_vtx_semantic[0] = 0x00;
    shader.sq_vtx_semantic[1] = 0x01;
    shader.vgt_strmout_buffer_en = 0;
    shader.vgt_vertex_reuse_block_cntl = 14;
    shader.vgt_hos_reuse_depth = 16;
    shader.size = 0x140;
    shader.ring_item_size = 0x20;
    return shader;
}

} // namespace

TEST_CASE("binding a vertex shader writes decaf's registers", "[hle][gx2][shader]") {
    const gx2::VertexShader shader = sample_vertex_shader();
    std::vector<u32> words;
    gx2::append_vertex_shader(words, shader, gx2::Program{.address = 0x00123400, .size = 0x140});

    Ran ran;
    // Semantics the shader does not use, set before: the clear empties them.
    const std::vector<u32> earlier{latte::pm4::type3_header(Opcode::SetContextReg, 2),
                                   (reg::sq_vtx_semantic_0 + (4 * 10) - 0x28000) / 4,
                                   0x22};
    ran.run(earlier);
    ran.run(words);

    CHECK(ran[reg::sq_pgm_start_vs] == 0x1234);
    CHECK(ran[reg::sq_pgm_start_vs + 4] == 0x140 >> 3);
    CHECK(ran[reg::sq_pgm_start_vs + 8] == gx2::program_unnamed_value);
    CHECK(ran[reg::sq_pgm_resources_vs] == 0x20C);
    CHECK(ran[reg::vgt_primitiveid_en] == 1);
    CHECK(ran[reg::spi_vs_out_config] == 4);
    CHECK(ran[reg::pa_cl_vs_out_cntl] == 0x100);
    CHECK(ran[reg::spi_vs_out_id_0] == 0x03020100);
    CHECK(ran[reg::spi_vs_out_id_0 + 8] == 0x0B0A0908);
    CHECK(ran[reg::sq_pgm_cf_offset_vs] == 0);
    CHECK(ran[reg::sq_vtx_semantic_0] == 0);
    CHECK(ran[reg::sq_vtx_semantic_0 + 4] == 1);
    CHECK(ran[reg::sq_vtx_semantic_0 + 40] == 0xFFFFFFFF); // cleared
    CHECK(ran[reg::vgt_vertex_reuse_block_cntl] == 14);
    CHECK(ran[reg::vgt_hos_reuse_depth] == 16);
    CHECK(ran[reg::sq_pgm_start_es] == 0); // not geometry mode
}

TEST_CASE("in geometry mode a vertex shader is the ES stage", "[hle][gx2][shader]") {
    gx2::VertexShader shader = sample_vertex_shader();
    shader.mode = static_cast<u32>(gx2::ShaderMode::GeometryShader);
    shader.num_spi_vs_out_id = 50; // past the array: held to its ten
    std::vector<u32> words;
    gx2::append_vertex_shader(words, shader, gx2::Program{.address = 0x00200000, .size = 0x80});
    const Ran ran{words};
    CHECK(ran[reg::sq_pgm_start_es] == 0x2000);
    CHECK(ran[reg::sq_esgs_ring_itemsize] == 0x20);
    CHECK(ran[reg::sq_pgm_start_vs] == 0); // the copy shader's, set with the geometry shader
    CHECK(ran[reg::spi_vs_out_config] == 0);
    CHECK(ran[reg::sq_vtx_semantic_0 + 4] == 1);
}

TEST_CASE("binding a pixel shader writes its exports and turns dual export on",
          "[hle][gx2][shader]") {
    gx2::PixelShader shader{};
    shader.sq_pgm_resources_ps = 0x00000105;
    shader.sq_pgm_exports_ps = 0x2;
    shader.spi_ps_in_control_0 = 0x10000002;
    shader.spi_ps_in_control_1 = 0x1;
    shader.num_spi_ps_input_cntl = 2;
    shader.spi_ps_input_cntls[0] = 0x100;
    shader.spi_ps_input_cntls[1] = 0x101;
    shader.cb_shader_mask = 0xF;
    shader.cb_shader_control = 0x1;
    shader.db_shader_control = 0x10;
    shader.spi_input_z = 0;
    std::vector<u32> words;
    gx2::append_pixel_shader(words, shader, gx2::Program{.address = 0x00300000, .size = 0x100});
    const Ran ran{words};

    CHECK(ran[reg::sq_pgm_start_ps] == 0x3000);
    CHECK(ran[reg::sq_pgm_start_ps + 4] == 0x20);
    CHECK(ran[reg::sq_pgm_resources_ps] == 0x105);
    CHECK(ran[reg::sq_pgm_exports_ps] == 2); // Cemu leaves this one out
    CHECK(ran[reg::spi_ps_in_control_0] == 0x10000002);
    CHECK(ran[reg::spi_ps_in_control_0 + 4] == 1);
    CHECK(ran[reg::spi_ps_input_cntl_0 + 4] == 0x101);
    CHECK(ran[reg::cb_shader_mask] == 0xF);
    CHECK(ran[reg::cb_shader_control] == 1);
    CHECK(ran[reg::db_shader_control] == (0x10U | (1U << 9)));
}

TEST_CASE("a geometry shader sets its copy shader as the VS stage", "[hle][gx2][shader]") {
    gx2::GeometryShader shader{};
    shader.sq_pgm_resources_gs = 0x8;
    shader.vgt_gs_out_prim_type = 2;
    shader.vgt_gs_mode = 0x1;
    shader.pa_cl_vs_out_cntl = 0x200;
    shader.sq_pgm_resources_vs = 0x6;
    shader.sq_gs_vert_itemsize = 4;
    shader.spi_vs_out_config = 0x2;
    shader.num_spi_vs_out_id = 1;
    shader.spi_vs_out_id[0] = 0x0100;
    shader.ring_item_size = 0x18010; // held to 15 bits
    shader.has_stream_out = 1;
    shader.stream_out_stride[1] = 32;
    std::vector<u32> words;
    gx2::append_geometry_shader(words,
                                shader,
                                gx2::Program{.address = 0x00400000, .size = 0x200},
                                gx2::Program{.address = 0x00500000, .size = 0x40});
    const Ran ran{words};

    CHECK(ran[reg::sq_pgm_start_gs] == 0x4000);
    CHECK(ran[reg::sq_pgm_start_gs + 16] == 0x8); // SQ_PGM_RESOURCES_GS
    CHECK(ran[reg::vgt_gs_out_prim_type] == 2);
    CHECK(ran[0x28A40] == 1); // VGT_GS_MODE
    CHECK(ran[reg::sq_gs_vert_itemsize] == 4);
    CHECK(ran[reg::vgt_strmout_vtx_stride_0 + 16] == 8); // 32 bytes, in words
    CHECK(ran[reg::sq_pgm_start_vs] == 0x5000);
    CHECK(ran[reg::sq_pgm_resources_vs] == 6);
    CHECK(ran[reg::pa_cl_vs_out_cntl] == 0x200);
    CHECK(ran[reg::spi_vs_out_id_0] == 0x100);
    CHECK(ran[reg::spi_vs_out_config] == 2);
    CHECK(ran[reg::sq_gsvs_ring_itemsize] == 0x0010);
}

TEST_CASE("decaf's example triangle, from GX2's registers to the intermediate form",
          "[hle][gx2][shader]") {
    // tools/latte-assembler/resources/example_shader.vsh and .psh: their
    // header comments are the shader structures' register values, and the
    // fetch shader is GX2's own for the two attributes the vertex shader
    // names -- aColour at location 0 and aPosition at 1, both four floats.
    gx2::VertexShader vs{};
    vs.sq_pgm_resources_vs = 3U | (1U << 8);
    vs.num_spi_vs_out_id = 1;
    vs.spi_vs_out_id[0] = 0; // SEMANTIC_0 = 0
    vs.sq_vtx_semantic_clear = 0xFFFFFFFC;
    vs.num_sq_vtx_semantic = 2;
    vs.sq_vtx_semantic[0] = 0;
    vs.sq_vtx_semantic[1] = 1;
    vs.vgt_vertex_reuse_block_cntl = 14;
    vs.vgt_hos_reuse_depth = 16;
    gx2::PixelShader ps{};
    ps.sq_pgm_resources_ps = 1;
    ps.spi_ps_in_control_0 = 1; // NUM_INTERP
    ps.num_spi_ps_input_cntl = 1;
    ps.spi_ps_input_cntls[0] = 0 | (1U << 8); // semantic 0, default 0001
    ps.cb_shader_mask = 0xF;                  // one colour output, as a compiled shader has
    std::vector<u32> words;
    gx2::append_vertex_shader(words, vs, gx2::Program{.address = 0x1000, .size = 24});
    gx2::append_pixel_shader(words, ps, gx2::Program{.address = 0x2000, .size = 8});
    const Ran ran{words};

    gx2::AttribStream colour{};
    colour.location = 0;
    colour.buffer = 0;
    colour.format = 0x813; // GX2_ATTRIB_FORMAT_FLOAT_32_32_32_32
    colour.mask = 0x00010203;
    colour.endian_swap = 3;
    gx2::AttribStream position = colour;
    position.location = 1;
    position.buffer = 1;
    const std::array attribs{colour, position};
    const gx2::FetchShaderProgram fetch = gx2::generate_fetch_shader(attribs);

    constexpr u32 xyzw = 0x688;
    const std::vector<u32> vertex_words{0,
                                        0x13U << 23,
                                        60U | (1U << 13) | (2U << 15),
                                        xyzw | (0x28U << 23) | (1U << 31),
                                        (2U << 13) | (1U << 15),
                                        xyzw | (0x28U << 23) | (1U << 21)};
    const std::vector<u32> pixel_words{0, xyzw | (0x28U << 23) | (1U << 31) | (1U << 21)};

    std::string error;
    latte::ShaderProgram vertex_program;
    latte::ShaderProgram fetch_program;
    latte::ShaderProgram pixel_program;
    REQUIRE(
        latte::decode_program(vertex_words, latte::ProgramShape::Shader, vertex_program, error));
    REQUIRE(
        latte::decode_program(fetch.words, latte::ProgramShape::FetchShader, fetch_program, error));
    REQUIRE(latte::decode_program(pixel_words, latte::ProgramShape::Shader, pixel_program, error));

    latte::ir::Shader vertex;
    latte::ir::Shader pixel;
    INFO(error);
    REQUIRE(latte::ir::build_vertex_shader(
        vertex_program, &fetch_program, *ran.registers, vertex, error));
    REQUIRE(latte::ir::build_pixel_shader(pixel_program, *ran.registers, pixel, error));
    CHECK(latte::ir::describe(vertex) ==
          "vertex shader\n"
          "  attribute R1.xyzw <- semantic 0: buffer 0 + 0, 16 bytes as FMT 0x23 scaled, endian "
          "swap 2, per vertex\n"
          "  attribute R2.xyzw <- semantic 1: buffer 1 + 0, 16 bytes as FMT 0x23 scaled, endian "
          "swap 2, per vertex\n"
          "  fetch attributes\n"
          "  position <- R2.xyzw\n"
          "  parameter 0 (semantic 0) <- R1.xyzw\n");
    CHECK(latte::ir::describe(pixel) == "pixel shader\n"
                                        "  input R0 <- parameter 0 (semantic 0), perspective\n"
                                        "  colour 0 (target 0) <- R0.xyzw\n");
}

TEST_CASE("GX2Invalidate's SURFACE_SYNC", "[hle][gx2][shader]") {
    namespace coher = latte::pm4::coher;
    // A texture written by the CPU: the texture cache, over its bytes.
    const std::vector<u32> texture =
        gx2::invalidate_packets(gx2::invalidate::texture, 0x00100000, 0x101);
    REQUIRE(texture.size() == 5);
    CHECK(texture[0] == latte::pm4::type3_header(Opcode::SurfaceSync, 4));
    CHECK(texture[1] == (coher::engine_me | coher::tc_action_ena));
    CHECK(texture[2] == 2); // 0x101 bytes is two 256-byte units
    CHECK(texture[3] == 0x1000);
    CHECK(texture[4] == 4);

    // Shaders, everywhere: the full cache too (decaf).
    const std::vector<u32> shaders =
        gx2::invalidate_packets(gx2::invalidate::shader, 0, 0xFFFFFFFF);
    CHECK(shaders[1] == (coher::engine_me | coher::full_cache_ena | coher::sh_action_ena));
    CHECK(shaders[2] == 0xFFFFFF);

    // Render targets: the colour block's eight destinations and its action.
    const std::vector<u32> colour =
        gx2::invalidate_packets(gx2::invalidate::color_buffer, 0, 0xFFFFFFFF);
    CHECK(colour[1] == (coher::engine_me | (0xFFU << 6) | coher::cb_action_ena));

    // The CPU's cache alone, or nothing: no packet.
    CHECK(gx2::invalidate_packets(gx2::invalidate::cpu, 0x100000, 64).empty());
    CHECK(gx2::invalidate_packets(0, 0, 0).empty());
}

// --------------------------------------------------------------------------
// Through GX2
// --------------------------------------------------------------------------

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kShader = 0x10004000;
constexpr ppcaddr_t kProgram = 0x10010000;
constexpr ppcaddr_t kValues = 0x10006000;

struct ShaderFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    ShaderFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
        call("GX2Init", {0});
    }

    u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    void run_gpu() {
        call("GX2Flush", {});
        state.timebase += os.gpu.timeline().latency();
        static_cast<void>(os.scheduler.reschedule(state));
    }

    [[nodiscard]] u32 reg(u32 address) const { return os.gpu.registers().read(address); }

    [[nodiscard]] u32 physical(ppcaddr_t address) const {
        const memory::Region* const region = memory.region_of(address);
        REQUIRE(region != nullptr);
        physaddr_t out = 0;
        REQUIRE(memory::to_physical(*region, address, out));
        return out;
    }
};

} // namespace

TEST_CASE_METHOD(ShaderFixture,
                 "GX2SetVertexShader and GX2SetPixelShader reach the register file",
                 "[hle][gx2][shader]") {
    gx2::VertexShader vertex = sample_vertex_shader();
    vertex.data = kProgram;
    REQUIRE(gx2::write_guest(memory, kShader, vertex));
    call("GX2SetVertexShader", {kShader});

    // A pixel shader whose program is in a GX2R buffer: 0x10 elements of 0x10.
    gx2::PixelShader pixel{};
    pixel.sq_pgm_resources_ps = 0x0304;
    pixel.gx2r.buffer = kProgram + 0x1000;
    pixel.gx2r.element_size = 0x10;
    pixel.gx2r.element_count = 0x10;
    REQUIRE(gx2::write_guest(memory, kShader + 0x200, pixel));
    call("GX2SetPixelShader", {kShader + 0x200});
    run_gpu();

    CHECK(reg(reg::sq_pgm_start_vs) == physical(kProgram) >> 8);
    CHECK(reg(reg::sq_vtx_semantic_0 + 4) == 1);
    CHECK(reg(reg::sq_pgm_start_ps) == physical(kProgram + 0x1000) >> 8);
    CHECK(reg(reg::sq_pgm_start_ps + 4) == 0x100 >> 3);
    CHECK(os.gpu.processor().stats().errors == 0);

    CHECK(call("GX2GetVertexShaderGPRs", {kShader}) == 12);
    CHECK(call("GX2GetVertexShaderStackEntries", {kShader}) == 2);
    CHECK(call("GX2GetPixelShaderGPRs", {kShader + 0x200}) == 4);
    CHECK(call("GX2GetPixelShaderStackEntries", {kShader + 0x200}) == 3);
}

TEST_CASE_METHOD(ShaderFixture, "a shader with no program is refused", "[hle][gx2][shader]") {
    run_gpu();
    const u64 before = os.gpu.processor().stats().packets;
    REQUIRE(gx2::write_guest(memory, kShader, gx2::PixelShader{}));
    call("GX2SetPixelShader", {kShader});
    run_gpu();
    CHECK(os.gpu.processor().stats().packets == before);
}

TEST_CASE_METHOD(ShaderFixture,
                 "uniform registers land in the stage's ALU constants",
                 "[hle][gx2][shader]") {
    for (u32 i = 0; i < 8; ++i) {
        REQUIRE(memory.try_write<u32>(kValues + (4 * i), 0x3F800000 + i));
    }
    call("GX2SetVertexUniformReg", {4, 8, kValues});
    call("GX2SetPixelUniformReg", {0, 4, kValues});
    call("GX2SetPixelUniformReg", {0x10000, 4, kValues}); // a loop constant: not yet
    run_gpu();
    CHECK(reg(reg::sq_alu_constant0_256 + 16) == 0x3F800000);
    CHECK(reg(reg::sq_alu_constant0_256 + 44) == 0x3F800007);
    CHECK(reg(reg::sq_alu_constant0_0 + 12) == 0x3F800003);
    CHECK(reg(reg::sq_alu_constant0_0 + 16) == 0);
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(ShaderFixture,
                 "GX2SetFetchShader and GX2Invalidate reach the GPU",
                 "[hle][gx2][shader]") {
    gx2::FetchShader fetch{};
    fetch.sq_pgm_resources_fs = 2;
    fetch.size = 0x40;
    fetch.data = kProgram;
    fetch.divisors[0] = 3;
    REQUIRE(gx2::write_guest(memory, kShader, fetch));
    call("GX2SetFetchShader", {kShader});
    call("GX2Invalidate", {gx2::invalidate::shader, kProgram, 0x40});
    run_gpu();

    CHECK(reg(reg::sq_pgm_start_fs) == physical(kProgram) >> 8);
    CHECK(reg(reg::sq_pgm_start_fs + 4) == 8);
    CHECK(reg(reg::sq_pgm_resources_fs) == 2);
    CHECK(reg(reg::vgt_instance_step_rate_0) == 3);
    CHECK(os.gpu.processor().unhandled().at(static_cast<u8>(Opcode::SurfaceSync)) >= 1);
}
