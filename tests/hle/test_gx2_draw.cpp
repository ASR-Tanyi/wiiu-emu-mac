// Phase 8.1 — GX2's draw calls.
//
// What is guarded: the packets each draw writes, run through a command
// processor and caught by its draw listener with the register file as it
// stood at the draw -- the base vertex and instance, the primitive, the
// index format (GX2's four values being VGT_DMA_INDEX_TYPE's, which is what
// Cemu writes them as), the index buffer found again from its physical
// address, and the initiator's tessellation mode; the predicate bit on the
// draw packet (Cemu's); the Ex2 forms putting the base instance back; and
// the calls refusing an index type that is not one, or indices the GPU
// cannot see.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_draw.hpp>
#include <wiiu/hle/gx2_fetch_shader.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/latte/draw.hpp>
#include <wiiu/latte/draw_shaders.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using latte::DrawCommand;
using latte::pm4::Opcode;
namespace reg = latte::reg;

namespace {

/// A draw as the listener saw it, with the registers it reads around it.
struct SeenDraw {
    DrawCommand command;
    u32 base_vertex = 0;
    u32 start_instance = 0;
};

[[nodiscard]] SeenDraw seen(const DrawCommand& command, const latte::RegisterFile& registers) {
    SeenDraw draw{.command = command};
    draw.command.immediate_indices = {};
    draw.base_vertex = registers.read(reg::sq_vtx_base_vtx_loc);
    draw.start_instance = registers.read(reg::sq_vtx_start_inst_loc);
    return draw;
}

/// A command processor over guest memory, collecting draws.
struct Ran {
    memory::Memory memory;
    std::unique_ptr<latte::RegisterFile> registers = std::make_unique<latte::RegisterFile>();
    latte::CommandProcessor processor{*registers};
    std::vector<SeenDraw> draws;

    explicit Ran(std::span<const u32> words) {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        processor.set_memory(&memory);
        processor.set_draw_listener(
            [this](const DrawCommand& command, const latte::RegisterFile& file) {
                draws.push_back(seen(command, file));
            });
        REQUIRE(processor.execute(words));
        CHECK(processor.stats().errors == 0);
        CHECK(processor.stats().unknown == 0);
    }
};

/// The opcodes of a stream's type-3 packets, and whether the last was
/// predicated.
struct Shape {
    std::vector<u8> opcodes;
    bool last_predicated = false;
};

[[nodiscard]] Shape shape(std::span<const u32> words) {
    Shape out;
    latte::pm4::Reader reader{words};
    while (const std::optional<latte::pm4::Packet> packet = reader.next()) {
        out.opcodes.push_back(packet->header.opcode);
        out.last_predicated = packet->header.predicate;
    }
    REQUIRE_FALSE(reader.failed());
    return out;
}

[[nodiscard]] constexpr u8 op(Opcode opcode) noexcept {
    return static_cast<u8>(opcode);
}

} // namespace

TEST_CASE("GX2's index types are VGT_DMA_INDEX_TYPE as they stand", "[hle][gx2][draw]") {
    for (const u32 type : {0U, 1U, 4U, 9U}) {
        CHECK(gx2::dma_index_type(type) == type);
    }
    latte::DmaIndexType u16{.raw = *gx2::dma_index_type(4)};
    CHECK(u16.index_size == latte::IndexSize::Bits16);
    CHECK(u16.swap_mode == latte::IndexSwap::Swap16);
    latte::DmaIndexType u32_le{.raw = *gx2::dma_index_type(1)};
    CHECK(u32_le.index_size == latte::IndexSize::Bits32);
    CHECK(u32_le.swap_mode == latte::IndexSwap::None);

    for (const u32 type : {2U, 3U, 5U, 8U, 13U, 0x10U}) {
        CHECK_FALSE(gx2::dma_index_type(type).has_value());
    }
}

TEST_CASE("an auto-indexed draw", "[hle][gx2][draw]") {
    std::vector<u32> words;
    gx2::append_draw(words, static_cast<u32>(latte::PrimitiveType::TriangleList), 3, 7, 2);

    const Shape packets = shape(words);
    CHECK(packets.opcodes == std::vector<u8>{op(Opcode::SetCtlConst),
                                             op(Opcode::SetConfigReg),
                                             op(Opcode::NumInstances),
                                             op(Opcode::DrawIndexAuto)});
    CHECK(packets.last_predicated);

    const Ran ran{words};
    REQUIRE(ran.draws.size() == 1);
    const SeenDraw& draw = ran.draws[0];
    CHECK(draw.command.kind == DrawCommand::Kind::Auto);
    CHECK(draw.command.count == 3);
    CHECK(draw.command.instances == 2);
    CHECK(draw.command.primitive == 4);
    CHECK(draw.command.draw_initiator == 0);
    CHECK(draw.base_vertex == 7);
}

TEST_CASE("an indexed draw finds its indices again", "[hle][gx2][draw]") {
    std::vector<u32> words;
    // Indices in MEM1, whose physical address 0x100 is 0xF4000100 to the CPU.
    gx2::append_draw_indexed(words,
                             static_cast<u32>(latte::PrimitiveType::TriangleStrip),
                             6,
                             *gx2::dma_index_type(9),
                             0x100,
                             0,
                             1);

    const Shape packets = shape(words);
    CHECK(packets.opcodes == std::vector<u8>{op(Opcode::SetCtlConst),
                                             op(Opcode::SetConfigReg),
                                             op(Opcode::IndexType),
                                             op(Opcode::NumInstances),
                                             op(Opcode::DrawIndex2)});
    CHECK(packets.last_predicated);
    CHECK(words[words.size() - 5] == 0xFFFFFFFF); // the maximum size
    CHECK(words[words.size() - 3] == 0);          // the address's high bits

    const Ran ran{words};
    REQUIRE(ran.draws.size() == 1);
    const DrawCommand& draw = ran.draws[0].command;
    CHECK(draw.kind == DrawCommand::Kind::Indexed);
    CHECK(draw.count == 6);
    CHECK(draw.primitive == 6);
    CHECK(draw.index_type == 9);
    CHECK(draw.indices_are_32bit());
    CHECK(draw.index_address == memory::mem1_base + 0x100);

    const latte::DrawInitiator initiator{.raw = draw.draw_initiator};
    CHECK(initiator.source_select == latte::SourceSelect::Dma);
    CHECK(initiator.major_mode == 0);
}

TEST_CASE("a tessellated indexed draw is major mode 1", "[hle][gx2][draw]") {
    std::vector<u32> words;
    const u32 mode = gx2::primitive_tessellate | 0x04;
    gx2::append_draw_indexed(words, mode, 3, 0, 0x100, 0, 1);
    const Ran ran{words};
    REQUIRE(ran.draws.size() == 1);
    CHECK(latte::DrawInitiator{.raw = ran.draws[0].command.draw_initiator}.major_mode == 1);
    CHECK(ran.draws[0].command.primitive == 0x84); // the mode as GX2 was given it (decaf)
}

// --------------------------------------------------------------------------
// Through GX2
// --------------------------------------------------------------------------

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kIndices = 0x10010000;
constexpr ppcaddr_t kRBuffer = 0x10004000;

struct DrawFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;
    std::vector<SeenDraw> draws;

    DrawFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
        call("GX2Init", {0});
        os.gpu.processor().set_draw_listener(
            [this](const DrawCommand& command, const latte::RegisterFile& file) {
                draws.push_back(seen(command, file));
            });
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
};

} // namespace

TEST_CASE_METHOD(DrawFixture, "GX2DrawEx and GX2DrawEx2 reach the GPU", "[hle][gx2][draw]") {
    call("GX2DrawEx", {4, 3, 0, 1});
    call("GX2DrawEx2", {0x13, 8, 16, 4, 2}); // quads, from vertex 16 and instance 2
    run_gpu();

    REQUIRE(draws.size() == 2);
    CHECK(draws[0].command.count == 3);
    CHECK(draws[0].base_vertex == 0);
    CHECK(draws[0].start_instance == 0);

    CHECK(draws[1].command.kind == DrawCommand::Kind::Auto);
    CHECK(draws[1].command.primitive == 0x13);
    CHECK(draws[1].command.count == 8);
    CHECK(draws[1].command.instances == 4);
    CHECK(draws[1].base_vertex == 16);
    CHECK(draws[1].start_instance == 2);
    CHECK(reg(reg::sq_vtx_start_inst_loc) == 0); // put back after the draw
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(DrawFixture,
                 "GX2DrawIndexedEx reads indices where the title put them",
                 "[hle][gx2][draw]") {
    call("GX2DrawIndexedEx", {4, 6, 4, kIndices, 10, 1});
    call("GX2DrawIndexedEx2", {4, 3, 1, kIndices + 0x40, 0, 2, 5});
    run_gpu();

    REQUIRE(draws.size() == 2);
    CHECK(draws[0].command.kind == DrawCommand::Kind::Indexed);
    CHECK(draws[0].command.index_address == kIndices);
    CHECK(draws[0].command.index_type == 4);
    CHECK_FALSE(draws[0].command.indices_are_32bit());
    CHECK(draws[0].base_vertex == 10);

    CHECK(draws[1].command.index_address == kIndices + 0x40);
    CHECK(draws[1].command.indices_are_32bit());
    CHECK(draws[1].command.instances == 2);
    CHECK(draws[1].start_instance == 5);
    CHECK(reg(reg::sq_vtx_start_inst_loc) == 0);
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(DrawFixture,
                 "a draw with no index type or no indices is refused",
                 "[hle][gx2][draw]") {
    run_gpu();
    const u64 before = os.gpu.processor().stats().packets;
    call("GX2DrawIndexedEx", {4, 3, 2, kIndices, 0, 1}); // 2 is not a GX2IndexType
    call("GX2DrawIndexedEx", {4, 3, 4, 0, 0, 1});        // null indices
    run_gpu();
    CHECK(os.gpu.processor().stats().packets == before);
    CHECK(draws.empty());
}

TEST_CASE_METHOD(DrawFixture,
                 "GX2RDrawIndexed starts its indices elements into the buffer",
                 "[hle][gx2][draw]") {
    gx2::RBuffer r{};
    r.element_size = 2;
    r.element_count = 0x100;
    r.buffer = kIndices;
    REQUIRE(gx2::write_guest(memory, kRBuffer, r));

    call("GX2RDrawIndexed", {4, kRBuffer, 4, 9, 3, 100, 1});
    run_gpu();

    REQUIRE(draws.size() == 1);
    CHECK(draws[0].command.index_address == kIndices + 6);
    CHECK(draws[0].command.count == 9);
    CHECK(draws[0].base_vertex == 100);
}

// --------------------------------------------------------------------------
// The translator's first slice (8.11): a draw's shaders, from GX2's calls
// --------------------------------------------------------------------------

namespace {

constexpr ppcaddr_t kFetchStruct = 0x10020000;
constexpr ppcaddr_t kFetchProgram = 0x10020100;
constexpr ppcaddr_t kAttribs = 0x10020400;
constexpr ppcaddr_t kVertexStruct = 0x10021000;
constexpr ppcaddr_t kVertexProgram = 0x10021400;
constexpr ppcaddr_t kPixelStruct = 0x10022000;
constexpr ppcaddr_t kPixelProgram = 0x10022400;

/// A program in the title's memory as the GPU reads it: little-endian words.
void put_program(memory::Memory& memory, ppcaddr_t address, const std::vector<u32>& words) {
    std::vector<u8> bytes(words.size() * 4);
    for (std::size_t i = 0; i < words.size(); ++i) {
        for (std::size_t b = 0; b < 4; ++b) {
            bytes[(4 * i) + b] = static_cast<u8>(words[i] >> (8 * b));
        }
    }
    REQUIRE(memory.write_bytes(address, bytes));
}

} // namespace

TEST_CASE_METHOD(DrawFixture,
                 "decaf's example triangle, from GX2's calls to MSL at the draw",
                 "[hle][gx2][draw][translator]") {
    // The fetch shader, from GX2's own generator: aColour at location 0 from
    // buffer 0, aPosition at 1 from buffer 1, four floats each.
    gx2::AttribStream colour{};
    colour.location = 0;
    colour.buffer = 0;
    colour.format = 0x813; // GX2_ATTRIB_FORMAT_FLOAT_32_32_32_32
    colour.mask = 0x00010203;
    colour.endian_swap = 3;
    gx2::AttribStream position = colour;
    position.location = 1;
    position.buffer = 1;
    REQUIRE(gx2::write_guest(memory, kAttribs, colour));
    REQUIRE(gx2::write_guest(memory, kAttribs + sizeof(gx2::AttribStream), position));
    call("GX2InitFetchShaderEx", {kFetchStruct, kFetchProgram, 2, kAttribs, 0, 0});
    call("GX2SetFetchShader", {kFetchStruct});

    // decaf's example vertex and pixel shaders (latte-assembler's
    // resources): their programs, and their headers' register values.
    constexpr u32 xyzw = 0x688;
    put_program(memory,
                kVertexProgram,
                {0,
                 0x13U << 23,
                 60U | (1U << 13) | (2U << 15),
                 xyzw | (0x28U << 23) | (1U << 31),
                 (2U << 13) | (1U << 15),
                 xyzw | (0x28U << 23) | (1U << 21)});
    gx2::VertexShader vs{};
    vs.sq_pgm_resources_vs = 3U | (1U << 8);
    vs.num_spi_vs_out_id = 1;
    vs.spi_vs_out_id[0] = 0xFFFFFF00; // parameter 0 is semantic 0
    vs.sq_vtx_semantic_clear = 0xFFFFFFFC;
    vs.num_sq_vtx_semantic = 2;
    vs.sq_vtx_semantic[0] = 0;
    vs.sq_vtx_semantic[1] = 1;
    vs.size = 24;
    vs.data = kVertexProgram;
    REQUIRE(gx2::write_guest(memory, kVertexStruct, vs));
    call("GX2SetVertexShader", {kVertexStruct});

    put_program(memory, kPixelProgram, {0, xyzw | (0x28U << 23) | (1U << 31) | (1U << 21)});
    gx2::PixelShader ps{};
    ps.sq_pgm_resources_ps = 1;
    ps.spi_ps_in_control_0 = 1;
    ps.num_spi_ps_input_cntl = 1;
    ps.spi_ps_input_cntls[0] = 0 | (1U << 8);
    ps.cb_shader_mask = 0xF;
    ps.size = 8;
    ps.data = kPixelProgram;
    REQUIRE(gx2::write_guest(memory, kPixelStruct, ps));
    call("GX2SetPixelShader", {kPixelStruct});

    // The translator runs at the draw, with the registers as they stand.
    bool built = false;
    std::string error;
    latte::DrawShaders shaders;
    os.gpu.processor().set_draw_listener(
        [&](const DrawCommand&, const latte::RegisterFile& registers) {
            built = latte::build_draw_shaders(registers, memory, shaders, error);
        });
    call("GX2DrawEx", {4, 3, 0, 1});
    run_gpu();

    INFO(error);
    REQUIRE(built);
    CHECK(shaders.has_fetch);
    CHECK(shaders.vertex_location.size == 24);
    CHECK(shaders.pixel_location.size == 8);
    CHECK(latte::ir::describe(shaders.vertex_ir) ==
          "vertex shader\n"
          "  attribute R1.xyzw <- semantic 0: buffer 0 + 0, 16 bytes as FMT 0x23 scaled, endian "
          "swap 2, per vertex\n"
          "  attribute R2.xyzw <- semantic 1: buffer 1 + 0, 16 bytes as FMT 0x23 scaled, endian "
          "swap 2, per vertex\n"
          "  fetch attributes\n"
          "  position <- R2.xyzw\n"
          "  parameter 0 (semantic 0) <- R1.xyzw\n");
    CHECK(latte::ir::describe(shaders.pixel_ir) ==
          "pixel shader\n"
          "  input R0 <- parameter 0 (semantic 0), perspective\n"
          "  colour 0 (target 0) <- R0.xyzw\n");
    // The MSL is the emitter's (test_msl.cpp has it word for word): here,
    // that it is there and names both buffers and the target.
    CHECK(shaders.vertex_msl.find("device const uchar* attrib1 [[buffer(2)]]") !=
          std::string::npos);
    CHECK(shaders.pixel_msl.find("float4 color0 [[color(0)]];") != std::string::npos);
}

TEST_CASE_METHOD(DrawFixture,
                 "a draw whose shaders cannot be translated says which and why",
                 "[hle][gx2][draw][translator]") {
    std::string error;
    bool built = true;
    latte::DrawShaders shaders;
    os.gpu.processor().set_draw_listener(
        [&](const DrawCommand&, const latte::RegisterFile& registers) {
            built = latte::build_draw_shaders(registers, memory, shaders, error);
        });

    // No vertex shader bound at all.
    call("GX2DrawEx", {4, 3, 0, 1});
    run_gpu();
    CHECK_FALSE(built);
    CHECK(error.find("no vertex shader") != std::string::npos);

    // A vertex shader whose program has an ALU clause.
    put_program(
        memory,
        kVertexProgram,
        {2, (8U << 26) | (1U << 31), 60U | (1U << 13), 0x688U | (0x28U << 23) | (1U << 21), 0, 0});
    gx2::VertexShader vs{};
    vs.size = 24;
    vs.data = kVertexProgram;
    REQUIRE(gx2::write_guest(memory, kVertexStruct, vs));
    call("GX2SetVertexShader", {kVertexStruct});
    put_program(memory, kPixelProgram, {0, 0x688U | (0x28U << 23) | (1U << 21)});
    gx2::PixelShader ps{};
    ps.size = 8;
    ps.data = kPixelProgram;
    REQUIRE(gx2::write_guest(memory, kPixelStruct, ps));
    call("GX2SetPixelShader", {kPixelStruct});
    call("GX2DrawEx", {4, 3, 0, 1});
    run_gpu();
    CHECK_FALSE(built);
    CHECK(error.find("the vertex shader at physical") != std::string::npos);
    CHECK(error.find("ALU clauses") != std::string::npos);
}
