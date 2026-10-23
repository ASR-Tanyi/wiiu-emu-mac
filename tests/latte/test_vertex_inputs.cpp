// Phase 8.2 / 8.11 — guest memory on the GPU, and where a draw's vertices
// come from.
//
// What is guarded: guest RAM is wrapped in place, never copied (P22) -- the
// buffer's contents are the guest's own bytes -- in windows made once and
// found again, any range up to a window step lying whole in the window its
// start falls in, the last window of a region cut at the region's end; what
// no region holds whole is refused. And from the registers, the buffers a
// vertex shader reads -- each buffer its attributes name, found through its
// resource, with its stride and size and the instance step rates for the
// shader's FetchConstants -- bound at the emitter's slots. And the uniforms
// a shader's ALU reads: its stage's uniform blocks from the stage's
// constant cache, read in place, and its uniform registers copied as bytes,
// each at the emitter's slot; a block with nothing bound, or out of RAM,
// refused.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/latte/guest_buffers.hpp>
#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_state.hpp>
#include <wiiu/latte/uniform_inputs.hpp>
#include <wiiu/latte/vertex_inputs.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

constexpr u64 kMiB = 1ULL << 20;

/// The data heap: virtual 0x10000000, physical 0x20000000.
constexpr ppcaddr_t kData = memory::data_heap_base;
constexpr physaddr_t kDataPhysical = 0x20000000;

struct GuestFixture {
    memory::Memory memory;
    gpu::NullBackend backend;
    std::unique_ptr<GuestBuffers> buffers;

    GuestFixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        REQUIRE(backend.initialize(error));
        buffers = std::make_unique<GuestBuffers>(backend, memory);
    }

    ~GuestFixture() { buffers.reset(); }

    GuestFixture(const GuestFixture&) = delete;
    GuestFixture& operator=(const GuestFixture&) = delete;

    [[nodiscard]] GuestBuffers::Range find(physaddr_t address, u64 size) {
        std::string error;
        const std::optional<GuestBuffers::Range> range = buffers->find(address, size, error);
        INFO(error);
        REQUIRE(range.has_value());
        return *range;
    }

    [[nodiscard]] bool refused(physaddr_t address, u64 size, std::string_view because) {
        std::string error;
        const bool found = buffers->find(address, size, error).has_value();
        INFO(error);
        return !found && error.find(because) != std::string::npos;
    }
};

/// Attribute buffer `index`'s resource, as `GX2SetAttribBuffer` writes it.
void put_attribute_buffer(
    RegisterFile& registers, u32 index, physaddr_t address, u32 size, u32 stride) {
    const u32 base = resource::address(resource::attribute_buffer_0 + index);
    REQUIRE(registers.write(base, address));
    REQUIRE(registers.write(base + 4, size - 1));
    REQUIRE(registers.write(base + 8, stride << 8));
    REQUIRE(registers.write(base + 24, static_cast<u32>(ResourceType::ValidBuffer) << 30));
}

} // namespace

TEST_CASE_METHOD(GuestFixture,
                 "guest memory is wrapped in place, a window at a time",
                 "[latte][gpu][p22]") {
    const GuestBuffers::Range first = find(kDataPhysical + 0x1000, 64);
    CHECK(first.offset == 0x1000);
    CHECK(backend.recorded("size=268435456 label=guest data 0x10000000"));
    // The guest's own bytes, not a copy of them.
    CHECK(backend.buffer_contents(first.buffer) == memory.translate_unchecked(kData));

    // Found again, not wrapped again.
    const GuestBuffers::Range second = find(kDataPhysical + 0x2000, 64);
    CHECK(second.buffer == first.buffer);
    CHECK(second.offset == 0x2000);
    CHECK(buffers->size() == 1);

    // 192 MiB in starts in the second window, which begins 128 MiB in.
    const GuestBuffers::Range later = find(kDataPhysical + (192 * kMiB), 64);
    CHECK(later.buffer != first.buffer);
    CHECK(later.offset == 64 * kMiB);
    CHECK(buffers->size() == 2);

    // A range across the 128 MiB mark is whole in the window it starts in.
    const GuestBuffers::Range across = find(kDataPhysical + (120 * kMiB), 16 * kMiB);
    CHECK(across.buffer == first.buffer);
    CHECK(across.offset == 120 * kMiB);

    // The data heap is 1312 MiB: its last window starts at 1280 and is cut
    // at the end, 32 MiB long.
    const GuestBuffers::Range last = find(kDataPhysical + (1300 * kMiB), 4 * kMiB);
    CHECK(last.offset == 20 * kMiB);
    CHECK(backend.recorded("size=33554432 label=guest data 0x60000000"));

    // MEM1 is a region of its own, at physical zero.
    const GuestBuffers::Range mem1 = find(0x100, 16);
    CHECK(mem1.offset == 0x100);
    CHECK(backend.buffer_contents(mem1.buffer) == memory.translate_unchecked(memory::mem1_base));
    CHECK(backend.rejected() == 0);

    buffers->clear();
    CHECK(buffers->size() == 0);
    CHECK(backend.live_count(gpu::ResourceKind::Buffer) == 0);
}

TEST_CASE_METHOD(GuestFixture,
                 "what no region of RAM holds whole is not wrapped",
                 "[latte][gpu][p22]") {
    CHECK(refused(0x0A000000, 64, "in no region of RAM"));
    CHECK(refused(memory::mem1_size - 16, 64, "run past the end of mem1"));
    CHECK(refused(kDataPhysical, 0, "not a range"));
    CHECK(refused(kDataPhysical, GuestBuffers::max_range + 1, "not a range"));
    CHECK(buffers->size() == 0);
}

TEST_CASE_METHOD(GuestFixture,
                 "a draw binds the buffers its vertex shader reads, from their resources",
                 "[latte][gpu][draw]") {
    RegisterFile registers;
    put_attribute_buffer(registers, 0, kDataPhysical + 0x1000, 48, 12);
    put_attribute_buffer(registers, 2, kDataPhysical + 0x2000, 64, 16);
    put_attribute_buffer(registers, 3, kDataPhysical + 0x3000, 64, 16); // not read
    REQUIRE(registers.write(reg::vgt_instance_step_rate_0, 1));
    REQUIRE(registers.write(reg::vgt_instance_step_rate_0 + 4, 3));

    ir::Shader vertex;
    vertex.attributes = {ir::VertexAttribute{.buffer = 2},
                         ir::VertexAttribute{.buffer = 0},
                         ir::VertexAttribute{.buffer = 2}};

    VertexInputs inputs;
    std::string error;
    REQUIRE(vertex_inputs(registers, vertex, *buffers, inputs, error));
    REQUIRE(inputs.buffers.size() == 2);
    CHECK(inputs.buffers[0].index == 0);
    CHECK(inputs.buffers[0].range.offset == 0x1000);
    CHECK(inputs.buffers[1].index == 2);
    CHECK(inputs.buffers[1].range.offset == 0x2000);
    CHECK(inputs.constants.stride[0] == 12);
    CHECK(inputs.constants.size[0] == 48);
    CHECK(inputs.constants.stride[2] == 16);
    CHECK(inputs.constants.size[2] == 64);
    CHECK(inputs.constants.stride[3] == 0); // bound by the title, read by nothing
    CHECK(inputs.constants.step_rate == std::array<u32, 2>{1, 3});

    // Bound at the emitter's slots: the constants at 0, buffer n at 1 + n.
    gpu::TextureDesc desc;
    desc.width = 4;
    desc.height = 4;
    desc.format = gpu::TextureFormat::RGBA8Unorm;
    desc.usage = gpu::TextureUsage::RenderTarget;
    gpu::RenderPassDesc pass;
    pass.color_count = 1;
    pass.colors[0].texture = backend.create_texture(desc);
    pass.colors[0].load = gpu::LoadAction::DontCare;
    pass.colors[0].store = gpu::StoreAction::Store;
    REQUIRE(backend.begin_render_pass(pass));
    bind_vertex_inputs(backend, inputs);
    backend.end_render_pass();

    const std::string buffer = std::to_string(inputs.buffers[0].range.buffer.value());
    CHECK(backend.recorded("bind_bytes vertex [0] 144 bytes"));
    CHECK(backend.recorded("bind_buffer vertex [1] = " + buffer + " +4096"));
    CHECK(backend.recorded("bind_buffer vertex [3] = " + buffer + " +8192"));
    msl::FetchConstants bound{};
    REQUIRE(backend.last_bytes().size() == sizeof(bound));
    std::memcpy(&bound, backend.last_bytes().data(), sizeof(bound));
    CHECK(bound.stride == inputs.constants.stride);
    CHECK(bound.size == inputs.constants.size);
    CHECK(bound.step_rate == inputs.constants.step_rate);
    CHECK(backend.rejected() == 0);
}

TEST_CASE_METHOD(GuestFixture,
                 "a buffer the shader reads and cannot have is refused",
                 "[latte][gpu][draw]") {
    RegisterFile registers;
    VertexInputs inputs;
    std::string error;

    ir::Shader vertex;
    vertex.attributes = {ir::VertexAttribute{.buffer = 5}};
    CHECK_FALSE(vertex_inputs(registers, vertex, *buffers, inputs, error));
    CHECK(error.find("attribute buffer 5 (resource 0x145)") != std::string::npos);

    put_attribute_buffer(registers, 1, 0x0A000000, 64, 16);
    vertex.attributes = {ir::VertexAttribute{.buffer = 1}};
    CHECK_FALSE(vertex_inputs(registers, vertex, *buffers, inputs, error));
    CHECK(error.find("attribute buffer 1: physical 0x0a000000") != std::string::npos);

    vertex.attributes = {ir::VertexAttribute{.buffer = 16}};
    CHECK_FALSE(vertex_inputs(registers, vertex, *buffers, inputs, error));
    CHECK(error.find("attribute buffer 16") != std::string::npos);
}

// --------------------------------------------------------------------------
// 8.12 — uniforms
// --------------------------------------------------------------------------

namespace {

/// A shader of one ALU group whose sources are `sources`, one MOV each.
[[nodiscard]] ir::Shader reading(ir::Stage stage, const std::vector<ir::AluSource>& sources) {
    ir::Shader shader;
    shader.stage = stage;
    ir::AluGroup group;
    u32 unit = 0;
    for (const ir::AluSource& source : sources) {
        ir::AluOperation op;
        op.unit = static_cast<AluUnit>(unit++);
        op.opcode = ir::AluOpcode::Mov;
        op.sources[0] = source;
        group.operations.push_back(op);
    }
    shader.body.emplace_back(group);
    return shader;
}

[[nodiscard]] ir::AluSource block_source(u32 block, u32 index) {
    return {.kind = ir::SourceKind::UniformBlock, .block = block, .index = index};
}

[[nodiscard]] ir::AluSource register_source(u32 index) {
    return {.kind = ir::SourceKind::UniformRegister, .index = index};
}

/// Uniform block `block` of the stage as `GX2Set…UniformBlock` leaves the
/// constant cache: the address >> 8, and the size in 256-byte units.
void put_uniform_block(
    RegisterFile& registers, ir::Stage stage, u32 block, physaddr_t address, u32 size) {
    const bool vertex = stage == ir::Stage::Vertex;
    REQUIRE(registers.write((vertex ? reg::sq_alu_const_cache_vs_0 : reg::sq_alu_const_cache_ps_0) +
                                (4 * block),
                            address >> 8));
    REQUIRE(registers.write(
        (vertex ? reg::sq_alu_const_buffer_size_vs_0 : reg::sq_alu_const_buffer_size_ps_0) +
            (4 * block),
        (size + 255) >> 8));
}

void in_a_pass(gpu::NullBackend& backend, const auto& bind) {
    gpu::TextureDesc desc;
    desc.width = 4;
    desc.height = 4;
    desc.format = gpu::TextureFormat::RGBA8Unorm;
    desc.usage = gpu::TextureUsage::RenderTarget;
    gpu::RenderPassDesc pass;
    pass.color_count = 1;
    pass.colors[0].texture = backend.create_texture(desc);
    pass.colors[0].load = gpu::LoadAction::DontCare;
    pass.colors[0].store = gpu::StoreAction::Store;
    REQUIRE(backend.begin_render_pass(pass));
    bind();
    backend.end_render_pass();
}

} // namespace

TEST_CASE_METHOD(GuestFixture,
                 "a draw binds the uniform blocks its shader reads, from the constant cache",
                 "[latte][gpu][draw][uniforms]") {
    RegisterFile registers;
    put_uniform_block(registers, ir::Stage::Pixel, 1, kDataPhysical + 0x4000, 64);
    put_uniform_block(registers, ir::Stage::Pixel, 4, kDataPhysical + 0x5000, 512);
    put_uniform_block(registers, ir::Stage::Pixel, 6, kDataPhysical + 0x6000, 64); // not read
    put_uniform_block(registers, ir::Stage::Vertex, 1, kDataPhysical + 0x7000, 64);

    const ir::Shader pixel =
        reading(ir::Stage::Pixel, {block_source(4, 20), block_source(1, 0), block_source(4, 2)});
    UniformInputs inputs;
    std::string error;
    REQUIRE(uniform_inputs(registers, pixel, *buffers, inputs, error));
    CHECK(inputs.stage == gpu::ShaderStage::Fragment);
    CHECK(inputs.registers.empty());
    REQUIRE(inputs.blocks.size() == 2);
    CHECK(inputs.blocks[0].block == 1);
    CHECK(inputs.blocks[0].range.offset == 0x4000);
    CHECK(inputs.blocks[1].block == 4);
    CHECK(inputs.blocks[1].range.offset == 0x5000);

    // At the emitter's slots: block n at 18 + n.
    in_a_pass(backend, [&] { bind_uniform_inputs(backend, inputs); });
    const std::string buffer = std::to_string(inputs.blocks[0].range.buffer.value());
    CHECK(backend.recorded("bind_buffer fragment [19] = " + buffer + " +16384"));
    CHECK(backend.recorded("bind_buffer fragment [22] = " + buffer + " +20480"));
    CHECK(backend.rejected() == 0);

    // The vertex shader's come from its own constant cache.
    const ir::Shader vertex = reading(ir::Stage::Vertex, {block_source(1, 0)});
    REQUIRE(uniform_inputs(registers, vertex, *buffers, inputs, error));
    CHECK(inputs.stage == gpu::ShaderStage::Vertex);
    REQUIRE(inputs.blocks.size() == 1);
    CHECK(inputs.blocks[0].range.offset == 0x7000);
}

TEST_CASE_METHOD(GuestFixture,
                 "a draw binds the stage's uniform registers as bytes",
                 "[latte][gpu][draw][uniforms]") {
    RegisterFile registers;
    // C3 of the pixel shader, and C3 of the vertex shader, 256 registers on.
    for (u32 word = 0; word < 4; ++word) {
        REQUIRE(registers.write(reg::sq_alu_constant0_0 + (16 * 3) + (4 * word), 0x100 + word));
        REQUIRE(registers.write(reg::sq_alu_constant0_256 + (16 * 3) + (4 * word), 0x200 + word));
    }
    const auto word_at = [](const std::vector<u8>& bytes, std::size_t index) {
        u32 value = 0;
        std::memcpy(&value, bytes.data() + (index * 4), 4);
        return value;
    };

    UniformInputs inputs;
    std::string error;
    REQUIRE(uniform_inputs(
        registers, reading(ir::Stage::Pixel, {register_source(3)}), *buffers, inputs, error));
    REQUIRE(inputs.registers.size() == 4096);
    CHECK(word_at(inputs.registers, 12) == 0x100);
    CHECK(word_at(inputs.registers, 15) == 0x103);
    CHECK(inputs.blocks.empty());

    REQUIRE(uniform_inputs(
        registers, reading(ir::Stage::Vertex, {register_source(3)}), *buffers, inputs, error));
    CHECK(word_at(inputs.registers, 12) == 0x200);
    CHECK(word_at(inputs.registers, 15) == 0x203);

    in_a_pass(backend, [&] { bind_uniform_inputs(backend, inputs); });
    CHECK(backend.recorded("bind_bytes vertex [17] 4096 bytes"));
    CHECK(backend.last_bytes() == inputs.registers);
    CHECK(backend.rejected() == 0);

    // A shader that reads no uniforms binds nothing.
    REQUIRE(uniform_inputs(
        registers, reading(ir::Stage::Pixel, {ir::AluSource{}}), *buffers, inputs, error));
    CHECK(inputs.registers.empty());
    CHECK(inputs.blocks.empty());
}

TEST_CASE_METHOD(GuestFixture,
                 "a uniform block the shader reads and cannot have is refused",
                 "[latte][gpu][draw][uniforms]") {
    RegisterFile registers;
    UniformInputs inputs;
    std::string error;
    const ir::Shader pixel = reading(ir::Stage::Pixel, {block_source(2, 0)});
    CHECK_FALSE(uniform_inputs(registers, pixel, *buffers, inputs, error));
    CHECK(error.find("uniform block 2 is read, and none is bound (SQ_ALU_CONST_CACHE_PS_2") !=
          std::string::npos);

    put_uniform_block(registers, ir::Stage::Pixel, 2, 0x0A000000, 64);
    CHECK_FALSE(uniform_inputs(registers, pixel, *buffers, inputs, error));
    CHECK(error.find("uniform block 2: physical 0x0a000000") != std::string::npos);
}
