// Phase 8.1 — attribute buffers and uniform blocks.
//
// What is guarded: where each buffer lands in the resource window -- the
// stage bases, against the fetch shader's view of its own buffers and the
// window offsets Cemu writes -- and the words in it, against decaf's fields,
// Linux's stride position and Cemu's literal type word; the constant-cache
// registers a uniform block also sets; and the calls' handling of the
// title's arguments: null buffers, indices past sixteen, strides wider than
// the register, and GX2R buffers read from an offset.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_buffers.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/latte/resources.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
namespace res = latte::resource;
namespace reg = latte::reg;

namespace {

/// A register file after running `words`, checked to have run cleanly.
struct Ran {
    std::unique_ptr<latte::RegisterFile> registers = std::make_unique<latte::RegisterFile>();
    latte::CommandProcessor processor{*registers};

    explicit Ran(std::span<const u32> words) {
        REQUIRE(processor.execute(words));
        CHECK(processor.stats().errors == 0);
        CHECK(processor.stats().unknown == 0);
    }

    [[nodiscard]] u32 operator[](u32 address) const { return registers->read(address); }
};

} // namespace

TEST_CASE("the resource window is shared out as decaf and Cemu share it", "[hle][gx2][buffers]") {
    // The fetch shader names attribute buffer n as its stage's 0xA0 + n
    // (gx2_fetch_shader.cpp); counted from the vertex stage's first
    // resource, that is the attribute buffers' first.
    CHECK(res::vs_base + 0xA0 == res::attribute_buffer_0);
    CHECK(res::attribute_buffer_0 == 0x140);
    CHECK(res::attribute_buffer_0 + res::attribute_buffer_count == res::gs_base);
    CHECK(res::gs_base + res::per_stage == 0x1F0);

    // Cemu's SET_RESOURCE offsets, in words from the window's start.
    CHECK(res::address(res::attribute_buffer_0) == res::window_base + (4 * 0x8C0));
    CHECK(res::address(res::vs_uniform_block_0) == res::window_base + (4 * 0x7E0));
    CHECK(res::address(res::gs_uniform_block_0) == res::window_base + (4 * 0xCB0));
}

TEST_CASE("an attribute buffer is a valid buffer resource with its stride", "[hle][gx2][buffers]") {
    std::vector<u32> words;
    gx2::append_attribute_buffer(words, 3, 0x20123400, 0x300, 12);
    const Ran ran{words};

    const u32 base = res::address(res::attribute_buffer_0 + 3);
    CHECK(ran[base] == 0x20123400);
    CHECK(ran[base + 4] == 0x2FF);
    CHECK(ran[base + 8] == 12U << 8); // Linux's SQ_VTXC_STRIDE; nothing else
    CHECK(ran[base + 12] == 0);
    CHECK(ran[base + 24] == 0xC0000000); // Cemu's literal: VALID_BUFFER

    const latte::BufferResource buffer = latte::buffer_resource(*ran.registers, 0x143);
    CHECK(buffer.valid());
    CHECK(buffer.address == 0x20123400);
    CHECK(buffer.size == 0x300);
    CHECK(buffer.word2.stride == 12);
    CHECK(buffer.word2.data_format == 0);

    // Its neighbours are untouched.
    CHECK_FALSE(latte::buffer_resource(*ran.registers, 0x142).valid());
    CHECK_FALSE(latte::buffer_resource(*ran.registers, 0x144).valid());
}

TEST_CASE("a stride wider than the register wraps", "[hle][gx2][buffers]") {
    std::vector<u32> words;
    gx2::append_attribute_buffer(words, 0, 0x20000000, 0x100, 0x900);
    const Ran ran{words};
    CHECK(latte::buffer_resource(*ran.registers, res::attribute_buffer_0).word2.stride == 0x100);
    CHECK((ran[res::address(res::attribute_buffer_0) + 8] & 0xFFF800FFU) == 0);
}

TEST_CASE("a uniform block is a resource and a constant-cache entry", "[hle][gx2][buffers]") {
    struct Expected {
        gx2::UniformStage stage;
        u32 first_resource;
        u32 cache;
        u32 size;
    };

    const std::vector<Expected> stages{
        {gx2::UniformStage::Vertex,
         0x120,
         reg::sq_alu_const_cache_vs_0,
         reg::sq_alu_const_buffer_size_vs_0},
        {gx2::UniformStage::Pixel,
         0x080,
         reg::sq_alu_const_cache_ps_0,
         reg::sq_alu_const_buffer_size_ps_0},
        {gx2::UniformStage::Geometry,
         0x1D0,
         reg::sq_alu_const_cache_gs_0,
         reg::sq_alu_const_buffer_size_gs_0},
    };
    for (const Expected& expected : stages) {
        std::vector<u32> words;
        gx2::append_uniform_block(words, expected.stage, 2, 0x20200000, 0x110);
        const Ran ran{words};

        const u32 base = res::address(expected.first_resource + 2);
        CHECK(ran[base] == 0x20200000);
        CHECK(ran[base + 4] == 0x10F);
        // decaf: a 16-byte stride, FMT_32_32_32_32 (0x22), signed.
        CHECK(ran[base + 8] == ((16U << 8) | (0x22U << 20) | (1U << 28)));
        CHECK(ran[base + 12] == 1); // MEM_REQUEST_SIZE, which Cemu writes too
        CHECK(ran[base + 24] == 0xC0000000);

        CHECK(ran[expected.cache + 8] == 0x20200000U >> 8);
        CHECK(ran[expected.size + 8] == 2); // 0x110 bytes, in 256-byte units, rounded up
        CHECK(ran[expected.cache] == 0);
        CHECK(ran[expected.size + 12] == 0);
    }
}

TEST_CASE("the constant cache's size holds nine bits", "[hle][gx2][buffers]") {
    std::vector<u32> words;
    gx2::append_uniform_block(words, gx2::UniformStage::Pixel, 15, 0x20000000, 0x10000);
    const Ran ran{words};
    CHECK(ran[reg::sq_alu_const_buffer_size_ps_0 + 60] == 0x100);

    words.clear();
    gx2::append_uniform_block(words, gx2::UniformStage::Pixel, 15, 0x20000000, 0x20000);
    const Ran wrapped{words};
    CHECK(wrapped[reg::sq_alu_const_buffer_size_ps_0 + 60] == 0); // 0x200, masked as decaf does
}

// --------------------------------------------------------------------------
// Through GX2
// --------------------------------------------------------------------------

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kBuffer = 0x10010000;
constexpr ppcaddr_t kRBuffer = 0x10004000;

struct BufferFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    BufferFixture() {
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

    [[nodiscard]] latte::BufferResource resource(u32 id) const {
        return latte::buffer_resource(os.gpu.registers(), id);
    }

    [[nodiscard]] u32 reg(u32 address) const { return os.gpu.registers().read(address); }

    [[nodiscard]] u32 physical(ppcaddr_t address) const {
        const memory::Region* const region = memory.region_of(address);
        REQUIRE(region != nullptr);
        physaddr_t out = 0;
        REQUIRE(memory::to_physical(*region, address, out));
        return out;
    }

    [[nodiscard]] u64 packets() { return os.gpu.processor().stats().packets; }
};

} // namespace

TEST_CASE_METHOD(BufferFixture,
                 "GX2SetAttribBuffer binds a buffer by its physical address",
                 "[hle][gx2][buffers]") {
    call("GX2SetAttribBuffer", {5, 0x240, 24, kBuffer});
    call("GX2SetAttribBuffer", {6, 0, 0, 0}); // null: base zero, as on hardware
    run_gpu();

    const latte::BufferResource five = resource(res::attribute_buffer_0 + 5);
    CHECK(five.valid());
    CHECK(five.address == physical(kBuffer));
    CHECK(five.address != kBuffer); // MEM2 is not where it is mapped
    CHECK(five.size == 0x240);
    CHECK(five.word2.stride == 24);

    const latte::BufferResource six = resource(res::attribute_buffer_0 + 6);
    CHECK(six.valid());
    CHECK(six.address == 0);
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(BufferFixture,
                 "past sixteen, or out of the GPU's sight, nothing is written",
                 "[hle][gx2][buffers]") {
    run_gpu();
    const u64 before = packets();
    call("GX2SetAttribBuffer", {16, 0x100, 16, kBuffer});
    call("GX2SetVertexUniformBlock", {16, 0x100, kBuffer});
    call("GX2SetPixelUniformBlock", {0, 0x100, 0x00000100}); // nothing is mapped there
    run_gpu();
    CHECK(packets() == before);
    CHECK_FALSE(resource(res::gs_base).valid()); // where a seventeenth would have gone
}

TEST_CASE_METHOD(BufferFixture,
                 "GX2Set…UniformBlock reach each stage's resources and cache",
                 "[hle][gx2][buffers]") {
    call("GX2SetVertexUniformBlock", {0, 0x40, kBuffer});
    call("GX2SetPixelUniformBlock", {1, 0x100, kBuffer + 0x100});
    call("GX2SetGeometryUniformBlock", {15, 0x101, kBuffer + 0x200});
    run_gpu();

    CHECK(resource(res::vs_uniform_block_0).address == physical(kBuffer));
    CHECK(resource(res::vs_uniform_block_0).size == 0x40);
    CHECK(reg(reg::sq_alu_const_cache_vs_0) == physical(kBuffer) >> 8);
    CHECK(reg(reg::sq_alu_const_buffer_size_vs_0) == 1);

    CHECK(resource(res::ps_uniform_block_0 + 1).address == physical(kBuffer + 0x100));
    CHECK(reg(reg::sq_alu_const_cache_ps_0 + 4) == physical(kBuffer + 0x100) >> 8);
    CHECK(reg(reg::sq_alu_const_buffer_size_ps_0 + 4) == 1);

    CHECK(resource(res::gs_uniform_block_0 + 15).size == 0x101);
    CHECK(reg(reg::sq_alu_const_cache_gs_0 + 60) == physical(kBuffer + 0x200) >> 8);
    CHECK(reg(reg::sq_alu_const_buffer_size_gs_0 + 60) == 2);
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(BufferFixture,
                 "the GX2R forms bind a GX2R buffer from an offset",
                 "[hle][gx2][buffers]") {
    gx2::RBuffer r{};
    r.element_size = 0x20;
    r.element_count = 0x10; // 0x200 bytes
    r.buffer = kBuffer;
    REQUIRE(gx2::write_guest(memory, kRBuffer, r));

    call("GX2RSetAttributeBuffer", {kRBuffer, 2, 0x20, 0x40});
    call("GX2RSetPixelUniformBlock", {kRBuffer, 3, 0x100});
    run_gpu();

    // Cemu's size: what is left past the offset.
    const latte::BufferResource attrib = resource(res::attribute_buffer_0 + 2);
    CHECK(attrib.address == physical(kBuffer + 0x40));
    CHECK(attrib.size == 0x1C0);
    CHECK(attrib.word2.stride == 0x20);

    const latte::BufferResource block = resource(res::ps_uniform_block_0 + 3);
    CHECK(block.address == physical(kBuffer + 0x100));
    CHECK(block.size == 0x100);
    CHECK(reg(reg::sq_alu_const_cache_ps_0 + 12) == physical(kBuffer + 0x100) >> 8);

    // An offset past the end is refused.
    const u64 before = packets();
    call("GX2RSetVertexUniformBlock", {kRBuffer, 0, 0x201});
    run_gpu();
    CHECK(packets() == before);
}
