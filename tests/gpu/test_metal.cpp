// Phases 7.2–7.8 — the Metal backend, on the machine that has Metal.
//
// Built only on Apple hosts, and the point of it is that it talks to the real
// GPU. Everything else in the GPU tests runs against the null backend, which
// proves the interface's rules but cannot prove that a device was created,
// that the capability probe reports the truth, or that a clear actually
// cleared.
//
// It needs no window. That is deliberate: a test that opened one could not
// run over ssh or in CI, and the headless path it uses instead is the same
// one the compatibility runner (10.5) will boot titles through.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/gpu/presenter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_fetch_shader.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/formats.hpp>
#include <wiiu/latte/guest_buffers.hpp>
#include <wiiu/latte/msl.hpp>
#include <wiiu/latte/surface_copy.hpp>
#include <wiiu/memory/map.hpp>
#include <wiiu/memory/memory.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <tuple>
#include <utility>
#include <vector>

using namespace wiiu;
using namespace wiiu::gpu;

namespace {

// Every test here runs under Metal's API validation and shader validation
// (7.8), and with capture allowed. They are set before main, so before the
// first device exists, which is when Metal reads them -- and without
// overwriting, so `MTL_DEBUG_LAYER=0 ctest` still turns them off for a run
// that wants to measure something. With argument buffers coming (ADR-003),
// a forgotten `useHeap` is a GPU fault rather than an error unless the debug
// layer is there to say so; that is the bug these are on for.
const bool kMetalTooling = [] {
    setenv("MTL_DEBUG_LAYER", "1", 0);
    setenv("MTL_DEBUG_LAYER_ERROR_MODE", "assert", 0);
    setenv("MTL_SHADER_VALIDATION", "1", 0);
    // Shader validation reports to the system log by default, where a test
    // run never looks. A fault also fails its command buffer, which every
    // test checks through `command_errors()`.
    setenv("MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "1", 0);
    setenv("MTL_CAPTURE_ENABLED", "1", 0);
    return true;
}();

/// A live Metal backend, or a failed test saying why.
[[nodiscard]] std::unique_ptr<GpuBackend> live_backend() {
    static_cast<void>(kMetalTooling);
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    std::unique_ptr<GpuBackend> backend = create_metal_backend();
    REQUIRE(backend != nullptr);

    std::string error;
    if (!backend->initialize(error)) {
        // The capability shortfall is the most useful thing this file can
        // say, so it goes in the failure rather than being swallowed by a
        // bare REQUIRE.
        FAIL("Metal backend did not initialize: " << error);
    }
    return backend;
}

} // namespace

TEST_CASE("the Metal device comes up and reports what it can do", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    const Capabilities& capabilities = backend->capabilities();

    INFO(capabilities.describe());

    CHECK_FALSE(capabilities.device_name.empty());
    CHECK(capabilities.sufficient());

    // ADR-002's floor. If this fails the emulator cannot run here at all,
    // and `initialize` above would already have refused -- so this is really
    // a check that the probe agrees with itself.
    CHECK(capabilities.metal3);
    CHECK(capabilities.mesh_shaders);

    // ADR-003 binds everything through argument buffers, which needs tier 2.
    CHECK(capabilities.argument_buffer_tier == 2);

    // The premise of P22's zero-copy guest memory. True on all Apple
    // silicon; if it is ever false here, that plan needs rethinking.
    CHECK(capabilities.unified_memory);
}

TEST_CASE("resources can be created on the real device", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    BufferDesc buffer_desc;
    buffer_desc.size = 1024;
    buffer_desc.storage = Storage::Shared;
    buffer_desc.label = "test buffer";
    const BufferHandle buffer = backend->create_buffer(buffer_desc);
    REQUIRE(buffer.valid());

    // Shared storage means the CPU can see the bytes directly, which is the
    // whole of P22 in one assertion.
    const std::array<u8, 4> bytes{1, 2, 3, 4};
    REQUIRE(backend->write_buffer(buffer, 0, bytes));
    const auto* const contents = static_cast<const u8*>(backend->buffer_contents(buffer));
    REQUIRE(contents != nullptr);
    CHECK(contents[2] == 3);

    TextureDesc texture_desc;
    texture_desc.width = 64;
    texture_desc.height = 64;
    texture_desc.format = TextureFormat::BGRA8Unorm;
    texture_desc.usage = TextureUsage::Sampled | TextureUsage::RenderTarget;
    texture_desc.label = "test texture";
    CHECK(backend->create_texture(texture_desc).valid());

    SamplerDesc sampler;
    sampler.label = "test sampler";
    CHECK(backend->create_sampler(sampler).valid());

    backend->destroy(buffer);
    backend->shutdown();
}

TEST_CASE("a clear really clears, on the GPU", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    constexpr u32 kSize = 8;
    TextureDesc desc;
    desc.width = kSize;
    desc.height = kSize;
    desc.format = TextureFormat::BGRA8Unorm;
    desc.usage = TextureUsage::RenderTarget;
    desc.storage = Storage::Shared; // so it can be read back
    desc.label = "offscreen";
    const TextureHandle target = backend->create_texture(desc);
    REQUIRE(target.valid());

    RenderPassDesc pass;
    pass.label = "offscreen clear";
    pass.color_count = 1;
    pass.colors[0].texture = target;
    pass.colors[0].load = LoadAction::Clear;
    pass.colors[0].store = StoreAction::Store;
    pass.colors[0].clear = ClearColor{1.0F, 0.0F, 0.0F, 1.0F};

    REQUIRE(backend->begin_render_pass(pass));
    backend->end_render_pass();
    backend->submit();

    std::vector<u8> pixels(static_cast<std::size_t>(kSize) * kSize * 4, 0);
    REQUIRE(backend->read_texture(target, pixels));

    // Opaque red, in BGRA order. This is the assertion the whole Metal path
    // exists for: a device was made, a command buffer was encoded, a render
    // pass ran on the GPU, and the result came back.
    CHECK(pixels[0] == 0x00);
    CHECK(pixels[1] == 0x00);
    CHECK(pixels[2] == 0xFF);
    CHECK(pixels[3] == 0xFF);

    const std::size_t last = pixels.size() - 4;
    CHECK(pixels[last + 2] == 0xFF);
    CHECK(pixels[last + 3] == 0xFF);

    backend->shutdown();
}

TEST_CASE("an unset store action is refused by the real backend too", "[gpu][metal][p20]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    TextureDesc desc;
    desc.width = 8;
    desc.height = 8;
    desc.format = TextureFormat::BGRA8Unorm;
    desc.usage = TextureUsage::RenderTarget;
    desc.storage = Storage::Shared;

    RenderPassDesc pass;
    pass.color_count = 1;
    pass.colors[0].texture = backend->create_texture(desc);
    pass.colors[0].load = LoadAction::Clear;
    // store left Unspecified

    // P20 is enforced by the descriptor, so it holds for every backend rather
    // than being something the Metal one remembers to check.
    CHECK_FALSE(backend->begin_render_pass(pass));

    backend->shutdown();
}

// ---------------------------------------------------------------------------
// 7.8 — the tooling is actually on
// ---------------------------------------------------------------------------

TEST_CASE("the tests run under Metal's API validation", "[gpu][metal][tooling]") {
    if (const char* const setting = std::getenv("MTL_DEBUG_LAYER");
        setting != nullptr && std::string_view{setting} == "0") {
        SKIP("MTL_DEBUG_LAYER=0 was asked for");
    }
    const std::unique_ptr<GpuBackend> backend = live_backend();

    // If this fails, the variables above were read before they were set, and
    // they belong in the test's environment instead (tests/CMakeLists.txt).
    INFO(backend->capabilities().describe());
    CHECK(backend->capabilities().api_validation);
}

// ---------------------------------------------------------------------------
// 7.6 — pipelines and draws
// ---------------------------------------------------------------------------

namespace {

constexpr u32 kTarget = 64;

/// A shared, readable render target, cleared to `clear`.
[[nodiscard]] TextureHandle
make_target(GpuBackend& backend, u32 width = kTarget, u32 height = kTarget) {
    TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = TextureFormat::BGRA8Unorm;
    desc.usage = TextureUsage::RenderTarget;
    desc.storage = Storage::Shared;
    desc.label = "target";
    const TextureHandle target = backend.create_texture(desc);
    REQUIRE(target.valid());
    return target;
}

[[nodiscard]] RenderPassDesc clear_pass(TextureHandle target, ClearColor colour) {
    RenderPassDesc pass;
    pass.label = "test pass";
    pass.color_count = 1;
    pass.colors[0].texture = target;
    pass.colors[0].load = LoadAction::Clear;
    pass.colors[0].store = StoreAction::Store;
    pass.colors[0].clear = colour;
    return pass;
}

/// BGRA bytes at (x, y) of a `width`-wide readback.
[[nodiscard]] std::array<u8, 4>
at(const std::vector<u8>& pixels, u32 x, u32 y, u32 width = kTarget) {
    const std::size_t offset = ((static_cast<std::size_t>(y) * width) + x) * 4;
    return {pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]};
}

constexpr std::array<u8, 4> kBlue{0xFF, 0x00, 0x00, 0xFF}; // BGRA
constexpr std::array<u8, 4> kGreen{0x00, 0xFF, 0x00, 0xFF};
constexpr std::array<u8, 4> kRed{0x00, 0x00, 0xFF, 0xFF};

// Vertices are fetched by the shader from a bound buffer, by vertex id --
// the Phase 7 rule, since Latte's vertex fetch has no alignment Metal's
// vertex descriptors would accept. The colour comes from a fragment buffer,
// so both stages' bindings are exercised.
constexpr std::string_view kTriangleShader = R"msl(
#include <metal_stdlib>
using namespace metal;

struct Out {
    float4 position [[position]];
};

vertex Out vertex_main(uint id [[vertex_id]], const device float2* positions [[buffer(0)]]) {
    Out out;
    out.position = float4(positions[id], 0.0, 1.0);
    return out;
}

fragment float4 fragment_main(Out in [[stage_in]], constant float4& colour [[buffer(0)]]) {
    return colour;
}
)msl";

[[nodiscard]] PipelineHandle triangle_pipeline(GpuBackend& backend) {
    PipelineDesc desc;
    desc.label = "triangle";
    desc.vertex_source = std::string{kTriangleShader};
    desc.color_formats[0] = TextureFormat::BGRA8Unorm;
    std::string error;
    const PipelineHandle pipeline = backend.create_pipeline(desc, error);
    INFO(error);
    REQUIRE(pipeline.valid());
    return pipeline;
}

template <typename T, std::size_t N>
[[nodiscard]] BufferHandle shared_buffer(GpuBackend& backend, const std::array<T, N>& values) {
    BufferDesc desc;
    desc.size = sizeof(T) * N;
    desc.storage = Storage::Shared;
    const BufferHandle buffer = backend.create_buffer(desc);
    REQUIRE(buffer.valid());
    std::array<u8, sizeof(T) * N> bytes{};
    std::memcpy(bytes.data(), values.data(), bytes.size());
    REQUIRE(backend.write_buffer(buffer, 0, bytes));
    return buffer;
}

} // namespace

TEST_CASE("a triangle is rasterised where it should be", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    const TextureHandle target = make_target(*backend);
    const PipelineHandle pipeline = triangle_pipeline(*backend);

    // The lower-left half of the target, in clip space (y up).
    const BufferHandle positions =
        shared_buffer(*backend, std::array<f32, 6>{-1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F});
    const BufferHandle colour = shared_buffer(*backend, std::array<f32, 4>{0.0F, 1.0F, 0.0F, 1.0F});

    REQUIRE(backend->begin_render_pass(clear_pass(target, ClearColor{0.0F, 0.0F, 1.0F, 1.0F})));
    backend->set_pipeline(pipeline);
    backend->bind_buffer(ShaderStage::Vertex, 0, positions, 0);
    backend->bind_buffer(ShaderStage::Fragment, 0, colour, 0);
    DrawCall call;
    call.vertex_count = 3;
    backend->draw(call);
    backend->end_render_pass();
    backend->submit();

    std::vector<u8> pixels(static_cast<std::size_t>(kTarget) * kTarget * 4);
    REQUIRE(backend->read_texture(target, pixels));

    // Texture rows run top to bottom, clip space bottom to top: the
    // triangle's corner is at the *bottom* left of the image. Getting that
    // backwards is the classic upside-down first frame.
    CHECK(at(pixels, 2, kTarget - 3) == kGreen);
    CHECK(at(pixels, kTarget - 3, 2) == kBlue);
    // Well inside, either side of the long edge. Pixels *on* it -- (2,2) and
    // the like, whose centres lie exactly on y = x -- belong to whichever
    // side the fill rule says, which is not what this test is about.
    CHECK(at(pixels, 2, 40) == kGreen);
    CHECK(at(pixels, 40, 61) == kGreen);
    CHECK(at(pixels, 40, 2) == kBlue);
    CHECK(backend->last_frame_stats().draws == 0); // a headless draw is not a frame's
    CHECK(backend->command_errors() == 0);

    backend->shutdown();
}

TEST_CASE("blending and the write mask are the pipeline's", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    // The triangle over a blue clear, with each pipeline in turn; the pixel
    // looked at is well inside it.
    const auto draw = [&](const BlendState& blend,
                          std::array<f32, 4> colour,
                          std::optional<ClearColor> blend_color) {
        PipelineDesc desc;
        desc.label = "blended triangle";
        desc.vertex_source = std::string{kTriangleShader};
        desc.color_formats[0] = TextureFormat::BGRA8Unorm;
        desc.blend[0] = blend;
        std::string error;
        const PipelineHandle pipeline = backend->create_pipeline(desc, error);
        INFO(error);
        REQUIRE(pipeline.valid());

        const TextureHandle target = make_target(*backend);
        const BufferHandle positions =
            shared_buffer(*backend, std::array<f32, 6>{-1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F});
        const BufferHandle fill = shared_buffer(*backend, colour);
        REQUIRE(backend->begin_render_pass(clear_pass(target, ClearColor{0.0F, 0.0F, 1.0F, 1.0F})));
        backend->set_pipeline(pipeline);
        if (blend_color.has_value()) {
            backend->set_blend_color(*blend_color);
        }
        backend->bind_buffer(ShaderStage::Vertex, 0, positions, 0);
        backend->bind_buffer(ShaderStage::Fragment, 0, fill, 0);
        DrawCall call;
        call.vertex_count = 3;
        backend->draw(call);
        backend->end_render_pass();
        backend->submit();

        std::vector<u8> pixels(static_cast<std::size_t>(kTarget) * kTarget * 4);
        REQUIRE(backend->read_texture(target, pixels));
        CHECK(backend->command_errors() == 0);
        return at(pixels, 2, 40);
    };
    const auto near = [](u8 a, u8 b) { return (a > b ? a - b : b - a) <= 1; };

    // Ordinary alpha blending: 40% green over the blue.
    const BlendState alpha_blend{.enabled = true,
                                 .source_rgb = BlendFactor::SourceAlpha,
                                 .destination_rgb = BlendFactor::OneMinusSourceAlpha,
                                 .rgb = BlendOperation::Add,
                                 .source_alpha = BlendFactor::SourceAlpha,
                                 .destination_alpha = BlendFactor::OneMinusSourceAlpha,
                                 .alpha = BlendOperation::Add};
    std::array<u8, 4> pixel = draw(alpha_blend, {0.0F, 1.0F, 0.0F, 0.4F}, std::nullopt);
    INFO("BGRA " << int{pixel[0]} << " " << int{pixel[1]} << " " << int{pixel[2]});
    CHECK(near(pixel[0], 153)); // blue: 0.6 of the clear
    CHECK(near(pixel[1], 102)); // green: 0.4 of the source
    CHECK(pixel[2] == 0);

    // The constant colour, from the pass.
    const BlendState constant{.enabled = true,
                              .source_rgb = BlendFactor::BlendColor,
                              .destination_rgb = BlendFactor::Zero};
    pixel = draw(constant, {1.0F, 1.0F, 1.0F, 1.0F}, ClearColor{0.2F, 0.4F, 0.6F, 1.0F});
    CHECK(near(pixel[0], 153)); // BGRA: 0.6, 0.4, 0.2
    CHECK(near(pixel[1], 102));
    CHECK(near(pixel[2], 51));

    // Writing green alone: red stays the clear's zero, blue its one.
    pixel =
        draw(BlendState{.write_mask = color_write::green}, {1.0F, 1.0F, 0.0F, 1.0F}, std::nullopt);
    CHECK(pixel == std::array<u8, 4>{255, 255, 0, 255});
}

TEST_CASE("a culled face is not drawn", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    const PipelineHandle pipeline = triangle_pipeline(*backend);

    // The same triangle -- counter-clockwise in clip space -- with back faces
    // culled and each winding in front in turn: in exactly one of the two it
    // is a back face. Which one is how Metal judges winding, and what the
    // translator's mapping of Latte's PA_FACE rests on (8.11).
    const auto drawn = [&](Winding front) {
        const TextureHandle target = make_target(*backend);
        const BufferHandle positions =
            shared_buffer(*backend, std::array<f32, 6>{-1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F});
        const BufferHandle colour =
            shared_buffer(*backend, std::array<f32, 4>{0.0F, 1.0F, 0.0F, 1.0F});
        REQUIRE(backend->begin_render_pass(clear_pass(target, ClearColor{0.0F, 0.0F, 1.0F, 1.0F})));
        backend->set_pipeline(pipeline);
        backend->set_cull_mode(CullMode::Back);
        backend->set_front_facing(front);
        backend->bind_buffer(ShaderStage::Vertex, 0, positions, 0);
        backend->bind_buffer(ShaderStage::Fragment, 0, colour, 0);
        DrawCall call;
        call.vertex_count = 3;
        backend->draw(call);
        backend->end_render_pass();
        backend->submit();
        std::vector<u8> pixels(static_cast<std::size_t>(kTarget) * kTarget * 4);
        REQUIRE(backend->read_texture(target, pixels));
        CHECK(backend->command_errors() == 0);
        return at(pixels, 2, 40) == kGreen;
    };
    const bool clockwise_front = drawn(Winding::Clockwise);
    const bool counter_clockwise_front = drawn(Winding::CounterClockwise);
    INFO("drawn with clockwise in front: " << clockwise_front << "; with counter-clockwise: "
                                           << counter_clockwise_front);
    CHECK(clockwise_front != counter_clockwise_front);
    // In window coordinates, y down, the triangle is clockwise; in clip
    // space's orientation, y up, counter-clockwise. Which Metal goes by is
    // what this reports. Cemu hands Latte's PA_FACE to Metal as it is, and
    // its titles draw right, so the translator does the same; this is the
    // fact to hold that against if a title's faces come out inside out.
    WARN("Metal judged the clip-space counter-clockwise triangle "
         << (counter_clockwise_front ? "counter-clockwise" : "clockwise"));

    backend->shutdown();
}

TEST_CASE("an indexed draw reads its indices", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    const TextureHandle target = make_target(*backend);
    const PipelineHandle pipeline = triangle_pipeline(*backend);

    // Four corners and six 16-bit indices: the whole target, as two
    // triangles sharing an edge.
    const BufferHandle positions = shared_buffer(
        *backend, std::array<f32, 8>{-1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F});
    const BufferHandle indices = shared_buffer(*backend, std::array<u16, 6>{0, 1, 2, 2, 1, 3});
    const BufferHandle colour = shared_buffer(*backend, std::array<f32, 4>{1.0F, 0.0F, 0.0F, 1.0F});

    REQUIRE(backend->begin_render_pass(clear_pass(target, ClearColor{0.0F, 0.0F, 1.0F, 1.0F})));
    backend->set_pipeline(pipeline);
    backend->bind_buffer(ShaderStage::Vertex, 0, positions, 0);
    backend->bind_buffer(ShaderStage::Fragment, 0, colour, 0);
    DrawCall call;
    call.index_buffer = indices;
    call.index_count = 6;
    backend->draw(call);
    backend->end_render_pass();
    backend->submit();

    std::vector<u8> pixels(static_cast<std::size_t>(kTarget) * kTarget * 4);
    REQUIRE(backend->read_texture(target, pixels));
    CHECK(at(pixels, 0, 0) == kRed);
    CHECK(at(pixels, kTarget - 1, 0) == kRed);
    CHECK(at(pixels, 0, kTarget - 1) == kRed);
    CHECK(at(pixels, kTarget - 1, kTarget - 1) == kRed);
    CHECK(at(pixels, kTarget / 2, kTarget / 2) == kRed);
    CHECK(backend->command_errors() == 0);

    backend->shutdown();
}

TEST_CASE("a broken shader is refused with the line it is about", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    PipelineDesc desc;
    desc.label = "broken";
    desc.vertex_source = "#include <metal_stdlib>\n"
                         "using namespace metal;\n"
                         "vertex float4 vertex_main() {\n"
                         "    return undeclared_thing;\n"
                         "}\n"
                         "fragment float4 fragment_main() { return float4(1.0); }\n";
    desc.color_formats[0] = TextureFormat::BGRA8Unorm;

    std::string error;
    CHECK_FALSE(backend->create_pipeline(desc, error).valid());
    INFO(error);
    // The compiler's own complaint, and the line it is about quoted under
    // it -- whichever of the two put it there.
    CHECK(error.find("undeclared_thing") != std::string::npos);
    CHECK(error.find("return undeclared_thing;") != std::string::npos);

    backend->shutdown();
}

TEST_CASE("a pipeline naming a missing function says what there is", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    PipelineDesc desc;
    desc.vertex_source = std::string{kTriangleShader};
    desc.vertex_entry = "vertex_mian";
    desc.color_formats[0] = TextureFormat::BGRA8Unorm;

    std::string error;
    CHECK_FALSE(backend->create_pipeline(desc, error).valid());
    INFO(error);
    CHECK(error.find("vertex_mian") != std::string::npos);
    CHECK(error.find("vertex_main") != std::string::npos); // what there was

    backend->shutdown();
}

// ---------------------------------------------------------------------------
// 7.4 — heaps, and host memory in place
// ---------------------------------------------------------------------------

TEST_CASE("resources come out of heaps", "[gpu][metal][p22]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    static_cast<void>(make_target(*backend));
    TextureDesc private_desc;
    private_desc.width = 256;
    private_desc.height = 256;
    private_desc.format = TextureFormat::RGBA8Unorm;
    private_desc.usage = TextureUsage::Sampled | TextureUsage::RenderTarget;
    REQUIRE(backend->create_texture(private_desc).valid());
    static_cast<void>(shared_buffer(*backend, std::array<u32, 4>{1, 2, 3, 4}));

    MemoryStats stats = backend->memory_stats();
    INFO("heaps " << stats.heaps << ", " << stats.heap_used_bytes << " of " << stats.heap_bytes
                  << " bytes used, " << stats.dedicated << " dedicated");
    // One heap per storage mode used: private and shared.
    CHECK(stats.heaps == 2);
    CHECK(stats.heap_used_bytes > 0);
    CHECK(stats.heap_used_bytes <= stats.heap_bytes);
    CHECK(stats.dedicated == 0);

    // Memoryless cannot come from a heap, and is counted rather than hidden.
    TextureDesc depth;
    depth.width = 64;
    depth.height = 64;
    depth.format = TextureFormat::Depth32Float;
    depth.usage = TextureUsage::RenderTarget;
    depth.storage = Storage::Memoryless;
    const TextureHandle memoryless = backend->create_texture(depth);
    REQUIRE(memoryless.valid());
    CHECK(backend->memory_stats().dedicated == 1);
    backend->destroy(memoryless);
    CHECK(backend->memory_stats().dedicated == 0);

    backend->shutdown();
}

TEST_CASE("the GPU reads host memory in place", "[gpu][metal][p22]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    // One page from mmap, the way the guest address space is made (3.1):
    // Metal's no-copy buffers want page-aligned memory the VM system gave out.
    const auto page = static_cast<std::size_t>(host_page_size());
    void* const memory = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    REQUIRE(memory != MAP_FAILED);

    const BufferHandle wrapped = backend->import_host_memory(memory, page, "host page");
    REQUIRE(wrapped.valid());
    CHECK(backend->buffer_contents(wrapped) == memory);
    CHECK(backend->memory_stats().imported_bytes == page);

    // Written *after* wrapping, through the host pointer and nothing else.
    // If the buffer were a copy, the draw below would see zeroes and draw
    // nothing.
    const std::array<f32, 6> triangle{-1.0F, -1.0F, 3.0F, -1.0F, -1.0F, 3.0F};
    std::memcpy(memory, triangle.data(), sizeof(triangle));

    const TextureHandle target = make_target(*backend);
    const PipelineHandle pipeline = triangle_pipeline(*backend);
    const BufferHandle colour = shared_buffer(*backend, std::array<f32, 4>{0.0F, 1.0F, 0.0F, 1.0F});

    REQUIRE(backend->begin_render_pass(clear_pass(target, ClearColor{0.0F, 0.0F, 1.0F, 1.0F})));
    backend->set_pipeline(pipeline);
    backend->bind_buffer(ShaderStage::Vertex, 0, wrapped, 0);
    backend->bind_buffer(ShaderStage::Fragment, 0, colour, 0);
    DrawCall call;
    call.vertex_count = 3;
    backend->draw(call);
    backend->end_render_pass();
    backend->submit();

    std::vector<u8> pixels(static_cast<std::size_t>(kTarget) * kTarget * 4);
    REQUIRE(backend->read_texture(target, pixels));
    CHECK(at(pixels, kTarget / 2, kTarget / 2) == kGreen);
    CHECK(backend->command_errors() == 0);

    backend->destroy(wrapped);
    backend->shutdown();
    munmap(memory, page);
}

// ---------------------------------------------------------------------------
// 7.6 — the presenter, end to end
// ---------------------------------------------------------------------------

TEST_CASE("a screen image reaches the target the right way up", "[gpu][metal][present]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    // 4x2 RGBA, top row red and bottom row blue, with the fourth byte zero --
    // as an OSScreen title that cleared with 0xRRGGBB00 leaves it. Rows are
    // 32 bytes apart, wider than the 16 they hold, as the DRC's are.
    std::vector<u8> image(2 * 32, 0);
    for (u32 x = 0; x < 4; ++x) {
        image[(x * 4) + 0] = 0xFF;      // top row: red
        image[32 + (x * 4) + 2] = 0xFF; // bottom row: blue
    }

    {
        Presenter presenter;
        std::string error;
        REQUIRE(presenter.initialize(*backend, TextureFormat::BGRA8Unorm, error));
        presenter.update(PresentScreen::Tv, ScreenImage{image, 4, 2, 32});

        // 2:1, like the image, so it fills the target exactly.
        const TextureHandle target = make_target(*backend, 64, 32);
        REQUIRE(presenter.draw(target, 64, 32));
        backend->submit();

        std::vector<u8> pixels(64U * 32U * 4U);
        REQUIRE(backend->read_texture(target, pixels));

        // Top rows red, bottom rows blue: the image is upright. And opaque,
        // although every source pixel said zero.
        CHECK(at(pixels, 32, 2, 64) == kRed);
        CHECK(at(pixels, 32, 29, 64) == kBlue);
        CHECK(at(pixels, 1, 2, 64)[3] == 0xFF);
        CHECK(backend->command_errors() == 0);
    }

    backend->shutdown();
}

TEST_CASE("a texture the GPU drew is shown without leaving the GPU", "[gpu][metal][present]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    {
        // A scan-out frame as ADR-006 keeps it: private, rendered to, then
        // sampled by the presenter where it lies.
        TextureDesc desc;
        desc.width = 32;
        desc.height = 16;
        desc.format = TextureFormat::RGBA8Unorm;
        desc.usage = TextureUsage::RenderTarget | TextureUsage::Sampled;
        desc.storage = Storage::Private;
        desc.label = "scan";
        const TextureHandle scan = backend->create_texture(desc);
        REQUIRE(scan.valid());

        // Fifths of 255 survive the trip to bytes and back exactly; the alpha
        // of zero is a scan-out byte the display ignores.
        REQUIRE(backend->begin_render_pass(clear_pass(scan, ClearColor{0.2F, 0.4F, 0.6F, 0.0F})));
        backend->end_render_pass();

        Presenter presenter;
        std::string error;
        REQUIRE(presenter.initialize(*backend, TextureFormat::BGRA8Unorm, error));
        presenter.show(PresentScreen::Tv, ScreenTexture{scan, 32, 16});

        const TextureHandle target = make_target(*backend, 64, 32);
        REQUIRE(presenter.draw(target, 64, 32));
        backend->submit();

        std::vector<u8> pixels(64U * 32U * 4U);
        REQUIRE(backend->read_texture(target, pixels));
        CHECK(at(pixels, 32, 16, 64) == std::array<u8, 4>{153, 102, 51, 0xFF}); // BGRA
        CHECK(at(pixels, 1, 1, 64) == std::array<u8, 4>{153, 102, 51, 0xFF});
        CHECK(backend->command_errors() == 0);

        presenter.forget(scan);
        backend->destroy(scan);
    }

    backend->shutdown();
}

TEST_CASE("a GPU clear stores what GX2's CPU clear does", "[gpu][metal][surface]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    // ADR-006: with a GPU attached GX2ClearColor is a render pass, and the
    // CPU clear (`latte::encode_color`) is the oracle it is held to. Unorm
    // has to match exactly. sRGB is encoded by the hardware, which the
    // graphics APIs let be off by up to 0.6 of a step -- so one either way.
    const std::array<f32, 4> colour{0.5F, 0.25F, 0.75F, 1.0F};
    for (const auto& [format, gx2_format, tolerance] :
         {std::tuple{TextureFormat::RGBA8Unorm, 0x01AU, 0},
          std::tuple{TextureFormat::RGBA8UnormSrgb, 0x41AU, 1}}) {
        TextureDesc desc;
        desc.width = 4;
        desc.height = 4;
        desc.format = format;
        desc.usage = TextureUsage::RenderTarget;
        desc.storage = Storage::Shared; // so it can be read back
        desc.label = "colour buffer";
        const TextureHandle target = backend->create_texture(desc);
        REQUIRE(target.valid());

        RenderPassDesc pass =
            clear_pass(target, ClearColor{colour[0], colour[1], colour[2], colour[3]});
        REQUIRE(backend->begin_render_pass(pass));
        backend->end_render_pass();
        backend->submit();

        std::vector<u8> pixels(4U * 4U * 4U);
        REQUIRE(backend->read_texture(target, pixels));
        const std::optional<latte::EncodedColor> expected = latte::encode_color(gx2_format, colour);
        REQUIRE(expected.has_value());
        for (std::size_t channel = 0; channel < 4; ++channel) {
            INFO(texture_format_name(format)
                 << " channel " << channel << ": the GPU stored " << int{pixels[channel]}
                 << ", the CPU clear " << int{expected->bytes[channel]});
            CHECK(std::abs(int{pixels[channel]} - int{expected->bytes[channel]}) <= tolerance);
        }
        CHECK(pixels[60] == pixels[0]); // the last pixel too
        CHECK(backend->command_errors() == 0);
        backend->destroy(target);
    }

    backend->shutdown();
}

namespace {

/// A `Shared` RGBA8 texture holding `bytes`, rows packed, for a copy to read.
[[nodiscard]] TextureHandle source_texture(GpuBackend& backend,
                                           TextureFormat format,
                                           u32 width,
                                           u32 height,
                                           const std::vector<u8>& bytes) {
    TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = format;
    desc.usage = TextureUsage::Sampled;
    desc.storage = Storage::Shared;
    desc.label = "copy source";
    const TextureHandle texture = backend.create_texture(desc);
    REQUIRE(texture.valid());
    REQUIRE(backend.write_texture(texture, Rect{0, 0, width, height}, bytes, width * 4));
    return texture;
}

/// A `Shared` RGBA8 UNORM render target, for a copy to write and a test to
/// read back.
[[nodiscard]] TextureHandle copy_target(GpuBackend& backend, u32 width, u32 height) {
    TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = latte::SurfaceCopier::target_format;
    desc.usage = TextureUsage::RenderTarget;
    desc.storage = Storage::Shared;
    desc.label = "copy target";
    const TextureHandle texture = backend.create_texture(desc);
    REQUIRE(texture.valid());
    return texture;
}

} // namespace

TEST_CASE("a surface copy keeps every byte, sRGB or not", "[gpu][metal][surface][copy]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    {
        latte::SurfaceCopier copier;
        std::string error;
        REQUIRE(copier.initialize(*backend, error));

        // All 256 values in every channel, each channel in a different order.
        std::vector<u8> bytes(256U * 4U);
        for (u32 i = 0; i < 256; ++i) {
            bytes[(i * 4) + 0] = static_cast<u8>(i);
            bytes[(i * 4) + 1] = static_cast<u8>(255 - i);
            bytes[(i * 4) + 2] = static_cast<u8>(i * 7);
            bytes[(i * 4) + 3] = static_cast<u8>(i ^ 0xA5);
        }

        for (const TextureFormat format :
             {TextureFormat::RGBA8Unorm, TextureFormat::RGBA8UnormSrgb}) {
            const TextureHandle source = source_texture(*backend, format, 256, 1, bytes);
            const TextureHandle target = copy_target(*backend, 256, 1);
            REQUIRE(copier.copy(latte::CopySource{source, format, 256, 1},
                                latte::CopyTarget{target, 256, 1},
                                error));
            backend->submit();

            std::vector<u8> out(bytes.size());
            REQUIRE(backend->read_texture(target, out));
            std::size_t wrong = 0;
            std::size_t first = out.size();
            for (std::size_t i = 0; i < out.size(); ++i) {
                if (out[i] != bytes[i]) {
                    ++wrong;
                    first = std::min(first, i);
                }
            }
            INFO(texture_format_name(format)
                 << ": " << wrong << " bytes changed"
                 << (wrong == 0 ? std::string{}
                                : ", the first at " + std::to_string(first) + ": " +
                                      std::to_string(bytes[first]) + " became " +
                                      std::to_string(out[first])));
            CHECK(wrong == 0);
            CHECK(backend->command_errors() == 0);
            backend->destroy(source);
            backend->destroy(target);
        }
    }
    backend->shutdown();
}

TEST_CASE("a surface copy scales by nearest pixel, as the CPU copy does",
          "[gpu][metal][surface][copy]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    {
        latte::SurfaceCopier copier;
        std::string error;
        REQUIRE(copier.initialize(*backend, error));

        // A 6x3 picture in an 8x4 texture, as a colour buffer is padded; the
        // padding holds what no copy should read.
        constexpr u32 kWidth = 8;
        constexpr u32 kHeight = 4;
        std::vector<u8> bytes(std::size_t{kWidth} * kHeight * 4, 0xEE);
        for (u32 y = 0; y < 3; ++y) {
            for (u32 x = 0; x < 6; ++x) {
                const std::size_t at = ((std::size_t{y} * kWidth) + x) * 4;
                bytes[at + 0] = static_cast<u8>(x * 40);
                bytes[at + 1] = static_cast<u8>(y * 80);
                bytes[at + 2] = static_cast<u8>(x + (y * 6));
                bytes[at + 3] = 0xFF;
            }
        }
        const TextureHandle source =
            source_texture(*backend, TextureFormat::RGBA8Unorm, kWidth, kHeight, bytes);

        // Up, and down: 10x5, then 4x2.
        for (const auto& [width, height] : {std::pair{10U, 5U}, std::pair{4U, 2U}}) {
            const TextureHandle target = copy_target(*backend, width, height);
            REQUIRE(copier.copy(latte::CopySource{source, TextureFormat::RGBA8Unorm, 6, 3},
                                latte::CopyTarget{target, width, height},
                                error));
            backend->submit();

            std::vector<u8> out(std::size_t{width} * height * 4);
            REQUIRE(backend->read_texture(target, out));
            std::size_t wrong = 0;
            for (u32 y = 0; y < height; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    // gx2::copy_to_scan_frame's arithmetic.
                    const u32 sx = (x * 6) / width;
                    const u32 sy = (y * 3) / height;
                    const std::size_t from = ((std::size_t{sy} * kWidth) + sx) * 4;
                    const std::size_t to = ((std::size_t{y} * width) + x) * 4;
                    wrong += std::memcmp(&out[to], &bytes[from], 4) != 0 ? 1U : 0U;
                }
            }
            INFO(width << "x" << height);
            CHECK(wrong == 0);
            CHECK(backend->command_errors() == 0);
            backend->destroy(target);
        }
        backend->destroy(source);
    }
    backend->shutdown();
}

TEST_CASE("guest memory is wrapped for the GPU in place", "[gpu][metal][p22]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    {
        // Metal takes no-copy buffers only over whole pages of memory the
        // process mapped itself; guest RAM is that, a window at a time.
        memory::Memory memory;
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        latte::GuestBuffers buffers(*backend, memory);

        for (const physaddr_t address : {physaddr_t{0x20001000}, physaddr_t{0x00000100}}) {
            const std::optional<latte::GuestBuffers::Range> range =
                buffers.find(address, 64, error);
            INFO(error);
            REQUIRE(range.has_value());
            ppcaddr_t guest = 0;
            REQUIRE(memory::to_virtual(memory.regions(), address, guest));
            // The guest's bytes themselves.
            CHECK(static_cast<u8*>(backend->buffer_contents(range->buffer)) + range->offset ==
                  memory.translate_unchecked(guest));
        }
        CHECK(backend->memory_stats().imported == 2);
        buffers.clear();
        CHECK(backend->memory_stats().imported == 0);
    }
    backend->shutdown();
}

TEST_CASE("GX2's clear and scan-out copy run on the GPU, and the flip shows them",
          "[gpu][metal][gx2]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    {
        // WHB's frame, through GX2's own calls, with Metal attached
        // (ADR-006): clear the TV's colour buffer, copy it to the TV, swap.
        constexpr ppcaddr_t kBuffer = 0x10005000;
        constexpr ppcaddr_t kImage = 0x10100000;
        constexpr ppcaddr_t kTvScan = memory::mem1_base + 0x100000;
        constexpr u32 kFrameBytes = 1280 * 720 * 4;

        memory::Memory memory;
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        cpu::CoreState state;
        hle::Registry registry;
        hle::coreinit::Context os;
        hle::coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, 0x10001000));
        std::vector<hle::coreinit::ScreenFlip> shown;
        os.screens.on_flip = [&shown](const hle::coreinit::ScreenFlip& flip,
                                      const memory::Memory&) {
            shown.push_back(flip);
            return true;
        };
        os.gpu.attach_backend(backend.get());

        const auto call =
            [&](std::string_view name, std::vector<u32> arguments, std::vector<f32> floats = {}) {
                hle::Function* const function = registry.find("gx2", name);
                REQUIRE(function != nullptr);
                for (std::size_t i = 0; i < arguments.size(); ++i) {
                    state.gpr[3 + i] = arguments[i];
                }
                for (std::size_t i = 0; i < floats.size(); ++i) {
                    state.fpr[1 + i].ps0 = static_cast<f64>(floats[i]);
                }
                hle::Call context{.state = state, .memory = memory};
                function->handler(context);
            };

        call("GX2Init", {0});
        call("GX2SetTVBuffer", {kTvScan, 2 * kFrameBytes, 3, 0x01A, 2}); // 720p, RGBA8, double
        call("GX2SetTVEnable", {1});

        hle::gx2::ColorBuffer buffer{};
        buffer.surface.dim = 1;
        buffer.surface.width = 1280;
        buffer.surface.height = 720;
        buffer.surface.depth = 1;
        buffer.surface.mip_levels = 1;
        buffer.surface.format = 0x01A;
        buffer.surface.use = hle::gx2::surface_use_texture | hle::gx2::surface_use_color_buffer;
        buffer.view_num_slices = 1;
        REQUIRE(hle::gx2::calc_surface_size_and_alignment(buffer.surface));
        buffer.surface.image = kImage;
        const auto registers = hle::gx2::color_buffer_registers(buffer);
        REQUIRE(registers.has_value());
        buffer.cb_color_size = registers->size;
        buffer.cb_color_info = registers->info;
        buffer.cb_color_view = registers->view;
        REQUIRE(hle::gx2::write_guest(memory, kBuffer, buffer));

        // 0.4 is 102 exactly, in a byte.
        call("GX2ClearColor", {kBuffer}, {0.0F, 0.4F, 1.0F, 1.0F});
        call("GX2CopyColorBufferToScanBuffer", {kBuffer, 1});
        call("GX2SwapScanBuffers", {});
        call("GX2Flush", {});
        state.timebase = hle::gx2::vsync_tick(1);
        static_cast<void>(os.scheduler.reschedule(state));

        REQUIRE(shown.size() == 1);
        REQUIRE(shown[0].texture.valid());
        backend->submit();

        std::vector<u8> pixels(kFrameBytes);
        REQUIRE(backend->read_texture(shown[0].texture, pixels));
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            const std::array<u8, 4> pixel{pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]};
            wrong += pixel != std::array<u8, 4>{0, 102, 255, 255} ? 1U : 0U;
        }
        INFO("the first pixel: " << int{pixels[0]} << ", " << int{pixels[1]} << ", "
                                 << int{pixels[2]} << ", " << int{pixels[3]});
        CHECK(wrong == 0);
        // Guest memory never saw it.
        CHECK(memory.read<u32>(kTvScan) == 0);
        CHECK(backend->command_errors() == 0);

        os.gpu.attach_backend(nullptr);
    }
    backend->shutdown();
}

TEST_CASE("decaf's example triangle, drawn through GX2's own calls", "[gpu][metal][gx2][draw]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    {
        // The whole road: GX2's calls write PM4, the packets retire on the
        // guest's clock, the draw's shaders are found and translated, its
        // vertices read in place from guest memory, and it is drawn on Metal
        // into the surface a clear made; the scan-out copy takes that to the
        // TV's frame, and the flip hands the frame over.
        constexpr ppcaddr_t kBuffer = 0x10005000;
        constexpr ppcaddr_t kImage = 0x10100000;
        constexpr ppcaddr_t kTvScan = memory::mem1_base + 0x100000;
        constexpr u32 kFrameBytes = 1280 * 720 * 4;
        constexpr ppcaddr_t kFetchStruct = 0x10020000;
        constexpr ppcaddr_t kFetchProgram = 0x10020100;
        constexpr ppcaddr_t kAttribs = 0x10020400;
        constexpr ppcaddr_t kVertexStruct = 0x10021000;
        constexpr ppcaddr_t kVertexProgram = 0x10021400;
        constexpr ppcaddr_t kPixelStruct = 0x10022000;
        constexpr ppcaddr_t kPixelProgram = 0x10022400;
        constexpr ppcaddr_t kColours = 0x10040000;
        constexpr ppcaddr_t kPositions = 0x10040100;

        memory::Memory memory;
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        cpu::CoreState state;
        hle::Registry registry;
        hle::coreinit::Context os;
        hle::coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, 0x10001000));
        std::vector<hle::coreinit::ScreenFlip> shown;
        os.screens.on_flip = [&shown](const hle::coreinit::ScreenFlip& flip,
                                      const memory::Memory&) {
            shown.push_back(flip);
            return true;
        };
        os.gpu.attach_backend(backend.get());

        const auto call =
            [&](std::string_view name, std::vector<u32> arguments, std::vector<f32> floats = {}) {
                hle::Function* const function = registry.find("gx2", name);
                REQUIRE(function != nullptr);
                for (std::size_t i = 0; i < arguments.size(); ++i) {
                    state.gpr[3 + i] = arguments[i];
                }
                for (std::size_t i = 0; i < floats.size(); ++i) {
                    state.fpr[1 + i].ps0 = static_cast<f64>(floats[i]);
                }
                hle::Call context{.state = state, .memory = memory};
                function->handler(context);
            };
        // Programs as the GPU reads them, little-endian; vertex data as the
        // title writes it, big-endian.
        const auto put_program = [&](ppcaddr_t address, std::initializer_list<u32> words) {
            for (const u32 word : words) {
                const std::array<u8, 4> bytes{static_cast<u8>(word),
                                              static_cast<u8>(word >> 8),
                                              static_cast<u8>(word >> 16),
                                              static_cast<u8>(word >> 24)};
                REQUIRE(memory.write_bytes(address, bytes));
                address += 4;
            }
        };
        const auto put_floats = [&](ppcaddr_t address, std::initializer_list<f32> values) {
            for (const f32 value : values) {
                memory.write<u32>(address, std::bit_cast<u32>(value));
                address += 4;
            }
        };

        call("GX2Init", {0});
        call("GX2SetTVBuffer", {kTvScan, 2 * kFrameBytes, 3, 0x01A, 2}); // 720p, RGBA8, double
        call("GX2SetTVEnable", {1});

        // WHB's TV colour buffer, bound as render target 0, with the
        // viewport and scissor over it.
        hle::gx2::ColorBuffer buffer{};
        buffer.surface.dim = 1;
        buffer.surface.width = 1280;
        buffer.surface.height = 720;
        buffer.surface.depth = 1;
        buffer.surface.mip_levels = 1;
        buffer.surface.format = 0x01A;
        buffer.surface.use = hle::gx2::surface_use_texture | hle::gx2::surface_use_color_buffer;
        buffer.view_num_slices = 1;
        REQUIRE(hle::gx2::calc_surface_size_and_alignment(buffer.surface));
        buffer.surface.image = kImage;
        const auto registers = hle::gx2::color_buffer_registers(buffer);
        REQUIRE(registers.has_value());
        buffer.cb_color_size = registers->size;
        buffer.cb_color_info = registers->info;
        buffer.cb_color_view = registers->view;
        REQUIRE(hle::gx2::write_guest(memory, kBuffer, buffer));
        call("GX2SetColorBuffer", {kBuffer, 0});
        call("GX2SetViewport", {}, {0.0F, 0.0F, 1280.0F, 720.0F, 0.0F, 1.0F});
        call("GX2SetScissor", {0, 0, 1280, 720});

        // decaf's example shaders (latte-assembler's resources), the fetch
        // shader from GX2's own generator: colour from buffer 0, position
        // from buffer 1, four big-endian floats each.
        hle::gx2::AttribStream colour{};
        colour.location = 0;
        colour.buffer = 0;
        colour.format = 0x813; // GX2_ATTRIB_FORMAT_FLOAT_32_32_32_32
        colour.mask = 0x00010203;
        colour.endian_swap = 3;
        hle::gx2::AttribStream position = colour;
        position.location = 1;
        position.buffer = 1;
        REQUIRE(hle::gx2::write_guest(memory, kAttribs, colour));
        REQUIRE(hle::gx2::write_guest(memory, kAttribs + sizeof(hle::gx2::AttribStream), position));
        call("GX2InitFetchShaderEx", {kFetchStruct, kFetchProgram, 2, kAttribs, 0, 0});
        call("GX2SetFetchShader", {kFetchStruct});

        constexpr u32 xyzw = 0x688;
        put_program(kVertexProgram,
                    {0,
                     0x13U << 23,
                     60U | (1U << 13) | (2U << 15),
                     xyzw | (0x28U << 23) | (1U << 31),
                     (2U << 13) | (1U << 15),
                     xyzw | (0x28U << 23) | (1U << 21)});
        hle::gx2::VertexShader vs{};
        vs.sq_pgm_resources_vs = 3U | (1U << 8);
        vs.num_spi_vs_out_id = 1;
        vs.spi_vs_out_id[0] = 0xFFFFFF00;
        vs.sq_vtx_semantic_clear = 0xFFFFFFFC;
        vs.num_sq_vtx_semantic = 2;
        vs.sq_vtx_semantic[0] = 0;
        vs.sq_vtx_semantic[1] = 1;
        vs.size = 24;
        vs.data = kVertexProgram;
        REQUIRE(hle::gx2::write_guest(memory, kVertexStruct, vs));
        call("GX2SetVertexShader", {kVertexStruct});

        put_program(kPixelProgram, {0, xyzw | (0x28U << 23) | (1U << 31) | (1U << 21)});
        hle::gx2::PixelShader ps{};
        ps.sq_pgm_resources_ps = 1;
        ps.spi_ps_in_control_0 = 1;
        ps.num_spi_ps_input_cntl = 1;
        ps.spi_ps_input_cntls[0] = 0 | (1U << 8);
        ps.cb_shader_mask = 0xF;
        ps.size = 8;
        ps.data = kPixelProgram;
        REQUIRE(hle::gx2::write_guest(memory, kPixelStruct, ps));
        call("GX2SetPixelShader", {kPixelStruct});

        // Green, over the lower-left half.
        put_floats(kColours, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
        put_floats(kPositions, {-1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 1});
        call("GX2SetAttribBuffer", {0, 48, 16, kColours});
        call("GX2SetAttribBuffer", {1, 48, 16, kPositions});

        // The frame: clear to blue, draw, copy to the TV, swap. The draw
        // either way GX2 has: auto-indexed, or from indices in the title's
        // byte order, which are converted for Metal (latte/index_cache.hpp).
        call("GX2ClearColor", {kBuffer}, {0.0F, 0.0F, 1.0F, 1.0F});
        SECTION("auto-indexed") {
            call("GX2DrawEx", {4, 3, 0, 1});
        }
        SECTION("from GX2's big-endian 16-bit indices") {
            constexpr ppcaddr_t kIndices = 0x10040200;
            memory.write<u16>(kIndices, 0);
            memory.write<u16>(kIndices + 2, 1);
            memory.write<u16>(kIndices + 4, 2);
            call("GX2DrawIndexedEx", {4, 3, 4, kIndices, 0, 1});
        }
        call("GX2CopyColorBufferToScanBuffer", {kBuffer, 1});
        call("GX2SwapScanBuffers", {});
        call("GX2Flush", {});
        state.timebase = hle::gx2::vsync_tick(1);
        static_cast<void>(os.scheduler.reschedule(state));

        for (const std::string& problem : os.gpu.problems()) {
            UNSCOPED_INFO("GX2: " << problem);
        }
        CHECK(os.gpu.draw_stats().drawn == 1);
        REQUIRE(shown.size() == 1);
        REQUIRE(shown[0].texture.valid());
        backend->submit();

        std::vector<u8> pixels(kFrameBytes);
        REQUIRE(backend->read_texture(shown[0].texture, pixels));
        const auto rgba = [&pixels](u32 x, u32 y) {
            const std::size_t at = ((std::size_t{y} * 1280) + x) * 4;
            return std::array<u8, 4>{pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]};
        };
        constexpr std::array<u8, 4> green{0, 255, 0, 255};
        constexpr std::array<u8, 4> blue{0, 0, 255, 255};
        // Clip space's -1 is the bottom of the viewport, so the lower-left
        // half is green; the rest is the clear. (Points well away from the
        // diagonal, where x + y in clip space is zero.)
        CHECK(rgba(10, 709) == green);
        CHECK(rgba(10, 360) == green);
        CHECK(rgba(600, 600) == green);
        CHECK(rgba(1269, 10) == blue);
        CHECK(rgba(900, 100) == blue);
        CHECK(rgba(700, 300) == blue);
        CHECK(backend->command_errors() == 0);

        os.gpu.attach_backend(nullptr);
    }
    backend->shutdown();
}

// ---------------------------------------------------------------------------
// 8.6 — the decompiler's MSL, drawing
// ---------------------------------------------------------------------------

namespace {

/// Floats as a title's memory holds them: big-endian.
template <std::size_t N>
[[nodiscard]] std::array<u8, N * 4> big_endian(const std::array<f32, N>& values) {
    std::array<u8, N * 4> bytes{};
    for (std::size_t i = 0; i < N; ++i) {
        u32 bits = 0;
        std::memcpy(&bits, &values[i], 4);
        bytes[(i * 4) + 0] = static_cast<u8>(bits >> 24);
        bytes[(i * 4) + 1] = static_cast<u8>(bits >> 16);
        bytes[(i * 4) + 2] = static_cast<u8>(bits >> 8);
        bytes[(i * 4) + 3] = static_cast<u8>(bits);
    }
    return bytes;
}

} // namespace

TEST_CASE("decaf's example triangle draws through the emitted MSL", "[gpu][metal][msl]") {
    using namespace wiiu::latte;
    const std::unique_ptr<GpuBackend> backend = live_backend();
    const TextureHandle target = make_target(*backend);

    // The example's shaders as the intermediate form has them (latte's
    // tests take them there from the microcode): colour fetched into R1
    // from attribute buffer 0, position into R2 from buffer 1, both four
    // big-endian floats; R2 exported as the position, R1 as parameter 0,
    // which the pixel shader reads into R0 and writes to target 0.
    constexpr ir::Swizzle xyzw{Sel::X, Sel::Y, Sel::Z, Sel::W};
    const auto attribute = [&](u32 semantic, u32 gpr, u32 buffer) {
        ir::VertexAttribute a;
        a.semantic = semantic;
        a.gpr = gpr;
        a.buffer = buffer;
        a.bytes = 16;
        a.data_format = 0x23; // FMT_32_32_32_32_FLOAT
        a.num_format = NumFormat::Scaled;
        a.endian = Endian::Swap8In32;
        a.components = xyzw;
        return a;
    };
    ir::Shader vs;
    vs.stage = ir::Stage::Vertex;
    vs.attributes = {attribute(0, 1, 0), attribute(1, 2, 1)};
    vs.body = {ir::FetchAttributes{},
               ir::ExportPosition{.gpr = 2, .swizzle = xyzw},
               ir::ExportParameter{.index = 0, .semantic = 0, .gpr = 1, .swizzle = xyzw}};
    ir::Shader ps;
    ps.stage = ir::Stage::Pixel;
    ps.inputs = {ir::PixelInput{.gpr = 0, .source = ir::InputSource::Parameter, .semantic = 0}};
    ps.body = {ir::ExportColor{.index = 0, .target = 0, .gpr = 0, .swizzle = xyzw}};

    PipelineDesc desc;
    desc.label = "emitted triangle";
    std::string error;
    REQUIRE(msl::emit_vertex(vs, &ps, desc.vertex_source, error));
    REQUIRE(msl::emit_pixel(ps, desc.fragment_source, error));
    desc.color_formats[0] = TextureFormat::BGRA8Unorm;
    const PipelineHandle pipeline = backend->create_pipeline(desc, error);
    INFO(error);
    INFO(desc.vertex_source);
    INFO(desc.fragment_source);
    REQUIRE(pipeline.valid());

    // The lower-left half of the target, green at every corner.
    const BufferHandle colours =
        shared_buffer(*backend,
                      big_endian(std::array<f32, 12>{
                          0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F}));
    const BufferHandle positions = shared_buffer(
        *backend,
        big_endian(std::array<f32, 12>{
            -1.0F, -1.0F, 0.0F, 1.0F, 1.0F, -1.0F, 0.0F, 1.0F, -1.0F, 1.0F, 0.0F, 1.0F}));
    msl::FetchConstants constants;
    constants.stride[0] = 16;
    constants.stride[1] = 16;
    constants.size[0] = 48;
    constants.size[1] = 48;
    const BufferHandle fetch =
        shared_buffer(*backend, std::array<msl::FetchConstants, 1>{constants});

    REQUIRE(backend->begin_render_pass(clear_pass(target, ClearColor{0.0F, 0.0F, 1.0F, 1.0F})));
    backend->set_pipeline(pipeline);
    backend->bind_buffer(ShaderStage::Vertex, msl::fetch_constants_slot, fetch, 0);
    backend->bind_buffer(ShaderStage::Vertex, msl::first_attribute_buffer_slot + 0, colours, 0);
    backend->bind_buffer(ShaderStage::Vertex, msl::first_attribute_buffer_slot + 1, positions, 0);
    DrawCall call;
    call.vertex_count = 3;
    backend->draw(call);
    backend->end_render_pass();
    backend->submit();

    std::vector<u8> pixels(static_cast<std::size_t>(kTarget) * kTarget * 4);
    REQUIRE(backend->read_texture(target, pixels));
    // As the hand-written triangle above: the lower-left half green, the
    // rest the clear colour.
    CHECK(at(pixels, 2, kTarget - 3) == kGreen);
    CHECK(at(pixels, 2, 40) == kGreen);
    CHECK(at(pixels, 40, 61) == kGreen);
    CHECK(at(pixels, kTarget - 3, 2) == kBlue);
    CHECK(at(pixels, 40, 2) == kBlue);
    CHECK(backend->command_errors() == 0);

    backend->shutdown();
}

// ---------------------------------------------------------------------------
// 7.8 — capture
// ---------------------------------------------------------------------------
//
// Hidden from the normal run, which has shader validation on: Metal refuses
// to capture such a process ("Capturing Metal Shader Validation is not
// supported"). tests/CMakeLists.txt runs it on its own as `gpu.capture`, with
// shader validation off and API validation still on.

TEST_CASE("a GPU capture is written where it was asked for", "[gpu][metal][tooling][.capture]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "wiiu-emu-test-capture.gputrace";

    std::string error;
    if (!backend->begin_capture(path.string(), error)) {
        // Only acceptable for the one reason a process cannot fix by itself:
        // capture was not enabled before Metal started. Anything else is a
        // capture that should have worked and did not.
        INFO(error);
        REQUIRE(error.find("MTL_CAPTURE_ENABLED") != std::string::npos);
        SKIP("GPU capture is not available to this process: " << error);
    }

    const TextureHandle target = make_target(*backend);
    REQUIRE(backend->begin_render_pass(clear_pass(target, ClearColor{1.0F, 0.0F, 0.0F, 1.0F})));
    backend->end_render_pass();
    backend->submit();
    backend->end_capture();

    // A .gputrace is a directory, and the thing Xcode opens.
    CHECK(std::filesystem::exists(path));
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);

    backend->shutdown();
}
