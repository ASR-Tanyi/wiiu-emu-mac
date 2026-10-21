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
#include <wiiu/gpu/backend.hpp>
#include <wiiu/gpu/presenter.hpp>
#include <wiiu/latte/formats.hpp>
#include <wiiu/latte/msl.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <tuple>
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
