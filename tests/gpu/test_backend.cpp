// Phase 7.1 — the backend interface, and the null backend behind it.
//
// Two things are being guarded. The first is the interface's own rules: an
// unset load or store action is refused (**P20**), a primitive Metal cannot
// draw never reaches a pipeline, a capability shortfall produces a sentence
// rather than a boolean. The second is that the null backend is strict enough
// to be worth testing against -- a recording backend that accepted everything
// would make the Phase 8 tests pass for the wrong reason.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/gpu/presenter.hpp>
#include <wiiu/gpu/shader_diagnostics.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::gpu;

namespace {

struct BackendFixture {
    NullBackend backend;

    BackendFixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        std::string error;
        REQUIRE(backend.initialize(error));
        REQUIRE(backend.attach_surface(NativeSurface{&marker, 1280, 720}, error));
    }

    /// A colour attachment pointing at a freshly made texture, with both
    /// actions set, which is the shortest legal pass.
    [[nodiscard]] RenderPassDesc simple_pass() {
        TextureDesc texture;
        texture.width = 1280;
        texture.height = 720;
        texture.format = TextureFormat::BGRA8Unorm;
        texture.usage = TextureUsage::RenderTarget;
        texture.label = "target";

        RenderPassDesc pass;
        pass.label = "test";
        pass.color_count = 1;
        pass.colors[0].texture = backend.create_texture(texture);
        pass.colors[0].load = LoadAction::Clear;
        pass.colors[0].store = StoreAction::Store;
        return pass;
    }

    /// The smallest pipeline the null backend accepts: a source that names
    /// both entry points.
    [[nodiscard]] PipelineHandle simple_pipeline() {
        PipelineDesc desc;
        desc.label = "simple";
        desc.vertex_source = "vertex vertex_main ... fragment fragment_main";
        desc.color_formats[0] = TextureFormat::BGRA8Unorm;
        std::string error;
        const PipelineHandle pipeline = backend.create_pipeline(desc, error);
        REQUIRE(pipeline.valid());
        return pipeline;
    }

    int marker = 0; ///< stands in for a CAMetalLayer the null backend ignores
};

} // namespace

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

TEST_CASE("a default handle names nothing", "[gpu]") {
    CHECK_FALSE(BufferHandle{}.valid());
    CHECK_FALSE(static_cast<bool>(TextureHandle{}));
    CHECK(BufferHandle{} == BufferHandle{});

    // Zero is reserved so that a zeroed descriptor does not accidentally
    // refer to the first resource anything created.
    CHECK_FALSE(BufferHandle{0}.valid());
    CHECK(BufferHandle{1}.valid());
    CHECK_FALSE(BufferHandle{1} == BufferHandle{2});
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

TEST_CASE("a machine that cannot run this says which part is missing", "[gpu]") {
    Capabilities capabilities;
    capabilities.device_name = "Some GPU";

    const std::string shortfall = capabilities.shortfall();
    CHECK_FALSE(capabilities.sufficient());
    CHECK(shortfall.find("Some GPU") != std::string::npos);
    CHECK(shortfall.find("Metal 3") != std::string::npos);
    CHECK(shortfall.find("Apple7") != std::string::npos);
    CHECK(shortfall.find("tier 2") != std::string::npos);

    // ADR-002's case exactly: Metal 3 is reported, mesh shaders are not. A
    // check for Metal 3 alone would let this machine through and then render
    // nothing where the geometry shaders were.
    Capabilities intel;
    intel.device_name = "Intel Iris";
    intel.metal3 = true;
    intel.argument_buffer_tier = 2;
    CHECK_FALSE(intel.sufficient());
    CHECK(intel.shortfall().find("mesh shaders") != std::string::npos);

    Capabilities apple = intel;
    apple.mesh_shaders = true;
    CHECK(apple.sufficient());
    CHECK(apple.shortfall().empty());
}

TEST_CASE("the capability report names what a bug report needs", "[gpu]") {
    Capabilities capabilities;
    capabilities.device_name = "Apple M3 Max";
    capabilities.api_version = "Apple9";
    capabilities.argument_buffer_tier = 2;

    const std::string text = capabilities.describe();
    CHECK(text.find("Apple M3 Max") != std::string::npos);
    CHECK(text.find("Apple9") != std::string::npos);
    CHECK(text.find("tier 2") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Render pass validation — P20
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(BackendFixture, "an unset load or store action is refused", "[gpu][p20]") {
    RenderPassDesc pass = simple_pass();

    SECTION("load") {
        pass.colors[0].load = LoadAction::Unspecified;
        const std::string problem = pass.validate();
        REQUIRE_FALSE(problem.empty());
        // The message has to say why, not just that: the wrong action is
        // still *correct*, so whoever reads this has no symptom to go on.
        CHECK(problem.find("load action") != std::string::npos);
        CHECK(problem.find("P20") != std::string::npos);
    }

    SECTION("store") {
        pass.colors[0].store = StoreAction::Unspecified;
        CHECK(pass.validate().find("store action") != std::string::npos);
    }

    SECTION("and the backend refuses it too, not just the descriptor") {
        pass.colors[0].load = LoadAction::Unspecified;
        CHECK_FALSE(backend.begin_render_pass(pass));
        CHECK(backend.rejected() == 1);
        CHECK_FALSE(backend.in_render_pass());
    }
}

TEST_CASE_METHOD(BackendFixture, "a render pass has to attach something", "[gpu]") {
    RenderPassDesc empty;
    CHECK(empty.validate().find("no attachments") != std::string::npos);

    RenderPassDesc pass = simple_pass();
    pass.colors[0].texture = TextureHandle{};
    CHECK(pass.validate().find("no texture") != std::string::npos);

    RenderPassDesc too_many = simple_pass();
    too_many.color_count = max_color_attachments + 1;
    CHECK_FALSE(too_many.validate().empty());
}

TEST_CASE_METHOD(BackendFixture, "resolving without somewhere to resolve to", "[gpu]") {
    RenderPassDesc pass = simple_pass();
    pass.colors[0].store = StoreAction::Resolve;
    CHECK(pass.validate().find("resolve texture") != std::string::npos);

    pass.colors[0].resolve_texture = pass.colors[0].texture;
    CHECK(pass.validate().empty());
}

TEST_CASE_METHOD(BackendFixture, "a depth attachment needs actions as well", "[gpu][p20]") {
    TextureDesc depth;
    depth.width = 1280;
    depth.height = 720;
    depth.format = TextureFormat::Depth32Float;
    depth.usage = TextureUsage::RenderTarget;

    RenderPassDesc pass = simple_pass();
    pass.depth_stencil.texture = backend.create_texture(depth);
    CHECK(pass.validate().find("depth/stencil") != std::string::npos);

    pass.depth_stencil.load = LoadAction::Clear;
    // A depth buffer nothing reads back is the textbook P21/P20 case: clear
    // it, use it within the pass, and throw it away.
    pass.depth_stencil.store = StoreAction::DontCare;
    CHECK(pass.validate().empty());
}

// ---------------------------------------------------------------------------
// Pipeline validation
// ---------------------------------------------------------------------------

TEST_CASE("a primitive Metal cannot draw never reaches a pipeline", "[gpu]") {
    PipelineDesc desc;
    desc.vertex_source = "// msl";
    desc.color_formats[0] = TextureFormat::BGRA8Unorm;

    CHECK(desc.validate().empty());

    // Quads are expanded by the translator and RECTS goes through the
    // object+mesh path (8.8). Either arriving here means Phase 8 skipped a
    // step, and a pipeline created from one would draw the wrong shape.
    desc.primitive = PrimitiveType::Quads;
    CHECK(desc.validate().find("cannot be drawn directly") != std::string::npos);

    desc.primitive = PrimitiveType::Rects;
    CHECK(desc.validate().find("mesh path") != std::string::npos);

    CHECK(needs_primitive_emulation(PrimitiveType::Rects));
    CHECK(needs_primitive_emulation(PrimitiveType::Quads));
    CHECK_FALSE(needs_primitive_emulation(PrimitiveType::Triangles));
}

TEST_CASE("a pipeline needs a vertex shader and a format", "[gpu]") {
    PipelineDesc desc;
    CHECK(desc.validate().find("no vertex shader") != std::string::npos);

    desc.vertex_source = "// msl";
    CHECK(desc.validate().find("no format") != std::string::npos);
}

// ---------------------------------------------------------------------------
// The null backend
// ---------------------------------------------------------------------------

TEST_CASE("a backend on a machine that cannot run refuses to start", "[gpu]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    NullBackend backend;
    Capabilities poor;
    poor.device_name = "Ancient GPU";
    backend.set_capabilities(poor);

    std::string error;
    CHECK_FALSE(backend.initialize(error));
    CHECK(error.find("Ancient GPU") != std::string::npos);
}

TEST_CASE_METHOD(BackendFixture, "a frame runs end to end", "[gpu]") {
    FrameInfo frame;
    REQUIRE(backend.begin_frame(frame));
    CHECK(frame.drawable.valid());
    CHECK(frame.width == 1280);
    CHECK(frame.index == 0);

    RenderPassDesc pass;
    pass.label = "clear";
    pass.color_count = 1;
    pass.colors[0].texture = frame.drawable;
    pass.colors[0].load = LoadAction::Clear;
    pass.colors[0].store = StoreAction::Store;
    REQUIRE(backend.begin_render_pass(pass));
    backend.end_render_pass();
    backend.end_frame();

    CHECK(backend.rejected() == 0);
    CHECK(backend.last_frame_stats().render_passes == 1);
    CHECK(backend.recorded("begin_render_pass 'clear' colors=1 [0:clear/store]"));

    // The drawable is a real texture handle, so the screen and an offscreen
    // target go down one code path rather than two.
    FrameInfo second;
    REQUIRE(backend.begin_frame(second));
    CHECK(second.index == 1);
    backend.end_frame();
}

TEST_CASE_METHOD(BackendFixture, "a pass left open at the end of a frame is caught", "[gpu]") {
    FrameInfo frame;
    REQUIRE(backend.begin_frame(frame));

    RenderPassDesc pass = simple_pass();
    REQUIRE(backend.begin_render_pass(pass));
    backend.end_frame();

    // On Metal this is an encoder never ended, which is a hang or a
    // validation failure rather than a wrong pixel -- so it is worth being a
    // line in a test instead.
    CHECK(backend.rejected() == 1);
    CHECK(backend.recorded("render pass still open"));
}

TEST_CASE_METHOD(BackendFixture, "drawing outside a pass is refused", "[gpu]") {
    DrawCall call;
    call.vertex_count = 3;
    backend.draw(call);
    CHECK(backend.rejected() == 1);
    CHECK(backend.last_frame_stats().draws == 0);
}

TEST_CASE_METHOD(BackendFixture, "a draw of nothing is a translation bug", "[gpu]") {
    FrameInfo frame;
    REQUIRE(backend.begin_frame(frame));
    RenderPassDesc pass = simple_pass();
    REQUIRE(backend.begin_render_pass(pass));

    backend.set_pipeline(simple_pipeline());
    backend.draw(DrawCall{});
    CHECK(backend.rejected() == 1);

    DrawCall real;
    real.vertex_count = 3;
    backend.draw(real);
    CHECK(backend.recorded("draw vertices 3"));

    backend.end_render_pass();
    backend.end_frame();
    CHECK(backend.last_frame_stats().draws == 1);
}

TEST_CASE_METHOD(BackendFixture, "a destroyed handle stops working", "[gpu]") {
    BufferDesc desc;
    desc.size = 256;
    desc.storage = Storage::Shared;
    const BufferHandle buffer = backend.create_buffer(desc);
    REQUIRE(buffer.valid());
    CHECK(backend.live_count(ResourceKind::Buffer) == 1);

    backend.destroy(buffer);
    CHECK(backend.live_count(ResourceKind::Buffer) == 0);

    // The second destroy is the interesting one: silently succeeding would
    // mean a double free looked fine right up until the handle number was
    // reused by something else.
    backend.destroy(buffer);
    CHECK(backend.rejected() == 1);

    const std::array<u8, 4> bytes{1, 2, 3, 4};
    CHECK_FALSE(backend.write_buffer(buffer, 0, bytes));
}

TEST_CASE_METHOD(BackendFixture, "a buffer cannot be memoryless", "[gpu][p21]") {
    // Tile memory holds render targets, nothing else. Metal refuses a
    // memoryless buffer, so the null backend does too.
    BufferDesc desc;
    desc.size = 64;
    desc.storage = Storage::Memoryless;
    CHECK_FALSE(backend.create_buffer(desc).valid());
    CHECK(backend.rejected() == 1);
}

TEST_CASE_METHOD(BackendFixture, "a buffer write stays inside the buffer", "[gpu]") {
    BufferDesc desc;
    desc.size = 8;
    desc.storage = Storage::Shared;
    const BufferHandle buffer = backend.create_buffer(desc);

    const std::array<u8, 4> bytes{0xDE, 0xAD, 0xBE, 0xEF};
    CHECK(backend.write_buffer(buffer, 4, bytes));
    CHECK_FALSE(backend.write_buffer(buffer, 6, bytes)); // two bytes over
    CHECK(backend.rejected() == 1);

    const auto* const contents = static_cast<const u8*>(backend.buffer_contents(buffer));
    REQUIRE(contents != nullptr);
    CHECK(contents[4] == 0xDE);
    CHECK(contents[7] == 0xEF);
    CHECK(contents[0] == 0);
}

TEST_CASE_METHOD(BackendFixture, "a memoryless texture cannot also be sampled", "[gpu][p21]") {
    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = TextureFormat::Depth32Float;
    desc.storage = Storage::Memoryless;
    desc.usage = TextureUsage::RenderTarget | TextureUsage::Sampled;
    desc.label = "depth";

    // Memoryless means it never leaves tile memory (P21). Sampling it later
    // is exactly the thing that cannot work, and a backend that accepted it
    // would hand back a texture full of nothing.
    CHECK_FALSE(backend.create_texture(desc).valid());
    CHECK(backend.rejected() == 1);

    desc.usage = TextureUsage::RenderTarget;
    CHECK(backend.create_texture(desc).valid());
}

TEST_CASE_METHOD(BackendFixture, "a border colour Metal cannot express is noted", "[gpu]") {
    SamplerDesc desc;
    desc.address_u = AddressMode::ClampToBorder;
    desc.border_color = ClearColor{0.5F, 0.0F, 0.0F, 1.0F}; // dark red: not one of the three
    REQUIRE(backend.create_sampler(desc).valid());
    CHECK(backend.recorded("border_emulated=true"));

    // Opaque black is one Metal has, so there is nothing for the shader
    // analysis pass to do.
    backend.clear_events();
    SamplerDesc plain;
    plain.address_u = AddressMode::ClampToBorder;
    plain.border_color = ClearColor{0.0F, 0.0F, 0.0F, 1.0F};
    REQUIRE(backend.create_sampler(plain).valid());
    CHECK(backend.recorded("border_emulated=false"));
}

TEST_CASE_METHOD(BackendFixture, "binding a resource that is gone is refused", "[gpu]") {
    RenderPassDesc pass = simple_pass();
    REQUIRE(backend.begin_render_pass(pass));
    backend.bind_texture(ShaderStage::Fragment, 0, TextureHandle{9999});
    backend.bind_buffer(ShaderStage::Vertex, 0, BufferHandle{9999}, 0);
    backend.bind_sampler(ShaderStage::Fragment, 0, SamplerHandle{9999});
    backend.set_pipeline(PipelineHandle{9999});
    CHECK(backend.rejected() == 4);
    backend.end_render_pass();
}

TEST_CASE_METHOD(BackendFixture, "pipelines and bindings belong to a pass", "[gpu]") {
    const PipelineHandle pipeline = simple_pipeline();
    SamplerDesc sampler_desc;
    const SamplerHandle sampler = backend.create_sampler(sampler_desc);

    // On Metal these are encoder calls, and there is no encoder outside a
    // pass. Remembering them for the next one would make the null backend
    // accept code that does nothing on the real one.
    backend.set_pipeline(pipeline);
    backend.bind_sampler(ShaderStage::Fragment, 0, sampler);
    CHECK(backend.rejected() == 2);
    CHECK(backend.recorded("set_pipeline outside a render pass"));

    // And a new pass starts with nothing set: a draw needs the pipeline
    // again, even though the last pass had one.
    RenderPassDesc pass = simple_pass();
    REQUIRE(backend.begin_render_pass(pass));
    backend.set_pipeline(pipeline);
    backend.end_render_pass();
    REQUIRE(backend.begin_render_pass(pass));
    DrawCall call;
    call.vertex_count = 3;
    backend.draw(call);
    CHECK(backend.recorded("draw with no pipeline"));
    backend.end_render_pass();
}

TEST_CASE_METHOD(BackendFixture, "a binding index past the stage's table is refused", "[gpu]") {
    SamplerDesc sampler_desc;
    const SamplerHandle sampler = backend.create_sampler(sampler_desc);
    RenderPassDesc pass = simple_pass();
    REQUIRE(backend.begin_render_pass(pass));

    // Sixteen samplers: the limit ADR-003 exists to lift, enforced until it
    // does.
    backend.bind_sampler(ShaderStage::Fragment, max_bound_samplers - 1, sampler);
    CHECK(backend.rejected() == 0);
    backend.bind_sampler(ShaderStage::Fragment, max_bound_samplers, sampler);
    CHECK(backend.rejected() == 1);
    backend.end_render_pass();
}

TEST_CASE_METHOD(BackendFixture, "a pipeline names functions its source has", "[gpu]") {
    PipelineDesc desc;
    desc.vertex_source = "vertex vertex_main";
    desc.fragment_source = "fragment something_else";
    desc.color_formats[0] = TextureFormat::BGRA8Unorm;

    std::string error;
    CHECK_FALSE(backend.create_pipeline(desc, error).valid());
    CHECK(error.find("fragment_main") != std::string::npos);

    desc.fragment_source = "fragment fragment_main";
    CHECK(backend.create_pipeline(desc, error).valid());
    CHECK(error.empty());
}

TEST_CASE_METHOD(BackendFixture, "an indexed draw needs an aligned index offset", "[gpu]") {
    BufferDesc indices;
    indices.size = 64;
    const BufferHandle buffer = backend.create_buffer(indices);
    RenderPassDesc pass = simple_pass();
    REQUIRE(backend.begin_render_pass(pass));
    backend.set_pipeline(simple_pipeline());

    DrawCall call;
    call.index_buffer = buffer;
    call.index_count = 3;
    call.index_offset = 2;
    call.index_is_32bit = true;
    backend.draw(call);
    CHECK(backend.recorded("not a multiple of the index size"));

    call.index_offset = 4;
    backend.draw(call);
    CHECK(backend.recorded("draw indexed 3"));
    backend.end_render_pass();
}

// ---------------------------------------------------------------------------
// Textures written from the CPU (7.6)
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(BackendFixture, "a texture write lands where it says", "[gpu]") {
    TextureDesc desc;
    desc.width = 4;
    desc.height = 2;
    desc.format = TextureFormat::RGBA8Unorm;
    desc.storage = Storage::Shared;
    const TextureHandle texture = backend.create_texture(desc);

    // A 2x2 block at (1,0), from rows eight bytes apart that carry 12 bytes
    // of padding each: the stride is the source's, not the texture's.
    std::vector<u8> source(2 * 20, 0);
    for (u32 y = 0; y < 2; ++y) {
        for (u32 x = 0; x < 8; ++x) {
            source[(y * 20) + x] = static_cast<u8>(0x10 * (y + 1) + x);
        }
    }
    REQUIRE(backend.write_texture(texture, Rect{1, 0, 2, 2}, source, 20));

    std::vector<u8> pixels(4 * 2 * 4, 0);
    REQUIRE(backend.read_texture(texture, pixels));
    CHECK(pixels[0] == 0);         // (0,0) untouched
    CHECK(pixels[4] == 0x10);      // (1,0) is the block's first byte
    CHECK(pixels[11] == 0x17);     // (2,0)'s last byte
    CHECK(pixels[16 + 4] == 0x20); // (1,1): the second source row
    CHECK(pixels[12] == 0);        // (3,0) untouched
}

TEST_CASE_METHOD(BackendFixture, "a texture write is refused where Metal refuses it", "[gpu]") {
    TextureDesc desc;
    desc.width = 4;
    desc.height = 4;
    desc.format = TextureFormat::RGBA8Unorm;
    const TextureHandle private_texture = backend.create_texture(desc);
    std::vector<u8> bytes(64, 0);

    // Private storage has no CPU address: that upload is a blit (8.2).
    CHECK_FALSE(backend.write_texture(private_texture, Rect{0, 0, 4, 4}, bytes, 16));

    desc.storage = Storage::Shared;
    const TextureHandle shared = backend.create_texture(desc);
    CHECK_FALSE(backend.write_texture(shared, Rect{2, 2, 4, 4}, bytes, 16)); // off the edge
    CHECK_FALSE(backend.write_texture(shared, Rect{0, 0, 4, 4}, bytes, 8));  // rows overlap
    CHECK_FALSE(backend.write_texture(shared, Rect{0, 0, 4, 4}, std::span{bytes}.first(32), 16));
    CHECK(backend.rejected() == 4);
    CHECK(backend.write_texture(shared, Rect{0, 0, 4, 4}, bytes, 16));
}

// ---------------------------------------------------------------------------
// Zero-copy host memory (7.4, P22)
// ---------------------------------------------------------------------------

namespace {

/// Whole pages of host memory, page-aligned, as the guest regions are.
struct Pages {
    explicit Pages(std::size_t count)
        : size(count * host_page_size()),
          memory(static_cast<u8*>(std::aligned_alloc(host_page_size(), size))) {}

    ~Pages() { std::free(memory); }

    Pages(const Pages&) = delete;
    Pages& operator=(const Pages&) = delete;

    std::size_t size;
    u8* memory;
};

} // namespace

TEST_CASE_METHOD(BackendFixture, "host memory is wrapped, not copied", "[gpu][p22]") {
    Pages pages(2);
    REQUIRE(pages.memory != nullptr);
    pages.memory[100] = 0x42;

    const BufferHandle buffer = backend.import_host_memory(pages.memory, pages.size, "guest");
    REQUIRE(buffer.valid());

    // The buffer *is* the memory: a write through either is seen by the
    // other, with no upload in between.
    CHECK(backend.buffer_contents(buffer) == pages.memory);
    const std::array<u8, 1> byte{0x99};
    REQUIRE(backend.write_buffer(buffer, 200, byte));
    CHECK(pages.memory[200] == 0x99);

    const MemoryStats stats = backend.memory_stats();
    CHECK(stats.imported == 1);
    CHECK(stats.imported_bytes == pages.size);

    // Destroying the wrapper leaves the memory alone; it was never ours.
    backend.destroy(buffer);
    CHECK(pages.memory[100] == 0x42);
    CHECK(backend.memory_stats().imported == 0);
}

TEST_CASE_METHOD(BackendFixture, "only whole pages can be wrapped", "[gpu][p22]") {
    Pages pages(2);
    CHECK_FALSE(backend.import_host_memory(pages.memory + 64, host_page_size(), "skewed").valid());
    CHECK_FALSE(backend.import_host_memory(pages.memory, host_page_size() + 1, "ragged").valid());
    CHECK_FALSE(backend.import_host_memory(nullptr, host_page_size(), "nothing").valid());
    CHECK(backend.rejected() == 3);
}

// ---------------------------------------------------------------------------
// Capture (7.8)
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(BackendFixture, "a capture is a bracket", "[gpu]") {
    std::string error;
    REQUIRE(backend.begin_capture("frame.gputrace", error));
    CHECK(backend.capturing());
    CHECK_FALSE(backend.begin_capture("again.gputrace", error));
    CHECK(error.find("already") != std::string::npos);
    backend.end_capture();
    CHECK_FALSE(backend.capturing());
    backend.end_capture();
    CHECK(backend.rejected() == 2);
}

// ---------------------------------------------------------------------------
// Shader diagnostics (7.6)
// ---------------------------------------------------------------------------

TEST_CASE("a shader error quotes the line it is about", "[gpu]") {
    const std::string source = "#include <metal_stdlib>\n"
                               "vertex float4 vertex_main() {\n"
                               "    return nope;\n"
                               "}\n";
    const std::string output = "program_source:3:12: error: use of undeclared identifier 'nope'\n"
                               "1 error generated.";

    const std::string annotated = annotate_shader_errors(source, output);
    INFO(annotated);
    CHECK(annotated.find("use of undeclared identifier") != std::string::npos);
    CHECK(annotated.find("    3 |     return nope;") != std::string::npos);
    // The caret sits under column 12, the 'n'.
    CHECK(annotated.find("      |            ^") != std::string::npos);
    CHECK(annotated.find("1 error generated.") != std::string::npos);
}

TEST_CASE("a shader error the compiler already quoted is not quoted twice", "[gpu]") {
    const std::string source = "a\nb\n    return nope;\n";
    const std::string output = "program_source:3:12: error: bad\n"
                               "    return nope;\n"
                               "           ^\n";
    const std::string annotated = annotate_shader_errors(source, output);
    CHECK(annotated == output);
}

TEST_CASE("a shader error with no usable location is passed through", "[gpu]") {
    const std::string source = "one line";
    CHECK(annotate_shader_errors(source, "program_source:9:1: error: past the end") ==
          "program_source:9:1: error: past the end");
    CHECK(annotate_shader_errors(source, "error: no location at all") ==
          "error: no location at all");
    CHECK(annotate_shader_errors(source, "elsewhere:1:1: error: another file") ==
          "elsewhere:1:1: error: another file");
}

// ---------------------------------------------------------------------------
// The presenter (7.6 / 7.7)
// ---------------------------------------------------------------------------

TEST_CASE("one screen fills the window with its aspect kept", "[gpu][present]") {
    const std::array<Rect, present_screen_count> sizes{Rect{0, 0, 1280, 720}, Rect{0, 0, 854, 480}};

    // A 16:9 window: the TV fills it exactly.
    PresentLayout layout = layout_screens(1920, 1080, sizes, {true, false});
    CHECK(layout.visible[0]);
    CHECK_FALSE(layout.visible[1]);
    CHECK(layout.rects[0] == Rect{0, 0, 1920, 1080});

    // A 4:3 window: full width, bars above and below.
    layout = layout_screens(1024, 768, sizes, {true, false});
    CHECK(layout.rects[0] == Rect{0, 96, 1024, 576});

    // A very wide one: full height, bars at the sides.
    layout = layout_screens(2560, 720, sizes, {true, false});
    CHECK(layout.rects[0] == Rect{640, 0, 1280, 720});
}

TEST_CASE("two screens stack, TV above GamePad", "[gpu][present]") {
    const std::array<Rect, present_screen_count> sizes{Rect{0, 0, 1280, 720}, Rect{0, 0, 854, 480}};
    const PresentLayout layout = layout_screens(1280, 1440, sizes, {true, true});
    REQUIRE(layout.visible[0]);
    REQUIRE(layout.visible[1]);
    CHECK(layout.rects[0] == Rect{0, 0, 1280, 720});
    CHECK(layout.rects[1].y >= 720);
    CHECK(layout.rects[1].y + layout.rects[1].height <= 1440);
    // 854x480 is not quite 16:9; it keeps its own shape rather than the TV's.
    CHECK(layout.rects[1].width == 1280);
    CHECK(layout.rects[1].height == 719);
}

TEST_CASE("a screen too small to see is not shown", "[gpu][present]") {
    const std::array<Rect, present_screen_count> sizes{Rect{0, 0, 1280, 720}, Rect{}};
    const PresentLayout layout = layout_screens(1, 1, sizes, {true, true});
    CHECK_FALSE(layout.visible[0]);
    CHECK_FALSE(layout.visible[1]);
}

TEST_CASE_METHOD(BackendFixture, "the presenter uploads and draws each screen", "[gpu][present]") {
    Presenter presenter;
    std::string error;
    REQUIRE(presenter.initialize(backend, TextureFormat::BGRA8Unorm, error));

    // A 4x2 image in rows of 24 bytes, as OSScreen's DRC has a pitch wider
    // than its picture.
    std::vector<u8> pixels(2 * 24, 0xAB);
    presenter.update(PresentScreen::Tv, ScreenImage{pixels, 4, 2, 24});
    CHECK(presenter.has_image(PresentScreen::Tv));
    CHECK_FALSE(presenter.has_image(PresentScreen::Drc));

    FrameInfo frame;
    REQUIRE(backend.begin_frame(frame));
    REQUIRE(presenter.draw(frame.drawable, frame.width, frame.height));
    backend.end_frame();

    CHECK(backend.rejected() == 0);
    CHECK(backend.recorded("write_texture"));
    CHECK(backend.recorded("bind_texture fragment [0]"));
    CHECK(backend.recorded("bind_sampler fragment [0]"));
    CHECK(backend.recorded("draw vertices 3"));
    CHECK(backend.last_frame_stats().draws == 1);

    // The bars are cleared rather than loaded: last frame's contents are
    // never needed (P20).
    CHECK(backend.recorded("begin_render_pass 'present' colors=1 [0:clear/store]"));
    CHECK(presenter.last_layout().visible[0]);
}

TEST_CASE_METHOD(BackendFixture,
                 "the presenter never writes the texture the last frame read",
                 "[gpu][present]") {
    Presenter presenter;
    std::string error;
    REQUIRE(presenter.initialize(backend, TextureFormat::BGRA8Unorm, error));

    std::vector<u8> pixels(4 * 4 * 4, 0);
    std::vector<std::string> written;
    for (u32 frame_number = 0; frame_number < 2 * frames_in_flight; ++frame_number) {
        pixels[0] = static_cast<u8>(frame_number);
        presenter.update(PresentScreen::Tv, ScreenImage{pixels, 4, 4, 16});

        backend.clear_events();
        FrameInfo frame;
        REQUIRE(backend.begin_frame(frame));
        REQUIRE(presenter.draw(frame.drawable, frame.width, frame.height));
        backend.end_frame();

        for (const std::string& line : backend.events()) {
            if (line.rfind("write_texture ", 0) == 0) {
                written.push_back(line.substr(14, line.find(' ', 14) - 14));
            }
        }
    }

    // One upload a frame, rotating: no texture is written twice in a row,
    // or within `frames_in_flight` frames of last being written.
    REQUIRE(written.size() == 2 * frames_in_flight);
    for (std::size_t index = 1; index < written.size(); ++index) {
        for (std::size_t back = 1; back < frames_in_flight && back <= index; ++back) {
            CHECK(written[index] != written[index - back]);
        }
    }
    CHECK(backend.rejected() == 0);
}

TEST_CASE_METHOD(BackendFixture, "a frame with nothing new uploads nothing", "[gpu][present]") {
    Presenter presenter;
    std::string error;
    REQUIRE(presenter.initialize(backend, TextureFormat::BGRA8Unorm, error));
    std::vector<u8> pixels(16, 0);
    presenter.update(PresentScreen::Drc, ScreenImage{pixels, 2, 2, 8});

    FrameInfo frame;
    REQUIRE(backend.begin_frame(frame));
    REQUIRE(presenter.draw(frame.drawable, frame.width, frame.height));
    backend.end_frame();

    backend.clear_events();
    REQUIRE(backend.begin_frame(frame));
    REQUIRE(presenter.draw(frame.drawable, frame.width, frame.height));
    backend.end_frame();

    // The same image is drawn again from the texture it is already in.
    CHECK_FALSE(backend.recorded("write_texture"));
    CHECK(backend.recorded("draw vertices 3"));
}

TEST_CASE_METHOD(BackendFixture, "a hidden screen gives its space away", "[gpu][present]") {
    Presenter presenter;
    std::string error;
    REQUIRE(presenter.initialize(backend, TextureFormat::BGRA8Unorm, error));
    std::vector<u8> pixels(16 * 9 * 4, 0);
    presenter.update(PresentScreen::Tv, ScreenImage{pixels, 16, 9, 64});
    presenter.update(PresentScreen::Drc, ScreenImage{pixels, 16, 9, 64});
    presenter.set_shown(PresentScreen::Drc, false);

    FrameInfo frame;
    REQUIRE(backend.begin_frame(frame));
    REQUIRE(presenter.draw(frame.drawable, frame.width, frame.height));
    backend.end_frame();

    CHECK(presenter.last_layout().visible[0]);
    CHECK_FALSE(presenter.last_layout().visible[1]);
    CHECK(presenter.last_layout().rects[0] == Rect{0, 0, 1280, 720});
    CHECK(backend.last_frame_stats().draws == 1);
}

TEST_CASE_METHOD(BackendFixture,
                 "the presenter shows a texture the GPU already holds",
                 "[gpu][present]") {
    Presenter presenter;
    std::string error;
    REQUIRE(presenter.initialize(backend, TextureFormat::BGRA8Unorm, error));

    // A scan-out frame as ADR-006 has it: GPU-only, drawn to, sampled.
    TextureDesc desc;
    desc.width = 64;
    desc.height = 48;
    desc.format = TextureFormat::RGBA8Unorm;
    desc.usage = TextureUsage::RenderTarget | TextureUsage::Sampled;
    desc.storage = Storage::Private;
    desc.label = "scan";
    const TextureHandle scan = backend.create_texture(desc);
    REQUIRE(scan.valid());

    presenter.show(PresentScreen::Tv, ScreenTexture{scan, 64, 48});
    CHECK(presenter.has_image(PresentScreen::Tv));

    backend.clear_events();
    FrameInfo frame;
    REQUIRE(backend.begin_frame(frame));
    REQUIRE(presenter.draw(frame.drawable, frame.width, frame.height));
    backend.end_frame();

    // Sampled where it is: nothing made, nothing uploaded.
    CHECK(backend.recorded("bind_texture fragment [0] = " + std::to_string(scan.value())));
    CHECK_FALSE(backend.recorded("create_texture"));
    CHECK_FALSE(backend.recorded("write_texture"));
    CHECK(backend.last_frame_stats().draws == 1);
    // Laid out by the texture's size: 4:3 in a 16:9 window.
    CHECK(presenter.last_layout().rects[0] == Rect{160, 0, 960, 720});
    CHECK(backend.rejected() == 0);
}

TEST_CASE_METHOD(BackendFixture,
                 "bytes and textures take turns, and a forgotten texture is not drawn",
                 "[gpu][present]") {
    Presenter presenter;
    std::string error;
    REQUIRE(presenter.initialize(backend, TextureFormat::BGRA8Unorm, error));

    TextureDesc desc;
    desc.width = 16;
    desc.height = 9;
    desc.format = TextureFormat::RGBA8Unorm;
    desc.usage = TextureUsage::RenderTarget | TextureUsage::Sampled;
    desc.label = "scan";
    const TextureHandle scan = backend.create_texture(desc);
    REQUIRE(scan.valid());
    const std::string bound_scan = "bind_texture fragment [0] = " + std::to_string(scan.value());

    const auto draw_frame = [&] {
        backend.clear_events();
        FrameInfo frame;
        REQUIRE(backend.begin_frame(frame));
        REQUIRE(presenter.draw(frame.drawable, frame.width, frame.height));
        backend.end_frame();
    };

    // Bytes first: uploaded and drawn from the presenter's own texture.
    std::vector<u8> pixels(16 * 9 * 4, 0x40);
    presenter.update(PresentScreen::Tv, ScreenImage{pixels, 16, 9, 64});
    draw_frame();
    CHECK(backend.recorded("write_texture"));
    CHECK_FALSE(backend.recorded(bound_scan));

    // Then a texture: drawn instead, nothing uploaded.
    presenter.show(PresentScreen::Tv, ScreenTexture{scan, 16, 9});
    draw_frame();
    CHECK_FALSE(backend.recorded("write_texture"));
    CHECK(backend.recorded(bound_scan));

    // Bytes again replace it.
    presenter.update(PresentScreen::Tv, ScreenImage{pixels, 16, 9, 64});
    draw_frame();
    CHECK(backend.recorded("write_texture"));
    CHECK_FALSE(backend.recorded(bound_scan));

    // Taken back before its owner destroys it: the screen has nothing to
    // show, and is not drawn.
    presenter.show(PresentScreen::Tv, ScreenTexture{scan, 16, 9});
    presenter.forget(scan);
    backend.destroy(scan);
    CHECK_FALSE(presenter.has_image(PresentScreen::Tv));
    draw_frame();
    CHECK_FALSE(presenter.last_layout().visible[0]);
    CHECK(backend.last_frame_stats().draws == 0);

    // Nothing to show is refused, and changes nothing.
    presenter.show(PresentScreen::Drc, ScreenTexture{TextureHandle{}, 16, 9});
    CHECK_FALSE(presenter.has_image(PresentScreen::Drc));
    CHECK(backend.rejected() == 0);
}

TEST_CASE_METHOD(BackendFixture, "a clear can be rendered and read back", "[gpu]") {
    TextureDesc desc;
    desc.width = 4;
    desc.height = 4;
    desc.format = TextureFormat::BGRA8Unorm;
    desc.usage = TextureUsage::RenderTarget;
    desc.storage = Storage::Shared;
    desc.label = "offscreen";
    const TextureHandle target = backend.create_texture(desc);
    REQUIRE(target.valid());

    RenderPassDesc pass;
    pass.label = "offscreen clear";
    pass.color_count = 1;
    pass.colors[0].texture = target;
    pass.colors[0].load = LoadAction::Clear;
    pass.colors[0].store = StoreAction::Store;
    pass.colors[0].clear = ClearColor{1.0F, 0.0F, 0.0F, 1.0F};

    // No frame, and no window: rendering does not require a swapchain, which
    // is what the headless compatibility runner (10.5) depends on.
    REQUIRE(backend.begin_render_pass(pass));
    backend.end_render_pass();
    backend.submit();
    backend.wait_idle();

    std::vector<u8> pixels(4U * 4U * 4U, 0);
    REQUIRE(backend.read_texture(target, pixels));

    // BGRA: opaque red is 00 00 FF FF, and getting the channel order wrong
    // is the classic way to read back a plausible but mirrored image.
    CHECK(pixels[0] == 0x00);
    CHECK(pixels[1] == 0x00);
    CHECK(pixels[2] == 0xFF);
    CHECK(pixels[3] == 0xFF);

    // ... and every other texel too, not just the first.
    CHECK(pixels[(15 * 4) + 2] == 0xFF);
    CHECK(backend.rejected() == 0);
}

TEST_CASE_METHOD(BackendFixture, "a readback that does not fit is refused", "[gpu]") {
    TextureDesc desc;
    desc.width = 8;
    desc.height = 8;
    desc.format = TextureFormat::RGBA8Unorm;
    desc.usage = TextureUsage::RenderTarget;
    const TextureHandle target = backend.create_texture(desc);

    std::vector<u8> small(16, 0);
    CHECK_FALSE(backend.read_texture(target, small));
    CHECK(backend.rejected() == 1);
}

TEST_CASE("bytes per pixel is zero where there is no such thing", "[gpu]") {
    CHECK(format_bytes_per_pixel(TextureFormat::BGRA8Unorm) == 4);
    CHECK(format_bytes_per_pixel(TextureFormat::R8Unorm) == 1);
    CHECK(format_bytes_per_pixel(TextureFormat::RGBA32Float) == 16);

    // Compressed formats have bytes per *block*. Zero rather than a guess:
    // a row stride computed from a wrong answer allocates a fraction of what
    // is needed and reads past the end of it.
    CHECK(format_bytes_per_pixel(TextureFormat::BC3Unorm) == 0);
    CHECK(format_bytes_per_pixel(TextureFormat::Unknown) == 0);
    CHECK(format_is_compressed(TextureFormat::BC3Unorm));
}

TEST_CASE("the metal backend is absent on a host without Metal", "[gpu]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

#if WIIU_PLATFORM_APPLE
    // Creating it is allowed anywhere; whether it *initializes* depends on
    // the GPU, and that is what `shortfall()` reports.
    CHECK(create_metal_backend() != nullptr);
#else
    // The core-only build has to answer this question rather than fail to
    // link, which is what keeps the emulator testable without a Mac.
    CHECK(create_metal_backend() == nullptr);
#endif
}
