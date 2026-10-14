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

#include <catch2/catch_test_macros.hpp>

#include <array>
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
    backend.bind_texture(ShaderStage::Fragment, 0, TextureHandle{9999});
    backend.bind_buffer(ShaderStage::Vertex, 0, BufferHandle{9999}, 0);
    backend.bind_sampler(ShaderStage::Fragment, 0, SamplerHandle{9999});
    backend.set_pipeline(PipelineHandle{9999});
    CHECK(backend.rejected() == 4);
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
