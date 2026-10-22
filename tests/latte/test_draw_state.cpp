// Phase 8.10 / 8.11 — a draw's fixed state, and the pipelines it is drawn
// with.
//
// What is guarded: Latte's primitives map to Metal's where Metal has them and
// are refused where it does not; viewport 0 comes back out of its scale and
// offset as the rectangle GX2SetViewport was given, flipped ones included,
// with the depth range by the clip-space convention; the scissor is cut to
// the render target, and one that leaves nothing is none; and a pipeline is
// built once and found after, and so is a failure, with its error kept.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/latte/draw.hpp>
#include <wiiu/latte/draw_state.hpp>
#include <wiiu/latte/pipeline_cache.hpp>
#include <wiiu/latte/render_state.hpp>
#include <wiiu/latte/viewport.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <string>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

/// Viewport 0's registers for a rectangle and depth range, as GX2SetViewport
/// writes them (hle/gx2_state.cpp).
void put_viewport(
    RegisterFile& registers, f32 x, f32 y, f32 width, f32 height, f32 near_z, f32 far_z) {
    const std::array<f32, 6> transform{width * 0.5F,
                                       x + (width * 0.5F),
                                       height * -0.5F,
                                       y + (height * 0.5F),
                                       (far_z - near_z) * 0.5F,
                                       (far_z + near_z) * 0.5F};
    for (u32 i = 0; i < transform.size(); ++i) {
        REQUIRE(
            registers.write(reg::pa_cl_vport_xscale_0 + (i * 4), std::bit_cast<u32>(transform[i])));
    }
}

void put_scissor(RegisterFile& registers, u32 left, u32 top, u32 right, u32 bottom) {
    REQUIRE(registers.write(reg::pa_sc_generic_scissor_tl, left | (top << 16) | (1U << 31)));
    REQUIRE(registers.write(reg::pa_sc_generic_scissor_br, right | (bottom << 16)));
}

} // namespace

TEST_CASE("Latte's primitives map to Metal's where Metal has them", "[latte][draw]") {
    const auto map = [](PrimitiveType type) { return primitive_type(static_cast<u32>(type)); };
    CHECK(map(PrimitiveType::PointList) == gpu::PrimitiveType::Points);
    CHECK(map(PrimitiveType::LineList) == gpu::PrimitiveType::Lines);
    CHECK(map(PrimitiveType::LineStrip) == gpu::PrimitiveType::LineStrip);
    CHECK(map(PrimitiveType::TriangleList) == gpu::PrimitiveType::Triangles);
    CHECK(map(PrimitiveType::TriangleStrip) == gpu::PrimitiveType::TriangleStrip);
    // The backend's own names, which only the mesh path will draw (8.8).
    CHECK(map(PrimitiveType::QuadList) == gpu::PrimitiveType::Quads);
    CHECK(map(PrimitiveType::RectList) == gpu::PrimitiveType::Rects);
    // Their indices want rewriting first.
    CHECK_FALSE(map(PrimitiveType::TriangleFan).has_value());
    CHECK_FALSE(map(PrimitiveType::LineLoop).has_value());
    CHECK_FALSE(map(PrimitiveType::Polygon).has_value());
    CHECK_FALSE(map(PrimitiveType::QuadStrip).has_value());
    CHECK_FALSE(map(PrimitiveType::TriangleListAdjacency).has_value());
    CHECK_FALSE(map(PrimitiveType::None).has_value());
}

TEST_CASE("the viewport comes back as the rectangle GX2 was given", "[latte][draw]") {
    RegisterFile registers;

    // WHB's: the whole 720p target, depth 0 to 1, OpenGL's clip space.
    put_viewport(registers, 0.0F, 0.0F, 1280.0F, 720.0F, 0.0F, 1.0F);
    CHECK(viewport(registers) == gpu::Viewport{0.0F, 0.0F, 1280.0F, 720.0F, 0.0F, 1.0F});

    // Somewhere inside it.
    put_viewport(registers, 100.0F, 50.0F, 200.0F, 100.0F, 0.0F, 1.0F);
    CHECK(viewport(registers) == gpu::Viewport{100.0F, 50.0F, 200.0F, 100.0F, 0.0F, 1.0F});

    // Upside down, as a title flips one: the height comes back negative.
    put_viewport(registers, 0.0F, 720.0F, 1280.0F, -720.0F, 0.0F, 1.0F);
    const gpu::Viewport flipped = viewport(registers);
    CHECK(flipped.y == 720.0F);
    CHECK(flipped.height == -720.0F);

    // Depth: with OpenGL's clip space z runs from -1, so the near plane is
    // the offset less the scale; with DirectX's ("half Z") it is the offset.
    put_viewport(registers, 0.0F, 0.0F, 64.0F, 64.0F, 0.25F, 0.75F);
    CHECK(viewport(registers).near_z == 0.25F);
    CHECK(viewport(registers).far_z == 0.75F);
    REQUIRE(registers.write(reg::pa_cl_clip_cntl, 1U << 19));
    CHECK(viewport(registers).near_z == 0.5F);
    CHECK(viewport(registers).far_z == 0.75F);
}

TEST_CASE("the scissor is cut to the render target", "[latte][draw]") {
    RegisterFile registers;
    put_scissor(registers, 0, 0, 1280, 720);
    CHECK(scissor(registers, 1280, 720) == gpu::Rect{0, 0, 1280, 720});

    // GX2's default reaches 8192 either way; Metal refuses past the target.
    put_scissor(registers, 0, 0, 8192, 8192);
    CHECK(scissor(registers, 854, 480) == gpu::Rect{0, 0, 854, 480});

    put_scissor(registers, 100, 50, 300, 150);
    CHECK(scissor(registers, 1280, 720) == gpu::Rect{100, 50, 200, 100});

    // Nothing left: nothing to draw.
    put_scissor(registers, 900, 0, 1000, 100);
    CHECK_FALSE(scissor(registers, 854, 480).has_value());
    put_scissor(registers, 10, 10, 10, 20);
    CHECK_FALSE(scissor(registers, 64, 64).has_value());
}

TEST_CASE("a pipeline is built once, and so is a failure", "[latte][draw][pipeline]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);
    gpu::NullBackend backend;
    std::string error;
    REQUIRE(backend.initialize(error));
    PipelineCache cache(backend);

    const PipelineKey key{.vertex_msl = "vertex VertexOut vertex_main(...)",
                          .pixel_msl = "fragment float4 fragment_main(...)",
                          .color_format = gpu::TextureFormat::RGBA8Unorm,
                          .primitive = gpu::PrimitiveType::Triangles};
    const gpu::PipelineHandle first = cache.pipeline(key, error);
    REQUIRE(first.valid());
    CHECK(cache.pipeline(key, error) == first);
    CHECK(cache.stats().builds == 1);
    CHECK(cache.stats().hits == 1);

    // Another colour format is another pipeline.
    PipelineKey srgb = key;
    srgb.color_format = gpu::TextureFormat::RGBA8UnormSrgb;
    CHECK(cache.pipeline(srgb, error) != first);
    CHECK(cache.stats().builds == 2);

    // A shader that does not build is refused once, and its error kept.
    PipelineKey broken = key;
    broken.pixel_msl = "no entry point here";
    CHECK_FALSE(cache.pipeline(broken, error).valid());
    CHECK(error.find("fragment_main") != std::string::npos);
    const u64 rejected = backend.rejected();
    error.clear();
    CHECK_FALSE(cache.pipeline(broken, error).valid());
    CHECK(error.find("fragment_main") != std::string::npos);
    CHECK(backend.rejected() == rejected); // not compiled again
    CHECK(cache.stats().failures == 1);
    CHECK(cache.size() == 3);

    cache.clear();
    CHECK(cache.size() == 0);
    CHECK(backend.live_count(gpu::ResourceKind::Pipeline) == 0);
}
