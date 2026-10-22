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
                          .primitive = gpu::PrimitiveType::Triangles,
                          .blend = {}};
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

    // So is another way of blending.
    PipelineKey blended = key;
    blended.blend.enabled = true;
    CHECK(cache.pipeline(blended, error) != first);
    CHECK(cache.stats().builds == 3);

    cache.clear();
    CHECK(cache.size() == 0);
    CHECK(backend.live_count(gpu::ResourceKind::Pipeline) == 0);
}

// --------------------------------------------------------------------------
// Blending and culling
// --------------------------------------------------------------------------

namespace {

/// CB_COLOR_CONTROL as GX2SetColorControl packs it: ROP3, blend enables,
/// and colour writes on or off.
void put_color_control(RegisterFile& registers, u32 rop3, u32 blend_enable, bool write) {
    REQUIRE(registers.write(reg::cb_color_control,
                            (rop3 << 16) | (blend_enable << 8) | ((write ? 0U : 1U) << 4)));
}

/// CB_BLENDn_CONTROL: colour source, function, destination; then alpha's.
void put_blend(RegisterFile& registers,
               u32 target,
               u32 source,
               u32 combine,
               u32 destination,
               bool separate = false,
               u32 alpha_source = 0,
               u32 alpha_combine = 0,
               u32 alpha_destination = 0) {
    REQUIRE(registers.write(reg::cb_blend0_control + (4 * target),
                            source | (combine << 5) | (destination << 8) | (alpha_source << 16) |
                                (alpha_combine << 21) | (alpha_destination << 24) |
                                ((separate ? 1U : 0U) << 29)));
}

[[nodiscard]] gpu::BlendState blend_of(const RegisterFile& registers, u32 target = 0) {
    std::string error;
    const std::optional<gpu::BlendState> blend = blend_state(registers, target, error);
    INFO(error);
    REQUIRE(blend.has_value());
    return *blend;
}

} // namespace

TEST_CASE("blending is translated from the colour registers", "[latte][draw][blend]") {
    RegisterFile registers;
    REQUIRE(registers.write(reg::cb_target_mask, 0xFFFFFFFF));

    // GX2's default: a plain copy, nothing blended, every channel written.
    put_color_control(registers, 0xCC, 0, true);
    put_blend(registers, 0, 4, 0, 5); // SRC_ALPHA + ONE_MINUS_SRC_ALPHA, not enabled
    CHECK(blend_of(registers) == gpu::BlendState{});

    // Enabled for target 0: alpha blends as colour does without
    // SEPARATE_ALPHA_BLEND.
    put_color_control(registers, 0xCC, 0x1, true);
    const gpu::BlendState alpha = blend_of(registers);
    CHECK(alpha.enabled);
    CHECK(alpha.source_rgb == gpu::BlendFactor::SourceAlpha);
    CHECK(alpha.destination_rgb == gpu::BlendFactor::OneMinusSourceAlpha);
    CHECK(alpha.rgb == gpu::BlendOperation::Add);
    CHECK(alpha.source_alpha == gpu::BlendFactor::SourceAlpha);
    CHECK(alpha.destination_alpha == gpu::BlendFactor::OneMinusSourceAlpha);
    CHECK(alpha.alpha == gpu::BlendOperation::Add);

    // Each target its own register and bit: target 1 is not enabled.
    put_blend(registers, 1, 1, 0, 1);
    CHECK_FALSE(blend_of(registers, 1).enabled);

    // Separate alpha, the reversed subtraction, the constant colour, and the
    // numbers that differ between Latte and Metal (DST_ALPHA is 6 on Latte,
    // DST_COLOR 8).
    put_blend(registers, 0, 13, 4, 6, true, 8, 1, 19);
    const gpu::BlendState separate = blend_of(registers);
    CHECK(separate.source_rgb == gpu::BlendFactor::BlendColor);
    CHECK(separate.destination_rgb == gpu::BlendFactor::DestinationAlpha);
    CHECK(separate.rgb == gpu::BlendOperation::ReverseSubtract);
    CHECK(separate.source_alpha == gpu::BlendFactor::DestinationColor);
    CHECK(separate.destination_alpha == gpu::BlendFactor::BlendAlpha);
    CHECK(separate.alpha == gpu::BlendOperation::Subtract);

    // D3D9's "both" factors stand for source and destination at once.
    put_blend(registers, 0, 11, 0, 0);
    const gpu::BlendState both = blend_of(registers);
    CHECK(both.source_rgb == gpu::BlendFactor::SourceAlpha);
    CHECK(both.destination_rgb == gpu::BlendFactor::OneMinusSourceAlpha);
    put_blend(registers, 0, 12, 0, 1);
    CHECK(blend_of(registers).source_rgb == gpu::BlendFactor::OneMinusSourceAlpha);
    CHECK(blend_of(registers).destination_rgb == gpu::BlendFactor::SourceAlpha);
}

TEST_CASE("the channels written come from the target mask, or none", "[latte][draw][blend]") {
    RegisterFile registers;
    put_color_control(registers, 0xCC, 0, true);
    REQUIRE(registers.write(reg::cb_target_mask, 0x00000052)); // target 0: green; 1: red, blue
    CHECK(blend_of(registers, 0).write_mask == gpu::color_write::green);
    CHECK(blend_of(registers, 1).write_mask == (gpu::color_write::red | gpu::color_write::blue));

    // Colour writes off (GX2SetColorControl's last argument).
    put_color_control(registers, 0xCC, 0, false);
    CHECK(blend_of(registers, 0).write_mask == 0);
}

TEST_CASE("what blending cannot express yet is refused by name", "[latte][draw][blend]") {
    RegisterFile registers;
    std::string error;
    put_color_control(registers, 0x66, 0, true); // XOR
    CHECK_FALSE(blend_state(registers, 0, error).has_value());
    CHECK(error.find("logic op (ROP3 0x66)") != std::string::npos);

    put_color_control(registers, 0xCC, 0x1, true);
    put_blend(registers, 0, 4, 7, 5); // no combine function 7
    CHECK_FALSE(blend_state(registers, 0, error).has_value());
    CHECK(error.find("combine function 7") != std::string::npos);

    put_blend(registers, 0, 4, 0, 11); // a "both" factor as the destination
    CHECK_FALSE(blend_state(registers, 0, error).has_value());
    CHECK(error.find("blend factors 4 and 11") != std::string::npos);
}

TEST_CASE("the blend colour and the faces culled come from their registers",
          "[latte][draw][blend]") {
    RegisterFile registers;
    const std::array<f32, 4> colour{0.25F, 0.5F, 0.75F, 1.0F};
    for (u32 i = 0; i < 4; ++i) {
        REQUIRE(registers.write(reg::cb_blend_red + (4 * i), std::bit_cast<u32>(colour[i])));
    }
    CHECK(blend_color(registers) == gpu::ClearColor{0.25F, 0.5F, 0.75F, 1.0F});

    std::string error;
    const auto raster = [&](u32 value) {
        REQUIRE(registers.write(reg::pa_su_sc_mode_cntl, value));
        const std::optional<RasterState> state = raster_state(registers, error);
        REQUIRE(state.has_value());
        return *state;
    };
    // GX2's default: nothing culled, counter-clockwise in front.
    RasterState state = raster(0);
    CHECK(state.cull == gpu::CullMode::None);
    CHECK(state.front == gpu::Winding::CounterClockwise);
    CHECK_FALSE(state.cull_all);

    state = raster(0x2 | 0x4); // back culled, clockwise in front
    CHECK(state.cull == gpu::CullMode::Back);
    CHECK(state.front == gpu::Winding::Clockwise);
    CHECK(raster(0x1).cull == gpu::CullMode::Front);

    // Both: Metal has no word for it; the draw is not made.
    state = raster(0x3);
    CHECK(state.cull_all);

    // Polygon mode is refused.
    REQUIRE(registers.write(reg::pa_su_sc_mode_cntl, 1U << 3));
    CHECK_FALSE(raster_state(registers, error).has_value());
    CHECK(error.find("polygon mode") != std::string::npos);
}
