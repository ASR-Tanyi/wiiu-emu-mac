// Phase 8.13, first form (ADR-006) — the surface cache.
//
// What is guarded: a colour buffer's identity, from register values and from
// the register file alike; the texture's size from CB_COLOR_SIZE; one
// texture per surface, counted as hits and misses; the formats not rendered
// to yet refused and counted; and that a cached surface is a render target
// the backend clears, which is what GX2's clear will become (ADR-006).

#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/latte/surface_cache.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

/// CB_COLOR_SIZE for a surface `width` by `height` pixels, both multiples of
/// eight: the pitch and the slice, in 8x8 tiles, less one.
[[nodiscard]] constexpr u32 size_register(u32 width, u32 height) noexcept {
    const u32 pitch = width / 8;
    const u32 slice = pitch * (height / 8);
    return (pitch - 1) | ((slice - 1) << 10);
}

/// CB_COLOR_INFO for 8_8_8_8, 2D-tiled, in `number`, standard component order.
[[nodiscard]] constexpr u32 info_register(CbNumberType number, CbCompSwap swap = CbCompSwap::Std) {
    return (0x1AU << 2) | (static_cast<u32>(TileMode::Tiled2DThin1) << 8) |
           (static_cast<u32>(number) << 12) | (static_cast<u32>(swap) << 16);
}

struct Fixture {
    std::unique_ptr<gpu::GpuBackend> backend = gpu::create_null_backend();
    std::unique_ptr<SurfaceCache> cache;

    Fixture() {
        std::string error;
        REQUIRE(backend->initialize(error));
        cache = std::make_unique<SurfaceCache>(*backend);
    }
};

} // namespace

TEST_CASE("a colour buffer's identity and size come from its registers", "[latte][surface]") {
    const ColorBufferKey key = ColorBufferKey::from(
        0x00123400 >> 8, size_register(1280, 720), info_register(CbNumberType::Unorm), 0);
    CHECK(key.format == 0x1A);
    CHECK(key.number_type == static_cast<u32>(CbNumberType::Unorm));
    CHECK(key.array_mode == static_cast<u32>(TileMode::Tiled2DThin1));
    CHECK(key.width() == 1280);
    CHECK(key.height() == 720);

    // The same values in render target 2's registers name the same surface.
    RegisterFile registers;
    REQUIRE(registers.write(reg::cb_color0_base + 8, 0x00123400 >> 8));
    REQUIRE(registers.write(reg::cb_color0_size + 8, size_register(1280, 720)));
    REQUIRE(registers.write(reg::cb_color0_info + 8, info_register(CbNumberType::Unorm)));
    CHECK(ColorBufferKey::from(registers, 2) == key);
    CHECK(ColorBufferKey::from(registers, 0) != key);

    // Blend and rounding bits in CB_COLOR_INFO change how it is written, not
    // which surface it is.
    REQUIRE(registers.write(reg::cb_color0_info + 8,
                            info_register(CbNumberType::Unorm) | (1U << 20) | (1U << 25)));
    CHECK(ColorBufferKey::from(registers, 2) == key);
}

TEST_CASE_METHOD(Fixture, "one texture a surface, counted", "[latte][surface]") {
    const ColorBufferKey tv = ColorBufferKey::from(
        0x1000, size_register(1280, 720), info_register(CbNumberType::Unorm), 0);
    const ColorBufferKey drc =
        ColorBufferKey::from(0x2000, size_register(856, 480), info_register(CbNumberType::Srgb), 0);

    CHECK_FALSE(cache->find(tv).has_value());
    std::string error;
    const std::optional<ColorSurface> first = cache->color_buffer(tv, error);
    INFO(error);
    REQUIRE(first.has_value());
    CHECK(first->width == 1280);
    CHECK(first->height == 720);
    CHECK(first->format == gpu::TextureFormat::RGBA8Unorm);

    const std::optional<ColorSurface> again = cache->color_buffer(tv, error);
    REQUIRE(again.has_value());
    CHECK(again->texture == first->texture);
    CHECK(cache->find(tv).has_value());

    const std::optional<ColorSurface> other = cache->color_buffer(drc, error);
    REQUIRE(other.has_value());
    CHECK(other->texture != first->texture);
    CHECK(other->format == gpu::TextureFormat::RGBA8UnormSrgb);

    CHECK(cache->stats().misses == 2);
    CHECK(cache->stats().hits == 1);
    CHECK(cache->size() == 2);

    cache->clear();
    CHECK(cache->size() == 0);
    CHECK_FALSE(cache->find(tv).has_value());
}

TEST_CASE_METHOD(Fixture,
                 "what is not rendered to yet is refused, and counted",
                 "[latte][surface]") {
    std::string error;
    // 32_32_32_32 float.
    const ColorBufferKey wide =
        ColorBufferKey::from(0x1000, size_register(64, 64), (0x23U << 2) | (7U << 12), 0);
    CHECK_FALSE(cache->color_buffer(wide, error).has_value());
    CHECK(error.find("format 0x23") != std::string::npos);

    // 8_8_8_8 with the alternate component order, and as UINT.
    CHECK_FALSE(
        cache
            ->color_buffer(ColorBufferKey::from(0x1000,
                                                size_register(64, 64),
                                                info_register(CbNumberType::Unorm, CbCompSwap::Alt),
                                                0),
                           error)
            .has_value());
    CHECK_FALSE(
        cache
            ->color_buffer(ColorBufferKey::from(
                               0x1000, size_register(64, 64), info_register(CbNumberType::Uint), 0),
                           error)
            .has_value());

    // A slice smaller than a row of tiles: a register misread.
    CHECK_FALSE(
        cache
            ->color_buffer(ColorBufferKey::from(0x1000, 159, info_register(CbNumberType::Unorm), 0),
                           error)
            .has_value());
    CHECK(error.find("makes no texture") != std::string::npos);

    CHECK(cache->stats().refused == 4);
    CHECK(cache->size() == 0);
}

TEST_CASE_METHOD(Fixture,
                 "a cached surface is a render target the GPU clears",
                 "[latte][surface]") {
    std::string error;
    const std::optional<ColorSurface> surface = cache->color_buffer(
        ColorBufferKey::from(0x1000, size_register(16, 8), info_register(CbNumberType::Unorm), 0),
        error);
    REQUIRE(surface.has_value());

    gpu::RenderPassDesc pass;
    pass.label = "clear";
    pass.color_count = 1;
    pass.colors[0].texture = surface->texture;
    pass.colors[0].load = gpu::LoadAction::Clear;
    pass.colors[0].store = gpu::StoreAction::Store;
    pass.colors[0].clear = gpu::ClearColor{1.0F, 0.0F, 0.0F, 1.0F};
    REQUIRE(backend->begin_render_pass(pass));
    backend->end_render_pass();
    backend->submit();

    std::vector<u8> pixels(std::size_t{16} * 8 * 4);
    REQUIRE(backend->read_texture(surface->texture, pixels));
    CHECK(pixels[0] == 0xFF); // RGBA: red
    CHECK(pixels[1] == 0x00);
    CHECK(pixels[3] == 0xFF);
}
