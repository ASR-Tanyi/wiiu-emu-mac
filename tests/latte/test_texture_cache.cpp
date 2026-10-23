// Phase 8.13, textures' first form — a texture resource's image as a texture.
//
// What is guarded: what a resource is read as -- the Metal format its data
// format and WORD4 pick, its size, and where a macro-tiled level really is
// once the swizzle in its base register is taken out -- and what is refused
// by name: invalid resources, the dimensions, formats and number types not
// uploaded yet, and a pitch the layout does not agree with. And the upload:
// an image written to guest memory in its tiled layout, with a swizzle,
// comes out as plain rows in a texture of the right format; it is made once
// and found after, made again -- the old one destroyed -- when the bytes
// change, and a second resource is a second texture.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/latte/surface.hpp>
#include <wiiu/latte/texture_cache.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

constexpr ppcaddr_t kImage = 0x10100000;

/// The resource GX2 would write for a `width` x `height` 2D texture of
/// `data_format` in `tile_mode`, laid out as the layout says, its level 0 at
/// physical `address` with GX2's swizzle word `swizzle` XORed in.
[[nodiscard]] TextureResource make_resource(u32 width,
                                            u32 height,
                                            u32 data_format,
                                            TileMode tile_mode,
                                            physaddr_t address,
                                            u32 swizzle = 0) {
    SurfaceDesc desc;
    desc.tile_mode = tile_mode;
    desc.format = data_format;
    desc.width = width;
    desc.height = height;
    const std::optional<SurfaceLayout> layout = compute_layout(desc);
    REQUIRE(layout.has_value());

    TextureResource out;
    out.word0.raw = 0;
    out.word0.dim = TexDim::D2;
    out.word0.tile_mode = static_cast<u32>(tile_mode);
    out.word0.pitch = (layout->pitch / 8) - 1;
    out.word0.width = width - 1;
    out.word1.raw = 0;
    out.word1.height = height - 1;
    out.word1.data_format = data_format;
    out.base = address ^ swizzle;
    out.mip = out.base;
    out.word4.raw = 0;
    out.word4.request_size = 2;
    out.word5.raw = 0;
    out.word6.raw = 0;
    out.word6.type = ResourceType::ValidTexture;
    return out;
}

struct TextureFixture {
    memory::Memory memory;
    gpu::NullBackend backend;
    std::unique_ptr<TextureCache> cache;

    TextureFixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        REQUIRE(backend.initialize(error));
        cache = std::make_unique<TextureCache>(backend, memory);
    }

    [[nodiscard]] physaddr_t physical(ppcaddr_t address) const {
        const memory::Region* const region = memory.region_of(address);
        REQUIRE(region != nullptr);
        physaddr_t out = 0;
        REQUIRE(memory::to_physical(*region, address, out));
        return out;
    }

    /// Write pixel (x, y) = (x, y, x ^ y, 0xA5) of a `width` x `height`
    /// RGBA8 image into guest memory where the tiled layout puts it, with
    /// GX2's swizzle word `swizzle`.
    void put_image(u32 width, u32 height, TileMode tile_mode, u32 swizzle, u8 salt = 0) {
        SurfaceDesc desc;
        desc.tile_mode = tile_mode;
        desc.format = 0x1A;
        desc.width = width;
        desc.height = height;
        const std::optional<SurfaceLayout> layout = compute_layout(desc);
        REQUIRE(layout.has_value());
        const BankPipeSwizzle split = split_swizzle(swizzle);
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u64 at = element_offset(desc, *layout, x, y, 0, 0, split.pipe, split.bank);
                const std::array<u8, 4> pixel{
                    static_cast<u8>(x), static_cast<u8>(y), static_cast<u8>((x ^ y) + salt), 0xA5};
                REQUIRE(memory.write_bytes(kImage + static_cast<u32>(at), pixel));
            }
        }
    }
};

} // namespace

TEST_CASE("a texture resource is read as the layout and Metal take it", "[latte][texture]") {
    std::string error;
    const physaddr_t address = 0x50100000;
    TextureResource rgba = make_resource(64, 32, 0x1A, TileMode::Tiled2DThin1, address, 0x300);
    std::optional<TextureImage> image = texture_image(rgba, error);
    INFO(error);
    REQUIRE(image.has_value());
    CHECK(image->format == gpu::TextureFormat::RGBA8Unorm);
    CHECK(image->width == 64);
    CHECK(image->height == 32);
    CHECK(image->bytes_per_element == 4);
    // The swizzle comes out of the base: the level is where GX2 put it.
    CHECK(image->address == address);
    CHECK(image->swizzle.pipe == split_swizzle(0x300).pipe);
    CHECK(image->swizzle.bank == split_swizzle(0x300).bank);

    // sRGB, and the one- and two-channel formats.
    const auto format = [&error](const TextureResource& r) {
        const std::optional<TextureImage> read = texture_image(r, error);
        return read.has_value() ? read->format : gpu::TextureFormat::Unknown;
    };
    rgba.word4.force_degamma = true;
    CHECK(format(rgba) == gpu::TextureFormat::RGBA8UnormSrgb);
    CHECK(format(make_resource(64, 64, 0x01, TileMode::Tiled2DThin1, address)) ==
          gpu::TextureFormat::R8Unorm);
    CHECK(format(make_resource(64, 64, 0x07, TileMode::Tiled2DThin1, address)) ==
          gpu::TextureFormat::RG8Unorm);

    // A linear or 1D level has no swizzle in its base.
    const std::optional<TextureImage> linear =
        texture_image(make_resource(64, 8, 0x1A, TileMode::LinearAligned, address + 0x300), error);
    REQUIRE(linear.has_value());
    CHECK(linear->address == address + 0x300);
    CHECK(linear->swizzle.pipe == 0);
    CHECK(linear->swizzle.bank == 0);
}

TEST_CASE("what is not uploaded yet is refused by name", "[latte][texture]") {
    std::string error;
    const TextureResource good = make_resource(64, 64, 0x1A, TileMode::Tiled2DThin1, 0x50100000);

    TextureResource invalid = good;
    invalid.word6.type = ResourceType::ValidBuffer;
    CHECK_FALSE(texture_image(invalid, error).has_value());
    CHECK(error.find("not a valid texture") != std::string::npos);

    TextureResource cube = good;
    cube.word0.dim = TexDim::Cube;
    CHECK_FALSE(texture_image(cube, error).has_value());
    CHECK(error == "a cube texture is not uploaded yet");

    TextureResource wide = good;
    wide.word1.data_format = 0x22; // 32_32_32_32
    CHECK_FALSE(texture_image(wide, error).has_value());
    CHECK(error.find("format 0x22") != std::string::npos);

    TextureResource snorm = good;
    snorm.word4.format_comp_x = FormatComp::Signed;
    CHECK_FALSE(texture_image(snorm, error).has_value());

    TextureResource integer = good;
    integer.word4.num_format_all = NumFormat::Int;
    CHECK_FALSE(texture_image(integer, error).has_value());

    TextureResource swapped = good;
    swapped.word4.endian_swap = Endian::Swap8In32;
    CHECK_FALSE(texture_image(swapped, error).has_value());
    CHECK(error.find("byte-swapped") != std::string::npos);

    // A pitch the layout does not agree with: misread, not drawn sheared.
    TextureResource sheared = good;
    sheared.word0.pitch = sheared.word0.pitch + 8;
    CHECK_FALSE(texture_image(sheared, error).has_value());
    CHECK(error.find("pixels to a row") != std::string::npos);
}

TEST_CASE_METHOD(TextureFixture,
                 "a tiled, swizzled image comes out as plain rows, once",
                 "[latte][texture]") {
    constexpr u32 kSwizzle = 0x300;
    put_image(64, 64, TileMode::Tiled2DThin1, kSwizzle);
    const TextureResource tiled =
        make_resource(64, 64, 0x1A, TileMode::Tiled2DThin1, physical(kImage), kSwizzle);

    std::string error;
    const std::optional<CachedTexture> texture = cache->texture(tiled, error);
    INFO(error);
    REQUIRE(texture.has_value());
    CHECK(texture->format == gpu::TextureFormat::RGBA8Unorm);
    CHECK(texture->width == 64);
    CHECK(texture->height == 64);

    std::vector<u8> pixels(std::size_t{64} * 64 * 4);
    REQUIRE(backend.read_texture(texture->texture, pixels));
    u32 wrong = 0;
    for (u32 y = 0; y < 64; ++y) {
        for (u32 x = 0; x < 64; ++x) {
            const std::size_t at = ((std::size_t{y} * 64) + x) * 4;
            if (pixels[at] != x || pixels[at + 1] != y || pixels[at + 2] != (x ^ y) ||
                pixels[at + 3] != 0xA5) {
                ++wrong;
            }
        }
    }
    CHECK(wrong == 0);

    // Found again while the bytes are the same.
    const std::optional<CachedTexture> again = cache->texture(tiled, error);
    REQUIRE(again.has_value());
    CHECK(again->texture == texture->texture);
    CHECK(cache->stats().uploads == 1);
    CHECK(cache->stats().hits == 1);

    // Rewritten: a new texture, the old one gone.
    put_image(64, 64, TileMode::Tiled2DThin1, kSwizzle, 1);
    const std::optional<CachedTexture> rewritten = cache->texture(tiled, error);
    REQUIRE(rewritten.has_value());
    CHECK(rewritten->texture != texture->texture);
    CHECK(cache->stats().uploads == 2);
    CHECK(backend.live_count(gpu::ResourceKind::Texture) == 1);
    REQUIRE(backend.read_texture(rewritten->texture, pixels));
    CHECK(pixels[(((std::size_t{3} * 64) + 5) * 4) + 2] == ((5 ^ 3) + 1));

    // The same memory read as sRGB is another texture.
    TextureResource srgb = tiled;
    srgb.word4.force_degamma = true;
    const std::optional<CachedTexture> other = cache->texture(srgb, error);
    REQUIRE(other.has_value());
    CHECK(other->texture != rewritten->texture);
    CHECK(other->format == gpu::TextureFormat::RGBA8UnormSrgb);
    CHECK(cache->size() == 2);

    cache->clear();
    CHECK(cache->size() == 0);
    CHECK(backend.live_count(gpu::ResourceKind::Texture) == 0);
    CHECK(backend.rejected() == 0);
}

TEST_CASE_METHOD(TextureFixture,
                 "a texture where no memory is, or that is refused, makes nothing",
                 "[latte][texture]") {
    std::string error;
    // Physical memory nothing maps.
    CHECK_FALSE(
        cache->texture(make_resource(64, 64, 0x1A, TileMode::Tiled2DThin1, 0x0E000000), error)
            .has_value());
    CHECK(error.find("no memory the CPU maps") != std::string::npos);

    TextureResource cube = make_resource(64, 64, 0x1A, TileMode::Tiled2DThin1, physical(kImage));
    cube.word0.dim = TexDim::Cube;
    CHECK_FALSE(cache->texture(cube, error).has_value());
    CHECK(cache->size() == 0);
    CHECK(backend.live_count(gpu::ResourceKind::Texture) == 0);
}
