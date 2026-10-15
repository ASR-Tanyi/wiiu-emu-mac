// Phase 8.5 — Latte's surface layout.
//
// The pinned values below were checked when this was written against a second,
// independently written implementation of the same addressing (Cemu's): 720
// layouts across eight formats, nine sizes and ten tile modes, and 92 million
// element addresses with bank and pipe swizzles, all identical. Those values
// are what this file holds onto. The properties after them -- every element
// lands somewhere distinct, inside the surface, on an element boundary -- are
// what would break first if the library or the configuration handed to it
// changed.

#include <wiiu/latte/surface.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <set>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

constexpr u32 kRgba8 = 0x01A;
constexpr u32 kBc1 = 0x031;

[[nodiscard]] SurfaceDesc surface(TileMode mode, u32 format, u32 width, u32 height) {
    SurfaceDesc desc;
    desc.tile_mode = mode;
    desc.format = format;
    desc.width = width;
    desc.height = height;
    return desc;
}

} // namespace

TEST_CASE("2D-tiled colour buffers are laid out as the console lays them out", "[latte][surface]") {
    // 720p: no padding at all, so two of them are exactly the 0x708000 that
    // OSScreen's TV buffer and GX2CalcTVSize agree on.
    const auto tv720 = compute_layout(surface(TileMode::Tiled2DThin1, kRgba8, 1280, 720));
    REQUIRE(tv720.has_value());
    CHECK(tv720->tile_mode == TileMode::Tiled2DThin1);
    CHECK(tv720->pitch == 1280);
    CHECK(tv720->height == 720);
    CHECK(tv720->size == 3'686'400);
    CHECK(tv720->size * 2 == 0x708000);
    CHECK(tv720->base_align == 2048);
    CHECK(tv720->pitch_align == 32);
    CHECK(tv720->height_align == 16);

    // 1080p: 1080 rows round up to 1088.
    const auto tv1080 = compute_layout(surface(TileMode::Tiled2DThin1, kRgba8, 1920, 1080));
    REQUIRE(tv1080.has_value());
    CHECK(tv1080->pitch == 1920);
    CHECK(tv1080->height == 1088);
    CHECK(tv1080->size == 8'355'840);

    // The GamePad: 854 pixels round up to a pitch of 864.
    const auto drc = compute_layout(surface(TileMode::Tiled2DThin1, kRgba8, 854, 480));
    REQUIRE(drc.has_value());
    CHECK(drc->pitch == 864);
    CHECK(drc->size == 1'658'880);
}

TEST_CASE("the other modes pad as their tiles need", "[latte][surface]") {
    const auto one_d = compute_layout(surface(TileMode::Tiled1DThin1, kRgba8, 100, 37));
    REQUIRE(one_d.has_value());
    CHECK(one_d->pitch == 104); // 8x8 micro-tiles
    CHECK(one_d->height == 40);
    CHECK(one_d->size == 16'640);
    CHECK(one_d->base_align == 256);

    const auto linear = compute_layout(surface(TileMode::LinearAligned, kRgba8, 100, 37));
    REQUIRE(linear.has_value());
    CHECK(linear->pitch == 128); // 64 elements
    CHECK(linear->height == 37);
    CHECK(linear->size == 18'944);

    // GX2's own linear-special: nothing padded, nothing aligned.
    const auto special = compute_layout(surface(TileMode::LinearSpecial, kRgba8, 100, 37));
    REQUIRE(special.has_value());
    CHECK(special->pitch == 100);
    CHECK(special->size == 100U * 37 * 4);
    CHECK(special->base_align == 1);

    // A compressed format is measured in 4x4 blocks of 64 bits.
    const auto bc1 = compute_layout(surface(TileMode::Tiled2DThin1, kBc1, 256, 256));
    REQUIRE(bc1.has_value());
    CHECK(bc1->bits_per_element == 64);
    CHECK(bc1->pitch == 64);
    CHECK(bc1->height == 64);
    CHECK(bc1->size == 32'768);
}

TEST_CASE("a surface with no element size cannot be laid out", "[latte][surface]") {
    CHECK_FALSE(compute_layout(surface(TileMode::Tiled2DThin1, 0x000, 64, 64)).has_value());
    CHECK_FALSE(compute_layout(surface(TileMode::Tiled2DThin1, 0x081, 64, 64)).has_value());
}

TEST_CASE("elements of a micro-tile go x, x, y, x, y, y", "[latte][surface]") {
    // A 32-bit thin micro-tile: the element index takes x0, x1, y0, x2, y1,
    // y2 from low bit to high, and the micro-tile is 8x8 elements, 256 bytes.
    const SurfaceDesc desc = surface(TileMode::Tiled1DThin1, kRgba8, 1280, 720);
    const auto layout = compute_layout(desc);
    REQUIRE(layout.has_value());
    CHECK(element_offset(desc, *layout, 0, 0) == 0);
    CHECK(element_offset(desc, *layout, 1, 0) == 4);
    CHECK(element_offset(desc, *layout, 2, 0) == 8);
    CHECK(element_offset(desc, *layout, 0, 1) == 16);
    CHECK(element_offset(desc, *layout, 4, 0) == 32);
    CHECK(element_offset(desc, *layout, 0, 2) == 64);
    CHECK(element_offset(desc, *layout, 8, 0) == 256);       // the next micro-tile
    CHECK(element_offset(desc, *layout, 0, 8) == 1280 * 32); // the next row of them
}

TEST_CASE("2D tiling spreads micro-tiles across banks and pipes", "[latte][surface]") {
    const SurfaceDesc desc = surface(TileMode::Tiled2DThin1, kRgba8, 1280, 720);
    const auto layout = compute_layout(desc);
    REQUIRE(layout.has_value());
    CHECK(element_offset(desc, *layout, 0, 0) == 0);
    CHECK(element_offset(desc, *layout, 7, 7) == 252); // inside the first micro-tile
    CHECK(element_offset(desc, *layout, 8, 0) == 768); // the next one is elsewhere
    CHECK(element_offset(desc, *layout, 0, 8) == 256);
    CHECK(element_offset(desc, *layout, 37, 91) == 413'044);
    CHECK(element_offset(desc, *layout, 1279, 719) == 3'686'140);
}

TEST_CASE("every element has its own place inside the surface", "[latte][surface]") {
    struct Case {
        TileMode mode;
        u32 format;
        u32 width;
        u32 height;
        u32 swizzle;
    };

    const Case cases[] = {
        {TileMode::LinearAligned, kRgba8, 100, 37, 0},
        {TileMode::Tiled1DThin1, kRgba8, 100, 37, 0},
        {TileMode::Tiled1DThin1, 0x001, 64, 40, 0}, // 8-bit
        {TileMode::Tiled2DThin1, kRgba8, 256, 128, 0},
        {TileMode::Tiled2DThin1, kRgba8, 256, 128, 0x500},
        {TileMode::Tiled2BThin1, 0x007, 200, 64, 0x300}, // 16-bit, bank-swapped
        {TileMode::Tiled2DThin1, 0x01F, 128, 64, 0x700}, // 64-bit
        {TileMode::Tiled2DThin1, kBc1, 512, 256, 0},
    };

    for (const Case& c : cases) {
        const SurfaceDesc desc = surface(c.mode, c.format, c.width, c.height);
        const auto layout = compute_layout(desc);
        REQUIRE(layout.has_value());
        const BankPipeSwizzle swizzle = split_swizzle(c.swizzle);
        const u32 bytes = layout->bits_per_element / 8;

        std::set<u64> seen;
        bool inside = true;
        bool aligned = true;
        for (u32 y = 0; y < layout->height; ++y) {
            for (u32 x = 0; x < layout->pitch; ++x) {
                const u64 offset =
                    element_offset(desc, *layout, x, y, 0, 0, swizzle.pipe, swizzle.bank);
                seen.insert(offset);
                inside = inside && offset + bytes <= layout->size;
                aligned = aligned && offset % bytes == 0;
            }
        }
        INFO(tile_mode_name(c.mode) << " " << c.width << "x" << c.height << " format " << c.format
                                    << " swizzle " << c.swizzle);
        CHECK(seen.size() == static_cast<std::size_t>(layout->pitch) * layout->height);
        CHECK(inside);
        CHECK(aligned);
    }
}

TEST_CASE("a GX2 swizzle splits into a pipe and a bank", "[latte][surface]") {
    // Two pipes and four banks: the 256-byte base's low bit picks the pipe,
    // the next two the bank.
    for (u32 base = 0; base < 8; ++base) {
        const BankPipeSwizzle split = split_swizzle(base << 8);
        CHECK(split.pipe == (base & 1));
        CHECK(split.bank == ((base >> 1) & 3));
    }
    CHECK(split_swizzle(0x000D0000).pipe == 0); // bits outside the second byte do nothing
}

TEST_CASE("the detiler puts every element where element_offset says it is", "[latte][surface]") {
    // The detiler's fast path places whole micro-tiles by one tile's
    // pattern; this holds it, element by element, to the slow way. The level
    // is filled with noise, so an element read from the wrong place shows.
    struct Case {
        TileMode mode;
        u32 format;
        u32 width;
        u32 height;
        u32 slices;
        u32 slice;
        u32 swizzle;
    };

    const Case cases[] = {
        {TileMode::LinearAligned, kRgba8, 100, 37, 1, 0, 0},
        {TileMode::LinearSpecial, kRgba8, 5, 3, 1, 0, 0},
        {TileMode::Tiled1DThin1, kRgba8, 100, 37, 1, 0, 0},
        {TileMode::Tiled1DThin1, 0x001, 64, 40, 3, 2, 0},
        {TileMode::Tiled2DThin1, kRgba8, 1280, 720, 1, 0, 0},
        {TileMode::Tiled2DThin1, kRgba8, 300, 200, 1, 0, 0x500},
        {TileMode::Tiled2DThin1, kRgba8, 256, 128, 4, 3, 0x700},
        {TileMode::Tiled2DThin1, 0x007, 200, 64, 1, 0, 0x300},
        {TileMode::Tiled2DThin1, 0x001, 130, 70, 1, 0, 0x100},
        {TileMode::Tiled2BThin1, kRgba8, 512, 64, 1, 0, 0x300},
        {TileMode::Tiled2DThin2, kRgba8, 256, 256, 1, 0, 0x200},
        {TileMode::Tiled3DThin1, kRgba8, 256, 128, 2, 1, 0x400},
        {TileMode::Tiled2DThin1, kBc1, 512, 256, 1, 0, 0},
        // Past the fast path: 64-bit elements split across pipes, and thick
        // modes interleave slices. These take the slow way.
        {TileMode::Tiled2DThin1, 0x01F, 128, 64, 1, 0, 0x700},
        {TileMode::Tiled2DThick, kRgba8, 64, 64, 8, 5, 0},
        {TileMode::Tiled1DThick, kRgba8, 32, 32, 4, 1, 0},
    };

    for (const Case& c : cases) {
        SurfaceDesc desc = surface(c.mode, c.format, c.width, c.height);
        desc.slices = c.slices;
        const auto layout = compute_layout(desc);
        REQUIRE(layout.has_value());
        const BankPipeSwizzle swizzle = split_swizzle(c.swizzle);
        const u32 bytes = layout->bits_per_element / 8;

        std::vector<u8> level(layout->size);
        u32 noise = 0x12345678;
        for (u8& byte : level) {
            noise = (noise * 1664525U) + 1013904223U;
            byte = static_cast<u8>(noise >> 24);
        }

        const Extent extent = level_extent(desc);
        std::vector<u8> rows(static_cast<std::size_t>(extent.width) * extent.height * bytes);
        REQUIRE(detile(desc, *layout, level, c.slice, 0, swizzle, rows));

        u64 wrong = 0;
        for (u32 y = 0; y < extent.height; ++y) {
            for (u32 x = 0; x < extent.width; ++x) {
                const u64 offset =
                    element_offset(desc, *layout, x, y, c.slice, 0, swizzle.pipe, swizzle.bank);
                const u64 at = ((static_cast<u64>(y) * extent.width) + x) * bytes;
                if (std::memcmp(rows.data() + at, level.data() + offset, bytes) != 0) {
                    ++wrong;
                }
            }
        }
        INFO(tile_mode_name(c.mode) << " " << c.width << "x" << c.height << " format " << c.format
                                    << " slice " << c.slice << " swizzle " << c.swizzle);
        CHECK(wrong == 0);
    }
}

TEST_CASE("the detiler refuses what it cannot fit", "[latte][surface]") {
    const SurfaceDesc desc = surface(TileMode::Tiled2DThin1, kRgba8, 64, 64);
    const auto layout = compute_layout(desc);
    REQUIRE(layout.has_value());
    const std::vector<u8> level(layout->size);

    std::vector<u8> small(64 * 63 * 4);
    CHECK_FALSE(detile(desc, *layout, level, 0, 0, {}, small));

    std::vector<u8> rows(64 * 64 * 4);
    const std::vector<u8> short_level(layout->size / 2);
    CHECK_FALSE(detile(desc, *layout, short_level, 0, 0, {}, rows));
    CHECK(detile(desc, *layout, level, 0, 0, {}, rows));
}

TEST_CASE("a level's extent is in elements, before padding", "[latte][surface]") {
    SurfaceDesc desc = surface(TileMode::Tiled2DThin1, kRgba8, 100, 37);
    CHECK(level_extent(desc).width == 100);
    CHECK(level_extent(desc).height == 37);
    desc.mip_level = 3;
    CHECK(level_extent(desc).width == 12);
    CHECK(level_extent(desc).height == 4);
    desc.mip_level = 9;
    CHECK(level_extent(desc).width == 1);

    const SurfaceDesc bc1 = surface(TileMode::Tiled2DThin1, kBc1, 30, 9);
    CHECK(level_extent(bc1).width == 8); // 4x4 blocks, rounded up
    CHECK(level_extent(bc1).height == 3);
}
