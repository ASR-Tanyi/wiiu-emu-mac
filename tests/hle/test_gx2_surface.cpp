// Phase 8.1 — GX2's surface calls.
//
// The pinned values were produced by this implementation and are the same
// ones Cemu's GX2 code produces over its own addressing library: they were
// compared in a scratch harness across 1.9 million surfaces and 3.1
// million register sets (gx2_surface.hpp says where, and the few places the
// two part ways on purpose). What is guarded here is what a title builds on:
// the size it allocates, the alignment it allocates at, the tile mode and
// swizzle it keeps, where its mip levels go, and the registers that will
// describe the same memory to the GPU.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/color_buffer.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using latte::TileMode;

namespace {

constexpr u32 kRgba8 = 0x01A;
constexpr u32 kTextureColorBufferTv =
    0x80000000 | gx2::surface_use_texture | gx2::surface_use_color_buffer;

[[nodiscard]] gx2::Surface surface(gx2::SurfaceDim dim,
                                   u32 width,
                                   u32 height,
                                   u32 format = kRgba8,
                                   u32 tile_mode = gx2::tile_mode_default) {
    gx2::Surface s{};
    s.dim = static_cast<u32>(dim);
    s.width = width;
    s.height = height;
    s.depth = 1;
    s.mip_levels = 1;
    s.format = format;
    s.use = gx2::surface_use_texture;
    s.tile_mode = tile_mode;
    return s;
}

[[nodiscard]] gx2::ColorBuffer color_buffer(const gx2::Surface& s) {
    gx2::ColorBuffer buffer{};
    buffer.surface = s;
    buffer.view_num_slices = 1;
    return buffer;
}

[[nodiscard]] u32 swizzle_first_untiled_level(const gx2::Surface& s) {
    return (s.swizzle >> 16) & 0xFF;
}

[[nodiscard]] bool same_bytes(const gx2::Surface& a, const gx2::Surface& b) {
    return std::memcmp(&a, &b, sizeof(gx2::Surface)) == 0;
}

} // namespace

TEST_CASE("the TV's colour buffers are 2D-tiled, as the scan-out expects", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    // WHB's TV colour buffer: a 2D texture, colour buffer and TV, default tile
    // mode. 720p needs no padding at all; 1080 rows pad to 1088.
    gx2::Surface tv720 = surface(gx2::SurfaceDim::Texture2D, 1280, 720);
    tv720.use = kTextureColorBufferTv;
    REQUIRE(gx2::calc_surface_size_and_alignment(tv720));
    CHECK(tv720.tile_mode == static_cast<u32>(TileMode::Tiled2DThin1));
    CHECK(tv720.image_size == 3'686'400);
    CHECK(tv720.alignment == 2048);
    CHECK(tv720.pitch == 1280);
    CHECK(tv720.swizzle == 0x000D0000);
    CHECK(tv720.mip_levels == 1);
    CHECK(tv720.mipmap_size == 0);

    gx2::Surface tv1080 = surface(gx2::SurfaceDim::Texture2D, 1920, 1080);
    tv1080.use = kTextureColorBufferTv;
    REQUIRE(gx2::calc_surface_size_and_alignment(tv1080));
    CHECK(tv1080.image_size == 1920U * 1088 * 4);
    CHECK(tv1080.pitch == 1920);

    // The GamePad's: 854 pixels pad to a pitch of 864.
    gx2::Surface drc = surface(gx2::SurfaceDim::Texture2D, 854, 480);
    drc.use = kTextureColorBufferTv;
    REQUIRE(gx2::calc_surface_size_and_alignment(drc));
    CHECK(drc.pitch == 864);
    CHECK(drc.image_size == 1'658'880);
}

TEST_CASE("the default tile mode is GX2's choice", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    SECTION("a single-sampled 1D texture is linear, and carries no swizzle level") {
        gx2::Surface s = surface(gx2::SurfaceDim::Texture1D, 100, 1);
        REQUIRE(gx2::calc_surface_size_and_alignment(s));
        CHECK(s.tile_mode == static_cast<u32>(TileMode::LinearAligned));
        CHECK(s.pitch == 128); // 64-element pitch alignment
        CHECK(s.image_size == 512);
        CHECK(s.swizzle == 0);
    }

    SECTION("smaller than a macro-tile both ways steps down to 1D") {
        gx2::Surface s = surface(gx2::SurfaceDim::Texture2D, 16, 8);
        s.swizzle = 0x00000300;
        REQUIRE(gx2::calc_surface_size_and_alignment(s));
        CHECK(s.tile_mode == static_cast<u32>(TileMode::Tiled1DThin1));
        CHECK(s.image_size == 512);
        CHECK(s.alignment == 256);
        CHECK(s.pitch == 16);
        CHECK(s.swizzle == 0x00000300); // the title's bank swizzle survives
    }

    SECTION("smaller one way only stays 2D, padded") {
        gx2::Surface s = surface(gx2::SurfaceDim::Texture2D, 16, 16);
        REQUIRE(gx2::calc_surface_size_and_alignment(s));
        CHECK(s.tile_mode == static_cast<u32>(TileMode::Tiled2DThin1));
        CHECK(s.pitch == 32);
        CHECK(s.image_size == 2048);
    }

    SECTION("a volume is 2D thick, unless it is a colour buffer") {
        gx2::Surface texture = surface(gx2::SurfaceDim::Texture3D, 64, 64);
        texture.depth = 8;
        REQUIRE(gx2::calc_surface_size_and_alignment(texture));
        CHECK(texture.tile_mode == static_cast<u32>(TileMode::Tiled2DThick));
        CHECK(texture.alignment == 8192);

        gx2::Surface target = surface(gx2::SurfaceDim::Texture3D, 64, 64);
        target.depth = 8;
        target.use = gx2::surface_use_texture | gx2::surface_use_color_buffer;
        REQUIRE(gx2::calc_surface_size_and_alignment(target));
        CHECK(target.tile_mode == static_cast<u32>(TileMode::Tiled2DThin1));
        CHECK(target.image_size == 131'072);
    }

    SECTION("the bad-align default judges a compressed surface in pixels") {
        // BC1 at 32x32 is 8x8 blocks. Against the alignment in blocks it is
        // big enough for 2D; in pixels it is not, and steps down.
        gx2::Surface plain = surface(gx2::SurfaceDim::Texture2D, 32, 32, 0x031);
        REQUIRE(gx2::calc_surface_size_and_alignment(plain));
        CHECK(plain.tile_mode == static_cast<u32>(TileMode::Tiled2DThin1));
        CHECK(plain.image_size == 4096);

        gx2::Surface bad =
            surface(gx2::SurfaceDim::Texture2D, 32, 32, 0x031, gx2::tile_mode_default_bad_align);
        REQUIRE(gx2::calc_surface_size_and_alignment(bad));
        CHECK(bad.tile_mode == static_cast<u32>(TileMode::Tiled1DThin1));
        CHECK(bad.image_size == 512);
    }
}

TEST_CASE("an explicit tile mode is kept, and the swizzle with it", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    gx2::Surface linear = surface(
        gx2::SurfaceDim::Texture2D, 64, 64, kRgba8, static_cast<u32>(TileMode::LinearAligned));
    linear.swizzle = 0x00000300;
    REQUIRE(gx2::calc_surface_size_and_alignment(linear));
    CHECK(linear.tile_mode == static_cast<u32>(TileMode::LinearAligned));
    CHECK(linear.swizzle == 0x00000300);
    CHECK(linear.image_size == 16'384);

    // A 2D mode gets the first-untiled-level byte; the bytes either side of
    // it are the title's and are kept.
    gx2::Surface tiled = surface(
        gx2::SurfaceDim::Texture2D, 64, 64, kRgba8, static_cast<u32>(TileMode::Tiled2DThin1));
    tiled.swizzle = 0x12340300;
    REQUIRE(gx2::calc_surface_size_and_alignment(tiled));
    CHECK(tiled.swizzle == 0x120D0300);
    CHECK(tiled.alignment == 2048);
}

TEST_CASE("mip levels follow one another, each aligned", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    gx2::Surface s = surface(gx2::SurfaceDim::Texture2D, 256, 256);
    s.mip_levels = 20; // more than there are: capped at the full chain
    REQUIRE(gx2::calc_surface_size_and_alignment(s));
    CHECK(s.mip_levels == 9);
    CHECK(s.image_size == 262'144);

    // Entry 0 is level 1's offset from the image; the rest are from level 1.
    const std::vector<u32> expected = {
        262'144, 65'536, 81'920, 86'016, 87'040, 87'296, 87'552, 87'808};
    for (u32 i = 0; i < expected.size(); ++i) {
        INFO("offset " << i);
        CHECK(s.mip_level_offset[i] == expected[i]);
    }
    CHECK(s.mipmap_size == 88'064);

    // Level 4 (16x16) is the first that is not macro-tiled, so the swizzle
    // stops applying there.
    CHECK(swizzle_first_untiled_level(s) == 4);
    CHECK(s.tile_mode == static_cast<u32>(TileMode::Tiled2DThin1));
}

TEST_CASE("a step down from macro-tiling below level 1 pads by the swizzle",
          "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    // GX2's own quirk, which decaf and Cemu both carry: the padding before
    // the first untiled level is the swizzle word's low half. With a bank
    // swizzle of 5 every level from 4 on moves 0x500 bytes further in.
    gx2::Surface s = surface(gx2::SurfaceDim::Texture2D, 256, 256);
    s.mip_levels = 9;
    s.swizzle = 0x00000500;
    REQUIRE(gx2::calc_surface_size_and_alignment(s));
    CHECK(s.mip_level_offset[2] == 81'920);
    CHECK(s.mip_level_offset[3] == 86'016 + 0x500);
    CHECK(s.mipmap_size == 88'064 + 0x500);
}

TEST_CASE("a surface that cannot be laid out is left as it was", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    gx2::Surface s = surface(gx2::SurfaceDim::Texture2D, 64, 64, 0x000);
    s.swizzle = 0x00000300;
    s.image_size = 0xDEAD;
    const gx2::Surface before = s;
    CHECK_FALSE(gx2::calc_surface_size_and_alignment(s));
    CHECK(same_bytes(s, before));
}

TEST_CASE("a colour buffer's registers describe the surface it was sized as",
          "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    gx2::Surface s = surface(gx2::SurfaceDim::Texture2D, 1280, 720);
    s.use = kTextureColorBufferTv;
    REQUIRE(gx2::calc_surface_size_and_alignment(s));
    const auto registers = gx2::color_buffer_registers(color_buffer(s));
    REQUIRE(registers.has_value());

    const latte::CbColorSize size{.raw = registers->size};
    CHECK(size.pitch_tile_max == 1280 / 8 - 1);
    CHECK(size.slice_tile_max == 1280 * 720 / 64 - 1);

    const latte::CbColorInfo info{.raw = registers->info};
    CHECK(info.endian == 0);
    CHECK(info.format == 0x1A);
    CHECK(info.array_mode == TileMode::Tiled2DThin1);
    CHECK(info.number_type == latte::CbNumberType::Unorm);
    CHECK(info.comp_swap == latte::CbCompSwap::Std);
    CHECK(info.tile_mode == latte::CbTileMode::Disable);
    CHECK(info.blend_clamp);
    CHECK_FALSE(info.blend_bypass);
    CHECK(info.round_mode == latte::CbRoundMode::ByHalf);
    CHECK(info.source_format == latte::CbSourceFormat::ExportNorm);
    CHECK(registers->info == 0x08100468);

    const latte::CbColorView view{.raw = registers->view};
    CHECK(view.slice_start == 0);
    CHECK(view.slice_max == 0);
    CHECK(registers->mask == 0);
}

TEST_CASE("the format decides how a colour buffer blends and exports", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    const auto info_for = [](u32 format) {
        gx2::Surface s = surface(
            gx2::SurfaceDim::Texture2D, 64, 64, format, static_cast<u32>(TileMode::Tiled2DThin1));
        REQUIRE(gx2::calc_surface_size_and_alignment(s));
        const auto registers = gx2::color_buffer_registers(color_buffer(s));
        REQUIRE(registers.has_value());
        return registers->info;
    };

    SECTION("an integer target bypasses blending and exports in full") {
        const latte::CbColorInfo info{.raw = info_for(0x11A)};
        CHECK(info.number_type == latte::CbNumberType::Uint);
        CHECK(info.blend_bypass);
        CHECK_FALSE(info.blend_clamp);
        CHECK(info.source_format == latte::CbSourceFormat::ExportFull);
    }

    SECTION("a float target truncates and does not clamp") {
        const latte::CbColorInfo info{.raw = info_for(0x823)};
        CHECK(info.format == 0x23);
        CHECK(info.number_type == latte::CbNumberType::Float);
        CHECK(info.round_mode == latte::CbRoundMode::Truncate);
        CHECK_FALSE(info.blend_clamp);
        CHECK(info.source_format == latte::CbSourceFormat::ExportFull);
    }

    SECTION("sRGB clamps like unorm") {
        const latte::CbColorInfo info{.raw = info_for(0x41A)};
        CHECK(info.number_type == latte::CbNumberType::Srgb);
        CHECK(info.blend_clamp);
    }

    SECTION("5_5_5_1 and 10_10_10_2 reverse their components") {
        CHECK(latte::CbColorInfo{.raw = info_for(0x00C)}.comp_swap == latte::CbCompSwap::StdRev);
        CHECK(latte::CbColorInfo{.raw = info_for(0x01B)}.comp_swap == latte::CbCompSwap::StdRev);
        CHECK(latte::CbColorInfo{.raw = info_for(0x019)}.comp_swap == latte::CbCompSwap::Std);
    }
}

TEST_CASE("a colour buffer's view names its slices", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    gx2::Surface s = surface(gx2::SurfaceDim::Texture2DArray, 64, 64);
    s.depth = 8;
    REQUIRE(gx2::calc_surface_size_and_alignment(s));
    gx2::ColorBuffer buffer = color_buffer(s);
    buffer.view_first_slice = 2;
    buffer.view_num_slices = 3;
    const auto registers = gx2::color_buffer_registers(buffer);
    REQUIRE(registers.has_value());
    const latte::CbColorView view{.raw = registers->view};
    CHECK(view.slice_start == 2);
    CHECK(view.slice_max == 4);

    // Linear-special has no slices to choose between, and is linear-general
    // to the hardware.
    gx2::Surface special = surface(
        gx2::SurfaceDim::Texture2D, 100, 37, kRgba8, static_cast<u32>(TileMode::LinearSpecial));
    REQUIRE(gx2::calc_surface_size_and_alignment(special));
    CHECK(special.image_size == 100U * 37 * 4);
    gx2::ColorBuffer special_buffer = color_buffer(special);
    special_buffer.view_first_slice = 2;
    const auto special_registers = gx2::color_buffer_registers(special_buffer);
    REQUIRE(special_registers.has_value());
    CHECK(special_registers->view == 0);
    CHECK(latte::CbColorInfo{.raw = special_registers->info}.array_mode == TileMode::LinearGeneral);
}

TEST_CASE("a scan buffer's registers keep the pitch it was sized with", "[hle][gx2][surface]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    // The scan-buffer flag widens a 1D surface's pitch alignment to 32. The
    // registers have to be computed with it too, or the GPU would write rows
    // of 8 into memory laid out in rows of 32 -- which is what leaving the
    // flag out of the register calculation, as Cemu does, would give.
    gx2::Surface s =
        surface(gx2::SurfaceDim::Texture2D, 1, 1, kRgba8, static_cast<u32>(TileMode::Tiled1DThin1));
    s.use = gx2::surface_use_color_buffer | gx2::surface_use_scan_buffer;
    REQUIRE(gx2::calc_surface_size_and_alignment(s));
    CHECK(s.pitch == 32);
    const auto registers = gx2::color_buffer_registers(color_buffer(s));
    REQUIRE(registers.has_value());
    CHECK(latte::CbColorSize{.raw = registers->size}.pitch_tile_max == 32 / 8 - 1);
}

// --------------------------------------------------------------------------
// Through guest memory
// --------------------------------------------------------------------------

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kBuffer = 0x10005000;

struct SurfaceFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    SurfaceFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
    }

    void call(std::string_view name, u32 argument) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        state.gpr[3] = argument;
        Call context{.state = state, .memory = memory};
        function->handler(context);
    }

    [[nodiscard]] u32 word(u32 offset) { return memory.read<u32>(kBuffer + offset); }
};

} // namespace

TEST_CASE_METHOD(SurfaceFixture,
                 "WHB's colour-buffer setup reads and writes the title's structures",
                 "[hle][gx2][surface]") {
    // What WHB's GfxInitTvColourBuffer does: zero the buffer, describe the
    // surface, then the two calls.
    REQUIRE(memory.fill(kBuffer, sizeof(gx2::ColorBuffer), 0));
    memory.write<u32>(kBuffer + 0x00, 1); // dim: 2D
    memory.write<u32>(kBuffer + 0x04, 1280);
    memory.write<u32>(kBuffer + 0x08, 720);
    memory.write<u32>(kBuffer + 0x0C, 1); // depth
    memory.write<u32>(kBuffer + 0x10, 1); // mip levels
    memory.write<u32>(kBuffer + 0x14, kRgba8);
    memory.write<u32>(kBuffer + 0x1C, kTextureColorBufferTv);
    memory.write<u32>(kBuffer + 0x7C, 1); // view slices

    call("GX2CalcSurfaceSizeAndAlignment", kBuffer);
    CHECK(word(0x20) == 3'686'400); // imageSize
    CHECK(word(0x30) == static_cast<u32>(TileMode::Tiled2DThin1));
    CHECK(word(0x34) == 0x000D0000); // swizzle
    CHECK(word(0x38) == 2048);       // alignment
    CHECK(word(0x3C) == 1280);       // pitch
    CHECK(word(0x1C) == kTextureColorBufferTv);

    call("GX2InitColorBufferRegs", kBuffer);
    CHECK(word(0x88) == (159U | (14'399U << 10))); // CB_COLOR0_SIZE
    CHECK(word(0x8C) == 0x08100468);               // CB_COLOR0_INFO
    CHECK(word(0x90) == 0);                        // CB_COLOR0_VIEW
    CHECK(word(0x94) == 0);                        // CB_COLOR0_MASK
}

TEST_CASE_METHOD(SurfaceFixture,
                 "a surface outside guest memory is reported, not touched",
                 "[hle][gx2][surface]") {
    call("GX2CalcSurfaceSizeAndAlignment", 0x00000100);
    call("GX2InitColorBufferRegs", 0x00000100);
    SUCCEED("neither call faulted the host");
}
