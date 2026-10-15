// Latte's depth-buffer registers: the fields GX2InitDepthBufferRegs packs,
// at the positions Linux's r600d.h gives them.

#include <wiiu/latte/depth_buffer.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace wiiu;
using namespace wiiu::latte;

TEST_CASE("the depth formats are the five GX2 accepts", "[latte][depth]") {
    CHECK(depth_format(0x005) == DbFormat::Depth16);
    CHECK(depth_format(0x011) == DbFormat::Depth8_24);
    CHECK(depth_format(0x80E) == DbFormat::Depth32Float);
    CHECK(depth_format(0x811) == DbFormat::Depth8_24Float);
    CHECK(depth_format(0x81C) == DbFormat::DepthX24_8_32Float);
    CHECK_FALSE(depth_format(0x01A).has_value());
}

TEST_CASE("DB_DEPTH_INFO's fields sit where r600d.h puts them", "[latte][depth]") {
    // S_028010_FORMAT << 0, READ_SIZE << 3, ARRAY_MODE << 15,
    // TILE_SURFACE_ENABLE << 25, TILE_COMPACT << 26, ZRANGE_PRECISION << 31.
    DbDepthInfo info{.raw = 0};
    info.format = DbFormat::Depth32Float;
    CHECK(info.raw == 6);
    info.read_size_512 = true;
    CHECK(info.raw == (6U | (1U << 3)));
    info.array_mode = TileMode::Tiled2DThin1;
    CHECK(info.raw == (6U | (1U << 3) | (4U << 15)));
    info.tile_surface_enable = true;
    info.tile_compact = true;
    info.zrange_precision = true;
    CHECK(info.raw == (6U | (1U << 3) | (4U << 15) | (1U << 25) | (1U << 26) | (1U << 31)));

    // S_028000: pitch in 10 bits, slice in the 20 above. S_028004: slice
    // start in 11 bits, the last slice from bit 13.
    DbDepthSize size{.raw = 0};
    size.pitch_tile_max = 0x3FF;
    size.slice_tile_max = 0xFFFFF;
    CHECK(size.raw == 0x3FFFFFFF);
    DbDepthView view{.raw = 0};
    view.slice_start = 2;
    view.slice_max = 4;
    CHECK(view.raw == (2U | (4U << 13)));
}
