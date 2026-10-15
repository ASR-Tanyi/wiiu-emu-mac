// Latte's surface formats: the element sizes the scan buffers (8.1) and, later,
// every surface (8.5) are measured in.

#include <wiiu/latte/formats.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace wiiu;
using namespace wiiu::latte;

TEST_CASE("a surface format's number type does not change its size", "[latte][formats]") {
    // wut's GX2SurfaceFormat values: the same bits as unorm, uint, snorm,
    // sint, sRGB and float.
    CHECK(bytes_per_element(0x01A) == 4); // UNORM_R8_G8_B8_A8
    CHECK(bytes_per_element(0x11A) == 4); // UINT_R8_G8_B8_A8
    CHECK(bytes_per_element(0x21A) == 4); // SNORM_R8_G8_B8_A8
    CHECK(bytes_per_element(0x41A) == 4); // SRGB_R8_G8_B8_A8
    CHECK(color_format(0x41A) == 0x1A);
}

TEST_CASE("element sizes follow the R600 format names", "[latte][formats]") {
    CHECK(bytes_per_element(0x001) == 1);  // UNORM_R8
    CHECK(bytes_per_element(0x002) == 1);  // UNORM_R4_G4
    CHECK(bytes_per_element(0x007) == 2);  // UNORM_R8_G8
    CHECK(bytes_per_element(0x008) == 2);  // UNORM_R5_G6_B5
    CHECK(bytes_per_element(0x806) == 2);  // FLOAT_R16
    CHECK(bytes_per_element(0x019) == 4);  // UNORM_R10_G10_B10_A2
    CHECK(bytes_per_element(0x01B) == 4);  // UNORM_A2_B10_G10_R10
    CHECK(bytes_per_element(0x816) == 4);  // FLOAT_R11_G11_B10
    CHECK(bytes_per_element(0x811) == 4);  // FLOAT_D24_S8
    CHECK(bytes_per_element(0x81C) == 8);  // FLOAT_X8_X24
    CHECK(bytes_per_element(0x01F) == 8);  // UNORM_R16_G16_B16_A16
    CHECK(bytes_per_element(0x820) == 8);  // FLOAT_R16_G16_B16_A16
    CHECK(bytes_per_element(0x81E) == 8);  // FLOAT_R32_G32
    CHECK(bytes_per_element(0x122) == 16); // UINT_R32_G32_B32_A32
    CHECK(bytes_per_element(0x823) == 16); // FLOAT_R32_G32_B32_A32
}

TEST_CASE("formats without an element size say so", "[latte][formats]") {
    CHECK(bytes_per_element(0x000) == 0); // INVALID
    CHECK(bytes_per_element(0x031) == 0); // UNORM_BC1: measured in blocks
    CHECK(bytes_per_element(0x035) == 0); // UNORM_BC5
    CHECK(bytes_per_element(0x081) == 0); // UNORM_NV12
}

TEST_CASE("compressed formats are measured in blocks", "[latte][formats]") {
    CHECK(is_block_compressed(0x031)); // UNORM_BC1
    CHECK(is_block_compressed(0x433)); // SRGB_BC3
    CHECK_FALSE(is_block_compressed(0x01A));
    CHECK_FALSE(is_block_compressed(0x081)); // NV12

    CHECK(bits_per_element(0x031) == 64);  // BC1
    CHECK(bits_per_element(0x032) == 128); // BC2
    CHECK(bits_per_element(0x033) == 128); // BC3
    CHECK(bits_per_element(0x234) == 64);  // SNORM_BC4
    CHECK(bits_per_element(0x235) == 128); // SNORM_BC5
    CHECK(bits_per_element(0x01A) == 32);
    CHECK(bits_per_element(0x823) == 128);
    CHECK(bits_per_element(0x000) == 0);
}
