// Latte's surface formats: the element sizes the scan buffers (8.1) and, later,
// every surface (8.5) are measured in.

#include <wiiu/latte/formats.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>

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

TEST_CASE("a clear colour is encoded as the format stores it", "[latte][formats]") {
    // Red first, a byte each; unorm rounds to nearest: 0.5 is 127.5, so 128.
    const auto rgba8 = encode_color(0x01A, {0.0F, 0.5F, 1.0F, 0.25F});
    REQUIRE(rgba8.has_value());
    CHECK(rgba8->size == 4);
    CHECK(rgba8->bytes[0] == 0);
    CHECK(rgba8->bytes[1] == 128);
    CHECK(rgba8->bytes[2] == 255);
    CHECK(rgba8->bytes[3] == 64);

    // Out of range clamps, and NaN is zero.
    const auto clamped = encode_color(0x01A, {-1.0F, 2.0F, std::nanf(""), 1.0F});
    REQUIRE(clamped.has_value());
    CHECK(clamped->bytes[0] == 0);
    CHECK(clamped->bytes[1] == 255);
    CHECK(clamped->bytes[2] == 0);

    // Narrower unorm formats take the first components.
    const auto r8 = encode_color(0x001, {1.0F, 0.0F, 0.0F, 0.0F});
    REQUIRE(r8.has_value());
    CHECK(r8->size == 1);
    CHECK(r8->bytes[0] == 255);
    const auto rg8 = encode_color(0x007, {0.0F, 1.0F, 0.0F, 0.0F});
    REQUIRE(rg8.has_value());
    CHECK(rg8->size == 2);
    CHECK(rg8->bytes[1] == 255);
}

TEST_CASE("an sRGB clear is encoded with the sRGB curve", "[latte][formats]") {
    // Linear 0.5 is sRGB 0.7354, which is 187.5 of 255: 188. Alpha stays
    // linear. The curve's ends are fixed points, and its linear toe is 12.92x.
    const auto srgb = encode_color(0x41A, {0.5F, 0.0F, 1.0F, 0.5F});
    REQUIRE(srgb.has_value());
    CHECK(srgb->bytes[0] == 188);
    CHECK(srgb->bytes[1] == 0);
    CHECK(srgb->bytes[2] == 255);
    CHECK(srgb->bytes[3] == 128);
    CHECK(encode_color(0x41A, {0.002F, 0, 0, 0})->bytes[0] == 7); // 0.02584 * 255 = 6.59

    // Formats whose byte order has not been checked are refused, not guessed.
    CHECK_FALSE(encode_color(0x823, {0, 0, 0, 0}).has_value()); // FLOAT_R32_G32_B32_A32
    CHECK_FALSE(encode_color(0x019, {0, 0, 0, 0}).has_value()); // UNORM_R10_G10_B10_A2
    CHECK_FALSE(encode_color(0x11A, {0, 0, 0, 0}).has_value()); // UINT_R8_G8_B8_A8
}

TEST_CASE("a depth clear is encoded as the depth format stores it", "[latte][formats]") {
    // Little-endian words. R32: the float itself.
    const auto r32 = encode_depth_stencil(0x80E, 1.0F, 0x5A, true, true);
    REQUIRE(r32.has_value());
    CHECK(r32->size == 4);
    CHECK(r32->bytes == std::array<u8, 4>{0x00, 0x00, 0x80, 0x3F});
    CHECK(r32->write == std::array<bool, 4>{true, true, true, true});

    // R16: a 16-bit unorm; 0.5 is 32767.5, so 32768.
    const auto r16 = encode_depth_stencil(0x005, 0.5F, 0, true, false);
    REQUIRE(r16.has_value());
    CHECK(r16->size == 2);
    CHECK(r16->bytes[0] == 0x00);
    CHECK(r16->bytes[1] == 0x80);

    // R24_X8: 24 bits of depth under the stencil byte, and a clear of one
    // leaves the other's bytes alone.
    const auto both = encode_depth_stencil(0x011, 1.0F, 0x5A, true, true);
    REQUIRE(both.has_value());
    CHECK(both->bytes == std::array<u8, 4>{0xFF, 0xFF, 0xFF, 0x5A});
    CHECK(both->write == std::array<bool, 4>{true, true, true, true});
    CHECK(encode_depth_stencil(0x011, 0.0F, 7, true, false)->write ==
          std::array<bool, 4>{true, true, true, false});
    CHECK(encode_depth_stencil(0x011, 0.0F, 7, false, true)->write ==
          std::array<bool, 4>{false, false, false, true});

    // Unorm depth clamps; NaN is zero.
    CHECK(encode_depth_stencil(0x005, 2.0F, 0, true, false)->bytes[1] == 0xFF);
    CHECK(encode_depth_stencil(0x005, std::nanf(""), 0, true, false)->bytes[1] == 0x00);

    // The float D24_S8 and the 64-bit X8_X24 are not encoded yet.
    CHECK_FALSE(encode_depth_stencil(0x811, 1.0F, 0, true, true).has_value());
    CHECK_FALSE(encode_depth_stencil(0x81C, 1.0F, 0, true, true).has_value());
    CHECK_FALSE(encode_depth_stencil(0x01A, 1.0F, 0, true, true).has_value());
}
