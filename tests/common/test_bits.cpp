#include <wiiu/common/bits.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace wiiu;

TEST_CASE("is_pow2", "[common][bits]") {
    STATIC_REQUIRE(is_pow2(1U));
    STATIC_REQUIRE(is_pow2(2U));
    STATIC_REQUIRE(is_pow2(32U));
    STATIC_REQUIRE(is_pow2(u32{1} << 31));

    STATIC_REQUIRE_FALSE(is_pow2(0U));
    STATIC_REQUIRE_FALSE(is_pow2(3U));
    STATIC_REQUIRE_FALSE(is_pow2(48U));
}

TEST_CASE("align_up", "[common][bits]") {
    STATIC_REQUIRE(align_up(0U, 32U) == 0U);
    STATIC_REQUIRE(align_up(1U, 32U) == 32U);
    STATIC_REQUIRE(align_up(31U, 32U) == 32U);
    STATIC_REQUIRE(align_up(32U, 32U) == 32U);
    STATIC_REQUIRE(align_up(33U, 32U) == 64U);

    // Page-sized, as the Phase 3 memory map will use it.
    STATIC_REQUIRE(align_up(0x1001U, 0x1000U) == 0x2000U);
}

TEST_CASE("align_down and is_aligned", "[common][bits]") {
    STATIC_REQUIRE(align_down(33U, 32U) == 32U);
    STATIC_REQUIRE(align_down(32U, 32U) == 32U);
    STATIC_REQUIRE(align_down(31U, 32U) == 0U);

    STATIC_REQUIRE(is_aligned(64U, 32U));
    STATIC_REQUIRE_FALSE(is_aligned(65U, 32U));
}

TEST_CASE("align_up and align_down agree on cache lines", "[common][bits]") {
    for (u32 address = 0; address < 128; ++address) {
        const u32 up = align_up(address, cache_line_size);
        const u32 down = align_down(address, cache_line_size);
        CHECK(is_aligned(up, cache_line_size));
        CHECK(is_aligned(down, cache_line_size));
        CHECK(down <= address);
        CHECK(up >= address);
        CHECK(up - down <= cache_line_size);
    }
}

TEST_CASE("extract_bits", "[common][bits]") {
    STATIC_REQUIRE(extract_bits(0xDEADBEEFU, 0, 4) == 0xFU);
    STATIC_REQUIRE(extract_bits(0xDEADBEEFU, 4, 4) == 0xEU);
    STATIC_REQUIRE(extract_bits(0xDEADBEEFU, 16, 16) == 0xDEADU);
    STATIC_REQUIRE(extract_bits(0xDEADBEEFU, 0, 32) == 0xDEADBEEFU);

    // A full-width request must not shift by the type width (UB).
    STATIC_REQUIRE(extract_bits(u8{0xAB}, 0, 8) == 0xABU);
}

TEST_CASE("sign_extend", "[common][bits]") {
    // The PowerPC 16-bit displacement case (Phase 4.4).
    STATIC_REQUIRE(sign_extend(0x0000U, 16) == 0);
    STATIC_REQUIRE(sign_extend(0x7FFFU, 16) == 32767);
    STATIC_REQUIRE(sign_extend(0x8000U, 16) == -32768);
    STATIC_REQUIRE(sign_extend(0xFFFFU, 16) == -1);

    // The 26-bit branch displacement case (Phase 4.5). Bit 25 is the sign,
    // so 0x03FFFFFC is -4 (a short backward branch), not a large positive
    // offset -- exactly the mistake this test exists to catch.
    STATIC_REQUIRE(sign_extend(0x01FFFFFCU, 26) == 33554428);
    STATIC_REQUIRE(sign_extend(0x03FFFFFCU, 26) == -4);
    STATIC_REQUIRE(sign_extend(0x02000000U, 26) == -33554432);
}
