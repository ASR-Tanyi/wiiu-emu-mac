#include <wiiu/common/types.hpp>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <limits>
#include <type_traits>

using namespace wiiu;

TEST_CASE("fixed-width types have the expected sizes", "[common][types]") {
    STATIC_REQUIRE(sizeof(u8) == 1);
    STATIC_REQUIRE(sizeof(u16) == 2);
    STATIC_REQUIRE(sizeof(u32) == 4);
    STATIC_REQUIRE(sizeof(u64) == 8);
    STATIC_REQUIRE(sizeof(s8) == 1);
    STATIC_REQUIRE(sizeof(s16) == 2);
    STATIC_REQUIRE(sizeof(s32) == 4);
    STATIC_REQUIRE(sizeof(s64) == 8);
    STATIC_REQUIRE(sizeof(f32) == 4);
    STATIC_REQUIRE(sizeof(f64) == 8);
}

TEST_CASE("guest addresses are 32-bit", "[common][types]") {
    // Widening these hides truncation bugs at the host boundary, so the
    // property is asserted rather than assumed.
    STATIC_REQUIRE(sizeof(ppcaddr_t) == 4);
    STATIC_REQUIRE(sizeof(physaddr_t) == 4);
    STATIC_REQUIRE(std::is_unsigned_v<ppcaddr_t>);
    STATIC_REQUIRE(std::is_unsigned_v<physaddr_t>);
}

TEST_CASE("the host is little-endian", "[common][types]") {
    // The guest is big-endian; every byte-swap decision in Phase 2.1 assumes
    // the host is not. If this ever fails, be<T> needs revisiting.
    STATIC_REQUIRE(std::endian::native == std::endian::little);
}

TEST_CASE("float is IEEE-754", "[common][types]") {
    STATIC_REQUIRE(std::numeric_limits<f32>::is_iec559);
    STATIC_REQUIRE(std::numeric_limits<f64>::is_iec559);
}
