#include <wiiu/common/bitfield.hpp>

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace wiiu;

namespace {

/// A PowerPC D-form instruction, as Phase 4.2 will declare it.
///
/// PowerPC numbers bits from the most significant end, so the documented
/// fields map to little-end positions like this:
///
///   documented bits 0-5   (opcode) -> position 26, 6 bits
///   documented bits 6-10  (rD/rS)  -> position 21, 5 bits
///   documented bits 11-15 (rA)     -> position 16, 5 bits
///   documented bits 16-31 (D)      -> position 0, 16 bits, signed
union DForm {
    u32 raw;
    BitField<26, 6, u32> opcode;
    BitField<21, 5, u32> rd;
    BitField<16, 5, u32> ra;
    BitField<0, 16, s32> displacement;
};

static_assert(sizeof(DForm) == 4);
static_assert(std::is_trivially_copyable_v<DForm>);

/// An X-form instruction with the record bit, to exercise single-bit fields.
union XForm {
    u32 raw;
    BitField<26, 6, u32> opcode;
    BitField<21, 5, u32> rs;
    BitField<1, 10, u32> extended_opcode;
    BitField<0, 1, bool> rc;
};

enum class SurfaceFormat : u32 { Invalid = 0, R8G8B8A8 = 0x1A, Bc1 = 0x31 };

union LatteRegister {
    u32 raw;
    BitField<0, 6, SurfaceFormat> format;
    BitField<8, 4, u32> tile_mode;
    BitField<31, 1, bool> enabled;
};

} // namespace

TEST_CASE("mask and max_value", "[common][bitfield]") {
    STATIC_REQUIRE(BitField<26, 6, u32>::mask == 0xFC000000U);
    STATIC_REQUIRE(BitField<0, 16, u32>::mask == 0x0000FFFFU);
    STATIC_REQUIRE(BitField<0, 32, u32>::mask == 0xFFFFFFFFU);
    STATIC_REQUIRE(BitField<31, 1, u32>::mask == 0x80000000U);

    STATIC_REQUIRE(BitField<26, 6, u32>::max_value == 0x3FU);
    STATIC_REQUIRE(BitField<0, 32, u32>::max_value == 0xFFFFFFFFU);
}

TEST_CASE("a full-width field does not shift by the type width", "[common][bitfield]") {
    // The UB trap: (1 << 32) on a 32-bit type.
    union Whole {
        u32 raw;
        BitField<0, 32, u32> all;
    };

    Whole word{.raw = 0xDEADBEEF};
    CHECK(word.all == 0xDEADBEEFU);

    word.all = 0x12345678;
    CHECK(word.raw == 0x12345678U);
}

TEST_CASE("D-form decode", "[common][bitfield]") {
    // lwz r3, 8(r4)  ->  opcode 32, rD 3, rA 4, D 8
    const DForm inst{.raw = 0x80640008};

    CHECK(inst.opcode == 32U);
    CHECK(inst.rd == 3U);
    CHECK(inst.ra == 4U);
    CHECK(inst.displacement == 8);
}

TEST_CASE("D-form displacement is sign-extended", "[common][bitfield]") {
    // stw r0, -4(r1)  ->  opcode 36, rS 0, rA 1, D -4 (0xFFFC)
    const DForm inst{.raw = 0x9001FFFC};

    CHECK(inst.opcode == 36U);
    CHECK(inst.rd == 0U);
    CHECK(inst.ra == 1U);
    CHECK(inst.displacement == -4);

    // The boundary cases of a 16-bit signed field.
    DForm edge{.raw = 0};
    edge.displacement = 32767;
    CHECK(edge.displacement == 32767);
    CHECK((edge.raw & 0xFFFFU) == 0x7FFFU);

    edge.displacement = -32768;
    CHECK(edge.displacement == -32768);
    CHECK((edge.raw & 0xFFFFU) == 0x8000U);

    edge.displacement = -1;
    CHECK(edge.displacement == -1);
    CHECK((edge.raw & 0xFFFFU) == 0xFFFFU);
}

TEST_CASE("single-bit fields read as bool", "[common][bitfield]") {
    // or. r3, r4, r4  (mr. r3, r4) -> opcode 31, extended 444, Rc set
    XForm inst{.raw = 0x7C832379};

    CHECK(inst.opcode == 31U);
    CHECK(inst.extended_opcode == 444U);
    CHECK(inst.rc == true);

    inst.rc = false;
    CHECK(inst.rc == false);
    CHECK(inst.raw == 0x7C832378U);

    inst.rc = true;
    CHECK(inst.raw == 0x7C832379U);
}

TEST_CASE("writing a field leaves its neighbours alone", "[common][bitfield]") {
    DForm inst{.raw = 0xFFFFFFFF};

    inst.rd = 0;
    CHECK(inst.rd == 0U);
    CHECK(inst.opcode == 0x3FU);    // untouched
    CHECK(inst.ra == 0x1FU);        // untouched
    CHECK(inst.displacement == -1); // untouched
    CHECK(inst.raw == 0xFC1FFFFFU); // only bits 21-25 cleared

    inst.opcode = 0;
    CHECK(inst.raw == 0x001FFFFFU); // and now bits 26-31 too
}

TEST_CASE("an oversized write truncates, as the hardware would", "[common][bitfield]") {
    DForm inst{.raw = 0};

    inst.rd = 0xFFFFFFFF; // 5-bit field
    CHECK(inst.rd == 0x1FU);
    CHECK(inst.raw == 0x03E00000U); // nothing spilled into rA or opcode
}

TEST_CASE("enum-typed fields", "[common][bitfield]") {
    LatteRegister reg{.raw = 0};

    reg.format = SurfaceFormat::R8G8B8A8;
    reg.tile_mode = 4;
    reg.enabled = true;

    CHECK(reg.format == SurfaceFormat::R8G8B8A8);
    CHECK(reg.tile_mode == 4U);
    CHECK(reg.enabled == true);
    CHECK(reg.raw == 0x8000041AU);

    reg.format = SurfaceFormat::Bc1;
    CHECK(reg.format == SurfaceFormat::Bc1);
    CHECK(reg.tile_mode == 4U); // untouched
}

TEST_CASE("free get_field and set_field", "[common][bitfield]") {
    STATIC_REQUIRE(get_field<26, 6>(0x80640008U) == 32U);
    STATIC_REQUIRE(get_field<0, 16, s32>(0x9001FFFCU) == -4);
    STATIC_REQUIRE(set_field<21, 5>(0U, 3U) == 0x00600000U);

    // Round-trip through both.
    constexpr u32 word = set_field<16, 5>(set_field<26, 6>(0U, 32U), 4U);
    STATIC_REQUIRE(get_field<26, 6>(word) == 32U);
    STATIC_REQUIRE(get_field<16, 5>(word) == 4U);
}

TEST_CASE("fields work over wider storage", "[common][bitfield]") {
    union Wide {
        u64 raw;
        BitField<0, 32, u32, u64> low;
        BitField<32, 32, u32, u64> high;
    };

    Wide value{.raw = 0};
    value.low = 0xDEADBEEF;
    value.high = 0x12345678;

    CHECK(value.raw == 0x12345678DEADBEEFULL);
    CHECK(value.low == 0xDEADBEEFU);
    CHECK(value.high == 0x12345678U);
}

TEST_CASE("BitField is not copy-assignable", "[common][bitfield]") {
    // `a = b` would copy the whole word, not the field. Deleting it means the
    // mistake is a compile error rather than a silently clobbered register.
    STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<BitField<0, 4, u32>>);
}
