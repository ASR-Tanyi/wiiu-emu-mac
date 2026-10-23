#include <wiiu/common/endian.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

using namespace wiiu;

namespace {

/// The bytes a value actually occupies in memory — the property that matters,
/// and the one a round-trip test alone would not catch. (A be<T> that swapped
/// on neither read nor write would round-trip perfectly and be useless.)
template <typename T>
[[nodiscard]] std::vector<u8> bytes_of(const be<T>& value) {
    std::vector<u8> out(sizeof(T));
    std::memcpy(out.data(), &value, sizeof(T));
    return out;
}

} // namespace

TEST_CASE("swap_bytes", "[common][endian]") {
    STATIC_REQUIRE(swap_bytes(u8{0xAB}) == 0xAB);
    STATIC_REQUIRE(swap_bytes(u16{0x1234}) == 0x3412);
    STATIC_REQUIRE(swap_bytes(u32{0x12345678}) == 0x78563412);
    STATIC_REQUIRE(swap_bytes(u64{0x0123456789ABCDEF}) == 0xEFCDAB8967452301);

    // Swapping twice is the identity — the property the encode/decode path
    // relies on.
    STATIC_REQUIRE(swap_bytes(swap_bytes(u32{0xDEADBEEF})) == 0xDEADBEEF);
}

TEST_CASE("be<T> stores big-endian bytes", "[common][endian]") {
    const u32be value = 0x12345678U;
    const auto bytes = bytes_of(value);

    REQUIRE(bytes.size() == 4);
    CHECK(bytes[0] == 0x12);
    CHECK(bytes[1] == 0x34);
    CHECK(bytes[2] == 0x56);
    CHECK(bytes[3] == 0x78);

    // ... and reads back in host order.
    CHECK(value == 0x12345678U);
}

TEST_CASE("be<T> round-trips every width", "[common][endian]") {
    CHECK(be<u16>{0xBEEF} == 0xBEEF);
    CHECK(be<u32>{0xDEADBEEF} == 0xDEADBEEF);
    CHECK(be<u64>{0x0123456789ABCDEF} == 0x0123456789ABCDEF);

    CHECK(be<s16>{-2} == -2);
    CHECK(be<s32>{-2} == -2);
    CHECK(be<s64>{-2} == -2);

    CHECK(be<u32>{0} == 0);
    CHECK(be<u32>{std::numeric_limits<u32>::max()} == std::numeric_limits<u32>::max());
    CHECK(be<s32>{std::numeric_limits<s32>::min()} == std::numeric_limits<s32>::min());
    CHECK(be<s32>{std::numeric_limits<s32>::max()} == std::numeric_limits<s32>::max());
}

TEST_CASE("be<T> preserves negative values bit-exactly", "[common][endian]") {
    const s32be negative = -1;
    const auto bytes = bytes_of(negative);
    CHECK(bytes[0] == 0xFF);
    CHECK(bytes[3] == 0xFF);
    CHECK(negative == -1);

    const s16be small = -2; // 0xFFFE
    const auto small_bytes = bytes_of(small);
    CHECK(small_bytes[0] == 0xFF);
    CHECK(small_bytes[1] == 0xFE);
}

TEST_CASE("be<float> swaps the bit pattern, not the value", "[common][endian]") {
    // 1.0f is 0x3F800000. Big-endian in memory: 3F 80 00 00.
    const f32be one = 1.0F;
    const auto bytes = bytes_of(one);
    CHECK(bytes[0] == 0x3F);
    CHECK(bytes[1] == 0x80);
    CHECK(bytes[2] == 0x00);
    CHECK(bytes[3] == 0x00);
    CHECK(one == 1.0F);

    const f64be half = 0.5; // 0x3FE0000000000000
    const auto wide = bytes_of(half);
    CHECK(wide[0] == 0x3F);
    CHECK(wide[1] == 0xE0);
    CHECK(half == 0.5);

    CHECK(f32be{-0.0F} == -0.0F);
    CHECK(f32be{3.5F} == 3.5F);
}

TEST_CASE("a signalling NaN survives the round trip", "[common][endian]") {
    // Guest memory holds bit patterns, not numbers. Swapping through a
    // floating-point register could quieten a signalling NaN; going through
    // the integer storage cannot.
    constexpr u32 snan_bits = 0x7FA00000;
    const f32be value = f32be::from_raw(swap_bytes(snan_bits));
    CHECK(value.raw() == swap_bytes(snan_bits));

    u32 recovered = 0;
    const f32 as_float = value.value();
    std::memcpy(&recovered, &as_float, sizeof(recovered));
    CHECK(recovered == snan_bits);
}

TEST_CASE("raw() and from_raw() expose the stored pattern", "[common][endian]") {
    const u32be value = 0x12345678U;
    CHECK(value.raw() == 0x78563412U); // as stored on a little-endian host

    const auto rebuilt = u32be::from_raw(value.raw());
    CHECK(rebuilt == 0x12345678U);
}

TEST_CASE("compound assignment operates in host order", "[common][endian]") {
    u32be counter = 10;

    counter += 5;
    CHECK(counter == 15U);

    counter -= 3;
    CHECK(counter == 12U);

    counter *= 2;
    CHECK(counter == 24U);

    counter /= 4;
    CHECK(counter == 6U);

    counter |= 0x10U;
    CHECK(counter == 0x16U);

    counter &= 0x1FU;
    CHECK(counter == 0x16U);

    counter ^= 0xFFU;
    CHECK(counter == 0xE9U);

    counter <<= 4;
    CHECK(counter == 0xE90U);

    counter >>= 8;
    CHECK(counter == 0xEU);

    // ... and the result is still stored big-endian.
    const auto bytes = bytes_of(counter);
    CHECK(bytes[0] == 0x00);
    CHECK(bytes[3] == 0x0E);
}

TEST_CASE("increment and decrement", "[common][endian]") {
    u32be value = 1;

    CHECK(++value == 2U);
    CHECK(value++ == 2U);
    CHECK(value == 3U);
    CHECK(--value == 2U);
    CHECK(value-- == 2U);
    CHECK(value == 1U);
}

TEST_CASE("be<T> is usable in constant expressions", "[common][endian]") {
    constexpr u32be value = 0xCAFEBABEU;
    STATIC_REQUIRE(value == 0xCAFEBABEU);
    STATIC_REQUIRE(value.value() == 0xCAFEBABEU);
    STATIC_REQUIRE(u32be{1} + u32be{2} == 3U);
}

TEST_CASE("an array of be<T> is contiguous guest data", "[common][endian]") {
    const std::array<u32be, 3> values{u32be{1}, u32be{2}, u32be{3}};
    STATIC_REQUIRE(sizeof(values) == 12);

    std::array<u8, 12> raw{};
    std::memcpy(raw.data(), values.data(), raw.size());

    CHECK(raw[3] == 1);
    CHECK(raw[7] == 2);
    CHECK(raw[11] == 3);
    CHECK(raw[0] == 0);
}

// ---------------------------------------------------------------------------
// Phase 2.6 done-criterion: a hand-written guest structure with mixed be<>
// fields parses a known byte blob correctly.
//
// The structure is the real one Phase 5.1 will need — an Elf32 header as it
// appears at the front of an RPX. Transcribed field-for-field from the format
// documentation, with no manual swapping anywhere.
// ---------------------------------------------------------------------------

namespace {

struct Elf32Header {
    std::array<u8, 16> ident; // no swapping: bytes are bytes
    u16be type;
    u16be machine;
    u32be version;
    u32be entry;
    u32be phoff;
    u32be shoff;
    u32be flags;
    u16be ehsize;
    u16be phentsize;
    u16be phnum;
    u16be shentsize;
    u16be shnum;
    u16be shstrndx;
};

static_assert(sizeof(Elf32Header) == 52, "Elf32 header is 52 bytes on the wire");
static_assert(std::is_trivially_copyable_v<Elf32Header>);
static_assert(std::is_standard_layout_v<Elf32Header>);
static_assert(offsetof(Elf32Header, type) == 16);
static_assert(offsetof(Elf32Header, entry) == 24);
static_assert(offsetof(Elf32Header, shoff) == 32);
static_assert(offsetof(Elf32Header, shstrndx) == 50);

/// A real RPX header prefix. e_type 0xFE01 is ET_CAFE_RPL; the 0xCA 0xFE in
/// e_ident at EI_OSABI/EI_ABIVERSION is the Cafe marker.
constexpr std::array<u8, 52> kRpxHeader{
    // e_ident
    0x7F, 0x45, 0x4C, 0x46,                               // \x7F E L F
    0x01,                                                 // EI_CLASS   = ELFCLASS32
    0x02,                                                 // EI_DATA    = ELFDATA2MSB (big-endian)
    0x01,                                                 // EI_VERSION = 1
    0xCA, 0xFE,                                           // EI_OSABI, EI_ABIVERSION
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x01, // e_type      = ET_CAFE_RPL
    0x00, 0x14,                                           // e_machine   = EM_PPC (20)
    0x00, 0x00, 0x00, 0x01,                               // e_version   = 1
    0x02, 0x00, 0x00, 0x00,                               // e_entry     = 0x02000000 (app region)
    0x00, 0x00, 0x00, 0x00, // e_phoff     = 0 (RPX has no program headers)
    0x00, 0x00, 0x00, 0x40, // e_shoff     = 0x40
    0x00, 0x00, 0x00, 0x00, // e_flags     = 0
    0x00, 0x34,             // e_ehsize    = 52
    0x00, 0x00,             // e_phentsize = 0
    0x00, 0x00,             // e_phnum     = 0
    0x00, 0x28,             // e_shentsize = 40
    0x00, 0x0B,             // e_shnum     = 11
    0x00, 0x0A,             // e_shstrndx  = 10
};

} // namespace

TEST_CASE("a guest struct parses a known blob", "[common][endian][guest-struct]") {
    Elf32Header header{};
    std::memcpy(&header, kRpxHeader.data(), sizeof(header));

    CHECK(header.ident[0] == 0x7F);
    CHECK(header.ident[1] == 'E');
    CHECK(header.ident[2] == 'L');
    CHECK(header.ident[3] == 'F');
    CHECK(header.ident[4] == 1);    // ELFCLASS32
    CHECK(header.ident[5] == 2);    // big-endian
    CHECK(header.ident[7] == 0xCA); // Cafe marker
    CHECK(header.ident[8] == 0xFE);

    CHECK(header.type == 0xFE01); // ET_CAFE_RPL
    CHECK(header.machine == 20);  // EM_PPC
    CHECK(header.version == 1U);
    CHECK(header.entry == 0x02000000U);
    CHECK(header.phoff == 0U); // RPX has no program headers
    CHECK(header.shoff == 0x40U);
    CHECK(header.flags == 0U);
    CHECK(header.ehsize == 52);
    CHECK(header.phnum == 0);
    CHECK(header.shentsize == 40);
    CHECK(header.shnum == 11);
    CHECK(header.shstrndx == 10);
}

TEST_CASE("writing a guest struct reproduces the blob", "[common][endian][guest-struct]") {
    Elf32Header header{};
    std::memcpy(header.ident.data(), kRpxHeader.data(), header.ident.size());
    header.type = 0xFE01;
    header.machine = 20;
    header.version = 1;
    header.entry = 0x02000000;
    header.phoff = 0;
    header.shoff = 0x40;
    header.flags = 0;
    header.ehsize = 52;
    header.phentsize = 0;
    header.phnum = 0;
    header.shentsize = 40;
    header.shnum = 11;
    header.shstrndx = 10;

    std::array<u8, 52> written{};
    std::memcpy(written.data(), &header, written.size());

    CHECK(written == kRpxHeader);
}
