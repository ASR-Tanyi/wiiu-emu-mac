// Phase 8.3 — reading PM4.
//
// The reader's job is narrow and unforgiving: split a stream of words into
// packets, exactly, and stop at the first word it cannot account for. An
// off-by-one in a header's count misreads every packet after it, so the
// counts are what these tests are mostly about.

#include <wiiu/latte/pm4.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte::pm4;

TEST_CASE("a Type 3 header decodes", "[latte][pm4]") {
    // SET_CONTEXT_REG with three words of payload, predicated: count 2.
    const Header header = decode_header(0xC0026901);
    CHECK(header.type == PacketType::Type3);
    CHECK(header.opcode == 0x69);
    CHECK(header.payload_words == 3);
    CHECK(header.predicate);
    CHECK(opcode_name(header.opcode) == "SET_CONTEXT_REG");
}

TEST_CASE("a Type 0 header decodes", "[latte][pm4]") {
    // Four registers from index 0x2200 -- byte address 0x8800.
    const Header header = decode_header(0x00032200);
    CHECK(header.type == PacketType::Type0);
    CHECK(header.base_index == 0x2200);
    CHECK(header.payload_words == 4);
}

TEST_CASE("headers built here decode to what was asked for", "[latte][pm4]") {
    const Header three = decode_header(type3_header(Opcode::DrawIndexAuto, 2));
    CHECK(three.type == PacketType::Type3);
    CHECK(three.opcode == static_cast<u8>(Opcode::DrawIndexAuto));
    CHECK(three.payload_words == 2);
    CHECK_FALSE(three.predicate);

    const Header zero = decode_header(type0_header(0xA000, 5));
    CHECK(zero.type == PacketType::Type0);
    CHECK(zero.base_index == 0xA000);
    CHECK(zero.payload_words == 5);

    CHECK(decode_header(type2_filler).type == PacketType::Type2);
    CHECK(decode_header(type2_filler).payload_words == 0);

    // The largest payload a header can describe: fourteen bits of count.
    CHECK(decode_header(type3_header(Opcode::Nop, 0x4000)).payload_words == 0x4000);
}

TEST_CASE("an opcode nobody named still has a number", "[latte][pm4]") {
    CHECK(opcode_name(0x5D).empty());
    const std::array<u32, 2> words{0xC0005D00, 0x12345678};
    Reader reader{words};
    const auto packet = reader.next();
    REQUIRE(packet.has_value());
    CHECK(packet->header.opcode == 0x5D);
    CHECK(describe(*packet).find("unknown (0x5d)") != std::string::npos);
}

TEST_CASE("a stream splits into its packets exactly", "[latte][pm4]") {
    const std::vector<u32> words{
        type3_header(Opcode::SetContextReg, 3),
        0x00000010,
        0xAAAAAAAA,
        0xBBBBBBBB,
        type2_filler,
        type0_header(0x2300, 2),
        0x11111111,
        0x22222222,
        type3_header(Opcode::DrawIndexAuto, 2),
        6,
        2,
    };

    Reader reader{words};

    auto packet = reader.next();
    REQUIRE(packet.has_value());
    CHECK(packet->is(Opcode::SetContextReg));
    CHECK(packet->offset == 0);
    REQUIRE(packet->payload.size() == 3);
    CHECK(packet->payload[0] == 0x10);
    CHECK(packet->payload[2] == 0xBBBBBBBB);

    packet = reader.next();
    REQUIRE(packet.has_value());
    CHECK(packet->header.type == PacketType::Type2);
    CHECK(packet->offset == 4);
    CHECK(packet->payload.empty());

    packet = reader.next();
    REQUIRE(packet.has_value());
    CHECK(packet->header.type == PacketType::Type0);
    CHECK(packet->header.base_index == 0x2300);
    CHECK(packet->offset == 5);
    CHECK(packet->payload.size() == 2);

    packet = reader.next();
    REQUIRE(packet.has_value());
    CHECK(packet->is(Opcode::DrawIndexAuto));
    CHECK(packet->offset == 8);
    CHECK(packet->payload[0] == 6);

    CHECK_FALSE(reader.next().has_value());
    CHECK_FALSE(reader.failed());
    CHECK(reader.position() == words.size());
}

TEST_CASE("a payload that runs past the end stops the walk", "[latte][pm4]") {
    // Says four words of payload, has two.
    const std::array<u32, 4> words{
        type3_header(Opcode::Nop, 1), 0, type3_header(Opcode::SetContextReg, 4), 0x10};
    Reader reader{words};
    CHECK(reader.next().has_value());
    CHECK_FALSE(reader.next().has_value());
    REQUIRE(reader.failed());
    CHECK(reader.error().find("word 2") != std::string::npos);
    CHECK(reader.error().find("wants 4") != std::string::npos);

    // And it stays stopped: a misread header misreads everything after it.
    CHECK_FALSE(reader.next().has_value());
    CHECK(reader.position() == 2);
}

TEST_CASE("a Type 1 header is refused, not skipped", "[latte][pm4]") {
    const std::array<u32, 2> words{0x40000000, 0};
    Reader reader{words};
    CHECK_FALSE(reader.next().has_value());
    REQUIRE(reader.failed());
    CHECK(reader.error().find("Type 1") != std::string::npos);
}

TEST_CASE("an empty stream is empty, not broken", "[latte][pm4]") {
    Reader reader{std::span<const u32>{}};
    CHECK_FALSE(reader.next().has_value());
    CHECK_FALSE(reader.failed());
}

TEST_CASE("a packet describes itself in one line", "[latte][pm4]") {
    const std::array<u32, 3> words{type3_header(Opcode::DrawIndexAuto, 2, true), 3, 2};
    Reader reader{words};
    const auto packet = reader.next();
    REQUIRE(packet.has_value());
    CHECK(describe(*packet) == "@0 DRAW_INDEX_AUTO (0x2d) 2 words predicated");
}
