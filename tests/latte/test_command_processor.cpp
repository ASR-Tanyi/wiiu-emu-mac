// Phase 8.3 — running PM4.
//
// The property that matters most is ordering: a draw must see the register
// file exactly as the packets *before* it left it, not as the stream leaves
// it at the end. A translator that read state after the fact would draw
// every object in a frame with the last object's blend mode.

#include <wiiu/common/log.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;
using pm4::Opcode;
using pm4::type0_header;
using pm4::type3_header;

namespace {

/// VGT_PRIMITIVE_TYPE as an offset into the config window.
constexpr u32 kPrimitiveTypeOffset = (reg::vgt_primitive_type - 0x08000) / 4;

struct Fixture {
    std::unique_ptr<RegisterFile> registers = std::make_unique<RegisterFile>();
    CommandProcessor processor{*registers};
    std::vector<DrawCommand> draws;
    std::vector<std::vector<u32>> immediate;

    Fixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        processor.set_draw_listener([this](const DrawCommand& draw, const RegisterFile&) {
            draws.push_back(draw);
            immediate.emplace_back(draw.immediate_indices.begin(), draw.immediate_indices.end());
        });
    }
};

/// Guest memory, for the packets that address it.
struct MemoryFixture : Fixture {
    memory::Memory memory;

    MemoryFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        processor.set_memory(&memory);
    }

    /// Write a stream into guest memory the way GX2 leaves it: big-endian.
    void put(ppcaddr_t address, const std::vector<u32>& words) {
        for (std::size_t index = 0; index < words.size(); ++index) {
            REQUIRE(memory.try_write<u32>(address + static_cast<u32>(index * 4), words[index]));
        }
    }
};

/// MEM1: virtual 0xF4000000 is physical 0.
constexpr ppcaddr_t kMem1 = memory::mem1_base;

} // namespace

TEST_CASE_METHOD(Fixture, "state packets write the register file", "[latte][cp]") {
    const std::vector<u32> stream{
        type3_header(Opcode::SetContextReg, 3),
        0x10,
        0xAAAA,
        0xBBBB,
        type0_header(0x2300, 1),
        0x1234,
        type3_header(Opcode::SetAluConst, 5),
        4,
        1,
        2,
        3,
        4,
    };
    REQUIRE(processor.execute(stream));
    CHECK(registers->read(0x28040) == 0xAAAA);
    CHECK(registers->read(0x28044) == 0xBBBB);
    CHECK(registers->read(0x08C00) == 0x1234);
    // ALU constant 1 (four words each), its .x component.
    CHECK(registers->read(0x30010) == 1);
    CHECK(processor.stats().packets == 3);
    CHECK(processor.stats().errors == 0);
}

TEST_CASE_METHOD(Fixture, "a draw sees the state as it was at the draw", "[latte][cp]") {
    const std::vector<u32> stream{
        type3_header(Opcode::SetConfigReg, 2),
        kPrimitiveTypeOffset,
        4, // triangle list
        type3_header(Opcode::NumInstances, 1),
        3,
        type3_header(Opcode::DrawIndexAuto, 2),
        36,
        0x2,
        type3_header(Opcode::SetConfigReg, 2),
        kPrimitiveTypeOffset,
        0x11, // RECTS
        type3_header(Opcode::DrawIndexAuto, 2),
        3,
        0x2,
    };
    REQUIRE(processor.execute(stream));
    REQUIRE(draws.size() == 2);

    CHECK(draws[0].kind == DrawCommand::Kind::Auto);
    CHECK(draws[0].count == 36);
    CHECK(draws[0].primitive == 4);
    CHECK(draws[0].instances == 3);
    CHECK(draws[0].draw_initiator == 0x2);

    // The second draw got the second primitive type, and the instance count
    // carried over -- NUM_INSTANCES is state, not part of one draw.
    CHECK(draws[1].primitive == 0x11);
    CHECK(draws[1].instances == 3);
    CHECK(processor.stats().draws == 2);
}

TEST_CASE_METHOD(Fixture, "immediate indices come with their draw", "[latte][cp]") {
    const std::vector<u32> stream{
        type3_header(Opcode::IndexType, 1),
        0, // 16-bit
        type3_header(Opcode::DrawIndexImmediate, 4),
        3,
        0x1,
        0x00010000,
        0x00000002,
    };
    REQUIRE(processor.execute(stream));
    REQUIRE(draws.size() == 1);
    CHECK(draws[0].kind == DrawCommand::Kind::Immediate);
    CHECK(draws[0].count == 3);
    CHECK_FALSE(draws[0].indices_are_32bit());
    REQUIRE(immediate[0].size() == 2);
    CHECK(immediate[0][0] == 0x00010000);
}

TEST_CASE_METHOD(MemoryFixture,
                 "an indexed draw finds its indices in guest memory",
                 "[latte][cp]") {
    // The packet carries a physical address -- 0x2000, in MEM1 -- and the
    // draw reports the virtual one the rest of the emulator uses.
    const std::vector<u32> stream{
        type3_header(Opcode::IndexType, 1),
        1, // 32-bit
        type3_header(Opcode::DrawIndex2, 5),
        0xFFFF,
        0x2000,
        0,
        6,
        0x0,
    };
    REQUIRE(processor.execute(stream));
    REQUIRE(draws.size() == 1);
    CHECK(draws[0].kind == DrawCommand::Kind::Indexed);
    CHECK(draws[0].count == 6);
    CHECK(draws[0].indices_are_32bit());
    CHECK(draws[0].index_address == kMem1 + 0x2000);
}

TEST_CASE_METHOD(Fixture, "an indexed draw with nowhere to read from is an error", "[latte][cp]") {
    // No memory attached: the draw cannot be made, and is not reported as if
    // it could.
    const std::vector<u32> stream{type3_header(Opcode::DrawIndex2, 5), 0xFFFF, 0x2000, 0, 6, 0};
    REQUIRE(processor.execute(stream));
    CHECK(draws.empty());
    CHECK(processor.stats().errors == 1);
    CHECK(processor.last_error().find("0x00002000") != std::string::npos);
}

TEST_CASE_METHOD(MemoryFixture, "an indirect buffer runs and comes back", "[latte][cp]") {
    // A display list in MEM1 at physical 0x1000: one state write and a draw.
    put(kMem1 + 0x1000,
        {
            type3_header(Opcode::SetContextReg, 2),
            0x20,
            0xCAFE,
            type3_header(Opcode::DrawIndexAuto, 2),
            6,
            0x2,
        });

    const std::vector<u32> stream{
        type3_header(Opcode::IndirectBufferPriv, 3),
        0x1000,
        0,
        6,
        type3_header(Opcode::DrawIndexAuto, 2),
        3,
        0x2, // after the call returns
    };
    REQUIRE(processor.execute(stream));

    CHECK(registers->read(0x28080) == 0xCAFE);
    REQUIRE(draws.size() == 2);
    CHECK(draws[0].count == 6); // the one inside the list, first
    CHECK(draws[1].count == 3);
    CHECK(processor.stats().indirect_buffers == 1);
}

TEST_CASE_METHOD(MemoryFixture, "a buffer that calls itself stops, not the host", "[latte][cp]") {
    // A buffer at physical 0x3000 whose only packet calls physical 0x3000.
    put(kMem1 + 0x3000, {type3_header(Opcode::IndirectBuffer, 3), 0x3000, 0, 4});

    const std::vector<u32> stream{type3_header(Opcode::IndirectBuffer, 3), 0x3000, 0, 4};
    REQUIRE(processor.execute(stream));
    CHECK(processor.stats().indirect_buffers == CommandProcessor::max_indirect_depth);
    CHECK(processor.last_error().find("nested") != std::string::npos);
}

TEST_CASE_METHOD(MemoryFixture,
                 "a buffer somewhere unmapped is an error, not a read",
                 "[latte][cp]") {
    const std::vector<u32> stream{type3_header(Opcode::IndirectBufferPriv, 3), 0xF0000000, 0, 16};
    REQUIRE(processor.execute(stream));
    CHECK(processor.stats().indirect_buffers == 0);
    CHECK(processor.stats().errors == 1);

    // And an absurd size is refused before anything is allocated.
    const std::vector<u32> huge{type3_header(Opcode::IndirectBufferPriv, 3), 0x1000, 0, 0xFFFFFFFF};
    REQUIRE(processor.execute(huge));
    CHECK(processor.stats().errors == 2);
}

TEST_CASE_METHOD(Fixture,
                 "a write past its window is an error the stream survives",
                 "[latte][cp]") {
    const std::vector<u32> stream{
        type3_header(Opcode::SetContextReg, 3),
        0x3FF,
        1,
        2, // the second word is past the end
        type3_header(Opcode::DrawIndexAuto, 2),
        3,
        0x2,
    };
    REQUIRE(processor.execute(stream));
    CHECK(processor.stats().errors == 1);
    CHECK(processor.last_error().find("context") != std::string::npos);
    // The framing was fine, so the packet after it still ran.
    CHECK(draws.size() == 1);
}

TEST_CASE_METHOD(Fixture, "what cannot be acted on yet is counted by opcode", "[latte][cp]") {
    const std::vector<u32> stream{
        type3_header(Opcode::EventWrite, 1),
        0x16,
        type3_header(Opcode::EventWrite, 1),
        0x16,
        type3_header(Opcode::SurfaceSync, 4),
        0,
        0,
        0,
        10,
        0xC0005D00,
        0, // an opcode with no name
    };
    REQUIRE(processor.execute(stream));
    CHECK(processor.stats().deferred == 3);
    CHECK(processor.stats().unknown == 1);
    CHECK(processor.unhandled().at(static_cast<u8>(Opcode::EventWrite)) == 2);
    CHECK(processor.unhandled().at(0x5D) == 1);
    CHECK(processor.stats().errors == 0);
}

TEST_CASE_METHOD(Fixture, "a stream that cannot be framed stops and says so", "[latte][cp]") {
    const std::vector<u32> stream{
        type3_header(Opcode::DrawIndexAuto, 2),
        3,
        0x2,
        type3_header(Opcode::SetContextReg, 8),
        0, // eight words promised, one there
    };
    CHECK_FALSE(processor.execute(stream));
    CHECK(draws.size() == 1); // what came before still happened
    CHECK(processor.last_error().find("word 3") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Writes back to guest memory (8.2)
// ---------------------------------------------------------------------------

TEST_CASE("a GPU write is swapped the way its address says", "[latte][cp]") {
    constexpr u64 kValue = 0x1122334455667788ULL;

    // No swap: the GPU's own little-endian order.
    CHECK(encode_write(kValue, 4, 0) ==
          std::array<u8, 8>{0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11});
    // Within 32 bits, on a 32-bit write: big-endian, which is what GX2 asks
    // for so that the guest reads the value back as written.
    const auto swapped32 = encode_write(kValue, 4, 2);
    CHECK(swapped32[0] == 0x55);
    CHECK(swapped32[3] == 0x88);
    // Within 64 bits, on a 64-bit write: the whole value big-endian.
    const auto swapped64 = encode_write(kValue, 8, 3);
    CHECK(swapped64[0] == 0x11);
    CHECK(swapped64[7] == 0x88);
    // Within 16 bits: pairs.
    const auto swapped16 = encode_write(kValue, 4, 1);
    CHECK(swapped16[0] == 0x77);
    CHECK(swapped16[1] == 0x88);
    // A unit wider than the write is the whole write.
    const auto wide = encode_write(0xAABBCCDD, 4, 3);
    CHECK(wide[0] == 0xAA);
    CHECK(wide[3] == 0xDD);
}

TEST_CASE_METHOD(MemoryFixture, "MEM_WRITE puts a value in guest memory", "[latte][cp]") {
    // A 32-bit write to physical 0x4000 with the 32-bit swap: the guest reads
    // exactly the value the packet carried.
    const std::vector<u32> stream{
        type3_header(Opcode::MemWrite, 4),
        0x4000 | 2,
        1U << 18,
        0xDEADBEEF,
        0,
    };
    REQUIRE(processor.execute(stream));
    CHECK(memory.read<u32>(kMem1 + 0x4000) == 0xDEADBEEF);
    CHECK(memory.read<u32>(kMem1 + 0x4004) == 0); // 32 bits, not 64
    CHECK(processor.stats().memory_writes == 1);
}

TEST_CASE_METHOD(MemoryFixture, "MEM_WRITE of the clock writes the guest's clock", "[latte][cp]") {
    // ADR-004: the "GPU clock" is the timebase the stream was run at, so two
    // runs of the same stream write the same value.
    processor.set_clock(0x0000000123456789ULL);
    const std::vector<u32> stream{
        type3_header(Opcode::MemWrite, 4),
        0x4100 | 3,
        1U << 16,
        0,
        0,
    };
    REQUIRE(processor.execute(stream));
    CHECK(memory.read<u64>(kMem1 + 0x4100) == 0x0000000123456789ULL);
}

TEST_CASE_METHOD(MemoryFixture, "an end-of-pipe event writes and interrupts", "[latte][cp]") {
    std::vector<std::pair<u32, u64>> interrupts;
    processor.set_interrupt_listener(
        [&interrupts](u32 event, u64 value) { interrupts.emplace_back(event, value); });

    // 64-bit data (select 2), interrupt (select 2), 64-bit swap: a timestamp
    // the way GX2 writes one.
    const u32 address_hi = (2U << 29) | (2U << 24);
    const std::vector<u32> stream{
        type3_header(Opcode::EventWriteEop, 5),
        0x515,
        0x4200 | 3,
        address_hi,
        0x00000042,
        0x1,
    };
    REQUIRE(processor.execute(stream));
    CHECK(memory.read<u64>(kMem1 + 0x4200) == 0x0000000100000042ULL);
    REQUIRE(interrupts.size() == 1);
    CHECK(interrupts[0].first == 0x515);
    CHECK(interrupts[0].second == 0x0000000100000042ULL);
    CHECK(processor.stats().interrupts == 1);
}

TEST_CASE_METHOD(MemoryFixture,
                 "an end-of-pipe event can interrupt without writing",
                 "[latte][cp]") {
    std::vector<u64> values;
    processor.set_interrupt_listener([&values](u32, u64 value) { values.push_back(value); });

    // Data select 0 discards; interrupt select 1 still interrupts.
    const std::vector<u32> stream{
        type3_header(Opcode::EventWriteEop, 5),
        0x515,
        0x4300 | 2,
        1U << 24,
        7,
        0,
    };
    REQUIRE(processor.execute(stream));
    CHECK(memory.read<u32>(kMem1 + 0x4300) == 0);
    CHECK(processor.stats().memory_writes == 0);
    CHECK(values.size() == 1);
}

TEST_CASE_METHOD(MemoryFixture,
                 "a write somewhere unmapped is an error, not a write",
                 "[latte][cp]") {
    const std::vector<u32> stream{
        type3_header(Opcode::MemWrite, 4),
        0xF0000000 | 2,
        1U << 18,
        1,
        0,
        type3_header(Opcode::MemWrite, 4),
        0x4000 | 2,
        (1U << 18) | 0x1,
        1,
        0, // a 40-bit address
    };
    REQUIRE(processor.execute(stream));
    CHECK(processor.stats().errors == 2);
    CHECK(processor.stats().memory_writes == 0);
}

TEST_CASE_METHOD(MemoryFixture,
                 "the emulator's own packets go to their listener, in stream order",
                 "[latte][cp]") {
    std::vector<std::pair<u8, std::vector<u32>>> seen;
    processor.set_hle_listener([&seen](const pm4::Packet& packet) {
        seen.emplace_back(packet.header.opcode,
                          std::vector<u32>(packet.payload.begin(), packet.payload.end()));
    });

    // One from a called buffer, as from a display list: GX2 writes those
    // big-endian into the title's memory, and they arrive the same.
    put(kMem1 + 0x1000, {type3_header(Opcode::HleCopyColorToScan, 2), 1, 0xABCD});
    const std::vector<u32> stream{
        type3_header(Opcode::HleClearColor, 1),
        0x3F800000,
        type3_header(Opcode::IndirectBufferPriv, 3),
        0x1000,
        0,
        3,
        type3_header(Opcode::HleSwap, 1),
        0x53574150,
    };
    REQUIRE(processor.execute(stream));
    REQUIRE(seen.size() == 3);
    CHECK(seen[0].first == static_cast<u8>(Opcode::HleClearColor));
    CHECK(seen[0].second == std::vector<u32>{0x3F800000});
    CHECK(seen[1].first == static_cast<u8>(Opcode::HleCopyColorToScan));
    CHECK(seen[1].second == std::vector<u32>{1, 0xABCD});
    CHECK(seen[2].first == static_cast<u8>(Opcode::HleSwap));
    CHECK(processor.stats().hle == 3);
    CHECK(processor.stats().unknown == 0);
}

TEST_CASE_METHOD(Fixture,
                 "with nobody listening, the emulator's packets are deferred",
                 "[latte][cp]") {
    const std::vector<u32> stream{type3_header(Opcode::HleSwap, 1), 0x53574150};
    REQUIRE(processor.execute(stream));
    CHECK(processor.stats().hle == 0);
    CHECK(processor.stats().deferred == 1);
    CHECK(processor.stats().errors == 0);
    CHECK(pm4::opcode_name(static_cast<u8>(Opcode::HleSwap)) == "HLE_SWAP");
    CHECK(pm4::is_hle_opcode(static_cast<u8>(Opcode::HleCopyColorToScan)));
    CHECK_FALSE(pm4::is_hle_opcode(static_cast<u8>(Opcode::SurfaceBaseUpdate)));
}
