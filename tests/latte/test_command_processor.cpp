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

#include <memory>
#include <string>
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
