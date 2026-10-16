// Phase 8.6 — decoding a shader's control flow.
//
// What is guarded: the three instruction forms' fields, through programs
// whose words are worked out by hand from the field positions, read back
// as AMD's syntax in decaf's spelling -- decaf's own example shaders
// (latte-assembler's resources) among them, which must come out as the
// text they were written as; the clauses found and where; subroutines
// found through CALL; and every refusal: a program that runs off its end,
// instructions R700 does not have, clauses and jumps outside the program,
// and a subroutine that never returns.

#include <wiiu/latte/control_flow.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

// Field positions, as microcode.hpp has them.
[[nodiscard]] constexpr u32 cf(CfInst inst) noexcept {
    return static_cast<u32>(inst) << 23;
}

[[nodiscard]] constexpr u32 exp(ExportInst inst) noexcept {
    return static_cast<u32>(inst) << 23;
}

[[nodiscard]] constexpr u32 alu(AluCfInst inst) noexcept {
    return static_cast<u32>(inst) << 26;
}

constexpr u32 barrier = 1U << 31;
constexpr u32 end_of_program = 1U << 21;
constexpr u32 xyzw = 0 | (1U << 3) | (2U << 6) | (3U << 9); // 0x688

/// An export's first word.
[[nodiscard]] constexpr u32 exp0(u32 base, u32 type, u32 gpr) noexcept {
    return base | (type << 13) | (gpr << 15);
}

[[nodiscard]] ControlFlow decoded(const std::vector<u32>& words,
                                  ProgramShape shape = ProgramShape::Shader) {
    ControlFlow flow;
    std::string error;
    INFO(error);
    REQUIRE(decode_control_flow(words, shape, flow, error));
    return flow;
}

[[nodiscard]] std::string refused(const std::vector<u32>& words,
                                  ProgramShape shape = ProgramShape::Shader) {
    ControlFlow flow;
    std::string error;
    CHECK_FALSE(decode_control_flow(words, shape, flow, error));
    CHECK_FALSE(error.empty());
    return error;
}

/// One instruction's text.
[[nodiscard]] std::string text(u32 word0, u32 word1) {
    return disassemble(CfInstruction{.index = 0, .word0 = word0, .word1 = word1});
}

} // namespace

TEST_CASE("decaf's example vertex shader reads as it was written", "[latte][shader][cf]") {
    // tools/latte-assembler/resources/example_shader.vsh: the fetch shader's
    // call, then the position and one parameter.
    const std::vector<u32> words{
        0x00000000,
        cf(CfInst::CallFs), // no barrier
        exp0(60, 1, 2),
        xyzw | exp(ExportInst::ExportDone) | barrier, // POS0, R2
        exp0(0, 2, 1),
        xyzw | exp(ExportInst::ExportDone) | end_of_program, // PARAM0, R1
    };
    // The words themselves, as the fields put them.
    CHECK(words[2] == 0x0001203C);
    CHECK(words[3] == 0x94000688);
    CHECK(words[5] == 0x14200688);

    const ControlFlow flow = decoded(words);
    CHECK(flow.main.size() == 3);
    CHECK(flow.clauses.empty());
    CHECK(flow.subroutines.empty());
    CHECK(flow.main[2].ends_program());
    CHECK(disassemble(flow) == "00 CALL_FS NO_BARRIER\n"
                               "01 EXP_DONE: POS0, R2.xyzw\n"
                               "02 EXP_DONE: PARAM0, R1.xyzw NO_BARRIER\n"
                               "END_OF_PROGRAM\n");
}

TEST_CASE("decaf's example pixel shader reads as it was written", "[latte][shader][cf]") {
    const std::vector<u32> words{exp0(0, 0, 0),
                                 xyzw | exp(ExportInst::ExportDone) | barrier | end_of_program};
    CHECK(words[1] == 0x94200688);
    CHECK(disassemble(decoded(words)) == "00 EXP_DONE: PIX0, R0.xyzw\nEND_OF_PROGRAM\n");
}

TEST_CASE("an ALU clause, its constant caches and the clause list", "[latte][shader][cf]") {
    std::vector<u32> words(72, 0);
    // 00 ALU at 64-bit word 32, four slots, the first sixteen uniforms of
    // block 1 in cache 0 and 32 of block 3, from uniform 32, in cache 1.
    words[0] = 32U | (1U << 22) | (3U << 26) | (1U << 30);
    words[1] = 2U | (0U << 2) | (2U << 10) | (3U << 18) | alu(AluCfInst::Alu) | barrier;
    words[2] = exp0(0, 0, 0);
    words[3] = xyzw | exp(ExportInst::ExportDone) | barrier | end_of_program;

    const ControlFlow flow = decoded(words);
    REQUIRE(flow.clauses.size() == 1);
    CHECK(flow.clauses[0].kind == ClauseKind::Alu);
    CHECK(flow.clauses[0].cf_index == 0);
    CHECK(flow.clauses[0].address == 32);
    CHECK(flow.clauses[0].count == 4);
    CHECK(flow.clauses[0].offset_bytes() == 256);
    CHECK(flow.clauses[0].size_bytes() == 32);
    CHECK(disassemble(flow.main[0]) == "ALU: ADDR(32) CNT(4) KCACHE0(CB1:0-15) KCACHE1(CB3:32-63)");
    CHECK_FALSE(flow.main[0].ends_program()); // an ALU clause has no END_OF_PROGRAM
}

TEST_CASE("loops indent, and a CALL's subroutine is found", "[latte][shader][cf]") {
    std::vector<u32> words(22, 0);
    const auto put = [&words](u32 index, u32 word0, u32 word1) {
        words[2 * index] = word0;
        words[(2 * index) + 1] = word1;
    };
    put(0, 5, cf(CfInst::Call) | barrier);
    put(1, 4, cf(CfInst::LoopStartDx10) | barrier);
    put(2, 8, cf(CfInst::Tex) | barrier);
    put(3, 2, cf(CfInst::LoopEnd) | barrier);
    put(4, exp0(0, 0, 0), xyzw | exp(ExportInst::ExportDone) | barrier | end_of_program);
    put(5, 10, alu(AluCfInst::Alu) | barrier);
    put(6, 0, cf(CfInst::Return) | barrier);

    const ControlFlow flow = decoded(words);
    CHECK(flow.main.size() == 5);
    REQUIRE(flow.subroutines.size() == 1);
    CHECK(flow.subroutines[0].address == 5);
    CHECK(flow.subroutines[0].instructions.size() == 2);

    REQUIRE(flow.clauses.size() == 2);
    CHECK(flow.clauses[0].kind == ClauseKind::Tex);
    CHECK(flow.clauses[0].cf_index == 2);
    CHECK(flow.clauses[0].size_bytes() == 16);
    CHECK(flow.clauses[1].kind == ClauseKind::Alu);
    CHECK(flow.clauses[1].cf_index == 5);

    // decaf indents a loop's body, its LOOP_END included, and prints
    // LOOP_START's and LOOP_END's CF_CONST even when it is zero.
    CHECK(disassemble(flow) == "00 CALL ADDR(5)\n"
                               "01 LOOP_START_DX10 FAIL_JUMP_ADDR(4)\n"
                               "  02 TEX: ADDR(8) CNT(1)\n"
                               "  03 LOOP_END CF_CONST(0) PASS_JUMP_ADDR(2)\n"
                               "04 EXP_DONE: PIX0, R0.xyzw\n"
                               "END_OF_PROGRAM\n"
                               "\n"
                               "SUBROUTINE 05\n"
                               "05 ALU: ADDR(10) CNT(1)\n"
                               "06 RETURN\n");
}

TEST_CASE("a fetch-shaped program ends at its RETURN", "[latte][shader][cf]") {
    std::vector<u32> words(12, 0);
    words[0] = 2;
    words[1] = cf(CfInst::VtxTc) | (1U << 10) | (1U << 19); // ten fetches: 9 = 0b1001
    words[3] = cf(CfInst::Return) | barrier;
    words.resize(4 + (10 * 4), 0);

    const ControlFlow flow = decoded(words, ProgramShape::FetchShader);
    CHECK(flow.main.size() == 2);
    REQUIRE(flow.clauses.size() == 1);
    CHECK(flow.clauses[0].kind == ClauseKind::Vtx);
    CHECK(flow.clauses[0].count == 10);
    CHECK(disassemble(flow) == "00 VTX_TC: ADDR(2) CNT(10) NO_BARRIER\n01 RETURN\n");

    // As a shader it would need an END_OF_PROGRAM it does not have.
    CHECK(refused(words).find("END_OF_PROGRAM") != std::string::npos);
}

TEST_CASE("the common form's fields, as decaf prints them", "[latte][shader][cf]") {
    constexpr u32 cond_bool = 2U << 8;
    constexpr u32 cf_const_5 = 5U << 3;
    constexpr u32 pop_2 = 2;
    constexpr u32 valid_pixel = 1U << 22;
    constexpr u32 wqm = 1U << 30;

    CHECK(text(3, cf(CfInst::Jump) | cond_bool | cf_const_5 | pop_2) ==
          "JUMP CND(BOOL) CF_CONST(5) POP_CNT(2) ADDR(3) NO_BARRIER");
    CHECK(text(7, cf(CfInst::PopJump) | pop_2 | barrier) == "POP_JUMP POP_CNT(2) ADDR(7)");
    CHECK(text(9, cf(CfInst::Call) | (3U << 13) | barrier) == "CALL CALL_COUNT(3) ADDR(9)");
    CHECK(text(0, cf(CfInst::Return) | (1U << 8) | barrier) == "RETURN CND(FALSE) CF_CONST(0)");
    CHECK(text(6, cf(CfInst::LoopStart) | cf_const_5 | barrier) ==
          "LOOP_START CF_CONST(5) FAIL_JUMP_ADDR(6)");
    CHECK(text(6, cf(CfInst::LoopBreak) | (3U << 8) | barrier) ==
          "LOOP_BREAK CND(NOT_BOOL) CF_CONST(0) ADDR(6)");
    CHECK(text(16, cf(CfInst::Tex) | (7U << 10) | (1U << 19) | wqm | valid_pixel) ==
          "TEX: ADDR(16) CNT(16) NO_BARRIER WHOLE_QUAD VALID_PIX");
    CHECK(text(0, cf(CfInst::Push) | cond_bool | pop_2 | valid_pixel | barrier) ==
          "PUSH CND(BOOL) CF_CONST(0) POP_CNT(2) VALID_PIX");
    CHECK(text(0, cf(CfInst::Pop) | pop_2 | barrier) == "POP POP_CNT(2)");
    CHECK(text(0, cf(CfInst::EmitVertex)) == "EMIT_VERTEX NO_BARRIER");
    CHECK(text(0, cf(CfInst::Nop) | barrier) == "NOP");
    CHECK(text(0, cf(CfInst::VtxAck) | barrier) == "VTX_ACK");
}

TEST_CASE("the exports' fields, as decaf prints them", "[latte][shader][cf]") {
    // A position export of two registers, R4 and R5, to POS1 and POS2.
    CHECK(text(exp0(61, 1, 4), xyzw | (1U << 17) | exp(ExportInst::Export) | barrier) ==
          "EXP: POS1, R4.xyzw BURSTCNT(1)");
    // Constants and masked components, from a relative register.
    const u32 swizzle = 4U | (5U << 3) | (7U << 6) | (6U << 9);
    CHECK(text(exp0(3, 2, 1) | (1U << 22), swizzle | exp(ExportInst::Export) | barrier) ==
          "EXP: PARAM3, R[AL + 1].01_?");
    // Stream-out: words, so the base is four bytes a unit.
    const u32 buffer = 4U | (3U << 12);
    CHECK(text(exp0(4, 0, 3), buffer | (1U << 17) | exp(ExportInst::MemStream0) | barrier) ==
          "MEM_STREAM0: WRITE(16), R3.xy__ ARRAY_SIZE(4) BURSTCNT(1)");
    // The ring, indexed, with an element size: and scratch counts in vec4s.
    CHECK(text(exp0(2, 1, 6) | (7U << 23) | (3U << 30), 0xFU << 12 | exp(ExportInst::MemRing)) ==
          "MEM_RING: WRITE(8 + R7), R6.xyzw ARRAY_SIZE(0) ELEM_SIZE(3) NO_BARRIER");
    CHECK(text(exp0(2, 0, 0), exp(ExportInst::MemScratch) | barrier) ==
          "MEM_SCRATCH: WRITE(32), R0.____ ARRAY_SIZE(0)");
    CHECK(text(exp0(0, 2, 0), exp(ExportInst::MemStream1) | barrier) ==
          "MEM_STREAM1: INVALID_READ, R0.____ ARRAY_SIZE(0)");
}

TEST_CASE("programs the later stages could not trust are refused", "[latte][shader][cf]") {
    CHECK(refused({}).find("runs off the end") != std::string::npos);
    CHECK(refused({0, cf(CfInst::Nop)}).find("without END_OF_PROGRAM") != std::string::npos);

    // 0x1E is Evergreen's, and an ALU clause's 0xC is reserved.
    CHECK(refused({0, (0x1EU << 23) | end_of_program}).find("not an R700") != std::string::npos);
    CHECK(refused({0, alu(AluCfInst::AluExtended)}).find("not an R700") != std::string::npos);
    CHECK(refused({0, exp(static_cast<ExportInst>(0x29)) | end_of_program}).find("not an R700") !=
          std::string::npos);

    // A clause and a jump outside the program.
    CHECK(refused({1, cf(CfInst::Vtx) | end_of_program}).find("past the end") != std::string::npos);
    CHECK(refused({0, alu(AluCfInst::Alu) | (2U << 18), 0, cf(CfInst::Nop) | end_of_program})
              .find("past the end") != std::string::npos);
    CHECK(refused({9, cf(CfInst::Jump) | end_of_program}).find("jumps to 9") != std::string::npos);

    // A subroutine that reaches the end of the program instead of returning.
    CHECK(refused({1, cf(CfInst::Call), 0, cf(CfInst::Nop) | end_of_program})
              .find("before its RETURN") != std::string::npos);
    CHECK(refused({0, cf(CfInst::Nop)}, ProgramShape::FetchShader).find("without a RETURN") !=
          std::string::npos);
}
