#include <wiiu/cpu/disassembler.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <string_view>

using namespace wiiu;
using namespace wiiu::cpu;

namespace {

/// Every encoding below was produced by binutils' assembler, so the expected
/// text is checked against what a real toolchain calls the same word.
[[nodiscard]] std::string dis(u32 word, ppcaddr_t address = 0) {
    return disassemble(word, address);
}

[[nodiscard]] std::string raw(u32 word, ppcaddr_t address = 0) {
    return disassemble(
        word,
        address,
        {.simplified = false, .register_names = true, .absolute_branch_targets = true});
}

} // namespace

TEST_CASE("integer instructions", "[cpu][disasm]") {
    CHECK(dis(0x7C642A14) == "add r3,r4,r5");
    CHECK(dis(0x7C642851) == "subf. r3,r4,r5");
    CHECK(dis(0x7C642E15) == "addo. r3,r4,r5");
    CHECK(dis(0x7C6429D6) == "mullw r3,r4,r5");
    CHECK(dis(0x7C642BD6) == "divw r3,r4,r5");
    CHECK(dis(0x38630001) == "addi r3,r3,1");
    CHECK(dis(0x7C642838) == "and r4,r3,r5");
    CHECK(dis(0x7C642A79) == "xor. r4,r3,r5");
    CHECK(dis(0x54644228) == "rlwinm r4,r3,8,8,20");
    CHECK(dis(0x7C643E70) == "srawi r4,r3,7");
}

TEST_CASE("loads and stores", "[cpu][disasm]") {
    CHECK(dis(0x80640008) == "lwz r3,8(r4)");
    CHECK(dis(0x9421FFE0) == "stwu r1,-32(r1)");
    CHECK(dis(0x90010024) == "stw r0,36(r1)");
    CHECK(dis(0x93E1001C) == "stw r31,28(r1)");
    CHECK(dis(0xC0240008) == "lfs f1,8(r4)");
    CHECK(dis(0xD8450010) == "stfd f2,16(r5)");
}

TEST_CASE("floating point", "[cpu][disasm]") {
    CHECK(dis(0xEC22182A) == "fadds f1,f2,f3");
    CHECK(dis(0xFC22193A) == "fmadd f1,f2,f4,f3");
    CHECK(dis(0xFC201890) == "fmr f1,f3");
}

TEST_CASE("paired singles", "[cpu][disasm]") {
    CHECK(dis(0x1022182A) == "ps_add f1,f2,f3");
    CHECK(dis(0xE0242008) == "psq_l f1,8(r4),0,2");
    CHECK(dis(0xF024BFF0) == "psq_st f1,-16(r4),1,3");
}

TEST_CASE("cache, sync and reservations", "[cpu][disasm]") {
    CHECK(dis(0x7C642828) == "lwarx r3,r4,r5");
    CHECK(dis(0x7C64292D) == "stwcx. r3,r4,r5");
    CHECK(dis(0x7C0428AC) == "dcbf r4,r5");
    CHECK(dis(0x7C042FAC) == "icbi r4,r5");
    CHECK(dis(0x7C0004AC) == "sync");
    CHECK(dis(0x4C00012C) == "isync");
    CHECK(dis(0x7C0006AC) == "eieio");
    CHECK(dis(0x44000002) == "sc");
}

TEST_CASE("compare and branch mnemonics", "[cpu][disasm]") {
    // These two dominate compiled code, so printing them raw makes every
    // disassembly harder to read than it needs to be.
    CHECK(dis(0x2C04000B) == "cmpwi r4,11");
    CHECK(dis(0x28030064) == "cmplwi r3,100");
    CHECK(dis(0x7C032000) == "cmpw r3,r4");
    CHECK(dis(0x7C032040) == "cmplw r3,r4");

    CHECK(dis(0x41820020, 0) == "beq 0x20");
    CHECK(dis(0x40820020, 0) == "bne 0x20");
    CHECK(dis(0x41800020, 0) == "blt 0x20");
    CHECK(dis(0x41810020, 0) == "bgt 0x20");

    // A non-zero CR field is named rather than dropped.
    CHECK(dis(0x41860020, 0) == "beq cr1,0x20");

    // Counter-based forms have no simplified spelling here and stay raw.
    CHECK(dis(0x4200FFFC, 0x1000) == "bc 16,0,0xffc");
}

TEST_CASE("condition register and special registers", "[cpu][disasm]") {
    CHECK(dis(0x4C221B82) == "cror 1,2,3");
    CHECK(dis(0x7C680120) == "mtcrf 128,r3");
    CHECK(dis(0x7C600026) == "mfcr r3");
    CHECK(dis(0x7C6C42E6) == "mftb r3,268");
}

TEST_CASE("simplified mnemonics", "[cpu][disasm]") {
    CHECK(dis(0x60000000) == "nop");         // ori 0,0,0
    CHECK(dis(0x7C3F0B78) == "mr r31,r1");   // or 31,1,1
    CHECK(dis(0x3860FFFF) == "li r3,-1");    // addi 3,0,-1
    CHECK(dis(0x3C601234) == "lis r3,4660"); // addis 3,0,0x1234
    CHECK(dis(0x4E800020) == "blr");         // bclr 20,0
    CHECK(dis(0x4E800421) == "bctrl");       // bcctrl 20,0
    CHECK(dis(0x7C0802A6) == "mflr r0");     // mfspr 0,8
    CHECK(dis(0x7C6903A6) == "mtctr r3");    // mtspr 9,3
}

TEST_CASE("simplified mnemonics can be turned off", "[cpu][disasm]") {
    CHECK(raw(0x60000000) == "ori r0,r0,0");
    CHECK(raw(0x4E800020) == "bclr 20,0");
    CHECK(raw(0x7C0802A6) == "mfspr r0,lr");
    CHECK(raw(0x3860FFFF) == "addi r3,r0,-1");
}

TEST_CASE("branch targets are resolved against the instruction's address", "[cpu][disasm]") {
    // bl 0x100, assembled at 0.
    CHECK(dis(0x48000101, 0) == "bl 0x100");
    // The same word 0x2000000 further along points 0x100 further along too.
    CHECK(dis(0x48000101, 0x02000000) == "bl 0x2000100");

    // beq 0x20 == bc 12,2,+0x20, which now prints with its simplified name.
    CHECK(dis(0x41820020, 0x1000) == "beq 0x1020");
    CHECK(raw(0x41820020, 0x1000) == "bc 12,2,0x1020");

    // A backward branch.
    CHECK(dis(0x4BFFFFF8, 0x1000) == "b 0xff8");
}

TEST_CASE("an unknown word is emitted as data", "[cpu][disasm]") {
    CHECK(dis(0x00000000) == ".long 0x00000000");
    CHECK(dis(0x14000000) == ".long 0x14000000");

    // 0xFFFFFFFF looks like it should be invalid and is not.
    CHECK(dis(0xFFFFFFFF) == "fnmadd. f31,f31,f31,f31");
}

TEST_CASE("a function prologue reads correctly", "[cpu][disasm]") {
    // What every compiled Wii U function starts with. If this is wrong,
    // nothing later in the project is debuggable.
    const std::array<u32, 5> prologue{0x9421FFE0, 0x7C0802A6, 0x90010024, 0x93E1001C, 0x7C3F0B78};
    const std::array<std::string_view, 5> expected{
        "stwu r1,-32(r1)", "mflr r0", "stw r0,36(r1)", "stw r31,28(r1)", "mr r31,r1"};

    for (std::size_t i = 0; i < prologue.size(); ++i) {
        CHECK(dis(prologue[i], static_cast<ppcaddr_t>(0x02000000 + i * 4)) == expected[i]);
    }
}
