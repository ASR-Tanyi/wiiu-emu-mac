#include <wiiu/cpu/disassembler.hpp>
#include <wiiu/loader/relocations.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::loader;

namespace {

[[nodiscard]] u32 relocate(RelocationType type, u32 word, ppcaddr_t target, ppcaddr_t at = 0) {
    u32 out = 0;
    REQUIRE(apply_relocation(type, word, target, at, out));
    return out;
}

} // namespace

TEST_CASE("the adjusted high half compensates for the low half's sign", "[loader][reloc]") {
    // This is the subtle one. `lis` loads the high half and `ori`/`addi` adds
    // the low half, but the low half is *sign-extended*. When it is 0x8000 or
    // above it subtracts 0x10000, so HA adds one to compensate.
    //
    // Using the plain top 16 bits works for exactly half of all addresses,
    // which is the worst way for a bug like this to behave.
    STATIC_REQUIRE(high_adjusted(0x12345678) == 0x1234); // low half positive
    STATIC_REQUIRE(high_adjusted(0x1234FFFF) == 0x1235); // low half negative
    STATIC_REQUIRE(high_adjusted(0x12348000) == 0x1235); // exactly the boundary
    STATIC_REQUIRE(high_adjusted(0x12347FFF) == 0x1234); // just below it
    STATIC_REQUIRE(high_adjusted(0x0000FFFF) == 0x0001);
}

TEST_CASE("lis/addi reassembles the right address", "[loader][reloc]") {
    // The property that matters: whatever the address, the pair of
    // instructions has to compute it exactly.
    for (const u32 address : {0x02000000U, 0x1000FFFFU, 0x10008000U, 0xFFFFFFFFU, 0x00000001U}) {
        INFO("address " << std::to_string(address));

        const u16 high = high_adjusted(address);
        const auto low = static_cast<s16>(address & 0xFFFF);

        // lis rX,high ; addi rX,rX,low  -- the addi sign-extends.
        const u32 reassembled = (static_cast<u32>(high) << 16) + static_cast<u32>(low);
        CHECK(reassembled == address);
    }
}

TEST_CASE("ADDR32 writes the whole address", "[loader][reloc]") {
    CHECK(relocate(RelocationType::Addr32, 0x00000000, 0x02001234) == 0x02001234U);
    CHECK(relocate(RelocationType::Addr32, 0xFFFFFFFF, 0x10000000) == 0x10000000U);
}

TEST_CASE("the 16-bit forms keep the instruction's opcode", "[loader][reloc]") {
    // lis r3,0 == addis r3,r0,0
    const u32 lis = 0x3C600000;
    const u32 patched = relocate(RelocationType::Addr16Ha, lis, 0x1000FFFF);
    CHECK((patched >> 16) == (lis >> 16)); // opcode and registers untouched
    CHECK((patched & 0xFFFF) == 0x1001U);  // adjusted high half
    CHECK(cpu::disassemble(patched) == "lis r3,4097");

    // addi r3,r3,0
    const u32 addi = 0x38630000;
    const u32 low = relocate(RelocationType::Addr16Lo, addi, 0x1000FFFF);
    CHECK((low >> 16) == (addi >> 16));
    CHECK((low & 0xFFFF) == 0xFFFFU);

    // Unadjusted HI is the plain top half.
    CHECK((relocate(RelocationType::Addr16Hi, lis, 0x1000FFFF) & 0xFFFF) == 0x1000U);
}

TEST_CASE("REL24 computes a branch displacement", "[loader][reloc]") {
    // bl 0 at 0x02000000, retargeted to 0x02000100.
    const u32 bl = 0x48000001;
    const u32 patched = relocate(RelocationType::Rel24, bl, 0x02000100, 0x02000000);

    CHECK(cpu::disassemble(patched, 0x02000000) == "bl 0x2000100");
    CHECK((patched & 1) == 1U); // the link bit survives

    // Backwards too.
    const u32 back = relocate(RelocationType::Rel24, bl, 0x02000000, 0x02000100);
    CHECK(cpu::disassemble(back, 0x02000100) == "bl 0x2000000");
}

TEST_CASE("a branch out of range is refused rather than truncated", "[loader][reloc]") {
    // A 26-bit signed displacement reaches +/-32 MiB. Beyond that the linker
    // needed a trampoline; silently wrapping would produce a branch into
    // nowhere that looks like a working one.
    u32 out = 0;
    CHECK_FALSE(apply_relocation(RelocationType::Rel24, 0x48000001, 0x10000000, 0x02000000, out));
    CHECK(apply_relocation(RelocationType::Rel24, 0x48000001, 0x03FFFFFC, 0x02000000, out));

    CHECK_FALSE(apply_relocation(RelocationType::Rel14, 0x41820000, 0x02010000, 0x02000000, out));
    CHECK(apply_relocation(RelocationType::Rel14, 0x41820000, 0x02000100, 0x02000000, out));
}

TEST_CASE("relocations move a section to a new base", "[loader][reloc]") {
    Section section;
    section.name = ".text";
    section.address = 0x02000000;

    // Two words: an ADDR32 pointer and a lis/addi pair.
    const std::vector<u32> words{0x02001000, 0x3C600000, 0x38630000};
    for (const u32 word : words) {
        section.data.push_back(static_cast<u8>(word >> 24));
        section.data.push_back(static_cast<u8>(word >> 16));
        section.data.push_back(static_cast<u8>(word >> 8));
        section.data.push_back(static_cast<u8>(word));
    }
    section.size = static_cast<u32>(section.data.size());

    // The 16-bit entries address the immediate field, which is the
    // instruction address plus two -- exactly as an assembler emits them.
    const std::vector<Relocation> relocations{
        {0x02000000, 0, RelocationType::Addr32, 0x02001000},
        {0x02000006, 0, RelocationType::Addr16Ha, 0x0200FFFF},
        {0x0200000A, 0, RelocationType::Addr16Lo, 0x0200FFFF},
    };

    // Loaded 0x1000 higher than it was linked.
    RelocationContext context;
    context.delta = 0x1000;

    const RelocationResult result = apply_relocations(relocations, section, context);
    REQUIRE(result.ok);
    CHECK(result.applied == 3);
    CHECK(result.warnings.empty());

    const auto word_at = [&section](std::size_t index) {
        return (static_cast<u32>(section.data[index * 4]) << 24) |
               (static_cast<u32>(section.data[index * 4 + 1]) << 16) |
               (static_cast<u32>(section.data[index * 4 + 2]) << 8) |
               static_cast<u32>(section.data[index * 4 + 3]);
    };

    CHECK(word_at(0) == 0x02002000U); // pointer moved by the delta

    const u32 high = word_at(1) & 0xFFFF;
    const auto low = static_cast<s16>(word_at(2) & 0xFFFF);
    CHECK((static_cast<u32>(high) << 16) + static_cast<u32>(low) == 0x02010FFFU);
}

TEST_CASE("an unresolved import is reported, not silently skipped", "[loader][reloc]") {
    Section section;
    section.name = ".text";
    section.address = 0x02000000;
    section.data.assign(4, 0);
    section.size = 4;

    const std::vector<Relocation> relocations{
        {0x02000000, 7, RelocationType::Addr32, 0},
    };

    RelocationContext context;
    context.resolve = [](u32, ppcaddr_t&) { return false; };

    const RelocationResult result = apply_relocations(relocations, section, context);
    REQUIRE(result.ok);
    CHECK(result.applied == 0);
    REQUIRE(result.warnings.size() == 1);
    CHECK(result.warnings[0].find("unresolved symbol 7") != std::string::npos);
}

TEST_CASE("a relocation outside the section is reported", "[loader][reloc]") {
    Section section;
    section.name = ".text";
    section.address = 0x02000000;
    section.data.assign(8, 0);
    section.size = 8;

    const std::vector<Relocation> relocations{
        {0x01000000, 0, RelocationType::Addr32, 0}, // before
        {0x02000100, 0, RelocationType::Addr32, 0}, // after
    };

    const RelocationResult result = apply_relocations(relocations, section, {});
    REQUIRE(result.ok);
    CHECK(result.applied == 0);
    CHECK(result.warnings.size() == 2);
}

TEST_CASE("a halfword relocation leaves the opcode beside it alone", "[loader][reloc]") {
    Section section;
    section.name = ".text";
    section.address = 0x02000000;

    // lis r3,0 ; addi r3,r3,0
    for (const u32 word : {0x3C600000U, 0x38630000U}) {
        section.data.push_back(static_cast<u8>(word >> 24));
        section.data.push_back(static_cast<u8>(word >> 16));
        section.data.push_back(static_cast<u8>(word >> 8));
        section.data.push_back(static_cast<u8>(word));
    }
    section.size = static_cast<u32>(section.data.size());

    const std::vector<Relocation> relocations{
        {0x02000002, 0, RelocationType::Addr16Ha, 0x10000000},
        {0x02000006, 0, RelocationType::Addr16Lo, 0x10000000},
    };

    const RelocationResult result = apply_relocations(relocations, section, {});
    REQUIRE(result.applied == 2);

    // Both opcodes survive intact. Treating these as 32-bit writes would put
    // the high half into the middle of the first instruction and corrupt both.
    CHECK(section.data[0] == 0x3C);
    CHECK(section.data[1] == 0x60);
    CHECK(section.data[4] == 0x38);
    CHECK(section.data[5] == 0x63);
    CHECK(section.data[2] == 0x10); // the patched immediate
    CHECK(section.data[3] == 0x00);
}

TEST_CASE("relocation entries parse", "[loader][reloc]") {
    Section section;
    // offset 0x02000000, info (symbol 3, type ADDR16_HA), addend 0x40
    const std::vector<u8> bytes{
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x40};
    section.data = bytes;

    const auto relocations = parse_relocations(section);
    REQUIRE(relocations.size() == 1);
    CHECK(relocations[0].offset == 0x02000000U);
    CHECK(relocations[0].symbol == 3U);
    CHECK(relocations[0].type == RelocationType::Addr16Ha);
    CHECK(relocations[0].addend == 0x40);
    CHECK(relocation_name(relocations[0].type) == "ADDR16_HA");
}

TEST_CASE("a REL32 split by elf2rpl is two Green Hills halves of S + A - P", "[loader][reloc]") {
    // A switch's jump table entry in .rodata: the displacement from the
    // entry to a case label in .text. elf2rpl writes it as GHS_REL16_HI at
    // the word and GHS_REL16_LO at the word + 2, the LO's addend 2 more.
    Section section;
    section.name = ".rodata";
    section.address = 0x10000100;
    section.data.assign(8, 0);
    section.size = 8;
    constexpr u32 rodata_symbol = 3;
    constexpr u32 text_symbol = 4;
    constexpr s32 label = 0x5A74; // the case label, as an addend on .text's symbol
    const std::vector<Relocation> relocations{
        {0x10000104, text_symbol, RelocationType::GhsRel16Hi, label},
        {0x10000106, text_symbol, RelocationType::GhsRel16Lo, label + 2},
        // And the HA form, which elf2rpl does not write but the loader has.
        {0x10000102, rodata_symbol, RelocationType::GhsRel16Ha, 0x8102},
    };
    RelocationContext context;
    context.resolve = [](u32 symbol, ppcaddr_t& out) {
        out = symbol == text_symbol ? 0x02000000 : 0x10010000;
        return true;
    };
    const RelocationResult result = apply_relocations(relocations, section, context);
    CHECK(result.applied == 3);
    CHECK(result.warnings.empty());
    const auto word = [&section](std::size_t at) {
        return (static_cast<u32>(section.data[at]) << 24) |
               (static_cast<u32>(section.data[at + 1]) << 16) |
               (static_cast<u32>(section.data[at + 2]) << 8) | section.data[at + 3];
    };
    // 0x02005A74 - 0x10000104, as a 32-bit displacement.
    CHECK(word(4) == 0x02005A74U - 0x10000104U);
    // 0x10018102 - 0x10000102 = 0x18000: HA rounds it up to 2, where HI is 1.
    CHECK(((static_cast<u32>(section.data[2]) << 8) | section.data[3]) == 0x0002);
}

TEST_CASE("a branch to an undefined weak symbol is made a branch to itself", "[loader][reloc]") {
    // wut's crt: `if (&__init_wut_socket) __init_wut_socket();` with no
    // socket library linked. Zero is out of a branch's reach from 0x02000000;
    // Cafe OS's loader branches to the branch instead (decaf).
    Section section;
    section.name = ".text";
    section.address = 0x02000D1C;
    section.data = {0x48, 0x00, 0x00, 0x01}; // bl
    section.size = 4;
    const std::vector<Relocation> relocations{{0x02000D1C, 9, RelocationType::Rel24, 0}};
    RelocationContext context;
    context.resolve = [](u32, ppcaddr_t& out) {
        out = 0;
        return true;
    };
    context.undefined_weak = [](u32 symbol) { return symbol == 9; };
    const RelocationResult result = apply_relocations(relocations, section, context);
    CHECK(result.applied == 1);
    CHECK(result.warnings.empty());
    CHECK(section.data == std::vector<u8>{0x48, 0x00, 0x00, 0x01}); // bl .

    // A symbol that is merely zero, not undefined and weak, is still out of reach.
    context.undefined_weak = [](u32) { return false; };
    const RelocationResult refused = apply_relocations(relocations, section, context);
    CHECK(refused.applied == 0);
    CHECK(refused.warnings.size() == 1);
}
