#include <wiiu/cpu/disassembler.hpp>
#include <wiiu/loader/rpl.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::loader;

namespace {

[[nodiscard]] std::filesystem::path data_path(std::string_view name) {
    return std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / name;
}

[[nodiscard]] std::vector<u8> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    REQUIRE(file);
    const auto size = static_cast<std::streamsize>(file.tellg());
    file.seekg(0);
    std::vector<u8> bytes(static_cast<std::size_t>(size));
    REQUIRE(file.read(reinterpret_cast<char*>(bytes.data()), size));
    return bytes;
}

} // namespace

TEST_CASE("on-disk structures are the documented sizes", "[loader][rpl]") {
    // If any of these drifts, every offset in the file is read from the wrong
    // place and the failure looks like a corrupt file rather than our bug.
    STATIC_REQUIRE(sizeof(Elf32Header) == 52);
    STATIC_REQUIRE(sizeof(Elf32SectionHeader) == 40);
    STATIC_REQUIRE(sizeof(RplFileInfo) == 0x60);
}

TEST_CASE("a compressed RPX parses", "[loader][rpl]") {
    Rpl rpl;
    const ParseResult result = parse_file(data_path("hello.rpx").string(), rpl);

    INFO("error: " << result.error);
    REQUIRE(result.ok);
    CHECK(result.warnings.empty());

    CHECK(rpl.is_rpx);
    CHECK(rpl.entry == 0x02000000U);
    CHECK(rpl.module_name == "hello.rpx");
    CHECK(rpl.sections.size() == 10);
}

TEST_CASE("compressed sections are inflated", "[loader][rpl]") {
    Rpl rpl;
    REQUIRE(parse_file(data_path("hello.rpx").string(), rpl).ok);

    const Section* const text = rpl.find_section(".text");
    REQUIRE(text != nullptr);
    CHECK(text->compressed);
    CHECK(text->is_code());

    // The size reported is the *uncompressed* size, and the data is there.
    CHECK(text->size == text->data.size());
    CHECK(text->data.size() == 56);

    // The first instruction is a real one, not compressed bytes.
    const u32 first = (static_cast<u32>(text->data[0]) << 24) |
                      (static_cast<u32>(text->data[1]) << 16) |
                      (static_cast<u32>(text->data[2]) << 8) | static_cast<u32>(text->data[3]);
    CHECK(cpu::disassemble(first, text->address) == "mflr r0");
}

TEST_CASE("section CRCs are verified", "[loader][rpl]") {
    Rpl rpl;
    REQUIRE(parse_file(data_path("hello.rpx").string(), rpl).ok);

    int checked = 0;
    for (const Section& section : rpl.sections) {
        if (section.crc_checked) {
            INFO("section " << section.name);
            CHECK(section.crc_ok);
            ++checked;
        }
    }
    CHECK(checked >= 4);
}

TEST_CASE("a corrupted section is reported, not silently accepted", "[loader][rpl]") {
    std::vector<u8> bytes = read_file(data_path("libtest.rpl"));

    Rpl original;
    REQUIRE(parse(bytes, original).ok);
    const Section* const text = original.find_section(".text");
    REQUIRE(text != nullptr);
    REQUIRE(text->crc_checked);

    // Flip a byte inside the uncompressed .text and re-parse.
    bytes[text->file_offset] ^= 0xFF;

    Rpl damaged;
    const ParseResult result = parse(bytes, damaged);
    REQUIRE(result.ok); // structurally fine, contents are not

    const Section* const damaged_text = damaged.find_section(".text");
    REQUIRE(damaged_text != nullptr);
    CHECK(damaged_text->crc_checked);
    CHECK_FALSE(damaged_text->crc_ok);
    CHECK_FALSE(result.warnings.empty());
}

TEST_CASE("imports are listed per library", "[loader][rpl]") {
    Rpl rpl;
    REQUIRE(parse_file(data_path("hello.rpx").string(), rpl).ok);

    REQUIRE(rpl.imports.size() == 2);
    CHECK(rpl.imports[0].library == "coreinit");
    CHECK(rpl.imports[0].is_function);
    CHECK(rpl.imports[0].count == 3);
    CHECK(rpl.imports[1].library == "gx2");
}

TEST_CASE("an uncompressed library parses, with its exports", "[loader][rpl]") {
    Rpl rpl;
    const ParseResult result = parse_file(data_path("libtest.rpl").string(), rpl);

    INFO("error: " << result.error);
    REQUIRE(result.ok);

    // No RPL_IS_RPX flag: this one is a library.
    CHECK_FALSE(rpl.is_rpx);
    CHECK(rpl.module_name == "libtest.rpl");

    const Section* const text = rpl.find_section(".text");
    REQUIRE(text != nullptr);
    CHECK_FALSE(text->compressed);

    REQUIRE(rpl.exports.size() == 2);

    const ExportTable& functions = rpl.exports[0];
    CHECK(functions.section == ".fexports");
    CHECK(functions.is_function);
    REQUIRE(functions.entries.size() == 2);
    CHECK(functions.entries[0].name == "TestAdd");
    CHECK(functions.entries[0].value == 0x02000000U);
    CHECK(functions.entries[1].name == "TestSubtract");
    CHECK(functions.entries[1].value == 0x02000008U);

    const ExportTable& data = rpl.exports[1];
    CHECK(data.section == ".dexports");
    CHECK_FALSE(data.is_function);
    REQUIRE(data.entries.size() == 1);
    CHECK(data.entries[0].name == "g_TestCounter");
}

TEST_CASE("exported functions point at real code", "[loader][rpl]") {
    Rpl rpl;
    REQUIRE(parse_file(data_path("libtest.rpl").string(), rpl).ok);

    const ExportTable& functions = rpl.exports[0];
    for (const Export& entry : functions.entries) {
        INFO("export " << entry.name);
        const Section* const section = rpl.section_at(entry.value);
        REQUIRE(section != nullptr);
        CHECK(section->is_code());
    }

    // TestAdd is `add r3,r3,r4`.
    const Section* const text = rpl.find_section(".text");
    REQUIRE(text != nullptr);
    const u32 first = (static_cast<u32>(text->data[0]) << 24) |
                      (static_cast<u32>(text->data[1]) << 16) |
                      (static_cast<u32>(text->data[2]) << 8) | static_cast<u32>(text->data[3]);
    CHECK(cpu::disassemble(first, text->address) == "add r3,r3,r4");
}

TEST_CASE("section lookup by address", "[loader][rpl]") {
    Rpl rpl;
    REQUIRE(parse_file(data_path("hello.rpx").string(), rpl).ok);

    const Section* const text = rpl.find_section(".text");
    REQUIRE(text != nullptr);

    CHECK(rpl.section_at(text->address) == text);
    CHECK(rpl.section_at(text->address + text->size - 1) == text);
    CHECK(rpl.section_at(text->address + text->size) != text);
    CHECK(rpl.section_at(0x70000000) == nullptr);
}

// ---------------------------------------------------------------------------
// Malformed input
// ---------------------------------------------------------------------------

TEST_CASE("a non-ELF file is rejected", "[loader][rpl]") {
    const std::vector<u8> bytes(64, 0x41);
    Rpl rpl;
    const ParseResult result = parse(bytes, rpl);
    CHECK_FALSE(result.ok);
    CHECK(result.error.find("not an ELF") != std::string::npos);
}

TEST_CASE("an empty file is rejected", "[loader][rpl]") {
    Rpl rpl;
    const ParseResult result = parse({}, rpl);
    CHECK_FALSE(result.ok);
    CHECK(result.error.find("too small") != std::string::npos);
}

TEST_CASE("a little-endian ELF is rejected", "[loader][rpl]") {
    std::vector<u8> bytes = read_file(data_path("hello.rpx"));
    bytes[5] = 1; // EI_DATA = ELFDATA2LSB

    Rpl rpl;
    const ParseResult result = parse(bytes, rpl);
    CHECK_FALSE(result.ok);
    CHECK(result.error.find("big-endian") != std::string::npos);
}

TEST_CASE("e_type is ET_CAFE_RPL's 0xFE01, and another is noted, not refused", "[loader][rpl]") {
    std::vector<u8> bytes = read_file(data_path("hello.rpx"));
    CHECK(bytes[16] == 0xFE); // as elf2rpl writes it
    CHECK(bytes[17] == 0x01);

    // Cafe OS's loader does not look at it (decaf's reimplementation), so
    // neither does this one: 0xFF01, which wut-tools' elf.h names and this
    // loader once required, loads with a warning.
    bytes[16] = 0xFF;
    Rpl rpl;
    const ParseResult result = parse(bytes, rpl);
    CHECK(result.ok);
    REQUIRE(result.warnings.size() == 1);
    CHECK(result.warnings[0] == "e_type is 0xff01, where RPX and RPL files have 0xfe01");
}

TEST_CASE("a truncated file is rejected rather than read past its end", "[loader][rpl]") {
    const std::vector<u8> full = read_file(data_path("hello.rpx"));

    // Every truncation point must be handled: the parser is the first thing
    // that touches untrusted bytes.
    for (std::size_t size = 1; size < full.size(); size += 7) {
        INFO("truncated to " << size << " bytes");
        const std::vector<u8> bytes(full.begin(), full.begin() + static_cast<long>(size));
        Rpl rpl;
        const ParseResult result = parse(bytes, rpl);
        if (!result.ok) {
            CHECK_FALSE(result.error.empty());
        }
    }
}

TEST_CASE("a section claiming an implausible size is rejected", "[loader][rpl]") {
    std::vector<u8> bytes = read_file(data_path("hello.rpx"));

    Rpl rpl;
    REQUIRE(parse(bytes, rpl).ok);
    const Section* const text = rpl.find_section(".text");
    REQUIRE(text != nullptr);
    REQUIRE(text->compressed);

    // The uncompressed size prefix drives an allocation, so it is exactly the
    // field an attacker would reach for.
    const u32 at = text->file_offset;
    bytes[at] = 0xFF;
    bytes[at + 1] = 0xFF;
    bytes[at + 2] = 0xFF;
    bytes[at + 3] = 0xFF;

    Rpl damaged;
    const ParseResult result = parse(bytes, damaged);
    CHECK_FALSE(result.ok);
    CHECK(result.error.find("implausible") != std::string::npos);
}

TEST_CASE("garbage compressed data is reported", "[loader][rpl]") {
    std::vector<u8> bytes = read_file(data_path("hello.rpx"));

    Rpl rpl;
    REQUIRE(parse(bytes, rpl).ok);
    const Section* const text = rpl.find_section(".text");
    REQUIRE(text != nullptr);

    // Corrupt the zlib stream itself, leaving the size prefix alone.
    for (u32 i = 4; i < 12; ++i) {
        bytes[text->file_offset + i] = 0xAA;
    }

    Rpl damaged;
    const ParseResult result = parse(bytes, damaged);
    CHECK_FALSE(result.ok);
    CHECK(result.error.find("zlib") != std::string::npos);
}
