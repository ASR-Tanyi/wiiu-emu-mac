// rpl-dump — inspect an RPX or RPL.
//
//   rpl-dump game.rpx              sections, imports, exports, file info
//   rpl-dump --disasm game.rpx     also disassemble the executable sections
//   rpl-dump --section .text f.rpx hex dump one section
//
// Phase 5.10, and the reason it comes this early: every later phase debugs
// against real binaries, and "what is actually in this file" needs answering
// without a debugger.

#include <wiiu/cpu/disassembler.hpp>
#include <wiiu/loader/rpl.hpp>

#include <fmt/format.h>

#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace wiiu;
using namespace wiiu::loader;

[[nodiscard]] std::string section_type_name(u32 type) {
    switch (type) {
        case 0:
            return "NULL";
        case sht_progbits:
            return "PROGBITS";
        case sht_symtab:
            return "SYMTAB";
        case sht_strtab:
            return "STRTAB";
        case sht_rela:
            return "RELA";
        case sht_nobits:
            return "NOBITS";
        case sht_rpl_exports:
            return "RPL_EXPORTS";
        case sht_rpl_imports:
            return "RPL_IMPORTS";
        case sht_rpl_crcs:
            return "RPL_CRCS";
        case sht_rpl_fileinfo:
            return "RPL_FILEINFO";
        default:
            return fmt::format("{:#010x}", type);
    }
}

[[nodiscard]] std::string flag_letters(u32 flags) {
    std::string out;
    out += (flags & shf_alloc) != 0 ? 'A' : '-';
    out += (flags & shf_write) != 0 ? 'W' : '-';
    out += (flags & shf_execinstr) != 0 ? 'X' : '-';
    out += (flags & shf_deflated) != 0 ? 'Z' : '-';
    return out;
}

void print_sections(const Rpl& rpl) {
    fmt::print("\nsections:\n");
    fmt::print("  {:<3} {:<18} {:<13} {:<5} {:>10} {:>10}  {}\n",
               "#",
               "name",
               "type",
               "flags",
               "address",
               "size",
               "crc");

    for (std::size_t i = 0; i < rpl.sections.size(); ++i) {
        const Section& section = rpl.sections[i];

        std::string crc = "-";
        if (section.crc_checked) {
            crc = section.crc_ok ? fmt::format("{:#010x} ok", section.recorded_crc)
                                 : fmt::format("{:#010x} MISMATCH", section.recorded_crc);
        }

        fmt::print("  {:<3} {:<18} {:<13} {:<5} {:#010x} {:>10}  {}\n",
                   i,
                   section.name.empty() ? "(none)" : section.name,
                   section_type_name(section.type),
                   flag_letters(section.flags),
                   section.address,
                   section.size,
                   crc);
    }
}

void print_imports(const Rpl& rpl) {
    if (rpl.imports.empty()) {
        return;
    }
    fmt::print("\nimports:\n");
    for (const ImportTable& table : rpl.imports) {
        fmt::print("  {:<20} {:<9} {} entries  (section {})\n",
                   table.library,
                   table.is_function ? "function" : "data",
                   table.count,
                   table.section);
    }
}

void print_exports(const Rpl& rpl) {
    if (rpl.exports.empty()) {
        return;
    }
    fmt::print("\nexports:\n");
    for (const ExportTable& table : rpl.exports) {
        fmt::print("  {} ({}, {} entries)\n",
                   table.section,
                   table.is_function ? "function" : "data",
                   table.entries.size());
        for (const Export& entry : table.entries) {
            fmt::print("    {:#010x}  {}\n", entry.value, entry.name);
        }
    }
}

void print_file_info(const Rpl& rpl) {
    if (!rpl.file_info.has_value()) {
        return;
    }
    const RplFileInfo& info = *rpl.file_info;

    fmt::print("\nfile info:\n");
    fmt::print("  version .......... {:#06x}\n", info.version.value());
    fmt::print("  module ........... {}\n",
               rpl.module_name.empty() ? "(unnamed)" : rpl.module_name);
    fmt::print("  flags ............ {:#010x}{}\n",
               info.flags.value(),
               rpl.is_rpx ? "  (RPL_IS_RPX)" : "");
    fmt::print(
        "  text / data ...... {} / {} bytes\n", info.text_size.value(), info.data_size.value());
    fmt::print("  load size ........ {} bytes\n", info.load_size.value());
    fmt::print(
        "  stack / heap ..... {} / {} bytes\n", info.stack_size.value(), info.heap_size.value());
    fmt::print("  SDA bases ........ {:#010x} / {:#010x}\n",
               info.sda_base.value(),
               info.sda2_base.value());
    fmt::print("  SDK version ...... {:#010x}\n", info.sdk_version.value());
}

void disassemble_section(const Section& section) {
    fmt::print("\ndisassembly of {}:\n", section.name);
    for (std::size_t offset = 0; offset + 4 <= section.data.size(); offset += 4) {
        const u32 word = (static_cast<u32>(section.data[offset]) << 24) |
                         (static_cast<u32>(section.data[offset + 1]) << 16) |
                         (static_cast<u32>(section.data[offset + 2]) << 8) |
                         static_cast<u32>(section.data[offset + 3]);
        const auto address = static_cast<ppcaddr_t>(section.address + offset);
        fmt::print("  {:08x}:  {:08x}  {}\n", address, word, cpu::disassemble(word, address));
    }
}

void hex_dump(const Section& section) {
    fmt::print("\ncontents of {} ({} bytes):\n", section.name, section.data.size());
    for (std::size_t offset = 0; offset < section.data.size(); offset += 16) {
        fmt::print("  {:08x} ", static_cast<u32>(section.address + offset));
        for (std::size_t i = 0; i < 16; ++i) {
            if (offset + i < section.data.size()) {
                fmt::print(" {:02x}", section.data[offset + i]);
            } else {
                fmt::print("   ");
            }
        }
        fmt::print("  ");
        for (std::size_t i = 0; i < 16 && offset + i < section.data.size(); ++i) {
            const u8 byte = section.data[offset + i];
            fmt::print("{}", (byte >= 0x20 && byte < 0x7F) ? static_cast<char>(byte) : '.');
        }
        fmt::print("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    bool disasm = false;
    std::string_view wanted_section;
    std::string_view path;

    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--disasm" || args[i] == "-d") {
            disasm = true;
        } else if ((args[i] == "--section" || args[i] == "-s") && i + 1 < args.size()) {
            wanted_section = args[++i];
        } else if (args[i].starts_with('-')) {
            fmt::print(stderr, "rpl-dump: unknown option '{}'\n", args[i]);
            return 2;
        } else {
            path = args[i];
        }
    }

    if (path.empty()) {
        fmt::print(stderr, "usage: rpl-dump [--disasm] [--section NAME] FILE\n");
        return 2;
    }

    Rpl rpl;
    const ParseResult result = parse_file(std::string{path}, rpl);
    if (!result.ok) {
        fmt::print(stderr, "rpl-dump: {}: {}\n", path, result.error);
        return 1;
    }

    fmt::print("{}\n", path);
    fmt::print("  kind ............. {}\n", rpl.is_rpx ? "RPX (executable)" : "RPL (library)");
    fmt::print("  entry ............ {:#010x}\n", rpl.entry);
    fmt::print("  sections ......... {}\n", rpl.sections.size());

    for (const std::string& warning : result.warnings) {
        fmt::print("  warning: {}\n", warning);
    }

    print_file_info(rpl);
    print_sections(rpl);
    print_imports(rpl);
    print_exports(rpl);

    if (!wanted_section.empty()) {
        const Section* const section = rpl.find_section(wanted_section);
        if (section == nullptr) {
            fmt::print(stderr, "rpl-dump: no section named '{}'\n", wanted_section);
            return 1;
        }
        hex_dump(*section);
    }

    if (disasm) {
        for (const Section& section : rpl.sections) {
            if (section.is_code() && !section.data.empty()) {
                disassemble_section(section);
            }
        }
    }

    return 0;
}
