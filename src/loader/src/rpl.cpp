#include <wiiu/common/log.hpp>
#include <wiiu/loader/rpl.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <zlib.h>

namespace wiiu::loader {
namespace {

using wiiu::log::Channel;

/// Read a structure from a byte span, checked. Guest structures are declared
/// with be<> fields, so this is a copy and nothing more.
template <typename T>
[[nodiscard]] bool read_struct(std::span<const u8> bytes, std::size_t offset, T& out) {
    if (offset + sizeof(T) > bytes.size()) {
        return false;
    }
    std::memcpy(&out, bytes.data() + offset, sizeof(T));
    return true;
}

/// A null-terminated string inside a section, bounded by the section's end.
[[nodiscard]] std::string read_string(std::span<const u8> data, std::size_t offset) {
    if (offset >= data.size()) {
        return {};
    }
    const auto* const begin = reinterpret_cast<const char*>(data.data()) + offset;
    const std::size_t available = data.size() - offset;
    const std::size_t length = ::strnlen(begin, available);
    return std::string{begin, length};
}

} // namespace

u32 crc32(std::span<const u8> bytes) noexcept {
    return static_cast<u32>(::crc32(0, bytes.data(), static_cast<uInt>(bytes.size())));
}

bool inflate_section(std::span<const u8> input, std::vector<u8>& out, std::string& error) {
    if (input.size() < 4) {
        error = "deflated section is too small to hold its size prefix";
        return false;
    }

    // The uncompressed size is stored big-endian ahead of the zlib stream.
    u32be stored_size{};
    std::memcpy(&stored_size, input.data(), sizeof(stored_size));
    const u32 expected = stored_size.value();

    // A corrupt header could otherwise ask for an arbitrary allocation.
    constexpr u32 sane_limit = 256U * 1024 * 1024;
    if (expected > sane_limit) {
        error = fmt::format("deflated section claims {} bytes, which is implausible", expected);
        return false;
    }

    out.assign(expected, 0);
    uLongf produced = expected;

    const uLong available = input.size() - 4;
    const int result = ::uncompress(out.data(), &produced, input.data() + 4, available);
    if (result != Z_OK) {
        error = fmt::format("zlib error {}", result);
        return false;
    }
    if (produced != expected) {
        error = fmt::format("inflated to {} bytes, header said {}", produced, expected);
        return false;
    }
    return true;
}

const Section* Rpl::find_section(std::string_view name) const noexcept {
    const auto it = std::ranges::find_if(
        sections, [name](const Section& section) { return section.name == name; });
    return it == sections.end() ? nullptr : &*it;
}

const Section* Rpl::find_section(u32 index) const noexcept {
    return index < sections.size() ? &sections[index] : nullptr;
}

std::string Rpl::import_library(const Symbol& symbol) const {
    const Section* const section = find_section(symbol.section_index);
    if (section == nullptr || section->type != sht_rpl_imports) {
        return {};
    }
    const auto underscore = section->name.find('_');
    return underscore == std::string::npos ? section->name : section->name.substr(underscore + 1);
}

bool Rpl::is_data_import(const Symbol& symbol) const {
    const Section* const section = find_section(symbol.section_index);
    return section != nullptr && section->type == sht_rpl_imports &&
           section->name.starts_with(".dimport_");
}

const Section* Rpl::section_at(ppcaddr_t address) const noexcept {
    for (const Section& section : sections) {
        if (section.occupies_memory() && section.size != 0 && address >= section.address &&
            address - section.address < section.size) {
            return &section;
        }
    }
    return nullptr;
}

// --------------------------------------------------------------------------
// Parsing
// --------------------------------------------------------------------------

ParseResult parse(std::span<const u8> bytes, Rpl& out) {
    ParseResult result;
    out = Rpl{};

    Elf32Header header{};
    if (!read_struct(bytes, 0, header)) {
        result.error = "file is too small to contain an ELF header";
        return result;
    }

    if (header.ident[0] != 0x7F || header.ident[1] != 'E' || header.ident[2] != 'L' ||
        header.ident[3] != 'F') {
        result.error = "not an ELF file";
        return result;
    }
    if (header.ident[4] != 1) {
        result.error = "not ELF32";
        return result;
    }
    if (header.ident[5] != 2) {
        result.error = "not big-endian";
        return result;
    }
    if (header.type != et_cafe_rpl) {
        // Not fatal either: Cafe OS's loader does not look at it (decaf's
        // reimplementation of it checks the class, byte order, OS/ABI,
        // version, machine and section count, and not the type).
        result.warnings.push_back(
            fmt::format("e_type is {:#06x}, where RPX and RPL files have {:#06x}",
                        header.type.value(),
                        et_cafe_rpl));
    }
    if (header.machine != em_ppc) {
        result.error =
            fmt::format("e_machine is {}, expected {} (EM_PPC)", header.machine.value(), em_ppc);
        return result;
    }

    if (header.ident[7] != ei_osabi_cafe || header.ident[8] != ei_abiversion_cafe) {
        // Not fatal: the marker is a convention rather than something the
        // loader depends on.
        result.warnings.emplace_back("missing the 0xCAFE marker in e_ident");
    }

    // RPX files carry no program headers; section headers drive loading.
    if (header.phnum != 0) {
        result.warnings.push_back(fmt::format(
            "{} program headers present; RPX files normally have none", header.phnum.value()));
    }

    if (header.shentsize != sizeof(Elf32SectionHeader)) {
        result.error = fmt::format("section header size is {}, expected {}",
                                   header.shentsize.value(),
                                   sizeof(Elf32SectionHeader));
        return result;
    }

    const u32 section_count = header.shnum;
    const u32 table_offset = header.shoff;
    if (static_cast<u64>(table_offset) +
            static_cast<u64>(section_count) * sizeof(Elf32SectionHeader) >
        bytes.size()) {
        result.error = "section header table runs past the end of the file";
        return result;
    }

    out.entry = header.entry;

    // --- read the raw headers ---------------------------------------------
    std::vector<Elf32SectionHeader> raw(section_count);
    for (u32 i = 0; i < section_count; ++i) {
        if (!read_struct(bytes, table_offset + i * sizeof(Elf32SectionHeader), raw[i])) {
            result.error = fmt::format("section header {} is truncated", i);
            return result;
        }
    }

    // --- load contents -----------------------------------------------------
    out.sections.resize(section_count);
    for (u32 i = 0; i < section_count; ++i) {
        const Elf32SectionHeader& source = raw[i];
        Section& section = out.sections[i];

        section.type = source.type;
        section.flags = source.flags;
        section.address = source.addr;
        section.file_offset = source.offset;
        section.link = source.link;
        section.info = source.info;
        section.alignment = source.addralign;
        section.entry_size = source.entsize;
        section.compressed = (source.flags & shf_deflated) != 0;
        section.size = source.size;

        if (source.type == sht_nobits || source.size == 0) {
            continue;
        }

        const u64 end = static_cast<u64>(source.offset) + source.size;
        if (end > bytes.size()) {
            result.error = fmt::format("section {} runs past the end of the file", i);
            return result;
        }

        const auto stored = bytes.subspan(source.offset, source.size);

        if (section.compressed) {
            std::string error;
            if (!inflate_section(stored, section.data, error)) {
                result.error = fmt::format("section {}: {}", i, error);
                return result;
            }
            section.size = static_cast<u32>(section.data.size());
        } else {
            section.data.assign(stored.begin(), stored.end());
        }
    }

    // --- names -------------------------------------------------------------
    if (header.shstrndx < section_count) {
        const std::vector<u8>& strings = out.sections[header.shstrndx].data;
        for (u32 i = 0; i < section_count; ++i) {
            out.sections[i].name = read_string(strings, raw[i].name);
        }
    } else {
        result.warnings.emplace_back("section name table index is out of range");
    }

    // --- CRCs --------------------------------------------------------------
    // One CRC32 per section, in section order. The CRCs section's own entry is
    // zero, since it cannot contain its own checksum.
    for (u32 i = 0; i < section_count; ++i) {
        if (out.sections[i].type != sht_rpl_crcs) {
            continue;
        }
        const std::vector<u8>& table = out.sections[i].data;
        const std::size_t entries = table.size() / sizeof(u32be);

        for (u32 j = 0; j < section_count && j < entries; ++j) {
            u32be recorded{};
            std::memcpy(&recorded, table.data() + j * sizeof(u32be), sizeof(recorded));

            Section& section = out.sections[j];
            section.recorded_crc = recorded.value();

            if (j == i || recorded == 0) {
                continue; // the CRC section itself, or an unchecksummed one
            }

            section.crc_checked = true;
            section.crc_ok = crc32(section.data) == section.recorded_crc;
            if (!section.crc_ok) {
                result.warnings.push_back(
                    fmt::format("section '{}' CRC mismatch: file says {:#010x}, contents give "
                                "{:#010x}",
                                section.name,
                                section.recorded_crc,
                                crc32(section.data)));
            }
        }
        break;
    }

    // --- file info ---------------------------------------------------------
    for (const Section& section : out.sections) {
        if (section.type != sht_rpl_fileinfo || section.data.size() < sizeof(RplFileInfo)) {
            continue;
        }
        RplFileInfo info{};
        std::memcpy(&info, section.data.data(), sizeof(info));
        out.file_info = info;
        out.flags = info.flags;
        out.is_rpx = (info.flags & rpl_is_rpx) != 0;

        // The filename field indexes a string table that follows the struct.
        if (info.filename >= sizeof(RplFileInfo)) {
            out.module_name = read_string(section.data, info.filename);
        }
        break;
    }

    // --- symbols -----------------------------------------------------------
    for (const Section& section : out.sections) {
        if (section.type != sht_symtab) {
            continue;
        }
        const Section* const strings = out.find_section(section.link);
        if (strings == nullptr) {
            result.warnings.emplace_back("symbol table has no string table");
            continue;
        }

        const std::size_t count = section.data.size() / sizeof(Elf32Symbol);
        out.symbols.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            Elf32Symbol entry{};
            std::memcpy(&entry, section.data.data() + i * sizeof(Elf32Symbol), sizeof(entry));

            Symbol symbol;
            symbol.name = read_string(strings->data, entry.name);
            symbol.value = entry.value;
            symbol.size = entry.size;
            symbol.info = entry.info;
            symbol.section_index = entry.shndx;
            out.symbols.push_back(std::move(symbol));
        }
        break;
    }

    // --- exports and imports ----------------------------------------------
    for (const Section& section : out.sections) {
        if (section.type == sht_rpl_exports) {
            if (section.data.size() < 8) {
                result.warnings.push_back(
                    fmt::format("export section '{}' is too small", section.name));
                continue;
            }

            ExportTable table;
            table.section = section.name;
            table.is_function = section.name.starts_with(".fexport");

            u32be count{};
            u32be signature{};
            std::memcpy(&count, section.data.data(), sizeof(count));
            std::memcpy(&signature, section.data.data() + 4, sizeof(signature));
            table.signature = signature;

            // Eight bytes of header, then eight per entry: value and a name
            // offset relative to the start of this section.
            for (u32 i = 0; i < count; ++i) {
                const std::size_t at = 8 + static_cast<std::size_t>(i) * 8;
                if (at + 8 > section.data.size()) {
                    result.warnings.push_back(
                        fmt::format("export section '{}' claims {} entries but is truncated",
                                    section.name,
                                    count.value()));
                    break;
                }
                u32be value{};
                u32be name_offset{};
                std::memcpy(&value, section.data.data() + at, sizeof(value));
                std::memcpy(&name_offset, section.data.data() + at + 4, sizeof(name_offset));

                table.entries.push_back({read_string(section.data, name_offset), value});
            }
            out.exports.push_back(std::move(table));

        } else if (section.type == sht_rpl_imports) {
            if (section.data.size() < 8) {
                result.warnings.push_back(
                    fmt::format("import section '{}' is too small", section.name));
                continue;
            }

            ImportTable table;
            table.section = section.name;
            table.is_function = section.name.starts_with(".fimport");

            u32be count{};
            u32be signature{};
            std::memcpy(&count, section.data.data(), sizeof(count));
            std::memcpy(&signature, section.data.data() + 4, sizeof(signature));
            table.count = count;
            table.signature = signature;

            // The library name follows the header, null-terminated.
            table.library = read_string(section.data, 8);
            if (table.library.empty()) {
                // Fall back to the section name: ".fimport_coreinit".
                const auto underscore = section.name.find('_');
                if (underscore != std::string::npos) {
                    table.library = section.name.substr(underscore + 1);
                }
            }
            out.imports.push_back(std::move(table));
        }
    }

    WIIU_LOG_DEBUG(Loader,
                   "parsed {}: {} sections, {} imports, {} export tables",
                   out.is_rpx ? "RPX" : "RPL",
                   out.sections.size(),
                   out.imports.size(),
                   out.exports.size());

    result.ok = true;
    return result;
}

ParseResult parse_file(const std::string& path, Rpl& out) {
    ParseResult result;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        result.error = fmt::format("cannot open {}", path);
        return result;
    }

    const auto size = static_cast<std::streamsize>(file.tellg());
    file.seekg(0);

    std::vector<u8> bytes(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        result.error = fmt::format("cannot read {}", path);
        return result;
    }

    result = parse(bytes, out);
    out.path = path;
    return result;
}

} // namespace wiiu::loader
