// Phase 5.1-5.3 — RPX and RPL files.
//
// A Cafe executable is ELF32, PowerPC, big-endian, with `e_type` 0xFF01 and
// **no program headers**: section headers alone drive loading. Sections may be
// individually zlib-compressed, each carries a CRC32, and imports and exports
// use PE-style tables rather than ELF dynamic symbols.
//
// RPX and RPL are the same format; the `RPL_IS_RPX` flag in the file-info
// section says which is the executable and which is a library.
//
// Caveat on provenance: the ELF parts are standard, but the RPL-specific
// section layouts below are documented by tooling (elf2rpl, wut) rather than
// by a published specification. Where a field's meaning is inferred it says
// so.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wiiu::loader {

// --------------------------------------------------------------------------
// On-disk structures, transcribed field for field.
// --------------------------------------------------------------------------

struct Elf32Header {
    std::array<u8, 16> ident;
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

static_assert(sizeof(Elf32Header) == 52);

struct Elf32SectionHeader {
    u32be name;
    u32be type;
    u32be flags;
    u32be addr;
    u32be offset;
    u32be size;
    u32be link;
    u32be info;
    u32be addralign;
    u32be entsize;
};

static_assert(sizeof(Elf32SectionHeader) == 40);

struct Elf32Symbol {
    u32be name; ///< offset into the linked string table
    u32be value;
    u32be size;
    u8 info; ///< binding in the high nibble, type in the low
    u8 other;
    u16be shndx; ///< the section it belongs to
};

static_assert(sizeof(Elf32Symbol) == 16);

/// The file-info section. 0x60 bytes; field names follow elf2rpl.
struct RplFileInfo {
    u32be version; ///< 0xCAFE
    u32be text_size;
    u32be text_align;
    u32be data_size;
    u32be data_align;
    u32be load_size;
    u32be load_align;
    u32be temp_size;
    u32be tramp_adjust;
    u32be sda_base;
    u32be sda2_base;
    u32be stack_size;
    u32be filename; ///< offset into the string table
    u32be flags;
    u32be heap_size;
    u32be tag_offset;
    u32be min_version;
    u32be compression_level;
    u32be tramp_addition;
    u32be file_info_pad;
    u32be sdk_version;
    u32be sdk_revision;
    u16be tls_module_index;
    u16be tls_align_shift;
    u32be runtime_file_info_size;
};

static_assert(sizeof(RplFileInfo) == 0x60);

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------

inline constexpr u16 et_cafe_rpl = 0xFF01;
inline constexpr u16 em_ppc = 20;

inline constexpr u32 sht_rpl_exports = 0x80000001;
inline constexpr u32 sht_rpl_imports = 0x80000002;
inline constexpr u32 sht_rpl_crcs = 0x80000003;
inline constexpr u32 sht_rpl_fileinfo = 0x80000004;

inline constexpr u32 sht_progbits = 1;
inline constexpr u32 sht_symtab = 2;
inline constexpr u32 sht_strtab = 3;
inline constexpr u32 sht_rela = 4;
inline constexpr u32 sht_nobits = 8;

inline constexpr u32 shf_write = 0x1;
inline constexpr u32 shf_alloc = 0x2;
inline constexpr u32 shf_execinstr = 0x4;
/// Section data is a 4-byte big-endian uncompressed size followed by a zlib
/// stream.
inline constexpr u32 shf_deflated = 0x08000000;

inline constexpr u32 rpl_is_rpx = 0x2;

/// The Cafe marker at EI_OSABI / EI_ABIVERSION.
inline constexpr u8 ei_osabi_cafe = 0xCA;
inline constexpr u8 ei_abiversion_cafe = 0xFE;

// --------------------------------------------------------------------------
// Parsed form
// --------------------------------------------------------------------------

struct Section {
    std::string name;
    u32 type = 0;
    u32 flags = 0;
    ppcaddr_t address = 0;
    u32 size = 0; ///< uncompressed size
    u32 file_offset = 0;
    u32 link = 0;
    u32 info = 0;
    u32 alignment = 0;
    u32 entry_size = 0;
    bool compressed = false;

    /// Inflated contents. Empty for SHT_NOBITS.
    std::vector<u8> data;

    /// CRC32 recorded in the CRCs section, and whether it matched.
    u32 recorded_crc = 0;
    bool crc_checked = false;
    bool crc_ok = false;

    [[nodiscard]] bool is_code() const noexcept { return (flags & shf_execinstr) != 0; }

    [[nodiscard]] bool occupies_memory() const noexcept { return (flags & shf_alloc) != 0; }
};

/// A symbol table entry, resolved against its string table.
///
/// Imports are the interesting case: an imported function appears here with
/// its `shndx` pointing at the `.fimport_<library>` section that names where it
/// comes from. That is the link between a call site's relocation and the
/// library it needs -- there is no separate table mapping one to the other.
struct Symbol {
    std::string name;
    u32 value = 0;
    u32 size = 0;
    u8 info = 0;
    u16 section_index = 0;

    [[nodiscard]] u8 type() const noexcept { return info & 0xF; }

    [[nodiscard]] u8 binding() const noexcept { return info >> 4; }
};

struct Export {
    std::string name;
    u32 value = 0; ///< address for a function, or address of the datum
};

struct ExportTable {
    std::string section; ///< ".fexports" or ".dexports"
    bool is_function = false;
    u32 signature = 0;
    std::vector<Export> entries;
};

struct ImportTable {
    std::string section; ///< ".fimport_coreinit" and friends
    std::string library; ///< "coreinit"
    bool is_function = false;
    u32 signature = 0;
    u32 count = 0;
};

struct Rpl {
    std::string path;
    bool is_rpx = false; ///< executable rather than library
    ppcaddr_t entry = 0;
    u32 flags = 0;

    std::vector<Section> sections;
    std::vector<Symbol> symbols;
    std::vector<ExportTable> exports;
    std::vector<ImportTable> imports;
    std::optional<RplFileInfo> file_info;

    /// Name from the file-info string table, when present.
    std::string module_name;

    [[nodiscard]] const Section* find_section(std::string_view name) const noexcept;
    [[nodiscard]] const Section* find_section(u32 index) const noexcept;

    /// The library an imported symbol comes from, from the name of the
    /// `.fimport_*` / `.dimport_*` section it points at. Empty if the symbol
    /// is not an import.
    [[nodiscard]] std::string import_library(const Symbol& symbol) const;
    [[nodiscard]] const Section* section_at(ppcaddr_t address) const noexcept;
};

struct ParseResult {
    bool ok = false;
    std::string error;
    std::vector<std::string> warnings; ///< CRC mismatches, odd sections
};

/// Parse an RPX or RPL held in memory.
[[nodiscard]] ParseResult parse(std::span<const u8> bytes, Rpl& out);

/// Read and parse a file.
[[nodiscard]] ParseResult parse_file(const std::string& path, Rpl& out);

/// CRC32 as the format uses it (the standard reflected polynomial).
[[nodiscard]] u32 crc32(std::span<const u8> bytes) noexcept;

/// Inflate a SHF_DEFLATED payload: a 4-byte big-endian size then a zlib
/// stream.
[[nodiscard]] bool
inflate_section(std::span<const u8> input, std::vector<u8>& out, std::string& error);

} // namespace wiiu::loader
