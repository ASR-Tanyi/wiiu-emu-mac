// Phase 5.5 — the XML files beside an executable.
//
// A decrypted title's /code directory carries three of them:
//
//   cos.xml   what the loader needs: heap sizes, the codegen area, permissions
//   app.xml   title id, version, the OS version it was built against
//   meta.xml  what a person sees: name, publisher, region, controller support
//
// They are read here rather than in the HLE layer because the loader needs
// cos.xml before any guest code runs: the heap layout it describes decides
// where the application's memory lives.
#pragma once

#include <wiiu/common/types.hpp>

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace wiiu::loader {

/// From cos.xml.
struct CosSettings {
    u32 version = 0;
    std::string argstr;

    u32 max_size = 0;
    u32 max_codesize = 0;
    u32 codegen_size = 0; ///< the JIT area at 0x01800000; 0 means none
    u32 codegen_core = 0;
    u32 max_data_size = 0;
    u32 avail_size = 0;

    /// Per-core stack sizes and priorities, indexed by core.
    std::array<u32, 3> stack_size{};
    std::array<u32, 3> priority{};

    u32 exception_stack0_size = 0;
    u32 exception_stack1_size = 0;
    u32 exception_stack2_size = 0;

    /// Raw permission entries, keyed by group name. The loader cares whether a
    /// title asked for something it should not get.
    std::map<std::string, u64> permissions;
};

/// From app.xml.
struct AppSettings {
    u32 version = 0;
    u64 os_version = 0;
    u64 title_id = 0;
    u32 title_version = 0;
    u32 sdk_version = 0;
    u32 app_type = 0;
    u32 group_id = 0;

    /// Title id as the 16 hex digits used in paths and per-title config.
    [[nodiscard]] std::string title_id_string() const;
};

/// From meta.xml. Only the fields anything actually uses.
struct MetaSettings {
    u64 title_id = 0;
    u32 title_version = 0;
    std::string product_code;
    std::string company_code;
    std::string region;

    /// Long name by language code, e.g. "en" -> "The Legend of ...".
    std::map<std::string, std::string> long_names;
    std::map<std::string, std::string> short_names;
    std::map<std::string, std::string> publishers;

    /// English if present, otherwise whatever there is.
    [[nodiscard]] std::string display_name() const;
};

struct MetadataResult {
    bool ok = false;
    std::string error;
    std::vector<std::string> warnings;
};

[[nodiscard]] MetadataResult parse_cos_xml(std::string_view xml, CosSettings& out);
[[nodiscard]] MetadataResult parse_app_xml(std::string_view xml, AppSettings& out);
[[nodiscard]] MetadataResult parse_meta_xml(std::string_view xml, MetaSettings& out);

[[nodiscard]] MetadataResult load_cos_xml(const std::filesystem::path& path, CosSettings& out);
[[nodiscard]] MetadataResult load_app_xml(const std::filesystem::path& path, AppSettings& out);
[[nodiscard]] MetadataResult load_meta_xml(const std::filesystem::path& path, MetaSettings& out);

} // namespace wiiu::loader
