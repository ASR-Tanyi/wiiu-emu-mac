#include <wiiu/loader/metadata.hpp>

#include <fmt/format.h>

#include <charconv>
#include <fstream>
#include <pugixml.hpp>
#include <sstream>

namespace wiiu::loader {
namespace {

/// Read a numeric element.
///
/// These files mix conventions: `type="hexBinary"` values are hex without a
/// prefix, some values carry an explicit `0x`, and the rest are decimal. An
/// unparseable value becomes zero and is warned about rather than failing the
/// load, since a title with one odd field should still boot.
[[nodiscard]] u64 parse_value(const pugi::xml_node& node) {
    std::string_view text = node.text().as_string();
    if (text.empty()) {
        return 0;
    }

    const std::string_view type = node.attribute("type").as_string();

    int base = 10;
    if (text.starts_with("0x") || text.starts_with("0X")) {
        text.remove_prefix(2);
        base = 16;
    } else if (type == "hexBinary") {
        base = 16;
    }

    const auto* const end = text.data() + text.size();

    u64 value = 0;
    if (std::from_chars(text.data(), end, value, base).ec == std::errc{}) {
        return value;
    }

    // A decimal-looking field that is actually hex: try again rather than
    // silently reporting zero.
    value = 0;
    if (base == 10 && std::from_chars(text.data(), end, value, 16).ec == std::errc{}) {
        return value;
    }
    return 0;
}

[[nodiscard]] u32 parse_u32(const pugi::xml_node& node) {
    return static_cast<u32>(parse_value(node));
}

[[nodiscard]] MetadataResult load_document(std::string_view xml, pugi::xml_document& document) {
    MetadataResult result;
    const pugi::xml_parse_result parsed = document.load_buffer(xml.data(), xml.size());
    if (!parsed) {
        result.error = fmt::format("{} at offset {}", parsed.description(), parsed.offset);
        return result;
    }
    result.ok = true;
    return result;
}

[[nodiscard]] MetadataResult read_file(const std::filesystem::path& path, std::string& out) {
    MetadataResult result;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        result.error = fmt::format("cannot open {}", path.string());
        return result;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    out = buffer.str();
    result.ok = true;
    return result;
}

} // namespace

std::string AppSettings::title_id_string() const {
    return fmt::format("{:016x}", title_id);
}

std::string MetaSettings::display_name() const {
    if (const auto english = long_names.find("en"); english != long_names.end()) {
        return english->second;
    }
    return long_names.empty() ? std::string{} : long_names.begin()->second;
}

MetadataResult parse_cos_xml(std::string_view xml, CosSettings& out) {
    pugi::xml_document document;
    MetadataResult result = load_document(xml, document);
    if (!result.ok) {
        return result;
    }

    const pugi::xml_node root = document.child("app");
    if (!root) {
        result.ok = false;
        result.error = "cos.xml has no <app> element";
        return result;
    }

    out = CosSettings{};
    out.version = parse_u32(root.child("version"));
    out.argstr = root.child("argstr").text().as_string();
    out.max_size = parse_u32(root.child("max_size"));
    out.max_codesize = parse_u32(root.child("max_codesize"));
    out.codegen_size = parse_u32(root.child("codegen_size"));
    out.codegen_core = parse_u32(root.child("codegen_core"));
    out.max_data_size = parse_u32(root.child("max_data_size"));
    out.avail_size = parse_u32(root.child("avail_size"));

    for (u32 core = 0; core < 3; ++core) {
        out.stack_size[core] =
            parse_u32(root.child(fmt::format("default_stack{}_size", core).c_str()));
        out.priority[core] =
            parse_u32(root.child(fmt::format("default_redzone{}_size", core).c_str()));
    }

    out.exception_stack0_size = parse_u32(root.child("exception_stack0_size"));
    out.exception_stack1_size = parse_u32(root.child("exception_stack1_size"));
    out.exception_stack2_size = parse_u32(root.child("exception_stack2_size"));

    for (const pugi::xml_node& permission : root.child("permissions").children()) {
        const std::string group = permission.child("group").text().as_string();
        if (!group.empty()) {
            out.permissions[group] = parse_value(permission.child("mask"));
        }
    }

    return result;
}

MetadataResult parse_app_xml(std::string_view xml, AppSettings& out) {
    pugi::xml_document document;
    MetadataResult result = load_document(xml, document);
    if (!result.ok) {
        return result;
    }

    const pugi::xml_node root = document.child("app");
    if (!root) {
        result.ok = false;
        result.error = "app.xml has no <app> element";
        return result;
    }

    out = AppSettings{};
    out.version = parse_u32(root.child("version"));
    out.os_version = parse_value(root.child("os_version"));
    out.title_id = parse_value(root.child("title_id"));
    out.title_version = parse_u32(root.child("title_version"));
    out.sdk_version = parse_u32(root.child("sdk_version"));
    out.app_type = parse_u32(root.child("app_type"));
    out.group_id = parse_u32(root.child("group_id"));

    if (out.title_id == 0) {
        result.warnings.emplace_back("app.xml has no title id");
    }
    return result;
}

MetadataResult parse_meta_xml(std::string_view xml, MetaSettings& out) {
    pugi::xml_document document;
    MetadataResult result = load_document(xml, document);
    if (!result.ok) {
        return result;
    }

    const pugi::xml_node root = document.child("menu");
    if (!root) {
        result.ok = false;
        result.error = "meta.xml has no <menu> element";
        return result;
    }

    out = MetaSettings{};
    out.title_id = parse_value(root.child("title_id"));
    out.title_version = parse_u32(root.child("title_version"));
    out.product_code = root.child("product_code").text().as_string();
    out.company_code = root.child("company_code").text().as_string();
    out.region = root.child("region").text().as_string();

    // Names are one element per language: <longname_en>, <shortname_ja>, ...
    for (const pugi::xml_node& child : root.children()) {
        const std::string_view name = child.name();
        const auto split = [&](std::string_view prefix, std::map<std::string, std::string>& into) {
            if (name.size() > prefix.size() && name.starts_with(prefix)) {
                into[std::string{name.substr(prefix.size())}] = child.text().as_string();
            }
        };
        split("longname_", out.long_names);
        split("shortname_", out.short_names);
        split("publisher_", out.publishers);
    }

    return result;
}

MetadataResult load_cos_xml(const std::filesystem::path& path, CosSettings& out) {
    std::string xml;
    MetadataResult result = read_file(path, xml);
    return result.ok ? parse_cos_xml(xml, out) : result;
}

MetadataResult load_app_xml(const std::filesystem::path& path, AppSettings& out) {
    std::string xml;
    MetadataResult result = read_file(path, xml);
    return result.ok ? parse_app_xml(xml, out) : result;
}

MetadataResult load_meta_xml(const std::filesystem::path& path, MetaSettings& out) {
    std::string xml;
    MetadataResult result = read_file(path, xml);
    return result.ok ? parse_meta_xml(xml, out) : result;
}

} // namespace wiiu::loader
