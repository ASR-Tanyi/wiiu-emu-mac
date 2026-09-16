#include <wiiu/common/config.hpp>
#include <wiiu/common/log.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <system_error>
#include <toml++/toml.hpp>

namespace wiiu::config {
namespace {

using wiiu::log::Channel;

[[nodiscard]] std::string lowercase(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const char c : input) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

[[nodiscard]] std::string_view trim(std::string_view input) {
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

bool parse_bool(std::string_view text, bool& out) {
    static constexpr std::array<std::string_view, 4> kTrue{"true", "1", "yes", "on"};
    static constexpr std::array<std::string_view, 4> kFalse{"false", "0", "no", "off"};

    const std::string needle = lowercase(trim(text));
    if (std::ranges::find(kTrue, needle) != kTrue.end()) {
        out = true;
        return true;
    }
    if (std::ranges::find(kFalse, needle) != kFalse.end()) {
        out = false;
        return true;
    }
    return false;
}

bool parse_u32(std::string_view text, u32& out) {
    std::string_view value = trim(text);
    int base = 10;
    if (value.size() > 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
        value.remove_prefix(2);
        base = 16;
    }
    if (value.empty()) {
        return false;
    }

    u32 parsed = 0;
    const auto* const begin = value.data();
    const auto* const end = begin + value.size();
    const auto result = std::from_chars(begin, end, parsed, base);
    if (result.ec != std::errc{} || result.ptr != end) {
        return false;
    }
    out = parsed;
    return true;
}

/// TOML scalars, as the text `set()` expects. Tables and arrays have no
/// meaning for a setting, and come back empty so the caller can warn.
[[nodiscard]] std::optional<std::string> node_to_string(const toml::node& node) {
    if (const auto* const value = node.as_string()) {
        return value->get();
    }
    if (const auto* const value = node.as_boolean()) {
        return value->get() ? "true" : "false";
    }
    if (const auto* const value = node.as_integer()) {
        return fmt::format("{}", value->get());
    }
    if (const auto* const value = node.as_floating_point()) {
        return fmt::format("{}", value->get());
    }
    return std::nullopt;
}

/// Flatten a TOML document to dotted key -> text.
///
/// TOML turns `graphics.resolution_scale = 1` into a nested `graphics` table,
/// so a per-title section arrives as sub-tables rather than as the dotted keys
/// it was written with. Flattening makes both spellings identical, which is
/// what the field table expects.
void flatten(const toml::table& table,
             const std::string& prefix,
             std::vector<std::pair<std::string, std::string>>& out,
             std::vector<std::string>& unsupported) {
    for (const auto& [key, value] : table) {
        const std::string full =
            prefix.empty() ? std::string{key.str()} : fmt::format("{}.{}", prefix, key.str());

        if (const auto* const nested = value.as_table()) {
            flatten(*nested, full, out, unsupported);
            continue;
        }

        auto text = node_to_string(value);
        if (!text.has_value()) {
            unsupported.push_back(full);
            continue;
        }
        out.emplace_back(full, *std::move(text));
    }
}

const std::vector<Field>& field_table() {
    // Declared once. Order here is the order fields are written back out.
    static const std::vector<Field> table = [] {
        std::vector<Field> f;

        f.push_back({"cpu.jit", &Settings::jit, "Recompile guest code instead of interpreting"});
        f.push_back({"cpu.idle_skip",
                     &Settings::idle_skip,
                     "Skip emulated time when a guest thread is spinning"});
        f.push_back({"cpu.free_running_cores",
                     &Settings::free_running_cores,
                     "Run the three cores independently; lock-step can be faster "
                     "for heavily synchronised titles"});
        f.push_back({"cpu.timebase_hz",
                     &Settings::timebase_hz,
                     "Guest timebase frequency; the default matches real hardware",
                     1000000U,
                     1000000000U});

        f.push_back({"graphics.resolution_scale",
                     &Settings::resolution_scale,
                     "Multiple of the native 1280x720 to render at",
                     1U,
                     8U});
        f.push_back(
            {"graphics.vsync", &Settings::vsync, "Synchronise presentation to the display"});
        f.push_back({"graphics.metalfx_upscaling",
                     &Settings::metalfx_upscaling,
                     "Upscale the final image with MetalFX spatial"});
        f.push_back({"graphics.shader_cache",
                     &Settings::shader_cache,
                     "Persist translated shaders and pipelines between runs"});
        f.push_back({"graphics.shader_compile_threads",
                     &Settings::shader_compile_threads,
                     "Background shader compile threads; 0 picks from the CPU count",
                     0U,
                     32U});
        f.push_back({"graphics.ubershader_fallback",
                     &Settings::ubershader_fallback,
                     "Draw with a generic shader while the specialised one compiles"});

        f.push_back({"audio.enabled", &Settings::audio_enabled, "Enable audio output"});
        f.push_back({"audio.volume", &Settings::volume, "Output volume, per cent", 0U, 100U});
        f.push_back({"audio.sample_rate",
                     &Settings::sample_rate,
                     "AX renderer rate; the hardware supports 32000 and 48000",
                     32000U,
                     48000U});

        f.push_back({"logging.level",
                     &Settings::log_level,
                     "Default level for every channel",
                     0U,
                     0xFFFFFFFFU,
                     {"trace", "debug", "info", "warn", "error", "critical", "off"}});
        f.push_back(
            {"logging.file", &Settings::log_file, "Also write the log here; empty to skip"});

        f.push_back({"paths.mlc", &Settings::mlc_path, "Emulated NAND storage directory"});
        f.push_back({"paths.cache", &Settings::cache_path, "Shader and pipeline cache directory"});

        return f;
    }();
    return table;
}

} // namespace

std::string_view Field::section() const noexcept {
    const auto dot = key.find('.');
    return dot == std::string_view::npos ? std::string_view{} : key.substr(0, dot);
}

std::string_view Field::name() const noexcept {
    const auto dot = key.find('.');
    return dot == std::string_view::npos ? key : key.substr(dot + 1);
}

std::span<const Field> fields() {
    return field_table();
}

const Field* find_field(std::string_view key) noexcept {
    const auto& table = field_table();
    const auto it = std::ranges::find_if(table, [key](const Field& f) { return f.key == key; });
    return it == table.end() ? nullptr : &*it;
}

bool set(Settings& settings, std::string_view key, std::string_view value, std::string& error) {
    const Field* const field = find_field(key);
    if (field == nullptr) {
        error = fmt::format("unknown setting '{}'", key);
        return false;
    }

    return std::visit(
        [&](auto member) {
            using MemberType = std::remove_cvref_t<decltype(settings.*member)>;

            if constexpr (std::is_same_v<MemberType, bool>) {
                bool parsed = false;
                if (!parse_bool(value, parsed)) {
                    error = fmt::format("'{}' expects a boolean, got '{}'", key, value);
                    return false;
                }
                settings.*member = parsed;
                return true;

            } else if constexpr (std::is_same_v<MemberType, u32>) {
                u32 parsed = 0;
                if (!parse_u32(value, parsed)) {
                    error = fmt::format("'{}' expects a number, got '{}'", key, value);
                    return false;
                }
                if (parsed < field->minimum || parsed > field->maximum) {
                    error = fmt::format("'{}' must be between {} and {}, got {}",
                                        key,
                                        field->minimum,
                                        field->maximum,
                                        parsed);
                    return false;
                }
                settings.*member = parsed;
                return true;

            } else {
                const std::string parsed{trim(value)};
                if (!field->choices.empty() &&
                    std::ranges::find(field->choices, parsed) == field->choices.end()) {
                    error = fmt::format("'{}' must be one of {}, got '{}'",
                                        key,
                                        fmt::join(field->choices, ", "),
                                        parsed);
                    return false;
                }
                settings.*member = parsed;
                return true;
            }
        },
        field->member);
}

std::string get(const Settings& settings, const Field& field) {
    return std::visit(
        [&settings](auto member) -> std::string {
            using MemberType = std::remove_cvref_t<decltype(settings.*member)>;
            if constexpr (std::is_same_v<MemberType, bool>) {
                return settings.*member ? "true" : "false";
            } else if constexpr (std::is_same_v<MemberType, u32>) {
                return fmt::format("{}", settings.*member);
            } else {
                return settings.*member;
            }
        },
        field.member);
}

std::optional<std::string> get(const Settings& settings, std::string_view key) {
    const Field* const field = find_field(key);
    if (field == nullptr) {
        return std::nullopt;
    }
    return get(settings, *field);
}

std::string to_toml(const Settings& settings) {
    std::string out = "# wiiu-emu settings\n";
    std::string_view current_section;

    for (const Field& field : fields()) {
        if (field.section() != current_section) {
            current_section = field.section();
            out += fmt::format("\n[{}]\n", current_section);
        }

        const std::string value = get(settings, field);
        const bool quoted = std::holds_alternative<std::string Settings::*>(field.member);

        out += fmt::format(
            "{} = {}{}{}\n", field.name(), quoted ? "\"" : "", value, quoted ? "\"" : "");
    }
    return out;
}

std::string describe_fields() {
    std::string out;
    std::size_t width = 0;
    for (const Field& field : fields()) {
        width = std::max(width, field.key.size());
    }
    for (const Field& field : fields()) {
        out += fmt::format("  {:<{}}  {}\n", field.key, width, field.help);
    }
    return out;
}

// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------

LoadResult Config::parse(std::string_view toml_text) {
    LoadResult result;

    toml::table table;
    try {
        table = toml::parse(toml_text);
    } catch (const toml::parse_error& error) {
        result.error = fmt::format("{} at line {}", error.description(), error.source().begin.line);
        return result;
    }

    base_ = Settings{};
    title_overrides_.clear();

    std::vector<std::pair<std::string, std::string>> flat;
    std::vector<std::string> unsupported;
    flatten(table, {}, flat, unsupported);

    for (const std::string& key : unsupported) {
        result.warnings.push_back(fmt::format("'{}' has an unsupported value type; ignored", key));
    }

    static constexpr std::string_view kTitlePrefix = "title.";

    for (const auto& [key, value] : flat) {
        if (!key.starts_with(kTitlePrefix)) {
            std::string error;
            if (!set(base_, key, value, error)) {
                result.warnings.push_back(error);
            }
            continue;
        }

        // title.<id>.<section>.<name>
        const std::string_view remainder{key};
        const std::string_view after_prefix = remainder.substr(kTitlePrefix.size());
        const auto separator = after_prefix.find('.');
        if (separator == std::string_view::npos) {
            result.warnings.push_back(fmt::format("'{}' is not a per-title setting; ignored", key));
            continue;
        }

        const std::string title_id = lowercase(after_prefix.substr(0, separator));
        const std::string setting_key{after_prefix.substr(separator + 1)};

        if (find_field(setting_key) == nullptr) {
            result.warnings.push_back(
                fmt::format("unknown setting '{}' in [title.{}]", setting_key, title_id));
            continue;
        }

        // Validate now, against a scratch copy, so a bad per-title value is
        // reported at load time rather than when that title is launched.
        Settings probe = base_;
        std::string error;
        if (!set(probe, setting_key, value, error)) {
            result.warnings.push_back(fmt::format("[title.{}]: {}", title_id, error));
            continue;
        }

        title_overrides_[title_id][setting_key] = value;
    }

    // Command-line overrides win over the file, and survive a reload.
    for (const auto& [key, value] : command_line_overrides_) {
        std::string error;
        if (!set(base_, key, value, error)) {
            result.warnings.push_back(error);
        }
    }

    result.ok = true;
    return result;
}

LoadResult Config::load_string(std::string_view toml_text, const std::filesystem::path& origin) {
    path_ = origin;
    loaded_mtime_.reset();
    return parse(toml_text);
}

LoadResult Config::load(const std::filesystem::path& path) {
    path_ = path;
    loaded_mtime_.reset();

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        // Not an error: first run. Defaults stand, and save() can create it.
        LoadResult result;
        result.ok = true;
        base_ = Settings{};
        title_overrides_.clear();
        for (const auto& [key, value] : command_line_overrides_) {
            std::string error;
            if (!set(base_, key, value, error)) {
                result.warnings.push_back(error);
            }
        }
        return result;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        LoadResult result;
        result.error = fmt::format("cannot read {}", path.string());
        return result;
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    LoadResult result = parse(buffer.str());
    if (result.ok) {
        loaded_mtime_ = std::filesystem::last_write_time(path, ec);
    }
    return result;
}

bool Config::apply_override(std::string_view assignment, std::string& error) {
    const auto separator = assignment.find('=');
    if (separator == std::string_view::npos) {
        error = fmt::format("expected key=value, got '{}'", assignment);
        return false;
    }

    const std::string key{trim(assignment.substr(0, separator))};
    const std::string value{trim(assignment.substr(separator + 1))};

    if (!set(base_, key, value, error)) {
        return false;
    }

    // Replace any earlier override of the same key so the list stays the
    // minimal set to re-apply after a reload.
    std::erase_if(command_line_overrides_,
                  [&key](const auto& entry) { return entry.first == key; });
    command_line_overrides_.emplace_back(key, value);
    return true;
}

bool Config::reload_if_changed() {
    if (path_.empty() || !loaded_mtime_.has_value()) {
        return false;
    }

    std::error_code ec;
    const auto current = std::filesystem::last_write_time(path_, ec);
    if (ec || current == *loaded_mtime_) {
        return false;
    }

    const LoadResult result = load(path_);
    if (!result.ok) {
        WIIU_LOG_WARN(Core, "settings reload failed: {}", result.error);
        return false;
    }
    for (const std::string& warning : result.warnings) {
        WIIU_LOG_WARN(Core, "settings: {}", warning);
    }
    WIIU_LOG_INFO(Core, "reloaded settings from {}", path_.string());
    return true;
}

bool Config::save(std::string& error) const {
    if (path_.empty()) {
        error = "no settings path set";
        return false;
    }

    std::error_code ec;
    if (path_.has_parent_path()) {
        std::filesystem::create_directories(path_.parent_path(), ec);
    }

    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    if (!file) {
        error = fmt::format("cannot write {}", path_.string());
        return false;
    }

    file << to_toml(base_);

    // Per-title sections are preserved verbatim; nothing here rewrites a
    // workaround a user put in by hand.
    for (const auto& [title, overrides] : title_overrides_) {
        file << fmt::format("\n[title.{}]\n", title);
        for (const auto& [key, value] : overrides) {
            const Field* const field = find_field(key);
            const bool quoted =
                field != nullptr && std::holds_alternative<std::string Settings::*>(field->member);
            file << fmt::format(
                "{} = {}{}{}\n", key, quoted ? "\"" : "", value, quoted ? "\"" : "");
        }
    }

    return file.good();
}

Settings Config::for_title(std::string_view title_id) const {
    Settings settings = base_;

    const auto it = title_overrides_.find(lowercase(title_id));
    if (it == title_overrides_.end()) {
        return settings;
    }

    for (const auto& [key, value] : it->second) {
        std::string error;
        if (!set(settings, key, value, error)) {
            WIIU_LOG_WARN(Core, "title {}: {}", title_id, error);
        }
    }

    // Command-line overrides outrank a per-title profile: the user asking for
    // something on this run beats a stored workaround.
    for (const auto& [key, value] : command_line_overrides_) {
        std::string error;
        static_cast<void>(set(settings, key, value, error));
    }

    return settings;
}

bool Config::has_title_overrides(std::string_view title_id) const {
    return title_overrides_.contains(lowercase(title_id));
}

std::vector<std::string> Config::title_ids() const {
    std::vector<std::string> ids;
    ids.reserve(title_overrides_.size());
    for (const auto& [title, overrides] : title_overrides_) {
        static_cast<void>(overrides);
        ids.push_back(title);
    }
    return ids;
}

} // namespace wiiu::config
