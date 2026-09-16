// Phase 2.5 — settings.
//
// One table describes every setting: its key, its type, where it lives in the
// `Settings` struct, and its valid range. TOML loading, TOML writing,
// command-line overrides, validation and `--help` output are all generated
// from that table, so adding a setting is one line and cannot leave the four
// code paths disagreeing with each other.
//
// Per-title overrides are a first-class part of the file rather than a bolt-on,
// because Phase 14.3 will need them: compatibility work produces per-game
// workarounds, and the alternative to a config layer is hacks hardcoded in
// engine code.
//
//     [graphics]
//     resolution_scale = 2
//
//     [title.0005000010143500]        # Breath of the Wild
//     graphics.resolution_scale = 1
#pragma once

#include <wiiu/common/types.hpp>

#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace wiiu::config {

/// Every tunable, with its default. Plain data: no invariants, no methods.
/// Validation lives in the field table, which is the only thing that writes
/// to these members from outside.
struct Settings {
    // [cpu]
    bool jit = true;                ///< P1
    bool idle_skip = true;          ///< P2
    bool free_running_cores = true; ///< P32 — not always faster; measure per title
    u32 timebase_hz = 62156250;     ///< 248.625 MHz bus / 4, per PowerPC 750

    // [graphics]
    u32 resolution_scale = 1; ///< 1 = native 1280x720
    bool vsync = true;
    bool metalfx_upscaling = false;  ///< P39 — spatial only; temporal is not usable here
    bool shader_cache = true;        ///< P5 / P24
    u32 shader_compile_threads = 0;  ///< 0 = pick from hardware_concurrency
    bool ubershader_fallback = true; ///< P25

    // [audio]
    bool audio_enabled = true;
    u32 volume = 100;
    u32 sample_rate = 48000; ///< AX supports 32000 or 48000

    // [logging]
    std::string log_level = "info";
    std::string log_file;

    // [paths]
    std::string mlc_path;   ///< emulated NAND storage
    std::string cache_path; ///< shader and pipeline caches
};

/// What a field's value is, and where it lives.
using FieldRef = std::variant<bool Settings::*, u32 Settings::*, std::string Settings::*>;

struct Field {
    std::string_view key; ///< dotted, e.g. "graphics.resolution_scale"
    FieldRef member;
    std::string_view help;
    u32 minimum = 0;                         ///< numeric fields only
    u32 maximum = 0xFFFFFFFFU;               ///< numeric fields only
    std::vector<std::string_view> choices{}; ///< string fields, empty = free-form

    [[nodiscard]] std::string_view section() const noexcept;
    [[nodiscard]] std::string_view name() const noexcept;
};

/// The table. Declaration order is the order values are written back out.
[[nodiscard]] std::span<const Field> fields();

[[nodiscard]] const Field* find_field(std::string_view key) noexcept;

/// Parse `value` according to the field's type and assign it. Returns false
/// and fills `error` on an unknown key, a malformed value, or one out of range.
bool set(Settings& settings, std::string_view key, std::string_view value, std::string& error);

/// The field's current value, formatted as it would be written to the file.
[[nodiscard]] std::string get(const Settings& settings, const Field& field);
[[nodiscard]] std::optional<std::string> get(const Settings& settings, std::string_view key);

[[nodiscard]] std::string to_toml(const Settings& settings);

/// Multi-line, for `--help`.
[[nodiscard]] std::string describe_fields();

struct LoadResult {
    bool ok = false;
    std::string error;                 ///< set when ok is false
    std::vector<std::string> warnings; ///< unknown keys and out-of-range values, which are skipped
};

/// A settings file, its parsed per-title sections, and any command-line
/// overrides applied on top.
class Config {
public:
    Config() = default;

    /// Load from disk. A missing file is not an error: defaults are used and
    /// `path()` is remembered so `save()` can create it.
    LoadResult load(const std::filesystem::path& path);
    LoadResult load_string(std::string_view toml, const std::filesystem::path& origin = {});

    /// Apply a `key=value` override, as passed on the command line. Overrides
    /// are remembered and re-applied after a reload, so a hot reload cannot
    /// silently undo what the user asked for on the command line.
    bool apply_override(std::string_view assignment, std::string& error);

    /// Re-read the file if its modification time changed. Returns true if it
    /// was reloaded. Intended for debug builds; nothing calls it automatically.
    bool reload_if_changed();

    bool save(std::string& error) const;

    [[nodiscard]] const Settings& base() const noexcept { return base_; }

    /// Base settings with `[title.<id>]` overrides merged on top. Title ids
    /// are compared case-insensitively, since they are written both ways.
    [[nodiscard]] Settings for_title(std::string_view title_id) const;

    [[nodiscard]] bool has_title_overrides(std::string_view title_id) const;
    [[nodiscard]] std::vector<std::string> title_ids() const;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    LoadResult parse(std::string_view toml);

    std::filesystem::path path_;
    Settings base_{};
    std::map<std::string, std::map<std::string, std::string>> title_overrides_;
    std::vector<std::pair<std::string, std::string>> command_line_overrides_;
    std::optional<std::filesystem::file_time_type> loaded_mtime_;
};

} // namespace wiiu::config
