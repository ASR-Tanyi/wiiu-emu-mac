#include <wiiu/common/config.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

using namespace wiiu;
using wiiu::config::Config;
using wiiu::config::Settings;

namespace {

/// A scratch file that removes itself, so tests leave nothing behind.
class TempFile {
public:
    explicit TempFile(std::string_view contents = {}) {
        path_ = std::filesystem::temp_directory_path() /
                std::filesystem::path{"wiiu-emu-test-" + std::to_string(counter()++) + ".toml"};
        if (!contents.empty()) {
            write(contents);
        }
    }

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    void write(std::string_view contents) const {
        std::ofstream file(path_, std::ios::binary | std::ios::trunc);
        file << contents;
    }

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    static int& counter() {
        static int value = 0;
        return value;
    }

    std::filesystem::path path_;
};

} // namespace

TEST_CASE("defaults are sane", "[common][config]") {
    const Settings settings;
    CHECK(settings.jit);
    CHECK(settings.idle_skip);
    CHECK(settings.resolution_scale == 1U);
    CHECK(settings.volume == 100U);
    CHECK(settings.log_level == "info");
}

TEST_CASE("the field table covers every key exactly once", "[common][config]") {
    const auto all = config::fields();
    REQUIRE_FALSE(all.empty());

    for (const auto& field : all) {
        CHECK_FALSE(field.key.empty());
        CHECK_FALSE(field.help.empty());
        CHECK(field.key.find('.') != std::string_view::npos); // section.name
        CHECK(config::find_field(field.key) == &field);
    }

    CHECK(config::find_field("graphics.nonexistent") == nullptr);
}

TEST_CASE("set parses each type", "[common][config]") {
    Settings settings;
    std::string error;

    REQUIRE(config::set(settings, "cpu.jit", "false", error));
    CHECK_FALSE(settings.jit);

    REQUIRE(config::set(settings, "cpu.jit", "on", error));
    CHECK(settings.jit);

    REQUIRE(config::set(settings, "graphics.resolution_scale", "4", error));
    CHECK(settings.resolution_scale == 4U);

    REQUIRE(config::set(settings, "cpu.timebase_hz", "0x12A05F20", error));
    CHECK(settings.timebase_hz == 0x12A05F20U);

    REQUIRE(config::set(settings, "paths.mlc", "/tmp/mlc", error));
    CHECK(settings.mlc_path == "/tmp/mlc");
}

TEST_CASE("set rejects bad input without modifying anything", "[common][config]") {
    Settings settings;
    std::string error;

    CHECK_FALSE(config::set(settings, "graphics.nonexistent", "1", error));
    CHECK(error.find("unknown setting") != std::string::npos);

    CHECK_FALSE(config::set(settings, "cpu.jit", "maybe", error));
    CHECK(error.find("boolean") != std::string::npos);
    CHECK(settings.jit); // unchanged

    CHECK_FALSE(config::set(settings, "graphics.resolution_scale", "twelve", error));
    CHECK(settings.resolution_scale == 1U);

    // Out of range: resolution_scale is capped at 8.
    CHECK_FALSE(config::set(settings, "graphics.resolution_scale", "99", error));
    CHECK(error.find("between 1 and 8") != std::string::npos);
    CHECK(settings.resolution_scale == 1U);

    // Trailing garbage is not a number.
    CHECK_FALSE(config::set(settings, "audio.volume", "50%", error));
    CHECK(settings.volume == 100U);

    // A field with an enumerated set of values rejects anything else.
    CHECK_FALSE(config::set(settings, "logging.level", "verbose", error));
    CHECK(error.find("must be one of") != std::string::npos);
    CHECK(config::set(settings, "logging.level", "trace", error));
    CHECK(settings.log_level == "trace");
}

TEST_CASE("settings round-trip through TOML", "[common][config]") {
    Settings original;
    std::string error;
    REQUIRE(config::set(original, "graphics.resolution_scale", "3", error));
    REQUIRE(config::set(original, "cpu.jit", "false", error));
    REQUIRE(config::set(original, "paths.mlc", "/games/mlc", error));
    REQUIRE(config::set(original, "logging.level", "debug", error));

    const std::string text = config::to_toml(original);
    CHECK(text.find("[graphics]") != std::string::npos);
    CHECK(text.find("resolution_scale = 3") != std::string::npos);
    CHECK(text.find("mlc = \"/games/mlc\"") != std::string::npos);

    Config reloaded;
    const auto result = reloaded.load_string(text);
    REQUIRE(result.ok);
    CHECK(result.warnings.empty());

    CHECK(reloaded.base().resolution_scale == 3U);
    CHECK_FALSE(reloaded.base().jit);
    CHECK(reloaded.base().mlc_path == "/games/mlc");
    CHECK(reloaded.base().log_level == "debug");
}

TEST_CASE("a malformed file is an error, not a crash", "[common][config]") {
    Config config;
    const auto result = config.load_string("[graphics\nvsync = ");
    CHECK_FALSE(result.ok);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("unknown keys warn and are skipped", "[common][config]") {
    Config config;
    const auto result = config.load_string(R"(
[graphics]
vsync = false
enable_raytracing = true

[nonsense]
what = 1
)");

    REQUIRE(result.ok);
    CHECK_FALSE(config.base().vsync);   // the valid key still applied
    CHECK(result.warnings.size() >= 2); // both unknown keys reported
}

TEST_CASE("an out-of-range value warns and leaves the default", "[common][config]") {
    Config config;
    const auto result = config.load_string("[audio]\nvolume = 500\n");

    REQUIRE(result.ok);
    REQUIRE(result.warnings.size() == 1);
    CHECK(config.base().volume == 100U);
}

TEST_CASE("command-line overrides beat the file", "[common][config]") {
    Config config;
    REQUIRE(config.load_string("[graphics]\nresolution_scale = 2\n").ok);
    CHECK(config.base().resolution_scale == 2U);

    std::string error;
    REQUIRE(config.apply_override("graphics.resolution_scale=4", error));
    CHECK(config.base().resolution_scale == 4U);

    CHECK_FALSE(config.apply_override("graphics.resolution_scale", error));
    CHECK(error.find("key=value") != std::string::npos);

    CHECK_FALSE(config.apply_override("graphics.bogus=1", error));
}

TEST_CASE("overrides survive a reload", "[common][config]") {
    // The hazard this guards: a hot reload silently undoing what the user
    // asked for on the command line.
    TempFile file("[graphics]\nresolution_scale = 1\n");

    Config config;
    REQUIRE(config.load(file.path()).ok);

    std::string error;
    REQUIRE(config.apply_override("graphics.resolution_scale=4", error));
    CHECK(config.base().resolution_scale == 4U);

    REQUIRE(config.load(file.path()).ok);
    CHECK(config.base().resolution_scale == 4U); // override re-applied
}

TEST_CASE("per-title overrides merge on top of the base", "[common][config]") {
    Config config;
    const auto result = config.load_string(R"(
[graphics]
resolution_scale = 2
vsync = true

[cpu]
jit = true

[title.0005000010143500]
graphics.resolution_scale = 1
cpu.free_running_cores = false
)");

    REQUIRE(result.ok);
    CHECK(result.warnings.empty());

    const Settings base = config.base();
    CHECK(base.resolution_scale == 2U);
    CHECK(base.free_running_cores);

    const Settings botw = config.for_title("0005000010143500");
    CHECK(botw.resolution_scale == 1U);   // overridden
    CHECK_FALSE(botw.free_running_cores); // overridden
    CHECK(botw.vsync);                    // inherited
    CHECK(botw.jit);                      // inherited

    // An unknown title just gets the base settings.
    const Settings other = config.for_title("0005000000000000");
    CHECK(other.resolution_scale == 2U);
    CHECK(other.free_running_cores);

    CHECK(config.has_title_overrides("0005000010143500"));
    CHECK_FALSE(config.has_title_overrides("0005000000000000"));
    CHECK(config.title_ids().size() == 1);
}

TEST_CASE("title ids match case-insensitively", "[common][config]") {
    Config config;
    REQUIRE(config
                .load_string(R"(
[title.000500001017BB00]
graphics.vsync = false
)")
                .ok);

    CHECK(config.has_title_overrides("000500001017bb00"));
    CHECK(config.has_title_overrides("000500001017BB00"));
    CHECK_FALSE(config.for_title("000500001017bb00").vsync);
}

TEST_CASE("a command-line override outranks a per-title profile", "[common][config]") {
    Config config;
    REQUIRE(config
                .load_string(R"(
[graphics]
resolution_scale = 2

[title.0005000010143500]
graphics.resolution_scale = 1
)")
                .ok);

    CHECK(config.for_title("0005000010143500").resolution_scale == 1U);

    std::string error;
    REQUIRE(config.apply_override("graphics.resolution_scale=4", error));
    CHECK(config.for_title("0005000010143500").resolution_scale == 4U);
}

TEST_CASE("a missing file is not an error", "[common][config]") {
    const auto missing =
        std::filesystem::temp_directory_path() / "wiiu-emu-definitely-not-here.toml";
    std::error_code ec;
    std::filesystem::remove(missing, ec);

    Config config;
    const auto result = config.load(missing);
    CHECK(result.ok);
    CHECK(config.base().resolution_scale == 1U); // defaults
    CHECK(config.path() == missing);
}

TEST_CASE("save writes a file that loads back", "[common][config]") {
    TempFile file;

    Config config;
    REQUIRE(config.load(file.path()).ok);

    std::string error;
    REQUIRE(config.apply_override("graphics.resolution_scale=3", error));
    REQUIRE(config.apply_override("paths.cache=/var/cache/wiiu", error));
    REQUIRE(config.save(error));

    Config reloaded;
    REQUIRE(reloaded.load(file.path()).ok);
    CHECK(reloaded.base().resolution_scale == 3U);
    CHECK(reloaded.base().cache_path == "/var/cache/wiiu");
}

TEST_CASE("save preserves per-title sections", "[common][config]") {
    TempFile file(R"(
[graphics]
resolution_scale = 2

[title.0005000010143500]
graphics.resolution_scale = 1
)");

    Config config;
    REQUIRE(config.load(file.path()).ok);

    std::string error;
    REQUIRE(config.save(error));

    Config reloaded;
    REQUIRE(reloaded.load(file.path()).ok);
    CHECK(reloaded.base().resolution_scale == 2U);
    CHECK(reloaded.for_title("0005000010143500").resolution_scale == 1U);
}

TEST_CASE("describe_fields lists every setting", "[common][config]") {
    const std::string text = config::describe_fields();
    for (const auto& field : config::fields()) {
        CHECK(text.find(std::string{field.key}) != std::string::npos);
    }
}
