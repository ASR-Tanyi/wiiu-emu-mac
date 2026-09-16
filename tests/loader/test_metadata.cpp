#include <wiiu/loader/metadata.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

using namespace wiiu;
using namespace wiiu::loader;

namespace {

[[nodiscard]] std::filesystem::path title_path(std::string_view relative) {
    return std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title" / relative;
}

} // namespace

TEST_CASE("cos.xml gives the loader its memory layout", "[loader][metadata]") {
    CosSettings cos;
    const MetadataResult result = load_cos_xml(title_path("code/cos.xml"), cos);

    INFO("error: " << result.error);
    REQUIRE(result.ok);

    // hexBinary values carry no 0x prefix, so a decimal reading would give
    // wildly wrong sizes rather than failing.
    CHECK(cos.max_size == 0x40000000U);
    CHECK(cos.max_codesize == 0x00800000U);
    CHECK(cos.codegen_size == 0x02000000U);
    CHECK(cos.codegen_core == 1U);

    CHECK(cos.stack_size[0] == 0x10000U);
    CHECK(cos.stack_size[2] == 0x10000U);
    CHECK(cos.exception_stack0_size == 0x1000U);

    REQUIRE(cos.permissions.size() == 2);
    CHECK(cos.permissions.contains("3"));
    CHECK(cos.permissions.at("11") == 0xFFU);
}

TEST_CASE("app.xml gives the title id", "[loader][metadata]") {
    AppSettings app;
    const MetadataResult result = load_app_xml(title_path("code/app.xml"), app);

    INFO("error: " << result.error);
    REQUIRE(result.ok);

    CHECK(app.title_id == 0x0005000010143500ULL);
    CHECK(app.title_id_string() == "0005000010143500");
    CHECK(app.os_version == 0x000500101000400AULL);
    CHECK(app.app_type == 0x80000000U);
    CHECK(app.group_id == 0x1435U);
}

TEST_CASE("the title id string matches the config-file spelling", "[loader][metadata]") {
    // Per-title config sections are keyed by this, so the two have to agree.
    AppSettings app;
    app.title_id = 0x000500001017BB00ULL;
    CHECK(app.title_id_string() == "000500001017bb00");
}

TEST_CASE("meta.xml gives the human-readable names", "[loader][metadata]") {
    MetaSettings meta;
    const MetadataResult result = load_meta_xml(title_path("meta/meta.xml"), meta);

    INFO("error: " << result.error);
    REQUIRE(result.ok);

    CHECK(meta.title_id == 0x0005000010143500ULL);
    CHECK(meta.product_code == "WUP-P-TEST");
    CHECK(meta.company_code == "0001");

    CHECK(meta.long_names.at("en") == "Test Title");
    CHECK(meta.short_names.at("en") == "Test");
    CHECK(meta.publishers.at("en") == "Alpha Star Research");
    CHECK(meta.display_name() == "Test Title");

    // Non-ASCII names survive: these files are UTF-8 and a game library will
    // show whatever the title actually calls itself.
    CHECK(meta.long_names.contains("ja"));
    CHECK_FALSE(meta.long_names.at("ja").empty());
}

TEST_CASE("display_name falls back when there is no English", "[loader][metadata]") {
    MetaSettings meta;
    meta.long_names["ja"] = "タイトル";
    CHECK(meta.display_name() == "タイトル");

    meta.long_names.clear();
    CHECK(meta.display_name().empty());
}

TEST_CASE("malformed XML is an error, not a crash", "[loader][metadata]") {
    CosSettings cos;
    const MetadataResult result = parse_cos_xml("<app><version>1</app>", cos);
    CHECK_FALSE(result.ok);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("the wrong root element is rejected", "[loader][metadata]") {
    AppSettings app;
    const MetadataResult result = parse_app_xml("<other><title_id>1</title_id></other>", app);
    CHECK_FALSE(result.ok);
    CHECK(result.error.find("<app>") != std::string::npos);
}

TEST_CASE("a missing file is reported", "[loader][metadata]") {
    CosSettings cos;
    const MetadataResult result = load_cos_xml(title_path("code/nothere.xml"), cos);
    CHECK_FALSE(result.ok);
    CHECK(result.error.find("cannot open") != std::string::npos);
}

TEST_CASE("an empty title id warns", "[loader][metadata]") {
    AppSettings app;
    const MetadataResult result = parse_app_xml("<app><version>1</version></app>", app);
    REQUIRE(result.ok);
    CHECK(app.title_id == 0U);
    CHECK_FALSE(result.warnings.empty());
}
