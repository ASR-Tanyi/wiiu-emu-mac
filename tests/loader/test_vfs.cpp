#include <wiiu/loader/vfs.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <string>

using namespace wiiu;
using namespace wiiu::loader;

namespace {

[[nodiscard]] std::filesystem::path title_directory() {
    return std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title";
}

} // namespace

TEST_CASE("path normalisation", "[loader][vfs]") {
    CHECK(normalize_path("/vol/content/file.bin") == "/vol/content/file.bin");
    CHECK(normalize_path("vol/content") == "/vol/content");
    CHECK(normalize_path("/vol//content///file") == "/vol/content/file");
    CHECK(normalize_path("/vol/./content/./file") == "/vol/content/file");
    CHECK(normalize_path("/vol/content/sub/../file") == "/vol/content/file");
    CHECK(normalize_path("/") == "/");
    CHECK(normalize_path("") == "/");
}

TEST_CASE("climbing out of the root is not possible", "[loader][vfs]") {
    // Guest code is untrusted. A title asking for this is broken or hostile,
    // and either way must not reach the host filesystem.
    CHECK(normalize_path("/vol/content/../../../etc/passwd") == "/etc/passwd");
    CHECK(normalize_path("/../../..") == "/");
    CHECK(normalize_path("../secret") == "/secret");
}

TEST_CASE("a title directory mounts", "[loader][vfs]") {
    VirtualFileSystem vfs;
    std::string error;
    INFO("error: " << error);
    REQUIRE(vfs.mount_title_directory(title_directory(), error));

    const auto mounts = vfs.mount_points();
    CHECK(std::ranges::find(mounts, "/vol/code") != mounts.end());
    CHECK(std::ranges::find(mounts, "/vol/content") != mounts.end());
    CHECK(std::ranges::find(mounts, "/vol/meta") != mounts.end());
}

TEST_CASE("a directory without code/content/meta is refused", "[loader][vfs]") {
    VirtualFileSystem vfs;
    std::string error;
    CHECK_FALSE(vfs.mount_title_directory(title_directory() / "code", error));
    CHECK(error.find("directory") != std::string::npos);

    CHECK_FALSE(vfs.mount_title_directory(title_directory() / "nothere", error));
    CHECK(error.find("not a directory") != std::string::npos);
}

TEST_CASE("files are readable through the guest path", "[loader][vfs]") {
    VirtualFileSystem vfs;
    std::string error;
    REQUIRE(vfs.mount_title_directory(title_directory(), error));

    CHECK(vfs.exists("/vol/content/data.bin"));
    CHECK(vfs.exists("/vol/code/cos.xml"));
    CHECK(vfs.exists("/vol/meta/meta.xml"));
    CHECK_FALSE(vfs.exists("/vol/content/nothere.bin"));

    std::vector<u8> contents;
    REQUIRE(vfs.read("/vol/content/data.bin", contents));
    CHECK(std::string(contents.begin(), contents.end()) == "content payload\n");

    const auto size = vfs.size("/vol/content/data.bin");
    REQUIRE(size.has_value());
    CHECK(*size == contents.size());
}

TEST_CASE("nested directories work", "[loader][vfs]") {
    VirtualFileSystem vfs;
    std::string error;
    REQUIRE(vfs.mount_title_directory(title_directory(), error));

    CHECK(vfs.is_directory("/vol/content/sub"));
    CHECK_FALSE(vfs.is_directory("/vol/content/data.bin"));

    std::vector<u8> contents;
    REQUIRE(vfs.read("/vol/content/sub/nested.bin", contents));
    CHECK(std::string(contents.begin(), contents.end()) == "nested\n");
}

TEST_CASE("listing is sorted", "[loader][vfs]") {
    // Directory iteration order is not defined, and a title that walks its
    // content directory should see the same thing every run.
    VirtualFileSystem vfs;
    std::string error;
    REQUIRE(vfs.mount_title_directory(title_directory(), error));

    const auto entries = vfs.list("/vol/content");
    REQUIRE(entries.size() == 2);
    CHECK(entries[0].name == "data.bin");
    CHECK_FALSE(entries[0].is_directory);
    CHECK(entries[1].name == "sub");
    CHECK(entries[1].is_directory);

    CHECK(std::ranges::is_sorted(
        entries, [](const FileInfo& lhs, const FileInfo& rhs) { return lhs.name < rhs.name; }));
}

TEST_CASE("traversal cannot reach the host filesystem", "[loader][vfs]") {
    VirtualFileSystem vfs;
    std::string error;
    REQUIRE(vfs.mount_title_directory(title_directory(), error));

    // Worth being precise about what the guarantee is. `..` is resolved inside
    // the *guest* namespace, so a path that walks up and back down reaches
    // another mount -- which the guest could have addressed directly anyway,
    // so nothing is gained by it.
    CHECK(vfs.exists("/vol/content/../code/cos.xml"));

    // What must never happen is reaching the host filesystem. Both layers stop
    // that: normalisation cannot climb above the guest root, and each mount
    // refuses a host path outside its own directory.
    CHECK_FALSE(vfs.exists("/vol/content/../../../../etc/passwd"));
    CHECK_FALSE(vfs.exists("/vol/content/../../etc/passwd"));

    std::vector<u8> contents;
    CHECK_FALSE(vfs.read("/vol/content/../../etc/hostname", contents));

    // And a mount handed a climbing path directly still refuses.
    const HostMount mount{title_directory() / "content"};
    CHECK_FALSE(mount.exists("/../code/cos.xml"));
    CHECK_FALSE(mount.exists("/../../../etc/passwd"));
}

TEST_CASE("an unmounted path finds nothing", "[loader][vfs]") {
    VirtualFileSystem vfs;
    CHECK_FALSE(vfs.exists("/vol/content/data.bin"));
    CHECK(vfs.list("/vol/content").empty());
    CHECK_FALSE(vfs.size("/anything").has_value());
}

TEST_CASE("a longer mount prefix wins", "[loader][vfs]") {
    VirtualFileSystem vfs;
    vfs.mount("/vol/content", std::make_shared<HostMount>(title_directory() / "content"));
    vfs.mount("/vol/content/sub", std::make_shared<HostMount>(title_directory() / "meta"));

    // The more specific mount shadows the general one.
    CHECK(vfs.exists("/vol/content/sub/meta.xml"));
    CHECK(vfs.exists("/vol/content/data.bin"));
}

TEST_CASE("a prefix only matches on a path boundary", "[loader][vfs]") {
    // "/vol/contentious" must not be served by the "/vol/content" mount.
    VirtualFileSystem vfs;
    vfs.mount("/vol/content", std::make_shared<HostMount>(title_directory() / "content"));

    CHECK(vfs.exists("/vol/content/data.bin"));
    CHECK_FALSE(vfs.exists("/vol/contentious/data.bin"));
}

TEST_CASE("unmounting removes the prefix", "[loader][vfs]") {
    VirtualFileSystem vfs;
    std::string error;
    REQUIRE(vfs.mount_title_directory(title_directory(), error));
    REQUIRE(vfs.exists("/vol/content/data.bin"));

    vfs.unmount("/vol/content");
    CHECK_FALSE(vfs.exists("/vol/content/data.bin"));
    CHECK(vfs.exists("/vol/meta/meta.xml"));
}
