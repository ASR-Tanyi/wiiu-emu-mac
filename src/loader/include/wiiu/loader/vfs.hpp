// Phase 5.6 / 5.9 — the guest filesystem.
//
// Guest code asks for `/vol/content/foo.bin`. Where that comes from — a
// decrypted folder on disk, a WUP install, a disc image — is this layer's
// problem and nobody else's.
//
// Only the decrypted `/code`, `/content`, `/meta` layout is implemented here.
// It is the cheapest input path, it unblocks everything downstream, and the
// container formats (5.7, 5.8) plug in behind the same interface.
#pragma once

#include <wiiu/common/types.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wiiu::loader {

struct FileInfo {
    std::string name;
    bool is_directory = false;
    u64 size = 0;
};

/// Where a guest path is served from.
class Mount {
public:
    Mount() = default;
    virtual ~Mount() = default;

    Mount(const Mount&) = delete;
    Mount& operator=(const Mount&) = delete;

    [[nodiscard]] virtual bool exists(std::string_view path) const = 0;
    [[nodiscard]] virtual bool is_directory(std::string_view path) const = 0;
    [[nodiscard]] virtual std::optional<u64> size(std::string_view path) const = 0;
    [[nodiscard]] virtual bool read(std::string_view path, std::vector<u8>& out) const = 0;
    [[nodiscard]] virtual std::vector<FileInfo> list(std::string_view path) const = 0;
};

/// A mount backed by a directory on the host.
class HostMount final : public Mount {
public:
    explicit HostMount(std::filesystem::path root) : root_(std::move(root)) {}

    [[nodiscard]] bool exists(std::string_view path) const override;
    [[nodiscard]] bool is_directory(std::string_view path) const override;
    [[nodiscard]] std::optional<u64> size(std::string_view path) const override;
    [[nodiscard]] bool read(std::string_view path, std::vector<u8>& out) const override;
    [[nodiscard]] std::vector<FileInfo> list(std::string_view path) const override;

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

private:
    /// Guest path to host path, or nothing if it would escape the root.
    ///
    /// Guest code is untrusted: a title that asks for `/vol/content/../../..`
    /// is either broken or hostile, and either way must not reach outside its
    /// own directory.
    [[nodiscard]] std::optional<std::filesystem::path> resolve(std::string_view path) const;

    std::filesystem::path root_;
};

/// The guest-visible filesystem: a set of prefixes, each served by a mount.
class VirtualFileSystem {
public:
    /// Mount `source` at a guest prefix such as "/vol/content".
    void mount(std::string prefix, std::shared_ptr<Mount> source);
    void unmount(std::string_view prefix);

    /// Mount a decrypted title directory, the one containing code/, content/
    /// and meta/. Returns false if those subdirectories are not there.
    [[nodiscard]] bool mount_title_directory(const std::filesystem::path& directory,
                                             std::string& error);

    [[nodiscard]] bool exists(std::string_view path) const;
    [[nodiscard]] bool is_directory(std::string_view path) const;
    [[nodiscard]] std::optional<u64> size(std::string_view path) const;
    [[nodiscard]] bool read(std::string_view path, std::vector<u8>& out) const;
    [[nodiscard]] std::vector<FileInfo> list(std::string_view path) const;

    [[nodiscard]] std::vector<std::string> mount_points() const;

private:
    struct Entry {
        std::string prefix;
        std::shared_ptr<Mount> source;
    };

    /// Longest matching prefix wins, so "/vol/content/sub" can be mounted
    /// separately from "/vol/content".
    [[nodiscard]] const Entry* find(std::string_view path, std::string& relative) const;

    std::vector<Entry> mounts_;
};

/// Collapse `.`, `..` and repeated separators. Leading `..` are dropped rather
/// than escaping, so the result is always inside the mount.
[[nodiscard]] std::string normalize_path(std::string_view path);

} // namespace wiiu::loader
