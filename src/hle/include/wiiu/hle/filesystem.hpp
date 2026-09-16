// Phase 6.9 — coreinit's FS API.
//
// Guest code opens `/vol/content/...` through FS calls; this layer turns those
// into the Phase 5 virtual filesystem, which is where the decision about
// containers (folder, WUP, disc image) actually lives.
//
// Only the synchronous forms are implemented. The async ones take a callback
// and a queue, and without a scheduler (Phase 6.4) there is nowhere to run the
// callback from — so they are stubs that say so rather than silently never
// completing, which is the failure mode that wastes a day.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/loader/vfs.hpp>

#include <map>
#include <string>
#include <vector>

namespace wiiu::hle {

/// FSStatus. Zero is success; failures are negative, as the API defines them.
enum class FsStatus : s32 {
    Ok = 0,
    Cancelled = -1,
    End = -2,
    Max = -3,
    AlreadyOpen = -4,
    Exists = -5,
    NotFound = -6,
    NotFile = -7,
    NotDirectory = -8,
    AccessError = -9,
    PermissionError = -10,
    FileTooBig = -11,
    StorageFull = -12,
    UnsupportedCommand = -14,
    MediaNotReady = -15,
    MediaError = -17,
    Corrupted = -18,
    FatalError = -0x400,
};

[[nodiscard]] std::string_view fs_status_name(FsStatus status) noexcept;

struct OpenFile {
    std::string path;
    std::vector<u8> contents;
    u64 position = 0;
};

struct OpenDirectory {
    std::string path;
    std::vector<loader::FileInfo> entries;
    std::size_t next = 0;
};

/// The FS state a title accumulates: clients, open files, open directories.
class FileSystem {
public:
    /// Handles start high enough that a zero or small value is obviously not
    /// one, which makes an uninitialised handle easy to spot.
    static constexpr u32 client_base = 0x5C000000;
    static constexpr u32 file_base = 0x5F000000;
    static constexpr u32 directory_base = 0x5D000000;

    void reset();

    void set_filesystem(loader::VirtualFileSystem* vfs) noexcept { vfs_ = vfs; }

    [[nodiscard]] loader::VirtualFileSystem* filesystem() const noexcept { return vfs_; }

    [[nodiscard]] u32 add_client();
    [[nodiscard]] bool remove_client(u32 handle);
    [[nodiscard]] bool has_client(u32 handle) const;

    [[nodiscard]] FsStatus open(std::string_view path, std::string_view mode, u32& handle);
    [[nodiscard]] FsStatus read(u32 handle, u32 size, std::vector<u8>& out);
    [[nodiscard]] FsStatus close(u32 handle);
    [[nodiscard]] FsStatus seek(u32 handle, u64 position);
    [[nodiscard]] FsStatus tell(u32 handle, u64& position) const;
    [[nodiscard]] FsStatus size_of(u32 handle, u64& size) const;

    [[nodiscard]] FsStatus open_directory(std::string_view path, u32& handle);
    [[nodiscard]] FsStatus read_directory(u32 handle, loader::FileInfo& out);
    [[nodiscard]] FsStatus close_directory(u32 handle);

    [[nodiscard]] FsStatus stat(std::string_view path, loader::FileInfo& out) const;

    [[nodiscard]] std::size_t open_file_count() const noexcept { return files_.size(); }

    [[nodiscard]] std::size_t client_count() const noexcept { return clients_.size(); }

private:
    loader::VirtualFileSystem* vfs_ = nullptr;

    std::map<u32, bool> clients_;
    std::map<u32, OpenFile> files_;
    std::map<u32, OpenDirectory> directories_;

    u32 next_client_ = client_base;
    u32 next_file_ = file_base;
    u32 next_directory_ = directory_base;
};

} // namespace wiiu::hle
