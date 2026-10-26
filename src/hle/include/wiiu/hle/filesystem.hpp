// Phase 6.9 — coreinit's FS API.
//
// Guest code opens `/vol/content/...` through FS calls; this layer turns those
// into the Phase 5 virtual filesystem, which is where the decision about
// containers (folder, WUP, disc image) actually lives.
//
// The SD card is a host directory, given to the emulator (`--sdcard`), which
// a title mounts at /vol/external01 the way it would on a console --
// FSGetMountSource for the SD card, then FSMount -- and which is not there
// until it does. Without one the slot is empty, and the mount source is not
// found.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/loader/vfs.hpp>

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
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

    /// Where the SD card's mount source mounts.
    static constexpr std::string_view sd_card_path = "/vol/external01";

    /// The SD card: a host directory, or an empty path for an empty slot.
    void set_sd_card(std::filesystem::path directory) { sd_card_ = std::move(directory); }

    [[nodiscard]] bool has_sd_card() const noexcept { return !sd_card_.empty(); }

    /// Mount the SD card at `sd_card_path`: `NotFound` with no card,
    /// `Exists` if it is mounted already, `FatalError` with no filesystem.
    [[nodiscard]] FsStatus mount_sd_card();

    /// FSUnmount: only the SD card is mounted this way.
    [[nodiscard]] FsStatus unmount(std::string_view target);

    [[nodiscard]] std::size_t open_file_count() const noexcept { return files_.size(); }

    [[nodiscard]] std::size_t client_count() const noexcept { return clients_.size(); }

private:
    loader::VirtualFileSystem* vfs_ = nullptr;

    std::map<u32, bool> clients_;
    std::map<u32, OpenFile> files_;
    std::map<u32, OpenDirectory> directories_;

    std::filesystem::path sd_card_;
    bool sd_card_mounted_ = false;

    u32 next_client_ = client_base;
    u32 next_file_ = file_base;
    u32 next_directory_ = directory_base;
};

} // namespace wiiu::hle
