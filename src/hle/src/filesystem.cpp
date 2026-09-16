#include <wiiu/common/log.hpp>
#include <wiiu/hle/filesystem.hpp>

#include <algorithm>

namespace wiiu::hle {
namespace {

using wiiu::log::Channel;

} // namespace

std::string_view fs_status_name(FsStatus status) noexcept {
    switch (status) {
        case FsStatus::Ok:
            return "ok";
        case FsStatus::Cancelled:
            return "cancelled";
        case FsStatus::End:
            return "end of file";
        case FsStatus::Max:
            return "too many open";
        case FsStatus::AlreadyOpen:
            return "already open";
        case FsStatus::Exists:
            return "exists";
        case FsStatus::NotFound:
            return "not found";
        case FsStatus::NotFile:
            return "not a file";
        case FsStatus::NotDirectory:
            return "not a directory";
        case FsStatus::AccessError:
            return "access error";
        case FsStatus::PermissionError:
            return "permission error";
        case FsStatus::FileTooBig:
            return "file too big";
        case FsStatus::StorageFull:
            return "storage full";
        case FsStatus::UnsupportedCommand:
            return "unsupported command";
        case FsStatus::MediaNotReady:
            return "media not ready";
        case FsStatus::MediaError:
            return "media error";
        case FsStatus::Corrupted:
            return "corrupted";
        case FsStatus::FatalError:
            return "fatal error";
    }
    return "?";
}

void FileSystem::reset() {
    clients_.clear();
    files_.clear();
    directories_.clear();
    next_client_ = client_base;
    next_file_ = file_base;
    next_directory_ = directory_base;
}

u32 FileSystem::add_client() {
    const u32 handle = next_client_;
    next_client_ += 4;
    clients_[handle] = true;
    return handle;
}

bool FileSystem::remove_client(u32 handle) {
    return clients_.erase(handle) != 0;
}

bool FileSystem::has_client(u32 handle) const {
    return clients_.contains(handle);
}

FsStatus FileSystem::open(std::string_view path, std::string_view mode, u32& handle) {
    if (vfs_ == nullptr) {
        return FsStatus::MediaNotReady;
    }

    // Writes are not supported: the content filesystem is read-only, and save
    // data (Phase 9.9) will be a separate mount rather than a write path
    // grafted on here.
    if (mode.find('w') != std::string_view::npos || mode.find('a') != std::string_view::npos) {
        WIIU_LOG_WARN(Coreinit, "FSOpenFile('{}', '{}'): writing is not supported", path, mode);
        return FsStatus::PermissionError;
    }

    if (!vfs_->exists(path)) {
        return FsStatus::NotFound;
    }
    if (vfs_->is_directory(path)) {
        return FsStatus::NotFile;
    }

    OpenFile file;
    file.path = std::string{path};
    // Read the whole file up front. Game content is a few megabytes at most
    // per file, and it keeps reads from having to hold host file handles
    // across an emulated frame.
    if (!vfs_->read(path, file.contents)) {
        return FsStatus::AccessError;
    }

    handle = next_file_;
    next_file_ += 4;
    files_.emplace(handle, std::move(file));
    return FsStatus::Ok;
}

FsStatus FileSystem::read(u32 handle, u32 size, std::vector<u8>& out) {
    const auto it = files_.find(handle);
    if (it == files_.end()) {
        return FsStatus::NotFound;
    }

    OpenFile& file = it->second;
    if (file.position >= file.contents.size()) {
        out.clear();
        return FsStatus::End;
    }

    const auto available = static_cast<u32>(file.contents.size() - file.position);
    const u32 count = std::min(size, available);

    out.assign(file.contents.begin() + static_cast<std::ptrdiff_t>(file.position),
               file.contents.begin() + static_cast<std::ptrdiff_t>(file.position + count));
    file.position += count;
    return FsStatus::Ok;
}

FsStatus FileSystem::close(u32 handle) {
    return files_.erase(handle) != 0 ? FsStatus::Ok : FsStatus::NotFound;
}

FsStatus FileSystem::seek(u32 handle, u64 position) {
    const auto it = files_.find(handle);
    if (it == files_.end()) {
        return FsStatus::NotFound;
    }
    // Seeking past the end is legal; the next read reports end of file.
    it->second.position = position;
    return FsStatus::Ok;
}

FsStatus FileSystem::tell(u32 handle, u64& position) const {
    const auto it = files_.find(handle);
    if (it == files_.end()) {
        return FsStatus::NotFound;
    }
    position = it->second.position;
    return FsStatus::Ok;
}

FsStatus FileSystem::size_of(u32 handle, u64& size) const {
    const auto it = files_.find(handle);
    if (it == files_.end()) {
        return FsStatus::NotFound;
    }
    size = it->second.contents.size();
    return FsStatus::Ok;
}

FsStatus FileSystem::open_directory(std::string_view path, u32& handle) {
    if (vfs_ == nullptr) {
        return FsStatus::MediaNotReady;
    }
    if (!vfs_->exists(path)) {
        return FsStatus::NotFound;
    }
    if (!vfs_->is_directory(path)) {
        return FsStatus::NotDirectory;
    }

    OpenDirectory directory;
    directory.path = std::string{path};
    directory.entries = vfs_->list(path);

    handle = next_directory_;
    next_directory_ += 4;
    directories_.emplace(handle, std::move(directory));
    return FsStatus::Ok;
}

FsStatus FileSystem::read_directory(u32 handle, loader::FileInfo& out) {
    const auto it = directories_.find(handle);
    if (it == directories_.end()) {
        return FsStatus::NotFound;
    }

    OpenDirectory& directory = it->second;
    if (directory.next >= directory.entries.size()) {
        return FsStatus::End;
    }
    out = directory.entries[directory.next++];
    return FsStatus::Ok;
}

FsStatus FileSystem::close_directory(u32 handle) {
    return directories_.erase(handle) != 0 ? FsStatus::Ok : FsStatus::NotFound;
}

FsStatus FileSystem::stat(std::string_view path, loader::FileInfo& out) const {
    if (vfs_ == nullptr) {
        return FsStatus::MediaNotReady;
    }
    if (!vfs_->exists(path)) {
        return FsStatus::NotFound;
    }

    out.name = std::string{path};
    out.is_directory = vfs_->is_directory(path);
    out.size = out.is_directory ? 0 : vfs_->size(path).value_or(0);
    return FsStatus::Ok;
}

} // namespace wiiu::hle
