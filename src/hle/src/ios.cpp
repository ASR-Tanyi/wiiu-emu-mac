// Phase 6.10 — IPC device table and the /dev/fsa device.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/ios.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <utility>

namespace wiiu::hle {

namespace {

/// A NUL-terminated guest string, bounded so a missing terminator cannot run
/// away through memory. The same rule `Call::read_string` follows, repeated
/// here because a device has memory but no `Call`.
[[nodiscard]] std::string
read_guest_string(const memory::Memory& memory, ppcaddr_t address, u32 limit) {
    std::string text;
    text.reserve(64);
    for (u32 offset = 0; offset < limit; ++offset) {
        u8 byte = 0;
        if (!memory.try_read<u8>(address + offset, byte) || byte == 0) {
            break;
        }
        text.push_back(static_cast<char>(byte));
    }
    return text;
}

/// The first few words of a buffer, for a log line that has to carry evidence
/// rather than a conclusion.
[[nodiscard]] std::string dump_words(const memory::Memory& memory, ppcaddr_t address, u32 size) {
    std::string text;
    const u32 words = std::min<u32>(size / 4, 8);
    for (u32 index = 0; index < words; ++index) {
        u32 value = 0;
        if (!memory.try_read<u32>(address + (index * 4), value)) {
            break;
        }
        if (!text.empty()) {
            text.push_back(' ');
        }
        text += fmt::format("{:08x}", value);
    }
    return text.empty() ? std::string{"<unreadable>"} : text;
}

/// Does this look like a path a title would ask for? Absolute, printable, and
/// not empty. Used to tell "the layout is right and the file is missing" from
/// "the layout is wrong and this is not a path at all", which are the same
/// `NotFound` otherwise and take very different work to fix.
[[nodiscard]] bool looks_like_path(std::string_view path) noexcept {
    if (path.empty() || path.front() != '/') {
        return false;
    }
    return std::all_of(path.begin(), path.end(), [](char character) {
        const auto byte = static_cast<unsigned char>(character);
        return byte >= 0x20 && byte < 0x7F;
    });
}

} // namespace

std::string_view ios_error_name(IosError error) noexcept {
    switch (error) {
        case IosError::Ok:
            return "OK";
        case IosError::Access:
            return "ACCESS";
        case IosError::Exists:
            return "EXISTS";
        case IosError::Interrupted:
            return "INTR";
        case IosError::Invalid:
            return "INVALID";
        case IosError::Max:
            return "MAX";
        case IosError::NoExists:
            return "NOEXISTS";
        case IosError::QueueEmpty:
            return "QEMPTY";
        case IosError::QueueFull:
            return "QFULL";
        case IosError::Unknown:
            return "UNKNOWN";
        case IosError::NotReady:
            return "NOTREADY";
    }
    return "?";
}

bool read_ios_vectors(const memory::Memory& memory,
                      ppcaddr_t address,
                      u32 count,
                      std::vector<IosVector>& out) {
    out.clear();
    if (count == 0) {
        return true;
    }
    if (address == 0) {
        return false;
    }

    out.reserve(count);
    for (u32 index = 0; index < count; ++index) {
        const ppcaddr_t base = address + (index * ios_vector_size);
        IosVector vector;
        if (!memory.try_read<u32>(base, vector.address) ||
            !memory.try_read<u32>(base + 4, vector.size) ||
            !memory.try_read<u32>(base + 8, vector.unused)) {
            return false;
        }
        out.push_back(vector);
    }
    return true;
}

// --------------------------------------------------------------------------
// IosDevice — the defaults
// --------------------------------------------------------------------------
//
// Refusing loudly rather than returning zero. A device that answers an
// unimplemented command with success is the failure mode that costs a day:
// the title carries on with a buffer nobody filled in.

IosError IosDevice::open(s32, IosOpenMode) {
    return IosError::Ok;
}

void IosDevice::close(s32) {}

s32 IosDevice::read(memory::Memory&, s32 handle, ppcaddr_t buffer, u32 size) {
    WIIU_LOG_WARN(Ios,
                  "{}: IOS_Read on handle {} ({:#010x}, {} bytes) is not implemented",
                  name(),
                  handle,
                  buffer,
                  size);
    return static_cast<s32>(IosError::Invalid);
}

s32 IosDevice::write(memory::Memory&, s32 handle, ppcaddr_t buffer, u32 size) {
    WIIU_LOG_WARN(Ios,
                  "{}: IOS_Write on handle {} ({:#010x}, {} bytes) is not implemented",
                  name(),
                  handle,
                  buffer,
                  size);
    return static_cast<s32>(IosError::Invalid);
}

s32 IosDevice::seek(s32 handle, s32 offset, IosSeekOrigin origin) {
    WIIU_LOG_WARN(Ios,
                  "{}: IOS_Seek on handle {} ({:+d} from {}) is not implemented",
                  name(),
                  handle,
                  offset,
                  static_cast<u32>(origin));
    return static_cast<s32>(IosError::Invalid);
}

IosError IosDevice::ioctl(memory::Memory&, const IosIoctl& request) {
    WIIU_LOG_WARN(Ios,
                  "{}: unimplemented ioctl {:#x} on handle {} (in {} bytes, out {} bytes)",
                  name(),
                  request.command,
                  request.handle,
                  request.input_size,
                  request.output_size);
    return IosError::Invalid;
}

IosError IosDevice::ioctlv(memory::Memory&, const IosIoctlv& request) {
    WIIU_LOG_WARN(Ios,
                  "{}: unimplemented ioctlv {:#x} on handle {} ({} in, {} out)",
                  name(),
                  request.command,
                  request.handle,
                  request.read_count,
                  request.write_count);
    return IosError::Invalid;
}

// --------------------------------------------------------------------------
// IosManager
// --------------------------------------------------------------------------

void IosManager::reset() {
    for (auto& [handle, open] : handles_) {
        if (open.device != nullptr) {
            open.device->close(handle);
        }
    }
    handles_.clear();
    next_handle_ = first_handle;
}

void IosManager::clear() {
    reset();
    devices_.clear();
}

void IosManager::add_device(std::string path, std::shared_ptr<IosDevice> device) {
    if (device == nullptr) {
        return;
    }
    WIIU_LOG_DEBUG(Ios, "device {} registered", path);
    devices_.insert_or_assign(std::move(path), std::move(device));
}

IosDevice* IosManager::device_for(std::string_view path) const {
    const auto entry = devices_.find(path);
    return entry == devices_.end() ? nullptr : entry->second.get();
}

s32 IosManager::open(std::string_view path, IosOpenMode mode) {
    IosDevice* const device = device_for(path);
    if (device == nullptr) {
        // The 6.12 policy, one level down: the gap is named rather than
        // papered over, because a handle that answers nothing is harder to
        // diagnose than an open that failed.
        WIIU_LOG_WARN(Ios, "IOS_Open('{}'): no such device", path);
        return static_cast<s32>(IosError::NoExists);
    }

    if (handles_.size() >= max_open_handles) {
        WIIU_LOG_ERROR(Ios,
                       "IOS_Open('{}'): {} handles are already open; refusing rather than growing",
                       path,
                       handles_.size());
        return static_cast<s32>(IosError::Max);
    }

    const s32 handle = next_handle_;
    const IosError accepted = device->open(handle, mode);
    if (accepted != IosError::Ok) {
        WIIU_LOG_WARN(Ios, "IOS_Open('{}'): device refused ({})", path, ios_error_name(accepted));
        return static_cast<s32>(accepted);
    }

    ++next_handle_;
    handles_.emplace(handle, OpenHandle{device, std::string{path}, mode});
    WIIU_LOG_DEBUG(Ios, "IOS_Open('{}') -> {}", path, handle);
    return handle;
}

IosError IosManager::close(s32 handle) {
    const auto entry = handles_.find(handle);
    if (entry == handles_.end()) {
        WIIU_LOG_WARN(Ios, "IOS_Close({}): not an open handle", handle);
        return IosError::Invalid;
    }
    if (entry->second.device != nullptr) {
        entry->second.device->close(handle);
    }
    handles_.erase(entry);
    return IosError::Ok;
}

const IosManager::OpenHandle* IosManager::find(s32 handle) const {
    const auto entry = handles_.find(handle);
    return entry == handles_.end() ? nullptr : &entry->second;
}

std::vector<std::string> IosManager::device_paths() const {
    std::vector<std::string> paths;
    paths.reserve(devices_.size());
    for (const auto& [path, device] : devices_) {
        static_cast<void>(device);
        paths.push_back(path);
    }
    return paths;
}

// --------------------------------------------------------------------------
// /dev/fsa
// --------------------------------------------------------------------------

namespace fsa {

std::string_view command_name(u32 command) noexcept {
    switch (static_cast<Command>(command)) {
        case Command::Mount:
            return "Mount";
        case Command::Unmount:
            return "Unmount";
        case Command::GetVolumeInfo:
            return "GetVolumeInfo";
        case Command::ChangeDir:
            return "ChangeDir";
        case Command::GetCwd:
            return "GetCwd";
        case Command::MakeDir:
            return "MakeDir";
        case Command::Remove:
            return "Remove";
        case Command::Rename:
            return "Rename";
        case Command::OpenDir:
            return "OpenDir";
        case Command::ReadDir:
            return "ReadDir";
        case Command::RewindDir:
            return "RewindDir";
        case Command::CloseDir:
            return "CloseDir";
        case Command::OpenFile:
            return "OpenFile";
        case Command::ReadFile:
            return "ReadFile";
        case Command::WriteFile:
            return "WriteFile";
        case Command::GetPosFile:
            return "GetPosFile";
        case Command::SetPosFile:
            return "SetPosFile";
        case Command::IsEof:
            return "IsEof";
        case Command::StatFile:
            return "StatFile";
        case Command::CloseFile:
            return "CloseFile";
        case Command::GetError:
            return "GetError";
        case Command::FlushFile:
            return "FlushFile";
        case Command::GetInfoByQuery:
            return "GetInfoByQuery";
        case Command::AppendFile:
            return "AppendFile";
        case Command::TruncateFile:
            return "TruncateFile";
    }
    return "?";
}

std::string_view status_name(Status status) noexcept {
    switch (status) {
        case Status::Ok:
            return "Ok";
        case Status::NotInit:
            return "NotInit";
        case Status::Busy:
            return "Busy";
        case Status::Cancelled:
            return "Cancelled";
        case Status::EndOfDir:
            return "EndOfDir";
        case Status::EndOfFile:
            return "EndOfFile";
        case Status::MaxMountpoints:
            return "MaxMountpoints";
        case Status::MaxVolumes:
            return "MaxVolumes";
        case Status::MaxClients:
            return "MaxClients";
        case Status::MaxFiles:
            return "MaxFiles";
        case Status::MaxDirs:
            return "MaxDirs";
        case Status::AlreadyOpen:
            return "AlreadyOpen";
        case Status::AlreadyExists:
            return "AlreadyExists";
        case Status::NotFound:
            return "NotFound";
        case Status::NotEmpty:
            return "NotEmpty";
        case Status::AccessError:
            return "AccessError";
        case Status::PermissionError:
            return "PermissionError";
        case Status::DataCorrupted:
            return "DataCorrupted";
        case Status::StorageFull:
            return "StorageFull";
        case Status::JournalFull:
            return "JournalFull";
        case Status::UnavailableCommand:
            return "UnavailableCommand";
        case Status::UnsupportedCommand:
            return "UnsupportedCommand";
        case Status::InvalidParam:
            return "InvalidParam";
        case Status::InvalidPath:
            return "InvalidPath";
        case Status::InvalidBuffer:
            return "InvalidBuffer";
        case Status::InvalidAlignment:
            return "InvalidAlignment";
        case Status::InvalidClientHandle:
            return "InvalidClientHandle";
        case Status::InvalidFileHandle:
            return "InvalidFileHandle";
        case Status::InvalidDirHandle:
            return "InvalidDirHandle";
        case Status::NotFile:
            return "NotFile";
        case Status::NotDir:
            return "NotDir";
        case Status::FileTooBig:
            return "FileTooBig";
        case Status::OutOfRange:
            return "OutOfRange";
        case Status::OutOfResources:
            return "OutOfResources";
        case Status::MediaNotReady:
            return "MediaNotReady";
        case Status::MediaError:
            return "MediaError";
        case Status::WriteProtected:
            return "WriteProtected";
        case Status::InvalidMedia:
            return "InvalidMedia";
    }
    return "?";
}

} // namespace fsa

fsa::Status fsa_status_for(FsStatus status, fsa::Status end) noexcept {
    using fsa::Status;
    switch (status) {
        case FsStatus::Ok:
            return Status::Ok;
        case FsStatus::Cancelled:
            return Status::Cancelled;
        case FsStatus::End:
            return end;
        case FsStatus::Max:
            return Status::MaxFiles;
        case FsStatus::AlreadyOpen:
            return Status::AlreadyOpen;
        case FsStatus::Exists:
            return Status::AlreadyExists;
        case FsStatus::NotFound:
            return Status::NotFound;
        case FsStatus::NotFile:
            return Status::NotFile;
        case FsStatus::NotDirectory:
            return Status::NotDir;
        case FsStatus::AccessError:
            return Status::AccessError;
        case FsStatus::PermissionError:
            return Status::PermissionError;
        case FsStatus::FileTooBig:
            return Status::FileTooBig;
        case FsStatus::StorageFull:
            return Status::StorageFull;
        case FsStatus::UnsupportedCommand:
            return Status::UnsupportedCommand;
        case FsStatus::MediaNotReady:
            return Status::MediaNotReady;
        case FsStatus::MediaError:
        case FsStatus::FatalError:
            return Status::MediaError;
        case FsStatus::Corrupted:
            return Status::DataCorrupted;
    }
    return Status::MediaError;
}

namespace {

/// A handle-based command's failure. The FS layer says `NotFound` for a
/// handle it has never heard of; IOSU says which kind of handle was wrong.
[[nodiscard]] fsa::Status handle_status(FsStatus status, bool directory) noexcept {
    if (status == FsStatus::NotFound) {
        return directory ? fsa::Status::InvalidDirHandle : fsa::Status::InvalidFileHandle;
    }
    return fsa_status_for(status, directory ? fsa::Status::EndOfDir : fsa::Status::EndOfFile);
}

} // namespace

IosError FsaDevice::open(s32 handle, IosOpenMode mode) {
    static_cast<void>(mode);
    WIIU_LOG_DEBUG(Ios, "/dev/fsa opened as handle {}", handle);
    return filesystem_ == nullptr ? IosError::NotReady : IosError::Ok;
}

void FsaDevice::close(s32 handle) {
    // Files opened through this device live in the same `FileSystem` the FS
    // API uses, and a title that closes /dev/fsa with files still open has
    // leaked them either way. Closing them here would also close the ones the
    // FS API opened, which is worse.
    WIIU_LOG_DEBUG(Ios, "/dev/fsa handle {} closed", handle);
}

void FsaDevice::report_mismatch(
    const memory::Memory& memory, u32 command, ppcaddr_t buffer, u32 size, std::string_view why) {
    ++layout_mismatches_;
    WIIU_LOG_ERROR(Ios,
                   "/dev/fsa {} ({:#x}): {}. The offsets in fsa::layout come from wut's "
                   "checked structures; if a real title disagrees, this is the evidence. "
                   "request {:#010x} ({} bytes): {}",
                   fsa::command_name(command),
                   command,
                   why,
                   buffer,
                   size,
                   dump_words(memory, buffer, size));
}

std::string FsaDevice::read_path(const memory::Memory& memory, const IosIoctl& request) {
    if (request.input == 0 || request.input_size <= fsa::layout::request_path) {
        report_mismatch(memory,
                        request.command,
                        request.input,
                        request.input_size,
                        "no request buffer to take a path from");
        return {};
    }

    const u32 available = request.input_size - fsa::layout::request_path;
    const std::string path = read_guest_string(memory,
                                               request.input + fsa::layout::request_path,
                                               std::min(available, fsa::layout::request_path_max));

    if (!looks_like_path(path)) {
        report_mismatch(memory,
                        request.command,
                        request.input,
                        request.input_size,
                        "the request does not hold a path where one is expected");
        return {};
    }
    return path;
}

bool FsaDevice::read_handle(const memory::Memory& memory, const IosIoctl& request, u32& handle) {
    if (request.input == 0 || request.input_size < fsa::layout::request_handle + sizeof(u32) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle)) {
        report_mismatch(
            memory, request.command, request.input, request.input_size, "no handle in the request");
        return false;
    }
    return true;
}

bool FsaDevice::write_stat(memory::Memory& memory, ppcaddr_t address, bool directory, u64 size) {
    using namespace fsa::layout;

    // Zero first: owner, group, quota, entry id and both timestamps are
    // reported as zero, as decaf reports them, rather than as whatever the
    // title's buffer held before.
    if (!memory.fill(address, stat_bytes, 0)) {
        return false;
    }
    bool ok = memory.try_write<u32>(address + stat_flags,
                                    directory ? stat_flag_directory : stat_flag_file);
    ok = memory.try_write<u32>(address + stat_mode, stat_default_mode) && ok;
    if (!directory) {
        // FSStat's sizes are 32 bits. Nothing on a Wii U disc is larger.
        ok = memory.try_write<u32>(address + stat_size, static_cast<u32>(size)) && ok;
        ok = memory.try_write<u32>(address + stat_alloc_size, static_cast<u32>(size)) && ok;
    }
    return ok;
}

fsa::Status FsaDevice::open_file(memory::Memory& memory, const IosIoctl& request) {
    const std::string path = read_path(memory, request);
    if (path.empty()) {
        return Status::InvalidPath;
    }

    std::string mode = "r";
    if (request.input_size > fsa::layout::request_open_mode) {
        const u32 available = request.input_size - fsa::layout::request_open_mode;
        const std::string given =
            read_guest_string(memory,
                              request.input + fsa::layout::request_open_mode,
                              std::min(available, fsa::layout::request_open_mode_max));
        if (!given.empty()) {
            mode = given;
        }
    }

    u32 handle = 0;
    const FsStatus status = filesystem_->open(path, mode, handle);
    if (status != FsStatus::Ok) {
        WIIU_LOG_DEBUG(
            Ios, "/dev/fsa OpenFile('{}', '{}'): {}", path, mode, fs_status_name(status));
        return fsa_status_for(status);
    }

    if (request.output == 0 ||
        !memory.try_write<u32>(request.output + fsa::layout::response_value, handle)) {
        static_cast<void>(filesystem_->close(handle));
        return Status::InvalidBuffer;
    }
    return Status::Ok;
}

IosError FsaDevice::read_file(memory::Memory& memory, const IosIoctlv& request) {
    using namespace fsa::layout;

    // One vector in (the request), two out (the data, then the response).
    if (request.read_count != 1 || request.write_count != 2 || request.vectors.size() < 3) {
        report_mismatch(memory,
                        request.command,
                        request.vectors.empty() ? 0 : request.vectors[0].address,
                        request.vectors.empty() ? 0 : request.vectors[0].size,
                        fmt::format("ReadFile wants 1 vector in and 2 out, not {} and {}",
                                    request.read_count,
                                    request.write_count));
        return fsa::as_reply(Status::InvalidParam);
    }

    const IosVector& header = request.vectors[rw_vector_request];
    const IosVector& data = request.vectors[rw_vector_data];

    u32 size = 0;
    u32 count = 0;
    u32 position = 0;
    u32 handle = 0;
    u32 flags = 0;
    if (header.size < request_rw_flags + sizeof(u32) ||
        !memory.try_read<u32>(header.address + request_rw_size, size) ||
        !memory.try_read<u32>(header.address + request_rw_count, count) ||
        !memory.try_read<u32>(header.address + request_rw_position, position) ||
        !memory.try_read<u32>(header.address + request_rw_handle, handle) ||
        !memory.try_read<u32>(header.address + request_rw_flags, flags)) {
        report_mismatch(memory,
                        request.command,
                        header.address,
                        header.size,
                        "the read request could not be decoded");
        return fsa::as_reply(Status::InvalidParam);
    }

    if ((flags & rw_flag_with_position) != 0) {
        if (const FsStatus sought = filesystem_->seek(handle, position); sought != FsStatus::Ok) {
            return fsa::as_reply(handle_status(sought, false));
        }
    }

    // The data vector is the real destination; the request's own buffer word
    // is coreinit's virtual address for the same bytes, and IOSU never reads
    // it. Reading no more than the vector holds keeps a short one from
    // turning into a write past its end.
    const u64 wanted = std::min<u64>(static_cast<u64>(size) * count, data.size);
    std::vector<u8> bytes;
    const FsStatus status = filesystem_->read(handle, static_cast<u32>(wanted), bytes);
    if (status == FsStatus::End) {
        return IosError::Ok; // zero bytes, which is what the caller will see
    }
    if (status != FsStatus::Ok) {
        return fsa::as_reply(handle_status(status, false));
    }

    // Whole items only, as IOSU counts them: a trailing partial item is not
    // reported as read. The position has still moved past it, as it does on
    // hardware's FAT layer.
    const auto moved = size == 0 ? 0U : static_cast<u32>((bytes.size() / size) * size);
    if (!memory.write_bytes(data.address, std::span<const u8>{bytes}.first(moved))) {
        return fsa::as_reply(Status::InvalidBuffer);
    }

    // On success the reply word is the number of bytes moved, not a status.
    return static_cast<IosError>(static_cast<s32>(moved));
}

fsa::Status FsaDevice::close_file(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (!read_handle(memory, request, handle)) {
        return Status::InvalidParam;
    }
    return handle_status(filesystem_->close(handle), false);
}

fsa::Status FsaDevice::get_pos_file(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (!read_handle(memory, request, handle)) {
        return Status::InvalidParam;
    }

    u64 position = 0;
    if (const FsStatus status = filesystem_->tell(handle, position); status != FsStatus::Ok) {
        return handle_status(status, false);
    }
    if (request.output == 0 || !memory.try_write<u32>(request.output + fsa::layout::response_value,
                                                      static_cast<u32>(position))) {
        return Status::InvalidBuffer;
    }
    return Status::Ok;
}

fsa::Status FsaDevice::set_pos_file(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    u32 position = 0;
    if (request.input == 0 || request.input_size < fsa::layout::request_position + sizeof(u32) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_position, position)) {
        report_mismatch(memory,
                        request.command,
                        request.input,
                        request.input_size,
                        "the seek request could not be decoded");
        return Status::InvalidParam;
    }
    return handle_status(filesystem_->seek(handle, position), false);
}

fsa::Status FsaDevice::is_eof(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (!read_handle(memory, request, handle)) {
        return Status::InvalidParam;
    }

    u64 position = 0;
    u64 size = 0;
    if (const FsStatus told = filesystem_->tell(handle, position); told != FsStatus::Ok) {
        return handle_status(told, false);
    }
    if (const FsStatus sized = filesystem_->size_of(handle, size); sized != FsStatus::Ok) {
        return handle_status(sized, false);
    }

    // The answer is the status itself: end-of-file when there is nothing left,
    // success when there is. Nothing is written to the response.
    return position >= size ? Status::EndOfFile : Status::Ok;
}

fsa::Status FsaDevice::stat_file(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (!read_handle(memory, request, handle)) {
        return Status::InvalidParam;
    }

    u64 size = 0;
    if (const FsStatus status = filesystem_->size_of(handle, size); status != FsStatus::Ok) {
        return handle_status(status, false);
    }
    if (request.output == 0 ||
        !write_stat(memory, request.output + fsa::layout::response_stat, false, size)) {
        return Status::InvalidBuffer;
    }
    return Status::Ok;
}

fsa::Status FsaDevice::get_info_by_query(memory::Memory& memory, const IosIoctl& request) {
    const std::string path = read_path(memory, request);
    if (path.empty()) {
        return Status::InvalidPath;
    }

    u32 type = 0;
    if (request.input_size < fsa::layout::request_query_type + sizeof(u32) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_query_type, type)) {
        report_mismatch(memory,
                        request.command,
                        request.input,
                        request.input_size,
                        "no query type after the path");
        return Status::InvalidParam;
    }

    // Stat is the question coreinit's `FSGetStat` asks, and the only one a
    // read-only virtual filesystem can answer honestly. Free space, journal
    // size and the rest are refused by name rather than answered with zero.
    if (type != fsa::layout::query_type_stat) {
        WIIU_LOG_WARN(
            Ios, "/dev/fsa GetInfoByQuery('{}'): query type {} is not supported", path, type);
        return Status::UnsupportedCommand;
    }

    loader::FileInfo info;
    if (const FsStatus status = filesystem_->stat(path, info); status != FsStatus::Ok) {
        return fsa_status_for(status);
    }
    if (request.output == 0 ||
        !write_stat(
            memory, request.output + fsa::layout::response_stat, info.is_directory, info.size)) {
        return Status::InvalidBuffer;
    }
    return Status::Ok;
}

fsa::Status FsaDevice::open_dir(memory::Memory& memory, const IosIoctl& request) {
    const std::string path = read_path(memory, request);
    if (path.empty()) {
        return Status::InvalidPath;
    }

    u32 handle = 0;
    const FsStatus status = filesystem_->open_directory(path, handle);
    if (status != FsStatus::Ok) {
        WIIU_LOG_DEBUG(Ios, "/dev/fsa OpenDir('{}'): {}", path, fs_status_name(status));
        return fsa_status_for(status, Status::EndOfDir);
    }
    if (request.output == 0 ||
        !memory.try_write<u32>(request.output + fsa::layout::response_value, handle)) {
        static_cast<void>(filesystem_->close_directory(handle));
        return Status::InvalidBuffer;
    }
    return Status::Ok;
}

fsa::Status FsaDevice::read_dir(memory::Memory& memory, const IosIoctl& request) {
    using namespace fsa::layout;

    u32 handle = 0;
    if (!read_handle(memory, request, handle)) {
        return Status::InvalidParam;
    }

    loader::FileInfo entry;
    if (const FsStatus status = filesystem_->read_directory(handle, entry);
        status != FsStatus::Ok) {
        return handle_status(status, true);
    }

    // Strictly greater: a response that ends exactly where the name begins
    // has no room for even the terminator, and computing the room as
    // `size - offset - 1` would wrap rather than come out negative.
    if (request.output == 0 || request.output_size <= response_entry_name) {
        report_mismatch(memory,
                        request.command,
                        request.input,
                        request.input_size,
                        "no room for a directory entry in the response");
        return Status::InvalidBuffer;
    }

    bool ok = write_stat(memory, request.output + response_entry, entry.is_directory, entry.size);

    const u32 room = std::min(request.output_size - response_entry_name, response_entry_name_max);
    const auto length = std::min<u32>(static_cast<u32>(entry.name.size()), room - 1);
    const std::vector<u8> name(entry.name.begin(), entry.name.begin() + length);
    ok = memory.write_bytes(request.output + response_entry_name, name) && ok;
    ok = memory.try_write<u8>(request.output + response_entry_name + length, 0) && ok;

    return ok ? Status::Ok : Status::InvalidBuffer;
}

fsa::Status FsaDevice::close_dir(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (!read_handle(memory, request, handle)) {
        return Status::InvalidParam;
    }
    return handle_status(filesystem_->close_directory(handle), true);
}

IosError FsaDevice::ioctl(memory::Memory& memory, const IosIoctl& request) {
    if (filesystem_ == nullptr) {
        return IosError::NotReady;
    }

    WIIU_LOG_TRACE(Ios,
                   "/dev/fsa {} ({:#x}) on handle {}: in {} bytes, out {} bytes",
                   fsa::command_name(request.command),
                   request.command,
                   request.handle,
                   request.input_size,
                   request.output_size);

    // coreinit's error injection: a negative first word is the answer.
    if (s32 injected = 0;
        request.input != 0 && request.input_size >= sizeof(u32) &&
        memory.try_read<s32>(request.input + fsa::layout::request_emulated_error, injected) &&
        injected < 0) {
        return static_cast<IosError>(injected);
    }

    switch (static_cast<fsa::Command>(request.command)) {
        case fsa::Command::OpenFile:
            return fsa::as_reply(open_file(memory, request));
        case fsa::Command::CloseFile:
            return fsa::as_reply(close_file(memory, request));
        case fsa::Command::GetPosFile:
            return fsa::as_reply(get_pos_file(memory, request));
        case fsa::Command::SetPosFile:
            return fsa::as_reply(set_pos_file(memory, request));
        case fsa::Command::IsEof:
            return fsa::as_reply(is_eof(memory, request));
        case fsa::Command::StatFile:
            return fsa::as_reply(stat_file(memory, request));
        case fsa::Command::GetInfoByQuery:
            return fsa::as_reply(get_info_by_query(memory, request));
        case fsa::Command::OpenDir:
            return fsa::as_reply(open_dir(memory, request));
        case fsa::Command::ReadDir:
            return fsa::as_reply(read_dir(memory, request));
        case fsa::Command::CloseDir:
            return fsa::as_reply(close_dir(memory, request));

        case fsa::Command::Mount:
        case fsa::Command::Unmount:
            // The Phase 5 VFS presents every mount from the start; there is
            // nothing to attach. Succeeding is honest here — the title asked
            // for a filesystem to be available, and it is.
            WIIU_LOG_DEBUG(Ios,
                           "/dev/fsa {}: the virtual filesystem is always mounted",
                           fsa::command_name(request.command));
            return IosError::Ok;

        case fsa::Command::FlushFile:
            // Nothing is written yet (6.9), so there is nothing buffered to push.
            return IosError::Ok;

        case fsa::Command::ReadFile:
        case fsa::Command::WriteFile:
            // These move bulk data and arrive as ioctlv on hardware. One sent
            // as a plain ioctl is not something IOSU would carry out, and
            // guessing where its data should go would be a specification.
            report_mismatch(memory,
                            request.command,
                            request.input,
                            request.input_size,
                            "arrived as an ioctl; IOSU takes this command as an ioctlv");
            return fsa::as_reply(Status::InvalidParam);

        case fsa::Command::GetVolumeInfo:
        case fsa::Command::ChangeDir:
        case fsa::Command::GetCwd:
        case fsa::Command::MakeDir:
        case fsa::Command::Remove:
        case fsa::Command::Rename:
        case fsa::Command::RewindDir:
        case fsa::Command::GetError:
        case fsa::Command::AppendFile:
        case fsa::Command::TruncateFile:
            break;
    }

    // Not implemented, or a command this table does not know. Either way the
    // default says so with its number and sizes, which is what makes the next
    // gap visible.
    return IosDevice::ioctl(memory, request);
}

IosError FsaDevice::ioctlv(memory::Memory& memory, const IosIoctlv& request) {
    if (filesystem_ == nullptr) {
        return IosError::NotReady;
    }

    if (static_cast<fsa::Command>(request.command) == fsa::Command::ReadFile) {
        return read_file(memory, request);
    }

    // WriteFile is refused by the filesystem underneath anyway (6.9, save
    // data in 9.8), and Mount's vectored form names a source device this
    // emulator does not have. Both report themselves through the default.
    return IosDevice::ioctlv(memory, request);
}

} // namespace wiiu::hle
