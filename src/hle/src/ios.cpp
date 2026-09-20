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
        case Command::GetStatFile:
            return "GetStatFile";
        case Command::CloseFile:
            return "CloseFile";
        case Command::FlushFile:
            return "FlushFile";
    }
    return "?";
}

} // namespace fsa

IosError ios_error_for(FsStatus status) noexcept {
    switch (status) {
        case FsStatus::Ok:
            return IosError::Ok;
        case FsStatus::NotFound:
            return IosError::NoExists;
        case FsStatus::Exists:
            return IosError::Exists;
        case FsStatus::AlreadyOpen:
            return IosError::Exists;
        case FsStatus::Max:
            return IosError::Max;
        case FsStatus::PermissionError:
        case FsStatus::AccessError:
            return IosError::Access;
        case FsStatus::MediaNotReady:
            return IosError::NotReady;
        case FsStatus::Cancelled:
            return IosError::Interrupted;
        case FsStatus::End:
        case FsStatus::NotFile:
        case FsStatus::NotDirectory:
        case FsStatus::FileTooBig:
        case FsStatus::StorageFull:
        case FsStatus::UnsupportedCommand:
        case FsStatus::MediaError:
        case FsStatus::Corrupted:
        case FsStatus::FatalError:
            break;
    }
    return IosError::Invalid;
}

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

void FsaDevice::report_mismatch(const memory::Memory& memory,
                                const IosIoctl& request,
                                std::string_view why) {
    ++layout_mismatches_;
    WIIU_LOG_ERROR(Ios,
                   "/dev/fsa {} ({:#x}): {}. The offsets in fsa::layout are this project's "
                   "reading of the format and have not been checked against a real title -- "
                   "this is the evidence to fix them with. request {:#010x} ({} bytes): {}",
                   fsa::command_name(request.command),
                   request.command,
                   why,
                   request.input,
                   request.input_size,
                   dump_words(memory, request.input, request.input_size));
}

std::string FsaDevice::read_path(const memory::Memory& memory, const IosIoctl& request) {
    if (request.input == 0 || request.input_size <= fsa::layout::request_path) {
        report_mismatch(memory, request, "no request buffer to take a path from");
        return {};
    }

    const u32 available = request.input_size - fsa::layout::request_path;
    const std::string path = read_guest_string(memory,
                                               request.input + fsa::layout::request_path,
                                               std::min(available, fsa::layout::request_path_max));

    if (!looks_like_path(path)) {
        report_mismatch(memory, request, "the request does not hold a path where one is expected");
        return {};
    }
    return path;
}

IosError FsaDevice::open_file(memory::Memory& memory, const IosIoctl& request) {
    const std::string path = read_path(memory, request);
    if (path.empty()) {
        return IosError::Invalid;
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
        return ios_error_for(status);
    }

    if (request.output == 0 ||
        !memory.try_write<u32>(request.output + fsa::layout::response_value, handle)) {
        static_cast<void>(filesystem_->close(handle));
        return IosError::Invalid;
    }
    return IosError::Ok;
}

IosError FsaDevice::read_file(memory::Memory& memory, const IosIoctl& request) {
    u32 buffer = 0;
    u32 item_size = 0;
    u32 count = 0;
    u32 handle = 0;
    if (request.input == 0 || request.input_size < fsa::layout::request_read_handle + sizeof(u32) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_buffer, buffer) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_item_size, item_size) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_item_count, count) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_read_handle, handle)) {
        report_mismatch(memory, request, "the read request could not be decoded");
        return IosError::Invalid;
    }

    const u64 wanted = static_cast<u64>(item_size) * count;
    std::vector<u8> data;
    const FsStatus status = filesystem_->read(handle, static_cast<u32>(wanted), data);
    if (status == FsStatus::End) {
        return IosError::Ok; // zero items, which is what the caller will read
    }
    if (status != FsStatus::Ok) {
        return ios_error_for(status);
    }
    if (!memory.write_bytes(buffer, data)) {
        return IosError::Invalid;
    }

    // Like the FS API, the count is in *items*.
    const u32 read = item_size == 0 ? 0 : static_cast<u32>(data.size() / item_size);
    if (request.output != 0) {
        static_cast<void>(
            memory.try_write<u32>(request.output + fsa::layout::response_value, read));
    }
    return IosError::Ok;
}

IosError FsaDevice::close_file(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (request.input == 0 ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle)) {
        report_mismatch(memory, request, "no handle in the request");
        return IosError::Invalid;
    }
    return ios_error_for(filesystem_->close(handle));
}

IosError FsaDevice::get_pos_file(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (request.input == 0 ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle)) {
        report_mismatch(memory, request, "no handle in the request");
        return IosError::Invalid;
    }

    u64 position = 0;
    const FsStatus status = filesystem_->tell(handle, position);
    if (status != FsStatus::Ok) {
        return ios_error_for(status);
    }
    if (request.output == 0 || !memory.try_write<u32>(request.output + fsa::layout::response_value,
                                                      static_cast<u32>(position))) {
        return IosError::Invalid;
    }
    return IosError::Ok;
}

IosError FsaDevice::set_pos_file(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    u32 position = 0;
    if (request.input == 0 || request.input_size < fsa::layout::request_position + sizeof(u32) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle) ||
        !memory.try_read<u32>(request.input + fsa::layout::request_position, position)) {
        report_mismatch(memory, request, "the seek request could not be decoded");
        return IosError::Invalid;
    }
    return ios_error_for(filesystem_->seek(handle, position));
}

IosError FsaDevice::is_eof(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (request.input == 0 ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle)) {
        report_mismatch(memory, request, "no handle in the request");
        return IosError::Invalid;
    }

    u64 position = 0;
    u64 size = 0;
    const FsStatus told = filesystem_->tell(handle, position);
    if (told != FsStatus::Ok) {
        return ios_error_for(told);
    }
    const FsStatus sized = filesystem_->size_of(handle, size);
    if (sized != FsStatus::Ok) {
        return ios_error_for(sized);
    }

    // Hardware answers this one with a status of its own rather than a flag.
    // `IosError` has no value for end-of-file, and inventing one would be a
    // guess on top of a guess, so the answer goes in the response word and
    // the call itself succeeds — it did, after all, find out.
    if (request.output == 0 || !memory.try_write<u32>(request.output + fsa::layout::response_value,
                                                      position >= size ? 1U : 0U)) {
        return IosError::Invalid;
    }
    return IosError::Ok;
}

IosError FsaDevice::open_dir(memory::Memory& memory, const IosIoctl& request) {
    const std::string path = read_path(memory, request);
    if (path.empty()) {
        return IosError::Invalid;
    }

    u32 handle = 0;
    const FsStatus status = filesystem_->open_directory(path, handle);
    if (status != FsStatus::Ok) {
        WIIU_LOG_DEBUG(Ios, "/dev/fsa OpenDir('{}'): {}", path, fs_status_name(status));
        return ios_error_for(status);
    }
    if (request.output == 0 ||
        !memory.try_write<u32>(request.output + fsa::layout::response_value, handle)) {
        static_cast<void>(filesystem_->close_directory(handle));
        return IosError::Invalid;
    }
    return IosError::Ok;
}

IosError FsaDevice::read_dir(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (request.input == 0 ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle)) {
        report_mismatch(memory, request, "no handle in the request");
        return IosError::Invalid;
    }

    loader::FileInfo entry;
    const FsStatus status = filesystem_->read_directory(handle, entry);
    if (status != FsStatus::Ok) {
        return ios_error_for(status);
    }

    // Strictly greater: a response that ends exactly where the name begins
    // has no room for even the terminator, and computing the room as
    // `size - offset - 1` would wrap rather than come out negative.
    if (request.output == 0 || request.output_size <= fsa::layout::response_entry_name) {
        report_mismatch(memory, request, "no room for a directory entry in the response");
        return IosError::Invalid;
    }

    const u32 flags = entry.is_directory ? fsa::layout::entry_flag_directory : 0U;
    bool ok = memory.try_write<u32>(request.output + fsa::layout::response_entry_flags, flags);
    ok = memory.try_write<u32>(request.output + fsa::layout::response_entry_size,
                               static_cast<u32>(entry.size)) &&
         ok;

    const u32 room = std::min(request.output_size - fsa::layout::response_entry_name,
                              fsa::layout::response_entry_name_max);
    const auto length = std::min<u32>(static_cast<u32>(entry.name.size()), room - 1);
    const std::vector<u8> name(entry.name.begin(), entry.name.begin() + length);
    ok = memory.write_bytes(request.output + fsa::layout::response_entry_name, name) && ok;
    ok = memory.try_write<u8>(request.output + fsa::layout::response_entry_name + length, 0) && ok;

    return ok ? IosError::Ok : IosError::Invalid;
}

IosError FsaDevice::close_dir(memory::Memory& memory, const IosIoctl& request) {
    u32 handle = 0;
    if (request.input == 0 ||
        !memory.try_read<u32>(request.input + fsa::layout::request_handle, handle)) {
        report_mismatch(memory, request, "no handle in the request");
        return IosError::Invalid;
    }
    return ios_error_for(filesystem_->close_directory(handle));
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

    switch (static_cast<fsa::Command>(request.command)) {
        case fsa::Command::OpenFile:
            return open_file(memory, request);
        case fsa::Command::ReadFile:
            return read_file(memory, request);
        case fsa::Command::CloseFile:
            return close_file(memory, request);
        case fsa::Command::GetPosFile:
            return get_pos_file(memory, request);
        case fsa::Command::SetPosFile:
            return set_pos_file(memory, request);
        case fsa::Command::IsEof:
            return is_eof(memory, request);
        case fsa::Command::OpenDir:
            return open_dir(memory, request);
        case fsa::Command::ReadDir:
            return read_dir(memory, request);
        case fsa::Command::CloseDir:
            return close_dir(memory, request);

        case fsa::Command::Mount:
        case fsa::Command::Unmount:
            // The Phase 5 VFS presents every mount from the start; there is
            // nothing to attach. Succeeding is honest here — the title asked for
            // a filesystem to be available, and it is.
            WIIU_LOG_DEBUG(Ios,
                           "/dev/fsa {}: the virtual filesystem is always mounted",
                           fsa::command_name(request.command));
            return IosError::Ok;

        case fsa::Command::FlushFile:
            // Nothing is written yet (6.9), so there is nothing buffered to push.
            return IosError::Ok;

        case fsa::Command::ChangeDir:
        case fsa::Command::GetCwd:
        case fsa::Command::MakeDir:
        case fsa::Command::Remove:
        case fsa::Command::Rename:
        case fsa::Command::RewindDir:
        case fsa::Command::WriteFile:
        case fsa::Command::GetStatFile:
            break;
    }

    // Not implemented, or a command this table does not know. Either way the
    // default says so with its number and sizes, which is what makes the next
    // gap visible.
    return IosDevice::ioctl(memory, request);
}

IosError FsaDevice::ioctlv(memory::Memory& memory, const IosIoctlv& request) {
    // FSA's vectored commands are the ones that move bulk data with the
    // buffer supplied separately. None are implemented, and inventing a
    // decode for them would be a specification rather than a device.
    return IosDevice::ioctlv(memory, request);
}

} // namespace wiiu::hle
