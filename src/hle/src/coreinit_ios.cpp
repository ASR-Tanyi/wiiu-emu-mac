// Phase 6.10 — coreinit's IOS_* entry points.
//
// The whole of the PowerPC side's conversation with IOSU comes through eleven
// functions. Each one builds an IPC request on hardware and waits for the ARM
// to answer; here each one reaches the device table in ios.hpp directly, so
// the request never becomes a packet and Starbuck never has to exist.
//
// The asynchronous forms are the interesting half. A title that calls
// `IOS_IoctlAsync` and then sets a flag is relying on its callback *not*
// running inside the call, so the work happens immediately — the devices are
// synchronous — and only the completion is scheduled, on the timebase,
// through the same timer list alarms and asynchronous FS completions use. A
// replay therefore delivers it at the same instruction (10.4).

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/ios.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace wiiu::hle::coreinit {

namespace {

[[nodiscard]] Context* active() noexcept {
    return detail::active_context();
}

/// Ids for the pending IPC completions, from the range coreinit.hpp reserves.
/// Held on the context rather than in a static, so two runs of the same title
/// in one process do not inherit each other's numbering.
[[nodiscard]] u32 next_ios_async_id(Context& context) {
    if (context.next_ios_async_id < ios_async_id_base ||
        context.next_ios_async_id >= ios_async_id_limit) {
        context.next_ios_async_id = ios_async_id_base;
    }
    return context.next_ios_async_id++;
}

/// Schedule `IOSAsyncCallbackFn(IOSError result, void *context)`.
///
/// A request with no callback has nowhere to deliver its answer. Hardware
/// would leave the caller waiting for a reply that never comes; saying so is
/// what makes that legible rather than a title that simply stops.
void complete_ios_async(Call& call, std::string name, s32 result, ppcaddr_t callback, u32 user) {
    Context& context = *active();
    context.memory = &call.memory;

    if (callback == 0) {
        WIIU_LOG_WARN(Ios, "{} was given no callback to deliver its result to", name);
        return;
    }

    const u32 id = next_ios_async_id(context);
    context.scheduler.set_timer(
        id,
        call.state.timebase + ios_async_latency_ticks,
        [&context, name = std::move(name), result, callback, user](u64) {
            if (context.memory == nullptr) {
                return;
            }
            const std::array<u32, 2> arguments{static_cast<u32>(result), user};
            static_cast<void>(
                invoke_guest_callback(context, *context.memory, name, callback, arguments));
        });
}

/// The device a handle names, or nothing — with the complaint already logged.
[[nodiscard]] const IosManager::OpenHandle* resolve(s32 handle, std::string_view what) {
    Context* const context = active();
    if (context == nullptr) {
        return nullptr;
    }
    const IosManager::OpenHandle* const open = context->ios.find(handle);
    if (open == nullptr) {
        WIIU_LOG_WARN(Ios, "{}: {} is not an open handle", what, handle);
    }
    return open;
}

// --- the synchronous forms -------------------------------------------------

[[nodiscard]] s32 do_open(Call& call, u32 path_address, u32 mode) {
    if (active() == nullptr) {
        return static_cast<s32>(IosError::NotReady);
    }
    const std::string path = call.read_string(path_address, 64);
    if (path.empty()) {
        WIIU_LOG_WARN(Ios, "IOS_Open was given no device name");
        return static_cast<s32>(IosError::Invalid);
    }
    return active()->ios.open(path, static_cast<IosOpenMode>(mode));
}

[[nodiscard]] s32 do_close(s32 handle) {
    if (active() == nullptr) {
        return static_cast<s32>(IosError::NotReady);
    }
    return static_cast<s32>(active()->ios.close(handle));
}

[[nodiscard]] s32 do_ioctl(
    Call& call, s32 handle, u32 command, u32 input, u32 input_size, u32 output, u32 output_size) {
    const IosManager::OpenHandle* const open = resolve(handle, "IOS_Ioctl");
    if (open == nullptr) {
        return static_cast<s32>(IosError::Invalid);
    }
    const IosIoctl request{.handle = handle,
                           .command = command,
                           .input = input,
                           .input_size = input_size,
                           .output = output,
                           .output_size = output_size};
    return static_cast<s32>(open->device->ioctl(call.memory, request));
}

[[nodiscard]] s32 do_ioctlv(
    Call& call, s32 handle, u32 command, u32 read_count, u32 write_count, u32 vectors_address) {
    const IosManager::OpenHandle* const open = resolve(handle, "IOS_Ioctlv");
    if (open == nullptr) {
        return static_cast<s32>(IosError::Invalid);
    }

    std::vector<IosVector> vectors;
    if (!read_ios_vectors(call.memory, vectors_address, read_count + write_count, vectors)) {
        WIIU_LOG_WARN(Ios,
                      "IOS_Ioctlv({:#x}) on handle {}: the vector array at {:#010x} could not be "
                      "read",
                      command,
                      handle,
                      vectors_address);
        return static_cast<s32>(IosError::Invalid);
    }

    const IosIoctlv request{.handle = handle,
                            .command = command,
                            .read_count = read_count,
                            .write_count = write_count,
                            .vectors = vectors};
    return static_cast<s32>(open->device->ioctlv(call.memory, request));
}

// --- the registered handlers ----------------------------------------------

s32 ios_open(Call& call, u32 path_address, u32 mode) {
    return do_open(call, path_address, mode);
}

s32 ios_open_async(Call& call, u32 path_address, u32 mode, u32 callback, u32 user) {
    const s32 result = do_open(call, path_address, mode);
    // For an open, the handle *is* the result the callback receives.
    complete_ios_async(call, "ios_open_async", result, callback, user);
    return static_cast<s32>(IosError::Ok);
}

s32 ios_close(Call&, s32 handle) {
    return do_close(handle);
}

s32 ios_close_async(Call& call, s32 handle, u32 callback, u32 user) {
    const s32 result = do_close(handle);
    complete_ios_async(call, "ios_close_async", result, callback, user);
    return static_cast<s32>(IosError::Ok);
}

s32 ios_read(Call& call, s32 handle, u32 buffer, u32 size) {
    const IosManager::OpenHandle* const open = resolve(handle, "IOS_Read");
    return open == nullptr ? static_cast<s32>(IosError::Invalid)
                           : open->device->read(call.memory, handle, buffer, size);
}

s32 ios_write(Call& call, s32 handle, u32 buffer, u32 size) {
    const IosManager::OpenHandle* const open = resolve(handle, "IOS_Write");
    return open == nullptr ? static_cast<s32>(IosError::Invalid)
                           : open->device->write(call.memory, handle, buffer, size);
}

s32 ios_seek(Call&, s32 handle, s32 offset, u32 origin) {
    const IosManager::OpenHandle* const open = resolve(handle, "IOS_Seek");
    return open == nullptr ? static_cast<s32>(IosError::Invalid)
                           : open->device->seek(handle, offset, static_cast<IosSeekOrigin>(origin));
}

s32 ios_ioctl(
    Call& call, s32 handle, u32 command, u32 input, u32 input_size, u32 output, u32 output_size) {
    return do_ioctl(call, handle, command, input, input_size, output, output_size);
}

s32 ios_ioctl_async(Call& call,
                    s32 handle,
                    u32 command,
                    u32 input,
                    u32 input_size,
                    u32 output,
                    u32 output_size,
                    u32 callback,
                    u32 user) {
    // Eight arguments exactly, so the whole call arrives in r3-r10 and none
    // of it is on the stack. Worth noticing rather than assuming: the
    // marshaller reads zero past r10, so a ninth argument would be silently
    // null instead of wrong in a way anyone would see.
    const s32 result = do_ioctl(call, handle, command, input, input_size, output, output_size);
    complete_ios_async(call, "ios_ioctl_async", result, callback, user);
    return static_cast<s32>(IosError::Ok);
}

s32 ios_ioctlv(
    Call& call, s32 handle, u32 command, u32 read_count, u32 write_count, u32 vectors_address) {
    return do_ioctlv(call, handle, command, read_count, write_count, vectors_address);
}

s32 ios_ioctlv_async(Call& call,
                     s32 handle,
                     u32 command,
                     u32 read_count,
                     u32 write_count,
                     u32 vectors_address,
                     u32 callback,
                     u32 user) {
    const s32 result = do_ioctlv(call, handle, command, read_count, write_count, vectors_address);
    complete_ios_async(call, "ios_ioctlv_async", result, callback, user);
    return static_cast<s32>(IosError::Ok);
}

} // namespace

void register_ios(Registry& registry, Context& context) {
    // /dev/fsa shares the context's filesystem rather than keeping one of its
    // own: a title that opens a file through the FS API and reads it through
    // the device — or the other way round — must see one set of handles.
    context.ios.add_device("/dev/fsa", std::make_shared<FsaDevice>(context.filesystem));

    registry.add("coreinit", "IOS_Open", ios_open);
    registry.add("coreinit", "IOS_OpenAsync", ios_open_async);
    registry.add("coreinit", "IOS_Close", ios_close);
    registry.add("coreinit", "IOS_CloseAsync", ios_close_async);
    registry.add("coreinit", "IOS_Read", ios_read);
    registry.add("coreinit", "IOS_Write", ios_write);
    registry.add("coreinit", "IOS_Seek", ios_seek);
    registry.add("coreinit", "IOS_Ioctl", ios_ioctl);
    registry.add("coreinit", "IOS_IoctlAsync", ios_ioctl_async);
    registry.add("coreinit", "IOS_Ioctlv", ios_ioctlv);
    registry.add("coreinit", "IOS_IoctlvAsync", ios_ioctlv_async);
}

} // namespace wiiu::hle::coreinit
