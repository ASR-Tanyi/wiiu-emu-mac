// Phase 6.10 — the IPC bridge to IOSU.
//
// Everything a Wii U title cannot do for itself goes through Starbuck: the
// filesystem, the account system, saves, the network stack, decryption. The
// PowerPC side asks for it over IPC — a request structure in the
// `0xE0000000` communication area and a doorbell through the PI registers —
// and an ARM926 inside Latte answers.
//
// None of that is emulated. The plan is explicit that LLE of the security
// processor is out of scope, so the interception happens one layer up, at the
// `IOS_*` functions coreinit exposes: `IOS_Open` names a device, `IOS_Ioctl`
// sends it a command, and a native handler answers. A title cannot tell the
// difference, because the request never had to become a packet.
//
// The raw doorbell path is deliberately *not* serviced. Only an LLE coreinit
// would take it, and Phase 3.5 leaves the IPC MMIO range unmapped, so a title
// that poked it would fault visibly rather than be quietly humoured.
//
// The device table follows the missing-function policy (6.12) one level down:
// opening a device nobody has written logs the path and fails, rather than
// handing back a handle that answers every command with silence. A device
// registered under a path no title opens is dead code that reads like a
// specification.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/hle/filesystem.hpp>
#include <wiiu/memory/memory.hpp>

#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wiiu::hle {

/// `IOSError`. Zero is success, a positive value from `IOS_Open` is a handle,
/// and everything negative is a failure.
///
/// Transcribed from public reverse-engineering of IOS; the first dozen are
/// inherited unchanged from the Wii and are the ones worth trusting. Only the
/// values this emulator actually returns are listed — a longer table would be
/// a specification nobody had checked.
enum class IosError : s32 {
    Ok = 0,
    Access = -1,
    Exists = -2,
    Interrupted = -3,
    Invalid = -4,
    Max = -5,
    NoExists = -6,
    QueueEmpty = -7,
    QueueFull = -8,
    Unknown = -9,
    NotReady = -10,
};

[[nodiscard]] std::string_view ios_error_name(IosError error) noexcept;

/// `IOSOpenMode`, the second argument to `IOS_Open`.
enum class IosOpenMode : u32 {
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3,
};

/// The IPC command ids, as they appear in a request header.
///
/// Nothing here builds a request header — that is the layer being skipped —
/// but the numbering is what a trace or a future LLE path would speak, and
/// naming it once is cheaper than recognising `6` later.
enum class IosCommand : u32 {
    Open = 1,
    Close = 2,
    Read = 3,
    Write = 4,
    Seek = 5,
    Ioctl = 6,
    Ioctlv = 7,
    Reply = 8,
};

/// Where `IOS_Seek` counts from.
enum class IosSeekOrigin : u32 {
    Set = 0,
    Current = 1,
    End = 2,
};

/// One `IOSVec`: a guest buffer and its length, plus a word the API reserves
/// and nothing sets. Twelve bytes, and an `IOS_Ioctlv` passes an array of
/// them.
struct IosVector {
    ppcaddr_t address = 0;
    u32 size = 0;
    u32 unused = 0;
};

inline constexpr u32 ios_vector_size = 12;

/// Read `count` vectors from guest memory. False if any of them could not be
/// read, in which case `out` is left as far as it got — a partially readable
/// vector array is a broken request, not half a request.
[[nodiscard]] bool read_ios_vectors(const memory::Memory& memory,
                                    ppcaddr_t address,
                                    u32 count,
                                    std::vector<IosVector>& out);

/// An `IOS_Ioctl`: one input buffer, one output buffer.
struct IosIoctl {
    s32 handle = 0;
    u32 command = 0;
    ppcaddr_t input = 0;
    u32 input_size = 0;
    ppcaddr_t output = 0;
    u32 output_size = 0;
};

/// An `IOS_Ioctlv`: a vector array, the first `read_count` of which the
/// device reads and the next `write_count` of which it writes.
struct IosIoctlv {
    s32 handle = 0;
    u32 command = 0;
    u32 read_count = 0;
    u32 write_count = 0;
    std::span<const IosVector> vectors;
};

/// A device behind `/dev/...`.
///
/// Every operation defaults to refusing loudly, so a device implements the
/// commands it knows about and the rest report themselves rather than
/// returning a plausible zero. The handle is passed to each call because one
/// device serves every title thread that opened it, and what a command means
/// often depends on which handle asked.
class IosDevice {
public:
    IosDevice() = default;
    virtual ~IosDevice() = default;

    IosDevice(const IosDevice&) = delete;
    IosDevice& operator=(const IosDevice&) = delete;

    [[nodiscard]] virtual std::string_view name() const = 0;

    /// Accept or refuse an open. The manager has already allocated `handle`;
    /// refusing releases it.
    [[nodiscard]] virtual IosError open(s32 handle, IosOpenMode mode);

    virtual void close(s32 handle);

    /// Bytes read, or a negative `IosError`.
    [[nodiscard]] virtual s32 read(memory::Memory& memory, s32 handle, ppcaddr_t buffer, u32 size);

    /// Bytes written, or a negative `IosError`.
    [[nodiscard]] virtual s32 write(memory::Memory& memory, s32 handle, ppcaddr_t buffer, u32 size);

    /// The new position, or a negative `IosError`.
    [[nodiscard]] virtual s32 seek(s32 handle, s32 offset, IosSeekOrigin origin);

    [[nodiscard]] virtual IosError ioctl(memory::Memory& memory, const IosIoctl& request);

    [[nodiscard]] virtual IosError ioctlv(memory::Memory& memory, const IosIoctlv& request);
};

/// The open devices and the handles into them.
class IosManager {
public:
    /// Handles count up from one. Hardware hands out zero as well; leaving it
    /// unused means an `IOSHandle` that was never assigned does not name a
    /// live device, which turns one class of title bug into an error instead
    /// of silent traffic to whatever opened first.
    static constexpr s32 first_handle = 1;

    /// A title that leaks handles should be told, not accommodated. Hardware
    /// has a per-process limit of its own; the number here only has to be far
    /// above what any title legitimately holds.
    static constexpr std::size_t max_open_handles = 256;

    struct OpenHandle {
        IosDevice* device = nullptr;
        std::string path;
        IosOpenMode mode = IosOpenMode::None;
    };

    /// Forget every open handle, closing each on its device. Registered
    /// devices survive: they belong to the emulator, not to the title, in the
    /// same way the HLE registry does.
    void reset();

    /// Drop the devices too. Only a fresh machine wants this.
    void clear();

    void add_device(std::string path, std::shared_ptr<IosDevice> device);

    [[nodiscard]] IosDevice* device_for(std::string_view path) const;

    /// A handle on success, or a negative `IosError`.
    [[nodiscard]] s32 open(std::string_view path, IosOpenMode mode);

    [[nodiscard]] IosError close(s32 handle);

    [[nodiscard]] const OpenHandle* find(s32 handle) const;

    [[nodiscard]] std::size_t open_count() const noexcept { return handles_.size(); }

    [[nodiscard]] std::size_t device_count() const noexcept { return devices_.size(); }

    [[nodiscard]] std::vector<std::string> device_paths() const;

private:
    std::map<std::string, std::shared_ptr<IosDevice>, std::less<>> devices_;
    std::map<s32, OpenHandle> handles_;
    s32 next_handle_ = first_handle;
};

// --------------------------------------------------------------------------
// /dev/fsa
// --------------------------------------------------------------------------
//
// The filesystem as IOSU presents it. coreinit's own `FS*` API is HLE'd
// directly (6.9) and never comes through here, so this path is for the title
// that talks to the device itself — which homebrew does, and which a few
// commercial titles do for `/vol/storage_mlc01`.

namespace fsa {

/// FSA command numbers.
///
/// Transcribed from public reverse-engineering and **not verified against
/// hardware or a real title**. They are here because the dispatch has to be
/// driven by something and a table in one place is correctable in one place —
/// not because this file knows them to be right.
enum class Command : u32 {
    Mount = 0x01,
    Unmount = 0x02,
    ChangeDir = 0x05,
    GetCwd = 0x06,
    MakeDir = 0x07,
    Remove = 0x08,
    Rename = 0x09,
    OpenDir = 0x0A,
    ReadDir = 0x0B,
    RewindDir = 0x0C,
    CloseDir = 0x0D,
    OpenFile = 0x0E,
    ReadFile = 0x0F,
    WriteFile = 0x10,
    GetPosFile = 0x11,
    SetPosFile = 0x12,
    IsEof = 0x13,
    GetStatFile = 0x14,
    CloseFile = 0x15,
    FlushFile = 0x16,
};

[[nodiscard]] std::string_view command_name(u32 command) noexcept;

/// Where the fields sit inside an FSA ioctl's request and response buffers.
///
/// ⚠ **These are this project's reading of the format, not hardware's.** The
/// FSA request and response layouts have not been checked against a real
/// title, and the numbers below are very likely wrong in detail.
///
/// They are written down anyway because the alternative is a device that
/// cannot be wired to anything: with them, the plumbing — handle allocation,
/// command dispatch, the VFS behind it, the asynchronous completion — is real
/// and testable, and only the offsets are in doubt. `FsaDevice` checks what it
/// reads rather than trusting it: a path that is not a plausible NUL-terminated
/// absolute path is reported as a layout mismatch, with the buffer, instead of
/// being passed to the filesystem as if it meant something. So the first real
/// FSA user turns this from a guess into a log line naming exactly what to fix.
///
/// Verify before trusting. This belongs in the plan's next-actions list
/// alongside the `fres`/`frsqrte` tables, for the same reason.
namespace layout {

inline constexpr u32 request_size = 0x520;
inline constexpr u32 response_size = 0x293;

/// Commands that name a path put it at the start of the request.
inline constexpr u32 request_path = 0x00;
inline constexpr u32 request_path_max = 0x280;

/// `OpenFile` follows its path with the mode string ("r", "rb", ...).
inline constexpr u32 request_open_mode = 0x280;
inline constexpr u32 request_open_mode_max = 0x10;

/// Commands that name something already open put the handle first instead.
inline constexpr u32 request_handle = 0x00;

/// `ReadFile`/`WriteFile`: the guest buffer, the item size and the count.
inline constexpr u32 request_buffer = 0x04;
inline constexpr u32 request_item_size = 0x08;
inline constexpr u32 request_item_count = 0x0C;
inline constexpr u32 request_read_handle = 0x10;

/// `SetPosFile`: the handle, then the position.
inline constexpr u32 request_position = 0x04;

/// The first word of a response is whatever the command produced: a handle
/// for the open commands, a position, a flag.
inline constexpr u32 response_value = 0x00;

/// `ReadDir` fills in a directory entry: the flags, the size, then the name.
inline constexpr u32 response_entry_flags = 0x00;
inline constexpr u32 response_entry_size = 0x04;
inline constexpr u32 response_entry_name = 0x08;
inline constexpr u32 response_entry_name_max = 0x100;

/// The directory-entry flag that marks a subdirectory.
inline constexpr u32 entry_flag_directory = 0x80000000;

} // namespace layout

} // namespace fsa

/// `/dev/fsa`, backed by the Phase 5 virtual filesystem through the same
/// `FileSystem` the `FS*` API uses — so a title that opens a file one way and
/// reads it the other sees one filesystem, with one set of open handles.
class FsaDevice final : public IosDevice {
public:
    explicit FsaDevice(FileSystem& filesystem) : filesystem_(&filesystem) {}

    [[nodiscard]] std::string_view name() const override { return "/dev/fsa"; }

    [[nodiscard]] IosError open(s32 handle, IosOpenMode mode) override;
    void close(s32 handle) override;

    [[nodiscard]] IosError ioctl(memory::Memory& memory, const IosIoctl& request) override;
    [[nodiscard]] IosError ioctlv(memory::Memory& memory, const IosIoctlv& request) override;

    /// How many ioctls arrived with a request buffer this file's layout could
    /// not make sense of. Nonzero means `fsa::layout` is wrong and the log
    /// says how.
    [[nodiscard]] u64 layout_mismatches() const noexcept { return layout_mismatches_; }

private:
    [[nodiscard]] IosError open_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError read_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError close_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError get_pos_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError set_pos_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError is_eof(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError open_dir(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError read_dir(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] IosError close_dir(memory::Memory& memory, const IosIoctl& request);

    /// Read the path a command names, checking it looks like one. Empty means
    /// the layout did not hold, which has already been logged.
    [[nodiscard]] std::string read_path(const memory::Memory& memory, const IosIoctl& request);

    void
    report_mismatch(const memory::Memory& memory, const IosIoctl& request, std::string_view why);

    FileSystem* filesystem_ = nullptr;
    u64 layout_mismatches_ = 0;
};

/// An `FsStatus` as an `IosError`. FSA speaks the same failures one layer
/// down, and mapping them here keeps the translation in one place.
[[nodiscard]] IosError ios_error_for(FsStatus status) noexcept;

} // namespace wiiu::hle
