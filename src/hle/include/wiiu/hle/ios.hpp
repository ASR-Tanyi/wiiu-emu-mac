// Phase 6.10 — the IPC bridge to IOSU.
//
// Everything a Wii U title cannot do for itself goes through Starbuck: the
// filesystem, the account system, saves, the network stack, decryption. The
// PowerPC side asks for it over IPC — a request structure handed over by
// physical address, and a doorbell through the PI registers — and an ARM926
// inside Latte answers.
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
/// From wut's `filesystem_fsa.h` (`FSACommand`), which homebrew uses against
/// real IOSU — so these are hardware's numbers, not a reading of them. decaf's
/// `ios_fs_enum.h` agrees value for value. Only the commands this device has
/// a reason to name are listed; anything else still reaches the default
/// refusal with its number.
enum class Command : u32 {
    Mount = 0x01,
    Unmount = 0x02,
    GetVolumeInfo = 0x03,
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
    ReadFile = 0x0F,  ///< ioctlv: request in, data and response out
    WriteFile = 0x10, ///< ioctlv: request and data in, response out
    GetPosFile = 0x11,
    SetPosFile = 0x12,
    IsEof = 0x13,
    StatFile = 0x14,
    CloseFile = 0x15,
    GetError = 0x16,
    FlushFile = 0x17,
    GetInfoByQuery = 0x18,
    AppendFile = 0x19,
    TruncateFile = 0x1A,
};

[[nodiscard]] std::string_view command_name(u32 command) noexcept;

/// What an FSA command replies with.
///
/// FSA does not answer in `IosError`s: the IPC reply word carries an FSA
/// status (negative), or for `ReadFile`/`WriteFile` the number of bytes moved
/// (positive), and coreinit's FS layer translates. Values from decaf's
/// `FSAStatus`, which wut's `FSError` matches.
enum class Status : s32 {
    Ok = 0,
    NotInit = -0x30001,
    Busy = -0x30002,
    Cancelled = -0x30003,
    EndOfDir = -0x30004,
    EndOfFile = -0x30005,
    MaxMountpoints = -0x30010,
    MaxVolumes = -0x30011,
    MaxClients = -0x30012,
    MaxFiles = -0x30013,
    MaxDirs = -0x30014,
    AlreadyOpen = -0x30015,
    AlreadyExists = -0x30016,
    NotFound = -0x30017,
    NotEmpty = -0x30018,
    AccessError = -0x30019,
    PermissionError = -0x3001A,
    DataCorrupted = -0x3001B,
    StorageFull = -0x3001C,
    JournalFull = -0x3001D,
    UnavailableCommand = -0x3001F,
    UnsupportedCommand = -0x30020,
    InvalidParam = -0x30021,
    InvalidPath = -0x30022,
    InvalidBuffer = -0x30023,
    InvalidAlignment = -0x30024,
    InvalidClientHandle = -0x30025,
    InvalidFileHandle = -0x30026,
    InvalidDirHandle = -0x30027,
    NotFile = -0x30028,
    NotDir = -0x30029,
    FileTooBig = -0x3002A,
    OutOfRange = -0x3002B,
    OutOfResources = -0x3002C,
    MediaNotReady = -0x30040,
    MediaError = -0x30041,
    WriteProtected = -0x30042,
    InvalidMedia = -0x30043,
};

[[nodiscard]] std::string_view status_name(Status status) noexcept;

/// An FSA status as the IPC reply word it travels in.
[[nodiscard]] constexpr IosError as_reply(Status status) noexcept {
    return static_cast<IosError>(static_cast<s32>(status));
}

/// Where the fields sit inside an FSA request and response.
///
/// From wut's `WUT_CHECK_OFFSET`/`WUT_CHECK_SIZE` lines for `FSARequest` and
/// `FSAResponse`, which are compile-time checks against structures homebrew
/// sends to real IOSU — decaf's `CHECK_OFFSET`s agree. Both structures begin
/// with one word of their own (`emulatedError` in the request, `word0` in the
/// response) and put the per-command union after it, so every field is four
/// bytes further in than the union's own offsets say. The constants here are
/// absolute, with that word already counted.
///
/// The first reading of this format (6.10) got the leading word wrong, read
/// the file handle of `ReadFile` from where its position is, put a directory
/// entry's name eight bytes in instead of after a full `FSStat`, and expected
/// `ReadFile` as an ioctl when it is an ioctlv. `FsaDevice` still checks what
/// it reads rather than trusting it, so a request that does not hold a
/// plausible path is reported as a layout mismatch with the buffer beside it.
namespace layout {

inline constexpr u32 request_size = 0x520;
inline constexpr u32 response_size = 0x293;

/// A request whose first word is negative is answered with that status
/// without being carried out — coreinit's hook for injecting errors.
inline constexpr u32 request_emulated_error = 0x000;

/// Commands that name a path put it at the start of the union.
inline constexpr u32 request_path = 0x004;
inline constexpr u32 request_path_max = 0x280;

/// `OpenFile` follows its path with the mode string ("r", "rb", ...).
inline constexpr u32 request_open_mode = 0x284;
inline constexpr u32 request_open_mode_max = 0x10;

/// Commands that name something already open put the handle first instead.
inline constexpr u32 request_handle = 0x004;

/// `SetPosFile`: the handle, then the position.
inline constexpr u32 request_position = 0x008;

/// `ReadFile`/`WriteFile`. The buffer word is coreinit's own virtual pointer;
/// IOSU moves the data through the ioctlv's second vector instead, which is
/// what this device reads.
inline constexpr u32 request_rw_buffer = 0x004;
inline constexpr u32 request_rw_size = 0x008;
inline constexpr u32 request_rw_count = 0x00C;
inline constexpr u32 request_rw_position = 0x010;
inline constexpr u32 request_rw_handle = 0x014;
inline constexpr u32 request_rw_flags = 0x018;

/// `ReadFile`/`WriteFile` flag: seek to `request_rw_position` first.
inline constexpr u32 rw_flag_with_position = 0x1;

/// `GetInfoByQuery`: the path, then which question is being asked.
inline constexpr u32 request_query_type = 0x284;
inline constexpr u32 query_type_stat = 5;

/// The ioctlv vectors of `ReadFile` and `WriteFile`.
inline constexpr u32 rw_vector_request = 0;
inline constexpr u32 rw_vector_data = 1;
inline constexpr u32 rw_vector_response = 2;

/// The first word of the response union: a handle for the open commands, a
/// position for `GetPosFile`.
inline constexpr u32 response_value = 0x004;

/// `StatFile` and `GetInfoByQuery(Stat)` fill in an `FSStat` here.
inline constexpr u32 response_stat = 0x004;

/// `ReadDir` fills in an `FSDirectoryEntry`: a whole `FSStat`, then the name.
inline constexpr u32 response_entry = 0x004;

/// `FSStat`, relative to its own start (wut's `filesystem.h`).
inline constexpr u32 stat_flags = 0x00;
inline constexpr u32 stat_mode = 0x04;
inline constexpr u32 stat_owner = 0x08;
inline constexpr u32 stat_group = 0x0C;
inline constexpr u32 stat_size = 0x10;
inline constexpr u32 stat_alloc_size = 0x14;
inline constexpr u32 stat_entry_id = 0x20;
inline constexpr u32 stat_bytes = 0x64;

inline constexpr u32 response_entry_flags = response_entry + stat_flags;
inline constexpr u32 response_entry_size = response_entry + stat_size;
inline constexpr u32 response_entry_name = response_entry + stat_bytes;
inline constexpr u32 response_entry_name_max = 0x100;

/// `FSStat` flags.
inline constexpr u32 stat_flag_directory = 0x80000000;
inline constexpr u32 stat_flag_file = 0x01000000;

/// The mode bits reported for everything: read and write for owner and
/// group, as decaf reports. Content is read-only in practice, and the write
/// path refuses on its own (6.9) rather than through these bits.
inline constexpr u32 stat_default_mode = 0x660;

static_assert(response_entry_name + response_entry_name_max <= response_size,
              "an FSDirectoryEntry has to fit in an FSAResponse");

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

    /// How many requests arrived that this file's layout could not make
    /// sense of. Nonzero means `fsa::layout` is wrong and the log says how.
    [[nodiscard]] u64 layout_mismatches() const noexcept { return layout_mismatches_; }

    /// Fill in an `FSStat` at `address`. coreinit's `FSGetStatFile` writes
    /// the same structure.
    [[nodiscard]] static bool
    write_stat(memory::Memory& memory, ppcaddr_t address, bool directory, u64 size);

private:
    using Status = fsa::Status;

    [[nodiscard]] Status open_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status close_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status get_pos_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status set_pos_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status is_eof(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status stat_file(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status get_info_by_query(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status open_dir(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status read_dir(memory::Memory& memory, const IosIoctl& request);
    [[nodiscard]] Status close_dir(memory::Memory& memory, const IosIoctl& request);

    /// `ReadFile`'s reply is a byte count on success, so it does not fit the
    /// `Status`-returning shape of the others.
    [[nodiscard]] IosError read_file(memory::Memory& memory, const IosIoctlv& request);

    /// Read the path a command names, checking it looks like one. Empty means
    /// the layout did not hold, which has already been logged.
    [[nodiscard]] std::string read_path(const memory::Memory& memory, const IosIoctl& request);

    /// Read the handle a command names. False means there was no request
    /// buffer to read it from, which has already been logged.
    [[nodiscard]] bool
    read_handle(const memory::Memory& memory, const IosIoctl& request, u32& handle);

    void report_mismatch(const memory::Memory& memory,
                         u32 command,
                         ppcaddr_t buffer,
                         u32 size,
                         std::string_view why);

    FileSystem* filesystem_ = nullptr;
    u64 layout_mismatches_ = 0;
};

/// An `FsStatus` as the FSA status IOSU would have replied with. The FS API
/// and FSA speak the same failures one layer apart; `end` is what running out
/// means for the command at hand, a file or a directory.
[[nodiscard]] fsa::Status fsa_status_for(FsStatus status,
                                         fsa::Status end = fsa::Status::EndOfFile) noexcept;

} // namespace wiiu::hle
