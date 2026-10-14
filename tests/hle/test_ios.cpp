// Phase 6.10 — the IPC bridge to IOSU.
//
// Two things are being guarded here, and they fail differently. The device
// table is the easy half: a handle names a device, a closed handle names
// nothing, and a device nobody wrote is refused rather than quietly handed a
// handle that answers everything with silence.
//
// The asynchronous half is the one worth the machinery. These tests run the
// whole path -- HLE call, timer, scheduler, interpreter -- because the bug
// being guarded against is a completion that is scheduled beautifully and
// never actually runs, which looks exactly like a title hanging for no reason.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/ios.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/loader/vfs.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;

namespace {

constexpr ppcaddr_t kMainProgram = 0x02000000;
constexpr ppcaddr_t kCallbackProgram = 0x02001000;
constexpr ppcaddr_t kScratch = 0x10002000;

constexpr ppcaddr_t kDevicePath = 0x10004000;
constexpr ppcaddr_t kRequest = 0x10005000;
constexpr ppcaddr_t kResponse = 0x10006000;
constexpr ppcaddr_t kBuffer = 0x10007000;
constexpr ppcaddr_t kVectors = 0x10008000;

constexpr u32 kEnc_mflr_r31 = 0x7FE802A6;
constexpr u32 kEnc_mtctr_r12 = 0x7D8903A6;
constexpr u32 kEnc_bctrl = 0x4E800421;
constexpr u32 kEnc_mtlr_r31 = 0x7FE803A6;
constexpr u32 kEnc_blr = 0x4E800020;

[[nodiscard]] constexpr u32 lis_r12(u32 value) {
    return 0x3D800000U | (value >> 16);
}

[[nodiscard]] constexpr u32 ori_r12(u32 value) {
    return 0x618C0000U | (value & 0xFFFF);
}

[[nodiscard]] constexpr u32 li(u32 reg, u32 value) {
    return 0x38000000U | (reg << 21) | (value & 0xFFFF);
}

/// The file the test title ships, and what is in it.
constexpr std::string_view kContentPath = "/vol/content/data.bin";
constexpr std::string_view kContent = "content payload\n";

struct IosFixture {
    memory::Memory memory;
    cpu::CoreState state;
    cpu::Interpreter interpreter{state, memory};
    Registry registry;
    coreinit::Context os;
    loader::VirtualFileSystem vfs;

    IosFixture() {
        std::vector<memory::Region> regions(memory::default_regions().begin(),
                                            memory::default_regions().end());
        regions.insert(regions.begin() + 1,
                       memory::Region{"hle",
                                      Registry::trampoline_base,
                                      Registry::trampoline_size,
                                      0,
                                      memory::Backing::None,
                                      memory::read_write_execute});

        std::string error;
        REQUIRE(memory.initialize(regions, error));
        wiiu::log::set_all_levels(std::getenv("WIIU_TEST_LOG") != nullptr ? wiiu::log::Level::Trace
                                                                          : wiiu::log::Level::Off);

        coreinit::register_all(registry, os);
        REQUIRE(os.heaps.create_default_heaps(memory, error));
        REQUIRE(registry.install(memory, error));

        REQUIRE(vfs.mount_title_directory(
            std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title", error));
        os.filesystem.set_filesystem(&vfs);
        os.memory = &memory;

        write_callback_program();

        state.pc = kMainProgram;
        state.gpr[1] = memory::data_heap_base + 0x20000;
        state.lr = return_sentinel;
        static_cast<void>(os.scheduler.adopt_main(state, main_thread_handle));
    }

    /// Counts its calls at kScratch and records its first two arguments after
    /// it, so a test can tell not just that it ran but what it was handed.
    void write_callback_program() {
        const std::array<u32, 8> program{
            0x3CA01000, // lis   r5,0x1000
            0x60A52000, // ori   r5,r5,0x2000
            0x80C50000, // lwz   r6,0(r5)
            0x38C60001, // addi  r6,r6,1
            0x90C50000, // stw   r6,0(r5)
            0x90650004, // stw   r3,4(r5)
            0x90850008, // stw   r4,8(r5)
            kEnc_blr,
        };
        ppcaddr_t address = kCallbackProgram;
        for (const u32 word : program) {
            REQUIRE(memory.try_write<u32>(address, word));
            address += 4;
        }
        REQUIRE(memory.fill(kScratch, 16, 0));
    }

    void write_main_calling(std::string_view function, u32 argument) {
        const ppcaddr_t trampoline = registry.resolve("coreinit", function);
        const std::array<u32, 9> program{
            kEnc_mflr_r31,
            lis_r12(trampoline),
            ori_r12(trampoline),
            kEnc_mtctr_r12,
            li(3, argument),
            kEnc_bctrl,
            kEnc_mtlr_r31,
            li(3, 0),
            kEnc_blr,
        };
        ppcaddr_t address = kMainProgram;
        for (const u32 word : program) {
            REQUIRE(memory.try_write<u32>(address, word));
            address += 4;
        }

        std::string error;
        REQUIRE(registry.install(memory, error));
    }

    [[nodiscard]] u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("coreinit", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size() && i < 8; ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    /// The IOS entry points all answer with a signed value: a handle, zero,
    /// or an IosError.
    [[nodiscard]] s32 icall(std::string_view name, std::vector<u32> arguments) {
        return static_cast<s32>(call(name, std::move(arguments)));
    }

    RunReport run_it(u64 budget = 200'000) {
        return run(interpreter,
                   memory,
                   registry,
                   {.scheduler = &os.scheduler, .budget = budget, .timeslice = 200});
    }

    ppcaddr_t put_string(std::string_view text, ppcaddr_t at) {
        const std::vector<u8> bytes(text.begin(), text.end());
        REQUIRE(memory.write_bytes(at, bytes));
        REQUIRE(memory.try_write<u8>(at + static_cast<u32>(bytes.size()), 0));
        return at;
    }

    /// Open /dev/fsa and return the handle, failing the test if it could not
    /// be opened -- every FSA test starts this way.
    [[nodiscard]] s32 open_fsa() {
        static_cast<void>(put_string("/dev/fsa", kDevicePath));
        const s32 handle =
            icall("IOS_Open", {kDevicePath, static_cast<u32>(IosOpenMode::ReadWrite)});
        REQUIRE(handle >= IosManager::first_handle);
        return handle;
    }

    /// An empty FSA request buffer.
    void clear_request() {
        REQUIRE(memory.fill(kRequest, fsa::layout::request_size, 0));
        REQUIRE(memory.fill(kResponse, fsa::layout::response_size, 0));
    }

    [[nodiscard]] s32 fsa_ioctl(s32 handle, fsa::Command command) {
        return icall("IOS_Ioctl",
                     {static_cast<u32>(handle),
                      static_cast<u32>(command),
                      kRequest,
                      fsa::layout::request_size,
                      kResponse,
                      fsa::layout::response_size});
    }

    /// What an FSA command replies with, as the signed word `icall` returns.
    [[nodiscard]] static s32 reply(fsa::Status status) {
        return static_cast<s32>(fsa::as_reply(status));
    }

    /// One IOSVec: wut's `{vaddr, len, paddr}`. The physical address is left
    /// zero, as a title leaves it -- coreinit fills it in on hardware.
    void put_vector(u32 index, ppcaddr_t address, u32 size) {
        const ppcaddr_t base = kVectors + (index * ios_vector_size);
        REQUIRE(memory.try_write<u32>(base, address));
        REQUIRE(memory.try_write<u32>(base + 4, size));
        REQUIRE(memory.try_write<u32>(base + 8, 0));
    }

    /// FSA ReadFile as coreinit sends it: an ioctlv with the request in, and
    /// the data and response out.
    [[nodiscard]] s32 fsa_read(s32 device,
                               u32 file,
                               u32 size,
                               u32 count,
                               u32 room,
                               std::optional<u32> position = std::nullopt) {
        clear_request();
        REQUIRE(memory.fill(kBuffer, room, 0));
        // coreinit's own pointer to the data; IOSU ignores it in favour of
        // the vector, and so does the device.
        REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_rw_buffer, 0xDEADBEEF));
        REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_rw_size, size));
        REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_rw_count, count));
        REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_rw_handle, file));
        if (position.has_value()) {
            REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_rw_position, *position));
            REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_rw_flags,
                                          fsa::layout::rw_flag_with_position));
        }
        put_vector(fsa::layout::rw_vector_request, kRequest, fsa::layout::request_size);
        put_vector(fsa::layout::rw_vector_data, kBuffer, room);
        put_vector(fsa::layout::rw_vector_response, kResponse, fsa::layout::response_size);
        return icall(
            "IOS_Ioctlv",
            {static_cast<u32>(device), static_cast<u32>(fsa::Command::ReadFile), 1, 2, kVectors});
    }

    /// FSA OpenFile, returning the file handle it produced.
    [[nodiscard]] u32 fsa_open(s32 device, std::string_view path, std::string_view mode = "r") {
        clear_request();
        static_cast<void>(put_string(path, kRequest + fsa::layout::request_path));
        static_cast<void>(put_string(mode, kRequest + fsa::layout::request_open_mode));
        REQUIRE(fsa_ioctl(device, fsa::Command::OpenFile) == static_cast<s32>(IosError::Ok));
        return memory.read<u32>(kResponse + fsa::layout::response_value);
    }

    [[nodiscard]] FsaDevice& fsa_device() {
        IosDevice* const device = os.ios.device_for("/dev/fsa");
        REQUIRE(device != nullptr);
        return *static_cast<FsaDevice*>(device);
    }

    [[nodiscard]] u32 callback_count() { return memory.read<u32>(kScratch); }

    [[nodiscard]] u32 callback_argument(u32 index) {
        return memory.read<u32>(kScratch + 4 + (index * 4));
    }
};

} // namespace

// ---------------------------------------------------------------------------
// The device table
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(IosFixture, "a device nobody wrote is refused, not stubbed", "[hle][ios]") {
    static_cast<void>(put_string("/dev/something_nobody_implemented", kDevicePath));

    // The 6.12 policy one level down. A handle that answered every command
    // with zero would send the title on with a buffer nobody filled in, which
    // surfaces much later and somewhere else.
    CHECK(icall("IOS_Open", {kDevicePath, 1}) == static_cast<s32>(IosError::NoExists));
    CHECK(os.ios.open_count() == 0);
}

TEST_CASE_METHOD(IosFixture, "IOS_Open hands out distinct handles", "[hle][ios]") {
    const s32 first = open_fsa();
    const s32 second = icall("IOS_Open", {kDevicePath, 1});

    CHECK(first >= IosManager::first_handle);
    CHECK(second != first);
    CHECK(os.ios.open_count() == 2);

    CHECK(icall("IOS_Close", {static_cast<u32>(first)}) == static_cast<s32>(IosError::Ok));
    CHECK(os.ios.open_count() == 1);

    // Closing it twice is a title bug, and saying so is more useful than
    // succeeding: the second close would otherwise be closing whatever had
    // been given that number since.
    CHECK(icall("IOS_Close", {static_cast<u32>(first)}) == static_cast<s32>(IosError::Invalid));
}

TEST_CASE_METHOD(IosFixture, "a handle is not reused after it is closed", "[hle][ios]") {
    const s32 first = open_fsa();
    CHECK(icall("IOS_Close", {static_cast<u32>(first)}) == static_cast<s32>(IosError::Ok));

    const s32 second = icall("IOS_Open", {kDevicePath, 1});
    CHECK(second != first);
}

TEST_CASE_METHOD(IosFixture, "commands on a stale handle are refused", "[hle][ios]") {
    const s32 handle = open_fsa();
    CHECK(icall("IOS_Close", {static_cast<u32>(handle)}) == static_cast<s32>(IosError::Ok));

    clear_request();
    CHECK(fsa_ioctl(handle, fsa::Command::OpenFile) == static_cast<s32>(IosError::Invalid));
    CHECK(icall("IOS_Read", {static_cast<u32>(handle), kBuffer, 16}) ==
          static_cast<s32>(IosError::Invalid));
    CHECK(icall("IOS_Seek", {static_cast<u32>(handle), 0, 0}) ==
          static_cast<s32>(IosError::Invalid));
}

TEST_CASE_METHOD(IosFixture, "an unimplemented command reports itself", "[hle][ios]") {
    const s32 handle = open_fsa();
    clear_request();

    // Rename is in the command table but not implemented. The default says so
    // with its number and its buffer sizes rather than returning zero.
    CHECK(fsa_ioctl(handle, fsa::Command::Rename) == static_cast<s32>(IosError::Invalid));

    // And so does a command the table has never heard of.
    CHECK(icall("IOS_Ioctl", {static_cast<u32>(handle), 0xDEAD, kRequest, 16, kResponse, 16}) ==
          static_cast<s32>(IosError::Invalid));
}

TEST_CASE_METHOD(IosFixture, "the device table survives a title being torn down", "[hle][ios]") {
    static_cast<void>(open_fsa());
    REQUIRE(os.ios.open_count() == 1);
    const std::size_t devices = os.ios.device_count();

    os.reset();

    // Handles belong to the title; devices belong to the emulator, the same
    // way registered HLE functions do.
    CHECK(os.ios.open_count() == 0);
    CHECK(os.ios.device_count() == devices);
    CHECK(os.ios.device_for("/dev/fsa") != nullptr);
}

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(IosFixture, "an IOSVec array decodes", "[hle][ios]") {
    REQUIRE(memory.try_write<u32>(kVectors + 0, kBuffer));
    REQUIRE(memory.try_write<u32>(kVectors + 4, 16));
    REQUIRE(memory.try_write<u32>(kVectors + 8, 0));
    REQUIRE(memory.try_write<u32>(kVectors + 12, kResponse));
    REQUIRE(memory.try_write<u32>(kVectors + 16, 32));
    REQUIRE(memory.try_write<u32>(kVectors + 20, 0));

    std::vector<IosVector> vectors;
    REQUIRE(read_ios_vectors(memory, kVectors, 2, vectors));
    REQUIRE(vectors.size() == 2);
    CHECK(vectors[0].address == kBuffer);
    CHECK(vectors[0].size == 16U);
    CHECK(vectors[1].address == kResponse);
    CHECK(vectors[1].size == 32U);

    // A request with no vector array at all is broken, not empty.
    CHECK_FALSE(read_ios_vectors(memory, 0, 1, vectors));
    CHECK(read_ios_vectors(memory, 0, 0, vectors));
}

TEST_CASE_METHOD(IosFixture, "an unimplemented ioctlv reports itself", "[hle][ios]") {
    const s32 handle = open_fsa();
    REQUIRE(memory.fill(kVectors, 3 * ios_vector_size, 0));

    CHECK(icall("IOS_Ioctlv", {static_cast<u32>(handle), 0x20, 1, 1, kVectors}) ==
          static_cast<s32>(IosError::Invalid));
}

// ---------------------------------------------------------------------------
// /dev/fsa over the Phase 5 filesystem
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(IosFixture, "a file opens, reads and closes through /dev/fsa", "[hle][ios][fs]") {
    const s32 device = open_fsa();
    const u32 file = fsa_open(device, kContentPath);
    REQUIRE(file != 0U);
    CHECK(os.filesystem.open_file_count() == 1);

    // The reply to a read is the number of bytes moved, not a status.
    CHECK(fsa_read(device, file, 1, 32, 32) == static_cast<s32>(kContent.size()));
    CHECK(fsa_device().layout_mismatches() == 0U);

    std::vector<u8> read(kContent.size());
    REQUIRE(memory.read_bytes(kBuffer, read));
    CHECK(std::string(read.begin(), read.end()) == kContent);

    // The position moved with the read.
    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, file));
    REQUIRE(fsa_ioctl(device, fsa::Command::GetPosFile) == static_cast<s32>(IosError::Ok));
    CHECK(memory.read<u32>(kResponse + fsa::layout::response_value) == kContent.size());

    // ... and it is at the end, which IsEof answers with its status.
    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, file));
    CHECK(fsa_ioctl(device, fsa::Command::IsEof) == reply(fsa::Status::EndOfFile));

    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, file));
    CHECK(fsa_ioctl(device, fsa::Command::CloseFile) == static_cast<s32>(IosError::Ok));
    CHECK(os.filesystem.open_file_count() == 0);
}

TEST_CASE_METHOD(IosFixture, "seeking through /dev/fsa moves the file", "[hle][ios][fs]") {
    const s32 device = open_fsa();
    const u32 file = fsa_open(device, kContentPath);

    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, file));
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_position, 8));
    REQUIRE(fsa_ioctl(device, fsa::Command::SetPosFile) == static_cast<s32>(IosError::Ok));

    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, file));
    REQUIRE(fsa_ioctl(device, fsa::Command::GetPosFile) == static_cast<s32>(IosError::Ok));
    CHECK(memory.read<u32>(kResponse + fsa::layout::response_value) == 8U);

    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, file));
    CHECK(fsa_ioctl(device, fsa::Command::IsEof) == reply(fsa::Status::Ok));
}

TEST_CASE_METHOD(IosFixture, "a missing file fails as a missing file", "[hle][ios][fs]") {
    const s32 device = open_fsa();

    clear_request();
    static_cast<void>(
        put_string("/vol/content/not_here.bin", kRequest + fsa::layout::request_path));
    static_cast<void>(put_string("r", kRequest + fsa::layout::request_open_mode));

    // NotFound, and no layout complaint: the request was understood, the
    // file was not there. Telling those two apart is the whole point of the
    // check.
    CHECK(fsa_ioctl(device, fsa::Command::OpenFile) == reply(fsa::Status::NotFound));
    CHECK(fsa_device().layout_mismatches() == 0U);
}

TEST_CASE_METHOD(IosFixture, "a directory walks through /dev/fsa", "[hle][ios][fs]") {
    const s32 device = open_fsa();

    clear_request();
    static_cast<void>(put_string("/vol/content", kRequest + fsa::layout::request_path));
    REQUIRE(fsa_ioctl(device, fsa::Command::OpenDir) == static_cast<s32>(IosError::Ok));
    const u32 directory = memory.read<u32>(kResponse + fsa::layout::response_value);
    REQUIRE(directory != 0U);

    std::vector<std::string> names;
    for (int index = 0; index < 2; ++index) {
        clear_request();
        REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, directory));
        REQUIRE(fsa_ioctl(device, fsa::Command::ReadDir) == static_cast<s32>(IosError::Ok));

        std::vector<u8> name(16);
        REQUIRE(memory.read_bytes(kResponse + fsa::layout::response_entry_name, name));
        names.emplace_back(reinterpret_cast<const char*>(name.data()));

        // The entry is a whole FSStat followed by the name, so the flags say
        // which of the two this is and a file carries its size.
        const u32 flags = memory.read<u32>(kResponse + fsa::layout::response_entry_flags);
        if (names.back() == "sub") {
            CHECK(flags == fsa::layout::stat_flag_directory);
        } else {
            CHECK(flags == fsa::layout::stat_flag_file);
            CHECK(memory.read<u32>(kResponse + fsa::layout::response_entry_size) ==
                  kContent.size());
        }
    }

    // And then there are no more, which is a status rather than an entry.
    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, directory));
    CHECK(fsa_ioctl(device, fsa::Command::ReadDir) == reply(fsa::Status::EndOfDir));

    // Both entries turned up, and the subdirectory is marked as one. The
    // order is the filesystem's, so it is not asserted.
    CHECK(names.size() == 2);
    CHECK((names[0] == "data.bin" || names[1] == "data.bin"));
    CHECK((names[0] == "sub" || names[1] == "sub"));

    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, directory));
    CHECK(fsa_ioctl(device, fsa::Command::CloseDir) == static_cast<s32>(IosError::Ok));
}

TEST_CASE_METHOD(IosFixture,
                 "a directory entry never runs past the response buffer",
                 "[hle][ios][fs]") {
    const s32 device = open_fsa();

    clear_request();
    static_cast<void>(put_string("/vol/content", kRequest + fsa::layout::request_path));
    REQUIRE(fsa_ioctl(device, fsa::Command::OpenDir) == static_cast<s32>(IosError::Ok));
    const u32 directory = memory.read<u32>(kResponse + fsa::layout::response_value);

    // Only the request is rebuilt between calls: the response buffer carries
    // a filler this test checks was not written over.
    const auto read_dir_into = [&](u32 output_size) {
        REQUIRE(memory.fill(kRequest, fsa::layout::request_size, 0));
        REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, directory));
        return icall("IOS_Ioctl",
                     {static_cast<u32>(device),
                      static_cast<u32>(fsa::Command::ReadDir),
                      kRequest,
                      fsa::layout::request_size,
                      kResponse,
                      output_size});
    };

    // A response that ends exactly where the name would start has no room for
    // even a terminator. The arithmetic for "how much room is left" is
    // unsigned, so getting this wrong writes a whole filename past the end
    // rather than nothing.
    CHECK(read_dir_into(fsa::layout::response_entry_name) == reply(fsa::Status::InvalidBuffer));

    // One byte further, and the name is truncated to fit -- terminator
    // included -- rather than refused.
    REQUIRE(memory.fill(kResponse, fsa::layout::response_size, 0xAA));
    CHECK(read_dir_into(fsa::layout::response_entry_name + 4) == static_cast<s32>(IosError::Ok));
    CHECK(memory.read<u8>(kResponse + fsa::layout::response_entry_name + 3) == 0U);
    CHECK(memory.read<u8>(kResponse + fsa::layout::response_entry_name + 4) == 0xAAU);
}

TEST_CASE_METHOD(IosFixture,
                 "the device and the FS API share one set of open files",
                 "[hle][ios][fs]") {
    const s32 device = open_fsa();
    const u32 file = fsa_open(device, kContentPath);

    // Opened through IPC, read through coreinit. A title that mixes the two
    // is not doing anything unusual -- on hardware they are the same driver,
    // and here they are the same `FileSystem`.
    REQUIRE(memory.fill(kBuffer, 32, 0));
    const u32 items = call("FSReadFile", {0, 0, kBuffer, 1, 32, file, 0});
    CHECK(items == kContent.size());

    std::vector<u8> read(kContent.size());
    REQUIRE(memory.read_bytes(kBuffer, read));
    CHECK(std::string(read.begin(), read.end()) == kContent);
}

TEST_CASE_METHOD(IosFixture,
                 "a request the layout cannot explain is reported, not guessed at",
                 "[hle][ios][fs]") {
    const s32 device = open_fsa();
    clear_request(); // all zeroes: no path anywhere in it

    CHECK(fsa_ioctl(device, fsa::Command::OpenFile) == reply(fsa::Status::InvalidPath));
    CHECK(fsa_device().layout_mismatches() == 1U);

    // Nothing was opened on the strength of a buffer that made no sense.
    CHECK(os.filesystem.open_file_count() == 0);
}

TEST_CASE_METHOD(IosFixture, "a positioned read starts where it says", "[hle][ios][fs]") {
    const s32 device = open_fsa();
    const u32 file = fsa_open(device, kContentPath);

    // Two-byte items from offset 3: the read seeks first, and only whole
    // items count.
    const s32 moved = fsa_read(device, file, 2, 4, 16, 3U);
    CHECK(moved == 8);

    std::vector<u8> read(8);
    REQUIRE(memory.read_bytes(kBuffer, read));
    CHECK(std::string(read.begin(), read.end()) == kContent.substr(3, 8));
}

TEST_CASE_METHOD(IosFixture, "a read never writes past its data vector", "[hle][ios][fs]") {
    const s32 device = open_fsa();
    const u32 file = fsa_open(device, kContentPath);

    // The request asks for 32 bytes, the vector has room for 4. IOSU writes
    // through the vector, so the vector is the limit.
    REQUIRE(memory.fill(kBuffer, 32, 0xAA));
    CHECK(fsa_read(device, file, 1, 32, 4) == 4);
    CHECK(memory.read<u8>(kBuffer + 3) == static_cast<u8>(kContent[3]));
    CHECK(memory.read<u8>(kBuffer + 4) == 0xAAU);
}

TEST_CASE_METHOD(IosFixture, "ReadFile sent as a plain ioctl is reported", "[hle][ios][fs]") {
    const s32 device = open_fsa();
    clear_request();

    // IOSU only takes bulk data through a vector. A plain ioctl carrying a
    // read means whoever built it read the format differently, which is
    // worth a line in the log rather than a guess at where the data goes.
    CHECK(fsa_ioctl(device, fsa::Command::ReadFile) == reply(fsa::Status::InvalidParam));
    CHECK(fsa_device().layout_mismatches() == 1U);
}

TEST_CASE_METHOD(IosFixture, "StatFile and a stat query describe a file", "[hle][ios][fs]") {
    const s32 device = open_fsa();
    const u32 file = fsa_open(device, kContentPath);

    clear_request();
    REQUIRE(memory.fill(kResponse, fsa::layout::response_size, 0xAA));
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, file));
    REQUIRE(fsa_ioctl(device, fsa::Command::StatFile) == reply(fsa::Status::Ok));

    const ppcaddr_t stat = kResponse + fsa::layout::response_stat;
    CHECK(memory.read<u32>(stat + fsa::layout::stat_flags) == fsa::layout::stat_flag_file);
    CHECK(memory.read<u32>(stat + fsa::layout::stat_size) == kContent.size());
    // The rest of the FSStat is written, not left as whatever was there.
    CHECK(memory.read<u32>(stat + fsa::layout::stat_entry_id) == 0U);

    // The path-based form is GetInfoByQuery asking the Stat question, which
    // is what coreinit's FSGetStat sends.
    clear_request();
    static_cast<void>(put_string("/vol/content/sub", kRequest + fsa::layout::request_path));
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_query_type,
                                  fsa::layout::query_type_stat));
    REQUIRE(fsa_ioctl(device, fsa::Command::GetInfoByQuery) == reply(fsa::Status::Ok));
    CHECK(memory.read<u32>(stat + fsa::layout::stat_flags) == fsa::layout::stat_flag_directory);

    // Any other question is refused by name rather than answered with zero.
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_query_type, 0));
    CHECK(fsa_ioctl(device, fsa::Command::GetInfoByQuery) ==
          reply(fsa::Status::UnsupportedCommand));
}

TEST_CASE_METHOD(IosFixture, "a stale file handle is named as one", "[hle][ios][fs]") {
    const s32 device = open_fsa();
    clear_request();
    REQUIRE(memory.try_write<u32>(kRequest + fsa::layout::request_handle, 0x12345678));
    CHECK(fsa_ioctl(device, fsa::Command::CloseFile) == reply(fsa::Status::InvalidFileHandle));
    CHECK(fsa_ioctl(device, fsa::Command::CloseDir) == reply(fsa::Status::InvalidDirHandle));
}

TEST_CASE_METHOD(IosFixture,
                 "an emulated error is answered without being carried out",
                 "[hle][ios][fs]") {
    const s32 device = open_fsa();
    clear_request();
    static_cast<void>(put_string(kContentPath, kRequest + fsa::layout::request_path));
    static_cast<void>(put_string("r", kRequest + fsa::layout::request_open_mode));
    REQUIRE(memory.try_write<s32>(kRequest + fsa::layout::request_emulated_error,
                                  static_cast<s32>(fsa::Status::MediaNotReady)));

    CHECK(fsa_ioctl(device, fsa::Command::OpenFile) == reply(fsa::Status::MediaNotReady));
    CHECK(os.filesystem.open_file_count() == 0);
}

// ---------------------------------------------------------------------------
// The asynchronous forms
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(IosFixture, "IOS_OpenAsync calls back with the handle", "[hle][ios][async]") {
    static_cast<void>(put_string("/dev/fsa", kDevicePath));

    CHECK(icall("IOS_OpenAsync", {kDevicePath, 1, kCallbackProgram, 0xC0DE}) ==
          static_cast<s32>(IosError::Ok));

    // The work happened in the call -- the device is open -- but the reply
    // has not been delivered. A title that opens a device and then sets a
    // flag depends on that order.
    CHECK(os.ios.open_count() == 1);
    CHECK(callback_count() == 0U);
    CHECK(os.scheduler.timer_count() == 1U);

    write_main_calling("OSSleepTicks", 8000);
    static_cast<void>(run_it());

    CHECK(callback_count() == 1U);
    CHECK(callback_argument(0) == static_cast<u32>(IosManager::first_handle));
    CHECK(callback_argument(1) == 0xC0DEU);
}

TEST_CASE_METHOD(IosFixture, "IOS_IoctlAsync calls back with the status", "[hle][ios][async]") {
    const s32 device = open_fsa();
    clear_request();
    static_cast<void>(put_string(kContentPath, kRequest + fsa::layout::request_path));
    static_cast<void>(put_string("r", kRequest + fsa::layout::request_open_mode));

    CHECK(icall("IOS_IoctlAsync",
                {static_cast<u32>(device),
                 static_cast<u32>(fsa::Command::OpenFile),
                 kRequest,
                 fsa::layout::request_size,
                 kResponse,
                 fsa::layout::response_size,
                 kCallbackProgram,
                 0xBEEF}) == static_cast<s32>(IosError::Ok));

    // The file is already open; only the acknowledgement is in flight.
    CHECK(os.filesystem.open_file_count() == 1);
    CHECK(callback_count() == 0U);

    write_main_calling("OSSleepTicks", 8000);
    static_cast<void>(run_it());

    CHECK(callback_count() == 1U);
    CHECK(callback_argument(0) == static_cast<u32>(IosError::Ok));
    CHECK(callback_argument(1) == 0xBEEFU);
}

TEST_CASE_METHOD(IosFixture,
                 "an asynchronous failure reaches the callback as a failure",
                 "[hle][ios][async]") {
    static_cast<void>(put_string("/dev/nothing_here", kDevicePath));

    CHECK(icall("IOS_OpenAsync", {kDevicePath, 1, kCallbackProgram, 0}) ==
          static_cast<s32>(IosError::Ok));

    write_main_calling("OSSleepTicks", 8000);
    static_cast<void>(run_it());

    // The call was accepted and the *result* is the failure, which is the
    // shape of every asynchronous API here: refusing in the call would tell
    // the title at the wrong moment.
    CHECK(callback_count() == 1U);
    CHECK(static_cast<s32>(callback_argument(0)) == static_cast<s32>(IosError::NoExists));
}

TEST_CASE_METHOD(IosFixture,
                 "an asynchronous request with nowhere to reply is reported",
                 "[hle][ios][async]") {
    static_cast<void>(put_string("/dev/fsa", kDevicePath));

    CHECK(icall("IOS_OpenAsync", {kDevicePath, 1, 0, 0}) == static_cast<s32>(IosError::Ok));

    // Hardware would leave the caller waiting for a reply that never comes.
    // Nothing is scheduled, and the log says why -- which is the difference
    // between a findable bug and a title that simply stops.
    CHECK(os.scheduler.timer_count() == 0U);
}

TEST_CASE_METHOD(IosFixture, "IPC and FS completions do not share ids", "[hle][ios][async]") {
    static_cast<void>(put_string("/dev/fsa", kDevicePath));
    CHECK(icall("IOS_OpenAsync", {kDevicePath, 1, kCallbackProgram, 0}) ==
          static_cast<s32>(IosError::Ok));

    // An asynchronous FS call in flight at the same time. Both live in the
    // scheduler's one timer list, so an id collision would be one completion
    // quietly replacing the other.
    constexpr ppcaddr_t kAsyncData = 0x10009000;
    constexpr u32 kClient = 0x1000A000;
    constexpr u32 kBlock = 0x1000B000;
    REQUIRE(memory.try_write<u32>(kAsyncData, kCallbackProgram));
    REQUIRE(memory.try_write<u32>(kAsyncData + 4, 0));
    REQUIRE(memory.try_write<u32>(kAsyncData + 8, 0));

    const s32 device = open_fsa();
    const u32 file = fsa_open(device, kContentPath);
    static_cast<void>(
        call("FSReadFileAsync", {kClient, kBlock, kBuffer, 1, 16, file, 0, kAsyncData}));

    CHECK(os.scheduler.timer_count() == 2U);
    CHECK(os.scheduler.has_timer(coreinit::ios_async_id_base));
    CHECK(os.scheduler.has_timer(coreinit::async_id_base));

    write_main_calling("OSSleepTicks", 8000);
    static_cast<void>(run_it());

    CHECK(callback_count() == 2U);
}

TEST_CASE_METHOD(IosFixture, "IOS_CloseAsync closes before it replies", "[hle][ios][async]") {
    const s32 device = open_fsa();

    CHECK(icall("IOS_CloseAsync", {static_cast<u32>(device), kCallbackProgram, 0x1234}) ==
          static_cast<s32>(IosError::Ok));
    CHECK(os.ios.open_count() == 0);

    write_main_calling("OSSleepTicks", 8000);
    static_cast<void>(run_it());

    CHECK(callback_count() == 1U);
    CHECK(callback_argument(0) == static_cast<u32>(IosError::Ok));
    CHECK(callback_argument(1) == 0x1234U);
}
