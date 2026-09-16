#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/filesystem.hpp>
#include <wiiu/hle/heap.hpp>
#include <wiiu/loader/vfs.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;

namespace {

/// Enough of a machine to call coreinit functions directly, without going
/// through a guest program. Calling the handlers is the real interface, so
/// these tests set up registers and dispatch exactly as the run loop does.
struct CoreinitFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;
    loader::VirtualFileSystem vfs;

    CoreinitFixture() {
        std::string error;
        REQUIRE(memory.initialize(error));

        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        coreinit::register_all(registry, os);

        REQUIRE(os.heaps.create_default_heaps(memory, error));

        REQUIRE(vfs.mount_title_directory(
            std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title", error));
        os.filesystem.set_filesystem(&vfs);
    }

    /// Call a coreinit function with the given arguments, returning r3.
    u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("coreinit", name);
        REQUIRE(function != nullptr);

        for (std::size_t i = 0; i < arguments.size() && i < 8; ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        state.lr = 0x02000000;

        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    /// Write a string into guest memory and return its address.
    ppcaddr_t put_string(std::string_view text, ppcaddr_t at) {
        const std::vector<u8> bytes(text.begin(), text.end());
        REQUIRE(memory.write_bytes(at, bytes));
        REQUIRE(memory.try_write<u8>(at + static_cast<u32>(bytes.size()), 0));
        return at;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Heaps
// ---------------------------------------------------------------------------

TEST_CASE("an expanded heap allocates and frees", "[hle][heap]") {
    ExpandedHeap heap{"test", 0x10000000, 0x1000};

    const ppcaddr_t first = heap.allocate(64, 0);
    const ppcaddr_t second = heap.allocate(64, 0);

    CHECK(first == 0x10000000U);
    CHECK(second == first + 64);
    CHECK(heap.block_size(first) == 64U);
    CHECK(heap.live_blocks() == 2);

    CHECK(heap.free(second));
    CHECK(heap.live_blocks() == 1);
    CHECK(heap.free(first));
    CHECK(heap.total_free() == 0x1000U);
}

TEST_CASE("allocations honour alignment", "[hle][heap]") {
    ExpandedHeap heap{"test", 0x10000000, 0x1000};

    CHECK(heap.allocate(1, 0) == 0x10000000U);
    const ppcaddr_t aligned = heap.allocate(16, 256);
    CHECK(aligned % 256 == 0U);
    CHECK(aligned > 0x10000000U);
}

TEST_CASE("a negative alignment allocates from the tail", "[hle][heap]") {
    // That is the API's own convention for picking the end, and games rely on
    // it to keep scratch allocations away from long-lived ones.
    ExpandedHeap heap{"test", 0x10000000, 0x1000};

    const ppcaddr_t head = heap.allocate(64, 4);
    const ppcaddr_t tail = heap.allocate(64, -4);

    CHECK(head == 0x10000000U);
    CHECK(tail > head);
    CHECK(tail + 64 == 0x10001000U); // flush against the end
}

TEST_CASE("a heap reports exhaustion rather than overlapping", "[hle][heap]") {
    ExpandedHeap heap{"small", 0x10000000, 128};

    CHECK(heap.allocate(64, 0) != 0U);
    CHECK(heap.allocate(64, 0) != 0U);
    CHECK(heap.allocate(64, 0) == 0U); // no room, and no silent overlap
}

TEST_CASE("head and tail allocations cannot collide", "[hle][heap]") {
    ExpandedHeap heap{"small", 0x10000000, 128};

    CHECK(heap.allocate(96, 4) != 0U);
    CHECK(heap.allocate(64, -4) == 0U); // would run into the head
}

TEST_CASE("freeing an address that was never allocated fails", "[hle][heap]") {
    ExpandedHeap heap{"test", 0x10000000, 0x1000};
    CHECK_FALSE(heap.free(0x10000500));

    const ppcaddr_t block = heap.allocate(32, 0);
    CHECK(heap.free(block));
    CHECK_FALSE(heap.free(block)); // and not twice
}

TEST_CASE_METHOD(CoreinitFixture, "MEMAllocFromDefaultHeap returns usable memory", "[hle][heap]") {
    const u32 address = call("MEMAllocFromDefaultHeap", {256});
    REQUIRE(address != 0U);

    // The address is real: it can be written and read back.
    REQUIRE(memory.try_write<u32>(address, 0xDEADBEEF));
    CHECK(memory.read<u32>(address) == 0xDEADBEEFU);

    CHECK(call("MEMGetSizeForMBlockExpHeap", {address}) == 256U);

    call("MEMFreeToDefaultHeap", {address});
    CHECK(call("MEMGetSizeForMBlockExpHeap", {address}) == 0U);
}

TEST_CASE_METHOD(CoreinitFixture, "MEM1 and the default heap are distinct", "[hle][heap]") {
    const u32 mem1 = call("MEMGetBaseHeapHandle", {0});
    const u32 mem2 = call("MEMGetBaseHeapHandle", {1});
    CHECK(mem1 != mem2);

    const u32 from_mem1 = call("MEMAllocFromExpHeapEx", {mem1, 128, 32});
    REQUIRE(from_mem1 != 0U);
    CHECK(from_mem1 >= memory::mem1_base);
    CHECK(from_mem1 < memory::mem1_base + memory::mem1_size);
}

TEST_CASE_METHOD(CoreinitFixture,
                 "freeing through the default heap finds the right one",
                 "[hle][heap]") {
    // A title can free a MEM1 block through MEMFreeToDefaultHeap, so the heap
    // is looked up from the address rather than assumed.
    const u32 mem1 = call("MEMGetBaseHeapHandle", {0});
    const u32 address = call("MEMAllocFromExpHeapEx", {mem1, 128, 32});
    REQUIRE(address != 0U);

    call("MEMFreeToDefaultHeap", {address});
    CHECK(call("MEMGetSizeForMBlockExpHeap", {address}) == 0U);
}

// ---------------------------------------------------------------------------
// Filesystem
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(CoreinitFixture, "a file opens, reads and closes", "[hle][fs]") {
    constexpr ppcaddr_t scratch = memory::data_heap_base;
    const ppcaddr_t path = put_string("/vol/content/data.bin", scratch);
    const ppcaddr_t mode = put_string("r", scratch + 0x100);
    constexpr ppcaddr_t handle_out = scratch + 0x200;
    constexpr ppcaddr_t buffer = scratch + 0x300;

    CHECK(call("FSInit", {}) == 0U);
    CHECK(static_cast<s32>(call("FSOpenFile", {0, 0, path, mode, handle_out, 0})) == 0);

    const u32 handle = memory.read<u32>(handle_out);
    CHECK(handle != 0U);

    // "content payload\n" is 16 bytes.
    const auto read = static_cast<s32>(call("FSReadFile", {0, 0, buffer, 1, 64, handle, 0}));
    CHECK(read == 16);

    std::vector<u8> contents(16);
    REQUIRE(memory.read_bytes(buffer, contents));
    CHECK(std::string(contents.begin(), contents.end()) == "content payload\n");

    CHECK(static_cast<s32>(call("FSCloseFile", {0, 0, handle, 0})) == 0);
}

TEST_CASE_METHOD(CoreinitFixture, "reading past the end returns zero items", "[hle][fs]") {
    constexpr ppcaddr_t scratch = memory::data_heap_base;
    const ppcaddr_t path = put_string("/vol/content/data.bin", scratch);
    const ppcaddr_t mode = put_string("r", scratch + 0x100);
    constexpr ppcaddr_t handle_out = scratch + 0x200;
    constexpr ppcaddr_t buffer = scratch + 0x300;

    REQUIRE(static_cast<s32>(call("FSOpenFile", {0, 0, path, mode, handle_out, 0})) == 0);
    const u32 handle = memory.read<u32>(handle_out);

    CHECK(static_cast<s32>(call("FSReadFile", {0, 0, buffer, 1, 1024, handle, 0})) == 16);
    CHECK(static_cast<s32>(call("FSReadFile", {0, 0, buffer, 1, 1024, handle, 0})) == 0);
}

TEST_CASE_METHOD(CoreinitFixture, "seeking moves the read position", "[hle][fs]") {
    constexpr ppcaddr_t scratch = memory::data_heap_base;
    const ppcaddr_t path = put_string("/vol/content/data.bin", scratch);
    const ppcaddr_t mode = put_string("r", scratch + 0x100);
    constexpr ppcaddr_t handle_out = scratch + 0x200;
    constexpr ppcaddr_t buffer = scratch + 0x300;
    constexpr ppcaddr_t position_out = scratch + 0x400;

    REQUIRE(static_cast<s32>(call("FSOpenFile", {0, 0, path, mode, handle_out, 0})) == 0);
    const u32 handle = memory.read<u32>(handle_out);

    CHECK(static_cast<s32>(call("FSSetPosFile", {0, 0, handle, 8, 0})) == 0);
    CHECK(static_cast<s32>(call("FSGetPosFile", {0, 0, handle, position_out, 0})) == 0);
    CHECK(memory.read<u32>(position_out) == 8U);

    const auto read = static_cast<s32>(call("FSReadFile", {0, 0, buffer, 1, 64, handle, 0}));
    CHECK(read == 8);

    std::vector<u8> contents(8);
    REQUIRE(memory.read_bytes(buffer, contents));
    CHECK(std::string(contents.begin(), contents.end()) == "payload\n");
}

TEST_CASE_METHOD(CoreinitFixture, "opening a missing file reports not found", "[hle][fs]") {
    constexpr ppcaddr_t scratch = memory::data_heap_base;
    const ppcaddr_t path = put_string("/vol/content/nothere.bin", scratch);
    const ppcaddr_t mode = put_string("r", scratch + 0x100);

    const auto status =
        static_cast<s32>(call("FSOpenFile", {0, 0, path, mode, scratch + 0x200, 0}));
    CHECK(status == static_cast<s32>(FsStatus::NotFound));
}

TEST_CASE_METHOD(CoreinitFixture, "opening a directory as a file is refused", "[hle][fs]") {
    constexpr ppcaddr_t scratch = memory::data_heap_base;
    const ppcaddr_t path = put_string("/vol/content/sub", scratch);
    const ppcaddr_t mode = put_string("r", scratch + 0x100);

    const auto status =
        static_cast<s32>(call("FSOpenFile", {0, 0, path, mode, scratch + 0x200, 0}));
    CHECK(status == static_cast<s32>(FsStatus::NotFile));
}

TEST_CASE_METHOD(CoreinitFixture,
                 "opening for writing is refused, not silently ignored",
                 "[hle][fs]") {
    // Content is read-only. Returning success and dropping the writes would
    // look like a working save system right up until the data was needed.
    constexpr ppcaddr_t scratch = memory::data_heap_base;
    const ppcaddr_t path = put_string("/vol/content/data.bin", scratch);
    const ppcaddr_t mode = put_string("w", scratch + 0x100);

    const auto status =
        static_cast<s32>(call("FSOpenFile", {0, 0, path, mode, scratch + 0x200, 0}));
    CHECK(status == static_cast<s32>(FsStatus::PermissionError));
}

TEST_CASE("directory iteration walks every entry once", "[hle][fs]") {
    loader::VirtualFileSystem vfs;
    std::string error;
    REQUIRE(vfs.mount_title_directory(
        std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title", error));

    FileSystem fs;
    fs.set_filesystem(&vfs);

    u32 handle = 0;
    REQUIRE(fs.open_directory("/vol/content", handle) == FsStatus::Ok);

    std::vector<std::string> names;
    loader::FileInfo entry;
    while (fs.read_directory(handle, entry) == FsStatus::Ok) {
        names.push_back(entry.name);
    }

    REQUIRE(names.size() == 2);
    CHECK(names[0] == "data.bin");
    CHECK(names[1] == "sub");
    CHECK(fs.close_directory(handle) == FsStatus::Ok);
}

TEST_CASE("stat distinguishes files from directories", "[hle][fs]") {
    loader::VirtualFileSystem vfs;
    std::string error;
    REQUIRE(vfs.mount_title_directory(
        std::filesystem::path{WIIU_TEST_DATA_DIR} / "loader" / "data" / "title", error));

    FileSystem fs;
    fs.set_filesystem(&vfs);

    loader::FileInfo info;
    REQUIRE(fs.stat("/vol/content/data.bin", info) == FsStatus::Ok);
    CHECK_FALSE(info.is_directory);
    CHECK(info.size == 16);

    REQUIRE(fs.stat("/vol/content/sub", info) == FsStatus::Ok);
    CHECK(info.is_directory);

    CHECK(fs.stat("/vol/content/nothere", info) == FsStatus::NotFound);
}

TEST_CASE("FS without a mounted filesystem reports media not ready", "[hle][fs]") {
    // Better than NotFound: nothing is wrong with the path, there is just no
    // medium, and the two send a developer in different directions.
    FileSystem fs;
    u32 handle = 0;
    CHECK(fs.open("/vol/content/data.bin", "r", handle) == FsStatus::MediaNotReady);
}

// ---------------------------------------------------------------------------
// Time (6.7)
// ---------------------------------------------------------------------------

TEST_CASE("the epoch is 2000-01-01", "[hle][coreinit][time]") {
    const coreinit::CalendarTime zero = coreinit::ticks_to_calendar(0);
    CHECK(zero.year == 2000);
    CHECK(zero.month == 0);
    CHECK(zero.day == 1);
    CHECK(zero.hour == 0);
    CHECK(zero.minute == 0);
    CHECK(zero.second == 0);
    CHECK(zero.weekday == 6); // 2000-01-01 was a Saturday
    CHECK(zero.yearday == 0);
}

TEST_CASE("calendar time round-trips through ticks", "[hle][coreinit][time]") {
    // Dates chosen for the things that break date arithmetic: a leap day in a
    // century year that *is* a leap year, a leap day in an ordinary leap year,
    // the day either side of a century boundary, and the end of a year.
    const std::vector<coreinit::CalendarTime> dates{
        {.second = 0, .minute = 0, .hour = 0, .day = 1, .month = 0, .year = 2000},
        {.second = 59, .minute = 59, .hour = 23, .day = 29, .month = 1, .year = 2000},
        {.second = 30, .minute = 15, .hour = 12, .day = 29, .month = 1, .year = 2024},
        {.second = 1, .minute = 2, .hour = 3, .day = 28, .month = 1, .year = 2100},
        {.second = 0, .minute = 0, .hour = 0, .day = 1, .month = 2, .year = 2100},
        {.second = 59, .minute = 59, .hour = 23, .day = 31, .month = 11, .year = 2099},
        {.second = 33, .minute = 1, .hour = 8, .day = 16, .month = 8, .year = 2026},
    };

    for (const coreinit::CalendarTime& date : dates) {
        const u64 ticks = coreinit::calendar_to_ticks(date);
        const coreinit::CalendarTime back = coreinit::ticks_to_calendar(ticks);

        INFO(date.year << "-" << date.month + 1 << "-" << date.day);
        CHECK(back.year == date.year);
        CHECK(back.month == date.month);
        CHECK(back.day == date.day);
        CHECK(back.hour == date.hour);
        CHECK(back.minute == date.minute);
        CHECK(back.second == date.second);
    }
}

TEST_CASE("2100 is not a leap year", "[hle][coreinit][time]") {
    // The century rule is the one every hand-rolled calendar gets wrong, and
    // the Wii U's epoch means a title can reach it.
    const coreinit::CalendarTime feb28{
        .second = 0, .minute = 0, .hour = 0, .day = 28, .month = 1, .year = 2100};
    const coreinit::CalendarTime march1{
        .second = 0, .minute = 0, .hour = 0, .day = 1, .month = 2, .year = 2100};

    CHECK(coreinit::calendar_to_ticks(march1) - coreinit::calendar_to_ticks(feb28) ==
          86400ULL * cpu::timebase_hz);

    // 2000 is, because it divides by 400.
    const coreinit::CalendarTime y2k_feb28{
        .second = 0, .minute = 0, .hour = 0, .day = 28, .month = 1, .year = 2000};
    const coreinit::CalendarTime y2k_march1{
        .second = 0, .minute = 0, .hour = 0, .day = 1, .month = 2, .year = 2000};
    CHECK(coreinit::calendar_to_ticks(y2k_march1) - coreinit::calendar_to_ticks(y2k_feb28) ==
          2 * 86400ULL * cpu::timebase_hz);
}

TEST_CASE("weekdays and days-of-year are right", "[hle][coreinit][time]") {
    const coreinit::CalendarTime day{
        .second = 0, .minute = 0, .hour = 0, .day = 16, .month = 8, .year = 2026};
    const coreinit::CalendarTime got =
        coreinit::ticks_to_calendar(coreinit::calendar_to_ticks(day));

    CHECK(got.weekday == 3); // 2026-09-16 is a Wednesday
    CHECK(got.yearday == 258);
}

TEST_CASE("out-of-range calendar fields normalise", "[hle][coreinit][time]") {
    // A title that adds a month to a date and hands it straight back should
    // get the answer it meant rather than a rejection.
    const coreinit::CalendarTime december{
        .second = 0, .minute = 0, .hour = 0, .day = 1, .month = 11, .year = 2026};
    const coreinit::CalendarTime thirteenth{
        .second = 0, .minute = 0, .hour = 0, .day = 1, .month = 12, .year = 2026};

    const coreinit::CalendarTime rolled =
        coreinit::ticks_to_calendar(coreinit::calendar_to_ticks(thirteenth));
    CHECK(rolled.year == 2027);
    CHECK(rolled.month == 0);
    CHECK(coreinit::calendar_to_ticks(thirteenth) > coreinit::calendar_to_ticks(december));
}

TEST_CASE("OSGetTime is the real-time clock plus the timebase", "[hle][coreinit][time]") {
    CoreinitFixture fixture;

    // Pin the clock rather than reading the host's, so the arithmetic is the
    // only thing under test.
    fixture.os.rtc_epoch = 1'000'000;
    fixture.os.rtc_valid = true;

    fixture.state.timebase = 250;

    Function* const get_time = fixture.registry.find("coreinit", "OSGetTime");
    REQUIRE(get_time != nullptr);
    Call call{.state = fixture.state, .memory = fixture.memory};
    get_time->handler(call);

    const u64 returned = (static_cast<u64>(fixture.state.gpr[3]) << 32) | fixture.state.gpr[4];
    CHECK(returned == 1'000'250ULL);

    // OSGetSystemTime is the timebase alone: uptime, not wall-clock.
    Function* const system_time = fixture.registry.find("coreinit", "OSGetSystemTime");
    REQUIRE(system_time != nullptr);
    Call second{.state = fixture.state, .memory = fixture.memory};
    system_time->handler(second);
    CHECK(((static_cast<u64>(fixture.state.gpr[3]) << 32) | fixture.state.gpr[4]) == 250ULL);
}

TEST_CASE("OSTicksToCalendarTime writes big-endian fields", "[hle][coreinit][time]") {
    CoreinitFixture fixture;

    const coreinit::CalendarTime when{
        .second = 7, .minute = 6, .hour = 5, .day = 4, .month = 2, .year = 2015};
    const u64 ticks = coreinit::calendar_to_ticks(when);

    const ppcaddr_t out = memory::data_heap_base + 0x100;

    // OSTicksToCalendarTime(OSTime, OSCalendarTime*): the 64-bit argument takes
    // r3:r4 and the pointer follows in r5.
    Function* const function = fixture.registry.find("coreinit", "OSTicksToCalendarTime");
    REQUIRE(function != nullptr);
    fixture.state.gpr[3] = static_cast<u32>(ticks >> 32);
    fixture.state.gpr[4] = static_cast<u32>(ticks);
    fixture.state.gpr[5] = out;

    Call call{.state = fixture.state, .memory = fixture.memory};
    function->handler(call);

    u32 field = 0;
    REQUIRE(fixture.memory.try_read<u32>(out + 0, field));
    CHECK(field == 7U); // second
    REQUIRE(fixture.memory.try_read<u32>(out + 20, field));
    CHECK(field == 2015U); // year
    REQUIRE(fixture.memory.try_read<u32>(out + 16, field));
    CHECK(field == 2U); // month, counting from zero

    // And back the other way, through guest memory.
    Function* const inverse = fixture.registry.find("coreinit", "OSCalendarTimeToTicks");
    REQUIRE(inverse != nullptr);
    fixture.state.gpr[3] = out;
    Call back{.state = fixture.state, .memory = fixture.memory};
    inverse->handler(back);
    CHECK(((static_cast<u64>(fixture.state.gpr[3]) << 32) | fixture.state.gpr[4]) == ticks);
}

TEST_CASE("negative sub-second fields borrow, not wrap", "[hle][coreinit][time]") {
    // Every field is signed and comes from guest memory. Casting a negative
    // millisecond straight to u64 adds about three and a half days, and a
    // date three days in the future is a very hard wrong answer to trace back
    // to an unsigned cast.
    const coreinit::CalendarTime base{
        .second = 30, .minute = 0, .hour = 12, .day = 1, .month = 0, .year = 2020};
    coreinit::CalendarTime minus_a_millisecond = base;
    minus_a_millisecond.millisecond = -1;

    const u64 at = coreinit::calendar_to_ticks(base);
    const u64 before = coreinit::calendar_to_ticks(minus_a_millisecond);

    CHECK(before < at);
    // A millisecond, to within the one tick that microsecond-resolution
    // fields can round away.
    CHECK(at - before >= cpu::timebase_hz / 1000);
    CHECK(at - before <= cpu::timebase_hz / 1000 + 1);

    // And a millisecond past the epoch itself still clamps rather than
    // wrapping to something enormous.
    const coreinit::CalendarTime epoch_minus{
        .second = 0, .minute = 0, .hour = 0, .day = 1, .month = 0, .year = 2000, .millisecond = -1};
    CHECK(coreinit::calendar_to_ticks(epoch_minus) == 0U);
}

TEST_CASE("an absurd year saturates rather than wrapping", "[hle][coreinit][time]") {
    // read_calendar takes whatever the guest wrote, so every field can be
    // anything a 32-bit word can hold. Wrapping would return a plausible date
    // thousands of years out; saturating is at least obviously wrong.
    coreinit::CalendarTime absurd;
    absurd.year = 0x7FFFFFFF;
    CHECK(coreinit::calendar_to_ticks(absurd) == std::numeric_limits<u64>::max());

    absurd.year = -0x7FFFFFFF;
    CHECK(coreinit::calendar_to_ticks(absurd) == 0U);
}
