// Phase 6.8 — coreinit.
//
// The library everything else is built on. Only the parts a program needs to
// start up and say something are here; threads, heaps and the filesystem are
// 6.4 through 6.9.
#pragma once

#include <wiiu/hle/filesystem.hpp>
#include <wiiu/hle/heap.hpp>
#include <wiiu/hle/recording.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/hle/scheduler.hpp>

#include <functional>
#include <map>
#include <string>

namespace wiiu::hle::coreinit {

/// Where OSReport output goes. Defaults to the log; tests capture it.
using ReportSink = std::function<void(std::string_view)>;

/// The state coreinit owns on behalf of a running title.
///
/// Passed in rather than kept in globals: the emulator runs one title at a
/// time, but a test should be able to stand up a fresh one without the
/// previous one's heaps still being allocated.
/// A mutex, event or semaphore. These live in guest memory and the caller
/// passes the pointer back every time, so the pointer is the identity; the
/// state we care about is kept alongside rather than inside it.
struct Mutex {
    u32 owner = 0;     ///< the thread holding it, or 0
    u32 recursion = 0; ///< OSMutex is recursive for its owner
};

struct Event {
    bool signalled = false;
    bool auto_reset = false; ///< OS_EVENT_MODE_AUTO releases one waiter and resets
};

struct Semaphore {
    s32 count = 0;
};

struct SyncObjects {
    std::map<u32, Mutex> mutexes;
    std::map<u32, Event> events;
    std::map<u32, Semaphore> semaphores;

    void reset() {
        mutexes.clear();
        events.clear();
        semaphores.clear();
    }
};

struct Context {
    HeapManager heaps;
    FileSystem filesystem;
    Scheduler scheduler;
    SyncObjects sync;
    ReportSink report_sink;

    /// Record/replay (10.4). Attached before the run starts; null means
    /// neither, and every host value is simply taken as it comes.
    Recording* recording = nullptr;

    /// What OSGetTime would have returned at timebase 0: ticks since the Wii
    /// U's epoch of 2000-01-01. Read from the host clock on first use, and
    /// the only value in the whole emulator that differs between two runs of
    /// the same title — which is why it goes through the recording.
    u64 rtc_epoch = 0;
    bool rtc_valid = false;
};

/// Wii U calendar time. Laid out exactly as `OSCalendarTime` is in guest
/// memory: ten big-endian 32-bit fields, in this order.
struct CalendarTime {
    s32 second = 0;
    s32 minute = 0;
    s32 hour = 0;
    s32 day = 1;     ///< 1-31
    s32 month = 0;   ///< 0-11, as in C's tm_mon
    s32 year = 2000; ///< the full year, unlike C's tm_year
    s32 weekday = 0; ///< 0 = Sunday
    s32 yearday = 0; ///< 0-365
    s32 millisecond = 0;
    s32 microsecond = 0;
};

inline constexpr u32 calendar_time_size = 10 * sizeof(u32);

/// Ticks are counted from 2000-01-01 00:00:00 UTC at the timebase rate.
[[nodiscard]] CalendarTime ticks_to_calendar(u64 ticks) noexcept;
[[nodiscard]] u64 calendar_to_ticks(const CalendarTime& time) noexcept;

/// The real-time clock, read once and then held: every later call derives
/// from the deterministic timebase, so a title that asks the time twice a
/// frame does not drag host jitter into the emulation.
[[nodiscard]] u64 rtc_epoch_ticks(Context& context);

namespace detail {

/// The context the registered handlers act on.
///
/// The marshalling framework's whole point is that a handler is a plain
/// function, and a plain function has nowhere to keep a context — so the one
/// belonging to the running title is held here instead. There is exactly one,
/// shared by every coreinit translation unit, because two of them that could
/// disagree is a bug waiting to be written. It is also the reason the
/// emulator runs one title at a time, which is worth saying out loud rather
/// than discovering.
[[nodiscard]] Context* active_context() noexcept;

void set_active_context(Context* context) noexcept;

} // namespace detail

/// Register the implemented functions, plus stubs for the ones a typical
/// program touches on the way to main().
void register_all(Registry& registry, Context& context);

/// Threads, mutexes, events and semaphores (6.4 / 6.5). Called by
/// register_all; separate because the blocking rules differ.
void register_threads(Registry& registry, Context& context);

/// Time and alarms (6.7). Called by register_all.
void register_time(Registry& registry, Context& context);

/// printf-style formatting against guest varargs. Exposed because OSReport,
/// OSFatal and __os_snprintf all need the same thing, and because the
/// conversion rules are worth testing on their own.
[[nodiscard]] std::string format_guest(Call& call, std::string_view format, u32 first_gpr);

} // namespace wiiu::hle::coreinit
