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
#include <set>
#include <span>
#include <string>
#include <vector>

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

/// A condition variable. What it has to remember is the mutex its waiters
/// released, so that waking one can hand the mutex back: the whole contract of
/// OSWaitCond is that it returns with the mutex held again.
struct Condition {
    u32 mutex = 0;
};

/// OSFastMutex is the same idea as OSMutex with a cheaper uncontended path and
/// no recursion. Modelled separately rather than aliased, because a title that
/// locks one twice deadlocks on hardware and would quietly succeed if this
/// were an OSMutex.
struct FastMutex {
    u32 owner = 0;
};

/// A rendezvous is a barrier across a set of cores: every core named in the
/// mask has to arrive before any of them continues.
struct Rendezvous {
    u32 arrived = 0;
};

struct SyncObjects {
    std::map<u32, Mutex> mutexes;
    std::map<u32, Event> events;
    std::map<u32, Semaphore> semaphores;
    std::map<u32, Condition> conditions;
    std::map<u32, FastMutex> fast_mutexes;
    std::map<u32, Rendezvous> rendezvous;

    void reset() {
        mutexes.clear();
        events.clear();
        semaphores.clear();
        conditions.clear();
        fast_mutexes.clear();
        rendezvous.clear();
    }
};

/// An alarm: a guest callback to run at a point on the timebase, once or
/// repeatedly. The structure lives in guest memory and its address is the
/// identity, as with every other coreinit object.
struct Alarm {
    u64 deadline = 0;
    u64 period = 0; ///< zero for a one-shot
    ppcaddr_t callback = 0;
    u32 user_data = 0;
    u32 tag = 0;
    bool armed = false;
    u64 fire_count = 0;
};

/// An asynchronous FS request in flight. The work itself is already done --
/// the Phase 5 filesystem is synchronous -- so what is pending is the
/// completion: hardware would take some microseconds to come back, and a title
/// that assumes the callback cannot run before its own call returns is
/// relying on exactly that.
struct AsyncFsRequest {
    ppcaddr_t client = 0;
    ppcaddr_t block = 0;
    s32 status = 0;
    ppcaddr_t callback = 0;
    u32 context = 0;
};

/// A recycled callback thread: its stack and the handle that goes with it.
struct CallbackSlot {
    ppcaddr_t stack = 0;
    u32 handle = 0;
};

/// The most callback threads that may be live at once. Reached only by a
/// title whose callbacks never return, which is a bug worth reporting rather
/// than growing memory for.
inline constexpr u32 max_callback_threads = 4096;

/// Where the asynchronous FS completion ids start. High enough to be out of
/// the way of every guest address an alarm is keyed by, and there are 65536
/// of them before it wraps -- each released as its completion fires.
inline constexpr u32 async_id_base = 0xFFFF0000;

/// How long an asynchronous FS call takes to come back, in timebase ticks.
/// About 16 microseconds: long enough that the completion cannot be mistaken
/// for part of the call, short enough not to matter. Nothing waits for it in
/// real time -- the idle skip (P2) jumps straight there.
inline constexpr u64 fs_async_latency_ticks = 1000;

struct Context {
    HeapManager heaps;
    FileSystem filesystem;
    Scheduler scheduler;
    SyncObjects sync;
    ReportSink report_sink;

    std::map<u32, Alarm> alarms;

    /// Guest memory, remembered by the calls that can complete later.
    ///
    /// An alarm callback and an asynchronous FS completion both run outside
    /// any HLE call, so neither has a `Call` to take memory from -- but
    /// neither can exist without one having armed it first, so the pointer is
    /// always set by the time it is needed.
    memory::Memory* memory = nullptr;

    /// Stacks and handles for the threads OS callbacks run on, recycled
    /// together: a title with a periodic alarm fires one every frame, and
    /// neither a stack nor a handle per firing survives an afternoon.
    std::vector<CallbackSlot> free_callback_stacks;
    u32 next_callback_handle = callback_handle_base;

    /// The command blocks with a request in flight. One block carries one
    /// request: hardware refuses a second, and letting it through here would
    /// deliver two completions through the same structure.
    std::set<ppcaddr_t> busy_fs_blocks;

    /// Ids for the asynchronous FS completions in flight, counting up through
    /// a range nothing else uses -- alarms are keyed by their guest address.
    /// Each id is released when its completion fires.
    u32 next_async_id = async_id_base;

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

/// How big a stack an OS callback thread gets. Callbacks are meant to be
/// short; a title doing real work in one gets the same 16 KiB hardware would
/// give it and the same overflow if it asks for more.
inline constexpr u32 callback_stack_size = 0x4000;

/// Run a guest function on its own thread, with `arguments` in r3 onward.
///
/// This is how the OS calls back into the title: an alarm handler (6.7), an
/// asynchronous read completing (6.9), and later a ProcUI transition. Returns
/// the thread handle, or zero if no stack could be allocated -- which is worth
/// checking, because a callback that never runs is the failure mode that looks
/// like the title hanging for no reason.
[[nodiscard]] u32 invoke_guest_callback(Context& context,
                                        memory::Memory& memory,
                                        std::string name,
                                        ppcaddr_t entry,
                                        std::span<const u32> arguments,
                                        s32 priority = default_priority);

/// Release the stacks of callback threads that have finished, and forget the
/// threads. Called before starting a new callback, so the cost lands on the
/// thing creating the work rather than on a timer somewhere.
void reap_callback_threads(Context& context);

/// Time and alarms (6.7). Called by register_all.
void register_time(Registry& registry, Context& context);

/// printf-style formatting against guest varargs. Exposed because OSReport,
/// OSFatal and __os_snprintf all need the same thing, and because the
/// conversion rules are worth testing on their own.
[[nodiscard]] std::string format_guest(Call& call, std::string_view format, u32 first_gpr);

} // namespace wiiu::hle::coreinit
