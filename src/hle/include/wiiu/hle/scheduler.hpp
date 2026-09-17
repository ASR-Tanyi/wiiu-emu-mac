// Phase 6.3 — threads and scheduling.
//
// Cafe OS gives an application three cores and a priority-scheduled thread on
// each. This implementation runs them on **one host thread**, interleaving
// guest threads deterministically, rather than mapping each core onto a host
// thread.
//
// That is a deliberate first step, not an oversight. Deterministic scheduling
// means a failure reproduces exactly, which is what makes the record/replay in
// Phase 10.4 and the JIT lockstep comparison in 12.7 possible at all. P32
// already notes that free-running cores are not reliably faster; when real
// parallelism is worth measuring, it goes behind this same interface.
//
// Thread handles are the guest `OSThread*` the caller allocated, the way the
// API works: the caller owns the structure and passes it back every time. The
// state that matters to us is kept alongside it rather than inside it, for the
// same reason the heaps are (see heap.hpp).
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/state.hpp>

#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace wiiu::hle {

enum class ThreadState : u8 {
    Ready,
    Running,
    Sleeping,  ///< waiting for a deadline
    Blocked,   ///< waiting on a mutex, event or join
    Suspended, ///< created but not resumed, or explicitly suspended
    Exited,
};

[[nodiscard]] std::string_view thread_state_name(ThreadState state) noexcept;

/// Why a thread is blocked, so the debug UI can say something useful and a
/// deadlock is legible rather than just "stopped".
enum class BlockReason : u8 {
    None,
    Mutex,
    Event,
    Semaphore,
    Condition,
    Rendezvous,
    Join,
    Sleep,
};

[[nodiscard]] std::string_view block_reason_name(BlockReason reason) noexcept;

/// Cafe OS priorities run 0 (highest) to 31 (lowest).
inline constexpr s32 highest_priority = 0;
inline constexpr s32 lowest_priority = 31;
inline constexpr s32 default_priority = 16;

/// Where a thread's entry point returns to. Outside every mapped region, so
/// the run loop spots it without the thread noticing.
inline constexpr ppcaddr_t thread_exit_sentinel = 0x01FE0000;

/// The handle the main thread is adopted under. Guest OSThread pointers are
/// real allocations, so a value outside any mapped region cannot collide with
/// one.
inline constexpr u32 main_thread_handle = 0x01FD0000;

/// Handles for the threads the OS makes on the title's behalf -- alarm
/// callbacks, asynchronous FS completion. Same reasoning: outside every mapped
/// region, so they cannot be mistaken for an OSThread the title allocated.
inline constexpr u32 callback_handle_base = 0x01FC0000;

struct Thread {
    u32 handle = 0; ///< the guest OSThread pointer
    std::string name;

    cpu::CoreState context;
    ThreadState state = ThreadState::Suspended;
    BlockReason block_reason = BlockReason::None;
    u32 blocked_on = 0; ///< the object's guest address, for diagnostics

    s32 priority = default_priority;
    u32 affinity = 0x7; ///< bitmask of cores this may run on
    u32 core = 1;       ///< core 1 has the 2 MiB L2; Cafe OS puts the main thread there

    s32 exit_value = 0;
    u32 suspend_count = 1; ///< created suspended, as OSCreateThread leaves it
    u64 wake_time = 0;

    /// Where this thread's stack was allocated, for the threads the emulator
    /// allocated one for. Zero for a thread whose stack the title owns, which
    /// is every thread it created itself.
    ppcaddr_t stack_base = 0;

    /// Threads waiting on this one, with where each wants the exit value
    /// written.
    std::vector<std::pair<u32, ppcaddr_t>> joiners;

    [[nodiscard]] bool runnable() const noexcept { return state == ThreadState::Ready; }
};

/// Round-robin between guest threads, honouring priority.
class Scheduler {
public:
    /// How many instructions a thread runs before the scheduler looks again.
    ///
    /// Without this, a thread that never yields -- a spin on a flag another
    /// thread sets -- would run forever and the program would hang, which is
    /// exactly the bug that looks like an emulator fault rather than a
    /// scheduling one.
    static constexpr u64 default_timeslice = 10'000;

    void reset();

    [[nodiscard]] u32 create(u32 handle,
                             std::string name,
                             ppcaddr_t entry,
                             u32 argc,
                             ppcaddr_t argv,
                             ppcaddr_t stack_top,
                             u32 stack_size,
                             s32 priority,
                             u32 affinity);

    /// Adopt the already-running state as the main thread, so the module's
    /// entry point is schedulable like anything else.
    [[nodiscard]] u32 adopt_main(const cpu::CoreState& state, u32 handle);

    /// Create a thread that starts at `entry` with `arguments` in r3 onward
    /// and is runnable immediately.
    ///
    /// A callback from the OS into the title -- an alarm handler, an
    /// asynchronous read completing -- is an ordinary guest function that has
    /// to run on a stack somewhere. Hardware runs them on a system thread per
    /// core; here each one is a thread, which has the same observable shape
    /// without a pool whose state could disagree with the scheduler's.
    [[nodiscard]] u32 create_callback(u32 handle,
                                      std::string name,
                                      ppcaddr_t entry,
                                      std::span<const u32> arguments,
                                      ppcaddr_t stack_top,
                                      s32 priority = default_priority);

    /// Forget a thread that has exited, so a title firing an alarm every frame
    /// does not accumulate one dead thread per frame. Refuses anything still
    /// alive: a live thread that vanished would take its state with it.
    bool remove(u32 handle);

    [[nodiscard]] bool resume(u32 handle);
    [[nodiscard]] bool suspend(u32 handle);

    /// Give up the rest of this timeslice. Returns true if another thread was
    /// picked.
    bool yield(cpu::CoreState& state);

    void sleep(cpu::CoreState& state, u64 ticks);

    void block(cpu::CoreState& state, BlockReason reason, u32 object);

    /// Block a thread that is not the one running. Used when a wait is handed
    /// from one object to another -- a condition variable's waiters queue up
    /// on its mutex when it is signalled -- which cannot go through `block`,
    /// since that one blocks the caller.
    void block_thread(u32 handle, BlockReason reason, u32 object);

    void unblock(u32 handle);
    /// Wake every thread blocked on `object`. Returns how many.
    u32 unblock_all(u32 object);

    /// Wake the highest-priority thread blocked on `object`, and say which one
    /// it was -- zero if nobody was waiting.
    ///
    /// The handle is the point: whoever hands out a mutex has to record who
    /// now owns it. Returning only "somebody" left every woken waiter
    /// resuming *after* its OSLockMutex with the mutex unowned, which is two
    /// threads inside the same critical section.
    u32 unblock_one(u32 object);

    /// Everything blocked on `object`, highest priority first.
    [[nodiscard]] std::vector<u32> waiters(u32 object) const;

    void exit_current(cpu::CoreState& state, s32 value);

    /// Block the caller until `handle` exits. False if it has already exited,
    /// in which case `value` is filled in and nothing blocks.
    [[nodiscard]] bool join(cpu::CoreState& state, u32 handle, ppcaddr_t out_address, s32& value);

    /// Called for each joiner when a thread exits, so the exit value can be
    /// written into guest memory. The scheduler has no memory access of its
    /// own, and giving it some would make it harder to test.
    using ExitCallback = std::function<void(s32 exit_value, ppcaddr_t out_address)>;

    void set_exit_callback(ExitCallback callback) { on_exit_ = std::move(callback); }

    /// Pick the next thread and switch to it if that is not the current one.
    /// Also wakes any sleeper whose deadline has passed, and fires any timer
    /// that has come due.
    bool reschedule(cpu::CoreState& state);

    // --- deadlines --------------------------------------------------------
    //
    // Anything that has to happen at a particular point on the timebase --
    // an alarm (6.7), and later a vsync or an audio frame -- registers here.
    // Keeping them with the sleeping threads is what makes "nothing can run
    // until time moves" answerable in one place, which is both the idle skip
    // below and the difference between a deadlock and a wait.

    using TimerCallback = std::function<void(u64 now)>;

    /// Register or re-arm a deadline. An id that already exists is replaced,
    /// which is what re-arming a periodic alarm does.
    void set_timer(u32 id, u64 deadline, TimerCallback callback);

    void clear_timer(u32 id) noexcept;

    [[nodiscard]] bool has_timer(u32 id) const noexcept;

    [[nodiscard]] std::size_t timer_count() const noexcept { return timers_.size(); }

    /// Fire every timer due at `now`, oldest deadline first. Returns how many
    /// fired. A timer is removed before its callback runs, so a callback that
    /// re-arms the same id keeps the new deadline rather than losing it.
    u32 service_timers(u64 now);

    /// The next moment anything is waiting for: the earliest sleeping thread
    /// or timer deadline.
    [[nodiscard]] std::optional<u64> next_deadline() const noexcept;

    /// How far the clock would have to move before anything could happen, or
    /// zero if something can run now or nothing ever will (**P2**).
    ///
    /// Emulating a wait faithfully means running a spin loop or an idle core
    /// for however many million instructions the wait lasts, producing
    /// nothing. Skipping to the next deadline is both faster and *more*
    /// accurate about when the guest resumes. The run loop does the moving,
    /// not the scheduler: the decrementer counts down with the timebase, and
    /// only the interpreter owns that.
    [[nodiscard]] u64 idle_skip_ticks(const cpu::CoreState& state) const noexcept;

    [[nodiscard]] Thread* current() noexcept;
    [[nodiscard]] const Thread* current() const noexcept;
    [[nodiscard]] Thread* find(u32 handle) noexcept;

    [[nodiscard]] u32 current_handle() const noexcept { return current_; }

    [[nodiscard]] std::vector<const Thread*> threads() const;
    [[nodiscard]] std::size_t runnable_count() const noexcept;
    [[nodiscard]] bool all_exited() const noexcept;

    /// True when nothing can run but something is still waiting -- a deadlock,
    /// or a sleeper whose deadline has not arrived.
    [[nodiscard]] bool stalled() const noexcept;

    [[nodiscard]] u64 switch_count() const noexcept { return switches_; }

    /// Fold every thread into `seed`: its scheduling state, and the saved
    /// context of each thread that is not the one currently running. A hash
    /// of the live core says nothing about a thread that is blocked, so a
    /// divergence in one would stay invisible until it was scheduled back in
    /// -- long after the cause (10.4).
    [[nodiscard]] u64 hash(u64 seed) const noexcept;

private:
    void save_current(const cpu::CoreState& state);
    void restore(Thread& thread, cpu::CoreState& state);
    [[nodiscard]] Thread* pick_next();
    void wake_sleepers(u64 now);

    struct Timer {
        u64 deadline = 0;
        TimerCallback callback;
    };

    ExitCallback on_exit_;
    std::map<u32, Thread> threads_;
    std::map<u32, Timer> timers_;
    u32 current_ = 0;
    u32 next_core_ = 0;
    u64 switches_ = 0;
    u64 skipped_ticks_ = 0;
    bool servicing_timers_ = false;

public:
    /// Timebase ticks skipped rather than emulated (**P2**). A counter rather
    /// than a log line because the interesting question is what fraction of a
    /// run was spent waiting, which only shows up in aggregate.
    [[nodiscard]] u64 skipped_ticks() const noexcept { return skipped_ticks_; }

    void note_skipped(u64 ticks) noexcept { skipped_ticks_ += ticks; }
};

} // namespace wiiu::hle
