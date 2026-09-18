#include <wiiu/common/log.hpp>
#include <wiiu/hle/recording.hpp>
#include <wiiu/hle/scheduler.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace wiiu::hle {
namespace {

using wiiu::log::Channel;

} // namespace

std::string_view thread_state_name(ThreadState state) noexcept {
    switch (state) {
        case ThreadState::Ready:
            return "ready";
        case ThreadState::Running:
            return "running";
        case ThreadState::Sleeping:
            return "sleeping";
        case ThreadState::Blocked:
            return "blocked";
        case ThreadState::Suspended:
            return "suspended";
        case ThreadState::Exited:
            return "exited";
    }
    return "?";
}

std::string_view block_reason_name(BlockReason reason) noexcept {
    switch (reason) {
        case BlockReason::None:
            return "-";
        case BlockReason::Mutex:
            return "mutex";
        case BlockReason::Event:
            return "event";
        case BlockReason::Semaphore:
            return "semaphore";
        case BlockReason::Condition:
            return "condition";
        case BlockReason::Rendezvous:
            return "rendezvous";
        case BlockReason::MessageQueue:
            return "message queue";
        case BlockReason::Join:
            return "join";
        case BlockReason::Sleep:
            return "sleep";
    }
    return "?";
}

void Scheduler::reset() {
    threads_.clear();
    timers_.clear();
    current_ = 0;
    switches_ = 0;
    skipped_ticks_ = 0;
    next_block_sequence_ = 1;
    servicing_timers_ = false;
}

u32 Scheduler::create(u32 handle,
                      std::string name,
                      ppcaddr_t entry,
                      u32 argc,
                      ppcaddr_t argv,
                      ppcaddr_t stack_top,
                      u32 stack_size,
                      s32 priority,
                      u32 affinity) {
    Thread thread;
    thread.handle = handle;
    thread.name = name.empty() ? fmt::format("thread_{:#x}", handle) : std::move(name);
    thread.priority = std::clamp(priority, highest_priority, lowest_priority);
    thread.affinity = affinity == 0 ? 0x7U : affinity;

    // Which core this actually runs on. A thread pinned to one core has no
    // choice; a thread that may run anywhere is spread round, because putting
    // every "any core" thread on core 0 makes anything that coordinates
    // *between* cores -- a rendezvous, most obviously -- wait for a core that
    // is never going to arrive.
    thread.core = next_core_ % cpu::core_count;
    if ((thread.affinity & (1U << thread.core)) == 0) {
        for (u32 core = 0; core < cpu::core_count; ++core) {
            if ((thread.affinity & (1U << core)) != 0) {
                thread.core = core;
                break;
            }
        }
    }
    ++next_core_;

    thread.context.reset();
    thread.context.pc = entry;
    thread.context.next_pc = entry;
    thread.context.gpr[3] = argc;
    thread.context.gpr[4] = argv;
    thread.context.core_id = thread.core;

    // The ABI wants an 8-byte-aligned stack with a null back-chain, so walking
    // the frames terminates rather than wandering off.
    thread.context.gpr[1] = (stack_top - 8) & ~7U;

    // Returning from the entry point lands on the sentinel, which the run loop
    // turns into an exit.
    thread.context.lr = thread_exit_sentinel;

    static_cast<void>(stack_size);

    // Created suspended: OSCreateThread does not start a thread, OSResumeThread
    // does. Starting it here would run threads a title expected to set up
    // first.
    thread.state = ThreadState::Suspended;
    thread.suspend_count = 1;

    // Re-creating the handle that was running means the old thread is gone,
    // whatever the caller thought. Same reasoning as `remove`.
    if (current_ == handle) {
        current_ = 0;
    }

    threads_[handle] = std::move(thread);
    return handle;
}

u32 Scheduler::create_callback(u32 handle,
                               std::string name,
                               ppcaddr_t entry,
                               std::span<const u32> arguments,
                               ppcaddr_t stack_top,
                               s32 priority) {
    static_cast<void>(create(handle, std::move(name), entry, 0, 0, stack_top, 0, priority, 0x7));

    Thread* const thread = find(handle);
    if (thread == nullptr) {
        return 0;
    }

    // The ABI's first eight argument registers. More than eight would go on
    // the stack, and no OS callback takes that many.
    const std::size_t count = std::min<std::size_t>(arguments.size(), 8);
    for (std::size_t index = 0; index < count; ++index) {
        thread->context.gpr[3 + index] = arguments[index];
    }

    // A callback runs now; it is not something the title resumes.
    thread->state = ThreadState::Ready;
    thread->suspend_count = 0;
    return handle;
}

bool Scheduler::remove(u32 handle) {
    const auto it = threads_.find(handle);
    if (it == threads_.end() || it->second.state != ThreadState::Exited) {
        return false;
    }
    threads_.erase(it);

    // The core is not running it any more -- it does not exist. Leaving
    // `current_` pointing at the handle means the next thread created under
    // it inherits "is already running", and `reschedule` then declines to
    // restore its context: the new thread starts with the old one's registers
    // and exits without executing an instruction.
    if (current_ == handle) {
        current_ = 0;
    }
    return true;
}

u32 Scheduler::adopt_main(const cpu::CoreState& state, u32 handle) {
    Thread thread;
    thread.handle = handle;
    thread.name = "main";
    thread.context = state;
    thread.state = ThreadState::Running;
    thread.suspend_count = 0;
    thread.priority = default_priority;
    thread.core = state.core_id;

    threads_[handle] = std::move(thread);
    current_ = handle;
    return handle;
}

bool Scheduler::resume(u32 handle) {
    Thread* const thread = find(handle);
    if (thread == nullptr || thread->state == ThreadState::Exited) {
        return false;
    }
    if (thread->suspend_count > 0) {
        --thread->suspend_count;
    }
    if (thread->suspend_count == 0 && thread->state == ThreadState::Suspended) {
        thread->state = ThreadState::Ready;
    }
    return true;
}

bool Scheduler::suspend(u32 handle) {
    Thread* const thread = find(handle);
    if (thread == nullptr || thread->state == ThreadState::Exited) {
        return false;
    }
    ++thread->suspend_count;
    if (thread->state == ThreadState::Ready || thread->state == ThreadState::Running) {
        thread->state = ThreadState::Suspended;
    }
    return true;
}

Thread* Scheduler::current() noexcept {
    return find(current_);
}

const Thread* Scheduler::current() const noexcept {
    const auto it = threads_.find(current_);
    return it == threads_.end() ? nullptr : &it->second;
}

Thread* Scheduler::find(u32 handle) noexcept {
    const auto it = threads_.find(handle);
    return it == threads_.end() ? nullptr : &it->second;
}

void Scheduler::save_current(const cpu::CoreState& state) {
    Thread* const thread = current();
    if (thread != nullptr) {
        thread->context = state;
    }
}

void Scheduler::restore(Thread& thread, cpu::CoreState& state) {
    // The timebase belongs to the machine, not to a thread. It is in
    // `CoreState` because the interpreter counts it there, but it is a clock
    // that only ever goes forward, and a whole-struct copy would hand the core
    // whatever the incoming thread last saw -- zero, for one that has not run
    // yet. Every deadline in this file is measured against it, so a switch
    // that rewound it fired alarms and asynchronous completions immediately,
    // or never.
    const u64 now = state.timebase;
    state = thread.context;
    state.timebase = now;

    thread.state = ThreadState::Running;
    current_ = thread.handle;
}

Thread* Scheduler::pick_next() {
    // Highest priority wins; among equals, take turns.
    //
    // "Prefer anything that is not the current thread" is not enough: with
    // three equal-priority threads it alternates between two of them and
    // starves the third. Proper round-robin means scanning from just after the
    // current thread and wrapping, so every peer is reached in order.
    s32 best_priority = lowest_priority + 1;
    for (const auto& [handle, thread] : threads_) {
        if (thread.runnable()) {
            best_priority = std::min(best_priority, thread.priority);
        }
    }
    if (best_priority > lowest_priority) {
        return nullptr;
    }

    Thread* first = nullptr;
    for (auto& [handle, thread] : threads_) {
        if (!thread.runnable() || thread.priority != best_priority) {
            continue;
        }
        if (first == nullptr) {
            first = &thread;
        }
        if (handle > current_) {
            return &thread; // the next one round
        }
    }
    return first; // wrapped
}

void Scheduler::wake_sleepers(u64 now) {
    for (auto& [handle, thread] : threads_) {
        if (thread.state == ThreadState::Sleeping && now >= thread.wake_time) {
            thread.state = ThreadState::Ready;
            thread.block_reason = BlockReason::None;
        }
    }
}

void Scheduler::set_timer(u32 id, u64 deadline, TimerCallback callback) {
    timers_[id] = Timer{deadline, std::move(callback)};
}

void Scheduler::clear_timer(u32 id) noexcept {
    timers_.erase(id);
}

bool Scheduler::has_timer(u32 id) const noexcept {
    return timers_.contains(id);
}

u32 Scheduler::service_timers(u64 now) {
    // A callback can arm timers, create threads, or cause a reschedule that
    // would land back here. Re-entering would fire the same deadline twice.
    if (servicing_timers_) {
        return 0;
    }

    // A guard rather than a pair of assignments: a callback that throws --
    // and they allocate, so they can -- would otherwise leave the flag set
    // and no timer would ever fire again.
    struct Guard {
        bool& flag;

        ~Guard() { flag = false; }
    } const guard{servicing_timers_};

    servicing_timers_ = true;

    u32 fired = 0;
    while (true) {
        // Earliest deadline first, one at a time: the map is being modified by
        // the callbacks, so nothing may be held across one.
        u32 due_id = 0;
        u64 due_deadline = 0;
        bool found = false;
        for (const auto& [id, timer] : timers_) {
            if (timer.deadline <= now && (!found || timer.deadline < due_deadline)) {
                due_id = id;
                due_deadline = timer.deadline;
                found = true;
            }
        }
        if (!found) {
            break;
        }

        const auto it = timers_.find(due_id);
        TimerCallback callback = std::move(it->second.callback);
        timers_.erase(it);
        ++fired;

        if (callback) {
            callback(now);
        }
    }

    return fired;
}

std::optional<u64> Scheduler::next_deadline() const noexcept {
    std::optional<u64> earliest;
    const auto consider = [&earliest](u64 deadline) {
        if (!earliest.has_value() || deadline < *earliest) {
            earliest = deadline;
        }
    };

    for (const auto& [handle, thread] : threads_) {
        if (thread.state == ThreadState::Sleeping) {
            consider(thread.wake_time);
        }
    }
    for (const auto& [id, timer] : timers_) {
        consider(timer.deadline);
    }
    return earliest;
}

u64 Scheduler::idle_skip_ticks(const cpu::CoreState& state) const noexcept {
    const bool anything_runnable =
        std::ranges::any_of(threads_, [](const auto& entry) { return entry.second.runnable(); });
    if (anything_runnable) {
        return 0;
    }
    const Thread* const running = current();
    if (running != nullptr && running->state == ThreadState::Running) {
        return 0;
    }

    const std::optional<u64> deadline = next_deadline();
    if (!deadline.has_value()) {
        return 0; // nothing is waiting for time: this is a deadlock, not a wait
    }
    return *deadline > state.timebase ? *deadline - state.timebase : 0;
}

bool Scheduler::reschedule(cpu::CoreState& state) {
    static_cast<void>(service_timers(state.timebase));
    wake_sleepers(state.timebase);

    Thread* const running = current();
    if (running != nullptr && running->state == ThreadState::Running) {
        // Put it back in the queue before choosing, so an equal-priority peer
        // gets a turn.
        running->state = ThreadState::Ready;
    }

    Thread* const next = pick_next();
    if (next == nullptr) {
        return false;
    }

    if (next->handle == current_) {
        next->state = ThreadState::Running;
        return false;
    }

    save_current(state);
    restore(*next, state);
    ++switches_;

    WIIU_LOG_TRACE(
        Coreinit, "switch to '{}' ({:#x}) at {:#010x}", next->name, next->handle, state.pc);
    return true;
}

bool Scheduler::yield(cpu::CoreState& state) {
    return reschedule(state);
}

void Scheduler::sleep(cpu::CoreState& state, u64 ticks) {
    Thread* const thread = current();
    if (thread == nullptr) {
        return;
    }
    thread->wake_time = state.timebase + ticks;
    thread->state = ThreadState::Sleeping;
    thread->block_reason = BlockReason::Sleep;
    static_cast<void>(reschedule(state));
}

void Scheduler::block(cpu::CoreState& state, BlockReason reason, u32 object) {
    Thread* const thread = current();
    if (thread == nullptr) {
        // Nothing to block. This means the caller never registered a main
        // thread, and silently continuing turns every blocking call into a
        // no-op -- a program that joins a worker would sail past it and read
        // an exit value that was never written.
        WIIU_LOG_ERROR(Coreinit,
                       "blocking call with no current thread; "
                       "was adopt_main() called?");
        return;
    }
    thread->state = ThreadState::Blocked;
    thread->block_reason = reason;
    thread->blocked_on = object;
    thread->block_sequence = next_block_sequence_++;
    static_cast<void>(reschedule(state));
}

bool Scheduler::waits_before(const Thread& left, const Thread& right) noexcept {
    // Priority first, and within a priority whoever has been waiting longest.
    // Cafe OS thread queues are FIFO within a level, and a title that sends two
    // messages through two equal-priority threads gets them back in the order
    // it sent them -- not in the order its allocator handed out OSThreads.
    if (left.priority != right.priority) {
        return left.priority < right.priority;
    }
    return left.block_sequence < right.block_sequence;
}

void Scheduler::unblock(u32 handle) {
    Thread* const thread = find(handle);
    if (thread != nullptr && thread->state == ThreadState::Blocked) {
        thread->state = ThreadState::Ready;
        thread->block_reason = BlockReason::None;
        thread->blocked_on = 0;
    }
}

u32 Scheduler::unblock_all(u32 object) {
    u32 woken = 0;
    for (auto& [handle, thread] : threads_) {
        if (thread.state == ThreadState::Blocked && thread.blocked_on == object) {
            thread.state = ThreadState::Ready;
            thread.block_reason = BlockReason::None;
            thread.blocked_on = 0;
            ++woken;
        }
    }
    return woken;
}

u32 Scheduler::unblock_one(u32 object) {
    // Highest priority waiter wins, which is what a priority-scheduled OS
    // does and what stops a low-priority thread holding up the queue.
    Thread* best = nullptr;
    for (auto& [handle, thread] : threads_) {
        if (thread.state != ThreadState::Blocked || thread.blocked_on != object) {
            continue;
        }
        if (best == nullptr || waits_before(thread, *best)) {
            best = &thread;
        }
    }
    if (best == nullptr) {
        return 0;
    }
    best->state = ThreadState::Ready;
    best->block_reason = BlockReason::None;
    best->blocked_on = 0;
    return best->handle;
}

std::vector<u32> Scheduler::waiters(u32 object) const {
    std::vector<const Thread*> found;
    for (const auto& [handle, thread] : threads_) {
        if (thread.state == ThreadState::Blocked && thread.blocked_on == object) {
            found.push_back(&thread);
        }
    }
    std::ranges::sort(
        found, [](const Thread* lhs, const Thread* rhs) { return waits_before(*lhs, *rhs); });

    std::vector<u32> handles;
    handles.reserve(found.size());
    for (const Thread* thread : found) {
        handles.push_back(thread->handle);
    }
    return handles;
}

void Scheduler::block_thread(u32 handle, BlockReason reason, u32 object) {
    Thread* const thread = find(handle);
    if (thread == nullptr || thread->state == ThreadState::Exited) {
        return;
    }
    thread->state = ThreadState::Blocked;
    thread->block_reason = reason;
    thread->blocked_on = object;
    thread->block_sequence = next_block_sequence_++;
}

void Scheduler::exit_current(cpu::CoreState& state, s32 value) {
    Thread* const thread = current();
    if (thread == nullptr) {
        // Nothing is running: the core is sitting on a sentinel with no
        // thread behind it. Something else may still be runnable, and
        // returning without looking leaves the caller to meet the same
        // sentinel again, and again.
        static_cast<void>(reschedule(state));
        return;
    }

    thread->state = ThreadState::Exited;
    thread->exit_value = value;

    for (const auto& [joiner, out_address] : thread->joiners) {
        if (on_exit_ && out_address != 0) {
            on_exit_(value, out_address);
        }
        unblock(joiner);
    }
    thread->joiners.clear();

    WIIU_LOG_DEBUG(Coreinit, "thread '{}' exited with {}", thread->name, value);
    static_cast<void>(reschedule(state));
}

bool Scheduler::join(cpu::CoreState& state, u32 handle, ppcaddr_t out_address, s32& value) {
    Thread* const target = find(handle);
    if (target == nullptr) {
        value = 0;
        return false;
    }
    if (target->state == ThreadState::Exited) {
        value = target->exit_value;
        return false;
    }

    target->joiners.emplace_back(current_, out_address);
    block(state, BlockReason::Join, handle);
    return true;
}

std::vector<const Thread*> Scheduler::threads() const {
    std::vector<const Thread*> out;
    out.reserve(threads_.size());
    for (const auto& [handle, thread] : threads_) {
        out.push_back(&thread);
    }
    return out;
}

std::size_t Scheduler::runnable_count() const noexcept {
    return static_cast<std::size_t>(
        std::ranges::count_if(threads_, [](const auto& entry) { return entry.second.runnable(); }));
}

bool Scheduler::all_exited() const noexcept {
    return std::ranges::all_of(
        threads_, [](const auto& entry) { return entry.second.state == ThreadState::Exited; });
}

bool Scheduler::stalled() const noexcept {
    if (threads_.empty()) {
        return false;
    }
    const bool anything_runnable =
        std::ranges::any_of(threads_, [](const auto& entry) { return entry.second.runnable(); });
    if (anything_runnable) {
        return false;
    }
    const Thread* const running = current();
    if (running != nullptr && running->state == ThreadState::Running) {
        return false;
    }
    return !all_exited();
}

u64 Scheduler::hash(u64 seed) const noexcept {
    // std::map iterates in handle order, so the hash does not depend on the
    // order threads happened to be created in.
    u64 value = seed;
    const auto fold = [&value](u64 part) {
        for (u32 i = 0; i < 8; ++i) {
            value ^= (part >> (i * 8)) & 0xFF;
            value *= 0x100000001B3ULL;
        }
    };

    fold(current_);

    // Pending deadlines are part of the machine: two runs that disagree about
    // when an alarm fires will diverge later, when it does.
    for (const auto& [id, timer] : timers_) {
        fold(id);
        fold(timer.deadline);
    }

    for (const auto& [handle, thread] : threads_) {
        fold(handle);
        fold(static_cast<u64>(thread.state));
        fold(static_cast<u64>(thread.block_reason));
        fold(thread.blocked_on);
        fold(static_cast<u64>(static_cast<u32>(thread.priority)));
        fold(thread.affinity);
        fold(thread.core);
        fold(static_cast<u64>(static_cast<u32>(thread.exit_value)));
        fold(static_cast<u64>(static_cast<u32>(thread.suspend_count)));
        fold(thread.wake_time);
        fold(thread.stack_base);
        fold(thread.joiners.size());

        // The current thread's saved context is stale -- the live one is in
        // the core the caller is hashing -- so folding it in would be hashing
        // whatever was there at the last switch.
        if (handle != current_) {
            fold(Recording::hash_state(thread.context));
        }
    }
    return value;
}

} // namespace wiiu::hle
