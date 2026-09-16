// Phase 6.4 / 6.5 — threads, mutexes, events and semaphores.
//
// Split from coreinit.cpp because these have a rule the others do not: a
// handler that blocks switches the context out from under itself. Anything
// that can block therefore sets its return value *before* blocking and is
// registered as a raw handler rather than a wrapped function, since the
// wrapper writes the return value after the call -- by which time `state` may
// belong to a different thread.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>

#include <algorithm>

namespace wiiu::hle::coreinit {
namespace {

using wiiu::log::Channel;

[[nodiscard]] Scheduler& scheduler() {
    return detail::active_context()->scheduler;
}

[[nodiscard]] SyncObjects& sync() {
    return detail::active_context()->sync;
}

// --- threads ---------------------------------------------------------------

void os_create_thread(Call& call) {
    const u32 thread = call.next_word();
    const u32 entry = call.next_word();
    const u32 argc = call.next_word();
    const u32 argv = call.next_word();
    const u32 stack_top = call.next_word();
    const u32 stack_size = call.next_word();
    const auto priority = static_cast<s32>(call.next_word());
    const u32 attributes = call.next_word();

    // The low three bits of the attribute word are the core affinity mask.
    const u32 affinity = (attributes & 0x7) == 0 ? 0x7 : (attributes & 0x7);

    static_cast<void>(scheduler().create(
        thread, {}, entry, argc, argv, stack_top, stack_size, priority, affinity));
    call.state.gpr[3] = 1; // TRUE
}

void os_resume_thread(Call& call) {
    const u32 thread = call.next_word();
    const Thread* const existing = scheduler().find(thread);
    const u32 previous = existing == nullptr ? 0 : existing->suspend_count;
    static_cast<void>(scheduler().resume(thread));
    call.state.gpr[3] = previous;
}

void os_suspend_thread(Call& call) {
    const u32 thread = call.next_word();
    const Thread* const existing = scheduler().find(thread);
    const u32 previous = existing == nullptr ? 0 : existing->suspend_count;
    static_cast<void>(scheduler().suspend(thread));
    call.state.gpr[3] = previous;

    // Suspending the running thread has to take effect immediately.
    if (scheduler().current_handle() == thread) {
        static_cast<void>(scheduler().reschedule(call.state));
    }
}

void os_exit_thread(Call& call) {
    const auto value = static_cast<s32>(call.next_word());
    scheduler().exit_current(call.state, value);
}

void os_join_thread(Call& call) {
    const u32 thread = call.next_word();
    const u32 out_address = call.next_word();

    // Set before join, because join may block -- and blocking replaces
    // `state` with another thread's, so a value written afterwards would land
    // in the wrong thread's r3.
    call.state.gpr[3] = 1;

    s32 value = 0;
    const bool blocked = scheduler().join(call.state, thread, out_address, value);

    if (!blocked) {
        // Already finished: write the exit value and carry straight on.
        if (out_address != 0) {
            static_cast<void>(call.memory.try_write<u32>(out_address, static_cast<u32>(value)));
        }
        return;
    }

    // Blocked. The exit value is written through the scheduler's exit callback
    // when the target finishes.
}

void os_yield_thread(Call& call) {
    static_cast<void>(scheduler().yield(call.state));
}

void os_sleep_ticks(Call& call) {
    const u32 ticks = call.next_word();
    scheduler().sleep(call.state, ticks);
}

void os_get_current_thread(Call& call) {
    call.state.gpr[3] = scheduler().current_handle();
}

void os_is_thread_terminated(Call& call) {
    const u32 thread = call.next_word();
    const Thread* const existing = scheduler().find(thread);
    call.state.gpr[3] = (existing != nullptr && existing->state == ThreadState::Exited) ? 1U : 0U;
}

void os_get_thread_priority(Call& call) {
    const u32 thread = call.next_word();
    const Thread* const existing = scheduler().find(thread);
    call.state.gpr[3] =
        static_cast<u32>(existing == nullptr ? default_priority : existing->priority);
}

void os_set_thread_priority(Call& call) {
    const u32 thread = call.next_word();
    const auto priority = static_cast<s32>(call.next_word());
    Thread* const existing = scheduler().find(thread);
    if (existing != nullptr) {
        existing->priority = std::clamp(priority, highest_priority, lowest_priority);
    }
    call.state.gpr[3] = existing != nullptr ? 1U : 0U;
}

// --- mutexes ---------------------------------------------------------------

void os_init_mutex(Call& call) {
    const u32 mutex = call.next_word();
    sync().mutexes[mutex] = Mutex{};
}

void os_lock_mutex(Call& call) {
    const u32 handle = call.next_word();
    Mutex& mutex = sync().mutexes[handle];
    const u32 self = scheduler().current_handle();

    if (mutex.owner == 0 || mutex.owner == self) {
        // Recursive for the owner, as OSMutex is.
        mutex.owner = self;
        ++mutex.recursion;
        return;
    }

    scheduler().block(call.state, BlockReason::Mutex, handle);
}

void os_try_lock_mutex(Call& call) {
    const u32 handle = call.next_word();
    Mutex& mutex = sync().mutexes[handle];
    const u32 self = scheduler().current_handle();

    if (mutex.owner == 0 || mutex.owner == self) {
        mutex.owner = self;
        ++mutex.recursion;
        call.state.gpr[3] = 1;
        return;
    }
    call.state.gpr[3] = 0;
}

void os_unlock_mutex(Call& call) {
    const u32 handle = call.next_word();
    Mutex& mutex = sync().mutexes[handle];

    if (mutex.recursion > 0) {
        --mutex.recursion;
    }
    if (mutex.recursion != 0) {
        return;
    }

    mutex.owner = 0;
    // Hand it to the highest-priority waiter, which then acquires it on the
    // way out of its own OSLockMutex.
    if (scheduler().unblock_one(handle)) {
        mutex.owner = 0; // the woken thread re-runs its lock attempt
    }
}

// --- events ----------------------------------------------------------------

void os_init_event(Call& call) {
    const u32 handle = call.next_word();
    const u32 initial = call.next_word();
    const u32 mode = call.next_word();

    Event event;
    event.signalled = initial != 0;
    event.auto_reset = mode == 0; // OS_EVENT_MODE_AUTO is 0
    sync().events[handle] = event;
}

void os_wait_event(Call& call) {
    const u32 handle = call.next_word();
    Event& event = sync().events[handle];

    if (event.signalled) {
        if (event.auto_reset) {
            event.signalled = false;
        }
        return;
    }
    scheduler().block(call.state, BlockReason::Event, handle);
}

void os_signal_event(Call& call) {
    const u32 handle = call.next_word();
    Event& event = sync().events[handle];

    if (event.auto_reset) {
        // Auto-reset releases exactly one waiter, and only stays signalled if
        // nobody was waiting.
        if (!scheduler().unblock_one(handle)) {
            event.signalled = true;
        }
        return;
    }

    event.signalled = true;
    static_cast<void>(scheduler().unblock_all(handle));
}

void os_reset_event(Call& call) {
    const u32 handle = call.next_word();
    sync().events[handle].signalled = false;
}

// --- semaphores ------------------------------------------------------------

void os_init_semaphore(Call& call) {
    const u32 handle = call.next_word();
    const auto count = static_cast<s32>(call.next_word());
    sync().semaphores[handle] = Semaphore{count};
}

void os_wait_semaphore(Call& call) {
    const u32 handle = call.next_word();
    Semaphore& semaphore = sync().semaphores[handle];

    if (semaphore.count > 0) {
        --semaphore.count;
        call.state.gpr[3] = static_cast<u32>(semaphore.count);
        return;
    }
    call.state.gpr[3] = 0;
    scheduler().block(call.state, BlockReason::Semaphore, handle);
}

void os_try_wait_semaphore(Call& call) {
    const u32 handle = call.next_word();
    Semaphore& semaphore = sync().semaphores[handle];

    if (semaphore.count > 0) {
        --semaphore.count;
        call.state.gpr[3] = static_cast<u32>(semaphore.count);
        return;
    }
    call.state.gpr[3] = static_cast<u32>(-1);
}

void os_signal_semaphore(Call& call) {
    const u32 handle = call.next_word();
    Semaphore& semaphore = sync().semaphores[handle];

    const s32 previous = semaphore.count;
    if (!scheduler().unblock_one(handle)) {
        ++semaphore.count;
    }
    call.state.gpr[3] = static_cast<u32>(previous);
}

} // namespace

void register_threads(Registry& registry, Context& context) {
    // register_all does this too; doing it here as well means this function
    // also works on its own, which the tests rely on.
    detail::set_active_context(&context);

    context.scheduler.set_exit_callback(nullptr);

    registry.add("coreinit", "OSCreateThread", os_create_thread);
    registry.add("coreinit", "OSResumeThread", os_resume_thread);
    registry.add("coreinit", "OSSuspendThread", os_suspend_thread);
    registry.add("coreinit", "OSExitThread", os_exit_thread);
    registry.add("coreinit", "OSJoinThread", os_join_thread);
    registry.add("coreinit", "OSYieldThread", os_yield_thread);
    registry.add("coreinit", "OSSleepTicks", os_sleep_ticks);
    registry.add("coreinit", "OSGetCurrentThread", os_get_current_thread);
    registry.add("coreinit", "OSIsThreadTerminated", os_is_thread_terminated);
    registry.add("coreinit", "OSGetThreadPriority", os_get_thread_priority);
    registry.add("coreinit", "OSSetThreadPriority", os_set_thread_priority);

    registry.add("coreinit", "OSInitMutex", os_init_mutex);
    registry.add("coreinit", "OSInitMutexEx", os_init_mutex);
    registry.add("coreinit", "OSLockMutex", os_lock_mutex);
    registry.add("coreinit", "OSTryLockMutex", os_try_lock_mutex);
    registry.add("coreinit", "OSUnlockMutex", os_unlock_mutex);

    registry.add("coreinit", "OSInitEvent", os_init_event);
    registry.add("coreinit", "OSInitEventEx", os_init_event);
    registry.add("coreinit", "OSWaitEvent", os_wait_event);
    registry.add("coreinit", "OSSignalEvent", os_signal_event);
    registry.add("coreinit", "OSSignalEventAll", os_signal_event);
    registry.add("coreinit", "OSResetEvent", os_reset_event);

    registry.add("coreinit", "OSInitSemaphore", os_init_semaphore);
    registry.add("coreinit", "OSInitSemaphoreEx", os_init_semaphore);
    registry.add("coreinit", "OSWaitSemaphore", os_wait_semaphore);
    registry.add("coreinit", "OSTryWaitSemaphore", os_try_wait_semaphore);
    registry.add("coreinit", "OSSignalSemaphore", os_signal_semaphore);
}

} // namespace wiiu::hle::coreinit
