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
#include <functional>

namespace wiiu::hle::coreinit {
namespace {

using wiiu::log::Channel;

[[nodiscard]] Scheduler& scheduler() {
    return detail::active_context()->scheduler;
}

[[nodiscard]] SyncObjects& sync() {
    return detail::active_context()->sync;
}

/// The core the calling thread is on, for the rendezvous.
[[nodiscard]] u32 current_core() {
    const Thread* const thread = scheduler().current();
    return thread == nullptr ? 0 : thread->core;
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

    // Hand it to the highest-priority waiter *as the owner*. A woken thread
    // resumes after its OSLockMutex -- the return address was set before the
    // handler blocked it -- so it never gets another chance to take the lock.
    // Leaving the mutex unowned here put two threads inside the same critical
    // section, with nothing to show for it until their shared state was
    // corrupt.
    const u32 woken = scheduler().unblock_one(handle);
    mutex.owner = woken;
    mutex.recursion = woken != 0 ? 1 : 0;
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

void os_signal_event_all(Call& call) {
    const u32 handle = call.next_word();
    Event& event = sync().events[handle];

    // Unlike OSSignalEvent, this releases every waiter even on an auto-reset
    // event -- which is the whole reason it exists as a separate function.
    const u32 woken = scheduler().unblock_all(handle);
    if (!event.auto_reset) {
        event.signalled = true;
    } else if (woken == 0) {
        event.signalled = true;
    }
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

// --- condition variables ---------------------------------------------------

void os_init_cond(Call& call) {
    const u32 handle = call.next_word();
    sync().conditions[handle] = Condition{};
}

void os_wait_cond(Call& call) {
    const u32 handle = call.next_word();
    const u32 mutex_handle = call.next_word();

    // The contract: release the mutex, block, and hold it again on return.
    // Releasing it here rather than on the way out is the whole point --
    // whoever is going to signal the condition needs it.
    Condition& condition = sync().conditions[handle];
    condition.mutex = mutex_handle;

    Mutex& mutex = sync().mutexes[mutex_handle];
    const u32 next_owner = scheduler().unblock_one(mutex_handle);
    mutex.owner = next_owner;
    mutex.recursion = next_owner != 0 ? 1 : 0;

    scheduler().block(call.state, BlockReason::Condition, handle);
}

void os_signal_cond(Call& call) {
    const u32 handle = call.next_word();
    Condition& condition = sync().conditions[handle];

    // OSSignalCond releases every waiter. They all have to hold the mutex
    // again before they can run, though -- that is the contract that makes a
    // condition variable usable -- and only one of them can. So the
    // highest-priority waiter becomes the owner, and the rest simply move to
    // waiting on the mutex, which is the queue they would have joined by
    // calling OSLockMutex themselves.
    const std::vector<u32> waiting = scheduler().waiters(handle);
    if (waiting.empty()) {
        return;
    }

    if (condition.mutex == 0) {
        static_cast<void>(scheduler().unblock_all(handle));
        return;
    }

    Mutex& mutex = sync().mutexes[condition.mutex];
    bool handed_over = false;

    for (const u32 thread : waiting) {
        if (!handed_over && mutex.owner == 0) {
            scheduler().unblock(thread);
            mutex.owner = thread;
            mutex.recursion = 1;
            handed_over = true;
            continue;
        }
        scheduler().block_thread(thread, BlockReason::Mutex, condition.mutex);
    }
}

// --- fast mutexes ----------------------------------------------------------

void os_init_fast_mutex(Call& call) {
    const u32 handle = call.next_word();
    sync().fast_mutexes[handle] = FastMutex{};
}

void os_lock_fast_mutex(Call& call) {
    const u32 handle = call.next_word();
    FastMutex& mutex = sync().fast_mutexes[handle];
    const u32 self = scheduler().current_handle();

    if (mutex.owner == 0) {
        mutex.owner = self;
        return;
    }

    // Not recursive, unlike OSMutex: locking one twice on hardware deadlocks,
    // and pretending otherwise would hide the bug rather than the deadlock.
    scheduler().block(call.state, BlockReason::Mutex, handle);
}

void os_try_lock_fast_mutex(Call& call) {
    const u32 handle = call.next_word();
    FastMutex& mutex = sync().fast_mutexes[handle];

    if (mutex.owner == 0) {
        mutex.owner = scheduler().current_handle();
        call.state.gpr[3] = 1;
        return;
    }
    call.state.gpr[3] = 0;
}

void os_unlock_fast_mutex(Call& call) {
    const u32 handle = call.next_word();
    FastMutex& mutex = sync().fast_mutexes[handle];

    // As with OSMutex: the waiter this wakes is the new owner, because it
    // resumes past its own lock call rather than retrying it.
    mutex.owner = scheduler().unblock_one(handle);
}

// --- rendezvous ------------------------------------------------------------

void os_init_rendezvous(Call& call) {
    const u32 handle = call.next_word();
    sync().rendezvous[handle] = Rendezvous{};
}

void os_wait_rendezvous(Call& call) {
    const u32 handle = call.next_word();
    const u32 mask = call.next_word();

    Rendezvous& barrier = sync().rendezvous[handle];
    barrier.arrived |= 1U << current_core();

    if ((barrier.arrived & mask) == mask) {
        // Everyone is here. Reset it so the same rendezvous can be used again,
        // which is what a title does with one per frame.
        barrier.arrived = 0;
        call.state.gpr[3] = 1;
        static_cast<void>(scheduler().unblock_all(handle));
        return;
    }

    call.state.gpr[3] = 1;
    scheduler().block(call.state, BlockReason::Rendezvous, handle);
}

// --- atomics ---------------------------------------------------------------
//
// These are ordinary reads and writes here, and that is correct rather than a
// shortcut: guest threads are interleaved on one host thread and a handler
// runs to completion, so nothing can observe a torn value. When cores run in
// parallel (P32) these become the one place that has to change, which is why
// they go through a single helper.

[[nodiscard]] bool atomic_update(Call& call, u32 address, const std::function<u32(u32)>& update) {
    u32 value = 0;
    if (!call.memory.try_read<u32>(address, value)) {
        return false;
    }
    return call.memory.try_write<u32>(address, update(value));
}

void os_swap_atomic(Call& call) {
    const u32 address = call.next_word();
    const u32 value = call.next_word();

    u32 previous = 0;
    static_cast<void>(call.memory.try_read<u32>(address, previous));
    static_cast<void>(atomic_update(call, address, [value](u32) { return value; }));
    call.state.gpr[3] = previous;
}

void os_compare_and_swap_atomic(Call& call) {
    const u32 address = call.next_word();
    const u32 expected = call.next_word();
    const u32 value = call.next_word();

    u32 current = 0;
    static_cast<void>(call.memory.try_read<u32>(address, current));
    const bool matched = current == expected;
    if (matched) {
        static_cast<void>(call.memory.try_write<u32>(address, value));
    }
    call.state.gpr[3] = matched ? 1U : 0U;
}

/// The Ex form also reports what it found, which is how lock-free code avoids
/// a second read on failure.
void os_compare_and_swap_atomic_ex(Call& call) {
    const u32 address = call.next_word();
    const u32 expected = call.next_word();
    const u32 value = call.next_word();
    const u32 out_address = call.next_word();

    u32 current = 0;
    static_cast<void>(call.memory.try_read<u32>(address, current));
    const bool matched = current == expected;
    if (matched) {
        static_cast<void>(call.memory.try_write<u32>(address, value));
    }
    if (out_address != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_address, current));
    }
    call.state.gpr[3] = matched ? 1U : 0U;
}

void os_add_atomic(Call& call) {
    const u32 address = call.next_word();
    const u32 amount = call.next_word();

    u32 previous = 0;
    static_cast<void>(call.memory.try_read<u32>(address, previous));
    static_cast<void>(atomic_update(call, address, [amount](u32 value) { return value + amount; }));
    call.state.gpr[3] = previous + amount; // OSAddAtomic returns the new value
}

void os_and_atomic(Call& call) {
    const u32 address = call.next_word();
    const u32 mask = call.next_word();

    u32 previous = 0;
    static_cast<void>(call.memory.try_read<u32>(address, previous));
    static_cast<void>(atomic_update(call, address, [mask](u32 value) { return value & mask; }));
    call.state.gpr[3] = previous;
}

void os_or_atomic(Call& call) {
    const u32 address = call.next_word();
    const u32 mask = call.next_word();

    u32 previous = 0;
    static_cast<void>(call.memory.try_read<u32>(address, previous));
    static_cast<void>(atomic_update(call, address, [mask](u32 value) { return value | mask; }));
    call.state.gpr[3] = previous;
}

} // namespace

u32 invoke_guest_callback(Context& context,
                          memory::Memory& memory,
                          std::string name,
                          ppcaddr_t entry,
                          std::span<const u32> arguments,
                          s32 priority) {
    if (entry == 0) {
        return 0;
    }

    reap_callback_threads(context);

    ppcaddr_t stack = 0;
    u32 handle = 0;
    if (!context.free_callback_stacks.empty()) {
        // Handles are recycled with the stacks they belong to. Handing out a
        // fresh one every time walks up through the main thread's handle and
        // then into the addresses real OSThread pointers have -- about
        // eighteen minutes of a 60 Hz alarm away.
        const CallbackSlot slot = context.free_callback_stacks.back();
        context.free_callback_stacks.pop_back();
        stack = slot.stack;
        handle = slot.handle;
    } else {
        ExpandedHeap* const heap = context.heaps.find(context.heaps.default_handle());
        if (heap == nullptr) {
            WIIU_LOG_ERROR(Coreinit, "no heap to allocate a callback stack from");
            return 0;
        }
        stack = heap->allocate(callback_stack_size, 8);
        if (stack == 0) {
            WIIU_LOG_ERROR(Coreinit, "out of memory for a callback stack");
            return 0;
        }

        if (context.next_callback_handle >= callback_handle_base + max_callback_threads) {
            WIIU_LOG_ERROR(Coreinit,
                           "{} callbacks are live at once; refusing to start another",
                           max_callback_threads);
            return 0;
        }
        handle = context.next_callback_handle++;
    }

    static_cast<void>(memory.fill(stack, callback_stack_size, 0));

    const u32 created = context.scheduler.create_callback(
        handle, std::move(name), entry, arguments, stack + callback_stack_size, priority);
    if (created == 0) {
        context.free_callback_stacks.push_back(CallbackSlot{stack, handle});
        return 0;
    }

    Thread* const thread = context.scheduler.find(handle);
    if (thread != nullptr) {
        // Remember the stack on the thread so reaping can give it back. r13 is
        // the small-data pointer and is not used for that here, so the stack
        // base rides in the unused frame slot below the stack pointer instead.
        thread->stack_base = stack;
    }
    return handle;
}

void reap_callback_threads(Context& context) {
    for (const Thread* thread : context.scheduler.threads()) {
        // Only the handles this hands out. The main thread's handle is above
        // the callback range and reaping *it* leaves the core with no current
        // thread at all, which the run loop meets as a sentinel it can never
        // get off.
        const bool is_callback = thread->handle >= callback_handle_base &&
                                 thread->handle < callback_handle_base + max_callback_threads;
        if (!is_callback || thread->state != ThreadState::Exited) {
            continue;
        }
        const ppcaddr_t stack = thread->stack_base;
        const u32 handle = thread->handle;
        if (context.scheduler.remove(handle) && stack != 0) {
            context.free_callback_stacks.push_back(CallbackSlot{stack, handle});
        }
    }
}

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
    registry.add("coreinit", "OSSignalEventAll", os_signal_event_all);
    registry.add("coreinit", "OSResetEvent", os_reset_event);

    registry.add("coreinit", "OSInitSemaphore", os_init_semaphore);
    registry.add("coreinit", "OSInitSemaphoreEx", os_init_semaphore);
    registry.add("coreinit", "OSWaitSemaphore", os_wait_semaphore);
    registry.add("coreinit", "OSTryWaitSemaphore", os_try_wait_semaphore);
    registry.add("coreinit", "OSSignalSemaphore", os_signal_semaphore);

    registry.add("coreinit", "OSInitCond", os_init_cond);
    registry.add("coreinit", "OSInitCondEx", os_init_cond);
    registry.add("coreinit", "OSWaitCond", os_wait_cond);
    registry.add("coreinit", "OSSignalCond", os_signal_cond);

    registry.add("coreinit", "OSFastMutex_Init", os_init_fast_mutex);
    registry.add("coreinit", "OSFastMutex_Lock", os_lock_fast_mutex);
    registry.add("coreinit", "OSFastMutex_TryLock", os_try_lock_fast_mutex);
    registry.add("coreinit", "OSFastMutex_Unlock", os_unlock_fast_mutex);

    registry.add("coreinit", "OSInitRendezvous", os_init_rendezvous);
    registry.add("coreinit", "OSWaitRendezvous", os_wait_rendezvous);

    registry.add("coreinit", "OSSwapAtomic", os_swap_atomic);
    registry.add("coreinit", "OSCompareAndSwapAtomic", os_compare_and_swap_atomic);
    registry.add("coreinit", "OSCompareAndSwapAtomicEx", os_compare_and_swap_atomic_ex);
    registry.add("coreinit", "OSAddAtomic", os_add_atomic);
    registry.add("coreinit", "OSAndAtomic", os_and_atomic);
    registry.add("coreinit", "OSOrAtomic", os_or_atomic);
}

} // namespace wiiu::hle::coreinit
