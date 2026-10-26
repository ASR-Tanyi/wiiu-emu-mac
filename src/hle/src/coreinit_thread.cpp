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
    // OSTime is 64-bit, so it arrives in r3:r4. Reading r3 alone takes the
    // high half, which for every sleep shorter than about a minute is zero.
    const u64 ticks = call.next_long();
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

// --- thread-specific values and the cleanup callback ------------------------

/// `u32 OSGetThreadSpecific(u32 id)`, of the calling thread. wut keeps its
/// newlib reentrancy structure in one, so every `printf` comes through here.
void os_get_thread_specific(Call& call) {
    const u32 id = call.next_word();
    const Thread* const thread = scheduler().current();
    if (thread == nullptr || id >= thread->specific.size()) {
        WIIU_LOG_ERROR(Coreinit, "OSGetThreadSpecific({}): no such value", id);
        call.state.gpr[3] = 0;
        return;
    }
    call.state.gpr[3] = thread->specific[id];
}

/// `void OSSetThreadSpecific(u32 id, u32 value)`.
void os_set_thread_specific(Call& call) {
    const u32 id = call.next_word();
    const u32 value = call.next_word();
    Thread* const thread = scheduler().current();
    if (thread == nullptr || id >= thread->specific.size()) {
        WIIU_LOG_ERROR(Coreinit, "OSSetThreadSpecific({}, 0x{:08X}): no such value", id, value);
        return;
    }
    thread->specific[id] = value;
}

/// `OSThreadCleanupCallbackFn OSSetThreadCleanupCallback(OSThread*, fn)`:
/// the previous one back.
void os_set_thread_cleanup_callback(Call& call) {
    const u32 handle = call.next_word();
    const u32 callback = call.next_word();
    Thread* const thread = scheduler().find(handle);
    if (thread == nullptr) {
        call.state.gpr[3] = 0;
        return;
    }
    call.state.gpr[3] = thread->cleanup_callback;
    thread->cleanup_callback = callback;
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

// --- spin locks ------------------------------------------------------------
//
// An OSSpinLock is the title's: the owning OSThread* at 0x0 and the
// recursion count at 0x8, sixteen bytes in all (decaf's layout), and they
// are kept there, where the title can see them. A thread that finds one held
// waits blocked on its address rather than spinning, since threads here take
// turns on one host thread and a spin would only burn its timeslice; the
// release hands the lock to the waiter, which resumes past its acquire as
// a mutex's does. Two things Cafe OS does are not done yet: raising the
// holder to priority 0, and -- for the uninterruptible forms -- turning
// interrupts off, which are not modelled at all.

inline constexpr u32 spin_lock_owner_offset = 0x0;
inline constexpr u32 spin_lock_recursion_offset = 0x8;

/// Take `lock` for the calling thread, if it is free or already the
/// thread's. False if another thread holds it.
[[nodiscard]] bool try_take_spin_lock(Call& call, u32 lock) {
    const u32 self = scheduler().current_handle();
    u32 owner = 0;
    static_cast<void>(call.memory.try_read<u32>(lock + spin_lock_owner_offset, owner));
    if (owner != 0 && owner == self) {
        u32 recursion = 0;
        static_cast<void>(call.memory.try_read<u32>(lock + spin_lock_recursion_offset, recursion));
        static_cast<void>(
            call.memory.try_write<u32>(lock + spin_lock_recursion_offset, recursion + 1));
        return true;
    }
    if (owner != 0) {
        return false;
    }
    static_cast<void>(call.memory.try_write<u32>(lock + spin_lock_owner_offset, self));
    return true;
}

void release_spin_lock(Call& call, u32 lock) {
    u32 recursion = 0;
    static_cast<void>(call.memory.try_read<u32>(lock + spin_lock_recursion_offset, recursion));
    if (recursion > 0) {
        static_cast<void>(
            call.memory.try_write<u32>(lock + spin_lock_recursion_offset, recursion - 1));
        return;
    }

    u32 owner = 0;
    static_cast<void>(call.memory.try_read<u32>(lock + spin_lock_owner_offset, owner));
    if (owner != scheduler().current_handle()) {
        // Cafe OS stops here (decaf aborts); the lock is left as it is.
        WIIU_LOG_ERROR(Coreinit,
                       "spin lock 0x{:08X} released by 0x{:08X}, but 0x{:08X} holds it",
                       lock,
                       scheduler().current_handle(),
                       owner);
        return;
    }
    static_cast<void>(
        call.memory.try_write<u32>(lock + spin_lock_owner_offset, scheduler().unblock_one(lock)));
}

/// `void OSInitSpinLock(OSSpinLock*)`.
void os_init_spin_lock(Call& call) {
    const u32 lock = call.next_word();
    static_cast<void>(call.memory.try_write<u32>(lock + spin_lock_owner_offset, 0));
    static_cast<void>(call.memory.try_write<u32>(lock + spin_lock_recursion_offset, 0));
}

/// `BOOL OSAcquireSpinLock(OSSpinLock*)`, and the uninterruptible form:
/// TRUE, once it is the caller's.
void os_acquire_spin_lock(Call& call) {
    const u32 lock = call.next_word();
    call.state.gpr[3] = 1; // before blocking: see the top of the file
    if (!try_take_spin_lock(call, lock)) {
        scheduler().block(call.state, BlockReason::SpinLock, lock);
    }
}

/// `BOOL OSTryAcquireSpinLock(OSSpinLock*)`, and the uninterruptible form.
void os_try_acquire_spin_lock(Call& call) {
    const u32 lock = call.next_word();
    call.state.gpr[3] = try_take_spin_lock(call, lock) ? 1U : 0U;
}

/// `BOOL OSReleaseSpinLock(OSSpinLock*)`, and the uninterruptible form.
void os_release_spin_lock(Call& call) {
    const u32 lock = call.next_word();
    release_spin_lock(call, lock);
    call.state.gpr[3] = 1;
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

// --- message queues --------------------------------------------------------
//
// The queue a title waits on for anything the system hands it: an asynchronous
// read finishing (6.9), and later a ProcUI transition (6.11).
//
// The delivery rule is the one that shapes all of this. A woken thread resumes
// *after* the call it blocked in, so a receiver parked in OSReceiveMessage
// never gets to look at the queue again -- whoever wakes it has to have put
// the message where it asked for it already. So a blocked receiver records
// where it wants its message, a blocked sender records what it is trying to
// post, and a send with a receiver waiting hands the message straight over
// rather than going through the ring at all.

/// The queue at `handle`, or null if it was never initialised.
///
/// Unlike every other object here this does not create one on demand: a queue
/// with no capacity is not an empty queue, it is a queue nothing can ever be
/// sent through, and a title that hits that has a bug worth naming.
[[nodiscard]] MessageQueue* find_queue(SyncObjects& objects, u32 handle) {
    const auto entry = objects.message_queues.find(handle);
    if (entry == objects.message_queues.end() || entry->second.capacity == 0) {
        WIIU_LOG_WARN(Coreinit, "message queue {:#010x} used before OSInitMessageQueue", handle);
        return nullptr;
    }
    return &entry->second;
}

/// Forget records for threads that are no longer parked here. The scheduler's
/// waiter list is what actually decides who is waiting; these records only say
/// what each waiter wanted. Handles are guest OSThread pointers and do get
/// reused, so a record left behind by a thread that exited while parked could
/// otherwise be handed a message meant for nobody.
template <typename Record>
void prune_pending(std::map<u32, Record>& records, const std::vector<u32>& waiting) {
    for (auto entry = records.begin(); entry != records.end();) {
        entry = std::ranges::find(waiting, entry->first) == waiting.end() ? records.erase(entry)
                                                                          : std::next(entry);
    }
}

/// Hand `message` to the highest-priority thread parked in OSReceiveMessage,
/// and say whether there was one. The message goes into the address that
/// thread asked for before it blocked.
[[nodiscard]] bool hand_to_receiver(Scheduler& threads,
                                    memory::Memory& memory,
                                    u32 handle,
                                    MessageQueue& queue,
                                    const Message& message) {
    const std::vector<u32> waiting = threads.waiters(handle + message_queue_recv_wait_offset);
    prune_pending(queue.pending_receives, waiting);

    for (const u32 thread : waiting) {
        const auto record = queue.pending_receives.find(thread);
        if (record == queue.pending_receives.end()) {
            continue;
        }
        const ppcaddr_t out_address = record->second;
        queue.pending_receives.erase(record);
        static_cast<void>(write_message(memory, out_address, message));
        threads.unblock(thread);
        return true;
    }
    return false;
}

/// Let the highest-priority thread parked in OSSendMessage finish its send,
/// now that there is room. Called after a receive takes a message out, so the
/// admitted message takes its place among the ones already queued rather than
/// in front of all of them -- except a jammed one, which asked for the head
/// and does not lose it by having had to wait for room.
void admit_pending_sender(Scheduler& threads, u32 handle, MessageQueue& queue) {
    const std::vector<u32> waiting = threads.waiters(handle + message_queue_send_wait_offset);
    prune_pending(queue.pending_sends, waiting);

    for (const u32 thread : waiting) {
        const auto record = queue.pending_sends.find(thread);
        if (record == queue.pending_sends.end()) {
            continue;
        }
        if (record->second.jam) {
            queue.entries.push_front(record->second.message);
        } else {
            queue.entries.push_back(record->second.message);
        }
        queue.pending_sends.erase(record);
        threads.unblock(thread);
        return;
    }
}

void os_init_message_queue(Call& call) {
    const u32 handle = call.next_word();
    const u32 messages = call.next_word();
    const auto size = static_cast<s32>(call.next_word());

    if (size <= 0) {
        WIIU_LOG_WARN(Coreinit,
                      "OSInitMessageQueue({:#010x}) with {} entries; nothing can be sent through "
                      "it",
                      handle,
                      size);
    }

    if (handle != 0 && handle == detail::active_context()->system_message_queue) {
        // The system queue is the emulator's, and something may well be parked
        // on it. Replacing it wholesale would drop those waiters on the floor.
        WIIU_LOG_WARN(
            Coreinit, "OSInitMessageQueue on the system message queue {:#010x}; ignored", handle);
        return;
    }

    MessageQueue queue;
    queue.messages = messages;
    queue.capacity = size <= 0 ? 0 : static_cast<u32>(size);
    sync().message_queues[handle] = std::move(queue);
}

/// OSSendMessage and OSJamMessage, which differ only in where the message
/// goes. OS_MESSAGE_QUEUE_FLAG_HIGH_PRIORITY asks for the head too, so a send
/// carrying it is a jam.
void send_message(Call& call, bool force_jam) {
    const u32 handle = call.next_word();
    const u32 message_address = call.next_word();
    const u32 flags = call.next_word();

    MessageQueue* const queue = find_queue(sync(), handle);
    if (queue == nullptr) {
        call.state.gpr[3] = 0;
        return;
    }

    const Message message = read_message(call.memory, message_address);
    const bool jam = force_jam || (flags & message_flag_high_priority) != 0;

    // A queue with someone waiting on it is empty, so the message never needs
    // to touch the ring -- and handing it over directly is what makes the
    // receiver's return value true without it running again.
    if (hand_to_receiver(scheduler(), call.memory, handle, *queue, message)) {
        call.state.gpr[3] = 1;
        return;
    }

    if (queue->entries.size() < queue->capacity) {
        if (jam) {
            queue->entries.push_front(message);
        } else {
            queue->entries.push_back(message);
        }
        call.state.gpr[3] = 1;
        return;
    }

    if ((flags & message_flag_block) == 0) {
        call.state.gpr[3] = 0;
        return;
    }

    // Full, and the caller asked to wait. Park the message with the thread: a
    // receiver taking one out is what lets it in, and by then this handler is
    // long gone.
    const u32 thread = scheduler().current_handle();
    queue->pending_sends[thread] = PendingSend{message, jam};
    call.state.gpr[3] = 1; // a blocking send only ever comes back having sent
    scheduler().block(
        call.state, BlockReason::MessageQueue, handle + message_queue_send_wait_offset);
}

void os_send_message(Call& call) {
    send_message(call, false);
}

void os_jam_message(Call& call) {
    send_message(call, true);
}

void os_receive_message(Call& call) {
    const u32 handle = call.next_word();
    const u32 message_address = call.next_word();
    const u32 flags = call.next_word();

    MessageQueue* const queue = find_queue(sync(), handle);
    if (queue == nullptr) {
        call.state.gpr[3] = 0;
        return;
    }

    if (!queue->entries.empty()) {
        const Message message = queue->entries.front();
        queue->entries.pop_front();
        static_cast<void>(write_message(call.memory, message_address, message));

        // Taking one out made room, so whoever was waiting to send can.
        admit_pending_sender(scheduler(), handle, *queue);
        call.state.gpr[3] = 1;
        return;
    }

    if ((flags & message_flag_block) == 0) {
        call.state.gpr[3] = 0;
        return;
    }

    const u32 thread = scheduler().current_handle();
    queue->pending_receives[thread] = message_address;
    call.state.gpr[3] = 1; // a blocking receive only ever comes back with one
    scheduler().block(
        call.state, BlockReason::MessageQueue, handle + message_queue_recv_wait_offset);
}

void os_peek_message(Call& call) {
    const u32 handle = call.next_word();
    const u32 message_address = call.next_word();

    MessageQueue* const queue = find_queue(sync(), handle);
    if (queue == nullptr || queue->entries.empty()) {
        call.state.gpr[3] = 0;
        return;
    }
    static_cast<void>(write_message(call.memory, message_address, queue->entries.front()));
    call.state.gpr[3] = 1;
}

void os_get_system_message_queue(Call& call) {
    call.state.gpr[3] = system_message_queue(*detail::active_context(), call.memory);
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

ppcaddr_t system_message_queue(Context& context, memory::Memory& memory) {
    if (context.system_message_queue != 0) {
        return context.system_message_queue;
    }

    // coreinit's own memory, not the title's (heap.hpp).
    ExpandedHeap* const heap = context.heaps.find(context.heaps.system_handle());
    if (heap == nullptr) {
        WIIU_LOG_ERROR(Coreinit, "no heap to allocate the system message queue from");
        return 0;
    }

    // The structure and the ring it points at. Both are the title's to see --
    // it is handed the address and may pass it to OSReceiveMessage like any
    // other queue -- so they are real guest memory rather than a reserved
    // value that happens not to collide.
    const u32 ring_bytes = static_cast<u32>(system_message_queue_capacity) * message_size;
    const ppcaddr_t queue = heap->allocate(message_queue_size, 8);
    const ppcaddr_t ring = queue == 0 ? 0 : heap->allocate(ring_bytes, 8);
    if (queue == 0 || ring == 0) {
        // Give back whichever one succeeded. Nothing points at it, and a
        // caller that tries again should not be charged for the last attempt.
        if (queue != 0) {
            static_cast<void>(heap->free(queue));
        }
        WIIU_LOG_ERROR(Coreinit, "out of memory for the system message queue");
        return 0;
    }

    static_cast<void>(memory.fill(queue, message_queue_size, 0));
    static_cast<void>(memory.fill(ring, ring_bytes, 0));

    MessageQueue state;
    state.messages = ring;
    state.capacity = static_cast<u32>(system_message_queue_capacity);
    context.sync.message_queues[queue] = std::move(state);

    context.system_message_queue = queue;
    return queue;
}

void Context::reset() {
    // The scheduler goes first: the threads parked on the sync objects are
    // about to have nothing to be parked on, and forgetting them is what keeps
    // "nothing can run" from being reported as a deadlock in a title that has
    // simply been torn down.
    scheduler.reset();
    sync.reset();

    alarms.clear();
    busy_fs_blocks.clear();
    system_message_queue = 0;
    system_info = 0;
    home_button_menu = true;
    fsa_clients.clear();
    next_fsa_client = 1;

    // The trampoline is coreinit's, registered once, and stays.
    const ppcaddr_t preinit_returned = startup.returned;
    startup = Startup{};
    startup.returned = preinit_returned;
    next_async_id = async_id_base;

    // Handles the title opened, not the devices behind them: a device is the
    // emulator's, like a registered HLE function, and survives a title being
    // torn down and another being started.
    ios.reset();
    next_ios_async_id = ios_async_id_base;

    free_callback_stacks.clear();
    next_callback_handle = callback_handle_base;

    // The title's buffers; the frontend's flip hook stays.
    screens.reset();

    // The GPU's timer went with the scheduler's; what it would have retired
    // goes with the title.
    gpu.reset();
}

Message read_message(memory::Memory& memory, ppcaddr_t address) {
    Message message;
    if (address == 0) {
        return message;
    }
    static_cast<void>(memory.try_read<u32>(address, message.value));
    const auto count = static_cast<u32>(message.args.size());
    for (u32 index = 0; index < count; ++index) {
        static_cast<void>(memory.try_read<u32>(address + 4 + (index * 4), message.args[index]));
    }
    return message;
}

bool write_message(memory::Memory& memory, ppcaddr_t address, const Message& message) {
    if (address == 0) {
        return false;
    }
    bool ok = memory.try_write<u32>(address, message.value);
    const auto count = static_cast<u32>(message.args.size());
    for (u32 index = 0; index < count; ++index) {
        ok = memory.try_write<u32>(address + 4 + (index * 4), message.args[index]) && ok;
    }
    return ok;
}

bool take_message(Context& context, ppcaddr_t queue_address, Message& message) {
    MessageQueue* const queue = find_queue(context.sync, queue_address);
    if (queue == nullptr || queue->entries.empty()) {
        return false;
    }
    message = queue->entries.front();
    queue->entries.pop_front();
    admit_pending_sender(context.scheduler, queue_address, *queue);
    return true;
}

bool post_message(Context& context,
                  memory::Memory& memory,
                  ppcaddr_t queue_address,
                  const Message& message,
                  bool jam) {
    MessageQueue* const queue = find_queue(context.sync, queue_address);
    if (queue == nullptr) {
        return false;
    }

    if (hand_to_receiver(context.scheduler, memory, queue_address, *queue, message)) {
        return true;
    }

    if (queue->entries.size() >= queue->capacity) {
        // Nothing to block here -- this is the system delivering something,
        // not a thread that can wait -- so the message is dropped. Hardware
        // has the same problem and the same answer; saying so is what turns
        // "the title stopped getting events" into a line in the log.
        WIIU_LOG_WARN(Coreinit,
                      "message queue {:#010x} is full ({} entries); dropping a system message",
                      queue_address,
                      queue->capacity);
        return false;
    }

    if (jam) {
        queue->entries.push_front(message);
    } else {
        queue->entries.push_back(message);
    }
    return true;
}

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
        ExpandedHeap* const heap = context.heaps.find(context.heaps.system_handle());
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
    registry.add("coreinit", "OSGetThreadSpecific", os_get_thread_specific);
    registry.add("coreinit", "OSSetThreadSpecific", os_set_thread_specific);
    registry.add("coreinit", "OSSetThreadCleanupCallback", os_set_thread_cleanup_callback);

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

    registry.add("coreinit", "OSInitSpinLock", os_init_spin_lock);
    registry.add("coreinit", "OSAcquireSpinLock", os_acquire_spin_lock);
    registry.add("coreinit", "OSTryAcquireSpinLock", os_try_acquire_spin_lock);
    registry.add("coreinit", "OSReleaseSpinLock", os_release_spin_lock);
    registry.add("coreinit", "OSUninterruptibleSpinLock_Acquire", os_acquire_spin_lock);
    registry.add("coreinit", "OSUninterruptibleSpinLock_TryAcquire", os_try_acquire_spin_lock);
    registry.add("coreinit", "OSUninterruptibleSpinLock_Release", os_release_spin_lock);

    registry.add("coreinit", "OSInitRendezvous", os_init_rendezvous);
    registry.add("coreinit", "OSWaitRendezvous", os_wait_rendezvous);

    registry.add("coreinit", "OSInitMessageQueue", os_init_message_queue);
    registry.add("coreinit", "OSInitMessageQueueEx", os_init_message_queue);
    registry.add("coreinit", "OSSendMessage", os_send_message);
    registry.add("coreinit", "OSJamMessage", os_jam_message);
    registry.add("coreinit", "OSReceiveMessage", os_receive_message);
    registry.add("coreinit", "OSPeekMessage", os_peek_message);
    registry.add("coreinit", "OSGetSystemMessageQueue", os_get_system_message_queue);

    registry.add("coreinit", "OSSwapAtomic", os_swap_atomic);
    registry.add("coreinit", "OSCompareAndSwapAtomic", os_compare_and_swap_atomic);
    registry.add("coreinit", "OSCompareAndSwapAtomicEx", os_compare_and_swap_atomic_ex);
    registry.add("coreinit", "OSAddAtomic", os_add_atomic);
    registry.add("coreinit", "OSAndAtomic", os_and_atomic);
    registry.add("coreinit", "OSOrAtomic", os_or_atomic);
}

} // namespace wiiu::hle::coreinit
