// Phase 6.11 — proc_ui.rpl.
//
// Two things here are worth reading before the rest.
//
// The first is how a callback runs. ProcUI's callbacks are guest functions,
// and the title expects them to have finished by the time
// ProcUIProcessMessages returns -- a release callback frees the GPU resources
// the next call is about to hand back. A native handler cannot call guest code
// and wait for it, so instead the handler *redirects the thread it is already
// on*: it remembers where the title wanted to return to, points the thread at
// the first callback with the return address set to a trampoline of ours, and
// lets it run. The trampoline is an `sc` like any other HLE call, so when the
// callback returns we get control back, start the next one, and eventually put
// the thread back where it was going with the value it was going to get. The
// callbacks therefore run on the caller's own thread and its own stack, in
// order, exactly as on hardware -- no extra thread, no join.
//
// The second is what drives the transitions. Nothing in the emulator asks for
// the foreground yet, so the state machine is moved by the host side asking,
// and those requests arrive as messages on coreinit's system message queue --
// which is both what hardware does and what lets a title parked in the
// background be woken rather than polled.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/proc_ui.hpp>

#include <algorithm>

namespace wiiu::hle::procui {
namespace {

using wiiu::log::Channel;

/// The context and state the handlers act on. Same reasoning as coreinit's:
/// a handler is a plain function and has nowhere to keep them.
coreinit::Context* g_context = nullptr;
State* g_state = nullptr;

[[nodiscard]] bool ready() {
    return g_context != nullptr && g_state != nullptr;
}

/// A scratch OSMessage for `thread` to receive into, allocated on first use
/// and kept for as long as that thread exists.
///
/// One per thread, not one per title: two threads parked on the system queue
/// register their inbox address with it, and a shared slot means the second
/// delivery lands on top of the first before anyone has read it.
[[nodiscard]] ppcaddr_t
inbox(State& state, coreinit::Context& context, memory::Memory& memory, u32 thread) {
    const auto existing = state.inboxes.find(thread);
    if (existing != state.inboxes.end()) {
        return existing->second;
    }

    ExpandedHeap* const heap = context.heaps.find(context.heaps.default_handle());
    if (heap == nullptr) {
        return 0;
    }
    const ppcaddr_t address = heap->allocate(coreinit::message_size, 8);
    if (address == 0) {
        WIIU_LOG_ERROR(Coreinit, "out of memory for the ProcUI inbox");
        return 0;
    }
    static_cast<void>(memory.fill(address, coreinit::message_size, 0));
    state.inboxes[thread] = address;
    return address;
}

void apply(State& state, u32 value) {
    switch (static_cast<SystemMessage>(value)) {
        case SystemMessage::ReleaseForeground:
            state.release_pending = true;
            state.acquire_pending = false;
            break;
        case SystemMessage::AcquireForeground:
            state.acquire_pending = true;
            state.release_pending = false;
            break;
        case SystemMessage::Exit:
            state.exit_pending = true;
            break;
        default:
            WIIU_LOG_WARN(Coreinit, "unrecognised system message {:#x} for ProcUI", value);
            break;
    }
}

/// Fold everything waiting into the state machine: first whatever was
/// delivered while a thread was parked on the queue, then whatever is in it
/// now. That order is the order the messages arrived in.
void drain(State& state, coreinit::Context& context, memory::Memory& memory) {
    const ppcaddr_t queue = context.system_message_queue;

    // Anything handed to a parked thread comes first: it was taken out of the
    // queue before whatever is still in it. Each parked thread has its own
    // slot, so two deliveries cannot land on top of each other -- but the
    // state machine belongs to the title, not to a thread, so *whichever*
    // thread polls next folds in everything that has arrived. A message that
    // only its own thread could pick up would sit unread while another thread
    // was told nothing had happened.
    //
    // A thread still registered as a receiver has not been given anything yet;
    // one that is not, but is still in `awaiting`, has.
    const auto found = context.sync.message_queues.find(queue);
    const coreinit::MessageQueue* const entry =
        (queue == 0 || found == context.sync.message_queues.end()) ? nullptr : &found->second;

    for (auto waiting = state.awaiting.begin(); waiting != state.awaiting.end();) {
        const u32 parked = *waiting;
        const bool delivered = entry == nullptr || entry->pending_receives.find(parked) ==
                                                       entry->pending_receives.end();
        if (!delivered) {
            ++waiting;
            continue;
        }
        waiting = state.awaiting.erase(waiting);

        const auto where = state.inboxes.find(parked);
        if (where != state.inboxes.end()) {
            apply(state, coreinit::read_message(memory, where->second).value);
        }
    }

    if (queue == 0) {
        return;
    }
    // Through coreinit rather than off the deque, so anything parked on a full
    // system queue is let in as messages are taken out of it.
    coreinit::Message message;
    while (coreinit::take_message(context, queue, message)) {
        apply(state, message.value);
    }
}

/// The registered callbacks of one kind, in the order they should run:
/// highest priority first, and within a priority, registration order.
[[nodiscard]] std::vector<PendingCall> ordered(const State& state, CallbackType type) {
    const auto entry = state.callbacks.find(type);
    if (entry == state.callbacks.end()) {
        return {};
    }

    std::vector<Callback> sorted = entry->second;
    std::ranges::sort(sorted, [](const Callback& left, const Callback& right) {
        if (left.priority != right.priority) {
            return left.priority > right.priority;
        }
        return left.sequence < right.sequence;
    });

    std::vector<PendingCall> calls;
    calls.reserve(sorted.size());
    for (const Callback& callback : sorted) {
        if (callback.function != 0) {
            calls.push_back(PendingCall{callback.function, callback.param});
        }
    }
    return calls;
}

/// Point the current thread at `call`, with the return address set to our
/// trampoline so we get control back when it returns.
void enter(cpu::CoreState& state, const State& procui, const PendingCall& call) {
    state.gpr[3] = call.param;
    state.lr = procui.chain_return;
    state.pc = call.function;
    state.next_pc = call.function;
}

/// Put the thread back where ProcUIProcessMessages was going to send it.
void leave(cpu::CoreState& state, const Chain& chain) {
    state.gpr[3] = chain.resume_value;
    state.pc = chain.resume_pc;
    state.next_pc = chain.resume_pc;
}

/// Run `calls` on the calling thread and then return `result` from the call
/// that is running now. With nothing to run, this is just setting r3.
void run_chain(Call& call, State& state, std::vector<PendingCall> calls, u32 result) {
    call.state.gpr[3] = result;
    if (calls.empty()) {
        return;
    }

    if (state.chain_return == 0) {
        // Without the trampoline there is nowhere for a callback to return to,
        // and jumping into one anyway would lose the thread. Skipping them is
        // wrong, but it is wrong in a way that says so.
        WIIU_LOG_ERROR(
            Coreinit, "ProcUI has no callback trampoline; {} callback(s) not run", calls.size());
        return;
    }

    Chain chain;
    chain.calls = std::move(calls);
    chain.next = 1;
    chain.resume_pc = call.state.pc; // the dispatcher already put the caller's lr here
    chain.resume_value = result;

    const PendingCall first = chain.calls.front();
    const u32 thread = g_context->scheduler.current_handle();
    const Thread* const running = g_context->scheduler.find(thread);
    chain.generation = running == nullptr ? 0 : running->generation;
    state.chains[thread] = std::move(chain);
    enter(call.state, state, first);
}

/// Forget chains belonging to threads that are gone.
///
/// A chain is erased when it finishes, so a callback that exits its thread
/// instead of returning leaves one behind -- keyed by a guest OSThread
/// pointer, which titles reuse from a pool. The next thread at that address
/// then looks like it is in the middle of a chain it has never heard of.
void prune_chains(State& state, Scheduler& threads) {
    for (auto entry = state.chains.begin(); entry != state.chains.end();) {
        const Thread* const thread = threads.find(entry->first);
        const bool mine = thread != nullptr && thread->state != ThreadState::Exited &&
                          thread->generation == entry->second.generation;
        entry = mine ? std::next(entry) : state.chains.erase(entry);
    }
}

/// The trampoline every ProcUI callback returns to.
void chain_return(Call& call) {
    if (!ready()) {
        return;
    }
    State& state = *g_state;
    const u32 thread = g_context->scheduler.current_handle();

    const auto entry = state.chains.find(thread);
    if (entry == state.chains.end()) {
        // Nothing is running a chain on this thread, so whoever landed here
        // did not come from one. There is no sensible place to send it, and
        // the dispatcher has already pointed it back at the trampoline, which
        // would spin.
        WIIU_LOG_ERROR(
            Coreinit, "a ProcUI callback returned on thread {:#010x}, which has no chain", thread);
        call.state.halted = true;
        return;
    }

    Chain& chain = entry->second;
    if (chain.next < chain.calls.size()) {
        const PendingCall next = chain.calls[chain.next];
        ++chain.next;
        enter(call.state, state, next);
        return;
    }

    const Chain finished = chain;
    state.chains.erase(entry);
    leave(call.state, finished);
}

// --- the state machine -----------------------------------------------------

/// What the next call should report, running whatever the transition needs.
/// Returns the status, and the callbacks to run before the title sees it.
[[nodiscard]] Status advance(State& state, std::vector<PendingCall>& calls) {
    if (state.exit_pending && !state.exiting) {
        state.exiting = true;
        state.exit_pending = false;

        // The save callback runs first: it is the title's last chance to write
        // anything out, and it runs before the exit callbacks tear down what
        // it might need.
        if (state.save_callback != 0) {
            calls.push_back(PendingCall{
                state.save_callback, state.save_callback_takes_argument ? state.save_argument : 0});
        }
        for (const PendingCall& call : ordered(state, CallbackType::Exit)) {
            calls.push_back(call);
        }
        return Status::Exiting;
    }

    if (state.exiting) {
        return Status::Exiting;
    }

    if (state.foreground && state.release_pending) {
        if (!state.draw_done) {
            // Tell the title to finish up. It calls ProcUIDrawDoneRelease when
            // it has, and the transition completes on the next call.
            return Status::ReleaseForeground;
        }
        state.release_pending = false;
        state.draw_done = false;
        state.foreground = false;
        calls = ordered(state, CallbackType::Release);
        return Status::InBackground;
    }

    if (!state.foreground && state.acquire_pending) {
        state.acquire_pending = false;
        state.foreground = true;
        calls = ordered(state, CallbackType::Acquire);
        return Status::InForeground;
    }

    return state.foreground ? Status::InForeground : Status::InBackground;
}

/// Park the calling thread on the system message queue until something is
/// posted to it. The message goes to the inbox, which the next call folds in.
void wait_for_message(Call& call, State& state, coreinit::Context& context) {
    const u32 thread = context.scheduler.current_handle();
    const ppcaddr_t queue = coreinit::system_message_queue(context, call.memory);
    const ppcaddr_t where = inbox(state, context, call.memory, thread);
    if (queue == 0 || where == 0) {
        return;
    }

    const auto entry = context.sync.message_queues.find(queue);
    if (entry == context.sync.message_queues.end()) {
        return;
    }
    if (!entry->second.entries.empty()) {
        // Something arrived between the drain and here; there is nothing to
        // wait for, and parking would wait for the message after it.
        return;
    }

    // Register as a receiver the same way OSReceiveMessage does: whoever posts
    // the message writes it into `where` and wakes this thread, because by
    // then this call is long over. `awaiting` is what tells this thread's next
    // call to look there, and it is per thread, so no other thread's poll can
    // consume it.
    entry->second.pending_receives[thread] = where;
    state.awaiting.insert(thread);
    context.scheduler.block(
        call.state, BlockReason::MessageQueue, queue + coreinit::message_queue_recv_wait_offset);
}

// --- the library -----------------------------------------------------------

void procui_init(Call& call) {
    const u32 save_callback = call.next_word();
    if (!ready()) {
        return;
    }

    State& state = *g_state;
    const bool was_initialised = state.initialised;
    state.reset();
    state.initialised = true;
    state.save_callback = save_callback;
    state.save_callback_takes_argument = false;

    if (was_initialised) {
        // Re-initialising drops the callbacks the title registered, which is
        // what hardware does and what a title relaunching a scene relies on.
        WIIU_LOG_DEBUG(Coreinit, "ProcUIInit called again; callbacks cleared");
    }

    // Make sure the queue exists now rather than at the first transition: the
    // thing that asks for one is the host side, which has no guest call to
    // allocate from.
    static_cast<void>(coreinit::system_message_queue(*g_context, call.memory));
}

void procui_init_ex(Call& call) {
    const u32 save_callback = call.next_word();
    const u32 argument = call.next_word();
    if (!ready()) {
        return;
    }

    State& state = *g_state;
    state.reset();
    state.initialised = true;
    state.save_callback = save_callback;
    state.save_argument = argument;
    state.save_callback_takes_argument = true;

    static_cast<void>(coreinit::system_message_queue(*g_context, call.memory));
}

void procui_shutdown(Call&) {
    if (ready()) {
        g_state->initialised = false;
    }
}

void procui_process_messages(Call& call) {
    const bool block = call.next_word() != 0;
    if (!ready()) {
        call.state.gpr[3] = static_cast<u32>(Status::Exiting);
        return;
    }

    State& state = *g_state;
    ++state.poll_count;

    const u32 thread = g_context->scheduler.current_handle();
    prune_chains(state, g_context->scheduler);

    if (state.chains.find(thread) != state.chains.end()) {
        // Called from inside one of its own callbacks. Starting a second chain
        // here would overwrite the first one's saved return address, and the
        // outer call would never get back to the title -- it would come back
        // into the callback it was running, whose own return then lands on a
        // trampoline with no chain behind it. Report where things stand and
        // change nothing.
        WIIU_LOG_WARN(Coreinit,
                      "ProcUIProcessMessages called from inside a ProcUI callback; "
                      "reporting {} and running nothing",
                      status_name(state.status));
        call.state.gpr[3] = static_cast<u32>(state.status);
        return;
    }

    if (!state.initialised) {
        // A title that never called ProcUIInit is not going to get a coherent
        // answer, and telling it to exit is the one answer that cannot leave
        // it looping forever.
        WIIU_LOG_WARN(Coreinit, "ProcUIProcessMessages before ProcUIInit");
        call.state.gpr[3] = static_cast<u32>(Status::Exiting);
        return;
    }

    drain(state, *g_context, call.memory);

    std::vector<PendingCall> calls;
    const Status status = advance(state, calls);
    state.status = status;

    // The callbacks run before the title sees the status, on this thread, and
    // the status is what the call returns once they are done.
    run_chain(call, state, std::move(calls), static_cast<u32>(status));

    if (status == Status::InBackground && block && !state.acquire_pending && !state.exit_pending &&
        state.chains.find(g_context->scheduler.current_handle()) == state.chains.end()) {
        // Nothing to do until the system says otherwise. Parking here is what
        // keeps a backgrounded title from spinning through its main loop, and
        // the idle skip (P2) jumps the wait rather than emulating it.
        wait_for_message(call, state, *g_context);
    }
}

void procui_in_foreground(Call& call) {
    call.state.gpr[3] = (ready() && g_state->foreground) ? 1U : 0U;
}

void procui_is_running(Call& call) {
    call.state.gpr[3] = (ready() && g_state->initialised) ? 1U : 0U;
}

void procui_draw_done_release(Call& call) {
    // Only meaningful while the title is being asked to release. Latching it
    // at any other time -- a title that calls this once a frame after swapping
    // buffers, or a release callback that calls it on the way out -- meant the
    // *next* handover skipped ReleaseForeground entirely: the title was never
    // told to finish its frame, and its release callbacks ran underneath it
    // while it was still drawing.
    if (ready() && g_state->release_pending && g_state->foreground) {
        g_state->draw_done = true;
    }
    call.state.gpr[3] = 0;
}

void procui_register_callback(Call& call) {
    const u32 type = call.next_word();
    const u32 function = call.next_word();
    const u32 param = call.next_word();
    const u32 priority = call.next_word();

    if (!ready()) {
        return;
    }
    if (type >= callback_type_count) {
        WIIU_LOG_WARN(Coreinit, "ProcUIRegisterCallback with unknown type {}", type);
        return;
    }

    State& state = *g_state;
    Callback callback;
    callback.function = function;
    callback.param = param;
    callback.priority = priority;
    callback.sequence = state.next_sequence++;
    state.callbacks[static_cast<CallbackType>(type)].push_back(callback);
}

void procui_set_save_callback(Call& call) {
    const u32 function = call.next_word();
    const u32 argument = call.next_word();
    if (!ready()) {
        return;
    }
    g_state->save_callback = function;
    g_state->save_argument = argument;
    g_state->save_callback_takes_argument = true;
}

void procui_clear_callbacks(Call&) {
    if (ready()) {
        g_state->callbacks.clear();
    }
}

} // namespace

std::string_view status_name(Status status) noexcept {
    switch (status) {
        case Status::InForeground:
            return "foreground";
        case Status::InBackground:
            return "background";
        case Status::ReleaseForeground:
            return "releasing";
        case Status::Exiting:
            return "exiting";
    }
    return "?";
}

void State::reset() {
    // These are the emulator's, not the title's, and survive. The inboxes are
    // heap allocations that would leak if they were forgotten; the trampoline
    // is assigned once when the library is registered; and a chain in flight
    // is a thread that is *currently* inside a callback -- dropping its record
    // would strand it with nowhere to return to.
    auto kept_inboxes = std::move(inboxes);
    auto kept_chains = std::move(chains);
    const ppcaddr_t kept_trampoline = chain_return;

    *this = State{};

    inboxes = std::move(kept_inboxes);
    chains = std::move(kept_chains);
    chain_return = kept_trampoline;
}

bool State::request_release(coreinit::Context& context, memory::Memory& memory) {
    coreinit::Message message;
    message.value = static_cast<u32>(SystemMessage::ReleaseForeground);
    const ppcaddr_t queue = coreinit::system_message_queue(context, memory);
    return queue != 0 && coreinit::post_message(context, memory, queue, message);
}

bool State::request_acquire(coreinit::Context& context, memory::Memory& memory) {
    coreinit::Message message;
    message.value = static_cast<u32>(SystemMessage::AcquireForeground);
    const ppcaddr_t queue = coreinit::system_message_queue(context, memory);
    return queue != 0 && coreinit::post_message(context, memory, queue, message);
}

bool State::request_exit(coreinit::Context& context, memory::Memory& memory) {
    coreinit::Message message;
    message.value = static_cast<u32>(SystemMessage::Exit);
    const ppcaddr_t queue = coreinit::system_message_queue(context, memory);

    // An exit jumps the queue. A title in the middle of a foreground handover
    // has messages ahead of this one that it no longer needs to act on, and
    // the thing being asked for is that it stops.
    return queue != 0 && coreinit::post_message(context, memory, queue, message, true);
}

void register_all(Registry& registry, coreinit::Context& context, State& state) {
    g_context = &context;
    g_state = &state;

    registry.add("proc_ui", "ProcUIInit", procui_init);
    registry.add("proc_ui", "ProcUIInitEx", procui_init_ex);
    registry.add("proc_ui", "ProcUIShutdown", procui_shutdown);
    registry.add("proc_ui", "ProcUIProcessMessages", procui_process_messages);
    registry.add("proc_ui", "ProcUISubProcessMessages", procui_process_messages);
    registry.add("proc_ui", "ProcUIInForeground", procui_in_foreground);
    registry.add("proc_ui", "ProcUIIsRunning", procui_is_running);
    registry.add("proc_ui", "ProcUIDrawDoneRelease", procui_draw_done_release);
    registry.add("proc_ui", "ProcUIRegisterCallback", procui_register_callback);
    registry.add("proc_ui", "ProcUISetSaveCallback", procui_set_save_callback);
    registry.add("proc_ui", "ProcUISetSaveCallbackEx", procui_set_save_callback);
    registry.add("proc_ui", "ProcUIClearCallbacks", procui_clear_callbacks);

    // The address a callback returns to. Registered like an import so it gets
    // an `sc` word of its own, and resolved now so that the address exists
    // before the trampolines are written into guest memory.
    registry.add("proc_ui", "__procui_callback_returned", chain_return);
    state.chain_return = registry.resolve("proc_ui", "__procui_callback_returned");
}

} // namespace wiiu::hle::procui
