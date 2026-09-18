// Phase 6.11 — proc_ui.rpl.
//
// The library a title's main loop is built around. On hardware ProcUI is how
// an application finds out that the system wants the screen back -- the HOME
// button, a system overlay, a shutdown -- and it is the reason a Wii U title's
// main loop looks like this:
//
//     ProcUIInit(&save_callback);
//     while (ProcUIProcessMessages(true) != PROCUI_STATUS_EXITING) {
//         draw();
//     }
//     ProcUIShutdown();
//
// So a title that cannot get a sensible answer out of ProcUIProcessMessages
// does not draw a wrong frame; it never gets to the first one.
//
// Nothing in this emulator asks for the foreground yet -- there is no HOME
// button and no system menu -- so a title runs in the foreground until the
// emulator asks it to stop. The transitions are driven from the host side
// (`request_release`, `request_acquire`, `request_exit`) and delivered on
// coreinit's system message queue, which is the path the real thing uses and
// the path a HOME button will use when there is one.
#pragma once

#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/registry.hpp>

#include <map>
#include <set>
#include <vector>

namespace wiiu::hle::procui {

/// What ProcUIProcessMessages reports. The values are the guest's.
enum class Status : u32 {
    InForeground = 0,
    InBackground = 1,

    /// The system wants the screen back. The title finishes the frame it is
    /// on, releases what it has to, and calls ProcUIDrawDoneRelease.
    ReleaseForeground = 2,

    Exiting = 3,
};

[[nodiscard]] std::string_view status_name(Status status) noexcept;

/// The callback kinds a title can register for, with the guest's values.
enum class CallbackType : u32 {
    Acquire = 0, ///< the foreground is ours again: recreate what was released
    Release = 1, ///< give the foreground back: free GPU resources and unmap MEM1
    Exit = 2,
    NetIoStart = 3,
    NetIoStop = 4,
    HomeButtonDenied = 5,
};

inline constexpr u32 callback_type_count = 6;

/// The messages coreinit posts on the system message queue for ProcUI to read.
///
/// These values are this emulator's own, not hardware's: nothing outside this
/// file writes them and nothing outside it reads them. A title that goes
/// looking at the raw messages -- which no title has reason to do -- would see
/// something different from a console.
enum class SystemMessage : u32 {
    ReleaseForeground = 1,
    AcquireForeground = 2,
    Exit = 3,
};

/// One registered callback.
struct Callback {
    ppcaddr_t function = 0;
    u32 param = 0;
    u32 priority = 0;

    /// Registration order, so two callbacks at the same priority run in the
    /// order they were registered rather than in whatever order the container
    /// happens to hold them.
    u64 sequence = 0;
};

/// A guest call ProcUI has lined up to make on the caller's own thread.
struct PendingCall {
    ppcaddr_t function = 0;
    u32 param = 0;
};

/// Where a thread running a chain of callbacks came from, so it can be put
/// back when the chain finishes.
struct Chain {
    std::vector<PendingCall> calls;
    std::size_t next = 0;
    ppcaddr_t resume_pc = 0; ///< the address ProcUIProcessMessages would have returned to
    u32 resume_value = 0;    ///< and the value it would have returned

    /// Which thread this belongs to, beyond its handle. A callback that exits
    /// its thread instead of returning leaves the chain behind, and the next
    /// thread the title allocates at that address would otherwise look like it
    /// were half way through a chain it has never heard of.
    u64 generation = 0;
};

/// Everything ProcUI remembers for a running title.
struct State {
    bool initialised = false;
    bool foreground = true;

    /// The title has finished the frame it was releasing on and called
    /// ProcUIDrawDoneRelease; the transition to the background can complete.
    bool draw_done = false;

    bool release_pending = false;
    bool acquire_pending = false;
    bool exit_pending = false;

    /// Exiting has been reported at least once. Every call after that reports
    /// it again: a title's loop tests for it and must not be told it is back
    /// in the foreground half way out.
    bool exiting = false;

    Status status = Status::InForeground;

    ppcaddr_t save_callback = 0;
    u32 save_argument = 0;
    bool save_callback_takes_argument = false;

    std::map<CallbackType, std::vector<Callback>> callbacks;
    u64 next_sequence = 1;

    /// Where a message delivered to a parked thread lands, per thread.
    ///
    /// A woken thread resumes *after* its call, so the message it was waiting
    /// for has to be picked up by its next one -- and only by *its* next one.
    /// One shared slot meant any other thread's poll consumed the flag and
    /// applied an empty message, and a second thread parking overwrote the
    /// first one's message before it woke. Either way the transition was gone,
    /// and a backgrounded title never came back.
    std::map<u32, ppcaddr_t> inboxes;

    /// Threads that parked and have not yet folded in what they were sent.
    std::set<u32> awaiting;

    /// Callback chains in flight, by thread handle. A chain runs on the thread
    /// that called ProcUIProcessMessages, which is where hardware runs these
    /// too -- the title expects its release callback to have finished by the
    /// time the call returns.
    std::map<u32, Chain> chains;

    /// The trampoline a callback returns to, so the next one can be started.
    ppcaddr_t chain_return = 0;

    /// How many times ProcUIProcessMessages has been called, which is the
    /// cheapest way to see whether a title's main loop is actually turning.
    u64 poll_count = 0;

    void reset();

    // --- driving it from the host -----------------------------------------
    //
    // What a HOME button, a system overlay or a close request will call. They
    // post to coreinit's system message queue rather than setting a flag here,
    // because that is what wakes a title parked in the background -- and
    // because it is the path hardware uses.

    /// Ask the title to give the foreground back. It gets one more frame.
    bool request_release(coreinit::Context& context, memory::Memory& memory);

    bool request_acquire(coreinit::Context& context, memory::Memory& memory);

    /// Ask the title to shut down. Its loop sees Exiting and leaves.
    bool request_exit(coreinit::Context& context, memory::Memory& memory);
};

/// Register the library. `context` is coreinit's, for the system message
/// queue, the heaps and the scheduler.
void register_all(Registry& registry, coreinit::Context& context, State& state);

} // namespace wiiu::hle::procui
