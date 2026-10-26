// Phase 6.3 — what a title calls on its way to main(), past what Phase 6's
// first steps covered: the RPX's own `__preinit_user`, called before its
// entry point; the system information block; the debugger and HOME button
// queries; and the FSA client wut's newlib glue opens for its SD card device.
//
// Each is here because wut's start-up calls it -- every title built with
// wut, which is every homebrew title, goes through all of them before
// main() -- and each does what decaf's does, as far as that start-up can
// tell.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>

#include <fmt/format.h>

#include <array>
#include <utility>

namespace wiiu::hle::coreinit {
namespace {

using wiiu::log::Channel;

[[nodiscard]] Context* active() noexcept {
    return detail::active_context();
}

// --- __preinit_user ----------------------------------------------------------

/// The trampoline `__preinit_user` returns to: make the handles it left the
/// base heaps, then send the thread on to the entry point as the loader set
/// it up.
void preinit_returned(Call& call) {
    Context* const context = active();
    if (context == nullptr || context->startup.frame == 0) {
        // Nothing sent a thread through a preinit, so there is nowhere to go.
        WIIU_LOG_ERROR(Coreinit, "__preinit_user returned, but no preinit was running");
        call.state.halted = true;
        return;
    }
    Startup& startup = context->startup;

    constexpr std::array<std::pair<BaseHeap, u32>, 3> handles{{
        {BaseHeap::Mem1, preinit_mem1_offset},
        {BaseHeap::Foreground, preinit_foreground_offset},
        {BaseHeap::Mem2, preinit_mem2_offset},
    }};
    for (const auto& [type, offset] : handles) {
        u32 handle = 0;
        static_cast<void>(call.memory.try_read<u32>(startup.frame + offset, handle));
        // A null handle leaves the base heap alone, as Cafe OS's loader does.
        if (handle != 0 && handle != context->heaps.base(type)) {
            WIIU_LOG_INFO(Coreinit,
                          "__preinit_user made {:#010x} base heap {}",
                          handle,
                          static_cast<u32>(type));
            static_cast<void>(context->heaps.set_base(type, handle));
        }
    }

    call.state.gpr[1] = startup.frame + preinit_frame_size;
    call.state.gpr[3] = startup.r3;
    call.state.gpr[4] = startup.r4;
    call.state.lr = startup.lr;
    call.state.pc = startup.entry;
    call.state.next_pc = startup.entry;

    const ppcaddr_t returned = startup.returned;
    startup = Startup{};
    startup.returned = returned;
}

// --- system information ------------------------------------------------------

/// `OSSystemInfo`, as decaf lays it out: 0x20 bytes.
inline constexpr u32 system_info_size = 0x20;

/// `OSSystemInfo* OSGetSystemInfo()`. The block is coreinit's, in the system
/// heap, filled in on first use: the bus and core clocks, the time at
/// timebase zero, the three cores' L2 sizes, and the core-to-bus ratio --
/// decaf's values, which are the hardware's. wut reads the bus clock for its
/// tick conversions, so a wrong one makes every timeout the wrong length.
u32 os_get_system_info(Call& call) {
    Context* const context = active();
    if (context == nullptr) {
        return 0;
    }
    if (context->system_info != 0) {
        return context->system_info;
    }

    ExpandedHeap* const heap = context->heaps.find(context->heaps.system_handle());
    const ppcaddr_t info = heap == nullptr ? 0 : heap->allocate(system_info_size, 8);
    if (info == 0) {
        WIIU_LOG_ERROR(Coreinit, "no memory for OSGetSystemInfo's block");
        return 0;
    }

    const u64 base_time = rtc_epoch_ticks(*context);
    bool ok = call.memory.try_write<u32>(info + 0x00, static_cast<u32>(cpu::bus_clock_hz));
    ok = call.memory.try_write<u32>(info + 0x04, static_cast<u32>(cpu::cpu_clock_hz)) && ok;
    ok = call.memory.try_write<u32>(info + 0x08, static_cast<u32>(base_time >> 32)) && ok;
    ok = call.memory.try_write<u32>(info + 0x0C, static_cast<u32>(base_time)) && ok;
    for (u32 core = 0; core < cpu::core_count; ++core) {
        ok = call.memory.try_write<u32>(info + 0x10 + (core * 4), cpu::l2_cache_size(core)) && ok;
    }
    ok = call.memory.try_write<u32>(info + 0x1C,
                                    static_cast<u32>(cpu::cpu_clock_hz / cpu::bus_clock_hz)) &&
         ok;
    if (!ok) {
        WIIU_LOG_ERROR(Coreinit, "could not write OSGetSystemInfo's block at {:#010x}", info);
    }

    context->system_info = info;
    return info;
}

/// `BOOL OSIsDebuggerPresent()` and `OSIsDebuggerInitialized()`: there is
/// none, as on a retail console.
u32 os_no_debugger(Call&) {
    return 0;
}

/// `BOOL OSEnableHomeButtonMenu(BOOL)`: TRUE, as decaf has it. There is no
/// HOME button menu to enable yet (the plan's 6.11 has ProcUI's half); the
/// setting is kept for when there is.
u32 os_enable_home_button_menu(Call&, u32 enable) {
    if (Context* const context = active(); context != nullptr) {
        context->home_button_menu = enable != 0;
    }
    return 1;
}

/// `void OSSavesDone_ReadyToRelease()`: a title saying its saves are
/// flushed, on its way out. Nothing is waiting on it here.
void os_saves_done_ready_to_release(Call&) {}

// --- the FSA client ----------------------------------------------------------
//
// FSA is the filesystem's lower interface, which wut's newlib glue opens at
// start-up for an SD card device. A client can be opened and closed; there
// is no SD card behind it, so mounting one fails as a console with an empty
// slot does, and the title carries on without it. Everything else FSA has is
// the plan's filesystem work.

inline constexpr s32 fs_error_ok = 0;
inline constexpr s32 fs_error_not_found = -0x30017;
inline constexpr s32 fs_error_invalid_client_handle = -0x30025;

/// `FSError FSAInit()`.
s32 fsa_init(Call&) {
    return fs_error_ok;
}

/// `FSAClientHandle FSAAddClient(FSAClientAttachAsyncData*)`: a handle,
/// positive.
s32 fsa_add_client(Call&, u32 attach) {
    Context* const context = active();
    if (context == nullptr) {
        return fs_error_invalid_client_handle;
    }
    if (attach != 0) {
        // The asynchronous attach notifies the title when media comes and
        // goes. There are no media.
        WIIU_LOG_WARN(Coreinit, "FSAAddClient: the asynchronous attach is not done yet");
    }
    const s32 handle = context->next_fsa_client++;
    context->fsa_clients.insert(handle);
    return handle;
}

/// `FSError FSADelClient(FSAClientHandle)`.
s32 fsa_del_client(Call&, s32 handle) {
    Context* const context = active();
    if (context == nullptr || context->fsa_clients.erase(handle) == 0) {
        return fs_error_invalid_client_handle;
    }
    return fs_error_ok;
}

/// `FSError FSAMount(FSAClientHandle, const char* source, const char*
/// target, FSAMountFlags, void*, u32)`.
s32 fsa_mount(Call& call, s32 handle, u32 source, u32 target) {
    Context* const context = active();
    if (context == nullptr || !context->fsa_clients.contains(handle)) {
        return fs_error_invalid_client_handle;
    }
    WIIU_LOG_INFO(Coreinit,
                  "FSAMount(\"{}\" at \"{}\"): there is no such device",
                  call.read_string(source),
                  call.read_string(target));
    return fs_error_not_found;
}

} // namespace

bool start_with_preinit(Context& context,
                        memory::Memory& memory,
                        cpu::CoreState& state,
                        ppcaddr_t preinit,
                        std::string& error) {
    if (preinit == 0) {
        return true;
    }
    if (context.startup.returned == 0) {
        error = "coreinit is not registered, so __preinit_user has nowhere to return to";
        return false;
    }

    // A frame of its own below the loader's, holding the three handles as
    // they are now; whatever __preinit_user leaves there is applied when it
    // returns.
    const ppcaddr_t frame = state.gpr[1] - preinit_frame_size;
    bool ok = memory.fill(frame, preinit_frame_size, 0);
    ok = memory.try_write<u32>(frame, state.gpr[1]) && ok; // the back chain
    ok = memory.try_write<u32>(frame + preinit_mem1_offset, context.heaps.base(BaseHeap::Mem1)) &&
         ok;
    ok = memory.try_write<u32>(frame + preinit_foreground_offset,
                               context.heaps.base(BaseHeap::Foreground)) &&
         ok;
    ok = memory.try_write<u32>(frame + preinit_mem2_offset, context.heaps.base(BaseHeap::Mem2)) &&
         ok;
    if (!ok) {
        error = fmt::format("cannot put __preinit_user's frame on the stack at {:#010x}", frame);
        return false;
    }

    context.startup.frame = frame;
    context.startup.entry = state.pc;
    context.startup.r3 = state.gpr[3];
    context.startup.r4 = state.gpr[4];
    context.startup.lr = state.lr;

    state.gpr[1] = frame;
    state.gpr[3] = frame + preinit_mem1_offset;
    state.gpr[4] = frame + preinit_foreground_offset;
    state.gpr[5] = frame + preinit_mem2_offset;
    state.lr = context.startup.returned;
    state.pc = preinit;
    state.next_pc = preinit;

    WIIU_LOG_INFO(Coreinit, "calling __preinit_user at {:#010x} first", preinit);
    return true;
}

void register_startup(Registry& registry, Context& context) {
    // The address __preinit_user returns to. Registered like an import so it
    // has an `sc` word of its own, and resolved now so the address exists
    // before the loader writes the trampolines.
    registry.add("coreinit", "__preinit_user_returned", preinit_returned);
    context.startup.returned = registry.resolve("coreinit", "__preinit_user_returned");

    registry.add("coreinit", "OSGetSystemInfo", os_get_system_info);
    registry.add("coreinit", "OSIsDebuggerPresent", os_no_debugger);
    registry.add("coreinit", "OSIsDebuggerInitialized", os_no_debugger);
    registry.add("coreinit", "OSEnableHomeButtonMenu", os_enable_home_button_menu);
    registry.add("coreinit", "OSSavesDone_ReadyToRelease", os_saves_done_ready_to_release);

    registry.add("coreinit", "FSAInit", fsa_init);
    registry.add("coreinit", "FSAAddClient", fsa_add_client);
    registry.add("coreinit", "FSADelClient", fsa_del_client);
    registry.add("coreinit", "FSAMount", fsa_mount);
}

} // namespace wiiu::hle::coreinit
