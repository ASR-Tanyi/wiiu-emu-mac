// Phase 8.1 — GX2, the graphics library: command buffers and timestamps.
//
// GX2 is how a title talks to the GPU. Almost every call appends PM4 packets
// (8.3) to a command buffer; `GX2Flush` hands the buffer to the GPU and the
// title carries on. What comes back is a timestamp per submission, which the
// title polls (`GX2GetRetiredTimeStamp`) or waits on (`GX2WaitTimeStamp`,
// `GX2DrawDone`). This file is that skeleton -- the buffers, the flush, the
// timestamps, display lists -- and the calls that set state on the way to a
// draw (8.5 onward) are built on it.
//
// The model, following decaf where it has one:
//
//   * the **main command buffer** is the emulator's, not the title's. GX2
//     carves it out of a pool the title may name in `GX2Init`, but nothing
//     the title can see points into it, and the timeline (8.2) copies a
//     submission when it is made -- so it lives host-side, in host order,
//     and the pool is noted and left alone;
//   * a **display list** is the title's. It is written big-endian into the
//     title's memory, where the title can keep it, copy it, patch it and
//     call it later, and the GPU reads it from there when a call to it
//     retires -- which is what hardware does too;
//   * **timestamps count submissions**, from 1. One retires when the guest's
//     clock reaches its deadline (ADR-004), and a guest thread waiting for one
//     sleeps until then: the idle skip (P2) gets there without emulating the
//     wait, and the scheduler's timer list is what retires it.
//
// What is not here yet: the vsync and flip events (the callbacks are stored,
// nothing raises them), the overrun callback a full display list asks for,
// and the profiling and toss-stage modes `GX2Init` can ask for.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/timeline.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace wiiu::hle {

class Registry;

namespace coreinit {
struct Context;
} // namespace coreinit

namespace gx2 {

/// The scheduler timer the GPU's timeline retires on. One id, re-armed at
/// each next deadline; below the IOS completion ids, and above every guest
/// address an alarm is keyed by.
inline constexpr u32 gpu_timer_id = 0xFFFD0000;

/// GX2 pads every buffer it submits to a multiple of this many words.
inline constexpr u32 command_buffer_alignment_words = 8;

/// The most a single submission holds before GX2 submits it on its own and
/// starts another. decaf's figure; a title can tell only by counting
/// timestamps, which is why it is modelled at all.
inline constexpr u32 max_command_buffer_words = 0x20000;

/// `GX2InitAttributes`: the keys of the zero-terminated key/value list
/// `GX2Init` takes.
enum class InitAttribute : u32 {
    End = 0,
    CommandBufferBase = 1,
    CommandBufferPoolSize = 2,
    Argc = 7,
    Argv = 8,
    ProfileMode = 9,
    TossStage = 10,
    AppIoThreadStackSize = 11,
};

/// What `GX2Init` was told. Zero for anything not given.
struct InitAttributes {
    ppcaddr_t command_buffer_base = 0;
    u32 command_buffer_pool_size = 0;
    u32 argc = 0;
    ppcaddr_t argv = 0;
    u32 profile_mode = 0;
    u32 toss_stage = 0;
    u32 app_io_thread_stack_size = 0;
};

/// The longest attribute list read before giving up on finding its end. A
/// real one has a handful of pairs; one this long has no terminator.
inline constexpr u32 max_init_attributes = 64;

/// Read a `GX2Init` attribute list. A null list is an empty one -- wut's
/// default. Unknown keys are logged and skipped; an unreadable or
/// unterminated list is logged and what was read so far kept.
[[nodiscard]] InitAttributes read_init_attributes(memory::Memory& memory, ppcaddr_t address);

/// `GX2EventType`.
enum class EventType : u32 {
    StartOfPipe = 0,
    EndOfPipe = 1,
    Vsync = 2,
    Flip = 3,
    DisplayListOverrun = 4,
};

inline constexpr u32 event_type_count = 5;

[[nodiscard]] std::string_view event_type_name(u32 type) noexcept;

/// A `GX2SetEventCallback` registration: a guest function and its argument.
struct EventCallback {
    ppcaddr_t function = 0;
    ppcaddr_t user_data = 0;
};

/// A display list being written, in the title's memory.
struct DisplayList {
    ppcaddr_t base = 0;
    u32 capacity_words = 0;
    u32 used_words = 0;
    bool profiling = false;

    /// Set once a write did not fit. Everything after it is dropped, and the
    /// list ends short -- the overrun callback that would have supplied more
    /// room is not raised yet.
    bool overrun = false;
};

/// The cores a display list can be open on: one each, as on hardware.
inline constexpr u32 core_count = 3;

/// The GPU as GX2 sees it: the buffers being written, the timeline they go
/// to, and the register file and command processor behind that.
class Gpu {
public:
    Gpu();

    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;

    /// `GX2Init`: remember who called, and give the command processor the
    /// memory it reads indirect buffers from and writes back into.
    void initialize(memory::Memory& memory, u32 main_core, const InitAttributes& attributes);

    /// `GX2Shutdown`. Anything not yet flushed is dropped, as on hardware;
    /// what was submitted still retires.
    void shutdown() noexcept;

    [[nodiscard]] bool initialized() const noexcept { return initialized_; }

    [[nodiscard]] u32 main_core() const noexcept { return main_core_; }

    [[nodiscard]] const InitAttributes& attributes() const noexcept { return attributes_; }

    /// Append packets to whatever `core` is writing: its open display list,
    /// or, on the main core, the main command buffer -- submitting that first
    /// if they would not fit. Returns false, having logged why, if they went
    /// nowhere: a full display list, or a core with neither.
    bool write(u32 core, std::span<const u32> words, u64 now_ticks);

    /// `GX2Flush`: pad the main command buffer and submit it. Returns the
    /// timestamp it was given, or the last one handed out if there was
    /// nothing to submit.
    u64 flush(u64 now_ticks);

    /// Submit a stream as its own submission, unpadded and uncopied into the
    /// main buffer -- what `GX2DirectCallDisplayList` does after flushing.
    u64 submit(std::span<const u32> words, u64 now_ticks);

    /// The words written to the main command buffer and not yet flushed.
    [[nodiscard]] std::span<const u32> pending_commands() const noexcept { return commands_; }

    // --- display lists ------------------------------------------------------

    /// `GX2BeginDisplayListEx`. Returns false, having logged why, if `core`
    /// already has one open.
    bool begin_display_list(u32 core, ppcaddr_t base, u32 bytes, bool profiling);

    /// `GX2EndDisplayList`: pad the list and close it. Returns its size in
    /// bytes, or nothing if `core` had none open.
    [[nodiscard]] std::optional<u32> end_display_list(u32 core);

    /// The list `core` is writing, if any.
    [[nodiscard]] const DisplayList* display_list(u32 core) const noexcept;

    // --- events -------------------------------------------------------------

    void set_event_callback(u32 type, EventCallback callback) noexcept;

    [[nodiscard]] EventCallback event_callback(u32 type) const noexcept;

    // --- the GPU itself -----------------------------------------------------

    [[nodiscard]] latte::GpuTimeline& timeline() noexcept { return timeline_; }

    [[nodiscard]] const latte::GpuTimeline& timeline() const noexcept { return timeline_; }

    [[nodiscard]] latte::CommandProcessor& processor() noexcept { return processor_; }

    [[nodiscard]] const latte::RegisterFile& registers() const noexcept { return *registers_; }

    /// Back to before `GX2Init`, with nothing in flight: the register file
    /// cleared and a fresh timeline started. The command processor's
    /// listeners and counters are the emulator's, and stay.
    void reset();

private:
    bool write_display_list(DisplayList& list, std::span<const u32> words);

    // Heap-held: 256 KiB of registers does not belong inside a Context that
    // tests put on the stack.
    std::unique_ptr<latte::RegisterFile> registers_;
    latte::CommandProcessor processor_;
    latte::GpuTimeline timeline_;

    memory::Memory* memory_ = nullptr;
    std::vector<u32> commands_;
    std::array<std::optional<DisplayList>, core_count> display_lists_{};
    std::array<EventCallback, event_type_count> callbacks_{};
    InitAttributes attributes_{};
    bool initialized_ = false;
    u32 main_core_ = 1;
};

/// Pad a stream to a multiple of `command_buffer_alignment_words` with type-2
/// filler, as GX2 does before submitting anything.
void pad_command_buffer(std::vector<u32>& words);

/// Keep the scheduler's GPU timer at the timeline's next deadline: armed when
/// something is in flight, cleared when nothing is. Called after every
/// submission, and by the timer itself after it retires what was due.
void arm_retirement(coreinit::Context& context);

/// Register the GX2 entry points under the `gx2` library.
void register_gx2(Registry& registry, coreinit::Context& context);

} // namespace gx2
} // namespace wiiu::hle
