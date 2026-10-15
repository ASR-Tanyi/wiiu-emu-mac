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
//     wait, and the scheduler's timer list is what retires it;
//   * **vsync is a tick of the guest's clock**, 59.94 times a second of
//     timebase, at fixed ticks counted from zero -- not the host display's,
//     which would make every frame-paced title run differently on every
//     machine. A **swap** is counted when `GX2SwapScanBuffers` is called and
//     belongs to the submission that carries it; it **flips** at the first
//     vsync by which that submission has retired and the swap interval has
//     passed since the last flip. That is the order hardware does it in
//     (Cemu and decaf agree on it), and it makes a flip's tick a pure function
//     of the title's own calls.
//
// What is not here yet: what a flip shows (the scan buffers and the copy into
// them), swap interval 0's flip-without-waiting (it waits for vsync like 1),
// the overrun callback a full display list asks for, and the profiling and
// toss-stage modes `GX2Init` can ask for.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/timeline.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>
#include <deque>
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

/// The scheduler timer vsync runs on, re-armed every vsync from `GX2Init`.
inline constexpr u32 vsync_timer_id = 0xFFFD0001;

/// What a thread waiting for a flip blocks on. Outside guest memory, so it
/// cannot be mistaken for a sync object the title allocated.
inline constexpr u32 flip_wait_object = 0xFFFD0002;

/// Vsync `index` falls on this tick: 59.94 Hz -- 60000/1001, the rate the
/// Wii U scans out at, per Cemu -- against the timebase's 62.15625 MHz,
/// which is 1036973.4375 ticks a frame, or 16591575/16. Exact, so no drift
/// accumulates however long a title runs.
[[nodiscard]] constexpr u64 vsync_tick(u64 index) noexcept {
    return index * 16'591'575 / 16;
}

/// The first vsync strictly after `now`.
[[nodiscard]] constexpr u64 next_vsync_index(u64 now) noexcept {
    u64 index = (now * 16 / 16'591'575) + 1;
    while (index > 1 && vsync_tick(index - 1) > now) {
        --index;
    }
    while (vsync_tick(index) <= now) {
        ++index;
    }
    return index;
}

/// The default swap interval: flip at most once a vsync.
inline constexpr u32 default_swap_interval = 1;

/// The largest swap interval accepted. Cemu's bound; nothing a title would
/// mean asks for a third of a second between frames.
inline constexpr u32 max_swap_interval = 19;

/// How many swaps may be waiting for a flip before `GX2SwapScanBuffers`
/// waits for one. Cemu's figure, and the place it waits -- after counting the
/// new swap -- is the one it reports hardware using.
inline constexpr u32 max_pending_swaps = 5;

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

/// The two outputs. GX2's display calls name them by function -- the TV
/// calls and the DRC calls -- and this is the index for both.
enum class ScanTarget : u32 {
    Tv = 0,
    Drc = 1,
};

inline constexpr u32 scan_target_count = 2;

[[nodiscard]] std::string_view scan_target_name(ScanTarget target) noexcept;

/// A scan buffer's shape, in pixels.
struct ScanGeometry {
    u32 width = 0;
    u32 height = 0;
    u32 pitch = 0; ///< pixels from one row to the next
};

/// The TV's shape for a `GX2TVRenderMode`: 480p at 4:3 and 16:9, 720p, and
/// 1080p. Mode 4 is a second 720p that wut leaves unnamed; Cemu and decaf
/// both size it as 1280x720. Nothing for 0, which is "disabled", or for a
/// mode that does not exist.
[[nodiscard]] std::optional<ScanGeometry> tv_scan_geometry(u32 render_mode) noexcept;

/// The GamePad's shape for a `GX2DrcRenderMode`: 854x480, in rows of 864 --
/// decaf's pitch, which is the width rounded up to a multiple of 32. Nothing
/// for 0, "disabled", or a mode that does not exist.
[[nodiscard]] std::optional<ScanGeometry> drc_scan_geometry(u32 render_mode) noexcept;

/// How many frames a `GX2BufferingMode` holds -- the enum's own value, one to
/// three -- or zero for one that is not a mode.
[[nodiscard]] u32 scan_buffer_count(u32 buffering_mode) noexcept;

/// What `GX2CalcTVSize`/`GX2CalcDRCSize` report: every frame, at the format's
/// element size. Zero for a format without one.
///
/// This is decaf's linear figure. Hardware's is the size of a tiled surface,
/// a little larger for the alignment; that one needs Latte's surface layout
/// (8.13). The frames here are the emulator's own -- nothing but the
/// emulator reads a scan buffer -- so the only thing the figure has to be is
/// enough room for them, which it is.
[[nodiscard]] u32
scan_buffer_bytes(const ScanGeometry& geometry, u32 surface_format, u32 buffering_mode) noexcept;

/// What `GX2SetTVBuffer`/`GX2SetDRCBuffer` were told, and the switches and
/// scale that go with it.
struct ScanBuffer {
    ppcaddr_t address = 0;
    u32 size = 0;
    u32 render_mode = 0;    ///< GX2TVRenderMode or GX2DrcRenderMode
    u32 surface_format = 0; ///< GX2SurfaceFormat
    u32 buffering_mode = 0; ///< GX2BufferingMode: also the number of frames
    ScanGeometry geometry;

    bool enabled = false;
    u32 scale_x = 0;
    u32 scale_y = 0;

    [[nodiscard]] bool configured() const noexcept { return address != 0; }
};

/// What the TV and the GamePad report about themselves. The console reports
/// what its outputs are connected to; here that is a 1080p, 16:9 TV and one
/// GamePad, which is what Cemu and decaf report too.
inline constexpr u32 system_tv_scan_mode = 7;    ///< GX2_TV_SCAN_MODE_1080P
inline constexpr u32 system_tv_aspect_ratio = 1; ///< GX2_ASPECT_RATIO_16_9
inline constexpr u32 system_drc_mode = 1;        ///< GX2_DRC_RENDER_MODE_SINGLE

/// What `GX2GetSwapStatus` reports, and what decides the next flip.
struct SwapStatus {
    u32 interval = default_swap_interval;
    u32 swaps = 0;      ///< `GX2SwapScanBuffers` calls
    u32 flips = 0;      ///< flips performed
    u64 last_flip = 0;  ///< the tick of the vsync the last flip happened at
    u64 last_vsync = 0; ///< the tick of the last vsync

    /// The index of the vsync the last flip happened at; meaningless until
    /// there has been one.
    u64 last_flip_vsync = 0;

    [[nodiscard]] u32 pending() const noexcept { return swaps - flips; }
};

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

    // --- the outputs ---------------------------------------------------------

    [[nodiscard]] ScanBuffer& scan_buffer(ScanTarget target) noexcept {
        return scan_buffers_[static_cast<u32>(target)];
    }

    [[nodiscard]] const ScanBuffer& scan_buffer(ScanTarget target) const noexcept {
        return scan_buffers_[static_cast<u32>(target)];
    }

    /// The title's `GX2DRCConnectCallback`, as `GX2SetDRCConnectCallback`
    /// left it.
    [[nodiscard]] ppcaddr_t drc_connect_callback() const noexcept { return drc_connect_; }

    /// Returns the one it replaces.
    ppcaddr_t set_drc_connect_callback(ppcaddr_t callback) noexcept {
        const ppcaddr_t old = drc_connect_;
        drc_connect_ = callback;
        return old;
    }

    // --- swaps and vsync ----------------------------------------------------

    [[nodiscard]] const SwapStatus& swap_status() const noexcept { return swap_; }

    /// `GX2SetSwapInterval`. Returns false, having logged why, for one out of
    /// range, which is left as it was.
    bool set_swap_interval(u32 interval);

    /// `GX2SwapScanBuffers`: count a swap, carried by the submission with
    /// timestamp `timestamp`.
    void request_swap(u64 timestamp);

    /// Vsync number `index`: retire what the GPU has done by its tick, then
    /// flip the oldest swap if its submission has retired and the swap
    /// interval allows. Returns whether it flipped.
    bool vsync(u64 index);

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
    SwapStatus swap_;
    std::array<ScanBuffer, scan_target_count> scan_buffers_{};
    ppcaddr_t drc_connect_ = 0;

    /// The timestamps of the submissions carrying the swaps not yet flipped,
    /// oldest first.
    std::deque<u64> pending_swaps_;
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

/// Arm the vsync timer at the first vsync after `now`. From then on it keeps
/// itself armed, one vsync at a time, until `GX2Shutdown` or a reset.
void start_vsync(coreinit::Context& context, u64 now);

/// Register the GX2 entry points under the `gx2` library.
void register_gx2(Registry& registry, coreinit::Context& context);

/// The display half: scan buffers, the outputs' switches and what they
/// report. Called by `register_gx2`.
void register_gx2_display(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
