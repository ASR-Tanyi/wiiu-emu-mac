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
// What a flip shows is the scan buffers (gx2_display.cpp), filled by the
// scan-out copy (gx2_surface.cpp). The swap, the clears and that copy reach the
// GPU as the emulator's own packets (`pm4::Opcode::Hle*`), so each happens in
// stream order when its submission retires -- from a display list as well as
// the ring -- and the GPU, not the CPU, counts the swaps a copy lands
// between.
//
// With a host GPU attached (`Gpu::attach_backend`), what the GPU draws stays
// on it (ADR-006): the surfaces it renders to live in a surface cache as
// textures, and the work on them is done there -- the clear, and the scan-out
// copy of a surface the GPU holds, which fills a scan frame's texture rather
// than its bytes; a flip then hands the frontend that texture. Without one,
// as in most tests, all of it is done on guest memory, on the CPU.
//
// What is not here yet: swap interval 0's flip-without-waiting (it waits for
// vsync like 1), the overrun callback a full display list asks for, and the
// profiling and toss-stage modes `GX2Init` can ask for.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/surface_cache.hpp>
#include <wiiu/latte/surface_copy.hpp>
#include <wiiu/latte/timeline.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace wiiu::hle {

class Registry;
struct Call;

namespace coreinit {
struct Context;
} // namespace coreinit

namespace gx2 {

struct ColorBuffer; // gx2_surface.hpp

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

/// The output a `GX2ScanTarget` names: 1 is the TV and 4 the GamePad (wut's
/// and decaf's values). Nothing for anything else.
[[nodiscard]] std::optional<ScanTarget> scan_target_from_gx2(u32 value) noexcept;

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
/// a little larger where the height needs padding (1080 rows become 1088);
/// that one comes from Latte's surface layout (8.5). The frames here are the
/// emulator's own -- nothing but the emulator reads a scan buffer -- so the
/// only thing the figure has to be is enough room for them, which it is.
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

/// Where frame `flip_index` (counting flips from zero) of a scan buffer is:
/// the frames rotate, one per flip, as many as the buffering mode holds.
/// Nothing for a buffer that is not set, or whose format is narrower than the
/// four bytes a pixel is presented from -- a 16-bit scan buffer is legal and
/// nobody uses one, and showing it as RGBA8 would read past its frames.
[[nodiscard]] std::optional<ppcaddr_t> scan_frame(const ScanBuffer& buffer,
                                                  u32 flip_index) noexcept;

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

    /// The GPU's textures go back to the backend they were made on, which
    /// therefore has to outlive this -- or be detached first.
    ~Gpu();

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

    /// Swaps the GPU has reached in stream order: the `HleSwap` packets that
    /// have run. A scan-out copy lands in the frame the next of them will
    /// show -- frame `swaps_reached()` of the rotation, the same count the
    /// flips take their frames by.
    [[nodiscard]] u32 swaps_reached() const noexcept { return swaps_reached_; }

    // --- the host GPU (ADR-006) ---------------------------------------------

    /// Do the GPU's work on `backend` from here on, or, with null, on guest
    /// memory again. Either way the surface cache starts empty and the scan
    /// frames' textures are gone: what the GPU held is destroyed on the
    /// backend it was made on, so a backend must be detached -- this with
    /// null -- before it goes, and whoever was showing a scan texture must
    /// stop first.
    void attach_backend(gpu::GpuBackend* backend);

    [[nodiscard]] gpu::GpuBackend* backend() const noexcept { return backend_; }

    /// The surfaces the GPU holds, or none without a backend.
    [[nodiscard]] latte::SurfaceCache* surfaces() noexcept { return surfaces_.get(); }

    [[nodiscard]] const latte::SurfaceCache* surfaces() const noexcept { return surfaces_.get(); }

    /// What frame `flip_index` of `target` holds, as a texture, when the copy
    /// that last filled it ran on the GPU -- then that texture, not the
    /// frame's bytes in guest memory, is what a flip of it shows. None when
    /// guest memory holds it, or nothing has filled it.
    [[nodiscard]] gpu::TextureHandle scan_texture_shown(ScanTarget target,
                                                        u32 flip_index) const noexcept;

    /// Back to before `GX2Init`, with nothing in flight: the register file
    /// cleared, a fresh timeline started, and the surface cache emptied. The
    /// command processor's listeners and counters are the emulator's, and
    /// stay, and so does the backend.
    void reset();

private:
    bool write_display_list(DisplayList& list, std::span<const u32> words);

    /// One of the emulator's own packets, as it runs.
    void run_hle(const latte::pm4::Packet& packet);

    /// Log a problem with a packet the first time it is seen: a copy that
    /// cannot be done fails the same way every frame.
    void report(const std::string& problem);

    /// A scan-out copy into frame `index` of `target`, on the GPU, if the GPU
    /// holds the colour buffer's surface. False if it does not, and the copy
    /// is guest memory's to do.
    bool copy_on_gpu(ScanTarget target, u32 index, const ColorBuffer& buffer);

    /// Frame `index` of `target` as a texture, made on first use, as large
    /// as the scan buffer's picture. A scan buffer set anew gets new
    /// textures; the old ones may still be on screen, so they are kept until
    /// the GPU's textures all go.
    gpu::TextureHandle scan_texture(ScanTarget target, u32 index);

    /// Note where frame `index` of `target` was last filled.
    void mark_scan_frame(ScanTarget target, u32 index, bool on_gpu);

    /// Destroy every scan texture, the retired ones too.
    void release_scan_textures();

    /// A scan buffer's frames as textures.
    struct ScanTextures {
        std::vector<gpu::TextureHandle> frames;
        std::vector<bool> on_gpu; ///< filled last by a copy on the GPU
        u32 width = 0;
        u32 height = 0;

        [[nodiscard]] bool describe(const ScanBuffer& scan) const noexcept;
    };

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
    u32 swaps_reached_ = 0;
    std::set<std::string> reported_;

    gpu::GpuBackend* backend_ = nullptr;
    std::unique_ptr<latte::SurfaceCache> surfaces_;
    latte::SurfaceCopier copier_;
    std::array<ScanTextures, scan_target_count> scan_textures_{};
    std::vector<gpu::TextureHandle> retired_textures_;

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

/// Write packets for the calling core -- its open display list, or the main
/// command buffer -- and keep the GPU's timer armed if that submitted
/// anything. What every GX2 call that sends the GPU something goes through.
bool emit(Call& call, std::span<const u32> words);

/// Where the context registers start: SET_CONTEXT_REG addresses them in
/// words from here.
inline constexpr u32 context_register_base = 0x28000;

/// `SET_CONTEXT_REG` for consecutive registers from the byte address
/// `address`, through `emit`.
bool emit_context_registers(Call& call, u32 address, std::span<const u32> values);

/// Append a register-window write -- any `SET_*` opcode, or
/// `SET_ALL_CONTEXTS` -- for consecutive registers from the byte address
/// `address`, which must be in that opcode's window. For
/// state built up before it is emitted: the default state, and calls that
/// write more than one packet.
void append_registers(std::vector<u32>& out,
                      latte::pm4::Opcode opcode,
                      u32 address,
                      std::span<const u32> values);

/// `SET_CONTEXT_REG` of one register.
void append_context_register(std::vector<u32>& out, u32 address, u32 value);

/// Call a display list in the title's memory, `bytes` long, with
/// `INDIRECT_BUFFER_PRIV` through `emit` -- `GX2CallDisplayList`. False,
/// having logged why naming `what`, if the GPU cannot see it.
bool emit_display_list_call(Call& call, ppcaddr_t list, u32 bytes, std::string_view what);

/// The physical address GX2 writes into PM4 for something in the title's
/// memory, or nothing -- logged, naming `what` -- if the GPU cannot see it.
[[nodiscard]] std::optional<physaddr_t>
physical_address(const memory::Memory& memory, ppcaddr_t address, std::string_view what);

/// `GX2InvalidateMode`: what `GX2Invalidate` is told was written, and so
/// which caches must forget it.
namespace invalidate {
inline constexpr u32 attribute_buffer = 1U << 0;
inline constexpr u32 texture = 1U << 1;
inline constexpr u32 uniform_block = 1U << 2;
inline constexpr u32 shader = 1U << 3;
inline constexpr u32 color_buffer = 1U << 4;
inline constexpr u32 depth_buffer = 1U << 5;
inline constexpr u32 cpu = 1U << 6;
inline constexpr u32 stream_out_buffer = 1U << 7;
inline constexpr u32 export_buffer = 1U << 8;
} // namespace invalidate

/// The SURFACE_SYNC `GX2Invalidate` writes for `mode` over `size` bytes at
/// physical `address`: decaf's choice of cache actions and destinations,
/// which Cemu's agrees with but for one -- decaf also asks for the full
/// cache when the whole of memory is meant (a null address and a size of
/// all ones) and a GPU read cache is involved. Empty when the mode asks
/// nothing of the GPU: zero, or the CPU's cache alone.
[[nodiscard]] std::vector<u32> invalidate_packets(u32 mode, physaddr_t address, u32 size);

/// A structure of the title's -- made of `u32be` and `f32be` fields, laid
/// out as GX2 lays it out -- copied out of guest memory or into it. False if
/// the memory is not there.
template <typename T>
[[nodiscard]] bool read_guest(const memory::Memory& memory, ppcaddr_t address, T& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<u8, sizeof(T)> bytes{};
    if (!memory.read_bytes(address, bytes)) {
        return false;
    }
    std::memcpy(&out, bytes.data(), sizeof(T));
    return true;
}

template <typename T>
[[nodiscard]] bool write_guest(memory::Memory& memory, ppcaddr_t address, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<u8, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    return memory.write_bytes(address, bytes);
}

/// Register the GX2 entry points under the `gx2` library.
void register_gx2(Registry& registry, coreinit::Context& context);

/// The display half: scan buffers, the outputs' switches and what they
/// report. Called by `register_gx2`.
void register_gx2_display(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
