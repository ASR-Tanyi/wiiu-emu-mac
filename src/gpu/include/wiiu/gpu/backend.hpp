// Phase 7.1 — the GPU backend interface.
//
// One rule, and everything else follows from it: **no Metal type crosses this
// header**. Not a `MTL::Device*`, not a `MTLPixelFormat`, not a
// `CA::MetalLayer*`. The Phase 8 command processor is written against this
// interface and never learns which API is underneath, which is what makes a
// future Vulkan port — or, much more likely, a Metal 4 rewrite — a new file
// rather than a new emulator.
//
// It is also what makes the GPU testable without a GPU. `create_null_backend`
// returns an implementation that records what it was asked to do and draws
// nothing, so the translator's output can be checked on any machine, in CI,
// without a window.
//
// The binding model is *not* in this interface. ADR-003 chose argument
// buffers Tier 2, and the choice lives in the backend: callers bind by stage
// and index, and whether that index reaches the shader through a binding slot
// or a descriptor in an argument buffer is the backend's business.
#pragma once

#include <wiiu/gpu/types.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace wiiu::gpu {

/// What this machine can actually do.
///
/// Filled in by the backend at startup (7.3). The point is `sufficient()`:
/// refusing to start with a clear message beats rendering garbage, and beats
/// Cemu's behaviour of silently skipping the draws it cannot do — which
/// reaches the user as an invisible enemy rather than as an error.
struct Capabilities {
    std::string device_name;
    std::string api_version;

    bool metal3 = false;

    /// GPU family Apple7. Required: the geometry path and Latte's RECTS both
    /// go through object+mesh pipelines (8.8), and there is no fallback.
    bool mesh_shaders = false;

    /// GPU family Apple9. Optional, and worth knowing about: it allows mesh
    /// draws to be issued indirectly, which is what a PM4 stream would rather
    /// do than round-trip through the CPU.
    bool indirect_mesh_draws = false;

    /// Apple2+. The route to shader logic ops, which Metal does not expose
    /// outside private API (Phase 7 table).
    bool programmable_blending = false;

    /// Apple4+. Tile shaders, for resolves and reductions that never leave
    /// tile memory.
    bool tile_shaders = false;

    /// 2 means bindless is available, which ADR-003 depends on.
    u32 argument_buffer_tier = 0;

    /// CPU and GPU address the same bytes. True on all Apple silicon, and the
    /// premise of **P22**.
    bool unified_memory = false;

    u32 max_texture_size_2d = 0;

    /// Is the device wrapped by Metal's API validation layer (7.8)? Not a
    /// capability, but the first thing a GPU bug report should say: a crash
    /// with validation off and the same crash with it on are different
    /// reports.
    bool api_validation = false;

    /// Everything this machine is missing that the emulator cannot do
    /// without, in a sentence a user can act on. Empty when it can run.
    [[nodiscard]] std::string shortfall() const;

    [[nodiscard]] bool sufficient() const { return shortfall().empty(); }

    /// The whole report, for the log and for a bug report.
    [[nodiscard]] std::string describe() const;
};

/// A window's drawable surface, handed over by the platform layer.
///
/// `handle` is a `CAMetalLayer*` for the Metal backend and nothing at all for
/// the null one. It is `void*` precisely so that this header does not have to
/// know which — the platform layer creates it and the backend recognises it,
/// and nobody in between needs the type.
struct NativeSurface {
    void* handle = nullptr;
    u32 width = 0;
    u32 height = 0;

    [[nodiscard]] bool valid() const noexcept { return handle != nullptr; }
};

struct BufferDesc {
    u64 size = 0;
    Storage storage = Storage::Private;
    std::string_view label;
};

struct TextureDesc {
    u32 width = 0;
    u32 height = 0;
    u32 depth = 1;
    u32 mip_levels = 1;
    u32 array_length = 1;
    u32 sample_count = 1;
    TextureFormat format = TextureFormat::Unknown;
    TextureUsage usage = TextureUsage::Sampled;
    Storage storage = Storage::Private;
    std::string_view label;
};

/// Sampler state.
///
/// `border_color` carries the exact colour the guest asked for, not one of
/// the three Metal allows. A backend that cannot honour it sets
/// `border_color_emulated` on the sampler it returns, and the shader analysis
/// pass (8.7) picks that up and emulates the border in-shader. Rounding it to
/// the nearest opaque black here would be a silent wrong answer.
struct SamplerDesc {
    Filter min_filter = Filter::Linear;
    Filter mag_filter = Filter::Linear;
    Filter mip_filter = Filter::Linear;
    AddressMode address_u = AddressMode::Repeat;
    AddressMode address_v = AddressMode::Repeat;
    AddressMode address_w = AddressMode::Repeat;
    f32 lod_min = 0.0F;
    f32 lod_max = 1000.0F;
    u32 max_anisotropy = 1;
    CompareFunction compare = CompareFunction::Never;
    bool compare_enabled = false;
    ClearColor border_color{0.0F, 0.0F, 0.0F, 0.0F};
    std::string_view label;
};

struct ColorAttachment {
    TextureHandle texture;
    TextureHandle resolve_texture;
    u32 level = 0;
    u32 slice = 0;
    LoadAction load = LoadAction::Unspecified;
    StoreAction store = StoreAction::Unspecified;
    ClearColor clear;
};

struct DepthStencilAttachment {
    TextureHandle texture;
    LoadAction load = LoadAction::Unspecified;
    StoreAction store = StoreAction::Unspecified;
    f32 clear_depth = 1.0F;
    u32 clear_stencil = 0;
};

struct RenderPassDesc {
    std::array<ColorAttachment, max_color_attachments> colors{};
    u32 color_count = 0;
    DepthStencilAttachment depth_stencil;
    std::string_view label;

    /// Why this pass cannot be begun, or empty if it can.
    ///
    /// Checked by every backend before anything is encoded, so an unset load
    /// or store action (**P20**) is caught at the call site rather than
    /// becoming a correct-but-slow frame nobody ever investigates.
    [[nodiscard]] std::string validate() const;
};

/// Enough of a pipeline to clear a screen and draw a triangle. Phase 8.11
/// grows this into the full Latte state translation — blend, depth/stencil,
/// rasterizer, vertex fetch — and this is deliberately not that yet.
///
/// There is no vertex descriptor, and there will not be one: Latte's vertex
/// fetch has no alignment rules and Metal's has, so vertices are fetched by
/// the shader itself from bound buffers, indexed by `[[vertex_id]]` (the
/// Phase 7 table).
///
/// Shaders are compiled with fast math **off**. Metal's default is on, which
/// lets the compiler reassociate, contract and assume no NaNs -- the GPU-side
/// twin of the `-ffp-contract=off` rule for guest CPU arithmetic. Latte's
/// shaders have IEEE and DX9 multiplies that differ exactly where fast math
/// takes liberties (0 * inf), so the decompiler (8.6) has to be able to say
/// which one it meant and be obeyed.
struct PipelineDesc {
    std::string_view label;
    std::string vertex_source;   ///< MSL, as 8.6 emits it
    std::string fragment_source; ///< MSL
    std::string vertex_entry = "vertex_main";
    std::string fragment_entry = "fragment_main";
    PrimitiveType primitive = PrimitiveType::Triangles;
    std::array<TextureFormat, max_color_attachments> color_formats{};
    u32 color_count = 1;
    TextureFormat depth_format = TextureFormat::Unknown;
    u32 sample_count = 1;

    [[nodiscard]] std::string validate() const;
};

struct DrawCall {
    u32 vertex_count = 0;
    u32 instance_count = 1;
    u32 first_vertex = 0;
    u32 first_instance = 0;

    /// Set for an indexed draw. `index_count` then replaces `vertex_count`.
    BufferHandle index_buffer;
    u64 index_offset = 0;
    u32 index_count = 0;
    bool index_is_32bit = false;
};

/// What `begin_frame` hands back.
struct FrameInfo {
    TextureHandle drawable;
    u32 width = 0;
    u32 height = 0;
    TextureFormat format = TextureFormat::Unknown;
    u64 index = 0; ///< counts up for the life of the backend
};

/// The GPU, as the rest of the emulator sees it.
///
/// Called from the GPU thread (8.2) and from nowhere else. Methods that can
/// fail say so; methods that create resources return an invalid handle and
/// log, rather than throwing, because a failed texture should degrade a frame
/// and not end a session.
class GpuBackend {
public:
    GpuBackend() = default;
    virtual ~GpuBackend() = default;

    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;

    [[nodiscard]] virtual std::string_view name() const = 0;

    // --- lifecycle --------------------------------------------------------

    /// Create the device and queue. False means this machine cannot run the
    /// emulator, and `error` says why in a sentence worth showing a user.
    [[nodiscard]] virtual bool initialize(std::string& error) = 0;

    virtual void shutdown() = 0;

    [[nodiscard]] virtual const Capabilities& capabilities() const = 0;

    // --- swapchain --------------------------------------------------------

    /// Attach to a window's surface. Called again on resize.
    [[nodiscard]] virtual bool attach_surface(const NativeSurface& surface, std::string& error) = 0;

    virtual void set_vsync(bool enabled) = 0;

    /// Acquire the next drawable. False means there was none to be had this
    /// frame — the window is occluded or being resized — which is a reason to
    /// skip a frame, not an error.
    [[nodiscard]] virtual bool begin_frame(FrameInfo& out) = 0;

    /// Present, and make the frame's resources reclaimable.
    virtual void end_frame() = 0;

    /// Commit whatever has been recorded, without presenting anything.
    ///
    /// Rendering does not require a window. The headless compatibility runner
    /// (10.5) boots titles with no swapchain at all and hashes what they
    /// drew, and a save-state test (11.4) does the same — so a render pass
    /// outside a frame is an ordinary thing to want, not an edge case.
    virtual void submit() = 0;

    /// Block until everything submitted has finished.
    ///
    /// Unlike `wait(FenceHandle)` this is not something a guest ever asks
    /// for: it is for shutdown, and for reading back what was drawn.
    virtual void wait_idle() = 0;

    [[nodiscard]] virtual FrameStats last_frame_stats() const = 0;

    /// Command buffers the GPU reported as failed, ever: a fault, a timeout,
    /// a shader-validation error (7.8). Should be zero, and the GPU tests
    /// say so -- nothing else would notice, because a failed buffer leaves
    /// the CPU side looking exactly as if it had worked.
    [[nodiscard]] virtual u64 command_errors() const = 0;

    // --- resources --------------------------------------------------------

    [[nodiscard]] virtual BufferHandle create_buffer(const BufferDesc& desc) = 0;
    [[nodiscard]] virtual TextureHandle create_texture(const TextureDesc& desc) = 0;
    [[nodiscard]] virtual SamplerHandle create_sampler(const SamplerDesc& desc) = 0;

    /// Compile and link a pipeline. On failure `error` holds the compiler's
    /// own words, with each line it complains about quoted from the source --
    /// which is what makes a decompiler bug (8.6) a five-minute fix instead of
    /// a line number to count to.
    [[nodiscard]] virtual PipelineHandle create_pipeline(const PipelineDesc& desc,
                                                         std::string& error) = 0;

    virtual void destroy(BufferHandle handle) = 0;
    virtual void destroy(TextureHandle handle) = 0;
    virtual void destroy(SamplerHandle handle) = 0;
    virtual void destroy(PipelineHandle handle) = 0;

    /// Copy into a `Shared` buffer. False if the handle is unknown or the
    /// write would run past the end.
    [[nodiscard]] virtual bool
    write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) = 0;

    /// Copy a texture's pixels back to the host, tightly packed.
    ///
    /// Only for `Shared` storage, and it waits for the GPU first — so this is
    /// a debugging and testing path, not something a frame should do. It is
    /// what 10.5 hashes framebuffers with, and what proves a clear actually
    /// cleared.
    [[nodiscard]] virtual bool read_texture(TextureHandle handle, std::span<u8> out) = 0;

    /// The host pointer behind a `Shared` buffer, or null.
    ///
    /// This is the zero-copy path (**P22**): on unified memory a guest-owned
    /// range can be the buffer rather than be copied into one.
    [[nodiscard]] virtual void* buffer_contents(BufferHandle handle) = 0;

    /// Copy pixels into level 0 of a `Shared` texture, rows `bytes_per_row`
    /// apart in `data`.
    ///
    /// Immediate and CPU-side, like `write_buffer`, so the caller owns the
    /// hazard: a texture a frame still in flight is sampling must not be
    /// written. Rotating through `frames_in_flight` copies is the usual answer
    /// (the presenter does). A `Private` texture is refused -- that upload is
    /// a blit, and blits belong to the GPU thread (8.2).
    [[nodiscard]] virtual bool write_texture(TextureHandle handle,
                                             const Rect& region,
                                             std::span<const u8> data,
                                             u32 bytes_per_row) = 0;

    /// Wrap memory the emulator already owns as a `Shared` buffer, without
    /// copying it (**P22**).
    ///
    /// This is how guest RAM becomes GPU-visible on unified memory: the bytes
    /// the guest writes are the bytes the GPU reads, with no staging buffer
    /// and no upload. `pointer` and `size` must be multiples of
    /// `host_page_size()`, which Metal requires and the guest regions already
    /// are. The memory must outlive the buffer; destroying the buffer does not
    /// free it.
    ///
    /// Zero-copy is not always right. Anything the CPU will overwrite while a
    /// frame that reads it is still in flight needs a copy instead -- which is
    /// why OSScreen's scanout copies at the flip (7.6) and GX2's buffers, which
    /// carry their own synchronisation, are what this is for (8.2).
    [[nodiscard]] virtual BufferHandle
    import_host_memory(void* pointer, u64 size, std::string_view label) = 0;

    [[nodiscard]] virtual MemoryStats memory_stats() const = 0;

    // --- encoding ---------------------------------------------------------

    /// Begin a render pass. False if `desc.validate()` refused it, which has
    /// already been logged.
    [[nodiscard]] virtual bool begin_render_pass(const RenderPassDesc& desc) = 0;
    virtual void end_render_pass() = 0;

    /// The pipeline, and every binding below, belong to the render pass they
    /// were made in: Metal encodes them into the pass's encoder, so a new pass
    /// starts with nothing bound. Both backends refuse them outside a pass
    /// rather than remembering them for one.
    virtual void set_pipeline(PipelineHandle pipeline) = 0;
    virtual void set_viewport(const Viewport& viewport) = 0;
    virtual void set_scissor(const Rect& rect) = 0;

    /// Bind by stage and index. Until ADR-003's argument buffers arrive with
    /// the MSL emitter (8.6, 8.12), index `n` is `[[buffer(n)]]`,
    /// `[[texture(n)]]` or `[[sampler(n)]]` in that stage's function, up to
    /// `max_bound_buffers`, `max_bound_textures` and `max_bound_samplers`.
    virtual void bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) = 0;
    virtual void bind_texture(ShaderStage stage, u32 index, TextureHandle texture) = 0;
    virtual void bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) = 0;

    virtual void draw(const DrawCall& call) = 0;

    // --- synchronisation --------------------------------------------------

    [[nodiscard]] virtual FenceHandle create_fence() = 0;
    virtual void destroy(FenceHandle handle) = 0;

    /// Signal once the work submitted so far has completed.
    virtual void signal(FenceHandle handle) = 0;

    /// Has it been signalled? Asking is free; waiting is not.
    [[nodiscard]] virtual bool is_signalled(FenceHandle handle) = 0;

    /// Block until signalled. **P4**: only ever called where the guest itself
    /// asked to wait. Every other wait is a bug, and the time spent here
    /// lands in `FrameStats::cpu_stall_us` so that it is visible when it is.
    virtual void wait(FenceHandle handle) = 0;

    // --- tooling (7.8) ----------------------------------------------------

    /// Start writing a GPU capture to `path` (a `.gputrace` on Metal), which
    /// opens in Xcode's Metal debugger. Everything submitted until
    /// `end_capture` is in it. False, with the reason, if this process cannot
    /// capture -- on macOS that usually means `MTL_CAPTURE_ENABLED=1` was not
    /// set before the device was created.
    [[nodiscard]] virtual bool begin_capture(std::string_view path, std::string& error) = 0;

    virtual void end_capture() = 0;
};

/// The host's virtual memory page size: what `import_host_memory` wants its
/// pointer and size aligned to. 16 KiB on Apple silicon.
[[nodiscard]] u64 host_page_size() noexcept;

/// A backend that records what it is told and draws nothing.
///
/// Not a stub: it is how the Phase 8 translator gets tested without a GPU,
/// how the headless compatibility runner (10.5) will run, and how this
/// interface is exercised on a machine that has no Metal at all.
[[nodiscard]] std::unique_ptr<GpuBackend> create_null_backend();

/// The Metal backend, or null on a platform without one.
[[nodiscard]] std::unique_ptr<GpuBackend> create_metal_backend();

} // namespace wiiu::gpu
