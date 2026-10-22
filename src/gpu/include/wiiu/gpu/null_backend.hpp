// Phase 7.1 — a backend that records instead of rendering.
//
// Every interface wants one implementation that cannot fail, so that the
// things written against it can be tested without the thing underneath. Here
// that is worth more than usual: the Phase 8 command processor turns PM4 into
// backend calls, and "did this packet produce the right draw" is a question
// about the calls, not about the pixels. It also answers it on a Linux CI
// runner, which has no Metal at all.
//
// It is not a stub. It validates everything a real backend validates, tracks
// which handles are live, and complains about the same mistakes -- so code
// that works here is code that will not immediately fall over on Metal.
#pragma once

#include <wiiu/gpu/backend.hpp>

#include <array>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace wiiu::gpu {

class NullBackend final : public GpuBackend {
public:
    NullBackend();

    [[nodiscard]] std::string_view name() const override { return "null"; }

    [[nodiscard]] bool initialize(std::string& error) override;
    void shutdown() override;

    [[nodiscard]] const Capabilities& capabilities() const override { return capabilities_; }

    [[nodiscard]] bool attach_surface(const NativeSurface& surface, std::string& error) override;
    void set_vsync(bool enabled) override;
    [[nodiscard]] bool begin_frame(FrameInfo& out) override;
    void end_frame() override;
    void submit() override;
    void wait_idle() override;

    [[nodiscard]] FrameStats last_frame_stats() const override { return last_stats_; }

    /// Nothing runs, so nothing fails on a GPU.
    [[nodiscard]] u64 command_errors() const override { return 0; }

    [[nodiscard]] BufferHandle create_buffer(const BufferDesc& desc) override;
    [[nodiscard]] TextureHandle create_texture(const TextureDesc& desc) override;
    [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc& desc) override;
    [[nodiscard]] PipelineHandle create_pipeline(const PipelineDesc& desc,
                                                 std::string& error) override;

    void destroy(BufferHandle handle) override;
    void destroy(TextureHandle handle) override;
    void destroy(SamplerHandle handle) override;
    void destroy(PipelineHandle handle) override;
    void destroy(DepthStencilHandle handle) override;
    [[nodiscard]] DepthStencilHandle create_depth_stencil(const DepthStencilDesc& desc) override;
    void destroy(FenceHandle handle) override;

    [[nodiscard]] bool
    write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) override;
    [[nodiscard]] void* buffer_contents(BufferHandle handle) override;
    [[nodiscard]] bool read_texture(TextureHandle handle, std::span<u8> out) override;
    [[nodiscard]] bool write_texture(TextureHandle handle,
                                     const Rect& region,
                                     std::span<const u8> data,
                                     u32 bytes_per_row) override;
    [[nodiscard]] BufferHandle
    import_host_memory(void* pointer, u64 size, std::string_view label) override;
    [[nodiscard]] MemoryStats memory_stats() const override;

    [[nodiscard]] bool begin_render_pass(const RenderPassDesc& desc) override;
    void end_render_pass() override;

    void set_pipeline(PipelineHandle pipeline) override;
    void set_viewport(const Viewport& viewport) override;
    void set_scissor(const Rect& rect) override;
    void set_cull_mode(CullMode mode) override;
    void set_front_facing(Winding winding) override;
    void set_blend_color(const ClearColor& color) override;
    void set_depth_stencil(DepthStencilHandle state) override;
    void set_stencil_reference(u32 front, u32 back) override;

    void bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) override;
    void bind_bytes(ShaderStage stage, u32 index, std::span<const u8> bytes) override;
    void bind_texture(ShaderStage stage, u32 index, TextureHandle texture) override;
    void bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) override;

    void draw(const DrawCall& call) override;

    [[nodiscard]] FenceHandle create_fence() override;
    void signal(FenceHandle handle) override;
    [[nodiscard]] bool is_signalled(FenceHandle handle) override;
    void wait(FenceHandle handle) override;

    [[nodiscard]] bool begin_capture(std::string_view path, std::string& error) override;
    void end_capture() override;

    // --- what the tests look at -------------------------------------------

    /// Every call, in order, one line each. A test asserts on this the way it
    /// would read a trace.
    [[nodiscard]] const std::vector<std::string>& events() const noexcept { return events_; }

    /// Does the event list contain a line with this substring?
    [[nodiscard]] bool recorded(std::string_view fragment) const;

    [[nodiscard]] std::size_t live_count(ResourceKind kind) const;

    /// Calls that were refused: an unknown handle, a pass that failed
    /// validation, a write past the end of a buffer. Should be zero.
    [[nodiscard]] u64 rejected() const noexcept { return rejected_; }

    /// What the last `bind_bytes` was given, so a test can read the
    /// constants a draw was handed.
    [[nodiscard]] const std::vector<u8>& last_bytes() const noexcept { return last_bytes_; }

    [[nodiscard]] const RenderPassDesc& last_render_pass() const noexcept {
        return last_render_pass_;
    }

    [[nodiscard]] bool in_render_pass() const noexcept { return in_render_pass_; }

    [[nodiscard]] bool capturing() const noexcept { return capturing_; }

    /// The capabilities a test wants to pretend this machine has.
    void set_capabilities(Capabilities capabilities) { capabilities_ = std::move(capabilities); }

    void clear_events() { events_.clear(); }

private:
    void record(std::string line);
    void reject(std::string line);

    /// Encoding calls are made inside a render pass or not at all. False,
    /// having rejected the call, if there is no pass open.
    [[nodiscard]] bool require_pass(std::string_view what);

    /// A binding index inside the stage's table. False, having rejected it,
    /// if it is not.
    [[nodiscard]] bool check_binding(std::string_view what, u32 index, u32 limit);

    [[nodiscard]] u32 next_id(ResourceKind kind);

    [[nodiscard]] bool live(ResourceKind kind, u32 id) const;
    void kill(ResourceKind kind, u32 id, std::string_view what);

    Capabilities capabilities_;
    NativeSurface surface_;
    bool initialized_ = false;
    bool vsync_ = true;
    bool in_render_pass_ = false;
    bool in_frame_ = false;
    bool capturing_ = false;

    /// The pipeline set in the current pass. Cleared when a pass begins,
    /// because on Metal a new encoder starts with nothing set.
    PipelineHandle pass_pipeline_;

    /// The attachment formats each pipeline was built for, and each
    /// texture's format: Metal refuses a pipeline in a pass whose
    /// attachments are not those formats, and so does this.
    struct AttachmentFormats {
        std::array<TextureFormat, max_color_attachments> colors{};
        u32 color_count = 0;
        TextureFormat depth = TextureFormat::Unknown;
    };

    std::map<u32, AttachmentFormats> pipeline_formats_;
    std::map<u32, TextureFormat> texture_formats_;

    [[nodiscard]] TextureFormat format_of(TextureHandle texture) const;

    std::vector<std::string> events_;
    std::vector<u8> last_bytes_;
    u64 rejected_ = 0;
    u64 frame_index_ = 0;

    FrameStats stats_;
    FrameStats last_stats_;
    RenderPassDesc last_render_pass_;

    std::array<u32, resource_kind_count> next_id_{};
    std::array<std::map<u32, std::string>, resource_kind_count> live_;

    /// Buffer contents, so `write_buffer` and `buffer_contents` mean
    /// something and a test can check what the translator uploaded.
    std::map<u32, std::vector<u8>> buffer_bytes_;

    /// Buffers wrapping host memory (`import_host_memory`): the bytes are
    /// the caller's, so they are remembered rather than copied.
    std::map<u32, std::span<u8>> imported_;

    /// Texture contents, so a clear is something a test can look at rather
    /// than only something it can see recorded. The headless compatibility
    /// runner (10.5) hashes framebuffers; this is the same question asked of
    /// a backend with no GPU.
    struct TexturePixels {
        u32 width = 0;
        u32 height = 0;
        TextureFormat format = TextureFormat::Unknown;
        Storage storage = Storage::Private;
        std::vector<u8> bytes;
    };

    std::map<u32, TexturePixels> texture_pixels_;

    void clear_pixels(TextureHandle handle, const ClearColor& color);

    /// The drawable handed out by `begin_frame`, recreated per backend rather
    /// than per frame: a real swapchain rotates through a small set, and a
    /// test that holds one across frames should not be surprised.
    TextureHandle drawable_;

    std::map<u32, bool> fences_;
};

} // namespace wiiu::gpu
