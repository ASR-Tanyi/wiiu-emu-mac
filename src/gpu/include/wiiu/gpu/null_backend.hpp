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

#include <map>
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

    [[nodiscard]] FrameStats last_frame_stats() const override { return last_stats_; }

    [[nodiscard]] BufferHandle create_buffer(const BufferDesc& desc) override;
    [[nodiscard]] TextureHandle create_texture(const TextureDesc& desc) override;
    [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc& desc) override;
    [[nodiscard]] PipelineHandle create_pipeline(const PipelineDesc& desc) override;

    void destroy(BufferHandle handle) override;
    void destroy(TextureHandle handle) override;
    void destroy(SamplerHandle handle) override;
    void destroy(PipelineHandle handle) override;
    void destroy(FenceHandle handle) override;

    [[nodiscard]] bool
    write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) override;
    [[nodiscard]] void* buffer_contents(BufferHandle handle) override;

    [[nodiscard]] bool begin_render_pass(const RenderPassDesc& desc) override;
    void end_render_pass() override;

    void set_pipeline(PipelineHandle pipeline) override;
    void set_viewport(const Viewport& viewport) override;
    void set_scissor(const Rect& rect) override;

    void bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) override;
    void bind_texture(ShaderStage stage, u32 index, TextureHandle texture) override;
    void bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) override;

    void draw(const DrawCall& call) override;

    [[nodiscard]] FenceHandle create_fence() override;
    void signal(FenceHandle handle) override;
    [[nodiscard]] bool is_signalled(FenceHandle handle) override;
    void wait(FenceHandle handle) override;

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

    [[nodiscard]] const RenderPassDesc& last_render_pass() const noexcept {
        return last_render_pass_;
    }

    [[nodiscard]] bool in_render_pass() const noexcept { return in_render_pass_; }

    /// The capabilities a test wants to pretend this machine has.
    void set_capabilities(Capabilities capabilities) { capabilities_ = std::move(capabilities); }

    void clear_events() { events_.clear(); }

private:
    void record(std::string line);
    void reject(std::string line);

    [[nodiscard]] u32 next_id(ResourceKind kind);

    [[nodiscard]] bool live(ResourceKind kind, u32 id) const;
    void kill(ResourceKind kind, u32 id, std::string_view what);

    Capabilities capabilities_;
    NativeSurface surface_;
    bool initialized_ = false;
    bool vsync_ = true;
    bool in_render_pass_ = false;
    bool in_frame_ = false;

    std::vector<std::string> events_;
    u64 rejected_ = 0;
    u64 frame_index_ = 0;

    FrameStats stats_;
    FrameStats last_stats_;
    RenderPassDesc last_render_pass_;

    std::array<u32, 5> next_id_{};
    std::array<std::map<u32, std::string>, 5> live_;

    /// Buffer contents, so `write_buffer` and `buffer_contents` mean
    /// something and a test can check what the translator uploaded.
    std::map<u32, std::vector<u8>> buffer_bytes_;

    /// The drawable handed out by `begin_frame`, recreated per backend rather
    /// than per frame: a real swapchain rotates through a small set, and a
    /// test that holds one across frames should not be surprised.
    TextureHandle drawable_;

    std::map<u32, bool> fences_;
};

} // namespace wiiu::gpu
