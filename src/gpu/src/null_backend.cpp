// Phase 7.1 — the recording backend.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/null_backend.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <utility>

namespace wiiu::gpu {

namespace {

[[nodiscard]] std::size_t index_of(ResourceKind kind) noexcept {
    return static_cast<std::size_t>(kind);
}

[[nodiscard]] std::string label_or(std::string_view label, std::string_view fallback) {
    return label.empty() ? std::string{fallback} : std::string{label};
}

} // namespace

NullBackend::NullBackend() {
    // Enough to satisfy `sufficient()`. A headless run should exercise the
    // same paths a real one does, not the refusal path -- and a test that
    // wants the refusal path sets its own capabilities.
    capabilities_.device_name = "null device";
    capabilities_.api_version = "none";
    capabilities_.metal3 = true;
    capabilities_.mesh_shaders = true;
    capabilities_.indirect_mesh_draws = true;
    capabilities_.programmable_blending = true;
    capabilities_.tile_shaders = true;
    capabilities_.argument_buffer_tier = 2;
    capabilities_.unified_memory = true;
    capabilities_.max_texture_size_2d = 16384;

    next_id_.fill(1);
}

void NullBackend::record(std::string line) {
    WIIU_LOG_TRACE(Gpu, "null: {}", line);
    events_.push_back(std::move(line));
}

void NullBackend::reject(std::string line) {
    ++rejected_;
    WIIU_LOG_WARN(Gpu, "null: rejected: {}", line);
    events_.push_back("rejected: " + std::move(line));
}

u32 NullBackend::next_id(ResourceKind kind) {
    return next_id_[index_of(kind)]++;
}

bool NullBackend::live(ResourceKind kind, u32 id) const {
    const auto& table = live_[index_of(kind)];
    return id != 0 && table.find(id) != table.end();
}

void NullBackend::kill(ResourceKind kind, u32 id, std::string_view what) {
    auto& table = live_[index_of(kind)];
    const auto entry = table.find(id);
    if (id == 0 || entry == table.end()) {
        reject(fmt::format("destroy {} {}: not a live handle", what, id));
        return;
    }
    record(fmt::format("destroy {} {} ({})", what, id, entry->second));
    table.erase(entry);
    if (kind == ResourceKind::Buffer) {
        buffer_bytes_.erase(id);
    }
    if (kind == ResourceKind::Fence) {
        fences_.erase(id);
    }
}

bool NullBackend::recorded(std::string_view fragment) const {
    return std::any_of(events_.begin(), events_.end(), [fragment](const std::string& line) {
        return line.find(fragment) != std::string::npos;
    });
}

std::size_t NullBackend::live_count(ResourceKind kind) const {
    return live_[index_of(kind)].size();
}

// --------------------------------------------------------------------------
// Lifecycle
// --------------------------------------------------------------------------

bool NullBackend::initialize(std::string& error) {
    if (initialized_) {
        error = "the null backend is already initialized";
        return false;
    }
    if (const std::string shortfall = capabilities_.shortfall(); !shortfall.empty()) {
        error = shortfall;
        return false;
    }
    initialized_ = true;
    record("initialize");
    return true;
}

void NullBackend::shutdown() {
    if (!initialized_) {
        return;
    }
    record("shutdown");
    initialized_ = false;
}

bool NullBackend::attach_surface(const NativeSurface& surface, std::string& error) {
    if (!initialized_) {
        error = "attach_surface before initialize";
        return false;
    }
    if (surface.width == 0 || surface.height == 0) {
        error = fmt::format("a surface of {}x{} has no pixels", surface.width, surface.height);
        return false;
    }
    surface_ = surface;

    // The drawable is a texture like any other, so a test that renders to it
    // and a test that renders to an offscreen target go down the same path.
    if (!drawable_.valid()) {
        const u32 id = next_id(ResourceKind::Texture);
        live_[index_of(ResourceKind::Texture)].emplace(id, "drawable");
        drawable_ = TextureHandle{id};
    }
    record(fmt::format("attach_surface {}x{}", surface.width, surface.height));
    return true;
}

void NullBackend::set_vsync(bool enabled) {
    vsync_ = enabled;
    record(fmt::format("set_vsync {}", enabled));
}

bool NullBackend::begin_frame(FrameInfo& out) {
    if (!initialized_ || !drawable_.valid()) {
        reject("begin_frame with no surface attached");
        return false;
    }
    if (in_frame_) {
        reject("begin_frame inside a frame");
        return false;
    }

    in_frame_ = true;
    stats_ = FrameStats{};
    out.drawable = drawable_;
    out.width = surface_.width;
    out.height = surface_.height;
    out.format = TextureFormat::BGRA8Unorm;
    out.index = frame_index_;
    record(fmt::format("begin_frame {}", frame_index_));
    return true;
}

void NullBackend::end_frame() {
    if (!in_frame_) {
        reject("end_frame outside a frame");
        return;
    }
    if (in_render_pass_) {
        // On Metal this is an encoder left open, which is a hang or a
        // validation failure rather than a wrong pixel. Worth catching here,
        // where it is a line in a test instead.
        reject("end_frame with a render pass still open");
        in_render_pass_ = false;
    }
    record(fmt::format("end_frame {}", frame_index_));
    ++frame_index_;
    in_frame_ = false;
    last_stats_ = stats_;
}

// --------------------------------------------------------------------------
// Resources
// --------------------------------------------------------------------------

BufferHandle NullBackend::create_buffer(const BufferDesc& desc) {
    if (desc.size == 0) {
        reject("create_buffer of zero bytes");
        return {};
    }
    const u32 id = next_id(ResourceKind::Buffer);
    live_[index_of(ResourceKind::Buffer)].emplace(id, label_or(desc.label, "buffer"));
    buffer_bytes_[id].assign(static_cast<std::size_t>(desc.size), 0);
    record(fmt::format("create_buffer {} size={} storage={} label={}",
                       id,
                       desc.size,
                       storage_name(desc.storage),
                       label_or(desc.label, "-")));
    return BufferHandle{id};
}

TextureHandle NullBackend::create_texture(const TextureDesc& desc) {
    if (desc.width == 0 || desc.height == 0) {
        reject(fmt::format("create_texture {}x{}", desc.width, desc.height));
        return {};
    }
    if (desc.format == TextureFormat::Unknown) {
        reject("create_texture with no format");
        return {};
    }
    if (desc.storage == Storage::Memoryless && has_usage(desc.usage, TextureUsage::Sampled)) {
        // Memoryless means it never leaves tile memory (P21). Sampling it
        // later is exactly the thing that cannot work.
        reject(fmt::format("create_texture '{}' is memoryless but sampled",
                           label_or(desc.label, "-")));
        return {};
    }

    const u32 id = next_id(ResourceKind::Texture);
    live_[index_of(ResourceKind::Texture)].emplace(id, label_or(desc.label, "texture"));
    record(fmt::format("create_texture {} {}x{} format={} storage={} label={}",
                       id,
                       desc.width,
                       desc.height,
                       texture_format_name(desc.format),
                       storage_name(desc.storage),
                       label_or(desc.label, "-")));
    return TextureHandle{id};
}

SamplerHandle NullBackend::create_sampler(const SamplerDesc& desc) {
    const u32 id = next_id(ResourceKind::Sampler);
    live_[index_of(ResourceKind::Sampler)].emplace(id, label_or(desc.label, "sampler"));

    // A border colour Metal cannot express is not an error -- it is a note
    // for the shader analysis pass (8.7), which emulates the border in the
    // shader. Recording it is how a test proves the note was made.
    const bool custom_border = (desc.address_u == AddressMode::ClampToBorder ||
                                desc.address_v == AddressMode::ClampToBorder ||
                                desc.address_w == AddressMode::ClampToBorder) &&
                               !(desc.border_color == ClearColor{0.0F, 0.0F, 0.0F, 0.0F} ||
                                 desc.border_color == ClearColor{0.0F, 0.0F, 0.0F, 1.0F} ||
                                 desc.border_color == ClearColor{1.0F, 1.0F, 1.0F, 1.0F});

    record(fmt::format("create_sampler {} border_emulated={} label={}",
                       id,
                       custom_border,
                       label_or(desc.label, "-")));
    return SamplerHandle{id};
}

PipelineHandle NullBackend::create_pipeline(const PipelineDesc& desc) {
    if (const std::string problem = desc.validate(); !problem.empty()) {
        reject(fmt::format("create_pipeline '{}': {}", label_or(desc.label, "-"), problem));
        return {};
    }
    const u32 id = next_id(ResourceKind::Pipeline);
    live_[index_of(ResourceKind::Pipeline)].emplace(id, label_or(desc.label, "pipeline"));
    record(fmt::format("create_pipeline {} primitive={} label={}",
                       id,
                       primitive_type_name(desc.primitive),
                       label_or(desc.label, "-")));
    return PipelineHandle{id};
}

void NullBackend::destroy(BufferHandle handle) {
    kill(ResourceKind::Buffer, handle.value(), "buffer");
}

void NullBackend::destroy(TextureHandle handle) {
    kill(ResourceKind::Texture, handle.value(), "texture");
}

void NullBackend::destroy(SamplerHandle handle) {
    kill(ResourceKind::Sampler, handle.value(), "sampler");
}

void NullBackend::destroy(PipelineHandle handle) {
    kill(ResourceKind::Pipeline, handle.value(), "pipeline");
}

void NullBackend::destroy(FenceHandle handle) {
    kill(ResourceKind::Fence, handle.value(), "fence");
}

bool NullBackend::write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) {
    const auto entry = buffer_bytes_.find(handle.value());
    if (entry == buffer_bytes_.end()) {
        reject(fmt::format("write_buffer {}: not a live handle", handle.value()));
        return false;
    }
    if (offset + data.size() > entry->second.size()) {
        reject(fmt::format("write_buffer {}: {} bytes at {} runs past the end ({})",
                           handle.value(),
                           data.size(),
                           offset,
                           entry->second.size()));
        return false;
    }
    std::copy(
        data.begin(), data.end(), entry->second.begin() + static_cast<std::ptrdiff_t>(offset));
    record(fmt::format("write_buffer {} {} bytes at {}", handle.value(), data.size(), offset));
    return true;
}

void* NullBackend::buffer_contents(BufferHandle handle) {
    const auto entry = buffer_bytes_.find(handle.value());
    return entry == buffer_bytes_.end() ? nullptr : entry->second.data();
}

// --------------------------------------------------------------------------
// Encoding
// --------------------------------------------------------------------------

bool NullBackend::begin_render_pass(const RenderPassDesc& desc) {
    if (in_render_pass_) {
        reject("begin_render_pass inside a render pass");
        return false;
    }
    if (const std::string problem = desc.validate(); !problem.empty()) {
        reject(fmt::format("begin_render_pass '{}': {}", label_or(desc.label, "-"), problem));
        return false;
    }
    for (u32 index = 0; index < desc.color_count; ++index) {
        if (!live(ResourceKind::Texture, desc.colors[index].texture.value())) {
            reject(fmt::format("begin_render_pass: colour attachment {} is not a live texture",
                               index));
            return false;
        }
    }

    in_render_pass_ = true;
    last_render_pass_ = desc;
    ++stats_.render_passes;

    std::string line = fmt::format(
        "begin_render_pass '{}' colors={}", label_or(desc.label, "-"), desc.color_count);
    for (u32 index = 0; index < desc.color_count; ++index) {
        line += fmt::format(" [{}:{}/{}]",
                            index,
                            load_action_name(desc.colors[index].load),
                            store_action_name(desc.colors[index].store));
    }
    record(std::move(line));
    return true;
}

void NullBackend::end_render_pass() {
    if (!in_render_pass_) {
        reject("end_render_pass outside a render pass");
        return;
    }
    in_render_pass_ = false;
    record("end_render_pass");
}

void NullBackend::set_pipeline(PipelineHandle pipeline) {
    if (!live(ResourceKind::Pipeline, pipeline.value())) {
        reject(fmt::format("set_pipeline {}: not a live handle", pipeline.value()));
        return;
    }
    record(fmt::format("set_pipeline {}", pipeline.value()));
}

void NullBackend::set_viewport(const Viewport& viewport) {
    record(fmt::format("set_viewport {}x{} at ({},{}) depth [{},{}]",
                       viewport.width,
                       viewport.height,
                       viewport.x,
                       viewport.y,
                       viewport.near_z,
                       viewport.far_z));
}

void NullBackend::set_scissor(const Rect& rect) {
    record(fmt::format("set_scissor {}x{} at ({},{})", rect.width, rect.height, rect.x, rect.y));
}

void NullBackend::bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) {
    if (!live(ResourceKind::Buffer, buffer.value())) {
        reject(fmt::format("bind_buffer {}: not a live handle", buffer.value()));
        return;
    }
    record(fmt::format(
        "bind_buffer {} [{}] = {} +{}", shader_stage_name(stage), index, buffer.value(), offset));
}

void NullBackend::bind_texture(ShaderStage stage, u32 index, TextureHandle texture) {
    if (!live(ResourceKind::Texture, texture.value())) {
        reject(fmt::format("bind_texture {}: not a live handle", texture.value()));
        return;
    }
    record(
        fmt::format("bind_texture {} [{}] = {}", shader_stage_name(stage), index, texture.value()));
}

void NullBackend::bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) {
    if (!live(ResourceKind::Sampler, sampler.value())) {
        reject(fmt::format("bind_sampler {}: not a live handle", sampler.value()));
        return;
    }
    record(
        fmt::format("bind_sampler {} [{}] = {}", shader_stage_name(stage), index, sampler.value()));
}

void NullBackend::draw(const DrawCall& call) {
    if (!in_render_pass_) {
        reject("draw outside a render pass");
        return;
    }

    const bool indexed = call.index_buffer.valid();
    if (indexed && !live(ResourceKind::Buffer, call.index_buffer.value())) {
        reject(
            fmt::format("draw: index buffer {} is not a live handle", call.index_buffer.value()));
        return;
    }
    const u32 count = indexed ? call.index_count : call.vertex_count;
    if (count == 0) {
        // Harmless on hardware and almost always a translation bug: a draw
        // the guest asked for whose vertex count came out zero.
        reject("draw of nothing");
        return;
    }

    ++stats_.draws;
    record(fmt::format(
        "draw {} {} instances={}", indexed ? "indexed" : "vertices", count, call.instance_count));
}

// --------------------------------------------------------------------------
// Synchronisation
// --------------------------------------------------------------------------

FenceHandle NullBackend::create_fence() {
    const u32 id = next_id(ResourceKind::Fence);
    live_[index_of(ResourceKind::Fence)].emplace(id, "fence");
    fences_[id] = false;
    record(fmt::format("create_fence {}", id));
    return FenceHandle{id};
}

void NullBackend::signal(FenceHandle handle) {
    const auto entry = fences_.find(handle.value());
    if (entry == fences_.end()) {
        reject(fmt::format("signal {}: not a live handle", handle.value()));
        return;
    }
    entry->second = true;
    record(fmt::format("signal {}", handle.value()));
}

bool NullBackend::is_signalled(FenceHandle handle) {
    const auto entry = fences_.find(handle.value());
    return entry != fences_.end() && entry->second;
}

void NullBackend::wait(FenceHandle handle) {
    const auto entry = fences_.find(handle.value());
    if (entry == fences_.end()) {
        reject(fmt::format("wait {}: not a live handle", handle.value()));
        return;
    }
    // Nothing is running, so there is nothing to wait for -- but the wait is
    // still recorded and still counted, because **P4** is about how *often*
    // the CPU waits, and that question is answerable without a GPU.
    entry->second = true;
    record(fmt::format("wait {}", handle.value()));
}

std::unique_ptr<GpuBackend> create_null_backend() {
    return std::make_unique<NullBackend>();
}

} // namespace wiiu::gpu
