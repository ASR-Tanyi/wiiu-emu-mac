// Phase 7.1 — the recording backend.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/null_backend.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

bool NullBackend::require_pass(std::string_view what) {
    if (in_render_pass_) {
        return true;
    }
    reject(fmt::format("{} outside a render pass", what));
    return false;
}

bool NullBackend::check_binding(std::string_view what, u32 index, u32 limit) {
    if (index < limit) {
        return true;
    }
    reject(fmt::format("{} index {} is past the {} the stage has", what, index, limit));
    return false;
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
        imported_.erase(id);
    }
    if (kind == ResourceKind::Texture) {
        texture_pixels_.erase(id);
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

    // Resized along with the surface, so that what was presented can be read
    // back at the size it was presented at.
    TexturePixels pixels;
    pixels.width = surface.width;
    pixels.height = surface.height;
    pixels.format = TextureFormat::BGRA8Unorm;
    pixels.bytes.assign(static_cast<std::size_t>(surface.width) * surface.height * 4, 0);
    texture_pixels_.insert_or_assign(drawable_.value(), std::move(pixels));
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
    if (desc.storage == Storage::Memoryless) {
        // Only render targets can live in tile memory alone; Metal refuses
        // a memoryless buffer, so this does too.
        reject(fmt::format("create_buffer '{}' is memoryless", label_or(desc.label, "-")));
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

    if (const u32 pixel = format_bytes_per_pixel(desc.format); pixel != 0) {
        TexturePixels pixels;
        pixels.width = desc.width;
        pixels.height = desc.height;
        pixels.format = desc.format;
        pixels.storage = desc.storage;
        pixels.bytes.assign(static_cast<std::size_t>(desc.width) * desc.height * pixel, 0);
        texture_pixels_.emplace(id, std::move(pixels));
    }

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

PipelineHandle NullBackend::create_pipeline(const PipelineDesc& desc, std::string& error) {
    error = desc.validate();

    // There is no compiler here, but there is one question a compiler would
    // answer first that a string search can answer too: whether the function
    // the pipeline names is in the source at all.
    if (error.empty() && desc.vertex_source.find(desc.vertex_entry) == std::string::npos) {
        error = fmt::format("no function '{}' in the vertex source", desc.vertex_entry);
    }
    const std::string& fragment =
        desc.fragment_source.empty() ? desc.vertex_source : desc.fragment_source;
    if (error.empty() && fragment.find(desc.fragment_entry) == std::string::npos) {
        error = fmt::format("no function '{}' in the fragment source", desc.fragment_entry);
    }
    if (!error.empty()) {
        reject(fmt::format("create_pipeline '{}': {}", label_or(desc.label, "-"), error));
        return {};
    }
    const u32 id = next_id(ResourceKind::Pipeline);
    live_[index_of(ResourceKind::Pipeline)].emplace(id, label_or(desc.label, "pipeline"));
    record(fmt::format("create_pipeline {} primitive={} label={}",
                       id,
                       primitive_type_name(desc.primitive),
                       label_or(desc.label, "-")));
    // How each attachment is written, where it is not Metal's default, as
    // the numbers the enums have: a line each, for a test to look for.
    for (u32 index = 0; index < desc.color_count; ++index) {
        const BlendState& blend = desc.blend[index];
        if (blend == BlendState{}) {
            continue;
        }
        record(fmt::format("pipeline {} attachment {}: blend {} rgb {}*{} op {} alpha {}*{} op "
                           "{} mask {:#x}",
                           id,
                           index,
                           blend.enabled ? "on" : "off",
                           static_cast<u32>(blend.source_rgb),
                           static_cast<u32>(blend.destination_rgb),
                           static_cast<u32>(blend.rgb),
                           static_cast<u32>(blend.source_alpha),
                           static_cast<u32>(blend.destination_alpha),
                           static_cast<u32>(blend.alpha),
                           blend.write_mask));
    }
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

DepthStencilHandle NullBackend::create_depth_stencil(const DepthStencilDesc& desc) {
    const u32 id = next_id(ResourceKind::DepthStencil);
    live_[index_of(ResourceKind::DepthStencil)].emplace(id, "depth-stencil state");
    record(fmt::format("create_depth_stencil {} depth {} write {} stencil {}",
                       id,
                       static_cast<u32>(desc.depth_compare),
                       desc.depth_write ? "on" : "off",
                       desc.stencil ? "on" : "off"));
    if (desc.stencil) {
        for (const auto& [name, face] :
             {std::pair{"front", desc.front}, std::pair{"back", desc.back}}) {
            record(fmt::format("depth-stencil {} {}: compare {} fail {} depth-fail {} pass {} read "
                               "{:#04x} write {:#04x}",
                               id,
                               name,
                               static_cast<u32>(face.compare),
                               static_cast<u32>(face.stencil_fail),
                               static_cast<u32>(face.depth_fail),
                               static_cast<u32>(face.pass),
                               face.read_mask,
                               face.write_mask));
        }
    }
    return DepthStencilHandle{id};
}

void NullBackend::destroy(DepthStencilHandle handle) {
    kill(ResourceKind::DepthStencil, handle.value(), "depth-stencil state");
}

void NullBackend::destroy(FenceHandle handle) {
    kill(ResourceKind::Fence, handle.value(), "fence");
}

bool NullBackend::write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) {
    std::span<u8> bytes;
    if (const auto entry = buffer_bytes_.find(handle.value()); entry != buffer_bytes_.end()) {
        bytes = entry->second;
    } else if (const auto wrapped = imported_.find(handle.value()); wrapped != imported_.end()) {
        bytes = wrapped->second;
    } else {
        reject(fmt::format("write_buffer {}: not a live handle", handle.value()));
        return false;
    }
    if (offset + data.size() > bytes.size()) {
        reject(fmt::format("write_buffer {}: {} bytes at {} runs past the end ({})",
                           handle.value(),
                           data.size(),
                           offset,
                           bytes.size()));
        return false;
    }
    std::copy(data.begin(), data.end(), bytes.begin() + static_cast<std::ptrdiff_t>(offset));
    record(fmt::format("write_buffer {} {} bytes at {}", handle.value(), data.size(), offset));
    return true;
}

void* NullBackend::buffer_contents(BufferHandle handle) {
    if (const auto entry = buffer_bytes_.find(handle.value()); entry != buffer_bytes_.end()) {
        return entry->second.data();
    }
    const auto wrapped = imported_.find(handle.value());
    return wrapped == imported_.end() ? nullptr : wrapped->second.data();
}

bool NullBackend::write_texture(TextureHandle handle,
                                const Rect& region,
                                std::span<const u8> data,
                                u32 bytes_per_row) {
    const auto entry = texture_pixels_.find(handle.value());
    if (entry == texture_pixels_.end()) {
        reject(
            fmt::format("write_texture {}: nothing writable behind that handle", handle.value()));
        return false;
    }
    TexturePixels& pixels = entry->second;
    if (pixels.storage != Storage::Shared) {
        // The same refusal Metal makes: a private texture has no CPU
        // address, and filling one is a blit on the GPU thread (8.2).
        reject(fmt::format("write_texture {}: {} storage has no CPU address",
                           handle.value(),
                           storage_name(pixels.storage)));
        return false;
    }
    if (region.width == 0 || region.height == 0 || region.x + region.width > pixels.width ||
        region.y + region.height > pixels.height) {
        reject(fmt::format("write_texture {}: {}x{} at ({},{}) is not inside {}x{}",
                           handle.value(),
                           region.width,
                           region.height,
                           region.x,
                           region.y,
                           pixels.width,
                           pixels.height));
        return false;
    }

    const u32 pixel = format_bytes_per_pixel(pixels.format);
    const std::size_t row = static_cast<std::size_t>(region.width) * pixel;
    const std::size_t needed =
        (static_cast<std::size_t>(bytes_per_row) * (region.height - 1)) + row;
    if (bytes_per_row < row || data.size() < needed) {
        reject(fmt::format("write_texture {}: {} bytes, {} to a row, for a {}x{} region",
                           handle.value(),
                           data.size(),
                           bytes_per_row,
                           region.width,
                           region.height));
        return false;
    }

    const std::size_t stride = static_cast<std::size_t>(pixels.width) * pixel;
    for (u32 y = 0; y < region.height; ++y) {
        const auto source =
            data.begin() + static_cast<std::ptrdiff_t>(y * std::size_t{bytes_per_row});
        const auto target =
            pixels.bytes.begin() + static_cast<std::ptrdiff_t>(((region.y + y) * stride) +
                                                               (region.x * std::size_t{pixel}));
        std::copy(source, source + static_cast<std::ptrdiff_t>(row), target);
    }
    record(fmt::format("write_texture {} {}x{} at ({},{})",
                       handle.value(),
                       region.width,
                       region.height,
                       region.x,
                       region.y));
    return true;
}

BufferHandle NullBackend::import_host_memory(void* pointer, u64 size, std::string_view label) {
    const u64 page = host_page_size();
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || size == 0 || (address % page) != 0 || (size % page) != 0) {
        // Metal's rule, enforced here too so that code which works against
        // this backend does not fall over on the real one.
        reject(fmt::format("import_host_memory '{}': {} bytes at {:#x} are not whole {}-byte pages",
                           label_or(label, "-"),
                           size,
                           address,
                           page));
        return {};
    }
    const u32 id = next_id(ResourceKind::Buffer);
    live_[index_of(ResourceKind::Buffer)].emplace(id, label_or(label, "imported"));
    imported_.emplace(id, std::span<u8>{static_cast<u8*>(pointer), static_cast<std::size_t>(size)});
    record(fmt::format("import_host_memory {} size={} label={}", id, size, label_or(label, "-")));
    return BufferHandle{id};
}

MemoryStats NullBackend::memory_stats() const {
    // No heaps: everything here is a resource of its own, which is the
    // honest answer for a backend with no allocator to suballocate from.
    MemoryStats stats;
    stats.dedicated = buffer_bytes_.size() + live_[index_of(ResourceKind::Texture)].size();
    stats.imported = imported_.size();
    for (const auto& [id, bytes] : imported_) {
        static_cast<void>(id);
        stats.imported_bytes += bytes.size();
    }
    return stats;
}

void NullBackend::clear_pixels(TextureHandle handle, const ClearColor& color) {
    const auto entry = texture_pixels_.find(handle.value());
    if (entry == texture_pixels_.end()) {
        return;
    }
    TexturePixels& pixels = entry->second;
    if (format_bytes_per_pixel(pixels.format) != 4) {
        // Only the eight-bit four-channel formats are modelled. Anything else
        // keeps whatever it had, which is zeroes -- a wrong answer is worse
        // than an obviously absent one.
        return;
    }

    const auto quantize = [](f32 value) {
        const f32 clamped = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
        return static_cast<u8>((clamped * 255.0F) + 0.5F);
    };

    // A clear's colour is linear. An sRGB attachment stores it encoded with
    // the standard curve, red, green and blue, alpha as it is -- what Metal
    // does, and what GX2's CPU clear does (`latte::encode_color`).
    const bool srgb = pixels.format == TextureFormat::RGBA8UnormSrgb ||
                      pixels.format == TextureFormat::BGRA8UnormSrgb;
    const auto colour = [srgb, &quantize](f32 value) {
        if (!srgb) {
            return quantize(value);
        }
        const f32 clamped = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
        return quantize(clamped <= 0.0031308F ? clamped * 12.92F
                                              : (1.055F * std::pow(clamped, 1.0F / 2.4F)) - 0.055F);
    };

    const bool bgra = pixels.format == TextureFormat::BGRA8Unorm ||
                      pixels.format == TextureFormat::BGRA8UnormSrgb;
    const std::array<u8, 4> texel{bgra ? colour(color.blue) : colour(color.red),
                                  colour(color.green),
                                  bgra ? colour(color.red) : colour(color.blue),
                                  quantize(color.alpha)};

    for (std::size_t offset = 0; offset + 4 <= pixels.bytes.size(); offset += 4) {
        std::copy(
            texel.begin(), texel.end(), pixels.bytes.begin() + static_cast<std::ptrdiff_t>(offset));
    }
}

bool NullBackend::read_texture(TextureHandle handle, std::span<u8> out) {
    const auto entry = texture_pixels_.find(handle.value());
    if (entry == texture_pixels_.end()) {
        reject(fmt::format("read_texture {}: nothing readable behind that handle", handle.value()));
        return false;
    }
    if (out.size() < entry->second.bytes.size()) {
        reject(fmt::format("read_texture {}: {} bytes of room for {}",
                           handle.value(),
                           out.size(),
                           entry->second.bytes.size()));
        return false;
    }
    std::copy(entry->second.bytes.begin(), entry->second.bytes.end(), out.begin());
    return true;
}

void NullBackend::submit() {
    if (in_render_pass_) {
        reject("submit with a render pass still open");
        in_render_pass_ = false;
    }
    record("submit");
}

void NullBackend::wait_idle() {
    // Nothing is running, so this is instantaneous -- but it is still
    // recorded, because where the emulator waits is a question worth being
    // able to ask of a trace (**P4**).
    record("wait_idle");
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
    pass_pipeline_ = {};
    last_render_pass_ = desc;
    ++stats_.render_passes;

    // A clear is the one piece of rendering a backend without a GPU can
    // honestly carry out, and doing it means `read_texture` answers the same
    // question here as it does on Metal.
    for (u32 index = 0; index < desc.color_count; ++index) {
        if (desc.colors[index].load == LoadAction::Clear) {
            clear_pixels(desc.colors[index].texture, desc.colors[index].clear);
        }
    }

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
    if (!require_pass("set_pipeline")) {
        return;
    }
    if (!live(ResourceKind::Pipeline, pipeline.value())) {
        reject(fmt::format("set_pipeline {}: not a live handle", pipeline.value()));
        return;
    }
    pass_pipeline_ = pipeline;
    record(fmt::format("set_pipeline {}", pipeline.value()));
}

void NullBackend::set_viewport(const Viewport& viewport) {
    if (!require_pass("set_viewport")) {
        return;
    }
    record(fmt::format("set_viewport {}x{} at ({},{}) depth [{},{}]",
                       viewport.width,
                       viewport.height,
                       viewport.x,
                       viewport.y,
                       viewport.near_z,
                       viewport.far_z));
}

void NullBackend::set_scissor(const Rect& rect) {
    if (!require_pass("set_scissor")) {
        return;
    }
    record(fmt::format("set_scissor {}x{} at ({},{})", rect.width, rect.height, rect.x, rect.y));
}

void NullBackend::set_cull_mode(CullMode mode) {
    if (!require_pass("set_cull_mode")) {
        return;
    }
    record(fmt::format("set_cull_mode {}",
                       mode == CullMode::Front  ? "front"
                       : mode == CullMode::Back ? "back"
                                                : "none"));
}

void NullBackend::set_front_facing(Winding winding) {
    if (!require_pass("set_front_facing")) {
        return;
    }
    record(fmt::format("set_front_facing {}", winding == Winding::Clockwise ? "cw" : "ccw"));
}

void NullBackend::set_blend_color(const ClearColor& color) {
    if (!require_pass("set_blend_color")) {
        return;
    }
    record(fmt::format(
        "set_blend_color {} {} {} {}", color.red, color.green, color.blue, color.alpha));
}

void NullBackend::set_depth_stencil(DepthStencilHandle state) {
    if (!require_pass("set_depth_stencil")) {
        return;
    }
    if (!live(ResourceKind::DepthStencil, state.value())) {
        reject(fmt::format("set_depth_stencil {}: not a live handle", state.value()));
        return;
    }
    record(fmt::format("set_depth_stencil {}", state.value()));
}

void NullBackend::set_stencil_reference(u32 front, u32 back) {
    if (!require_pass("set_stencil_reference")) {
        return;
    }
    record(fmt::format("set_stencil_reference {} {}", front, back));
}

void NullBackend::bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) {
    if (!require_pass("bind_buffer") || !check_binding("bind_buffer", index, max_bound_buffers)) {
        return;
    }
    if (!live(ResourceKind::Buffer, buffer.value())) {
        reject(fmt::format("bind_buffer {}: not a live handle", buffer.value()));
        return;
    }
    record(fmt::format(
        "bind_buffer {} [{}] = {} +{}", shader_stage_name(stage), index, buffer.value(), offset));
}

void NullBackend::bind_bytes(ShaderStage stage, u32 index, std::span<const u8> bytes) {
    if (!require_pass("bind_bytes") || !check_binding("bind_bytes", index, max_bound_buffers)) {
        return;
    }
    if (bytes.empty() || bytes.size() > max_bound_bytes) {
        reject(fmt::format(
            "bind_bytes: {} bytes, where Metal takes 1 to {}", bytes.size(), max_bound_bytes));
        return;
    }
    last_bytes_.assign(bytes.begin(), bytes.end());
    record(
        fmt::format("bind_bytes {} [{}] {} bytes", shader_stage_name(stage), index, bytes.size()));
}

void NullBackend::bind_texture(ShaderStage stage, u32 index, TextureHandle texture) {
    if (!require_pass("bind_texture") ||
        !check_binding("bind_texture", index, max_bound_textures)) {
        return;
    }
    if (!live(ResourceKind::Texture, texture.value())) {
        reject(fmt::format("bind_texture {}: not a live handle", texture.value()));
        return;
    }
    record(
        fmt::format("bind_texture {} [{}] = {}", shader_stage_name(stage), index, texture.value()));
}

void NullBackend::bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) {
    if (!require_pass("bind_sampler") ||
        !check_binding("bind_sampler", index, max_bound_samplers)) {
        return;
    }
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

    if (!pass_pipeline_.valid()) {
        // On Metal this is a validation failure at best: an encoder with no
        // pipeline state has no idea what a vertex is.
        reject("draw with no pipeline set in this pass");
        return;
    }

    const bool indexed = call.index_buffer.valid();
    if (indexed && !live(ResourceKind::Buffer, call.index_buffer.value())) {
        reject(
            fmt::format("draw: index buffer {} is not a live handle", call.index_buffer.value()));
        return;
    }
    if (indexed && (call.index_offset % (call.index_is_32bit ? 4U : 2U)) != 0) {
        reject(fmt::format("draw: an index offset of {} is not a multiple of the index size",
                           call.index_offset));
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

// --------------------------------------------------------------------------
// Tooling
// --------------------------------------------------------------------------

bool NullBackend::begin_capture(std::string_view path, std::string& error) {
    if (capturing_) {
        error = "a capture is already running";
        reject(fmt::format("begin_capture {}: {}", path, error));
        return false;
    }
    // Nothing to capture, but the bracket is real: a frontend that starts a
    // capture and never ends it is the same bug here as on Metal.
    capturing_ = true;
    record(fmt::format("begin_capture {}", path));
    return true;
}

void NullBackend::end_capture() {
    if (!capturing_) {
        reject("end_capture with no capture running");
        return;
    }
    capturing_ = false;
    record("end_capture");
}

std::unique_ptr<GpuBackend> create_null_backend() {
    return std::make_unique<NullBackend>();
}

} // namespace wiiu::gpu
