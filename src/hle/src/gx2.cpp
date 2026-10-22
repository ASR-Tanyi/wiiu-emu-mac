#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_buffers.hpp>
#include <wiiu/hle/gx2_context.hpp>
#include <wiiu/hle/gx2_draw.hpp>
#include <wiiu/hle/gx2_sampler.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/gx2_state.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/gx2_texture.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/draw.hpp>
#include <wiiu/latte/draw_shaders.hpp>
#include <wiiu/latte/draw_state.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/vertex_inputs.hpp>
#include <wiiu/memory/map.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>

namespace wiiu::hle::gx2 {

// --------------------------------------------------------------------------
// GX2Init's attributes
// --------------------------------------------------------------------------

InitAttributes read_init_attributes(memory::Memory& memory, ppcaddr_t address) {
    InitAttributes attributes;
    if (address == 0) {
        return attributes;
    }

    for (u32 index = 0; index < max_init_attributes; ++index) {
        const ppcaddr_t entry = address + (index * 8);
        u32 key = 0;
        u32 value = 0;
        if (!memory.try_read<u32>(entry, key) ||
            (key != 0 && !memory.try_read<u32>(entry + 4, value))) {
            WIIU_LOG_ERROR(
                Gx2, "GX2Init's attribute list at {:#010x} runs into unreadable memory", address);
            return attributes;
        }

        switch (static_cast<InitAttribute>(key)) {
            case InitAttribute::End:
                return attributes;
            case InitAttribute::CommandBufferBase:
                attributes.command_buffer_base = value;
                break;
            case InitAttribute::CommandBufferPoolSize:
                attributes.command_buffer_pool_size = value;
                break;
            case InitAttribute::Argc:
                attributes.argc = value;
                break;
            case InitAttribute::Argv:
                attributes.argv = value;
                break;
            case InitAttribute::ProfileMode:
                attributes.profile_mode = value;
                break;
            case InitAttribute::TossStage:
                attributes.toss_stage = value;
                break;
            case InitAttribute::AppIoThreadStackSize:
                attributes.app_io_thread_stack_size = value;
                break;
            default:
                WIIU_LOG_WARN(
                    Gx2, "GX2Init attribute {} ({:#x}) is not one this knows", key, value);
                break;
        }
    }

    WIIU_LOG_ERROR(Gx2,
                   "GX2Init's attribute list at {:#010x} has no end within {} entries",
                   address,
                   max_init_attributes);
    return attributes;
}

std::string_view event_type_name(u32 type) noexcept {
    switch (static_cast<EventType>(type)) {
        case EventType::StartOfPipe:
            return "start of pipe";
        case EventType::EndOfPipe:
            return "end of pipe";
        case EventType::Vsync:
            return "vsync";
        case EventType::Flip:
            return "flip";
        case EventType::DisplayListOverrun:
            return "display list overrun";
    }
    return "unknown";
}

// --------------------------------------------------------------------------
// Gpu
// --------------------------------------------------------------------------

Gpu::Gpu()
    : registers_(std::make_unique<latte::RegisterFile>()), processor_(*registers_),
      timeline_(processor_) {
    processor_.set_hle_listener([this](const latte::pm4::Packet& packet) { run_hle(packet); });
    processor_.set_draw_listener(
        [this](const latte::DrawCommand& draw, const latte::RegisterFile& registers) {
            run_draw(draw, registers);
        });
}

void Gpu::initialize(memory::Memory& memory, u32 main_core, const InitAttributes& attributes) {
    if (memory_ != &memory) {
        // Windows onto the memory there was, and copies made from it.
        guest_buffers_.reset();
        index_cache_.reset();
    }
    memory_ = &memory;
    processor_.set_memory(&memory);
    main_core_ = main_core;
    attributes_ = attributes;
    initialized_ = true;
}

void Gpu::shutdown() noexcept {
    commands_.clear();
    for (auto& list : display_lists_) {
        list.reset();
    }
    swap_ = {};
    scan_buffers_ = {};
    drc_connect_ = 0;
    pending_swaps_.clear();
    swaps_reached_ = 0;
    initialized_ = false;
}

bool Gpu::write(u32 core, std::span<const u32> words, u64 now_ticks) {
    if (core < core_count && display_lists_[core].has_value()) {
        return write_display_list(*display_lists_[core], words);
    }

    if (core != main_core_) {
        // Only the main core has a command buffer of its own; the others can
        // build display lists and nothing else.
        WIIU_LOG_ERROR(Gx2,
                       "core {} wrote {} words with no display list open; only core {} has a "
                       "command buffer",
                       core,
                       words.size(),
                       main_core_);
        return false;
    }

    if (commands_.size() + words.size() > max_command_buffer_words) {
        static_cast<void>(flush(now_ticks));
    }
    commands_.insert(commands_.end(), words.begin(), words.end());
    return true;
}

u64 Gpu::flush(u64 now_ticks) {
    if (commands_.empty()) {
        return timeline_.submitted();
    }
    pad_command_buffer(commands_);
    const u64 timestamp = timeline_.submit(commands_, now_ticks);
    commands_.clear();
    return timestamp;
}

u64 Gpu::submit(std::span<const u32> words, u64 now_ticks) {
    return timeline_.submit(words, now_ticks);
}

bool Gpu::begin_display_list(u32 core, ppcaddr_t base, u32 bytes, bool profiling) {
    if (core >= core_count) {
        WIIU_LOG_ERROR(Gx2, "a display list was begun on core {}, which does not exist", core);
        return false;
    }
    if (display_lists_[core].has_value()) {
        WIIU_LOG_ERROR(Gx2,
                       "a display list at {:#010x} was begun on core {} while the one at "
                       "{:#010x} is still open",
                       base,
                       core,
                       display_lists_[core]->base);
        return false;
    }
    display_lists_[core] = DisplayList{
        .base = base,
        .capacity_words = bytes / 4,
        .used_words = 0,
        .profiling = profiling,
        .overrun = false,
    };
    return true;
}

std::optional<u32> Gpu::end_display_list(u32 core) {
    if (core >= core_count || !display_lists_[core].has_value()) {
        return std::nullopt;
    }
    DisplayList& list = *display_lists_[core];

    // Padded like any other buffer, so it can be called as one. Filler that
    // does not fit is left off rather than overrunning: the list is already
    // aligned as far as the GPU cares if it ends at the end of its buffer.
    const u32 padding =
        (command_buffer_alignment_words - (list.used_words % command_buffer_alignment_words)) %
        command_buffer_alignment_words;
    const u32 room = list.capacity_words - list.used_words;
    for (u32 i = 0; i < std::min(padding, room); ++i) {
        static_cast<void>(
            memory_->try_write<u32>(list.base + (list.used_words * 4), latte::pm4::type2_filler));
        ++list.used_words;
    }

    const u32 bytes = list.used_words * 4;
    display_lists_[core].reset();
    return bytes;
}

const DisplayList* Gpu::display_list(u32 core) const noexcept {
    if (core >= core_count || !display_lists_[core].has_value()) {
        return nullptr;
    }
    return &*display_lists_[core];
}

bool Gpu::write_display_list(DisplayList& list, std::span<const u32> words) {
    if (list.overrun) {
        return false;
    }
    if (words.size() > list.capacity_words - list.used_words) {
        list.overrun = true;
        WIIU_LOG_ERROR(Gx2,
                       "the display list at {:#010x} is full ({} of {} words used, {} more "
                       "asked for); the overrun callback that would give it more room is not "
                       "implemented, so the rest of the list is dropped",
                       list.base,
                       list.used_words,
                       list.capacity_words,
                       words.size());
        return false;
    }
    if (memory_ == nullptr) {
        WIIU_LOG_ERROR(Gx2, "a display list was written before GX2Init");
        return false;
    }

    for (const u32 word : words) {
        if (!memory_->try_write<u32>(list.base + (list.used_words * 4), word)) {
            list.overrun = true;
            WIIU_LOG_ERROR(Gx2,
                           "the display list at {:#010x} is not writable memory at word {}",
                           list.base,
                           list.used_words);
            return false;
        }
        ++list.used_words;
    }
    return true;
}

void Gpu::set_event_callback(u32 type, EventCallback callback) noexcept {
    if (type < event_type_count) {
        callbacks_[type] = callback;
    }
}

EventCallback Gpu::event_callback(u32 type) const noexcept {
    return type < event_type_count ? callbacks_[type] : EventCallback{};
}

bool Gpu::set_swap_interval(u32 interval) {
    if (interval > max_swap_interval) {
        WIIU_LOG_WARN(Gx2,
                      "GX2SetSwapInterval({}) is out of range (at most {}); keeping {}",
                      interval,
                      max_swap_interval,
                      swap_.interval);
        return false;
    }
    if (interval == 0) {
        WIIU_LOG_DEBUG(Gx2, "swap interval 0 flips at vsync, like 1, until it is modelled");
    }
    swap_.interval = interval;
    return true;
}

void Gpu::request_swap(u64 timestamp) {
    ++swap_.swaps;
    pending_swaps_.push_back(timestamp);
}

bool Gpu::vsync(u64 index) {
    // Retired up to the vsync's own tick, not the moment the timer happened to
    // be serviced: a swap whose submission retires a tick after the vsync
    // waits for the next one, on every run.
    const u64 tick = vsync_tick(index);
    static_cast<void>(timeline_.retire_until(tick));
    swap_.last_vsync = tick;

    if (pending_swaps_.empty() || pending_swaps_.front() > timeline_.retired()) {
        return false;
    }
    const u64 interval = std::max<u32>(swap_.interval, 1);
    if (swap_.flips != 0 && index - swap_.last_flip_vsync < interval) {
        return false;
    }

    pending_swaps_.pop_front();
    ++swap_.flips;
    swap_.last_flip = tick;
    swap_.last_flip_vsync = index;
    return true;
}

Gpu::~Gpu() {
    attach_backend(nullptr);
}

void Gpu::attach_backend(gpu::GpuBackend* backend) {
    // Destroyed on the backend that made them, before it is let go.
    release_scan_textures();
    copier_.shutdown();
    guest_buffers_.reset();
    index_cache_.reset();
    depth_stencils_.reset();
    pipelines_.reset();
    surfaces_.reset();
    backend_ = backend;
    if (backend_ == nullptr) {
        return;
    }
    surfaces_ = std::make_unique<latte::SurfaceCache>(*backend_);
    pipelines_ = std::make_unique<latte::PipelineCache>(*backend_);
    depth_stencils_ = std::make_unique<latte::DepthStencilCache>(*backend_);
    std::string error;
    if (!copier_.initialize(*backend_, error)) {
        // Clears still run on the GPU; copies of what they made cannot, and
        // say so when asked for.
        WIIU_LOG_ERROR(Gx2, "the GPU's surface copy is unavailable: {}", error);
    }
}

void Gpu::reset() {
    release_scan_textures();
    if (surfaces_ != nullptr) {
        surfaces_->clear();
    }
    // Pipelines hold nothing of the title's and stay; windows onto its
    // memory, and copies of its indices, do not.
    guest_buffers_.reset();
    index_cache_.reset();
    draw_stats_ = {};
    registers_->reset();
    processor_.reset_state();
    timeline_ = latte::GpuTimeline{processor_, timeline_.latency()};
    processor_.set_memory(nullptr);
    memory_ = nullptr;
    commands_.clear();
    for (auto& list : display_lists_) {
        list.reset();
    }
    callbacks_ = {};
    swap_ = {};
    scan_buffers_ = {};
    drc_connect_ = 0;
    pending_swaps_.clear();
    swaps_reached_ = 0;
    attributes_ = {};
    reported_.clear();
    initialized_ = false;
    main_core_ = 1;
}

void Gpu::run_hle(const latte::pm4::Packet& packet) {
    using latte::pm4::Opcode;
    const std::span<const u32> payload = packet.payload;

    switch (static_cast<Opcode>(packet.header.opcode)) {
        case Opcode::HleSwap:
            ++swaps_reached_;
            return;

        case Opcode::HleClearColor: {
            if (payload.size() != 4 + color_buffer_words || memory_ == nullptr) {
                report(fmt::format("a clear packet of {} words, or no memory to clear in",
                                   payload.size()));
                return;
            }
            const std::array<f32, 4> rgba{std::bit_cast<f32>(payload[0]),
                                          std::bit_cast<f32>(payload[1]),
                                          std::bit_cast<f32>(payload[2]),
                                          std::bit_cast<f32>(payload[3])};
            const ColorBuffer buffer =
                color_buffer_from_words(payload.subspan<4, color_buffer_words>());
            if (backend_ != nullptr) {
                std::string problem;
                if (clear_color_surface(*surfaces_, *backend_, *memory_, buffer, rgba, problem)) {
                    return;
                }
                report(problem);
            }
            report(clear_color_buffer(*memory_, buffer, rgba));
            return;
        }

        case Opcode::HleClearDepthStencil: {
            if (payload.size() != 3 + depth_buffer_words || memory_ == nullptr) {
                report(fmt::format("a depth clear packet of {} words, or no memory to clear in",
                                   payload.size()));
                return;
            }
            const DepthBuffer buffer =
                depth_buffer_from_words(payload.subspan<3, depth_buffer_words>());
            if (backend_ != nullptr) {
                std::string problem;
                if (clear_depth_stencil_surface(*surfaces_,
                                                *backend_,
                                                *memory_,
                                                buffer,
                                                std::bit_cast<f32>(payload[1]),
                                                static_cast<u8>(payload[2]),
                                                payload[0],
                                                problem)) {
                    return;
                }
                report(problem);
            }
            report(clear_depth_stencil_buffer(*memory_,
                                              buffer,
                                              std::bit_cast<f32>(payload[1]),
                                              static_cast<u8>(payload[2]),
                                              payload[0]));
            return;
        }

        case Opcode::HleCopyColorToScan: {
            if (payload.size() != 1 + color_buffer_words || memory_ == nullptr) {
                report(fmt::format("a scan-out copy packet of {} words, or no memory to copy in",
                                   payload.size()));
                return;
            }
            const std::optional<ScanTarget> target = scan_target_from_gx2(payload[0]);
            if (!target.has_value()) {
                report(fmt::format("a scan-out copy to scan target {}", payload[0]));
                return;
            }
            const ScanBuffer& scan = scan_buffer(*target);
            const std::optional<ppcaddr_t> frame = scan_frame(scan, swaps_reached_);
            if (!frame.has_value()) {
                report(fmt::format("a scan-out copy to the {}, which has no frame to copy into: "
                                   "no scan buffer set, or one narrower than four bytes a pixel",
                                   scan_target_name(*target)));
                return;
            }
            const ColorBuffer buffer =
                color_buffer_from_words(payload.subspan<1, color_buffer_words>());
            // The frame `scan_frame` chose, by its place in the rotation.
            const u32 index = swaps_reached_ % scan_buffer_count(scan.buffering_mode);
            if (backend_ != nullptr && copy_on_gpu(*target, index, buffer)) {
                return;
            }
            report(copy_to_scan_frame(*memory_, buffer, scan, *frame));
            mark_scan_frame(*target, index, false);
            return;
        }

        default:
            report(
                fmt::format("an emulator packet {:#04x} GX2 does not write", packet.header.opcode));
            return;
    }
}

void Gpu::run_draw(const latte::DrawCommand& draw, const latte::RegisterFile& registers) {
    if (backend_ == nullptr || memory_ == nullptr) {
        // Without a GPU nothing draws: guest memory has no rasteriser.
        return;
    }
    std::string problem;
    if (draw_on_gpu(draw, registers, problem)) {
        ++draw_stats_.drawn;
    } else if (problem.empty()) {
        ++draw_stats_.empty;
    } else {
        ++draw_stats_.refused;
        report(problem);
    }
}

bool Gpu::draw_on_gpu(const latte::DrawCommand& draw,
                      const latte::RegisterFile& registers,
                      std::string& problem) {
    if (draw.kind == latte::DrawCommand::Kind::Immediate) {
        problem = "a draw with its indices in the packet is not drawn yet";
        return false;
    }
    if (draw.count == 0 || draw.instances == 0) {
        return false;
    }
    const std::optional<gpu::PrimitiveType> primitive = latte::primitive_type(draw.primitive);
    if (!primitive.has_value() || gpu::needs_primitive_emulation(*primitive)) {
        problem = fmt::format("a draw of primitive type {:#x} is not drawn yet", draw.primitive);
        return false;
    }

    // Render target 0, as GX2SetColorBuffer left it: the surface a clear of
    // the same buffer made, or a new one.
    std::string error;
    const std::optional<latte::ColorSurface> surface =
        surfaces_->color_buffer(latte::ColorBufferKey::from(registers, 0), error);
    if (!surface.has_value()) {
        problem = "a draw into render target 0: " + error;
        return false;
    }

    // How it is written and rasterized (draw_state.hpp).
    const std::optional<gpu::BlendState> blend = latte::blend_state(registers, 0, error);
    if (!blend.has_value()) {
        problem = "a draw's blending: " + error;
        return false;
    }
    const std::optional<latte::RasterState> raster = latte::raster_state(registers, error);
    if (!raster.has_value()) {
        problem = "a draw's rasterizer state: " + error;
        return false;
    }
    if (raster->cull_all && (*primitive == gpu::PrimitiveType::Triangles ||
                             *primitive == gpu::PrimitiveType::TriangleStrip)) {
        return false; // every face culled: nothing to draw
    }

    // The depth and stencil tests, and the depth buffer they are made
    // against: the one GX2SetDepthBuffer named, from the surface a clear of
    // it made, or a new one. A draw that tests and writes nothing does not
    // need it, and one with no depth buffer bound -- DB_DEPTH_INFO's format
    // INVALID, as before GX2SetDepthBuffer -- tests nothing.
    latte::DepthStencilState depth = latte::depth_stencil_state(registers);
    std::optional<latte::DepthSurface> depth_surface;
    if (depth.uses_depth_buffer()) {
        const latte::DepthBufferKey key = latte::DepthBufferKey::from(registers);
        if (latte::depth_texture_format(key).has_value()) {
            depth_surface = surfaces_->depth_buffer(key, error);
            if (!depth_surface.has_value()) {
                problem = "a draw's depth buffer: " + error;
                return false;
            }
            // A depth buffer without a stencil passes the stencil test.
            depth.desc.stencil = depth.desc.stencil && depth_surface->has_stencil;
        }
    }

    latte::DrawShaders shaders;
    if (!latte::build_draw_shaders(registers, *memory_, shaders, error)) {
        problem = "a draw's shaders: " + error;
        return false;
    }
    const gpu::PipelineHandle pipeline = pipelines_->pipeline(
        latte::PipelineKey{.vertex_msl = shaders.vertex_msl,
                           .pixel_msl = shaders.pixel_msl,
                           .color_format = surface->format,
                           .depth_format = depth_surface.has_value() ? depth_surface->format
                                                                     : gpu::TextureFormat::Unknown,
                           .primitive = *primitive,
                           .blend = *blend},
        error);
    if (!pipeline.valid()) {
        problem = "a draw's pipeline did not build: " + error;
        return false;
    }

    if (guest_buffers_ == nullptr) {
        guest_buffers_ = std::make_unique<latte::GuestBuffers>(*backend_, *memory_);
    }
    latte::VertexInputs inputs;
    if (!latte::vertex_inputs(registers, shaders.vertex_ir, *guest_buffers_, inputs, error)) {
        problem = "a draw's vertices: " + error;
        return false;
    }

    // An indexed draw's indices, converted to the host's order (index_cache.hpp).
    std::optional<latte::IndexBuffer> indices;
    if (draw.kind == latte::DrawCommand::Kind::Indexed) {
        if (index_cache_ == nullptr) {
            index_cache_ = std::make_unique<latte::IndexCache>(*backend_, *memory_);
        }
        indices = index_cache_->indices(draw.index_address, draw.count, draw.index_type, error);
        if (!indices.has_value()) {
            problem = "an indexed draw's indices: " + error;
            return false;
        }
    }
    // The scissor stays inside every attachment, the depth buffer's too.
    u32 width = surface->width;
    u32 height = surface->height;
    if (depth_surface.has_value()) {
        width = std::min(width, depth_surface->width);
        height = std::min(height, depth_surface->height);
    }
    const std::optional<gpu::Rect> scissor = latte::scissor(registers, width, height);
    if (!scissor.has_value()) {
        return false;
    }

    gpu::RenderPassDesc pass;
    pass.label = "draw";
    pass.color_count = 1;
    pass.colors[0].texture = surface->texture;
    pass.colors[0].load = gpu::LoadAction::Load;
    pass.colors[0].store = gpu::StoreAction::Store;
    if (depth_surface.has_value()) {
        pass.depth_stencil.texture = depth_surface->texture;
        pass.depth_stencil.load = gpu::LoadAction::Load;
        pass.depth_stencil.store = gpu::StoreAction::Store;
    }
    if (!backend_->begin_render_pass(pass)) {
        problem = "the backend refused a draw's render pass";
        return false;
    }
    backend_->set_pipeline(pipeline);
    if (depth_surface.has_value()) {
        backend_->set_depth_stencil(depth_stencils_->state(depth.desc));
        if (depth.desc.stencil) {
            backend_->set_stencil_reference(depth.front_reference, depth.back_reference);
        }
    }
    backend_->set_viewport(latte::viewport(registers));
    backend_->set_scissor(*scissor);
    backend_->set_cull_mode(raster->cull);
    backend_->set_front_facing(raster->front);
    if (blend->enabled) {
        backend_->set_blend_color(latte::blend_color(registers));
    }
    latte::bind_vertex_inputs(*backend_, inputs);
    gpu::DrawCall call;
    if (indices.has_value()) {
        call.index_buffer = indices->buffer;
        call.index_count = draw.count;
        call.index_is_32bit = indices->is_32bit;
    } else {
        call.vertex_count = draw.count;
    }
    call.instance_count = draw.instances;
    call.first_vertex = registers.read(latte::reg::sq_vtx_base_vtx_loc);
    call.first_instance = registers.read(latte::reg::sq_vtx_start_inst_loc);
    backend_->draw(call);
    backend_->end_render_pass();
    return true;
}

bool Gpu::copy_on_gpu(ScanTarget target, u32 index, const ColorBuffer& buffer) {
    const std::optional<latte::ColorSurface> surface =
        held_color_surface(*surfaces_, *memory_, buffer);
    if (!surface.has_value()) {
        return false;
    }
    // From here the copy is the GPU's: guest memory under the surface is
    // stale, so a copy from it would show the wrong picture. A copy that
    // cannot be done leaves the frame as it was.
    const ScanBuffer& scan = scan_buffer(target);
    const gpu::TextureHandle texture = scan_texture(target, index);
    if (!texture.valid()) {
        report(fmt::format("GX2CopyColorBufferToScanBuffer: the backend made no {}x{} texture "
                           "for the {}'s frame",
                           scan.geometry.width,
                           scan.geometry.height,
                           scan_target_name(target)));
        return true;
    }
    std::string problem;
    if (!copy_surface_to_scan(copier_,
                              *surface,
                              buffer,
                              scan,
                              latte::CopyTarget{texture, scan.geometry.width, scan.geometry.height},
                              problem)) {
        report(problem);
        return true;
    }
    mark_scan_frame(target, index, true);
    return true;
}

bool Gpu::ScanTextures::describe(const ScanBuffer& scan) const noexcept {
    return frames.size() == scan_buffer_count(scan.buffering_mode) &&
           width == scan.geometry.width && height == scan.geometry.height;
}

gpu::TextureHandle Gpu::scan_texture(ScanTarget target, u32 index) {
    const ScanBuffer& scan = scan_buffer(target);
    ScanTextures& textures = scan_textures_[static_cast<u32>(target)];
    if (!textures.describe(scan)) {
        for (const gpu::TextureHandle texture : textures.frames) {
            if (texture.valid()) {
                retired_textures_.push_back(texture);
            }
        }
        const u32 count = scan_buffer_count(scan.buffering_mode);
        textures = ScanTextures{.frames = std::vector<gpu::TextureHandle>(count),
                                .on_gpu = std::vector<bool>(count, false),
                                .width = scan.geometry.width,
                                .height = scan.geometry.height};
    }
    if (index >= textures.frames.size()) {
        return {};
    }
    gpu::TextureHandle& texture = textures.frames[index];
    if (!texture.valid()) {
        // Shared, though only the GPU writes it and only the presenter reads
        // it: on unified memory that costs nothing worth measuring, and it is
        // what lets a test, or the headless runner (10.5), read a frame back.
        const std::string label = fmt::format("{} scan frame {}", scan_target_name(target), index);
        gpu::TextureDesc desc;
        desc.width = scan.geometry.width;
        desc.height = scan.geometry.height;
        desc.format = latte::SurfaceCopier::target_format;
        desc.usage = gpu::TextureUsage::RenderTarget | gpu::TextureUsage::Sampled;
        desc.storage = gpu::Storage::Shared;
        desc.label = label;
        texture = backend_->create_texture(desc);
    }
    return texture;
}

void Gpu::mark_scan_frame(ScanTarget target, u32 index, bool on_gpu) {
    ScanTextures& textures = scan_textures_[static_cast<u32>(target)];
    if (textures.describe(scan_buffer(target)) && index < textures.on_gpu.size()) {
        textures.on_gpu[index] = on_gpu;
    }
}

gpu::TextureHandle Gpu::scan_texture_shown(ScanTarget target, u32 flip_index) const noexcept {
    const ScanTextures& textures = scan_textures_[static_cast<u32>(target)];
    if (textures.frames.empty() || !textures.describe(scan_buffer(target))) {
        return {};
    }
    const std::size_t index = flip_index % textures.frames.size();
    return textures.on_gpu[index] ? textures.frames[index] : gpu::TextureHandle{};
}

void Gpu::release_scan_textures() {
    if (backend_ != nullptr) {
        for (const ScanTextures& textures : scan_textures_) {
            for (const gpu::TextureHandle texture : textures.frames) {
                if (texture.valid()) {
                    backend_->destroy(texture);
                }
            }
        }
        for (const gpu::TextureHandle texture : retired_textures_) {
            backend_->destroy(texture);
        }
    }
    scan_textures_ = {};
    retired_textures_.clear();
}

void Gpu::report(const std::string& problem) {
    if (!problem.empty() && reported_.insert(problem).second) {
        WIIU_LOG_WARN(Gx2, "{} (said once)", problem);
    }
}

void pad_command_buffer(std::vector<u32>& words) {
    while (words.size() % command_buffer_alignment_words != 0) {
        words.push_back(latte::pm4::type2_filler);
    }
}

void arm_retirement(coreinit::Context& context) {
    const std::optional<u64> deadline = context.gpu.timeline().next_deadline();
    if (!deadline.has_value()) {
        context.scheduler.clear_timer(gpu_timer_id);
        return;
    }
    // Retired up to its own deadline, not to whenever the timer is serviced:
    // the next deadline re-arms it, and if that is due too the scheduler fires
    // it again in the same pass, earliest first. So retirement interleaves
    // with vsync in deadline order however late the clock is looked at, and
    // a vsync never sees a submission retired that was due after it.
    context.scheduler.set_timer(gpu_timer_id, *deadline, [&context, due = *deadline](u64) {
        static_cast<void>(context.gpu.timeline().retire_until(due));
        arm_retirement(context);
    });
}

namespace {

/// Run a title's GX2 event callback, `void (*)(GX2EventType, void *)`, on
/// its own thread -- hardware runs them on GX2's AppIo thread, which is the
/// same thing from the title's side.
void raise_event(coreinit::Context& context, EventType type) {
    const EventCallback callback = context.gpu.event_callback(static_cast<u32>(type));
    if (callback.function == 0 || context.memory == nullptr) {
        return;
    }
    const std::array<u32, 2> arguments{static_cast<u32>(type), callback.user_data};
    static_cast<void>(coreinit::invoke_guest_callback(
        context,
        *context.memory,
        fmt::format("GX2 {} callback", event_type_name(static_cast<u32>(type))),
        callback.function,
        arguments));
}

/// Hand the frames a flip shows to the frontend, through the hook OSScreen's
/// flips use (6.8): the same frames, read from guest memory, presented the
/// same way.
void present_flip(coreinit::Context& context) {
    if (!context.screens.on_flip || context.memory == nullptr) {
        return;
    }
    const u32 flips = context.gpu.swap_status().flips;

    // The GamePad first. The frontend presents when the TV's frame arrives,
    // so this way that present carries the GamePad's new frame as well.
    for (const ScanTarget target : {ScanTarget::Drc, ScanTarget::Tv}) {
        const ScanBuffer& buffer = context.gpu.scan_buffer(target);
        const std::optional<ppcaddr_t> frame = scan_frame(buffer, flips - 1);
        if (!frame.has_value()) {
            continue;
        }
        const coreinit::ScreenFlip flip{
            .screen = target == ScanTarget::Tv ? coreinit::ScreenId::Tv : coreinit::ScreenId::Drc,
            .address = *frame,
            .geometry = {buffer.geometry.width, buffer.geometry.height, buffer.geometry.pitch},
            .enabled = buffer.enabled,
            .flip_count = flips,
            .texture = context.gpu.scan_texture_shown(target, flips - 1),
        };
        if (!context.screens.on_flip(flip, *context.memory)) {
            // The window was closed. There is no guest call to halt the core
            // from at a vsync; the scheduler halts it at the next switch.
            context.screens.stop_requested = true;
            context.scheduler.request_halt();
            return;
        }
    }
}

void on_vsync(coreinit::Context& context, u64 index) {
    if (context.gpu.vsync(index)) {
        static_cast<void>(context.scheduler.unblock_all(flip_wait_object));
        present_flip(context);
        raise_event(context, EventType::Flip);
    }
    // The vsync may have retired submissions itself.
    arm_retirement(context);
    raise_event(context, EventType::Vsync);
}

void arm_vsync(coreinit::Context& context, u64 index) {
    context.scheduler.set_timer(vsync_timer_id, vsync_tick(index), [&context, index](u64) {
        on_vsync(context, index);
        arm_vsync(context, index + 1);
    });
}

} // namespace

void start_vsync(coreinit::Context& context, u64 now) {
    arm_vsync(context, next_vsync_index(now));
}

// --------------------------------------------------------------------------
// The entry points
// --------------------------------------------------------------------------

namespace {

[[nodiscard]] coreinit::Context& active() {
    return *coreinit::detail::active_context();
}

[[nodiscard]] Gpu& gpu() {
    return active().gpu;
}

} // namespace

std::optional<physaddr_t>
physical_address(const memory::Memory& memory, ppcaddr_t address, std::string_view what) {
    const memory::Region* const region = memory.region_of(address);
    physaddr_t physical = 0;
    if (region == nullptr || !memory::to_physical(*region, address, physical)) {
        WIIU_LOG_ERROR(Gx2, "{} at {:#010x} is not in memory the GPU can see", what, address);
        return std::nullopt;
    }
    return physical;
}

void append_registers(std::vector<u32>& out,
                      latte::pm4::Opcode opcode,
                      u32 address,
                      std::span<const u32> values) {
    using latte::pm4::Opcode;
    latte::Window window = latte::Window::Context;
    switch (opcode) {
        case Opcode::SetConfigReg:
            window = latte::Window::Config;
            break;
        case Opcode::SetCtlConst:
            window = latte::Window::CtlConst;
            break;
        case Opcode::SetAluConst:
            window = latte::Window::AluConst;
            break;
        case Opcode::SetResource:
            window = latte::Window::Resource;
            break;
        case Opcode::SetSampler:
            window = latte::Window::Sampler;
            break;
        case Opcode::SetLoopConst:
            window = latte::Window::LoopConst;
            break;
        case Opcode::SetBoolConst:
            window = latte::Window::BoolConst;
            break;
        default: // SET_CONTEXT_REG and SET_ALL_CONTEXTS
            break;
    }
    out.push_back(latte::pm4::type3_header(opcode, static_cast<u32>(1 + values.size())));
    out.push_back((address - latte::window_range(window).base) / 4);
    out.insert(out.end(), values.begin(), values.end());
}

void append_context_register(std::vector<u32>& out, u32 address, u32 value) {
    const std::array<u32, 1> values{value};
    append_registers(out, latte::pm4::Opcode::SetContextReg, address, values);
}

bool emit_context_registers(Call& call, u32 address, std::span<const u32> values) {
    std::vector<u32> packet;
    packet.reserve(2 + values.size());
    append_registers(packet, latte::pm4::Opcode::SetContextReg, address, values);
    return emit(call, packet);
}

std::vector<u32> invalidate_packets(u32 mode, physaddr_t address, u32 size) {
    namespace coher = latte::pm4::coher;
    if ((mode & ~invalidate::cpu) == 0) {
        return {};
    }
    u32 control = coher::engine_me;
    const bool everything = address == 0 && size == 0xFFFFFFFF;
    if (everything && (mode & 0xFU) != 0) {
        control |= coher::full_cache_ena;
    }
    if ((mode & (invalidate::texture | invalidate::attribute_buffer)) != 0) {
        control |= coher::tc_action_ena;
    }
    if ((mode & invalidate::uniform_block) != 0) {
        control |= coher::tc_action_ena | coher::sh_action_ena;
    }
    if ((mode & invalidate::shader) != 0) {
        control |= coher::sh_action_ena;
    }
    if ((mode & invalidate::color_buffer) != 0) {
        control |= coher::cb_dest_base_ena_all | coher::cb_action_ena;
    }
    if ((mode & invalidate::depth_buffer) != 0) {
        control |= coher::db_dest_base_ena | coher::db_action_ena;
    }
    if ((mode & invalidate::stream_out_buffer) != 0) {
        control |= coher::so_dest_base_ena_all | coher::sx_action_ena;
    }
    if ((mode & invalidate::export_buffer) != 0) {
        control |= coher::dest_base_0_ena | coher::tc_action_ena | coher::cb_action_ena |
                   coher::db_action_ena | coher::sx_action_ena;
    }
    // In 256-byte units, rounded up; all ones for everything.
    const u32 units =
        size == 0xFFFFFFFF ? 0xFFFFFFU : static_cast<u32>((static_cast<u64>(size) + 0xFF) >> 8);
    return {
        latte::pm4::type3_header(latte::pm4::Opcode::SurfaceSync, 4),
        control,
        units,
        address >> 8,
        4, // the poll interval, as decaf and Cemu write it
    };
}

bool emit_display_list_call(Call& call, ppcaddr_t list, u32 bytes, std::string_view what) {
    const std::optional<physaddr_t> physical = physical_address(call.memory, list, what);
    if (!physical.has_value()) {
        return false;
    }
    const std::array<u32, 4> packet{
        latte::pm4::type3_header(latte::pm4::Opcode::IndirectBufferPriv, 3),
        *physical,
        0,
        bytes / 4,
    };
    return emit(call, packet);
}

// Every handler that writes PM4 goes through here: a write can fill the main
// buffer and submit it, and a submission nobody arms the timer for would sit
// until the next flush.
bool emit(Call& call, std::span<const u32> words) {
    coreinit::Context& context = active();
    const u64 before = context.gpu.timeline().submitted();
    const bool written = context.gpu.write(call.state.core_id, words, call.state.timebase);
    if (context.gpu.timeline().submitted() != before) {
        arm_retirement(context);
    }
    return written;
}

namespace {

/// Submit the main command buffer and keep the GPU's timer at the next
/// deadline.
u64 flush_and_arm(Call& call) {
    coreinit::Context& context = active();
    const u64 timestamp = context.gpu.flush(call.state.timebase);
    arm_retirement(context);
    return timestamp;
}

void gx2_init(Call& call, u32 attributes_address) {
    coreinit::Context& context = active();
    context.memory = &call.memory;

    if (context.gpu.initialized()) {
        WIIU_LOG_WARN(Gx2, "GX2Init called again without GX2Shutdown");
    }

    const InitAttributes attributes = read_init_attributes(call.memory, attributes_address);
    if (attributes.command_buffer_base != 0) {
        // The pool is where hardware's GX2 carves command buffers from. The
        // main buffer here is the emulator's (see gx2.hpp), so it is noted
        // and not used.
        WIIU_LOG_DEBUG(Gx2,
                       "GX2Init: command buffer pool of {:#x} bytes at {:#010x}, not used",
                       attributes.command_buffer_pool_size,
                       attributes.command_buffer_base);
    }
    if (attributes.profile_mode != 0 || attributes.toss_stage != 0) {
        WIIU_LOG_WARN(Gx2,
                      "GX2Init asked for profile mode {} and toss stage {}; neither is modelled",
                      attributes.profile_mode,
                      attributes.toss_stage);
    }

    context.gpu.initialize(call.memory, call.state.core_id, attributes);
    start_vsync(context, call.state.timebase);

    // GX2's own start, as decaf has it: shadowing off, the registers
    // initialised, the default state, and all of it submitted.
    static_cast<void>(emit(call, unbind_context_state_packets()));
    static_cast<void>(emit(call, initial_register_packets()));
    static_cast<void>(emit(call, default_state_packets()));
    static_cast<void>(flush_and_arm(call));
    WIIU_LOG_INFO(Gx2, "GX2Init on core {}", call.state.core_id);
}

void gx2_shutdown(Call&) {
    coreinit::Context& context = active();
    context.gpu.shutdown();
    context.scheduler.clear_timer(vsync_timer_id);

    // Nothing will flip again; a thread waiting for one would wait forever.
    static_cast<void>(context.scheduler.unblock_all(flip_wait_object));
}

u32 gx2_get_main_core_id(Call&) {
    return gpu().main_core();
}

void gx2_flush(Call& call) {
    static_cast<void>(flush_and_arm(call));
}

u64 gx2_get_last_submitted_time_stamp(Call&) {
    return gpu().timeline().submitted();
}

u64 gx2_get_retired_time_stamp(Call&) {
    return gpu().timeline().retired();
}

/// Park the calling thread until `timestamp` has retired, with TRUE in r3
/// for when it wakes. Written untyped because the thread may not be the one
/// in `call.state` by the time this returns.
void wait_for(Call& call, u64 timestamp) {
    coreinit::Context& context = active();
    const latte::GpuTimeline& timeline = context.gpu.timeline();

    if (timestamp <= timeline.retired()) {
        call.state.gpr[3] = 1;
        return;
    }

    const std::optional<u64> deadline = timeline.deadline_of(timestamp);
    if (!deadline.has_value()) {
        // Nothing this title has submitted will ever get there. Hardware
        // waits out GX2's GPU timeout and returns FALSE; so does this, minus
        // the wait -- which nothing could observe except as lost time.
        WIIU_LOG_WARN(Gx2,
                      "GX2WaitTimeStamp({}) waits for a timestamp not yet submitted (last is {})",
                      timestamp,
                      timeline.submitted());
        call.state.gpr[3] = 0;
        return;
    }

    // The timer that retires it fires at the same tick the thread wakes, and
    // the scheduler services timers before it wakes sleepers -- so by the
    // time this thread runs again, the timestamp has retired.
    call.state.gpr[3] = 1;
    const u64 now = call.state.timebase;
    context.scheduler.sleep(call.state, *deadline > now ? *deadline - now : 0);
}

void gx2_wait_time_stamp(Call& call) {
    const u64 timestamp = call.next_long();
    wait_for(call, timestamp);
}

void gx2_draw_done(Call& call) {
    const u64 timestamp = flush_and_arm(call);
    wait_for(call, timestamp);
}

void gx2_begin_display_list_ex(Call& call, u32 list, u32 bytes, bool profiling) {
    coreinit::Context& context = active();
    context.memory = &call.memory;
    static_cast<void>(context.gpu.begin_display_list(call.state.core_id, list, bytes, profiling));
}

void gx2_begin_display_list(Call& call, u32 list, u32 bytes) {
    gx2_begin_display_list_ex(call, list, bytes, true);
}

u32 gx2_end_display_list(Call& call, u32 list) {
    Gpu& state = gpu();
    const DisplayList* const open = state.display_list(call.state.core_id);
    if (open == nullptr) {
        WIIU_LOG_WARN(Gx2,
                      "GX2EndDisplayList({:#010x}) on core {} with no display list open",
                      list,
                      call.state.core_id);
        return 0;
    }
    if (open->base != list) {
        WIIU_LOG_WARN(
            Gx2, "GX2EndDisplayList({:#010x}) ends the list begun at {:#010x}", list, open->base);
    }
    return state.end_display_list(call.state.core_id).value_or(0);
}

u32 gx2_get_display_list_write_status(Call& call) {
    return gpu().display_list(call.state.core_id) != nullptr ? 1U : 0U;
}

u32 gx2_get_current_display_list(Call& call, u32 out_list, u32 out_size) {
    const DisplayList* const open = gpu().display_list(call.state.core_id);
    if (open == nullptr) {
        return 0;
    }
    if (out_list != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_list, open->base));
    }
    if (out_size != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_size, open->capacity_words * 4));
    }
    return 1;
}

void gx2_call_display_list(Call& call, u32 list, u32 bytes) {
    static_cast<void>(emit_display_list_call(call, list, bytes, "GX2CallDisplayList's list"));
}

void gx2_direct_call_display_list(Call& call, u32 list, u32 bytes) {
    const std::optional<physaddr_t> physical =
        physical_address(call.memory, list, "GX2DirectCallDisplayList's list");
    if (!physical.has_value()) {
        return;
    }

    coreinit::Context& context = active();
    if (context.gpu.display_list(call.state.core_id) != nullptr) {
        // Inside a display list there is nothing to submit to; decaf calls it
        // the same as GX2CallDisplayList, and so does this.
        gx2_call_display_list(call, list, bytes);
        return;
    }

    // Whatever was written before it runs before it.
    static_cast<void>(context.gpu.flush(call.state.timebase));
    const std::array<u32, 4> packet{
        latte::pm4::type3_header(latte::pm4::Opcode::IndirectBufferPriv, 3),
        *physical,
        0,
        bytes / 4,
    };
    static_cast<void>(context.gpu.submit(packet, call.state.timebase));
    arm_retirement(context);
}

void gx2_copy_display_list(Call& call, u32 list, u32 bytes) {
    // Copied now, word for word, into whatever is being written: unlike a
    // call, the title may reuse its buffer as soon as this returns.
    std::vector<u32> words(bytes / 4);
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (!call.memory.try_read<u32>(list + static_cast<ppcaddr_t>(i * 4), words[i])) {
            WIIU_LOG_ERROR(
                Gx2, "GX2CopyDisplayList({:#010x}, {:#x}) reads unreadable memory", list, bytes);
            return;
        }
    }
    static_cast<void>(emit(call, words));
}

/// `void GX2Invalidate(GX2InvalidateMode mode, void *buffer, uint32_t
/// size)`. The CPU's part -- a data-cache flush -- needs nothing here yet;
/// it will tell the texture and buffer caches (8.13) what changed.
void gx2_invalidate(Call& call, u32 mode, u32 buffer, u32 size) {
    physaddr_t physical = 0;
    if (buffer != 0) {
        const std::optional<physaddr_t> found =
            physical_address(call.memory, buffer, "GX2Invalidate's buffer");
        if (!found.has_value()) {
            return;
        }
        physical = *found;
    }
    const std::vector<u32> packet = invalidate_packets(mode, physical, size);
    if (!packet.empty()) {
        static_cast<void>(emit(call, packet));
    }
}

void gx2_set_event_callback(Call&, u32 type, u32 function, u32 user_data) {
    if (type >= event_type_count) {
        WIIU_LOG_WARN(Gx2, "GX2SetEventCallback for event type {}, which does not exist", type);
        return;
    }
    const bool raised =
        type == static_cast<u32>(EventType::Vsync) || type == static_cast<u32>(EventType::Flip);
    WIIU_LOG_DEBUG(Gx2,
                   "{} callback set to {:#010x} ({:#010x}){}",
                   event_type_name(type),
                   function,
                   user_data,
                   raised ? "" : "; nothing raises this event yet");
    gpu().set_event_callback(type, EventCallback{.function = function, .user_data = user_data});
}

void gx2_get_event_callback(Call& call, u32 type, u32 out_function, u32 out_user_data) {
    const EventCallback callback = gpu().event_callback(type);
    if (out_function != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_function, callback.function));
    }
    if (out_user_data != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_user_data, callback.user_data));
    }
}

/// Written untyped, like every call that may block: by the time a blocking
/// call returns, `call.state` may hold another thread.
void gx2_wait_for_vsync(Call& call) {
    const u64 now = call.state.timebase;
    active().scheduler.sleep(call.state, vsync_tick(next_vsync_index(now)) - now);
}

void gx2_wait_for_flip(Call& call) {
    coreinit::Context& context = active();
    if (context.gpu.swap_status().pending() == 0) {
        return; // nothing waiting to flip: hardware returns at once too
    }
    context.scheduler.block(call.state, BlockReason::Event, flip_wait_object);
}

/// What `GX2SwapScanBuffers` leaves in the stream: the emulator's swap packet,
/// carrying "SWAP". The swap then has a submission to belong to even when
/// nothing was drawn, a PM4 capture shows where each frame ends, and the GPU
/// counts it as it runs -- which is what tells a scan-out copy which frame it
/// is filling.
inline constexpr u32 swap_marker = 0x53574150;

void gx2_swap_scan_buffers(Call& call) {
    coreinit::Context& context = active();
    Gpu& state = context.gpu;
    const u32 core = call.state.core_id;

    if (core != state.main_core() || state.display_list(core) != nullptr) {
        // Cemu sees one title do this from another core and counts nothing,
        // because nothing reaches the GPU from there either.
        WIIU_LOG_WARN(Gx2,
                      "GX2SwapScanBuffers on core {}{}; only core {}'s command buffer can carry "
                      "a swap, so it is not counted",
                      core,
                      state.display_list(core) != nullptr ? " inside a display list" : "",
                      state.main_core());
        return;
    }

    const std::array<u32, 2> marker{
        latte::pm4::type3_header(latte::pm4::Opcode::HleSwap, 1),
        swap_marker,
    };
    static_cast<void>(emit(call, marker));

    // Whatever submission the main buffer goes out as next carries it.
    state.request_swap(state.timeline().submitted() + 1);

    if (state.swap_status().pending() > max_pending_swaps) {
        // Too far ahead of the display: wait for a flip. The buffer is flushed
        // first -- a title that never flushes would otherwise be waiting for
        // swaps that cannot retire, which is a hang rather than a pace.
        static_cast<void>(flush_and_arm(call));
        context.scheduler.block(call.state, BlockReason::Event, flip_wait_object);
    }
}

u32 gx2_get_swap_interval(Call&) {
    return gpu().swap_status().interval;
}

void gx2_set_swap_interval(Call&, u32 interval) {
    static_cast<void>(gpu().set_swap_interval(interval));
}

void gx2_get_swap_status(
    Call& call, u32 out_swaps, u32 out_flips, u32 out_last_flip, u32 out_last_vsync) {
    const SwapStatus& status = gpu().swap_status();
    if (out_swaps != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_swaps, status.swaps));
    }
    if (out_flips != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_flips, status.flips));
    }
    if (out_last_flip != 0) {
        static_cast<void>(call.memory.try_write<u64>(out_last_flip, status.last_flip));
    }
    if (out_last_vsync != 0) {
        static_cast<void>(call.memory.try_write<u64>(out_last_vsync, status.last_vsync));
    }
}

} // namespace

void register_gx2(Registry& registry, coreinit::Context& context) {
    coreinit::detail::set_active_context(&context);

    registry.add("gx2", "GX2Init", gx2_init);
    registry.add("gx2", "GX2Shutdown", gx2_shutdown);
    registry.add("gx2", "GX2GetMainCoreId", gx2_get_main_core_id);
    registry.add("gx2", "GX2Flush", gx2_flush);
    registry.add("gx2", "GX2Invalidate", gx2_invalidate);

    registry.add("gx2", "GX2GetLastSubmittedTimeStamp", gx2_get_last_submitted_time_stamp);
    registry.add("gx2", "GX2GetRetiredTimeStamp", gx2_get_retired_time_stamp);
    registry.add("gx2", "GX2WaitTimeStamp", gx2_wait_time_stamp);
    registry.add("gx2", "GX2DrawDone", gx2_draw_done);

    registry.add("gx2", "GX2BeginDisplayList", gx2_begin_display_list);
    registry.add("gx2", "GX2BeginDisplayListEx", gx2_begin_display_list_ex);
    registry.add("gx2", "GX2EndDisplayList", gx2_end_display_list);
    registry.add("gx2", "GX2GetDisplayListWriteStatus", gx2_get_display_list_write_status);
    registry.add("gx2", "GX2GetCurrentDisplayList", gx2_get_current_display_list);
    registry.add("gx2", "GX2CallDisplayList", gx2_call_display_list);
    registry.add("gx2", "GX2DirectCallDisplayList", gx2_direct_call_display_list);
    registry.add("gx2", "GX2CopyDisplayList", gx2_copy_display_list);

    registry.add("gx2", "GX2WaitForVsync", gx2_wait_for_vsync);
    registry.add("gx2", "GX2WaitForFlip", gx2_wait_for_flip);
    registry.add("gx2", "GX2SwapScanBuffers", gx2_swap_scan_buffers);
    registry.add("gx2", "GX2GetSwapInterval", gx2_get_swap_interval);
    registry.add("gx2", "GX2SetSwapInterval", gx2_set_swap_interval);
    registry.add("gx2", "GX2GetSwapStatus", gx2_get_swap_status);

    registry.add("gx2", "GX2SetEventCallback", gx2_set_event_callback);
    registry.add("gx2", "GX2GetEventCallback", gx2_get_event_callback);

    register_gx2_display(registry);
    register_gx2_surface(registry);
    register_gx2_state(registry);
    register_gx2_context(registry);
    register_gx2_shader(registry);
    register_gx2_buffers(registry);
    register_gx2_texture(registry);
    register_gx2_sampler(registry);
    register_gx2_draw(registry);
}

} // namespace wiiu::hle::gx2
