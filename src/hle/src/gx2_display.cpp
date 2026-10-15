// Phase 8.1 — GX2's display calls: the scan buffers, and the outputs'
// switches and what they report.
//
// A scan buffer is what a flip shows. The title asks how big one must be,
// allocates it, and hands it over; after that the scan-out copy
// (`GX2CopyColorBufferToScanBuffer`) fills it and a flip shows it. What is
// here is the bookkeeping that comes first: which buffer, what shape, which
// format, how many frames, switched on or not.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/formats.hpp>

#include <array>

namespace wiiu::hle::gx2 {

std::string_view scan_target_name(ScanTarget target) noexcept {
    switch (target) {
        case ScanTarget::Tv:
            return "TV";
        case ScanTarget::Drc:
            return "DRC";
    }
    return "unknown";
}

std::optional<ScanTarget> scan_target_from_gx2(u32 value) noexcept {
    switch (value) {
        case 1:
            return ScanTarget::Tv;
        case 4:
            return ScanTarget::Drc;
        default:
            return std::nullopt;
    }
}

std::optional<ScanGeometry> tv_scan_geometry(u32 render_mode) noexcept {
    switch (render_mode) {
        case 1:
            return ScanGeometry{640, 480, 640};
        case 2:
            return ScanGeometry{854, 480, 854};
        case 3:
        case 4:
            return ScanGeometry{1280, 720, 1280};
        case 5:
            return ScanGeometry{1920, 1080, 1920};
        default:
            return std::nullopt;
    }
}

std::optional<ScanGeometry> drc_scan_geometry(u32 render_mode) noexcept {
    if (render_mode == 1 || render_mode == 2) {
        return ScanGeometry{854, 480, 864};
    }
    return std::nullopt;
}

u32 scan_buffer_count(u32 buffering_mode) noexcept {
    return buffering_mode >= 1 && buffering_mode <= 3 ? buffering_mode : 0;
}

u32 scan_buffer_bytes(const ScanGeometry& geometry,
                      u32 surface_format,
                      u32 buffering_mode) noexcept {
    return geometry.pitch * geometry.height * latte::bytes_per_element(surface_format) *
           scan_buffer_count(buffering_mode);
}

std::optional<ppcaddr_t> scan_frame(const ScanBuffer& buffer, u32 flip_index) noexcept {
    const u32 bytes = latte::bytes_per_element(buffer.surface_format);
    const u32 frames = scan_buffer_count(buffer.buffering_mode);
    if (!buffer.configured() || buffer.geometry.height == 0 || bytes < 4 || frames == 0) {
        return std::nullopt;
    }
    const u32 stride = buffer.geometry.pitch * buffer.geometry.height * bytes;
    return buffer.address + ((flip_index % frames) * stride);
}

namespace {

[[nodiscard]] coreinit::Context& active() {
    return *coreinit::detail::active_context();
}

[[nodiscard]] Gpu& gpu() {
    return active().gpu;
}

[[nodiscard]] std::optional<ScanGeometry> geometry_for(ScanTarget target, u32 render_mode) {
    return target == ScanTarget::Tv ? tv_scan_geometry(render_mode)
                                    : drc_scan_geometry(render_mode);
}

/// What both Calc calls do, for one output.
void calc_size(Call& call,
               ScanTarget target,
               u32 render_mode,
               u32 surface_format,
               u32 buffering_mode,
               u32 out_size,
               u32 out_scale_needed) {
    u32 size = 0;
    if (const std::optional<ScanGeometry> geometry = geometry_for(target, render_mode)) {
        size = scan_buffer_bytes(*geometry, surface_format, buffering_mode);
    }
    if (size == 0 && render_mode != 0) {
        WIIU_LOG_WARN(Gx2,
                      "{} scan buffer size asked for render mode {}, format {:#x}, buffering {}: "
                      "not a combination this knows, so zero",
                      scan_target_name(target),
                      render_mode,
                      surface_format,
                      buffering_mode);
    }

    if (out_size != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_size, size));
    }
    // The second output says whether the output will scale the buffer to the
    // screen. Cemu and decaf both report no; nothing here scales.
    if (out_scale_needed != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_scale_needed, 0));
    }
}

/// What both Set*Buffer calls do, for one output.
void set_buffer(ScanTarget target,
                u32 address,
                u32 size,
                u32 render_mode,
                u32 surface_format,
                u32 buffering_mode) {
    ScanBuffer& buffer = gpu().scan_buffer(target);
    const std::optional<ScanGeometry> geometry = geometry_for(target, render_mode);
    const u32 needed =
        geometry.has_value() ? scan_buffer_bytes(*geometry, surface_format, buffering_mode) : 0;

    if (!geometry.has_value() || needed == 0) {
        WIIU_LOG_WARN(Gx2,
                      "{} scan buffer at {:#010x}: render mode {}, format {:#x}, buffering {} "
                      "is not a combination this knows; nothing will be shown from it",
                      scan_target_name(target),
                      address,
                      render_mode,
                      surface_format,
                      buffering_mode);
    } else if (size < needed) {
        WIIU_LOG_WARN(Gx2,
                      "{} scan buffer at {:#010x} is {:#x} bytes; its frames need {:#x}",
                      scan_target_name(target),
                      address,
                      size,
                      needed);
    }

    buffer.address = address;
    buffer.size = size;
    buffer.render_mode = render_mode;
    buffer.surface_format = surface_format;
    buffer.buffering_mode = buffering_mode;
    buffer.geometry = geometry.value_or(ScanGeometry{});

    WIIU_LOG_DEBUG(Gx2,
                   "{} scan buffer: {}x{} format {:#x}, {} frame(s) at {:#010x}",
                   scan_target_name(target),
                   buffer.geometry.width,
                   buffer.geometry.height,
                   surface_format,
                   scan_buffer_count(buffering_mode),
                   address);
}

void gx2_calc_tv_size(Call& call, u32 mode, u32 format, u32 buffering, u32 out, u32 scale) {
    calc_size(call, ScanTarget::Tv, mode, format, buffering, out, scale);
}

void gx2_calc_drc_size(Call& call, u32 mode, u32 format, u32 buffering, u32 out, u32 scale) {
    calc_size(call, ScanTarget::Drc, mode, format, buffering, out, scale);
}

void gx2_set_tv_buffer(Call&, u32 address, u32 size, u32 mode, u32 format, u32 buffering) {
    set_buffer(ScanTarget::Tv, address, size, mode, format, buffering);
}

void gx2_set_drc_buffer(Call&, u32 address, u32 size, u32 mode, u32 format, u32 buffering) {
    set_buffer(ScanTarget::Drc, address, size, mode, format, buffering);
}

void gx2_set_tv_enable(Call&, bool enable) {
    gpu().scan_buffer(ScanTarget::Tv).enabled = enable;
}

void gx2_set_drc_enable(Call&, bool enable) {
    gpu().scan_buffer(ScanTarget::Drc).enabled = enable;
}

void gx2_set_tv_scale(Call&, u32 x, u32 y) {
    ScanBuffer& buffer = gpu().scan_buffer(ScanTarget::Tv);
    buffer.scale_x = x;
    buffer.scale_y = y;
}

void gx2_set_drc_scale(Call&, u32 x, u32 y) {
    ScanBuffer& buffer = gpu().scan_buffer(ScanTarget::Drc);
    buffer.scale_x = x;
    buffer.scale_y = y;
}

u32 gx2_get_system_tv_scan_mode(Call&) {
    return system_tv_scan_mode;
}

u32 gx2_get_system_tv_aspect_ratio(Call&) {
    return system_tv_aspect_ratio;
}

u32 gx2_get_system_drc_mode(Call&) {
    return system_drc_mode;
}

u32 gx2_is_video_out_ready(Call&) {
    return 1;
}

/// `GX2DRCConnectCallback GX2SetDRCConnectCallback(uint32_t slot, callback)`.
///
/// A callback set is told straight away whether its GamePad is there, which
/// is how a title learns about the one it booted with: slot 0 always is, and
/// a second one never is here -- Cemu found a title that refuses input if it
/// is told a second GamePad exists that it cannot read.
u32 gx2_set_drc_connect_callback(Call& call, u32 slot, u32 callback) {
    coreinit::Context& context = active();
    context.memory = &call.memory;
    const ppcaddr_t old = context.gpu.set_drc_connect_callback(callback);

    if (callback != 0) {
        const std::array<u32, 2> arguments{slot, slot == 0 ? 1U : 0U};
        static_cast<void>(coreinit::invoke_guest_callback(
            context, call.memory, "GX2 DRC connect callback", callback, arguments));
    }
    return old;
}

} // namespace

void register_gx2_display(Registry& registry) {
    registry.add("gx2", "GX2CalcTVSize", gx2_calc_tv_size);
    registry.add("gx2", "GX2CalcDRCSize", gx2_calc_drc_size);
    registry.add("gx2", "GX2SetTVBuffer", gx2_set_tv_buffer);
    registry.add("gx2", "GX2SetDRCBuffer", gx2_set_drc_buffer);
    registry.add("gx2", "GX2SetTVEnable", gx2_set_tv_enable);
    registry.add("gx2", "GX2SetDRCEnable", gx2_set_drc_enable);
    registry.add("gx2", "GX2SetTVScale", gx2_set_tv_scale);
    registry.add("gx2", "GX2SetDRCScale", gx2_set_drc_scale);

    registry.add("gx2", "GX2GetSystemTVScanMode", gx2_get_system_tv_scan_mode);
    registry.add("gx2", "GX2GetSystemTVAspectRatio", gx2_get_system_tv_aspect_ratio);
    registry.add("gx2", "GX2GetSystemDRCMode", gx2_get_system_drc_mode);
    registry.add("gx2", "GX2GetSystemDRCScanMode", gx2_get_system_drc_mode);
    registry.add("gx2", "GX2IsVideoOutReady", gx2_is_video_out_ready);
    registry.add("gx2", "GX2SetDRCConnectCallback", gx2_set_drc_connect_callback);
}

} // namespace wiiu::hle::gx2
