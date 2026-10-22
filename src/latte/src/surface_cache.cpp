// Phase 8.13, first form — see surface_cache.hpp.

#include <wiiu/latte/surface_cache.hpp>

#include <fmt/format.h>

namespace wiiu::latte {

namespace {

/// FMT_8_8_8_8: the colour format of every GX2 R8_G8_B8_A8 surface format.
constexpr u32 format_8_8_8_8 = 0x1A;

/// The largest texture side Metal makes on Apple GPUs, and more than any
/// Latte surface needs: a size past it is a register misread.
constexpr u32 max_side = 16384;

} // namespace

ColorBufferKey ColorBufferKey::from(u32 base, u32 size, u32 info, u32 view) noexcept {
    const CbColorInfo fields{.raw = info};
    return ColorBufferKey{.base = base,
                          .size = size,
                          .format = fields.format,
                          .number_type = static_cast<u32>(fields.number_type.value()),
                          .array_mode = static_cast<u32>(fields.array_mode.value()),
                          .comp_swap = static_cast<u32>(fields.comp_swap.value()),
                          .slice = CbColorView{.raw = view}.slice_start};
}

ColorBufferKey ColorBufferKey::from(const RegisterFile& registers, u32 target) noexcept {
    const u32 offset = 4 * target;
    return from(registers.read(reg::cb_color0_base + offset),
                registers.read(reg::cb_color0_size + offset),
                registers.read(reg::cb_color0_info + offset),
                registers.read(reg::cb_color0_view + offset));
}

u32 ColorBufferKey::width() const noexcept {
    return (CbColorSize{.raw = size}.pitch_tile_max + 1) * 8;
}

u32 ColorBufferKey::height() const noexcept {
    const CbColorSize fields{.raw = size};
    const u32 pitch_tiles = fields.pitch_tile_max + 1;
    return ((fields.slice_tile_max + 1) / pitch_tiles) * 8;
}

std::optional<gpu::TextureFormat> color_texture_format(const ColorBufferKey& key) {
    if (key.format != format_8_8_8_8 || key.comp_swap != static_cast<u32>(CbCompSwap::Std)) {
        return std::nullopt;
    }
    switch (static_cast<CbNumberType>(key.number_type)) {
        case CbNumberType::Unorm:
            return gpu::TextureFormat::RGBA8Unorm;
        case CbNumberType::Srgb:
            return gpu::TextureFormat::RGBA8UnormSrgb;
        default:
            return std::nullopt;
    }
}

std::optional<ColorSurface> SurfaceCache::color_buffer(const ColorBufferKey& key,
                                                       std::string& error) {
    if (const auto found = colors_.find(key); found != colors_.end()) {
        ++stats_.hits;
        return found->second;
    }

    const std::optional<gpu::TextureFormat> format = color_texture_format(key);
    if (!format.has_value()) {
        ++stats_.refused;
        error = fmt::format("a colour buffer of format {:#x}, number type {} and component swap "
                            "{} is not rendered to yet",
                            key.format,
                            key.number_type,
                            key.comp_swap);
        return std::nullopt;
    }
    const u32 width = key.width();
    const u32 height = key.height();
    if (height == 0 || width > max_side || height > max_side) {
        ++stats_.refused;
        error = fmt::format("a colour buffer {} by {} pixels (CB_COLOR_SIZE {:#x}) makes no "
                            "texture",
                            width,
                            height,
                            key.size);
        return std::nullopt;
    }

    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = *format;
    desc.usage = gpu::TextureUsage::RenderTarget | gpu::TextureUsage::Sampled;
    desc.storage = gpu::Storage::Private;
    desc.label = "colour buffer";
    const gpu::TextureHandle texture = backend_->create_texture(desc);
    if (!texture.valid()) {
        ++stats_.refused;
        error =
            fmt::format("the backend made no {} by {} texture for a colour buffer", width, height);
        return std::nullopt;
    }

    ++stats_.misses;
    const ColorSurface surface{
        .texture = texture, .format = *format, .width = width, .height = height};
    colors_.emplace(key, surface);
    return surface;
}

DepthBufferKey DepthBufferKey::from(u32 base, u32 size, u32 info, u32 view) noexcept {
    const DbDepthInfo fields{.raw = info};
    return DepthBufferKey{.base = base,
                          .size = size,
                          .format = static_cast<u32>(fields.format.value()),
                          .array_mode = static_cast<u32>(fields.array_mode.value()),
                          .slice = DbDepthView{.raw = view}.slice_start};
}

DepthBufferKey DepthBufferKey::from(const RegisterFile& registers) noexcept {
    return from(registers.read(reg::db_depth_base),
                registers.read(reg::db_depth_size),
                registers.read(reg::db_depth_info),
                registers.read(reg::db_depth_view));
}

// DB_DEPTH_SIZE is laid out as CB_COLOR_SIZE is.
u32 DepthBufferKey::width() const noexcept {
    return (DbDepthSize{.raw = size}.pitch_tile_max + 1) * 8;
}

u32 DepthBufferKey::height() const noexcept {
    const DbDepthSize fields{.raw = size};
    const u32 pitch_tiles = fields.pitch_tile_max + 1;
    return ((fields.slice_tile_max + 1) / pitch_tiles) * 8;
}

bool DepthBufferKey::has_stencil() const noexcept {
    switch (static_cast<DbFormat>(format)) {
        case DbFormat::Depth8_24:
        case DbFormat::Depth8_24Float:
        case DbFormat::DepthX24_8_32Float:
            return true;
        default:
            return false;
    }
}

std::optional<gpu::TextureFormat> depth_texture_format(const DepthBufferKey& key) {
    switch (static_cast<DbFormat>(key.format)) {
        case DbFormat::Depth16:
        case DbFormat::DepthX8_24:
        case DbFormat::DepthX8_24Float:
        case DbFormat::Depth32Float:
            return gpu::TextureFormat::Depth32Float;
        case DbFormat::Depth8_24:
        case DbFormat::Depth8_24Float:
        case DbFormat::DepthX24_8_32Float:
            return gpu::TextureFormat::Depth32FloatStencil8;
        case DbFormat::Invalid:
            break;
    }
    return std::nullopt;
}

std::optional<DepthSurface> SurfaceCache::depth_buffer(const DepthBufferKey& key,
                                                       std::string& error) {
    if (const auto found = depths_.find(key); found != depths_.end()) {
        ++stats_.hits;
        return found->second;
    }
    const std::optional<gpu::TextureFormat> format = depth_texture_format(key);
    if (!format.has_value()) {
        ++stats_.refused;
        error = fmt::format("a depth buffer of depth format {} is not one there is", key.format);
        return std::nullopt;
    }
    const u32 width = key.width();
    const u32 height = key.height();
    if (height == 0 || width > max_side || height > max_side) {
        ++stats_.refused;
        error = fmt::format("a depth buffer {} by {} pixels (DB_DEPTH_SIZE {:#x}) makes no texture",
                            width,
                            height,
                            key.size);
        return std::nullopt;
    }

    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = *format;
    desc.usage = gpu::TextureUsage::RenderTarget | gpu::TextureUsage::Sampled;
    desc.storage = gpu::Storage::Private;
    desc.label = "depth buffer";
    const gpu::TextureHandle texture = backend_->create_texture(desc);
    if (!texture.valid()) {
        ++stats_.refused;
        error =
            fmt::format("the backend made no {} by {} texture for a depth buffer", width, height);
        return std::nullopt;
    }

    ++stats_.misses;
    const DepthSurface surface{.texture = texture,
                               .format = *format,
                               .width = width,
                               .height = height,
                               .has_stencil = key.has_stencil()};
    depths_.emplace(key, surface);
    return surface;
}

std::optional<DepthSurface> SurfaceCache::find(const DepthBufferKey& key) const {
    if (const auto found = depths_.find(key); found != depths_.end()) {
        return found->second;
    }
    return std::nullopt;
}

std::optional<ColorSurface> SurfaceCache::find(const ColorBufferKey& key) const {
    if (const auto found = colors_.find(key); found != colors_.end()) {
        return found->second;
    }
    return std::nullopt;
}

void SurfaceCache::clear() {
    for (const auto& [key, surface] : colors_) {
        backend_->destroy(surface.texture);
    }
    colors_.clear();
    for (const auto& [key, surface] : depths_) {
        backend_->destroy(surface.texture);
    }
    depths_.clear();
}

} // namespace wiiu::latte
