// Phase 8.13, textures' first form — see texture_cache.hpp.

#include <wiiu/latte/texture_cache.hpp>
#include <wiiu/memory/map.hpp>

#include <fmt/format.h>

#include <cstring>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace wiiu::latte {

namespace {

/// The bank and pipe swizzle a macro-tiled level's base register carries:
/// bits 8 to 10 of the address, as a GX2 swizzle word holds it.
constexpr u32 base_swizzle_mask = 0x700;

/// WORD4's fields that say how the image's bits are read: the components'
/// signs, the number format, SRF_MODE_ALL, FORCE_DEGAMMA and ENDIAN_SWAP.
constexpr u32 word4_format_mask = 0x3FFF;

/// FMT_8, FMT_8_8 and FMT_8_8_8_8.
constexpr u32 fmt_8 = 0x01;
constexpr u32 fmt_8_8 = 0x07;
constexpr u32 fmt_8_8_8_8 = 0x1A;

[[nodiscard]] std::string_view dim_name(TexDim dim) noexcept {
    switch (dim) {
        case TexDim::D1:
            return "1D";
        case TexDim::D2:
            return "2D";
        case TexDim::D3:
            return "3D";
        case TexDim::Cube:
            return "cube";
        case TexDim::D1Array:
            return "1D array";
        case TexDim::D2Array:
            return "2D array";
        case TexDim::D2Msaa:
            return "multisampled 2D";
        case TexDim::D2ArrayMsaa:
            return "multisampled 2D array";
    }
    return "unknown";
}

/// The Metal format that reads a Latte format's bytes as WORD4 says to, or
/// nothing for one not uploaded yet.
[[nodiscard]] std::optional<gpu::TextureFormat> metal_format(u32 data_format,
                                                             TexResourceWord4 word4) noexcept {
    const bool unsigned_norm = word4.format_comp_x == FormatComp::Unsigned &&
                               word4.format_comp_y == FormatComp::Unsigned &&
                               word4.format_comp_z == FormatComp::Unsigned &&
                               word4.format_comp_w == FormatComp::Unsigned &&
                               word4.num_format_all == NumFormat::Norm &&
                               word4.endian_swap == Endian::None;
    if (!unsigned_norm) {
        return std::nullopt;
    }
    const bool srgb = word4.force_degamma;
    switch (data_format) {
        case fmt_8:
            return srgb ? std::nullopt : std::optional{gpu::TextureFormat::R8Unorm};
        case fmt_8_8:
            return srgb ? std::nullopt : std::optional{gpu::TextureFormat::RG8Unorm};
        case fmt_8_8_8_8:
            return srgb ? gpu::TextureFormat::RGBA8UnormSrgb : gpu::TextureFormat::RGBA8Unorm;
        default:
            return std::nullopt;
    }
}

[[nodiscard]] constexpr bool macro_tiled(TileMode mode) noexcept {
    return mode >= TileMode::Tiled2DThin1 && mode != TileMode::LinearSpecial;
}

/// FNV-1a a 64-bit word at a time: any changed word changes the result, and
/// it is eight times quicker over a texture than a byte at a time.
[[nodiscard]] u64 hash_bytes(std::span<const u8> bytes) noexcept {
    constexpr u64 prime = 0x100000001B3ULL;
    u64 hash = 0xCBF29CE484222325ULL;
    std::size_t at = 0;
    for (; at + 8 <= bytes.size(); at += 8) {
        u64 word = 0;
        std::memcpy(&word, bytes.data() + at, sizeof(word));
        hash = (hash ^ word) * prime;
    }
    for (; at < bytes.size(); ++at) {
        hash = (hash ^ bytes[at]) * prime;
    }
    return hash;
}

} // namespace

std::optional<TextureImage> texture_image(const TextureResource& resource, std::string& error) {
    if (!resource.valid()) {
        error = fmt::format("resource type {} is not a valid texture",
                            static_cast<u32>(resource.word6.type.value()));
        return std::nullopt;
    }
    const TexDim dim = resource.word0.dim;
    if (dim != TexDim::D2) {
        error = fmt::format("a {} texture is not uploaded yet", dim_name(dim));
        return std::nullopt;
    }
    const u32 data_format = resource.word1.data_format;
    const std::optional<gpu::TextureFormat> format = metal_format(data_format, resource.word4);
    if (!format.has_value()) {
        const TexResourceWord4 w4 = resource.word4;
        error = fmt::format("format {:#04x} read as number format {}, components {}{}{}{}{}{} "
                            "is not uploaded yet",
                            data_format,
                            static_cast<u32>(w4.num_format_all.value()),
                            static_cast<u32>(w4.format_comp_x.value()),
                            static_cast<u32>(w4.format_comp_y.value()),
                            static_cast<u32>(w4.format_comp_z.value()),
                            static_cast<u32>(w4.format_comp_w.value()),
                            w4.force_degamma ? ", sRGB" : "",
                            w4.endian_swap.value() != Endian::None ? ", byte-swapped" : "");
        return std::nullopt;
    }

    TextureImage out;
    out.format = *format;
    out.desc.tile_mode = static_cast<TileMode>(resource.word0.tile_mode.value());
    out.desc.format = data_format;
    out.desc.width = resource.word0.width + 1;
    out.desc.height = resource.word1.height + 1;
    out.desc.depth = resource.word0.tile_type;
    const std::optional<SurfaceLayout> layout = compute_layout(out.desc);
    if (!layout.has_value()) {
        error = fmt::format("a {}x{} {} texture of format {:#04x} makes no layout",
                            out.desc.width,
                            out.desc.height,
                            tile_mode_name(out.desc.tile_mode),
                            data_format);
        return std::nullopt;
    }
    out.layout = *layout;
    const u32 pitch = (resource.word0.pitch + 1) * 8;
    if (layout->pitch != pitch) {
        error = fmt::format("a {}x{} {} texture is laid out {} pixels to a row, but its resource "
                            "says {}",
                            out.desc.width,
                            out.desc.height,
                            tile_mode_name(out.desc.tile_mode),
                            layout->pitch,
                            pitch);
        return std::nullopt;
    }

    out.address = resource.base;
    if (macro_tiled(out.desc.tile_mode)) {
        out.swizzle = split_swizzle(resource.base & base_swizzle_mask);
        out.address = resource.base & ~base_swizzle_mask;
    }
    const Extent extent = level_extent(out.desc);
    out.width = extent.width;
    out.height = extent.height;
    out.bytes_per_element = layout->bits_per_element / 8;
    return out;
}

std::optional<CachedTexture> TextureCache::texture(const TextureResource& resource,
                                                   std::string& error) {
    const std::optional<TextureImage> image = texture_image(resource, error);
    if (!image.has_value()) {
        return std::nullopt;
    }
    ppcaddr_t address = 0;
    if (!memory::to_virtual(memory_->regions(), image->address, address)) {
        error = fmt::format("the texture at physical {:#010x} is in no memory the CPU maps",
                            image->address);
        return std::nullopt;
    }
    if (image->layout.size > std::numeric_limits<u32>::max()) {
        error =
            fmt::format("a texture of {} bytes is past what one read can take", image->layout.size);
        return std::nullopt;
    }
    const auto size = static_cast<u32>(image->layout.size);
    const u8* const host = memory_->translate(address, size, memory::Access::Read);
    if (host == nullptr) {
        error = fmt::format(
            "the {}-byte texture at {:#010x} is not readable guest memory", size, address);
        return std::nullopt;
    }
    const std::span<const u8> level{host, size};
    const u64 hash = hash_bytes(level);

    const Key key{.base = resource.base,
                  .word0 = resource.word0.raw,
                  .word1 = resource.word1.raw,
                  .number = resource.word4.raw & word4_format_mask};
    auto found = entries_.find(key);
    if (found != entries_.end() && found->second.hash == hash) {
        ++stats_.hits;
        return found->second.texture;
    }

    const u32 row = image->width * image->bytes_per_element;
    std::vector<u8> rows(static_cast<std::size_t>(row) * image->height);
    if (!detile(image->desc, image->layout, level, 0, 0, image->swizzle, rows)) {
        error = fmt::format("the elements of the {}x{} texture at {:#010x} fall outside it",
                            image->width,
                            image->height,
                            address);
        return std::nullopt;
    }

    gpu::TextureDesc desc;
    desc.width = image->width;
    desc.height = image->height;
    desc.format = image->format;
    desc.usage = gpu::TextureUsage::Sampled;
    desc.storage = gpu::Storage::Shared;
    desc.label = "texture";
    const gpu::TextureHandle handle = backend_->create_texture(desc);
    if (!handle.valid() ||
        !backend_->write_texture(handle, gpu::Rect{0, 0, image->width, image->height}, rows, row)) {
        if (handle.valid()) {
            backend_->destroy(handle);
        }
        error = fmt::format("the backend made no {}x{} {} texture",
                            image->width,
                            image->height,
                            gpu::texture_format_name(image->format));
        return std::nullopt;
    }

    const CachedTexture texture{
        .texture = handle, .format = image->format, .width = image->width, .height = image->height};
    if (found != entries_.end()) {
        // Rewritten: the old texture goes, once the frames using it are done.
        backend_->destroy(found->second.texture.texture);
        found->second = Entry{.hash = hash, .texture = texture};
    } else {
        if (entries_.size() >= max_entries) {
            clear();
        }
        entries_.emplace(key, Entry{.hash = hash, .texture = texture});
    }
    ++stats_.uploads;
    stats_.uploaded_bytes += size;
    return texture;
}

void TextureCache::clear() {
    for (const auto& [key, entry] : entries_) {
        backend_->destroy(entry.texture.texture);
    }
    entries_.clear();
}

} // namespace wiiu::latte
