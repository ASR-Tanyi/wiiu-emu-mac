// Phase 8.1 — GX2's surface calls. See gx2_surface.hpp for the model.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/color_buffer.hpp>
#include <wiiu/latte/depth_buffer.hpp>
#include <wiiu/latte/formats.hpp>
#include <wiiu/latte/pm4.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

namespace wiiu::hle::gx2 {

namespace {

using latte::TileMode;

/// 2D- or 3D-tiled. Read the way GX2 reads a title's tile mode: anything from
/// 2D thin1 up but linear-special, so a value past the modes counts too, as it
/// does in decaf and Cemu (its layout uses the low four bits).
[[nodiscard]] constexpr bool is_macro_tiled(u32 mode) noexcept {
    return mode >= static_cast<u32>(TileMode::Tiled2DThin1) &&
           mode != static_cast<u32>(TileMode::LinearSpecial);
}

/// How many levels a full chain from this size has. GX2 caps a surface's
/// `mipLevels` at it, and leaves a surface asking for one level alone.
[[nodiscard]] u32 full_chain_levels(const Surface& surface) {
    if (surface.mip_levels <= 1) {
        return 1;
    }
    const auto bits = [](u32 value) { return static_cast<u32>(std::bit_width(value)); };
    u32 levels = std::max(bits(surface.width), bits(surface.height));
    if (static_cast<SurfaceDim>(surface.dim.value()) == SurfaceDim::Texture3D) {
        levels = std::max(levels, bits(surface.depth));
    }
    return levels;
}

/// The sample count a `GX2AAMode` names. Only the low two bits mean
/// anything; a value beyond them is a surface nobody initialised.
[[nodiscard]] u32 sample_count(u32 aa) noexcept {
    return 1U << (aa & 3);
}

/// CB_COLORn_INFO's number type, from a surface format's high bits.
[[nodiscard]] std::optional<latte::CbNumberType> number_type(u32 surface_format) noexcept {
    switch (surface_format >> 8) {
        case 0x0:
            return latte::CbNumberType::Unorm;
        case 0x1:
            return latte::CbNumberType::Uint;
        case 0x2:
            return latte::CbNumberType::Snorm;
        case 0x3:
            return latte::CbNumberType::Sint;
        case 0x4:
            return latte::CbNumberType::Srgb;
        case 0x8:
            return latte::CbNumberType::Float;
        default:
            return std::nullopt;
    }
}

/// Whether the shader's export of a colour format fits the 16-bit normalised
/// path, by colour format. decaf's and Cemu's tables, which are identical.
[[nodiscard]] latte::CbSourceFormat export_source_format(u32 color_format) noexcept {
    constexpr std::array<u8, 0x40> norm = {
        1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, //
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, //
        1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, //
        0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    };
    return norm[color_format & 0x3F] != 0 ? latte::CbSourceFormat::ExportNorm
                                          : latte::CbSourceFormat::ExportFull;
}

} // namespace

latte::SurfaceDesc surface_desc(const Surface& surface, u32 level) {
    latte::SurfaceDesc desc;
    const u32 mode = surface.tile_mode;
    desc.tile_mode = mode == static_cast<u32>(TileMode::LinearSpecial)
                         ? TileMode::LinearSpecial
                         : static_cast<TileMode>(mode & 0xF);
    desc.format = surface.format;
    desc.width = surface.width;
    desc.samples = sample_count(surface.aa);
    desc.mip_level = level;
    desc.depth = (surface.use & surface_use_depth_buffer) != 0;
    desc.display = (surface.use & surface_use_scan_buffer) != 0;

    // `latte::compute_layout` takes each level's width and height from level
    // 0's, but its slices as given -- so a volume's are halved here.
    desc.height = 1;
    desc.slices = 1;
    switch (static_cast<SurfaceDim>(surface.dim.value())) {
        case SurfaceDim::Texture1D:
            break;
        case SurfaceDim::Texture2D:
        case SurfaceDim::Texture2DMsaa:
            desc.height = surface.height;
            break;
        case SurfaceDim::Texture3D:
            desc.height = surface.height;
            desc.slices = std::max<u32>(1, surface.depth >> level);
            desc.volume = true;
            break;
        case SurfaceDim::TextureCube:
            desc.height = surface.height;
            desc.slices = std::max<u32>(6, surface.depth);
            desc.cube = true;
            break;
        case SurfaceDim::Texture1DArray:
            desc.slices = surface.depth;
            break;
        case SurfaceDim::Texture2DArray:
        case SurfaceDim::Texture2DMsaaArray:
            desc.height = surface.height;
            desc.slices = surface.depth;
            break;
    }
    return desc;
}

std::optional<latte::SurfaceLayout> surface_layout(const Surface& surface, u32 level) {
    return latte::compute_layout(surface_desc(surface, level));
}

std::optional<ppcaddr_t> level_address(const Surface& surface, u32 level) {
    if (level == 0) {
        if (surface.image == 0) {
            return std::nullopt;
        }
        return surface.image.value();
    }
    if (surface.mipmaps == 0 || level >= max_mip_levels) {
        return std::nullopt;
    }
    const u32 offset = level == 1 ? 0 : surface.mip_level_offset[level - 1].value();
    return surface.mipmaps + offset;
}

latte::BankPipeSwizzle level_swizzle(const Surface& surface, u32 level) {
    const u32 first_untiled = (surface.swizzle >> 16) & 0xFF;
    if (!is_macro_tiled(surface.tile_mode) || level >= first_untiled) {
        return {};
    }
    return latte::split_swizzle(surface.swizzle & 0xFFFF);
}

namespace {

/// A guest structure as packet words: each big-endian word of it, in host
/// order, as the command buffer holds words. And back.
template <typename T>
[[nodiscard]] std::array<u32, sizeof(T) / 4> struct_to_words(const T& value) {
    std::array<u8, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    std::array<u32, sizeof(T) / 4> words{};
    for (u32 i = 0; i < words.size(); ++i) {
        const u8* const at = bytes.data() + (i * 4);
        words[i] = (u32{at[0]} << 24) | (u32{at[1]} << 16) | (u32{at[2]} << 8) | u32{at[3]};
    }
    return words;
}

template <typename T>
[[nodiscard]] T struct_from_words(std::span<const u32, sizeof(T) / 4> words) {
    std::array<u8, sizeof(T)> bytes{};
    for (u32 i = 0; i < words.size(); ++i) {
        for (u32 b = 0; b < 4; ++b) {
            bytes[(i * 4) + b] = static_cast<u8>(words[i] >> (24 - (8 * b)));
        }
    }
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
}

} // namespace

std::array<u32, color_buffer_words> color_buffer_to_words(const ColorBuffer& buffer) {
    return struct_to_words(buffer);
}

ColorBuffer color_buffer_from_words(std::span<const u32, color_buffer_words> words) {
    return struct_from_words<ColorBuffer>(words);
}

std::array<u32, depth_buffer_words> depth_buffer_to_words(const DepthBuffer& buffer) {
    return struct_to_words(buffer);
}

DepthBuffer depth_buffer_from_words(std::span<const u32, depth_buffer_words> words) {
    return struct_from_words<DepthBuffer>(words);
}

namespace {

/// The level and slices a buffer's view names, laid out -- what the clears
/// and the copy start from. A view of no slices is taken as one, as decaf
/// takes it.
struct ViewedLevel {
    latte::SurfaceDesc desc;
    latte::SurfaceLayout layout;
    ppcaddr_t address = 0;
    u32 first_slice = 0;
    u32 slices = 0;
};

[[nodiscard]] std::string describe(const Surface& s) {
    return fmt::format("{}x{} format {:#x} tile mode {} at {:#010x}",
                       s.width.value(),
                       s.height.value(),
                       s.format.value(),
                       s.tile_mode.value(),
                       s.image.value());
}

[[nodiscard]] std::optional<ViewedLevel>
viewed_level(const Surface& s, u32 level, u32 first_slice, u32 slices, std::string& problem) {
    const std::optional<latte::SurfaceLayout> layout = surface_layout(s, level);
    if (!layout.has_value()) {
        problem = fmt::format("level {} of a {} cannot be laid out", level, describe(s));
        return std::nullopt;
    }
    const std::optional<ppcaddr_t> address = level_address(s, level);
    if (!address.has_value()) {
        problem = fmt::format("a {} has no memory for level {}", describe(s), level);
        return std::nullopt;
    }
    ViewedLevel viewed{
        .desc = surface_desc(s, level),
        .layout = *layout,
        .address = *address,
        .first_slice = first_slice,
        .slices = std::max<u32>(1, slices),
    };
    if (viewed.first_slice >= layout->depth) {
        problem = fmt::format(
            "slice {} of a {} with {} slices", viewed.first_slice, describe(s), layout->depth);
        return std::nullopt;
    }
    viewed.slices = std::min(viewed.slices, layout->depth - viewed.first_slice);
    return viewed;
}

[[nodiscard]] bool is_thick(TileMode mode) noexcept {
    switch (mode) {
        case TileMode::Tiled1DThick:
        case TileMode::Tiled2DThick:
        case TileMode::Tiled2BThick:
        case TileMode::Tiled3DThick:
        case TileMode::Tiled3BThick:
            return true;
        default:
            return false;
    }
}

/// Write one element's bytes -- those of them `write` names -- over every
/// element of the viewed slices. In the thin modes a slice is one run of
/// bytes, so walking the run element by element reaches every element of it,
/// whatever the tiling; thick modes interleave four slices, and are refused.
[[nodiscard]] std::string fill_viewed_slices(memory::Memory& memory,
                                             const ViewedLevel& viewed,
                                             std::span<const u8> element,
                                             std::span<const bool> write,
                                             std::string_view call,
                                             const Surface& s) {
    if (is_thick(viewed.layout.tile_mode)) {
        return fmt::format("{}: a thick-tiled buffer ({}) is not cleared yet", call, describe(s));
    }

    const u64 slice_bytes = viewed.layout.slice_size;
    const u64 start = viewed.address + (viewed.first_slice * slice_bytes);
    const u64 bytes = viewed.slices * slice_bytes;
    u8* const host = start + bytes <= 0x1'0000'0000ULL
                         ? memory.translate(static_cast<ppcaddr_t>(start),
                                            static_cast<u32>(bytes),
                                            memory::Access::Write)
                         : nullptr;
    if (host == nullptr) {
        return fmt::format("{}: {:#x} bytes at {:#010x} for a {} are not writable guest memory",
                           call,
                           bytes,
                           start,
                           describe(s));
    }

    const std::size_t size = element.size();
    const bool whole = std::all_of(write.begin(), write.end(), [](bool b) { return b; });
    for (u64 at = 0; at + size <= bytes; at += size) {
        if (whole) {
            std::memcpy(host + at, element.data(), size);
            continue;
        }
        for (std::size_t i = 0; i < size; ++i) {
            if (write[i]) {
                host[at + i] = element[i];
            }
        }
    }
    return {};
}

} // namespace

std::string clear_color_buffer(memory::Memory& memory,
                               const ColorBuffer& buffer,
                               const std::array<f32, 4>& rgba) {
    std::string problem;
    const std::optional<ViewedLevel> viewed = viewed_level(
        buffer.surface, buffer.view_mip, buffer.view_first_slice, buffer.view_num_slices, problem);
    if (!viewed.has_value()) {
        return "GX2ClearColor: " + problem;
    }
    const Surface& s = buffer.surface;

    const std::optional<latte::EncodedColor> color = latte::encode_color(s.format, rgba);
    if (!color.has_value() || color->size * 8 != viewed->layout.bits_per_element) {
        return fmt::format("GX2ClearColor: clearing format {:#x} is not supported yet",
                           s.format.value());
    }
    std::array<bool, 16> all{};
    all.fill(true);
    return fill_viewed_slices(memory,
                              *viewed,
                              std::span<const u8>(color->bytes.data(), color->size),
                              std::span<const bool>(all.data(), color->size),
                              "GX2ClearColor",
                              s);
}

std::string clear_depth_stencil_buffer(
    memory::Memory& memory, const DepthBuffer& buffer, f32 depth, u8 stencil, u32 flags) {
    const bool write_depth = (flags & clear_depth) != 0;
    const bool write_stencil = (flags & clear_stencil) != 0;
    if (!write_depth && !write_stencil) {
        return {};
    }

    std::string problem;
    const std::optional<ViewedLevel> viewed = viewed_level(
        buffer.surface, buffer.view_mip, buffer.view_first_slice, buffer.view_num_slices, problem);
    if (!viewed.has_value()) {
        return "GX2ClearDepthStencilEx: " + problem;
    }
    const Surface& s = buffer.surface;

    const std::optional<latte::EncodedDepthStencil> element =
        latte::encode_depth_stencil(s.format, depth, stencil, write_depth, write_stencil);
    if (!element.has_value() || element->size * 8 != viewed->layout.bits_per_element) {
        return fmt::format("GX2ClearDepthStencilEx: clearing depth format {:#x} is not "
                           "supported yet",
                           s.format.value());
    }
    return fill_viewed_slices(memory,
                              *viewed,
                              std::span<const u8>(element->bytes.data(), element->size),
                              std::span<const bool>(element->write.data(), element->size),
                              "GX2ClearDepthStencilEx",
                              s);
}

std::string copy_to_scan_frame(memory::Memory& memory,
                               const ColorBuffer& buffer,
                               const ScanBuffer& scan,
                               ppcaddr_t frame) {
    std::string problem;
    const std::optional<ViewedLevel> viewed = viewed_level(
        buffer.surface, buffer.view_mip, buffer.view_first_slice, buffer.view_num_slices, problem);
    if (!viewed.has_value()) {
        return "GX2CopyColorBufferToScanBuffer: " + problem;
    }
    const Surface& s = buffer.surface;

    const u32 bytes = viewed->layout.bits_per_element / 8;
    if (latte::color_format(s.format) != latte::color_format(scan.surface_format) ||
        bytes != latte::bytes_per_element(scan.surface_format)) {
        return fmt::format("GX2CopyColorBufferToScanBuffer: copying format {:#x} into a scan "
                           "buffer of format {:#x} is not supported yet",
                           s.format.value(),
                           scan.surface_format);
    }

    const u8* const level = memory.translate(
        viewed->address, static_cast<u32>(viewed->layout.size), memory::Access::Read);
    if (level == nullptr) {
        return fmt::format("GX2CopyColorBufferToScanBuffer: the {} is not readable guest memory",
                           describe(s));
    }
    const latte::Extent extent = latte::level_extent(viewed->desc);
    std::vector<u8> rows(static_cast<std::size_t>(extent.width) * extent.height * bytes);
    if (!latte::detile(viewed->desc,
                       viewed->layout,
                       std::span<const u8>(level, viewed->layout.size),
                       viewed->first_slice,
                       0,
                       level_swizzle(s, buffer.view_mip),
                       rows)) {
        return fmt::format("GX2CopyColorBufferToScanBuffer: the elements of a {} fall outside it",
                           describe(s));
    }

    const ScanGeometry& geometry = scan.geometry;
    const u64 frame_bytes = static_cast<u64>(geometry.pitch) * geometry.height * bytes;
    u8* const out = memory.translate(frame, static_cast<u32>(frame_bytes), memory::Access::Write);
    if (out == nullptr) {
        return fmt::format("GX2CopyColorBufferToScanBuffer: the {}x{} scan frame at {:#010x} is "
                           "not writable guest memory",
                           geometry.width,
                           geometry.height,
                           frame);
    }

    const u64 out_row = static_cast<u64>(geometry.pitch) * bytes;
    for (u32 y = 0; y < geometry.height; ++y) {
        const u32 sy = static_cast<u32>((static_cast<u64>(y) * extent.height) / geometry.height);
        const u8* const in = rows.data() + (static_cast<u64>(sy) * extent.width * bytes);
        u8* const row = out + (y * out_row);
        if (extent.width == geometry.width) {
            std::memcpy(row, in, static_cast<std::size_t>(geometry.width) * bytes);
            continue;
        }
        for (u32 x = 0; x < geometry.width; ++x) {
            const u32 sx = static_cast<u32>((static_cast<u64>(x) * extent.width) / geometry.width);
            std::memcpy(
                row + (static_cast<u64>(x) * bytes), in + (static_cast<u64>(sx) * bytes), bytes);
        }
    }
    return {};
}

bool calc_surface_size_and_alignment(Surface& surface) {
    // Worked on a copy, so that a level which cannot be laid out leaves the
    // title's surface as it was rather than half-updated.
    Surface s = surface;

    const bool depth_buffer = (s.use & surface_use_depth_buffer) != 0;
    const bool color_buffer = (s.use & surface_use_color_buffer) != 0;
    const bool bad_align = s.tile_mode == tile_mode_default_bad_align;
    const auto dim = static_cast<SurfaceDim>(s.dim.value());

    bool chosen = false;
    if (s.tile_mode == tile_mode_default || bad_align) {
        if (dim != SurfaceDim::Texture1D || s.aa != 0 || depth_buffer) {
            s.tile_mode = static_cast<u32>(dim != SurfaceDim::Texture3D || color_buffer
                                               ? TileMode::Tiled2DThin1
                                               : TileMode::Tiled2DThick);
            chosen = true;
        } else {
            s.tile_mode = static_cast<u32>(TileMode::LinearAligned);
        }
    }

    const u32 asked = std::max<u32>(s.mip_levels, 1);
    u32 levels = std::max<u32>(1, std::min(asked, full_chain_levels(s)));
    if (levels > max_mip_levels) {
        WIIU_LOG_WARN(Gx2,
                      "a {}x{} surface has {} mip levels; GX2 has room for {}",
                      s.width.value(),
                      s.height.value(),
                      levels,
                      max_mip_levels);
        levels = max_mip_levels;
    }
    s.mip_levels = levels;

    s.mip_level_offset[0] = 0;
    s.swizzle &= 0xFF00FFFF;
    if (is_macro_tiled(s.tile_mode)) {
        s.swizzle |= swizzle_first_untiled_level_unset << 16;
    }

    const u32 align_shift = bad_align && latte::is_block_compressed(s.format) ? 2 : 0;

    u32 last_mode = s.tile_mode;
    u32 previous_size = 0;
    u32 first_mip_offset = 0;

    for (u32 level = 0; level < levels; ++level) {
        std::optional<latte::SurfaceLayout> layout = surface_layout(s, level);
        if (!layout.has_value()) {
            WIIU_LOG_ERROR(Gx2,
                           "GX2CalcSurfaceSizeAndAlignment: level {} of a {}x{}x{} surface, "
                           "format {:#x}, tile mode {}, cannot be laid out; the surface is "
                           "left as it was",
                           level,
                           s.width.value(),
                           s.height.value(),
                           s.depth.value(),
                           s.format.value(),
                           s.tile_mode.value());
            return false;
        }

        if (level == 0) {
            if (chosen) {
                // addrlib may already have stepped a 2D mode down for this
                // size; take what it used.
                if (s.tile_mode != static_cast<u32>(layout->tile_mode)) {
                    s.tile_mode = static_cast<u32>(layout->tile_mode);
                    layout = surface_layout(s, 0);
                    if (!is_macro_tiled(s.tile_mode)) {
                        s.swizzle &= 0xFF00FFFF;
                    }
                    last_mode = s.tile_mode;
                }
                // Smaller than a macro-tile both ways: 1D wastes less.
                if (layout.has_value() && s.width < (layout->pitch_align << align_shift) &&
                    s.height < (layout->height_align << align_shift)) {
                    s.tile_mode =
                        static_cast<u32>(s.tile_mode == static_cast<u32>(TileMode::Tiled2DThick)
                                             ? TileMode::Tiled1DThick
                                             : TileMode::Tiled1DThin1);
                    layout = surface_layout(s, 0);
                    s.swizzle &= 0xFF00FFFF;
                    last_mode = s.tile_mode;
                }
                if (!layout.has_value()) {
                    WIIU_LOG_ERROR(Gx2,
                                   "GX2CalcSurfaceSizeAndAlignment: a {}x{} surface, format "
                                   "{:#x}, cannot be laid out in tile mode {}",
                                   s.width.value(),
                                   s.height.value(),
                                   s.format.value(),
                                   s.tile_mode.value());
                    return false;
                }
            }
            s.image_size = static_cast<u32>(layout->size);
            s.alignment = layout->base_align;
            s.pitch = layout->pitch;
        } else {
            u32 pad = 0;
            if (is_macro_tiled(last_mode) && !is_macro_tiled(static_cast<u32>(layout->tile_mode))) {
                // The first level that is no longer macro-tiled: the swizzle
                // stops applying here.
                s.swizzle = (level << 16) | (s.swizzle & 0xFF00FFFF);
                last_mode = static_cast<u32>(layout->tile_mode);
                if (level > 1) {
                    pad = s.swizzle & 0xFFFF;
                }
            }
            pad += (layout->base_align - (previous_size % layout->base_align)) % layout->base_align;

            if (level == 1) {
                first_mip_offset = pad + previous_size;
            } else {
                s.mip_level_offset[level - 1] = pad + previous_size + s.mip_level_offset[level - 2];
            }
        }

        previous_size = static_cast<u32>(layout->size);
    }

    // Level 1 sits at `mipmaps` itself, so the chain's size is the last
    // level's offset from there plus its size -- measured before entry 0
    // becomes level 1's offset from `image`.
    s.mipmap_size = levels <= 1 ? 0 : previous_size + s.mip_level_offset[levels - 2];
    s.mip_level_offset[0] = first_mip_offset;

    surface = s;
    return true;
}

std::optional<ColorBufferRegisters> color_buffer_registers(const ColorBuffer& buffer) {
    const Surface& surface = buffer.surface;
    const std::optional<latte::SurfaceLayout> layout = surface_layout(surface, buffer.view_mip);
    if (!layout.has_value()) {
        WIIU_LOG_ERROR(Gx2,
                       "GX2InitColorBufferRegs: level {} of a {}x{} colour buffer, format {:#x}, "
                       "cannot be laid out; its registers are left as they were",
                       buffer.view_mip.value(),
                       surface.width.value(),
                       surface.height.value(),
                       surface.format.value());
        return std::nullopt;
    }

    const u32 format = surface.format;
    const u32 color_format = latte::color_format(format);
    if (!latte::is_color_renderable(color_format)) {
        WIIU_LOG_WARN(Gx2, "GX2InitColorBufferRegs: format {:#x} cannot be rendered to", format);
    }

    latte::CbColorSize size{.raw = 0};
    size.pitch_tile_max = (layout->pitch / 8) - 1;
    size.slice_tile_max = ((layout->pitch * layout->height) / 64) - 1;

    std::optional<latte::CbNumberType> type = number_type(format);
    if (!type.has_value()) {
        WIIU_LOG_WARN(Gx2,
                      "GX2InitColorBufferRegs: format {:#x} has no number type this knows; "
                      "treated as unorm",
                      format);
        type = latte::CbNumberType::Unorm;
    }

    const bool integer = *type == latte::CbNumberType::Uint || *type == latte::CbNumberType::Sint;
    const bool normalised = *type == latte::CbNumberType::Unorm ||
                            *type == latte::CbNumberType::Snorm ||
                            *type == latte::CbNumberType::Srgb;
    // The integer formats, and the depth layouts read as colour: nothing the
    // blender can do anything with.
    const bool bypass = integer || format == 0x011 || format == 0x811 || format == 0x81C;

    latte::CbColorInfo info{.raw = 0};
    info.endian = 0; // GX2 swaps nothing on the way out
    info.format = color_format;
    // GX2's linear-special is linear-general to the hardware.
    info.array_mode =
        layout->tile_mode == TileMode::LinearSpecial ? TileMode::LinearGeneral : layout->tile_mode;
    info.number_type = *type;
    info.comp_swap = color_format == 0x0C || color_format == 0x1B ? latte::CbCompSwap::StdRev
                                                                  : latte::CbCompSwap::Std;
    info.tile_mode = surface.aa != 0 ? latte::CbTileMode::FragEnable : latte::CbTileMode::Disable;
    info.blend_clamp = normalised && !bypass;
    info.blend_bypass = bypass;
    info.round_mode = *type == latte::CbNumberType::Float ? latte::CbRoundMode::Truncate
                                                          : latte::CbRoundMode::ByHalf;
    info.source_format =
        integer ? latte::CbSourceFormat::ExportFull : export_source_format(color_format);

    latte::CbColorView view{.raw = 0};
    if (surface.tile_mode != static_cast<u32>(TileMode::LinearSpecial)) {
        view.slice_start = buffer.view_first_slice;
        view.slice_max = buffer.view_first_slice + buffer.view_num_slices - 1;
    }

    if (surface.aa != 0) {
        WIIU_LOG_WARN(Gx2,
                      "GX2InitColorBufferRegs: a multisampled colour buffer; its CMASK and "
                      "FMASK state is not computed yet");
    }

    return ColorBufferRegisters{
        .size = size.raw,
        .info = info.raw,
        .view = view.raw,
        .mask = 0,
    };
}

std::optional<DepthBufferRegisters> depth_buffer_registers(const DepthBuffer& buffer) {
    const Surface& surface = buffer.surface;
    const std::optional<latte::SurfaceLayout> layout = surface_layout(surface, buffer.view_mip);
    if (!layout.has_value()) {
        WIIU_LOG_ERROR(Gx2,
                       "GX2InitDepthBufferRegs: level {} of a {}x{} depth buffer, format {:#x}, "
                       "cannot be laid out; its registers are left as they were",
                       buffer.view_mip.value(),
                       surface.width.value(),
                       surface.height.value(),
                       surface.format.value());
        return std::nullopt;
    }

    latte::DbDepthSize size{.raw = 0};
    size.pitch_tile_max = (layout->pitch / 8) - 1;
    size.slice_tile_max = ((layout->pitch * layout->height) / 64) - 1;

    latte::DbDepthView view{.raw = 0};
    view.slice_start = buffer.view_first_slice;
    view.slice_max = buffer.view_first_slice + buffer.view_num_slices - 1;

    latte::DbHtileSurface htile{.raw = 0};
    htile.htile_width = true;
    htile.htile_height = true;
    htile.full_cache = true;

    latte::DbPrefetchLimit prefetch{.raw = 0};
    prefetch.depth_height_tile_max = (surface.height / 8) - 1;

    latte::DbPreloadControl preload{.raw = 0};
    preload.max_x = (surface.width / 32) & 0xFF;
    preload.max_y = (surface.height / 32) & 0xFF;

    latte::DbDepthInfo info{.raw = 0};
    info.read_size_512 = true;
    info.array_mode =
        layout->tile_mode == TileMode::LinearSpecial ? TileMode::LinearGeneral : layout->tile_mode;
    info.tile_surface_enable = buffer.hi_z != 0;

    // The polygon offset is scaled by the depth format's precision: minus
    // the bits of a unorm depth, and the float formats' own figures.
    latte::PaSuPolyOffsetDbFmtCntl poly{.raw = 0};
    const std::optional<latte::DbFormat> format = latte::depth_format(surface.format);
    info.format = format.value_or(latte::DbFormat::Invalid);
    switch (format.value_or(latte::DbFormat::Invalid)) {
        case latte::DbFormat::Depth16:
            poly.neg_num_db_bits = 240; // -16
            break;
        case latte::DbFormat::Depth8_24:
            poly.neg_num_db_bits = 232; // -24
            break;
        case latte::DbFormat::Depth32Float:
        case latte::DbFormat::DepthX24_8_32Float:
            poly.neg_num_db_bits = 233; // -23
            poly.db_is_float = true;
            break;
        case latte::DbFormat::Depth8_24Float:
            poly.neg_num_db_bits = 236; // -20
            poly.db_is_float = true;
            break;
        default:
            WIIU_LOG_WARN(Gx2,
                          "GX2InitDepthBufferRegs: format {:#x} is not a depth format",
                          surface.format.value());
            break;
    }

    return DepthBufferRegisters{
        .size = size.raw,
        .view = view.raw,
        .info = info.raw,
        .htile_surface = htile.raw,
        .prefetch_limit = prefetch.raw,
        .preload_control = preload.raw,
        .poly_offset_control = poly.raw,
    };
}

namespace {

/// `void GX2CalcSurfaceSizeAndAlignment(GX2Surface *surface)`.
void gx2_calc_surface_size_and_alignment(Call& call, u32 address) {
    Surface surface{};
    if (!read_guest(call.memory, address, surface)) {
        WIIU_LOG_ERROR(Gx2, "GX2CalcSurfaceSizeAndAlignment: no surface at {:#010x}", address);
        return;
    }
    if (calc_surface_size_and_alignment(surface)) {
        static_cast<void>(write_guest(call.memory, address, surface));
    }
}

/// `void GX2InitColorBufferRegs(GX2ColorBuffer *buffer)`.
void gx2_init_color_buffer_regs(Call& call, u32 address) {
    ColorBuffer buffer{};
    if (!read_guest(call.memory, address, buffer)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitColorBufferRegs: no colour buffer at {:#010x}", address);
        return;
    }
    const std::optional<ColorBufferRegisters> registers = color_buffer_registers(buffer);
    if (!registers.has_value()) {
        return;
    }
    buffer.cb_color_size = registers->size;
    buffer.cb_color_info = registers->info;
    buffer.cb_color_view = registers->view;
    buffer.cb_color_mask = registers->mask;
    static_cast<void>(write_guest(call.memory, address, buffer));
}

/// `void GX2ClearColor(GX2ColorBuffer *buffer, float r, float g, float b, float a)`.
void gx2_clear_color(Call& call, u32 address, f32 red, f32 green, f32 blue, f32 alpha) {
    ColorBuffer buffer{};
    if (!read_guest(call.memory, address, buffer)) {
        WIIU_LOG_ERROR(Gx2, "GX2ClearColor: no colour buffer at {:#010x}", address);
        return;
    }
    std::vector<u32> packet;
    packet.reserve(1 + 4 + color_buffer_words);
    packet.push_back(
        latte::pm4::type3_header(latte::pm4::Opcode::HleClearColor, 4 + color_buffer_words));
    for (const f32 component : {red, green, blue, alpha}) {
        packet.push_back(std::bit_cast<u32>(component));
    }
    const std::array<u32, color_buffer_words> words = color_buffer_to_words(buffer);
    packet.insert(packet.end(), words.begin(), words.end());
    static_cast<void>(emit(call, packet));
}

/// `void GX2CopyColorBufferToScanBuffer(GX2ColorBuffer *buffer, GX2ScanTarget target)`.
void gx2_copy_color_buffer_to_scan_buffer(Call& call, u32 address, u32 target) {
    if (!scan_target_from_gx2(target).has_value()) {
        WIIU_LOG_WARN(
            Gx2, "GX2CopyColorBufferToScanBuffer: {} is not a scan target; nothing copied", target);
        return;
    }
    ColorBuffer buffer{};
    if (!read_guest(call.memory, address, buffer)) {
        WIIU_LOG_ERROR(
            Gx2, "GX2CopyColorBufferToScanBuffer: no colour buffer at {:#010x}", address);
        return;
    }
    std::vector<u32> packet;
    packet.reserve(1 + 1 + color_buffer_words);
    packet.push_back(
        latte::pm4::type3_header(latte::pm4::Opcode::HleCopyColorToScan, 1 + color_buffer_words));
    packet.push_back(target);
    const std::array<u32, color_buffer_words> words = color_buffer_to_words(buffer);
    packet.insert(packet.end(), words.begin(), words.end());
    static_cast<void>(emit(call, packet));
}

/// `void GX2InitDepthBufferRegs(GX2DepthBuffer *buffer)`.
void gx2_init_depth_buffer_regs(Call& call, u32 address) {
    DepthBuffer buffer{};
    if (!read_guest(call.memory, address, buffer)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitDepthBufferRegs: no depth buffer at {:#010x}", address);
        return;
    }
    const std::optional<DepthBufferRegisters> registers = depth_buffer_registers(buffer);
    if (!registers.has_value()) {
        return;
    }
    buffer.db_depth_size = registers->size;
    buffer.db_depth_view = registers->view;
    buffer.db_depth_info = registers->info;
    buffer.db_htile_surface = registers->htile_surface;
    buffer.db_prefetch_limit = registers->prefetch_limit;
    buffer.db_preload_control = registers->preload_control;
    buffer.pa_poly_offset_control = registers->poly_offset_control;
    static_cast<void>(write_guest(call.memory, address, buffer));
}

/// `void GX2ClearDepthStencilEx(GX2DepthBuffer *buffer, float depth, uint8_t
/// stencil, GX2ClearFlags flags)`.
///
/// Two halves, as Cemu has them: the clear registers, written only where the
/// flags ask (decaf writes both every time), and the clear itself.
void gx2_clear_depth_stencil_ex(Call& call, u32 address, f32 depth, u32 stencil, u32 flags) {
    DepthBuffer buffer{};
    if (!read_guest(call.memory, address, buffer)) {
        WIIU_LOG_ERROR(Gx2, "GX2ClearDepthStencilEx: no depth buffer at {:#010x}", address);
        return;
    }

    const u32 stencil_value = stencil & 0xFF;
    const u32 depth_value = std::bit_cast<u32>(depth);
    const bool set_stencil = (flags & clear_set_stencil_register) != 0;
    const bool set_depth = (flags & clear_set_depth_register) != 0;
    constexpr u32 stencil_offset = (latte::reg::db_stencil_clear - 0x28000) / 4;
    constexpr u32 depth_offset = (latte::reg::db_depth_clear - 0x28000) / 4;
    using latte::pm4::Opcode;
    using latte::pm4::type3_header;
    if (set_stencil && set_depth) {
        const std::array<u32, 4> packet{
            type3_header(Opcode::SetContextReg, 3), stencil_offset, stencil_value, depth_value};
        static_cast<void>(emit(call, packet));
    } else if (set_stencil || set_depth) {
        const std::array<u32, 3> packet{type3_header(Opcode::SetContextReg, 2),
                                        set_depth ? depth_offset : stencil_offset,
                                        set_depth ? depth_value : stencil_value};
        static_cast<void>(emit(call, packet));
    }

    if ((flags & (clear_depth | clear_stencil)) == 0) {
        return;
    }
    std::vector<u32> packet;
    packet.reserve(1 + 3 + depth_buffer_words);
    packet.push_back(type3_header(Opcode::HleClearDepthStencil, 3 + depth_buffer_words));
    packet.push_back(flags);
    packet.push_back(depth_value);
    packet.push_back(stencil_value);
    const std::array<u32, depth_buffer_words> words = depth_buffer_to_words(buffer);
    packet.insert(packet.end(), words.begin(), words.end());
    static_cast<void>(emit(call, packet));
}

/// What a render target's base register holds: the level's physical
/// address in 256-byte units, with the surface's bank and pipe swizzle in its
/// low bits while the level is still macro-tiled -- decaf's and Cemu's
/// arithmetic. Nothing, logged, for a level with no memory the GPU can see.
[[nodiscard]] std::optional<u32> base_register(const memory::Memory& memory,
                                               const Surface& surface,
                                               u32 level,
                                               std::string_view what) {
    const std::optional<ppcaddr_t> address = level_address(surface, level);
    if (!address.has_value()) {
        WIIU_LOG_ERROR(Gx2, "{}: the surface has no memory for level {}", what, level);
        return std::nullopt;
    }
    std::optional<physaddr_t> physical = physical_address(memory, *address, what);
    if (!physical.has_value()) {
        return std::nullopt;
    }
    if (is_macro_tiled(surface.tile_mode) && level < ((surface.swizzle >> 16) & 0xFF)) {
        *physical ^= surface.swizzle & 0xFFFF;
    }
    return *physical >> 8;
}

/// `void GX2SetColorBuffer(GX2ColorBuffer *buffer, GX2RenderTarget target)`.
///
/// The registers `GX2InitColorBufferRegs` worked out, as the title's
/// structure holds them now, and the base address -- decaf's seven, one
/// packet each. Cemu leaves out CMASK and FMASK and writes the base as a
/// byte address for its own renderer; on hardware it is in 256-byte units,
/// which is what Linux writes too.
void gx2_set_color_buffer(Call& call, u32 address, u32 target) {
    if (target >= latte::reg::render_target_count) {
        WIIU_LOG_WARN(Gx2, "GX2SetColorBuffer: render target {} does not exist", target);
        return;
    }
    ColorBuffer buffer{};
    if (!read_guest(call.memory, address, buffer)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetColorBuffer: no colour buffer at {:#010x}", address);
        return;
    }
    const u32 base =
        base_register(call.memory, buffer.surface, buffer.view_mip, "GX2SetColorBuffer")
            .value_or(0);

    u32 tile = 0;
    u32 frag = 0;
    if (buffer.surface.aa != 0 && buffer.aa_buffer != 0) {
        if (const auto aa = physical_address(call.memory, buffer.aa_buffer, "GX2SetColorBuffer")) {
            frag = *aa >> 8;
            tile = (*aa + buffer.cmask_offset) >> 8;
        }
    }

    const u32 at = target * 4;
    const std::array<std::pair<u32, u32>, 7> registers{{
        {latte::reg::cb_color0_base + at, base},
        {latte::reg::cb_color0_size + at, buffer.cb_color_size},
        {latte::reg::cb_color0_info + at, buffer.cb_color_info},
        {latte::reg::cb_color0_tile + at, tile},
        {latte::reg::cb_color0_frag + at, frag},
        {latte::reg::cb_color0_view + at, buffer.cb_color_view},
        {latte::reg::cb_color0_mask + at, buffer.cb_color_mask},
    }};
    for (const auto& [register_address, value] : registers) {
        const std::array<u32, 1> values{value};
        static_cast<void>(emit_context_registers(call, register_address, values));
    }
}

/// `void GX2SetDepthBuffer(GX2DepthBuffer *buffer)`: decaf's registers and
/// order. Cemu's own notes call its version a hack for its renderer.
void gx2_set_depth_buffer(Call& call, u32 address) {
    DepthBuffer buffer{};
    if (!read_guest(call.memory, address, buffer)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetDepthBuffer: no depth buffer at {:#010x}", address);
        return;
    }
    const u32 base =
        base_register(call.memory, buffer.surface, buffer.view_mip, "GX2SetDepthBuffer")
            .value_or(0);
    u32 hi_z = 0;
    if (buffer.hi_z != 0) {
        if (const auto physical = physical_address(call.memory, buffer.hi_z, "GX2SetDepthBuffer")) {
            hi_z = *physical >> 8;
        }
    }

    const std::array<u32, 2> size_view{buffer.db_depth_size, buffer.db_depth_view};
    const std::array<u32, 3> base_info{base, buffer.db_depth_info, hi_z};
    const std::array<u32, 1> htile{buffer.db_htile_surface};
    const std::array<u32, 1> prefetch{buffer.db_prefetch_limit};
    const std::array<u32, 1> preload{buffer.db_preload_control};
    const std::array<u32, 1> poly{buffer.pa_poly_offset_control};
    const std::array<u32, 2> clear{buffer.stencil_clear,
                                   std::bit_cast<u32>(buffer.depth_clear.value())};
    static_cast<void>(emit_context_registers(call, latte::reg::db_depth_size, size_view));
    static_cast<void>(emit_context_registers(call, latte::reg::db_depth_base, base_info));
    static_cast<void>(emit_context_registers(call, latte::reg::db_htile_surface, htile));
    static_cast<void>(emit_context_registers(call, latte::reg::db_prefetch_limit, prefetch));
    static_cast<void>(emit_context_registers(call, latte::reg::db_preload_control, preload));
    static_cast<void>(
        emit_context_registers(call, latte::reg::pa_su_poly_offset_db_fmt_cntl, poly));
    static_cast<void>(emit_context_registers(call, latte::reg::db_stencil_clear, clear));
}

} // namespace

std::optional<latte::ColorBufferKey> color_buffer_key(const memory::Memory& memory,
                                                      const ColorBuffer& buffer) {
    const std::optional<u32> base =
        base_register(memory, buffer.surface, buffer.view_mip, "a colour buffer");
    if (!base.has_value()) {
        return std::nullopt;
    }
    return latte::ColorBufferKey::from(
        *base, buffer.cb_color_size, buffer.cb_color_info, buffer.cb_color_view);
}

bool clear_color_surface(latte::SurfaceCache& cache,
                         gpu::GpuBackend& backend,
                         const memory::Memory& memory,
                         const ColorBuffer& buffer,
                         const std::array<f32, 4>& rgba,
                         std::string& problem) {
    // The cache holds one single-sampled slice a surface so far. A clear of
    // anything else is done on guest memory, as before ADR-006.
    if (buffer.surface.aa != 0) {
        problem = "GX2ClearColor: a multisampled colour buffer is cleared on the CPU for now";
        return false;
    }
    if (buffer.view_num_slices != 1) {
        problem = fmt::format("GX2ClearColor: a clear of {} slices is done on the CPU for now",
                              buffer.view_num_slices.value());
        return false;
    }
    const std::optional<latte::ColorBufferKey> key = color_buffer_key(memory, buffer);
    if (!key.has_value()) {
        problem = "GX2ClearColor: the colour buffer has no memory the GPU can see";
        return false;
    }
    std::string error;
    const std::optional<latte::ColorSurface> surface = cache.color_buffer(*key, error);
    if (!surface.has_value()) {
        problem = fmt::format("GX2ClearColor: {}; cleared on the CPU instead", error);
        return false;
    }

    // NaN clears to zero, as `encode_color` has it; out of range is clamped
    // by the conversion to the format, on both sides.
    const auto component = [](f32 value) { return std::isnan(value) ? 0.0F : value; };
    gpu::RenderPassDesc pass;
    pass.label = "GX2ClearColor";
    pass.color_count = 1;
    pass.colors[0].texture = surface->texture;
    pass.colors[0].load = gpu::LoadAction::Clear;
    pass.colors[0].store = gpu::StoreAction::Store;
    pass.colors[0].clear = gpu::ClearColor{
        component(rgba[0]), component(rgba[1]), component(rgba[2]), component(rgba[3])};
    if (!backend.begin_render_pass(pass)) {
        problem = "GX2ClearColor: the backend refused the clear's render pass";
        return false;
    }
    backend.end_render_pass();
    return true;
}

std::optional<latte::DepthBufferKey> depth_buffer_key(const memory::Memory& memory,
                                                      const DepthBuffer& buffer) {
    const std::optional<u32> base =
        base_register(memory, buffer.surface, buffer.view_mip, "a depth buffer");
    if (!base.has_value()) {
        return std::nullopt;
    }
    return latte::DepthBufferKey::from(
        *base, buffer.db_depth_size, buffer.db_depth_info, buffer.db_depth_view);
}

bool clear_depth_stencil_surface(latte::SurfaceCache& cache,
                                 gpu::GpuBackend& backend,
                                 const memory::Memory& memory,
                                 const DepthBuffer& buffer,
                                 f32 depth,
                                 u8 stencil,
                                 u32 flags,
                                 std::string& problem) {
    const bool depth_cleared = (flags & clear_depth) != 0;
    const bool stencil_cleared = (flags & clear_stencil) != 0;
    if (!depth_cleared && !stencil_cleared) {
        return true; // nothing asked, here or on guest memory
    }
    if (buffer.surface.aa != 0) {
        problem =
            "GX2ClearDepthStencilEx: a multisampled depth buffer is cleared on the CPU for now";
        return false;
    }
    if (buffer.view_num_slices != 1) {
        problem =
            fmt::format("GX2ClearDepthStencilEx: a clear of {} slices is done on the CPU for now",
                        buffer.view_num_slices.value());
        return false;
    }
    const std::optional<latte::DepthBufferKey> key = depth_buffer_key(memory, buffer);
    if (!key.has_value()) {
        problem = "GX2ClearDepthStencilEx: the depth buffer has no memory the GPU can see";
        return false;
    }
    // A surface made now holds nothing worth loading.
    const bool fresh = !cache.find(*key).has_value();
    std::string error;
    const std::optional<latte::DepthSurface> surface = cache.depth_buffer(*key, error);
    if (!surface.has_value()) {
        problem = fmt::format("GX2ClearDepthStencilEx: {}; cleared on the CPU instead", error);
        return false;
    }

    gpu::RenderPassDesc pass;
    pass.label = "GX2ClearDepthStencilEx";
    pass.color_count = 0;
    gpu::DepthStencilAttachment& attachment = pass.depth_stencil;
    attachment.texture = surface->texture;
    attachment.load = depth_cleared || fresh ? gpu::LoadAction::Clear : gpu::LoadAction::Load;
    attachment.store = gpu::StoreAction::Store;
    if (surface->has_stencil) {
        attachment.stencil_load =
            stencil_cleared || fresh ? gpu::LoadAction::Clear : gpu::LoadAction::Load;
        attachment.stencil_store = gpu::StoreAction::Store;
    }
    // Metal's depth is [0, 1], as a unorm depth's is; NaN clears to zero, as
    // the CPU clear has it.
    attachment.clear_depth = std::isnan(depth) ? 0.0F : std::clamp(depth, 0.0F, 1.0F);
    attachment.clear_stencil = stencil;
    if (!backend.begin_render_pass(pass)) {
        problem = "GX2ClearDepthStencilEx: the backend refused the clear's render pass";
        return false;
    }
    backend.end_render_pass();
    return true;
}

std::optional<latte::ColorSurface> held_color_surface(const latte::SurfaceCache& cache,
                                                      const memory::Memory& memory,
                                                      const ColorBuffer& buffer) {
    const std::optional<latte::ColorBufferKey> key = color_buffer_key(memory, buffer);
    if (!key.has_value()) {
        return std::nullopt;
    }
    return cache.find(*key);
}

bool copy_surface_to_scan(latte::SurfaceCopier& copier,
                          const latte::ColorSurface& surface,
                          const ColorBuffer& buffer,
                          const ScanBuffer& scan,
                          const latte::CopyTarget& target,
                          std::string& problem) {
    // The CPU copy's rule: one colour format, one element size. Every surface
    // the GPU holds is 8_8_8_8, four bytes an element.
    const u32 format = buffer.surface.format;
    if (latte::color_format(format) != latte::color_format(scan.surface_format) ||
        latte::bytes_per_element(format) != latte::bytes_per_element(scan.surface_format)) {
        problem = fmt::format("GX2CopyColorBufferToScanBuffer: copying format {:#x} into a scan "
                              "buffer of format {:#x} is not supported yet",
                              format,
                              scan.surface_format);
        return false;
    }

    // The picture is the viewed level's own size; the texture, the padded
    // size its registers describe, is at least as large.
    const latte::Extent extent = latte::level_extent(surface_desc(buffer.surface, buffer.view_mip));
    const latte::CopySource source{.texture = surface.texture,
                                   .format = surface.format,
                                   .width = std::min(extent.width, surface.width),
                                   .height = std::min(extent.height, surface.height)};
    std::string error;
    if (!copier.copy(source, target, error)) {
        problem = "GX2CopyColorBufferToScanBuffer: " + error;
        return false;
    }
    return true;
}

void register_gx2_surface(Registry& registry) {
    registry.add("gx2", "GX2CalcSurfaceSizeAndAlignment", gx2_calc_surface_size_and_alignment);
    registry.add("gx2", "GX2InitColorBufferRegs", gx2_init_color_buffer_regs);
    registry.add("gx2", "GX2ClearColor", gx2_clear_color);
    registry.add("gx2", "GX2CopyColorBufferToScanBuffer", gx2_copy_color_buffer_to_scan_buffer);
    registry.add("gx2", "GX2InitDepthBufferRegs", gx2_init_depth_buffer_regs);
    registry.add("gx2", "GX2ClearDepthStencilEx", gx2_clear_depth_stencil_ex);
    registry.add("gx2", "GX2SetColorBuffer", gx2_set_color_buffer);
    registry.add("gx2", "GX2SetDepthBuffer", gx2_set_depth_buffer);
}

} // namespace wiiu::hle::gx2
