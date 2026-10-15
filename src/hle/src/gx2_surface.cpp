// Phase 8.1 — GX2's surface calls. See gx2_surface.hpp for the model.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/color_buffer.hpp>
#include <wiiu/latte/formats.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <span>

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

template <typename T>
[[nodiscard]] bool read_guest(const memory::Memory& memory, ppcaddr_t address, T& out) {
    std::array<u8, sizeof(T)> bytes{};
    if (!memory.read_bytes(address, bytes)) {
        return false;
    }
    std::memcpy(&out, bytes.data(), sizeof(T));
    return true;
}

template <typename T>
[[nodiscard]] bool write_guest(memory::Memory& memory, ppcaddr_t address, const T& value) {
    std::array<u8, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    return memory.write_bytes(address, bytes);
}

} // namespace

std::optional<latte::SurfaceLayout> surface_layout(const Surface& surface, u32 level) {
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

    return latte::compute_layout(desc);
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

} // namespace

void register_gx2_surface(Registry& registry) {
    registry.add("gx2", "GX2CalcSurfaceSizeAndAlignment", gx2_calc_surface_size_and_alignment);
    registry.add("gx2", "GX2InitColorBufferRegs", gx2_init_color_buffer_regs);
}

} // namespace wiiu::hle::gx2
