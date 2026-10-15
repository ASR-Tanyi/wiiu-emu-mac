// Latte's surface formats: what GX2's `GX2SurfaceFormat` values are made of.
//
// A GX2 surface format is a Latte colour format in its low six bits -- the
// R600 `COLOR_*`/`FMT_*` number -- with the number type above it: 0x000
// unorm, 0x100 uint, 0x200 snorm, 0x300 sint, 0x400 sRGB, 0x800 float. So
// `0x01A` is 8_8_8_8 unorm and `0x41A` the same bits read as sRGB. Only the
// low six bits decide how big an element is; how a surface is laid out is
// surface.hpp's (8.5). What a colour looks like as an element is
// `encode_color`, for the few formats a clear can write so far; how a texture
// is read is still to come.
#pragma once

#include <wiiu/common/types.hpp>

#include <array>
#include <optional>

namespace wiiu::latte {

/// The Latte colour format inside a GX2 surface format.
[[nodiscard]] constexpr u32 color_format(u32 surface_format) noexcept {
    return surface_format & 0x3F;
}

/// Bytes per element of a GX2 surface format, or zero for one that is not
/// an uncompressed format this knows -- block-compressed formats measure in
/// 4x4 blocks, which `bits_per_element` covers.
///
/// The sizes follow from the R600 format names: 8 and 4_4 are a byte; 16,
/// 8_8 and the five- and six-bit packings are two; everything packed into 32
/// bits is four; 32_32, 16_16_16_16 and X24_8_32 are eight; 32_32_32_32 is
/// sixteen. Bits 6 and 7 are not part of either field; NV12 (0x081) sets
/// one, and is two planes rather than one size of element.
[[nodiscard]] constexpr u32 bytes_per_element(u32 surface_format) noexcept {
    if ((surface_format & 0xC0) != 0) {
        return 0;
    }
    const u32 format = color_format(surface_format);
    if (format >= 0x01 && format <= 0x03) {
        return 1;
    }
    if (format >= 0x05 && format <= 0x0C) {
        return 2;
    }
    if (format >= 0x0D && format <= 0x1B) {
        return 4;
    }
    if (format >= 0x1C && format <= 0x20) {
        return 8;
    }
    if (format == 0x22 || format == 0x23) {
        return 16;
    }
    return 0;
}

/// BC1 to BC5: 0x31 to 0x35, each element a 4x4 block of pixels.
[[nodiscard]] constexpr bool is_block_compressed(u32 surface_format) noexcept {
    const u32 format = color_format(surface_format);
    return (surface_format & 0xC0) == 0 && format >= 0x31 && format <= 0x35;
}

/// Bits per element, counting a compressed format's 4x4 block as one
/// element: BC1 and BC4 are 64 bits a block, BC2, BC3 and BC5 are 128. What
/// AMD's addrlib takes as `bpp`. Zero for a format this does not know.
[[nodiscard]] constexpr u32 bits_per_element(u32 surface_format) noexcept {
    if (is_block_compressed(surface_format)) {
        const u32 format = color_format(surface_format);
        return format == 0x31 || format == 0x34 ? 64 : 128;
    }
    return bytes_per_element(surface_format) * 8;
}

/// A colour as one element of a format: the bytes the GPU stores, in memory
/// order.
struct EncodedColor {
    std::array<u8, 16> bytes{};
    u32 size = 0;
};

/// What a clear to (red, green, blue, alpha) stores in a surface of this
/// format, or nothing for a format this does not encode yet.
///
/// So far: the 8-bit unorm formats -- 8, 8_8, 8_8_8_8 -- and 8_8_8_8 sRGB,
/// which between them are every colour buffer and scan buffer a WHB title
/// makes. Components go into memory red first, a byte each: an 8-bit-per-
/// component format on a little-endian GPU, and the same order OSScreen's
/// buffers hold (6.8). Unorm clamps to [0, 1] and rounds to nearest; NaN is
/// zero. sRGB encodes red, green and blue with the standard curve, as a draw
/// into an sRGB target does, and leaves alpha linear. Wider and float formats
/// wait until their byte order has been checked against something real.
[[nodiscard]] std::optional<EncodedColor> encode_color(u32 surface_format,
                                                       const std::array<f32, 4>& rgba) noexcept;

/// A depth and stencil clear as one element of a depth format: the bytes,
/// and which of them to write -- a clear of depth alone leaves a packed
/// stencil as it was, and the other way round.
struct EncodedDepthStencil {
    std::array<u8, 4> bytes{};
    std::array<bool, 4> write{};
    u32 size = 0;
};

/// What a depth/stencil clear stores in a depth buffer of this format, or
/// nothing for a format this does not encode yet.
///
/// Elements are little-endian words, which is how the GPU stores them and how
/// Cemu reads depth back as a texture: R16 is a 16-bit unorm; R24_X8 packs a
/// 24-bit unorm depth under an 8-bit stencil; R32 is the float itself. The
/// float D24_S8 and the 64-bit X8_X24 wait until something needs them. Unorm
/// depth clamps to [0, 1] and rounds to nearest; float depth is written as
/// given.
[[nodiscard]] std::optional<EncodedDepthStencil> encode_depth_stencil(
    u32 surface_format, f32 depth, u8 stencil, bool write_depth, bool write_stencil) noexcept;

} // namespace wiiu::latte
