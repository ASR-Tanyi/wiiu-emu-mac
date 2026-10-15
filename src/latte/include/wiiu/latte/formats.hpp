// Latte's surface formats: what GX2's `GX2SurfaceFormat` values are made of.
//
// A GX2 surface format is a Latte colour format in its low six bits -- the
// R600 `COLOR_*`/`FMT_*` number -- with the number type above it: 0x000
// unorm, 0x100 uint, 0x200 snorm, 0x300 sint, 0x400 sRGB, 0x800 float. So
// `0x01A` is 8_8_8_8 unorm and `0x41A` the same bits read as sRGB. Only the
// low six bits decide how big an element is, which is all this file knows so
// far; the rest of the format story -- tiling, compression, how a texture is
// read -- is 8.13's.
#pragma once

#include <wiiu/common/types.hpp>

namespace wiiu::latte {

/// The Latte colour format inside a GX2 surface format.
[[nodiscard]] constexpr u32 color_format(u32 surface_format) noexcept {
    return surface_format & 0x3F;
}

/// Bytes per element of a GX2 surface format, or zero for one that is not
/// an uncompressed format this knows -- block-compressed formats measure in
/// 4x4 blocks, not elements, and are left to 8.13.
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

} // namespace wiiu::latte
