// Latte's colour-buffer registers: CB_COLORn_SIZE, CB_COLORn_INFO and
// CB_COLORn_VIEW.
//
// A render target is described to the GPU by a few context registers per
// target: where it is, how big it is in tiles, its format and tiling, and
// which slices are drawn to. GX2 works their values out once, in
// `GX2InitColorBufferRegs`, keeps them in the title's `GX2ColorBuffer`, and
// writes them with SET_CONTEXT_REG whenever that buffer is bound. So the two
// halves of the emulator meet here: GX2 (8.1) encodes these words and the
// translator (8.11) will decode them, through the same fields.
//
// Field positions and enum values are the R600 family's; decaf's
// `latte_registers_cb.h` and Cemu's `GX2InitColorBufferRegs` agree on all of
// them.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/latte/surface.hpp>

namespace wiiu::latte {

/// How the colour buffer's values are interpreted. GX2 takes it from the
/// number type in a surface format's high bits (formats.hpp).
enum class CbNumberType : u32 {
    Unorm = 0,
    Snorm = 1,
    Uscaled = 2,
    Sscaled = 3,
    Uint = 4,
    Sint = 5,
    Srgb = 6,
    Float = 7,
};

/// Which shader output component goes to which channel of the format.
enum class CbCompSwap : u32 {
    Std = 0,
    Alt = 1,
    StdRev = 2,
    AltRev = 3,
};

/// The fast-clear and multisample-fragment modes. GX2 sets fragments on for a
/// multisampled buffer and leaves everything else off.
enum class CbTileMode : u32 {
    Disable = 0,
    ClearEnable = 1,
    FragEnable = 2,
};

enum class CbRoundMode : u32 {
    ByHalf = 0,
    Truncate = 1,
};

/// Whether the shader's export is full precision, or fits the 16-bit
/// normalised path.
enum class CbSourceFormat : u32 {
    ExportFull = 0,
    ExportNorm = 1,
};

/// CB_COLORn_SIZE: the level's pitch and slice, in 8x8 micro-tiles, minus one.
union CbColorSize {
    u32 raw;
    BitField<0, 10, u32> pitch_tile_max;
    BitField<10, 20, u32> slice_tile_max;
};

/// CB_COLORn_INFO: everything about the target but its size and address.
union CbColorInfo {
    u32 raw;
    BitField<0, 2, u32> endian;
    BitField<2, 6, u32> format; ///< the colour format: a surface format's low six bits
    BitField<8, 4, TileMode> array_mode;
    BitField<12, 3, CbNumberType> number_type;
    BitField<15, 1, bool> read_size;
    BitField<16, 2, CbCompSwap> comp_swap;
    BitField<18, 2, CbTileMode> tile_mode;
    BitField<20, 1, bool> blend_clamp;
    BitField<21, 1, bool> clear_color;
    BitField<22, 1, bool> blend_bypass;
    BitField<23, 1, bool> blend_float32;
    BitField<24, 1, bool> simple_float;
    BitField<25, 1, CbRoundMode> round_mode;
    BitField<26, 1, bool> tile_compact;
    BitField<27, 1, CbSourceFormat> source_format;
};

/// CB_COLORn_VIEW: the slices drawn to, first and last.
union CbColorView {
    u32 raw;
    BitField<0, 11, u32> slice_start;
    BitField<13, 11, u32> slice_max;
};

/// Whether a colour format can be a render target at all: the CB_FORMAT
/// values the colour block has, which are the data formats' numbers for the
/// same layouts -- everything from 8 to 32_32_32_32_FLOAT but the gaps at 4
/// and 0x21. Compressed and the three-component 32-bit formats are textures
/// only.
[[nodiscard]] constexpr bool is_color_renderable(u32 color_format) noexcept {
    return (color_format >= 0x01 && color_format <= 0x03) ||
           (color_format >= 0x05 && color_format <= 0x20) || color_format == 0x22 ||
           color_format == 0x23;
}

} // namespace wiiu::latte
