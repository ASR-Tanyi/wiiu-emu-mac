// Phase 8.5 — Latte's surface layout: how big a surface is, how it must be
// aligned, and where each element of it lives.
//
// Latte keeps almost every surface tiled. Elements are grouped into 8x8
// micro-tiles, and the 2D modes group micro-tiles again into macro-tiles
// spread across the memory's banks and pipes, with a per-surface swizzle on
// top. Get any of it wrong and a texture comes out as diagonal noise, or a
// colour buffer the title allocated is a few kilobytes short of what the GPU
// writes.
//
// The arithmetic is AMD's addrlib (ADR-005), which is also what GX2 itself is
// built on. This file is the narrow, typed door to it: one place that knows
// the Wii U's memory configuration, and two questions -- how is a surface laid
// out, and where in it is an element. Everything above (GX2's surface calls,
// the scan-out copy, later the texture cache) asks here.
#pragma once

#include <wiiu/common/types.hpp>

#include <optional>
#include <string_view>

namespace wiiu::latte {

/// Latte's array modes, which GX2 calls tile modes and numbers the same.
enum class TileMode : u32 {
    LinearGeneral = 0,
    LinearAligned = 1,
    Tiled1DThin1 = 2,
    Tiled1DThick = 3,
    Tiled2DThin1 = 4,
    Tiled2DThin2 = 5,
    Tiled2DThin4 = 6,
    Tiled2DThick = 7,
    Tiled2BThin1 = 8,
    Tiled2BThin2 = 9,
    Tiled2BThin4 = 10,
    Tiled2BThick = 11,
    Tiled3DThin1 = 12,
    Tiled3DThick = 13,
    Tiled3BThin1 = 14,
    Tiled3BThick = 15,

    /// GX2's own: linear with no alignment at all. addrlib has no such mode;
    /// GX2 sizes it by hand, and so does this.
    LinearSpecial = 16,
};

[[nodiscard]] std::string_view tile_mode_name(TileMode mode) noexcept;

/// What a surface is, as far as its layout goes. Sizes are in pixels; a
/// compressed format's are rounded to its 4x4 blocks inside.
struct SurfaceDesc {
    TileMode tile_mode = TileMode::LinearAligned;
    u32 format = 0; ///< GX2SurfaceFormat
    u32 width = 1;
    u32 height = 1;
    u32 slices = 1; ///< array slices, cube faces, or depth for a volume
    u32 samples = 1;
    u32 mip_level = 0;

    bool depth = false;   ///< a depth buffer: its own micro-tile order
    bool display = false; ///< a scan buffer
    bool cube = false;
    bool volume = false;
};

/// How a surface is laid out, for one mip level.
struct SurfaceLayout {
    /// The mode actually used: addrlib steps a 2D mode down to 1D for a level
    /// too small to fill a macro-tile, and a thick mode down to thin.
    TileMode tile_mode = TileMode::LinearAligned;

    u32 bits_per_element = 0;
    u32 pitch = 0;  ///< elements per row, padded
    u32 height = 0; ///< rows, padded
    u32 depth = 0;  ///< slices, padded
    u64 size = 0;   ///< bytes, every slice and sample

    u32 base_align = 0; ///< bytes the start must be aligned to
    u32 pitch_align = 0;
    u32 height_align = 0;
    u32 depth_align = 0;
    u32 slice_size = 0;
};

/// Lay a surface out. Nothing for a format addrlib cannot size, or a
/// description it refuses.
[[nodiscard]] std::optional<SurfaceLayout> compute_layout(const SurfaceDesc& surface);

/// Where element (x, y) of slice `slice` and sample `sample` is, in bytes
/// from the start of the level, for a surface laid out as `layout` with
/// GX2's pipe and bank swizzle. Coordinates are in elements -- blocks, for a
/// compressed format.
[[nodiscard]] u64 element_offset(const SurfaceDesc& surface,
                                 const SurfaceLayout& layout,
                                 u32 x,
                                 u32 y,
                                 u32 slice = 0,
                                 u32 sample = 0,
                                 u32 pipe_swizzle = 0,
                                 u32 bank_swizzle = 0);

/// The pipe and bank swizzle a GX2 surface's `swizzle` word carries in its
/// second byte -- a starting offset in units of 256 bytes -- split the way
/// Latte's two pipes and four banks divide memory.
struct BankPipeSwizzle {
    u32 pipe = 0;
    u32 bank = 0;
};

[[nodiscard]] BankPipeSwizzle split_swizzle(u32 gx2_swizzle) noexcept;

} // namespace wiiu::latte
