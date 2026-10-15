// Phase 8.1 — GX2's state-setting calls: the registers a draw reads.
//
// Each of these is the same shape: work out a few register values -- once,
// into a structure the title keeps (`GX2Init*Reg`), or on the spot -- and
// write them with SET_CONTEXT_REG (`GX2Set*`). Nothing happens until the
// packet runs; then the register file holds them, and the translator (8.11)
// reads them at the draw.
//
// So far: the viewport and the scissor, which every title sets for its
// render targets before drawing anything.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>

#include <array>
#include <cstddef>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// `GX2ViewportReg`: viewport 0's transform, the guard band and the depth
/// range, as the registers take them -- decaf's layout, which wut's size
/// agrees with.
struct ViewportReg {
    std::array<f32be, 6> transform;  ///< x scale, x offset, y scale, y offset, z scale, z offset
    std::array<f32be, 4> guard_band; ///< vertical clip, vertical discard, horizontal clip, discard
    f32be z_min;
    f32be z_max;
};

static_assert(sizeof(ViewportReg) == 0x30);
static_assert(offsetof(ViewportReg, guard_band) == 0x18);
static_assert(offsetof(ViewportReg, z_min) == 0x28);

/// `GX2ScissorReg`.
struct ScissorReg {
    u32be top_left;
    u32be bottom_right;
};

static_assert(sizeof(ScissorReg) == 0x8);

/// What `GX2InitViewportReg` computes, which decaf and Cemu agree on for the
/// transform: x and y scale by half the size and offset to the centre, with
/// y flipped; depth maps [-1, 1] onto [near, far].
///
/// The rest is decaf's alone -- Cemu leaves it as a to-do. The guard band:
/// how far past the viewport, as a multiple of its size, the clipper may let
/// a primitive run before clipping it, measured to Latte's 8192-pixel edge;
/// 1 for an empty viewport. Discard adjustments of 1. And the depth range as
/// the least and greatest of near and far.
[[nodiscard]] ViewportReg
viewport_registers(f32 x, f32 y, f32 width, f32 height, f32 near_z, f32 far_z) noexcept;

/// What `GX2InitScissorReg` computes: the rectangle's corners, the far one
/// exclusive, each clamped to 8192, with the window offset turned off --
/// Cemu's reading. decaf neither clamps nor sets the bit.
[[nodiscard]] ScissorReg scissor_registers(u32 x, u32 y, u32 width, u32 height) noexcept;

/// Register the viewport and scissor calls. Called by `register_gx2`.
void register_gx2_state(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
