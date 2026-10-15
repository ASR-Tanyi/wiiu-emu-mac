// Latte's viewport and scissor registers: where clip space lands on the
// render target, the guard band around it, the depth range, and the
// rectangle outside which nothing is drawn.
//
// Addresses are R600's, as Linux's register list for the family
// (`reg_srcs/r600`) gives them; decaf and, where it has them, Cemu agree.
// GX2 sets viewport 0 and the generic scissor; the other fifteen viewports
// are for geometry shaders that pick one, which GX2 does not offer.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>

namespace wiiu::latte {

namespace reg {
/// Viewport 0's transform: x, y and z scale and offset, as floats, in
/// order (XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET).
inline constexpr u32 pa_cl_vport_xscale_0 = 0x2843C;

/// The guard band, as floats, in order: vertical clip and discard, then
/// horizontal clip and discard adjustments.
inline constexpr u32 pa_cl_gb_vert_clip_adj = 0x28C0C;

/// Viewport 0's depth range, as floats: the least and greatest depth kept.
inline constexpr u32 pa_sc_vport_zmin_0 = 0x282D0;
inline constexpr u32 pa_sc_vport_zmax_0 = 0x282D4;

/// The generic scissor: top left, then bottom right (exclusive).
inline constexpr u32 pa_sc_generic_scissor_tl = 0x28240;
inline constexpr u32 pa_sc_generic_scissor_br = 0x28244;
} // namespace reg

/// PA_SC_GENERIC_SCISSOR_TL. Fourteen bits a coordinate, as R600 has them
/// (decaf's width; Cemu gives fifteen, which only differs past 16383).
union PaScGenericScissorTl {
    u32 raw;
    BitField<0, 14, u32> x;
    BitField<16, 14, u32> y;
    BitField<31, 1, bool> window_offset_disable;
};

/// PA_SC_GENERIC_SCISSOR_BR.
union PaScGenericScissorBr {
    u32 raw;
    BitField<0, 14, u32> x;
    BitField<16, 14, u32> y;
};

} // namespace wiiu::latte
