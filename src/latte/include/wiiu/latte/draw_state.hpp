// Phase 8.11 — the fixed state a draw needs from the registers, as Metal
// takes it: the primitive, the viewport and the scissor.
//
// **The viewport.** Latte keeps a scale and an offset per axis
// (PA_CL_VPORT_*), what GX2SetViewport wrote (8.1): x scale half the width
// and offset its centre; y scale *minus* half the height, offset its centre
// -- so clip space's +y is the top of the viewport, as in Metal. Metal wants
// the rectangle back: x is the offset less the scale and the width twice the
// scale; y is the offset plus the (negative) scale and the height minus twice
// it. Cemu's Metal renderer reads them the same way. Depth: with DX clip
// space (PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF, GX2's "half Z") the near plane is
// the z offset and the far is offset plus scale; without it, z runs from -1,
// so near is offset less scale (Cemu's again). Clip space's own z, which
// Metal clips to [0, 1], is the emitter's to bring over (msl.hpp, not done).
//
// **The scissor** is the generic scissor (PA_SC_GENERIC_SCISSOR_*), which
// GX2SetScissor writes, cut to the render target: Metal refuses a scissor
// that leaves it. One that leaves nothing draws nothing.
//
// **Primitives** map where Metal has them. Fans, loops, polygons, quad strips
// and the adjacency types need their indices rewritten, which is not done
// yet; quads and RECTS map to the backend's own names for them, which only
// the mesh path (8.8) will draw.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/types.hpp>
#include <wiiu/latte/registers.hpp>

#include <optional>

namespace wiiu::latte {

/// The backend's primitive for VGT_PRIMITIVE_TYPE's, or nothing for one
/// with no counterpart yet.
[[nodiscard]] std::optional<gpu::PrimitiveType> primitive_type(u32 latte) noexcept;

/// Viewport 0, as Metal takes it.
[[nodiscard]] gpu::Viewport viewport(const RegisterFile& registers) noexcept;

/// The generic scissor, cut to a `width` x `height` render target; nothing
/// if that leaves nothing.
[[nodiscard]] std::optional<gpu::Rect>
scissor(const RegisterFile& registers, u32 width, u32 height) noexcept;

} // namespace wiiu::latte
