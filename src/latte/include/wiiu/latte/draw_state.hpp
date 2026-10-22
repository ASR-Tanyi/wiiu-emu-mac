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
// Metal clips to [0, 1], is the emitter's to bring over (msl.hpp).
//
// **The scissor** is the generic scissor (PA_SC_GENERIC_SCISSOR_*), which
// GX2SetScissor writes, cut to the render target: Metal refuses a scissor
// that leaves it. One that leaves nothing draws nothing.
//
// **Blending** is target n's CB_BLENDn_CONTROL, when CB_COLOR_CONTROL's
// blend-enable bit n is set -- each target's own register, whatever
// PER_MRT_BLEND says, as GX2 writes them and Cemu reads them. The factors and
// combine functions are decaf's and Cemu's (which agree) mapped to Metal's
// as Cemu's Metal renderer maps them, but for D3D9's two "both" factors,
// which Cemu leaves as zero: BOTH_SRC_ALPHA as the source factor means
// source alpha for the source and one less it for the destination, and
// BOTH_INV_SRC_ALPHA the other way about -- D3D9's definition. Without
// SEPARATE_ALPHA_BLEND alpha blends as colour does. The channels written are
// target n's nibble of CB_TARGET_MASK, none when CB_COLOR_CONTROL's special
// op turns colour writes off. Refused: a ROP3 other than a plain copy (Metal
// pipelines have no logic ops), the other special ops, and numbers Latte
// has no factor or function for.
//
// **Culling** is PA_SU_SC_MODE_CNTL's: the culled faces and the winding of
// the front, handed to Metal as they are (Cemu's choice; the Metal tests
// report which way Metal judges winding). Both faces culled culls every
// polygon, which Metal's cull mode cannot say, so a draw of triangles is not
// encoded at all. Polygon mode -- drawing triangles as points or lines -- is
// refused for now.
//
// **Depth and stencil** are DB_DEPTH_CONTROL's, with the stencil's masks
// and references in DB_STENCILREFMASK and _BF, read as Cemu's Metal renderer
// reads them: depth disabled tests nothing and writes nothing; the stencil,
// when enabled, is the front face's fields for both faces unless
// BACKFACE_ENABLE gives the back its own (and its own masks and reference).
// Latte's comparisons and stencil operations are Metal's, in Metal's order.
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
#include <string>

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

/// How render target `target` is written. Nothing, with `error` saying
/// what, for state not expressed yet.
[[nodiscard]] std::optional<gpu::BlendState>
blend_state(const RegisterFile& registers, u32 target, std::string& error);

/// CB_BLEND_RED to _ALPHA: the colour the constant factors read.
[[nodiscard]] gpu::ClearColor blend_color(const RegisterFile& registers) noexcept;

/// The rasterizer's face state.
struct RasterState {
    gpu::CullMode cull = gpu::CullMode::None;
    gpu::Winding front = gpu::Winding::CounterClockwise;
    bool cull_all = false; ///< both faces: no polygon is drawn
};

/// PA_SU_SC_MODE_CNTL's culling. Nothing, with `error`, for polygon mode.
[[nodiscard]] std::optional<RasterState> raster_state(const RegisterFile& registers,
                                                      std::string& error);

/// The depth and stencil tests, and the references the stencil compares
/// with -- which Metal sets in a pass rather than in the state.
struct DepthStencilState {
    gpu::DepthStencilDesc desc;
    u8 front_reference = 0;
    u8 back_reference = 0;

    /// Whether a draw with this state reads or writes a depth buffer at all.
    [[nodiscard]] bool uses_depth_buffer() const noexcept {
        return desc.depth_compare != gpu::CompareFunction::Always || desc.depth_write ||
               desc.stencil;
    }
};

/// DB_DEPTH_CONTROL and DB_STENCILREFMASK(_BF), as Metal takes them.
[[nodiscard]] DepthStencilState depth_stencil_state(const RegisterFile& registers) noexcept;

} // namespace wiiu::latte
