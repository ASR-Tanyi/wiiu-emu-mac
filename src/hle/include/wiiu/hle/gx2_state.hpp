// Phase 8.1 — GX2's state-setting calls: the registers a draw reads.
//
// Each of these is the same shape: work out a few register values -- once,
// into a structure the title keeps (`GX2Init*Reg`), or on the spot -- and
// write them with SET_CONTEXT_REG (`GX2Set*`). Nothing happens until the
// packet runs; then the register file holds them, and the translator (8.11)
// reads them at the draw.
//
// So far: the viewport and the scissor, which every title sets for its
// render targets before drawing anything; the depth, stencil and rasterizer
// state -- the depth test, the stencil test and its masks, culling and
// polygon mode, polygon offset, point and line sizes, clipping, and the
// primitive restart index; and the colour state -- colour control, blending
// and its constant, the channel masks, the alpha test, alpha-to-mask and the
// multisample mask; and stream out on or off and the tessellation levels'
// bounds. `GX2SetTessellation` itself is a stub, as it is in decaf. Each comes as GX2 has it: the
// call that sets the state, and for most an `Init…Reg` that computes it into the title's structure
// and a `Set…Reg` that writes that. The `Get…Reg` calls that read a structure back are not here
// yet.
//
// Where decaf and Cemu compute a value differently it is named, with the
// one taken; the field layouts are latte/render_state.hpp's.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>

#include <array>
#include <cstddef>
#include <vector>

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

// --- depth and stencil -----------------------------------------------------

/// `GX2DepthStencilControlReg`, `GX2StencilMaskReg`.
struct DepthStencilControlReg {
    u32be db_depth_control;
};

struct StencilMaskReg {
    u32be front; ///< DB_STENCILREFMASK
    u32be back;  ///< DB_STENCILREFMASK_BF
};

static_assert(sizeof(DepthStencilControlReg) == 0x4);
static_assert(sizeof(StencilMaskReg) == 0x8);

/// `GX2SetDepthStencilControl`'s thirteen arguments: GX2's comparison and
/// stencil-operation values, which are Latte's.
struct DepthStencil {
    bool depth_test = false;
    bool depth_write = false;
    u32 depth_compare = 0;
    bool stencil_test = false;
    bool backface_stencil = false;
    u32 front_compare = 0;
    u32 front_depth_pass = 0;
    u32 front_depth_fail = 0;
    u32 front_stencil_fail = 0;
    u32 back_compare = 0;
    u32 back_depth_pass = 0;
    u32 back_depth_fail = 0;
    u32 back_stencil_fail = 0;
};

/// DB_DEPTH_CONTROL for it. decaf and Cemu pack it the same way; so does
/// `GX2SetDepthOnlyControl`, which is this with the stencil fields zero.
[[nodiscard]] u32 depth_stencil_control(const DepthStencil& state) noexcept;

[[nodiscard]] StencilMaskReg stencil_mask_registers(u32 front_compare_mask,
                                                    u32 front_write_mask,
                                                    u32 front_ref,
                                                    u32 back_compare_mask,
                                                    u32 back_write_mask,
                                                    u32 back_ref) noexcept;

// --- the rasterizer --------------------------------------------------------

/// `GX2PolygonControlReg`, `GX2PolygonOffsetReg`, `GX2PointSizeReg`,
/// `GX2PointLimitsReg`, `GX2LineWidthReg`.
struct PolygonControlReg {
    u32be pa_su_sc_mode_cntl;
};

struct PolygonOffsetReg {
    f32be front_scale; ///< times 16, as the register takes it
    f32be front_offset;
    f32be back_scale; ///< likewise
    f32be back_offset;
    f32be clamp;
};

struct PointSizeReg {
    u32be pa_su_point_size;
};

struct PointLimitsReg {
    u32be pa_su_point_minmax;
};

struct LineWidthReg {
    u32be pa_su_line_cntl;
};

static_assert(sizeof(PolygonControlReg) == 0x4);
static_assert(sizeof(PolygonOffsetReg) == 0x14);
static_assert(sizeof(PointSizeReg) == 0x4);
static_assert(sizeof(PointLimitsReg) == 0x4);
static_assert(sizeof(LineWidthReg) == 0x4);

/// `GX2SetPolygonControl`'s nine arguments.
struct PolygonControl {
    u32 front_face = 0; ///< GX2FrontFace: 0 counter-clockwise, 1 clockwise
    bool cull_front = false;
    bool cull_back = false;
    u32 poly_mode = 0;  ///< a BOOL that GX2 passes straight through
    u32 front_mode = 0; ///< GX2PolygonMode
    u32 back_mode = 0;
    bool offset_front = false;
    bool offset_back = false;
    bool offset_para = false;
};

/// PA_SU_SC_MODE_CNTL for it, as decaf and Cemu pack it.
[[nodiscard]] u32 polygon_control(const PolygonControl& state) noexcept;

/// The scales times 16, everything else as given; decaf and Cemu agree.
[[nodiscard]] PolygonOffsetReg polygon_offset_registers(
    f32 front_offset, f32 front_scale, f32 back_offset, f32 back_scale, f32 clamp) noexcept;

/// A size in the units PA_SU_POINT_SIZE, PA_SU_POINT_MINMAX and
/// PA_SU_LINE_CNTL take: times 8, truncated, held to 16 bits -- Cemu's
/// clamp; decaf lets the value wrap. Negative and NaN sizes are zero.
[[nodiscard]] u32 size_eighths(f32 size) noexcept;

[[nodiscard]] u32 point_size_register(f32 width, f32 height) noexcept;
[[nodiscard]] u32 point_limits_register(f32 min_size, f32 max_size) noexcept;
[[nodiscard]] u32 line_width_register(f32 width) noexcept;

/// PA_CL_CLIP_CNTL for `GX2SetRasterizerClipControlEx`: the rasterizer
/// killed if it is off, near and far clipping off together, half-Z as the
/// DX clip space. Cemu also sets DX_LINEAR_ATTR_CLIP_ENA every time, and so
/// does this; decaf leaves it clear.
[[nodiscard]] u32 clip_control_register(bool rasterizer, bool z_clip, bool half_z) noexcept;

// --- colour -----------------------------------------------------------------

/// `GX2ColorControlReg`, `GX2BlendControlReg`, `GX2BlendConstantColorReg`,
/// `GX2TargetChannelMaskReg`, `GX2AlphaTestReg`, `GX2AlphaToMaskReg`,
/// `GX2AAMaskReg`.
struct ColorControlReg {
    u32be cb_color_control;
};

struct BlendControlReg {
    u32be target; ///< which CB_BLENDn_CONTROL
    u32be cb_blend_control;
};

struct BlendConstantColorReg {
    f32be red;
    f32be green;
    f32be blue;
    f32be alpha;
};

struct TargetChannelMaskReg {
    u32be cb_target_mask;
};

struct AlphaTestReg {
    u32be sx_alpha_test_control;
    f32be sx_alpha_ref;
};

struct AlphaToMaskReg {
    u32be db_alpha_to_mask;
};

struct AaMaskReg {
    u32be pa_sc_aa_mask;
};

static_assert(sizeof(ColorControlReg) == 0x4);
static_assert(sizeof(BlendControlReg) == 0x8);
static_assert(sizeof(BlendConstantColorReg) == 0x10);
static_assert(sizeof(TargetChannelMaskReg) == 0x4);
static_assert(sizeof(AlphaTestReg) == 0x8);
static_assert(sizeof(AlphaToMaskReg) == 0x4);
static_assert(sizeof(AaMaskReg) == 0x4);

/// CB_COLOR_CONTROL for `GX2SetColorControl`: the logic op (a ROP3 code;
/// `GX2LogicOp` is one), a blend-enable bit per target, multiwrite, and
/// colour writes on or off -- SPECIAL_OP normal or disable. decaf and Cemu
/// agree.
[[nodiscard]] u32 color_control_register(u32 logic_op,
                                         u32 blend_enable_mask,
                                         bool multiwrite,
                                         bool color_write) noexcept;

/// `GX2SetBlendControl`'s arguments after the target.
struct BlendControl {
    u32 color_source = 0; ///< GX2BlendMode
    u32 color_destination = 0;
    u32 color_combine = 0; ///< GX2BlendCombineMode
    bool separate_alpha = false;
    u32 alpha_source = 0;
    u32 alpha_destination = 0;
    u32 alpha_combine = 0;
};

/// CB_BLENDn_CONTROL for it. decaf and Cemu agree.
[[nodiscard]] u32 blend_control_register(const BlendControl& state) noexcept;

/// CB_TARGET_MASK: target n's `GX2ChannelMask` in bits 4n to 4n+3.
[[nodiscard]] u32 target_channel_masks_register(const std::array<u32, 8>& masks) noexcept;

[[nodiscard]] AlphaTestReg alpha_test_registers(bool enable, u32 compare, f32 reference) noexcept;

/// DB_ALPHA_TO_MASK for a `GX2AlphaToMaskMode`: decaf's dither offsets for
/// each mode (it alone has the call). An unknown mode is non-dithered.
[[nodiscard]] u32 alpha_to_mask_register(bool enable, u32 mode) noexcept;

[[nodiscard]] u32
aa_mask_register(u32 upper_left, u32 upper_right, u32 lower_left, u32 lower_right) noexcept;

// --- the packets ------------------------------------------------------------

/// What the calls that write more than one packet write, appended to `out`
/// -- for `GX2SetDefaultState`, which writes them too. Two packets each for
/// the stencil masks (front, back), the polygon offset (the four scales and
/// offsets, then the clamp) and the alpha test (control, reference), as
/// decaf and Cemu write them; one for the blend constant. The blend
/// control's target must be below 8.
void append_stencil_mask(std::vector<u32>& out, const StencilMaskReg& reg);
void append_polygon_offset(std::vector<u32>& out, const PolygonOffsetReg& reg);
void append_alpha_test(std::vector<u32>& out, const AlphaTestReg& reg);
void append_blend_control(std::vector<u32>& out, u32 target, u32 value);
void append_blend_constant_color(std::vector<u32>& out, const BlendConstantColorReg& reg);

/// Register the state calls. Called by `register_gx2`.
void register_gx2_state(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
