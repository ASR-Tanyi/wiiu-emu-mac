// Latte's fixed-function state registers: depth and stencil, and the
// rasterizer's polygon, point, line and clip controls.
//
// GX2's state calls pack these (8.1) and the translator (8.11) will unpack
// them into Metal's depth-stencil and rasterizer state, through the same
// fields. Every register address is in Linux's R600 register list
// (`reg_srcs/r600`). DB_DEPTH_CONTROL's fields are also in Linux's `r600d.h`,
// and PA_CL_CLIP_CNTL's, VGT_MULTI_PRIM_IB_RESET_INDX's and PA_CL_VTE_CNTL's
// in AMD's R6xx register reference; the rest are where decaf's and Cemu's
// tables, which agree on every one, put them.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>

namespace wiiu::latte {

/// A depth or stencil comparison (REF_FUNC). GX2's `GX2CompareFunction` has
/// the same values.
enum class CompareFunction : u32 {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
};

/// What a stencil test does to the stencil value (DB_STENCIL_FUNC). GX2's
/// `GX2StencilFunction` has the same values.
enum class StencilOp : u32 {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementClamp = 3,
    DecrementClamp = 4,
    Invert = 5,
    IncrementWrap = 6,
    DecrementWrap = 7,
};

/// Which winding is the front (PA_FACE). GX2's `GX2FrontFace` likewise.
enum class FrontFace : u32 {
    CounterClockwise = 0,
    Clockwise = 1,
};

/// What a polygon is drawn as when polygon mode is on (PA_PTYPE). GX2's
/// `GX2PolygonMode` likewise.
enum class PolygonMode : u32 {
    Points = 0,
    Lines = 1,
    Triangles = 2,
};

/// DB_DEPTH_CONTROL.
union DbDepthControl {
    u32 raw;
    BitField<0, 1, bool> stencil_enable;
    BitField<1, 1, bool> z_enable;
    BitField<2, 1, bool> z_write_enable;
    BitField<4, 3, CompareFunction> z_func;
    BitField<7, 1, bool> backface_enable;
    BitField<8, 3, CompareFunction> stencil_func;
    BitField<11, 3, StencilOp> stencil_fail;
    BitField<14, 3, StencilOp> stencil_zpass;
    BitField<17, 3, StencilOp> stencil_zfail;
    BitField<20, 3, CompareFunction> stencil_func_bf;
    BitField<23, 3, StencilOp> stencil_fail_bf;
    BitField<26, 3, StencilOp> stencil_zpass_bf;
    BitField<29, 3, StencilOp> stencil_zfail_bf;
};

/// DB_STENCILREFMASK, and DB_STENCILREFMASK_BF for back faces.
union DbStencilRefMask {
    u32 raw;
    BitField<0, 8, u32> ref;
    BitField<8, 8, u32> mask;
    BitField<16, 8, u32> write_mask;
};

/// PA_SU_SC_MODE_CNTL: culling, winding, polygon mode and offset enables.
union PaSuScModeCntl {
    u32 raw;
    BitField<0, 1, bool> cull_front;
    BitField<1, 1, bool> cull_back;
    BitField<2, 1, FrontFace> face;
    BitField<3, 2, u32> poly_mode; ///< 0 off; GX2 passes its BOOL through
    BitField<5, 3, PolygonMode> polymode_front_ptype;
    BitField<8, 3, PolygonMode> polymode_back_ptype;
    BitField<11, 1, bool> poly_offset_front_enable;
    BitField<12, 1, bool> poly_offset_back_enable;
    BitField<13, 1, bool> poly_offset_para_enable;
    BitField<16, 1, bool> vtx_window_offset_enable;
    BitField<19, 1, bool> provoking_vtx_last;
    BitField<20, 1, bool> persp_corr_dis;
    BitField<21, 1, bool> multi_prim_ib_ena;
};

/// PA_CL_CLIP_CNTL.
union PaClClipCntl {
    u32 raw;
    BitField<0, 6, u32> ucp_ena; ///< user clip planes 0-5
    BitField<16, 1, bool> clip_disable;
    BitField<19, 1, bool> dx_clip_space_def; ///< depth clips to [0, 1]; GX2's "half Z"
    BitField<22, 1, bool> dx_rasterization_kill;
    BitField<24, 1, bool> dx_linear_attr_clip_ena;
    BitField<26, 1, bool> zclip_near_disable;
    BitField<27, 1, bool> zclip_far_disable;
};

/// PA_SU_POINT_SIZE. GX2 writes the size it is given times 8; decaf notes
/// the fields as 12.4 fixed point, which makes that half the size -- the
/// translator will settle which (8.11).
union PaSuPointSize {
    u32 raw;
    BitField<0, 16, u32> height;
    BitField<16, 16, u32> width;
};

/// PA_SU_POINT_MINMAX, written the same way.
union PaSuPointMinMax {
    u32 raw;
    BitField<0, 16, u32> min_size;
    BitField<16, 16, u32> max_size;
};

/// PA_SU_LINE_CNTL, written the same way.
union PaSuLineCntl {
    u32 raw;
    BitField<0, 16, u32> width;
};

namespace reg {
inline constexpr u32 db_depth_control = 0x28800;
inline constexpr u32 db_stencilrefmask = 0x28430;
inline constexpr u32 db_stencilrefmask_bf = 0x28434;
inline constexpr u32 pa_cl_clip_cntl = 0x28810;
inline constexpr u32 pa_su_sc_mode_cntl = 0x28814;
inline constexpr u32 pa_su_point_size = 0x28A00;
inline constexpr u32 pa_su_point_minmax = 0x28A04;
inline constexpr u32 pa_su_line_cntl = 0x28A08;

/// Front scale, front offset, back scale, back offset: consecutive, and
/// written as one packet. Floats.
inline constexpr u32 pa_su_poly_offset_front_scale = 0x28E00;
inline constexpr u32 pa_su_poly_offset_clamp = 0x28DFC;

/// The index that restarts a strip.
inline constexpr u32 vgt_multi_prim_ib_reset_indx = 0x2840C;
} // namespace reg

} // namespace wiiu::latte
