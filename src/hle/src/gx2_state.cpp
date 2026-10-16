// Phase 8.1 — GX2's state-setting calls. See gx2_state.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_state.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/color_buffer.hpp>
#include <wiiu/latte/render_state.hpp>
#include <wiiu/latte/viewport.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <string_view>

namespace wiiu::hle::gx2 {

ViewportReg
viewport_registers(f32 x, f32 y, f32 width, f32 height, f32 near_z, f32 far_z) noexcept {
    ViewportReg reg{};
    reg.transform = {width * 0.5F,
                     x + (width * 0.5F),
                     height * -0.5F,
                     y + (height * 0.5F),
                     (far_z - near_z) * 0.5F,
                     (far_z + near_z) * 0.5F};

    f32 vertical_clip = 1.0F;
    f32 horizontal_clip = 1.0F;
    if (width != 0.0F && height != 0.0F) {
        if (height < 0.0F) {
            y += height;
            height = -height;
        }
        // From the viewport's near edge to the far side of Latte's 8192
        // pixels, or the other way if the viewport starts past -8192.
        f32 horizontal = x + 8192.0F;
        if (horizontal <= 0.0F) {
            horizontal = 8192.0F - (width + x);
        }
        horizontal_clip = (horizontal + horizontal + width) / width;
        f32 vertical = y + 8192.0F;
        if (vertical <= 0.0F) {
            vertical = 8192.0F - (height + y);
        }
        vertical_clip = (vertical + vertical + height) / height;
    }
    reg.guard_band = {vertical_clip, 1.0F, horizontal_clip, 1.0F};
    reg.z_min = std::min(near_z, far_z);
    reg.z_max = std::max(near_z, far_z);
    return reg;
}

ScissorReg scissor_registers(u32 x, u32 y, u32 width, u32 height) noexcept {
    constexpr u32 limit = 8192;
    latte::PaScGenericScissorTl top_left{.raw = 0};
    top_left.x = std::min(x, limit);
    top_left.y = std::min(y, limit);
    top_left.window_offset_disable = true;
    latte::PaScGenericScissorBr bottom_right{.raw = 0};
    bottom_right.x = std::min(x + width, limit);
    bottom_right.y = std::min(y + height, limit);
    return ScissorReg{.top_left = top_left.raw, .bottom_right = bottom_right.raw};
}

u32 depth_stencil_control(const DepthStencil& state) noexcept {
    using latte::CompareFunction;
    using latte::StencilOp;
    latte::DbDepthControl reg{.raw = 0};
    reg.z_enable = state.depth_test;
    reg.z_write_enable = state.depth_write;
    reg.z_func = static_cast<CompareFunction>(state.depth_compare & 7U);
    reg.stencil_enable = state.stencil_test;
    reg.backface_enable = state.backface_stencil;
    reg.stencil_func = static_cast<CompareFunction>(state.front_compare & 7U);
    reg.stencil_zpass = static_cast<StencilOp>(state.front_depth_pass & 7U);
    reg.stencil_zfail = static_cast<StencilOp>(state.front_depth_fail & 7U);
    reg.stencil_fail = static_cast<StencilOp>(state.front_stencil_fail & 7U);
    reg.stencil_func_bf = static_cast<CompareFunction>(state.back_compare & 7U);
    reg.stencil_zpass_bf = static_cast<StencilOp>(state.back_depth_pass & 7U);
    reg.stencil_zfail_bf = static_cast<StencilOp>(state.back_depth_fail & 7U);
    reg.stencil_fail_bf = static_cast<StencilOp>(state.back_stencil_fail & 7U);
    return reg.raw;
}

StencilMaskReg stencil_mask_registers(u32 front_compare_mask,
                                      u32 front_write_mask,
                                      u32 front_ref,
                                      u32 back_compare_mask,
                                      u32 back_write_mask,
                                      u32 back_ref) noexcept {
    latte::DbStencilRefMask front{.raw = 0};
    front.ref = front_ref & 0xFFU;
    front.mask = front_compare_mask & 0xFFU;
    front.write_mask = front_write_mask & 0xFFU;
    latte::DbStencilRefMask back{.raw = 0};
    back.ref = back_ref & 0xFFU;
    back.mask = back_compare_mask & 0xFFU;
    back.write_mask = back_write_mask & 0xFFU;
    return StencilMaskReg{.front = front.raw, .back = back.raw};
}

u32 polygon_control(const PolygonControl& state) noexcept {
    latte::PaSuScModeCntl reg{.raw = 0};
    reg.face =
        state.front_face != 0 ? latte::FrontFace::Clockwise : latte::FrontFace::CounterClockwise;
    reg.cull_front = state.cull_front;
    reg.cull_back = state.cull_back;
    reg.poly_mode = state.poly_mode & 3U;
    reg.polymode_front_ptype = static_cast<latte::PolygonMode>(state.front_mode & 7U);
    reg.polymode_back_ptype = static_cast<latte::PolygonMode>(state.back_mode & 7U);
    reg.poly_offset_front_enable = state.offset_front;
    reg.poly_offset_back_enable = state.offset_back;
    reg.poly_offset_para_enable = state.offset_para;
    return reg.raw;
}

PolygonOffsetReg polygon_offset_registers(
    f32 front_offset, f32 front_scale, f32 back_offset, f32 back_scale, f32 clamp) noexcept {
    return PolygonOffsetReg{
        .front_scale = front_scale * 16.0F,
        .front_offset = front_offset,
        .back_scale = back_scale * 16.0F,
        .back_offset = back_offset,
        .clamp = clamp,
    };
}

u32 size_eighths(f32 size) noexcept {
    const f32 scaled = size * 8.0F;
    if (!(scaled > 0.0F)) {
        return 0;
    }
    if (scaled >= 65535.0F) {
        return 0xFFFF;
    }
    return static_cast<u32>(scaled);
}

u32 point_size_register(f32 width, f32 height) noexcept {
    latte::PaSuPointSize reg{.raw = 0};
    reg.width = size_eighths(width);
    reg.height = size_eighths(height);
    return reg.raw;
}

u32 point_limits_register(f32 min_size, f32 max_size) noexcept {
    latte::PaSuPointMinMax reg{.raw = 0};
    reg.min_size = size_eighths(min_size);
    reg.max_size = size_eighths(max_size);
    return reg.raw;
}

u32 line_width_register(f32 width) noexcept {
    latte::PaSuLineCntl reg{.raw = 0};
    reg.width = size_eighths(width);
    return reg.raw;
}

u32 clip_control_register(bool rasterizer, bool z_clip, bool half_z) noexcept {
    latte::PaClClipCntl reg{.raw = 0};
    reg.dx_rasterization_kill = !rasterizer;
    reg.zclip_near_disable = !z_clip;
    reg.zclip_far_disable = !z_clip;
    reg.dx_clip_space_def = half_z;
    reg.dx_linear_attr_clip_ena = true;
    return reg.raw;
}

u32 color_control_register(u32 logic_op,
                           u32 blend_enable_mask,
                           bool multiwrite,
                           bool color_write) noexcept {
    latte::CbColorControl reg{.raw = 0};
    reg.rop3 = logic_op & 0xFFU;
    reg.target_blend_enable = blend_enable_mask & 0xFFU;
    reg.multiwrite_enable = multiwrite;
    reg.special_op = color_write ? latte::cb_special_normal : latte::cb_special_disable;
    return reg.raw;
}

u32 blend_control_register(const BlendControl& state) noexcept {
    latte::CbBlendControl reg{.raw = 0};
    reg.color_srcblend = state.color_source & 0x1FU;
    reg.color_destblend = state.color_destination & 0x1FU;
    reg.color_comb_fcn = state.color_combine & 7U;
    reg.separate_alpha_blend = state.separate_alpha;
    reg.alpha_srcblend = state.alpha_source & 0x1FU;
    reg.alpha_destblend = state.alpha_destination & 0x1FU;
    reg.alpha_comb_fcn = state.alpha_combine & 7U;
    return reg.raw;
}

u32 target_channel_masks_register(const std::array<u32, 8>& masks) noexcept {
    u32 value = 0;
    for (u32 target = 0; target < masks.size(); ++target) {
        value |= (masks[target] & 0xFU) << (4 * target);
    }
    return value;
}

AlphaTestReg alpha_test_registers(bool enable, u32 compare, f32 reference) noexcept {
    latte::SxAlphaTestControl control{.raw = 0};
    control.alpha_test_enable = enable;
    control.alpha_func = static_cast<latte::CompareFunction>(compare & 7U);
    return AlphaTestReg{.sx_alpha_test_control = control.raw, .sx_alpha_ref = reference};
}

u32 alpha_to_mask_register(bool enable, u32 mode) noexcept {
    // The four quad offsets for each mode, offset 0 first: decaf's table.
    constexpr std::array<std::array<u32, 4>, 5> offsets{{
        {2, 2, 2, 2}, // non-dithered
        {0, 2, 3, 1}, // dither 0
        {2, 1, 0, 3}, // dither 90
        {1, 3, 2, 0}, // dither 180
        {3, 0, 1, 2}, // dither 270
    }};
    const std::array<u32, 4>& chosen = offsets[mode < offsets.size() ? mode : 0];
    latte::DbAlphaToMask reg{.raw = 0};
    reg.enable = enable;
    reg.offset0 = chosen[0];
    reg.offset1 = chosen[1];
    reg.offset2 = chosen[2];
    reg.offset3 = chosen[3];
    return reg.raw;
}

u32 aa_mask_register(u32 upper_left, u32 upper_right, u32 lower_left, u32 lower_right) noexcept {
    latte::PaScAaMask reg{.raw = 0};
    reg.upper_left = upper_left & 0xFFU;
    reg.upper_right = upper_right & 0xFFU;
    reg.lower_left = lower_left & 0xFFU;
    reg.lower_right = lower_right & 0xFFU;
    return reg.raw;
}

namespace {

[[nodiscard]] u32 float_bits(const f32be& value) noexcept {
    return std::bit_cast<u32>(value.value());
}

} // namespace

void append_stencil_mask(std::vector<u32>& out, const StencilMaskReg& reg) {
    append_context_register(out, latte::reg::db_stencilrefmask, reg.front.value());
    append_context_register(out, latte::reg::db_stencilrefmask_bf, reg.back.value());
}

void append_polygon_offset(std::vector<u32>& out, const PolygonOffsetReg& reg) {
    const std::array<u32, 4> values{float_bits(reg.front_scale),
                                    float_bits(reg.front_offset),
                                    float_bits(reg.back_scale),
                                    float_bits(reg.back_offset)};
    append_registers(
        out, latte::pm4::Opcode::SetContextReg, latte::reg::pa_su_poly_offset_front_scale, values);
    append_context_register(out, latte::reg::pa_su_poly_offset_clamp, float_bits(reg.clamp));
}

void append_alpha_test(std::vector<u32>& out, const AlphaTestReg& reg) {
    append_context_register(
        out, latte::reg::sx_alpha_test_control, reg.sx_alpha_test_control.value());
    append_context_register(out, latte::reg::sx_alpha_ref, float_bits(reg.sx_alpha_ref));
}

void append_blend_control(std::vector<u32>& out, u32 target, u32 value) {
    append_context_register(out, latte::reg::cb_blend0_control + (4 * target), value);
}

void append_blend_constant_color(std::vector<u32>& out, const BlendConstantColorReg& reg) {
    const std::array<u32, 4> values{
        float_bits(reg.red), float_bits(reg.green), float_bits(reg.blue), float_bits(reg.alpha)};
    append_registers(out, latte::pm4::Opcode::SetContextReg, latte::reg::cb_blend_red, values);
}

namespace {

[[nodiscard]] u32 bits(const f32be& value) noexcept {
    return std::bit_cast<u32>(value.value());
}

/// GX2SetViewportReg's three packets: the transform, the guard band, the
/// depth range.
void set_viewport(Call& call, const ViewportReg& reg) {
    std::array<u32, 6> transform{};
    std::transform(reg.transform.begin(), reg.transform.end(), transform.begin(), bits);
    std::array<u32, 4> guard_band{};
    std::transform(reg.guard_band.begin(), reg.guard_band.end(), guard_band.begin(), bits);
    const std::array<u32, 2> depth{bits(reg.z_min), bits(reg.z_max)};
    static_cast<void>(emit_context_registers(call, latte::reg::pa_cl_vport_xscale_0, transform));
    static_cast<void>(emit_context_registers(call, latte::reg::pa_cl_gb_vert_clip_adj, guard_band));
    static_cast<void>(emit_context_registers(call, latte::reg::pa_sc_vport_zmin_0, depth));
}

void set_scissor(Call& call, const ScissorReg& reg) {
    const std::array<u32, 2> corners{reg.top_left, reg.bottom_right};
    static_cast<void>(emit_context_registers(call, latte::reg::pa_sc_generic_scissor_tl, corners));
}

/// `void GX2InitViewportReg(GX2ViewportReg *reg, float x, float y, float
/// width, float height, float nearZ, float farZ)`.
void gx2_init_viewport_reg(
    Call& call, u32 address, f32 x, f32 y, f32 width, f32 height, f32 near_z, f32 far_z) {
    if (!write_guest(
            call.memory, address, viewport_registers(x, y, width, height, near_z, far_z))) {
        WIIU_LOG_ERROR(Gx2, "GX2InitViewportReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_viewport_reg(Call& call, u32 address) {
    ViewportReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetViewportReg: no viewport at {:#010x}", address);
        return;
    }
    set_viewport(call, reg);
}

void gx2_set_viewport(Call& call, f32 x, f32 y, f32 width, f32 height, f32 near_z, f32 far_z) {
    set_viewport(call, viewport_registers(x, y, width, height, near_z, far_z));
}

void gx2_init_scissor_reg(Call& call, u32 address, u32 x, u32 y, u32 width, u32 height) {
    if (!write_guest(call.memory, address, scissor_registers(x, y, width, height))) {
        WIIU_LOG_ERROR(Gx2, "GX2InitScissorReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_scissor_reg(Call& call, u32 address) {
    ScissorReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetScissorReg: no scissor at {:#010x}", address);
        return;
    }
    set_scissor(call, reg);
}

void gx2_set_scissor(Call& call, u32 x, u32 y, u32 width, u32 height) {
    set_scissor(call, scissor_registers(x, y, width, height));
}

// --- depth and stencil -----------------------------------------------------

/// One register, in a packet of its own.
void set_word(Call& call, u32 address, u32 value) {
    const std::array<u32, 1> values{value};
    static_cast<void>(emit_context_registers(call, address, values));
}

void set_depth_stencil_control(Call& call, u32 value) {
    set_word(call, latte::reg::db_depth_control, value);
}

/// `void GX2SetDepthStencilControl(BOOL depthTest, BOOL depthWrite,
/// GX2CompareFunction depthCompare, BOOL stencilTest, BOOL backfaceStencil,
/// GX2CompareFunction frontStencilFunc, GX2StencilFunction frontStencilZPass,
/// GX2StencilFunction frontStencilZFail, GX2StencilFunction
/// frontStencilFail, GX2CompareFunction backStencilFunc, GX2StencilFunction
/// backStencilZPass, GX2StencilFunction backStencilZFail, GX2StencilFunction
/// backStencilFail)` -- the last five on the stack.
void gx2_set_depth_stencil_control(Call& call,
                                   bool depth_test,
                                   bool depth_write,
                                   u32 depth_compare,
                                   bool stencil_test,
                                   bool backface_stencil,
                                   u32 front_compare,
                                   u32 front_depth_pass,
                                   u32 front_depth_fail,
                                   u32 front_stencil_fail,
                                   u32 back_compare,
                                   u32 back_depth_pass,
                                   u32 back_depth_fail,
                                   u32 back_stencil_fail) {
    set_depth_stencil_control(call,
                              depth_stencil_control(DepthStencil{
                                  .depth_test = depth_test,
                                  .depth_write = depth_write,
                                  .depth_compare = depth_compare,
                                  .stencil_test = stencil_test,
                                  .backface_stencil = backface_stencil,
                                  .front_compare = front_compare,
                                  .front_depth_pass = front_depth_pass,
                                  .front_depth_fail = front_depth_fail,
                                  .front_stencil_fail = front_stencil_fail,
                                  .back_compare = back_compare,
                                  .back_depth_pass = back_depth_pass,
                                  .back_depth_fail = back_depth_fail,
                                  .back_stencil_fail = back_stencil_fail,
                              }));
}

void gx2_init_depth_stencil_control_reg(Call& call,
                                        u32 address,
                                        bool depth_test,
                                        bool depth_write,
                                        u32 depth_compare,
                                        bool stencil_test,
                                        bool backface_stencil,
                                        u32 front_compare,
                                        u32 front_depth_pass,
                                        u32 front_depth_fail,
                                        u32 front_stencil_fail,
                                        u32 back_compare,
                                        u32 back_depth_pass,
                                        u32 back_depth_fail,
                                        u32 back_stencil_fail) {
    const DepthStencilControlReg reg{.db_depth_control = depth_stencil_control(DepthStencil{
                                         .depth_test = depth_test,
                                         .depth_write = depth_write,
                                         .depth_compare = depth_compare,
                                         .stencil_test = stencil_test,
                                         .backface_stencil = backface_stencil,
                                         .front_compare = front_compare,
                                         .front_depth_pass = front_depth_pass,
                                         .front_depth_fail = front_depth_fail,
                                         .front_stencil_fail = front_stencil_fail,
                                         .back_compare = back_compare,
                                         .back_depth_pass = back_depth_pass,
                                         .back_depth_fail = back_depth_fail,
                                         .back_stencil_fail = back_stencil_fail,
                                     })};
    if (!write_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitDepthStencilControlReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_depth_stencil_control_reg(Call& call, u32 address) {
    DepthStencilControlReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetDepthStencilControlReg: nothing at {:#010x}", address);
        return;
    }
    set_depth_stencil_control(call, reg.db_depth_control.value());
}

void gx2_set_depth_only_control(Call& call, bool depth_test, bool depth_write, u32 compare) {
    set_depth_stencil_control(call,
                              depth_stencil_control(DepthStencil{.depth_test = depth_test,
                                                                 .depth_write = depth_write,
                                                                 .depth_compare = compare}));
}

/// Emit what an `append_*` builds.
template <typename... Args>
void emit_built(Call& call, void (*append)(std::vector<u32>&, Args...), Args... args) {
    std::vector<u32> packets;
    append(packets, args...);
    static_cast<void>(emit(call, packets));
}

void set_stencil_mask(Call& call, const StencilMaskReg& reg) {
    emit_built<const StencilMaskReg&>(call, append_stencil_mask, reg);
}

/// `void GX2SetStencilMask(uint8_t frontMask, uint8_t frontWriteMask,
/// uint8_t frontRef, uint8_t backMask, uint8_t backWriteMask, uint8_t
/// backRef)`.
void gx2_set_stencil_mask(Call& call,
                          u32 front_mask,
                          u32 front_write_mask,
                          u32 front_ref,
                          u32 back_mask,
                          u32 back_write_mask,
                          u32 back_ref) {
    set_stencil_mask(
        call,
        stencil_mask_registers(
            front_mask, front_write_mask, front_ref, back_mask, back_write_mask, back_ref));
}

void gx2_init_stencil_mask_reg(Call& call,
                               u32 address,
                               u32 front_mask,
                               u32 front_write_mask,
                               u32 front_ref,
                               u32 back_mask,
                               u32 back_write_mask,
                               u32 back_ref) {
    if (!write_guest(
            call.memory,
            address,
            stencil_mask_registers(
                front_mask, front_write_mask, front_ref, back_mask, back_write_mask, back_ref))) {
        WIIU_LOG_ERROR(Gx2, "GX2InitStencilMaskReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_stencil_mask_reg(Call& call, u32 address) {
    StencilMaskReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetStencilMaskReg: nothing at {:#010x}", address);
        return;
    }
    set_stencil_mask(call, reg);
}

// --- the rasterizer --------------------------------------------------------

void set_polygon_control(Call& call, u32 value) {
    set_word(call, latte::reg::pa_su_sc_mode_cntl, value);
}

/// `void GX2SetPolygonControl(GX2FrontFace frontFace, BOOL cullFront, BOOL
/// cullBack, BOOL polyMode, GX2PolygonMode polyModeFront, GX2PolygonMode
/// polyModeBack, BOOL polyOffsetFrontEnable, BOOL polyOffsetBackEnable, BOOL
/// polyOffsetParaEnable)` -- the last on the stack.
void gx2_set_polygon_control(Call& call,
                             u32 front_face,
                             bool cull_front,
                             bool cull_back,
                             u32 poly_mode,
                             u32 front_mode,
                             u32 back_mode,
                             bool offset_front,
                             bool offset_back,
                             bool offset_para) {
    set_polygon_control(call,
                        polygon_control(PolygonControl{.front_face = front_face,
                                                       .cull_front = cull_front,
                                                       .cull_back = cull_back,
                                                       .poly_mode = poly_mode,
                                                       .front_mode = front_mode,
                                                       .back_mode = back_mode,
                                                       .offset_front = offset_front,
                                                       .offset_back = offset_back,
                                                       .offset_para = offset_para}));
}

void gx2_init_polygon_control_reg(Call& call,
                                  u32 address,
                                  u32 front_face,
                                  bool cull_front,
                                  bool cull_back,
                                  u32 poly_mode,
                                  u32 front_mode,
                                  u32 back_mode,
                                  bool offset_front,
                                  bool offset_back,
                                  bool offset_para) {
    const PolygonControlReg reg{.pa_su_sc_mode_cntl =
                                    polygon_control(PolygonControl{.front_face = front_face,
                                                                   .cull_front = cull_front,
                                                                   .cull_back = cull_back,
                                                                   .poly_mode = poly_mode,
                                                                   .front_mode = front_mode,
                                                                   .back_mode = back_mode,
                                                                   .offset_front = offset_front,
                                                                   .offset_back = offset_back,
                                                                   .offset_para = offset_para})};
    if (!write_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitPolygonControlReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_polygon_control_reg(Call& call, u32 address) {
    PolygonControlReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetPolygonControlReg: nothing at {:#010x}", address);
        return;
    }
    set_polygon_control(call, reg.pa_su_sc_mode_cntl.value());
}

/// Culling alone: polygon mode and offsets off, as decaf and Cemu have it.
void gx2_set_cull_only_control(Call& call, u32 front_face, bool cull_front, bool cull_back) {
    set_polygon_control(call,
                        polygon_control(PolygonControl{.front_face = front_face,
                                                       .cull_front = cull_front,
                                                       .cull_back = cull_back}));
}

void set_polygon_offset(Call& call, const PolygonOffsetReg& reg) {
    emit_built<const PolygonOffsetReg&>(call, append_polygon_offset, reg);
}

void gx2_set_polygon_offset(
    Call& call, f32 front_offset, f32 front_scale, f32 back_offset, f32 back_scale, f32 clamp) {
    set_polygon_offset(
        call, polygon_offset_registers(front_offset, front_scale, back_offset, back_scale, clamp));
}

void gx2_init_polygon_offset_reg(Call& call,
                                 u32 address,
                                 f32 front_offset,
                                 f32 front_scale,
                                 f32 back_offset,
                                 f32 back_scale,
                                 f32 clamp) {
    if (!write_guest(
            call.memory,
            address,
            polygon_offset_registers(front_offset, front_scale, back_offset, back_scale, clamp))) {
        WIIU_LOG_ERROR(Gx2, "GX2InitPolygonOffsetReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_polygon_offset_reg(Call& call, u32 address) {
    PolygonOffsetReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetPolygonOffsetReg: nothing at {:#010x}", address);
        return;
    }
    set_polygon_offset(call, reg);
}

// The point size, point limits and line width are each one register, kept
// in a one-word structure.

void init_word(Call& call, std::string_view name, u32 address, u32 value) {
    if (!call.memory.try_write<u32>(address, value)) {
        WIIU_LOG_ERROR(Gx2, "{}: nowhere to write at {:#010x}", name, address);
    }
}

void set_word_from(Call& call, std::string_view name, u32 reg_address, u32 address) {
    u32 value = 0;
    if (!call.memory.try_read<u32>(address, value)) {
        WIIU_LOG_ERROR(Gx2, "{}: nothing at {:#010x}", name, address);
        return;
    }
    set_word(call, reg_address, value);
}

void gx2_set_point_size(Call& call, f32 width, f32 height) {
    set_word(call, latte::reg::pa_su_point_size, point_size_register(width, height));
}

void gx2_init_point_size_reg(Call& call, u32 address, f32 width, f32 height) {
    init_word(call, "GX2InitPointSizeReg", address, point_size_register(width, height));
}

void gx2_set_point_size_reg(Call& call, u32 address) {
    set_word_from(call, "GX2SetPointSizeReg", latte::reg::pa_su_point_size, address);
}

void gx2_set_point_limits(Call& call, f32 min_size, f32 max_size) {
    set_word(call, latte::reg::pa_su_point_minmax, point_limits_register(min_size, max_size));
}

void gx2_init_point_limits_reg(Call& call, u32 address, f32 min_size, f32 max_size) {
    init_word(call, "GX2InitPointLimitsReg", address, point_limits_register(min_size, max_size));
}

void gx2_set_point_limits_reg(Call& call, u32 address) {
    set_word_from(call, "GX2SetPointLimitsReg", latte::reg::pa_su_point_minmax, address);
}

void gx2_set_line_width(Call& call, f32 width) {
    set_word(call, latte::reg::pa_su_line_cntl, line_width_register(width));
}

void gx2_init_line_width_reg(Call& call, u32 address, f32 width) {
    init_word(call, "GX2InitLineWidthReg", address, line_width_register(width));
}

void gx2_set_line_width_reg(Call& call, u32 address) {
    set_word_from(call, "GX2SetLineWidthReg", latte::reg::pa_su_line_cntl, address);
}

void gx2_set_rasterizer_clip_control_ex(Call& call, bool rasterizer, bool z_clip, bool half_z) {
    set_word(call, latte::reg::pa_cl_clip_cntl, clip_control_register(rasterizer, z_clip, half_z));
}

void gx2_set_rasterizer_clip_control(Call& call, bool rasterizer, bool z_clip) {
    gx2_set_rasterizer_clip_control_ex(call, rasterizer, z_clip, false);
}

/// The whole of the index register, as decaf and Cemu write it.
void gx2_set_primitive_restart_index(Call& call, u32 index) {
    set_word(call, latte::reg::vgt_multi_prim_ib_reset_indx, index);
}

// --- colour -----------------------------------------------------------------

void gx2_set_color_control(
    Call& call, u32 logic_op, u32 blend_enable_mask, bool multiwrite, bool color_write) {
    set_word(call,
             latte::reg::cb_color_control,
             color_control_register(logic_op, blend_enable_mask, multiwrite, color_write));
}

void gx2_init_color_control_reg(Call& call,
                                u32 address,
                                u32 logic_op,
                                u32 blend_enable_mask,
                                bool multiwrite,
                                bool color_write) {
    init_word(call,
              "GX2InitColorControlReg",
              address,
              color_control_register(logic_op, blend_enable_mask, multiwrite, color_write));
}

void gx2_set_color_control_reg(Call& call, u32 address) {
    set_word_from(call, "GX2SetColorControlReg", latte::reg::cb_color_control, address);
}

void set_blend_control(Call& call, u32 target, u32 value) {
    if (target >= latte::reg::render_target_count) {
        WIIU_LOG_ERROR(Gx2, "a blend control for render target {}, which does not exist", target);
        return;
    }
    emit_built<u32, u32>(call, append_blend_control, target, value);
}

/// `void GX2SetBlendControl(GX2RenderTarget target, GX2BlendMode
/// colorSrcBlend, GX2BlendMode colorDstBlend, GX2BlendCombineMode
/// colorCombine, BOOL useAlphaBlend, GX2BlendMode alphaSrcBlend,
/// GX2BlendMode alphaDstBlend, GX2BlendCombineMode alphaCombine)`.
void gx2_set_blend_control(Call& call,
                           u32 target,
                           u32 color_source,
                           u32 color_destination,
                           u32 color_combine,
                           bool separate_alpha,
                           u32 alpha_source,
                           u32 alpha_destination,
                           u32 alpha_combine) {
    set_blend_control(call,
                      target,
                      blend_control_register(BlendControl{.color_source = color_source,
                                                          .color_destination = color_destination,
                                                          .color_combine = color_combine,
                                                          .separate_alpha = separate_alpha,
                                                          .alpha_source = alpha_source,
                                                          .alpha_destination = alpha_destination,
                                                          .alpha_combine = alpha_combine}));
}

/// Nine arguments: the alpha combine mode is on the stack.
void gx2_init_blend_control_reg(Call& call,
                                u32 address,
                                u32 target,
                                u32 color_source,
                                u32 color_destination,
                                u32 color_combine,
                                bool separate_alpha,
                                u32 alpha_source,
                                u32 alpha_destination,
                                u32 alpha_combine) {
    const BlendControlReg reg{.target = target,
                              .cb_blend_control = blend_control_register(
                                  BlendControl{.color_source = color_source,
                                               .color_destination = color_destination,
                                               .color_combine = color_combine,
                                               .separate_alpha = separate_alpha,
                                               .alpha_source = alpha_source,
                                               .alpha_destination = alpha_destination,
                                               .alpha_combine = alpha_combine})};
    if (!write_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitBlendControlReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_blend_control_reg(Call& call, u32 address) {
    BlendControlReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetBlendControlReg: nothing at {:#010x}", address);
        return;
    }
    set_blend_control(call, reg.target.value(), reg.cb_blend_control.value());
}

void set_blend_constant_color(Call& call, const BlendConstantColorReg& reg) {
    emit_built<const BlendConstantColorReg&>(call, append_blend_constant_color, reg);
}

void gx2_set_blend_constant_color(Call& call, f32 red, f32 green, f32 blue, f32 alpha) {
    set_blend_constant_color(
        call, BlendConstantColorReg{.red = red, .green = green, .blue = blue, .alpha = alpha});
}

void gx2_init_blend_constant_color_reg(
    Call& call, u32 address, f32 red, f32 green, f32 blue, f32 alpha) {
    if (!write_guest(
            call.memory,
            address,
            BlendConstantColorReg{.red = red, .green = green, .blue = blue, .alpha = alpha})) {
        WIIU_LOG_ERROR(Gx2, "GX2InitBlendConstantColorReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_blend_constant_color_reg(Call& call, u32 address) {
    BlendConstantColorReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetBlendConstantColorReg: nothing at {:#010x}", address);
        return;
    }
    set_blend_constant_color(call, reg);
}

void gx2_set_target_channel_masks(Call& call,
                                  u32 mask0,
                                  u32 mask1,
                                  u32 mask2,
                                  u32 mask3,
                                  u32 mask4,
                                  u32 mask5,
                                  u32 mask6,
                                  u32 mask7) {
    set_word(
        call,
        latte::reg::cb_target_mask,
        target_channel_masks_register({mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7}));
}

/// Nine arguments: target 7's mask is on the stack.
void gx2_init_target_channel_masks_reg(Call& call,
                                       u32 address,
                                       u32 mask0,
                                       u32 mask1,
                                       u32 mask2,
                                       u32 mask3,
                                       u32 mask4,
                                       u32 mask5,
                                       u32 mask6,
                                       u32 mask7) {
    init_word(
        call,
        "GX2InitTargetChannelMasksReg",
        address,
        target_channel_masks_register({mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7}));
}

void gx2_set_target_channel_masks_reg(Call& call, u32 address) {
    set_word_from(call, "GX2SetTargetChannelMasksReg", latte::reg::cb_target_mask, address);
}

void set_alpha_test(Call& call, const AlphaTestReg& reg) {
    emit_built<const AlphaTestReg&>(call, append_alpha_test, reg);
}

void gx2_set_alpha_test(Call& call, bool enable, u32 compare, f32 reference) {
    set_alpha_test(call, alpha_test_registers(enable, compare, reference));
}

void gx2_init_alpha_test_reg(Call& call, u32 address, bool enable, u32 compare, f32 reference) {
    if (!write_guest(call.memory, address, alpha_test_registers(enable, compare, reference))) {
        WIIU_LOG_ERROR(Gx2, "GX2InitAlphaTestReg: nowhere to write at {:#010x}", address);
    }
}

void gx2_set_alpha_test_reg(Call& call, u32 address) {
    AlphaTestReg reg{};
    if (!read_guest(call.memory, address, reg)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetAlphaTestReg: nothing at {:#010x}", address);
        return;
    }
    set_alpha_test(call, reg);
}

void gx2_set_alpha_to_mask(Call& call, bool enable, u32 mode) {
    set_word(call, latte::reg::db_alpha_to_mask, alpha_to_mask_register(enable, mode));
}

void gx2_init_alpha_to_mask_reg(Call& call, u32 address, bool enable, u32 mode) {
    init_word(call, "GX2InitAlphaToMaskReg", address, alpha_to_mask_register(enable, mode));
}

void gx2_set_alpha_to_mask_reg(Call& call, u32 address) {
    set_word_from(call, "GX2SetAlphaToMaskReg", latte::reg::db_alpha_to_mask, address);
}

void gx2_set_aa_mask(Call& call, u32 upper_left, u32 upper_right, u32 lower_left, u32 lower_right) {
    set_word(call,
             latte::reg::pa_sc_aa_mask,
             aa_mask_register(upper_left, upper_right, lower_left, lower_right));
}

void gx2_init_aa_mask_reg(
    Call& call, u32 address, u32 upper_left, u32 upper_right, u32 lower_left, u32 lower_right) {
    init_word(call,
              "GX2InitAAMaskReg",
              address,
              aa_mask_register(upper_left, upper_right, lower_left, lower_right));
}

void gx2_set_aa_mask_reg(Call& call, u32 address) {
    set_word_from(call, "GX2SetAAMaskReg", latte::reg::pa_sc_aa_mask, address);
}

// --- stream out and tessellation ------------------------------------------

void gx2_set_stream_out_enable(Call& call, bool enable) {
    set_word(call, latte::reg::vgt_strmout_en, enable ? 1U : 0U);
}

void gx2_set_max_tessellation_level(Call& call, f32 level) {
    set_word(call, latte::reg::vgt_hos_max_tess_level, std::bit_cast<u32>(level));
}

void gx2_set_min_tessellation_level(Call& call, f32 level) {
    set_word(call, latte::reg::vgt_hos_min_tess_level, std::bit_cast<u32>(level));
}

} // namespace

void register_gx2_state(Registry& registry) {
    registry.add("gx2", "GX2InitViewportReg", gx2_init_viewport_reg);
    registry.add("gx2", "GX2SetViewportReg", gx2_set_viewport_reg);
    registry.add("gx2", "GX2SetViewport", gx2_set_viewport);
    registry.add("gx2", "GX2InitScissorReg", gx2_init_scissor_reg);
    registry.add("gx2", "GX2SetScissorReg", gx2_set_scissor_reg);
    registry.add("gx2", "GX2SetScissor", gx2_set_scissor);

    registry.add("gx2", "GX2SetDepthStencilControl", gx2_set_depth_stencil_control);
    registry.add("gx2", "GX2InitDepthStencilControlReg", gx2_init_depth_stencil_control_reg);
    registry.add("gx2", "GX2SetDepthStencilControlReg", gx2_set_depth_stencil_control_reg);
    registry.add("gx2", "GX2SetDepthOnlyControl", gx2_set_depth_only_control);
    registry.add("gx2", "GX2SetStencilMask", gx2_set_stencil_mask);
    registry.add("gx2", "GX2InitStencilMaskReg", gx2_init_stencil_mask_reg);
    registry.add("gx2", "GX2SetStencilMaskReg", gx2_set_stencil_mask_reg);

    registry.add("gx2", "GX2SetPolygonControl", gx2_set_polygon_control);
    registry.add("gx2", "GX2InitPolygonControlReg", gx2_init_polygon_control_reg);
    registry.add("gx2", "GX2SetPolygonControlReg", gx2_set_polygon_control_reg);
    registry.add("gx2", "GX2SetCullOnlyControl", gx2_set_cull_only_control);
    registry.add("gx2", "GX2SetPolygonOffset", gx2_set_polygon_offset);
    registry.add("gx2", "GX2InitPolygonOffsetReg", gx2_init_polygon_offset_reg);
    registry.add("gx2", "GX2SetPolygonOffsetReg", gx2_set_polygon_offset_reg);
    registry.add("gx2", "GX2SetPointSize", gx2_set_point_size);
    registry.add("gx2", "GX2InitPointSizeReg", gx2_init_point_size_reg);
    registry.add("gx2", "GX2SetPointSizeReg", gx2_set_point_size_reg);
    registry.add("gx2", "GX2SetPointLimits", gx2_set_point_limits);
    registry.add("gx2", "GX2InitPointLimitsReg", gx2_init_point_limits_reg);
    registry.add("gx2", "GX2SetPointLimitsReg", gx2_set_point_limits_reg);
    registry.add("gx2", "GX2SetLineWidth", gx2_set_line_width);
    registry.add("gx2", "GX2InitLineWidthReg", gx2_init_line_width_reg);
    registry.add("gx2", "GX2SetLineWidthReg", gx2_set_line_width_reg);
    registry.add("gx2", "GX2SetRasterizerClipControl", gx2_set_rasterizer_clip_control);
    registry.add("gx2", "GX2SetRasterizerClipControlEx", gx2_set_rasterizer_clip_control_ex);
    registry.add("gx2", "GX2SetRasterizerClipControlHalfZ", gx2_set_rasterizer_clip_control_ex);
    registry.add("gx2", "GX2SetPrimitiveRestartIndex", gx2_set_primitive_restart_index);

    registry.add("gx2", "GX2SetColorControl", gx2_set_color_control);
    registry.add("gx2", "GX2InitColorControlReg", gx2_init_color_control_reg);
    registry.add("gx2", "GX2SetColorControlReg", gx2_set_color_control_reg);
    registry.add("gx2", "GX2SetBlendControl", gx2_set_blend_control);
    registry.add("gx2", "GX2InitBlendControlReg", gx2_init_blend_control_reg);
    registry.add("gx2", "GX2SetBlendControlReg", gx2_set_blend_control_reg);
    registry.add("gx2", "GX2SetBlendConstantColor", gx2_set_blend_constant_color);
    registry.add("gx2", "GX2InitBlendConstantColorReg", gx2_init_blend_constant_color_reg);
    registry.add("gx2", "GX2SetBlendConstantColorReg", gx2_set_blend_constant_color_reg);
    registry.add("gx2", "GX2SetTargetChannelMasks", gx2_set_target_channel_masks);
    registry.add("gx2", "GX2InitTargetChannelMasksReg", gx2_init_target_channel_masks_reg);
    registry.add("gx2", "GX2SetTargetChannelMasksReg", gx2_set_target_channel_masks_reg);
    registry.add("gx2", "GX2SetAlphaTest", gx2_set_alpha_test);
    registry.add("gx2", "GX2InitAlphaTestReg", gx2_init_alpha_test_reg);
    registry.add("gx2", "GX2SetAlphaTestReg", gx2_set_alpha_test_reg);
    registry.add("gx2", "GX2SetAlphaToMask", gx2_set_alpha_to_mask);
    registry.add("gx2", "GX2InitAlphaToMaskReg", gx2_init_alpha_to_mask_reg);
    registry.add("gx2", "GX2SetAlphaToMaskReg", gx2_set_alpha_to_mask_reg);
    registry.add("gx2", "GX2SetAAMask", gx2_set_aa_mask);
    registry.add("gx2", "GX2InitAAMaskReg", gx2_init_aa_mask_reg);
    registry.add("gx2", "GX2SetAAMaskReg", gx2_set_aa_mask_reg);

    registry.add("gx2", "GX2SetStreamOutEnable", gx2_set_stream_out_enable);
    registry.add("gx2", "GX2SetMaxTessellationLevel", gx2_set_max_tessellation_level);
    registry.add("gx2", "GX2SetMinTessellationLevel", gx2_set_min_tessellation_level);
    // Eight registers decaf names and does not fill in; Cemu has no call.
    // Logged and counted until a title needs it.
    registry.add_stub("gx2", "GX2SetTessellation");
}

} // namespace wiiu::hle::gx2
