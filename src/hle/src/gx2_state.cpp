// Phase 8.1 — GX2's state-setting calls. See gx2_state.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_state.hpp>
#include <wiiu/hle/registry.hpp>
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

namespace {

template <typename T>
[[nodiscard]] bool read_guest(const memory::Memory& memory, ppcaddr_t address, T& out) {
    std::array<u8, sizeof(T)> bytes{};
    if (!memory.read_bytes(address, bytes)) {
        return false;
    }
    std::memcpy(&out, bytes.data(), sizeof(T));
    return true;
}

template <typename T>
[[nodiscard]] bool write_guest(memory::Memory& memory, ppcaddr_t address, const T& value) {
    std::array<u8, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    return memory.write_bytes(address, bytes);
}

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

/// Two packets, front then back, as decaf and Cemu both write it.
void set_stencil_mask(Call& call, const StencilMaskReg& reg) {
    set_word(call, latte::reg::db_stencilrefmask, reg.front.value());
    set_word(call, latte::reg::db_stencilrefmask_bf, reg.back.value());
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

/// The four scales and offsets in one packet, the clamp in another.
void set_polygon_offset(Call& call, const PolygonOffsetReg& reg) {
    const std::array<u32, 4> values{
        bits(reg.front_scale), bits(reg.front_offset), bits(reg.back_scale), bits(reg.back_offset)};
    static_cast<void>(
        emit_context_registers(call, latte::reg::pa_su_poly_offset_front_scale, values));
    set_word(call, latte::reg::pa_su_poly_offset_clamp, bits(reg.clamp));
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
}

} // namespace wiiu::hle::gx2
