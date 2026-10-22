// Phase 8.11 — see draw_state.hpp.

#include <wiiu/latte/draw.hpp>
#include <wiiu/latte/draw_state.hpp>
#include <wiiu/latte/render_state.hpp>
#include <wiiu/latte/viewport.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>

namespace wiiu::latte {

std::optional<gpu::PrimitiveType> primitive_type(u32 latte) noexcept {
    switch (static_cast<PrimitiveType>(latte)) {
        case PrimitiveType::PointList:
            return gpu::PrimitiveType::Points;
        case PrimitiveType::LineList:
            return gpu::PrimitiveType::Lines;
        case PrimitiveType::LineStrip:
            return gpu::PrimitiveType::LineStrip;
        case PrimitiveType::TriangleList:
            return gpu::PrimitiveType::Triangles;
        case PrimitiveType::TriangleStrip:
            return gpu::PrimitiveType::TriangleStrip;
        case PrimitiveType::QuadList:
            return gpu::PrimitiveType::Quads;
        case PrimitiveType::RectList:
            return gpu::PrimitiveType::Rects;
        default:
            return std::nullopt;
    }
}

gpu::Viewport viewport(const RegisterFile& registers) noexcept {
    const auto value = [&registers](u32 index) {
        return std::bit_cast<f32>(registers.read(reg::pa_cl_vport_xscale_0 + (index * 4)));
    };
    const f32 x_scale = value(0);
    const f32 x_offset = value(1);
    const f32 y_scale = value(2);
    const f32 y_offset = value(3);
    const f32 z_scale = value(4);
    const f32 z_offset = value(5);
    const bool half_z = PaClClipCntl{.raw = registers.read(reg::pa_cl_clip_cntl)}.dx_clip_space_def;

    gpu::Viewport out;
    out.x = x_offset - x_scale;
    out.width = x_scale * 2.0F;
    out.y = y_offset + y_scale;
    out.height = y_scale * -2.0F;
    out.near_z = half_z ? z_offset : z_offset - z_scale;
    out.far_z = z_offset + z_scale;
    return out;
}

std::optional<gpu::Rect> scissor(const RegisterFile& registers, u32 width, u32 height) noexcept {
    const PaScGenericScissorTl top_left{.raw = registers.read(reg::pa_sc_generic_scissor_tl)};
    const PaScGenericScissorBr bottom_right{.raw = registers.read(reg::pa_sc_generic_scissor_br)};
    const u32 left = std::min<u32>(top_left.x, width);
    const u32 top = std::min<u32>(top_left.y, height);
    const u32 right = std::min<u32>(bottom_right.x, width);
    const u32 bottom = std::min<u32>(bottom_right.y, height);
    if (right <= left || bottom <= top) {
        return std::nullopt;
    }
    return gpu::Rect{left, top, right - left, bottom - top};
}

namespace {

/// CB_COLOR_CONTROL's ROP3 for a plain copy: GX2_LOGIC_OP_COPY.
constexpr u32 rop3_copy = 0xCC;

/// D3D9's "both" factors (decaf's BOTH_SRC_ALPHA and BOTH_INV_SRC_ALPHA).
constexpr u32 both_source_alpha = 11;
constexpr u32 both_inverse_source_alpha = 12;

/// Latte's blend factors (CB_BLEND_FUNC), by number, as Metal's; the two
/// "both" factors are dealt with before this is asked.
[[nodiscard]] std::optional<gpu::BlendFactor> blend_factor(u32 latte) noexcept {
    using F = gpu::BlendFactor;
    constexpr std::array<std::optional<F>, 21> table{F::Zero,
                                                     F::One,
                                                     F::SourceColor,
                                                     F::OneMinusSourceColor,
                                                     F::SourceAlpha,
                                                     F::OneMinusSourceAlpha,
                                                     F::DestinationAlpha,
                                                     F::OneMinusDestinationAlpha,
                                                     F::DestinationColor,
                                                     F::OneMinusDestinationColor,
                                                     F::SourceAlphaSaturated,
                                                     std::nullopt, // BOTH_SRC_ALPHA
                                                     std::nullopt, // BOTH_INV_SRC_ALPHA
                                                     F::BlendColor,
                                                     F::OneMinusBlendColor,
                                                     F::Source1Color,
                                                     F::OneMinusSource1Color,
                                                     F::Source1Alpha,
                                                     F::OneMinusSource1Alpha,
                                                     F::BlendAlpha,
                                                     F::OneMinusBlendAlpha};
    return latte < table.size() ? table[latte] : std::nullopt;
}

/// CB_COMB_FUNC, as Metal's.
[[nodiscard]] std::optional<gpu::BlendOperation> blend_operation(u32 latte) noexcept {
    switch (latte) {
        case 0:
            return gpu::BlendOperation::Add; // DST_PLUS_SRC
        case 1:
            return gpu::BlendOperation::Subtract; // SRC_MINUS_DST
        case 2:
            return gpu::BlendOperation::Min;
        case 3:
            return gpu::BlendOperation::Max;
        case 4:
            return gpu::BlendOperation::ReverseSubtract; // DST_MINUS_SRC
        default:
            return std::nullopt;
    }
}

/// One half of a blend -- colour or alpha -- as Metal's: a source factor, a
/// destination factor and an operation, D3D9's "both" source factors
/// standing for both.
bool blend_half(u32 source,
                u32 destination,
                u32 combine,
                gpu::BlendFactor& source_out,
                gpu::BlendFactor& destination_out,
                gpu::BlendOperation& operation_out,
                std::string_view half,
                std::string& error) {
    const std::optional<gpu::BlendOperation> operation = blend_operation(combine);
    if (!operation.has_value()) {
        error = fmt::format("{} combine function {} is not one Latte has", half, combine);
        return false;
    }
    operation_out = *operation;
    if (source == both_source_alpha || source == both_inverse_source_alpha) {
        const bool inverse = source == both_inverse_source_alpha;
        source_out =
            inverse ? gpu::BlendFactor::OneMinusSourceAlpha : gpu::BlendFactor::SourceAlpha;
        destination_out =
            inverse ? gpu::BlendFactor::SourceAlpha : gpu::BlendFactor::OneMinusSourceAlpha;
        return true;
    }
    const std::optional<gpu::BlendFactor> from = blend_factor(source);
    const std::optional<gpu::BlendFactor> to = blend_factor(destination);
    if (!from.has_value() || !to.has_value()) {
        error = fmt::format("{} blend factors {} and {} are not both ones Metal has a source and "
                            "destination for",
                            half,
                            source,
                            destination);
        return false;
    }
    source_out = *from;
    destination_out = *to;
    return true;
}

} // namespace

std::optional<gpu::BlendState>
blend_state(const RegisterFile& registers, u32 target, std::string& error) {
    const CbColorControl control{.raw = registers.read(reg::cb_color_control)};
    if (control.rop3.value() != rop3_copy) {
        error = fmt::format("logic op (ROP3 {:#04x}) is not drawn yet", control.rop3.value());
        return std::nullopt;
    }
    gpu::BlendState out;
    switch (control.special_op.value()) {
        case cb_special_normal:
            out.write_mask = static_cast<u8>((registers.read(reg::cb_target_mask) >> (4 * target)) &
                                             gpu::color_write::all);
            break;
        case cb_special_disable:
            out.write_mask = 0;
            break;
        default:
            error =
                fmt::format("colour special op {} is not drawn yet", control.special_op.value());
            return std::nullopt;
    }

    if ((control.target_blend_enable.value() & (1U << target)) == 0) {
        return out;
    }
    const CbBlendControl blend{.raw = registers.read(reg::cb_blend0_control + (4 * target))};
    out.enabled = true;
    if (!blend_half(blend.color_srcblend.value(),
                    blend.color_destblend.value(),
                    blend.color_comb_fcn.value(),
                    out.source_rgb,
                    out.destination_rgb,
                    out.rgb,
                    "colour",
                    error)) {
        return std::nullopt;
    }
    if (!blend.separate_alpha_blend.value()) {
        out.source_alpha = out.source_rgb;
        out.destination_alpha = out.destination_rgb;
        out.alpha = out.rgb;
        return out;
    }
    if (!blend_half(blend.alpha_srcblend.value(),
                    blend.alpha_destblend.value(),
                    blend.alpha_comb_fcn.value(),
                    out.source_alpha,
                    out.destination_alpha,
                    out.alpha,
                    "alpha",
                    error)) {
        return std::nullopt;
    }
    return out;
}

gpu::ClearColor blend_color(const RegisterFile& registers) noexcept {
    const auto value = [&registers](u32 index) {
        return std::bit_cast<f32>(registers.read(reg::cb_blend_red + (index * 4)));
    };
    return gpu::ClearColor{value(0), value(1), value(2), value(3)};
}

std::optional<RasterState> raster_state(const RegisterFile& registers, std::string& error) {
    const PaSuScModeCntl mode{.raw = registers.read(reg::pa_su_sc_mode_cntl)};
    if (mode.poly_mode != 0) {
        error = "polygon mode (triangles drawn as points or lines) is not drawn yet";
        return std::nullopt;
    }
    RasterState out;
    out.front = mode.face == FrontFace::Clockwise ? gpu::Winding::Clockwise
                                                  : gpu::Winding::CounterClockwise;
    out.cull_all = mode.cull_front && mode.cull_back;
    if (mode.cull_front && !mode.cull_back) {
        out.cull = gpu::CullMode::Front;
    } else if (mode.cull_back && !mode.cull_front) {
        out.cull = gpu::CullMode::Back;
    }
    return out;
}

namespace {

// Latte's numbers are Metal's order (render_state.hpp, gpu/types.hpp).
static_assert(static_cast<u32>(CompareFunction::Always) ==
              static_cast<u32>(gpu::CompareFunction::Always));
static_assert(static_cast<u32>(CompareFunction::GreaterEqual) ==
              static_cast<u32>(gpu::CompareFunction::GreaterEqual));
static_assert(static_cast<u32>(StencilOp::DecrementWrap) ==
              static_cast<u32>(gpu::StencilOperation::DecrementWrap));
static_assert(static_cast<u32>(StencilOp::Invert) ==
              static_cast<u32>(gpu::StencilOperation::Invert));

[[nodiscard]] gpu::CompareFunction compare(CompareFunction latte) noexcept {
    return static_cast<gpu::CompareFunction>(static_cast<u32>(latte));
}

[[nodiscard]] gpu::StencilOperation stencil_operation(StencilOp latte) noexcept {
    return static_cast<gpu::StencilOperation>(static_cast<u32>(latte));
}

[[nodiscard]] gpu::StencilFace stencil_face(CompareFunction function,
                                            StencilOp fail,
                                            StencilOp depth_fail,
                                            StencilOp pass,
                                            DbStencilRefMask masks) noexcept {
    return gpu::StencilFace{.compare = compare(function),
                            .stencil_fail = stencil_operation(fail),
                            .depth_fail = stencil_operation(depth_fail),
                            .pass = stencil_operation(pass),
                            .read_mask = static_cast<u8>(masks.mask.value()),
                            .write_mask = static_cast<u8>(masks.write_mask.value())};
}

} // namespace

DepthStencilState depth_stencil_state(const RegisterFile& registers) noexcept {
    const DbDepthControl control{.raw = registers.read(reg::db_depth_control)};
    DepthStencilState out;
    if (control.z_enable) {
        out.desc.depth_compare = compare(control.z_func);
        out.desc.depth_write = control.z_write_enable;
    }
    if (!control.stencil_enable) {
        return out;
    }
    const DbStencilRefMask front{.raw = registers.read(reg::db_stencilrefmask)};
    out.desc.stencil = true;
    out.desc.front = stencil_face(control.stencil_func,
                                  control.stencil_fail,
                                  control.stencil_zfail,
                                  control.stencil_zpass,
                                  front);
    out.front_reference = static_cast<u8>(front.ref.value());
    if (control.backface_enable) {
        const DbStencilRefMask back{.raw = registers.read(reg::db_stencilrefmask_bf)};
        out.desc.back = stencil_face(control.stencil_func_bf,
                                     control.stencil_fail_bf,
                                     control.stencil_zfail_bf,
                                     control.stencil_zpass_bf,
                                     back);
        out.back_reference = static_cast<u8>(back.ref.value());
    } else {
        out.desc.back = out.desc.front;
        out.back_reference = out.front_reference;
    }
    return out;
}

} // namespace wiiu::latte
