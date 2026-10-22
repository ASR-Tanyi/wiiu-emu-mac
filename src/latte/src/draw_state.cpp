// Phase 8.11 — see draw_state.hpp.

#include <wiiu/latte/draw.hpp>
#include <wiiu/latte/draw_state.hpp>
#include <wiiu/latte/render_state.hpp>
#include <wiiu/latte/viewport.hpp>

#include <algorithm>
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

} // namespace wiiu::latte
