// Phase 8.1 — GX2's state-setting calls. See gx2_state.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_state.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/viewport.hpp>

#include <algorithm>
#include <bit>
#include <cstring>

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

} // namespace

void register_gx2_state(Registry& registry) {
    registry.add("gx2", "GX2InitViewportReg", gx2_init_viewport_reg);
    registry.add("gx2", "GX2SetViewportReg", gx2_set_viewport_reg);
    registry.add("gx2", "GX2SetViewport", gx2_set_viewport);
    registry.add("gx2", "GX2InitScissorReg", gx2_init_scissor_reg);
    registry.add("gx2", "GX2SetScissorReg", gx2_set_scissor_reg);
    registry.add("gx2", "GX2SetScissor", gx2_set_scissor);
}

} // namespace wiiu::hle::gx2
