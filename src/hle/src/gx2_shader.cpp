// Phase 8.1 — GX2's shader calls. See gx2_shader.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/render_state.hpp>

#include <array>

namespace wiiu::hle::gx2 {

namespace {

using latte::pm4::Opcode;

// Register addresses: Linux's `r600d.h` has every one.
constexpr u32 sq_config = 0x8C00;
constexpr u32 sq_esgs_ring_base = 0x8C40;      ///< then its size, GSVS's base and size
constexpr u32 sq_esgs_ring_itemsize = 0x288A8; ///< then SQ_GSVS_RING_ITEMSIZE
constexpr u32 vgt_gs_mode = 0x28A40;

// Field positions, from the same header.
constexpr u32 dx9_consts = 1U << 2;
constexpr u32 alu_inst_prefer_vector = 1U << 3;

[[nodiscard]] constexpr u32 priorities(u32 ps, u32 vs, u32 gs, u32 es) noexcept {
    return (ps << 24) | (vs << 26) | (gs << 28) | (es << 30);
}

[[nodiscard]] constexpr u32 pair(u32 low, u32 high, u32 mask) noexcept {
    return (low & mask) | ((high & mask) << 16);
}

constexpr u32 gpr_mask = 0xFF;
constexpr u32 stack_mask = 0xFFF;
constexpr u32 four_clause_temps = 4U << 28;

[[nodiscard]] constexpr u32 threads(u32 ps, u32 vs, u32 gs, u32 es) noexcept {
    return ps | (vs << 8) | (gs << 16) | (es << 24);
}

/// VGT_GS_MODE for compute: scenario G (3), compute mode in bits 14 and 15
/// -- decaf's COMPUTE_MODE and FAST_COMPUTE_MODE, Cemu's two-bit field set
/// to 3 -- and PARTIAL_THD_AT_EOI (bit 17).
constexpr u32 gs_mode_compute = 3U | (3U << 14) | (1U << 17);

/// `void GX2SetShaderModeEx(GX2ShaderMode mode, uint32_t numVsGpr, uint32_t
/// numVsStackEntries, uint32_t numGsGpr, uint32_t numGsStackEntries,
/// uint32_t numPsGpr, uint32_t numPsStackEntries)`.
void gx2_set_shader_mode_ex(Call& call,
                            u32 mode,
                            u32 vs_gprs,
                            u32 vs_stack,
                            u32 gs_gprs,
                            u32 gs_stack,
                            u32 ps_gprs,
                            u32 ps_stack) {
    if (mode > static_cast<u32>(ShaderMode::ComputeShader)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetShaderModeEx({}): no such shader mode", mode);
        return;
    }
    std::vector<u32> packets;
    append_shader_mode(packets,
                       static_cast<ShaderMode>(mode),
                       ShaderResources{.vs_gprs = vs_gprs,
                                       .vs_stack = vs_stack,
                                       .gs_gprs = gs_gprs,
                                       .gs_stack = gs_stack,
                                       .ps_gprs = ps_gprs,
                                       .ps_stack = ps_stack});
    static_cast<void>(emit(call, packets));
}

} // namespace

ShaderModeRegisters shader_mode_registers(ShaderMode mode,
                                          const ShaderResources& resources) noexcept {
    ShaderModeRegisters regs;
    regs.sq_config = alu_inst_prefer_vector;
    regs.sq_config |=
        mode == ShaderMode::ComputeShader ? priorities(0, 1, 2, 3) : priorities(3, 2, 1, 0);
    if (mode == ShaderMode::UniformRegister) {
        regs.sq_config |= dx9_consts;
    }

    switch (mode) {
        case ShaderMode::GeometryShader:
            regs.sq_gpr_resource_mgmt_1 = pair(resources.ps_gprs, 64, gpr_mask) | four_clause_temps;
            regs.sq_gpr_resource_mgmt_2 = pair(resources.gs_gprs, resources.vs_gprs, gpr_mask);
            regs.sq_thread_resource_mgmt = threads(124, 32, 8, 28);
            regs.sq_stack_resource_mgmt_1 = resources.ps_stack & stack_mask;
            regs.sq_stack_resource_mgmt_2 =
                pair(resources.gs_stack, resources.vs_stack, stack_mask);
            break;
        case ShaderMode::ComputeShader:
            regs.sq_gpr_resource_mgmt_1 = four_clause_temps;
            regs.sq_gpr_resource_mgmt_2 = pair(0, 248, gpr_mask);
            regs.sq_thread_resource_mgmt = threads(1, 1, 1, 189);
            regs.sq_stack_resource_mgmt_2 = pair(0, 256, stack_mask);
            break;
        case ShaderMode::UniformRegister:
        case ShaderMode::UniformBlock:
            regs.sq_gpr_resource_mgmt_1 =
                pair(resources.ps_gprs, resources.vs_gprs, gpr_mask) | four_clause_temps;
            regs.sq_thread_resource_mgmt = threads(136, 48, 4, 4);
            regs.sq_stack_resource_mgmt_1 =
                pair(resources.ps_stack, resources.vs_stack, stack_mask);
            break;
    }
    return regs;
}

void append_shader_mode(std::vector<u32>& out, ShaderMode mode, const ShaderResources& resources) {
    if (mode != ShaderMode::GeometryShader) {
        append_context_register(
            out, vgt_gs_mode, mode == ShaderMode::ComputeShader ? gs_mode_compute : 0U);
    }
    const ShaderModeRegisters regs = shader_mode_registers(mode, resources);
    const std::array<u32, 6> values{regs.sq_config,
                                    regs.sq_gpr_resource_mgmt_1,
                                    regs.sq_gpr_resource_mgmt_2,
                                    regs.sq_thread_resource_mgmt,
                                    regs.sq_stack_resource_mgmt_1,
                                    regs.sq_stack_resource_mgmt_2};
    append_registers(out, Opcode::SetConfigReg, sq_config, values);

    if (mode == ShaderMode::ComputeShader) {
        const std::array<u32, 4> rings{0, 0xFFFFFF, 0, 0xFFFFFF};
        append_registers(out, Opcode::SetConfigReg, sq_esgs_ring_base, rings);
        const std::array<u32, 2> item_sizes{0, 1};
        append_registers(out, Opcode::SetContextReg, sq_esgs_ring_itemsize, item_sizes);
        append_context_register(out, latte::reg::vgt_strmout_en, 0);
    }
}

void register_gx2_shader(Registry& registry) {
    registry.add("gx2", "GX2SetShaderModeEx", gx2_set_shader_mode_ex);
}

} // namespace wiiu::hle::gx2
