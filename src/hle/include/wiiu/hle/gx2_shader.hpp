// Phase 8.1 — GX2's shader calls.
//
// So far only the shader mode: how the shader core divides its registers,
// threads and stack between the stages, and whether uniforms come from
// registers or from uniform blocks. `GX2SetDefaultState` sets it, and a
// title changes it when it switches to geometry or compute shaders. The
// shaders themselves, and binding them, come next.
#pragma once

#include <wiiu/common/types.hpp>

#include <vector>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// `GX2ShaderMode`.
enum class ShaderMode : u32 {
    UniformRegister = 0,
    UniformBlock = 1,
    GeometryShader = 2,
    ComputeShader = 3,
};

/// `GX2SetShaderModeEx`'s resource split: general-purpose registers and
/// stack entries for each stage.
struct ShaderResources {
    u32 vs_gprs = 0;
    u32 vs_stack = 0;
    u32 gs_gprs = 0;
    u32 gs_stack = 0;
    u32 ps_gprs = 0;
    u32 ps_stack = 0;
};

/// SQ_CONFIG to SQ_STACK_RESOURCE_MGMT_2, which `GX2SetShaderModeEx` writes
/// as one SET_CONFIG_REG, in register order.
struct ShaderModeRegisters {
    u32 sq_config = 0;
    u32 sq_gpr_resource_mgmt_1 = 0;
    u32 sq_gpr_resource_mgmt_2 = 0;
    u32 sq_thread_resource_mgmt = 0;
    u32 sq_stack_resource_mgmt_1 = 0;
    u32 sq_stack_resource_mgmt_2 = 0;
};

/// The registers for a mode. decaf and Cemu agree on all of it -- the
/// priorities, the fixed thread counts, which stage gets which share --
/// except that Cemu also asks for four clause-temporary registers in every
/// mode, as decaf does only for geometry and compute; Cemu's is taken. In
/// geometry mode the vertex shader runs as the ES stage and the VS stage
/// runs the copy shader, so the vertex shader's share goes to the ES fields
/// and the VS gets a fixed 64 registers. Field positions are Linux's
/// `r600d.h`'s.
[[nodiscard]] ShaderModeRegisters shader_mode_registers(ShaderMode mode,
                                                        const ShaderResources& resources) noexcept;

/// The packets `GX2SetShaderModeEx` writes: VGT_GS_MODE (off, or compute's
/// scenario G) unless the mode is geometry -- `GX2SetGeometryShader` sets it
/// then -- the SQ registers, and in compute mode the ring and stream-out
/// registers decaf and Cemu both clear. Cemu also writes SPI_THREAD_GROUPING
/// for titles built with SDK 2.11.4 or later, which needs the title's SDK
/// version; GX2's own register initialisation writes it anyway. The shader
/// cache invalidation both write outside uniform-register mode waits for
/// `GX2Invalidate`.
void append_shader_mode(std::vector<u32>& out, ShaderMode mode, const ShaderResources& resources);

/// Register the shader calls. Called by `register_gx2`.
void register_gx2_shader(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
