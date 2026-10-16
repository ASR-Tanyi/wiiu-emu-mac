// Phase 8.1 — GX2's shader calls.
//
// The shader mode: how the shader core divides its registers, threads and
// stack between the stages, and whether uniforms come from registers or
// from uniform blocks. `GX2SetDefaultState` sets it, and a title changes it
// when it switches to geometry or compute shaders.
//
// And binding shaders. A title's shaders come compiled, in GX2's shader
// structures -- loaded from a GFD file, usually, which is the title's own
// code and not GX2's -- each holding the program's address and size and the
// register values the compiler worked out. Binding one is writing those
// registers: the program's start, size and resources, then the stage's own
// (the vertex shader's outputs and semantics, the pixel shader's inputs and
// exports, the geometry shader's copy shader). The sequences are decaf's,
// which read them out of gx2.rpl; Cemu writes a subset of each, and where
// the two put different values in the same register, the one taken is named.
//
// The fetch shader GX2 generates itself is gx2_fetch_shader.hpp's. Not yet:
// uniform blocks, samplers and textures, and loop variables, whose offsets
// decaf adds to the loop constants as bytes and Cemu does not write at all.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>

#include <array>
#include <cstddef>
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

// --- the shader structures ---------------------------------------------------

/// `GX2RBuffer`: where a GX2R-managed program is, when the structure's own
/// pointer is null.
struct RBuffer {
    u32be flags;
    u32be element_size;
    u32be element_count;
    u32be buffer;
};

static_assert(sizeof(RBuffer) == 0x10);

/// `GX2FetchShader`: the program that fetches vertex attributes, which
/// `GX2InitFetchShaderEx` writes and the vertex shader calls.
struct FetchShader {
    u32be type;
    u32be sq_pgm_resources_fs;
    u32be size;
    u32be data;
    u32be attrib_count;
    u32be divisor_count;
    std::array<u32be, 2> divisors; ///< VGT_INSTANCE_STEP_RATE_0 and _1
};

static_assert(sizeof(FetchShader) == 0x20);

/// `GX2VertexShader`: decaf's layout, field for field.
struct VertexShader {
    u32be sq_pgm_resources_vs;
    u32be vgt_primitiveid_en;
    u32be spi_vs_out_config;
    u32be num_spi_vs_out_id;
    std::array<u32be, 10> spi_vs_out_id;
    u32be pa_cl_vs_out_cntl;
    u32be sq_vtx_semantic_clear;
    u32be num_sq_vtx_semantic;
    std::array<u32be, 32> sq_vtx_semantic;
    u32be vgt_strmout_buffer_en;
    u32be vgt_vertex_reuse_block_cntl;
    u32be vgt_hos_reuse_depth;
    u32be size;
    u32be data;
    u32be mode;
    u32be uniform_block_count;
    u32be uniform_blocks;
    u32be uniform_var_count;
    u32be uniform_vars;
    u32be initial_value_count;
    u32be initial_values;
    u32be loop_var_count;
    u32be loop_vars;
    u32be sampler_var_count;
    u32be sampler_vars;
    u32be attrib_var_count;
    u32be attrib_vars;
    u32be ring_item_size;
    u32be has_stream_out;
    std::array<u32be, 4> stream_out_stride;
    RBuffer gx2r;
};

static_assert(offsetof(VertexShader, pa_cl_vs_out_cntl) == 0x38);
static_assert(offsetof(VertexShader, sq_vtx_semantic) == 0x44);
static_assert(offsetof(VertexShader, size) == 0xD0);
static_assert(offsetof(VertexShader, loop_var_count) == 0xF4);
static_assert(offsetof(VertexShader, ring_item_size) == 0x10C);
static_assert(offsetof(VertexShader, gx2r) == 0x124);
static_assert(sizeof(VertexShader) == 0x134);

/// `GX2PixelShader`.
struct PixelShader {
    u32be sq_pgm_resources_ps;
    u32be sq_pgm_exports_ps;
    u32be spi_ps_in_control_0;
    u32be spi_ps_in_control_1;
    u32be num_spi_ps_input_cntl;
    std::array<u32be, 32> spi_ps_input_cntls;
    u32be cb_shader_mask;
    u32be cb_shader_control;
    u32be db_shader_control;
    u32be spi_input_z;
    u32be size;
    u32be data;
    u32be mode;
    u32be uniform_block_count;
    u32be uniform_blocks;
    u32be uniform_var_count;
    u32be uniform_vars;
    u32be initial_value_count;
    u32be initial_values;
    u32be loop_var_count;
    u32be loop_vars;
    u32be sampler_var_count;
    u32be sampler_vars;
    RBuffer gx2r;
};

static_assert(offsetof(PixelShader, cb_shader_mask) == 0x94);
static_assert(offsetof(PixelShader, size) == 0xA4);
static_assert(offsetof(PixelShader, gx2r) == 0xD8);
static_assert(sizeof(PixelShader) == 0xE8);

/// `GX2GeometryShader`: the geometry program, and the copy shader that runs
/// as the VS stage after it.
struct GeometryShader {
    u32be sq_pgm_resources_gs;
    u32be vgt_gs_out_prim_type;
    u32be vgt_gs_mode;
    u32be pa_cl_vs_out_cntl;
    u32be sq_pgm_resources_vs;
    u32be sq_gs_vert_itemsize;
    u32be spi_vs_out_config;
    u32be num_spi_vs_out_id;
    std::array<u32be, 10> spi_vs_out_id;
    u32be vgt_strmout_buffer_en;
    u32be size;
    u32be data;
    u32be vertex_shader_size;
    u32be vertex_shader_data;
    u32be mode;
    u32be uniform_block_count;
    u32be uniform_blocks;
    u32be uniform_var_count;
    u32be uniform_vars;
    u32be initial_value_count;
    u32be initial_values;
    u32be loop_var_count;
    u32be loop_vars;
    u32be sampler_var_count;
    u32be sampler_vars;
    u32be ring_item_size;
    u32be has_stream_out;
    std::array<u32be, 4> stream_out_stride;
    RBuffer gx2r;
    RBuffer gx2r_vertex_shader;
};

static_assert(offsetof(GeometryShader, vgt_strmout_buffer_en) == 0x48);
static_assert(offsetof(GeometryShader, size) == 0x4C);
static_assert(offsetof(GeometryShader, ring_item_size) == 0x88);
static_assert(offsetof(GeometryShader, gx2r) == 0xA0);
static_assert(sizeof(GeometryShader) == 0xC0);

/// A program as the GPU sees it: physical address and size in bytes.
struct Program {
    physaddr_t address = 0;
    u32 size = 0;
};

/// What the program's two registers after its size hold. decaf and Cemu
/// both write 0x100000 into them for the vertex and pixel shaders; for the
/// fetch shader Cemu writes 0x10000 and for the geometry shader and its copy
/// shader decaf writes 0. Nothing documents them; 0x100000 everywhere.
inline constexpr u32 program_unnamed_value = 0x100000;

/// The packets each `GX2Set…Shader` writes, for a program already found.
/// Counts past the structure's arrays are held to them (Cemu's guard).
void append_fetch_shader(std::vector<u32>& out, const FetchShader& shader, physaddr_t program);
void append_vertex_shader(std::vector<u32>& out, const VertexShader& shader, Program program);
void append_pixel_shader(std::vector<u32>& out, const PixelShader& shader, Program program);
void append_geometry_shader(std::vector<u32>& out,
                            const GeometryShader& shader,
                            Program program,
                            Program copy_shader);

/// Register the shader calls. Called by `register_gx2`.
void register_gx2_shader(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
