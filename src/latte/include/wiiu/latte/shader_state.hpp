// Latte's shader registers: where each stage's program is, its resources,
// and what it reads and writes -- vertex semantics, the vertex shader's
// outputs, the pixel shader's inputs and exports.
//
// GX2 writes them from the shader structures a title loads (8.1); the
// translator (8.6, 8.11) reads them to find the programs and wire the
// stages together. Addresses are Linux's: `r600d.h` has the program starts
// and the CF offsets, CB_SHADER_MASK, VGT_STRMOUT_BUFFER_EN and
// SQ_GS_VERT_ITEMSIZE, and its R600 register list (`reg_srcs/r600`) the
// rest, CB_SHADER_CONTROL from the R700 list.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>

namespace wiiu::latte {

/// SQ_PGM_RESOURCES_{PS,VS,GS,ES,FS}: the registers and stack a program
/// needs. decaf's and Cemu's tables agree.
union SqPgmResources {
    u32 raw;
    BitField<0, 8, u32> num_gprs;
    BitField<8, 8, u32> stack_size;
    BitField<21, 1, bool> dx10_clamp;
    BitField<24, 3, u32> fetch_cache_lines;
    BitField<28, 1, bool> uncached_first_inst;
};

namespace reg {

/// A program is five registers from its start: the address >> 8, the size
/// >> 3, two GX2 always writes the same values into, and the resources.
/// The pixel shader's sixth is SQ_PGM_EXPORTS_PS.
inline constexpr u32 sq_pgm_start_ps = 0x28840;
inline constexpr u32 sq_pgm_start_vs = 0x28858;
inline constexpr u32 sq_pgm_start_gs = 0x2886C;
inline constexpr u32 sq_pgm_start_es = 0x28880;
inline constexpr u32 sq_pgm_start_fs = 0x28894;
inline constexpr u32 sq_pgm_resources_ps = 0x28850;
inline constexpr u32 sq_pgm_exports_ps = 0x28854;
inline constexpr u32 sq_pgm_resources_vs = 0x28868;
inline constexpr u32 sq_pgm_resources_fs = 0x288A4;
inline constexpr u32 sq_pgm_cf_offset_vs = 0x288D0;

/// The vertex shader's ring item size in geometry mode (it runs as ES), the
/// geometry shader's vertex item size, and the copy shader's ring.
inline constexpr u32 sq_esgs_ring_itemsize = 0x288A8;
inline constexpr u32 sq_gsvs_ring_itemsize = 0x288AC;
inline constexpr u32 sq_gs_vert_itemsize = 0x288C8;

/// SQ_VTX_SEMANTIC_0 to _31: the semantic each vertex input is fetched for.
/// Writing SQ_VTX_SEMANTIC_CLEAR sets each one whose bit is set to all
/// ones -- unused -- which the command processor does (decaf's reading).
inline constexpr u32 sq_vtx_semantic_0 = 0x28380;
inline constexpr u32 sq_vtx_semantic_count = 32;
inline constexpr u32 sq_vtx_semantic_clear = 0x288E0;

inline constexpr u32 vgt_primitiveid_en = 0x28A84;
inline constexpr u32 vgt_instance_step_rate_0 = 0x28AA0; ///< and _1 after it
inline constexpr u32 vgt_gs_out_prim_type = 0x28A6C;
inline constexpr u32 vgt_strmout_vtx_stride_0 = 0x28AD4; ///< _1 to _3 every 16 bytes
inline constexpr u32 vgt_strmout_buffer_en = 0x28B20;
inline constexpr u32 vgt_vertex_reuse_block_cntl = 0x28C58;
inline constexpr u32 vgt_hos_reuse_depth = 0x28A20;

/// The vertex shader's outputs: how many and which, and the clip and
/// cull distances it writes.
inline constexpr u32 spi_vs_out_config = 0x286C4;
inline constexpr u32 spi_vs_out_id_0 = 0x28614; ///< ten of them
inline constexpr u32 pa_cl_vs_out_cntl = 0x2881C;

/// The pixel shader's inputs, and what it exports.
inline constexpr u32 spi_ps_in_control_0 = 0x286CC; ///< and _1 after it
inline constexpr u32 spi_ps_input_cntl_0 = 0x28644; ///< thirty-two of them
inline constexpr u32 spi_input_z = 0x286D8;
inline constexpr u32 cb_shader_mask = 0x2823C;
inline constexpr u32 cb_shader_control = 0x287A0;
inline constexpr u32 db_shader_control = 0x2880C;

/// The uniform registers, four words each: the pixel shader's 256 first,
/// then the vertex shader's.
inline constexpr u32 sq_alu_constant0_0 = 0x30000;
inline constexpr u32 sq_alu_constant0_256 = 0x31000;

} // namespace reg

/// DB_SHADER_CONTROL's DUAL_EXPORT_ENABLE, which GX2 sets on every pixel
/// shader it binds (decaf).
inline constexpr u32 db_shader_dual_export_enable = 1U << 9;

} // namespace wiiu::latte
