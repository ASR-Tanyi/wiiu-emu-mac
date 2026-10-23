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
inline constexpr u32 spi_ps_in_control_0 = 0x286CC;
inline constexpr u32 spi_ps_in_control_1 = 0x286D0;
inline constexpr u32 spi_ps_input_cntl_0 = 0x28644; ///< thirty-two of them
inline constexpr u32 spi_input_z = 0x286D8;
inline constexpr u32 cb_shader_mask = 0x2823C;
inline constexpr u32 cb_shader_control = 0x287A0;
inline constexpr u32 db_shader_control = 0x2880C;

/// The uniform registers, four words each: the pixel shader's 256 first,
/// then the vertex shader's.
inline constexpr u32 sq_alu_constant0_0 = 0x30000;
inline constexpr u32 sq_alu_constant0_256 = 0x31000;

/// SQ_CONFIG: among other things, whether shaders read their uniforms from
/// the uniform registers or from uniform blocks (Linux's `r600d.h`).
inline constexpr u32 sq_config = 0x8C00;

} // namespace reg

/// SQ_CONFIG's DX9_CONSTS: set, the ALU reads the uniform registers (the
/// constant file, sources 256 to 511); clear, uniform blocks, through
/// KCACHE. GX2's shader mode decides it (decaf).
inline constexpr u32 sq_config_dx9_consts = 1U << 2;

/// The uniform registers a stage has, of four words each.
inline constexpr u32 uniform_registers_per_stage = 256;

/// DB_SHADER_CONTROL's DUAL_EXPORT_ENABLE, which GX2 sets on every pixel
/// shader it binds (decaf).
inline constexpr u32 db_shader_dual_export_enable = 1U << 9;

/// A semantic that names nothing: what SQ_VTX_SEMANTIC_CLEAR leaves in a
/// vertex semantic, and what SPI_VS_OUT_ID holds for an unused parameter.
inline constexpr u32 semantic_unused = 0xFF;

/// SPI_VS_OUT_ID_0 to _9 hold a byte each for the vertex shader's forty
/// parameter exports: the semantic the pixel shader matches it by.
inline constexpr u32 spi_vs_out_id_count = 10;
inline constexpr u32 vs_parameter_count = spi_vs_out_id_count * 4;

/// SPI_PS_IN_CONTROL_0: how many inputs the pixel shader has, and whether
/// one of them is the pixel's position. Linux's `r600d.h` has every field.
union SpiPsInControl0 {
    u32 raw;
    BitField<0, 6, u32> num_interp;
    BitField<8, 1, bool> position_ena;
    BitField<9, 1, bool> position_centroid;
    BitField<10, 5, u32> position_addr;
    BitField<15, 4, u32> param_gen; ///< point-sprite coordinates
    BitField<19, 7, u32> param_gen_addr;
    BitField<26, 2, u32> baryc_sample_cntl;
    BitField<28, 1, bool> persp_gradient_ena;
    BitField<29, 1, bool> linear_gradient_ena;
    BitField<30, 1, bool> position_sample;
    BitField<31, 1, bool> baryc_at_sample_ena;
};

/// SPI_PS_IN_CONTROL_1: the front-face bit and the rest. Linux's `r600d.h`.
union SpiPsInControl1 {
    u32 raw;
    BitField<0, 1, bool> gen_index_pix;
    BitField<1, 7, u32> gen_index_pix_addr;
    BitField<8, 1, bool> front_face_ena;
    BitField<9, 2, u32> front_face_chan;
    BitField<11, 1, bool> front_face_all_bits;
    BitField<12, 5, u32> front_face_addr;
    BitField<17, 7, u32> fog_addr;
    BitField<24, 1, bool> fixed_pt_position_ena;
    BitField<25, 5, u32> fixed_pt_position_addr;
};

/// SPI_PS_INPUT_CNTL_n: pixel shader input n -- the semantic it takes from
/// the vertex shader, the value it has when no vertex output has that
/// semantic, and how it is interpolated. decaf's and Cemu's fields, which
/// agree.
union SpiPsInputCntl {
    u32 raw;
    BitField<0, 8, u32> semantic;
    BitField<8, 2, u32> default_val; ///< 0000, 0001, 1110 or 1111, as XYZW
    BitField<10, 1, bool> flat_shade;
    BitField<11, 1, bool> sel_centroid;
    BitField<12, 1, bool> sel_linear; ///< without perspective correction
    BitField<13, 4, u32> cyl_wrap;
    BitField<17, 1, bool> pt_sprite_tex;
    BitField<18, 1, bool> sel_sample;
};

inline constexpr u32 spi_ps_input_cntl_count = 32;

} // namespace wiiu::latte
