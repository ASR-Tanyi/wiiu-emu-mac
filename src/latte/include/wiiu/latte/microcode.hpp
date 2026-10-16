// Latte's shader microcode: the R700 instruction words a program is made of.
//
// GX2 writes one kind of program itself -- the fetch shader (8.1) -- and the
// decompiler (8.6) will read every kind, so the words are named here, as
// unions over the raw word like the registers. A program is 32-bit words,
// little-endian in memory, which is how the GPU reads them: the control-flow
// program first, two words an instruction, then the clauses it points at,
// four words a fetch.
//
// Only what the fetch shader needs is here so far: the common form of a
// control-flow instruction, and a vertex fetch. The field positions are
// decaf's and Cemu's, which agree on every one of them.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>

namespace wiiu::latte {

/// CF_INST: what a control-flow instruction does. The ones GX2 writes.
enum class CfInst : u32 {
    Nop = 0x00,
    Vtx = 0x02,
    VtxTc = 0x03,  ///< a vertex fetch clause, through the texture cache
    CallFs = 0x13, ///< the vertex shader calling its fetch shader
    Return = 0x14,
};

/// A control-flow instruction's first word: where its clause starts, in
/// 64-bit words from the start of the program.
union CfWord0 {
    u32 raw;
    BitField<0, 32, u32> addr;
};

/// The second word. A clause holds `count` + 1 instructions, `count_3`
/// being the fourth bit of the count.
union CfWord1 {
    u32 raw;
    BitField<0, 3, u32> pop_count;
    BitField<3, 5, u32> cf_const;
    BitField<8, 2, u32> cond;
    BitField<10, 3, u32> count;
    BitField<13, 6, u32> call_count;
    BitField<19, 1, u32> count_3;
    BitField<21, 1, bool> end_of_program;
    BitField<22, 1, bool> valid_pixel_mode;
    BitField<23, 7, CfInst> cf_inst;
    BitField<30, 1, bool> whole_quad_mode;
    BitField<31, 1, bool> barrier;
};

/// VTX_INST: a plain fetch into a register, or a semantic fetch, whose
/// destination is the register the vertex shader's SQ_VTX_SEMANTIC_n maps
/// semantic n to.
enum class VtxInst : u32 {
    Fetch = 0,
    Semantic = 1,
};

/// Which index a vertex fetch reads at.
enum class VtxFetchType : u32 {
    VertexData = 0,
    InstanceData = 1,
    NoIndexOffset = 2,
};

/// A component select: a source channel, or a constant.
enum class Sel : u32 {
    X = 0,
    Y = 1,
    Z = 2,
    W = 3,
    Zero = 4,
    One = 5,
    Mask = 7, ///< not written
};

/// How a fetched value's bits become a number.
enum class NumFormat : u32 {
    Norm = 0,   ///< normalised to [0, 1], or [-1, 1] signed
    Int = 1,    ///< an integer
    Scaled = 2, ///< an integer, converted to float
};

/// The byte swap a fetch applies to what it reads.
enum class Endian : u32 {
    None = 0,
    Swap8In16 = 1,
    Swap8In32 = 2,
};

/// A vertex fetch's first word: the resource, and the index it reads at.
/// The resource is counted from the stage's first: the vertex shader's
/// attribute buffers are its 0xA0 onwards, 0x140 onwards overall (decaf).
/// For an instance fetch, `src_sel_x` picks the instance index's divisor:
/// W is the instance index itself, Y and Z divide it by
/// VGT_INSTANCE_STEP_RATE_0 and _1.
union VtxWord0 {
    u32 raw;
    BitField<0, 5, VtxInst> vtx_inst;
    BitField<5, 2, VtxFetchType> fetch_type;
    BitField<7, 1, bool> fetch_whole_quad;
    BitField<8, 8, u32> buffer_id;
    BitField<16, 7, u32> src_gpr;
    BitField<23, 1, bool> src_rel;
    BitField<24, 2, Sel> src_sel_x;
    BitField<26, 6, u32> mega_fetch_count; ///< bytes fetched, less one
};

/// The second: where the value goes and what it is. A semantic fetch puts
/// its semantic in the low byte where a plain fetch has its register.
union VtxWord1 {
    u32 raw;
    BitField<0, 8, u32> semantic_id;
    BitField<0, 7, u32> dst_gpr;
    BitField<7, 1, bool> dst_rel;
    BitField<9, 3, Sel> dst_sel_x;
    BitField<12, 3, Sel> dst_sel_y;
    BitField<15, 3, Sel> dst_sel_z;
    BitField<18, 3, Sel> dst_sel_w;
    BitField<21, 1, bool> use_const_fields;
    BitField<22, 6, u32> data_format; ///< an R600 FMT_* number
    BitField<28, 2, NumFormat> num_format_all;
    BitField<30, 1, bool> format_comp_signed;
    BitField<31, 1, bool> srf_mode_all;
};

/// The third: where in the element, and the byte swap. The fourth word is
/// padding.
union VtxWord2 {
    u32 raw;
    BitField<0, 16, u32> offset;
    BitField<16, 2, Endian> endian_swap;
    BitField<18, 1, bool> const_buf_no_stride;
    BitField<19, 1, bool> mega_fetch;
    BitField<20, 1, bool> alt_const;
};

/// Bytes in a control-flow instruction and in a vertex fetch.
inline constexpr u32 cf_instruction_bytes = 8;
inline constexpr u32 vtx_instruction_bytes = 16;

/// The most instructions one fetch clause holds.
inline constexpr u32 max_fetches_per_clause = 16;

} // namespace wiiu::latte
