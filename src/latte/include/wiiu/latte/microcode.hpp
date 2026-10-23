// Latte's shader microcode: the R700 instruction words a program is made of.
//
// GX2 writes one kind of program itself -- the fetch shader (8.1) -- and the
// decompiler (8.6) will read every kind, so the words are named here, as
// unions over the raw word like the registers. A program is 32-bit words,
// little-endian in memory, which is how the GPU reads them: the control-flow
// program first, two words an instruction, then the clauses it points at,
// four words a fetch.
//
// So far: the three forms of a control-flow instruction -- the common one,
// the ALU clause's and the export's -- a vertex fetch, and a texture fetch.
// The field positions are decaf's and Cemu's, which agree on every one of
// them but a texture fetch's LOD bias, which Cemu reads from word 2 on top
// of the source selects and decaf from word 1's free bits 21 to 27; decaf's
// is taken. The instruction numbers are decaf's, which follow AMD's R700
// ISA, and Cemu's agree on every one it has. A word's top two CF_INST bits are its form:
// 0 the common one, 1 an export, 2 or 3 an ALU clause, whose CF_INST is the
// word's top four bits rather than seven.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>

namespace wiiu::latte {

/// Which form a control-flow instruction is: CF_INST's top two bits.
enum class CfInstType : u32 {
    Normal = 0,
    Export = 1,
    Alu = 2,
    AluExtended = 3, ///< an ALU clause whose CF_INST has bit 2 set
};

/// CF_INST, for the common form.
enum class CfInst : u32 {
    Nop = 0x00,
    Tex = 0x01,
    Vtx = 0x02,
    VtxTc = 0x03, ///< a vertex fetch clause, through the texture cache
    LoopStart = 0x04,
    LoopEnd = 0x05,
    LoopStartDx10 = 0x06,
    LoopStartNoAl = 0x07,
    LoopContinue = 0x08,
    LoopBreak = 0x09,
    Jump = 0x0A,
    Push = 0x0B,
    PushElse = 0x0C,
    Else = 0x0D,
    Pop = 0x0E,
    PopJump = 0x0F,
    PopPush = 0x10,
    PopPushElse = 0x11,
    Call = 0x12,
    CallFs = 0x13, ///< the vertex shader calling its fetch shader
    Return = 0x14,
    EmitVertex = 0x15,
    EmitCutVertex = 0x16,
    CutVertex = 0x17,
    Kill = 0x18,
    EndProgram = 0x19,
    WaitAck = 0x1A,
    TexAck = 0x1B,
    VtxAck = 0x1C,
    VtxTcAck = 0x1D, ///< the last R700 has; 0x1E on are Evergreen's
};

/// CF_INST, for an export. The memory exports' TYPE is a `MemExportType`.
enum class ExportInst : u32 {
    MemStream0 = 0x20,
    MemStream1 = 0x21,
    MemStream2 = 0x22,
    MemStream3 = 0x23,
    MemScratch = 0x24,
    MemReduction = 0x25,
    MemRing = 0x26,
    Export = 0x27,     ///< EXP
    ExportDone = 0x28, ///< EXP_DONE: the last export of its type
    MemExport = 0x3A,
};

/// CF_INST, for an ALU clause: four bits, at the top of the word.
enum class AluCfInst : u32 {
    Alu = 0x8,
    AluPushBefore = 0x9,
    AluPopAfter = 0xA,
    AluPop2After = 0xB,
    AluExtended = 0xC, ///< reserved on R700; Evergreen's ALU_EXT
    AluContinue = 0xD,
    AluBreak = 0xE,
    AluElseAfter = 0xF,
};

/// COND: when a control-flow instruction runs. CF_BOOL and CF_NOT_BOOL
/// test the bool constant CF_CONST names.
enum class CfCond : u32 {
    Active = 0,
    AlwaysFalse = 1,
    Bool = 2,
    NotBool = 3,
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
    BitField<28, 2, CfInstType> cf_inst_type;
    BitField<30, 1, bool> whole_quad_mode;
    BitField<31, 1, bool> barrier;
};

/// KCACHE_MODE: how many sets of sixteen uniforms an ALU clause's constant
/// cache locks from a uniform block -- none, one or two -- or two counted
/// from the loop index.
enum class KcacheMode : u32 {
    Nop = 0,
    Lock1 = 1,
    Lock2 = 2,
    LockLoopIndex = 3,
};

/// An ALU clause's first word: where it starts, in 64-bit words, and the
/// uniform blocks (banks) its two constant caches read.
union CfAluWord0 {
    u32 raw;
    BitField<0, 22, u32> addr;
    BitField<22, 4, u32> kcache_bank0;
    BitField<26, 4, u32> kcache_bank1;
    BitField<30, 2, KcacheMode> kcache_mode0;
};

/// The second: the caches' first sets of sixteen, and the clause's length
/// in 64-bit slots -- instructions and literals both -- less one.
union CfAluWord1 {
    u32 raw;
    BitField<0, 2, KcacheMode> kcache_mode1;
    BitField<2, 8, u32> kcache_addr0;
    BitField<10, 8, u32> kcache_addr1;
    BitField<18, 7, u32> count;
    BitField<25, 1, bool> alt_const;
    BitField<26, 4, AluCfInst> cf_inst;
    BitField<30, 1, bool> whole_quad_mode;
    BitField<31, 1, bool> barrier;
};

/// An export's TYPE, for EXP and EXP_DONE.
enum class ExportType : u32 {
    Pixel = 0,
    Position = 1,
    Parameter = 2,
};

/// And for the memory exports.
enum class MemExportType : u32 {
    Write = 0,
    WriteIndexed = 1,
    Read = 2,
    ReadIndexed = 3,
};

/// The array base a position export counts from: POS0 is 60.
inline constexpr u32 export_position_base = 60;

/// An export's first word: where it goes, and which register it sends.
union CfExportWord0 {
    u32 raw;
    BitField<0, 13, u32> array_base;
    BitField<13, 2, u32> type; ///< an `ExportType`, or a `MemExportType`
    BitField<15, 7, u32> rw_gpr;
    BitField<22, 1, bool> rw_rel;
    BitField<23, 7, u32> index_gpr;
    BitField<30, 2, u32> elem_size;
};

/// The second: BURST_COUNT + 1 registers, from RW_GPR on, to as many array
/// entries from ARRAY_BASE on.
union CfExportWord1 {
    u32 raw;
    BitField<17, 4, u32> burst_count;
    BitField<21, 1, bool> end_of_program;
    BitField<22, 1, bool> valid_pixel_mode;
    BitField<23, 7, ExportInst> cf_inst;
    BitField<30, 1, bool> whole_quad_mode;
    BitField<31, 1, bool> barrier;
};

/// Its low half, for EXP and EXP_DONE: which component goes where.
union CfExportWord1Swizzle {
    u32 raw;
    BitField<0, 3, u32> sel_x; ///< a `Sel`
    BitField<3, 3, u32> sel_y;
    BitField<6, 3, u32> sel_z;
    BitField<9, 3, u32> sel_w;
};

/// And for the memory exports: how many elements, and which components.
union CfExportWord1Buffer {
    u32 raw;
    BitField<0, 12, u32> array_size;
    BitField<12, 4, u32> comp_mask;
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

/// TEX_INST: what a texture fetch does, by decaf's names for R700's
/// numbers. 0 and 1 are the vertex fetches, which a texture clause may also
/// hold, in a vertex fetch's words.
enum class TexInst : u32 {
    VtxFetch = 0x00,
    VtxSemantic = 0x01,
    Mem = 0x02,
    Ld = 0x03,
    GetTextureInfo = 0x04,
    GetSampleInfo = 0x05,
    GetCompTexLod = 0x06,
    GetGradientsH = 0x07,
    GetGradientsV = 0x08,
    GetLerp = 0x09,
    KeepGradients = 0x0A,
    SetGradientsH = 0x0B,
    SetGradientsV = 0x0C,
    Pass = 0x0D,
    SetCubemapIndex = 0x0E,
    Fetch4 = 0x0F,
    Sample = 0x10,
    SampleL = 0x11,
    SampleLb = 0x12,
    SampleLz = 0x13,
    SampleG = 0x14,
    SampleGL = 0x15,
    SampleGLb = 0x16,
    SampleGLz = 0x17,
    SampleC = 0x18,
    SampleCL = 0x19,
    SampleCLb = 0x1A,
    SampleCLz = 0x1B,
    SampleCG = 0x1C,
    SampleCGL = 0x1D,
    SampleCGLb = 0x1E,
    SampleCGLz = 0x1F,
};

/// A texture fetch's first word: what it does, from which resource, with
/// which register's coordinates. The resource is counted from the stage's
/// first, as a vertex fetch's buffer is.
union TexWord0 {
    u32 raw;
    BitField<0, 5, TexInst> tex_inst;
    BitField<5, 1, bool> bc_frac_mode;
    BitField<7, 1, bool> fetch_whole_quad;
    BitField<8, 8, u32> resource_id;
    BitField<16, 7, u32> src_gpr;
    BitField<23, 1, bool> src_rel;
    BitField<24, 1, bool> alt_const;
};

/// The second: where the result goes, the LOD bias (signed, three
/// fractional bits), and whether each coordinate is normalised (set) or in
/// texels.
union TexWord1 {
    u32 raw;
    BitField<0, 7, u32> dst_gpr;
    BitField<7, 1, bool> dst_rel;
    BitField<9, 3, Sel> dst_sel_x;
    BitField<12, 3, Sel> dst_sel_y;
    BitField<15, 3, Sel> dst_sel_z;
    BitField<18, 3, Sel> dst_sel_w;
    BitField<21, 7, u32> lod_bias;
    BitField<28, 1, bool> coord_type_x;
    BitField<29, 1, bool> coord_type_y;
    BitField<30, 1, bool> coord_type_z;
    BitField<31, 1, bool> coord_type_w;
};

/// The third: the texel offsets (signed, one fractional bit), the sampler,
/// and which of the source register's components are the coordinates. The
/// fourth word is padding.
union TexWord2 {
    u32 raw;
    BitField<0, 5, u32> offset_x;
    BitField<5, 5, u32> offset_y;
    BitField<10, 5, u32> offset_z;
    BitField<15, 5, u32> sampler_id;
    BitField<20, 3, Sel> src_sel_x;
    BitField<23, 3, Sel> src_sel_y;
    BitField<26, 3, Sel> src_sel_z;
    BitField<29, 3, Sel> src_sel_w;
};

/// Bytes in a control-flow instruction and in a vertex or texture fetch.
inline constexpr u32 cf_instruction_bytes = 8;
inline constexpr u32 vtx_instruction_bytes = 16;
inline constexpr u32 tex_instruction_bytes = 16;

/// The most instructions one fetch clause holds.
inline constexpr u32 max_fetches_per_clause = 16;

} // namespace wiiu::latte
