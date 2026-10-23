// Phase 8.6 — the decompiler's intermediate form: a shader as the emitter
// will see it.
//
// The third stage. A decoded program (shader_program.hpp) says what each
// instruction is; what it *means* also depends on registers GX2 wrote when
// the shaders were bound -- which register a vertex attribute lands in
// (SQ_VTX_SEMANTIC), which semantic each of the vertex shader's parameter
// exports carries (SPI_VS_OUT_ID), where each pixel shader input comes from
// (SPI_PS_INPUT_CNTL), which colour target each pixel export is
// (CB_SHADER_MASK). This stage does those lookups once and writes down the
// result, so the emitter (MSL, next) reads one self-contained description.
//
// The model stays R700's: statements read and write numbered four-component
// registers, R0 to R127. What it holds so far is what decaf's example
// triangle needs and no more:
//
//   * a vertex shader's attributes, from its fetch shader's semantic
//     fetches, each with the register it is fetched into;
//   * a pixel shader's inputs, the registers R0 onwards hold on entry;
//   * statements: fetching the attributes (CALL_FS), sampling a texture
//     (SAMPLE, in a texture clause), an ALU instruction group (in an ALU
//     clause), and exporting the position, a parameter or a colour.
//
// An ALU group keeps the hardware's shape, because its meaning depends on
// it: every instruction in a group reads its sources before any of the
// group's results is written, and a source may name the previous group's
// results, unit by unit (PV.x to PV.w, and PS for the T unit), whether or
// not they were written to a register. Translated so far is float
// arithmetic -- MOV, ADD, MUL and MUL_IEEE, MAX and MIN, the MULADDs, DOT4
// and DOT4_IEEE -- from registers, literals, the inline constants, PV and PS,
// and uniforms, with each source's absolute value and negation and each
// result's
// output modifier and clamp. MUL, MULADD and DOT4 are DX9's multiply, where
// zero times anything, infinity and NaN included, is zero; the _IEEE forms
// are IEEE's.
//
// **Uniforms** come one of two ways, and SQ_CONFIG's DX9_CONSTS -- which
// GX2's shader mode sets -- says which (decaf's reading): set, a source of
// 256 to 511 is one of the stage's 256 uniform registers, C0 to C255;
// clear, a source of 128 to 191 is a constant of a uniform block, through
// the clause's two constant caches. Each cache locks one bank -- a uniform
// block -- and a run of sixteen constants (LOCK_1) or thirty-two (LOCK_2)
// from KCACHE_ADDR * 16, so KC0[n] is constant KCACHE_ADDR0 * 16 + n of
// bank KCACHE_BANK0 (decaf's and Cemu's arithmetic, which agree). A source
// of the other kind than SQ_CONFIG says, a cache the clause does not lock,
// an index past what it locks, and a lock from the loop index are refused.
//
// A sample is written down with what the texture resource bound to its unit
// says about it at the draw: its shape, and its component selects (WORD4's
// DST_SEL, GX2's component map), which the hardware applies to what the
// texture returns before the fetch's own selects pick the destination's
// components. So a shader is translated for the textures it is drawn with,
// as it is for the other registers here.
//
// Everything else is refused, saying which instruction and that it is not
// done yet, rather than translated into something that draws wrongly: in an
// ALU clause, every other instruction, relative registers and indexing
// (of uniforms too), predication, PV or PS where the
// previous group of the clause left nothing, a vertex shader reading R0
// before writing it (R0 holds the vertex and instance indices, not modelled
// yet), and ALU clauses that push, pop or break; loops, jumps and
// subroutines, memory exports, point size, clip
// distances and depth exports, relative registers, plain (non-semantic)
// fetches; in a texture clause, every fetch but SAMPLE, coordinates in
// texels, a LOD bias or offsets, and textures that are not 2D. Each later
// step takes something off that list.
//
// Two rules that come from the hardware and are easy to get wrong, both
// decaf's and Cemu's: a semantic fetch lands in R(n + 1), where
// SQ_VTX_SEMANTIC_n names its semantic (R0 holds the vertex and instance
// indices), and a fetch whose semantic no SQ_VTX_SEMANTIC names is skipped;
// and pixel export n goes to the n-th colour target CB_SHADER_MASK enables,
// not to target n.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/microcode.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_program.hpp>

#include <array>
#include <string>
#include <variant>
#include <vector>

namespace wiiu::latte::ir {

enum class Stage : u8 {
    Vertex,
    Pixel,
};

/// For each of a destination's components, where it comes from: a source
/// component, a constant, or nothing (not written).
using Swizzle = std::array<Sel, 4>;

/// Which index a vertex attribute is read at.
enum class AttributeIndex : u8 {
    Vertex,        ///< the vertex index
    Instance,      ///< the instance index
    InstanceRate0, ///< the instance index over VGT_INSTANCE_STEP_RATE_0
    InstanceRate1, ///< over VGT_INSTANCE_STEP_RATE_1
};

/// A vertex attribute: one of the fetch shader's semantic fetches.
struct VertexAttribute {
    u32 semantic = 0;
    u32 gpr = 0;         ///< the register it is fetched into
    u32 buffer = 0;      ///< attribute buffer, 0 to 15
    u32 offset = 0;      ///< bytes into the element
    u32 bytes = 0;       ///< what the fetch reads: MEGA_FETCH_COUNT + 1
    u32 data_format = 0; ///< an R600 FMT_* number
    NumFormat num_format = NumFormat::Norm;
    bool is_signed = false;
    Endian endian = Endian::None;
    AttributeIndex index = AttributeIndex::Vertex;
    Swizzle components{}; ///< the register's components, from the fetched value's
};

/// Where a pixel shader input comes from.
enum class InputSource : u8 {
    Parameter, ///< a vertex shader parameter export
    Position,  ///< the pixel's position
    Default,   ///< a constant: no vertex output has the semantic
};

enum class Interpolation : u8 {
    Perspective,
    Linear, ///< without perspective correction
    Flat,
};

/// A pixel shader input: what register `gpr` holds on entry.
struct PixelInput {
    u32 gpr = 0;
    InputSource source = InputSource::Default;
    u32 semantic = 0;
    u32 parameter = 0;     ///< `Parameter`: which of the vertex shader's
    u32 default_value = 0; ///< `Default`: SPI_PS_INPUT_CNTL's DEFAULT_VAL
    Interpolation interpolation = Interpolation::Perspective;
    bool centroid = false;
    bool sample = false;
};

/// The statements.
struct FetchAttributes {}; ///< every attribute into its register (CALL_FS)

struct ExportPosition {
    u32 gpr = 0;
    Swizzle swizzle{};
};

struct ExportParameter {
    u32 index = 0; ///< PARAMn
    u32 semantic = 0;
    u32 gpr = 0;
    Swizzle swizzle{};
};

struct ExportColor {
    u32 index = 0;  ///< PIXn
    u32 target = 0; ///< the colour target CB_SHADER_MASK makes it
    u32 gpr = 0;
    Swizzle swizzle{};
};

/// A SAMPLE from a 2D texture: the stage's texture unit and sampler, at
/// normalised coordinates from two of a register's components.
struct SampleTexture {
    u32 texture = 0; ///< the stage's texture unit: the fetch's resource
    u32 sampler = 0; ///< the stage's sampler
    u32 src_gpr = 0;
    std::array<Sel, 2> coordinates{}; ///< the source register's U and V
    Swizzle texture_select{};         ///< the resource's DST_SEL, applied first
    u32 dst_gpr = 0;
    Swizzle swizzle{}; ///< the fetch's DST_SEL: the destination's components
};

/// What an ALU source reads.
enum class SourceKind : u8 {
    Register,        ///< a component of R0 to R127
    Value,           ///< a literal or an inline constant: these 32 bits
    PreviousVector,  ///< PV: the previous group's result on the X, Y, Z or W unit
    PreviousScalar,  ///< PS: its result on the T unit
    UniformRegister, ///< one of the stage's 256 uniform registers (the constant file)
    UniformBlock,    ///< a constant of one of the stage's uniform blocks, through KCACHE
};

/// One of an ALU instruction's sources, with its modifiers: the absolute
/// value first, then the negation, as decaf applies them.
struct AluSource {
    SourceKind kind = SourceKind::Value;
    u32 gpr = 0;   ///< `Register`
    u32 chan = 0;  ///< every kind but `Value` and `PreviousScalar`: x, y, z or w
    u32 bits = 0;  ///< `Value`: as stored -- a float's bits, or an integer's
    u32 block = 0; ///< `UniformBlock`: which of the stage's sixteen (the KCACHE bank)
    u32 index = 0; ///< `UniformRegister` and `UniformBlock`: which four-component constant
    bool absolute = false;
    bool negate = false;
};

/// The ALU instructions translated so far: float arithmetic.
enum class AluOpcode : u8 {
    Mov,
    Add,
    Mul, ///< DX9's: zero times anything is zero
    MulIeee,
    Max,    ///< (a >= b) ? a : b
    Min,    ///< (a < b) ? a : b
    MulAdd, ///< DX9's multiply, then the add
    MulAddIeee,
    Dot4, ///< a reduction over the X, Y, Z and W units; DX9's multiply
    Dot4Ieee,
};

/// What is done to a result before it is clamped: an OP2's OMOD, or what an
/// OP3 MULADD's _M2, _M4 or _D2 asks for.
enum class OutputModifier : u8 {
    None,
    Times2,
    Times4,
    Half,
};

/// One unit's instruction. A reduction is four of these, one a unit, all
/// with the same opcode: each unit's result is the reduction's, written
/// where that unit's own destination says. That is Cemu's reading; decaf
/// gives it to PV.x and the first written unit's destination alone, and
/// Cemu's own note says AMD's documentation may agree with decaf about PV.
/// Cemu's is taken because it runs real titles; a shader that tells the two
/// apart is worth keeping when one turns up.
struct AluOperation {
    AluUnit unit = AluUnit::X;
    AluOpcode opcode = AluOpcode::Mov;
    std::array<AluSource, 3> sources{}; ///< as many as the opcode reads
    bool write = true;                  ///< to the register; PV or PS is set either way
    u32 dst_gpr = 0;
    u32 dst_chan = 0;
    OutputModifier modifier = OutputModifier::None;
    bool clamp = false; ///< to [0, 1], after the modifier
};

/// An ALU instruction group: every source is read before any result is
/// written, and the results become PV and PS for the group after.
struct AluGroup {
    std::vector<AluOperation> operations; ///< by unit; NOPs left out
};

using Statement = std::
    variant<FetchAttributes, SampleTexture, AluGroup, ExportPosition, ExportParameter, ExportColor>;

struct Shader {
    Stage stage = Stage::Vertex;
    std::vector<VertexAttribute> attributes; ///< a vertex shader's
    std::vector<PixelInput> inputs;          ///< a pixel shader's
    std::vector<Statement> body;
};

/// The uniforms a shader's ALU reads: whether it reads the stage's uniform
/// registers, and which of its uniform blocks.
struct UniformUse {
    bool registers = false;
    std::array<bool, resource::uniform_blocks_per_stage> blocks{};

    [[nodiscard]] bool any() const noexcept;
};

[[nodiscard]] UniformUse uniform_use(const Shader& shader);

/// A vertex shader, with the fetch shader it calls (or none, if it calls
/// none). False, with `error` saying what and where, for one this cannot
/// describe yet or that makes no sense.
[[nodiscard]] bool build_vertex_shader(const ShaderProgram& vertex,
                                       const ShaderProgram* fetch,
                                       const RegisterFile& registers,
                                       Shader& out,
                                       std::string& error);

/// A pixel shader. Its inputs are matched against the vertex shader's
/// parameter semantics, which are registers too.
[[nodiscard]] bool build_pixel_shader(const ShaderProgram& pixel,
                                      const RegisterFile& registers,
                                      Shader& out,
                                      std::string& error);

/// A readable account of a shader, a line a fact -- for tests and the GPU
/// debugger.
[[nodiscard]] std::string describe(const Shader& shader);

} // namespace wiiu::latte::ir
