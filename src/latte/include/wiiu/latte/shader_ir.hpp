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
//   * statements: fetching the attributes (CALL_FS), and exporting the
//     position, a parameter or a colour.
//
// Everything else is refused, saying which instruction and that it is not
// done yet, rather than translated into something that draws wrongly: ALU
// and texture clauses, loops, jumps and subroutines, memory exports, point
// size, clip distances and depth exports, relative registers, plain
// (non-semantic) fetches. Each later step takes something off that list.
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

using Statement = std::variant<FetchAttributes, ExportPosition, ExportParameter, ExportColor>;

struct Shader {
    Stage stage = Stage::Vertex;
    std::vector<VertexAttribute> attributes; ///< a vertex shader's
    std::vector<PixelInput> inputs;          ///< a pixel shader's
    std::vector<Statement> body;
};

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
