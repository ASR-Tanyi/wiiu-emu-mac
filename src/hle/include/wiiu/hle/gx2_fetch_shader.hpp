// Phase 8.1 — the fetch shader GX2 writes.
//
// A vertex shader does not read its attributes itself: it calls a fetch
// shader, which GX2 generates at run time from the title's list of attribute
// streams (`GX2InitFetchShaderEx`) and `GX2SetFetchShader` binds. The program
// is one vertex fetch per attribute, sixteen to a VTX_TC clause, then a
// RETURN -- R700 microcode in the title's buffer, little-endian, as the GPU
// reads it (latte/microcode.hpp). Each fetch says where its attribute is
// (the buffer, the offset in the element, per vertex or per instance), how
// to read it (the data format and byte swap) and where it goes (the
// semantic, and which components).
//
// decaf and Cemu both generate it, and agree on the program word for word
// but for the differences named where they arise. Only the untessellated
// fetch shader is generated: decaf refuses the tessellated ones and Cemu
// writes the untessellated one regardless, which is what this does, logged.
// The sizes `GX2CalcFetchShaderSizeEx` gives for them are decaf's, so a
// title's buffer is as large as on hardware.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/latte/microcode.hpp>

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// `GX2FetchShaderType`.
enum class FetchShaderType : u32 {
    NoTessellation = 0,
    LineTessellation = 1,
    TriangleTessellation = 2,
    QuadTessellation = 3,
};

/// `GX2TessellationMode`.
enum class TessellationMode : u32 {
    Discrete = 0,
    Continuous = 1,
    Adaptive = 2,
};

/// `GX2AttribIndexType`.
enum class AttribIndexType : u32 {
    PerVertex = 0,
    PerInstance = 1,
};

/// `GX2AttribStream`: one attribute, as the title describes it.
struct AttribStream {
    u32be location;    ///< the semantic the vertex shader reads it as
    u32be buffer;      ///< which attribute buffer
    u32be offset;      ///< bytes into the element
    u32be format;      ///< `GX2AttribFormat`
    u32be type;        ///< `GX2AttribIndexType`
    u32be alu_divisor; ///< per-instance: instances per element
    u32be mask;        ///< component selects, X in the top byte
    u32be endian_swap; ///< `GX2EndianSwapMode`; 3 is the format's own
};

static_assert(sizeof(AttribStream) == 0x20);

/// A `GX2AttribFormat`: the type in the low bits, flags above. The type
/// numbers its layouts 8, 4_4, 16, 16_FLOAT, 8_8, 32, 32_FLOAT, 16_16,
/// 16_16_FLOAT, 10_11_11_FLOAT, 8_8_8_8, 10_10_10_2, 32_32, 32_32_FLOAT,
/// 16_16_16_16, 16_16_16_16_FLOAT, 32_32_32, 32_32_32_FLOAT, 32_32_32_32 and
/// 32_32_32_32_FLOAT, from 0 to 0x13.
namespace attrib_format {
inline constexpr u32 type_mask = 0x3F;
inline constexpr u32 type_count = 0x14;
inline constexpr u32 integer = 0x100;
inline constexpr u32 is_signed = 0x200;
inline constexpr u32 degamma = 0x400;
inline constexpr u32 scaled = 0x800;
} // namespace attrib_format

/// `GX2GetAttribFormatBits`: the bits one attribute of a format takes. Zero
/// for a type GX2 does not have.
[[nodiscard]] u32 attrib_format_bits(u32 format) noexcept;

/// The data format a fetch reads an attribute format as: an R600 FMT_*
/// number, or nothing for a type GX2 does not have. Cemu's table, which
/// decaf's switch matches but for one entry: GX2's 10_10_10_2 is
/// FMT_2_10_10_10 (0x19) in Cemu and FMT_10_10_10_2 (0x1B) in decaf. Cemu's
/// is taken -- it is the number GX2's surface formats give R10_G10_B10_A2.
[[nodiscard]] std::optional<u32> attrib_data_format(u32 format) noexcept;

/// The byte swap an attribute gets when its stream asks for the format's
/// own: none for bytes, 8-in-16 for 16-bit components, 8-in-32 for the
/// rest. decaf and Cemu agree.
[[nodiscard]] latte::Endian attrib_default_endian(u32 format) noexcept;

/// `GX2CalcFetchShaderSizeEx`: the bytes the program takes. Untessellated,
/// the control-flow instructions -- one per sixteen fetches, and the RETURN
/// -- padded to sixteen bytes, then sixteen bytes a fetch.
[[nodiscard]] u32
fetch_shader_size(u32 attribs, FetchShaderType type, TessellationMode mode) noexcept;

/// What `GX2InitFetchShaderEx` produces: the program's words, in the order
/// the GPU reads them, and the instance divisors it collected.
struct FetchShaderProgram {
    std::vector<u32> words;
    u32 divisor_count = 0;
    std::array<u32, 2> divisors{};
};

/// The untessellated fetch shader for `attribs`.
[[nodiscard]] FetchShaderProgram generate_fetch_shader(std::span<const AttribStream> attribs);

/// Register `GX2CalcFetchShaderSizeEx`, `GX2InitFetchShaderEx` and
/// `GX2GetAttribFormatBits`. Called by `register_gx2_shader`.
void register_gx2_fetch_shader(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
