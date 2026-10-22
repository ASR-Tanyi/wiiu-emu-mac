// Phase 8.6 — the emitter: the intermediate form (shader_ir.hpp) as Metal
// Shading Language, text, straight from the IR -- not through SPIR-V (see
// the plan, 8.6).
//
// The model carries over unchanged: each register the shader uses is a
// `float4 R<n>`, and statements become assignments, one component at a
// time, so a masked component is simply not written.
//
// **Vertex fetch is done in the shader** (the Phase 7 rule): Latte reads an
// attribute at any byte offset, in either byte order, and Metal's vertex
// descriptors would accept neither. Each attribute buffer is bound as bytes
// -- attribute buffer n at vertex buffer slot `first_attribute_buffer_slot
// + n`, bound at the buffer's base address -- and each component is put
// together a byte at a time in the order the fetch's ENDIAN_SWAP says:
// as stored (little-endian) for none, big-endian for 8IN32, which is what
// GX2 asks for. A read past the buffer's size reads zero, as Latte's does.
// What the shader needs to know per draw -- each buffer's stride and size,
// from its resource, and the two instance step rates -- comes in a
// `FetchConstants` at `fetch_constants_slot`, so one compiled shader serves
// however the title binds its buffers. Missing components of a fetched value
// read zero, as Cemu has it; the fetch's own component selects supply the
// constants a title wants.
//
// Vertex outputs are matched to pixel inputs by semantic: each is a `float4`
// named `sem<S>` with `[[user(sem<S>)]]`. A vertex shader is given the pixel
// shader it will be linked with, and also outputs, as zero, any semantic the
// pixel shader reads and it does not write, since Metal will not link a
// fragment input nothing produces. A pixel input's interpolation is its
// Metal qualifier.
//
// Refused, as not translated yet: attribute formats other than 32-bit
// floats (32, 32_32, 32_32_32 and 32_32_32_32, SCALED, as GX2 writes them),
// byte swaps other than none and 8IN32, a semantic read twice with two
// interpolations, and integer colour targets -- every colour output is a
// `float4` for now. Not yet done either, and needing register state the
// translator (8.11) has: R0's vertex and instance indices for a shader that
// reads them.
//
// **Clip-space depth.** Metal clips z to [0, w]; a title that leaves
// PA_CL_CLIP_CNTL's DX_CLIP_SPACE_DEF clear -- GX2's default -- writes
// OpenGL's [-w, w]. With `VertexOptions::depth_from_minus_one` the position
// is brought over as it is returned, z' = (z + w) / 2, which is Cemu's
// SET_POSITION; the viewport's depth range (draw_state.hpp) is already read
// the OpenGL way for such a title, so the depth that lands is OpenGL's.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_ir.hpp>

#include <array>
#include <string>
#include <string_view>

namespace wiiu::latte::msl {

inline constexpr std::string_view vertex_entry = "vertex_main";
inline constexpr std::string_view fragment_entry = "fragment_main";

/// The vertex stage's buffer slots.
inline constexpr u32 fetch_constants_slot = 0;
inline constexpr u32 first_attribute_buffer_slot = 1;

/// What a vertex shader reads at `fetch_constants_slot`. The MSL structure
/// is written from the same layout.
struct FetchConstants {
    std::array<u32, resource::attribute_buffer_count> stride{}; ///< bytes an element
    std::array<u32, resource::attribute_buffer_count> size{};   ///< bytes in all
    std::array<u32, 2> step_rate{};                             ///< VGT_INSTANCE_STEP_RATE_0 and _1
    std::array<u32, 2> pad{};
};

static_assert(sizeof(FetchConstants) == 144);

/// What a vertex shader's source depends on beyond the shader itself.
struct VertexOptions {
    /// Clip-space z runs from -w (OpenGL's convention), to be brought to
    /// Metal's 0.
    bool depth_from_minus_one = false;
};

/// A vertex shader's source, given the pixel shader it will be linked with
/// (or none). False, with `error` saying what, for one this cannot emit yet.
[[nodiscard]] bool emit_vertex(const ir::Shader& vertex,
                               const ir::Shader* pixel,
                               std::string& out,
                               std::string& error,
                               const VertexOptions& options = {});

/// A pixel shader's source.
[[nodiscard]] bool emit_pixel(const ir::Shader& pixel, std::string& out, std::string& error);

} // namespace wiiu::latte::msl
