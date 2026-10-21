// Phase 8.11 — the translator's first slice: a draw's shaders, from the
// registers that name them to MSL.
//
// A draw packet says nothing about its shaders. They are wherever
// SQ_PGM_START_VS, _PS and _FS last said -- each the program's physical
// address >> 8, with its size >> 3 in the register after it (decaf's and
// Cemu's reading, and what GX2's shader calls write, 8.1). This finds each
// program, reads it out of guest memory -- as little-endian words, which is
// how the GPU reads a program -- and takes it through the decompiler's three
// stages: decode (shader_program.hpp), describe (shader_ir.hpp) and emit
// (msl.hpp). What comes back is what a pipeline is made of, and everything
// along the way, so a failure can be looked at stage by stage.
//
// A vertex shader without a fetch shader is allowed -- SQ_PGM_START_FS's
// size zero means none -- and is an error only if it calls one. Refused, not
// yet translated: geometry shaders (VGT_GS_MODE other than off, Cemu's test,
// in which the vertex shader runs as the ES stage). Not done yet: caching
// (8.10) -- the same programs under the same registers make the same
// shaders, so this will be looked up rather than rebuilt per draw -- and
// making the pipeline itself, which needs the render targets.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/shader_ir.hpp>
#include <wiiu/latte/shader_program.hpp>
#include <wiiu/memory/memory.hpp>

#include <string>
#include <vector>

namespace wiiu::latte {

namespace reg {

/// VGT_GS_MODE: its MODE, the low two bits, is zero when there is no
/// geometry shader. In Linux's R600 register list.
inline constexpr u32 vgt_gs_mode = 0x28A40;

} // namespace reg

/// Where a stage's program is.
struct ProgramLocation {
    physaddr_t address = 0;
    u32 size = 0; ///< bytes
};

/// The program a SQ_PGM_START register and the one after it name.
[[nodiscard]] ProgramLocation program_location(const RegisterFile& registers, u32 start);

/// A program's words, read little-endian out of guest memory. False, with
/// `error` saying where, if the memory is not there.
[[nodiscard]] bool read_program(const memory::Memory& memory,
                                ProgramLocation location,
                                std::vector<u32>& words,
                                std::string& error);

/// A draw's shaders, every stage of the way.
struct DrawShaders {
    ProgramLocation vertex_location;
    ProgramLocation fetch_location;
    ProgramLocation pixel_location;

    ShaderProgram vertex;
    ShaderProgram fetch; ///< empty if there is none
    bool has_fetch = false;
    ShaderProgram pixel;

    ir::Shader vertex_ir;
    ir::Shader pixel_ir;

    std::string vertex_msl;
    std::string pixel_msl;
};

/// Build a draw's shaders from the registers as they stand at the draw.
/// False, with `error` naming the stage and the shader, for one that cannot
/// be translated yet or makes no sense.
[[nodiscard]] bool build_draw_shaders(const RegisterFile& registers,
                                      const memory::Memory& memory,
                                      DrawShaders& out,
                                      std::string& error);

} // namespace wiiu::latte
