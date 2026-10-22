// Phase 8.11 — see draw_shaders.hpp.

#include <wiiu/latte/draw_shaders.hpp>
#include <wiiu/latte/msl.hpp>
#include <wiiu/latte/render_state.hpp>
#include <wiiu/latte/shader_state.hpp>
#include <wiiu/memory/map.hpp>

#include <fmt/format.h>

#include <string_view>

namespace wiiu::latte {

namespace {

/// Read, decode and name a stage's program; `what` is for the messages.
[[nodiscard]] bool decode_stage(const memory::Memory& memory,
                                ProgramLocation location,
                                ProgramShape shape,
                                ShaderProgram& out,
                                std::string_view what,
                                std::string& error) {
    if (location.size == 0) {
        error = fmt::format("there is no {} (its size register is zero)", what);
        return false;
    }
    std::vector<u32> words;
    if (!read_program(memory, location, words, error)) {
        error = fmt::format("the {}: {}", what, error);
        return false;
    }
    if (!decode_program(words, shape, out, error)) {
        error = fmt::format("the {} at physical {:#010x}: {}", what, location.address, error);
        return false;
    }
    return true;
}

} // namespace

ProgramLocation program_location(const RegisterFile& registers, u32 start) {
    return ProgramLocation{.address = registers.read(start) << 8,
                           .size = registers.read(start + 4) << 3};
}

bool read_program(const memory::Memory& memory,
                  ProgramLocation location,
                  std::vector<u32>& words,
                  std::string& error) {
    ppcaddr_t address = 0;
    if (!memory::to_virtual(memory.regions(), location.address, address)) {
        error = fmt::format(
            "{} bytes at physical {:#010x}, which nothing maps", location.size, location.address);
        return false;
    }
    std::vector<u8> bytes(location.size);
    if (!memory.read_bytes(address, bytes)) {
        error = fmt::format(
            "{} bytes at physical {:#010x} are not all readable", location.size, location.address);
        return false;
    }
    words.assign(bytes.size() / 4, 0);
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::size_t at = 4 * i;
        words[i] = static_cast<u32>(bytes[at]) | (static_cast<u32>(bytes[at + 1]) << 8) |
                   (static_cast<u32>(bytes[at + 2]) << 16) |
                   (static_cast<u32>(bytes[at + 3]) << 24);
    }
    return true;
}

bool build_draw_shaders(const RegisterFile& registers,
                        const memory::Memory& memory,
                        DrawShaders& out,
                        std::string& error) {
    out = DrawShaders{};
    if ((registers.read(reg::vgt_gs_mode) & 3U) != 0) {
        error = "the draw has a geometry shader (VGT_GS_MODE); not translated yet";
        return false;
    }

    out.vertex_location = program_location(registers, reg::sq_pgm_start_vs);
    out.fetch_location = program_location(registers, reg::sq_pgm_start_fs);
    out.pixel_location = program_location(registers, reg::sq_pgm_start_ps);

    if (!decode_stage(memory,
                      out.vertex_location,
                      ProgramShape::Shader,
                      out.vertex,
                      "vertex shader",
                      error) ||
        !decode_stage(
            memory, out.pixel_location, ProgramShape::Shader, out.pixel, "pixel shader", error)) {
        return false;
    }
    out.has_fetch = out.fetch_location.size != 0;
    if (out.has_fetch && !decode_stage(memory,
                                       out.fetch_location,
                                       ProgramShape::FetchShader,
                                       out.fetch,
                                       "fetch shader",
                                       error)) {
        return false;
    }

    if (!ir::build_vertex_shader(
            out.vertex, out.has_fetch ? &out.fetch : nullptr, registers, out.vertex_ir, error)) {
        error = fmt::format(
            "the vertex shader at physical {:#010x}: {}", out.vertex_location.address, error);
        return false;
    }
    if (!ir::build_pixel_shader(out.pixel, registers, out.pixel_ir, error)) {
        error = fmt::format(
            "the pixel shader at physical {:#010x}: {}", out.pixel_location.address, error);
        return false;
    }
    const msl::VertexOptions options{
        .depth_from_minus_one =
            !PaClClipCntl{.raw = registers.read(reg::pa_cl_clip_cntl)}.dx_clip_space_def};
    if (!msl::emit_vertex(out.vertex_ir, &out.pixel_ir, out.vertex_msl, error, options)) {
        error = fmt::format("emitting the vertex shader: {}", error);
        return false;
    }
    if (!msl::emit_pixel(out.pixel_ir, out.pixel_msl, error)) {
        error = fmt::format("emitting the pixel shader: {}", error);
        return false;
    }
    return true;
}

} // namespace wiiu::latte
