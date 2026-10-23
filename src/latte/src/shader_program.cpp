// Phase 8.6 — see shader_program.hpp.

#include <wiiu/latte/shader_program.hpp>

#include <fmt/format.h>

#include <array>
#include <iterator>
#include <string_view>

namespace wiiu::latte {

namespace {

/// Words in one vertex fetch: three and a word of padding.
constexpr u32 vtx_words = vtx_instruction_bytes / 4;

[[nodiscard]] bool known(const VertexFetch& fetch) noexcept {
    const u32 inst = fetch.word0 & 0x1FU;
    return inst == static_cast<u32>(VtxInst::Fetch) || inst == static_cast<u32>(VtxInst::Semantic);
}

[[nodiscard]] bool decode_vertex_fetches(std::span<const u32> program,
                                         const Clause& clause,
                                         std::vector<VertexFetch>& out,
                                         std::string& error) {
    // The control flow has checked the clause lies inside the program.
    const std::size_t first = clause.offset_bytes() / 4;
    for (u32 i = 0; i < clause.count; ++i) {
        const std::size_t at = first + (std::size_t{i} * vtx_words);
        const VertexFetch fetch{
            .word0 = program[at], .word1 = program[at + 1], .word2 = program[at + 2]};
        if (!known(fetch)) {
            error = fmt::format("fetch {} of the clause instruction {} runs ({:#010x}) is "
                                "VTX_INST {}, not an R700 vertex fetch",
                                i,
                                clause.cf_index,
                                fetch.word0,
                                fetch.word0 & 0x1FU);
            return false;
        }
        out.push_back(fetch);
    }
    return true;
}

/// Words in one texture fetch: three and a word of padding.
constexpr u32 tex_words = tex_instruction_bytes / 4;

/// A texture clause's fetches. Every five-bit TEX_INST names an R700
/// instruction, and the two that are vertex fetches are the two vertex
/// fetch numbers there are, so nothing here can be refused.
[[nodiscard]] std::vector<TextureFetch> decode_texture_fetches(std::span<const u32> program,
                                                               const Clause& clause) {
    std::vector<TextureFetch> out;
    const std::size_t first = clause.offset_bytes() / 4;
    for (u32 i = 0; i < clause.count; ++i) {
        const std::size_t at = first + (std::size_t{i} * tex_words);
        out.push_back(
            TextureFetch{.word0 = program[at], .word1 = program[at + 1], .word2 = program[at + 2]});
    }
    return out;
}

[[nodiscard]] char select_char(u32 sel) noexcept {
    constexpr std::string_view chars = "xyzw01?_";
    return chars[sel & 7U];
}

[[nodiscard]] char select_char(Sel sel) noexcept {
    return select_char(static_cast<u32>(sel));
}

/// A signed field of `bits` bits, with `fraction` of them after the point.
[[nodiscard]] f32 signed_fixed(u32 raw, u32 bits, u32 fraction) noexcept {
    const u32 sign = 1U << (bits - 1);
    const s32 value = static_cast<s32>((raw ^ sign) - sign);
    return static_cast<f32>(value) / static_cast<f32>(1U << fraction);
}

} // namespace

f32 TextureFetch::lod_bias() const noexcept {
    return signed_fixed(tex1().lod_bias, 7, 3);
}

f32 TextureFetch::offset(u32 axis) const noexcept {
    const TexWord2 word = tex2();
    const u32 raw = axis == 0   ? word.offset_x.value()
                    : axis == 1 ? word.offset_y.value()
                                : word.offset_z.value();
    return signed_fixed(raw, 5, 1);
}

bool decode_program(std::span<const u32> program,
                    ProgramShape shape,
                    ShaderProgram& out,
                    std::string& error) {
    out = ShaderProgram{};
    if (!decode_control_flow(program, shape, out.control_flow, error)) {
        return false;
    }
    for (const Clause& clause : out.control_flow.clauses) {
        DecodedClause decoded{.clause = clause, .vertex_fetches = {}, .texture_fetches = {}};
        if (clause.kind == ClauseKind::Vtx &&
            !decode_vertex_fetches(program, clause, decoded.vertex_fetches, error)) {
            return false;
        }
        if (clause.kind == ClauseKind::Tex) {
            decoded.texture_fetches = decode_texture_fetches(program, clause);
        }
        out.clauses.push_back(std::move(decoded));
    }
    return true;
}

std::string disassemble(const VertexFetch& fetch) {
    const VtxWord0 word0 = fetch.fetch0();
    const VtxWord1 word1 = fetch.fetch1();
    const VtxWord2 word2 = fetch.fetch2();
    std::string text;
    const auto out = std::back_inserter(text);

    fmt::format_to(out, "{:<15} ", fetch.is_semantic() ? "SEMANTIC" : "FETCH");

    const std::array<u32, 4> dst{static_cast<u32>(word1.dst_sel_x.value()),
                                 static_cast<u32>(word1.dst_sel_y.value()),
                                 static_cast<u32>(word1.dst_sel_z.value()),
                                 static_cast<u32>(word1.dst_sel_w.value())};
    const u32 masked = static_cast<u32>(Sel::Mask);
    if (dst[0] != masked || dst[1] != masked || dst[2] != masked || dst[3] != masked) {
        if (fetch.is_semantic()) {
            fmt::format_to(out, "SEM[{}]", word1.semantic_id.value());
        } else {
            fmt::format_to(out, "R{}{}", word1.dst_gpr.value(), word1.dst_rel ? "[AL]" : "");
        }
        fmt::format_to(out,
                       ".{}{}{}{}",
                       select_char(dst[0]),
                       select_char(dst[1]),
                       select_char(dst[2]),
                       select_char(dst[3]));
    } else {
        fmt::format_to(out, "____");
    }

    fmt::format_to(out,
                   ", R{}{}.{}, b{}",
                   word0.src_gpr.value(),
                   word0.src_rel ? "[AL]" : "",
                   select_char(static_cast<u32>(word0.src_sel_x.value())),
                   word0.buffer_id.value());

    constexpr std::array<std::string_view, 3> fetch_types{
        "VERTEX_DATA", "INSTANCE_DATA", "NO_INDEX_OFFSET"};
    const u32 fetch_type = static_cast<u32>(word0.fetch_type.value());
    if (fetch_type < fetch_types.size()) {
        fmt::format_to(out, " FETCH_TYPE({})", fetch_types[fetch_type]);
    } else {
        fmt::format_to(out, " FETCH_TYPE({})", fetch_type);
    }

    if (!word1.use_const_fields) {
        // decaf's spacing, a number format of 3 included.
        fmt::format_to(out, " FORMAT( {}", word1.data_format.value());
        constexpr std::array<std::string_view, 3> num_formats{" NORM", " INT", " SCALED"};
        const u32 num_format = static_cast<u32>(word1.num_format_all.value());
        if (num_format < num_formats.size()) {
            fmt::format_to(out, "{}", num_formats[num_format]);
        } else {
            fmt::format_to(out, "{}", num_format);
        }
        fmt::format_to(out,
                       "{} {})",
                       word1.format_comp_signed ? " SIGNED" : " UNSIGNED",
                       word1.srf_mode_all ? 1 : 0);
    }

    fmt::format_to(out,
                   " {}({}) OFFSET({})",
                   word2.mega_fetch ? "MEGA" : "MINI",
                   word0.mega_fetch_count + 1,
                   word2.offset.value());
    if (word0.fetch_whole_quad) {
        fmt::format_to(out, " WHOLE_QUAD");
    }
    switch (word2.endian_swap.value()) {
        case Endian::None:
            break;
        case Endian::Swap8In16:
            fmt::format_to(out, " ENDIAN_SWAP(8IN16)");
            break;
        case Endian::Swap8In32:
            fmt::format_to(out, " ENDIAN_SWAP(8IN32)");
            break;
        default:
            fmt::format_to(out, " ENDIAN_SWAP({})", static_cast<u32>(word2.endian_swap.value()));
            break;
    }
    if (word2.const_buf_no_stride) {
        fmt::format_to(out, " CONST_BUF_NO_STRIDE");
    }
    if (word2.alt_const) {
        fmt::format_to(out, " ALT_CONST");
    }
    return text;
}

std::string_view tex_inst_name(TexInst inst) noexcept {
    constexpr std::array<std::string_view, 32> names{"VTX_FETCH",
                                                     "VTX_SEMANTIC",
                                                     "MEM",
                                                     "LD",
                                                     "GET_TEXTURE_INFO",
                                                     "GET_SAMPLE_INFO",
                                                     "GET_COMP_TEX_LOD",
                                                     "GET_GRADIENTS_H",
                                                     "GET_GRADIENTS_V",
                                                     "GET_LERP",
                                                     "KEEP_GRADIENTS",
                                                     "SET_GRADIENTS_H",
                                                     "SET_GRADIENTS_V",
                                                     "PASS",
                                                     "SET_CUBEMAP_INDEX",
                                                     "FETCH4",
                                                     "SAMPLE",
                                                     "SAMPLE_L",
                                                     "SAMPLE_LB",
                                                     "SAMPLE_LZ",
                                                     "SAMPLE_G",
                                                     "SAMPLE_G_L",
                                                     "SAMPLE_G_LB",
                                                     "SAMPLE_G_LZ",
                                                     "SAMPLE_C",
                                                     "SAMPLE_C_L",
                                                     "SAMPLE_C_LB",
                                                     "SAMPLE_C_LZ",
                                                     "SAMPLE_C_G",
                                                     "SAMPLE_C_G_L",
                                                     "SAMPLE_C_G_LB",
                                                     "SAMPLE_C_G_LZ"};
    return names[static_cast<u32>(inst) & 0x1FU];
}

std::string disassemble(const TextureFetch& fetch) {
    if (fetch.is_vertex_fetch()) {
        return disassemble(fetch.as_vertex_fetch());
    }
    const TexWord0 word0 = fetch.tex0();
    const TexWord1 word1 = fetch.tex1();
    const TexWord2 word2 = fetch.tex2();
    std::string text;
    const auto out = std::back_inserter(text);

    fmt::format_to(out, "{:<15} ", tex_inst_name(fetch.inst()));

    const std::array<Sel, 4> dst{
        word1.dst_sel_x, word1.dst_sel_y, word1.dst_sel_z, word1.dst_sel_w};
    if (dst[0] != Sel::Mask || dst[1] != Sel::Mask || dst[2] != Sel::Mask || dst[3] != Sel::Mask) {
        fmt::format_to(out,
                       "R{}{}.{}{}{}{}",
                       word1.dst_gpr.value(),
                       word1.dst_rel ? "[AL]" : "",
                       select_char(dst[0]),
                       select_char(dst[1]),
                       select_char(dst[2]),
                       select_char(dst[3]));
    } else {
        fmt::format_to(out, "____");
    }

    fmt::format_to(out,
                   ", R{}{}.{}{}{}{}, t{}, s{}",
                   word0.src_gpr.value(),
                   word0.src_rel ? "[AL]" : "",
                   select_char(word2.src_sel_x),
                   select_char(word2.src_sel_y),
                   select_char(word2.src_sel_z),
                   select_char(word2.src_sel_w),
                   word0.resource_id.value(),
                   word2.sampler_id.value());

    if (word1.lod_bias != 0) {
        fmt::format_to(out, " LOD({})", fetch.lod_bias());
    }
    if (word0.fetch_whole_quad) {
        fmt::format_to(out, " WHOLE_QUAD");
    }
    if (word0.bc_frac_mode) {
        fmt::format_to(out, " BC_FRAC_MODE");
    }
    if (word0.alt_const) {
        fmt::format_to(out, " ALT_CONST");
    }
    // COORD_TYPE is set for a normalised coordinate: the others are texels.
    const std::array<bool, 4> normalised{
        word1.coord_type_x, word1.coord_type_y, word1.coord_type_z, word1.coord_type_w};
    if (!normalised[0] || !normalised[1] || !normalised[2] || !normalised[3]) {
        fmt::format_to(out, " DENORM(");
        constexpr std::string_view axes = "XYZW";
        for (std::size_t i = 0; i < normalised.size(); ++i) {
            if (!normalised[i]) {
                fmt::format_to(out, "{}", axes[i]);
            }
        }
        fmt::format_to(out, ")");
    }
    constexpr std::array<std::string_view, 3> offsets{"XOFFSET", "YOFFSET", "ZOFFSET"};
    const std::array<u32, 3> raw{word2.offset_x, word2.offset_y, word2.offset_z};
    for (u32 axis = 0; axis < offsets.size(); ++axis) {
        if (raw[axis] != 0) {
            fmt::format_to(out, " {}({})", offsets[axis], fetch.offset(axis));
        }
    }
    return text;
}

std::string disassemble(const ShaderProgram& program) {
    // decaf numbers every fetch, and every ALU group, in one count across the
    // program, in the order the control flow lists them.
    u32 counter = 0;
    const ClauseLines after = [&program, &counter](std::string& out,
                                                   const CfInstruction& instruction,
                                                   std::string_view indent) {
        for (const DecodedClause& decoded : program.clauses) {
            if (decoded.clause.cf_index != instruction.index) {
                continue;
            }
            for (const VertexFetch& fetch : decoded.vertex_fetches) {
                fmt::format_to(std::back_inserter(out),
                               "{}  {:<3}    {}\n",
                               indent,
                               counter++,
                               disassemble(fetch));
            }
            for (const TextureFetch& fetch : decoded.texture_fetches) {
                fmt::format_to(std::back_inserter(out),
                               "{}  {:<3}    {}\n",
                               indent,
                               counter++,
                               disassemble(fetch));
            }
            return;
        }
    };
    return disassemble(program.control_flow, after);
}

} // namespace wiiu::latte
