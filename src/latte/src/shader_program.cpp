// Phase 8.6 — see shader_program.hpp.

#include <wiiu/latte/shader_program.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

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

/// Words in one ALU slot, and the most instructions a group has.
constexpr u32 alu_slot_words = 2;
constexpr u32 max_group_instructions = 5;

/// The most bank swizzles a vector instruction and a transcendental-only one
/// have; PRED_SEL's reserved value.
constexpr u32 vector_bank_swizzles = 6;
constexpr u32 scalar_bank_swizzles = 4;
constexpr u32 pred_sel_reserved = 1;

/// The power of ten from which fmt 8 prints a float with an exponent.
constexpr int fixed_exponent_limit = 16;

[[nodiscard]] bool decode_alu_groups(std::span<const u32> program,
                                     const Clause& clause,
                                     std::vector<AluGroup>& out,
                                     std::string& error) {
    const std::size_t first = clause.offset_bytes() / 4;
    const auto refuse = [&](u32 slot, std::string_view what) {
        error = fmt::format("the ALU group at slot {} of the clause instruction {} {}",
                            slot,
                            clause.cf_index,
                            what);
        return false;
    };
    u32 slot = 0;
    while (slot < clause.count) {
        const u32 start = slot;
        AluGroup group;
        std::array<bool, 5> taken{};
        u32 literal_count = 0;
        bool last = false;
        while (!last) {
            if (group.instructions.size() == max_group_instructions) {
                return refuse(start, "has more than five instructions");
            }
            if (slot >= clause.count) {
                return refuse(start, "runs past the end of its clause");
            }
            const std::size_t at = first + (std::size_t{slot} * alu_slot_words);
            AluInstruction inst{
                .word0 = program[at], .word1 = program[at + 1], .op = {}, .unit = AluUnit::X};
            ++slot;
            const u32 number =
                inst.is_op3() ? inst.op3().alu_inst.value() : inst.op2().alu_inst.value();
            const std::optional<AluOp> op = inst.is_op3() ? alu_op3(number) : alu_op2(number);
            if (!op.has_value()) {
                return refuse(start,
                              fmt::format("holds {} {:#x}, not an R700 ALU instruction",
                                          inst.is_op3() ? "OP3" : "OP2",
                                          number));
            }
            inst.op = *op;

            // Its channel's unit; T if only T runs it, or if that unit is
            // taken and T can run it.
            u32 unit = inst.alu1().dst_chan;
            if (op->transcendental_only() || (!op->vector_only() && taken[unit])) {
                unit = static_cast<u32>(AluUnit::T);
            }
            if (taken[unit]) {
                return refuse(start,
                              fmt::format("gives two instructions the {} unit", "xyzwt"[unit]));
            }
            taken[unit] = true;
            inst.unit = static_cast<AluUnit>(unit);

            const u32 swizzles =
                op->transcendental_only() ? scalar_bank_swizzles : vector_bank_swizzles;
            if (inst.alu1().bank_swizzle >= swizzles) {
                return refuse(start,
                              fmt::format("gives {} bank swizzle {}",
                                          op->name,
                                          inst.alu1().bank_swizzle.value()));
            }
            if (inst.alu0().pred_sel == pred_sel_reserved) {
                return refuse(start, fmt::format("gives {} PRED_SEL 1", op->name));
            }

            const AluWord0 word0 = inst.alu0();
            const auto literal = [&](u32 index, u32 sel, u32 chan) {
                if (op->sources > index && sel == alu_src::literal) {
                    literal_count = std::max(literal_count, chan + 1);
                }
            };
            literal(0, word0.src0_sel, word0.src0_chan);
            literal(1, word0.src1_sel, word0.src1_chan);
            if (inst.is_op3()) {
                literal(2, inst.op3().src2_sel, inst.op3().src2_chan);
            }
            last = word0.last;
            group.instructions.push_back(inst);
        }
        const u32 literal_slots = (literal_count + 1) / 2;
        if (slot + literal_slots > clause.count) {
            return refuse(start, "has literals past the end of its clause");
        }
        const std::size_t at = first + (std::size_t{slot} * alu_slot_words);
        group.literals.assign(program.begin() + static_cast<std::ptrdiff_t>(at),
                              program.begin() + static_cast<std::ptrdiff_t>(at + literal_count));
        slot += literal_slots;
        out.push_back(std::move(group));
    }
    return true;
}

[[nodiscard]] char select_char(u32 sel) noexcept {
    constexpr std::string_view chars = "xyzw01?_";
    return chars[sel & 7U];
}

/// A literal read as a float, as decaf prints it: the fewest digits that
/// read back as the same float, in fixed notation from 1e-4 to 1e16. That is
/// fmt 8's rule, decaf's; later fmt switches to an exponent from 1e7 for a
/// float, which would print the common 0x4B000000 as 8.388608e+06, not
/// 8388608.
[[nodiscard]] std::string literal_float(u32 bits) {
    std::string text = fmt::format("{}", std::bit_cast<f32>(bits));
    const std::size_t e = text.find("e+");
    int exponent = 0;
    if (e == std::string::npos ||
        std::from_chars(text.data() + e + 2, text.data() + text.size(), exponent).ec !=
            std::errc{} ||
        exponent >= fixed_exponent_limit) {
        return text;
    }
    const bool negative = text.front() == '-';
    std::string digits;
    for (std::size_t i = negative ? 1 : 0; i < e; ++i) {
        if (text[i] != '.') {
            digits += text[i];
        }
    }
    const auto whole = static_cast<std::size_t>(exponent) + 1;
    if (digits.size() > whole) {
        digits.insert(whole, ".");
    } else {
        digits.resize(whole, '0');
    }
    return negative ? "-" + digits : digits;
}

/// A source, or the destination, as decaf prints one: negated, absolute,
/// relative, and with its channel where it has one.
void alu_source(std::string& out,
                u64 group,
                u32 index_mode,
                u32 sel,
                bool relative,
                u32 chan,
                u32 literal,
                bool negate,
                bool absolute) {
    const auto to = std::back_inserter(out);
    bool channel = true;
    if (negate) {
        out += '-';
    }
    if (absolute) {
        out += '|';
    }
    if (sel >= alu_src::kcache0_first && sel <= alu_src::kcache0_last) {
        fmt::format_to(to, "KC0[{}]", sel - alu_src::kcache0_first);
    } else if (sel >= alu_src::kcache1_first && sel <= alu_src::kcache1_last) {
        fmt::format_to(to, "KC1[{}]", sel - alu_src::kcache1_first);
    } else if (sel <= alu_src::register_last) {
        fmt::format_to(to, "R{}", sel);
    } else if (sel >= alu_src::const_file_first && sel <= alu_src::const_file_last) {
        fmt::format_to(to, "C{}", sel - alu_src::const_file_first);
    } else {
        channel = false;
        switch (sel) {
            case 219:
                out += "LDS_OQ_A";
                break;
            case 220:
                out += "LDS_OQ_B";
                break;
            case 221:
                out += "LDS_OQ_A_POP";
                break;
            case 222:
                out += "LDS_OQ_B_POP";
                break;
            case 223:
                out += "LDS_DIRECT_A";
                break;
            case 224:
                out += "LDS_DIRECT_B";
                break;
            case 227:
                out += "TIME_HI";
                break;
            case 228:
                out += "TIME_LO";
                break;
            case 229:
                out += "MASK_HI";
                break;
            case 230:
                out += "MASK_LO";
                break;
            case 231:
                out += "HW_WAVE_ID";
                break;
            case 232:
                out += "SIMD_ID";
                break;
            case 233:
                out += "SE_ID";
                break;
            case 234:
                out += "HW_THREADGRP_ID";
                break;
            case 235:
                out += "WAVE_ID_IN_GRP";
                break;
            case 236:
                out += "NUM_THREADGRP_WAVES";
                break;
            case 237:
                out += "HW_ALU_ODD";
                break;
            case alu_src::loop_index:
                out += "AL";
                break;
            case 240:
                out += "PARAM_BASE_ADDR";
                break;
            case 241:
                out += "NEW_PRIM_MASK";
                break;
            case 242:
                out += "PRIM_MASK_HI";
                break;
            case 243:
                out += "PRIM_MASK_LO";
                break;
            case 244:
                out += "1.0_L";
                break;
            case 245:
                out += "1.0_M";
                break;
            case 246:
                out += "0.5_L";
                break;
            case 247:
                out += "0.5_M";
                break;
            case alu_src::zero:
                out += "0.0f";
                break;
            case alu_src::one:
                out += "1.0f";
                break;
            case alu_src::one_int:
                out += "1";
                break;
            case alu_src::minus_one:
                out += "-1";
                break;
            case alu_src::half:
                out += "0.5f";
                break;
            case alu_src::literal:
                fmt::format_to(to, "(0x{:08X}, {})", literal, literal_float(literal));
                break;
            case alu_src::previous_vector:
                // decaf's count is unsigned: before the first group, it wraps.
                fmt::format_to(to, "PV{}", group - 1);
                channel = true;
                break;
            case alu_src::previous_scalar:
                fmt::format_to(to, "PS{}", group - 1);
                break;
            default:
                out += "UNKNOWN";
                break;
        }
    }
    if (relative) {
        constexpr std::array<std::string_view, 5> modes{
            "[AR.x]", "[AR.y]", "[AR.z]", "[AR.w]", "[AL]"};
        out += index_mode < modes.size() ? modes[index_mode] : "[UNKNOWN]";
    }
    if (channel) {
        fmt::format_to(to, ".{}", "xyzw"[chan & 3U]);
    }
    if (absolute) {
        out += '|';
    }
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
        DecodedClause decoded{
            .clause = clause, .vertex_fetches = {}, .texture_fetches = {}, .alu_groups = {}};
        if (clause.kind == ClauseKind::Vtx &&
            !decode_vertex_fetches(program, clause, decoded.vertex_fetches, error)) {
            return false;
        }
        if (clause.kind == ClauseKind::Tex) {
            decoded.texture_fetches = decode_texture_fetches(program, clause);
        }
        if (clause.kind == ClauseKind::Alu &&
            !decode_alu_groups(program, clause, decoded.alu_groups, error)) {
            return false;
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

std::string
disassemble(const AluInstruction& instruction, u32 group, std::span<const u32> literals) {
    const AluWord0 word0 = instruction.alu0();
    const AluWord1 word1 = instruction.alu1();
    const AluWord1Op2 op2 = instruction.op2();
    const AluWord1Op3 op3 = instruction.op3();
    const bool is_op3 = instruction.is_op3();
    const AluOp& op = instruction.op;
    std::string text;
    const auto to = std::back_inserter(text);

    std::string name{op.name};
    if (!is_op3 && !op2.update_execute_mask) {
        constexpr std::array<std::string_view, 4> modifiers{"", "*2", "*4", "/2"};
        name += modifiers[op2.omod];
    }
    fmt::format_to(to, "{:<15} ", name);

    const u64 count = group;
    const u32 index_mode = word0.index_mode;
    const auto literal_at = [&literals](u32 sel, u32 chan) {
        return sel == alu_src::literal && chan < literals.size() ? literals[chan] : 0U;
    };
    if (!is_op3 && !op2.write_mask) {
        text += "____";
    } else {
        alu_source(
            text, count, index_mode, word1.dst_gpr, word1.dst_rel, word1.dst_chan, 0, false, false);
    }
    if (op.sources > 0) {
        text += ", ";
        alu_source(text,
                   count,
                   index_mode,
                   word0.src0_sel,
                   word0.src0_rel,
                   word0.src0_chan,
                   literal_at(word0.src0_sel, word0.src0_chan),
                   word0.src0_neg,
                   !is_op3 && op2.src0_abs);
    }
    if (op.sources > 1) {
        text += ", ";
        alu_source(text,
                   count,
                   index_mode,
                   word0.src1_sel,
                   word0.src1_rel,
                   word0.src1_chan,
                   literal_at(word0.src1_sel, word0.src1_chan),
                   word0.src1_neg,
                   !is_op3 && op2.src1_abs);
    }
    if (op.sources > 2) {
        text += ", ";
        alu_source(text,
                   count,
                   index_mode,
                   op3.src2_sel,
                   op3.src2_rel,
                   op3.src2_chan,
                   literal_at(op3.src2_sel, op3.src2_chan),
                   op3.src2_neg,
                   false);
    }

    if (word1.clamp) {
        text += " CLAMP";
    }
    if (op.transcendental_only()) {
        constexpr std::array<std::string_view, 4> scalar{
            " SCL_210", " SCL_122", " SCL_212", " SCL_221"};
        text += scalar[word1.bank_swizzle & 3U];
    } else if (word1.bank_swizzle != 0) {
        constexpr std::array<std::string_view, 6> vector{
            "", " VEC_021", " VEC_120", " VEC_102", " VEC_201", " VEC_210"};
        text += word1.bank_swizzle < vector.size() ? vector[word1.bank_swizzle] : "";
    }
    if (word0.pred_sel == 2) {
        text += " PRED_SEL_ZERO";
    } else if (word0.pred_sel == 3) {
        text += " PRED_SEL_ONE";
    }
    if (!is_op3) {
        if (op2.update_execute_mask) {
            text += " UPDATE_EXEC_MASK";
        }
        if (op2.update_pred) {
            text += " UPDATE_PRED";
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
            // A group's first instruction beside its number, the rest
            // beneath, each with its unit -- decaf's layout.
            for (const AluGroup& group : decoded.alu_groups) {
                for (std::size_t i = 0; i < group.instructions.size(); ++i) {
                    const AluInstruction& alu = group.instructions[i];
                    const char unit = "xyzwt"[static_cast<u32>(alu.unit)];
                    const std::string text = disassemble(alu, counter, group.literals);
                    if (i == 0) {
                        fmt::format_to(std::back_inserter(out),
                                       "{}  {:<3} {}: {}\n",
                                       indent,
                                       counter,
                                       unit,
                                       text);
                    } else {
                        fmt::format_to(
                            std::back_inserter(out), "{}      {}: {}\n", indent, unit, text);
                    }
                }
                ++counter;
            }
            return;
        }
    };
    return disassemble(program.control_flow, after);
}

} // namespace wiiu::latte
