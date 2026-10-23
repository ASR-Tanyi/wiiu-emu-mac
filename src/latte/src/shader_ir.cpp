// Phase 8.6 — see shader_ir.hpp.

#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_ir.hpp>
#include <wiiu/latte/shader_state.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <optional>
#include <string_view>
#include <utility>

namespace wiiu::latte::ir {

namespace {

/// The fetch shader names attribute buffer n as its stage's resource 0xA0 + n.
constexpr u32 first_attribute_buffer = resource::attribute_buffer_0 - resource::vs_base;

/// The array base a position export uses for the position itself: POS0.
constexpr u32 position_export = export_position_base;

/// The array base of a pixel shader's depth export.
constexpr u32 depth_export = 61;

constexpr u32 color_targets = 8;

[[nodiscard]] std::string where(const CfInstruction& inst) {
    return fmt::format("instruction {} ({})", inst.index, cf_inst_name(inst));
}

[[nodiscard]] Swizzle export_swizzle(const CfInstruction& inst) {
    const CfExportWord1Swizzle swizzle{.raw = inst.word1};
    return {static_cast<Sel>(swizzle.sel_x.value()),
            static_cast<Sel>(swizzle.sel_y.value()),
            static_cast<Sel>(swizzle.sel_z.value()),
            static_cast<Sel>(swizzle.sel_w.value())};
}

/// A semantic's byte in a run of registers holding four each
/// (SPI_VS_OUT_ID).
[[nodiscard]] u32 packed_semantic(const RegisterFile& registers, u32 base, u32 index) {
    return (registers.read(base + (4 * (index / 4))) >> (8 * (index % 4))) & 0xFFU;
}

/// The register SQ_VTX_SEMANTIC maps `semantic` to: R(n + 1) for the first
/// n that names it. Nothing if none does.
[[nodiscard]] std::optional<u32> semantic_gpr(const RegisterFile& registers, u32 semantic) {
    if (semantic == semantic_unused) {
        return std::nullopt;
    }
    for (u32 n = 0; n < reg::sq_vtx_semantic_count; ++n) {
        if ((registers.read(reg::sq_vtx_semantic_0 + (4 * n)) & 0xFFU) == semantic) {
            return n + 1;
        }
    }
    return std::nullopt;
}

/// The vertex shader parameter whose SPI_VS_OUT_ID semantic is `semantic`.
[[nodiscard]] std::optional<u32> parameter_with(const RegisterFile& registers, u32 semantic) {
    if (semantic == semantic_unused) {
        return std::nullopt;
    }
    for (u32 index = 0; index < vs_parameter_count; ++index) {
        if (packed_semantic(registers, reg::spi_vs_out_id_0, index) == semantic) {
            return index;
        }
    }
    return std::nullopt;
}

/// The colour target pixel export `index` goes to: the index-th target with
/// any bit of CB_SHADER_MASK set (Cemu).
[[nodiscard]] std::optional<u32> color_target(const RegisterFile& registers, u32 index) {
    const u32 mask = registers.read(reg::cb_shader_mask);
    u32 enabled = 0;
    for (u32 target = 0; target < color_targets; ++target) {
        if (((mask >> (4 * target)) & 0xFU) == 0) {
            continue;
        }
        if (enabled == index) {
            return target;
        }
        ++enabled;
    }
    return std::nullopt;
}

// The ALU instructions translated so far, by number (decaf's table).
constexpr u32 alu_nop = 0x1A;
constexpr u32 op3_muladd = 0x10;      ///< then _M2, _M4, _D2
constexpr u32 op3_muladd_ieee = 0x14; ///< then _M2, _M4, _D2
constexpr u32 op3_muladd_ieee_d2 = 0x17;

/// An OP2's OMOD, and the order of each MULADD's four forms.
constexpr std::array output_modifiers{
    OutputModifier::None, OutputModifier::Times2, OutputModifier::Times4, OutputModifier::Half};

[[nodiscard]] std::optional<AluOpcode> op2_opcode(u32 number) noexcept {
    switch (number) {
        case 0x00:
            return AluOpcode::Add;
        case 0x01:
            return AluOpcode::Mul;
        case 0x02:
            return AluOpcode::MulIeee;
        case 0x03:
            return AluOpcode::Max;
        case 0x04:
            return AluOpcode::Min;
        case 0x19:
            return AluOpcode::Mov;
        case 0x50:
            return AluOpcode::Dot4;
        case 0x51:
            return AluOpcode::Dot4Ieee;
        default:
            return std::nullopt;
    }
}

[[nodiscard]] std::string_view opcode_name(AluOpcode opcode) noexcept {
    switch (opcode) {
        case AluOpcode::Mov:
            return "MOV";
        case AluOpcode::Add:
            return "ADD";
        case AluOpcode::Mul:
            return "MUL";
        case AluOpcode::MulIeee:
            return "MUL_IEEE";
        case AluOpcode::Max:
            return "MAX";
        case AluOpcode::Min:
            return "MIN";
        case AluOpcode::MulAdd:
            return "MULADD";
        case AluOpcode::MulAddIeee:
            return "MULADD_IEEE";
        case AluOpcode::Dot4:
            return "DOT4";
        case AluOpcode::Dot4Ieee:
            return "DOT4_IEEE";
    }
    return "?";
}

[[nodiscard]] u32 source_count(AluOpcode opcode) noexcept {
    switch (opcode) {
        case AluOpcode::Mov:
            return 1;
        case AluOpcode::MulAdd:
        case AluOpcode::MulAddIeee:
            return 3;
        default:
            return 2;
    }
}

class Builder {
public:
    Builder(const RegisterFile& registers, Shader& out, std::string& error)
        : registers_(registers), out_(out), error_(error),
          uniform_registers_((registers.read(reg::sq_config) & sq_config_dx9_consts) != 0) {}

    [[nodiscard]] bool fail(std::string message) {
        error_ = std::move(message);
        return false;
    }

    [[nodiscard]] bool not_yet(const CfInstruction& inst, std::string_view what) {
        return fail(fmt::format("{}: {} are not translated yet", where(inst), what));
    }

    /// What every program is checked for before its instructions: no
    /// subroutines yet, and no ALU or texture clause.
    [[nodiscard]] bool supported_shape(const ShaderProgram& program) {
        if (!program.control_flow.subroutines.empty()) {
            return fail(fmt::format("the program calls a subroutine at {}; subroutines are not "
                                    "translated yet",
                                    program.control_flow.subroutines.front().address));
        }
        return true;
    }

    [[nodiscard]] bool fetch_attributes(const ShaderProgram& fetch) {
        for (const CfInstruction& inst : fetch.control_flow.main) {
            if (inst.is(CfInst::Vtx) || inst.is(CfInst::VtxTc) || inst.is(CfInst::Return) ||
                inst.is(CfInst::Nop)) {
                continue;
            }
            return fail(
                fmt::format("the fetch shader's {} is not one a fetch shader has", where(inst)));
        }
        for (const DecodedClause& clause : fetch.clauses) {
            for (const VertexFetch& vertex : clause.vertex_fetches) {
                if (!attribute(clause.clause, vertex)) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool attribute(const Clause& clause, const VertexFetch& fetch) {
        const VtxWord0 word0 = fetch.fetch0();
        const VtxWord1 word1 = fetch.fetch1();
        const VtxWord2 word2 = fetch.fetch2();
        const std::string at =
            fmt::format("a fetch in the fetch shader's clause at {}", clause.cf_index);
        if (!fetch.is_semantic()) {
            return fail(fmt::format("{} is a plain fetch; those are not translated yet", at));
        }
        const std::optional<u32> gpr = semantic_gpr(registers_, word1.semantic_id);
        if (!gpr.has_value()) {
            return true; // no register takes this semantic: the fetch does nothing (decaf)
        }
        const u32 buffer = word0.buffer_id;
        if (buffer < first_attribute_buffer ||
            buffer >= first_attribute_buffer + resource::attribute_buffer_count) {
            return fail(
                fmt::format("{} reads resource {:#x}, not an attribute buffer", at, buffer));
        }
        if (word0.src_gpr != 0 || word0.src_rel) {
            return fail(fmt::format("{} takes its index from R{}, not R0; not translated yet",
                                    at,
                                    word0.src_gpr.value()));
        }
        if (word1.use_const_fields) {
            return fail(
                fmt::format("{} takes its format from the resource; not translated yet", at));
        }

        VertexAttribute attribute;
        switch (word0.fetch_type.value()) {
            case VtxFetchType::VertexData:
                if (word0.src_sel_x != Sel::X) {
                    return fail(
                        fmt::format("{} reads its vertex index from R0.{}; not "
                                    "translated yet",
                                    at,
                                    "xyzw"[static_cast<u32>(word0.src_sel_x.value()) & 3U]));
                }
                attribute.index = AttributeIndex::Vertex;
                break;
            case VtxFetchType::InstanceData:
                // R0.w is the instance index; R0.y and R0.z are it over
                // VGT_INSTANCE_STEP_RATE_0 and _1.
                switch (word0.src_sel_x.value()) {
                    case Sel::W:
                        attribute.index = AttributeIndex::Instance;
                        break;
                    case Sel::Y:
                        attribute.index = AttributeIndex::InstanceRate0;
                        break;
                    case Sel::Z:
                        attribute.index = AttributeIndex::InstanceRate1;
                        break;
                    default:
                        return fail(fmt::format("{} reads its instance index from R0.x", at));
                }
                break;
            default:
                return fail(fmt::format("{} has no index offset; not translated yet", at));
        }

        attribute.semantic = word1.semantic_id;
        attribute.gpr = *gpr;
        attribute.buffer = buffer - first_attribute_buffer;
        attribute.offset = word2.offset;
        attribute.bytes = word0.mega_fetch_count + 1;
        attribute.data_format = word1.data_format;
        attribute.num_format = word1.num_format_all;
        attribute.is_signed = word1.format_comp_signed;
        attribute.endian = word2.endian_swap;
        attribute.components = {word1.dst_sel_x.value(),
                                word1.dst_sel_y.value(),
                                word1.dst_sel_z.value(),
                                word1.dst_sel_w.value()};
        out_.attributes.push_back(attribute);
        return true;
    }

    /// An EXP or EXP_DONE, its burst one statement a register.
    [[nodiscard]] bool exports(const CfInstruction& inst) {
        const CfExportWord0 word0 = inst.export0();
        const CfExportWord1 word1 = inst.export1();
        if (word1.cf_inst != ExportInst::Export && word1.cf_inst != ExportInst::ExportDone) {
            return not_yet(inst, "memory exports");
        }
        if (word0.rw_rel) {
            return not_yet(inst, "exports from relative registers");
        }
        const Swizzle swizzle = export_swizzle(inst);
        const auto type = static_cast<ExportType>(word0.type.value());
        for (u32 burst = 0; burst <= word1.burst_count; ++burst) {
            const u32 base = word0.array_base + burst;
            const u32 gpr = word0.rw_gpr + burst;
            if (!one_export(inst, type, base, gpr, swizzle)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool
    one_export(const CfInstruction& inst, ExportType type, u32 base, u32 gpr, Swizzle swizzle) {
        if (out_.stage == Stage::Vertex) {
            if (type == ExportType::Position) {
                if (base != position_export) {
                    return not_yet(inst, "point-size and clip-distance exports");
                }
                out_.body.emplace_back(ExportPosition{.gpr = gpr, .swizzle = swizzle});
                return true;
            }
            if (type == ExportType::Parameter) {
                if (base >= vs_parameter_count) {
                    return fail(fmt::format("{} exports parameter {}, past the {} there are",
                                            where(inst),
                                            base,
                                            vs_parameter_count));
                }
                const u32 semantic = packed_semantic(registers_, reg::spi_vs_out_id_0, base);
                if (semantic != semantic_unused) { // unused: nothing receives it (Cemu)
                    out_.body.emplace_back(ExportParameter{
                        .index = base, .semantic = semantic, .gpr = gpr, .swizzle = swizzle});
                }
                return true;
            }
            return fail(fmt::format("{} is a pixel export in a vertex shader", where(inst)));
        }

        if (type != ExportType::Pixel) {
            return fail(fmt::format("{} is not a pixel export, in a pixel shader", where(inst)));
        }
        if (base == depth_export) {
            return not_yet(inst, "depth exports");
        }
        if (base >= color_targets) {
            return fail(fmt::format("{} exports PIX{}, which is no colour", where(inst), base));
        }
        if (const std::optional<u32> target = color_target(registers_, base)) {
            out_.body.emplace_back(
                ExportColor{.index = base, .target = *target, .gpr = gpr, .swizzle = swizzle});
        }
        return true; // no target enabled for it: nothing receives it
    }

    /// The main program's instructions. CALL_FS only in a vertex shader.
    [[nodiscard]] bool body(const ShaderProgram& program, const ShaderProgram* fetch) {
        for (const CfInstruction& inst : program.control_flow.main) {
            if (inst.is_alu()) {
                if (!alu_clause(program, inst)) {
                    return false;
                }
                continue;
            }
            if (inst.type() == CfInstType::Export) {
                if (!exports(inst)) {
                    return false;
                }
                continue;
            }
            if (inst.is(CfInst::Nop)) {
                continue;
            }
            if (inst.is(CfInst::CallFs) && out_.stage == Stage::Vertex) {
                if (fetch == nullptr) {
                    return fail(
                        fmt::format("{} calls a fetch shader, and there is none", where(inst)));
                }
                if (out_.attributes.empty() && !fetch_attributes(*fetch)) {
                    return false;
                }
                out_.body.emplace_back(FetchAttributes{});
                continue;
            }
            if (inst.is(CfInst::Tex)) {
                if (!texture_clause(program, inst)) {
                    return false;
                }
                continue;
            }
            if (inst.is(CfInst::Vtx) || inst.is(CfInst::VtxTc)) {
                return not_yet(inst, "vertex fetch clauses outside a fetch shader");
            }
            return not_yet(inst, "instructions of this kind");
        }
        return true;
    }

    /// A texture clause: each SAMPLE one statement.
    [[nodiscard]] bool texture_clause(const ShaderProgram& program, const CfInstruction& inst) {
        for (const DecodedClause& decoded : program.clauses) {
            if (decoded.clause.cf_index != inst.index) {
                continue;
            }
            for (std::size_t i = 0; i < decoded.texture_fetches.size(); ++i) {
                if (!sample(inst, i, decoded.texture_fetches[i])) {
                    return false;
                }
            }
            return true;
        }
        return fail(fmt::format("{} runs a clause the program does not hold", where(inst)));
    }

    [[nodiscard]] bool sample(const CfInstruction& inst, std::size_t index, const TextureFetch& f) {
        const std::string at = fmt::format("{}, fetch {}", where(inst), index);
        if (f.is_vertex_fetch()) {
            return fail(fmt::format("{} is a vertex fetch; vertex fetches in a texture clause are "
                                    "not translated yet",
                                    at));
        }
        if (f.inst() != TexInst::Sample) {
            return fail(fmt::format(
                "{} is {}; only SAMPLE is translated yet", at, tex_inst_name(f.inst())));
        }
        const TexWord0 word0 = f.tex0();
        const TexWord1 word1 = f.tex1();
        const TexWord2 word2 = f.tex2();
        if (word0.src_rel || word1.dst_rel) {
            return fail(fmt::format("{} uses a relative register; not translated yet", at));
        }
        if (!word1.coord_type_x || !word1.coord_type_y) {
            return fail(fmt::format("{} samples at texel coordinates; not translated yet", at));
        }
        if (word1.lod_bias != 0 || word2.offset_x != 0 || word2.offset_y != 0 ||
            word2.offset_z != 0) {
            return fail(fmt::format("{} has a LOD bias or offsets; not translated yet", at));
        }

        const u32 first = out_.stage == Stage::Vertex ? resource::vs_base : resource::ps_base;
        const TextureResource texture = texture_resource(registers_, first + word0.resource_id);
        if (!texture.valid()) {
            return fail(fmt::format("{} samples texture {}, which is not a valid texture",
                                    at,
                                    word0.resource_id.value()));
        }
        if (texture.word0.dim != TexDim::D2) {
            return fail(fmt::format("{} samples texture {}, which is not 2D (SQ_TEX_DIM {}); not "
                                    "translated yet",
                                    at,
                                    word0.resource_id.value(),
                                    static_cast<u32>(texture.word0.dim.value())));
        }
        const Swizzle destination{word1.dst_sel_x.value(),
                                  word1.dst_sel_y.value(),
                                  word1.dst_sel_z.value(),
                                  word1.dst_sel_w.value()};
        out_.body.emplace_back(
            SampleTexture{.texture = word0.resource_id,
                          .sampler = word2.sampler_id,
                          .src_gpr = word0.src_gpr,
                          .coordinates = {word2.src_sel_x.value(), word2.src_sel_y.value()},
                          .texture_select = {texture.word4.dst_sel_x.value(),
                                             texture.word4.dst_sel_y.value(),
                                             texture.word4.dst_sel_z.value(),
                                             texture.word4.dst_sel_w.value()},
                          .dst_gpr = word1.dst_gpr,
                          .swizzle = destination});
        if (word1.dst_gpr == 0) {
            for (std::size_t c = 0; c < destination.size(); ++c) {
                r0_written_[c] = r0_written_[c] || destination[c] != Sel::Mask;
            }
        }
        return true;
    }

    // --- ALU clauses ----------------------------------------------------------

    /// An ALU clause: each group one statement. PV and PS do not reach from
    /// one clause into the next.
    [[nodiscard]] bool alu_clause(const ShaderProgram& program, const CfInstruction& inst) {
        if (inst.alu1().cf_inst != AluCfInst::Alu) {
            return not_yet(inst, "ALU clauses that push, pop or branch");
        }
        for (const DecodedClause& decoded : program.clauses) {
            if (decoded.clause.cf_index != inst.index) {
                continue;
            }
            previous_vector_ = {};
            previous_scalar_ = false;
            for (std::size_t i = 0; i < decoded.alu_groups.size(); ++i) {
                if (!alu_group(inst, i, decoded.alu_groups[i])) {
                    return false;
                }
            }
            return true;
        }
        return fail(fmt::format("{} runs a clause the program does not hold", where(inst)));
    }

    [[nodiscard]] bool
    alu_group(const CfInstruction& inst, std::size_t index, const latte::AluGroup& group) {
        const std::string at = fmt::format("{}, group {}", where(inst), index);
        AluGroup out;
        for (const AluInstruction& instruction : group.instructions) {
            if (is_nop(instruction)) {
                continue; // no result: PV or PS of its unit is left unset
            }
            const std::optional<AluOperation> operation =
                alu_operation(at, inst, instruction, group);
            if (!operation.has_value()) {
                return false;
            }
            out.operations.push_back(*operation);
        }
        if (!reduction_complete(at, out)) {
            return false;
        }

        // What this group leaves for the next: its units' results, and its
        // writes.
        previous_vector_ = {};
        previous_scalar_ = false;
        for (const AluOperation& op : out.operations) {
            if (op.unit == AluUnit::T) {
                previous_scalar_ = true;
            } else {
                previous_vector_[static_cast<u32>(op.unit)] = true;
            }
            if (op.write && op.dst_gpr == 0) {
                r0_written_[op.dst_chan] = true;
            }
        }
        out_.body.emplace_back(std::move(out));
        return true;
    }

    [[nodiscard]] static bool is_nop(const AluInstruction& instruction) {
        return !instruction.is_op3() && instruction.op2().alu_inst == alu_nop;
    }

    /// A reduction takes the four vector units, each running it.
    [[nodiscard]] bool reduction_complete(const std::string& at, const AluGroup& group) {
        std::optional<AluOpcode> reduction;
        for (const AluOperation& op : group.operations) {
            if (op.opcode == AluOpcode::Dot4 || op.opcode == AluOpcode::Dot4Ieee) {
                reduction = op.opcode;
            }
        }
        if (!reduction.has_value()) {
            return true;
        }
        std::array<bool, 4> units{};
        for (const AluOperation& op : group.operations) {
            if (op.unit == AluUnit::T) {
                continue;
            }
            if (op.opcode != *reduction) {
                return fail(fmt::format("{} runs {} on the {} unit beside a reduction",
                                        at,
                                        opcode_name(op.opcode),
                                        "xyzw"[static_cast<u32>(op.unit)]));
            }
            units[static_cast<u32>(op.unit)] = true;
        }
        if (units != std::array{true, true, true, true}) {
            return fail(fmt::format(
                "{} runs {} on fewer than the four vector units", at, opcode_name(*reduction)));
        }
        return true;
    }

    /// One instruction, or nothing (with the error set) for one not
    /// translated yet.
    [[nodiscard]] std::optional<AluOperation> alu_operation(const std::string& group_at,
                                                            const CfInstruction& clause,
                                                            const AluInstruction& instruction,
                                                            const latte::AluGroup& group) {
        const std::string at = fmt::format("{}, {} on the {} unit",
                                           group_at,
                                           instruction.op.name,
                                           "xyzwt"[static_cast<u32>(instruction.unit)]);
        const AluWord0 word0 = instruction.alu0();
        const AluWord1 word1 = instruction.alu1();
        const bool op3 = instruction.is_op3();

        AluOperation operation;
        operation.unit = instruction.unit;
        if (op3) {
            const u32 number = instruction.op3().alu_inst;
            if (number < op3_muladd || number > op3_muladd_ieee_d2) {
                return refuse(fmt::format("{}: {} is not translated yet", at, instruction.op.name));
            }
            const u32 variant = (number - op3_muladd) % 4;
            operation.opcode = number < op3_muladd_ieee ? AluOpcode::MulAdd : AluOpcode::MulAddIeee;
            operation.modifier = output_modifiers[variant];
            operation.write = true;
        } else {
            const AluWord1Op2 op2 = instruction.op2();
            const std::optional<AluOpcode> opcode = op2_opcode(op2.alu_inst);
            if (!opcode.has_value()) {
                return refuse(fmt::format("{}: {} is not translated yet", at, instruction.op.name));
            }
            if (op2.update_pred || op2.update_execute_mask) {
                return refuse(fmt::format(
                    "{} updates the predicate or the execute mask; not translated yet", at));
            }
            operation.opcode = *opcode;
            operation.modifier = output_modifiers[op2.omod];
            operation.write = op2.write_mask;
        }
        if (word0.pred_sel != 0) {
            return refuse(fmt::format("{} is predicated; not translated yet", at));
        }
        if (word1.dst_rel) {
            return refuse(fmt::format("{} writes a relative register; not translated yet", at));
        }
        operation.dst_gpr = word1.dst_gpr;
        operation.dst_chan = word1.dst_chan;
        operation.clamp = word1.clamp;

        struct Raw {
            u32 sel;
            bool rel;
            u32 chan;
            bool neg;
            bool abs;
        };

        const std::array<Raw, 3> raw{
            Raw{word0.src0_sel, word0.src0_rel, word0.src0_chan, word0.src0_neg, false},
            Raw{word0.src1_sel, word0.src1_rel, word0.src1_chan, word0.src1_neg, false},
            Raw{instruction.op3().src2_sel,
                instruction.op3().src2_rel,
                instruction.op3().src2_chan,
                instruction.op3().src2_neg,
                false}};
        for (u32 i = 0; i < instruction.op.sources; ++i) {
            Raw source = raw[i];
            if (!op3 && i < 2) {
                source.abs = i == 0 ? instruction.op2().src0_abs : instruction.op2().src1_abs;
            }
            const std::optional<AluSource> decoded = alu_source(fmt::format("{}, source {}", at, i),
                                                                clause,
                                                                source.sel,
                                                                source.rel,
                                                                source.chan,
                                                                group);
            if (!decoded.has_value()) {
                return std::nullopt;
            }
            operation.sources[i] = *decoded;
            operation.sources[i].absolute = source.abs;
            operation.sources[i].negate = source.neg;
        }
        return operation;
    }

    [[nodiscard]] std::optional<AluSource> alu_source(const std::string& at,
                                                      const CfInstruction& clause,
                                                      u32 sel,
                                                      bool relative,
                                                      u32 chan,
                                                      const latte::AluGroup& group) {
        if (relative) {
            return refuse(fmt::format("{} is a relative register; not translated yet", at));
        }
        AluSource source;
        if (sel <= alu_src::register_last) {
            if (out_.stage == Stage::Vertex && sel == 0 && !r0_written_[chan]) {
                return refuse(fmt::format("{} reads R0.{} before the shader writes it: the vertex "
                                          "and instance indices, which are not translated yet",
                                          at,
                                          "xyzw"[chan]));
            }
            source.kind = SourceKind::Register;
            source.gpr = sel;
            source.chan = chan;
            return source;
        }
        if (sel >= alu_src::kcache0_first && sel <= alu_src::kcache1_last) {
            return kcache_source(at, clause, sel, chan);
        }
        if (sel >= alu_src::const_file_first && sel <= alu_src::const_file_last) {
            if (!uniform_registers_) {
                return refuse(fmt::format("{} reads uniform register C{}, and SQ_CONFIG says "
                                          "uniform blocks (DX9_CONSTS is clear)",
                                          at,
                                          sel - alu_src::const_file_first));
            }
            source.kind = SourceKind::UniformRegister;
            source.index = sel - alu_src::const_file_first;
            source.chan = chan;
            return source;
        }
        source.kind = SourceKind::Value;
        switch (sel) {
            case alu_src::zero:
                source.bits = 0x00000000;
                return source;
            case alu_src::one:
                source.bits = 0x3F800000;
                return source;
            case alu_src::one_int:
                source.bits = 1;
                return source;
            case alu_src::minus_one:
                source.bits = 0xFFFFFFFF;
                return source;
            case alu_src::half:
                source.bits = 0x3F000000;
                return source;
            case alu_src::literal:
                source.bits = group.literals.at(chan); // the decoder read every one
                return source;
            case alu_src::previous_vector:
                if (!previous_vector_[chan]) {
                    return refuse(fmt::format("{} reads PV.{}, which the clause's previous group "
                                              "did not set",
                                              at,
                                              "xyzw"[chan]));
                }
                source.kind = SourceKind::PreviousVector;
                source.chan = chan;
                return source;
            case alu_src::previous_scalar:
                if (!previous_scalar_) {
                    return refuse(fmt::format(
                        "{} reads PS, which the clause's previous group did not set", at));
                }
                source.kind = SourceKind::PreviousScalar;
                return source;
            default:
                return refuse(
                    fmt::format("{} reads source select {}; not translated yet", at, sel));
        }
    }

    /// KC0[n] or KC1[n]: constant KCACHE_ADDR * 16 + n of the cache's
    /// bank, if the clause locks that far.
    [[nodiscard]] std::optional<AluSource>
    kcache_source(const std::string& at, const CfInstruction& clause, u32 sel, u32 chan) {
        const bool second = sel >= alu_src::kcache1_first;
        const u32 n = sel - (second ? alu_src::kcache1_first : alu_src::kcache0_first);
        const CfAluWord0 word0 = clause.alu0();
        const CfAluWord1 word1 = clause.alu1();
        const KcacheMode mode = second ? word1.kcache_mode1.value() : word0.kcache_mode0.value();
        const u32 bank = second ? word0.kcache_bank1 : word0.kcache_bank0;
        const u32 address = second ? word1.kcache_addr1 : word1.kcache_addr0;
        const std::string name = fmt::format("KC{}[{}]", second ? 1 : 0, n);
        if (uniform_registers_) {
            return refuse(fmt::format("{} reads {}, a uniform block, and SQ_CONFIG says uniform "
                                      "registers (DX9_CONSTS is set)",
                                      at,
                                      name));
        }
        switch (mode) {
            case KcacheMode::Nop:
                return refuse(fmt::format(
                    "{} reads {}, and the clause locks nothing in that cache", at, name));
            case KcacheMode::Lock1:
                if (n >= 16) {
                    return refuse(fmt::format(
                        "{} reads {}, past the sixteen constants the clause locks", at, name));
                }
                break;
            case KcacheMode::Lock2:
                break;
            case KcacheMode::LockLoopIndex:
                return refuse(fmt::format(
                    "{} reads {}, locked from the loop index; not translated yet", at, name));
        }
        AluSource source;
        source.kind = SourceKind::UniformBlock;
        source.block = bank;
        source.index = (address * 16) + n;
        source.chan = chan;
        return source;
    }

    [[nodiscard]] std::nullopt_t refuse(std::string message) {
        error_ = std::move(message);
        return std::nullopt;
    }

    [[nodiscard]] bool pixel_inputs() {
        const SpiPsInControl0 control0{.raw = registers_.read(reg::spi_ps_in_control_0)};
        const SpiPsInControl1 control1{.raw = registers_.read(reg::spi_ps_in_control_1)};
        if (control0.param_gen != 0) {
            return fail("the pixel shader asks for point-sprite coordinates (SPI_PS_IN_CONTROL_0 "
                        "PARAM_GEN); not translated yet");
        }
        if (control1.front_face_ena || control1.gen_index_pix || control1.fixed_pt_position_ena) {
            return fail(
                "the pixel shader asks for the front face, the pixel index or a fixed-point "
                "position (SPI_PS_IN_CONTROL_1); not translated yet");
        }
        const u32 count = control0.num_interp;
        if (count > spi_ps_input_cntl_count) {
            return fail(fmt::format("the pixel shader has {} inputs; there are {} at most",
                                    count,
                                    spi_ps_input_cntl_count));
        }
        for (u32 index = 0; index < count; ++index) {
            const SpiPsInputCntl control{
                .raw = registers_.read(reg::spi_ps_input_cntl_0 + (4 * index))};
            if (control.cyl_wrap != 0 || control.pt_sprite_tex) {
                return fail(fmt::format("pixel shader input {} asks for cylindrical wrapping or a "
                                        "point-sprite coordinate; not translated yet",
                                        index));
            }
            PixelInput input;
            input.gpr = index;
            input.semantic = control.semantic;
            input.default_value = control.default_val;
            input.centroid = control.sel_centroid;
            input.sample = control.sel_sample;
            if (control.flat_shade) {
                input.interpolation = Interpolation::Flat;
            } else if (control.sel_linear) {
                input.interpolation = Interpolation::Linear;
            }
            if (control0.position_ena && control0.position_addr == index) {
                input.source = InputSource::Position;
            } else if (const std::optional<u32> parameter =
                           parameter_with(registers_, control.semantic)) {
                input.source = InputSource::Parameter;
                input.parameter = *parameter;
            } else {
                input.source = InputSource::Default;
            }
            out_.inputs.push_back(input);
        }
        return true;
    }

private:
    const RegisterFile& registers_;
    Shader& out_;
    std::string& error_;
    std::array<bool, 4> previous_vector_{}; ///< which of PV.x to .w the last group set
    bool previous_scalar_ = false;          ///< whether it set PS
    std::array<bool, 4> r0_written_{};      ///< a vertex shader's R0, component by component
    bool uniform_registers_ = false;        ///< SQ_CONFIG's DX9_CONSTS
};

// --- describe ----------------------------------------------------------------

[[nodiscard]] std::string swizzle_text(const Swizzle& swizzle) {
    constexpr std::string_view chars = "xyzw01?_";
    std::string out;
    for (const Sel sel : swizzle) {
        out += chars[static_cast<u32>(sel) & 7U];
    }
    return out;
}

[[nodiscard]] std::string_view index_text(AttributeIndex index) {
    switch (index) {
        case AttributeIndex::Vertex:
            return "per vertex";
        case AttributeIndex::Instance:
            return "per instance";
        case AttributeIndex::InstanceRate0:
            return "per instance / step rate 0";
        case AttributeIndex::InstanceRate1:
            return "per instance / step rate 1";
    }
    return "?";
}

[[nodiscard]] std::string_view interpolation_text(Interpolation interpolation) {
    switch (interpolation) {
        case Interpolation::Perspective:
            return "perspective";
        case Interpolation::Linear:
            return "linear";
        case Interpolation::Flat:
            return "flat";
    }
    return "?";
}

struct StatementText {
    std::string& out;

    void operator()(const FetchAttributes&) const { out += "  fetch attributes\n"; }

    void operator()(const SampleTexture& s) const {
        constexpr std::string_view chars = "xyzw01?_";
        fmt::format_to(std::back_inserter(out),
                       "  R{}.{} <- sample texture {} ({}) with sampler {} at R{}.{}{}\n",
                       s.dst_gpr,
                       swizzle_text(s.swizzle),
                       s.texture,
                       swizzle_text(s.texture_select),
                       s.sampler,
                       s.src_gpr,
                       chars[static_cast<u32>(s.coordinates[0]) & 7U],
                       chars[static_cast<u32>(s.coordinates[1]) & 7U]);
    }

    void operator()(const AluGroup& group) const {
        const auto to = std::back_inserter(out);
        out += "  alu group\n";
        for (const AluOperation& op : group.operations) {
            fmt::format_to(to, "    {}: ", "xyzwt"[static_cast<u32>(op.unit)]);
            if (op.write) {
                fmt::format_to(to, "R{}.{}", op.dst_gpr, "xyzw"[op.dst_chan & 3U]);
            } else {
                out += "____";
            }
            fmt::format_to(to, " <- {}(", opcode_name(op.opcode));
            for (u32 i = 0; i < source_count(op.opcode); ++i) {
                if (i != 0) {
                    out += ", ";
                }
                source_text(op.sources[i]);
            }
            out += ')';
            switch (op.modifier) {
                case OutputModifier::None:
                    break;
                case OutputModifier::Times2:
                    out += " * 2";
                    break;
                case OutputModifier::Times4:
                    out += " * 4";
                    break;
                case OutputModifier::Half:
                    out += " / 2";
                    break;
            }
            if (op.clamp) {
                out += ", clamped";
            }
            out += '\n';
        }
    }

    void source_text(const AluSource& s) const {
        const auto to = std::back_inserter(out);
        if (s.negate) {
            out += '-';
        }
        if (s.absolute) {
            out += '|';
        }
        switch (s.kind) {
            case SourceKind::Register:
                fmt::format_to(to, "R{}.{}", s.gpr, "xyzw"[s.chan & 3U]);
                break;
            case SourceKind::Value:
                fmt::format_to(to, "0x{:08X}", s.bits);
                break;
            case SourceKind::PreviousVector:
                fmt::format_to(to, "PV.{}", "xyzw"[s.chan & 3U]);
                break;
            case SourceKind::PreviousScalar:
                out += "PS";
                break;
            case SourceKind::UniformRegister:
                fmt::format_to(to, "C{}.{}", s.index, "xyzw"[s.chan & 3U]);
                break;
            case SourceKind::UniformBlock:
                fmt::format_to(to, "UB{}[{}].{}", s.block, s.index, "xyzw"[s.chan & 3U]);
                break;
        }
        if (s.absolute) {
            out += '|';
        }
    }

    void operator()(const ExportPosition& e) const {
        fmt::format_to(
            std::back_inserter(out), "  position <- R{}.{}\n", e.gpr, swizzle_text(e.swizzle));
    }

    void operator()(const ExportParameter& e) const {
        fmt::format_to(std::back_inserter(out),
                       "  parameter {} (semantic {}) <- R{}.{}\n",
                       e.index,
                       e.semantic,
                       e.gpr,
                       swizzle_text(e.swizzle));
    }

    void operator()(const ExportColor& e) const {
        fmt::format_to(std::back_inserter(out),
                       "  colour {} (target {}) <- R{}.{}\n",
                       e.index,
                       e.target,
                       e.gpr,
                       swizzle_text(e.swizzle));
    }
};

} // namespace

bool build_vertex_shader(const ShaderProgram& vertex,
                         const ShaderProgram* fetch,
                         const RegisterFile& registers,
                         Shader& out,
                         std::string& error) {
    out = Shader{.stage = Stage::Vertex, .attributes = {}, .inputs = {}, .body = {}};
    Builder builder{registers, out, error};
    if (!builder.supported_shape(vertex)) {
        return false;
    }
    if (fetch != nullptr && !builder.supported_shape(*fetch)) {
        return false;
    }
    return builder.body(vertex, fetch);
}

bool build_pixel_shader(const ShaderProgram& pixel,
                        const RegisterFile& registers,
                        Shader& out,
                        std::string& error) {
    out = Shader{.stage = Stage::Pixel, .attributes = {}, .inputs = {}, .body = {}};
    Builder builder{registers, out, error};
    return builder.supported_shape(pixel) && builder.pixel_inputs() && builder.body(pixel, nullptr);
}

bool UniformUse::any() const noexcept {
    return registers || std::ranges::any_of(blocks, [](bool read) { return read; });
}

UniformUse uniform_use(const Shader& shader) {
    UniformUse use;
    for (const Statement& statement : shader.body) {
        const auto* group = std::get_if<AluGroup>(&statement);
        if (group == nullptr) {
            continue;
        }
        for (const AluOperation& op : group->operations) {
            for (u32 i = 0; i < source_count(op.opcode); ++i) {
                const AluSource& source = op.sources[i];
                if (source.kind == SourceKind::UniformRegister) {
                    use.registers = true;
                } else if (source.kind == SourceKind::UniformBlock &&
                           source.block < use.blocks.size()) {
                    use.blocks[source.block] = true;
                }
            }
        }
    }
    return use;
}

std::string describe(const Shader& shader) {
    std::string out = shader.stage == Stage::Vertex ? "vertex shader\n" : "pixel shader\n";
    const auto to = std::back_inserter(out);
    for (const VertexAttribute& a : shader.attributes) {
        fmt::format_to(to,
                       "  attribute R{}.{} <- semantic {}: buffer {} + {}, {} bytes as FMT {:#x} "
                       "{}{}, endian swap {}, {}\n",
                       a.gpr,
                       swizzle_text(a.components),
                       a.semantic,
                       a.buffer,
                       a.offset,
                       a.bytes,
                       a.data_format,
                       a.num_format == NumFormat::Norm  ? "norm"
                       : a.num_format == NumFormat::Int ? "int"
                                                        : "scaled",
                       a.is_signed ? " signed" : "",
                       static_cast<u32>(a.endian),
                       index_text(a.index));
    }
    for (const PixelInput& input : shader.inputs) {
        fmt::format_to(to, "  input R{} <- ", input.gpr);
        switch (input.source) {
            case InputSource::Parameter:
                fmt::format_to(to, "parameter {} (semantic {})", input.parameter, input.semantic);
                break;
            case InputSource::Position:
                fmt::format_to(to, "position");
                break;
            case InputSource::Default:
                fmt::format_to(
                    to, "default {} (semantic {} unmatched)", input.default_value, input.semantic);
                break;
        }
        fmt::format_to(to,
                       ", {}{}{}\n",
                       interpolation_text(input.interpolation),
                       input.centroid ? ", centroid" : "",
                       input.sample ? ", per sample" : "");
    }
    for (const Statement& statement : shader.body) {
        std::visit(StatementText{out}, statement);
    }
    return out;
}

} // namespace wiiu::latte::ir
