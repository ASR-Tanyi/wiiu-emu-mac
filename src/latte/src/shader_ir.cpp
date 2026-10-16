// Phase 8.6 — see shader_ir.hpp.

#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_ir.hpp>
#include <wiiu/latte/shader_state.hpp>

#include <fmt/format.h>

#include <iterator>
#include <optional>
#include <string_view>

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

class Builder {
public:
    Builder(const RegisterFile& registers, Shader& out, std::string& error)
        : registers_(registers), out_(out), error_(error) {}

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
                return not_yet(inst, "ALU clauses");
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
                return not_yet(inst, "texture clauses");
            }
            if (inst.is(CfInst::Vtx) || inst.is(CfInst::VtxTc)) {
                return not_yet(inst, "vertex fetch clauses outside a fetch shader");
            }
            return not_yet(inst, "instructions of this kind");
        }
        return true;
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
