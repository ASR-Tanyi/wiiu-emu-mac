// Phase 8.6 — see msl.hpp.

#include <wiiu/latte/msl.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace wiiu::latte::msl {

namespace {

// The 32-bit float formats (R600 FMT_* numbers, decaf's table).
constexpr u32 fmt_32_float = 0x0E;
constexpr u32 fmt_32_32_float = 0x1E;
constexpr u32 fmt_32_32_32_32_float = 0x23;
constexpr u32 fmt_32_32_32_float = 0x30;

[[nodiscard]] std::optional<u32> float32_components(u32 format) noexcept {
    switch (format) {
        case fmt_32_float:
            return 1;
        case fmt_32_32_float:
            return 2;
        case fmt_32_32_32_float:
            return 3;
        case fmt_32_32_32_32_float:
            return 4;
        default:
            return std::nullopt;
    }
}

constexpr std::string_view prelude = R"msl(#include <metal_stdlib>
using namespace metal;
)msl";

// The fetch constants and the one fetch routine every vertex shader shares.
// `FetchConstants` must match msl.hpp's, field for field.
constexpr std::string_view fetch_helpers = R"msl(
struct FetchConstants {
    uint stride[16];
    uint size[16];
    uint step_rate[2];
    uint pad[2];
};

// `count` 32-bit components of element `index` of an attribute buffer,
// `offset` bytes into the element, each put together from its bytes: no
// alignment is assumed, and the order is big-endian for ENDIAN_SWAP 8IN32,
// as stored otherwise. A component past the buffer's size reads zero.
static uint4 fetch32(device const uchar* buffer, uint size, uint stride, uint index,
                     uint offset, uint count, bool big_endian) {
    uint4 value = uint4(0u);
    const ulong base = ulong(index) * ulong(stride) + ulong(offset);
    for (uint c = 0u; c < count; ++c) {
        const ulong at = base + ulong(4u * c);
        if (at + 4ul > ulong(size)) {
            break;
        }
        const uint b0 = buffer[at];
        const uint b1 = buffer[at + 1ul];
        const uint b2 = buffer[at + 2ul];
        const uint b3 = buffer[at + 3ul];
        value[c] = big_endian ? ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
                              : (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
    return value;
}
)msl";

// The ALU's helpers, each written into a shader that uses it. DX9's
// multiply: zero times anything -- infinity and NaN too -- is zero. MAX and
// MIN as R700 defines them, by comparison, so a NaN picks the second source.
constexpr std::string_view mul_dx9_helper = R"msl(
static float mul_dx9(float a, float b) {
    return (a == 0.0f || b == 0.0f) ? 0.0f : a * b;
}
)msl";

constexpr std::string_view max_helper = R"msl(
static float max_r700(float a, float b) {
    return a >= b ? a : b;
}
)msl";

constexpr std::string_view min_helper = R"msl(
static float min_r700(float a, float b) {
    return a < b ? a : b;
}
)msl";

/// What a shader's ALU groups need declared.
struct AluUse {
    bool any = false; ///< PV and PS
    bool mul_dx9 = false;
    bool max = false;
    bool min = false;
};

void emit_alu_helpers(std::string& out, const AluUse& use) {
    if (use.mul_dx9) {
        out += mul_dx9_helper;
    }
    if (use.max) {
        out += max_helper;
    }
    if (use.min) {
        out += min_helper;
    }
}

constexpr std::string_view chans = "xyzw";

/// PV and PS, declared with the registers in a shader that has ALU groups.
constexpr std::string_view previous_results = "    float4 PV = float4(0.0);\n"
                                              "    float PS = 0.0;\n";

/// A source's value, with its absolute value and negation.
[[nodiscard]] std::string alu_source(const ir::AluSource& source) {
    std::string value;
    switch (source.kind) {
        case ir::SourceKind::Register:
            value = fmt::format("R{}.{}", source.gpr, chans[source.chan & 3U]);
            break;
        case ir::SourceKind::Value:
            switch (source.bits) {
                case 0x00000000:
                    value = "0.0f";
                    break;
                case 0x3F800000:
                    value = "1.0f";
                    break;
                case 0x3F000000:
                    value = "0.5f";
                    break;
                default:
                    value = fmt::format("as_type<float>(0x{:08X}u)", source.bits);
                    break;
            }
            break;
        case ir::SourceKind::PreviousVector:
            value = fmt::format("PV.{}", chans[source.chan & 3U]);
            break;
        case ir::SourceKind::PreviousScalar:
            value = "PS";
            break;
        case ir::SourceKind::UniformRegister:
            value = fmt::format("uniforms[{}].{}", source.index, chans[source.chan & 3U]);
            break;
        case ir::SourceKind::UniformBlock:
            value = fmt::format("ub{}[{}].{}", source.block, source.index, chans[source.chan & 3U]);
            break;
    }
    if (source.absolute) {
        value = fmt::format("abs({})", value);
    }
    if (source.negate) {
        value = fmt::format("-{}", value);
    }
    return value;
}

/// An operation's result, before its modifier and clamp. A reduction's is
/// the reduction's, from the four operations' sources.
[[nodiscard]] std::string alu_result(const ir::AluOperation& op, const ir::AluGroup& group) {
    const auto s = [&op](std::size_t i) { return alu_source(op.sources[i]); };
    switch (op.opcode) {
        case ir::AluOpcode::Mov:
            return s(0);
        case ir::AluOpcode::Add:
            return fmt::format("{} + {}", s(0), s(1));
        case ir::AluOpcode::Mul:
            return fmt::format("mul_dx9({}, {})", s(0), s(1));
        case ir::AluOpcode::MulIeee:
            return fmt::format("{} * {}", s(0), s(1));
        case ir::AluOpcode::Max:
            return fmt::format("max_r700({}, {})", s(0), s(1));
        case ir::AluOpcode::Min:
            return fmt::format("min_r700({}, {})", s(0), s(1));
        case ir::AluOpcode::MulAdd:
            return fmt::format("mul_dx9({}, {}) + {}", s(0), s(1), s(2));
        case ir::AluOpcode::MulAddIeee:
            return fmt::format("{} * {} + {}", s(0), s(1), s(2));
        case ir::AluOpcode::Dot4:
        case ir::AluOpcode::Dot4Ieee: {
            std::array<std::string, 4> terms;
            for (const ir::AluOperation& unit : group.operations) {
                if (unit.unit == AluUnit::T) {
                    continue;
                }
                const std::string a = alu_source(unit.sources[0]);
                const std::string b = alu_source(unit.sources[1]);
                terms[static_cast<u32>(unit.unit)] = op.opcode == ir::AluOpcode::Dot4
                                                         ? fmt::format("mul_dx9({}, {})", a, b)
                                                         : fmt::format("{} * {}", a, b);
            }
            return fmt::format("{} + {} + {} + {}", terms[0], terms[1], terms[2], terms[3]);
        }
    }
    return "0.0f";
}

/// The parameters a shader's uniforms are bound at. False for a block past
/// the slots there are.
[[nodiscard]] bool
uniform_parameters(const ir::Shader& shader, std::vector<std::string>& out, std::string& error) {
    const ir::UniformUse use = ir::uniform_use(shader);
    if (use.registers) {
        out.push_back(
            fmt::format("constant float4* uniforms [[buffer({})]]", uniform_registers_slot));
    }
    for (u32 block = 0; block < use.blocks.size(); ++block) {
        if (!use.blocks[block]) {
            continue;
        }
        if (block >= max_uniform_blocks) {
            error = fmt::format("uniform block {} is past the {} Metal has buffer slots for; not "
                                "emitted yet",
                                block,
                                max_uniform_blocks);
            return false;
        }
        out.push_back(fmt::format(
            "constant float4* ub{} [[buffer({})]]", block, first_uniform_block_slot + block));
    }
    return true;
}

/// One group: every result into its own constant, from sources read before
/// any is written; then the writes; then PV and PS for the next group.
void emit_alu_group(std::string& out, const ir::AluGroup& group) {
    const auto to = std::back_inserter(out);
    constexpr std::string_view units = "xyzwt";
    out += "    {\n";
    bool reduced = false;
    for (const ir::AluOperation& op : group.operations) {
        const bool reduction =
            op.opcode == ir::AluOpcode::Dot4 || op.opcode == ir::AluOpcode::Dot4Ieee;
        std::string value;
        if (reduction) {
            if (!reduced) {
                fmt::format_to(to, "        const float reduction = {};\n", alu_result(op, group));
                reduced = true;
            }
            value = "reduction";
        } else {
            value = alu_result(op, group);
        }
        switch (op.modifier) {
            case ir::OutputModifier::None:
                break;
            case ir::OutputModifier::Times2:
                value = fmt::format("({}) * 2.0f", value);
                break;
            case ir::OutputModifier::Times4:
                value = fmt::format("({}) * 4.0f", value);
                break;
            case ir::OutputModifier::Half:
                value = fmt::format("({}) * 0.5f", value);
                break;
        }
        if (op.clamp) {
            value = fmt::format("clamp({}, 0.0f, 1.0f)", value);
        }
        fmt::format_to(
            to, "        const float alu_{} = {};\n", units[static_cast<u32>(op.unit)], value);
    }
    for (const ir::AluOperation& op : group.operations) {
        if (op.write) {
            fmt::format_to(to,
                           "        R{}.{} = alu_{};\n",
                           op.dst_gpr,
                           chans[op.dst_chan & 3U],
                           units[static_cast<u32>(op.unit)]);
        }
    }
    for (const ir::AluOperation& op : group.operations) {
        if (op.unit == AluUnit::T) {
            out += "        PS = alu_t;\n";
        } else {
            fmt::format_to(to, "        PV.{0} = alu_{0};\n", units[static_cast<u32>(op.unit)]);
        }
    }
    out += "    }\n";
}

/// The registers an ALU group reads and writes, and the helpers it uses.
/// False for a destination no register has.
[[nodiscard]] bool
scan_alu(const ir::AluGroup& group, std::set<u32>& registers, AluUse& use, std::string& error) {
    use.any = true;
    for (const ir::AluOperation& op : group.operations) {
        switch (op.opcode) {
            case ir::AluOpcode::Mul:
            case ir::AluOpcode::MulAdd:
            case ir::AluOpcode::Dot4:
                use.mul_dx9 = true;
                break;
            case ir::AluOpcode::Max:
                use.max = true;
                break;
            case ir::AluOpcode::Min:
                use.min = true;
                break;
            default:
                break;
        }
        if (op.write) {
            if (op.dst_gpr > 127 || op.dst_chan > 3) {
                error = fmt::format("an ALU result is written to R{}.{}", op.dst_gpr, op.dst_chan);
                return false;
            }
            registers.insert(op.dst_gpr);
        }
        for (const ir::AluSource& source : op.sources) {
            if (source.kind == ir::SourceKind::Register) {
                registers.insert(source.gpr);
            }
        }
    }
    return true;
}

/// Component `sel` of `source`, or nothing for one not written.
[[nodiscard]] std::optional<std::string> component(Sel sel, std::string_view source) {
    switch (sel) {
        case Sel::X:
            return fmt::format("{}.x", source);
        case Sel::Y:
            return fmt::format("{}.y", source);
        case Sel::Z:
            return fmt::format("{}.z", source);
        case Sel::W:
            return fmt::format("{}.w", source);
        case Sel::Zero:
            return std::string{"0.0"};
        case Sel::One:
            return std::string{"1.0"};
        case Sel::Mask:
            return std::nullopt;
    }
    return std::nullopt; // 6: no select; the builders below refuse it first
}

[[nodiscard]] bool valid(const ir::Swizzle& swizzle) noexcept {
    return std::none_of(
        swizzle.begin(), swizzle.end(), [](Sel sel) { return static_cast<u32>(sel) == 6; });
}

/// `destination.c = <selected component of source>;` for each component
/// written.
void assign(std::string& out,
            std::string_view indent,
            std::string_view destination,
            const ir::Swizzle& swizzle,
            std::string_view source) {
    constexpr std::string_view names = "xyzw";
    for (std::size_t i = 0; i < swizzle.size(); ++i) {
        if (const std::optional<std::string> value = component(swizzle[i], source)) {
            fmt::format_to(
                std::back_inserter(out), "{}{}.{} = {};\n", indent, destination, names[i], *value);
        }
    }
}

[[nodiscard]] std::string index_expression(ir::AttributeIndex index) {
    switch (index) {
        case ir::AttributeIndex::Vertex:
            return "vid";
        case ir::AttributeIndex::Instance:
            return "iid";
        case ir::AttributeIndex::InstanceRate0:
            return "iid / max(fetch.step_rate[0], 1u)";
        case ir::AttributeIndex::InstanceRate1:
            return "iid / max(fetch.step_rate[1], 1u)";
    }
    return "vid";
}

/// A sample: the texture's value at the source's two components, through
/// the resource's selects (a masked one reads zero), then into the
/// destination's components through the fetch's.
void emit_sample(std::string& out, const ir::SampleTexture& sample) {
    const std::string source = fmt::format("R{}", sample.src_gpr);
    const auto either = [](const std::optional<std::string>& value) {
        return value.value_or("0.0");
    };
    const auto to = std::back_inserter(out);
    fmt::format_to(to,
                   "    {{\n"
                   "        const float4 t = tex{}.sample(smp{}, float2({}, {}));\n"
                   "        const float4 v = float4({}, {}, {}, {});\n",
                   sample.texture,
                   sample.sampler,
                   either(component(sample.coordinates[0], source)),
                   either(component(sample.coordinates[1], source)),
                   either(component(sample.texture_select[0], "t")),
                   either(component(sample.texture_select[1], "t")),
                   either(component(sample.texture_select[2], "t")),
                   either(component(sample.texture_select[3], "t")));
    assign(out, "        ", fmt::format("R{}", sample.dst_gpr), sample.swizzle, "v");
    out += "    }\n";
}

void declare_registers(std::string& out, const std::set<u32>& registers) {
    for (const u32 gpr : registers) {
        fmt::format_to(std::back_inserter(out), "    float4 R{} = float4(0.0);\n", gpr);
    }
}

[[nodiscard]] std::string_view qualifier(const ir::PixelInput& input) {
    switch (input.interpolation) {
        case ir::Interpolation::Flat:
            return " [[flat]]";
        case ir::Interpolation::Linear:
            return input.centroid ? " [[centroid_no_perspective]]"
                   : input.sample ? " [[sample_no_perspective]]"
                                  : " [[center_no_perspective]]";
        case ir::Interpolation::Perspective:
            return input.centroid ? " [[centroid_perspective]]"
                   : input.sample ? " [[sample_perspective]]"
                                  : " [[center_perspective]]";
    }
    return "";
}

[[nodiscard]] std::string default_value(u32 value) {
    switch (value & 3U) {
        case 0:
            return "float4(0.0, 0.0, 0.0, 0.0)";
        case 1:
            return "float4(0.0, 0.0, 0.0, 1.0)";
        case 2:
            return "float4(1.0, 1.0, 1.0, 0.0)";
        default:
            return "float4(1.0, 1.0, 1.0, 1.0)";
    }
}

} // namespace

bool emit_vertex(const ir::Shader& vertex,
                 const ir::Shader* pixel,
                 std::string& out,
                 std::string& error,
                 const VertexOptions& options) {
    out.clear();
    if (vertex.stage != ir::Stage::Vertex) {
        error = "emit_vertex was given a pixel shader";
        return false;
    }

    std::set<u32> registers;
    std::set<u32> buffers;
    AluUse alu;
    for (const ir::VertexAttribute& attribute : vertex.attributes) {
        if (!float32_components(attribute.data_format).has_value() ||
            attribute.num_format != NumFormat::Scaled) {
            error = fmt::format("attribute R{} (semantic {}) is FMT {:#x} read as {}; only "
                                "32-bit float formats are emitted yet",
                                attribute.gpr,
                                attribute.semantic,
                                attribute.data_format,
                                static_cast<u32>(attribute.num_format));
            return false;
        }
        if (attribute.endian != Endian::None && attribute.endian != Endian::Swap8In32) {
            error = fmt::format("attribute R{} (semantic {}) is read with byte swap {}; only none "
                                "and 8IN32 are emitted yet",
                                attribute.gpr,
                                attribute.semantic,
                                static_cast<u32>(attribute.endian));
            return false;
        }
        if (!valid(attribute.components)) {
            error = fmt::format("attribute R{} has a component select of 6", attribute.gpr);
            return false;
        }
        registers.insert(attribute.gpr);
        buffers.insert(attribute.buffer);
    }

    // Outputs: every semantic the shader exports, and every one the pixel
    // shader reads.
    std::set<u32> semantics;
    for (const ir::Statement& statement : vertex.body) {
        if (const auto* position = std::get_if<ir::ExportPosition>(&statement)) {
            registers.insert(position->gpr);
            if (!valid(position->swizzle)) {
                error = "the position export has a component select of 6";
                return false;
            }
        } else if (const auto* parameter = std::get_if<ir::ExportParameter>(&statement)) {
            registers.insert(parameter->gpr);
            semantics.insert(parameter->semantic);
            if (!valid(parameter->swizzle)) {
                error = fmt::format("parameter {} has a component select of 6", parameter->index);
                return false;
            }
        } else if (std::holds_alternative<ir::ExportColor>(statement)) {
            error = "a colour export in a vertex shader";
            return false;
        } else if (std::holds_alternative<ir::SampleTexture>(statement)) {
            error = "sampling a texture in a vertex shader is not emitted yet";
            return false;
        } else if (const auto* group = std::get_if<ir::AluGroup>(&statement)) {
            if (!scan_alu(*group, registers, alu, error)) {
                return false;
            }
        }
    }
    if (pixel != nullptr) {
        for (const ir::PixelInput& input : pixel->inputs) {
            if (input.source == ir::InputSource::Parameter) {
                semantics.insert(input.semantic);
            }
        }
    }

    std::vector<std::string> uniforms;
    if (!uniform_parameters(vertex, uniforms, error)) {
        return false;
    }

    const auto to = std::back_inserter(out);
    out += prelude;
    out += fetch_helpers;
    emit_alu_helpers(out, alu);
    out += "\nstruct VertexOut {\n    float4 position [[position]];\n";
    for (const u32 semantic : semantics) {
        fmt::format_to(to, "    float4 sem{0} [[user(sem{0})]];\n", semantic);
    }
    out += "};\n\n";

    fmt::format_to(to,
                   "vertex VertexOut {}(uint vid [[vertex_id]],\n"
                   "                    uint iid [[instance_id]],\n"
                   "                    constant FetchConstants& fetch [[buffer({})]]",
                   vertex_entry,
                   fetch_constants_slot);
    for (const u32 buffer : buffers) {
        fmt::format_to(to,
                       ",\n                    device const uchar* attrib{} [[buffer({})]]",
                       buffer,
                       first_attribute_buffer_slot + buffer);
    }
    for (const std::string& parameter : uniforms) {
        fmt::format_to(to, ",\n                    {}", parameter);
    }
    out += ") {\n";
    declare_registers(out, registers);
    if (alu.any) {
        out += previous_results;
    }
    out += "    VertexOut out = {};\n";

    for (const ir::Statement& statement : vertex.body) {
        if (std::holds_alternative<ir::FetchAttributes>(statement)) {
            for (const ir::VertexAttribute& a : vertex.attributes) {
                fmt::format_to(to,
                               "    {{\n"
                               "        const float4 v = as_type<float4>(fetch32(attrib{0}, "
                               "fetch.size[{0}], fetch.stride[{0}], {1}, {2}u, {3}u, {4}));\n",
                               a.buffer,
                               index_expression(a.index),
                               a.offset,
                               *float32_components(a.data_format),
                               a.endian == Endian::Swap8In32 ? "true" : "false");
                assign(out, "        ", fmt::format("R{}", a.gpr), a.components, "v");
                out += "    }\n";
            }
        } else if (const auto* group = std::get_if<ir::AluGroup>(&statement)) {
            emit_alu_group(out, *group);
        } else if (const auto* position = std::get_if<ir::ExportPosition>(&statement)) {
            assign(
                out, "    ", "out.position", position->swizzle, fmt::format("R{}", position->gpr));
        } else if (const auto* parameter = std::get_if<ir::ExportParameter>(&statement)) {
            assign(out,
                   "    ",
                   fmt::format("out.sem{}", parameter->semantic),
                   parameter->swizzle,
                   fmt::format("R{}", parameter->gpr));
        }
    }
    if (options.depth_from_minus_one) {
        out += "    out.position.z = (out.position.z + out.position.w) * 0.5;\n";
    }
    out += "    return out;\n}\n";
    return true;
}

bool emit_pixel(const ir::Shader& pixel, std::string& out, std::string& error) {
    out.clear();
    if (pixel.stage != ir::Stage::Pixel) {
        error = "emit_pixel was given a vertex shader";
        return false;
    }

    std::set<u32> registers;
    std::map<u32, std::string_view> semantics; // semantic -> qualifier
    bool position = false;
    for (const ir::PixelInput& input : pixel.inputs) {
        registers.insert(input.gpr);
        if (input.source == ir::InputSource::Position) {
            position = true;
        } else if (input.source == ir::InputSource::Parameter) {
            const std::string_view wanted = qualifier(input);
            const auto [entry, inserted] = semantics.emplace(input.semantic, wanted);
            if (!inserted && entry->second != wanted) {
                error = fmt::format("semantic {} is read twice with two interpolations; not "
                                    "emitted yet",
                                    input.semantic);
                return false;
            }
        }
    }
    std::set<u32> targets;
    std::set<u32> textures;
    std::set<u32> samplers;
    AluUse alu;
    for (const ir::Statement& statement : pixel.body) {
        if (const auto* color = std::get_if<ir::ExportColor>(&statement)) {
            registers.insert(color->gpr);
            targets.insert(color->target);
            if (!valid(color->swizzle)) {
                error = fmt::format("colour {} has a component select of 6", color->index);
                return false;
            }
        } else if (const auto* sample = std::get_if<ir::SampleTexture>(&statement)) {
            if (sample->sampler >= max_samplers) {
                error = fmt::format("sampler {} is past the {} Metal binds a stage; not emitted "
                                    "yet",
                                    sample->sampler,
                                    max_samplers);
                return false;
            }
            if (!valid(sample->swizzle) || !valid(sample->texture_select) ||
                static_cast<u32>(sample->coordinates[0]) == 6 ||
                static_cast<u32>(sample->coordinates[1]) == 6) {
                error = fmt::format("a sample of texture {} has a component select of 6",
                                    sample->texture);
                return false;
            }
            registers.insert(sample->src_gpr);
            registers.insert(sample->dst_gpr);
            textures.insert(sample->texture);
            samplers.insert(sample->sampler);
        } else if (const auto* group = std::get_if<ir::AluGroup>(&statement)) {
            if (!scan_alu(*group, registers, alu, error)) {
                return false;
            }
        } else {
            error = "a pixel shader statement that is not a sample, an ALU group or a colour "
                    "export";
            return false;
        }
    }

    const auto to = std::back_inserter(out);
    out += prelude;
    emit_alu_helpers(out, alu);
    const bool has_inputs = position || !semantics.empty();
    if (has_inputs) {
        out += "\nstruct PixelIn {\n";
        if (position) {
            out += "    float4 position [[position]];\n";
        }
        for (const auto& [semantic, interpolation] : semantics) {
            fmt::format_to(to, "    float4 sem{0} [[user(sem{0})]]{1};\n", semantic, interpolation);
        }
        out += "};\n";
    }
    if (!targets.empty()) {
        out += "\nstruct PixelOut {\n";
        for (const u32 target : targets) {
            fmt::format_to(to, "    float4 color{0} [[color({0})]];\n", target);
        }
        out += "};\n";
    }

    std::vector<std::string> parameters;
    if (has_inputs) {
        parameters.emplace_back("PixelIn in [[stage_in]]");
    }
    if (!uniform_parameters(pixel, parameters, error)) {
        out.clear();
        return false;
    }
    for (const u32 texture : textures) {
        parameters.push_back(fmt::format("texture2d<float> tex{0} [[texture({0})]]", texture));
    }
    for (const u32 sampler : samplers) {
        parameters.push_back(fmt::format("sampler smp{0} [[sampler({0})]]", sampler));
    }
    fmt::format_to(to,
                   "\nfragment {} {}({}) {{\n",
                   targets.empty() ? "void" : "PixelOut",
                   fragment_entry,
                   fmt::join(parameters, ",\n                    "));
    declare_registers(out, registers);
    if (alu.any) {
        out += previous_results;
    }
    for (const ir::PixelInput& input : pixel.inputs) {
        switch (input.source) {
            case ir::InputSource::Parameter:
                fmt::format_to(to, "    R{} = in.sem{};\n", input.gpr, input.semantic);
                break;
            case ir::InputSource::Position:
                fmt::format_to(to, "    R{} = in.position;\n", input.gpr);
                break;
            case ir::InputSource::Default:
                fmt::format_to(
                    to, "    R{} = {};\n", input.gpr, default_value(input.default_value));
                break;
        }
    }
    if (!targets.empty()) {
        out += "    PixelOut out = {};\n";
    }
    for (const ir::Statement& statement : pixel.body) {
        if (const auto* sample = std::get_if<ir::SampleTexture>(&statement)) {
            emit_sample(out, *sample);
            continue;
        }
        if (const auto* group = std::get_if<ir::AluGroup>(&statement)) {
            emit_alu_group(out, *group);
            continue;
        }
        const auto& color = std::get<ir::ExportColor>(statement);
        assign(out,
               "    ",
               fmt::format("out.color{}", color.target),
               color.swizzle,
               fmt::format("R{}", color.gpr));
    }
    if (!targets.empty()) {
        out += "    return out;\n";
    }
    out += "}\n";
    return true;
}

} // namespace wiiu::latte::msl
