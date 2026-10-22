// Phase 8.6 — see msl.hpp.

#include <wiiu/latte/msl.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <set>
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
        }
    }
    if (pixel != nullptr) {
        for (const ir::PixelInput& input : pixel->inputs) {
            if (input.source == ir::InputSource::Parameter) {
                semantics.insert(input.semantic);
            }
        }
    }

    const auto to = std::back_inserter(out);
    out += prelude;
    out += fetch_helpers;
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
    out += ") {\n";
    declare_registers(out, registers);
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
    for (const ir::Statement& statement : pixel.body) {
        if (const auto* color = std::get_if<ir::ExportColor>(&statement)) {
            registers.insert(color->gpr);
            targets.insert(color->target);
            if (!valid(color->swizzle)) {
                error = fmt::format("colour {} has a component select of 6", color->index);
                return false;
            }
        } else {
            error = "a pixel shader statement that is not a colour export";
            return false;
        }
    }

    const auto to = std::back_inserter(out);
    out += prelude;
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

    fmt::format_to(to,
                   "\nfragment {} {}({}) {{\n",
                   targets.empty() ? "void" : "PixelOut",
                   fragment_entry,
                   has_inputs ? "PixelIn in [[stage_in]]" : "");
    declare_registers(out, registers);
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
