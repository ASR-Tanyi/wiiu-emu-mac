// Phase 8.1 — the fetch shader GX2 writes. See gx2_fetch_shader.hpp.

#include <wiiu/common/bits.hpp>
#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_fetch_shader.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>

#include <algorithm>
#include <limits>

namespace wiiu::hle::gx2 {

namespace {

using latte::Endian;

[[nodiscard]] constexpr u32 type_of(u32 format) noexcept {
    return format & attrib_format::type_mask;
}

/// Bits per attribute, by type: decaf's `GX2GetAttribFormatBits`.
constexpr std::array<u8, attrib_format::type_count> type_bits{
    8, 8, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 96, 96, 128, 128};

/// FMT_* by type: Cemu's `rawFormatToFetchFormat`.
constexpr std::array<u8, attrib_format::type_count> type_data_formats{
    0x01, 0x02, 0x05, 0x06, 0x07, 0x0D, 0x0E, 0x0F, 0x10, 0x16,
    0x1A, 0x19, 0x1D, 0x1E, 0x1F, 0x20, 0x2F, 0x30, 0x22, 0x23};

[[nodiscard]] constexpr u32 clauses_for(u32 fetches) noexcept {
    return (fetches + latte::max_fetches_per_clause - 1) / latte::max_fetches_per_clause;
}

/// A title's count no vertex shader could tell apart: the semantic is a
/// byte. Anything past it is taken for garbage.
constexpr u32 max_attribs = 256;

/// decaf's tessellated fetch shaders: fetches per attribute, less the two
/// that carry the tessellation's own values, and the ALU instructions that
/// go with them. Zero fetches for the untessellated one.
[[nodiscard]] constexpr u32 tessellation_fetches_per_attrib(FetchShaderType type) noexcept {
    switch (type) {
        case FetchShaderType::LineTessellation:
            return 2;
        case FetchShaderType::TriangleTessellation:
            return 3;
        case FetchShaderType::QuadTessellation:
            return 4;
        case FetchShaderType::NoTessellation:
            break;
    }
    return 0;
}

[[nodiscard]] constexpr u32 tessellation_alu_instructions(FetchShaderType type,
                                                          TessellationMode mode) noexcept {
    u32 count = 4;
    if (mode == TessellationMode::Adaptive) {
        switch (type) {
            case FetchShaderType::LineTessellation:
                count = 11;
                break;
            case FetchShaderType::TriangleTessellation:
                count = 57;
                break;
            case FetchShaderType::QuadTessellation:
                count = 43;
                break;
            case FetchShaderType::NoTessellation:
                break;
        }
    }
    return count + 4;
}

/// The instance divisors a fetch shader has collected, two at most: they
/// become VGT_INSTANCE_STEP_RATE_0 and _1.
struct Divisors {
    u32 count = 0;
    std::array<u32, 2> values{};

    /// Which index an instance fetch with `divisor` reads: the instance
    /// index itself (W) for a divisor of one, otherwise the step rate it
    /// shares or is given -- Y for the first, Z for the second. A third
    /// divisor has no step rate left; Cemu's handling, logged: it keeps X.
    /// (decaf compares against both entries whatever the count, and writes
    /// a third past the array.)
    [[nodiscard]] latte::Sel select(u32 divisor) {
        if (divisor == 1) {
            return latte::Sel::W;
        }
        for (u32 i = 0; i < count; ++i) {
            if (values[i] == divisor) {
                return i == 0 ? latte::Sel::Y : latte::Sel::Z;
            }
        }
        if (count == values.size()) {
            WIIU_LOG_WARN(Gx2,
                          "GX2InitFetchShaderEx: a third instance divisor ({}) has no step rate",
                          divisor);
            return latte::Sel::X;
        }
        values[count] = divisor;
        return count++ == 0 ? latte::Sel::Y : latte::Sel::Z;
    }
};

[[nodiscard]] constexpr latte::Sel select(u32 mask, u32 shift) noexcept {
    return static_cast<latte::Sel>((mask >> shift) & 7U);
}

/// One attribute's fetch: four words, the last padding.
[[nodiscard]] std::array<u32, 4> vertex_fetch(const AttribStream& attrib, Divisors& divisors) {
    const u32 format = attrib.format.value();

    latte::VtxWord0 word0{.raw = 0};
    word0.vtx_inst = latte::VtxInst::Semantic;
    word0.buffer_id = 0xA0 + attrib.buffer.value();
    // decaf treats every type that is not per-instance as per-vertex; Cemu
    // writes the type into the field as it is.
    if (attrib.type.value() == static_cast<u32>(AttribIndexType::PerInstance)) {
        word0.fetch_type = latte::VtxFetchType::InstanceData;
        word0.src_sel_x = divisors.select(attrib.alu_divisor.value());
    } else {
        word0.fetch_type = latte::VtxFetchType::VertexData;
        word0.src_sel_x = latte::Sel::X;
    }
    // Mega-fetch: the attribute's bytes, as decaf writes them; Cemu leaves
    // both fields clear.
    const u32 bytes = attrib_format_bits(format) / 8;
    word0.mega_fetch_count = bytes == 0 ? 0 : bytes - 1;

    latte::VtxWord1 word1{.raw = 0};
    word1.semantic_id = attrib.location.value() & 0xFFU;
    const u32 mask = attrib.mask.value();
    word1.dst_sel_x = select(mask, 24);
    word1.dst_sel_y = select(mask, 16);
    word1.dst_sel_z = select(mask, 8);
    word1.dst_sel_w = select(mask, 0);
    const std::optional<u32> data_format = attrib_data_format(format);
    if (!data_format.has_value()) {
        WIIU_LOG_ERROR(Gx2, "GX2InitFetchShaderEx: no such attribute format {:#x}", format);
    }
    word1.data_format = data_format.value_or(0); // FMT_INVALID
    if ((format & attrib_format::scaled) != 0) {
        word1.num_format_all = latte::NumFormat::Scaled;
    } else if ((format & attrib_format::integer) != 0) {
        word1.num_format_all = latte::NumFormat::Int;
    } else {
        word1.num_format_all = latte::NumFormat::Norm;
    }
    word1.format_comp_signed = (format & attrib_format::is_signed) != 0;

    latte::VtxWord2 word2{.raw = 0};
    word2.offset = attrib.offset.value();
    const u32 swap = attrib.endian_swap.value() & 3U;
    word2.endian_swap = swap == 3 ? attrib_default_endian(format) : static_cast<Endian>(swap);
    word2.mega_fetch = true;

    return {word0.raw, word1.raw, word2.raw, 0};
}

/// `uint32_t GX2CalcFetchShaderSizeEx(uint32_t attribs, GX2FetchShaderType
/// type, GX2TessellationMode mode)`. A type GX2 does not have is sized as
/// untessellated; decaf aborts.
u32 gx2_calc_fetch_shader_size_ex(Call& /*call*/, u32 attribs, u32 type, u32 mode) {
    if (type > static_cast<u32>(FetchShaderType::QuadTessellation)) {
        WIIU_LOG_ERROR(Gx2, "GX2CalcFetchShaderSizeEx: no such fetch shader type {}", type);
        type = static_cast<u32>(FetchShaderType::NoTessellation);
    }
    return fetch_shader_size(
        attribs, static_cast<FetchShaderType>(type), static_cast<TessellationMode>(mode));
}

/// `void GX2InitFetchShaderEx(GX2FetchShader *fetchShader, uint8_t *buffer,
/// uint32_t attribCount, const GX2AttribStream *attribs, GX2FetchShaderType
/// type, GX2TessellationMode mode)`. The structure is cleared first, as
/// Cemu does -- decaf keeps what was there, reading its divisor count
/// before setting it -- and its resources are left clear: decaf's register
/// count for an untessellated fetch shader is zero, Cemu's two. The data
/// cache flush decaf ends with has nothing to do here.
void gx2_init_fetch_shader_ex(Call& call,
                              u32 shader_address,
                              u32 buffer,
                              u32 count,
                              u32 attribs_address,
                              u32 type,
                              u32 mode) {
    if (count > max_attribs) {
        WIIU_LOG_ERROR(Gx2, "GX2InitFetchShaderEx: {} attributes", count);
        return;
    }
    std::vector<AttribStream> attribs(count);
    for (u32 i = 0; i < count; ++i) {
        constexpr u32 stride = sizeof(AttribStream);
        if (!read_guest(call.memory, attribs_address + (i * stride), attribs[i])) {
            WIIU_LOG_ERROR(Gx2, "GX2InitFetchShaderEx: no attributes at {:#010x}", attribs_address);
            return;
        }
    }
    if (type != static_cast<u32>(FetchShaderType::NoTessellation)) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            WIIU_LOG_WARN(Gx2,
                          "GX2InitFetchShaderEx: fetch shader type {} is tessellated; writing the "
                          "untessellated one",
                          type);
        }
    }
    if ((buffer & 0xFFU) != 0) {
        WIIU_LOG_WARN(Gx2,
                      "GX2InitFetchShaderEx: the program at {:#010x} is not 256-byte aligned, "
                      "and the GPU will not find it",
                      buffer);
    }

    const FetchShaderProgram program = generate_fetch_shader(attribs);
    const u32 size = static_cast<u32>(program.words.size() * 4);
    if (type <= static_cast<u32>(FetchShaderType::QuadTessellation) &&
        size > fetch_shader_size(count,
                                 static_cast<FetchShaderType>(type),
                                 static_cast<TessellationMode>(mode))) {
        WIIU_LOG_ERROR(Gx2, "GX2InitFetchShaderEx: the program does not fit the title's buffer");
        return;
    }
    // Little-endian: the GPU's order.
    std::vector<u8> bytes(size);
    for (std::size_t i = 0; i < program.words.size(); ++i) {
        for (std::size_t b = 0; b < 4; ++b) {
            bytes[(4 * i) + b] = static_cast<u8>(program.words[i] >> (8 * b));
        }
    }
    if (!call.memory.write_bytes(buffer, bytes)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitFetchShaderEx: the buffer at {:#010x} is not writable", buffer);
        return;
    }

    FetchShader shader{};
    shader.type = type;
    shader.sq_pgm_resources_fs = 0;
    shader.size = size;
    shader.data = buffer;
    shader.attrib_count = count;
    shader.divisor_count = program.divisor_count;
    shader.divisors[0] = program.divisors[0];
    shader.divisors[1] = program.divisors[1];
    if (!write_guest(call.memory, shader_address, shader)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitFetchShaderEx: no fetch shader at {:#010x}", shader_address);
    }
}

u32 gx2_get_attrib_format_bits(Call& /*call*/, u32 format) {
    const u32 bits = attrib_format_bits(format);
    if (bits == 0) {
        WIIU_LOG_ERROR(Gx2, "GX2GetAttribFormatBits: no such attribute format {:#x}", format);
    }
    return bits;
}

} // namespace

u32 attrib_format_bits(u32 format) noexcept {
    const u32 type = type_of(format);
    return type < type_bits.size() ? type_bits[type] : 0;
}

std::optional<u32> attrib_data_format(u32 format) noexcept {
    const u32 type = type_of(format);
    if (type >= type_data_formats.size()) {
        return std::nullopt;
    }
    return type_data_formats[type];
}

Endian attrib_default_endian(u32 format) noexcept {
    switch (type_of(format)) {
        case 0x00: // 8
        case 0x01: // 4_4
        case 0x04: // 8_8
        case 0x0A: // 8_8_8_8
            return Endian::None;
        case 0x02: // 16
        case 0x03: // 16_FLOAT
        case 0x07: // 16_16
        case 0x08: // 16_16_FLOAT
        case 0x0E: // 16_16_16_16
        case 0x0F: // 16_16_16_16_FLOAT
            return Endian::Swap8In16;
        default:
            return Endian::Swap8In32;
    }
}

u32 fetch_shader_size(u32 attribs, FetchShaderType type, TessellationMode mode) noexcept {
    if (attribs > max_attribs) {
        return std::numeric_limits<u32>::max(); // no buffer is that large
    }
    u32 fetches = attribs;
    u32 alu = 0;
    u32 extra = 0;
    if (const u32 per_attrib = tessellation_fetches_per_attrib(type); per_attrib != 0) {
        fetches = per_attrib * (attribs >= 2 ? attribs - 2 : 0);
        alu = tessellation_alu_instructions(type, mode);
        extra = 2; // decaf counts two more control-flow instructions
    }
    const u32 control_flow = (clauses_for(fetches) + extra + 1) * latte::cf_instruction_bytes;
    return align_up(control_flow + (alu * 8), 16U) + (fetches * latte::vtx_instruction_bytes);
}

FetchShaderProgram generate_fetch_shader(std::span<const AttribStream> attribs) {
    const u32 count = static_cast<u32>(attribs.size());
    // The control flow, padded to sixteen bytes, then the fetches.
    const u32 head = align_up((clauses_for(count) + 1) * latte::cf_instruction_bytes, 16U);
    FetchShaderProgram out;
    out.words.assign((head + (count * latte::vtx_instruction_bytes)) / 4, 0);

    // A VTX_TC per sixteen fetches, then the RETURN, with a barrier.
    std::size_t word = 0;
    for (u32 first = 0; first < count; first += latte::max_fetches_per_clause) {
        const u32 fetches = std::min(count - first, latte::max_fetches_per_clause);
        latte::CfWord0 word0{.raw = 0};
        word0.addr = (head + (first * latte::vtx_instruction_bytes)) / 8;
        latte::CfWord1 word1{.raw = 0};
        word1.count = (fetches - 1) & 7U;
        word1.count_3 = ((fetches - 1) >> 3) & 1U;
        word1.cf_inst = latte::CfInst::VtxTc;
        out.words[word++] = word0.raw;
        out.words[word++] = word1.raw;
    }
    latte::CfWord1 end{.raw = 0};
    end.cf_inst = latte::CfInst::Return;
    end.barrier = true;
    out.words[word + 1] = end.raw;

    Divisors divisors;
    word = head / 4;
    for (const AttribStream& attrib : attribs) {
        for (const u32 value : vertex_fetch(attrib, divisors)) {
            out.words[word++] = value;
        }
    }
    out.divisor_count = divisors.count;
    out.divisors = divisors.values;
    return out;
}

void register_gx2_fetch_shader(Registry& registry) {
    registry.add("gx2", "GX2CalcFetchShaderSizeEx", gx2_calc_fetch_shader_size_ex);
    registry.add("gx2", "GX2InitFetchShaderEx", gx2_init_fetch_shader_ex);
    registry.add("gx2", "GX2GetAttribFormatBits", gx2_get_attrib_format_bits);
}

} // namespace wiiu::hle::gx2
