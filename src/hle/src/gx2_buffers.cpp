// Phase 8.1 — attribute buffers and uniform blocks. See gx2_buffers.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_buffers.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/resources.hpp>

#include <array>
#include <optional>
#include <string_view>

namespace wiiu::hle::gx2 {

namespace {

using latte::pm4::Opcode;

/// The resource words for a buffer, words 0 to 6.
using ResourceWords = std::array<u32, latte::resource::words>;

[[nodiscard]] u32 valid_buffer() noexcept {
    latte::VtxConstantWord6 word6{.raw = 0};
    word6.type = latte::ResourceType::ValidBuffer;
    return word6.raw;
}

struct StageRegisters {
    u32 first_resource = 0;
    u32 cache = 0;
    u32 size = 0;
};

[[nodiscard]] StageRegisters stage_registers(UniformStage stage) noexcept {
    namespace res = latte::resource;
    namespace reg = latte::reg;
    switch (stage) {
        case UniformStage::Vertex:
            return {res::vs_uniform_block_0,
                    reg::sq_alu_const_cache_vs_0,
                    reg::sq_alu_const_buffer_size_vs_0};
        case UniformStage::Pixel:
            return {res::ps_uniform_block_0,
                    reg::sq_alu_const_cache_ps_0,
                    reg::sq_alu_const_buffer_size_ps_0};
        case UniformStage::Geometry:
            return {res::gs_uniform_block_0,
                    reg::sq_alu_const_cache_gs_0,
                    reg::sq_alu_const_buffer_size_gs_0};
    }
    return {};
}

/// The physical address of a buffer the title names: zero for a null one,
/// which unbinds as it does on hardware, and nothing, logged, for one the
/// GPU cannot see.
[[nodiscard]] std::optional<physaddr_t>
buffer_address(const Call& call, ppcaddr_t address, std::string_view what) {
    if (address == 0) {
        return physaddr_t{0};
    }
    return physical_address(call.memory, address, what);
}

/// `void GX2SetAttribBuffer(uint32_t index, uint32_t size, uint32_t stride,
/// const void *buffer)`.
void set_attrib_buffer(
    Call& call, u32 index, u32 size, u32 stride, ppcaddr_t buffer, std::string_view what) {
    if (index >= latte::resource::attribute_buffer_count) {
        WIIU_LOG_ERROR(Gx2, "{}({}): there are only sixteen attribute buffers", what, index);
        return;
    }
    const std::optional<physaddr_t> address = buffer_address(call, buffer, what);
    if (!address.has_value()) {
        return;
    }
    if (stride > latte::max_vtx_stride) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            WIIU_LOG_WARN(
                Gx2, "{}: a stride of {} bytes is wider than the register; it wraps", what, stride);
        }
    }
    std::vector<u32> packets;
    append_attribute_buffer(packets, index, *address, size, stride);
    static_cast<void>(emit(call, packets));
}

void gx2_set_attrib_buffer(Call& call, u32 index, u32 size, u32 stride, u32 buffer) {
    set_attrib_buffer(call, index, size, stride, buffer, "GX2SetAttribBuffer");
}

/// `void GX2Set{Vertex,Pixel,Geometry}UniformBlock(uint32_t location,
/// uint32_t size, const void *data)`.
void set_uniform_block(
    Call& call, UniformStage stage, u32 location, u32 size, ppcaddr_t data, std::string_view what) {
    if (location >= latte::resource::uniform_blocks_per_stage) {
        WIIU_LOG_ERROR(
            Gx2, "{}({}): there are only sixteen uniform blocks a stage", what, location);
        return;
    }
    const std::optional<physaddr_t> address = buffer_address(call, data, what);
    if (!address.has_value()) {
        return;
    }
    // decaf asserts this: the constant cache holds the address >> 8, so a
    // block that is not 256-byte aligned is read from below where it is.
    if ((*address & 0xFFU) != 0) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            WIIU_LOG_WARN(Gx2,
                          "{}: a uniform block at {:#010x} is not 256-byte aligned; the ALU "
                          "reads it from {:#010x}",
                          what,
                          data,
                          data & ~0xFFU);
        }
    }
    std::vector<u32> packets;
    append_uniform_block(packets, stage, location, *address, size);
    static_cast<void>(emit(call, packets));
}

void gx2_set_vertex_uniform_block(Call& call, u32 location, u32 size, u32 data) {
    set_uniform_block(call, UniformStage::Vertex, location, size, data, "GX2SetVertexUniformBlock");
}

void gx2_set_pixel_uniform_block(Call& call, u32 location, u32 size, u32 data) {
    set_uniform_block(call, UniformStage::Pixel, location, size, data, "GX2SetPixelUniformBlock");
}

void gx2_set_geometry_uniform_block(Call& call, u32 location, u32 size, u32 data) {
    set_uniform_block(
        call, UniformStage::Geometry, location, size, data, "GX2SetGeometryUniformBlock");
}

// --- GX2R --------------------------------------------------------------------

/// Where a GX2R buffer is from `offset` on, and how much of it there is
/// past it. Nothing, logged, if the buffer is not there or the offset is
/// past its end.
struct RRange {
    ppcaddr_t address = 0;
    u32 size = 0;
};

[[nodiscard]] std::optional<RRange>
r_range(const Call& call, ppcaddr_t buffer, u32 offset, std::string_view what) {
    RBuffer r{};
    if (!read_guest(call.memory, buffer, r)) {
        WIIU_LOG_ERROR(Gx2, "{}: no GX2RBuffer at {:#010x}", what, buffer);
        return std::nullopt;
    }
    const u32 size = r.element_size.value() * r.element_count.value();
    if (offset > size) {
        WIIU_LOG_ERROR(
            Gx2, "{}: offset {:#x} is past the end of a {:#x}-byte buffer", what, offset, size);
        return std::nullopt;
    }
    return RRange{.address = r.buffer.value() + offset, .size = size - offset};
}

/// `void GX2RSetAttributeBuffer(const GX2RBuffer *buffer, uint32_t index,
/// uint32_t stride, uint32_t offset)`.
void gx2r_set_attribute_buffer(Call& call, u32 buffer, u32 index, u32 stride, u32 offset) {
    constexpr std::string_view what = "GX2RSetAttributeBuffer";
    const std::optional<RRange> range = r_range(call, buffer, offset, what);
    if (range.has_value()) {
        set_attrib_buffer(call, index, range->size, stride, range->address, what);
    }
}

/// `void GX2RSet{Vertex,Pixel,Geometry}UniformBlock(const GX2RBuffer
/// *buffer, uint32_t location, uint32_t offset)`.
void r_set_uniform_block(
    Call& call, UniformStage stage, u32 buffer, u32 location, u32 offset, std::string_view what) {
    const std::optional<RRange> range = r_range(call, buffer, offset, what);
    if (range.has_value()) {
        set_uniform_block(call, stage, location, range->size, range->address, what);
    }
}

void gx2r_set_vertex_uniform_block(Call& call, u32 buffer, u32 location, u32 offset) {
    r_set_uniform_block(
        call, UniformStage::Vertex, buffer, location, offset, "GX2RSetVertexUniformBlock");
}

void gx2r_set_pixel_uniform_block(Call& call, u32 buffer, u32 location, u32 offset) {
    r_set_uniform_block(
        call, UniformStage::Pixel, buffer, location, offset, "GX2RSetPixelUniformBlock");
}

void gx2r_set_geometry_uniform_block(Call& call, u32 buffer, u32 location, u32 offset) {
    r_set_uniform_block(
        call, UniformStage::Geometry, buffer, location, offset, "GX2RSetGeometryUniformBlock");
}

} // namespace

void append_attribute_buffer(
    std::vector<u32>& out, u32 index, physaddr_t address, u32 size, u32 stride) {
    latte::VtxConstantWord2 word2{.raw = 0};
    word2.stride = stride & latte::max_vtx_stride;
    const ResourceWords words{address, size - 1, word2.raw, 0, 0, 0, valid_buffer()};
    append_registers(out,
                     Opcode::SetResource,
                     latte::resource::address(latte::resource::attribute_buffer_0 + index),
                     words);
}

void append_uniform_block(
    std::vector<u32>& out, UniformStage stage, u32 location, physaddr_t address, u32 size) {
    const StageRegisters regs = stage_registers(stage);

    // Read as an array of 16-byte, four-component signed elements (decaf);
    // the sign matters only to a fetch that converts, which a float does not.
    latte::VtxConstantWord2 word2{.raw = 0};
    word2.stride = 16;
    word2.data_format = latte::fmt_32_32_32_32;
    word2.format_comp_signed = true;
    latte::VtxConstantWord3 word3{.raw = 0};
    word3.mem_request_size = 1;
    const ResourceWords words{address, size - 1, word2.raw, word3.raw, 0, 0, valid_buffer()};
    append_registers(
        out, Opcode::SetResource, latte::resource::address(regs.first_resource + location), words);

    append_context_register(out, regs.cache + (4 * location), address >> 8);
    append_context_register(
        out, regs.size + (4 * location), ((size + 255) >> 8) & latte::alu_const_buffer_size_mask);
}

void register_gx2_buffers(Registry& registry) {
    registry.add("gx2", "GX2SetAttribBuffer", gx2_set_attrib_buffer);
    registry.add("gx2", "GX2SetVertexUniformBlock", gx2_set_vertex_uniform_block);
    registry.add("gx2", "GX2SetPixelUniformBlock", gx2_set_pixel_uniform_block);
    registry.add("gx2", "GX2SetGeometryUniformBlock", gx2_set_geometry_uniform_block);
    registry.add("gx2", "GX2RSetAttributeBuffer", gx2r_set_attribute_buffer);
    registry.add("gx2", "GX2RSetVertexUniformBlock", gx2r_set_vertex_uniform_block);
    registry.add("gx2", "GX2RSetPixelUniformBlock", gx2r_set_pixel_uniform_block);
    registry.add("gx2", "GX2RSetGeometryUniformBlock", gx2r_set_geometry_uniform_block);
}

} // namespace wiiu::hle::gx2
