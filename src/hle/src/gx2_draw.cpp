// Phase 8.1 — GX2's draw calls. See gx2_draw.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_draw.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/draw.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/registers.hpp>

#include <array>
#include <string_view>

namespace wiiu::hle::gx2 {

namespace {

using latte::pm4::Opcode;
using latte::pm4::type3_header;

/// The base vertex, the primitive and -- for an indexed draw -- the index
/// format, and the instance count: what every draw writes before its packet.
void append_draw_state(std::vector<u32>& out,
                       u32 mode,
                       u32 base_vertex,
                       std::optional<u32> index_type,
                       u32 instances) {
    const std::array<u32, 1> base{base_vertex};
    append_registers(out, Opcode::SetCtlConst, latte::reg::sq_vtx_base_vtx_loc, base);
    const std::array<u32, 1> primitive{mode};
    append_registers(out, Opcode::SetConfigReg, latte::reg::vgt_primitive_type, primitive);
    if (index_type.has_value()) {
        out.push_back(type3_header(Opcode::IndexType, 1));
        out.push_back(*index_type);
    }
    out.push_back(type3_header(Opcode::NumInstances, 1));
    out.push_back(instances);
}

/// Physical indices for an indexed draw, or nothing, logged.
[[nodiscard]] bool indexed_arguments(const Call& call,
                                     u32 index_type,
                                     ppcaddr_t indices,
                                     u32& dma_type,
                                     physaddr_t& physical,
                                     std::string_view what) {
    const std::optional<u32> type = dma_index_type(index_type);
    if (!type.has_value()) {
        WIIU_LOG_ERROR(Gx2, "{}: {:#x} is not a GX2IndexType", what, index_type);
        return false;
    }
    const std::optional<physaddr_t> address = physical_address(call.memory, indices, what);
    if (!address.has_value()) {
        return false;
    }
    dma_type = *type;
    physical = *address;
    return true;
}

/// `void GX2DrawEx(GX2PrimitiveMode mode, uint32_t count, uint32_t offset,
/// uint32_t numInstances)`.
void gx2_draw_ex(Call& call, u32 mode, u32 count, u32 offset, u32 instances) {
    std::vector<u32> packets;
    append_draw(packets, mode, count, offset, instances);
    static_cast<void>(emit(call, packets));
}

/// `GX2DrawEx2(..., uint32_t baseInstance)`.
void gx2_draw_ex2(Call& call, u32 mode, u32 count, u32 offset, u32 instances, u32 base_instance) {
    std::vector<u32> packets;
    append_base_instance(packets, base_instance);
    append_draw(packets, mode, count, offset, instances);
    append_base_instance(packets, 0);
    static_cast<void>(emit(call, packets));
}

/// `void GX2DrawIndexedEx(GX2PrimitiveMode mode, uint32_t count,
/// GX2IndexType indexType, const void *indices, uint32_t offset, uint32_t
/// numInstances)`.
void gx2_draw_indexed_ex(
    Call& call, u32 mode, u32 count, u32 index_type, u32 indices, u32 offset, u32 instances) {
    u32 dma_type = 0;
    physaddr_t physical = 0;
    if (!indexed_arguments(call, index_type, indices, dma_type, physical, "GX2DrawIndexedEx")) {
        return;
    }
    std::vector<u32> packets;
    append_draw_indexed(packets, mode, count, dma_type, physical, offset, instances);
    static_cast<void>(emit(call, packets));
}

/// `GX2DrawIndexedEx2(..., uint32_t baseInstance)`.
void gx2_draw_indexed_ex2(Call& call,
                          u32 mode,
                          u32 count,
                          u32 index_type,
                          u32 indices,
                          u32 offset,
                          u32 instances,
                          u32 base_instance) {
    u32 dma_type = 0;
    physaddr_t physical = 0;
    if (!indexed_arguments(call, index_type, indices, dma_type, physical, "GX2DrawIndexedEx2")) {
        return;
    }
    std::vector<u32> packets;
    append_base_instance(packets, base_instance);
    append_draw_indexed(packets, mode, count, dma_type, physical, offset, instances);
    append_base_instance(packets, 0);
    static_cast<void>(emit(call, packets));
}

/// `void GX2RDrawIndexed(GX2PrimitiveMode mode, const GX2RBuffer *buffer,
/// GX2IndexType indexType, uint32_t count, uint32_t indexOffset, uint32_t
/// vertexOffset, uint32_t numInstances)`: the indices from `indexOffset`
/// elements into the buffer (decaf and Cemu agree).
void gx2r_draw_indexed(Call& call,
                       u32 mode,
                       u32 buffer,
                       u32 index_type,
                       u32 count,
                       u32 index_offset,
                       u32 vertex_offset,
                       u32 instances) {
    constexpr std::string_view what = "GX2RDrawIndexed";
    RBuffer r{};
    if (!read_guest(call.memory, buffer, r)) {
        WIIU_LOG_ERROR(Gx2, "{}: no GX2RBuffer at {:#010x}", what, buffer);
        return;
    }
    const ppcaddr_t indices = r.buffer.value() + (index_offset * r.element_size.value());
    u32 dma_type = 0;
    physaddr_t physical = 0;
    if (!indexed_arguments(call, index_type, indices, dma_type, physical, what)) {
        return;
    }
    std::vector<u32> packets;
    append_draw_indexed(packets, mode, count, dma_type, physical, vertex_offset, instances);
    static_cast<void>(emit(call, packets));
}

} // namespace

std::optional<u32> dma_index_type(u32 index_type) noexcept {
    latte::DmaIndexType type{.raw = 0};
    switch (static_cast<IndexType>(index_type)) {
        case IndexType::U16:
            type.index_size = latte::IndexSize::Bits16;
            type.swap_mode = latte::IndexSwap::Swap16;
            break;
        case IndexType::U16Le:
            type.index_size = latte::IndexSize::Bits16;
            break;
        case IndexType::U32:
            type.index_size = latte::IndexSize::Bits32;
            type.swap_mode = latte::IndexSwap::Swap32;
            break;
        case IndexType::U32Le:
            type.index_size = latte::IndexSize::Bits32;
            break;
        default:
            return std::nullopt;
    }
    return type.raw;
}

void append_draw(std::vector<u32>& out, u32 mode, u32 count, u32 base_vertex, u32 instances) {
    append_draw_state(out, mode, base_vertex, std::nullopt, instances);
    out.push_back(type3_header(Opcode::DrawIndexAuto, 2, true));
    out.push_back(count);
    out.push_back(0); // the initiator: zero, as decaf and Cemu write it
}

void append_draw_indexed(std::vector<u32>& out,
                         u32 mode,
                         u32 count,
                         u32 dma_index_type,
                         physaddr_t indices,
                         u32 base_vertex,
                         u32 instances) {
    append_draw_state(out, mode, base_vertex, dma_index_type, instances);
    latte::DrawInitiator initiator{.raw = 0};
    initiator.source_select = latte::SourceSelect::Dma;
    if ((mode & primitive_tessellate) != 0) {
        initiator.major_mode = 1;
    }
    out.push_back(type3_header(Opcode::DrawIndex2, 5, true));
    out.push_back(0xFFFFFFFF); // VGT_DMA_MAX_SIZE
    out.push_back(indices);
    out.push_back(0); // the address's high bits
    out.push_back(count);
    out.push_back(initiator.raw);
}

void append_base_instance(std::vector<u32>& out, u32 base_instance) {
    const std::array<u32, 1> value{base_instance};
    append_registers(out, Opcode::SetCtlConst, latte::reg::sq_vtx_start_inst_loc, value);
}

void register_gx2_draw(Registry& registry) {
    registry.add("gx2", "GX2DrawEx", gx2_draw_ex);
    registry.add("gx2", "GX2DrawEx2", gx2_draw_ex2);
    registry.add("gx2", "GX2DrawIndexedEx", gx2_draw_indexed_ex);
    registry.add("gx2", "GX2DrawIndexedEx2", gx2_draw_indexed_ex2);
    registry.add("gx2", "GX2RDrawIndexed", gx2r_draw_indexed);
}

} // namespace wiiu::hle::gx2
