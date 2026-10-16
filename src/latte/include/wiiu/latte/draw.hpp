// Latte's draw state: what a draw packet and the few registers written just
// before it say about the draw itself -- the primitive, the index buffer's
// format, the base vertex and instance.
//
// GX2's draw calls write them (8.1) and the translator (8.11) reads them,
// through `DrawCommand` (command_processor.hpp) and these fields. The
// primitive types are decaf's and Cemu's, which agree; GX2's
// `GX2PrimitiveMode` is the same numbers, with 0x80 for a tessellated draw.
// VGT_DMA_INDEX_TYPE and VGT_DRAW_INITIATOR are decaf's fields: Cemu agrees
// on the index type, whose four GX2 values it writes as they are, and does
// not decode the initiator. SQ_VTX_START_INST_LOC is in Linux's R600
// register list, the first of the control constants, SQ_VTX_BASE_VTX_LOC,
// just before it -- the address decaf and Cemu both write it at.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>

namespace wiiu::latte {

/// VGT_PRIMITIVE_TYPE's PRIM_TYPE (DI_PT_*).
enum class PrimitiveType : u32 {
    None = 0x00,
    PointList = 0x01,
    LineList = 0x02,
    LineStrip = 0x03,
    TriangleList = 0x04,
    TriangleFan = 0x05,
    TriangleStrip = 0x06,
    LineListAdjacency = 0x0A,
    LineStripAdjacency = 0x0B,
    TriangleListAdjacency = 0x0C,
    TriangleStripAdjacency = 0x0D,
    RectList = 0x11,
    LineLoop = 0x12,
    QuadList = 0x13,
    QuadStrip = 0x14,
    Polygon = 0x15,
};

/// VGT_DMA_INDEX_TYPE, which `INDEX_TYPE` carries: the size of an index,
/// and the byte swap the index fetch applies to what it reads. The GPU
/// reads memory little-endian, so big-endian indices are fetched with a
/// swap the size of an index.
enum class IndexSize : u32 {
    Bits16 = 0,
    Bits32 = 1,
};

enum class IndexSwap : u32 {
    None = 0,
    Swap16 = 1,
    Swap32 = 2,
    SwapWord = 3,
};

union DmaIndexType {
    u32 raw;
    BitField<0, 2, IndexSize> index_size;
    BitField<2, 2, IndexSwap> swap_mode;
};

/// VGT_DRAW_INITIATOR, a draw packet's last word before any indices.
enum class SourceSelect : u32 {
    Dma = 0,
    Immediate = 1,
    AutoIndex = 2,
};

union DrawInitiator {
    u32 raw;
    BitField<0, 2, SourceSelect> source_select;
    BitField<2, 2, u32> major_mode; ///< 1 for a tessellated draw
    BitField<4, 1, bool> sprite_enable;
    BitField<5, 1, bool> not_eop;
    BitField<6, 1, bool> use_opaque; ///< a stream-out draw's own vertex count
};

namespace reg {

/// The first two control constants: added to every vertex index, and to
/// every instance index, before the fetch.
inline constexpr u32 sq_vtx_base_vtx_loc = 0x3CFF0;
inline constexpr u32 sq_vtx_start_inst_loc = 0x3CFF4;

} // namespace reg

} // namespace wiiu::latte
