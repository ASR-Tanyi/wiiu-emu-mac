// Phase 8.1 — GX2's draw calls.
//
// A draw is four or five packets: the base vertex (a control constant), the
// primitive type (a config register), the index format for an indexed draw,
// the instance count, and the draw packet itself -- DRAW_INDEX_AUTO for
// vertices 0 to count - 1, DRAW_INDEX_2 for indices read from the title's
// memory. Everything else a draw needs is already in the register file,
// written by the state, shader and buffer calls before it. The `Ex2` forms
// set a base instance around the draw and put it back to zero after.
//
// decaf's sequences, and Cemu's agree with them but for three things. Cemu
// writes an index format before an auto-indexed draw too (32-bit,
// big-endian), which such a draw does not read; decaf's leaves it out. Cemu
// sets the predicate bit on every draw packet, which decaf cannot write
// (its stream-out draw carries a to-do saying gx2.rpl sets it there); it
// is Cemu's that is taken, since a draw without it would ignore
// predication when that arrives. And decaf marks a tessellated indexed draw
// with the initiator's major mode 1. Both write an auto-indexed draw's
// initiator as zero, though its source select is not DMA, and an indexed
// draw's maximum size as all ones.
//
// The translator (8.6, 8.11) is the draw listener the command processor
// hands these to; until it exists, nothing draws.
#pragma once

#include <wiiu/common/types.hpp>

#include <optional>
#include <vector>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// `GX2IndexType`: which is VGT_DMA_INDEX_TYPE itself -- the size in bit
/// 0, the swap in bits 2-3 (latte/draw.hpp).
enum class IndexType : u32 {
    U16Le = 0,
    U32Le = 1,
    U16 = 4,
    U32 = 9,
};

/// `GX2PrimitiveMode`'s flag for a tessellated draw. The rest of the mode
/// is VGT_PRIMITIVE_TYPE's value.
inline constexpr u32 primitive_tessellate = 0x80;

/// VGT_DMA_INDEX_TYPE for a `GX2IndexType`, as decaf works it out: a size,
/// and a swap of that size for big-endian indices. Nothing for a value
/// that is not one of the four.
[[nodiscard]] std::optional<u32> dma_index_type(u32 index_type) noexcept;

/// The packets `GX2DrawEx` writes.
void append_draw(std::vector<u32>& out, u32 mode, u32 count, u32 base_vertex, u32 instances);

/// The packets `GX2DrawIndexedEx` writes, for indices at physical
/// `indices` in the format `dma_index_type` gave.
void append_draw_indexed(std::vector<u32>& out,
                         u32 mode,
                         u32 count,
                         u32 dma_index_type,
                         physaddr_t indices,
                         u32 base_vertex,
                         u32 instances);

/// SQ_VTX_START_INST_LOC, which the `Ex2` forms set around their draw.
void append_base_instance(std::vector<u32>& out, u32 base_instance);

/// Register the draw calls. Called by `register_gx2`.
void register_gx2_draw(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
