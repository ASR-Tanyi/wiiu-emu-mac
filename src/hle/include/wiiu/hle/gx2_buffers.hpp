// Phase 8.1 — the buffers a shader reads: attribute buffers and uniform
// blocks.
//
// Both are buffer resources (latte/resources.hpp): seven words in the
// resource window, written with one SET_RESOURCE, saying where the buffer
// is, how big it is and how an element is read. An attribute buffer is what
// the fetch shader (gx2_fetch_shader.hpp) fetches from, at the stride the
// title gives. A uniform block is a buffer of four-float constants, and is
// told to the GPU twice: as a resource, for a shader that fetches from it,
// and in the stage's constant-cache registers, for one whose ALU reads it as
// constants -- which is how GX2's compiler reads a uniform block.
//
// decaf's sequences, which read them out of gx2.rpl. Cemu agrees on every
// resource word but two: it shifts an attribute buffer's stride by 11 rather
// than the 8 Linux's `r600d.h` gives, and writes a uniform block's word 2 as
// zero where decaf has its 16-byte stride and 32_32_32_32 format; and it
// leaves the constant cache alone. Its pixel uniform blocks also start at
// word 0x250 of the window, not on a resource boundary; decaf's 0x80th
// resource is taken. The GX2R forms are the same calls on a GX2R buffer,
// from an offset into it; for an attribute buffer decaf passes the whole
// buffer's size and Cemu the size past the offset, as both do for uniform
// blocks, and Cemu's is taken.
#pragma once

#include <wiiu/common/types.hpp>

#include <vector>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// The shader stage a uniform block is bound for.
enum class UniformStage : u8 {
    Vertex,
    Pixel,
    Geometry,
};

/// The SET_RESOURCE `GX2SetAttribBuffer` writes for attribute buffer
/// `index`, `size` bytes at physical `address`, `stride` bytes an element.
/// The index is the caller's to check; the stride is held to its 11 bits.
void append_attribute_buffer(
    std::vector<u32>& out, u32 index, physaddr_t address, u32 size, u32 stride);

/// The packets `GX2Set…UniformBlock` writes for uniform block `location`
/// of `stage`: the resource, then the constant-cache base (the address >>
/// 8) and size (in 256-byte units, rounded up). The location is the
/// caller's to check.
void append_uniform_block(
    std::vector<u32>& out, UniformStage stage, u32 location, physaddr_t address, u32 size);

/// Register the buffer calls. Called by `register_gx2`.
void register_gx2_buffers(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
