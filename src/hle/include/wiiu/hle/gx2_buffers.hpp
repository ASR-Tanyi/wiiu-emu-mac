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
//
// **GX2R's buffers** (gx2r_buffer.cpp) are memory GX2 gets for the title:
// `GX2RCreateBuffer` calls an allocator -- the title's, from
// `GX2RSetAllocator`, or GX2R's own, which goes through coreinit's pointer to
// the default heap's allocator, which wut points at its malloc -- and that
// is a guest function, so the call goes out to the guest and comes back
// (coreinit's `call_guest`). Locking marks the buffer and hands back its
// memory; unlocking publishes what the CPU wrote, invalidating the buffer for
// what it is bound as. decaf's and Cemu's agree on the shape; where they
// differ: the alignment is Cemu's table (decaf's is 256 for everything), the
// free function is given the buffer's own flags (Cemu's; decaf's gives only
// the option flags, and WHB's free function picks its heap by the bindings),
// a destroyed buffer's pointer is cleared whatever owned it (Cemu's), and
// locking keeps decaf's bookkeeping of the locked flags (Cemu keeps none).
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

/// GX2RResourceFlags (decaf's names).
namespace r_flags {
inline constexpr u32 bind_texture = 1U << 0;
inline constexpr u32 bind_color_buffer = 1U << 1;
inline constexpr u32 bind_depth_buffer = 1U << 2;
inline constexpr u32 bind_scan_buffer = 1U << 3;
inline constexpr u32 bind_vertex_buffer = 1U << 4;
inline constexpr u32 bind_index_buffer = 1U << 5;
inline constexpr u32 bind_uniform_block = 1U << 6;
inline constexpr u32 bind_shader_program = 1U << 7;
inline constexpr u32 bind_stream_output = 1U << 8;
inline constexpr u32 bind_display_list = 1U << 9;
inline constexpr u32 bind_gs_ring = 1U << 10;
inline constexpr u32 usage_cpu_read = 1U << 11;
inline constexpr u32 usage_cpu_write = 1U << 12;
inline constexpr u32 usage_gpu_read = 1U << 13;
inline constexpr u32 usage_gpu_write = 1U << 14;
inline constexpr u32 disable_cpu_invalidate = 1U << 20;
inline constexpr u32 disable_gpu_invalidate = 1U << 21;
inline constexpr u32 locked_read_only = 1U << 22;
inline constexpr u32 destroy_no_free = 1U << 23;
inline constexpr u32 gx2r_allocated = 1U << 29;
inline constexpr u32 locked = 1U << 30;
/// What the `…Ex` calls take from their flags argument: bits 19 to 23.
inline constexpr u32 option_mask = 0x00F80000;
} // namespace r_flags

/// Where a `GX2RBuffer`'s flags and memory pointer are (gx2_shader.hpp has
/// the structure).
inline constexpr u32 r_buffer_flags = 0x0;
inline constexpr u32 r_buffer_pointer = 0xC;

/// `GX2RGetBufferAlignment`: what a buffer bound as `flags` is aligned to.
[[nodiscard]] u32 r_alignment(u32 flags) noexcept;

/// `GX2RGetBufferAllocationSize`: a buffer's bytes rounded up to 64.
[[nodiscard]] constexpr u32 r_allocation_size(u32 bytes) noexcept {
    return (bytes + 0x3FU) & ~0x3FU;
}

/// Register GX2R's buffer calls. Called by `register_gx2`.
void register_gx2r_buffers(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
