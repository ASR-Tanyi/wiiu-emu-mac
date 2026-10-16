// Latte's resources: the seven-word descriptors SET_RESOURCE writes, which
// tell a shader's fetches where a texture or a buffer is and how to read it.
//
// The resource window is one array of them, shared out between the stages:
// each stage has 160, textures first and its uniform blocks from 0x80 into
// its share, and the vertex shader's share is followed by sixteen for the
// attribute buffers its fetch shader reads. A fetch names a resource from
// its own stage's first, so the fetch shader's buffer 0xA0 is resource 0x140
// overall. The bases are decaf's (`SQ_RES_OFFSET`); the vertex stage's 160
// and the attribute buffers' 0x140 are also where Cemu puts them.
//
// The buffer descriptor's fields: STRIDE, ENDIAN_SWAP and TYPE, with TYPE's
// values, are Linux's `r600d.h`'s (SQ_VTX_CONSTANT_WORD2_0 and _WORD6_0);
// the rest of word 2, and word 3, are decaf's. Cemu shifts the stride by 11
// rather than 8, which its own renderer reads back the same way; Linux's
// position is taken. The constant-cache registers -- where a uniform block
// is for the ALU's constant reads, as opposed to a fetch's -- are in Linux's
// `r600d.h` (SQ_ALU_CONST_CACHE_*) and its R600 register list
// (SQ_ALU_CONST_BUFFER_SIZE_*).
//
// GX2 writes these for attribute buffers and uniform blocks (8.1); the
// translator (8.11, 8.12) reads them back with `buffer_resource`.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/latte/microcode.hpp>
#include <wiiu/latte/registers.hpp>

#include <array>

namespace wiiu::latte {

namespace resource {

/// Words in one resource, and where the window starts.
inline constexpr u32 words = 7;
inline constexpr u32 window_base = 0x38000;

/// Where each stage's resources start, counted in resources.
inline constexpr u32 ps_base = 0x000;
inline constexpr u32 vs_base = 0x0A0;
inline constexpr u32 gs_base = 0x150;
inline constexpr u32 per_stage = 160;

/// A stage's uniform blocks, from 0x80 into its share.
inline constexpr u32 uniform_block_offset = 0x80;
inline constexpr u32 ps_uniform_block_0 = ps_base + uniform_block_offset; ///< 0x80
inline constexpr u32 vs_uniform_block_0 = vs_base + uniform_block_offset; ///< 0x120
inline constexpr u32 gs_uniform_block_0 = gs_base + uniform_block_offset; ///< 0x1D0

/// The attribute buffers, after the vertex stage's share.
inline constexpr u32 attribute_buffer_0 = vs_base + per_stage; ///< 0x140

/// The rings between the geometry stages: the last resource of the vertex
/// shader's share and of the geometry shader's (decaf).
inline constexpr u32 vs_gs_out = vs_base + per_stage - 1; ///< 0x13F
inline constexpr u32 gs_gs_in = gs_base + per_stage - 1;  ///< 0x1EF

/// How many of each GX2 binds: sixteen attribute buffers fit before the
/// geometry stage's share, and the constant cache has sixteen registers a
/// stage.
inline constexpr u32 attribute_buffer_count = 16;
inline constexpr u32 uniform_blocks_per_stage = 16;

/// The byte address of resource `id`'s first word.
[[nodiscard]] constexpr u32 address(u32 id) noexcept {
    return window_base + (id * words * 4);
}

} // namespace resource

/// SQ_TEX_VTX_TYPE: what a resource is, in word 6's top two bits.
enum class ResourceType : u32 {
    InvalidTexture = 0,
    InvalidBuffer = 1,
    ValidTexture = 2,
    ValidBuffer = 3,
};

/// SQ_VTX_CONSTANT_WORD0 is the base address and WORD1 the size in bytes,
/// less one; both are whole words. WORD2 says how to read an element.
union VtxConstantWord2 {
    u32 raw;
    BitField<0, 8, u32> base_address_hi;
    BitField<8, 11, u32> stride;
    BitField<19, 1, bool> clamp_x;
    BitField<20, 6, u32> data_format; ///< an R600 FMT_* number
    BitField<26, 2, NumFormat> num_format_all;
    BitField<28, 1, bool> format_comp_signed;
    BitField<29, 1, bool> srf_mode_all;
    BitField<30, 2, Endian> endian_swap;
};

union VtxConstantWord3 {
    u32 raw;
    BitField<0, 2, u32> mem_request_size;
    BitField<2, 1, bool> uncached;
};

/// WORD4 and WORD5 are unused by a buffer; WORD6 holds its type.
union VtxConstantWord6 {
    u32 raw;
    BitField<30, 2, ResourceType> type;
};

/// FMT_32_32_32_32, which a uniform block is read as.
inline constexpr u32 fmt_32_32_32_32 = 0x22;

/// The widest stride WORD2 holds.
inline constexpr u32 max_vtx_stride = 0x7FF;

/// A buffer resource, read back out of the register file.
struct BufferResource {
    physaddr_t address = 0;
    u32 size = 0; ///< bytes: WORD1 plus one
    VtxConstantWord2 word2{};
    VtxConstantWord3 word3{};
    ResourceType type = ResourceType::InvalidTexture;

    [[nodiscard]] bool valid() const noexcept { return type == ResourceType::ValidBuffer; }
};

[[nodiscard]] inline BufferResource buffer_resource(const RegisterFile& registers,
                                                    u32 id) noexcept {
    const u32 base = resource::address(id);
    BufferResource out;
    out.address = registers.read(base);
    out.size = registers.read(base + 4) + 1;
    out.word2.raw = registers.read(base + 8);
    out.word3.raw = registers.read(base + 12);
    out.type = VtxConstantWord6{.raw = registers.read(base + 24)}.type;
    return out;
}

namespace reg {

/// The constant cache: a uniform block's address >> 8 and its size in
/// 256-byte units, sixteen a stage, for the ALU's constant reads.
inline constexpr u32 sq_alu_const_buffer_size_ps_0 = 0x28140;
inline constexpr u32 sq_alu_const_buffer_size_vs_0 = 0x28180;
inline constexpr u32 sq_alu_const_buffer_size_gs_0 = 0x281C0;
inline constexpr u32 sq_alu_const_cache_ps_0 = 0x28940;
inline constexpr u32 sq_alu_const_cache_vs_0 = 0x28980;
inline constexpr u32 sq_alu_const_cache_gs_0 = 0x289C0;

} // namespace reg

/// SQ_ALU_CONST_BUFFER_SIZE's width: nine bits of 256-byte units (decaf).
inline constexpr u32 alu_const_buffer_size_mask = 0x1FF;

} // namespace wiiu::latte
