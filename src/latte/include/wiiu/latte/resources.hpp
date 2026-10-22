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
//
// A texture's descriptor is the same seven words read differently
// (SQ_TEX_RESOURCE_WORD0-6): its dimension, tile mode, pitch and size; its
// data format; where level 0 and level 1 are, in 256-byte units with the
// bank and pipe swizzle in the low bits; how its components are read and
// which lands in each channel; and the levels and slices of it in view.
// Every field is where Linux's `r600d.h` and decaf put it (they agree);
// `texture_resource` reads them back. Each stage has eighteen texture units,
// the first eighteen of its share (Cemu's limit, which GX2 keeps to).
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
/// geometry stage's share, the constant cache has sixteen registers a
/// stage, and a stage has eighteen texture units.
inline constexpr u32 attribute_buffer_count = 16;
inline constexpr u32 uniform_blocks_per_stage = 16;
inline constexpr u32 textures_per_stage = 18;

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

/// SQ_TEX_DIM: what shape a texture is.
enum class TexDim : u32 {
    D1 = 0,
    D2 = 1,
    D3 = 2,
    Cube = 3,
    D1Array = 4,
    D2Array = 5,
    D2Msaa = 6,
    D2ArrayMsaa = 7,
};

/// SQ_FORMAT_COMP: how a component's bits are read.
enum class FormatComp : u32 {
    Unsigned = 0,
    Signed = 1,
    UnsignedBiased = 2,
};

/// WORD0: the shape and layout. PITCH is in eight-pixel units less one
/// (for a block-compressed format, pixels: four a block); TILE_TYPE is set
/// for a depth layout read as a texture.
union TexResourceWord0 {
    u32 raw;
    BitField<0, 3, TexDim> dim;
    BitField<3, 4, u32> tile_mode; ///< a `TileMode`
    BitField<7, 1, bool> tile_type;
    BitField<8, 11, u32> pitch;
    BitField<19, 13, u32> width; ///< less one
};

/// WORD1: the rest of the size, and the format (an R600 FMT_* number).
union TexResourceWord1 {
    u32 raw;
    BitField<0, 13, u32> height; ///< less one
    BitField<13, 13, u32> depth; ///< less one: slices, cube arrays' cubes, or depth
    BitField<26, 6, u32> data_format;
};

/// WORD2 and WORD3 are level 0's and level 1's base address, >> 8.

/// WORD4: how each component is read and where it lands.
union TexResourceWord4 {
    u32 raw;
    BitField<0, 2, FormatComp> format_comp_x;
    BitField<2, 2, FormatComp> format_comp_y;
    BitField<4, 2, FormatComp> format_comp_z;
    BitField<6, 2, FormatComp> format_comp_w;
    BitField<8, 2, NumFormat> num_format_all;
    BitField<10, 1, bool> srf_mode_all;
    BitField<11, 1, bool> force_degamma; ///< sRGB
    BitField<12, 2, Endian> endian_swap;
    BitField<14, 2, u32> request_size;
    BitField<16, 3, Sel> dst_sel_x;
    BitField<19, 3, Sel> dst_sel_y;
    BitField<22, 3, Sel> dst_sel_z;
    BitField<25, 3, Sel> dst_sel_w;
    BitField<28, 4, u32> base_level;
};

/// WORD5: the levels and slices in view. For a multisampled texture
/// LAST_LEVEL is the log2 of its samples instead.
union TexResourceWord5 {
    u32 raw;
    BitField<0, 4, u32> last_level;
    BitField<4, 13, u32> base_array;
    BitField<17, 13, u32> last_array;
    BitField<30, 2, u32> yuv_conv; ///< GX2 sets it for a cube array
};

/// WORD6: filtering hints, and the type.
union TexResourceWord6 {
    u32 raw;
    BitField<0, 2, u32> mpeg_clamp;
    BitField<2, 3, u32> max_aniso_ratio;
    BitField<5, 3, u32> perf_modulation;
    BitField<8, 1, bool> interlaced;
    BitField<30, 2, ResourceType> type;
};

/// A texture resource, read back out of the register file.
struct TextureResource {
    TexResourceWord0 word0{};
    TexResourceWord1 word1{};
    physaddr_t base = 0; ///< level 0, the swizzle in its low bits
    physaddr_t mip = 0;  ///< level 1, likewise
    TexResourceWord4 word4{};
    TexResourceWord5 word5{};
    TexResourceWord6 word6{};

    [[nodiscard]] bool valid() const noexcept {
        return word6.type.value() == ResourceType::ValidTexture;
    }
};

[[nodiscard]] inline TextureResource texture_resource(const RegisterFile& registers,
                                                      u32 id) noexcept {
    const u32 base = resource::address(id);
    TextureResource out;
    out.word0.raw = registers.read(base);
    out.word1.raw = registers.read(base + 4);
    out.base = registers.read(base + 8) << 8;
    out.mip = registers.read(base + 12) << 8;
    out.word4.raw = registers.read(base + 16);
    out.word5.raw = registers.read(base + 20);
    out.word6.raw = registers.read(base + 24);
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
