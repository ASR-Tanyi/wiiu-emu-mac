// Phase 8.1 — GX2's textures: a surface with a view of it, packed into the
// texture resource a shader samples (latte/resources.hpp).
//
// A `GX2Texture` is a `GX2Surface` with the levels and slices in view and a
// component map, and five words of resource the title has
// `GX2InitTextureRegs` work out once: WORD0, 1, 4, 5 and 6 -- shape and
// pitch, size and format, how components are read and where they land, the
// view, and the type. `GX2Set{Pixel,Vertex,Geometry}Texture` adds the two
// words only known at the call, where level 0 and level 1 are, and writes
// all seven to the stage's texture unit with one SET_RESOURCE.
//
// decaf and Cemu pack the words alike: the data format is the surface
// format's low six bits; its SIGNED, FLOAT and INTEGER bits make the
// components signed and the number format scaled or integer, its sRGB bit
// forces degamma; nothing is byte-swapped (every surface format's swap mode
// is none); a block-compressed pitch, in blocks, is counted in pixels; a
// cube's depth is its faces over six, an array's or volume's its slices; a
// multisampled texture's last level is its sample count's log2; a cube array
// sets YUV_CONV; the depth bit of the surface's use sets TILE_TYPE; and every
// texture asks for anisotropy 4 and modulation 7. Where they differ, this is
// Cemu's: each word starts from zero (decaf keeps whatever WORD5 and 6 held),
// a zero width or pitch wraps as the arithmetic does rather than being held
// at one (decaf), and level 1 is the image when `mipmaps` is null, with the
// swizzle applied to each level only while it is still macro-tiled -- the
// rule the colour buffer's base follows (gx2_surface.hpp) -- where decaf
// applies it to both whenever there is one. `GX2SetComputeTexture` is the
// vertex stage's, as Cemu has it.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/hle/gx2_surface.hpp>

#include <cstddef>
#include <vector>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// `GX2Texture`, as it sits in guest memory: wut's offsets, which decaf's
/// agree with.
struct Texture {
    Surface surface;
    u32be view_first_mip;
    u32be view_num_mips;
    u32be view_first_slice;
    u32be view_num_slices;
    u32be comp_map; ///< the destination selects, X in the top byte
    u32be word0;
    u32be word1;
    u32be word4;
    u32be word5;
    u32be word6;
};

static_assert(sizeof(Texture) == 0x9C);
static_assert(offsetof(Texture, view_first_mip) == 0x74);
static_assert(offsetof(Texture, comp_map) == 0x84);
static_assert(offsetof(Texture, word0) == 0x88);
static_assert(offsetof(Texture, word6) == 0x98);

/// `GX2InitTextureRegs`: the five words, and the zero counts it holds at one
/// (levels and slices in view, height, depth and the surface's levels).
void init_texture_regs(Texture& texture);

/// Level 0's and level 1's base registers -- physical address >> 8, the
/// swizzle in the low bits while the level is macro-tiled -- for words 2 and
/// 3. Level 1 is the image when `mipmaps` is null.
struct TextureAddresses {
    u32 base = 0;
    u32 mip = 0;
};

/// The SET_RESOURCE `GX2Set…Texture` writes to resource `resource`.
void append_texture(std::vector<u32>& out,
                    u32 resource,
                    const Texture& texture,
                    TextureAddresses addresses);

/// Register the texture calls. Called by `register_gx2`.
void register_gx2_texture(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
