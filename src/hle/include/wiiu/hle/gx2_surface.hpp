// Phase 8.1 — GX2's surfaces: how big one is, the registers that describe a
// colour buffer, and the two things done to one before anything is drawn --
// clearing it, and copying it out to a scan buffer.
//
// Every image GX2 deals in -- texture, colour buffer, depth buffer -- is a
// `GX2Surface`. The title fills in what it is (dimension, size, format,
// samples, what it is for) and calls `GX2CalcSurfaceSizeAndAlignment`, which
// fills in how it is laid out: bytes, alignment, pitch, the tile mode, the
// swizzle and where each mip level goes. The title then allocates that many
// bytes at that alignment and points `image` at them. A colour buffer wraps a
// surface with the slices it renders to, and `GX2InitColorBufferRegs` packs
// all of it into the CB registers (latte/color_buffer.hpp) that
// `GX2SetColorBuffer` will write.
//
// Each level's layout is Latte's (8.5, ADR-005). What this file adds is GX2's
// policy on top of it:
//
//   * the **default tile mode** (0, or 0x20) means 2D-tiled for anything but
//     a single-sampled 1D texture, which is linear -- and a surface smaller
//     than one macro-tile in both directions steps down to 1D, which pads
//     far less;
//   * a 2D-tiled surface carries **0xD in the swizzle's third byte**: the
//     first mip level that is no longer macro-tiled, which GX2 moves down to
//     wherever that actually happens;
//   * **mip levels** follow one another, each aligned as its own layout
//     requires. `mipLevelOffset[0]` is level 1's offset from `image`; the
//     rest are offsets from `mipmaps`, which is where level 1 is.
//
// decaf and Cemu implement this policy the same way, line for line, quirks
// included (a step down from macro-tiling at level 2 or later pads by the
// swizzle word itself), and this follows them. Checked against Cemu's -- its
// own GX2 code over its own addressing library, independent of AMD's -- in a
// scratch harness: 1,900,800 surfaces across sixteen formats, every
// dimension, fifteen sizes, mip chains, sample counts, uses and twelve tile
// modes (two of them out of range, as an uninitialised surface would have),
// with every field the call writes compared. All identical but
// 57,660 linear-special surfaces where Cemu's own arithmetic fails (a 32-bit
// overflow at 512 MiB, and no case for the multisampled dimensions, which it
// sizes at zero); there this follows decaf. Where the two disagree about the
// colour-buffer registers is set out at `color_buffer_registers`.
//
// `GX2ClearColor` and `GX2CopyColorBufferToScanBuffer` are GPU work, so
// neither touches memory when called: each writes one of the emulator's own
// packets carrying a snapshot of the colour buffer as it was at the call --
// the whole structure, so a title free to reuse it afterwards changes
// nothing -- and the work happens when the packet runs, on the CPU, in
// stream order (ADR-004). That is `clear_color_buffer` and
// `copy_to_scan_frame` below, which is all of it that needs guest memory.
// The snapshot keeps the title's virtual addresses: these packets are the
// emulator's and no hardware reads them.
//
// With a host GPU attached (ADR-006) the clear is done there instead:
// `clear_color_surface` makes the colour buffer's surface in the surface
// cache, if it is not there yet, and clears it with a render pass. A clear
// defines every pixel of what it clears, so nothing has to come from guest
// memory first, and nothing goes back: from then on the texture is that
// surface. `color_buffer_key` is how the clear finds it -- the same key a
// draw finds the surface by from the registers `GX2SetColorBuffer` wrote.
// The depth buffer's clear is done there the same way
// (`clear_depth_stencil_surface`), keyed by `depth_buffer_key`. The scan-out
// copy of a surface the GPU holds is done there too
// (`copy_surface_to_scan`), into a texture standing for the scan frame; of
// one it does not hold, on guest memory, as before.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/latte/surface.hpp>
#include <wiiu/latte/surface_cache.hpp>
#include <wiiu/latte/surface_copy.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// `GX2SurfaceDim`.
enum class SurfaceDim : u32 {
    Texture1D = 0,
    Texture2D = 1,
    Texture3D = 2,
    TextureCube = 3,
    Texture1DArray = 4,
    Texture2DArray = 5,
    Texture2DMsaa = 6,
    Texture2DMsaaArray = 7,
};

/// `GX2SurfaceUse`: what a surface is for. Only the depth and scan-buffer
/// bits change its layout; the colour-buffer bit changes the default tile
/// mode of a volume.
inline constexpr u32 surface_use_texture = 1U << 0;
inline constexpr u32 surface_use_color_buffer = 1U << 1;
inline constexpr u32 surface_use_depth_buffer = 1U << 2;
inline constexpr u32 surface_use_scan_buffer = 1U << 3;

/// `GX2TileMode`'s two defaults: let GX2 choose. Every other value is a
/// `latte::TileMode`. The two differ only in the 1D step-down for a
/// compressed format: the second judges the width against four times the
/// alignment -- pixels against pixels -- where the first compares pixels
/// against blocks.
inline constexpr u32 tile_mode_default = 0;
inline constexpr u32 tile_mode_default_bad_align = 0x20;

/// The swizzle word's third byte on a 2D-tiled surface before any level has
/// stepped down from macro-tiling.
inline constexpr u32 swizzle_first_untiled_level_unset = 0xD;

/// Level 0 plus the thirteen `mipLevelOffset` entries: 8192 pixels, the
/// largest dimension Latte has, has fourteen levels.
inline constexpr u32 max_mip_levels = 14;

/// `GX2Surface`, as it sits in guest memory. The offsets are wut's
/// `WUT_CHECK_OFFSET` lines; decaf's and Cemu's agree.
struct Surface {
    u32be dim;
    u32be width;
    u32be height;
    u32be depth; ///< slices, faces or depth, by `dim`
    u32be mip_levels;
    u32be format; ///< GX2SurfaceFormat
    u32be aa;     ///< GX2AAMode: log2 of the sample count
    u32be use;    ///< GX2SurfaceUse, or GX2R resource flags
    u32be image_size;
    u32be image;
    u32be mipmap_size;
    u32be mipmaps;
    u32be tile_mode;
    u32be swizzle;
    u32be alignment;
    u32be pitch;
    std::array<u32be, max_mip_levels - 1> mip_level_offset;
};

static_assert(sizeof(Surface) == 0x74);
static_assert(offsetof(Surface, format) == 0x14);
static_assert(offsetof(Surface, use) == 0x1C);
static_assert(offsetof(Surface, image) == 0x24);
static_assert(offsetof(Surface, tile_mode) == 0x30);
static_assert(offsetof(Surface, pitch) == 0x3C);
static_assert(offsetof(Surface, mip_level_offset) == 0x40);

/// `GX2ColorBuffer`. wut calls the last five words `regs[5]`; they are the
/// CB registers `GX2SetColorBuffer` writes, and a word of multisample state.
struct ColorBuffer {
    Surface surface;
    u32be view_mip;
    u32be view_first_slice;
    u32be view_num_slices;
    u32be aa_buffer;
    u32be aa_size;
    u32be cb_color_size;
    u32be cb_color_info;
    u32be cb_color_view;
    u32be cb_color_mask;
    u32be cmask_offset;
};

static_assert(sizeof(ColorBuffer) == 0x9C);
static_assert(offsetof(ColorBuffer, view_mip) == 0x74);
static_assert(offsetof(ColorBuffer, aa_buffer) == 0x80);
static_assert(offsetof(ColorBuffer, cb_color_size) == 0x88);
static_assert(offsetof(ColorBuffer, cmask_offset) == 0x98);

/// `GX2DepthBuffer`. wut's offsets again, which decaf's agree with: the last
/// seven words are the DB registers `GX2SetDepthBuffer` writes, and the
/// polygon-offset control that goes with the format.
struct DepthBuffer {
    Surface surface;
    u32be view_mip;
    u32be view_first_slice;
    u32be view_num_slices;
    u32be hi_z; ///< hierarchical-Z buffer, or none
    u32be hi_z_size;
    f32be depth_clear;   ///< what `GX2SetClearDepthStencil` last set
    u32be stencil_clear; ///< likewise
    u32be db_depth_size;
    u32be db_depth_view;
    u32be db_depth_info;
    u32be db_htile_surface;
    u32be db_prefetch_limit;
    u32be db_preload_control;
    u32be pa_poly_offset_control;
};

static_assert(sizeof(DepthBuffer) == 0xAC);
static_assert(offsetof(DepthBuffer, hi_z) == 0x80);
static_assert(offsetof(DepthBuffer, depth_clear) == 0x88);
static_assert(offsetof(DepthBuffer, db_depth_size) == 0x90);
static_assert(offsetof(DepthBuffer, pa_poly_offset_control) == 0xA8);

/// `GX2ClearFlags`: what a depth/stencil clear clears, and which clear
/// registers it sets on the way. Cemu's four; decaf names only the first two.
inline constexpr u32 clear_depth = 1U << 0;
inline constexpr u32 clear_stencil = 1U << 1;
inline constexpr u32 clear_set_depth_register = 1U << 2;
inline constexpr u32 clear_set_stencil_register = 1U << 3;

/// Level `level` of a surface as Latte's layout reads it: GX2's dimension,
/// samples and use turned into width, height, slices and flags.
[[nodiscard]] latte::SurfaceDesc surface_desc(const Surface& surface, u32 level);

/// How level `level` of a surface is laid out. Nothing for a format with no
/// element size, or a description addrlib refuses.
[[nodiscard]] std::optional<latte::SurfaceLayout> surface_layout(const Surface& surface, u32 level);

/// Where level `level` starts in guest memory: `image` for level 0,
/// `mipmaps` for level 1, and `mipmaps` plus the level's offset after that.
/// Nothing for a level the surface has no memory for.
[[nodiscard]] std::optional<ppcaddr_t> level_address(const Surface& surface, u32 level);

/// The bank and pipe swizzle that applies to level `level`: the surface's
/// own, in the swizzle word's second byte, up to the first level that is not
/// macro-tiled (its third byte), and none from there -- GX2's
/// `GX2GetSurfaceSwizzleOffset`.
[[nodiscard]] latte::BankPipeSwizzle level_swizzle(const Surface& surface, u32 level);

/// `GX2CalcSurfaceSizeAndAlignment`. Returns false, having logged why, if a
/// level could not be laid out -- and then the surface is left exactly as it
/// was, rather than half-updated.
bool calc_surface_size_and_alignment(Surface& surface);

/// The CB registers for a colour buffer, as `GX2InitColorBufferRegs` packs
/// them.
///
/// decaf and Cemu disagree in four places. Three are bits of CB_COLORn_INFO,
/// and this follows Cemu, whose choices are the ones a reading of the
/// hardware would make: blending is bypassed for every integer format, signed
/// or not (decaf: unsigned only); unorm, snorm and sRGB targets clamp
/// blending (decaf: never); and an integer target exports at full precision
/// (decaf: unsigned ones at 16 bits, too narrow for 32-bit integers). Nothing
/// reads those bits yet; the translator (8.11) is where they will matter.
///
/// The fourth is which level is described, and there this follows decaf: the
/// viewed level is laid out with the same depth and scan-buffer flags the
/// surface was sized with. Cemu leaves them out here, so for a 1D-tiled scan
/// buffer its CB_COLORn_SIZE has a narrower pitch than the surface it sized,
/// and a depth-flagged thick surface keeps a mode its sizing degraded. With
/// the flags given to both, the harness above found all 3,124,680 register
/// sets identical.
struct ColorBufferRegisters {
    u32 size = 0;
    u32 info = 0;
    u32 view = 0;
    u32 mask = 0;
};

/// Nothing, having logged why, if the viewed level cannot be laid out.
[[nodiscard]] std::optional<ColorBufferRegisters> color_buffer_registers(const ColorBuffer& buffer);

/// The DB registers for a depth buffer, as `GX2InitDepthBufferRegs` packs
/// them -- decaf's reading. Cemu fills in only the size, and on that the two
/// agree; the size, view and info fields are checked against Linux's
/// `r600d.h` (latte/depth_buffer.hpp).
struct DepthBufferRegisters {
    u32 size = 0;
    u32 view = 0;
    u32 info = 0;
    u32 htile_surface = 0;
    u32 prefetch_limit = 0;
    u32 preload_control = 0;
    u32 poly_offset_control = 0;
};

/// Nothing, having logged why, if the viewed level cannot be laid out.
[[nodiscard]] std::optional<DepthBufferRegisters> depth_buffer_registers(const DepthBuffer& buffer);

/// Words a `ColorBuffer` takes in a packet: the structure, field by field.
inline constexpr u32 color_buffer_words = sizeof(ColorBuffer) / 4;

[[nodiscard]] std::array<u32, color_buffer_words> color_buffer_to_words(const ColorBuffer& buffer);

[[nodiscard]] ColorBuffer color_buffer_from_words(std::span<const u32, color_buffer_words> words);

inline constexpr u32 depth_buffer_words = sizeof(DepthBuffer) / 4;

[[nodiscard]] std::array<u32, depth_buffer_words> depth_buffer_to_words(const DepthBuffer& buffer);

[[nodiscard]] DepthBuffer depth_buffer_from_words(std::span<const u32, depth_buffer_words> words);

/// What `GX2ClearColor`'s packet does when it runs: every element of the
/// viewed level and slices becomes the colour, encoded for the format
/// (`latte::encode_color`). Returns what went wrong, or nothing.
///
/// A clear writes one value everywhere, so where each element sits does not
/// matter: in the thin modes a slice is one run of bytes, and filling the run
/// fills every element of it, padding and all. Thick modes interleave four
/// slices, and are not cleared yet.
[[nodiscard]] std::string clear_color_buffer(memory::Memory& memory,
                                             const ColorBuffer& buffer,
                                             const std::array<f32, 4>& rgba);

/// The key a colour buffer's surface has in the surface cache: the base
/// `GX2SetColorBuffer` writes -- the viewed level's physical address, with
/// the swizzle -- and the size, info and view registers the snapshot holds.
/// Nothing, logged, for a level with no memory the GPU can see.
[[nodiscard]] std::optional<latte::ColorBufferKey> color_buffer_key(const memory::Memory& memory,
                                                                    const ColorBuffer& buffer);

/// What `GX2ClearColor`'s packet does when a host GPU is attached: the
/// colour buffer's surface, made in `cache` if it is not there yet, cleared
/// to the colour by a render pass on `backend` -- linear, as a clear's colour
/// is, so an sRGB surface stores it encoded, as `encode_color` does.
///
/// False, with `problem` saying why, for a clear the GPU does not do yet --
/// a format or size the cache refuses, a multisampled buffer, more than one
/// slice -- which the caller does on guest memory instead.
[[nodiscard]] bool clear_color_surface(latte::SurfaceCache& cache,
                                       gpu::GpuBackend& backend,
                                       const memory::Memory& memory,
                                       const ColorBuffer& buffer,
                                       const std::array<f32, 4>& rgba,
                                       std::string& problem);

/// The key a depth buffer's surface has in the surface cache: the base
/// `GX2SetDepthBuffer` writes and the size, info and view registers the
/// snapshot holds. Nothing, logged, for a level with no memory the GPU can
/// see.
[[nodiscard]] std::optional<latte::DepthBufferKey> depth_buffer_key(const memory::Memory& memory,
                                                                    const DepthBuffer& buffer);

/// What `GX2ClearDepthStencilEx`'s packet does when a host GPU is attached:
/// the depth buffer's surface, made in `cache` if it is not there yet,
/// cleared by a render pass -- depth, stencil or both, as `flags` asks, the
/// other loaded as it was; a surface just made has nothing to keep, and is
/// cleared whole. False, with `problem` saying why, for a clear the GPU does
/// not do yet, which the caller does on guest memory instead.
[[nodiscard]] bool clear_depth_stencil_surface(latte::SurfaceCache& cache,
                                               gpu::GpuBackend& backend,
                                               const memory::Memory& memory,
                                               const DepthBuffer& buffer,
                                               f32 depth,
                                               u8 stencil,
                                               u32 flags,
                                               std::string& problem);

/// The surface the GPU holds for this colour buffer, if it holds one
/// (ADR-006): what decides whether a scan-out copy runs on the GPU or on
/// guest memory.
[[nodiscard]] std::optional<latte::ColorSurface> held_color_surface(
    const latte::SurfaceCache& cache, const memory::Memory& memory, const ColorBuffer& buffer);

/// What `GX2CopyColorBufferToScanBuffer`'s packet does for a surface the GPU
/// holds: the viewed level's picture -- its own size, inside the texture the
/// registers padded -- copied by `copier` into `target`, the scan frame's
/// texture, as `copy_to_scan_frame` copies it into guest memory: the same
/// formats agreeing, scaled by nearest pixel, the bytes kept. False, with
/// `problem` saying why, for a copy refused before anything is encoded.
[[nodiscard]] bool copy_surface_to_scan(latte::SurfaceCopier& copier,
                                        const latte::ColorSurface& surface,
                                        const ColorBuffer& buffer,
                                        const ScanBuffer& scan,
                                        const latte::CopyTarget& target,
                                        std::string& problem);

/// What `GX2ClearDepthStencilEx`'s packet does when it runs: depth, stencil
/// or both -- as `flags` asks -- written into every element of the viewed
/// level and slices (`latte::encode_depth_stencil`), a packed stencil left
/// alone by a depth-only clear and the other way round. Returns what went
/// wrong, or nothing.
[[nodiscard]] std::string clear_depth_stencil_buffer(
    memory::Memory& memory, const DepthBuffer& buffer, f32 depth, u8 stencil, u32 flags);

/// What `GX2CopyColorBufferToScanBuffer`'s packet does when it runs: the
/// viewed level and first slice, detiled (`latte::detile`) into `frame` of
/// the scan buffer. Returns what went wrong, or nothing.
///
/// The formats have to agree on their colour format and element size --
/// unorm and sRGB 8_8_8_8 are the same bytes, and the scan-out does not
/// convert. A buffer of another size is scaled to the frame by nearest
/// pixel, which stands in for whatever filter the hardware's copy uses -- a
/// WHB title never asks, its colour buffers being its frames' size. A
/// multisampled buffer shows its first sample rather than a resolve.
[[nodiscard]] std::string copy_to_scan_frame(memory::Memory& memory,
                                             const ColorBuffer& buffer,
                                             const ScanBuffer& scan,
                                             ppcaddr_t frame);

/// Register `GX2CalcSurfaceSizeAndAlignment`, `GX2InitColorBufferRegs`,
/// `GX2ClearColor`, `GX2CopyColorBufferToScanBuffer`, `GX2InitDepthBufferRegs`,
/// `GX2ClearDepthStencilEx`, `GX2SetColorBuffer` and `GX2SetDepthBuffer`.
/// Called by `register_gx2`.
void register_gx2_surface(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
