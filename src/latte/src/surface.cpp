#include <wiiu/common/log.hpp>
#include <wiiu/latte/formats.hpp>
#include <wiiu/latte/surface.hpp>

#include <addrlib/addrinterface.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace wiiu::latte {

namespace {

// --------------------------------------------------------------------------
// The Wii U's memory, as addrlib needs to be told it
// --------------------------------------------------------------------------
//
// An R600-family engine, and the tiling configuration decaf gives addrlib
// for the Wii U: 0x44902, which addrlib's own decoder reads as two pipes,
// four banks, a 256-byte pipe interleave and 2 KiB rows (with one bank swap
// and a sample split of one). Family 0x51 and revision 71 only select the
// R600 code path.

constexpr u32 chip_family = 0x51;
constexpr u32 chip_revision = 71;
constexpr u32 gb_addr_config = 0x44902;

void* allocate(const ADDR_ALLOCSYSMEM_INPUT* input) {
    return std::malloc(input->sizeInBytes);
}

ADDR_E_RETURNCODE release(const ADDR_FREESYSMEM_INPUT* input) {
    std::free(input->pVirtAddr);
    return ADDR_OK;
}

/// One handle for the process, made on first use. addrlib's queries do not
/// change it, so it is shared.
[[nodiscard]] ADDR_HANDLE handle() {
    static std::once_flag once;
    static ADDR_HANDLE library = nullptr;
    std::call_once(once, [] {
        ADDR_CREATE_INPUT input{};
        ADDR_CREATE_OUTPUT output{};
        input.size = sizeof(input);
        output.size = sizeof(output);
        input.chipEngine = CIASICIDGFXENGINE_R600;
        input.chipFamily = chip_family;
        input.chipRevision = chip_revision;
        input.createFlags.fillSizeFields = 1;
        input.regValue.gbAddrConfig = gb_addr_config;
        input.callbacks.allocSysMem = &allocate;
        input.callbacks.freeSysMem = &release;
        if (AddrCreate(&input, &output) == ADDR_OK) {
            library = output.hLib;
        } else {
            WIIU_LOG_ERROR(Gpu, "addrlib refused the Wii U's memory configuration");
        }
    });
    return library;
}

/// GX2's linear-special mode, which addrlib does not have: no alignment at
/// all, so the pitch is the width and the size is what the elements need.
[[nodiscard]] SurfaceLayout linear_special(const SurfaceDesc& surface, u32 bits) {
    const u32 block = is_block_compressed(surface.format) ? 4 : 1;
    const u32 width = std::max<u32>(1, surface.width >> surface.mip_level);
    const u32 height = std::max<u32>(1, surface.height >> surface.mip_level);

    SurfaceLayout layout;
    layout.tile_mode = TileMode::LinearSpecial;
    layout.bits_per_element = bits;
    layout.pitch = std::max<u32>(1, (width + block - 1) / block);
    layout.height = std::max<u32>(1, (height + block - 1) / block);
    layout.depth = std::max<u32>(1, surface.slices);
    layout.size = static_cast<u64>(layout.pitch) * layout.height * layout.depth *
                  std::max<u32>(1, surface.samples) * (bits / 8);
    layout.base_align = 1;
    layout.pitch_align = 1;
    layout.height_align = 1;
    layout.depth_align = 1;
    layout.slice_size = static_cast<u32>(layout.size / layout.depth);
    return layout;
}

} // namespace

std::string_view tile_mode_name(TileMode mode) noexcept {
    switch (mode) {
        case TileMode::LinearGeneral:
            return "linear general";
        case TileMode::LinearAligned:
            return "linear aligned";
        case TileMode::Tiled1DThin1:
            return "1D thin1";
        case TileMode::Tiled1DThick:
            return "1D thick";
        case TileMode::Tiled2DThin1:
            return "2D thin1";
        case TileMode::Tiled2DThin2:
            return "2D thin2";
        case TileMode::Tiled2DThin4:
            return "2D thin4";
        case TileMode::Tiled2DThick:
            return "2D thick";
        case TileMode::Tiled2BThin1:
            return "2B thin1";
        case TileMode::Tiled2BThin2:
            return "2B thin2";
        case TileMode::Tiled2BThin4:
            return "2B thin4";
        case TileMode::Tiled2BThick:
            return "2B thick";
        case TileMode::Tiled3DThin1:
            return "3D thin1";
        case TileMode::Tiled3DThick:
            return "3D thick";
        case TileMode::Tiled3BThin1:
            return "3B thin1";
        case TileMode::Tiled3BThick:
            return "3B thick";
        case TileMode::LinearSpecial:
            return "linear special";
    }
    return "unknown";
}

std::optional<SurfaceLayout> compute_layout(const SurfaceDesc& surface) {
    const u32 bits = bits_per_element(surface.format);
    if (bits == 0) {
        return std::nullopt;
    }
    if (surface.tile_mode == TileMode::LinearSpecial) {
        return linear_special(surface, bits);
    }
    if (static_cast<u32>(surface.tile_mode) > static_cast<u32>(TileMode::Tiled3BThick)) {
        return std::nullopt;
    }

    ADDR_HANDLE const library = handle();
    if (library == nullptr) {
        return std::nullopt;
    }

    ADDR_COMPUTE_SURFACE_INFO_INPUT input{};
    input.size = sizeof(input);
    input.tileMode = static_cast<AddrTileMode>(surface.tile_mode);
    input.format = static_cast<AddrFormat>(color_format(surface.format));
    input.bpp = bits;
    input.width = std::max<u32>(1, surface.width >> surface.mip_level);
    input.height = std::max<u32>(1, surface.height >> surface.mip_level);
    input.numSlices = std::max<u32>(1, surface.slices);
    input.numSamples = std::max<u32>(1, surface.samples);
    input.numFrags = input.numSamples;
    input.mipLevel = surface.mip_level;
    input.flags.depth = surface.depth ? 1 : 0;
    input.flags.display = surface.display ? 1 : 0;
    input.flags.cube = surface.cube ? 1 : 0;
    input.flags.volume = surface.volume ? 1 : 0;
    input.flags.inputBaseMap = surface.mip_level == 0 ? 1 : 0;

    ADDR_COMPUTE_SURFACE_INFO_OUTPUT output{};
    output.size = sizeof(output);
    if (AddrComputeSurfaceInfo(library, &input, &output) != ADDR_OK) {
        return std::nullopt;
    }

    SurfaceLayout layout;
    layout.tile_mode = static_cast<TileMode>(output.tileMode);
    layout.bits_per_element = output.bpp;
    layout.pitch = output.pitch;
    layout.height = output.height;
    layout.depth = output.depth;
    layout.size = output.surfSize;
    layout.base_align = output.baseAlign;
    layout.pitch_align = output.pitchAlign;
    layout.height_align = output.heightAlign;
    layout.depth_align = output.depthAlign;
    layout.slice_size = output.sliceSize;
    return layout;
}

u64 element_offset(const SurfaceDesc& surface,
                   const SurfaceLayout& layout,
                   u32 x,
                   u32 y,
                   u32 slice,
                   u32 sample,
                   u32 pipe_swizzle,
                   u32 bank_swizzle) {
    if (layout.tile_mode == TileMode::LinearSpecial) {
        const u64 element = (static_cast<u64>(slice) * layout.height + y) * layout.pitch + x;
        return element * (layout.bits_per_element / 8);
    }

    ADDR_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT input{};
    input.size = sizeof(input);
    input.x = x;
    input.y = y;
    input.slice = slice;
    input.sample = sample;
    input.bpp = layout.bits_per_element;
    input.pitch = layout.pitch;
    input.height = layout.height;
    input.numSlices = layout.depth;
    input.numSamples = std::max<u32>(1, surface.samples);
    input.numFrags = input.numSamples;
    input.tileMode = static_cast<AddrTileMode>(layout.tile_mode);
    input.isDepth = surface.depth;
    input.pipeSwizzle = pipe_swizzle;
    input.bankSwizzle = bank_swizzle;

    ADDR_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT output{};
    output.size = sizeof(output);
    static_cast<void>(AddrComputeSurfaceAddrFromCoord(handle(), &input, &output));
    return output.addr;
}

Extent level_extent(const SurfaceDesc& surface) noexcept {
    const u32 block = is_block_compressed(surface.format) ? 4 : 1;
    const u32 width = std::max<u32>(1, surface.width >> surface.mip_level);
    const u32 height = std::max<u32>(1, surface.height >> surface.mip_level);
    return Extent{.width = (width + block - 1) / block, .height = (height + block - 1) / block};
}

namespace {

/// Whether every 8x8 micro-tile of the level sits in one piece, laid out the
/// same way as every other -- so one tile's pattern places them all.
[[nodiscard]] bool micro_tiles_are_whole(const SurfaceDesc& surface,
                                         const SurfaceLayout& layout) noexcept {
    switch (layout.tile_mode) {
        case TileMode::LinearGeneral:
        case TileMode::LinearAligned:
        case TileMode::LinearSpecial:
        case TileMode::Tiled1DThin1:
            // Linear is a pitch and nothing else, and a 1D micro-tile is
            // always stored whole.
            return true;
        case TileMode::Tiled2DThin1:
        case TileMode::Tiled2DThin2:
        case TileMode::Tiled2DThin4:
        case TileMode::Tiled2BThin1:
        case TileMode::Tiled2BThin2:
        case TileMode::Tiled2BThin4:
        case TileMode::Tiled3DThin1:
        case TileMode::Tiled3BThin1: {
            // Bank and pipe are chosen per 256-byte group; a micro-tile that
            // fits in one is never split between them.
            const u64 tile_bits =
                64ULL * layout.bits_per_element * std::max<u32>(1, surface.samples);
            return tile_bits <= 256 * 8;
        }
        case TileMode::Tiled1DThick:
        case TileMode::Tiled2DThick:
        case TileMode::Tiled2BThick:
        case TileMode::Tiled3DThick:
        case TileMode::Tiled3BThick:
            return false;
    }
    return false;
}

} // namespace

bool detile(const SurfaceDesc& surface,
            const SurfaceLayout& layout,
            std::span<const u8> level,
            u32 slice,
            u32 sample,
            BankPipeSwizzle swizzle,
            std::span<u8> rows) {
    const Extent extent = level_extent(surface);
    const u32 bytes = layout.bits_per_element / 8;
    if (bytes == 0 || rows.size() < static_cast<u64>(extent.width) * extent.height * bytes) {
        return false;
    }

    const auto offset_of = [&](u32 x, u32 y) {
        return element_offset(surface, layout, x, y, slice, sample, swizzle.pipe, swizzle.bank);
    };
    const auto place = [&](u32 x, u32 y, u64 offset) {
        if (offset + bytes > level.size()) {
            return false;
        }
        const u64 at = ((static_cast<u64>(y) * extent.width) + x) * bytes;
        std::memcpy(rows.data() + at, level.data() + offset, bytes);
        return true;
    };

    if (!micro_tiles_are_whole(surface, layout)) {
        for (u32 y = 0; y < extent.height; ++y) {
            for (u32 x = 0; x < extent.width; ++x) {
                if (!place(x, y, offset_of(x, y))) {
                    return false;
                }
            }
        }
        return true;
    }

    // Where each element of a micro-tile is, from the tile's corner. The
    // corner is its first element in every mode, so these are never negative.
    constexpr u32 tile = 8;
    std::array<u64, tile * tile> pattern{};
    const u64 origin = offset_of(0, 0);
    for (u32 j = 0; j < tile; ++j) {
        for (u32 i = 0; i < tile; ++i) {
            pattern[(j * tile) + i] = offset_of(i, j) - origin;
        }
    }

    for (u32 ty = 0; ty < extent.height; ty += tile) {
        for (u32 tx = 0; tx < extent.width; tx += tile) {
            const u64 base = offset_of(tx, ty);
            const u32 rows_here = std::min(tile, extent.height - ty);
            const u32 columns_here = std::min(tile, extent.width - tx);
            for (u32 j = 0; j < rows_here; ++j) {
                for (u32 i = 0; i < columns_here; ++i) {
                    if (!place(tx + i, ty + j, base + pattern[(j * tile) + i])) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

BankPipeSwizzle split_swizzle(u32 gx2_swizzle) noexcept {
    ADDR_EXTRACT_BANKPIPE_SWIZZLE_INPUT input{};
    ADDR_EXTRACT_BANKPIPE_SWIZZLE_OUTPUT output{};
    input.size = sizeof(input);
    output.size = sizeof(output);
    input.base256b = (gx2_swizzle >> 8) & 0xFF;
    if (ADDR_HANDLE const library = handle(); library != nullptr) {
        static_cast<void>(AddrExtractBankPipeSwizzle(library, &input, &output));
    }
    return BankPipeSwizzle{.pipe = output.pipeSwizzle, .bank = output.bankSwizzle};
}

} // namespace wiiu::latte
