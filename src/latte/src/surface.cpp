#include <wiiu/common/log.hpp>
#include <wiiu/latte/formats.hpp>
#include <wiiu/latte/surface.hpp>

#include <addrlib/addrinterface.h>
#include <algorithm>
#include <cstdlib>
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
