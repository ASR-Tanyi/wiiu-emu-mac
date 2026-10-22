// Phase 8.1 — textures. See gx2_texture.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/gx2_texture.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/resources.hpp>

#include <array>
#include <optional>
#include <string_view>

namespace wiiu::hle::gx2 {

namespace {

using latte::pm4::Opcode;

/// `GX2SurfaceFormat`'s type bits, above the FMT_* number in its low six.
constexpr u32 format_number_mask = 0x3F;
constexpr u32 format_integer = 0x100;
constexpr u32 format_signed = 0x200;
constexpr u32 format_srgb = 0x400;
constexpr u32 format_float = 0x800;

/// FMT_BC1 to FMT_BC5: a pitch counted in 4x4 blocks.
constexpr u32 fmt_bc1 = 0x31;
constexpr u32 fmt_bc5 = 0x35;

/// What every texture asks of the filter, in WORD6 (decaf and Cemu).
constexpr u32 default_max_aniso_ratio = 4;
constexpr u32 default_perf_modulation = 7;

/// REQUEST_SIZE in WORD4 (decaf and Cemu).
constexpr u32 default_request_size = 2;

/// `GX2AAMode`'s 2x, 4x and 8x.
constexpr u32 max_aa_mode = 3;

/// A count GX2 holds at one when the title leaves it zero.
void at_least_one(u32be& value) {
    if (value == 0) {
        value = 1;
    }
}

/// Level 0's and level 1's base registers, or nothing -- logged -- if the
/// GPU cannot see them.
[[nodiscard]] std::optional<TextureAddresses>
texture_addresses(const memory::Memory& memory, const Surface& surface, std::string_view what) {
    const std::optional<physaddr_t> image = physical_address(memory, surface.image, what);
    const ppcaddr_t mip_virtual = surface.mipmaps == 0 ? surface.image : surface.mipmaps;
    const std::optional<physaddr_t> mip = physical_address(memory, mip_virtual, what);
    if (!image.has_value() || !mip.has_value()) {
        return std::nullopt;
    }
    physaddr_t base = *image;
    physaddr_t mip_base = *mip;
    if (is_macro_tiled(surface.tile_mode)) {
        const u32 first_untiled = (surface.swizzle >> 16) & 0xFF;
        const u32 swizzle = surface.swizzle & 0xFFFF;
        if (first_untiled > 0) {
            base ^= swizzle;
        }
        if (first_untiled > 1) {
            mip_base ^= swizzle;
        }
    }
    return TextureAddresses{.base = base >> 8, .mip = mip_base >> 8};
}

/// `void GX2InitTextureRegs(GX2Texture *texture)`.
void gx2_init_texture_regs(Call& call, u32 address) {
    Texture texture{};
    if (!read_guest(call.memory, address, texture)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitTextureRegs: no texture at {:#010x}", address);
        return;
    }
    init_texture_regs(texture);
    static_cast<void>(write_guest(call.memory, address, texture));
}

/// `void GX2Set{Pixel,Vertex,Geometry}Texture(const GX2Texture *texture,
/// uint32_t unit)`.
void set_texture(Call& call, u32 address, u32 unit, u32 first_resource, std::string_view what) {
    if (unit >= latte::resource::textures_per_stage) {
        WIIU_LOG_ERROR(Gx2, "{}({}): there are only eighteen texture units a stage", what, unit);
        return;
    }
    Texture texture{};
    if (!read_guest(call.memory, address, texture)) {
        WIIU_LOG_ERROR(Gx2, "{}: no texture at {:#010x}", what, address);
        return;
    }
    const std::optional<TextureAddresses> addresses =
        texture_addresses(call.memory, texture.surface, what);
    if (!addresses.has_value()) {
        return;
    }
    std::vector<u32> packets;
    append_texture(packets, first_resource + unit, texture, *addresses);
    static_cast<void>(emit(call, packets));
}

void gx2_set_pixel_texture(Call& call, u32 texture, u32 unit) {
    set_texture(call, texture, unit, latte::resource::ps_base, "GX2SetPixelTexture");
}

void gx2_set_vertex_texture(Call& call, u32 texture, u32 unit) {
    set_texture(call, texture, unit, latte::resource::vs_base, "GX2SetVertexTexture");
}

void gx2_set_geometry_texture(Call& call, u32 texture, u32 unit) {
    set_texture(call, texture, unit, latte::resource::gs_base, "GX2SetGeometryTexture");
}

void gx2_set_compute_texture(Call& call, u32 texture, u32 unit) {
    set_texture(call, texture, unit, latte::resource::vs_base, "GX2SetComputeTexture");
}

} // namespace

void init_texture_regs(Texture& texture) {
    at_least_one(texture.view_num_mips);
    at_least_one(texture.view_num_slices);
    at_least_one(texture.surface.height);
    at_least_one(texture.surface.depth);
    at_least_one(texture.surface.mip_levels);

    const Surface& surface = texture.surface;
    const u32 format = surface.format;
    const u32 data_format = format & format_number_mask;
    const auto dim = static_cast<SurfaceDim>(surface.dim.value());

    latte::TexResourceWord0 word0{.raw = 0};
    word0.dim = static_cast<latte::TexDim>(surface.dim & 7);
    word0.tile_mode = surface.tile_mode & 0xF; // linear-special is linear to the hardware
    word0.tile_type = (surface.use & surface_use_depth_buffer) != 0;
    u32 pitch = surface.pitch;
    if (data_format >= fmt_bc1 && data_format <= fmt_bc5) {
        pitch *= 4;
    }
    word0.pitch = (pitch >> 3) - 1;
    word0.width = surface.width - 1;

    latte::TexResourceWord1 word1{.raw = 0};
    word1.height = surface.height - 1;
    switch (dim) {
        case SurfaceDim::TextureCube:
            word1.depth = (surface.depth / 6) - 1;
            break;
        case SurfaceDim::Texture3D:
        case SurfaceDim::Texture1DArray:
        case SurfaceDim::Texture2DArray:
        case SurfaceDim::Texture2DMsaaArray:
            word1.depth = surface.depth - 1;
            break;
        default:
            word1.depth = 0;
            break;
    }
    word1.data_format = data_format;

    latte::TexResourceWord4 word4{.raw = 0};
    const latte::FormatComp comp =
        (format & format_signed) != 0 ? latte::FormatComp::Signed : latte::FormatComp::Unsigned;
    word4.format_comp_x = comp;
    word4.format_comp_y = comp;
    word4.format_comp_z = comp;
    word4.format_comp_w = comp;
    if ((format & format_float) != 0) {
        word4.num_format_all = latte::NumFormat::Scaled;
    } else if ((format & format_integer) != 0) {
        word4.num_format_all = latte::NumFormat::Int;
    } else {
        word4.num_format_all = latte::NumFormat::Norm;
    }
    word4.force_degamma = (format & format_srgb) != 0;
    word4.endian_swap = latte::Endian::None;
    word4.request_size = default_request_size;
    const u32 comp_map = texture.comp_map;
    word4.dst_sel_x = static_cast<latte::Sel>((comp_map >> 24) & 7);
    word4.dst_sel_y = static_cast<latte::Sel>((comp_map >> 16) & 7);
    word4.dst_sel_z = static_cast<latte::Sel>((comp_map >> 8) & 7);
    word4.dst_sel_w = static_cast<latte::Sel>(comp_map & 7);
    word4.base_level = texture.view_first_mip;

    latte::TexResourceWord5 word5{.raw = 0};
    word5.last_level = texture.view_first_mip + texture.view_num_mips - 1;
    word5.base_array = texture.view_first_slice;
    word5.last_array = texture.view_first_slice + texture.view_num_slices - 1;
    if (dim == SurfaceDim::TextureCube && word1.depth.value() != 0) {
        word5.yuv_conv = 1;
    }
    if (surface.aa >= 1 && surface.aa <= max_aa_mode) {
        word5.last_level = surface.aa;
    }

    latte::TexResourceWord6 word6{.raw = 0};
    word6.max_aniso_ratio = default_max_aniso_ratio;
    word6.perf_modulation = default_perf_modulation;
    word6.type = latte::ResourceType::ValidTexture;

    texture.word0 = word0.raw;
    texture.word1 = word1.raw;
    texture.word4 = word4.raw;
    texture.word5 = word5.raw;
    texture.word6 = word6.raw;
}

void append_texture(std::vector<u32>& out,
                    u32 resource,
                    const Texture& texture,
                    TextureAddresses addresses) {
    const std::array<u32, latte::resource::words> words{texture.word0,
                                                        texture.word1,
                                                        addresses.base,
                                                        addresses.mip,
                                                        texture.word4,
                                                        texture.word5,
                                                        texture.word6};
    append_registers(out, Opcode::SetResource, latte::resource::address(resource), words);
}

void register_gx2_texture(Registry& registry) {
    registry.add("gx2", "GX2InitTextureRegs", gx2_init_texture_regs);
    registry.add("gx2", "GX2SetPixelTexture", gx2_set_pixel_texture);
    registry.add("gx2", "GX2SetVertexTexture", gx2_set_vertex_texture);
    registry.add("gx2", "GX2SetGeometryTexture", gx2_set_geometry_texture);
    registry.add("gx2", "GX2SetComputeTexture", gx2_set_compute_texture);
}

} // namespace wiiu::hle::gx2
