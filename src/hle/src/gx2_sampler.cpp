// Phase 8.1 — samplers. See gx2_sampler.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_sampler.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/samplers.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <string_view>

namespace wiiu::hle::gx2 {

namespace {

using latte::pm4::Opcode;

/// The three words, unpacked.
struct Words {
    latte::TexSamplerWord0 word0{};
    latte::TexSamplerWord1 word1{};
    latte::TexSamplerWord2 word2{};
};

/// Change the title's sampler at `address` in place: read it, let
/// `change` set its fields, write it back.
template <typename Change>
void change_sampler(Call& call, u32 address, std::string_view what, Change change) {
    Sampler sampler{};
    if (!read_guest(call.memory, address, sampler)) {
        WIIU_LOG_ERROR(Gx2, "{}: no sampler at {:#010x}", what, address);
        return;
    }
    Words words;
    words.word0.raw = sampler.word0;
    words.word1.raw = sampler.word1;
    words.word2.raw = sampler.word2;
    change(words);
    sampler.word0 = words.word0.raw;
    sampler.word1 = words.word1.raw;
    sampler.word2 = words.word2.raw;
    static_cast<void>(write_guest(call.memory, address, sampler));
}

/// `value` as fixed point with `scale` steps to one: floored, held to
/// [low, high]; zero, held likewise, for a NaN.
[[nodiscard]] s32 to_fixed(f32 value, f32 scale, s32 low, s32 high) {
    const f64 scaled = std::floor(static_cast<f64>(value) * static_cast<f64>(scale));
    if (std::isnan(scaled)) {
        return std::clamp(0, low, high);
    }
    return static_cast<s32>(std::clamp(scaled, static_cast<f64>(low), static_cast<f64>(high)));
}

/// The field limits: ten bits of LOD, twelve of signed bias, six of
/// anisotropy bias.
constexpr s32 max_lod = 1023;
constexpr s32 min_lod_bias = -2048;
constexpr s32 max_lod_bias = 2047;
constexpr s32 max_aniso_bias = 63;

[[nodiscard]] latte::TexClamp clamp_mode(u32 value) noexcept {
    return static_cast<latte::TexClamp>(value & 7);
}

[[nodiscard]] latte::TexXYFilter xy_filter(u32 value) noexcept {
    return static_cast<latte::TexXYFilter>(value & 7);
}

[[nodiscard]] latte::TexZFilter z_filter(u32 value) noexcept {
    return static_cast<latte::TexZFilter>(value & 3);
}

/// A filter's anisotropic form (Cemu): point and bilinear have one; any
/// other is taken as point.
[[nodiscard]] latte::TexXYFilter anisotropic(u32 value) noexcept {
    switch (xy_filter(value)) {
        case latte::TexXYFilter::Bilinear:
            return latte::TexXYFilter::AnisoBilinear;
        default:
            return latte::TexXYFilter::AnisoPoint;
    }
}

/// `void GX2InitSampler(GX2Sampler *sampler, GX2TexClampMode clampMode,
/// GX2TexXYFilterMode minMagFilterMode)`.
void gx2_init_sampler(Call& call, u32 address, u32 clamp, u32 filter) {
    Words words;
    words.word0.raw = 0;
    words.word0.clamp_x = clamp_mode(clamp);
    words.word0.clamp_y = clamp_mode(clamp);
    words.word0.clamp_z = clamp_mode(clamp);
    words.word0.xy_mag_filter = xy_filter(filter);
    words.word0.xy_min_filter = xy_filter(filter);
    words.word0.z_filter = latte::TexZFilter::Point;
    words.word0.mip_filter = latte::TexZFilter::Point;
    words.word0.tex_array_override = true;
    words.word1.raw = 0;
    words.word1.max_lod = static_cast<u32>(max_lod);
    words.word2.raw = 0;
    words.word2.type = true;
    const Sampler sampler{
        .word0 = words.word0.raw, .word1 = words.word1.raw, .word2 = words.word2.raw};
    if (!write_guest(call.memory, address, sampler)) {
        WIIU_LOG_ERROR(Gx2, "GX2InitSampler: no sampler at {:#010x}", address);
    }
}

/// `void GX2InitSamplerXYFilter(GX2Sampler *sampler, GX2TexXYFilterMode
/// magFilter, GX2TexXYFilterMode minFilter, GX2TexAnisoRatio maxAniso)`.
void gx2_init_sampler_xy_filter(Call& call, u32 address, u32 mag, u32 min, u32 max_aniso) {
    change_sampler(call, address, "GX2InitSamplerXYFilter", [&](Words& words) {
        if (max_aniso == 0) {
            words.word0.xy_mag_filter = xy_filter(mag);
            words.word0.xy_min_filter = xy_filter(min);
        } else {
            words.word0.xy_mag_filter = anisotropic(mag);
            words.word0.xy_min_filter = anisotropic(min);
        }
        words.word0.max_aniso_ratio = max_aniso;
    });
}

/// `void GX2InitSamplerZMFilter(GX2Sampler *sampler, GX2TexZFilterMode
/// zFilter, GX2TexMipFilterMode mipFilter)`.
void gx2_init_sampler_zm_filter(Call& call, u32 address, u32 z, u32 mip) {
    change_sampler(call, address, "GX2InitSamplerZMFilter", [&](Words& words) {
        words.word0.z_filter = z_filter(z);
        words.word0.mip_filter = z_filter(mip);
    });
}

/// `void GX2InitSamplerLOD(GX2Sampler *sampler, float minLod, float
/// maxLod, float lodBias)`.
void gx2_init_sampler_lod(Call& call, u32 address, f32 min, f32 max, f32 bias) {
    change_sampler(call, address, "GX2InitSamplerLOD", [&](Words& words) {
        words.word1.min_lod = static_cast<u32>(to_fixed(min, latte::lod_scale, 0, max_lod));
        words.word1.max_lod = static_cast<u32>(to_fixed(max, latte::lod_scale, 0, max_lod));
        words.word1.lod_bias =
            static_cast<u32>(to_fixed(bias, latte::lod_scale, min_lod_bias, max_lod_bias));
    });
}

/// `void GX2InitSamplerClamping(GX2Sampler *sampler, GX2TexClampMode clampX,
/// GX2TexClampMode clampY, GX2TexClampMode clampZ)`.
void gx2_init_sampler_clamping(Call& call, u32 address, u32 x, u32 y, u32 z) {
    change_sampler(call, address, "GX2InitSamplerClamping", [&](Words& words) {
        words.word0.clamp_x = clamp_mode(x);
        words.word0.clamp_y = clamp_mode(y);
        words.word0.clamp_z = clamp_mode(z);
    });
}

/// `void GX2InitSamplerBorderType(GX2Sampler *sampler, GX2TexBorderType
/// borderType)`.
void gx2_init_sampler_border_type(Call& call, u32 address, u32 type) {
    change_sampler(call, address, "GX2InitSamplerBorderType", [&](Words& words) {
        words.word0.border_color_type = static_cast<latte::TexBorderColor>(type & 3);
    });
}

/// `void GX2InitSamplerDepthCompare(GX2Sampler *sampler, GX2CompareFunction
/// depthCompare)`.
void gx2_init_sampler_depth_compare(Call& call, u32 address, u32 compare) {
    change_sampler(call, address, "GX2InitSamplerDepthCompare", [&](Words& words) {
        words.word0.depth_compare_function = static_cast<latte::CompareFunction>(compare & 7);
    });
}

/// `void GX2InitSamplerFilterAdjust(GX2Sampler *sampler, BOOL
/// highPrecision, GX2TexMipPerfMode perfMip, GX2TexZPerfMode perfZ)`.
void gx2_init_sampler_filter_adjust(
    Call& call, u32 address, bool high_precision, u32 perf_mip, u32 perf_z) {
    change_sampler(call, address, "GX2InitSamplerFilterAdjust", [&](Words& words) {
        words.word2.high_precision_filter = high_precision;
        words.word2.perf_mip = perf_mip;
        words.word2.perf_z = perf_z;
    });
}

/// `void GX2InitSamplerLODAdjust(GX2Sampler *sampler, float anisoBias, BOOL
/// lodUsesMinorAxis)`.
void gx2_init_sampler_lod_adjust(Call& call, u32 address, f32 aniso_bias, bool minor_axis) {
    change_sampler(call, address, "GX2InitSamplerLODAdjust", [&](Words& words) {
        words.word2.aniso_bias =
            static_cast<u32>(to_fixed(aniso_bias, latte::aniso_bias_scale, 0, max_aniso_bias));
        words.word0.lod_uses_minor_axis = minor_axis;
    });
}

/// `void GX2InitSamplerRoundingMode(GX2Sampler *sampler, GX2RoundingMode
/// roundingMode)`: round to even, or truncate.
void gx2_init_sampler_rounding_mode(Call& call, u32 address, u32 mode) {
    change_sampler(call, address, "GX2InitSamplerRoundingMode", [&](Words& words) {
        words.word2.truncate_coord = (mode & 1) != 0;
    });
}

/// `void GX2Set{Pixel,Vertex,Geometry}Sampler(const GX2Sampler *sampler,
/// uint32_t unit)`.
void set_sampler(Call& call, u32 address, u32 unit, u32 first, std::string_view what) {
    if (unit >= latte::sampler::per_stage) {
        WIIU_LOG_ERROR(Gx2, "{}({}): there are only eighteen samplers a stage", what, unit);
        return;
    }
    Sampler sampler{};
    if (!read_guest(call.memory, address, sampler)) {
        WIIU_LOG_ERROR(Gx2, "{}: no sampler at {:#010x}", what, address);
        return;
    }
    std::vector<u32> packets;
    append_sampler(packets, first + unit, sampler);
    static_cast<void>(emit(call, packets));
}

void gx2_set_pixel_sampler(Call& call, u32 sampler, u32 unit) {
    set_sampler(call, sampler, unit, latte::sampler::ps_base, "GX2SetPixelSampler");
}

void gx2_set_vertex_sampler(Call& call, u32 sampler, u32 unit) {
    set_sampler(call, sampler, unit, latte::sampler::vs_base, "GX2SetVertexSampler");
}

void gx2_set_geometry_sampler(Call& call, u32 sampler, u32 unit) {
    set_sampler(call, sampler, unit, latte::sampler::gs_base, "GX2SetGeometrySampler");
}

void gx2_set_compute_sampler(Call& call, u32 sampler, u32 unit) {
    set_sampler(call, sampler, unit, latte::sampler::vs_base, "GX2SetComputeSampler");
}

/// `void GX2Set{Pixel,Vertex,Geometry}SamplerBorderColor(uint32_t unit,
/// float red, float green, float blue, float alpha)`.
void set_border_color(
    Call& call, u32 unit, std::array<f32, 4> color, u32 border_base, std::string_view what) {
    if (unit >= latte::sampler::per_stage) {
        WIIU_LOG_ERROR(Gx2, "{}({}): there are only eighteen samplers a stage", what, unit);
        return;
    }
    std::array<u32, 4> values{};
    std::ranges::transform(color, values.begin(), [](f32 c) { return std::bit_cast<u32>(c); });
    std::vector<u32> packets;
    append_registers(packets,
                     Opcode::SetConfigReg,
                     border_base + (unit * latte::sampler::border_stride),
                     values);
    static_cast<void>(emit(call, packets));
}

void gx2_set_pixel_sampler_border_color(Call& call, u32 unit, f32 r, f32 g, f32 b, f32 a) {
    set_border_color(
        call, unit, {r, g, b, a}, latte::sampler::ps_border_base, "GX2SetPixelSamplerBorderColor");
}

void gx2_set_vertex_sampler_border_color(Call& call, u32 unit, f32 r, f32 g, f32 b, f32 a) {
    set_border_color(
        call, unit, {r, g, b, a}, latte::sampler::vs_border_base, "GX2SetVertexSamplerBorderColor");
}

void gx2_set_geometry_sampler_border_color(Call& call, u32 unit, f32 r, f32 g, f32 b, f32 a) {
    set_border_color(call,
                     unit,
                     {r, g, b, a},
                     latte::sampler::gs_border_base,
                     "GX2SetGeometrySamplerBorderColor");
}

} // namespace

void append_sampler(std::vector<u32>& out, u32 id, const Sampler& sampler) {
    const std::array<u32, latte::sampler::words> words{sampler.word0, sampler.word1, sampler.word2};
    append_registers(out, Opcode::SetSampler, latte::sampler::address(id), words);
}

void register_gx2_sampler(Registry& registry) {
    registry.add("gx2", "GX2InitSampler", gx2_init_sampler);
    registry.add("gx2", "GX2InitSamplerXYFilter", gx2_init_sampler_xy_filter);
    registry.add("gx2", "GX2InitSamplerZMFilter", gx2_init_sampler_zm_filter);
    registry.add("gx2", "GX2InitSamplerLOD", gx2_init_sampler_lod);
    registry.add("gx2", "GX2InitSamplerClamping", gx2_init_sampler_clamping);
    registry.add("gx2", "GX2InitSamplerBorderType", gx2_init_sampler_border_type);
    registry.add("gx2", "GX2InitSamplerDepthCompare", gx2_init_sampler_depth_compare);
    registry.add("gx2", "GX2InitSamplerFilterAdjust", gx2_init_sampler_filter_adjust);
    registry.add("gx2", "GX2InitSamplerLODAdjust", gx2_init_sampler_lod_adjust);
    registry.add("gx2", "GX2InitSamplerRoundingMode", gx2_init_sampler_rounding_mode);
    registry.add("gx2", "GX2SetPixelSampler", gx2_set_pixel_sampler);
    registry.add("gx2", "GX2SetVertexSampler", gx2_set_vertex_sampler);
    registry.add("gx2", "GX2SetGeometrySampler", gx2_set_geometry_sampler);
    registry.add("gx2", "GX2SetComputeSampler", gx2_set_compute_sampler);
    registry.add("gx2", "GX2SetPixelSamplerBorderColor", gx2_set_pixel_sampler_border_color);
    registry.add("gx2", "GX2SetVertexSamplerBorderColor", gx2_set_vertex_sampler_border_color);
    registry.add("gx2", "GX2SetGeometrySamplerBorderColor", gx2_set_geometry_sampler_border_color);
}

} // namespace wiiu::hle::gx2
