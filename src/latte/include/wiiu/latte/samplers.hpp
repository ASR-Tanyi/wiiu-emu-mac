// Latte's samplers: the three-word descriptors SET_SAMPLER writes, which
// tell a texture fetch how to address and filter what it reads, and the
// border colours a sampler of border type REGISTER reads.
//
// The sampler window is one array of them, eighteen a stage: the pixel
// stage's first, then the vertex stage's from 18, then the geometry stage's
// from 36 (decaf and Cemu). A fetch names a sampler from its own stage's
// first.
//
// The fields are decaf's. Cemu agrees on every one GX2's calls set but
// three: in WORD0 it puts CHROMA_KEY at 28 and LOD_USES_MINOR_AXIS at 25,
// each on top of another field, and in WORD2 ANISO_BIAS two bits higher;
// decaf's layout, which overlaps nowhere, is taken. The LODs are unsigned
// 4.6 fixed point (0 to just under 16) and the bias signed 1.5.6 (-32 to
// just under 32).
//
// The border colours are config registers, four floats a sampler:
// TD_PS_SAMPLER0_BORDER_RED at 0xA400, the vertex stage's at 0xA600 and the
// geometry stage's at 0xA800 -- Linux's R600 register list, and Cemu. decaf
// has the geometry stage's at 0xA900.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/render_state.hpp>

#include <array>
#include <bit>

namespace wiiu::latte {

namespace sampler {

/// Words in one sampler, and where the window starts.
inline constexpr u32 words = 3;
inline constexpr u32 window_base = 0x3C000;

/// Each stage's eighteen, counted in samplers.
inline constexpr u32 per_stage = 18;
inline constexpr u32 ps_base = 0;
inline constexpr u32 vs_base = 18;
inline constexpr u32 gs_base = 36;

/// The byte address of sampler `id`'s first word.
[[nodiscard]] constexpr u32 address(u32 id) noexcept {
    return window_base + (id * words * 4);
}

/// Each stage's border colours: red, green, blue and alpha, sixteen bytes a
/// sampler.
inline constexpr u32 ps_border_base = 0xA400;
inline constexpr u32 vs_border_base = 0xA600;
inline constexpr u32 gs_border_base = 0xA800;
inline constexpr u32 border_stride = 16;

} // namespace sampler

/// SQ_TEX_CLAMP: how a coordinate outside [0, 1] is brought back.
enum class TexClamp : u32 {
    Wrap = 0,
    Mirror = 1,
    ClampLastTexel = 2,
    MirrorOnceLastTexel = 3,
    ClampHalfBorder = 4,
    MirrorOnceHalfBorder = 5,
    ClampBorder = 6,
    MirrorOnceBorder = 7,
};

/// SQ_TEX_XY_FILTER: within a level.
enum class TexXYFilter : u32 {
    Point = 0,
    Bilinear = 1,
    Bicubic = 2,
    AnisoPoint = 4,
    AnisoBilinear = 5,
};

/// SQ_TEX_Z_FILTER: between slices of a volume, and between levels.
enum class TexZFilter : u32 {
    None = 0, ///< for levels: the base level alone
    Point = 1,
    Linear = 2,
};

/// SQ_TEX_BORDER_COLOR: what a border clamp reads.
enum class TexBorderColor : u32 {
    TransparentBlack = 0,
    OpaqueBlack = 1,
    OpaqueWhite = 2,
    Register = 3, ///< the stage's TD_*_SAMPLERn_BORDER_* registers
};

union TexSamplerWord0 {
    u32 raw;
    BitField<0, 3, TexClamp> clamp_x;
    BitField<3, 3, TexClamp> clamp_y;
    BitField<6, 3, TexClamp> clamp_z;
    BitField<9, 3, TexXYFilter> xy_mag_filter;
    BitField<12, 3, TexXYFilter> xy_min_filter;
    BitField<15, 2, TexZFilter> z_filter;
    BitField<17, 2, TexZFilter> mip_filter;
    BitField<19, 3, u32> max_aniso_ratio; ///< log2 of the ratio
    BitField<22, 2, TexBorderColor> border_color_type;
    BitField<24, 1, bool> point_sampling_clamp;
    BitField<25, 1, bool> tex_array_override;
    BitField<26, 3, CompareFunction> depth_compare_function;
    BitField<29, 2, u32> chroma_key;
    BitField<31, 1, bool> lod_uses_minor_axis;
};

union TexSamplerWord1 {
    u32 raw;
    BitField<0, 10, u32> min_lod;   ///< unsigned 4.6
    BitField<10, 10, u32> max_lod;  ///< unsigned 4.6
    BitField<20, 12, u32> lod_bias; ///< signed 1.5.6
};

union TexSamplerWord2 {
    u32 raw;
    BitField<0, 12, u32> lod_bias_sec;
    BitField<12, 1, bool> mc_coord_truncate;
    BitField<13, 1, bool> force_degamma;
    BitField<14, 1, bool> high_precision_filter;
    BitField<15, 3, u32> perf_mip;
    BitField<18, 2, u32> perf_z;
    BitField<20, 6, u32> aniso_bias; ///< unsigned 1.5
    BitField<26, 1, bool> fetch_4;
    BitField<27, 1, bool> sample_is_pcf;
    BitField<28, 1, bool> truncate_coord;
    BitField<29, 1, bool> disable_cube_wrap;
    BitField<31, 1, bool> type; ///< GX2 always sets it
};

/// Fixed-point fractions: six bits for the LODs and the bias, five for
/// the anisotropy bias.
inline constexpr f32 lod_scale = 64.0F;
inline constexpr f32 aniso_bias_scale = 32.0F;

/// A sampler, read back out of the register file.
struct TexSampler {
    TexSamplerWord0 word0{};
    TexSamplerWord1 word1{};
    TexSamplerWord2 word2{};

    [[nodiscard]] f32 min_lod() const noexcept {
        return static_cast<f32>(word1.min_lod.value()) / lod_scale;
    }

    [[nodiscard]] f32 max_lod() const noexcept {
        return static_cast<f32>(word1.max_lod.value()) / lod_scale;
    }

    /// The bias, its twelve bits sign-extended.
    [[nodiscard]] f32 lod_bias() const noexcept {
        const auto bits = static_cast<s32>(word1.lod_bias.value() << 20) >> 20;
        return static_cast<f32>(bits) / lod_scale;
    }
};

[[nodiscard]] inline TexSampler texture_sampler(const RegisterFile& registers, u32 id) noexcept {
    const u32 base = sampler::address(id);
    TexSampler out;
    out.word0.raw = registers.read(base);
    out.word1.raw = registers.read(base + 4);
    out.word2.raw = registers.read(base + 8);
    return out;
}

/// The border colour a stage's sampler `unit` reads, from the stage's
/// `border_base`.
[[nodiscard]] inline std::array<f32, 4>
border_color(const RegisterFile& registers, u32 border_base, u32 unit) noexcept {
    const u32 base = border_base + (unit * sampler::border_stride);
    std::array<f32, 4> out{};
    for (u32 i = 0; i < out.size(); ++i) {
        out[i] = std::bit_cast<f32>(registers.read(base + (i * 4)));
    }
    return out;
}

} // namespace wiiu::latte
