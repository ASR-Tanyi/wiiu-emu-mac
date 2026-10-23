// Phase 8.11 — a Latte sampler as the backend's.
//
// What is guarded: GX2InitSampler's words as Metal's sampler -- filters,
// the mip filter and its LOD range (none holding to level 0, as Cemu), the
// clamp modes by Cemu's table, the anisotropy ratio, and the border colour
// by type or from the stage's registers; and the cache, which makes a
// sampler once and tells two apart by their words, and by the registers'
// border only when the sampler reads it.

#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/latte/sampler_cache.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

/// GX2InitSampler(sampler, GX2_TEX_CLAMP_MODE_CLAMP, GX2_TEX_XY_FILTER_MODE_LINEAR),
/// as its test has the words.
[[nodiscard]] TexSampler init_sampler() {
    TexSampler out;
    out.word0.raw = 0x02029292;
    out.word1.raw = 1023U << 10;
    out.word2.raw = 0x80000000;
    return out;
}

constexpr std::array<f32, 4> kNoBorder{};

} // namespace

TEST_CASE("GX2InitSampler's words are Metal's sampler", "[latte][sampler]") {
    const gpu::SamplerDesc desc = sampler_desc(init_sampler(), kNoBorder);
    CHECK(desc.min_filter == gpu::Filter::Linear);
    CHECK(desc.mag_filter == gpu::Filter::Linear);
    CHECK(desc.mip_filter == gpu::Filter::Nearest); // point
    CHECK(desc.lod_min == 0.0F);
    CHECK(desc.lod_max == 1023.0F / 64.0F);
    CHECK(desc.address_u == gpu::AddressMode::ClampToEdge);
    CHECK(desc.address_v == gpu::AddressMode::ClampToEdge);
    CHECK(desc.address_w == gpu::AddressMode::ClampToEdge);
    CHECK(desc.max_anisotropy == 1);
    CHECK(desc.border_color == gpu::ClearColor{0.0F, 0.0F, 0.0F, 0.0F});
    CHECK_FALSE(desc.compare_enabled);
}

TEST_CASE("each field of a sampler, as Metal takes it", "[latte][sampler]") {
    TexSampler sampler = init_sampler();

    // No mip filter: level 0 alone.
    sampler.word0.mip_filter = TexZFilter::None;
    CHECK(sampler_desc(sampler, kNoBorder).lod_max == 0.25F);
    sampler.word0.mip_filter = TexZFilter::Linear;
    sampler.word1.min_lod = 64;
    sampler.word1.max_lod = 256;
    gpu::SamplerDesc desc = sampler_desc(sampler, kNoBorder);
    CHECK(desc.mip_filter == gpu::Filter::Linear);
    CHECK(desc.lod_min == 1.0F);
    CHECK(desc.lod_max == 4.0F);

    // Anisotropic point and bilinear, 8:1.
    sampler.word0.xy_mag_filter = TexXYFilter::AnisoPoint;
    sampler.word0.xy_min_filter = TexXYFilter::AnisoBilinear;
    sampler.word0.max_aniso_ratio = 3;
    desc = sampler_desc(sampler, kNoBorder);
    CHECK(desc.mag_filter == gpu::Filter::Nearest);
    CHECK(desc.min_filter == gpu::Filter::Linear);
    CHECK(desc.max_anisotropy == 8);

    // Cemu's clamp table.
    const auto clamp = [&sampler](TexClamp mode) {
        sampler.word0.clamp_x = mode;
        return sampler_desc(sampler, kNoBorder).address_u;
    };
    CHECK(clamp(TexClamp::Wrap) == gpu::AddressMode::Repeat);
    CHECK(clamp(TexClamp::Mirror) == gpu::AddressMode::MirrorRepeat);
    CHECK(clamp(TexClamp::ClampLastTexel) == gpu::AddressMode::ClampToEdge);
    CHECK(clamp(TexClamp::MirrorOnceLastTexel) == gpu::AddressMode::MirrorClampToEdge);
    CHECK(clamp(TexClamp::ClampHalfBorder) == gpu::AddressMode::ClampToEdge);
    CHECK(clamp(TexClamp::MirrorOnceHalfBorder) == gpu::AddressMode::ClampToBorder);
    CHECK(clamp(TexClamp::ClampBorder) == gpu::AddressMode::ClampToBorder);
    CHECK(clamp(TexClamp::MirrorOnceBorder) == gpu::AddressMode::ClampToBorder);

    // Border colours: the three types, and the registers'.
    const std::array<f32, 4> registers{0.25F, 0.5F, 0.75F, 1.0F};
    const auto border = [&](TexBorderColor type) {
        sampler.word0.border_color_type = type;
        return sampler_desc(sampler, registers).border_color;
    };
    CHECK(border(TexBorderColor::TransparentBlack) == gpu::ClearColor{0.0F, 0.0F, 0.0F, 0.0F});
    CHECK(border(TexBorderColor::OpaqueBlack) == gpu::ClearColor{0.0F, 0.0F, 0.0F, 1.0F});
    CHECK(border(TexBorderColor::OpaqueWhite) == gpu::ClearColor{1.0F, 1.0F, 1.0F, 1.0F});
    CHECK(border(TexBorderColor::Register) == gpu::ClearColor{0.25F, 0.5F, 0.75F, 1.0F});
}

TEST_CASE("a sampler is made once, and told apart by what it reads", "[latte][sampler]") {
    gpu::NullBackend backend;
    std::string error;
    REQUIRE(backend.initialize(error));
    SamplerCache cache(backend);

    const TexSampler sampler = init_sampler();
    const gpu::SamplerHandle first = cache.sampler(sampler, kNoBorder);
    REQUIRE(first.valid());
    // The registers' border does not matter to a sampler that does not read it.
    CHECK(cache.sampler(sampler, {1.0F, 0.0F, 0.0F, 1.0F}) == first);
    CHECK(cache.size() == 1);

    TexSampler wrapped = sampler;
    wrapped.word0.clamp_y = TexClamp::Wrap;
    CHECK(cache.sampler(wrapped, kNoBorder) != first);

    // To one that does, it does.
    TexSampler bordered = sampler;
    bordered.word0.border_color_type = TexBorderColor::Register;
    const gpu::SamplerHandle red = cache.sampler(bordered, {1.0F, 0.0F, 0.0F, 1.0F});
    CHECK(cache.sampler(bordered, {0.0F, 1.0F, 0.0F, 1.0F}) != red);
    CHECK(cache.sampler(bordered, {1.0F, 0.0F, 0.0F, 1.0F}) == red);
    CHECK(cache.size() == 4);
    CHECK(backend.live_count(gpu::ResourceKind::Sampler) == 4);

    cache.clear();
    CHECK(cache.size() == 0);
    CHECK(backend.live_count(gpu::ResourceKind::Sampler) == 0);
}
