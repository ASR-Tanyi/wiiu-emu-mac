// Phase 8.11 — see sampler_cache.hpp.

#include <wiiu/latte/sampler_cache.hpp>

#include <bit>

namespace wiiu::latte {

namespace {

/// The LOD range a mip filter of none clamps to: level 0 alone (Cemu).
constexpr f32 base_level_only = 0.25F;

[[nodiscard]] gpu::Filter filter(TexXYFilter latte) noexcept {
    return latte == TexXYFilter::Point || latte == TexXYFilter::AnisoPoint ? gpu::Filter::Nearest
                                                                           : gpu::Filter::Linear;
}

[[nodiscard]] gpu::AddressMode address(TexClamp latte) noexcept {
    switch (latte) {
        case TexClamp::Wrap:
            return gpu::AddressMode::Repeat;
        case TexClamp::Mirror:
            return gpu::AddressMode::MirrorRepeat;
        case TexClamp::ClampLastTexel:
        case TexClamp::ClampHalfBorder:
            return gpu::AddressMode::ClampToEdge;
        case TexClamp::MirrorOnceLastTexel:
            return gpu::AddressMode::MirrorClampToEdge;
        case TexClamp::MirrorOnceHalfBorder:
        case TexClamp::ClampBorder:
        case TexClamp::MirrorOnceBorder:
            return gpu::AddressMode::ClampToBorder;
    }
    return gpu::AddressMode::Repeat;
}

[[nodiscard]] gpu::ClearColor border_of(TexBorderColor type,
                                        const std::array<f32, 4>& registers) noexcept {
    switch (type) {
        case TexBorderColor::TransparentBlack:
            return {0.0F, 0.0F, 0.0F, 0.0F};
        case TexBorderColor::OpaqueBlack:
            return {0.0F, 0.0F, 0.0F, 1.0F};
        case TexBorderColor::OpaqueWhite:
            return {1.0F, 1.0F, 1.0F, 1.0F};
        case TexBorderColor::Register:
            return {registers[0], registers[1], registers[2], registers[3]};
    }
    return {};
}

} // namespace

gpu::SamplerDesc sampler_desc(const TexSampler& sampler,
                              const std::array<f32, 4>& border) noexcept {
    const TexSamplerWord0 word0 = sampler.word0;
    gpu::SamplerDesc out;
    out.label = "sampler";
    out.min_filter = filter(word0.xy_min_filter);
    out.mag_filter = filter(word0.xy_mag_filter);
    switch (word0.mip_filter.value()) {
        case TexZFilter::None:
            out.mip_filter = gpu::Filter::Nearest;
            out.lod_min = 0.0F;
            out.lod_max = base_level_only;
            break;
        case TexZFilter::Point:
            out.mip_filter = gpu::Filter::Nearest;
            out.lod_min = sampler.min_lod();
            out.lod_max = sampler.max_lod();
            break;
        default: // linear, and the fourth value, as Cemu falls back
            out.mip_filter = gpu::Filter::Linear;
            out.lod_min = sampler.min_lod();
            out.lod_max = sampler.max_lod();
            break;
    }
    out.address_u = address(word0.clamp_x);
    out.address_v = address(word0.clamp_y);
    out.address_w = address(word0.clamp_z);
    out.max_anisotropy = 1U << word0.max_aniso_ratio.value();
    out.border_color = border_of(word0.border_color_type, border);
    return out;
}

gpu::SamplerHandle SamplerCache::sampler(const TexSampler& sampler,
                                         const std::array<f32, 4>& border) {
    Key key{.words = {sampler.word0.raw, sampler.word1.raw, sampler.word2.raw}, .border = {}};
    if (sampler.word0.border_color_type == TexBorderColor::Register) {
        for (std::size_t i = 0; i < border.size(); ++i) {
            key.border[i] = std::bit_cast<u32>(border[i]);
        }
    }
    if (const auto found = samplers_.find(key); found != samplers_.end()) {
        return found->second;
    }
    const gpu::SamplerHandle handle = backend_->create_sampler(sampler_desc(sampler, border));
    if (handle.valid()) {
        samplers_.emplace(key, handle);
    }
    return handle;
}

void SamplerCache::clear() {
    for (const auto& [key, handle] : samplers_) {
        backend_->destroy(handle);
    }
    samplers_.clear();
}

} // namespace wiiu::latte
