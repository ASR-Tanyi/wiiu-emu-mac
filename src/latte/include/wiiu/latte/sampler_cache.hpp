// Phase 8.11 — a Latte sampler (samplers.hpp) as the backend's: made once,
// found after.
//
// The translation is Cemu's Metal renderer's, field by field. Point and its
// anisotropic form are nearest, bilinear and its anisotropic form linear
// (and bicubic, which Metal has not, linear too); the anisotropy ratio is
// two to the field. A mip filter of none samples level 0 alone, as Cemu has
// it -- nearest, with the LOD held to [0, 0.25] -- and the others clamp the
// LOD to the sampler's own range, in its 4.6 fixed point. The clamp modes
// are Cemu's table: wrap, mirror, clamp to the edge, mirror once to the edge
// (Metal's mirror-clamp-to-edge); the half-border modes and mirroring once
// to the border, which Metal has not, become the nearest it has. The border
// colour is the type's -- transparent black, opaque black, opaque white --
// or the stage's border registers' for REGISTER, which the backend emulates
// where Metal's three do not reach.
//
// Not yet: the LOD bias, which Metal takes in the shader rather than the
// sampler (Cemu leaves it too), and depth comparison, which only SAMPLE_C
// reads -- not translated yet.
//
// A sampler is kept by its three words, and the border colour when it is
// the registers'; the backend's object for it is made the first time.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/latte/samplers.hpp>

#include <array>
#include <compare>
#include <cstddef>
#include <map>

namespace wiiu::latte {

/// A sampler as the backend takes it, with the border colour its stage's
/// registers hold for it (read only for border type REGISTER).
[[nodiscard]] gpu::SamplerDesc sampler_desc(const TexSampler& sampler,
                                            const std::array<f32, 4>& border) noexcept;

class SamplerCache {
public:
    explicit SamplerCache(gpu::GpuBackend& backend) : backend_(&backend) {}

    ~SamplerCache() { clear(); }

    SamplerCache(const SamplerCache&) = delete;
    SamplerCache& operator=(const SamplerCache&) = delete;

    /// The backend's sampler for `sampler`, made the first time. Invalid if
    /// the backend made none.
    [[nodiscard]] gpu::SamplerHandle sampler(const TexSampler& sampler,
                                             const std::array<f32, 4>& border);

    /// Every sampler destroyed.
    void clear();

    [[nodiscard]] std::size_t size() const noexcept { return samplers_.size(); }

private:
    struct Key {
        std::array<u32, 3> words{};
        std::array<u32, 4> border{}; ///< bits; zero unless the border is the registers'

        auto operator<=>(const Key&) const = default;
    };

    gpu::GpuBackend* backend_;
    std::map<Key, gpu::SamplerHandle> samplers_;
};

} // namespace wiiu::latte
