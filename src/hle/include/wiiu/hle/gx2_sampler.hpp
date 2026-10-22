// Phase 8.1 — GX2's samplers: how a texture is addressed and filtered,
// packed into the three words SET_SAMPLER writes (latte/samplers.hpp).
//
// A `GX2Sampler` is just those three words. The title builds them with
// `GX2InitSampler` -- one clamp mode and one filter for everything -- and
// adjusts them with the `GX2InitSampler…` calls, each of which changes its
// own fields and leaves the rest; `GX2Set{Pixel,Vertex,Geometry}Sampler`
// writes them to one of the stage's eighteen samplers, and
// `GX2Set…SamplerBorderColor` sets the four floats a border of type
// REGISTER reads.
//
// decaf and Cemu agree on most of this. Where they differ, this is Cemu's,
// which reads as gx2.rpl's own output rather than a reading of the fields:
//
//   * `GX2InitSampler` also sets the Z and mip filters to point and
//     TEX_ARRAY_OVERRIDE (decaf leaves all three zero -- no mip filtering);
//   * `GX2InitSamplerXYFilter` with an anisotropy ratio turns point and
//     bilinear into their anisotropic forms (decaf keeps the filters as
//     given);
//   * `GX2InitSamplerLOD` floors each value to its fixed point and clamps
//     the result to the field (decaf converts by truncation, so a negative
//     bias rounds the other way, and its clamps let a bias of 32 wrap).
//     Cemu's own min-only and max-only float clamps are replaced by clamping
//     to the field, which is the same for every value but a negative maximum
//     or a NaN, where Cemu's conversion is undefined; those read zero.
//
// `GX2InitSamplerFilterAdjust`, `…LODAdjust` and `…RoundingMode` are
// decaf's alone (Cemu does not export them), on decaf's fields; the
// anisotropy bias is held below its six bits' wrap. `GX2SetComputeSampler`
// is the vertex stage's, as Cemu has it.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>

#include <vector>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// `GX2Sampler`, as it sits in guest memory.
struct Sampler {
    u32be word0;
    u32be word1;
    u32be word2;
};

static_assert(sizeof(Sampler) == 12);

/// The SET_SAMPLER `GX2Set…Sampler` writes to sampler `id` of the window.
void append_sampler(std::vector<u32>& out, u32 id, const Sampler& sampler);

/// Register the sampler calls. Called by `register_gx2`.
void register_gx2_sampler(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
