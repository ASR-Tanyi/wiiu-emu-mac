// Phase 8.10, first form — the pipelines draws are made with, built once.
//
// A Metal pipeline is the two shaders compiled and linked for one colour
// format, one class of primitive and one way of blending, and compiling is
// the slow part of a draw by a long way: a title draws with the same few
// shaders thousands of times a frame. So a pipeline is built the first time a combination is
// drawn with and found after that -- and so is a failure: a shader the
// compiler refuses is refused once, with its error kept, rather than
// compiled again every draw (P5's counters say how often each happens).
//
// The key is the shaders' MSL itself, which is exactly what decides what
// gets compiled; 8.10 proper will key by a hash of the programs and their
// registers, and keep pipelines on disk, so a title's second run compiles
// nothing.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>

#include <compare>
#include <map>
#include <string>

namespace wiiu::latte {

struct PipelineKey {
    std::string vertex_msl;
    std::string pixel_msl;
    gpu::TextureFormat color_format = gpu::TextureFormat::Unknown;
    gpu::PrimitiveType primitive = gpu::PrimitiveType::Triangles;
    gpu::BlendState blend; ///< render target 0's

    auto operator<=>(const PipelineKey&) const = default;
};

struct PipelineCacheStats {
    u64 hits = 0;
    u64 builds = 0;
    u64 failures = 0; ///< builds the backend refused, each counted once
};

class PipelineCache {
public:
    explicit PipelineCache(gpu::GpuBackend& backend) : backend_(&backend) {}

    ~PipelineCache() { clear(); }

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    /// The pipeline for `key`, built the first time. An invalid handle, with
    /// `error` saying why -- the compiler's words, the first time and every
    /// time after -- for one that does not build.
    [[nodiscard]] gpu::PipelineHandle pipeline(const PipelineKey& key, std::string& error);

    /// Every pipeline destroyed, every failure forgotten.
    void clear();

    [[nodiscard]] const PipelineCacheStats& stats() const noexcept { return stats_; }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Entry {
        gpu::PipelineHandle pipeline;
        std::string error; ///< why it did not build, if it did not
    };

    gpu::GpuBackend* backend_;
    std::map<PipelineKey, Entry> entries_;
    PipelineCacheStats stats_;
};

} // namespace wiiu::latte
