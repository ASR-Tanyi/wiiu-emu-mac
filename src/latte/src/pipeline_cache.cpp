// Phase 8.10, first form — see pipeline_cache.hpp.

#include <wiiu/latte/msl.hpp>
#include <wiiu/latte/pipeline_cache.hpp>

namespace wiiu::latte {

gpu::PipelineHandle PipelineCache::pipeline(const PipelineKey& key, std::string& error) {
    if (const auto found = entries_.find(key); found != entries_.end()) {
        ++stats_.hits;
        if (!found->second.pipeline.valid()) {
            error = found->second.error;
        }
        return found->second.pipeline;
    }

    gpu::PipelineDesc desc;
    desc.label = "draw";
    desc.vertex_source = key.vertex_msl;
    desc.fragment_source = key.pixel_msl;
    desc.vertex_entry = std::string{msl::vertex_entry};
    desc.fragment_entry = std::string{msl::fragment_entry};
    desc.primitive = key.primitive;
    desc.color_formats[0] = key.color_format;
    desc.blend[0] = key.blend;
    desc.color_count = 1;

    Entry entry;
    entry.pipeline = backend_->create_pipeline(desc, entry.error);
    if (entry.pipeline.valid()) {
        ++stats_.builds;
        entry.error.clear();
    } else {
        ++stats_.failures;
        error = entry.error;
    }
    const gpu::PipelineHandle pipeline = entry.pipeline;
    entries_.emplace(key, std::move(entry));
    return pipeline;
}

void PipelineCache::clear() {
    for (const auto& [key, entry] : entries_) {
        if (entry.pipeline.valid()) {
            backend_->destroy(entry.pipeline);
        }
    }
    entries_.clear();
}

} // namespace wiiu::latte
