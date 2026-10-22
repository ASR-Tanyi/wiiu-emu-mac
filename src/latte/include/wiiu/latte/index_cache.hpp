// Phase 8.11 — an indexed draw's indices, as Metal can read them.
//
// Latte fetches indices from guest memory with a byte swap the draw's
// VGT_DMA_INDEX_TYPE names (draw.hpp): GX2's own index types are 16- and
// 32-bit, little-endian as stored or big-endian -- the title's byte order,
// and what nearly every title uses -- by a swap the size of an index.
// Metal's index fetch reads the host's order and swaps nothing, so the
// indices are converted on the CPU, the swap applied as the fetch would
// apply it, into a buffer of the GPU's own.
//
// Once: the converted copy is kept, keyed by where the indices are, how many
// and how they are read, and found again for as long as the bytes are the
// same. A hash of them says so at every draw -- cheap beside a conversion,
// and the only way to see a title rewrite its indices, which nothing
// announces. Changed bytes are converted again and the old copy destroyed;
// Metal's command buffers keep what they reference alive, so a frame still
// in flight finishes with the old one. Static geometry, which is most of it,
// converts once.
//
// The cache is bounded crudely for now: past `max_entries` it is emptied.
// Not done yet: a primitive restart index other than all ones, which Metal
// assumes for strips (GX2SetPrimitiveRestartIndex can set another).
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/memory/memory.hpp>

#include <compare>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wiiu::latte {

/// Indices converted for Metal: in `buffer`, from its start.
struct IndexBuffer {
    gpu::BufferHandle buffer;
    bool is_32bit = false;
};

struct IndexCacheStats {
    u64 hits = 0;
    u64 conversions = 0;
    u64 converted_bytes = 0;
};

/// `count` indices of `index_type` (VGT_DMA_INDEX_TYPE) in host order, from
/// `stored` as guest memory holds them: the fetch's swap applied to the
/// bytes -- within each 16 bits, each 32, or the two halves of each 32 --
/// and the result read little-endian. Empty for a size Latte does not have.
[[nodiscard]] std::vector<u8> convert_indices(std::span<const u8> stored, u32 index_type);

class IndexCache {
public:
    /// The most converted copies kept before the cache starts again.
    static constexpr std::size_t max_entries = 4096;

    /// The most indices one draw is converted for: 16 MiB of them.
    static constexpr u64 max_bytes = 16ULL << 20;

    IndexCache(gpu::GpuBackend& backend, const memory::Memory& memory)
        : backend_(&backend), memory_(&memory) {}

    ~IndexCache() { clear(); }

    IndexCache(const IndexCache&) = delete;
    IndexCache& operator=(const IndexCache&) = delete;

    /// The draw's indices -- `count` of `index_type` at guest `address` --
    /// converted, the first time or when they have changed. Nothing, with
    /// `error` saying why, for indices it cannot read or convert.
    [[nodiscard]] std::optional<IndexBuffer>
    indices(ppcaddr_t address, u32 count, u32 index_type, std::string& error);

    /// Every converted copy destroyed.
    void clear();

    [[nodiscard]] const IndexCacheStats& stats() const noexcept { return stats_; }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Key {
        ppcaddr_t address = 0;
        u32 count = 0;
        u32 index_type = 0;

        auto operator<=>(const Key&) const = default;
    };

    struct Entry {
        u64 hash = 0;
        IndexBuffer indices;
    };

    gpu::GpuBackend* backend_;
    const memory::Memory* memory_;
    std::map<Key, Entry> entries_;
    IndexCacheStats stats_;
};

} // namespace wiiu::latte
