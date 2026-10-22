// Phase 8.11 — see index_cache.hpp.

#include <wiiu/latte/draw.hpp>
#include <wiiu/latte/index_cache.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <bit>
#include <utility>

namespace wiiu::latte {

namespace {

// The converted bytes are handed to Metal as they are, and Metal reads them
// in the host's order.
static_assert(std::endian::native == std::endian::little);

/// Bytes an index of this type takes, or zero for a size Latte has not.
[[nodiscard]] u32 index_bytes(u32 index_type) noexcept {
    switch (DmaIndexType{.raw = index_type}.index_size.value()) {
        case IndexSize::Bits16:
            return 2;
        case IndexSize::Bits32:
            return 4;
    }
    return 0;
}

/// FNV-1a, as the recorder hashes memory (hle/recording.hpp): enough to see
/// that a title rewrote its indices, which is all it is asked.
[[nodiscard]] u64 hash_bytes(std::span<const u8> bytes) noexcept {
    u64 hash = 0xCBF29CE484222325ULL;
    for (const u8 byte : bytes) {
        hash ^= byte;
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

} // namespace

std::vector<u8> convert_indices(std::span<const u8> stored, u32 index_type) {
    if (index_bytes(index_type) == 0) {
        return {};
    }
    // The fetch swaps whole words, so the bytes are padded to them and the
    // padding dropped after.
    std::vector<u8> bytes(stored.begin(), stored.end());
    bytes.resize((stored.size() + 3) & ~std::size_t{3}, 0);
    switch (DmaIndexType{.raw = index_type}.swap_mode.value()) {
        case IndexSwap::None:
            break;
        case IndexSwap::Swap16:
            for (std::size_t i = 0; i < bytes.size(); i += 2) {
                std::swap(bytes[i], bytes[i + 1]);
            }
            break;
        case IndexSwap::Swap32:
            for (std::size_t i = 0; i < bytes.size(); i += 4) {
                std::swap(bytes[i], bytes[i + 3]);
                std::swap(bytes[i + 1], bytes[i + 2]);
            }
            break;
        case IndexSwap::SwapWord:
            for (std::size_t i = 0; i < bytes.size(); i += 4) {
                std::swap(bytes[i], bytes[i + 2]);
                std::swap(bytes[i + 1], bytes[i + 3]);
            }
            break;
    }
    bytes.resize(stored.size());
    return bytes;
}

std::optional<IndexBuffer>
IndexCache::indices(ppcaddr_t address, u32 count, u32 index_type, std::string& error) {
    const u32 size = index_bytes(index_type);
    if (size == 0) {
        error = fmt::format("index type {:#x} has no index size Latte knows", index_type);
        return std::nullopt;
    }
    const u64 bytes = u64{count} * size;
    if (count == 0 || bytes > max_bytes) {
        error = fmt::format(
            "{} indices is not a draw's worth (1 to {} bytes of them)", count, max_bytes);
        return std::nullopt;
    }
    const u8* const host =
        memory_->translate(address, static_cast<u32>(bytes), memory::Access::Read);
    if (host == nullptr) {
        error = fmt::format("{} indices at {:#010x} are not readable guest memory", count, address);
        return std::nullopt;
    }
    const std::span<const u8> stored{host, static_cast<std::size_t>(bytes)};
    const u64 hash = hash_bytes(stored);

    const Key key{.address = address, .count = count, .index_type = index_type};
    auto found = entries_.find(key);
    if (found != entries_.end() && found->second.hash == hash) {
        ++stats_.hits;
        return found->second.indices;
    }

    std::vector<u8> converted = convert_indices(stored, index_type);
    gpu::BufferDesc desc;
    desc.size = (converted.size() + 3) & ~u64{3};
    desc.storage = gpu::Storage::Shared;
    desc.label = "indices";
    const gpu::BufferHandle buffer = backend_->create_buffer(desc);
    converted.resize(desc.size, 0);
    if (!buffer.valid() || !backend_->write_buffer(buffer, 0, converted)) {
        if (buffer.valid()) {
            backend_->destroy(buffer);
        }
        error = fmt::format("the backend made no {}-byte buffer for indices", desc.size);
        return std::nullopt;
    }

    const IndexBuffer indices{.buffer = buffer, .is_32bit = size == 4};
    if (found != entries_.end()) {
        // Rewritten: the old copy goes, once the frames using it are done.
        backend_->destroy(found->second.indices.buffer);
        found->second = Entry{.hash = hash, .indices = indices};
    } else {
        if (entries_.size() >= max_entries) {
            clear();
        }
        entries_.emplace(key, Entry{.hash = hash, .indices = indices});
    }
    ++stats_.conversions;
    stats_.converted_bytes += bytes;
    return indices;
}

void IndexCache::clear() {
    for (const auto& [key, entry] : entries_) {
        backend_->destroy(entry.indices.buffer);
    }
    entries_.clear();
}

} // namespace wiiu::latte
