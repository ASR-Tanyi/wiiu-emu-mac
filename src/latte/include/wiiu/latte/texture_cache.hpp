// Phase 8.13, textures' first form — a texture resource's image, as a Metal
// texture.
//
// A shader samples whatever a texture resource (resources.hpp) describes:
// an image in guest memory, laid out as Latte lays it out (surface.hpp) and
// in a Latte format. Metal samples neither layout, so the image is detiled
// on the CPU into plain rows and written to a texture of the backend's own,
// in the Metal format that reads the same bytes.
//
// **What the resource says.** `texture_image` reads it as Latte's layout
// takes it: the dimension, tile mode and TILE_TYPE (a depth layout), the
// size, and the format, which is the data format with the number type --
// WORD4's NUM_FORMAT_ALL, FORMAT_COMP and FORCE_DEGAMMA -- deciding which
// Metal format reads it. A macro-tiled level's base register carries the
// bank and pipe swizzle in its low bits (GX2 XORs it in, gx2_texture.hpp):
// the level is at the base with those bits clear, and its swizzle is those
// bits, as a GX2 swizzle word's second byte. The pitch the layout works out
// must be the resource's own; if it is not, the resource has been misread,
// and it is refused rather than drawn sheared.
//
// **Once.** The texture is kept, keyed by where the image is and the words
// that say how it is laid out and read, and found again for as long as its
// bytes are the same -- a hash of them at each lookup, as the index cache
// does (index_cache.hpp), until memory-range tracking (P5) can say instead
// which pages a title wrote. Changed bytes make a new texture, the old one
// destroyed once the frames using it are done; a texture written by the CPU
// while the GPU reads it would tear.
//
// Not yet, and refused by name: any dimension but 2D, multisampled
// textures, levels past 0, and formats other than 8, 8_8 and 8_8_8_8 read
// as unsigned normalised (and 8_8_8_8 as sRGB) -- the rest want their byte
// order checked against something real, as the clear's did. Nor yet a
// texture the GPU drew (ADR-006), which lives in the surface cache, not in
// guest memory. The component selects (DST_SEL) are the shader's to apply.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/surface.hpp>
#include <wiiu/memory/memory.hpp>

#include <compare>
#include <map>
#include <optional>
#include <string>

namespace wiiu::latte {

/// A texture resource's level 0, as the layout and Metal take it.
struct TextureImage {
    SurfaceDesc desc;
    SurfaceLayout layout;
    physaddr_t address = 0; ///< where the level starts, the swizzle bits clear
    BankPipeSwizzle swizzle;
    gpu::TextureFormat format = gpu::TextureFormat::Unknown;
    u32 width = 0;
    u32 height = 0;
    u32 bytes_per_element = 0;
};

/// What a texture resource describes. Nothing, with `error` saying what,
/// for one that is not valid or not uploaded yet.
[[nodiscard]] std::optional<TextureImage> texture_image(const TextureResource& resource,
                                                        std::string& error);

/// A texture made from guest memory.
struct CachedTexture {
    gpu::TextureHandle texture;
    gpu::TextureFormat format = gpu::TextureFormat::Unknown;
    u32 width = 0;
    u32 height = 0;
};

struct TextureCacheStats {
    u64 hits = 0;
    u64 uploads = 0;
    u64 uploaded_bytes = 0; ///< of guest memory read
};

class TextureCache {
public:
    /// The most textures kept before the cache starts again.
    static constexpr std::size_t max_entries = 1024;

    TextureCache(gpu::GpuBackend& backend, const memory::Memory& memory)
        : backend_(&backend), memory_(&memory) {}

    ~TextureCache() { clear(); }

    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;

    /// The texture `resource` describes, uploaded the first time or when its
    /// bytes have changed. Nothing, with `error` saying why, for one it
    /// cannot read or upload.
    [[nodiscard]] std::optional<CachedTexture> texture(const TextureResource& resource,
                                                       std::string& error);

    /// Every texture destroyed.
    void clear();

    [[nodiscard]] const TextureCacheStats& stats() const noexcept { return stats_; }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Key {
        physaddr_t base = 0; ///< swizzle bits and all
        u32 word0 = 0;
        u32 word1 = 0;
        u32 number = 0; ///< WORD4's format bits: comps, number format, degamma, swap

        auto operator<=>(const Key&) const = default;
    };

    struct Entry {
        u64 hash = 0;
        CachedTexture texture;
    };

    gpu::GpuBackend* backend_;
    const memory::Memory* memory_;
    std::map<Key, Entry> entries_;
    TextureCacheStats stats_;
};

} // namespace wiiu::latte
