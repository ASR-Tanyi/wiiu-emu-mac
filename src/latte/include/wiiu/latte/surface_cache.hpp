// Phase 8.13, first form (ADR-006) — the surface cache: the Metal texture
// a colour buffer lives in once the GPU renders to it.
//
// A colour buffer is found by what the colour-buffer registers say
// identifies it -- `ColorBufferKey`: CB_COLORn_BASE (the address >> 8, with
// the swizzle), CB_COLORn_SIZE (pitch and slice in 8x8 tiles), and from
// CB_COLORn_INFO the format, number type, tile mode and component swap,
// with CB_COLORn_VIEW's first slice. A draw reads those from the register
// file; GX2's clear and scan-out copy carry the same values in the
// `GX2ColorBuffer` they snapshot (GX2InitColorBufferRegs, 8.1), so all three
// find one entry. The texture is as large as the registers describe -- the
// pitch by the padded height -- and each user takes its own rectangle.
//
// Depth buffers are cached the same way, by their own registers
// (`DepthBufferKey`: DB_DEPTH_BASE, DB_DEPTH_SIZE, DB_DEPTH_INFO's format
// and tile mode, DB_DEPTH_VIEW's first slice), so GX2's depth clear (from
// the `GX2DepthBuffer` it snapshots) and a draw (from the register file) find
// one entry. Apple's GPUs have no 24-bit depth, so Latte's 24-bit depths,
// fixed or float, are 32-bit float on Metal -- with 8 bits of stencil when
// the format has a stencil -- and its 16-bit depth is 32-bit float too, for
// now; a depth test compares the same way at either precision, near enough.
//
// The first form keeps to what the first frames need: colour buffers in the
// formats WHB's samples render to -- 8_8_8_8 UNORM and sRGB, with the
// standard component order -- and every depth format, everything else
// refused by name; one slice; no coherency with guest memory yet (ADR-006:
// CPU writes under a cached surface, and readback, come with 8.13 proper).
// It counts hits, misses and refusals from the start (P5).
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/latte/color_buffer.hpp>
#include <wiiu/latte/depth_buffer.hpp>
#include <wiiu/latte/registers.hpp>

#include <compare>
#include <map>
#include <optional>
#include <string>

namespace wiiu::latte {

/// What identifies a colour buffer.
struct ColorBufferKey {
    u32 base = 0; ///< CB_COLORn_BASE: the address >> 8, the swizzle in its low bits
    u32 size = 0; ///< CB_COLORn_SIZE
    u32 format = 0;
    u32 number_type = 0;
    u32 array_mode = 0; ///< the tile mode
    u32 comp_swap = 0;
    u32 slice = 0; ///< CB_COLORn_VIEW's SLICE_START

    auto operator<=>(const ColorBufferKey&) const = default;

    /// From the four registers' values.
    [[nodiscard]] static ColorBufferKey from(u32 base, u32 size, u32 info, u32 view) noexcept;

    /// Render target `target`'s, as the register file stands.
    [[nodiscard]] static ColorBufferKey from(const RegisterFile& registers, u32 target) noexcept;

    /// The texture's size: the pitch, and the slice over it, in pixels.
    [[nodiscard]] u32 width() const noexcept;
    [[nodiscard]] u32 height() const noexcept;
};

/// Metal's format for a colour buffer's, or nothing for one the first form
/// does not render to yet.
[[nodiscard]] std::optional<gpu::TextureFormat> color_texture_format(const ColorBufferKey& key);

/// What identifies a depth buffer.
struct DepthBufferKey {
    u32 base = 0;       ///< DB_DEPTH_BASE: the address >> 8, the swizzle in its low bits
    u32 size = 0;       ///< DB_DEPTH_SIZE
    u32 format = 0;     ///< DB_DEPTH_INFO's `DbFormat`
    u32 array_mode = 0; ///< the tile mode
    u32 slice = 0;      ///< DB_DEPTH_VIEW's SLICE_START

    auto operator<=>(const DepthBufferKey&) const = default;

    /// From the four registers' values.
    [[nodiscard]] static DepthBufferKey from(u32 base, u32 size, u32 info, u32 view) noexcept;

    /// The depth buffer, as the register file stands.
    [[nodiscard]] static DepthBufferKey from(const RegisterFile& registers) noexcept;

    [[nodiscard]] u32 width() const noexcept;
    [[nodiscard]] u32 height() const noexcept;

    /// Whether the format has a stencil beside its depth.
    [[nodiscard]] bool has_stencil() const noexcept;
};

/// Metal's format for a depth buffer's, or nothing for one that is not a
/// depth format.
[[nodiscard]] std::optional<gpu::TextureFormat> depth_texture_format(const DepthBufferKey& key);

/// A cached colour buffer.
struct ColorSurface {
    gpu::TextureHandle texture;
    gpu::TextureFormat format = gpu::TextureFormat::Unknown;
    u32 width = 0;
    u32 height = 0;
};

/// A cached depth buffer.
struct DepthSurface {
    gpu::TextureHandle texture;
    gpu::TextureFormat format = gpu::TextureFormat::Unknown;
    u32 width = 0;
    u32 height = 0;
    bool has_stencil = false;
};

struct SurfaceCacheStats {
    u64 hits = 0;
    u64 misses = 0;  ///< made on first use
    u64 refused = 0; ///< a format, or a size, there is no texture for yet
};

class SurfaceCache {
public:
    explicit SurfaceCache(gpu::GpuBackend& backend) : backend_(&backend) {}

    SurfaceCache(const SurfaceCache&) = delete;
    SurfaceCache& operator=(const SurfaceCache&) = delete;

    ~SurfaceCache() { clear(); }

    /// The colour buffer's texture, made on first use. Nothing, with `error`
    /// saying why, for one there is no texture for yet.
    [[nodiscard]] std::optional<ColorSurface> color_buffer(const ColorBufferKey& key,
                                                           std::string& error);

    /// The entry for `key` if there is one, without making it: whether the
    /// GPU holds this surface, which decides whether a clear or a copy runs
    /// on the GPU or on guest memory (ADR-006).
    [[nodiscard]] std::optional<ColorSurface> find(const ColorBufferKey& key) const;

    /// A depth buffer's texture, made on first use; nothing, with `error`,
    /// for one there is no texture for.
    [[nodiscard]] std::optional<DepthSurface> depth_buffer(const DepthBufferKey& key,
                                                           std::string& error);

    [[nodiscard]] std::optional<DepthSurface> find(const DepthBufferKey& key) const;

    /// Every texture destroyed, as at a GPU reset.
    void clear();

    [[nodiscard]] const SurfaceCacheStats& stats() const noexcept { return stats_; }

    /// Surfaces held: colour and depth.
    [[nodiscard]] std::size_t size() const noexcept { return colors_.size() + depths_.size(); }

private:
    gpu::GpuBackend* backend_;
    std::map<ColorBufferKey, ColorSurface> colors_;
    std::map<DepthBufferKey, DepthSurface> depths_;
    SurfaceCacheStats stats_;
};

} // namespace wiiu::latte
