// Phase 8.13 (ADR-006) — copying a colour surface on the GPU, which is what
// GX2's scan-out copy becomes for a surface the GPU holds.
//
// The scan-out copy hands the display bytes, and the display shows them as
// they are. So this copies bytes, not colours, and does exactly what the CPU
// copy (`gx2::copy_to_scan_frame`) does to them:
//
//   * the target is RGBA8 UNORM, whatever the source: the bytes a scan
//     buffer holds, which is how the presenter shows a texture too
//     (`gpu::ScreenTexture`);
//   * an sRGB source is read decoded -- Metal reads sRGB no other way -- and
//     encoded again in the shader before it is written, which brings back
//     the bytes it held: the standard curve's encode is its decode's inverse,
//     with half a step to spare at every one of the 256 values;
//   * a picture of another size is scaled by nearest pixel with the CPU
//     copy's arithmetic, in integers: target pixel (x, y) takes source pixel
//     (x * sw / tw, y * sh / th);
//   * the source is the top-left `width` x `height` of its texture, which is
//     the padded size the colour buffer's registers describe -- an 854-pixel
//     GamePad picture in a texture 864 wide (ADR-006: each user takes its own
//     rectangle of a surface).
//
// The sizes reach the shader with `bind_bytes`, so two copies in one command
// buffer -- the TV's and the GamePad's -- each read their own. Each copy is a
// render pass of its own that overwrites the whole target.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>

#include <string>

namespace wiiu::latte {

/// What a copy reads: the top-left `width` x `height` of `texture`.
struct CopySource {
    gpu::TextureHandle texture;
    gpu::TextureFormat format = gpu::TextureFormat::Unknown; ///< RGBA8 UNORM or sRGB
    u32 width = 0;
    u32 height = 0;
};

/// What it writes: all of `texture`, an RGBA8 UNORM render target
/// `width` x `height`.
struct CopyTarget {
    gpu::TextureHandle texture;
    u32 width = 0;
    u32 height = 0;
};

class SurfaceCopier {
public:
    /// The one format a copy writes.
    static constexpr gpu::TextureFormat target_format = gpu::TextureFormat::RGBA8Unorm;

    SurfaceCopier() = default;

    ~SurfaceCopier() { shutdown(); }

    SurfaceCopier(const SurfaceCopier&) = delete;
    SurfaceCopier& operator=(const SurfaceCopier&) = delete;

    /// Build the two pipelines, one for each kind of source.
    [[nodiscard]] bool initialize(gpu::GpuBackend& backend, std::string& error);

    void shutdown();

    [[nodiscard]] bool ready() const noexcept { return backend_ != nullptr; }

    /// Encode the copy, as a render pass of its own. False, with `error`
    /// saying why, before anything is encoded: a source format it does not
    /// copy, a size of zero, or no pipelines.
    [[nodiscard]] bool copy(const CopySource& source, const CopyTarget& target, std::string& error);

    /// The MSL a copy from an sRGB source, or a UNORM one, runs. For the
    /// tests.
    [[nodiscard]] static std::string shader_source(bool srgb_source);

private:
    gpu::GpuBackend* backend_ = nullptr;
    gpu::PipelineHandle unorm_;
    gpu::PipelineHandle srgb_;
};

} // namespace wiiu::latte
