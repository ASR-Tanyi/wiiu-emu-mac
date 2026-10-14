// Phase 7.6 / 7.7 — putting the Wii U's screens in a window.
//
// The Wii U has two outputs, the TV and the GamePad (DRC), and the window has
// one drawable. The presenter takes the latest image of each, scales it into
// the drawable with its aspect kept, and draws nothing else -- the bars around
// it are cleared to black, which is what a TV shows around a 16:9 picture.
//
// It is written against `GpuBackend` alone, so it runs, and is tested, on the
// null backend too. Its first client is OSScreen (6.8); GX2's scan-out
// (8.1) hands it a colour buffer the same way.
//
// Two things are worth knowing before changing it:
//
//   * **An image is copied when it is handed over**, not read later. A
//     flipped OSScreen frame is the one the title will draw into two flips
//     from now, and with `frames_in_flight` frames queued the GPU can still be
//     reading the last copy when that happens. Scan-out on hardware is a
//     snapshot too.
//   * **Uploads rotate through `frames_in_flight` textures per screen.**
//     Writing the texture a queued frame is still sampling is a race the GPU
//     wins sometimes, which shows up as a torn or flickering frame that no
//     test catches. The next texture in the ring is never one a frame in
//     flight is reading, because the backend never has more than
//     `frames_in_flight` frames outstanding (7.2).
#pragma once

#include <wiiu/gpu/backend.hpp>

#include <array>
#include <span>
#include <string>
#include <vector>

namespace wiiu::gpu {

enum class PresentScreen : u8 {
    Tv = 0,
    Drc = 1,
};

inline constexpr u32 present_screen_count = 2;

/// One screen's image: RGBA8, rows `bytes_per_row` apart, the fourth byte
/// ignored -- which is OSScreen's layout exactly, so no conversion is needed
/// on the way in.
struct ScreenImage {
    std::span<const u8> pixels;
    u32 width = 0;
    u32 height = 0;
    u32 bytes_per_row = 0;
};

/// Where each screen lands in a drawable, in pixels.
struct PresentLayout {
    std::array<Rect, present_screen_count> rects{};
    std::array<bool, present_screen_count> visible{};
};

/// Fit the shown screens into a `width` x `height` drawable, aspect kept and
/// centred. One screen gets the whole drawable; two are stacked, TV above
/// GamePad, each in its half. A screen that would come out with no pixels is
/// not shown.
[[nodiscard]] PresentLayout layout_screens(u32 width,
                                           u32 height,
                                           const std::array<Rect, present_screen_count>& sizes,
                                           const std::array<bool, present_screen_count>& shown);

class Presenter {
public:
    Presenter() = default;

    ~Presenter() { shutdown(); }

    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    /// Build the pipeline and sampler. `target` is the format of whatever
    /// `draw` will be given -- the drawable's, normally.
    [[nodiscard]] bool initialize(GpuBackend& backend, TextureFormat target, std::string& error);

    void shutdown();

    /// Take a copy of a screen's newest image. Nothing reaches the GPU until
    /// `draw`, so this can be called from anywhere -- including from inside
    /// an HLE call, where a flip arrives.
    void update(PresentScreen screen, const ScreenImage& image);

    /// Show or hide a screen. A hidden one is not drawn, and its space goes
    /// to the other: an OSScreen title that disables the DRC gets the TV in
    /// the whole window.
    void set_shown(PresentScreen screen, bool shown);

    /// Has this screen been given an image yet?
    [[nodiscard]] bool has_image(PresentScreen screen) const noexcept;

    /// Upload what changed since the last draw and draw every shown screen
    /// into `target`, which is `width` x `height`. Call inside a frame; the
    /// slot rotation relies on the backend's frame pacing.
    [[nodiscard]] bool draw(TextureHandle target, u32 width, u32 height);

    /// The layout the last `draw` used. For the tests, and for mapping a
    /// click back to GamePad touch coordinates later (9.7).
    [[nodiscard]] const PresentLayout& last_layout() const noexcept { return layout_; }

private:
    struct Screen {
        std::vector<u8> pixels; ///< the latest image, tightly packed
        u32 width = 0;
        u32 height = 0;
        bool dirty = false;
        bool shown = true;

        std::array<TextureHandle, frames_in_flight> slots{};
        std::array<Rect, frames_in_flight> slot_sizes{};
        u32 current = 0; ///< the slot the last draw sampled
        bool uploaded = false;
    };

    [[nodiscard]] bool upload(Screen& screen, std::string_view name);

    GpuBackend* backend_ = nullptr;
    TextureFormat target_format_ = TextureFormat::Unknown;
    PipelineHandle pipeline_;
    SamplerHandle sampler_;
    std::array<Screen, present_screen_count> screens_{};
    PresentLayout layout_;
};

} // namespace wiiu::gpu
