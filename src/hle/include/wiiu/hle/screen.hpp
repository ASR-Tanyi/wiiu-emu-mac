// Phase 6.8 — OSScreen, the framebuffer API.
//
// The simplest way a Wii U program can put pixels in front of someone: the
// title hands coreinit a buffer, draws into it with plain stores (or with
// `OSScreenPutPixelEx`/`OSScreenPutFontEx`), and flips. No GPU state, no
// shaders, no command buffers -- which is why it is the first thing this
// emulator draws (7.6), and why homebrew's console (`WHBLogConsole`) is built
// on it.
//
// The model is wut's documentation plus Cemu's implementation, which agree:
//
//   * the buffer the title supplies holds *two* frames back to back, and
//     `OSScreenGetBufferSizeEx` says how big the pair is;
//   * every drawing call writes the *work* buffer, and a flip makes it the
//     visible one while the previously visible one becomes the work buffer;
//   * a colour is `0xRRGGBBXX`, stored big-endian, so memory holds R, G, B, X
//     in that order -- an RGBA8 surface, with the last byte ignored.
//
// decaf models a single buffer and forces "alpha" by OR-ing 0xFF000000 into
// the colour, which in this layout is red. Both are worth not copying.
//
// The DRC's buffer is 896 pixels to a row and 854 of them are shown: Cemu
// uses that pitch, homebrew that writes the buffer directly uses it, and the
// odd width is the GamePad's panel.
//
// What reaches the screen is the frontend's business, not coreinit's. A flip
// calls `Screens::on_flip` with the visible buffer, and the frontend copies
// it out and presents it (7.6) -- or, in a test, checks what was drawn.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>
#include <functional>
#include <string_view>

namespace wiiu::hle::coreinit {

enum class ScreenId : u32 {
    Tv = 0,
    Drc = 1,
};

inline constexpr u32 screen_count = 2;

[[nodiscard]] std::string_view screen_name(ScreenId screen) noexcept;

/// One screen's shape, in pixels. `pitch` is the distance between rows.
struct ScreenGeometry {
    u32 width = 0;
    u32 height = 0;
    u32 pitch = 0;

    /// Bytes in one of the two frames.
    [[nodiscard]] constexpr u32 frame_bytes() const noexcept { return pitch * height * 4; }

    /// What `OSScreenGetBufferSizeEx` returns: both frames.
    [[nodiscard]] constexpr u32 buffer_bytes() const noexcept { return frame_bytes() * 2; }
};

inline constexpr std::array<ScreenGeometry, screen_count> screen_geometry{{
    {1280, 720, 1280}, // TV: 720p
    {854, 480, 896},   // DRC: the GamePad's panel, in a wider buffer
}};

static_assert(screen_geometry[0].buffer_bytes() == 0x708000);
static_assert(screen_geometry[1].buffer_bytes() == 0x348000);

/// The OSScreen font's character cell, in pixels. Neither the console's own
/// grid nor its font is documented anywhere reachable; this is DejaVu Sans
/// Mono in a 12x24 cell, anti-aliased as wut says the real one is, with the
/// grid's origin at the top-left corner of the screen.
inline constexpr u32 screen_font_width = 12;
inline constexpr u32 screen_font_height = 24;

struct ScreenState {
    /// What `OSScreenSetBufferEx` was given, or zero.
    ppcaddr_t buffer = 0;
    bool enabled = false;

    /// Flips so far. The work buffer is the one `flips` is even for.
    u64 flips = 0;

    [[nodiscard]] ppcaddr_t frame(u32 index, ScreenId screen) const noexcept;

    /// Where drawing goes.
    [[nodiscard]] ppcaddr_t work_buffer(ScreenId screen) const noexcept {
        return frame(static_cast<u32>(flips & 1U), screen);
    }

    /// What is on screen.
    [[nodiscard]] ppcaddr_t visible_buffer(ScreenId screen) const noexcept {
        return frame(static_cast<u32>((flips & 1U) ^ 1U), screen);
    }
};

/// What a flip hands the frontend.
struct ScreenFlip {
    ScreenId screen = ScreenId::Tv;

    /// The frame that has just become visible.
    ppcaddr_t address = 0;
    ScreenGeometry geometry;

    /// A disabled screen shows black, whatever is in its buffer.
    bool enabled = false;

    u64 flip_count = 0;
};

struct Screens {
    bool initialized = false;
    std::array<ScreenState, screen_count> state{};

    /// Called after every flip, from inside the guest's call, with memory to
    /// read the frame from. Returns false to stop the title -- which is how a
    /// window being closed reaches a run loop that otherwise would not look
    /// up until the title returned.
    std::function<bool(const ScreenFlip&, const memory::Memory&)> on_flip;

    /// Set when `on_flip` asked to stop, so whoever ran the title can tell a
    /// requested stop from a fault.
    bool stop_requested = false;

    [[nodiscard]] ScreenState& operator[](ScreenId screen) noexcept {
        return state[static_cast<u32>(screen)];
    }

    [[nodiscard]] const ScreenState& operator[](ScreenId screen) const noexcept {
        return state[static_cast<u32>(screen)];
    }

    /// Forget the title's buffers. The flip callback is the frontend's and
    /// survives, the way a registered HLE function does.
    void reset() noexcept;
};

/// Draw one string into a screen's work buffer, as `OSScreenPutFontEx` does.
/// Exposed for the tests, which check pixels rather than trust a call.
void put_font(memory::Memory& memory,
              const ScreenState& state,
              ScreenId screen,
              u32 column,
              u32 row,
              std::string_view text);

} // namespace wiiu::hle::coreinit
