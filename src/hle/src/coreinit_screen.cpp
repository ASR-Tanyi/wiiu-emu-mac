// Phase 6.8 — OSScreen.
//
// See screen.hpp for the model. Everything here writes guest memory through
// the ordinary accessors, never through a host pointer: a debugger watchpoint
// on a framebuffer (3.7) should see an HLE clear the same way it sees a
// guest store.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/screen.hpp>

#include <algorithm>
#include <array>
#include <vector>

namespace wiiu::hle::coreinit {

namespace {

#include "screen_font.inc"

static_assert(kFontCellWidth == screen_font_width && kFontCellHeight == screen_font_height,
              "the generated font and the grid it is drawn on have to agree");

[[nodiscard]] bool valid_screen(u32 screen) noexcept {
    return screen < screen_count;
}

[[nodiscard]] const ScreenGeometry& geometry_of(ScreenId screen) noexcept {
    return screen_geometry[static_cast<u32>(screen)];
}

/// The screen a call names, or nothing -- which is logged, because a title
/// passing a third screen is a title whose arguments are not where we think.
[[nodiscard]] ScreenState* screen_for(u32 screen, std::string_view function) {
    if (!valid_screen(screen)) {
        WIIU_LOG_WARN(Coreinit, "{}: there is no screen {}", function, screen);
        return nullptr;
    }
    return &detail::active_context()->screens[static_cast<ScreenId>(screen)];
}

/// A colour as it lands in memory: big-endian, so R, G, B, X.
[[nodiscard]] std::array<u8, 4> colour_bytes(u32 colour) noexcept {
    return {static_cast<u8>(colour >> 24),
            static_cast<u8>(colour >> 16),
            static_cast<u8>(colour >> 8),
            static_cast<u8>(colour)};
}

void os_screen_init(Call&) {
    Context& context = *detail::active_context();
    context.screens.initialized = true;
}

void os_screen_shutdown(Call&) {
    Context& context = *detail::active_context();
    context.screens.initialized = false;
}

u32 os_screen_get_buffer_size_ex(Call&, u32 screen) {
    if (!valid_screen(screen)) {
        WIIU_LOG_WARN(Coreinit, "OSScreenGetBufferSizeEx: there is no screen {}", screen);
        return 0;
    }
    return screen_geometry[screen].buffer_bytes();
}

void os_screen_set_buffer_ex(Call&, u32 screen, u32 address) {
    ScreenState* const state = screen_for(screen, "OSScreenSetBufferEx");
    if (state == nullptr) {
        return;
    }
    // wut documents 0x100 alignment and its own console allocates with 4.
    // Nothing here depends on it, so it is noted rather than enforced.
    if ((address & 0xFFU) != 0) {
        WIIU_LOG_DEBUG(Coreinit,
                       "OSScreenSetBufferEx: {} buffer {:#010x} is not 0x100-aligned",
                       screen_name(static_cast<ScreenId>(screen)),
                       address);
    }
    state->buffer = address;
}

void os_screen_enable_ex(Call&, u32 screen, bool enable) {
    if (ScreenState* const state = screen_for(screen, "OSScreenEnableEx"); state != nullptr) {
        state->enabled = enable;
    }
}

void os_screen_clear_buffer_ex(Call& call, u32 screen, u32 colour) {
    const ScreenState* const state = screen_for(screen, "OSScreenClearBufferEx");
    if (state == nullptr || state->buffer == 0) {
        return;
    }
    const auto id = static_cast<ScreenId>(screen);
    const ScreenGeometry& geometry = geometry_of(id);

    // The whole frame, pitch included: hardware clears the bytes, not the
    // visible rectangle.
    std::vector<u8> bytes(geometry.frame_bytes());
    const std::array<u8, 4> texel = colour_bytes(colour);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 4) {
        std::copy(texel.begin(), texel.end(), bytes.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    if (!call.memory.write_bytes(state->work_buffer(id), bytes)) {
        WIIU_LOG_WARN(Coreinit,
                      "OSScreenClearBufferEx: the {} buffer at {:#010x} is not writable",
                      screen_name(id),
                      state->work_buffer(id));
    }
}

void os_screen_put_pixel_ex(Call& call, u32 screen, u32 x, u32 y, u32 colour) {
    const ScreenState* const state = screen_for(screen, "OSScreenPutPixelEx");
    if (state == nullptr || state->buffer == 0) {
        return;
    }
    const auto id = static_cast<ScreenId>(screen);
    const ScreenGeometry& geometry = geometry_of(id);

    // Out of range returns immediately, as the hardware does.
    if (x >= geometry.width || y >= geometry.height) {
        return;
    }
    const ppcaddr_t address = state->work_buffer(id) + ((x + (y * geometry.pitch)) * 4);
    static_cast<void>(call.memory.try_write<u32>(address, colour));
}

void os_screen_put_font_ex(Call& call, u32 screen, u32 column, u32 row, u32 text) {
    const ScreenState* const state = screen_for(screen, "OSScreenPutFontEx");
    if (state == nullptr || state->buffer == 0 || text == 0) {
        return;
    }
    put_font(
        call.memory, *state, static_cast<ScreenId>(screen), column, row, call.read_string(text));
}

void os_screen_flip_buffers_ex(Call& call, u32 screen) {
    Context& context = *detail::active_context();
    ScreenState* const state = screen_for(screen, "OSScreenFlipBuffersEx");
    if (state == nullptr) {
        return;
    }
    const auto id = static_cast<ScreenId>(screen);
    ++state->flips;

    if (!context.screens.on_flip || state->buffer == 0) {
        return;
    }
    const ScreenFlip flip{
        .screen = id,
        .address = state->visible_buffer(id),
        .geometry = geometry_of(id),
        .enabled = state->enabled,
        .flip_count = state->flips,
        .texture = {}, // OSScreen's frames are bytes
    };
    if (!context.screens.on_flip(flip, call.memory)) {
        // The frontend wants the title stopped -- its window was closed.
        // Halting the core is how a handler ends a run (OSFatal does the
        // same); the flag is what tells the caller it was asked for.
        context.screens.stop_requested = true;
        call.state.halted = true;
    }
}

} // namespace

std::string_view screen_name(ScreenId screen) noexcept {
    switch (screen) {
        case ScreenId::Tv:
            return "TV";
        case ScreenId::Drc:
            return "DRC";
    }
    return "?";
}

ppcaddr_t ScreenState::frame(u32 index, ScreenId screen) const noexcept {
    if (buffer == 0) {
        return 0;
    }
    return buffer + (index * screen_geometry[static_cast<u32>(screen)].frame_bytes());
}

void Screens::reset() noexcept {
    initialized = false;
    state = {};
    stop_requested = false;
}

void put_font(memory::Memory& memory,
              const ScreenState& state,
              ScreenId screen,
              u32 column,
              u32 row,
              std::string_view text) {
    const ScreenGeometry& geometry = geometry_of(screen);
    const ppcaddr_t base = state.work_buffer(screen);
    if (base == 0) {
        return;
    }

    u32 cell_x = column;
    u32 cell_y = row;
    for (const char character : text) {
        if (character == '\n') {
            // Back to the column the string started in, one row down. Not
            // documented; Cemu does it, and a title that embeds a newline
            // means it.
            cell_x = column;
            ++cell_y;
            continue;
        }

        const auto code = static_cast<u8>(character);
        const u32 left = cell_x * screen_font_width;
        const u32 top = cell_y * screen_font_height;
        ++cell_x;

        // Unprintable characters take their cell and draw nothing.
        if (code < kFontFirst || code > kFontLast) {
            continue;
        }
        if (left >= geometry.width || top >= geometry.height) {
            continue;
        }

        const std::size_t glyph = static_cast<std::size_t>(code - kFontFirst) * kFontGlyphBytes;
        const u32 columns = std::min(screen_font_width, geometry.width - left);
        const u32 rows = std::min(screen_font_height, geometry.height - top);

        for (u32 y = 0; y < rows; ++y) {
            const ppcaddr_t line = base + (((top + y) * geometry.pitch) + left) * 4;
            for (u32 x = 0; x < columns; ++x) {
                const u32 pixel = (y * screen_font_width) + x;
                const u8 packed = kFont[glyph + (pixel / 2)];
                const u32 coverage = (pixel % 2) == 0 ? (packed >> 4U) : (packed & 0xFU);
                if (coverage == 0) {
                    continue;
                }

                // White, anti-aliased over what is there. The fourth byte is
                // ignored by the display and left alone here.
                const ppcaddr_t address = line + (x * 4);
                std::array<u8, 3> rgb{};
                if (!memory.read_bytes(address, rgb)) {
                    continue;
                }
                for (u8& channel : rgb) {
                    channel = static_cast<u8>(
                        ((channel * (15U - coverage)) + (255U * coverage) + 7U) / 15U);
                }
                static_cast<void>(memory.write_bytes(address, rgb));
            }
        }
    }
}

void register_screen(Registry& registry, Context& context) {
    detail::set_active_context(&context);

    registry.add("coreinit", "OSScreenInit", os_screen_init);
    registry.add("coreinit", "OSScreenShutdown", os_screen_shutdown);
    registry.add("coreinit", "OSScreenGetBufferSizeEx", os_screen_get_buffer_size_ex);
    registry.add("coreinit", "OSScreenSetBufferEx", os_screen_set_buffer_ex);
    registry.add("coreinit", "OSScreenEnableEx", os_screen_enable_ex);
    registry.add("coreinit", "OSScreenClearBufferEx", os_screen_clear_buffer_ex);
    registry.add("coreinit", "OSScreenPutPixelEx", os_screen_put_pixel_ex);
    registry.add("coreinit", "OSScreenPutFontEx", os_screen_put_font_ex);
    registry.add("coreinit", "OSScreenFlipBuffersEx", os_screen_flip_buffers_ex);
}

} // namespace wiiu::hle::coreinit
