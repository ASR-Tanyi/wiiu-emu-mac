// Phase 6.8 — OSScreen.
//
// The pixels are checked, not the calls: what this API is for is bytes in a
// buffer that the frontend will scan out, and the ways it goes wrong are all
// about which bytes -- the wrong one of the two frames, a colour in the wrong
// byte order, a row stride of the visible width instead of the pitch.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/hle/screen.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using coreinit::ScreenId;

namespace {

constexpr ppcaddr_t kTvBuffer = 0xF4000000; // MEM1, where WHB's console puts it
constexpr ppcaddr_t kDrcBuffer = 0xF4800000;
constexpr ppcaddr_t kText = 0x10002000;

constexpr u32 kTv = 0;
constexpr u32 kDrc = 1;

struct ScreenFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    std::vector<coreinit::ScreenFlip> flips;
    bool keep_running = true;

    ScreenFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);

        coreinit::register_all(registry, os);
        os.memory = &memory;
        os.screens.on_flip = [this](const coreinit::ScreenFlip& flip, const memory::Memory&) {
            flips.push_back(flip);
            return keep_running;
        };
    }

    u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("coreinit", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size() && i < 8; ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    void set_up_screens() {
        call("OSScreenInit", {});
        call("OSScreenSetBufferEx", {kTv, kTvBuffer});
        call("OSScreenSetBufferEx", {kDrc, kDrcBuffer});
        call("OSScreenEnableEx", {kTv, 1});
        call("OSScreenEnableEx", {kDrc, 1});
    }

    [[nodiscard]] std::array<u8, 4> pixel(ppcaddr_t frame, u32 pitch, u32 x, u32 y) const {
        std::array<u8, 4> bytes{};
        REQUIRE(memory.read_bytes(frame + ((x + (y * pitch)) * 4), bytes));
        return bytes;
    }

    void put_string(std::string_view text) {
        const std::vector<u8> bytes(text.begin(), text.end());
        REQUIRE(memory.write_bytes(kText, bytes));
        REQUIRE(memory.try_write<u8>(kText + static_cast<u32>(bytes.size()), 0));
    }
};

constexpr u32 kTvFrame = coreinit::screen_geometry[0].frame_bytes();
constexpr u32 kTvPitch = coreinit::screen_geometry[0].pitch;
constexpr u32 kDrcFrame = coreinit::screen_geometry[1].frame_bytes();
constexpr u32 kDrcPitch = coreinit::screen_geometry[1].pitch;

} // namespace

TEST_CASE_METHOD(ScreenFixture, "a screen's buffer holds two frames", "[hle][screen]") {
    // 1280x720 and 896x480 (the DRC's pitch, not its 854 visible columns),
    // four bytes a pixel, twice over.
    CHECK(call("OSScreenGetBufferSizeEx", {kTv}) == 0x708000U);
    CHECK(call("OSScreenGetBufferSizeEx", {kDrc}) == 0x348000U);
    CHECK(call("OSScreenGetBufferSizeEx", {2}) == 0U);
}

TEST_CASE_METHOD(ScreenFixture, "a clear fills the work frame, big-endian", "[hle][screen]") {
    set_up_screens();
    call("OSScreenClearBufferEx", {kTv, 0x993333FF});

    // 0xRRGGBBXX stored big-endian: R, G, B, X in memory, which is what an
    // RGBA8 surface reads.
    CHECK(pixel(kTvBuffer, kTvPitch, 0, 0) == std::array<u8, 4>{0x99, 0x33, 0x33, 0xFF});
    CHECK(pixel(kTvBuffer, kTvPitch, 1279, 719) == std::array<u8, 4>{0x99, 0x33, 0x33, 0xFF});

    // Only the work frame: the visible one is left alone.
    CHECK(memory.read<u32>(kTvBuffer + kTvFrame) == 0U);

    // The DRC clear covers its pitch, not just the 854 visible columns.
    call("OSScreenClearBufferEx", {kDrc, 0x00FF0000});
    CHECK(pixel(kDrcBuffer, kDrcPitch, 895, 479) == std::array<u8, 4>{0x00, 0xFF, 0x00, 0x00});
    CHECK(memory.read<u32>(kDrcBuffer + kDrcFrame) == 0U);
}

TEST_CASE_METHOD(ScreenFixture, "a flip swaps the work and visible frames", "[hle][screen]") {
    set_up_screens();

    call("OSScreenClearBufferEx", {kTv, 0xFF0000FF});
    call("OSScreenFlipBuffersEx", {kTv});
    REQUIRE(flips.size() == 1);
    CHECK(flips[0].screen == ScreenId::Tv);
    CHECK(flips[0].address == kTvBuffer); // what was just drawn is what is shown
    CHECK(flips[0].enabled);

    // Now the second frame is the one drawn into.
    call("OSScreenClearBufferEx", {kTv, 0x0000FFFF});
    CHECK(memory.read<u32>(kTvBuffer) == 0xFF0000FFU);
    CHECK(memory.read<u32>(kTvBuffer + kTvFrame) == 0x0000FFFFU);

    call("OSScreenFlipBuffersEx", {kTv});
    REQUIRE(flips.size() == 2);
    CHECK(flips[1].address == kTvBuffer + kTvFrame);
    CHECK(flips[1].flip_count == 2U);
}

TEST_CASE_METHOD(ScreenFixture, "a pixel lands at pitch, and nowhere else", "[hle][screen]") {
    set_up_screens();
    call("OSScreenPutPixelEx", {kDrc, 3, 2, 0x12345678});
    CHECK(memory.read<u32>(kDrcBuffer + ((3 + (2 * kDrcPitch)) * 4)) == 0x12345678U);

    // Column 854 is inside the DRC's pitch but outside its panel: hardware
    // returns without writing, and so does this.
    call("OSScreenPutPixelEx", {kDrc, 854, 0, 0xFFFFFFFF});
    CHECK(memory.read<u32>(kDrcBuffer + (854 * 4)) == 0U);
    call("OSScreenPutPixelEx", {kTv, 0, 720, 0xFFFFFFFF});
    CHECK(memory.read<u32>(kTvBuffer + (720 * kTvPitch * 4)) == 0U);
}

TEST_CASE_METHOD(ScreenFixture, "text is white and anti-aliased in its cell", "[hle][screen]") {
    set_up_screens();
    call("OSScreenClearBufferEx", {kTv, 0x000000AA});
    put_string("H");
    call("OSScreenPutFontEx", {kTv, 2, 1, kText});

    // Column 2, row 1: the cell at (24, 24), 12 by 24.
    const u32 left = 2 * coreinit::screen_font_width;
    const u32 top = 1 * coreinit::screen_font_height;

    u32 white = 0;
    u32 blended = 0;
    for (u32 y = 0; y < coreinit::screen_font_height; ++y) {
        for (u32 x = 0; x < coreinit::screen_font_width; ++x) {
            const std::array<u8, 4> bytes = pixel(kTvBuffer, kTvPitch, left + x, top + y);
            // Grey, never tinted: white over black at any coverage.
            CHECK(bytes[0] == bytes[1]);
            CHECK(bytes[1] == bytes[2]);
            // The ignored byte is left as the clear wrote it.
            CHECK(bytes[3] == 0xAA);
            white += bytes[0] == 0xFF ? 1U : 0U;
            blended += (bytes[0] != 0 && bytes[0] != 0xFF) ? 1U : 0U;
        }
    }
    CHECK(white > 20U);  // an H has two solid stems
    CHECK(blended > 0U); // and edges that are neither on nor off

    // Nothing outside the cell.
    CHECK(pixel(kTvBuffer, kTvPitch, left - 1, top + 10)[0] == 0U);
    CHECK(pixel(kTvBuffer, kTvPitch, left + 12, top + 10)[0] == 0U);
}

TEST_CASE_METHOD(ScreenFixture, "a newline goes back to the starting column", "[hle][screen]") {
    set_up_screens();
    put_string("|\n|");
    call("OSScreenPutFontEx", {kTv, 5, 0, kText});

    // Both bars in the same column, one row apart.
    const u32 middle = (5 * coreinit::screen_font_width) + (coreinit::screen_font_width / 2);
    CHECK(pixel(kTvBuffer, kTvPitch, middle, 12)[0] > 0U);
    CHECK(pixel(kTvBuffer, kTvPitch, middle, 24 + 12)[0] > 0U);
    // And not in the column after the first bar.
    CHECK(pixel(kTvBuffer, kTvPitch, middle + coreinit::screen_font_width, 12)[0] == 0U);
}

TEST_CASE_METHOD(ScreenFixture, "text past the edge is clipped, not wrapped", "[hle][screen]") {
    set_up_screens();
    // The DRC is 854 wide: column 71 starts at 852 and has two pixels left.
    // Writing past it would land in the pitch padding, or the next row.
    put_string("MM");
    call("OSScreenPutFontEx", {kDrc, 71, 0, kText});
    for (u32 y = 0; y < coreinit::screen_font_height; ++y) {
        for (u32 x = 854; x < kDrcPitch; ++x) {
            CHECK(memory.read<u32>(kDrcBuffer + ((x + (y * kDrcPitch)) * 4)) == 0U);
        }
    }
}

TEST_CASE_METHOD(ScreenFixture, "a disabled screen still flips, and says so", "[hle][screen]") {
    set_up_screens();
    call("OSScreenEnableEx", {kDrc, 0});
    call("OSScreenFlipBuffersEx", {kDrc});
    REQUIRE(flips.size() == 1);
    CHECK(flips[0].screen == ScreenId::Drc);
    CHECK_FALSE(flips[0].enabled);
}

TEST_CASE_METHOD(ScreenFixture, "the frontend can stop the title at a flip", "[hle][screen]") {
    set_up_screens();
    keep_running = false;
    call("OSScreenFlipBuffersEx", {kTv});

    // Halting is how a handler ends a run; the flag is how the caller tells
    // that from OSFatal.
    CHECK(state.halted);
    CHECK(os.screens.stop_requested);
}

TEST_CASE_METHOD(ScreenFixture, "reset forgets the buffers and keeps the hook", "[hle][screen]") {
    set_up_screens();
    os.reset();
    CHECK(os.screens[ScreenId::Tv].buffer == 0U);
    CHECK_FALSE(os.screens.initialized);
    CHECK(static_cast<bool>(os.screens.on_flip));

    // With no buffer there is nothing to draw into, and nothing is.
    call("OSScreenClearBufferEx", {kTv, 0xFFFFFFFF});
    CHECK(memory.read<u32>(kTvBuffer) == 0U);
}
