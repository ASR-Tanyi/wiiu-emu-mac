// Phase 8.1 — clearing a colour buffer and copying it out to a scan buffer.
//
// Together these are what a WHB title does every frame before it draws
// anything -- clear the TV's colour buffer, copy it to the TV's scan buffer,
// swap -- so with them a title that only clears shows its colour. What is
// guarded: the clear writes every element of the slice it names and nothing
// else; the copy puts each pixel where the scan-out reads it, however the
// colour buffer is tiled; neither touches memory before its packet runs; and
// a copy lands in the frame the next swap shows, counted on the GPU's side,
// so a copy inside a display list goes to the right frame too. And the depth
// buffer's clear, which WHB does beside the colour one every frame: depth,
// stencil or both, a packed stencil left alone by a clear of depth.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/depth_buffer.hpp>
#include <wiiu/latte/surface.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using latte::TileMode;

namespace {

constexpr u32 kRgba8 = 0x01A;
constexpr u32 kDouble = 2; // GX2_BUFFERING_MODE_DOUBLE
constexpr u32 kTv720p = 3; // GX2_TV_RENDER_MODE_WIDE_720P
constexpr u32 kScanTv = 1; // GX2_SCAN_TARGET_TV
constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kBuffer = 0x10005000; // the GX2ColorBuffer
constexpr ppcaddr_t kList = 0x10006000;   // a display list
constexpr ppcaddr_t kImage = 0x10100000;  // the colour buffer's pixels
constexpr ppcaddr_t kTvScan = memory::mem1_base + 0x100000;
constexpr u32 kFrameBytes = 1280 * 720 * 4;

using Pixel = std::array<u8, 4>;

constexpr Pixel kBlue{0, 0, 255, 255};
constexpr Pixel kRed{255, 0, 0, 255};

[[nodiscard]] gx2::ColorBuffer
color_buffer(u32 dim, u32 width, u32 height, u32 depth, u32 format, ppcaddr_t image) {
    gx2::ColorBuffer buffer{};
    buffer.surface.dim = dim;
    buffer.surface.width = width;
    buffer.surface.height = height;
    buffer.surface.depth = depth;
    buffer.surface.mip_levels = 1;
    buffer.surface.format = format;
    buffer.surface.use = gx2::surface_use_texture | gx2::surface_use_color_buffer;
    buffer.surface.tile_mode = gx2::tile_mode_default;
    buffer.view_num_slices = 1;
    REQUIRE(gx2::calc_surface_size_and_alignment(buffer.surface));
    buffer.surface.image = image;
    return buffer;
}

[[nodiscard]] gx2::ScanBuffer tv_scan_buffer(u32 format = kRgba8) {
    gx2::ScanBuffer scan;
    scan.address = kTvScan;
    scan.size = 2 * kFrameBytes;
    scan.render_mode = kTv720p;
    scan.surface_format = format;
    scan.buffering_mode = kDouble;
    scan.geometry = {1280, 720, 1280};
    scan.enabled = true;
    return scan;
}

struct MemoryFixture {
    memory::Memory memory;

    MemoryFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
    }

    /// A pixel's bytes, or a value no test writes if they cannot be read.
    [[nodiscard]] Pixel pixel(ppcaddr_t address) const {
        Pixel out{0xEE, 0xEE, 0xEE, 0x01};
        static_cast<void>(memory.read_bytes(address, out));
        return out;
    }

    void put(ppcaddr_t address, const Pixel& value) {
        static_cast<void>(memory.write_bytes(address, value));
    }
};

/// Where each element of a colour buffer's level 0 is, by the slow, sure
/// way: `element_offset`, with the layout worked out once.
class Elements {
public:
    explicit Elements(const gx2::Surface& surface)
        : image_(surface.image), desc_(gx2::surface_desc(surface, 0)),
          swizzle_(gx2::level_swizzle(surface, 0)) {
        const auto layout = gx2::surface_layout(surface, 0);
        REQUIRE(layout.has_value());
        layout_ = *layout;
    }

    explicit Elements(const gx2::ColorBuffer& buffer) : Elements(buffer.surface) {}

    [[nodiscard]] ppcaddr_t at(u32 x, u32 y, u32 slice = 0) const {
        return image_ + static_cast<u32>(latte::element_offset(
                            desc_, layout_, x, y, slice, 0, swizzle_.pipe, swizzle_.bank));
    }

private:
    ppcaddr_t image_;
    latte::SurfaceDesc desc_;
    latte::SurfaceLayout layout_;
    latte::BankPipeSwizzle swizzle_;
};

} // namespace

TEST_CASE_METHOD(MemoryFixture,
                 "a clear fills every element of the slice it names, and only that",
                 "[hle][gx2][scanout]") {
    // An array of three 2D-tiled slices, with a bank swizzle the title chose.
    gx2::ColorBuffer buffer = color_buffer(5, 256, 64, 3, kRgba8, kImage);
    buffer.surface.swizzle = buffer.surface.swizzle | 0x300;
    REQUIRE(memory.fill(kImage, buffer.surface.image_size, 0xAB));
    buffer.view_first_slice = 1;

    CHECK(gx2::clear_color_buffer(memory, buffer, {1.0F, 0.0F, 0.0F, 1.0F}).empty());

    const Elements elements(buffer);
    u64 wrong = 0;
    for (u32 slice = 0; slice < 3; ++slice) {
        const Pixel expected = slice == 1 ? kRed : Pixel{0xAB, 0xAB, 0xAB, 0xAB};
        for (u32 y = 0; y < 64; ++y) {
            for (u32 x = 0; x < 256; ++x) {
                wrong += pixel(elements.at(x, y, slice)) != expected ? 1U : 0U;
            }
        }
    }
    CHECK(wrong == 0);
}

TEST_CASE_METHOD(MemoryFixture,
                 "a clear the emulator cannot encode says so and writes nothing",
                 "[hle][gx2][scanout]") {
    gx2::ColorBuffer buffer = color_buffer(1, 64, 64, 1, 0x823, kImage); // FLOAT_R32_G32_B32_A32
    REQUIRE(memory.fill(kImage, buffer.surface.image_size, 0xAB));

    const std::string problem = gx2::clear_color_buffer(memory, buffer, {1, 1, 1, 1});
    CHECK(problem.find("not supported") != std::string::npos);
    CHECK(pixel(kImage) == Pixel{0xAB, 0xAB, 0xAB, 0xAB});

    gx2::ColorBuffer nowhere = color_buffer(1, 64, 64, 1, kRgba8, 0);
    CHECK_FALSE(gx2::clear_color_buffer(memory, nowhere, {1, 1, 1, 1}).empty());
}

TEST_CASE_METHOD(MemoryFixture,
                 "the scan-out copy puts each pixel where the scan-out reads it",
                 "[hle][gx2][scanout]") {
    // A 2D-tiled 720p buffer, swizzled, holding a pattern that names each
    // pixel; the frame is linear, rows of 1280.
    gx2::ColorBuffer buffer = color_buffer(1, 1280, 720, 1, kRgba8, kImage);
    buffer.surface.swizzle = buffer.surface.swizzle | 0x500;
    const auto pattern = [](u32 x, u32 y) {
        return Pixel{static_cast<u8>(x),
                     static_cast<u8>(y),
                     static_cast<u8>((x >> 8) | ((y >> 8) << 4)),
                     0xFF};
    };
    const Elements elements(buffer);
    for (u32 y = 0; y < 720; ++y) {
        for (u32 x = 0; x < 1280; ++x) {
            put(elements.at(x, y), pattern(x, y));
        }
    }

    CHECK(gx2::copy_to_scan_frame(memory, buffer, tv_scan_buffer(), kTvScan).empty());

    u64 wrong = 0;
    for (u32 y = 0; y < 720; ++y) {
        for (u32 x = 0; x < 1280; ++x) {
            wrong += pixel(kTvScan + ((y * 1280) + x) * 4) != pattern(x, y) ? 1U : 0U;
        }
    }
    CHECK(wrong == 0);
}

TEST_CASE_METHOD(MemoryFixture,
                 "a smaller colour buffer is scaled to the frame",
                 "[hle][gx2][scanout]") {
    gx2::ColorBuffer buffer = color_buffer(1, 640, 360, 1, kRgba8, kImage);
    const Elements elements(buffer);
    for (u32 y = 0; y < 360; ++y) {
        for (u32 x = 0; x < 640; ++x) {
            put(elements.at(x, y), Pixel{static_cast<u8>(x), static_cast<u8>(y), 0, 0xFF});
        }
    }
    CHECK(gx2::copy_to_scan_frame(memory, buffer, tv_scan_buffer(), kTvScan).empty());

    // Each source pixel covers two by two of the frame's.
    CHECK(pixel(kTvScan + ((101 * 1280) + 301) * 4) == Pixel{150, 50, 0, 0xFF});
    CHECK(pixel(kTvScan + ((719 * 1280) + 1279) * 4) ==
          Pixel{static_cast<u8>(639), 359 & 0xFF, 0, 0xFF});
}

TEST_CASE_METHOD(MemoryFixture,
                 "a copy between formats the scan-out cannot convert is refused",
                 "[hle][gx2][scanout]") {
    const gx2::ColorBuffer buffer = color_buffer(1, 1280, 720, 1, kRgba8, kImage);
    REQUIRE(memory.fill(kTvScan, 16, 0x5A));
    // 10_10_10_2 is four bytes a pixel too, but not the same four.
    const std::string problem =
        gx2::copy_to_scan_frame(memory, buffer, tv_scan_buffer(0x019), kTvScan);
    CHECK(problem.find("not supported") != std::string::npos);
    CHECK(pixel(kTvScan) == Pixel{0x5A, 0x5A, 0x5A, 0x5A});

    // sRGB and unorm 8_8_8_8 are the same bytes, and copy as they are.
    CHECK(gx2::copy_to_scan_frame(memory, buffer, tv_scan_buffer(0x41A), kTvScan).empty());
}

TEST_CASE("a colour buffer survives the trip through a packet", "[hle][gx2][scanout]") {
    gx2::ColorBuffer buffer{};
    buffer.surface.width = 1280;
    buffer.surface.image = 0x12345678;
    buffer.surface.mip_level_offset[12] = 0xCAFEF00D;
    buffer.view_first_slice = 7;
    buffer.cmask_offset = 0x89ABCDEF;

    const std::array<u32, gx2::color_buffer_words> words = gx2::color_buffer_to_words(buffer);
    CHECK(gx2::color_buffer_words == 39);
    CHECK(words[1] == 1280);       // width, at 0x04
    CHECK(words[9] == 0x12345678); // image, at 0x24
    CHECK(words[38] == 0x89ABCDEF);

    const gx2::ColorBuffer back = gx2::color_buffer_from_words(words);
    CHECK(back.surface.image == 0x12345678);
    CHECK(back.surface.mip_level_offset[12] == 0xCAFEF00D);
    CHECK(back.view_first_slice == 7);
    CHECK(back.cmask_offset == 0x89ABCDEF);
}

// --------------------------------------------------------------------------
// Through GX2, frame by frame
// --------------------------------------------------------------------------

namespace {

struct ScanOutFixture : MemoryFixture {
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;
    std::vector<coreinit::ScreenFlip> shown;

    ScanOutFixture() {
        coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
        os.screens.on_flip = [this](const coreinit::ScreenFlip& flip, const memory::Memory&) {
            shown.push_back(flip);
            return true;
        };

        call("GX2Init", {0});
        call("GX2SetTVBuffer", {kTvScan, 2 * kFrameBytes, kTv720p, kRgba8, kDouble});
        call("GX2SetTVEnable", {1});

        // WHB's TV colour buffer, in guest memory.
        const gx2::ColorBuffer buffer = color_buffer(1, 1280, 720, 1, kRgba8, kImage);
        std::array<u8, sizeof(gx2::ColorBuffer)> bytes{};
        std::memcpy(bytes.data(), &buffer, sizeof(buffer));
        REQUIRE(memory.write_bytes(kBuffer, bytes));
    }

    void call(std::string_view name, std::vector<u32> arguments, std::vector<f32> floats = {}) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        for (std::size_t i = 0; i < floats.size(); ++i) {
            state.fpr[1 + i].ps0 = static_cast<f64>(floats[i]);
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
    }

    void clear(const Pixel& colour) {
        call("GX2ClearColor",
             {kBuffer},
             {static_cast<f32>(colour[0]) / 255.0F,
              static_cast<f32>(colour[1]) / 255.0F,
              static_cast<f32>(colour[2]) / 255.0F,
              static_cast<f32>(colour[3]) / 255.0F});
    }

    void swap_and_flush() {
        call("GX2SwapScanBuffers", {});
        call("GX2Flush", {});
    }

    void advance_to(u64 tick) {
        state.timebase = tick;
        static_cast<void>(os.scheduler.reschedule(state));
    }
};

} // namespace

TEST_CASE_METHOD(ScanOutFixture,
                 "a title that only clears shows its colour, frame after frame",
                 "[hle][gx2][scanout]") {
    // WHB's frame: clear, copy to the TV, swap.
    clear(kBlue);
    call("GX2CopyColorBufferToScanBuffer", {kBuffer, kScanTv});
    swap_and_flush();

    // GPU work: nothing has happened yet.
    CHECK(pixel(kImage) == Pixel{0, 0, 0, 0});
    CHECK(pixel(kTvScan) == Pixel{0, 0, 0, 0});

    advance_to(gx2::vsync_tick(1));
    REQUIRE(shown.size() == 1);
    CHECK(shown[0].address == kTvScan);
    CHECK(pixel(kTvScan) == kBlue);
    CHECK(pixel(kTvScan + kFrameBytes - 4) == kBlue);
    CHECK(os.gpu.swaps_reached() == 1);

    // The next frame goes to the other half of the double buffer, and the
    // one on screen is left alone.
    clear(kRed);
    call("GX2CopyColorBufferToScanBuffer", {kBuffer, kScanTv});
    swap_and_flush();
    advance_to(gx2::vsync_tick(2));
    REQUIRE(shown.size() == 2);
    CHECK(shown[1].address == kTvScan + kFrameBytes);
    CHECK(pixel(kTvScan + kFrameBytes) == kRed);
    CHECK(pixel(kTvScan) == kBlue);
    CHECK(os.gpu.swaps_reached() == 2);
}

TEST_CASE_METHOD(ScanOutFixture,
                 "a copy in a display list fills the frame of the swap it reaches",
                 "[hle][gx2][scanout]") {
    // The packet names no frame: the GPU picks it when the packet runs, by
    // the swaps it has reached. So one display list, called every frame,
    // fills each frame in turn.
    call("GX2BeginDisplayList", {kList, 0x400});
    call("GX2CopyColorBufferToScanBuffer", {kBuffer, kScanTv});
    call("GX2EndDisplayList", {kList});
    const u32 list_bytes = state.gpr[3];
    REQUIRE(list_bytes > 0);

    clear(kBlue);
    call("GX2CallDisplayList", {kList, list_bytes});
    swap_and_flush();
    clear(kRed);
    call("GX2CallDisplayList", {kList, list_bytes});
    swap_and_flush();

    advance_to(gx2::vsync_tick(2));
    REQUIRE(shown.size() == 2);
    CHECK(pixel(shown[0].address) == kBlue);
    CHECK(pixel(shown[1].address) == kRed);
    CHECK(shown[1].address == kTvScan + kFrameBytes);
}

TEST_CASE_METHOD(ScanOutFixture,
                 "a copy to a scan target that does not exist is dropped",
                 "[hle][gx2][scanout]") {
    clear(kBlue);
    call("GX2CopyColorBufferToScanBuffer", {kBuffer, 2});
    swap_and_flush();
    advance_to(gx2::vsync_tick(1));
    CHECK(pixel(kImage) == kBlue);              // the clear still happened
    CHECK(pixel(kTvScan) == Pixel{0, 0, 0, 0}); // the copy did not
}

// --------------------------------------------------------------------------
// Depth and stencil
// --------------------------------------------------------------------------

namespace {

constexpr ppcaddr_t kDepth = 0x10007000;      // the GX2DepthBuffer
constexpr ppcaddr_t kDepthImage = 0x10500000; // its elements

[[nodiscard]] gx2::DepthBuffer depth_buffer(u32 width, u32 height, u32 format, ppcaddr_t image) {
    gx2::DepthBuffer buffer{};
    buffer.surface.dim = 1;
    buffer.surface.width = width;
    buffer.surface.height = height;
    buffer.surface.depth = 1;
    buffer.surface.mip_levels = 1;
    buffer.surface.format = format;
    buffer.surface.use = gx2::surface_use_depth_buffer;
    buffer.surface.tile_mode = gx2::tile_mode_default;
    buffer.view_num_slices = 1;
    REQUIRE(gx2::calc_surface_size_and_alignment(buffer.surface));
    buffer.surface.image = image;
    return buffer;
}

} // namespace

TEST_CASE_METHOD(MemoryFixture,
                 "a depth clear writes depth or stencil, and leaves the other",
                 "[hle][gx2][scanout]") {
    // R24_X8: depth in the low three bytes of each element, stencil above.
    const gx2::DepthBuffer buffer = depth_buffer(256, 64, 0x011, kDepthImage);
    REQUIRE(memory.fill(kDepthImage, buffer.surface.image_size, 0xAB));
    const Elements elements(buffer.surface);

    const auto count_not = [&](const Pixel& expected) {
        u64 wrong = 0;
        for (u32 y = 0; y < 64; ++y) {
            for (u32 x = 0; x < 256; ++x) {
                wrong += pixel(elements.at(x, y)) != expected ? 1U : 0U;
            }
        }
        return wrong;
    };

    CHECK(gx2::clear_depth_stencil_buffer(memory, buffer, 1.0F, 0x12, gx2::clear_depth).empty());
    CHECK(count_not(Pixel{0xFF, 0xFF, 0xFF, 0xAB}) == 0);

    CHECK(gx2::clear_depth_stencil_buffer(memory, buffer, 0.0F, 0x12, gx2::clear_stencil).empty());
    CHECK(count_not(Pixel{0xFF, 0xFF, 0xFF, 0x12}) == 0);

    // Flags that clear nothing write nothing.
    CHECK(gx2::clear_depth_stencil_buffer(memory, buffer, 0.0F, 0, 0).empty());
    CHECK(count_not(Pixel{0xFF, 0xFF, 0xFF, 0x12}) == 0);
}

TEST_CASE_METHOD(MemoryFixture,
                 "a float depth clear writes the float, and an unknown format says so",
                 "[hle][gx2][scanout]") {
    const gx2::DepthBuffer buffer = depth_buffer(128, 128, 0x80E, kDepthImage);
    CHECK(gx2::clear_depth_stencil_buffer(
              memory, buffer, 1.0F, 0, gx2::clear_depth | gx2::clear_stencil)
              .empty());
    const Elements elements(buffer.surface);
    CHECK(pixel(elements.at(0, 0)) == Pixel{0x00, 0x00, 0x80, 0x3F});
    CHECK(pixel(elements.at(127, 127)) == Pixel{0x00, 0x00, 0x80, 0x3F});

    const gx2::DepthBuffer float24 = depth_buffer(128, 128, 0x811, kDepthImage);
    const std::string problem =
        gx2::clear_depth_stencil_buffer(memory, float24, 0.0F, 0, gx2::clear_depth);
    CHECK(problem.find("not supported") != std::string::npos);
    CHECK(pixel(elements.at(0, 0)) == Pixel{0x00, 0x00, 0x80, 0x3F});
}

TEST_CASE_METHOD(ScanOutFixture,
                 "GX2ClearDepthStencilEx clears when its packet runs, and sets what it is told",
                 "[hle][gx2][scanout]") {
    const gx2::DepthBuffer buffer = depth_buffer(1280, 720, 0x80E, kDepthImage);
    std::array<u8, sizeof(gx2::DepthBuffer)> bytes{};
    std::memcpy(bytes.data(), &buffer, sizeof(buffer));
    REQUIRE(memory.write_bytes(kDepth, bytes));

    // WHB's: depth and stencil, and both clear registers.
    const u32 flags = gx2::clear_depth | gx2::clear_stencil | gx2::clear_set_depth_register |
                      gx2::clear_set_stencil_register;
    call("GX2ClearDepthStencilEx", {kDepth, 0x5A, flags}, {1.0F});
    call("GX2Flush", {});
    CHECK(pixel(kDepthImage) == Pixel{0, 0, 0, 0}); // GPU work: not yet

    advance_to(gx2::vsync_tick(1));
    CHECK(pixel(kDepthImage) == Pixel{0x00, 0x00, 0x80, 0x3F});
    CHECK(pixel(kDepthImage + buffer.surface.image_size - 4) == Pixel{0x00, 0x00, 0x80, 0x3F});
    CHECK(os.gpu.registers().read(latte::reg::db_stencil_clear) == 0x5A);
    CHECK(os.gpu.registers().read(latte::reg::db_depth_clear) == 0x3F800000);
}

TEST_CASE_METHOD(ScanOutFixture,
                 "a depth clear that does not ask to set the clear registers leaves them",
                 "[hle][gx2][scanout]") {
    const gx2::DepthBuffer buffer = depth_buffer(256, 64, 0x80E, kDepthImage);
    std::array<u8, sizeof(gx2::DepthBuffer)> bytes{};
    std::memcpy(bytes.data(), &buffer, sizeof(buffer));
    REQUIRE(memory.write_bytes(kDepth, bytes));

    call("GX2ClearDepthStencilEx", {kDepth, 0x5A, gx2::clear_depth}, {0.5F});
    call("GX2Flush", {});
    advance_to(gx2::vsync_tick(1));
    CHECK(pixel(kDepthImage) == Pixel{0x00, 0x00, 0x00, 0x3F}); // 0.5f
    CHECK(os.gpu.registers().read(latte::reg::db_stencil_clear) == 0);
    CHECK(os.gpu.registers().read(latte::reg::db_depth_clear) == 0);
}
