// Phase 8.1 — the state a draw reads: render targets, viewport, scissor.
//
// What is guarded: the values, against decaf's and Cemu's arithmetic where
// they agree and with the choice named where they do not; and that each call
// reaches the register file through the command buffer, when its submission
// retires, at the addresses Linux's register list gives.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_state.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/color_buffer.hpp>
#include <wiiu/latte/depth_buffer.hpp>
#include <wiiu/latte/viewport.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstring>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using Catch::Approx;

namespace {

[[nodiscard]] u32 bits(f32 value) {
    return std::bit_cast<u32>(value);
}

} // namespace

TEST_CASE("a viewport is a transform, a guard band and a depth range", "[hle][gx2][state]") {
    const gx2::ViewportReg reg = gx2::viewport_registers(0, 0, 1280, 720, 0, 1);

    // Half the size, centred, y flipped; depth from [-1, 1] to [0, 1].
    CHECK(reg.transform[0] == 640.0F);
    CHECK(reg.transform[1] == 640.0F);
    CHECK(reg.transform[2] == -360.0F);
    CHECK(reg.transform[3] == 360.0F);
    CHECK(reg.transform[4] == 0.5F);
    CHECK(reg.transform[5] == 0.5F);

    // decaf's guard band: to Latte's 8192-pixel edge and back, over the size.
    CHECK(reg.guard_band[0] == Approx((8192.0 * 2 + 720) / 720));
    CHECK(reg.guard_band[1] == 1.0F);
    CHECK(reg.guard_band[2] == Approx((8192.0 * 2 + 1280) / 1280));
    CHECK(reg.guard_band[3] == 1.0F);
    CHECK(reg.z_min == 0.0F);
    CHECK(reg.z_max == 1.0F);
}

TEST_CASE("an empty or flipped viewport keeps a sane guard band", "[hle][gx2][state]") {
    const gx2::ViewportReg empty = gx2::viewport_registers(0, 0, 0, 0, 0, 1);
    CHECK(empty.guard_band[0] == 1.0F);
    CHECK(empty.guard_band[2] == 1.0F);

    // Upside down: the transform follows the sign, the guard band does not.
    const gx2::ViewportReg flipped = gx2::viewport_registers(0, 720, 1280, -720, 1, 0);
    CHECK(flipped.transform[2] == 360.0F);
    CHECK(flipped.transform[3] == 360.0F);
    CHECK(flipped.guard_band[0] == Approx((8192.0 * 2 + 720) / 720));
    CHECK(flipped.transform[4] == -0.5F);
    CHECK(flipped.z_min == 0.0F); // the least of near and far, whichever is which
    CHECK(flipped.z_max == 1.0F);
}

TEST_CASE("a scissor is two corners, clamped to 8192", "[hle][gx2][state]") {
    const gx2::ScissorReg reg = gx2::scissor_registers(0, 0, 1280, 720);
    const latte::PaScGenericScissorTl top_left{.raw = reg.top_left};
    const latte::PaScGenericScissorBr bottom_right{.raw = reg.bottom_right};
    CHECK(top_left.x == 0);
    CHECK(top_left.y == 0);
    CHECK(top_left.window_offset_disable);
    CHECK(bottom_right.x == 1280);
    CHECK(bottom_right.y == 720);
    CHECK(reg.top_left == 0x80000000);
    CHECK(reg.bottom_right == (1280U | (720U << 16)));

    const gx2::ScissorReg big = gx2::scissor_registers(100, 9000, 9000, 50);
    CHECK(latte::PaScGenericScissorTl{.raw = big.top_left}.y == 8192);
    CHECK(latte::PaScGenericScissorBr{.raw = big.bottom_right}.x == 8192);
    CHECK(latte::PaScGenericScissorBr{.raw = big.bottom_right}.y == 8192);
}

// --------------------------------------------------------------------------
// Through GX2
// --------------------------------------------------------------------------

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kReg = 0x10004000;
constexpr ppcaddr_t kBuffer = 0x10005000;
constexpr ppcaddr_t kImage = 0x10100000;
constexpr ppcaddr_t kHiZ = 0x10600000;

struct StateFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    StateFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
        call("GX2Init", {0});
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

    /// Flush, and let the submission retire.
    void run_gpu() {
        call("GX2Flush", {});
        state.timebase += os.gpu.timeline().latency();
        static_cast<void>(os.scheduler.reschedule(state));
    }

    [[nodiscard]] u32 reg(u32 address) const { return os.gpu.registers().read(address); }

    [[nodiscard]] u32 physical(ppcaddr_t address) const {
        const memory::Region* const region = memory.region_of(address);
        REQUIRE(region != nullptr);
        physaddr_t out = 0;
        REQUIRE(memory::to_physical(*region, address, out));
        return out;
    }

    template <typename T>
    void put(ppcaddr_t address, const T& value) {
        std::array<u8, sizeof(T)> bytes{};
        std::memcpy(bytes.data(), &value, sizeof(T));
        REQUIRE(memory.write_bytes(address, bytes));
    }
};

} // namespace

TEST_CASE_METHOD(StateFixture,
                 "GX2SetViewport and GX2SetScissor reach the register file when they retire",
                 "[hle][gx2][state]") {
    call("GX2SetViewport", {}, {0, 0, 1280, 720, 0, 1});
    call("GX2SetScissor", {0, 0, 1280, 720});
    CHECK(reg(latte::reg::pa_cl_vport_xscale_0) == 0); // not before the GPU gets there

    run_gpu();
    CHECK(reg(latte::reg::pa_cl_vport_xscale_0) == bits(640.0F));
    CHECK(reg(latte::reg::pa_cl_vport_xscale_0 + 4) == bits(640.0F));  // XOFFSET
    CHECK(reg(latte::reg::pa_cl_vport_xscale_0 + 8) == bits(-360.0F)); // YSCALE
    CHECK(reg(latte::reg::pa_cl_vport_xscale_0 + 20) == bits(0.5F));   // ZOFFSET
    CHECK(reg(latte::reg::pa_cl_gb_vert_clip_adj + 4) == bits(1.0F));  // VERT_DISC
    CHECK(reg(latte::reg::pa_sc_vport_zmax_0) == bits(1.0F));
    CHECK(reg(latte::reg::pa_sc_generic_scissor_tl) == 0x80000000);
    CHECK(reg(latte::reg::pa_sc_generic_scissor_br) == (1280U | (720U << 16)));
}

TEST_CASE_METHOD(StateFixture,
                 "the Init and Set halves write the same registers as the one call",
                 "[hle][gx2][state]") {
    call("GX2InitViewportReg", {kReg}, {10, 20, 640, 360, 0.25F, 0.75F});
    call("GX2SetViewportReg", {kReg});
    call("GX2InitScissorReg", {kReg + 0x40, 10, 20, 640, 360});
    call("GX2SetScissorReg", {kReg + 0x40});
    run_gpu();

    const gx2::ViewportReg expected = gx2::viewport_registers(10, 20, 640, 360, 0.25F, 0.75F);
    for (u32 i = 0; i < 6; ++i) {
        CHECK(reg(latte::reg::pa_cl_vport_xscale_0 + (i * 4)) == bits(expected.transform[i]));
    }
    CHECK(reg(latte::reg::pa_cl_gb_vert_clip_adj) == bits(expected.guard_band[0]));
    CHECK(reg(latte::reg::pa_sc_vport_zmin_0) == bits(0.25F));
    CHECK(reg(latte::reg::pa_sc_generic_scissor_br) == (650U | (380U << 16)));
    CHECK(memory.read<u32>(kReg + 0x28) == bits(0.25F)); // the title's copy, big-endian
}

TEST_CASE_METHOD(StateFixture,
                 "GX2SetColorBuffer writes a render target's registers",
                 "[hle][gx2][state]") {
    gx2::ColorBuffer buffer{};
    buffer.surface.dim = 1;
    buffer.surface.width = 1280;
    buffer.surface.height = 720;
    buffer.surface.depth = 1;
    buffer.surface.mip_levels = 1;
    buffer.surface.format = 0x01A;
    buffer.surface.use = gx2::surface_use_color_buffer;
    buffer.view_num_slices = 1;
    REQUIRE(gx2::calc_surface_size_and_alignment(buffer.surface));
    buffer.surface.image = kImage;
    buffer.surface.swizzle = buffer.surface.swizzle | 0x300; // a bank swizzle of 3
    const auto registers = gx2::color_buffer_registers(buffer);
    REQUIRE(registers.has_value());
    buffer.cb_color_size = registers->size;
    buffer.cb_color_info = registers->info;
    buffer.cb_color_view = registers->view;
    put(kBuffer, buffer);

    call("GX2SetColorBuffer", {kBuffer, 0});
    call("GX2SetColorBuffer", {kBuffer, 2});
    run_gpu();

    // In 256-byte units, with the swizzle in the low bits.
    const u32 base = (physical(kImage) ^ 0x300) >> 8;
    for (const u32 target : {0U, 2U}) {
        INFO("target " << target);
        const u32 at = target * 4;
        CHECK(reg(latte::reg::cb_color0_base + at) == base);
        CHECK(reg(latte::reg::cb_color0_size + at) == registers->size);
        CHECK(reg(latte::reg::cb_color0_info + at) == registers->info);
        CHECK(reg(latte::reg::cb_color0_view + at) == registers->view);
        CHECK(reg(latte::reg::cb_color0_mask + at) == 0);
        CHECK(reg(latte::reg::cb_color0_tile + at) == 0); // single-sampled: no CMASK
    }
    CHECK(reg(latte::reg::cb_color0_base + 4) == 0); // target 1 untouched
}

TEST_CASE_METHOD(StateFixture,
                 "GX2SetDepthBuffer writes the depth buffer's registers and clear values",
                 "[hle][gx2][state]") {
    gx2::DepthBuffer buffer{};
    buffer.surface.dim = 1;
    buffer.surface.width = 1280;
    buffer.surface.height = 720;
    buffer.surface.depth = 1;
    buffer.surface.mip_levels = 1;
    buffer.surface.format = 0x80E;
    buffer.surface.use = gx2::surface_use_depth_buffer;
    buffer.view_num_slices = 1;
    buffer.depth_clear = 1.0F;
    buffer.stencil_clear = 0x40;
    REQUIRE(gx2::calc_surface_size_and_alignment(buffer.surface));
    buffer.surface.image = kImage;
    buffer.hi_z = kHiZ;
    const auto registers = gx2::depth_buffer_registers(buffer);
    REQUIRE(registers.has_value());
    buffer.db_depth_size = registers->size;
    buffer.db_depth_view = registers->view;
    buffer.db_depth_info = registers->info;
    buffer.db_htile_surface = registers->htile_surface;
    buffer.db_prefetch_limit = registers->prefetch_limit;
    buffer.db_preload_control = registers->preload_control;
    buffer.pa_poly_offset_control = registers->poly_offset_control;
    put(kBuffer, buffer);

    call("GX2SetDepthBuffer", {kBuffer});
    run_gpu();

    CHECK(reg(latte::reg::db_depth_size) == registers->size);
    CHECK(reg(latte::reg::db_depth_view) == registers->view);
    CHECK(reg(latte::reg::db_depth_base) == physical(kImage) >> 8);
    CHECK(reg(latte::reg::db_depth_info) == registers->info);
    CHECK(reg(latte::reg::db_htile_data_base) == physical(kHiZ) >> 8);
    CHECK(reg(latte::reg::db_htile_surface) == registers->htile_surface);
    CHECK(reg(latte::reg::db_prefetch_limit) == registers->prefetch_limit);
    CHECK(reg(latte::reg::db_preload_control) == registers->preload_control);
    CHECK(reg(latte::reg::pa_su_poly_offset_db_fmt_cntl) == registers->poly_offset_control);
    CHECK(reg(latte::reg::db_stencil_clear) == 0x40);
    CHECK(reg(latte::reg::db_depth_clear) == bits(1.0F));
}

TEST_CASE_METHOD(StateFixture, "a render target past the eighth is refused", "[hle][gx2][state]") {
    call("GX2SetColorBuffer", {kBuffer, 8});
    run_gpu();
    CHECK(os.gpu.processor().stats().packets == 0);
}
