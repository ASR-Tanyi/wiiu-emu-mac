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
#include <wiiu/latte/render_state.hpp>
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
constexpr ppcaddr_t kStack = 0x10008000;

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

    /// Call a GX2 function as a title would: the first eight integer
    /// arguments in r3-r10, the rest in the caller's parameter area at
    /// 8(r1).
    void call(std::string_view name, std::vector<u32> arguments, std::vector<f32> floats = {}) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        const u32 saved_sp = state.gpr[1];
        state.gpr[1] = kStack;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            if (i < 8) {
                state.gpr[3 + i] = arguments[i];
            } else {
                REQUIRE(memory.try_write<u32>(kStack + 8 + static_cast<u32>((i - 8) * 4),
                                              arguments[i]));
            }
        }
        for (std::size_t i = 0; i < floats.size(); ++i) {
            state.fpr[1 + i].ps0 = static_cast<f64>(floats[i]);
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        state.gpr[1] = saved_sp;
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
    run_gpu(); // GX2Init's own state
    const u64 before = os.gpu.processor().stats().packets;
    call("GX2SetColorBuffer", {kBuffer, 8});
    run_gpu();
    CHECK(os.gpu.processor().stats().packets == before);
}

// --------------------------------------------------------------------------
// Depth, stencil and the rasterizer
// --------------------------------------------------------------------------

TEST_CASE("depth and stencil control pack where Linux's r600d.h puts the fields",
          "[hle][gx2][state]") {
    // Depth alone: Z_ENABLE bit 1, Z_WRITE_ENABLE bit 2, ZFUNC bits 4-6.
    CHECK(gx2::depth_stencil_control(
              {.depth_test = true, .depth_write = true, .depth_compare = 1}) == 0x16);

    // Every stencil field a different value, so a swapped pair shows.
    const gx2::DepthStencil all{.depth_test = true,
                                .depth_write = false,
                                .depth_compare = 6,
                                .stencil_test = true,
                                .backface_stencil = true,
                                .front_compare = 3,
                                .front_depth_pass = 2,
                                .front_depth_fail = 4,
                                .front_stencil_fail = 1,
                                .back_compare = 5,
                                .back_depth_pass = 7,
                                .back_depth_fail = 3,
                                .back_stencil_fail = 6};
    const u32 expected = (1U << 0) | (1U << 1) | (6U << 4) | (1U << 7) | (3U << 8) | (1U << 11) |
                         (2U << 14) | (4U << 17) | (5U << 20) | (6U << 23) | (7U << 26) |
                         (3U << 29);
    CHECK(gx2::depth_stencil_control(all) == expected);
    const latte::DbDepthControl decoded{.raw = expected};
    CHECK(decoded.stencil_zfail_bf == latte::StencilOp::IncrementClamp);
    CHECK(decoded.z_func == latte::CompareFunction::GreaterEqual);

    // Masks and references: a byte each, ref low.
    const gx2::StencilMaskReg masks = gx2::stencil_mask_registers(0xFF, 0x0F, 0x1FF, 0xF0, 0x3C, 2);
    CHECK(masks.front == (0xFFU | (0xFFU << 8) | (0x0FU << 16)));
    CHECK(masks.back == (0x02U | (0xF0U << 8) | (0x3CU << 16)));
}

TEST_CASE("polygon control, offset, sizes and clipping pack as decaf and Cemu do",
          "[hle][gx2][state]") {
    const u32 polygon = gx2::polygon_control({.front_face = 1,
                                              .cull_front = false,
                                              .cull_back = true,
                                              .poly_mode = 1,
                                              .front_mode = 2,
                                              .back_mode = 1,
                                              .offset_front = true,
                                              .offset_back = false,
                                              .offset_para = true});
    CHECK(polygon ==
          ((1U << 1) | (1U << 2) | (1U << 3) | (2U << 5) | (1U << 8) | (1U << 11) | (1U << 13)));

    const gx2::PolygonOffsetReg offset = gx2::polygon_offset_registers(1, 2, 3, 4, 5);
    CHECK(offset.front_scale == 32.0F); // scales times 16
    CHECK(offset.front_offset == 1.0F);
    CHECK(offset.back_scale == 64.0F);
    CHECK(offset.back_offset == 3.0F);
    CHECK(offset.clamp == 5.0F);

    CHECK(gx2::size_eighths(1.0F) == 8);
    CHECK(gx2::size_eighths(0.1F) == 0); // 0.8, truncated
    CHECK(gx2::size_eighths(2.06F) == 16);
    CHECK(gx2::size_eighths(-1.0F) == 0);
    CHECK(gx2::size_eighths(std::bit_cast<f32>(0x7FC00000U)) == 0); // NaN
    CHECK(gx2::size_eighths(10000.0F) == 0xFFFF);                   // Cemu's clamp
    CHECK(gx2::point_size_register(2, 1) == ((16U << 16) | 8U));    // width high, height low
    CHECK(gx2::point_limits_register(1, 4) == ((32U << 16) | 8U));  // max high, min low
    CHECK(gx2::line_width_register(1.5F) == 12);

    // Everything on: only Cemu's linear attribute clipping bit.
    CHECK(gx2::clip_control_register(true, true, false) == (1U << 24));
    // Everything off, half Z: the rasterizer killed, near and far clipping off.
    CHECK(gx2::clip_control_register(false, false, true) ==
          ((1U << 19) | (1U << 22) | (1U << 24) | (1U << 26) | (1U << 27)));
}

TEST_CASE_METHOD(StateFixture,
                 "GX2SetDepthStencilControl reads its last five arguments from the stack",
                 "[hle][gx2][state]") {
    call("GX2SetDepthStencilControl", {1, 0, 6, 1, 1, 3, 2, 4, 1, 5, 7, 3, 6});
    run_gpu();
    CHECK(reg(latte::reg::db_depth_control) ==
          gx2::depth_stencil_control({.depth_test = true,
                                      .depth_write = false,
                                      .depth_compare = 6,
                                      .stencil_test = true,
                                      .backface_stencil = true,
                                      .front_compare = 3,
                                      .front_depth_pass = 2,
                                      .front_depth_fail = 4,
                                      .front_stencil_fail = 1,
                                      .back_compare = 5,
                                      .back_depth_pass = 7,
                                      .back_depth_fail = 3,
                                      .back_stencil_fail = 6}));

    // The Init/Set halves: fourteen arguments, six of them on the stack.
    call("GX2InitDepthStencilControlReg", {kReg, 1, 1, 1, 0, 0, 0, 0, 0, 0, 7, 2, 2, 2});
    CHECK(memory.read<u32>(kReg) == (0x16U | (7U << 20) | (2U << 23) | (2U << 26) | (2U << 29)));
    call("GX2SetDepthStencilControlReg", {kReg});
    run_gpu();
    CHECK(reg(latte::reg::db_depth_control) == memory.read<u32>(kReg));

    call("GX2SetDepthOnlyControl", {1, 0, 3});
    run_gpu();
    CHECK(reg(latte::reg::db_depth_control) == ((1U << 1) | (3U << 4)));

    call("GX2SetStencilMask", {0xFF, 0x0F, 1, 0xF0, 0x3C, 2});
    run_gpu();
    CHECK(reg(latte::reg::db_stencilrefmask) == (0x01U | (0xFFU << 8) | (0x0FU << 16)));
    CHECK(reg(latte::reg::db_stencilrefmask_bf) == (0x02U | (0xF0U << 8) | (0x3CU << 16)));
}

TEST_CASE_METHOD(StateFixture, "the rasterizer calls reach their registers", "[hle][gx2][state]") {
    // Nine arguments: the polygon-offset-para flag is on the stack.
    call("GX2SetPolygonControl", {1, 0, 1, 1, 2, 1, 1, 0, 1});
    call("GX2SetPolygonOffset", {}, {1, 2, 3, 4, 5});
    call("GX2SetPointSize", {}, {2, 1});
    call("GX2SetPointLimits", {}, {1, 4});
    call("GX2SetLineWidth", {}, {1.5F});
    call("GX2SetRasterizerClipControlEx", {0, 0, 1});
    call("GX2SetPrimitiveRestartIndex", {0xFFFF});
    run_gpu();

    CHECK(reg(latte::reg::pa_su_sc_mode_cntl) ==
          ((1U << 1) | (1U << 2) | (1U << 3) | (2U << 5) | (1U << 8) | (1U << 11) | (1U << 13)));
    CHECK(reg(latte::reg::pa_su_poly_offset_front_scale) == bits(32.0F));
    CHECK(reg(latte::reg::pa_su_poly_offset_front_scale + 4) == bits(1.0F));  // front offset
    CHECK(reg(latte::reg::pa_su_poly_offset_front_scale + 8) == bits(64.0F)); // back scale
    CHECK(reg(latte::reg::pa_su_poly_offset_front_scale + 12) == bits(3.0F)); // back offset
    CHECK(reg(latte::reg::pa_su_poly_offset_clamp) == bits(5.0F));
    CHECK(reg(latte::reg::pa_su_point_size) == ((16U << 16) | 8U));
    CHECK(reg(latte::reg::pa_su_point_minmax) == ((32U << 16) | 8U));
    CHECK(reg(latte::reg::pa_su_line_cntl) == 12);
    CHECK(reg(latte::reg::pa_cl_clip_cntl) ==
          ((1U << 19) | (1U << 22) | (1U << 24) | (1U << 26) | (1U << 27)));
    CHECK(reg(latte::reg::vgt_multi_prim_ib_reset_indx) == 0xFFFF);

    call("GX2SetCullOnlyControl", {0, 1, 0});
    call("GX2SetRasterizerClipControl", {1, 1});
    run_gpu();
    CHECK(reg(latte::reg::pa_su_sc_mode_cntl) == 1U); // cull front, nothing else
    CHECK(reg(latte::reg::pa_cl_clip_cntl) == (1U << 24));
}

TEST_CASE_METHOD(StateFixture,
                 "each Init/Set pair writes what the one call does",
                 "[hle][gx2][state]") {
    call("GX2InitPolygonControlReg", {kReg, 1, 1, 0, 0, 0, 0, 0, 0, 1});
    call("GX2SetPolygonControlReg", {kReg});
    call("GX2InitPolygonOffsetReg", {kReg + 0x10}, {0.5F, 0.25F, 0, 0, 1});
    call("GX2SetPolygonOffsetReg", {kReg + 0x10});
    call("GX2InitStencilMaskReg", {kReg + 0x40, 1, 2, 3, 4, 5, 6});
    call("GX2SetStencilMaskReg", {kReg + 0x40});
    call("GX2InitPointSizeReg", {kReg + 0x50}, {3, 3});
    call("GX2SetPointSizeReg", {kReg + 0x50});
    call("GX2InitPointLimitsReg", {kReg + 0x54}, {0.5F, 8});
    call("GX2SetPointLimitsReg", {kReg + 0x54});
    call("GX2InitLineWidthReg", {kReg + 0x58}, {2});
    call("GX2SetLineWidthReg", {kReg + 0x58});
    run_gpu();

    CHECK(memory.read<u32>(kReg) == ((1U << 0) | (1U << 2) | (1U << 13)));
    CHECK(reg(latte::reg::pa_su_sc_mode_cntl) == memory.read<u32>(kReg));
    CHECK(memory.read<u32>(kReg + 0x10) == bits(4.0F)); // front scale first
    CHECK(reg(latte::reg::pa_su_poly_offset_front_scale) == bits(4.0F));
    CHECK(reg(latte::reg::pa_su_poly_offset_front_scale + 4) == bits(0.5F));
    CHECK(reg(latte::reg::pa_su_poly_offset_clamp) == bits(1.0F));
    CHECK(reg(latte::reg::db_stencilrefmask) == (3U | (1U << 8) | (2U << 16)));
    CHECK(reg(latte::reg::db_stencilrefmask_bf) == (6U | (4U << 8) | (5U << 16)));
    CHECK(reg(latte::reg::pa_su_point_size) == ((24U << 16) | 24U));
    CHECK(reg(latte::reg::pa_su_point_minmax) == ((64U << 16) | 4U));
    CHECK(reg(latte::reg::pa_su_line_cntl) == 16);
}

// --------------------------------------------------------------------------
// Colour
// --------------------------------------------------------------------------

TEST_CASE("colour, blend and mask state pack as decaf and Cemu do", "[hle][gx2][state]") {
    // GX2SetDefaultState's: copy, no blending, colour writes on.
    CHECK(gx2::color_control_register(0xCC, 0, false, true) == (0xCCU << 16));
    // Writes off is SPECIAL_OP 1, bits 4-6 (Linux's r600d.h); the blend
    // mask is a bit per target from bit 8.
    CHECK(gx2::color_control_register(0x66, 0x81, true, false) ==
          ((0x66U << 16) | (0x81U << 8) | (1U << 4) | (1U << 1)));

    // Source alpha, one minus source alpha, add -- for colour and alpha both.
    const u32 blend = gx2::blend_control_register({.color_source = 4,
                                                   .color_destination = 5,
                                                   .color_combine = 0,
                                                   .separate_alpha = true,
                                                   .alpha_source = 1,
                                                   .alpha_destination = 0,
                                                   .alpha_combine = 4});
    CHECK(blend == (4U | (5U << 8) | (1U << 16) | (4U << 21) | (1U << 29)));
    const latte::CbBlendControl decoded{.raw = blend};
    CHECK(decoded.color_destblend == 5);
    CHECK(decoded.alpha_comb_fcn == 4);

    // A nibble a target, target 0 lowest (Linux's r600d.h).
    CHECK(gx2::target_channel_masks_register({0xF, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8}) ==
          0x8654321FU);

    const gx2::AlphaTestReg alpha = gx2::alpha_test_registers(true, 4, 0.5F);
    CHECK(alpha.sx_alpha_test_control == (4U | (1U << 3)));
    CHECK(alpha.sx_alpha_ref == 0.5F);

    // decaf's dither tables, read back as the byte its Get call matches.
    CHECK(gx2::alpha_to_mask_register(true, 0) == (1U | (0xAAU << 8)));
    CHECK(((gx2::alpha_to_mask_register(false, 1) >> 8) & 0xFF) == 0x78);
    CHECK(((gx2::alpha_to_mask_register(false, 2) >> 8) & 0xFF) == 0xC6);
    CHECK(((gx2::alpha_to_mask_register(false, 3) >> 8) & 0xFF) == 0x2D);
    CHECK(((gx2::alpha_to_mask_register(false, 4) >> 8) & 0xFF) == 0x93);
    CHECK(gx2::alpha_to_mask_register(false, 99) == (0xAAU << 8));

    CHECK(gx2::aa_mask_register(0x01, 0x02, 0x04, 0x108) == 0x08040201U);
}

TEST_CASE_METHOD(StateFixture, "the colour calls reach their registers", "[hle][gx2][state]") {
    call("GX2SetColorControl", {0xCC, 0x01, 0, 1});
    call("GX2SetBlendControl", {2, 4, 5, 0, 1, 1, 0, 4});
    call("GX2SetBlendConstantColor", {}, {0.25F, 0.5F, 0.75F, 1.0F});
    call("GX2SetTargetChannelMasks", {0xF, 0xF, 0, 0, 0, 0, 0, 0x7});
    call("GX2SetAlphaTest", {1, 4}, {0.5F});
    call("GX2SetAlphaToMask", {1, 2});
    call("GX2SetAAMask", {0xFF, 0xFF, 0xFF, 0xFF});
    run_gpu();

    CHECK(reg(latte::reg::cb_color_control) == ((0xCCU << 16) | (1U << 8)));
    // Target 2's, not 0's, which is still the default state's.
    CHECK(reg(latte::reg::cb_blend0_control) == gx2::blend_control_register({.color_source = 4,
                                                                             .color_destination = 5,
                                                                             .color_combine = 0,
                                                                             .separate_alpha = true,
                                                                             .alpha_source = 4,
                                                                             .alpha_destination = 5,
                                                                             .alpha_combine = 0}));
    CHECK(reg(latte::reg::cb_blend0_control + 8) ==
          (4U | (5U << 8) | (1U << 16) | (4U << 21) | (1U << 29)));
    CHECK(reg(latte::reg::cb_blend_red) == bits(0.25F));
    CHECK(reg(latte::reg::cb_blend_red + 12) == bits(1.0F)); // alpha
    CHECK(reg(latte::reg::cb_target_mask) == 0x700000FFU);
    CHECK(reg(latte::reg::sx_alpha_test_control) == (4U | (1U << 3)));
    CHECK(reg(latte::reg::sx_alpha_ref) == bits(0.5F));
    CHECK(reg(latte::reg::db_alpha_to_mask) == (1U | (0xC6U << 8)));
    CHECK(reg(latte::reg::pa_sc_aa_mask) == 0xFFFFFFFFU);

    // A target past the eighth is refused, and writes nothing.
    call("GX2SetBlendControl", {8, 1, 1, 0, 0, 0, 0, 0});
    run_gpu();
    CHECK(reg(latte::reg::cb_blend0_control + 32) == 0);
}

TEST_CASE_METHOD(StateFixture,
                 "the colour Init/Set pairs write what the one call does",
                 "[hle][gx2][state]") {
    // Nine arguments each: the last on the stack.
    call("GX2InitBlendControlReg", {kReg, 7, 2, 3, 1, 0, 0, 0, 3});
    call("GX2InitTargetChannelMasksReg", {kReg + 0x10, 1, 2, 3, 4, 5, 6, 7, 8});
    call("GX2InitColorControlReg", {kReg + 0x20, 0x00, 0xFF, 0, 0});
    call("GX2InitBlendConstantColorReg", {kReg + 0x30}, {1, 0, 0, 0.5F});
    call("GX2InitAlphaTestReg", {kReg + 0x40, 1, 7}, {0.25F});
    call("GX2InitAlphaToMaskReg", {kReg + 0x50, 1, 3});
    call("GX2InitAAMaskReg", {kReg + 0x54, 1, 2, 3, 4});
    CHECK(memory.read<u32>(kReg) == 7); // the target first
    CHECK(memory.read<u32>(kReg + 4) ==
          (2U | (1U << 5) | (3U << 8) | (3U << 21))); // from the stack
    CHECK(memory.read<u32>(kReg + 0x10) == 0x87654321U);

    call("GX2SetBlendControlReg", {kReg});
    call("GX2SetTargetChannelMasksReg", {kReg + 0x10});
    call("GX2SetColorControlReg", {kReg + 0x20});
    call("GX2SetBlendConstantColorReg", {kReg + 0x30});
    call("GX2SetAlphaTestReg", {kReg + 0x40});
    call("GX2SetAlphaToMaskReg", {kReg + 0x50});
    call("GX2SetAAMaskReg", {kReg + 0x54});
    run_gpu();

    CHECK(reg(latte::reg::cb_blend0_control + 28) == (2U | (1U << 5) | (3U << 8) | (3U << 21)));
    CHECK(reg(latte::reg::cb_target_mask) == 0x87654321U);
    CHECK(reg(latte::reg::cb_color_control) == ((0xFFU << 8) | (1U << 4)));
    CHECK(reg(latte::reg::cb_blend_red) == bits(1.0F));
    CHECK(reg(latte::reg::cb_blend_red + 12) == bits(0.5F));
    CHECK(reg(latte::reg::sx_alpha_test_control) == (7U | (1U << 3)));
    CHECK(reg(latte::reg::sx_alpha_ref) == bits(0.25F));
    CHECK(reg(latte::reg::db_alpha_to_mask) == (1U | (0x2DU << 8)));
    CHECK(reg(latte::reg::pa_sc_aa_mask) == 0x04030201U);
}
