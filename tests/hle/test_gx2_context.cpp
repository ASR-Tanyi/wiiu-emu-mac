// Phase 8.1 — GX2's context states.
//
// What is guarded: the binding packets, against decaf's and Cemu's tables
// and the block's layout; that a new state is cleared, named and given its
// binding display list; and the point of it all -- state written while a
// block is bound lands in it, and binding the block again brings it back,
// through the GPU, when the submissions retire.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_context.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/viewport.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <memory>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using latte::pm4::Opcode;

namespace {

[[nodiscard]] u32 bits(f32 value) {
    return std::bit_cast<u32>(value);
}

/// Every packet in a stream.
[[nodiscard]] std::vector<latte::pm4::Packet> packets(std::span<const u32> words) {
    std::vector<latte::pm4::Packet> out;
    latte::pm4::Reader reader{words};
    while (const std::optional<latte::pm4::Packet> next = reader.next()) {
        out.push_back(*next);
    }
    REQUIRE_FALSE(reader.failed());
    return out;
}

} // namespace

TEST_CASE("binding a context state is CONTEXT_CONTROL and six LOADs", "[hle][gx2][context]") {
    constexpr physaddr_t state = 0x00100000;
    const std::vector<u32> words = gx2::bind_context_state_packets(state, true);
    const std::vector<latte::pm4::Packet> found = packets(words);
    REQUIRE(found.size() == 7);

    CHECK(found[0].header.opcode == static_cast<u8>(Opcode::ContextControl));
    CHECK(found[0].payload[0] == 0x80000077);
    CHECK(found[0].payload[1] == 0x80000077);

    const std::array<Opcode, 6> order{Opcode::LoadConfigReg,
                                      Opcode::LoadContextReg,
                                      Opcode::LoadAluConst,
                                      Opcode::LoadLoopConst,
                                      Opcode::LoadResource,
                                      Opcode::LoadSampler};
    const std::array<u32, 6> offsets{0x0, 0x2C00, 0x3C00, 0x5C00, 0x5E00, 0x9500};
    const std::array<std::size_t, 6> ranges{15, 45, 1, 1, 9, 3}; // decaf's and Cemu's tables
    for (std::size_t i = 0; i < 6; ++i) {
        const latte::pm4::Packet& load = found[i + 1];
        CHECK(load.header.opcode == static_cast<u8>(order[i]));
        CHECK(load.payload[0] == state + offsets[i]);
        CHECK(load.payload[1] == (i == 0 ? 0x80000000U : 0U));
        CHECK(load.payload.size() == 2 + (ranges[i] * 2));
    }
    // Two of them, word for word: the first config range, the last context
    // one, and the single ALU constant range.
    CHECK(found[1].payload[2] == 0x300);
    CHECK(found[1].payload[3] == 6);
    CHECK(found[2].payload[found[2].payload.size() - 2] == 0x284);
    CHECK(found[2].payload[found[2].payload.size() - 1] == 0xC);
    CHECK(found[3].payload[3] == 0x800);

    // 169 words: what fits in the block's 192-word display list.
    CHECK(words.size() == 169);
    CHECK(words.size() <= gx2::context_state_list_words);

    // Naming without loading: one empty range each.
    const std::vector<u32> naming = gx2::bind_context_state_packets(state, false);
    CHECK(naming.size() == 3 + (6 * 5));
    const std::vector<latte::pm4::Packet> named = packets(naming);
    for (std::size_t i = 1; i < named.size(); ++i) {
        const latte::pm4::Packet& load = named[i];
        CHECK(load.payload.size() == 4);
        CHECK(load.payload[2] == 0);
        CHECK(load.payload[3] == 0);
    }

    const std::vector<u32> off = gx2::unbind_context_state_packets();
    CHECK(off == std::vector<u32>{
                     latte::pm4::type3_header(Opcode::ContextControl, 2), 0x80000000, 0x80000000});
}

TEST_CASE("each shadow area fits its window", "[hle][gx2][context]") {
    for (const gx2::ShadowArea& area : gx2::shadow_areas()) {
        const std::optional<latte::Window> window =
            latte::load_window(static_cast<Opcode>(area.opcode));
        REQUIRE(window.has_value());
        CHECK(area.registers <= latte::window_range(*window).words());
        for (const gx2::LoadRange& range : area.ranges) {
            CHECK(range.offset + range.count <= area.registers);
        }
    }
}

// --------------------------------------------------------------------------
// Through GX2
// --------------------------------------------------------------------------

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kOut = 0x10004000;
constexpr ppcaddr_t kStateA = 0x10200000;
constexpr ppcaddr_t kStateB = 0x10210000;

/// PA_CL_VPORT_XSCALE_0's word in a block's context image.
constexpr u32 kXScaleInBlock = 0x2C00 + (latte::reg::pa_cl_vport_xscale_0 - 0x28000);

struct ContextFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    ContextFixture() {
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

    void viewport(f32 width) { call("GX2SetViewport", {}, {0, 0, width, 720, 0, 1}); }

    [[nodiscard]] u32 xscale() const {
        return os.gpu.registers().read(latte::reg::pa_cl_vport_xscale_0);
    }

    [[nodiscard]] u32 physical(ppcaddr_t address) const {
        const memory::Region* const region = memory.region_of(address);
        REQUIRE(region != nullptr);
        physaddr_t out = 0;
        REQUIRE(memory::to_physical(*region, address, out));
        return out;
    }
};

} // namespace

TEST_CASE_METHOD(ContextFixture,
                 "GX2SetupContextStateEx clears the block and records its binding",
                 "[hle][gx2][context]") {
    REQUIRE(memory.write_bytes(kStateA, std::vector<u8>(gx2::context_state_bytes, 0xAB)));
    call("GX2SetupContextStateEx", {kStateA, 0});

    CHECK(memory.read<u32>(kStateA) == 0);
    CHECK(memory.read<u32>(kStateA + 0x97FC) == 0);
    CHECK(memory.read<u32>(kStateA + 0x9808) == 0);
    CHECK(memory.read<u32>(kStateA + gx2::context_state_profiling) == 0);

    // The binding, padded to eight words like any display list.
    const std::vector<u32> expected = gx2::bind_context_state_packets(physical(kStateA), true);
    const u32 bytes = memory.read<u32>(kStateA + gx2::context_state_list_size);
    CHECK(bytes == 176 * 4);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(memory.read<u32>(kStateA + gx2::context_state_list + static_cast<u32>(i * 4)) ==
              expected[i]);
    }
    CHECK(memory.read<u32>(kStateA + gx2::context_state_list + (175 * 4)) ==
          latte::pm4::type2_filler);
    CHECK(os.gpu.display_list(state.core_id) == nullptr); // closed again

    call("GX2GetContextStateDisplayList", {kStateA, kOut, kOut + 4});
    CHECK(memory.read<u32>(kOut) == kStateA + gx2::context_state_list);
    CHECK(memory.read<u32>(kOut + 4) == bytes);

    // The flags: profiling is recorded; without the list, there is none.
    call("GX2SetupContextStateEx", {kStateB, 3});
    CHECK(memory.read<u32>(kStateB + gx2::context_state_profiling) == 1);
    CHECK(memory.read<u32>(kStateB + gx2::context_state_list_size) == 0);
    CHECK(memory.read<u32>(kStateB + gx2::context_state_list) == 0);
}

TEST_CASE_METHOD(ContextFixture,
                 "state written after setting a block up lands in the block",
                 "[hle][gx2][context]") {
    call("GX2SetupContextStateEx", {kStateA, 0});
    viewport(1280);
    CHECK(memory.read<u32>(kStateA + kXScaleInBlock) == 0); // not before the GPU gets there

    run_gpu();
    CHECK(xscale() == bits(640.0F));
    CHECK(memory.read<u32>(kStateA + kXScaleInBlock) == bits(640.0F));
    CHECK(os.gpu.processor().shadow_base(latte::Window::Context) ==
          std::optional<physaddr_t>{physical(kStateA) + 0x2C00});
}

TEST_CASE_METHOD(ContextFixture,
                 "binding a context state brings back what was written into it",
                 "[hle][gx2][context]") {
    call("GX2SetupContextStateEx", {kStateA, 0});
    viewport(1280);
    call("GX2SetupContextStateEx", {kStateB, 0});
    viewport(640);
    run_gpu();
    CHECK(xscale() == bits(320.0F));

    call("GX2SetContextState", {kStateA});
    run_gpu();
    CHECK(xscale() == bits(640.0F));

    // Bound again, A is where writes go; B keeps its own.
    viewport(100);
    call("GX2SetContextState", {kStateB});
    run_gpu();
    CHECK(xscale() == bits(320.0F));
    CHECK(memory.read<u32>(kStateA + kXScaleInBlock) == bits(50.0F));

    call("GX2SetContextState", {kStateA});
    run_gpu();
    CHECK(xscale() == bits(50.0F));
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(ContextFixture,
                 "a state set up without its display list binds the long way",
                 "[hle][gx2][context]") {
    call("GX2SetupContextStateEx", {kStateA, gx2::context_state_no_display_list_flag});
    viewport(1280);
    call("GX2SetupContextStateEx", {kStateB, gx2::context_state_no_display_list_flag});
    viewport(640);
    call("GX2SetContextState", {kStateA});
    run_gpu();
    CHECK(xscale() == bits(640.0F));
    CHECK(os.gpu.processor().stats().indirect_buffers == 0);
}

TEST_CASE_METHOD(ContextFixture,
                 "GX2SetContextState(NULL) stops writes reaching the block",
                 "[hle][gx2][context]") {
    call("GX2SetupContextStateEx", {kStateA, 0});
    viewport(1280);
    call("GX2SetContextState", {0});
    viewport(200);
    run_gpu();
    CHECK(xscale() == bits(100.0F));
    CHECK(memory.read<u32>(kStateA + kXScaleInBlock) == bits(640.0F));
    CHECK(os.gpu.processor().shadow_enable() == gx2::shadowing_off);
}

// --------------------------------------------------------------------------
// The default state
// --------------------------------------------------------------------------

namespace {

/// A register file after running `words`, checked to have run cleanly.
struct Ran {
    std::unique_ptr<latte::RegisterFile> registers = std::make_unique<latte::RegisterFile>();
    latte::CommandProcessor processor{*registers};

    explicit Ran(std::span<const u32> words) {
        REQUIRE(processor.execute(words));
        CHECK(processor.stats().errors == 0);
        CHECK(processor.stats().unknown == 0);
        CHECK(processor.stats().deferred == 0);
    }

    [[nodiscard]] u32 operator[](u32 address) const { return registers->read(address); }
};

} // namespace

TEST_CASE("GX2's register initialisation is decaf's, at Linux's addresses", "[hle][gx2][context]") {
    const std::vector<u32> words = gx2::initial_register_packets();
    CHECK(packets(words).size() == 45);
    const Ran ran{words};

    CHECK(ran[0x28034] == 0x20002000);          // PA_SC_SCREEN_SCISSOR_BR: 8192, 8192
    CHECK(ran[0x28C08] == 0x29);                // PA_SU_VTX_CNTL
    CHECK(ran[0x28818] == 0x43F);               // PA_CL_VTE_CNTL
    CHECK(ran[0x28204] == 0x80000000);          // PA_SC_WINDOW_SCISSOR_TL
    CHECK(ran[0x28A4C] == 0x14001);             // PA_SC_MODE_CNTL, where its name says
    CHECK(ran[0x28A10] == 1);                   // VGT_OUTPUT_PATH_CNTL, not the mode control
    CHECK(ran[0x8B24] == 0xFF3FFF);             // PA_SC_FORCE_EOV_MAX_CNTS, a config register
    CHECK(ran[0x28A94] == 1);                   // VGT_MULTI_PRIM_IB_RESET_EN
    CHECK(ran[0x28400] == 0xFFFFFFFF);          // VGT_MAX_VTX_INDX
    CHECK(ran[0x9508] == 0x07000002);           // TA_CNTL_AUX
    CHECK(ran[0x286D4] == 0x4869);              // SPI_INTERP_CONTROL_0
    CHECK(ran[0x286C8] == 1);                   // SPI_THREAD_GROUPING, by SET_ALL_CONTEXTS
    CHECK(ran[0x28C30] == 0x01000000);          // CB_CLRCMP_CONTROL
    CHECK(ran[0x28C3C] == 0xFFFFFFFF);          // CB_CLRCMP_MSK
    CHECK(ran[0x28C48] == 0xFFFFFFFF);          // PA_SC_AA_MASK
    CHECK(ran[0x28230] == 0xAAAAAAAA);          // PA_SC_EDGERULE
    CHECK(ran.processor.stats().packets == 45); // SQ_VTX_START_INST_LOC's SET_CTL_CONST among them
}

TEST_CASE("the default state is decaf's sequence of GX2's calls", "[hle][gx2][context]") {
    const std::vector<u32> words = gx2::default_state_packets();
    // Two each for the shader mode, the stencil masks, the polygon offset and
    // the alpha test; eight blend controls; one each for the other fifteen.
    CHECK(packets(words).size() == 31);
    const Ran ran{words};

    CHECK(ran[0x28A40] == 0);          // VGT_GS_MODE off
    CHECK(ran[0x8C00] == 0x1B00000C);  // SQ_CONFIG: priorities 3, 2, 1, 0; DX9 constants
    CHECK(ran[0x8C04] == 0x403000C8);  // 200 pixel and 48 vertex GPRs, 4 clause temporaries
    CHECK(ran[0x8C0C] == 0x04043088);  // SQ_THREAD_RESOURCE_MGMT
    CHECK(ran[0x8C10] == 0x004000C0);  // 192 and 64 stack entries
    CHECK(ran[0x28430] == 0x00FFFF01); // DB_STENCILREFMASK: ref 1, masks 0xFF
    CHECK(ran[0x28A00] == 0x00080008); // PA_SU_POINT_SIZE: 1 x 1
    CHECK(ran[0x28A08] == 8);          // PA_SU_LINE_CNTL: 1
    CHECK(ran[0x28808] == 0x00CC0000); // CB_COLOR_CONTROL: copy, writes on
    CHECK(ran[0x28238] == 0xFFFFFFFF); // CB_TARGET_MASK
    for (u32 target = 0; target < 8; ++target) {
        CHECK(ran[0x28780 + (4 * target)] == 0x25040504); // source alpha over, both
    }
    CHECK(ran[0x28810] == 0x01000000); // PA_CL_CLIP_CNTL
    CHECK(ran[0x28A18] == bits(1.0F)); // tessellation levels
    CHECK(ran[0x28A1C] == bits(1.0F));
    CHECK(ran[0x28D0C] == 0); // DB_RENDER_CONTROL
}

TEST_CASE("a binding restores the shader mode: the config ranges are its registers",
          "[hle][gx2][context]") {
    // The first config range, 0x300 for 6, is SQ_CONFIG to
    // SQ_STACK_RESOURCE_MGMT_2: what GX2SetShaderModeEx writes.
    const gx2::ShadowArea& config = gx2::shadow_areas()[0];
    CHECK(config.ranges[0].offset == (0x8C00 - 0x8000) / 4);
    CHECK(config.ranges[0].count == 6);
}

TEST_CASE("the shader modes divide the shader core as decaf and Cemu do", "[hle][gx2][context]") {
    const gx2::ShaderResources resources{.vs_gprs = 40,
                                         .vs_stack = 32,
                                         .gs_gprs = 30,
                                         .gs_stack = 16,
                                         .ps_gprs = 100,
                                         .ps_stack = 48};

    const gx2::ShaderModeRegisters block =
        gx2::shader_mode_registers(gx2::ShaderMode::UniformBlock, resources);
    CHECK(block.sq_config == 0x1B000008); // no DX9 constants
    CHECK(block.sq_gpr_resource_mgmt_1 == (100U | (40U << 16) | (4U << 28)));
    CHECK(block.sq_stack_resource_mgmt_1 == (48U | (32U << 16)));

    // Geometry: the vertex shader is the ES stage; the VS runs the copy shader.
    const gx2::ShaderModeRegisters geometry =
        gx2::shader_mode_registers(gx2::ShaderMode::GeometryShader, resources);
    CHECK(geometry.sq_gpr_resource_mgmt_1 == (100U | (64U << 16) | (4U << 28)));
    CHECK(geometry.sq_gpr_resource_mgmt_2 == (30U | (40U << 16)));
    CHECK(geometry.sq_thread_resource_mgmt == 0x1C08207C);
    CHECK(geometry.sq_stack_resource_mgmt_2 == (16U | (32U << 16)));

    const gx2::ShaderModeRegisters compute =
        gx2::shader_mode_registers(gx2::ShaderMode::ComputeShader, resources);
    CHECK(compute.sq_config == 0xE4000008); // priorities reversed
    CHECK(compute.sq_gpr_resource_mgmt_2 == (248U << 16));
    CHECK(compute.sq_thread_resource_mgmt == 0xBD010101);
    CHECK(compute.sq_stack_resource_mgmt_2 == (256U << 16));

    // Geometry mode leaves VGT_GS_MODE to GX2SetGeometryShader; compute
    // clears the rings and stream out as well.
    std::vector<u32> words;
    gx2::append_shader_mode(words, gx2::ShaderMode::GeometryShader, resources);
    CHECK(packets(words).size() == 1);
    words.clear();
    gx2::append_shader_mode(words, gx2::ShaderMode::ComputeShader, resources);
    CHECK(packets(words).size() == 5);
    const Ran ran{words};
    CHECK(ran[0x28A40] == (3U | (3U << 14) | (1U << 17)));
    CHECK(ran[0x8C44] == 0xFFFFFF); // SQ_ESGS_RING_SIZE
    CHECK(ran[0x288AC] == 1);       // SQ_GSVS_RING_ITEMSIZE
}

TEST_CASE_METHOD(ContextFixture,
                 "a new context state holds the default state, and gives it back",
                 "[hle][gx2][context]") {
    call("GX2SetupContextStateEx", {kStateA, 0});
    run_gpu();
    // In the block: the config image's shader mode, the context image's masks.
    CHECK(memory.read<u32>(kStateA + (0x8C00 - 0x8000)) == 0x1B00000C);
    CHECK(memory.read<u32>(kStateA + 0x2C00 + (0x28238 - 0x28000)) == 0xFFFFFFFF);

    // A different shader mode and channel mask, then A again.
    call("GX2SetShaderModeEx", {1, 10, 20, 0, 0, 30, 40});
    call("GX2SetTargetChannelMasks", {1, 0, 0, 0, 0, 0, 0, 0});
    call("GX2SetupContextStateEx", {kStateB, 0});
    run_gpu();
    CHECK(os.gpu.registers().read(0x8C00) == 0x1B00000C); // B's default
    call("GX2SetContextState", {kStateA});
    run_gpu();
    CHECK(os.gpu.registers().read(0x8C00) == 0x1B000008); // A's: uniform blocks
    CHECK(os.gpu.registers().read(0x8C04) == (30U | (10U << 16) | (4U << 28)));
    CHECK(os.gpu.registers().read(0x28238) == 1);
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(ContextFixture,
                 "GX2SetDefaultState, stream out and the tessellation levels reach the GPU",
                 "[hle][gx2][context]") {
    call("GX2SetStreamOutEnable", {1});
    call("GX2SetMaxTessellationLevel", {}, {8});
    call("GX2SetMinTessellationLevel", {}, {2});
    call("GX2SetShaderModeEx", {4, 1, 1, 1, 1, 1, 1}); // no such mode: refused
    run_gpu();
    CHECK(os.gpu.registers().read(0x28AB0) == 1);
    CHECK(os.gpu.registers().read(0x28A18) == bits(8.0F));
    CHECK(os.gpu.registers().read(0x28A1C) == bits(2.0F));
    CHECK(os.gpu.registers().read(0x8C00) == 0x1B00000C); // still GX2Init's

    call("GX2SetDefaultState", {});
    run_gpu();
    CHECK(os.gpu.registers().read(0x28AB0) == 0);
    CHECK(os.gpu.registers().read(0x28A18) == bits(1.0F));
}
