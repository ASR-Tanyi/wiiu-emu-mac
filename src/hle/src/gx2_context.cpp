// Phase 8.1 — GX2's context states. See gx2_context.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_context.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/gx2_state.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/color_buffer.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/render_state.hpp>

#include <array>
#include <bit>
#include <optional>

namespace wiiu::hle::gx2 {

namespace {

using latte::pm4::Opcode;

// The registers a binding loads, window by window. decaf's and Cemu's tables
// are the same, entry for entry and in the same order -- which is not
// address order, and is kept as found.

constexpr std::array<LoadRange, 15> config_ranges{{
    {0x300, 0x6},
    {0x900, 0x48},
    {0x980, 0x48},
    {0xA00, 0x48},
    {0x310, 0xC},
    {0x542, 0x1},
    {0x235, 0x1},
    {0x232, 0x2},
    {0x23A, 0x1},
    {0x256, 0x1},
    {0x60C, 0x1},
    {0x5C5, 0x1},
    {0x2C8, 0x1},
    {0x363, 0x1},
    {0x404, 0x2},
}};

constexpr std::array<LoadRange, 45> context_ranges{{
    {0x0, 0x2},   {0x3, 0x3},   {0xA, 0x4},   {0x10, 0x38},  {0x50, 0x34},  {0x8E, 0x4},
    {0x94, 0x40}, {0x100, 0x9}, {0x10C, 0x3}, {0x10F, 0x60}, {0x185, 0xA},  {0x191, 0x27},
    {0x1E0, 0x9}, {0x200, 0x1}, {0x202, 0x7}, {0xE0, 0x20},  {0x210, 0x29}, {0x250, 0x34},
    {0x290, 0x1}, {0x292, 0x2}, {0x2A1, 0x1}, {0x2A5, 0x1},  {0x2A8, 0x2},  {0x2AC, 0x3},
    {0x2CA, 0x1}, {0x2CC, 0x1}, {0x2CE, 0x1}, {0x300, 0x9},  {0x30C, 0x1},  {0x312, 0x1},
    {0x316, 0x2}, {0x343, 0x2}, {0x349, 0x3}, {0x34C, 0x2},  {0x351, 0x1},  {0x37E, 0x6},
    {0x2B4, 0x3}, {0x2B8, 0x3}, {0x2BC, 0x3}, {0x2C0, 0x3},  {0x2C8, 0x1},  {0x29B, 0x1},
    {0x8C, 0x1},  {0xD5, 0x1},  {0x284, 0xC},
}};

constexpr std::array<LoadRange, 1> alu_const_ranges{{{0, 0x800}}};

constexpr std::array<LoadRange, 1> loop_const_ranges{{{0, 0x60}}};

// Seven words a resource. Cemu names three of these runs: the pixel, vertex
// and geometry shaders' textures.
constexpr std::array<LoadRange, 9> resource_ranges{{
    {0x0, 0x70}, // pixel shader textures
    {0x380, 0x70},
    {0x460, 0x70}, // vertex shader textures
    {0x7E0, 0x70},
    {0x8B9, 0x7},
    {0x8C0, 0x70},
    {0x930, 0x70}, // geometry shader textures
    {0xCB0, 0x70},
    {0xD89, 0x7},
}};

// Three words a sampler, eighteen samplers a shader stage.
constexpr std::array<LoadRange, 3> sampler_ranges{{
    {0x0, 0x36},
    {0x36, 0x36},
    {0x6C, 0x36},
}};

constexpr std::array<ShadowArea, 6> areas{{
    {static_cast<u8>(Opcode::LoadConfigReg), 0x0000, 0xB00, config_ranges},
    {static_cast<u8>(Opcode::LoadContextReg), 0x2C00, 0x400, context_ranges},
    {static_cast<u8>(Opcode::LoadAluConst), 0x3C00, 0x800, alu_const_ranges},
    {static_cast<u8>(Opcode::LoadLoopConst), 0x5C00, 0x60, loop_const_ranges},
    {static_cast<u8>(Opcode::LoadResource), 0x5E00, 0xD9E, resource_ranges},
    {static_cast<u8>(Opcode::LoadSampler), 0x9500, 0xA2, sampler_ranges},
}};

/// Each image starts where the one before, rounded up to 256 bytes, ends,
/// and the last ends where GX2's own fields begin.
[[nodiscard]] constexpr bool areas_are_packed() noexcept {
    u32 next = 0;
    for (const ShadowArea& area : areas) {
        if (area.offset != next) {
            return false;
        }
        for (const LoadRange& range : area.ranges) {
            if (range.offset + range.count > area.registers) {
                return false;
            }
        }
        next = (area.offset + (area.registers * 4) + 0xFF) & ~0xFFU;
    }
    return next == context_state_shadow_bytes;
}

static_assert(areas_are_packed());

[[nodiscard]] coreinit::Context& active() {
    return *coreinit::detail::active_context();
}

/// `void GX2SetupContextStateEx(GX2ContextState *state, GX2ContextStateFlags
/// flags)`.
void gx2_setup_context_state_ex(Call& call, u32 state, u32 flags) {
    const std::optional<physaddr_t> physical =
        physical_address(call.memory, state, "GX2SetupContextStateEx's state");
    if (!physical.has_value()) {
        return;
    }
    const std::vector<u8> zeroes(context_state_bytes, 0);
    if (!call.memory.write_bytes(state, zeroes)) {
        WIIU_LOG_ERROR(
            Gx2, "GX2SetupContextStateEx: the state at {:#010x} is not writable memory", state);
        return;
    }
    const bool profiling = (flags & context_state_profiling_flag) != 0;
    static_cast<void>(
        call.memory.try_write<u32>(state + context_state_profiling, profiling ? 1U : 0U));

    // Name the block, so what is written from here on lands in it.
    static_cast<void>(emit(call, bind_context_state_packets(*physical, false)));

    // The new block's state: written now, with the block named, it is what
    // the block holds.
    static_cast<void>(emit(call, initial_register_packets()));
    static_cast<void>(emit(call, default_state_packets()));

    if ((flags & context_state_no_display_list_flag) != 0) {
        return;
    }
    Gpu& gpu = active().gpu;
    const u32 core = call.state.core_id;
    if (!gpu.begin_display_list(
            core, state + context_state_list, context_state_list_words * 4, profiling)) {
        // A list is already open on this core. Binding will load the block
        // the long way instead, which comes to the same thing.
        return;
    }
    static_cast<void>(emit(call, bind_context_state_packets(*physical, true)));
    const u32 bytes = gpu.end_display_list(core).value_or(0);
    static_cast<void>(call.memory.try_write<u32>(state + context_state_list_size, bytes));
}

/// `void GX2SetContextState(GX2ContextState *state)`: bind it, or with null,
/// turn shadowing off.
void gx2_set_context_state(Call& call, u32 state) {
    if (state == 0) {
        static_cast<void>(emit(call, unbind_context_state_packets()));
        return;
    }
    u32 bytes = 0;
    if (!call.memory.try_read<u32>(state + context_state_list_size, bytes)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetContextState: no state at {:#010x}", state);
        return;
    }
    if (bytes == 0) {
        // Set up without its display list: the same packets, written here.
        const std::optional<physaddr_t> physical =
            physical_address(call.memory, state, "GX2SetContextState's state");
        if (physical.has_value()) {
            static_cast<void>(emit(call, bind_context_state_packets(*physical, true)));
        }
        return;
    }
    static_cast<void>(emit_display_list_call(
        call, state + context_state_list, bytes, "GX2SetContextState's display list"));
}

/// `void GX2GetContextStateDisplayList(GX2ContextState *state, void
/// **outDisplayList, uint32_t *outSize)`.
void gx2_get_context_state_display_list(Call& call, u32 state, u32 out_list, u32 out_size) {
    u32 bytes = 0;
    if (!call.memory.try_read<u32>(state + context_state_list_size, bytes)) {
        WIIU_LOG_ERROR(Gx2, "GX2GetContextStateDisplayList: no state at {:#010x}", state);
        return;
    }
    if (out_list != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_list, state + context_state_list));
    }
    if (out_size != 0) {
        static_cast<void>(call.memory.try_write<u32>(out_size, bytes));
    }
}

} // namespace

std::span<const ShadowArea> shadow_areas() noexcept {
    return areas;
}

std::vector<u32> bind_context_state_packets(physaddr_t state, bool load) {
    std::vector<u32> words{
        latte::pm4::type3_header(Opcode::ContextControl, 2),
        shadowing_on,
        shadowing_on,
    };
    constexpr std::array<LoadRange, 1> nothing{{{0, 0}}};
    for (const ShadowArea& area : areas) {
        const std::span<const LoadRange> ranges =
            load ? area.ranges : std::span<const LoadRange>{nothing};
        words.push_back(latte::pm4::type3_header(static_cast<Opcode>(area.opcode),
                                                 static_cast<u32>(2 + (ranges.size() * 2))));
        words.push_back(state + area.offset);
        // Cemu's second word: bit 31 on the config load, which it reads as
        // wait-for-idle; decaf writes zero, as the address's high bits.
        words.push_back(area.opcode == static_cast<u8>(Opcode::LoadConfigReg) ? 0x80000000U : 0U);
        for (const LoadRange& range : ranges) {
            words.push_back(range.offset);
            words.push_back(range.count);
        }
    }
    return words;
}

std::vector<u32> unbind_context_state_packets() {
    return {
        latte::pm4::type3_header(Opcode::ContextControl, 2),
        shadowing_off,
        shadowing_off,
    };
}

namespace {

void gx2_set_default_state(Call& call) {
    static_cast<void>(emit(call, default_state_packets()));
}

/// A register write, for the table below.
struct Write {
    Opcode opcode;
    u32 address;
    std::vector<u32> values;
};

} // namespace

std::vector<u32> initial_register_packets() {
    constexpr Opcode context = Opcode::SetContextReg;
    constexpr Opcode config = Opcode::SetConfigReg;

    // A scissor corner at 8192, 8192; and the same top-left corner with
    // WINDOW_OFFSET_DISABLE (bit 31).
    constexpr u32 corner_8192 = 8192U | (8192U << 16);
    constexpr u32 no_window_offset = 1U << 31;

    const std::vector<Write> writes{
        // PA_SC_SCREEN_SCISSOR_TL, _BR
        {context, 0x28030, {0, corner_8192}},
        // PA_SC_LINE_CNTL
        {context, 0x28C00, {0}},
        // PA_SU_VTX_CNTL: pixel centres OpenGL's (1), round by truncation (0),
        // quantised to 1/256 (5, in bits 3-5)
        {context, 0x28C08, {1U | (5U << 3)}},
        // PA_CL_POINT_X_RAD .. PA_CL_POINT_CULL_RAD
        {context, 0x28E10, std::vector<u32>(4, 0)},
        // PA_CL_UCP_0_X .. PA_CL_UCP_5_W: the user clip planes
        {context, 0x28E20, std::vector<u32>(24, 0)},
        // PA_CL_VTE_CNTL: the viewport's scale and offset on in x, y and z,
        // and W0 in the vertex (bit 10)
        {context, 0x28818, {0x3FU | (1U << 10)}},
        // PA_CL_NANINF_CNTL
        {context, 0x28820, {0}},
        // PA_SC_WINDOW_OFFSET, PA_SC_WINDOW_SCISSOR_TL, _BR
        {context, 0x28200, {0, no_window_offset, corner_8192}},
        // PA_SC_LINE_STIPPLE
        {context, 0x28A0C, {0}},
        // PA_SC_MPASS_PS_CNTL, PA_SC_MODE_CNTL: MSAA on (bit 0),
        // FORCE_EOV_CNTDWN_ENABLE (14) and FORCE_EOV_REZ_ENABLE (16)
        {context, 0x28A48, {0, 1U | (1U << 14) | (1U << 16)}},
        // PA_SC_VPORT_SCISSOR_0_TL, _BR
        {context, 0x28250, {no_window_offset, corner_8192}},
        // PA_SC_FORCE_EOV_MAX_CNTS (Linux's rv770d.h)
        {config, 0x8B24, {0xFF3FFF}},
        // PA_SC_CLIPRECT_RULE: every pixel passes
        {context, 0x2820C, {0xFFFF}},
        // VGT_GS_VERTEX_REUSE
        {config, 0x88D4, {16}},
        // VGT_OUTPUT_PATH_CNTL: tessellation's path (1)
        {context, 0x28A10, {1}},
        // VGT_ES_PER_GS (decaf: 16 times two values from GX2's state, both 1
        // as far as it knows), VGT_GS_PER_ES, VGT_GS_PER_VS
        {config, 0x88CC, {16}},
        {config, 0x88C8, {256}},
        {config, 0x88E8, {4}},
        // VGT_INDX_OFFSET, VGT_REUSE_OFF
        {context, 0x28408, {0}},
        {context, 0x28AB4, {0}},
        // VGT_MULTI_PRIM_IB_RESET_EN: primitive restart on
        {context, 0x28A94, {1}},
        // VGT_VERTEX_REUSE_BLOCK_CNTL, VGT_OUT_DEALLOC_CNTL
        {context, 0x28C58, {14, 16}},
        // VGT_HOS_REUSE_DEPTH, VGT_STRMOUT_DRAW_OPAQUE_OFFSET, VGT_VTX_CNT_EN
        {context, 0x28A20, {16}},
        {context, 0x28B28, {0}},
        {context, 0x28AB8, {0}},
        // VGT_MAX_VTX_INDX, VGT_MIN_VTX_INDX
        {context, 0x28400, {0xFFFFFFFF, 0}},
        // TA_CNTL_AUX: bit 1, and the three SYNC bits (24-26)
        {config, 0x9508, {(1U << 1) | (7U << 24)}},
        // VC_ENHANCE (Linux's r600d.h)
        {config, 0x9714, {1}},
        // SQ_DYN_GPR_CNTL_PS_FLUSH_REQ (Linux's R7xx register list)
        {config, 0x8D8C, {0x4000}},
        // SQ_ESTMP_RING_BASE .. SQ_REDUC_RING_SIZE
        {config, 0x8C50, std::vector<u32>(12, 0)},
        // SQ_ESTMP_RING_ITEMSIZE .. SQ_REDUC_RING_ITEMSIZE
        {context, 0x288B0, std::vector<u32>(6, 0)},
        // SQ_VTX_START_INST_LOC
        {Opcode::SetCtlConst, 0x3CFF4, {0}},
        // SPI_FOG_CNTL, SPI_FOG_FUNC_SCALE, SPI_FOG_FUNC_BIAS
        {context, 0x286DC, {0, 0, 0}},
        // SPI_INTERP_CONTROL_0: flat shading on, point sprites off and
        // overridden to s, t, 0, 1, top at 1
        {context, 0x286D4, {1U | (2U << 2) | (3U << 5) | (0U << 8) | (1U << 11) | (1U << 14)}},
        // SPI_CONFIG_CNTL_1
        {config, 0x913C, {0}},
        // SPI_THREAD_GROUPING (Linux's R7xx register list), in every context
        {Opcode::SetAllContexts, 0x286C8, {1}},
        // SX_SURFACE_SYNC (likewise): 0x1FF, decaf's branch for a value it
        // reads as zero
        {context, 0x28354, {0x1FF}},
        // DB_SRESULTS_COMPARE_STATE0, _1; DB_RENDER_OVERRIDE
        {context, 0x28D28, {0, 0}},
        {context, 0x28D10, {0}},
        // DB_DEBUG (Linux's r600d.h), then an unnamed one
        {config, 0x9830, {0}},
        {config, 0x983C, {0x1000000}},
        // CB_CLRCMP_CONTROL (select the source, 1 in bits 24-25), _SRC, _DST,
        // _MSK
        {context, 0x28C30, {1U << 24, 0, 0, 0xFFFFFFFF}},
        // unnamed
        {config, 0x9A1C, {0}},
        // PA_SC_AA_MASK: every sample
        {context, 0x28C48, {0xFFFFFFFF}},
        // PA_SC_EDGERULE (Linux's rv770d.h, which sets the same)
        {context, 0x28230, {0xAAAAAAAA}},
    };

    std::vector<u32> out;
    for (const Write& write : writes) {
        append_registers(out, write.opcode, write.address, write.values);
    }
    return out;
}

std::vector<u32> default_state_packets() {
    std::vector<u32> out;

    append_shader_mode(out,
                       ShaderMode::UniformRegister,
                       ShaderResources{.vs_gprs = 48,
                                       .vs_stack = 64,
                                       .gs_gprs = 0,
                                       .gs_stack = 0,
                                       .ps_gprs = 200,
                                       .ps_stack = 192});

    constexpr u32 less = 1;
    constexpr u32 always = 7;
    constexpr u32 replace = 2;
    append_context_register(out,
                            latte::reg::db_depth_control,
                            depth_stencil_control(DepthStencil{.depth_test = true,
                                                               .depth_write = true,
                                                               .depth_compare = less,
                                                               .stencil_test = false,
                                                               .backface_stencil = false,
                                                               .front_compare = always,
                                                               .front_depth_pass = replace,
                                                               .front_depth_fail = replace,
                                                               .front_stencil_fail = replace,
                                                               .back_compare = always,
                                                               .back_depth_pass = replace,
                                                               .back_depth_fail = replace,
                                                               .back_stencil_fail = replace}));

    constexpr u32 triangles = 2;
    append_context_register(out,
                            latte::reg::pa_su_sc_mode_cntl,
                            polygon_control(PolygonControl{
                                .front_face = 0, .front_mode = triangles, .back_mode = triangles}));

    append_stencil_mask(out, stencil_mask_registers(0xFF, 0xFF, 1, 0xFF, 0xFF, 1));
    append_polygon_offset(out, polygon_offset_registers(0, 0, 0, 0, 0));
    append_context_register(out, latte::reg::pa_su_point_size, point_size_register(1, 1));
    append_context_register(out, latte::reg::pa_su_point_minmax, point_limits_register(1, 1));
    append_context_register(out, latte::reg::pa_su_line_cntl, line_width_register(1));
    append_context_register(out, latte::reg::vgt_multi_prim_ib_reset_indx, 0xFFFFFFFF);
    append_alpha_test(out, alpha_test_registers(false, less, 0));
    append_context_register(out, latte::reg::db_alpha_to_mask, alpha_to_mask_register(false, 0));
    append_context_register(
        out,
        latte::reg::cb_target_mask,
        target_channel_masks_register({0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF}));

    constexpr u32 copy = 0xCC;
    append_context_register(
        out, latte::reg::cb_color_control, color_control_register(copy, 0, false, true));

    constexpr u32 source_alpha = 4;
    constexpr u32 inverse_source_alpha = 5;
    constexpr u32 add = 0;
    const u32 blend = blend_control_register(BlendControl{.color_source = source_alpha,
                                                          .color_destination = inverse_source_alpha,
                                                          .color_combine = add,
                                                          .separate_alpha = true,
                                                          .alpha_source = source_alpha,
                                                          .alpha_destination = inverse_source_alpha,
                                                          .alpha_combine = add});
    for (u32 target = 0; target < latte::reg::render_target_count; ++target) {
        append_blend_control(out, target, blend);
    }

    append_blend_constant_color(out,
                                BlendConstantColorReg{.red = 0, .green = 0, .blue = 0, .alpha = 0});
    append_context_register(out, latte::reg::vgt_strmout_en, 0);
    append_context_register(
        out, latte::reg::pa_cl_clip_cntl, clip_control_register(true, true, false));
    // GX2SetTessellation is a stub (gx2_state.hpp), so it writes nothing.
    append_context_register(out, latte::reg::vgt_hos_max_tess_level, std::bit_cast<u32>(1.0F));
    append_context_register(out, latte::reg::vgt_hos_min_tess_level, std::bit_cast<u32>(1.0F));
    append_context_register(out, latte::reg::db_render_control, 0);
    return out;
}

void register_gx2_context(Registry& registry) {
    registry.add("gx2", "GX2SetDefaultState", gx2_set_default_state);
    registry.add("gx2", "GX2SetupContextStateEx", gx2_setup_context_state_ex);
    registry.add("gx2", "GX2SetContextState", gx2_set_context_state);
    registry.add("gx2", "GX2GetContextStateDisplayList", gx2_get_context_state_display_list);
}

} // namespace wiiu::hle::gx2
