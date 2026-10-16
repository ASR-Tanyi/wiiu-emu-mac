// Phase 8.1 — GX2's context states. See gx2_context.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_context.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/pm4.hpp>

#include <array>
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

    // The default state is written here next (see the header).

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

void register_gx2_context(Registry& registry) {
    registry.add("gx2", "GX2SetupContextStateEx", gx2_setup_context_state_ex);
    registry.add("gx2", "GX2SetContextState", gx2_set_context_state);
    registry.add("gx2", "GX2GetContextStateDisplayList", gx2_get_context_state_display_list);
}

} // namespace wiiu::hle::gx2
