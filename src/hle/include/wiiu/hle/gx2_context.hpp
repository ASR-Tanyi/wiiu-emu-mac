// Phase 8.1 — GX2's context states.
//
// A title keeps its GPU state in a `GX2ContextState`: a block of its own
// memory holding an image of the register windows, built once
// (`GX2SetupContextStateEx`) and bound whenever the title wants that state
// back (`GX2SetContextState`) -- WHB binds its one every frame. The GPU does
// the work: binding is a `CONTEXT_CONTROL` that turns state shadowing on and
// a `LOAD_*` packet per window, which loads the registers GX2 cares about
// from the block and makes it where every later `SET_*` is mirrored, until
// another state is bound (command_processor.hpp). So a title can switch
// between states and find each as it left it.
//
// Setting one up names the block without loading it, so the state GX2 then
// writes lands in it, and records the binding packets in the block's own
// display list, so binding is one call. The block's layout and the register
// ranges are decaf's and Cemu's, which agree word for word; where they differ
// on a value, which is named below, Cemu's is taken, as the reading of what
// GX2 itself writes.
//
// A new block is given GX2's **default state**: the register initialisation
// GX2 does before anything else, then `GX2SetDefaultState`. `GX2Init` writes
// the same, with shadowing off, and submits it -- so the GPU starts in a
// known state and the first timestamp a title sees is 2. Both sequences are
// decaf's, which read them out of gx2.rpl; Cemu writes much less, and only in
// `GX2SetDefaultState`.
//
// Not yet: the shader-cache invalidation decaf ends a binding with, which
// waits for `GX2Invalidate`, and `GX2SetAAMode`, which decaf's
// initialisation leaves as a to-do too.
#pragma once

#include <wiiu/common/types.hpp>

#include <span>
#include <vector>

namespace wiiu::hle {

class Registry;

namespace gx2 {

/// One run of registers a binding loads: `count` from `offset` words into a
/// window.
struct LoadRange {
    u32 offset = 0;
    u32 count = 0;
};

/// A window's part of a context state: where in the block its image is, how
/// big that is, and which of its registers a binding loads.
struct ShadowArea {
    u8 opcode = 0;     ///< the `LOAD_*` that loads it
    u32 offset = 0;    ///< bytes into the block
    u32 registers = 0; ///< registers the image holds
    std::span<const LoadRange> ranges;
};

/// `GX2ContextState`'s layout: the six images, then GX2's own fields. Each
/// image is padded to 256 bytes.
inline constexpr u32 context_state_bytes = 0xA100;
inline constexpr u32 context_state_shadow_bytes = 0x9800;
inline constexpr u32 context_state_profiling = 0x9800; ///< BOOL
inline constexpr u32 context_state_list_size = 0x9804; ///< bytes of the list below
inline constexpr u32 context_state_list = 0x9E00;      ///< the binding's display list
inline constexpr u32 context_state_list_words = 192;   ///< room for it

/// The six windows a context state holds, in the order a binding loads them:
/// config, context, ALU constants, loop constants, resources, samplers.
[[nodiscard]] std::span<const ShadowArea> shadow_areas() noexcept;

/// `GX2ContextStateFlags`.
inline constexpr u32 context_state_profiling_flag = 1U << 0;
inline constexpr u32 context_state_no_display_list_flag = 1U << 1;

/// What `CONTEXT_CONTROL` is written with to turn shadowing on: load and
/// shadow the six windows a context state holds, and bit 31. Cemu's value;
/// decaf turns the bool and control constants on as well, which no block
/// has room for and so are never given a base.
inline constexpr u32 shadowing_on = 0x80000077;

/// ... and off: bit 31 alone, in both words (decaf and Cemu agree).
inline constexpr u32 shadowing_off = 0x80000000;

/// The packets that bind the context state at physical `state`: shadowing on,
/// then a `LOAD_*` per window. With `load` false they name each image as its
/// window's shadow base without loading anything -- one empty range each, as
/// decaf has it -- which is how a new state is set up. Cemu loads the zeroed
/// block instead; decaf's is taken, since a state's registers are written
/// into it straight after.
[[nodiscard]] std::vector<u32> bind_context_state_packets(physaddr_t state, bool load);

/// The packet that turns shadowing off, which `GX2SetContextState(NULL)`
/// writes.
[[nodiscard]] std::vector<u32> unbind_context_state_packets();

/// The registers GX2 initialises before writing any state: decaf's
/// `initialiseRegisters`, register for register and in its order. Where decaf
/// leaves one unnamed and Linux's R600/R700 headers name it, the name is
/// Linux's; three stay unnamed. One departure: decaf writes the multipass and
/// mode controls (PA_SC_MPASS_PS_CNTL, PA_SC_MODE_CNTL, 0x28A48) at 0x28A0C,
/// PA_SC_LINE_STIPPLE's address, which would put the mode control's value in
/// VGT_OUTPUT_PATH_CNTL; they are written where their names say.
[[nodiscard]] std::vector<u32> initial_register_packets();

/// `GX2SetDefaultState`'s packets: decaf's sequence of state calls with
/// their defaults -- shader mode, depth on with less-than, stencil off,
/// no culling, every channel written, blending set up (source alpha) but
/// off, and the rest -- each written as its call writes it.
[[nodiscard]] std::vector<u32> default_state_packets();

/// Register the context-state calls. Called by `register_gx2`.
void register_gx2_context(Registry& registry);

} // namespace gx2
} // namespace wiiu::hle
