// Phase 8.4 — Latte's register file.
//
// Everything a draw depends on that is not in the draw packet itself --
// render targets, blend state, viewport, shader addresses, every texture and
// sampler descriptor, every uniform -- is a register, written by PM4 before
// the draw that reads it. So the register file is the whole of the GPU's
// state, and the command processor's main job is keeping it up to date.
//
// It is one flat array over the register space, indexed by byte address / 4,
// because that is what Type 0 packets address and what the `SET_*` packets'
// windows are carved out of. Each `SET_*` opcode may only write inside its
// own window; a write outside it is a translation bug (or a corrupt stream)
// and is refused rather than landing on some other register.
//
// Window bounds are the R600 family's (`r600d.h`); decaf's Latte tables start
// every window at the same address.
#pragma once

#include <wiiu/common/types.hpp>

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wiiu::latte {

/// The register space, in bytes.
inline constexpr u32 register_space_bytes = 0x40000;
inline constexpr u32 register_count = register_space_bytes / 4;

/// The windows the `SET_*` packets write into.
enum class Window : u8 {
    Config,    ///< SET_CONFIG_REG: global state, the VGT's primitive type
    Context,   ///< SET_CONTEXT_REG: render targets, blend, depth, viewport, shaders
    AluConst,  ///< SET_ALU_CONST: uniform registers, four words each
    Resource,  ///< SET_RESOURCE: texture and buffer descriptors, seven words each
    Sampler,   ///< SET_SAMPLER: sampler descriptors, three words each
    CtlConst,  ///< SET_CTL_CONST
    LoopConst, ///< SET_LOOP_CONST
    BoolConst, ///< SET_BOOL_CONST
};

struct WindowRange {
    u32 base = 0; ///< first byte address
    u32 end = 0;  ///< one past the last

    [[nodiscard]] constexpr u32 words() const noexcept { return (end - base) / 4; }
};

[[nodiscard]] WindowRange window_range(Window window) noexcept;
[[nodiscard]] std::string_view window_name(Window window) noexcept;

/// Registers the command processor itself reads. Byte addresses.
namespace reg {

inline constexpr u32 vgt_primitive_type = 0x08958;
inline constexpr u32 vgt_index_type = 0x0895C;
inline constexpr u32 vgt_num_indices = 0x08970;

} // namespace reg

class RegisterFile {
public:
    RegisterFile() : values_(register_count, 0) {}

    /// A register by byte address. Zero outside the space, or unaligned.
    [[nodiscard]] u32 read(u32 address) const noexcept;

    /// Set one register by byte address. False outside the space or
    /// unaligned.
    bool write(u32 address, u32 value) noexcept;

    /// What a Type 0 packet does: consecutive registers from a register
    /// index (byte address / 4). False, with nothing written, if any of them
    /// would fall outside the space.
    [[nodiscard]] bool write_type0(u32 base_index, std::span<const u32> values, std::string& error);

    /// What a `SET_*` packet does: consecutive registers from `offset` words
    /// into `window`. False, with nothing written, if any would fall outside
    /// the window.
    [[nodiscard]] bool
    write_window(Window window, u32 offset, std::span<const u32> values, std::string& error);

    /// Every register back to zero, as after a GPU reset.
    void reset() noexcept;

    /// Registers written since construction or `reset`. A cheap way for a
    /// test, or a trace, to ask whether a stream did anything at all.
    [[nodiscard]] u64 writes() const noexcept { return writes_; }

private:
    std::vector<u32> values_;
    u64 writes_ = 0;
};

} // namespace wiiu::latte
