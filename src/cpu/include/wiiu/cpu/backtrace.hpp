// Phase 10.7 — reading a guest call stack.
//
// PowerPC's EABI links stack frames together: the word at the stack pointer is
// the previous frame's address, and a frame stores its *caller's* return
// address at offset 4. Walking that chain recovers a call stack without any
// symbol information, which is what makes a fault report useful rather than
// just an address.
//
// Guest memory is untrusted and a corrupted chain is exactly what a crash
// looks like, so the walk is bounded in every direction: frames must be
// aligned, must move upward, must stay mapped, and stop at a limit.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/memory/memory.hpp>

#include <string>
#include <vector>

namespace wiiu::cpu {

struct StackFrame {
    ppcaddr_t frame = 0;     ///< the frame's own stack address
    ppcaddr_t return_to = 0; ///< where it will return
};

struct BacktraceOptions {
    u32 max_frames = 32;
    /// A frame that does not move upward, or moves by more than this, is
    /// treated as the end of a corrupted chain rather than followed.
    u32 max_frame_size = 0x10000;
};

/// Walk the back-chain from the current stack pointer.
///
/// The first entry is the current function, whose return address is in LR
/// rather than in memory -- the prologue has not necessarily stored it yet.
[[nodiscard]] std::vector<StackFrame> backtrace(const CoreState& state,
                                                const memory::Memory& memory,
                                                const BacktraceOptions& options = {});

/// Registers, formatted for a human, with the non-zero ones first.
[[nodiscard]] std::string format_registers(const CoreState& state);

/// Instructions around `address`, with a marker on the one at it.
[[nodiscard]] std::string
format_code_window(const memory::Memory& memory, ppcaddr_t address, u32 before = 4, u32 after = 4);

/// Everything at once: where it stopped, the code there, the registers, and
/// the call stack.
[[nodiscard]] std::string format_state_dump(const CoreState& state, const memory::Memory& memory);

} // namespace wiiu::cpu
