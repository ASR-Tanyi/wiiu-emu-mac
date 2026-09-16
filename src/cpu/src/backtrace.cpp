#include <wiiu/cpu/backtrace.hpp>
#include <wiiu/cpu/disassembler.hpp>

#include <fmt/format.h>

namespace wiiu::cpu {

std::vector<StackFrame>
backtrace(const CoreState& state, const memory::Memory& memory, const BacktraceOptions& options) {
    std::vector<StackFrame> frames;

    // The innermost frame's return address lives in LR: a function's prologue
    // stores it into its caller's frame, and at the point of a fault it may
    // not have run yet.
    frames.push_back({state.gpr[1], state.lr & ~3U});

    ppcaddr_t sp = state.gpr[1];

    for (u32 depth = 1; depth < options.max_frames; ++depth) {
        if (sp == 0 || (sp % 4) != 0) {
            break;
        }

        u32 next = 0;
        if (!memory.try_read<u32>(sp, next)) {
            break;
        }

        // A null back-chain is the documented end of the stack. Anything that
        // does not move upward, or moves absurdly far, is a corrupted chain --
        // and following it would produce a plausible-looking stack of
        // nonsense, which is worse than stopping.
        if (next == 0 || next <= sp || next - sp > options.max_frame_size) {
            break;
        }

        u32 return_address = 0;
        if (!memory.try_read<u32>(next + 4, return_address)) {
            break;
        }
        if (return_address == 0) {
            break;
        }

        frames.push_back({next, return_address & ~3U});
        sp = next;
    }

    return frames;
}

std::string format_registers(const CoreState& state) {
    std::string out;

    out += fmt::format(
        "  pc  {:#010x}   lr  {:#010x}   ctr {:#010x}\n", state.pc, state.lr, state.ctr);
    out += fmt::format("  cr  {:#010x}   xer {:#010x}   msr {:#010x}\n",
                       state.cr.value,
                       state.xer.value,
                       state.msr);

    for (u32 i = 0; i < 32; i += 4) {
        out += "  ";
        for (u32 j = 0; j < 4; ++j) {
            out += fmt::format("r{:<2} {:#010x}  ", i + j, state.gpr[i + j]);
        }
        out += "\n";
    }
    return out;
}

std::string
format_code_window(const memory::Memory& memory, ppcaddr_t address, u32 before, u32 after) {
    std::string out;

    const ppcaddr_t start = address - before * 4;
    for (u32 i = 0; i < before + after + 1; ++i) {
        const ppcaddr_t at = start + i * 4;

        u32 word = 0;
        if (!memory.try_read<u32>(at, word)) {
            out += fmt::format("  {}{:08x}:  <unmapped>\n", at == address ? "->" : "  ", at);
            continue;
        }

        out += fmt::format("  {}{:08x}:  {:08x}  {}\n",
                           at == address ? "->" : "  ",
                           at,
                           word,
                           disassemble(word, at));
    }
    return out;
}

std::string format_state_dump(const CoreState& state, const memory::Memory& memory) {
    std::string out;

    out += fmt::format("stopped at {:#010x}\n\n", state.pc);
    out += "code:\n";
    out += format_code_window(memory, state.pc);
    out += "\nregisters:\n";
    out += format_registers(state);

    out += "\ncall stack:\n";
    const auto frames = backtrace(state, memory);
    if (frames.empty()) {
        out += "  (none)\n";
    }
    for (std::size_t i = 0; i < frames.size(); ++i) {
        out += fmt::format(
            "  #{:<2} {:#010x}   frame {:#010x}\n", i, frames[i].return_to, frames[i].frame);
    }

    return out;
}

} // namespace wiiu::cpu
