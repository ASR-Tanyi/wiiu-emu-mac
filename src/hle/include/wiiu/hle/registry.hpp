// Phase 6.1 — the HLE dispatch framework.
//
// Cafe OS libraries are reimplemented as native functions. The awkward part is
// the boundary: guest code calls `OSReport(const char*, ...)` following the
// PowerPC ABI, and the implementation should be an ordinary C++ function
// rather than something that picks arguments out of registers by hand.
//
//   registry.add("coreinit", "OSGetCoreId", [](Call&) -> u32 { return 1; });
//   registry.add("coreinit", "OSReport", os_report);
//
// Arguments are pulled from r3-r10 and f1-f8 in ABI order and the return value
// goes back in r3 or f1, all deduced from the function's signature. Getting
// this wrong once in the framework is much better than getting it wrong
// occasionally in four hundred stubs.
//
// How a call arrives: every resolved import gets a one-word trampoline
// containing `sc`, and call sites are relocated to it. The interpreter already
// stops on `sc`, so the dispatcher looks up the address and runs the handler —
// which is also how Cafe OS itself reaches the kernel.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>
#include <bit>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace wiiu::hle {

/// Everything a handler can reach: the calling core and guest memory.
struct Call {
    cpu::CoreState& state;
    memory::Memory& memory;

    /// Argument registers, consumed in order as the marshaller walks the
    /// parameter list.
    u32 next_gpr = 3;
    u32 next_fpr = 1;

    /// Bytes consumed of the caller's parameter area, where arguments go once
    /// their registers run out: from 8 bytes above the stack pointer (past
    /// the back chain and the LR save word), in order -- a word for an int
    /// or a float, eight bytes 8-aligned for a 64-bit value or a double.
    /// (Checked against powerpc-linux-gnu-gcc -m32 for 13 ints, a long long
    /// after the registers are full, and floats and doubles past f8.) A
    /// handler is called from the title's `bl`, with no frame of its own in
    /// between, so r1 is still the caller's. GX2's state calls take up to
    /// fourteen arguments.
    u32 stack_offset = 0;

    [[nodiscard]] u32 next_word() noexcept {
        return next_gpr <= 10 ? state.gpr[next_gpr++] : stack_word();
    }

    /// A 64-bit argument occupies a register pair starting at an *odd*
    /// register: r3:r4, r5:r6, r7:r8, r9:r10. So `f(long long)` uses r3:r4
    /// with no gap, while `f(int, long long)` puts the int in r3, skips r4,
    /// and uses r5:r6. (Verified against powerpc-linux-gnu-gcc -m32 for the
    /// leading, one-word-in and two-words-in cases.) Getting the parity
    /// backwards reads every 64-bit argument one register late, which looks
    /// like a plausible-but-wrong value rather than an obvious fault.
    [[nodiscard]] u64 next_long() noexcept {
        if ((next_gpr % 2) == 0) {
            ++next_gpr;
        }
        if (next_gpr <= 9) {
            const u64 high = state.gpr[next_gpr];
            const u64 low = state.gpr[next_gpr + 1];
            next_gpr += 2;
            return (high << 32) | low;
        }
        // No pair left: this and everything after it is on the stack, even
        // if r10 was free.
        next_gpr = 11;
        return stack_long();
    }

    [[nodiscard]] f64 next_double() noexcept {
        return next_fpr <= 8 ? state.fpr[next_fpr++].ps0 : std::bit_cast<f64>(stack_long());
    }

    /// A `float` parameter: in an FPR as a double, or on the stack as a
    /// single in a word of its own.
    [[nodiscard]] f32 next_single() noexcept {
        return next_fpr <= 8 ? static_cast<f32>(state.fpr[next_fpr++].ps0)
                             : std::bit_cast<f32>(stack_word());
    }

    /// The next word of the parameter area. Zero if it cannot be read, as a
    /// register past r10 used to be.
    [[nodiscard]] u32 stack_word() noexcept {
        u32 value = 0;
        static_cast<void>(memory.try_read<u32>(state.gpr[1] + 8 + stack_offset, value));
        stack_offset += 4;
        return value;
    }

    [[nodiscard]] u64 stack_long() noexcept {
        stack_offset = (stack_offset + 7) & ~7U;
        const u64 high = stack_word();
        const u64 low = stack_word();
        return (high << 32) | low;
    }

    /// A null-terminated guest string, bounded so a missing terminator cannot
    /// run away through memory.
    [[nodiscard]] std::string read_string(ppcaddr_t address, u32 limit = 4096) const;
};

using Handler = std::function<void(Call&)>;

// --------------------------------------------------------------------------
// Argument marshalling
// --------------------------------------------------------------------------

namespace detail {

template <typename T>
struct Argument;

template <>
struct Argument<u32> {
    static u32 read(Call& call) { return call.next_word(); }
};

template <>
struct Argument<s32> {
    static s32 read(Call& call) { return static_cast<s32>(call.next_word()); }
};

template <>
struct Argument<u64> {
    static u64 read(Call& call) { return call.next_long(); }
};

template <>
struct Argument<s64> {
    static s64 read(Call& call) { return static_cast<s64>(call.next_long()); }
};

template <>
struct Argument<bool> {
    static bool read(Call& call) { return call.next_word() != 0; }
};

template <>
struct Argument<f32> {
    static f32 read(Call& call) { return call.next_single(); }
};

template <>
struct Argument<f64> {
    static f64 read(Call& call) { return call.next_double(); }
};

template <typename T>
void write_return(Call& call, T value) {
    if constexpr (std::is_same_v<T, f32> || std::is_same_v<T, f64>) {
        call.state.fpr[1].splat(static_cast<f64>(value));
    } else if constexpr (std::is_same_v<T, u64> || std::is_same_v<T, s64>) {
        const auto wide = static_cast<u64>(value);
        call.state.gpr[3] = static_cast<u32>(wide >> 32);
        call.state.gpr[4] = static_cast<u32>(wide);
    } else {
        call.state.gpr[3] = static_cast<u32>(value);
    }
}

} // namespace detail

/// Wrap a plain function as a handler, marshalling from its signature.
template <typename Result, typename... Args>
[[nodiscard]] Handler wrap(Result (*function)(Call&, Args...)) {
    return [function](Call& call) {
        // Arguments are read left to right, which the ABI requires and
        // braced initialisation guarantees; a plain call expression would
        // leave the order up to the compiler.
        const std::tuple<Args...> arguments{detail::Argument<std::decay_t<Args>>::read(call)...};

        if constexpr (std::is_void_v<Result>) {
            std::apply([&](auto&&... unpacked) { function(call, unpacked...); }, arguments);
        } else {
            const Result value = std::apply(
                [&](auto&&... unpacked) { return function(call, unpacked...); }, arguments);
            detail::write_return(call, value);
        }
    };
}

// --------------------------------------------------------------------------
// Registry
// --------------------------------------------------------------------------

struct Function {
    std::string library;
    std::string name;
    Handler handler;
    ppcaddr_t trampoline = 0;
    u64 call_count = 0;
    bool is_stub = false; ///< registered as a placeholder, not implemented
};

/// One entry in the recent-call ring (Phase 10.3).
struct TraceEntry {
    const Function* function = nullptr;
    ppcaddr_t caller = 0;
    std::array<u32, 4> arguments{}; ///< r3-r6, enough to recognise a call
};

class Registry {
public:
    /// Where trampolines live. Outside every Cafe OS region, so a stray jump
    /// into it is obviously an HLE call and not guest code.
    static constexpr ppcaddr_t trampoline_base = 0x01900000;
    static constexpr u32 trampoline_size = 0x10000;

    void add(std::string library, std::string name, Handler handler);

    /// Register a plain function, deducing the marshalling.
    template <typename Result, typename... Args>
    void add(std::string library, std::string name, Result (*function)(Call&, Args...)) {
        add(std::move(library), std::move(name), wrap(function));
    }

    /// A function that is known to exist but does nothing yet. Calls are
    /// counted and logged so the next gap is always visible (6.12).
    void add_stub(std::string library, std::string name, u32 return_value = 0);

    [[nodiscard]] Function* find(std::string_view library, std::string_view name);
    [[nodiscard]] const Function* find_by_trampoline(ppcaddr_t address) const;

    /// Assign a trampoline address, registering the function as a stub if it
    /// is not implemented. Always succeeds: an unimplemented import must not
    /// stop a title loading, it must be visible when it is called.
    [[nodiscard]] ppcaddr_t resolve(std::string_view library, std::string_view name);

    /// Write the `sc` words backing the trampolines into guest memory.
    [[nodiscard]] bool install(memory::Memory& memory, std::string& error);

    /// Run the handler for a trampoline address. Returns false if the address
    /// is not one of ours.
    [[nodiscard]] bool dispatch(ppcaddr_t address, cpu::CoreState& state, memory::Memory& memory);

    [[nodiscard]] std::size_t size() const noexcept { return functions_.size(); }

    [[nodiscard]] std::vector<const Function*> unimplemented_called() const;

    /// Every function that was called, most-called first. The cheapest answer
    /// to "what is this title actually using".
    [[nodiscard]] std::vector<const Function*> called() const;

    [[nodiscard]] u64 total_calls() const noexcept { return total_calls_; }

    // --- tracing (Phase 10.3) ---------------------------------------------

    /// Keep the last `capacity` calls. Zero disables it, which is the default:
    /// a ring of every HLE call is not free, and most runs do not need one.
    void set_trace_capacity(std::size_t capacity);

    /// Log every call as it happens. Separate from the ring because the two
    /// answer different questions -- "what happened just before the crash"
    /// versus "what is it doing right now".
    void set_trace_logging(bool enabled) noexcept { trace_logging_ = enabled; }

    [[nodiscard]] std::vector<TraceEntry> recent_calls() const;

    /// Call counts and the recent-call ring, formatted for a fault report.
    [[nodiscard]] std::string format_summary(std::size_t top = 10) const;

private:
    [[nodiscard]] static std::string key(std::string_view library, std::string_view name);

    std::map<std::string, Function> functions_;
    std::map<ppcaddr_t, Function*> by_trampoline_;
    ppcaddr_t next_trampoline_ = trampoline_base;

    std::deque<TraceEntry> trace_;
    std::size_t trace_capacity_ = 0;
    bool trace_logging_ = false;
    u64 total_calls_ = 0;
};

} // namespace wiiu::hle
