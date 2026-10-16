// Phase 8.3 — the command processor: what a PM4 stream *does*.
//
// Three kinds of packet matter, and this is where each is told apart:
//
//   * **State.** Type 0 and the `SET_*` family write the register file (8.4),
//     which is all the state a draw reads. Most of a frame is this.
//   * **Draws.** `DRAW_INDEX_AUTO`, `DRAW_INDEX_2` and `DRAW_INDEX_IMMD` are
//     handed to a listener as a `DrawCommand`, with the register file as it
//     stands at that packet. The translator (8.11) is that listener; until it
//     exists, a test is.
//   * **Calls.** `INDIRECT_BUFFER` and `INDIRECT_BUFFER_PRIV` run another
//     stream from guest memory and come back -- which is how GX2 display
//     lists work, so a title's frame is usually a short ring stream calling a
//     long buffer the title built itself.
//
//   * **The emulator's own.** The `Hle*` opcodes (pm4.hpp) are handed, whole,
//     to a listener -- GX2 (8.1), which wrote them and knows what they carry.
//
//   * **Writes back.** `MEM_WRITE` and `EVENT_WRITE_EOP` put a value -- a
//     fence, a timestamp, the GPU's clock -- into guest memory, which is how a
//     title learns the GPU got somewhere. They are applied as the packet runs;
//     under ADR-004 packets run when their submission *retires*, so that is
//     the moment the guest can see them, on the guest's own clock.
//
//   * **Shadowing.** `CONTEXT_CONTROL` turns state shadowing on and off, and
//     the `LOAD_*` packets name, per register window, a block of guest
//     memory: they load registers from it, and while shadowing is on every
//     `SET_*` into that window is written there as well. That is how GX2's
//     context states work -- each is such a block, and switching to one is
//     loading it (see below).
//
// Everything else is either **deferred** -- recognised, counted, and not yet
// acted on: cache flushes, waits on memory -- or
// **unknown**, and logged once by number. Neither stops the stream: the packet framing is still
// right, so the packets after it are still worth reading. A stream that cannot be framed does stop,
// because then nothing after it is.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/memory/memory.hpp>

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>

namespace wiiu::latte {

/// A draw, as the command processor found it.
struct DrawCommand {
    enum class Kind : u8 {
        Auto,      ///< DRAW_INDEX_AUTO: vertices 0..count-1, no index buffer
        Indexed,   ///< DRAW_INDEX_2: indices fetched from guest memory
        Immediate, ///< DRAW_INDEX_IMMD: indices carried in the packet
    };

    Kind kind = Kind::Auto;

    /// Vertices for `Auto`, indices otherwise.
    u32 count = 0;

    /// From the last `NUM_INSTANCES`. One until a stream says otherwise.
    u32 instances = 1;

    /// `VGT_PRIMITIVE_TYPE` at the draw: Latte's encoding, which the
    /// translator maps (and expands, for quads and RECTS).
    u32 primitive = 0;

    /// From the last `INDEX_TYPE`: bit 0 set for 32-bit indices, bits 2-3
    /// the byte swap the index fetch applies.
    u32 index_type = 0;

    /// `Indexed`: where the indices are, as a guest virtual address -- the
    /// packet carries a physical one, and this has been translated back.
    ppcaddr_t index_address = 0;

    /// `Immediate`: the indices, packed as the packet packs them. A view into
    /// the stream, valid only for the listener call that receives it.
    std::span<const u32> immediate_indices;

    /// `VGT_DRAW_INITIATOR`: source select, major mode, and the rest.
    u32 draw_initiator = 0;

    [[nodiscard]] bool indices_are_32bit() const noexcept { return (index_type & 1U) != 0; }
};

/// The bytes the GPU stores for a `size`-byte (4 or 8) write of `value`.
///
/// The GPU is little-endian and stores the value that way, after swapping
/// within units the packet's address names: `endian_swap` 0 is none, 1 swaps
/// bytes within 16-bit units, 2 within 32-bit, 3 within 64-bit (`CB_ENDIAN`).
/// GX2 asks for the swap that makes a value land big-endian, so the guest
/// reads it back as written; a stream that asks for none leaves it
/// little-endian, and the guest sees it byte-reversed -- as on hardware. A
/// unit wider than the write is taken as the whole write.
[[nodiscard]] std::array<u8, 8> encode_write(u64 value, u32 size, u32 endian_swap) noexcept;

/// `CONTEXT_CONTROL`'s two words, LOAD_CONTROL and SHADOW_ENABLE: one bit
/// per register window, and bit 31, which both are written with. decaf's
/// field names; the one value Cemu knows, 0x80000077, is these bits for the
/// six windows a GX2 context state holds.
///
/// The model is decaf's: a `LOAD_*` packet for a window whose load bit is
/// set loads its ranges and makes its address that window's **shadow
/// base**; one whose bit is clear does nothing. A `SET_*` into a window whose
/// shadow bit is set, once it has a base, is written there too -- at the
/// base plus the register's offset into its window, so the block is an image
/// of the window. Cemu instead keeps an address per register, packing the
/// ranges one after another. The registers come out the same either way for
/// everything GX2 does, and only the GPU reads the block back, so either is
/// self-consistent. Shadow memory is big-endian, as Cemu keeps it and as
/// GX2 keeps the rest of what the GPU reads (decaf stores host order).
/// Type 0 writes are not shadowed.
namespace context_control {
inline constexpr u32 config = 1U << 0;
inline constexpr u32 context = 1U << 1;
inline constexpr u32 alu_const = 1U << 2;
inline constexpr u32 bool_const = 1U << 3;
inline constexpr u32 loop_const = 1U << 4;
inline constexpr u32 resource = 1U << 5;
inline constexpr u32 sampler = 1U << 6;
inline constexpr u32 ctl_const = 1U << 7;
inline constexpr u32 ordinal = 1U << 31;

/// The bit for `window`.
[[nodiscard]] constexpr u32 bit(Window window) noexcept {
    switch (window) {
        case Window::Config:
            return config;
        case Window::Context:
            return context;
        case Window::AluConst:
            return alu_const;
        case Window::Resource:
            return resource;
        case Window::Sampler:
            return sampler;
        case Window::CtlConst:
            return ctl_const;
        case Window::LoopConst:
            return loop_const;
        case Window::BoolConst:
            return bool_const;
    }
    return 0;
}
} // namespace context_control

/// The window a `LOAD_*` opcode loads, or nothing for any other opcode.
[[nodiscard]] std::optional<Window> load_window(pm4::Opcode opcode) noexcept;

/// The windows, for walking them all.
inline constexpr u32 window_count = 8;

struct CommandProcessorStats {
    u64 packets = 0;
    u64 draws = 0;
    u64 indirect_buffers = 0;

    /// Recognised, and not yet acted on (see the header comment).
    u64 deferred = 0;

    /// Opcodes this processor has no name for.
    u64 unknown = 0;

    /// The emulator's own packets, handed to the HLE listener.
    u64 hle = 0;

    /// Values written into guest memory by `MEM_WRITE` and `EVENT_WRITE_EOP`.
    u64 memory_writes = 0;

    /// Registers loaded from guest memory by the `LOAD_*` packets.
    u64 registers_loaded = 0;

    /// Registers a `SET_*` also wrote to a shadow block.
    u64 registers_shadowed = 0;

    /// `EVENT_WRITE_EOP`s that asked for an interrupt.
    u64 interrupts = 0;

    /// Packets that were framed correctly but made no sense -- a `SET_*`
    /// past its window, an indirect buffer at an address nothing maps.
    u64 errors = 0;
};

class CommandProcessor {
public:
    using DrawListener = std::function<void(const DrawCommand&, const RegisterFile&)>;

    /// An end-of-pipe interrupt: the event initiator word, and the value the
    /// packet carried (or wrote). GX2 (8.1) is what listens.
    using InterruptListener = std::function<void(u32 event, u64 value)>;

    /// One of the emulator's own packets (`pm4::Opcode::Hle*`). The payload
    /// is a view into the stream, valid only for the call.
    using HleListener = std::function<void(const pm4::Packet&)>;

    explicit CommandProcessor(RegisterFile& registers) : registers_(&registers) {}

    /// Guest memory, which `INDIRECT_BUFFER` and `DRAW_INDEX_2` read and
    /// `MEM_WRITE` and `EVENT_WRITE_EOP` write. Without it those packets are
    /// errors -- there is nowhere to read or write.
    void set_memory(memory::Memory* memory) noexcept { memory_ = memory; }

    void set_draw_listener(DrawListener listener) { draw_listener_ = std::move(listener); }

    void set_interrupt_listener(InterruptListener listener) {
        interrupt_listener_ = std::move(listener);
    }

    void set_hle_listener(HleListener listener) { hle_listener_ = std::move(listener); }

    /// The "GPU clock" a packet may ask to have written: the guest timebase
    /// at retirement (ADR-004), set by whoever runs the stream. Never the
    /// host's clock, which would make the value differ between two runs.
    void set_clock(u64 ticks) noexcept { clock_ = ticks; }

    /// Run a stream of host-order words. False if it could not be read to
    /// its end; `last_error()` says where and why.
    [[nodiscard]] bool execute(std::span<const u32> words);

    [[nodiscard]] const CommandProcessorStats& stats() const noexcept { return stats_; }

    /// The most recent problem, framing or otherwise, or empty.
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

    /// How many times each opcode was deferred or unknown -- the to-do list
    /// for 8.2 and beyond, ordered by what titles actually send.
    [[nodiscard]] const std::map<u8, u64>& unhandled() const noexcept { return unhandled_; }

    /// `CONTEXT_CONTROL`'s words as last set. Both zero -- no loading, no
    /// shadowing -- until a stream sets them.
    [[nodiscard]] u32 load_control() const noexcept { return load_control_; }

    [[nodiscard]] u32 shadow_enable() const noexcept { return shadow_enable_; }

    /// Where `window` is shadowed, as a physical address: the last `LOAD_*`
    /// for it that its load bit let through. Nothing before one has.
    [[nodiscard]] std::optional<physaddr_t> shadow_base(Window window) const noexcept {
        return shadow_bases_[static_cast<u32>(window)];
    }

    /// The GPU's own state back to how it starts: no shadowing, no shadow
    /// blocks, one instance, 16-bit indices. Listeners, memory and counters
    /// stay; the register file is its owner's to reset.
    void reset_state() noexcept;

    /// Nested buffers deeper than this are refused. GX2 display lists call
    /// one level down; a cycle in guest memory would otherwise recurse until
    /// the host stack ran out.
    static constexpr u32 max_indirect_depth = 8;

private:
    [[nodiscard]] bool execute_at_depth(std::span<const u32> words, u32 depth);

    void packet(const pm4::Packet& packet, u32 depth);
    void set_window(const pm4::Packet& packet, Window window);

    /// A context write that includes SQ_VTX_SEMANTIC_CLEAR: every semantic
    /// register whose bit it sets, and that the same write does not set,
    /// becomes all ones. Not shadowed: only what the packet carried is.
    void clear_semantics(Window window, u32 offset, std::span<const u32> values);
    void context_control(const pm4::Packet& packet);
    void load(const pm4::Packet& packet, Window window);

    /// Copy registers just written to `window` at `offset` into its shadow
    /// block, if it is being shadowed.
    void shadow(const pm4::Packet& packet, Window window, u32 offset, std::span<const u32> values);

    /// Guest memory at a physical address, `bytes` long, or null -- having
    /// said why, naming the packet -- if that is not memory the GPU can use.
    [[nodiscard]] u8*
    guest_bytes(const pm4::Packet& packet, physaddr_t physical, u32 bytes, memory::Access access);
    void draw(const pm4::Packet& packet);
    void indirect(const pm4::Packet& packet, u32 depth);
    void mem_write(const pm4::Packet& packet);
    void event_write_eop(const pm4::Packet& packet);

    /// Write `value` the way the GPU does: as a little-endian word of
    /// `bytes`, with the byte swap the packet's address asked for applied.
    /// False, having said why, if the address is not writable guest memory.
    bool
    write_back(const pm4::Packet& packet, u32 address_lo, u32 address_hi, u64 value, u32 bytes);
    void unhandled(const pm4::Packet& packet, bool deferred);
    void fail(std::string message);

    RegisterFile* registers_;
    memory::Memory* memory_ = nullptr;
    DrawListener draw_listener_;
    InterruptListener interrupt_listener_;
    HleListener hle_listener_;
    u64 clock_ = 0;

    u32 instances_ = 1;
    u32 index_type_ = 0;
    u32 load_control_ = 0;
    u32 shadow_enable_ = 0;
    std::array<std::optional<physaddr_t>, window_count> shadow_bases_{};

    CommandProcessorStats stats_;
    std::string last_error_;
    std::map<u8, u64> unhandled_;
};

} // namespace wiiu::latte
