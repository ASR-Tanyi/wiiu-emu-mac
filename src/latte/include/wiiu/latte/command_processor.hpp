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
//   * **Writes back.** `MEM_WRITE` and `EVENT_WRITE_EOP` put a value -- a
//     fence, a timestamp, the GPU's clock -- into guest memory, which is how a
//     title learns the GPU got somewhere. They are applied as the packet runs;
//     under ADR-004 packets run when their submission *retires*, so that is
//     the moment the guest can see them, on the guest's own clock.
//
// Everything else is either **deferred** -- recognised, counted, and not yet
// acted on: cache flushes, waits on memory, context shadowing -- or
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

struct CommandProcessorStats {
    u64 packets = 0;
    u64 draws = 0;
    u64 indirect_buffers = 0;

    /// Recognised, and not yet acted on (see the header comment).
    u64 deferred = 0;

    /// Opcodes this processor has no name for.
    u64 unknown = 0;

    /// Values written into guest memory by `MEM_WRITE` and `EVENT_WRITE_EOP`.
    u64 memory_writes = 0;

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

    explicit CommandProcessor(RegisterFile& registers) : registers_(&registers) {}

    /// Guest memory, which `INDIRECT_BUFFER` and `DRAW_INDEX_2` read and
    /// `MEM_WRITE` and `EVENT_WRITE_EOP` write. Without it those packets are
    /// errors -- there is nowhere to read or write.
    void set_memory(memory::Memory* memory) noexcept { memory_ = memory; }

    void set_draw_listener(DrawListener listener) { draw_listener_ = std::move(listener); }

    void set_interrupt_listener(InterruptListener listener) {
        interrupt_listener_ = std::move(listener);
    }

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

    /// Nested buffers deeper than this are refused. GX2 display lists call
    /// one level down; a cycle in guest memory would otherwise recurse until
    /// the host stack ran out.
    static constexpr u32 max_indirect_depth = 8;

private:
    [[nodiscard]] bool execute_at_depth(std::span<const u32> words, u32 depth);

    void packet(const pm4::Packet& packet, u32 depth);
    void set_window(const pm4::Packet& packet, Window window);
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
    u64 clock_ = 0;

    u32 instances_ = 1;
    u32 index_type_ = 0;

    CommandProcessorStats stats_;
    std::string last_error_;
    std::map<u8, u64> unhandled_;
};

} // namespace wiiu::latte
