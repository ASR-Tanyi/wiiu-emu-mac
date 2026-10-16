// Phase 8.3 — PM4, the language Latte's command processor reads.
//
// GX2 does not call the GPU; it writes packets into a buffer and the command
// processor reads them, asynchronously, on its own schedule (8.2). Every frame
// a title draws arrives here first, as a stream of 32-bit words:
//
//   * **Type 0** writes `count + 1` consecutive registers starting at a
//     register index (a byte address over four). Rare from GX2, common from
//     anything built on raw register pokes.
//   * **Type 2** is one word of filler: alignment, nothing else.
//   * **Type 3** is a command: an opcode and `count + 1` words of payload.
//     Every draw, every register-window write (`SET_CONTEXT_REG` and family),
//     every fence and every nested command buffer is one of these.
//
// Type 1 existed on older Radeons and does not on R600-class parts; a stream
// containing one is not something to guess about.
//
// This file only *reads*. Words come in host order -- GX2 writes them
// big-endian and whoever fetches them from guest memory swaps -- and packets
// come out with their payload as a view into the same words, so walking a
// stream copies nothing. What a packet *does* is the command processor's
// business.
#pragma once

#include <wiiu/common/types.hpp>

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace wiiu::latte::pm4 {

enum class PacketType : u8 {
    Type0 = 0,
    Type1 = 1,
    Type2 = 2,
    Type3 = 3,
};

/// Type 3 opcodes. The numbers are the R600 family's (Linux's `r600d.h`
/// agrees with decaf's Latte tables); only the ones anything here will name
/// are listed, and an unlisted one still decodes and still has a number.
/// Linux never sends the `LOAD_*` family, so for those it is decaf's and
/// Cemu's tables that agree.
enum class Opcode : u8 {
    Nop = 0x10,
    IndirectBufferEnd = 0x17,
    SetPredication = 0x20,
    RegRmw = 0x21,
    CondExec = 0x22,
    PredExec = 0x23,
    Start3dCmdbuf = 0x24,
    IndexBase = 0x26,
    DrawIndex2 = 0x27,
    ContextControl = 0x28,
    DrawIndexOffset = 0x29,
    IndexType = 0x2A,
    DrawIndex = 0x2B,
    DrawIndexAuto = 0x2D,
    DrawIndexImmediate = 0x2E,
    NumInstances = 0x2F,
    DrawIndexMultiAuto = 0x30,
    IndirectBufferPriv = 0x32,
    StreamOutBufferUpdate = 0x34,
    DrawIndexOffset2 = 0x35,
    MemSemaphore = 0x39,
    CopyDw = 0x3B,
    WaitRegMem = 0x3C,
    MemWrite = 0x3D,
    IndirectBuffer = 0x3F,
    CpDma = 0x41,
    PfpSyncMe = 0x42,
    SurfaceSync = 0x43,
    MeInitialize = 0x44,
    CondWrite = 0x45,
    EventWrite = 0x46,
    EventWriteEop = 0x47,
    OneRegWrite = 0x57,
    LoadConfigReg = 0x60,
    LoadContextReg = 0x61,
    LoadAluConst = 0x62,
    LoadBoolConst = 0x63,
    LoadLoopConst = 0x64,
    LoadResource = 0x65,
    LoadSampler = 0x66,
    LoadCtlConst = 0x67,
    SetConfigReg = 0x68,
    SetContextReg = 0x69,
    SetAluConst = 0x6A,
    SetBoolConst = 0x6B,
    SetLoopConst = 0x6C,
    SetResource = 0x6D,
    SetSampler = 0x6E,
    SetCtlConst = 0x6F,
    StreamOutBaseUpdate = 0x72,
    SurfaceBaseUpdate = 0x73,
    SetAllContexts = 0x74, ///< SET_CONTEXT_REG for every context; Latte here has one

    // The emulator's own commands, from 0xF0 up -- R600's highest opcode is
    // 0x74, so no packet meant for the hardware is one. Each stands for work GX2
    // does on hardware with a draw, a DMA or the display controller, and
    // which the emulator does in one step when the packet runs: in stream
    // order, on the guest's clock (ADR-004), from a display list as well as
    // the ring. Their payloads are GX2's business (8.1); the command processor
    // only hands them over. decaf and Cemu each have packets of their own for
    // the same jobs.
    HleSwap = 0xF0,              ///< GX2SwapScanBuffers' place in the stream
    HleClearColor = 0xF1,        ///< GX2ClearColor
    HleCopyColorToScan = 0xF2,   ///< GX2CopyColorBufferToScanBuffer
    HleClearDepthStencil = 0xF3, ///< GX2ClearDepthStencilEx
};

/// Whether an opcode is one of the emulator's own (`Hle*`).
[[nodiscard]] constexpr bool is_hle_opcode(u8 opcode) noexcept {
    return opcode >= 0xF0;
}

/// The opcode's name, or empty for a number this table does not know -- so
/// a log line can say `0x5D` rather than inventing a name for it.
[[nodiscard]] std::string_view opcode_name(u8 opcode) noexcept;

/// One decoded header.
struct Header {
    PacketType type = PacketType::Type2;

    /// Words of payload after the header: `count + 1` for types 0 and 3,
    /// none for type 2.
    u32 payload_words = 0;

    /// Type 0: the first register written, as a register index (byte
    /// address / 4).
    u32 base_index = 0;

    /// Type 3.
    u8 opcode = 0;
    bool predicate = false;
};

[[nodiscard]] Header decode_header(u32 word) noexcept;

/// Build a header word. The inverse of `decode_header`, for GX2 (8.1), which
/// writes packets, and for tests, which write streams by hand.
[[nodiscard]] u32 type0_header(u32 base_index, u32 registers) noexcept;
[[nodiscard]] u32 type3_header(Opcode opcode, u32 payload_words, bool predicate = false) noexcept;
inline constexpr u32 type2_filler = 0x80000000;

/// A packet, as found in a stream.
struct Packet {
    Header header;

    /// Where the header sat, in words from the start of the stream.
    std::size_t offset = 0;

    /// The words after the header. A view into the stream being read.
    std::span<const u32> payload;

    [[nodiscard]] bool is(Opcode opcode) const noexcept {
        return header.type == PacketType::Type3 && header.opcode == static_cast<u8>(opcode);
    }
};

/// Walks a stream of PM4 words, one packet at a time.
///
/// A stream that cannot be read further -- a packet whose payload runs past
/// the end, or a Type 1 header -- stops the walk and says why in `error()`.
/// It does not skip ahead and hope: once a header is misread, every word
/// after it is misread too, and the packets that would come out of that are
/// worse than none.
class Reader {
public:
    explicit Reader(std::span<const u32> words) noexcept : words_(words) {}

    /// The next packet, or nothing at the end of the stream or at the first
    /// word that cannot be read.
    [[nodiscard]] std::optional<Packet> next();

    [[nodiscard]] bool failed() const noexcept { return !error_.empty(); }

    [[nodiscard]] const std::string& error() const noexcept { return error_; }

    /// Words consumed so far.
    [[nodiscard]] std::size_t position() const noexcept { return position_; }

private:
    std::span<const u32> words_;
    std::size_t position_ = 0;
    std::string error_;
};

/// One line describing a packet, for traces and for the GPU debugger (8.16).
[[nodiscard]] std::string describe(const Packet& packet);

} // namespace wiiu::latte::pm4
