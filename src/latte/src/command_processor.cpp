// Phase 8.3 — see command_processor.hpp.

#include <wiiu/common/endian.hpp>
#include <wiiu/common/log.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/memory/map.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace wiiu::latte {

using pm4::Opcode;
using pm4::Packet;
using pm4::PacketType;

std::array<u8, 8> encode_write(u64 value, u32 size, u32 endian_swap) noexcept {
    std::array<u8, 8> bytes{};
    for (u32 index = 0; index < 8; ++index) {
        bytes[index] = static_cast<u8>(value >> (8 * index));
    }
    const u32 unit = endian_swap == 1 ? 2 : endian_swap == 2 ? 4 : endian_swap == 3 ? 8 : 1;
    const u32 span = unit > size ? size : unit;
    if (span > 1) {
        for (u32 start = 0; start + span <= size; start += span) {
            std::reverse(bytes.begin() + start, bytes.begin() + start + span);
        }
    }
    return bytes;
}

bool CommandProcessor::execute(std::span<const u32> words) {
    return execute_at_depth(words, 0);
}

bool CommandProcessor::execute_at_depth(std::span<const u32> words, u32 depth) {
    pm4::Reader reader{words};
    while (const std::optional<Packet> next = reader.next()) {
        ++stats_.packets;
        packet(*next, depth);
    }
    if (reader.failed()) {
        fail(reader.error());
        return false;
    }
    return true;
}

void CommandProcessor::fail(std::string message) {
    ++stats_.errors;
    WIIU_LOG_ERROR(Gpu, "pm4: {}", message);
    last_error_ = std::move(message);
}

void CommandProcessor::packet(const Packet& packet, u32 depth) {
    switch (packet.header.type) {
        case PacketType::Type0: {
            std::string error;
            if (!registers_->write_type0(packet.header.base_index, packet.payload, error)) {
                fail(std::move(error));
            }
            return;
        }
        case PacketType::Type2:
            return;
        case PacketType::Type1: // the reader refuses these before they get here
        case PacketType::Type3:
            break;
    }

    switch (static_cast<Opcode>(packet.header.opcode)) {
        case Opcode::Nop:
            // GX2 writes markers and debug strings this way. Nothing to do.
            return;

        case Opcode::HleSwap:
        case Opcode::HleClearColor:
        case Opcode::HleCopyColorToScan:
        case Opcode::HleClearDepthStencil:
            if (!hle_listener_) {
                unhandled(packet, true);
                return;
            }
            ++stats_.hle;
            hle_listener_(packet);
            return;

        case Opcode::SetConfigReg:
            set_window(packet, Window::Config);
            return;
        case Opcode::SetContextReg:
            set_window(packet, Window::Context);
            return;
        case Opcode::SetAluConst:
            set_window(packet, Window::AluConst);
            return;
        case Opcode::SetResource:
            set_window(packet, Window::Resource);
            return;
        case Opcode::SetSampler:
            set_window(packet, Window::Sampler);
            return;
        case Opcode::SetCtlConst:
            set_window(packet, Window::CtlConst);
            return;
        case Opcode::SetLoopConst:
            set_window(packet, Window::LoopConst);
            return;
        case Opcode::SetBoolConst:
            set_window(packet, Window::BoolConst);
            return;

        case Opcode::IndexType:
            index_type_ = packet.payload[0];
            return;
        case Opcode::NumInstances:
            instances_ = packet.payload[0];
            return;

        case Opcode::DrawIndexAuto:
        case Opcode::DrawIndex2:
        case Opcode::DrawIndexImmediate:
            draw(packet);
            return;

        case Opcode::IndirectBuffer:
        case Opcode::IndirectBufferPriv:
            indirect(packet, depth);
            return;

        case Opcode::MemWrite:
            mem_write(packet);
            return;
        case Opcode::EventWriteEop:
            event_write_eop(packet);
            return;

        // Recognised and not acted on yet: cache flushes and invalidations
        // (which matter once there is a texture cache, 8.13), waits on guest
        // memory, the context shadowing CONTEXT_CONTROL turns on, occlusion
        // queries and the other EVENT_WRITEs, and stream-out bookkeeping.
        case Opcode::ContextControl:
        case Opcode::SurfaceSync:
        case Opcode::EventWrite:
        case Opcode::WaitRegMem:
        case Opcode::MemSemaphore:
        case Opcode::PfpSyncMe:
        case Opcode::MeInitialize:
        case Opcode::Start3dCmdbuf:
        case Opcode::SetPredication:
        case Opcode::StreamOutBufferUpdate:
        case Opcode::StreamOutBaseUpdate:
        case Opcode::SurfaceBaseUpdate:
            unhandled(packet, true);
            return;

        // Named, and not yet anything here or in 8.2.
        case Opcode::IndirectBufferEnd:
        case Opcode::RegRmw:
        case Opcode::CondExec:
        case Opcode::PredExec:
        case Opcode::IndexBase:
        case Opcode::DrawIndexOffset:
        case Opcode::DrawIndex:
        case Opcode::DrawIndexMultiAuto:
        case Opcode::DrawIndexOffset2:
        case Opcode::CopyDw:
        case Opcode::CpDma:
        case Opcode::CondWrite:
        case Opcode::OneRegWrite:
            break;
    }
    unhandled(packet, false);
}

void CommandProcessor::set_window(const Packet& packet, Window window) {
    // The first word is the register offset, in words, from the window's
    // base; the rest are values for consecutive registers.
    std::string error;
    if (!registers_->write_window(window, packet.payload[0], packet.payload.subspan(1), error)) {
        fail(fmt::format("{} at word {}", error, packet.offset));
    }
}

void CommandProcessor::draw(const Packet& packet) {
    DrawCommand command;
    command.instances = instances_;
    command.index_type = index_type_;
    command.primitive = registers_->read(reg::vgt_primitive_type);

    const std::span<const u32> payload = packet.payload;
    switch (static_cast<Opcode>(packet.header.opcode)) {
        case Opcode::DrawIndexAuto:
            // count, initiator
            if (payload.size() < 2) {
                fail(fmt::format("DRAW_INDEX_AUTO at word {} is {} words, not 2",
                                 packet.offset,
                                 payload.size()));
                return;
            }
            command.kind = DrawCommand::Kind::Auto;
            command.count = payload[0];
            command.draw_initiator = payload[1];
            break;

        case Opcode::DrawIndex2: {
            // max indices, physical address (low, high), count, initiator
            if (payload.size() < 5) {
                fail(fmt::format(
                    "DRAW_INDEX_2 at word {} is {} words, not 5", packet.offset, payload.size()));
                return;
            }
            command.kind = DrawCommand::Kind::Indexed;
            command.count = payload[3];
            command.draw_initiator = payload[4];
            if (memory_ == nullptr ||
                !memory::to_virtual(memory_->regions(), payload[1], command.index_address)) {
                fail(fmt::format("DRAW_INDEX_2 at word {} reads indices at physical {:#010x}, "
                                 "which nothing maps",
                                 packet.offset,
                                 payload[1]));
                return;
            }
            break;
        }

        case Opcode::DrawIndexImmediate:
            // count, initiator, then the indices themselves
            if (payload.size() < 2) {
                fail(fmt::format("DRAW_INDEX_IMMD at word {} is {} words, not at least 2",
                                 packet.offset,
                                 payload.size()));
                return;
            }
            command.kind = DrawCommand::Kind::Immediate;
            command.count = payload[0];
            command.draw_initiator = payload[1];
            command.immediate_indices = payload.subspan(2);
            break;

        default:
            return;
    }

    ++stats_.draws;
    if (draw_listener_) {
        draw_listener_(command, *registers_);
    }
}

void CommandProcessor::indirect(const Packet& packet, u32 depth) {
    // physical address (low, high), size in words
    if (packet.payload.size() < 3) {
        fail(fmt::format("{} at word {} is {} words, not 3",
                         pm4::opcode_name(packet.header.opcode),
                         packet.offset,
                         packet.payload.size()));
        return;
    }
    if (depth + 1 > max_indirect_depth) {
        fail(fmt::format("indirect buffers nested more than {} deep at word {}; a cycle in "
                         "guest memory is the usual reason",
                         max_indirect_depth,
                         packet.offset));
        return;
    }

    const physaddr_t physical = packet.payload[0];
    const u32 words = packet.payload[2];

    // The size comes from the stream, so it is checked before it is
    // multiplied or allocated: a corrupt word would otherwise wrap the byte
    // count, or ask for gigabytes. 64 MiB is far past any real command buffer.
    constexpr u32 max_words = 16U << 20;
    if (words > max_words) {
        fail(fmt::format("{} at word {} calls {} words, more than any command buffer holds",
                         pm4::opcode_name(packet.header.opcode),
                         packet.offset,
                         words));
        return;
    }

    ppcaddr_t address = 0;
    const u8* host = nullptr;
    if (memory_ != nullptr && memory::to_virtual(memory_->regions(), physical, address)) {
        host = memory_->translate(address, words * 4, memory::Access::Read);
    }
    if (host == nullptr) {
        fail(fmt::format("{} at word {} calls {} words at physical {:#010x}, which is not "
                         "readable memory",
                         pm4::opcode_name(packet.header.opcode),
                         packet.offset,
                         words,
                         physical));
        return;
    }

    // GX2 writes command buffers big-endian; the reader wants host order.
    std::vector<u32> stream(words);
    std::memcpy(stream.data(), host, stream.size() * sizeof(u32));
    for (u32& word : stream) {
        word = swap_bytes(word);
    }

    ++stats_.indirect_buffers;
    static_cast<void>(execute_at_depth(stream, depth + 1));
}

bool CommandProcessor::write_back(
    const Packet& packet, u32 address_lo, u32 address_hi, u64 value, u32 bytes) {
    // The low two bits of the address word are the byte swap; the address is
    // the rest. Bits above 32 would be a 40-bit address, which the Wii U's
    // physical memory never needs.
    const physaddr_t physical = address_lo & ~3U;
    ppcaddr_t address = 0;
    if ((address_hi & 0xFF) != 0 || memory_ == nullptr ||
        !memory::to_virtual(memory_->regions(), physical, address)) {
        fail(fmt::format("{} at word {} writes to physical {:#04x}{:08x}, which nothing maps",
                         pm4::opcode_name(packet.header.opcode),
                         packet.offset,
                         address_hi & 0xFF,
                         physical));
        return false;
    }

    const std::array<u8, 8> encoded = encode_write(value, bytes, address_lo & 3U);
    if (!memory_->write_bytes(address, std::span<const u8>{encoded}.first(bytes))) {
        fail(fmt::format("{} at word {} writes to {:#010x}, which is not writable",
                         pm4::opcode_name(packet.header.opcode),
                         packet.offset,
                         address));
        return false;
    }
    ++stats_.memory_writes;
    return true;
}

void CommandProcessor::mem_write(const Packet& packet) {
    // address low (swap in bits 0-1), address high (bit 16: write the clock;
    // bit 18: 32-bit data), data low, data high
    if (packet.payload.size() < 4) {
        fail(fmt::format(
            "MEM_WRITE at word {} is {} words, not 4", packet.offset, packet.payload.size()));
        return;
    }
    const u32 address_hi = packet.payload[1];
    const bool clock = (address_hi & (1U << 16)) != 0;
    const bool data32 = (address_hi & (1U << 18)) != 0;

    u64 value = 0;
    if (clock) {
        value = clock_;
    } else if (data32) {
        value = packet.payload[2];
    } else {
        value = packet.payload[2] | (static_cast<u64>(packet.payload[3]) << 32);
    }
    static_cast<void>(write_back(packet, packet.payload[0], address_hi, value, data32 ? 4 : 8));
}

void CommandProcessor::event_write_eop(const Packet& packet) {
    // event initiator, address low (swap in bits 0-1), address high (bits
    // 24-25: interrupt select; bits 29-31: data select), data low, data high
    if (packet.payload.size() < 5) {
        fail(fmt::format(
            "EVENT_WRITE_EOP at word {} is {} words, not 5", packet.offset, packet.payload.size()));
        return;
    }
    const u32 event = packet.payload[0];
    const u32 address_hi = packet.payload[2];
    const u32 interrupt = (address_hi >> 24) & 3U;
    const u32 data = (address_hi >> 29) & 7U;

    u64 value = 0;
    switch (data) {
        case 0: // discard: nothing is written
            break;
        case 1:
            value = packet.payload[3];
            break;
        case 2:
            value = packet.payload[3] | (static_cast<u64>(packet.payload[4]) << 32);
            break;
        case 3:
            value = clock_;
            break;
        default:
            fail(fmt::format("EVENT_WRITE_EOP at word {} selects data {}, which does not exist",
                             packet.offset,
                             data));
            return;
    }
    if (data != 0) {
        static_cast<void>(
            write_back(packet, packet.payload[1], address_hi, value, data == 1 ? 4 : 8));
    }

    // 1 is an interrupt alone, 2 is one after the write is confirmed. Here
    // the write is done by the time the line above returns, so they are the
    // same thing.
    if (interrupt != 0) {
        ++stats_.interrupts;
        if (interrupt_listener_) {
            interrupt_listener_(event, value);
        }
    }
}

void CommandProcessor::unhandled(const Packet& packet, bool deferred) {
    if (deferred) {
        ++stats_.deferred;
    } else {
        ++stats_.unknown;
    }

    // Logged the first time each opcode is seen, so a frame of a thousand
    // EVENT_WRITEs is one line rather than a thousand.
    u64& seen = unhandled_[packet.header.opcode];
    if (seen++ == 0) {
        const std::string_view name = pm4::opcode_name(packet.header.opcode);
        WIIU_LOG_WARN(Gpu,
                      "pm4: {} ({:#04x}) is {}",
                      name.empty() ? std::string_view{"an unknown opcode"} : name,
                      packet.header.opcode,
                      deferred ? "recognised, and not acted on yet" : "not handled yet");
    }
}

} // namespace wiiu::latte
