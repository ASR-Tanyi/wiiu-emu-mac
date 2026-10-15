// Phase 8.3 — see command_processor.hpp.

#include <wiiu/common/endian.hpp>
#include <wiiu/common/log.hpp>
#include <wiiu/latte/command_processor.hpp>
#include <wiiu/memory/map.hpp>

#include <fmt/format.h>

#include <cstring>
#include <vector>

namespace wiiu::latte {

using pm4::Opcode;
using pm4::Packet;
using pm4::PacketType;

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

        // The ones whose effects need the GPU thread (8.2): fences and
        // timestamps written back to guest memory, waits on guest memory,
        // cache flushes, and the context shadowing CONTEXT_CONTROL turns on.
        case Opcode::ContextControl:
        case Opcode::SurfaceSync:
        case Opcode::EventWrite:
        case Opcode::EventWriteEop:
        case Opcode::MemWrite:
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
                      deferred ? "recognised, and not acted on until the GPU thread (8.2)"
                               : "not handled yet");
    }
}

} // namespace wiiu::latte
