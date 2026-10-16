// Phase 8.3 — see pm4.hpp.

#include <wiiu/latte/pm4.hpp>

#include <fmt/format.h>

namespace wiiu::latte::pm4 {

namespace {

constexpr u32 kCountMask = 0x3FFF;

} // namespace

std::string_view opcode_name(u8 opcode) noexcept {
    switch (static_cast<Opcode>(opcode)) {
        case Opcode::Nop:
            return "NOP";
        case Opcode::HleSwap:
            return "HLE_SWAP";
        case Opcode::HleClearColor:
            return "HLE_CLEAR_COLOR";
        case Opcode::HleCopyColorToScan:
            return "HLE_COPY_COLOR_TO_SCAN";
        case Opcode::HleClearDepthStencil:
            return "HLE_CLEAR_DEPTH_STENCIL";
        case Opcode::IndirectBufferEnd:
            return "INDIRECT_BUFFER_END";
        case Opcode::SetPredication:
            return "SET_PREDICATION";
        case Opcode::RegRmw:
            return "REG_RMW";
        case Opcode::CondExec:
            return "COND_EXEC";
        case Opcode::PredExec:
            return "PRED_EXEC";
        case Opcode::Start3dCmdbuf:
            return "START_3D_CMDBUF";
        case Opcode::IndexBase:
            return "INDEX_BASE";
        case Opcode::DrawIndex2:
            return "DRAW_INDEX_2";
        case Opcode::ContextControl:
            return "CONTEXT_CONTROL";
        case Opcode::DrawIndexOffset:
            return "DRAW_INDEX_OFFSET";
        case Opcode::IndexType:
            return "INDEX_TYPE";
        case Opcode::DrawIndex:
            return "DRAW_INDEX";
        case Opcode::DrawIndexAuto:
            return "DRAW_INDEX_AUTO";
        case Opcode::DrawIndexImmediate:
            return "DRAW_INDEX_IMMD";
        case Opcode::NumInstances:
            return "NUM_INSTANCES";
        case Opcode::DrawIndexMultiAuto:
            return "DRAW_INDEX_MULTI_AUTO";
        case Opcode::IndirectBufferPriv:
            return "INDIRECT_BUFFER_PRIV";
        case Opcode::StreamOutBufferUpdate:
            return "STRMOUT_BUFFER_UPDATE";
        case Opcode::DrawIndexOffset2:
            return "DRAW_INDEX_OFFSET_2";
        case Opcode::MemSemaphore:
            return "MEM_SEMAPHORE";
        case Opcode::CopyDw:
            return "COPY_DW";
        case Opcode::WaitRegMem:
            return "WAIT_REG_MEM";
        case Opcode::MemWrite:
            return "MEM_WRITE";
        case Opcode::IndirectBuffer:
            return "INDIRECT_BUFFER";
        case Opcode::CpDma:
            return "CP_DMA";
        case Opcode::PfpSyncMe:
            return "PFP_SYNC_ME";
        case Opcode::SurfaceSync:
            return "SURFACE_SYNC";
        case Opcode::MeInitialize:
            return "ME_INITIALIZE";
        case Opcode::CondWrite:
            return "COND_WRITE";
        case Opcode::EventWrite:
            return "EVENT_WRITE";
        case Opcode::EventWriteEop:
            return "EVENT_WRITE_EOP";
        case Opcode::OneRegWrite:
            return "ONE_REG_WRITE";
        case Opcode::LoadConfigReg:
            return "LOAD_CONFIG_REG";
        case Opcode::LoadContextReg:
            return "LOAD_CONTEXT_REG";
        case Opcode::LoadAluConst:
            return "LOAD_ALU_CONST";
        case Opcode::LoadBoolConst:
            return "LOAD_BOOL_CONST";
        case Opcode::LoadLoopConst:
            return "LOAD_LOOP_CONST";
        case Opcode::LoadResource:
            return "LOAD_RESOURCE";
        case Opcode::LoadSampler:
            return "LOAD_SAMPLER";
        case Opcode::LoadCtlConst:
            return "LOAD_CTL_CONST";
        case Opcode::SetConfigReg:
            return "SET_CONFIG_REG";
        case Opcode::SetContextReg:
            return "SET_CONTEXT_REG";
        case Opcode::SetAluConst:
            return "SET_ALU_CONST";
        case Opcode::SetBoolConst:
            return "SET_BOOL_CONST";
        case Opcode::SetLoopConst:
            return "SET_LOOP_CONST";
        case Opcode::SetResource:
            return "SET_RESOURCE";
        case Opcode::SetSampler:
            return "SET_SAMPLER";
        case Opcode::SetCtlConst:
            return "SET_CTL_CONST";
        case Opcode::StreamOutBaseUpdate:
            return "STRMOUT_BASE_UPDATE";
        case Opcode::SurfaceBaseUpdate:
            return "SURFACE_BASE_UPDATE";
        case Opcode::SetAllContexts:
            return "SET_ALL_CONTEXTS";
    }
    return {};
}

Header decode_header(u32 word) noexcept {
    Header header;
    header.type = static_cast<PacketType>(word >> 30);
    switch (header.type) {
        case PacketType::Type0:
            header.base_index = word & 0xFFFF;
            header.payload_words = ((word >> 16) & kCountMask) + 1;
            break;
        case PacketType::Type3:
            header.predicate = (word & 1U) != 0;
            header.opcode = static_cast<u8>((word >> 8) & 0xFF);
            header.payload_words = ((word >> 16) & kCountMask) + 1;
            break;
        case PacketType::Type1:
        case PacketType::Type2:
            break;
    }
    return header;
}

u32 type0_header(u32 base_index, u32 registers) noexcept {
    // A Type 0 packet writes at least one register; `registers - 1` is what
    // the header holds.
    const u32 count = registers == 0 ? 0 : registers - 1;
    return ((count & kCountMask) << 16) | (base_index & 0xFFFF);
}

u32 type3_header(Opcode opcode, u32 payload_words, bool predicate) noexcept {
    const u32 count = payload_words == 0 ? 0 : payload_words - 1;
    return (3U << 30) | ((count & kCountMask) << 16) | (static_cast<u32>(opcode) << 8) |
           (predicate ? 1U : 0U);
}

std::optional<Packet> Reader::next() {
    if (failed() || position_ >= words_.size()) {
        return std::nullopt;
    }

    Packet packet;
    packet.offset = position_;
    packet.header = decode_header(words_[position_]);

    if (packet.header.type == PacketType::Type1) {
        error_ = fmt::format("word {} ({:#010x}) is a Type 1 header, which Latte does not have",
                             position_,
                             words_[position_]);
        return std::nullopt;
    }

    const std::size_t available = words_.size() - position_ - 1;
    if (packet.header.payload_words > available) {
        error_ = fmt::format("the packet at word {} ({:#010x}) wants {} words of payload and the "
                             "stream has {} left",
                             position_,
                             words_[position_],
                             packet.header.payload_words,
                             available);
        return std::nullopt;
    }

    packet.payload = words_.subspan(position_ + 1, packet.header.payload_words);
    position_ += 1 + packet.header.payload_words;
    return packet;
}

std::string describe(const Packet& packet) {
    switch (packet.header.type) {
        case PacketType::Type0:
            return fmt::format("@{} type0 reg {:#07x} x{}",
                               packet.offset,
                               packet.header.base_index * 4,
                               packet.header.payload_words);
        case PacketType::Type2:
            return fmt::format("@{} type2 filler", packet.offset);
        case PacketType::Type3: {
            const std::string_view name = opcode_name(packet.header.opcode);
            return fmt::format("@{} {} ({:#04x}) {} words{}",
                               packet.offset,
                               name.empty() ? std::string_view{"unknown"} : name,
                               packet.header.opcode,
                               packet.header.payload_words,
                               packet.header.predicate ? " predicated" : "");
        }
        case PacketType::Type1:
            break;
    }
    return fmt::format("@{} type1", packet.offset);
}

} // namespace wiiu::latte::pm4
