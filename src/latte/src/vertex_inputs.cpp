// Phase 8.11 — see vertex_inputs.hpp.

#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_state.hpp>
#include <wiiu/latte/vertex_inputs.hpp>

#include <fmt/format.h>

#include <array>
#include <cstring>

namespace wiiu::latte {

bool vertex_inputs(const RegisterFile& registers,
                   const ir::Shader& vertex,
                   GuestBuffers& memory,
                   VertexInputs& out,
                   std::string& error) {
    out = VertexInputs{};

    std::array<bool, resource::attribute_buffer_count> read{};
    for (const ir::VertexAttribute& attribute : vertex.attributes) {
        if (attribute.buffer >= resource::attribute_buffer_count) {
            error = fmt::format("an attribute reads attribute buffer {}, past the {} there are",
                                attribute.buffer,
                                resource::attribute_buffer_count);
            return false;
        }
        read[attribute.buffer] = true;
    }

    for (u32 index = 0; index < resource::attribute_buffer_count; ++index) {
        if (!read[index]) {
            continue;
        }
        const u32 id = resource::attribute_buffer_0 + index;
        const BufferResource buffer = buffer_resource(registers, id);
        if (!buffer.valid()) {
            error = fmt::format("attribute buffer {} (resource {:#x}) is read, and is not a valid "
                                "buffer",
                                index,
                                id);
            return false;
        }
        std::string problem;
        const std::optional<GuestBuffers::Range> range =
            memory.find(buffer.address, buffer.size, problem);
        if (!range.has_value()) {
            error = fmt::format("attribute buffer {}: {}", index, problem);
            return false;
        }
        out.constants.stride[index] = buffer.word2.stride.value();
        out.constants.size[index] = buffer.size;
        out.buffers.push_back(VertexBuffer{.index = index, .range = *range});
    }

    out.constants.step_rate = {registers.read(reg::vgt_instance_step_rate_0),
                               registers.read(reg::vgt_instance_step_rate_0 + 4)};
    return true;
}

void bind_vertex_inputs(gpu::GpuBackend& backend, const VertexInputs& inputs) {
    std::array<u8, sizeof(msl::FetchConstants)> bytes{};
    std::memcpy(bytes.data(), &inputs.constants, sizeof(inputs.constants));
    backend.bind_bytes(gpu::ShaderStage::Vertex, msl::fetch_constants_slot, bytes);
    for (const VertexBuffer& buffer : inputs.buffers) {
        backend.bind_buffer(gpu::ShaderStage::Vertex,
                            msl::first_attribute_buffer_slot + buffer.index,
                            buffer.range.buffer,
                            buffer.range.offset);
    }
}

} // namespace wiiu::latte
