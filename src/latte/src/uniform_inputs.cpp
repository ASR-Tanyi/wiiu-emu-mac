// Phase 8.12 — see uniform_inputs.hpp.

#include <wiiu/latte/msl.hpp>
#include <wiiu/latte/resources.hpp>
#include <wiiu/latte/shader_state.hpp>
#include <wiiu/latte/uniform_inputs.hpp>

#include <fmt/format.h>

#include <cstring>
#include <optional>

namespace wiiu::latte {

namespace {

struct StageRegisters {
    u32 constants = 0; ///< the first uniform register's first word
    u32 cache = 0;     ///< SQ_ALU_CONST_CACHE_*_0
    u32 size = 0;      ///< SQ_ALU_CONST_BUFFER_SIZE_*_0
    std::string_view name;
};

[[nodiscard]] StageRegisters stage_registers(ir::Stage stage) noexcept {
    if (stage == ir::Stage::Vertex) {
        return {reg::sq_alu_constant0_256,
                reg::sq_alu_const_cache_vs_0,
                reg::sq_alu_const_buffer_size_vs_0,
                "VS"};
    }
    return {reg::sq_alu_constant0_0,
            reg::sq_alu_const_cache_ps_0,
            reg::sq_alu_const_buffer_size_ps_0,
            "PS"};
}

} // namespace

bool uniform_inputs(const RegisterFile& registers,
                    const ir::Shader& shader,
                    GuestBuffers& memory,
                    UniformInputs& out,
                    std::string& error) {
    out = UniformInputs{};
    out.stage =
        shader.stage == ir::Stage::Vertex ? gpu::ShaderStage::Vertex : gpu::ShaderStage::Fragment;
    const ir::UniformUse use = ir::uniform_use(shader);
    const StageRegisters regs = stage_registers(shader.stage);

    if (use.registers) {
        constexpr u32 words = uniform_registers_per_stage * 4;
        out.registers.resize(std::size_t{words} * 4);
        for (u32 word = 0; word < words; ++word) {
            const u32 value = registers.read(regs.constants + (4 * word));
            std::memcpy(out.registers.data() + (std::size_t{word} * 4), &value, 4);
        }
    }

    for (u32 block = 0; block < use.blocks.size(); ++block) {
        if (!use.blocks[block]) {
            continue;
        }
        const u32 cache = registers.read(regs.cache + (4 * block));
        const u32 units = registers.read(regs.size + (4 * block)) & alu_const_buffer_size_mask;
        const physaddr_t address = cache << 8;
        const u32 size = units << 8;
        if (address == 0 || size == 0) {
            error = fmt::format("uniform block {} is read, and none is bound "
                                "(SQ_ALU_CONST_CACHE_{}_{} is {:#x}, its size {} bytes)",
                                block,
                                regs.name,
                                block,
                                cache,
                                size);
            return false;
        }
        std::string problem;
        const std::optional<GuestBuffers::Range> range = memory.find(address, size, problem);
        if (!range.has_value()) {
            error = fmt::format("uniform block {}: {}", block, problem);
            return false;
        }
        out.blocks.push_back(UniformBlockBinding{.block = block, .range = *range});
    }
    return true;
}

void bind_uniform_inputs(gpu::GpuBackend& backend, const UniformInputs& inputs) {
    if (!inputs.registers.empty()) {
        backend.bind_bytes(inputs.stage, msl::uniform_registers_slot, inputs.registers);
    }
    for (const UniformBlockBinding& block : inputs.blocks) {
        backend.bind_buffer(inputs.stage,
                            msl::first_uniform_block_slot + block.block,
                            block.range.buffer,
                            block.range.offset);
    }
}

} // namespace wiiu::latte
