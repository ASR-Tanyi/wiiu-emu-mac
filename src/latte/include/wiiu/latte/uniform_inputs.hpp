// Phase 8.12 — what a draw's shaders read their uniforms from.
//
// The emitter (msl.hpp) has a shader read its uniform registers from a
// `constant float4*` at `msl::uniform_registers_slot` and uniform block n
// from one at `msl::first_uniform_block_slot + n`. This finds them, from the
// registers as they stand at a draw, for the uniforms the shader's ALU reads
// (`ir::uniform_use`) and no others:
//
//   * the uniform registers are the stage's 256 ALU constants -- the pixel
//     shader's from SQ_ALU_CONSTANT0_0, the vertex shader's from
//     SQ_ALU_CONSTANT0_256, which `GX2Set…UniformReg` writes -- copied as
//     they stand, to be bound as bytes (4 KiB, exactly what Metal's
//     `set*Bytes` takes);
//   * uniform block n is where the stage's SQ_ALU_CONST_CACHE_n says, its
//     address shifted by 8, for SQ_ALU_CONST_BUFFER_SIZE_n 256-byte units --
//     the constant cache KCACHE reads through, as decaf reads it, which
//     `GX2Set…UniformBlock` writes beside the block's resource -- found in
//     guest memory on the GPU (guest_buffers.hpp) and read in place.
//
// Refused: a uniform block the shader reads that has no address or no size,
// or whose memory the GPU cannot see. A geometry shader's uniforms come with
// geometry shaders (8.8).
//
// The uniform registers are bound by value at every draw, which is what
// GX2's uniform-register mode is: a title writes them between draws. Their
// 4 KiB a stage is a copy a draw; worth narrowing to the constants a shader
// reads if a profile says so.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/latte/guest_buffers.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/shader_ir.hpp>

#include <string>
#include <vector>

namespace wiiu::latte {

/// One uniform block a draw binds.
struct UniformBlockBinding {
    u32 block = 0; ///< 0 to 15
    GuestBuffers::Range range;
};

struct UniformInputs {
    gpu::ShaderStage stage = gpu::ShaderStage::Vertex;
    std::vector<u8> registers; ///< the stage's uniform registers; empty if none is read
    std::vector<UniformBlockBinding> blocks; ///< in block order
};

/// What `shader` reads, from the registers as they stand. False, with
/// `error` saying which block and why, for one it cannot bind.
[[nodiscard]] bool uniform_inputs(const RegisterFile& registers,
                                  const ir::Shader& shader,
                                  GuestBuffers& memory,
                                  UniformInputs& out,
                                  std::string& error);

/// Bind them, inside a render pass.
void bind_uniform_inputs(gpu::GpuBackend& backend, const UniformInputs& inputs);

} // namespace wiiu::latte
