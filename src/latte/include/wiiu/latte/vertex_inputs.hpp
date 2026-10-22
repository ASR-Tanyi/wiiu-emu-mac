// Phase 8.11 — where a draw's vertices come from.
//
// The vertex shader the emitter writes fetches its own attributes (msl.hpp):
// attribute buffer n is bound as bytes at vertex buffer slot
// `msl::first_attribute_buffer_slot + n`, and each buffer's stride and size,
// with the two instance step rates, arrive in a `msl::FetchConstants` at
// `msl::fetch_constants_slot`. This reads all of that from the registers as
// they stand at a draw: for each buffer the shader's attributes name, its
// resource -- attribute buffer n is resource 0x140 + n, which
// `GX2SetAttribBuffer` wrote (8.1) -- gives where the buffer is, how big and
// its stride, and the buffer is found in guest memory on the GPU
// (guest_buffers.hpp), read in place; VGT_INSTANCE_STEP_RATE_0 and _1 give
// the step rates. Buffers the shader does not read are not bound.
//
// Refused: a buffer the shader reads whose resource is not a valid buffer,
// or whose memory the GPU cannot see.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/latte/guest_buffers.hpp>
#include <wiiu/latte/msl.hpp>
#include <wiiu/latte/registers.hpp>
#include <wiiu/latte/shader_ir.hpp>

#include <string>
#include <vector>

namespace wiiu::latte {

/// One attribute buffer a draw binds.
struct VertexBuffer {
    u32 index = 0; ///< attribute buffer 0 to 15
    GuestBuffers::Range range;
};

struct VertexInputs {
    msl::FetchConstants constants;
    std::vector<VertexBuffer> buffers; ///< in index order
};

/// What `vertex`, a vertex shader's description, reads, from the registers
/// as they stand. False, with `error` saying which buffer and why, for one
/// it cannot bind.
[[nodiscard]] bool vertex_inputs(const RegisterFile& registers,
                                 const ir::Shader& vertex,
                                 GuestBuffers& memory,
                                 VertexInputs& out,
                                 std::string& error);

/// Bind them, inside a render pass: the constants with `bind_bytes`, each
/// buffer at its slot.
void bind_vertex_inputs(gpu::GpuBackend& backend, const VertexInputs& inputs);

} // namespace wiiu::latte
