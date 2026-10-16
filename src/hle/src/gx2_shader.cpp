// Phase 8.1 — GX2's shader calls. See gx2_shader.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_fetch_shader.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/pm4.hpp>
#include <wiiu/latte/render_state.hpp>
#include <wiiu/latte/shader_state.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string_view>

namespace wiiu::hle::gx2 {

namespace {

using latte::pm4::Opcode;

// Register addresses: Linux's `r600d.h` has every one.
constexpr u32 sq_config = 0x8C00;
constexpr u32 sq_esgs_ring_base = 0x8C40;      ///< then its size, GSVS's base and size
constexpr u32 sq_esgs_ring_itemsize = 0x288A8; ///< then SQ_GSVS_RING_ITEMSIZE
constexpr u32 vgt_gs_mode = 0x28A40;

// Field positions, from the same header.
constexpr u32 dx9_consts = 1U << 2;
constexpr u32 alu_inst_prefer_vector = 1U << 3;

[[nodiscard]] constexpr u32 priorities(u32 ps, u32 vs, u32 gs, u32 es) noexcept {
    return (ps << 24) | (vs << 26) | (gs << 28) | (es << 30);
}

[[nodiscard]] constexpr u32 pair(u32 low, u32 high, u32 mask) noexcept {
    return (low & mask) | ((high & mask) << 16);
}

constexpr u32 gpr_mask = 0xFF;
constexpr u32 stack_mask = 0xFFF;
constexpr u32 four_clause_temps = 4U << 28;

[[nodiscard]] constexpr u32 threads(u32 ps, u32 vs, u32 gs, u32 es) noexcept {
    return ps | (vs << 8) | (gs << 16) | (es << 24);
}

/// VGT_GS_MODE for compute: scenario G (3), compute mode in bits 14 and 15
/// -- decaf's COMPUTE_MODE and FAST_COMPUTE_MODE, Cemu's two-bit field set
/// to 3 -- and PARTIAL_THD_AT_EOI (bit 17).
constexpr u32 gs_mode_compute = 3U | (3U << 14) | (1U << 17);

/// `void GX2SetShaderModeEx(GX2ShaderMode mode, uint32_t numVsGpr, uint32_t
/// numVsStackEntries, uint32_t numGsGpr, uint32_t numGsStackEntries,
/// uint32_t numPsGpr, uint32_t numPsStackEntries)`.
void gx2_set_shader_mode_ex(Call& call,
                            u32 mode,
                            u32 vs_gprs,
                            u32 vs_stack,
                            u32 gs_gprs,
                            u32 gs_stack,
                            u32 ps_gprs,
                            u32 ps_stack) {
    if (mode > static_cast<u32>(ShaderMode::ComputeShader)) {
        WIIU_LOG_ERROR(Gx2, "GX2SetShaderModeEx({}): no such shader mode", mode);
        return;
    }
    std::vector<u32> packets;
    append_shader_mode(packets,
                       static_cast<ShaderMode>(mode),
                       ShaderResources{.vs_gprs = vs_gprs,
                                       .vs_stack = vs_stack,
                                       .gs_gprs = gs_gprs,
                                       .gs_stack = gs_stack,
                                       .ps_gprs = ps_gprs,
                                       .ps_stack = ps_stack});
    static_cast<void>(emit(call, packets));
}

using latte::reg::sq_pgm_start_es;
using latte::reg::sq_pgm_start_fs;
using latte::reg::sq_pgm_start_gs;
using latte::reg::sq_pgm_start_ps;
using latte::reg::sq_pgm_start_vs;

/// A program's first five registers: start, size, the two unnamed values
/// and its resources.
[[nodiscard]] std::array<u32, 5> program_registers(Program program, u32 resources) noexcept {
    return {program.address >> 8,
            program.size >> 3,
            program_unnamed_value,
            program_unnamed_value,
            resources};
}

/// The first `count` of `values`, held to the array.
template <std::size_t N>
[[nodiscard]] std::vector<u32> leading(const std::array<u32be, N>& values, u32 count) {
    const std::size_t n = std::min<std::size_t>(count, N);
    std::vector<u32> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = values[i].value();
    }
    return out;
}

void append_stream_out_strides(std::vector<u32>& out, const std::array<u32be, 4>& strides) {
    for (u32 i = 0; i < 4; ++i) {
        append_context_register(
            out, latte::reg::vgt_strmout_vtx_stride_0 + (16 * i), strides[i].value() >> 2);
    }
}

/// Where a shader's program is: its own pointer and size, or, when the
/// pointer is null, its GX2R buffer's. Nothing, logged, if the GPU cannot
/// see it.
[[nodiscard]] std::optional<Program>
find_program(const Call& call, u32 data, u32 size, const RBuffer& gx2r, std::string_view what) {
    ppcaddr_t address = data;
    u32 bytes = size;
    if (address == 0) {
        address = gx2r.buffer.value();
        bytes = gx2r.element_count.value() * gx2r.element_size.value();
    }
    if (address == 0 || bytes == 0) {
        WIIU_LOG_ERROR(Gx2, "{} has no program", what);
        return std::nullopt;
    }
    const std::optional<physaddr_t> physical = physical_address(call.memory, address, what);
    if (!physical.has_value()) {
        return std::nullopt;
    }
    return Program{.address = *physical, .size = bytes};
}

/// Read a shader structure; false, logged, if it is not there.
template <typename T>
[[nodiscard]] bool read_shader(const Call& call, u32 address, T& shader, std::string_view what) {
    if (!read_guest(call.memory, address, shader)) {
        WIIU_LOG_ERROR(Gx2, "{}: no shader at {:#010x}", what, address);
        return false;
    }
    return true;
}

/// Loop variables are not written yet (gx2_shader.hpp); say so once.
void report_loop_vars(u32 count, std::string_view what) {
    static bool reported = false;
    if (count != 0 && !reported) {
        reported = true;
        WIIU_LOG_WARN(Gx2, "{}: a shader with loop variables; they are not written yet", what);
    }
}

void gx2_set_fetch_shader(Call& call, u32 address) {
    FetchShader shader{};
    if (!read_shader(call, address, shader, "GX2SetFetchShader")) {
        return;
    }
    const std::optional<physaddr_t> program =
        physical_address(call.memory, shader.data.value(), "GX2SetFetchShader's program");
    if (!program.has_value()) {
        return;
    }
    std::vector<u32> packets;
    append_fetch_shader(packets, shader, *program);
    static_cast<void>(emit(call, packets));
}

void gx2_set_vertex_shader(Call& call, u32 address) {
    VertexShader shader{};
    if (!read_shader(call, address, shader, "GX2SetVertexShader")) {
        return;
    }
    const std::optional<Program> program = find_program(call,
                                                        shader.data.value(),
                                                        shader.size.value(),
                                                        shader.gx2r,
                                                        "GX2SetVertexShader's program");
    if (!program.has_value()) {
        return;
    }
    report_loop_vars(shader.loop_var_count.value(), "GX2SetVertexShader");
    std::vector<u32> packets;
    append_vertex_shader(packets, shader, *program);
    static_cast<void>(emit(call, packets));
}

void gx2_set_pixel_shader(Call& call, u32 address) {
    PixelShader shader{};
    if (!read_shader(call, address, shader, "GX2SetPixelShader")) {
        return;
    }
    const std::optional<Program> program = find_program(
        call, shader.data.value(), shader.size.value(), shader.gx2r, "GX2SetPixelShader's program");
    if (!program.has_value()) {
        return;
    }
    report_loop_vars(shader.loop_var_count.value(), "GX2SetPixelShader");
    std::vector<u32> packets;
    append_pixel_shader(packets, shader, *program);
    static_cast<void>(emit(call, packets));
}

void gx2_set_geometry_shader(Call& call, u32 address) {
    GeometryShader shader{};
    if (!read_shader(call, address, shader, "GX2SetGeometryShader")) {
        return;
    }
    const std::optional<Program> program = find_program(call,
                                                        shader.data.value(),
                                                        shader.size.value(),
                                                        shader.gx2r,
                                                        "GX2SetGeometryShader's program");
    const std::optional<Program> copy = find_program(call,
                                                     shader.vertex_shader_data.value(),
                                                     shader.vertex_shader_size.value(),
                                                     shader.gx2r_vertex_shader,
                                                     "GX2SetGeometryShader's copy shader");
    if (!program.has_value() || !copy.has_value()) {
        return;
    }
    report_loop_vars(shader.loop_var_count.value(), "GX2SetGeometryShader");
    std::vector<u32> packets;
    append_geometry_shader(packets, shader, *program, *copy);
    static_cast<void>(emit(call, packets));
}

// The register and stack counts a program asks for: SQ_PGM_RESOURCES_*'s
// low two bytes. What a title feeds GX2SetShaderModeEx with.

/// Every shader structure starts with its stage's SQ_PGM_RESOURCES.
u32 resources_field(Call& call, u32 address, u32 shift, std::string_view what) {
    u32 resources = 0;
    if (!call.memory.try_read<u32>(address, resources)) {
        WIIU_LOG_ERROR(Gx2, "{}: no shader at {:#010x}", what, address);
        return 0;
    }
    return (resources >> shift) & 0xFFU;
}

u32 gx2_get_vertex_shader_gprs(Call& call, u32 address) {
    return resources_field(call, address, 0, "GX2GetVertexShaderGPRs");
}

u32 gx2_get_vertex_shader_stack_entries(Call& call, u32 address) {
    return resources_field(call, address, 8, "GX2GetVertexShaderStackEntries");
}

u32 gx2_get_pixel_shader_gprs(Call& call, u32 address) {
    return resources_field(call, address, 0, "GX2GetPixelShaderGPRs");
}

u32 gx2_get_pixel_shader_stack_entries(Call& call, u32 address) {
    return resources_field(call, address, 8, "GX2GetPixelShaderStackEntries");
}

u32 gx2_get_geometry_shader_gprs(Call& call, u32 address) {
    return resources_field(call, address, 0, "GX2GetGeometryShaderGPRs");
}

u32 gx2_get_geometry_shader_stack_entries(Call& call, u32 address) {
    return resources_field(call, address, 8, "GX2GetGeometryShaderStackEntries");
}

/// `GX2SetVertexUniformReg(uint32_t offset, uint32_t count, const void
/// *data)` and the pixel shader's: `count` words of uniform registers from
/// word `offset` of the stage's, copied from the title's memory now.
void set_uniform_registers(
    Call& call, u32 base, u32 offset, u32 count, u32 data, std::string_view what) {
    if ((offset & 0xFFFF8000U) != 0) {
        // decaf writes a loop constant for the high half; Cemu does neither.
        WIIU_LOG_WARN(Gx2, "{}: offset {:#x} names a loop constant, not written yet", what, offset);
        return;
    }
    constexpr u32 stage_words = 256 * 4;
    if (offset + count > 2 * stage_words) {
        WIIU_LOG_ERROR(
            Gx2, "{}: {} words at {:#x} run past the uniform registers", what, count, offset);
        return;
    }
    std::vector<u32> values(count);
    for (u32 i = 0; i < count; ++i) {
        if (!call.memory.try_read<u32>(data + (4 * i), values[i])) {
            WIIU_LOG_ERROR(Gx2, "{}: the values at {:#010x} are not readable", what, data);
            return;
        }
    }
    std::vector<u32> packets;
    append_registers(packets, Opcode::SetAluConst, base + (4 * offset), values);
    static_cast<void>(emit(call, packets));
}

void gx2_set_vertex_uniform_reg(Call& call, u32 offset, u32 count, u32 data) {
    set_uniform_registers(
        call, latte::reg::sq_alu_constant0_256, offset, count, data, "GX2SetVertexUniformReg");
}

void gx2_set_pixel_uniform_reg(Call& call, u32 offset, u32 count, u32 data) {
    set_uniform_registers(
        call, latte::reg::sq_alu_constant0_0, offset, count, data, "GX2SetPixelUniformReg");
}

} // namespace

void append_fetch_shader(std::vector<u32>& out, const FetchShader& shader, physaddr_t program) {
    append_registers(out,
                     Opcode::SetContextReg,
                     sq_pgm_start_fs,
                     program_registers(Program{.address = program, .size = shader.size.value()},
                                       shader.sq_pgm_resources_fs.value()));
    const std::array<u32, 2> step_rates{shader.divisors[0].value(), shader.divisors[1].value()};
    append_registers(out, Opcode::SetContextReg, latte::reg::vgt_instance_step_rate_0, step_rates);
}

void append_vertex_shader(std::vector<u32>& out, const VertexShader& shader, Program program) {
    const auto resources = program_registers(program, shader.sq_pgm_resources_vs.value());
    if (shader.mode.value() != static_cast<u32>(ShaderMode::GeometryShader)) {
        append_registers(out, Opcode::SetContextReg, sq_pgm_start_vs, resources);
        append_context_register(
            out, latte::reg::vgt_primitiveid_en, shader.vgt_primitiveid_en.value());
        append_context_register(
            out, latte::reg::spi_vs_out_config, shader.spi_vs_out_config.value());
        append_context_register(
            out, latte::reg::pa_cl_vs_out_cntl, shader.pa_cl_vs_out_cntl.value());
        const std::vector<u32> ids =
            leading(shader.spi_vs_out_id, shader.num_spi_vs_out_id.value());
        if (!ids.empty()) {
            append_registers(out, Opcode::SetContextReg, latte::reg::spi_vs_out_id_0, ids);
        }
        append_context_register(out, latte::reg::sq_pgm_cf_offset_vs, 0);
        if (shader.has_stream_out.value() != 0) {
            append_stream_out_strides(out, shader.stream_out_stride);
        }
        append_context_register(
            out, latte::reg::vgt_strmout_buffer_en, shader.vgt_strmout_buffer_en.value());
    } else {
        // In geometry mode the vertex shader runs as the ES stage.
        append_registers(out, Opcode::SetContextReg, sq_pgm_start_es, resources);
        append_context_register(
            out, latte::reg::sq_esgs_ring_itemsize, shader.ring_item_size.value());
    }

    append_context_register(
        out, latte::reg::sq_vtx_semantic_clear, shader.sq_vtx_semantic_clear.value());
    const std::vector<u32> semantics =
        leading(shader.sq_vtx_semantic, shader.num_sq_vtx_semantic.value());
    if (!semantics.empty()) {
        append_registers(out, Opcode::SetContextReg, latte::reg::sq_vtx_semantic_0, semantics);
    }
    append_context_register(
        out, latte::reg::vgt_vertex_reuse_block_cntl, shader.vgt_vertex_reuse_block_cntl.value());
    append_context_register(
        out, latte::reg::vgt_hos_reuse_depth, shader.vgt_hos_reuse_depth.value());
}

void append_pixel_shader(std::vector<u32>& out, const PixelShader& shader, Program program) {
    const auto first = program_registers(program, shader.sq_pgm_resources_ps.value());
    const std::array<u32, 6> registers{
        first[0], first[1], first[2], first[3], first[4], shader.sq_pgm_exports_ps.value()};
    append_registers(out, Opcode::SetContextReg, sq_pgm_start_ps, registers);

    const std::array<u32, 2> input_control{shader.spi_ps_in_control_0.value(),
                                           shader.spi_ps_in_control_1.value()};
    append_registers(out, Opcode::SetContextReg, latte::reg::spi_ps_in_control_0, input_control);
    const std::vector<u32> inputs =
        leading(shader.spi_ps_input_cntls, shader.num_spi_ps_input_cntl.value());
    if (!inputs.empty()) {
        append_registers(out, Opcode::SetContextReg, latte::reg::spi_ps_input_cntl_0, inputs);
    }
    append_context_register(out, latte::reg::cb_shader_mask, shader.cb_shader_mask.value());
    append_context_register(out, latte::reg::cb_shader_control, shader.cb_shader_control.value());
    // decaf: GX2 turns dual export on, whatever the shader says. Cemu writes
    // the shader's value.
    append_context_register(out,
                            latte::reg::db_shader_control,
                            shader.db_shader_control.value() | latte::db_shader_dual_export_enable);
    append_context_register(out, latte::reg::spi_input_z, shader.spi_input_z.value());
}

void append_geometry_shader(std::vector<u32>& out,
                            const GeometryShader& shader,
                            Program program,
                            Program copy_shader) {
    append_registers(out,
                     Opcode::SetContextReg,
                     sq_pgm_start_gs,
                     program_registers(program, shader.sq_pgm_resources_gs.value()));
    append_context_register(
        out, latte::reg::vgt_gs_out_prim_type, shader.vgt_gs_out_prim_type.value());
    append_context_register(out, vgt_gs_mode, shader.vgt_gs_mode.value());
    append_context_register(
        out, latte::reg::sq_gs_vert_itemsize, shader.sq_gs_vert_itemsize.value());
    if (shader.has_stream_out.value() != 0) {
        append_stream_out_strides(out, shader.stream_out_stride);
    }
    append_context_register(
        out, latte::reg::vgt_strmout_buffer_en, shader.vgt_strmout_buffer_en.value());

    // The copy shader, as the VS stage.
    append_registers(out,
                     Opcode::SetContextReg,
                     sq_pgm_start_vs,
                     program_registers(copy_shader, shader.sq_pgm_resources_vs.value()));
    append_context_register(out, latte::reg::pa_cl_vs_out_cntl, shader.pa_cl_vs_out_cntl.value());
    const std::vector<u32> ids = leading(shader.spi_vs_out_id, shader.num_spi_vs_out_id.value());
    if (!ids.empty()) {
        append_registers(out, Opcode::SetContextReg, latte::reg::spi_vs_out_id_0, ids);
    }
    append_context_register(out, latte::reg::spi_vs_out_config, shader.spi_vs_out_config.value());
    // Cemu holds the ring item size to 15 bits; the field is no wider.
    append_context_register(
        out, latte::reg::sq_gsvs_ring_itemsize, shader.ring_item_size.value() & 0x7FFFU);
}

ShaderModeRegisters shader_mode_registers(ShaderMode mode,
                                          const ShaderResources& resources) noexcept {
    ShaderModeRegisters regs;
    regs.sq_config = alu_inst_prefer_vector;
    regs.sq_config |=
        mode == ShaderMode::ComputeShader ? priorities(0, 1, 2, 3) : priorities(3, 2, 1, 0);
    if (mode == ShaderMode::UniformRegister) {
        regs.sq_config |= dx9_consts;
    }

    switch (mode) {
        case ShaderMode::GeometryShader:
            regs.sq_gpr_resource_mgmt_1 = pair(resources.ps_gprs, 64, gpr_mask) | four_clause_temps;
            regs.sq_gpr_resource_mgmt_2 = pair(resources.gs_gprs, resources.vs_gprs, gpr_mask);
            regs.sq_thread_resource_mgmt = threads(124, 32, 8, 28);
            regs.sq_stack_resource_mgmt_1 = resources.ps_stack & stack_mask;
            regs.sq_stack_resource_mgmt_2 =
                pair(resources.gs_stack, resources.vs_stack, stack_mask);
            break;
        case ShaderMode::ComputeShader:
            regs.sq_gpr_resource_mgmt_1 = four_clause_temps;
            regs.sq_gpr_resource_mgmt_2 = pair(0, 248, gpr_mask);
            regs.sq_thread_resource_mgmt = threads(1, 1, 1, 189);
            regs.sq_stack_resource_mgmt_2 = pair(0, 256, stack_mask);
            break;
        case ShaderMode::UniformRegister:
        case ShaderMode::UniformBlock:
            regs.sq_gpr_resource_mgmt_1 =
                pair(resources.ps_gprs, resources.vs_gprs, gpr_mask) | four_clause_temps;
            regs.sq_thread_resource_mgmt = threads(136, 48, 4, 4);
            regs.sq_stack_resource_mgmt_1 =
                pair(resources.ps_stack, resources.vs_stack, stack_mask);
            break;
    }
    return regs;
}

void append_shader_mode(std::vector<u32>& out, ShaderMode mode, const ShaderResources& resources) {
    if (mode != ShaderMode::GeometryShader) {
        append_context_register(
            out, vgt_gs_mode, mode == ShaderMode::ComputeShader ? gs_mode_compute : 0U);
    }
    const ShaderModeRegisters regs = shader_mode_registers(mode, resources);
    const std::array<u32, 6> values{regs.sq_config,
                                    regs.sq_gpr_resource_mgmt_1,
                                    regs.sq_gpr_resource_mgmt_2,
                                    regs.sq_thread_resource_mgmt,
                                    regs.sq_stack_resource_mgmt_1,
                                    regs.sq_stack_resource_mgmt_2};
    append_registers(out, Opcode::SetConfigReg, sq_config, values);

    if (mode == ShaderMode::ComputeShader) {
        const std::array<u32, 4> rings{0, 0xFFFFFF, 0, 0xFFFFFF};
        append_registers(out, Opcode::SetConfigReg, sq_esgs_ring_base, rings);
        const std::array<u32, 2> item_sizes{0, 1};
        append_registers(out, Opcode::SetContextReg, sq_esgs_ring_itemsize, item_sizes);
        append_context_register(out, latte::reg::vgt_strmout_en, 0);
    }
}

void register_gx2_shader(Registry& registry) {
    registry.add("gx2", "GX2SetShaderModeEx", gx2_set_shader_mode_ex);
    registry.add("gx2", "GX2SetFetchShader", gx2_set_fetch_shader);
    registry.add("gx2", "GX2SetVertexShader", gx2_set_vertex_shader);
    registry.add("gx2", "GX2SetPixelShader", gx2_set_pixel_shader);
    registry.add("gx2", "GX2SetGeometryShader", gx2_set_geometry_shader);
    registry.add("gx2", "GX2GetVertexShaderGPRs", gx2_get_vertex_shader_gprs);
    registry.add("gx2", "GX2GetVertexShaderStackEntries", gx2_get_vertex_shader_stack_entries);
    registry.add("gx2", "GX2GetPixelShaderGPRs", gx2_get_pixel_shader_gprs);
    registry.add("gx2", "GX2GetPixelShaderStackEntries", gx2_get_pixel_shader_stack_entries);
    registry.add("gx2", "GX2GetGeometryShaderGPRs", gx2_get_geometry_shader_gprs);
    registry.add("gx2", "GX2GetGeometryShaderStackEntries", gx2_get_geometry_shader_stack_entries);
    registry.add("gx2", "GX2SetVertexUniformReg", gx2_set_vertex_uniform_reg);
    registry.add("gx2", "GX2SetPixelUniformReg", gx2_set_pixel_uniform_reg);
    register_gx2_fetch_shader(registry);
}

} // namespace wiiu::hle::gx2
