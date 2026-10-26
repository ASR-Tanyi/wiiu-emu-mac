// Phase 8.1 — GX2R's buffers: memory GX2 allocates for the title, through
// an allocator the title may give it, and the locking and invalidation
// around the CPU writing it. See gx2_buffers.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_buffers.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>

#include <array>
#include <optional>
#include <string_view>

namespace wiiu::hle::gx2 {

namespace {

using wiiu::log::Channel;

[[nodiscard]] Gpu& gpu() {
    return coreinit::detail::active_context()->gpu;
}

/// The GX2RBuffer at `address`, or nothing, logged.
[[nodiscard]] std::optional<RBuffer>
read_r_buffer(const Call& call, ppcaddr_t address, std::string_view what) {
    RBuffer r{};
    if (address == 0 || !read_guest(call.memory, address, r)) {
        WIIU_LOG_ERROR(Gx2, "{}: no GX2RBuffer at {:#010x}", what, address);
        return std::nullopt;
    }
    return r;
}

[[nodiscard]] u32 r_size(const RBuffer& r) noexcept {
    return r.element_size.value() * r.element_count.value();
}

void write_flags(Call& call, ppcaddr_t buffer, u32 flags) {
    static_cast<void>(call.memory.try_write<u32>(buffer + r_buffer_flags, flags));
}

void write_pointer(Call& call, ppcaddr_t buffer, ppcaddr_t pointer) {
    static_cast<void>(call.memory.try_write<u32>(buffer + r_buffer_pointer, pointer));
}

/// GX2RInvalidateMemory's choice of what to invalidate, from a buffer's
/// flags: its bindings, the CPU's cache for a buffer the CPU uses, and either
/// half turned off by the option flags (decaf's mapping).
[[nodiscard]] u32 invalidate_mode(u32 flags) noexcept {
    u32 mode = 0;
    if ((flags & r_flags::bind_texture) != 0) {
        mode |= invalidate::texture;
    }
    if ((flags & r_flags::bind_color_buffer) != 0) {
        mode |= invalidate::color_buffer;
    }
    if ((flags & r_flags::bind_depth_buffer) != 0) {
        mode |= invalidate::depth_buffer;
    }
    if ((flags & r_flags::bind_scan_buffer) != 0) {
        mode |= invalidate::color_buffer | invalidate::depth_buffer;
    }
    if ((flags & (r_flags::bind_vertex_buffer | r_flags::bind_index_buffer)) != 0) {
        mode |= invalidate::attribute_buffer;
    }
    if ((flags & r_flags::bind_uniform_block) != 0) {
        mode |= invalidate::uniform_block;
    }
    if ((flags & r_flags::bind_shader_program) != 0) {
        mode |= invalidate::shader;
    }
    if ((flags & r_flags::bind_stream_output) != 0) {
        mode |= invalidate::stream_out_buffer;
    }
    if ((flags & (r_flags::usage_cpu_read | r_flags::usage_cpu_write)) != 0) {
        mode |= invalidate::cpu;
    }
    if ((flags & r_flags::disable_cpu_invalidate) != 0) {
        mode &= ~invalidate::cpu;
    }
    if ((flags & r_flags::disable_gpu_invalidate) != 0) {
        mode &= invalidate::cpu;
    }
    return mode;
}

/// `void GX2RInvalidateMemory(GX2RResourceFlags, void*, uint32_t)`.
void invalidate_memory(Call& call, u32 flags, ppcaddr_t buffer, u32 size) {
    invalidate_range(call, invalidate_mode(flags), buffer, size);
}

// --- the allocator --------------------------------------------------------------

/// `void GX2RSetAllocator(GX2RAllocFunc, GX2RFreeFunc)`.
void gx2r_set_allocator(Call&, u32 alloc, u32 free) {
    gpu().set_r_allocator(Gpu::RAllocator{.alloc = alloc, .free = free});
}

/// `BOOL GX2RIsUserMemory(GX2RResourceFlags)`.
u32 gx2r_is_user_memory(Call&, u32 flags) {
    return (flags & r_flags::gx2r_allocated) == 0 ? 1U : 0U;
}

/// `uint32_t GX2RGetBufferAlignment(GX2RResourceFlags)`.
u32 gx2r_get_buffer_alignment(Call&, u32 flags) {
    return r_alignment(flags);
}

/// `uint32_t GX2RGetBufferAllocationSize(GX2RBuffer*)`.
u32 gx2r_get_buffer_allocation_size(Call& call, u32 buffer) {
    const std::optional<RBuffer> r = read_r_buffer(call, buffer, "GX2RGetBufferAllocationSize");
    return r.has_value() ? r_allocation_size(r_size(*r)) : 0;
}

// --- buffers --------------------------------------------------------------------

/// `BOOL GX2RCreateBuffer(GX2RBuffer*)`: memory from the allocator, through
/// a guest call -- the title's allocator, or GX2R's own, which goes through
/// coreinit's pointer to the default heap's allocator. A raw handler, for
/// that.
void gx2r_create_buffer(Call& call) {
    const ppcaddr_t buffer = call.next_word();
    const std::optional<RBuffer> r = read_r_buffer(call, buffer, "GX2RCreateBuffer");
    if (!r.has_value()) {
        call.state.gpr[3] = 0;
        return;
    }
    const u32 flags = (r->flags.value() & ~r_flags::locked) | r_flags::gx2r_allocated;
    write_flags(call, buffer, flags);
    const u32 size = r_allocation_size(r_size(*r));
    const u32 alignment = r_alignment(flags);

    auto created = [buffer, size](Call& back, u32 address) {
        write_pointer(back, buffer, address);
        if (address == 0) {
            WIIU_LOG_WARN(Gx2, "GX2RCreateBuffer: the allocator had no {:#x} bytes", size);
        }
        back.state.gpr[3] = address != 0 ? 1U : 0U;
    };

    const Gpu::RAllocator allocator = gpu().r_allocator();
    if (allocator.alloc != 0) {
        const std::array<u32, 3> arguments{flags, size, alignment};
        coreinit::call_guest(call, allocator.alloc, arguments, created);
        return;
    }
    const ppcaddr_t cell = coreinit::detail::active_context()->default_heap_cells.alloc_ex;
    u32 function = 0;
    static_cast<void>(call.memory.try_read<u32>(cell, function));
    const std::array<u32, 2> arguments{size, alignment};
    coreinit::call_guest(call, function, arguments, created);
}

/// `BOOL GX2RCreateBufferUserMemory(GX2RBuffer*, void*, uint32_t)`: the
/// title's own memory, which GX2R will not free.
u32 gx2r_create_buffer_user_memory(Call& call, u32 buffer, u32 memory, u32 size) {
    static_cast<void>(size);
    const std::optional<RBuffer> r = read_r_buffer(call, buffer, "GX2RCreateBufferUserMemory");
    if (!r.has_value()) {
        return 0;
    }
    write_flags(call, buffer, r->flags.value() & ~(r_flags::locked | r_flags::gx2r_allocated));
    write_pointer(call, buffer, memory);
    return 1;
}

/// `void GX2RDestroyBufferEx(GX2RBuffer*, GX2RResourceFlags)`: back to the
/// allocator, unless it was the title's memory or the flags say not to.
/// The free function is given the buffer's own flags (Cemu's), not just the
/// option flags (decaf's): WHB's picks the heap to free to by its bindings.
void gx2r_destroy_buffer_ex(Call& call) {
    const ppcaddr_t buffer = call.next_word();
    const u32 options = call.next_word() & r_flags::option_mask;
    const std::optional<RBuffer> r = read_r_buffer(call, buffer, "GX2RDestroyBufferEx");
    if (!r.has_value() || r->buffer.value() == 0) {
        return;
    }
    const ppcaddr_t address = r->buffer.value();
    const u32 flags = r->flags.value();
    write_pointer(call, buffer, 0);
    if ((flags & r_flags::gx2r_allocated) == 0 || (options & r_flags::destroy_no_free) != 0) {
        return;
    }

    const auto freed = [](Call&, u32) {};
    const Gpu::RAllocator allocator = gpu().r_allocator();
    if (allocator.free != 0) {
        const std::array<u32, 2> arguments{flags | options, address};
        coreinit::call_guest(call, allocator.free, arguments, freed);
        return;
    }
    const ppcaddr_t cell = coreinit::detail::active_context()->default_heap_cells.free;
    u32 function = 0;
    static_cast<void>(call.memory.try_read<u32>(cell, function));
    const std::array<u32, 1> arguments{address};
    coreinit::call_guest(call, function, arguments, freed);
}

/// `BOOL GX2RBufferExists(GX2RBuffer*)`.
u32 gx2r_buffer_exists(Call& call, u32 buffer) {
    RBuffer r{};
    return buffer != 0 && read_guest(call.memory, buffer, r) && r.buffer.value() != 0 ? 1U : 0U;
}

/// `void GX2RSetBufferName(GX2RBuffer*, const char*)`: for GX2's debugging
/// tools, which are not here.
void gx2r_set_buffer_name(Call&, u32, u32) {}

/// `void* GX2RLockBufferEx(GX2RBuffer*, GX2RResourceFlags)`: the memory,
/// marked as locked, read-only if the flags say so (decaf's bookkeeping).
u32 gx2r_lock_buffer_ex(Call& call, u32 buffer, u32 flags) {
    const std::optional<RBuffer> r = read_r_buffer(call, buffer, "GX2RLockBufferEx");
    if (!r.has_value()) {
        return 0;
    }
    const u32 all = r->flags.value() | (flags & r_flags::option_mask);
    write_flags(
        call, buffer, r->flags.value() | r_flags::locked | (all & r_flags::locked_read_only));
    return r->buffer.value();
}

/// `void GX2RUnlockBufferEx(GX2RBuffer*, GX2RResourceFlags)`: what the CPU
/// wrote published -- the buffer invalidated for what it is bound as --
/// unless it was locked to be read.
void gx2r_unlock_buffer_ex(Call& call, u32 buffer, u32 flags) {
    const std::optional<RBuffer> r = read_r_buffer(call, buffer, "GX2RUnlockBufferEx");
    if (!r.has_value()) {
        return;
    }
    const u32 own = r->flags.value();
    if ((own & r_flags::locked_read_only) == 0) {
        invalidate_memory(call,
                          (flags & r_flags::option_mask) | (own & ~r_flags::option_mask),
                          r->buffer.value(),
                          r_size(*r));
    }
    write_flags(call, buffer, own & ~(r_flags::locked | r_flags::locked_read_only));
}

/// `void GX2RInvalidateBuffer(GX2RBuffer*, GX2RResourceFlags)`.
void gx2r_invalidate_buffer(Call& call, u32 buffer, u32 flags) {
    const std::optional<RBuffer> r = read_r_buffer(call, buffer, "GX2RInvalidateBuffer");
    if (r.has_value()) {
        invalidate_memory(call,
                          (flags & r_flags::option_mask) |
                              (r->flags.value() & ~r_flags::option_mask),
                          r->buffer.value(),
                          r_size(*r));
    }
}

void gx2r_invalidate_memory(Call& call, u32 flags, u32 buffer, u32 size) {
    invalidate_memory(call, flags, buffer, size);
}

} // namespace

u32 r_alignment(u32 flags) noexcept {
    // Cemu's table. decaf's is 256 for everything, which no title can tell
    // apart except by the memory it costs.
    constexpr u32 wide = r_flags::bind_stream_output | r_flags::bind_uniform_block |
                         r_flags::bind_shader_program | r_flags::bind_gs_ring;
    constexpr u32 narrow =
        r_flags::bind_vertex_buffer | r_flags::bind_index_buffer | r_flags::bind_display_list;
    if ((flags & wide) != 0) {
        return 0x100;
    }
    if ((flags & narrow) != 0) {
        return 0x40;
    }
    return 0x100;
}

void register_gx2r_buffers(Registry& registry) {
    registry.add("gx2", "GX2RSetAllocator", gx2r_set_allocator);
    registry.add("gx2", "GX2RIsUserMemory", gx2r_is_user_memory);
    registry.add("gx2", "GX2RGetBufferAlignment", gx2r_get_buffer_alignment);
    registry.add("gx2", "GX2RGetBufferAllocationSize", gx2r_get_buffer_allocation_size);
    registry.add("gx2", "GX2RCreateBuffer", gx2r_create_buffer);
    registry.add("gx2", "GX2RCreateBufferUserMemory", gx2r_create_buffer_user_memory);
    registry.add("gx2", "GX2RDestroyBufferEx", gx2r_destroy_buffer_ex);
    registry.add("gx2", "GX2RBufferExists", gx2r_buffer_exists);
    registry.add("gx2", "GX2RSetBufferName", gx2r_set_buffer_name);
    registry.add("gx2", "GX2RLockBufferEx", gx2r_lock_buffer_ex);
    registry.add("gx2", "GX2RUnlockBufferEx", gx2r_unlock_buffer_ex);
    registry.add("gx2", "GX2RInvalidateBuffer", gx2r_invalidate_buffer);
    registry.add("gx2", "GX2RInvalidateMemory", gx2r_invalidate_memory);
}

} // namespace wiiu::hle::gx2
