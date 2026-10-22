// Phase 8.2's last piece (P22) — guest RAM as GPU buffers, without a copy.
//
// Latte reads its vertices, indices and uniform blocks straight out of
// memory, by physical address, and on unified memory the GPU here can do the
// same: the bytes the title wrote are wrapped as a Metal buffer in place
// (`GpuBackend::import_host_memory`), and a draw binds that buffer at the
// offset its data starts. Nothing is staged, nothing uploaded, and nothing
// has to notice the title writing the data again -- the next draw reads
// whatever is there, as Latte would. GX2's buffers carry their own
// synchronisation (the title invalidates and waits on timestamps), which is
// what makes reading them in place sound; it is not how surfaces work
// (ADR-006), and they are not done this way.
//
// Memory is wrapped in windows rather than whole regions, made the first
// time something inside one is asked for: 256 MiB each, one starting every
// 128 MiB of a region, so that any range up to 128 MiB lies whole inside
// one -- the window its start falls in -- and no single buffer is larger
// than a GPU wants to map. Windows overlap; wrapping the same bytes twice
// costs a mapping, not a copy. A range longer than that, which no vertex or
// uniform buffer is, is refused.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/memory/memory.hpp>

#include <map>
#include <optional>
#include <string>

namespace wiiu::latte {

class GuestBuffers {
public:
    /// How much each window wraps, and how far apart they start.
    static constexpr u64 window_bytes = 256ULL << 20;
    static constexpr u64 window_step = 128ULL << 20;

    /// The longest range `find` serves.
    static constexpr u64 max_range = window_step;

    /// Where some bytes are on the GPU: a buffer, and how far into it.
    struct Range {
        gpu::BufferHandle buffer;
        u64 offset = 0;
    };

    GuestBuffers(gpu::GpuBackend& backend, const memory::Memory& memory)
        : backend_(&backend), memory_(&memory) {}

    ~GuestBuffers() { clear(); }

    GuestBuffers(const GuestBuffers&) = delete;
    GuestBuffers& operator=(const GuestBuffers&) = delete;

    /// `size` bytes from physical `address`, as a buffer and an offset:
    /// the window they lie in, wrapped the first time. Nothing, with `error`
    /// saying why, for bytes no region of RAM holds whole, a range longer
    /// than `max_range`, or a window the backend would not wrap.
    [[nodiscard]] std::optional<Range> find(physaddr_t address, u64 size, std::string& error);

    /// Every window let go -- before the memory goes, or the backend.
    void clear();

    /// Windows wrapped so far.
    [[nodiscard]] std::size_t size() const noexcept { return windows_.size(); }

private:
    gpu::GpuBackend* backend_;
    const memory::Memory* memory_;
    std::map<ppcaddr_t, gpu::BufferHandle> windows_; ///< by the window's first guest address
};

} // namespace wiiu::latte
