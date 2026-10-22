// Phase 8.2's last piece (P22) — see guest_buffers.hpp.

#include <wiiu/latte/guest_buffers.hpp>
#include <wiiu/memory/map.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace wiiu::latte {

std::optional<GuestBuffers::Range>
GuestBuffers::find(physaddr_t address, u64 size, std::string& error) {
    if (size == 0 || size > max_range) {
        error = fmt::format("{} bytes at {:#010x} is not a range a buffer is wrapped for (1 to {})",
                            size,
                            address,
                            max_range);
        return std::nullopt;
    }
    ppcaddr_t virtual_address = 0;
    if (!memory::to_virtual(memory_->regions(), address, virtual_address)) {
        error = fmt::format("physical {:#010x} is in no region of RAM", address);
        return std::nullopt;
    }
    const memory::Region* const region = memory_->region_of(virtual_address);
    if (region == nullptr || region->backing == memory::Backing::None) {
        error = fmt::format("physical {:#010x} is in no region of RAM", address);
        return std::nullopt;
    }
    const u64 into_region = virtual_address - region->base;
    if (into_region + size > region->size) {
        error =
            fmt::format("{} bytes at {:#010x} run past the end of {}", size, address, region->name);
        return std::nullopt;
    }

    // The window the range starts in holds all of it: it starts less than a
    // step before the range and runs two steps, or to the end of the region,
    // which the range does not pass.
    const u64 index = into_region / window_step;
    const ppcaddr_t start = region->base + static_cast<u32>(index * window_step);
    auto window = windows_.find(start);
    if (window == windows_.end()) {
        const u64 length = std::min(window_bytes, region->size - (index * window_step));
        const std::string label = fmt::format("guest {} {:#010x}", region->name, start);
        const gpu::BufferHandle buffer =
            backend_->import_host_memory(memory_->translate_unchecked(start), length, label);
        if (!buffer.valid()) {
            error = fmt::format("the backend would not wrap {} bytes of {} at {:#010x}",
                                length,
                                region->name,
                                start);
            return std::nullopt;
        }
        window = windows_.emplace(start, buffer).first;
    }
    return Range{.buffer = window->second, .offset = virtual_address - start};
}

void GuestBuffers::clear() {
    for (const auto& [start, buffer] : windows_) {
        backend_->destroy(buffer);
    }
    windows_.clear();
}

} // namespace wiiu::latte
