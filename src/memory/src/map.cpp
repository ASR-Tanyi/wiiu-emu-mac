#include <wiiu/memory/map.hpp>

#include <algorithm>
#include <array>

namespace wiiu::memory {
namespace {

// Ordered by base address. The sizes are the documented ones; where a region
// is documented with two sizes (foreground versus background application), the
// foreground figure is used, since that is what a running game gets.
constexpr std::array<Region, 8> kDefaultRegions{{
    {"system", system_libraries_base, 0x00800000, 0x32000000, Backing::Mem2, read_execute},
    {"codegen", codegen_base, 0x00020000, 0, Backing::None, read_write_execute},
    {"application",
     application_base,
     application_end - application_base,
     0x72000000,
     Backing::Mem2,
     read_write_execute},
    {"data", data_heap_base, 0x52000000, 0x20000000, Backing::Mem2, read_write},
    {"hardware", hardware_base, 0x04000000, 0x14000000, Backing::Mem2, read_write},
    {"mem1", mem1_base, mem1_size, mem1_physical, Backing::Mem1, read_write},
    {"shared", shared_data_base, 0x03000000, 0x18000000, Backing::Mem2, read_only},
    {"kernel", kernel_heap_base, 0x00080000, 0x1B800000, Backing::Mem2, read_write},
}};

} // namespace

std::span<const Region> default_regions() noexcept {
    return kDefaultRegions;
}

const Region* find_region(std::span<const Region> regions, ppcaddr_t address) noexcept {
    const auto it = std::ranges::find_if(
        regions, [address](const Region& region) { return region.contains(address); });
    return it == regions.end() ? nullptr : &*it;
}

bool to_physical(const Region& region, ppcaddr_t address, physaddr_t& out) noexcept {
    if (region.backing == Backing::None || !region.contains(address)) {
        return false;
    }
    out = region.physical + (address - region.base);
    return true;
}

std::string_view backing_name(Backing backing) noexcept {
    switch (backing) {
        case Backing::Mem0:
            return "MEM0";
        case Backing::Mem1:
            return "MEM1";
        case Backing::Mem2:
            return "MEM2";
        case Backing::None:
            return "none";
    }
    return "?";
}

} // namespace wiiu::memory
