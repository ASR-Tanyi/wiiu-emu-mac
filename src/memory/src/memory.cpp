#include <wiiu/common/bits.hpp>
#include <wiiu/common/log.hpp>
#include <wiiu/memory/memory.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace wiiu::memory {
namespace {

using wiiu::log::Channel;

[[nodiscard]] constexpr u8 encode(Access access) noexcept {
    return static_cast<u8>(access);
}

} // namespace

bool Memory::initialize(std::span<const Region> regions, std::string& error) {
    if (space_.valid()) {
        error = "memory already initialised";
        return false;
    }

    if (!space_.reserve(error)) {
        return false;
    }

    pages_.assign(static_cast<std::size_t>(page_count), 0);
    regions_.assign(regions.begin(), regions.end());
    devices_.clear();

    for (const Region& region : regions_) {
        if (!space_.commit(region.base, region.size, region.access, error)) {
            shutdown();
            return false;
        }
        set_page_range(region.base, region.size, encode(region.access));

        WIIU_LOG_DEBUG(Mem,
                       "{:<12} {:#010x}-{:#010x} {:>6} KiB  {}",
                       region.name,
                       region.base,
                       region.end() - 1,
                       region.size / 1024,
                       backing_name(region.backing));
    }

    WIIU_LOG_INFO(Mem, "address space ready: {} regions", regions_.size());
    return true;
}

void Memory::shutdown() noexcept {
    space_.release();
    pages_.clear();
    regions_.clear();
    devices_.clear();
    watchpoints_.clear();
}

void Memory::set_page_range(ppcaddr_t base, u32 size, u8 entry) noexcept {
    const u64 first = base >> page_shift;
    const u64 last = (static_cast<u64>(base) + size + page_size - 1) >> page_shift;
    for (u64 page = first; page < last && page < page_count; ++page) {
        pages_[static_cast<std::size_t>(page)] = entry;
    }
}

Access Memory::access_of(ppcaddr_t address) const noexcept {
    if (pages_.empty()) {
        return Access::None;
    }
    return static_cast<Access>(page_entry(address) & ~kMmioFlag);
}

bool Memory::is_mapped(ppcaddr_t address) const noexcept {
    return !pages_.empty() && page_entry(address) != 0;
}

bool Memory::is_mmio(ppcaddr_t address) const noexcept {
    return !pages_.empty() && (page_entry(address) & kMmioFlag) != 0;
}

bool Memory::range_is(ppcaddr_t address, u32 size, Access access) const noexcept {
    if (pages_.empty() || size == 0) {
        return false;
    }
    if (static_cast<u64>(address) + size > address_space_size) {
        return false;
    }

    // Walk by page rather than checking only the endpoints: a range can span
    // a permission boundary without either end being on the wrong side of it.
    const u64 first = address >> page_shift;
    const u64 last = (static_cast<u64>(address) + size - 1) >> page_shift;
    for (u64 page = first; page <= last; ++page) {
        const u8 entry = pages_[static_cast<std::size_t>(page)];
        if ((entry & kMmioFlag) != 0) {
            return false;
        }
        if ((entry & encode(access)) != encode(access)) {
            return false;
        }
    }
    return true;
}

u8* Memory::translate(ppcaddr_t address, u32 size, Access access) const noexcept {
    if (!range_is(address, size, access)) {
        return nullptr;
    }
    return space_.translate(address);
}

const Region* Memory::region_of(ppcaddr_t address) const noexcept {
    return find_region(regions_, address);
}

// --- bulk -----------------------------------------------------------------

bool Memory::read_bytes(ppcaddr_t address, std::span<u8> out) const {
    const u8* const host = translate(address, static_cast<u32>(out.size()), Access::Read);
    if (host == nullptr) {
        record_fault({address, static_cast<u32>(out.size()), false, !is_mapped(address)});
        return false;
    }
    notify_watchpoints(address, static_cast<u32>(out.size()), false);
    std::memcpy(out.data(), host, out.size());
    return true;
}

bool Memory::write_bytes(ppcaddr_t address, std::span<const u8> data) {
    u8* const host = translate(address, static_cast<u32>(data.size()), Access::Write);
    if (host == nullptr) {
        record_fault({address, static_cast<u32>(data.size()), true, !is_mapped(address)});
        return false;
    }
    notify_watchpoints(address, static_cast<u32>(data.size()), true);
    std::memcpy(host, data.data(), data.size());
    return true;
}

bool Memory::fill(ppcaddr_t address, u32 size, u8 value) {
    u8* const host = translate(address, size, Access::Write);
    if (host == nullptr) {
        record_fault({address, size, true, !is_mapped(address)});
        return false;
    }
    notify_watchpoints(address, size, true);
    std::memset(host, value, size);
    return true;
}

// --- MMIO -----------------------------------------------------------------

bool Memory::map_device(ppcaddr_t base, u32 size, MmioDevice& device, std::string& error) {
    if (!space_.valid()) {
        error = "memory not initialised";
        return false;
    }
    if (size == 0 || (base % page_size) != 0 || (size % page_size) != 0) {
        error = fmt::format("device range {:#010x}+{:#x} is not page aligned", base, size);
        return false;
    }
    if (static_cast<u64>(base) + size > address_space_size) {
        error = fmt::format("device range {:#010x}+{:#x} runs past the address space", base, size);
        return false;
    }

    for (u64 page = base >> page_shift; page < (static_cast<u64>(base) + size) >> page_shift;
         ++page) {
        if (pages_[static_cast<std::size_t>(page)] != 0) {
            error = fmt::format("device range {:#010x}+{:#x} overlaps mapped memory", base, size);
            return false;
        }
    }

    devices_.push_back({base, size, &device});
    set_page_range(base, size, static_cast<u8>(kMmioFlag | encode(read_write)));

    WIIU_LOG_DEBUG(Mem, "device {:<10} {:#010x}-{:#010x}", device.name(), base, base + size - 1);
    return true;
}

void Memory::unmap_device(ppcaddr_t base, u32 size) {
    std::erase_if(devices_, [base](const Device& entry) { return entry.base == base; });
    set_page_range(base, size, 0);
}

const Memory::Device* Memory::device_for(ppcaddr_t address) const noexcept {
    const auto it = std::ranges::find_if(devices_, [address](const Device& entry) {
        return address >= entry.base && address - entry.base < entry.size;
    });
    return it == devices_.end() ? nullptr : &*it;
}

bool Memory::mmio_read(ppcaddr_t address, u32 size, u32& out) const {
    const Device* const entry = device_for(address);
    if (entry == nullptr) {
        record_fault({address, size, false, true});
        return false;
    }
    notify_watchpoints(address, size, false);
    out = entry->device->read(address - entry->base, size);
    return true;
}

bool Memory::mmio_write(ppcaddr_t address, u32 size, u32 value) {
    const Device* const entry = device_for(address);
    if (entry == nullptr) {
        record_fault({address, size, true, true});
        return false;
    }
    notify_watchpoints(address, size, true);
    entry->device->write(address - entry->base, value, size);
    return true;
}

// --- cache ----------------------------------------------------------------

void Memory::notify_cache(CacheOp op, ppcaddr_t address, u32 size) const {
    if (cache_listener_) {
        cache_listener_(op, address, size);
    }
}

void Memory::dc_flush_range(ppcaddr_t address, u32 size) {
    // Cache maintenance is line-granular on hardware, and guest code relies on
    // that: a flush of one byte publishes the whole 32-byte line.
    const ppcaddr_t start = align_down(address, cache_line_size);
    const u32 end = align_up(address + size, cache_line_size);
    notify_cache(CacheOp::Flush, start, end - start);
}

void Memory::dc_invalidate_range(ppcaddr_t address, u32 size) {
    const ppcaddr_t start = align_down(address, cache_line_size);
    const u32 end = align_up(address + size, cache_line_size);
    notify_cache(CacheOp::Invalidate, start, end - start);
}

void Memory::dc_zero_range(ppcaddr_t address, u32 size) {
    const ppcaddr_t start = align_down(address, cache_line_size);
    const u32 end = align_up(address + size, cache_line_size);
    static_cast<void>(fill(start, end - start, 0));
    notify_cache(CacheOp::Zero, start, end - start);
}

void Memory::ic_invalidate_range(ppcaddr_t address, u32 size) {
    const ppcaddr_t start = align_down(address, cache_line_size);
    const u32 end = align_up(address + size, cache_line_size);
    notify_cache(CacheOp::CodeChanged, start, end - start);
}

// --- watchpoints ----------------------------------------------------------

void Memory::add_watchpoint(Watchpoint watchpoint) {
    watchpoints_.push_back(std::move(watchpoint));
}

void Memory::remove_watchpoints(std::string_view name) {
    std::erase_if(watchpoints_, [name](const Watchpoint& entry) { return entry.name == name; });
}

void Memory::clear_watchpoints() {
    watchpoints_.clear();
}

void Memory::notify_watchpoints(ppcaddr_t address, u32 size, bool is_write) const {
    // The common case is no watchpoints at all, which costs one empty check.
    if (watchpoints_.empty() || !watch_listener_) {
        return;
    }

    const u8 wanted =
        is_write ? static_cast<u8>(WatchKind::Write) : static_cast<u8>(WatchKind::Read);

    for (const Watchpoint& watchpoint : watchpoints_) {
        if ((static_cast<u8>(watchpoint.kind) & wanted) == 0) {
            continue;
        }
        const bool overlaps =
            address < watchpoint.base + watchpoint.size && watchpoint.base < address + size;
        if (overlaps) {
            watch_listener_(watchpoint, address, size, is_write);
        }
    }
}

// --- faults ---------------------------------------------------------------

void Memory::record_fault(const Fault& fault) const {
    ++fault_count_;
    last_fault_ = fault;

    WIIU_LOG_WARN(Mem,
                  "{} fault at {:#010x} ({} bytes, {})",
                  fault.is_write ? "write" : "read",
                  fault.address,
                  fault.size,
                  fault.unmapped ? "unmapped" : "permission");

    if (fault_handler_) {
        fault_handler_(fault);
    }
}

} // namespace wiiu::memory
