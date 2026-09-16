// Phase 3.3 / 3.5 / 3.6 / 3.7 — the guest memory interface.
//
// Two ways in, deliberately:
//
//   translate_unchecked()  one add, no branch. The fastmem path (P3) the JIT
//                          will emit against, taking a host fault on anything
//                          unmapped.
//   read<T>() / write<T>() permission-checked, MMIO-aware, watchpoint-aware.
//                          What the interpreter, the loader and the tools use.
//
// Both exist from the start because the checked path has to be correct before
// the fast one can be trusted, and because the API shape is what makes fastmem
// reachable later — an accessor that assumed an explicit bounds check would
// have to be rewritten along with every JIT load and store.
#pragma once

#include <wiiu/common/endian.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/memory/address_space.hpp>
#include <wiiu/memory/map.hpp>

#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wiiu::memory {

/// A memory-mapped device. Offsets are relative to the device's own base, so
/// a device does not need to know where it was mapped.
class MmioDevice {
public:
    MmioDevice() = default;
    virtual ~MmioDevice() = default;

    MmioDevice(const MmioDevice&) = delete;
    MmioDevice& operator=(const MmioDevice&) = delete;

    [[nodiscard]] virtual u32 read(u32 offset, u32 size) = 0;
    virtual void write(u32 offset, u32 value, u32 size) = 0;
    [[nodiscard]] virtual std::string_view name() const = 0;
};

/// Why a cache maintenance call was made. The distinction matters because the
/// Espresso cores are coherent with each other but *not* with the GPU or IO:
/// software flushes are the only thing that makes a CPU write visible to
/// Latte, so they have to be modelled rather than ignored.
enum class CacheOp : u8 {
    Flush,       ///< DCFlushRange — CPU wrote; publish to the GPU
    Invalidate,  ///< DCInvalidateRange — GPU or DMA wrote; CPU must re-read
    Zero,        ///< DCZeroRange
    CodeChanged, ///< ICInvalidateRange — recompiled blocks must be dropped
};

enum class WatchKind : u8 {
    Read = 1U << 0,
    Write = 1U << 1,
    ReadWrite = Read | Write,
};

struct Watchpoint {
    std::string name;
    ppcaddr_t base;
    u32 size;
    WatchKind kind;
};

struct Fault {
    ppcaddr_t address;
    u32 size;
    bool is_write;
    bool unmapped; ///< false means mapped but lacking the required permission
};

class Memory {
public:
    Memory() = default;

    /// Reserve the address space and commit the given regions.
    [[nodiscard]] bool initialize(std::span<const Region> regions, std::string& error);

    [[nodiscard]] bool initialize(std::string& error) {
        return initialize(default_regions(), error);
    }

    void shutdown() noexcept;

    [[nodiscard]] bool ready() const noexcept { return space_.valid(); }

    // --- translation ------------------------------------------------------

    /// One add. Valid only for an address the caller already knows is mapped.
    [[nodiscard]] u8* translate_unchecked(ppcaddr_t address) const noexcept {
        return space_.translate(address);
    }

    /// Host pointer, or nullptr if the range is not fully mapped with the
    /// requested access. MMIO ranges always return nullptr: they have no
    /// backing bytes to point at.
    [[nodiscard]] u8* translate(ppcaddr_t address, u32 size, Access access) const noexcept;

    [[nodiscard]] Access access_of(ppcaddr_t address) const noexcept;
    [[nodiscard]] bool is_mapped(ppcaddr_t address) const noexcept;
    [[nodiscard]] bool is_mmio(ppcaddr_t address) const noexcept;
    [[nodiscard]] bool range_is(ppcaddr_t address, u32 size, Access access) const noexcept;

    [[nodiscard]] std::span<const Region> regions() const noexcept { return regions_; }

    [[nodiscard]] const Region* region_of(ppcaddr_t address) const noexcept;

    // --- typed access -----------------------------------------------------

    /// Read a big-endian `T` from guest memory, returning it in host order.
    /// A failed access records a fault, notifies the fault handler, and
    /// returns a zero value — the interpreter turns that into a DSI exception
    /// (Phase 4.10); tools generally prefer `try_read`.
    template <typename T>
    [[nodiscard]] T read(ppcaddr_t address) const {
        T value{};
        static_cast<void>(try_read(address, value));
        return value;
    }

    template <typename T>
    bool try_read(ppcaddr_t address, T& out) const {
        constexpr u32 size = sizeof(T);

        if (is_mmio(address)) {
            u32 raw = 0;
            if (!mmio_read(address, size, raw)) {
                return false;
            }
            out = static_cast<T>(raw);
            return true;
        }

        const u8* const host = translate(address, size, Access::Read);
        if (host == nullptr) {
            record_fault({address, size, false, !is_mapped(address)});
            out = T{};
            return false;
        }

        notify_watchpoints(address, size, false);

        if constexpr (size == 1) {
            std::memcpy(&out, host, size);
        } else {
            be<T> stored{};
            std::memcpy(&stored, host, size);
            out = stored.value();
        }
        return true;
    }

    template <typename T>
    void write(ppcaddr_t address, T value) {
        static_cast<void>(try_write(address, value));
    }

    template <typename T>
    bool try_write(ppcaddr_t address, T value) {
        constexpr u32 size = sizeof(T);

        if (is_mmio(address)) {
            return mmio_write(address, size, static_cast<u32>(value));
        }

        u8* const host = translate(address, size, Access::Write);
        if (host == nullptr) {
            record_fault({address, size, true, !is_mapped(address)});
            return false;
        }

        notify_watchpoints(address, size, true);

        if constexpr (size == 1) {
            std::memcpy(host, &value, size);
        } else {
            const be<T> stored = value;
            std::memcpy(host, &stored, size);
        }
        return true;
    }

    /// Bulk copy in and out. Bytes are not swapped: a block of guest data is
    /// whatever the guest put there.
    [[nodiscard]] bool read_bytes(ppcaddr_t address, std::span<u8> out) const;
    [[nodiscard]] bool write_bytes(ppcaddr_t address, std::span<const u8> data);
    [[nodiscard]] bool fill(ppcaddr_t address, u32 size, u8 value);

    // --- MMIO -------------------------------------------------------------

    /// Map a device over a virtual range. The range must be page aligned and
    /// must not overlap committed memory.
    [[nodiscard]] bool map_device(ppcaddr_t base, u32 size, MmioDevice& device, std::string& error);
    void unmap_device(ppcaddr_t base, u32 size);

    // --- cache model ------------------------------------------------------

    using CacheListener = std::function<void(CacheOp, ppcaddr_t, u32)>;

    /// Called when guest code publishes a range to, or re-reads it from, the
    /// GPU or IO. Phase 8.13's texture cache hangs off this.
    void set_cache_listener(CacheListener listener) { cache_listener_ = std::move(listener); }

    void dc_flush_range(ppcaddr_t address, u32 size);
    void dc_invalidate_range(ppcaddr_t address, u32 size);
    void dc_zero_range(ppcaddr_t address, u32 size);
    void ic_invalidate_range(ppcaddr_t address, u32 size);

    // --- watchpoints (Phase 3.7) -----------------------------------------

    using WatchListener = std::function<void(const Watchpoint&, ppcaddr_t, u32, bool)>;

    void set_watch_listener(WatchListener listener) { watch_listener_ = std::move(listener); }

    void add_watchpoint(Watchpoint watchpoint);
    void remove_watchpoints(std::string_view name);
    void clear_watchpoints();

    [[nodiscard]] std::span<const Watchpoint> watchpoints() const noexcept { return watchpoints_; }

    // --- faults -----------------------------------------------------------

    using FaultHandler = std::function<void(const Fault&)>;

    void set_fault_handler(FaultHandler handler) { fault_handler_ = std::move(handler); }

    [[nodiscard]] u64 fault_count() const noexcept { return fault_count_; }

    [[nodiscard]] const Fault& last_fault() const noexcept { return last_fault_; }

private:
    struct Device {
        ppcaddr_t base;
        u32 size;
        MmioDevice* device;
    };

    static constexpr u8 kMmioFlag = 1U << 7;

    [[nodiscard]] u8& page_entry(ppcaddr_t address) noexcept {
        return pages_[address >> page_shift];
    }

    [[nodiscard]] u8 page_entry(ppcaddr_t address) const noexcept {
        return pages_[address >> page_shift];
    }

    void set_page_range(ppcaddr_t base, u32 size, u8 entry) noexcept;

    [[nodiscard]] const Device* device_for(ppcaddr_t address) const noexcept;
    [[nodiscard]] bool mmio_read(ppcaddr_t address, u32 size, u32& out) const;
    [[nodiscard]] bool mmio_write(ppcaddr_t address, u32 size, u32 value);

    void record_fault(const Fault& fault) const;
    void notify_watchpoints(ppcaddr_t address, u32 size, bool is_write) const;
    void notify_cache(CacheOp op, ppcaddr_t address, u32 size) const;

    AddressSpace space_;
    std::vector<Region> regions_;
    std::vector<u8> pages_;
    std::vector<Device> devices_;
    std::vector<Watchpoint> watchpoints_;

    CacheListener cache_listener_;
    WatchListener watch_listener_;
    FaultHandler fault_handler_;

    mutable u64 fault_count_ = 0;
    mutable Fault last_fault_{};
};

} // namespace wiiu::memory
