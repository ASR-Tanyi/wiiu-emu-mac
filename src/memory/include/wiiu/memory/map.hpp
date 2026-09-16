// Phase 3.2 — the Cafe OS address map.
//
// Transcribed from the documented map. These are the addresses guest code
// actually sees, so they appear here as named constants rather than as
// literals scattered through the loader and the HLE layer.
#pragma once

#include <wiiu/common/types.hpp>

#include <span>
#include <string_view>

namespace wiiu::memory {

/// Host page granularity for permission tracking. Independent of the guest's
/// own page size, which the MMU model (Phase 3.4) deals with separately.
inline constexpr u32 page_size = 0x1000;
inline constexpr u32 page_shift = 12;

/// The guest address space is 32-bit, so the whole thing is 4 GiB.
inline constexpr u64 address_space_size = 0x100000000ULL;
inline constexpr u64 page_count = address_space_size / page_size;

enum class Access : u8 {
    None = 0,
    Read = 1U << 0,
    Write = 1U << 1,
    Execute = 1U << 2,
};

[[nodiscard]] constexpr Access operator|(Access lhs, Access rhs) noexcept {
    return static_cast<Access>(static_cast<u8>(lhs) | static_cast<u8>(rhs));
}

[[nodiscard]] constexpr bool has(Access value, Access flag) noexcept {
    return (static_cast<u8>(value) & static_cast<u8>(flag)) != 0;
}

inline constexpr Access read_only = Access::Read;
inline constexpr Access read_write = Access::Read | Access::Write;
inline constexpr Access read_execute = Access::Read | Access::Execute;
inline constexpr Access read_write_execute = Access::Read | Access::Write | Access::Execute;

/// Which physical memory a virtual region is backed by. The distinction is
/// not cosmetic: MEM1 is 32 MiB of fast on-die memory and games treat it very
/// differently from MEM2, and the GPU cares which one a surface lives in.
enum class Backing : u8 {
    Mem0, ///< 3 MiB SRAM — kernel and IOS images
    Mem1, ///< 32 MiB on-die
    Mem2, ///< 2 GiB DDR3
    None, ///< no physical backing (codegen area, MMIO windows)
};

struct Region {
    std::string_view name;
    ppcaddr_t base;
    u32 size;
    physaddr_t physical; ///< 0 when Backing::None
    Backing backing;
    Access access;

    [[nodiscard]] constexpr ppcaddr_t end() const noexcept { return base + size; }

    [[nodiscard]] constexpr bool contains(ppcaddr_t address) const noexcept {
        return address >= base && address - base < size;
    }
};

// --- physical memory -------------------------------------------------------

inline constexpr physaddr_t mem1_physical = 0x00000000;
inline constexpr u32 mem1_size = 0x02000000; ///< 32 MiB

inline constexpr physaddr_t mem0_physical = 0x08000000;
inline constexpr u32 mem0_size = 0x00300000; ///< 3 MiB

inline constexpr physaddr_t mem2_physical = 0x10000000;
inline constexpr u64 mem2_size = 0xC0000000ULL; ///< 2 GiB (ends at 0xCFFFFFFF)

// --- notable virtual addresses --------------------------------------------

inline constexpr ppcaddr_t system_libraries_base = 0x01000000;
inline constexpr ppcaddr_t codegen_base = 0x01800000;
inline constexpr ppcaddr_t application_base = 0x02000000;
inline constexpr ppcaddr_t application_end = 0x10000000; ///< always, whatever the start
inline constexpr ppcaddr_t data_heap_base = 0x10000000;
inline constexpr ppcaddr_t hardware_base = 0xE0000000; ///< IPC communication area
inline constexpr ppcaddr_t mem1_base = 0xF4000000;
inline constexpr ppcaddr_t shared_data_base = 0xF8000000; ///< system fonts, read-only
inline constexpr ppcaddr_t kernel_heap_base = 0xFF200000;

/// The default map for a foreground application.
///
/// Note the shape of the application region: it may start later than
/// 0x02000000 depending on `cos.xml`, but it always ends at 0x10000000, where
/// the data heap begins.
[[nodiscard]] std::span<const Region> default_regions() noexcept;

/// The region containing `address`, or nullptr.
[[nodiscard]] const Region* find_region(std::span<const Region> regions,
                                        ppcaddr_t address) noexcept;

/// Physical address for a virtual one, if the region has physical backing.
[[nodiscard]] bool to_physical(const Region& region, ppcaddr_t address, physaddr_t& out) noexcept;

[[nodiscard]] std::string_view backing_name(Backing backing) noexcept;

} // namespace wiiu::memory
