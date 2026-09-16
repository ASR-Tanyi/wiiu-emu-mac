// Phase 2.2 — fixed-width vocabulary types.
//
// Guest types are spelled the way the hardware documentation spells them, so
// that code transcribed from a register description reads like the register
// description. Host-side code uses the standard types.
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace wiiu {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using s8 = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;

using f32 = float;
using f64 = double;

static_assert(sizeof(f32) == 4, "f32 must be IEEE-754 binary32");
static_assert(sizeof(f64) == 8, "f64 must be IEEE-754 binary64");

/// Any unsigned integer except bool, which is unsigned but not an integer in
/// any sense this project cares about.
template <typename T>
concept UnsignedIntegral = std::unsigned_integral<T> && !std::same_as<T, bool>;

/// A Cafe OS virtual address. Always 32-bit: Espresso is a 32-bit core, and
/// widening this to size_t hides truncation bugs at the host boundary.
using ppcaddr_t = u32;

/// A Wii U physical address (MEM0/MEM1/MEM2/MMIO). Also 32-bit — the physical
/// map tops out below 4 GiB (MEM2 ends at 0xCFFFFFFF).
using physaddr_t = u32;

} // namespace wiiu
