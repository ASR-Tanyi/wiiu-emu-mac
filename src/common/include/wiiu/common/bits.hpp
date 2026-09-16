// Phase 2.4 — small bit and alignment utilities.
#pragma once

#include <wiiu/common/types.hpp>

#include <bit>
#include <type_traits>

namespace wiiu {

/// True for exactly one set bit. Zero is not a power of two.
template <UnsignedIntegral T>
[[nodiscard]] constexpr bool is_pow2(T value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

/// Round up to the next multiple of `alignment`, which must be a power of two.
template <UnsignedIntegral T>
[[nodiscard]] constexpr T align_up(T value, T alignment) noexcept {
    return (value + (alignment - 1)) & ~(alignment - 1);
}

/// Round down to the previous multiple of `alignment` (power of two).
template <UnsignedIntegral T>
[[nodiscard]] constexpr T align_down(T value, T alignment) noexcept {
    return value & ~(alignment - 1);
}

template <UnsignedIntegral T>
[[nodiscard]] constexpr bool is_aligned(T value, T alignment) noexcept {
    return (value & (alignment - 1)) == 0;
}

/// Extract `len` bits starting at `pos` (bit 0 = least significant).
///
/// Note: PowerPC and the Latte register documentation number bits from the
/// most significant end. Conversions belong at the call site, spelled out,
/// rather than hidden in here.
template <UnsignedIntegral T>
[[nodiscard]] constexpr T extract_bits(T value, unsigned pos, unsigned len) noexcept {
    const T mask =
        (len >= sizeof(T) * 8) ? static_cast<T>(~T{0}) : static_cast<T>((T{1} << len) - 1);
    return static_cast<T>((value >> pos) & mask);
}

/// Sign-extend the low `bits` of `value` to a signed 32-bit result.
[[nodiscard]] constexpr s32 sign_extend(u32 value, unsigned bits) noexcept {
    const unsigned shift = 32U - bits;
    return static_cast<s32>(value << shift) >> shift;
}

/// The Espresso cache line size. Every dcbz/dcbf/dcbst granularity decision
/// and the Phase 3.6 cache model key off this.
inline constexpr u32 cache_line_size = 32;

} // namespace wiiu
