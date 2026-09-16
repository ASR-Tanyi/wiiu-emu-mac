// Phase 2.1 — big-endian guest storage.
//
// The Wii U is big-endian and every host we target is little-endian, so every
// value that lives in guest memory has to be swapped on the way in and out.
// Doing that at the call site is how emulators end up with a permanent supply
// of byte-order bugs: the swap is invisible when you forget it, and the value
// is merely wrong, not obviously wrong.
//
// `be<T>` moves the swap into the type. A guest structure is transcribed
// field-for-field from the hardware documentation:
//
//     struct Elf32Header {
//         u8 ident[16];
//         u16be type;
//         u32be entry;
//     };
//
// and reading `header.entry` yields a host-order value. The layout matches the
// guest exactly, so the struct can be overlaid on guest memory directly.
//
// On a big-endian host the swap compiles away to nothing. On arm64 it is a
// single `rev`, which is why P11 (byte-swap elimination) is about removing
// *redundant* swaps in the JIT rather than the swap itself.
#pragma once

#include <wiiu/common/types.hpp>

#include <bit>
#include <concepts>
#include <cstring>
#include <type_traits>

namespace wiiu {

// --------------------------------------------------------------------------
// swap_bytes
// --------------------------------------------------------------------------

/// Reverse the bytes of an unsigned integer. `u8` is a no-op, kept so that
/// generic code does not need to special-case it.
template <UnsignedIntegral T>
[[nodiscard]] constexpr T swap_bytes(T value) noexcept {
    if constexpr (sizeof(T) == 1) {
        return value;
    } else if constexpr (sizeof(T) == 2) {
        return static_cast<T>(__builtin_bswap16(static_cast<u16>(value)));
    } else if constexpr (sizeof(T) == 4) {
        return static_cast<T>(__builtin_bswap32(static_cast<u32>(value)));
    } else {
        static_assert(sizeof(T) == 8, "unsupported integer width");
        return static_cast<T>(__builtin_bswap64(static_cast<u64>(value)));
    }
}

namespace detail {

/// The unsigned integer used to hold a `T` while it is byte-reversed.
/// Floating-point values are swapped through their bit pattern; swapping the
/// float itself would risk signalling-NaN normalisation on the way through a
/// floating-point register.
template <typename T>
struct StorageFor;

template <>
struct StorageFor<u8> {
    using type = u8;
};

template <>
struct StorageFor<u16> {
    using type = u16;
};

template <>
struct StorageFor<u32> {
    using type = u32;
};

template <>
struct StorageFor<u64> {
    using type = u64;
};

template <>
struct StorageFor<s8> {
    using type = u8;
};

template <>
struct StorageFor<s16> {
    using type = u16;
};

template <>
struct StorageFor<s32> {
    using type = u32;
};

template <>
struct StorageFor<s64> {
    using type = u64;
};

template <>
struct StorageFor<f32> {
    using type = u32;
};

template <>
struct StorageFor<f64> {
    using type = u64;
};

template <typename T>
using storage_for = typename StorageFor<T>::type;

template <typename T>
concept Swappable = requires { typename StorageFor<T>::type; };

} // namespace detail

// --------------------------------------------------------------------------
// be<T>
// --------------------------------------------------------------------------

/// A `T` stored in big-endian byte order.
///
/// Trivially copyable, standard layout, and the same size and alignment as
/// `T`, so it can appear in a structure overlaid on guest memory. Default
/// construction leaves the value uninitialised, exactly as the underlying
/// memory would be.
template <detail::Swappable T>
class be {
public:
    using value_type = T;
    using storage_type = detail::storage_for<T>;

    be() = default;

    constexpr be(T value) noexcept : storage_(encode(value)) {} // NOLINT(*-explicit-*)

    constexpr be& operator=(T value) noexcept {
        storage_ = encode(value);
        return *this;
    }

    constexpr operator T() const noexcept { return decode(storage_); } // NOLINT(*-explicit-*)

    [[nodiscard]] constexpr T value() const noexcept { return decode(storage_); }

    /// The raw big-endian bit pattern, as it sits in memory. For hashing,
    /// checksums and writing back out — never for arithmetic.
    [[nodiscard]] constexpr storage_type raw() const noexcept { return storage_; }

    [[nodiscard]] static constexpr be from_raw(storage_type raw) noexcept {
        be result;
        result.storage_ = raw;
        return result;
    }

    // Compound assignment, for arithmetic types. Each is a decode, an
    // operation in host order, and an encode; there is no cheaper correct way
    // in an object that must stay big-endian at rest.
    constexpr be& operator+=(T rhs) noexcept { return *this = static_cast<T>(value() + rhs); }

    constexpr be& operator-=(T rhs) noexcept { return *this = static_cast<T>(value() - rhs); }

    constexpr be& operator*=(T rhs) noexcept { return *this = static_cast<T>(value() * rhs); }

    constexpr be& operator/=(T rhs) noexcept { return *this = static_cast<T>(value() / rhs); }

    constexpr be& operator&=(T rhs) noexcept
        requires std::integral<T>
    {
        // Bitwise operations are order-independent, so they could act on the
        // raw storage. Going through the value keeps one code path and costs
        // nothing that the optimiser does not remove.
        return *this = static_cast<T>(value() & rhs);
    }

    constexpr be& operator|=(T rhs) noexcept
        requires std::integral<T>
    {
        return *this = static_cast<T>(value() | rhs);
    }

    constexpr be& operator^=(T rhs) noexcept
        requires std::integral<T>
    {
        return *this = static_cast<T>(value() ^ rhs);
    }

    constexpr be& operator%=(T rhs) noexcept
        requires std::integral<T>
    {
        return *this = static_cast<T>(value() % rhs);
    }

    constexpr be& operator<<=(unsigned rhs) noexcept
        requires std::integral<T>
    {
        return *this = static_cast<T>(value() << rhs);
    }

    constexpr be& operator>>=(unsigned rhs) noexcept
        requires std::integral<T>
    {
        return *this = static_cast<T>(value() >> rhs);
    }

    constexpr be& operator++() noexcept { return *this += T{1}; }

    constexpr be& operator--() noexcept { return *this -= T{1}; }

    constexpr T operator++(int) noexcept {
        const T previous = value();
        *this += T{1};
        return previous;
    }

    constexpr T operator--(int) noexcept {
        const T previous = value();
        *this -= T{1};
        return previous;
    }

private:
    [[nodiscard]] static constexpr storage_type encode(T value) noexcept {
        return to_guest_order(std::bit_cast<storage_type>(value));
    }

    [[nodiscard]] static constexpr T decode(storage_type stored) noexcept {
        return std::bit_cast<T>(to_guest_order(stored));
    }

    /// Host order to guest order, and back — the same operation either way.
    [[nodiscard]] static constexpr storage_type to_guest_order(storage_type value) noexcept {
        if constexpr (std::endian::native == std::endian::big) {
            return value;
        } else {
            static_assert(std::endian::native == std::endian::little,
                          "mixed-endian hosts are not supported");
            return swap_bytes(value);
        }
    }

    storage_type storage_;
};

// --------------------------------------------------------------------------
// Aliases, spelled to read naturally in a transcribed guest structure.
// --------------------------------------------------------------------------

using u16be = be<u16>;
using u32be = be<u32>;
using u64be = be<u64>;
using s16be = be<s16>;
using s32be = be<s32>;
using s64be = be<s64>;
using f32be = be<f32>;
using f64be = be<f64>;

// u8 needs no swapping; the alias exists so that a transcribed structure can
// be written uniformly.
using u8be = u8;
using s8be = s8;

// --------------------------------------------------------------------------
// Guarantees. These are the whole point of the type: if any of them breaks,
// every guest structure silently acquires the wrong layout.
// --------------------------------------------------------------------------

#define WIIU_ASSERT_BE_LAYOUT(T)                                                                   \
    static_assert(sizeof(be<T>) == sizeof(T), "be<" #T "> must not change size");                  \
    static_assert(alignof(be<T>) == alignof(T), "be<" #T "> must not change alignment");           \
    static_assert(std::is_trivially_copyable_v<be<T>>, "be<" #T "> must be trivially copyable");   \
    static_assert(std::is_standard_layout_v<be<T>>, "be<" #T "> must be standard layout");         \
    static_assert(std::is_trivially_default_constructible_v<be<T>>,                                \
                  "be<" #T "> must not initialise on construction")

WIIU_ASSERT_BE_LAYOUT(u16);
WIIU_ASSERT_BE_LAYOUT(u32);
WIIU_ASSERT_BE_LAYOUT(u64);
WIIU_ASSERT_BE_LAYOUT(s16);
WIIU_ASSERT_BE_LAYOUT(s32);
WIIU_ASSERT_BE_LAYOUT(s64);
WIIU_ASSERT_BE_LAYOUT(f32);
WIIU_ASSERT_BE_LAYOUT(f64);

#undef WIIU_ASSERT_BE_LAYOUT

} // namespace wiiu
