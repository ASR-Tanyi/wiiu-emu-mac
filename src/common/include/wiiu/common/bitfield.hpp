// Phase 2.3 — named bit ranges over a register or instruction word.
//
// Two subsystems spend their whole lives pulling fields out of 32-bit words:
// the PowerPC decoder (Phase 4.2) and the Latte register file (Phase 8.4).
// Writing the shifts and masks by hand is where off-by-one bit positions come
// from, and a wrong mask produces a plausible-looking wrong value rather than
// a crash.
//
// `BitField` names the range once, in a union over the raw word:
//
//     union Instruction {
//         u32 raw;
//         BitField<26, 6, u32> opcode;
//         BitField<21, 5, u32> rd;
//         BitField<0, 16, s32> simm;   // sign-extended
//         BitField<0, 1, bool> rc;
//     };
//
//     Instruction inst{.raw = word};
//     if (inst.opcode == 14) { ... inst.rd ... inst.simm ... }
//
// Bit 0 is the least significant. PowerPC documentation numbers bits from the
// most significant end, so a field documented as bits 16-20 of a 32-bit
// instruction is `BitField<11, 5>` here. That conversion is deliberately left
// at the declaration site, spelled out, rather than hidden in the template:
// making it implicit would mean every reader has to remember which convention
// a given header chose.
#pragma once

#include <wiiu/common/types.hpp>

#include <concepts>
#include <limits>
#include <type_traits>

namespace wiiu {

namespace detail {

/// A mask of `Bits` low bits, without shifting by the full type width (UB).
template <UnsignedIntegral Storage>
[[nodiscard]] constexpr Storage low_mask(std::size_t bits) noexcept {
    constexpr std::size_t width = sizeof(Storage) * 8;
    return bits >= width ? static_cast<Storage>(~Storage{0})
                         : static_cast<Storage>((Storage{1} << bits) - 1);
}

template <typename T>
concept BitFieldValue = std::integral<T> || std::is_enum_v<T>;

} // namespace detail

/// `Bits` bits starting at `Position` of a `Storage` word, read and written as
/// a `T`.
///
/// Holds the *entire* word, so it is only meaningful as a member of a union
/// (or a struct) alongside the raw value. It is trivially copyable and has no
/// constructors, so such a union stays an aggregate.
///
/// Copy assignment is deleted on purpose: `a = b` between two BitFields would
/// copy the whole word rather than the field, which is almost never what the
/// writer meant. Assign through the value instead — `a = b.value()`.
///
/// Consequence worth knowing: a union or struct containing a BitField is not
/// copy-assignable either. Copy *construction* still works, so pass such a
/// type by value freely; to reassign one, assign its raw word.
template <std::size_t Position,
          std::size_t Bits,
          detail::BitFieldValue T,
          UnsignedIntegral Storage = u32>
struct BitField {
    using value_type = T;
    using storage_type = Storage;

    static constexpr std::size_t position = Position;
    static constexpr std::size_t bits = Bits;
    static constexpr std::size_t storage_bits = sizeof(Storage) * 8;

    static_assert(Bits > 0, "a bit field needs at least one bit");
    static_assert(Position + Bits <= storage_bits, "bit field runs past the end of its word");

    /// The field's bits, in place.
    static constexpr Storage mask =
        static_cast<Storage>(detail::low_mask<Storage>(Bits) << Position);

    /// Largest value the field can hold, as the underlying integer.
    static constexpr auto max_value = detail::low_mask<Storage>(Bits);

    [[nodiscard]] constexpr T value() const noexcept {
        const auto raw = static_cast<Storage>((storage & mask) >> Position);

        if constexpr (std::same_as<T, bool>) {
            return raw != 0;
        } else if constexpr (std::is_enum_v<T>) {
            return static_cast<T>(raw);
        } else if constexpr (std::is_signed_v<T>) {
            // Sign-extend from the field's width. PowerPC displacement fields
            // (D, SIMM, BD, LI) all rely on this.
            if constexpr (Bits == sizeof(T) * 8) {
                return static_cast<T>(raw);
            } else {
                const auto sign_bit = static_cast<Storage>(Storage{1} << (Bits - 1));
                const auto extended = static_cast<Storage>((raw ^ sign_bit) - sign_bit);
                return static_cast<T>(extended);
            }
        } else {
            return static_cast<T>(raw);
        }
    }

    constexpr operator T() const noexcept { return value(); } // NOLINT(*-explicit-*)

    constexpr BitField& operator=(T value) noexcept {
        Storage raw{};
        if constexpr (std::same_as<T, bool>) {
            raw = value ? Storage{1} : Storage{0};
        } else if constexpr (std::is_enum_v<T>) {
            raw = static_cast<Storage>(static_cast<std::underlying_type_t<T>>(value));
        } else {
            raw = static_cast<Storage>(value);
        }

        // Values wider than the field are truncated, matching what writing the
        // hardware register would do.
        raw = static_cast<Storage>(raw & detail::low_mask<Storage>(Bits));
        storage = static_cast<Storage>((storage & ~mask) | static_cast<Storage>(raw << Position));
        return *this;
    }

    BitField& operator=(const BitField&) = delete;

    Storage storage;
};

// --------------------------------------------------------------------------
// Free access, for the cases where there is no struct to overlay — a PM4
// register decoded once in a switch, or a generated instruction table that
// carries positions as data rather than as template arguments.
// --------------------------------------------------------------------------

template <std::size_t Position,
          std::size_t Bits,
          detail::BitFieldValue T = u32,
          UnsignedIntegral Storage = u32>
[[nodiscard]] constexpr T get_field(Storage word) noexcept {
    const BitField<Position, Bits, T, Storage> field{word};
    return field.value();
}

template <std::size_t Position,
          std::size_t Bits,
          detail::BitFieldValue T = u32,
          UnsignedIntegral Storage = u32>
[[nodiscard]] constexpr Storage set_field(Storage word, T value) noexcept {
    BitField<Position, Bits, T, Storage> field{word};
    field = value;
    return field.storage;
}

} // namespace wiiu
