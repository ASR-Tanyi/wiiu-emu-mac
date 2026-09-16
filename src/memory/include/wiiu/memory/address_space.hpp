// Phase 3.1 — the host reservation backing guest memory.
//
// One 4 GiB PROT_NONE reservation, with the mapped regions committed inside
// it. Guest virtual to host pointer is then a single add, which is what makes
// fastmem (P3) possible: the JIT can emit a bare load against
// `base + zero_extend32(guest_address)` with no bounds check, and let the host
// MMU fault on anything unmapped.
//
// That is why the reservation is the whole 4 GiB rather than just the mapped
// parts. A guest address can never escape it, so a stray access faults instead
// of corrupting the emulator's own heap.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/memory/map.hpp>

#include <string>

namespace wiiu::memory {

/// Owns the host reservation. Move-only; the reservation is released on
/// destruction.
class AddressSpace {
public:
    AddressSpace() = default;
    ~AddressSpace();

    AddressSpace(const AddressSpace&) = delete;
    AddressSpace& operator=(const AddressSpace&) = delete;
    AddressSpace(AddressSpace&& other) noexcept;
    AddressSpace& operator=(AddressSpace&& other) noexcept;

    /// Reserve the address space. Nothing is readable until `commit`.
    [[nodiscard]] bool reserve(std::string& error);

    /// Back `[base, base + size)` with real pages.
    [[nodiscard]] bool commit(ppcaddr_t base, u32 size, Access access, std::string& error);

    /// Return a range to the unmapped state.
    [[nodiscard]] bool decommit(ppcaddr_t base, u32 size, std::string& error);

    void release() noexcept;

    [[nodiscard]] bool valid() const noexcept { return base_ != nullptr; }

    /// Host pointer for a guest address. No validation: this is the fastmem
    /// path, and the caller is asserting the access is legal (or is prepared
    /// to take the fault).
    [[nodiscard]] u8* translate(ppcaddr_t address) const noexcept { return base_ + address; }

    [[nodiscard]] u8* base() const noexcept { return base_; }

private:
    u8* base_ = nullptr;
};

} // namespace wiiu::memory
