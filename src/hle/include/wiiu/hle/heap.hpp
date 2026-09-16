// Phase 6.6 — coreinit heaps.
//
// Cafe OS hands applications several heaps: MEM1 (the 32 MiB of fast on-die
// memory), MEM2 (main RAM), and a foreground bucket, each managed as an
// "expanded heap" that allocates from either end.
//
// The bookkeeping here lives host-side rather than in guest memory. Real
// MEMExpHeap keeps its block headers in the heap itself, and a few games walk
// them; doing the same would mean matching an undocumented layout exactly, and
// getting it subtly wrong is worse than not offering it. This is noted rather
// than hidden — if a title turns out to need real headers, that is a
// contained change behind the same interface.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/memory/memory.hpp>

#include <map>
#include <string>
#include <vector>

namespace wiiu::hle {

/// Which end of the heap an allocation comes from. Games use the head for
/// long-lived data and the tail for scratch, so the two ends do not fragment
/// each other.
enum class HeapDirection : u8 {
    Head,
    Tail,
};

struct HeapBlock {
    ppcaddr_t address = 0;
    u32 size = 0;
    u32 alignment = 0;
    HeapDirection direction = HeapDirection::Head;
};

/// One expanded heap over a range of guest memory.
class ExpandedHeap {
public:
    ExpandedHeap() = default;

    ExpandedHeap(std::string name, ppcaddr_t base, u32 size)
        : name_(std::move(name)), base_(base), size_(size), head_(base), tail_(base + size) {}

    /// Allocate `size` bytes. A negative alignment means "from the tail",
    /// which is how the Cafe OS API spells it.
    [[nodiscard]] ppcaddr_t allocate(u32 size, s32 alignment);
    [[nodiscard]] bool free(ppcaddr_t address);

    [[nodiscard]] u32 block_size(ppcaddr_t address) const;
    [[nodiscard]] u32 total_free() const noexcept;
    [[nodiscard]] u32 largest_free() const noexcept;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    [[nodiscard]] ppcaddr_t base() const noexcept { return base_; }

    [[nodiscard]] u32 size() const noexcept { return size_; }

    [[nodiscard]] std::size_t live_blocks() const noexcept { return blocks_.size(); }

    /// Every outstanding allocation, for leak reporting at shutdown.
    [[nodiscard]] std::vector<HeapBlock> blocks() const;

private:
    std::string name_;
    ppcaddr_t base_ = 0;
    u32 size_ = 0;

    // Head grows up, tail grows down; the gap between them is what is free.
    // Freed head blocks are reused from a free list rather than coalesced,
    // which is enough for the allocation patterns a game actually has and
    // avoids pretending to a fidelity this does not have.
    ppcaddr_t head_ = 0;
    ppcaddr_t tail_ = 0;

    std::map<ppcaddr_t, HeapBlock> blocks_;
    std::vector<HeapBlock> free_list_;
};

/// The heaps an application sees, and the handles it refers to them by.
class HeapManager {
public:
    /// Handles are guest-visible values. They are deliberately not addresses:
    /// a handle that looks like a pointer invites code to dereference it, and
    /// these have nothing behind them.
    static constexpr u32 handle_base = 0xC0FFEE00;

    void reset();

    /// Create the standard heaps for a title. `memory` is only used to check
    /// the ranges are mapped.
    [[nodiscard]] bool create_default_heaps(memory::Memory& memory, std::string& error);

    [[nodiscard]] u32 create(std::string name, ppcaddr_t base, u32 size);
    [[nodiscard]] ExpandedHeap* find(u32 handle);
    [[nodiscard]] const ExpandedHeap* find(u32 handle) const;

    /// The heap that owns an address, for MEMFreeToDefaultHeap and friends.
    [[nodiscard]] ExpandedHeap* owner_of(ppcaddr_t address);

    [[nodiscard]] u32 default_handle() const noexcept { return default_handle_; }

    [[nodiscard]] u32 mem1_handle() const noexcept { return mem1_handle_; }

    [[nodiscard]] u32 mem2_handle() const noexcept { return mem2_handle_; }

    [[nodiscard]] std::vector<const ExpandedHeap*> all() const;

private:
    std::map<u32, ExpandedHeap> heaps_;
    u32 next_handle_ = handle_base;
    u32 default_handle_ = 0;
    u32 mem1_handle_ = 0;
    u32 mem2_handle_ = 0;
};

} // namespace wiiu::hle
