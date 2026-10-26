// Phase 6.6 — coreinit heaps.
//
// Cafe OS hands applications several heaps: MEM1 (the 32 MiB of fast on-die
// memory), MEM2 (main RAM), and a foreground bucket, each managed as an
// "expanded heap" that allocates from either end.
//
// An expanded heap's bookkeeping lives host-side rather than in guest memory.
// Real MEMExpHeap keeps its block headers in the heap itself, and a few games
// walk them; doing the same would mean matching an undocumented layout
// exactly, and getting it subtly wrong is worse than not offering it. This is
// noted rather than hidden — if a title turns out to need real headers, that
// is a contained change behind the same interface.
//
// A **frame heap** is the other way round: its whole state is a head, a tail
// and a list of recorded states, all of it in guest memory as `MEMFrmHeap`
// and `MEMFrmHeapState` (wut documents the layouts, and decaf keeps them
// there), so that is where it is kept here too, and its handle is its
// address. MEM1's base heap is one: Cafe OS makes MEM1 a frame heap and
// MEM2 the expanded default heap (decaf's CoreInitDefaultHeap), and WHB's
// console records a state on MEM1's heap and allocates its framebuffers
// from it.
//
// **coreinit's own objects** -- the system message queue, the stacks OS
// callbacks run on, ProcUI's inboxes, OSSystemInfo -- come from a system heap
// of their own, apart from the title's, as on hardware, where they live in
// coreinit's memory. A title may take the whole of its default heap (wut's
// sbrk heap does, in `__preinit_user`), and that must not starve them.
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

/// A frame heap in guest memory: `MEMFrmHeap` (0x4C bytes) at the start of
/// its range, its data after it. decaf's coreinit_memframeheap.cpp, in
/// guest-memory terms: a state is recorded by allocating a `MEMFrmHeapState`
/// from the head, and a zero-byte allocation takes one byte, as coreinit's
/// does. The debug fill a heap created with MEM_HEAP_FLAG_DEBUG_MODE asks for
/// is not done.
namespace frame_heap {

inline constexpr u32 tag = 0x46524D48; ///< MEM_FRAME_HEAP_TAG, "FRMH"
inline constexpr u32 header_size = 0x4C;
inline constexpr u32 state_size = 0x10;
inline constexpr u32 flag_zero_allocated = 1U << 0;
inline constexpr u32 free_head = 1U << 0;
inline constexpr u32 free_tail = 1U << 1;

/// MEMCreateFrmHeapEx: the heap's handle, or 0 for a range too small.
[[nodiscard]] ppcaddr_t create(memory::Memory& memory, ppcaddr_t base, u32 size, u32 flags);

/// Whether `handle` is a live frame heap.
[[nodiscard]] bool is_frame_heap(memory::Memory& memory, ppcaddr_t handle);

/// MEMAllocFromFrmHeapEx: a negative alignment allocates from the tail.
[[nodiscard]] ppcaddr_t allocate(memory::Memory& memory, ppcaddr_t heap, u32 size, s32 alignment);

/// MEMFreeToFrmHeap, with `free_head` and/or `free_tail`.
void free(memory::Memory& memory, ppcaddr_t heap, u32 mode);

/// MEMRecordStateForFrmHeap and MEMFreeByStateToFrmHeap (tag 0: the last).
[[nodiscard]] bool record_state(memory::Memory& memory, ppcaddr_t heap, u32 tag);
[[nodiscard]] bool free_by_state(memory::Memory& memory, ppcaddr_t heap, u32 tag);

/// MEMGetAllocatableSizeForFrmHeapEx.
[[nodiscard]] u32 allocatable_size(memory::Memory& memory, ppcaddr_t heap, s32 alignment);

/// MEMAdjustFrmHeap: the heap cut down to its head, if nothing is allocated
/// from the tail; its new size from its header, or 0.
[[nodiscard]] u32 adjust(memory::Memory& memory, ppcaddr_t heap);

/// MEMDestroyFrmHeap: the heap's tag cleared; its address.
[[nodiscard]] ppcaddr_t destroy(memory::Memory& memory, ppcaddr_t heap);

} // namespace frame_heap

/// MEMBaseHeapType: which arena a base heap is.
enum class BaseHeap : u32 {
    Mem1 = 0,
    Mem2 = 1,
    Foreground = 8,
};

/// The heaps an application sees, and the handles it refers to them by.
class HeapManager {
public:
    /// Handles are guest-visible values. They are deliberately not addresses:
    /// a handle that looks like a pointer invites code to dereference it, and
    /// these have nothing behind them.
    static constexpr u32 handle_base = 0xC0FFEE00;

    void reset();

    /// Create the standard heaps for a title, as Cafe OS does before it runs
    /// one: MEM1 a frame heap over the whole of MEM1, MEM2 an expanded heap
    /// that is the default heap, and coreinit's own system heap. The
    /// foreground bucket is not modelled, so its base heap is none.
    [[nodiscard]] bool create_default_heaps(memory::Memory& memory, std::string& error);

    [[nodiscard]] u32 create(std::string name, ppcaddr_t base, u32 size);
    [[nodiscard]] ExpandedHeap* find(u32 handle);
    [[nodiscard]] const ExpandedHeap* find(u32 handle) const;

    /// The heap that owns an address, for MEMFreeToDefaultHeap and friends.
    [[nodiscard]] ExpandedHeap* owner_of(ppcaddr_t address);

    [[nodiscard]] u32 default_handle() const noexcept { return default_handle_; }

    [[nodiscard]] u32 mem1_handle() const noexcept { return base(BaseHeap::Mem1); }

    [[nodiscard]] u32 mem2_handle() const noexcept { return base(BaseHeap::Mem2); }

    /// MEMGetBaseHeapHandle and MEMSetBaseHeapHandle (which returns the old
    /// one). A type that is none of the three has no heap.
    [[nodiscard]] u32 base(BaseHeap type) const noexcept;
    u32 set_base(BaseHeap type, u32 handle) noexcept;

    /// coreinit's own heap (see the top of this file).
    [[nodiscard]] u32 system_handle() const noexcept { return system_handle_; }

    [[nodiscard]] std::vector<const ExpandedHeap*> all() const;

private:
    std::map<u32, ExpandedHeap> heaps_;
    u32 next_handle_ = handle_base;
    u32 default_handle_ = 0;
    u32 system_handle_ = 0;
    u32 mem1_base_ = 0; ///< a frame heap's address
    u32 mem2_base_ = 0;
    u32 foreground_base_ = 0;
};

} // namespace wiiu::hle
