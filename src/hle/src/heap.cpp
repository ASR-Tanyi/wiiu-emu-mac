#include <wiiu/common/bits.hpp>
#include <wiiu/common/log.hpp>
#include <wiiu/hle/heap.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace wiiu::hle {
namespace {

using wiiu::log::Channel;

constexpr u32 kDefaultAlignment = 4;

} // namespace

ppcaddr_t ExpandedHeap::allocate(u32 size, s32 alignment) {
    if (size == 0) {
        return 0;
    }

    // The sign of the alignment picks the end to allocate from -- that is the
    // API's own convention, not an invention here.
    const HeapDirection direction = alignment < 0 ? HeapDirection::Tail : HeapDirection::Head;
    u32 align = alignment == 0 ? kDefaultAlignment
                               : static_cast<u32>(alignment < 0 ? -alignment : alignment);
    if (!is_pow2(align)) {
        align = kDefaultAlignment;
    }

    if (direction == HeapDirection::Head) {
        // Reuse an exactly-fitting freed block before growing.
        const auto reusable = std::ranges::find_if(free_list_, [&](const HeapBlock& block) {
            return block.direction == HeapDirection::Head && block.size >= size &&
                   is_aligned(block.address, align);
        });
        if (reusable != free_list_.end()) {
            HeapBlock block = *reusable;
            free_list_.erase(reusable);
            block.size = size;
            block.alignment = align;
            blocks_[block.address] = block;
            return block.address;
        }

        const ppcaddr_t address = align_up(head_, align);
        if (address + size > tail_) {
            WIIU_LOG_WARN(Coreinit,
                          "heap '{}' exhausted: wanted {} bytes, {} free",
                          name_,
                          size,
                          total_free());
            return 0;
        }
        head_ = address + size;
        blocks_[address] = {address, size, align, HeapDirection::Head};
        return address;
    }

    const ppcaddr_t candidate = align_down(tail_ - size, align);
    if (candidate < head_ || candidate > tail_) {
        WIIU_LOG_WARN(Coreinit, "heap '{}' exhausted from the tail: wanted {} bytes", name_, size);
        return 0;
    }
    tail_ = candidate;
    blocks_[candidate] = {candidate, size, align, HeapDirection::Tail};
    return candidate;
}

bool ExpandedHeap::free(ppcaddr_t address) {
    const auto it = blocks_.find(address);
    if (it == blocks_.end()) {
        // Freeing something that was never allocated is a guest bug worth
        // seeing, not something to ignore quietly.
        WIIU_LOG_WARN(
            Coreinit, "heap '{}': free of {:#010x}, which is not a live block", name_, address);
        return false;
    }

    const HeapBlock block = it->second;
    blocks_.erase(it);

    // Freeing the most recent head allocation simply rewinds; anything else
    // goes on the free list for exact reuse.
    if (block.direction == HeapDirection::Head && block.address + block.size == head_) {
        head_ = block.address;
    } else if (block.direction == HeapDirection::Tail && block.address == tail_) {
        tail_ = block.address + block.size;
    } else {
        free_list_.push_back(block);
    }
    return true;
}

u32 ExpandedHeap::block_size(ppcaddr_t address) const {
    const auto it = blocks_.find(address);
    return it == blocks_.end() ? 0 : it->second.size;
}

u32 ExpandedHeap::total_free() const noexcept {
    u32 free_bytes = tail_ - head_;
    for (const HeapBlock& block : free_list_) {
        free_bytes += block.size;
    }
    return free_bytes;
}

u32 ExpandedHeap::largest_free() const noexcept {
    u32 largest = tail_ - head_;
    for (const HeapBlock& block : free_list_) {
        largest = std::max(largest, block.size);
    }
    return largest;
}

std::vector<HeapBlock> ExpandedHeap::blocks() const {
    std::vector<HeapBlock> out;
    out.reserve(blocks_.size());
    for (const auto& [address, block] : blocks_) {
        out.push_back(block);
    }
    return out;
}

// --------------------------------------------------------------------------
// HeapManager
// --------------------------------------------------------------------------

void HeapManager::reset() {
    heaps_.clear();
    next_handle_ = handle_base;
    default_handle_ = 0;
    system_handle_ = 0;
    mem1_base_ = 0;
    mem2_base_ = 0;
    foreground_base_ = 0;
}

u32 HeapManager::base(BaseHeap type) const noexcept {
    switch (type) {
        case BaseHeap::Mem1:
            return mem1_base_;
        case BaseHeap::Mem2:
            return mem2_base_;
        case BaseHeap::Foreground:
            return foreground_base_;
    }
    return 0;
}

u32 HeapManager::set_base(BaseHeap type, u32 handle) noexcept {
    u32* slot = nullptr;
    switch (type) {
        case BaseHeap::Mem1:
            slot = &mem1_base_;
            break;
        case BaseHeap::Mem2:
            slot = &mem2_base_;
            break;
        case BaseHeap::Foreground:
            slot = &foreground_base_;
            break;
    }
    if (slot == nullptr) {
        return 0;
    }
    const u32 old = *slot;
    *slot = handle;
    return old;
}

u32 HeapManager::create(std::string name, ppcaddr_t base, u32 size) {
    const u32 handle = next_handle_;
    next_handle_ += 4;
    heaps_.emplace(handle, ExpandedHeap{std::move(name), base, size});
    return handle;
}

bool HeapManager::create_default_heaps(memory::Memory& memory, std::string& error) {
    reset();

    // The application heap sits above where a module and its data land. The
    // exact split comes from cos.xml (Phase 5.5); these are the fallbacks.
    // coreinit's own heap follows it.
    constexpr ppcaddr_t application_heap_base = memory::data_heap_base + 0x02000000;
    constexpr u32 application_heap_size = 0x08000000;
    constexpr ppcaddr_t system_heap_base = application_heap_base + application_heap_size;
    constexpr u32 system_heap_size = 0x01000000;

    if (!memory.range_is(
            application_heap_base, application_heap_size + system_heap_size, memory::read_write)) {
        error = fmt::format("application heap at {:#010x} is not mapped", application_heap_base);
        return false;
    }
    if (!memory.range_is(memory::mem1_base, memory::mem1_size, memory::read_write)) {
        error = "MEM1 is not mapped";
        return false;
    }

    default_handle_ = create("default", application_heap_base, application_heap_size);
    system_handle_ = create("system", system_heap_base, system_heap_size);
    mem2_base_ = default_handle_;
    mem1_base_ = frame_heap::create(memory, memory::mem1_base, memory::mem1_size, 0);
    if (mem1_base_ == 0) {
        error = "MEM1's frame heap could not be made";
        return false;
    }

    WIIU_LOG_INFO(Coreinit,
                  "heaps: default {} MiB at {:#010x}, MEM1 {} MiB at {:#010x} (a frame heap)",
                  application_heap_size / (1024 * 1024),
                  application_heap_base,
                  memory::mem1_size / (1024 * 1024),
                  memory::mem1_base);
    return true;
}

ExpandedHeap* HeapManager::find(u32 handle) {
    const auto it = heaps_.find(handle);
    return it == heaps_.end() ? nullptr : &it->second;
}

const ExpandedHeap* HeapManager::find(u32 handle) const {
    const auto it = heaps_.find(handle);
    return it == heaps_.end() ? nullptr : &it->second;
}

ExpandedHeap* HeapManager::owner_of(ppcaddr_t address) {
    for (auto& [handle, heap] : heaps_) {
        if (address >= heap.base() && address - heap.base() < heap.size()) {
            return &heap;
        }
    }
    return nullptr;
}

std::vector<const ExpandedHeap*> HeapManager::all() const {
    std::vector<const ExpandedHeap*> out;
    out.reserve(heaps_.size());
    for (const auto& [handle, heap] : heaps_) {
        out.push_back(&heap);
    }
    return out;
}

// --------------------------------------------------------------------------
// Frame heaps
// --------------------------------------------------------------------------

namespace frame_heap {
namespace {

// MEMHeapHeader's fields, and MEMFrmHeap's after it (wut's offsets).
constexpr u32 header_tag = 0x00;
constexpr u32 header_data_start = 0x18;
constexpr u32 header_data_end = 0x1C;
constexpr u32 header_flags = 0x30;
constexpr u32 heap_head = 0x40;
constexpr u32 heap_tail = 0x44;
constexpr u32 heap_previous_state = 0x48;

// MEMFrmHeapState's.
constexpr u32 state_tag = 0x00;
constexpr u32 state_head = 0x04;
constexpr u32 state_tail = 0x08;
constexpr u32 state_previous = 0x0C;

[[nodiscard]] u32 word(memory::Memory& memory, ppcaddr_t address) {
    u32 value = 0;
    static_cast<void>(memory.try_read<u32>(address, value));
    return value;
}

void put(memory::Memory& memory, ppcaddr_t address, u32 value) {
    static_cast<void>(memory.try_write<u32>(address, value));
}

[[nodiscard]] u32 alignment_of(s32 alignment) {
    const u32 magnitude = alignment < 0 ? static_cast<u32>(-static_cast<s64>(alignment))
                                        : static_cast<u32>(alignment);
    return magnitude == 0 || !is_pow2(magnitude) ? 4 : magnitude;
}

} // namespace

ppcaddr_t create(memory::Memory& memory, ppcaddr_t base, u32 size, u32 flags) {
    const ppcaddr_t start = align_up(base, 4U);
    const ppcaddr_t end = align_down(base + size, 4U);
    if (start >= end || end - start < header_size ||
        !memory.range_is(start, end - start, memory::read_write)) {
        return 0;
    }
    static_cast<void>(memory.fill(start, header_size, 0));
    put(memory, start + header_tag, tag);
    put(memory, start + header_data_start, start + header_size);
    put(memory, start + header_data_end, end);
    put(memory, start + header_flags, flags);
    put(memory, start + heap_head, start + header_size);
    put(memory, start + heap_tail, end);
    put(memory, start + heap_previous_state, 0);
    return start;
}

bool is_frame_heap(memory::Memory& memory, ppcaddr_t handle) {
    u32 value = 0;
    return handle != 0 && memory.try_read<u32>(handle + header_tag, value) && value == tag;
}

ppcaddr_t allocate(memory::Memory& memory, ppcaddr_t heap, u32 size, s32 alignment) {
    if (size == 0) {
        size = 1; // as coreinit does
    }
    const u32 align = alignment_of(alignment);
    const ppcaddr_t head = word(memory, heap + heap_head);
    const ppcaddr_t tail = word(memory, heap + heap_tail);
    ppcaddr_t block = 0;
    if (alignment < 0) {
        if (size > tail - head) {
            return 0;
        }
        block = align_down(tail - size, align);
        if (block < head) {
            return 0;
        }
        put(memory, heap + heap_tail, block);
    } else {
        block = align_up(head, align);
        if (block > tail || size > tail - block) {
            return 0;
        }
        put(memory, heap + heap_head, block + size);
    }
    if ((word(memory, heap + header_flags) & flag_zero_allocated) != 0) {
        static_cast<void>(memory.fill(block, size, 0));
    }
    return block;
}

void free(memory::Memory& memory, ppcaddr_t heap, u32 mode) {
    if ((mode & free_head) != 0) {
        put(memory, heap + heap_head, word(memory, heap + header_data_start));
        put(memory, heap + heap_previous_state, 0);
    }
    if ((mode & free_tail) != 0) {
        put(memory, heap + heap_tail, word(memory, heap + header_data_end));
        put(memory, heap + heap_previous_state, 0);
    }
}

bool record_state(memory::Memory& memory, ppcaddr_t heap, u32 state_tag_value) {
    // The head and tail before the state's own allocation, as coreinit
    // records them: freeing by this state frees the state too.
    const ppcaddr_t head = word(memory, heap + heap_head);
    const ppcaddr_t tail = word(memory, heap + heap_tail);
    const ppcaddr_t state = allocate(memory, heap, state_size, 4);
    if (state == 0) {
        return false;
    }
    put(memory, state + state_tag, state_tag_value);
    put(memory, state + state_head, head);
    put(memory, state + state_tail, tail);
    put(memory, state + state_previous, word(memory, heap + heap_previous_state));
    put(memory, heap + heap_previous_state, state);
    return true;
}

bool free_by_state(memory::Memory& memory, ppcaddr_t heap, u32 state_tag_value) {
    ppcaddr_t state = word(memory, heap + heap_previous_state);
    if (state_tag_value != 0) {
        while (state != 0 && word(memory, state + state_tag) != state_tag_value) {
            state = word(memory, state + state_previous);
        }
    }
    if (state == 0) {
        return false;
    }
    put(memory, heap + heap_head, word(memory, state + state_head));
    put(memory, heap + heap_tail, word(memory, state + state_tail));
    put(memory, heap + heap_previous_state, word(memory, state + state_previous));
    return true;
}

u32 allocatable_size(memory::Memory& memory, ppcaddr_t heap, s32 alignment) {
    const ppcaddr_t head = align_up(word(memory, heap + heap_head), alignment_of(alignment));
    const ppcaddr_t tail = word(memory, heap + heap_tail);
    return head < tail ? tail - head : 0;
}

u32 adjust(memory::Memory& memory, ppcaddr_t heap) {
    if (word(memory, heap + heap_tail) != word(memory, heap + header_data_end)) {
        return 0;
    }
    const ppcaddr_t head = word(memory, heap + heap_head);
    put(memory, heap + header_data_end, head);
    put(memory, heap + heap_tail, head);
    return head - heap;
}

ppcaddr_t destroy(memory::Memory& memory, ppcaddr_t heap) {
    put(memory, heap + header_tag, 0);
    return heap;
}

} // namespace frame_heap

} // namespace wiiu::hle
