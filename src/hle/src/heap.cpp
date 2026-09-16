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
    mem1_handle_ = 0;
    mem2_handle_ = 0;
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
    constexpr ppcaddr_t application_heap_base = memory::data_heap_base + 0x02000000;
    constexpr u32 application_heap_size = 0x08000000;

    if (!memory.range_is(application_heap_base, application_heap_size, memory::read_write)) {
        error = fmt::format("application heap at {:#010x} is not mapped", application_heap_base);
        return false;
    }
    if (!memory.range_is(memory::mem1_base, memory::mem1_size, memory::read_write)) {
        error = "MEM1 is not mapped";
        return false;
    }

    default_handle_ = create("default", application_heap_base, application_heap_size);
    mem2_handle_ = default_handle_;
    mem1_handle_ = create("MEM1", memory::mem1_base, memory::mem1_size);

    WIIU_LOG_INFO(Coreinit,
                  "heaps: default {} MiB at {:#010x}, MEM1 {} MiB at {:#010x}",
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

} // namespace wiiu::hle
