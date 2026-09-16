#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>

#include <fmt/format.h>

#include <cmath>
#include <vector>

namespace wiiu::hle::coreinit {
namespace {

using wiiu::log::Channel;

Context* g_active = nullptr;

/// The context these handlers act on. A short name because it appears in
/// almost every one of them; see `detail::active_context` for why it is not
/// simply a captured reference.
[[nodiscard]] Context* active() noexcept {
    return detail::active_context();
}

void emit(std::string_view text) {
    if (active() != nullptr && active()->report_sink) {
        active()->report_sink(text);
    } else {
        WIIU_LOG_INFO(Coreinit, "{}", text);
    }
}

} // namespace

namespace detail {

Context* active_context() noexcept {
    return g_active;
}

void set_active_context(Context* context) noexcept {
    g_active = context;
}

} // namespace detail

std::string format_guest(Call& call, std::string_view format, u32 first_gpr) {
    // Guest varargs live in the remaining argument registers and then on the
    // stack. Only the register window is handled; a call with more than a few
    // arguments spills, and that is worth knowing about rather than silently
    // printing rubbish.
    std::string out;
    out.reserve(format.size() + 32);

    u32 gpr = first_gpr;
    u32 fpr = 1;

    const auto next_word = [&]() -> u32 { return gpr <= 10 ? call.state.gpr[gpr++] : 0; };
    const auto next_double = [&]() -> f64 { return fpr <= 8 ? call.state.fpr[fpr++].ps0 : 0.0; };

    for (std::size_t i = 0; i < format.size(); ++i) {
        if (format[i] != '%') {
            out.push_back(format[i]);
            continue;
        }
        if (i + 1 >= format.size()) {
            break;
        }

        // Skip flags, width and precision: the conversion is what matters for
        // readability, and honouring every modifier is not worth the surface.
        std::size_t j = i + 1;
        while (j < format.size() && (std::string_view{"-+ #0123456789.*lhzjt"}.find(format[j]) !=
                                     std::string_view::npos)) {
            ++j;
        }
        if (j >= format.size()) {
            break;
        }

        const char conversion = format[j];
        switch (conversion) {
            case 'd':
            case 'i':
                out += fmt::format("{}", static_cast<s32>(next_word()));
                break;
            case 'u':
                out += fmt::format("{}", next_word());
                break;
            case 'x':
                out += fmt::format("{:x}", next_word());
                break;
            case 'X':
                out += fmt::format("{:X}", next_word());
                break;
            case 'o':
                out += fmt::format("{:o}", next_word());
                break;
            case 'p':
                out += fmt::format("{:#010x}", next_word());
                break;
            case 'c':
                out.push_back(static_cast<char>(next_word() & 0xFF));
                break;
            case 's': {
                const ppcaddr_t address = next_word();
                out += address == 0 ? "(null)" : call.read_string(address);
                break;
            }
            case 'f':
            case 'g':
            case 'e':
                out += fmt::format("{}", next_double());
                break;
            case '%':
                out.push_back('%');
                break;
            default:
                // Unrecognised: copy it through so the output shows what the
                // program actually asked for.
                out.push_back('%');
                out.push_back(conversion);
                break;
        }
        i = j;
    }

    return out;
}

// --------------------------------------------------------------------------
// Implemented functions
// --------------------------------------------------------------------------

namespace {

void os_report(Call& call, u32 format_address) {
    const std::string format = call.read_string(format_address);
    emit(format_guest(call, format, 4));
}

void os_fatal(Call& call, u32 message_address) {
    const std::string message = call.read_string(message_address);
    WIIU_LOG_ERROR(Coreinit, "OSFatal: {}", message);
    emit(fmt::format("OSFatal: {}", message));
    call.state.halted = true;
}

u32 os_get_core_id(Call& call) {
    return call.state.core_id;
}

u32 os_get_core_count(Call&) {
    return cpu::core_count;
}

u32 os_get_title_id(Call&) {
    return 0;
}

/// OSBlockMove / memcpy and friends (P16). Running the guest's own PowerPC
/// implementation of these through the interpreter is strictly worse than
/// calling the host's, and games lean on them constantly.
u32 memcpy_impl(Call& call, u32 destination, u32 source, u32 size) {
    std::vector<u8> buffer(size);
    if (size != 0 && call.memory.read_bytes(source, buffer)) {
        static_cast<void>(call.memory.write_bytes(destination, buffer));
    }
    return destination;
}

u32 memset_impl(Call& call, u32 destination, s32 value, u32 size) {
    static_cast<void>(call.memory.fill(destination, size, static_cast<u8>(value)));
    return destination;
}

u32 os_block_move(Call& call, u32 destination, u32 source, u32 size, bool flush) {
    static_cast<void>(flush);
    return memcpy_impl(call, destination, source, size);
}

s32 strlen_impl(Call& call, u32 address) {
    return static_cast<s32>(call.read_string(address).size());
}

void dc_flush_range(Call& call, u32 address, u32 size) {
    // Not a no-op: this is how a guest publishes a write to the GPU.
    call.memory.dc_flush_range(address, size);
}

void dc_invalidate_range(Call& call, u32 address, u32 size) {
    call.memory.dc_invalidate_range(address, size);
}

void ic_invalidate_range(Call& call, u32 address, u32 size) {
    call.memory.ic_invalidate_range(address, size);
}

} // namespace

// --------------------------------------------------------------------------
// Heaps (6.6)
// --------------------------------------------------------------------------

namespace {

u32 mem_get_base_heap_handle(Call&, u32 index) {
    if (active() == nullptr) {
        return 0;
    }
    // MEM_BASE_HEAP_MEM1 is 0, MEM2 is 1, FG is 2. The foreground bucket is
    // not modelled separately, so it shares MEM1's.
    switch (index) {
        case 0:
            return active()->heaps.mem1_handle();
        case 1:
            return active()->heaps.mem2_handle();
        default:
            return active()->heaps.mem1_handle();
    }
}

u32 mem_alloc_from_default_heap(Call&, u32 size) {
    if (active() == nullptr) {
        return 0;
    }
    ExpandedHeap* const heap = active()->heaps.find(active()->heaps.default_handle());
    return heap == nullptr ? 0 : heap->allocate(size, 0);
}

u32 mem_alloc_from_default_heap_ex(Call&, u32 size, s32 alignment) {
    if (active() == nullptr) {
        return 0;
    }
    ExpandedHeap* const heap = active()->heaps.find(active()->heaps.default_handle());
    return heap == nullptr ? 0 : heap->allocate(size, alignment);
}

void mem_free_to_default_heap(Call&, u32 address) {
    if (active() == nullptr || address == 0) {
        return;
    }
    // Which heap it came from is looked up rather than assumed: a title can
    // free a MEM1 allocation through the default-heap entry point.
    ExpandedHeap* const heap = active()->heaps.owner_of(address);
    if (heap != nullptr) {
        static_cast<void>(heap->free(address));
    }
}

u32 mem_create_exp_heap_ex(Call&, u32 base, u32 size, u32 flags) {
    static_cast<void>(flags);
    return active() == nullptr ? 0 : active()->heaps.create("user", base, size);
}

u32 mem_alloc_from_exp_heap_ex(Call&, u32 handle, u32 size, s32 alignment) {
    if (active() == nullptr) {
        return 0;
    }
    ExpandedHeap* const heap = active()->heaps.find(handle);
    return heap == nullptr ? 0 : heap->allocate(size, alignment);
}

void mem_free_to_exp_heap(Call&, u32 handle, u32 address) {
    if (active() == nullptr || address == 0) {
        return;
    }
    ExpandedHeap* const heap = active()->heaps.find(handle);
    if (heap != nullptr) {
        static_cast<void>(heap->free(address));
    }
}

u32 mem_get_size_for_mblock_exp_heap(Call&, u32 address) {
    if (active() == nullptr) {
        return 0;
    }
    const ExpandedHeap* const heap = active()->heaps.owner_of(address);
    return heap == nullptr ? 0 : heap->block_size(address);
}

u32 mem_get_total_free_size_for_exp_heap(Call&, u32 handle) {
    if (active() == nullptr) {
        return 0;
    }
    const ExpandedHeap* const heap = active()->heaps.find(handle);
    return heap == nullptr ? 0 : heap->total_free();
}

u32 mem_get_allocatable_size_for_exp_heap_ex(Call&, u32 handle, s32 alignment) {
    static_cast<void>(alignment);
    if (active() == nullptr) {
        return 0;
    }
    const ExpandedHeap* const heap = active()->heaps.find(handle);
    return heap == nullptr ? 0 : heap->largest_free();
}

// --------------------------------------------------------------------------
// Filesystem (6.9)
// --------------------------------------------------------------------------

s32 fs_init(Call&) {
    return 0;
}

s32 fs_add_client(Call&, u32 client, u32 error_mask) {
    static_cast<void>(error_mask);
    if (active() == nullptr) {
        return static_cast<s32>(FsStatus::FatalError);
    }
    // The client pointer is the handle: the API hands out a structure the
    // caller owns, and every later call passes it back.
    static_cast<void>(active()->filesystem.add_client());
    static_cast<void>(client);
    return 0;
}

s32 fs_del_client(Call&, u32 client, u32 error_mask) {
    static_cast<void>(client);
    static_cast<void>(error_mask);
    return 0;
}

s32 fs_open_file(Call& call,
                 u32 client,
                 u32 block,
                 u32 path_address,
                 u32 mode_address,
                 u32 handle_address,
                 u32 error_mask) {
    static_cast<void>(client);
    static_cast<void>(block);
    static_cast<void>(error_mask);
    if (active() == nullptr) {
        return static_cast<s32>(FsStatus::FatalError);
    }

    const std::string path = call.read_string(path_address);
    const std::string mode = call.read_string(mode_address, 8);

    u32 handle = 0;
    const FsStatus status = active()->filesystem.open(path, mode, handle);
    if (status != FsStatus::Ok) {
        WIIU_LOG_DEBUG(Coreinit, "FSOpenFile('{}'): {}", path, fs_status_name(status));
        return static_cast<s32>(status);
    }

    if (!call.memory.try_write<u32>(handle_address, handle)) {
        static_cast<void>(active()->filesystem.close(handle));
        return static_cast<s32>(FsStatus::FatalError);
    }
    return 0;
}

s32 fs_read_file(
    Call& call, u32 client, u32 block, u32 buffer, u32 size, u32 count, u32 handle, u32 flags) {
    static_cast<void>(client);
    static_cast<void>(block);
    static_cast<void>(flags);
    if (active() == nullptr) {
        return static_cast<s32>(FsStatus::FatalError);
    }

    const u64 wanted = static_cast<u64>(size) * count;
    std::vector<u8> data;
    const FsStatus status = active()->filesystem.read(handle, static_cast<u32>(wanted), data);
    if (status == FsStatus::End) {
        return 0;
    }
    if (status != FsStatus::Ok) {
        return static_cast<s32>(status);
    }

    if (!call.memory.write_bytes(buffer, data)) {
        return static_cast<s32>(FsStatus::FatalError);
    }

    // The API counts *items*, not bytes.
    return size == 0 ? 0 : static_cast<s32>(data.size() / size);
}

s32 fs_close_file(Call&, u32 client, u32 block, u32 handle, u32 error_mask) {
    static_cast<void>(client);
    static_cast<void>(block);
    static_cast<void>(error_mask);
    return active() == nullptr ? static_cast<s32>(FsStatus::FatalError)
                               : static_cast<s32>(active()->filesystem.close(handle));
}

s32 fs_set_pos_file(Call&, u32 client, u32 block, u32 handle, u32 position, u32 error_mask) {
    static_cast<void>(client);
    static_cast<void>(block);
    static_cast<void>(error_mask);
    return active() == nullptr ? static_cast<s32>(FsStatus::FatalError)
                               : static_cast<s32>(active()->filesystem.seek(handle, position));
}

s32 fs_get_pos_file(
    Call& call, u32 client, u32 block, u32 handle, u32 out_address, u32 error_mask) {
    static_cast<void>(client);
    static_cast<void>(block);
    static_cast<void>(error_mask);
    if (active() == nullptr) {
        return static_cast<s32>(FsStatus::FatalError);
    }

    u64 position = 0;
    const FsStatus status = active()->filesystem.tell(handle, position);
    if (status != FsStatus::Ok) {
        return static_cast<s32>(status);
    }
    return call.memory.try_write<u32>(out_address, static_cast<u32>(position))
               ? 0
               : static_cast<s32>(FsStatus::FatalError);
}

} // namespace

void register_all(Registry& registry, Context& context) {
    detail::set_active_context(&context);
    register_threads(registry, context);
    register_time(registry, context);

    registry.add("coreinit", "OSReport", os_report);
    registry.add("coreinit", "OSConsoleWrite", os_report);
    registry.add("coreinit", "OSFatal", os_fatal);

    registry.add("coreinit", "OSGetCoreId", os_get_core_id);
    registry.add("coreinit", "OSGetCoreCount", os_get_core_count);
    registry.add("coreinit", "OSGetTitleID", os_get_title_id);

    registry.add("coreinit", "memcpy", memcpy_impl);
    registry.add("coreinit", "memset", memset_impl);
    registry.add("coreinit", "OSBlockMove", os_block_move);
    registry.add("coreinit", "strlen", strlen_impl);

    registry.add("coreinit", "MEMGetBaseHeapHandle", mem_get_base_heap_handle);
    registry.add("coreinit", "MEMAllocFromDefaultHeap", mem_alloc_from_default_heap);
    registry.add("coreinit", "MEMAllocFromDefaultHeapEx", mem_alloc_from_default_heap_ex);
    registry.add("coreinit", "MEMFreeToDefaultHeap", mem_free_to_default_heap);
    registry.add("coreinit", "MEMCreateExpHeapEx", mem_create_exp_heap_ex);
    registry.add("coreinit", "MEMAllocFromExpHeapEx", mem_alloc_from_exp_heap_ex);
    registry.add("coreinit", "MEMFreeToExpHeap", mem_free_to_exp_heap);
    registry.add("coreinit", "MEMGetSizeForMBlockExpHeap", mem_get_size_for_mblock_exp_heap);
    registry.add("coreinit", "MEMGetTotalFreeSizeForExpHeap", mem_get_total_free_size_for_exp_heap);
    registry.add(
        "coreinit", "MEMGetAllocatableSizeForExpHeapEx", mem_get_allocatable_size_for_exp_heap_ex);

    registry.add("coreinit", "FSInit", fs_init);
    registry.add("coreinit", "FSAddClient", fs_add_client);
    registry.add("coreinit", "FSDelClient", fs_del_client);
    registry.add("coreinit", "FSOpenFile", fs_open_file);
    registry.add("coreinit", "FSReadFile", fs_read_file);
    registry.add("coreinit", "FSCloseFile", fs_close_file);
    registry.add("coreinit", "FSSetPosFile", fs_set_pos_file);
    registry.add("coreinit", "FSGetPosFile", fs_get_pos_file);

    registry.add("coreinit", "DCFlushRange", dc_flush_range);
    registry.add("coreinit", "DCStoreRange", dc_flush_range);
    registry.add("coreinit", "DCInvalidateRange", dc_invalidate_range);
    registry.add("coreinit", "ICInvalidateRange", ic_invalidate_range);

    // Known to exist, not implemented yet. Naming them here means a call
    // reports the function by name instead of as an unresolved import, and
    // the list doubles as the to-do for Phase 6.4 onwards.
    for (const char* name : {
             "OSDynLoad_Acquire",
             "OSDynLoad_FindExport",
             "OSDynLoad_Release",
             "OSScreenInit",
             "OSScreenFlipBuffersEx",
             "OSScreenClearBufferEx",
             "OSScreenPutFontEx",
             "exit",
             "__os_snprintf",
         }) {
        registry.add_stub("coreinit", name);
    }
}

} // namespace wiiu::hle::coreinit
