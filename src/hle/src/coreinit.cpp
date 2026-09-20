#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>

#include <fmt/format.h>

#include <array>
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

// --------------------------------------------------------------------------
// Asynchronous FS (6.9)
// --------------------------------------------------------------------------
//
// The asynchronous calls do the same work as the synchronous ones and then
// deliver the result later. "Later" is the part that matters: a title that
// starts a read, sets a flag, and expects its callback to run after the call
// returns would break if the callback ran inside it.
//
// The Phase 5 filesystem is synchronous, so the work happens immediately and
// only the completion is scheduled -- on the timebase, through the same timer
// list alarms use, so a replay delivers it at the same instruction (10.4).

/// FSAsyncData as the guest lays it out: the callback, its context word, and
/// the message queue to post to instead. Three big-endian words.
struct AsyncData {
    ppcaddr_t callback = 0;
    u32 context = 0;
    ppcaddr_t queue = 0;
};

[[nodiscard]] AsyncData read_async_data(Call& call, ppcaddr_t address) {
    AsyncData data;
    if (address == 0) {
        return data;
    }
    static_cast<void>(call.memory.try_read<u32>(address, data.callback));
    static_cast<void>(call.memory.try_read<u32>(address + 4, data.context));
    static_cast<void>(call.memory.try_read<u32>(address + 8, data.queue));
    return data;
}

/// Ids for the pending completions, from the range coreinit.hpp reserves for
/// them. Held on the context rather than in a static: two runs of the same
/// title in one process should not inherit each other's numbering.
u32 next_async_id(Context& context) {
    if (context.next_async_id < async_id_base) {
        context.next_async_id = async_id_base;
    }
    return context.next_async_id++;
}

/// Claim a command block for a request. False means it already has one in
/// flight: one block carries one request, and hardware refuses a second
/// rather than queueing it.
[[nodiscard]] bool claim_fs_block(ppcaddr_t block) {
    Context* const context = active();
    if (context == nullptr) {
        return false;
    }
    if (block != 0 && !context->busy_fs_blocks.insert(block).second) {
        WIIU_LOG_WARN(Coreinit, "FS command block {:#010x} already has a request in flight", block);
        return false;
    }
    return true;
}

/// Fill in the `FSAsyncResult` a queue-delivered completion points at. It
/// lives in the caller's command block, which is opaque to the title and
/// already claimed by this request.
[[nodiscard]] bool write_async_result(memory::Memory& memory,
                                      ppcaddr_t result,
                                      ppcaddr_t client,
                                      ppcaddr_t block,
                                      s32 status,
                                      const AsyncData& data,
                                      const Message& message) {
    bool ok = memory.try_write<u32>(result + fs_async_result_callback, data.callback);
    ok = memory.try_write<u32>(result + fs_async_result_context, data.context) && ok;
    ok = memory.try_write<u32>(result + fs_async_result_queue, data.queue) && ok;
    ok = write_message(memory, result + fs_async_result_message, message) && ok;
    ok = memory.try_write<u32>(result + fs_async_result_client, client) && ok;
    ok = memory.try_write<u32>(result + fs_async_result_block, block) && ok;
    ok = memory.try_write<u32>(result + fs_async_result_status, static_cast<u32>(status)) && ok;

    if (!ok) {
        WIIU_LOG_WARN(
            Coreinit, "could not write the FSAsyncResult into command block {:#010x}", block);
    }
    return ok;
}

void complete_async(Call& call, ppcaddr_t client, ppcaddr_t block, s32 status, AsyncData data) {
    Context& context = *active();
    context.memory = &call.memory;

    if (data.callback == 0 && data.queue == 0) {
        context.busy_fs_blocks.erase(block);
        // Neither half of the API was asked for. Hardware would leave the
        // caller waiting on something that never arrives; saying so here is
        // what makes that legible rather than a title that simply stops.
        WIIU_LOG_WARN(Coreinit,
                      "asynchronous FS request on command block {:#010x} completed with neither a "
                      "callback nor a queue to deliver it to",
                      block);
        return;
    }

    if (data.callback == 0 && block == 0) {
        // The queue half needs somewhere to put the FSAsyncResult, and the
        // command block is that somewhere.
        context.busy_fs_blocks.erase(block);
        WIIU_LOG_WARN(Coreinit,
                      "asynchronous FS request with a queue but no command block to put the "
                      "result in; dropping the completion");
        return;
    }

    const u32 id = next_async_id(context);

    context.scheduler.set_timer(
        id,
        call.state.timebase + fs_async_latency_ticks,
        [&context, client, block, status, data](u64) {
            context.busy_fs_blocks.erase(block);
            if (context.memory == nullptr) {
                return;
            }
            if (data.callback != 0) {
                // A request carrying both is delivered to the callback, which
                // is what hardware does: the queue coreinit posts to on the
                // way there is its own, not the title's.
                // FSAsyncCallback(FSClient*, FSCmdBlock*, FSStatus, void*).
                const std::array<u32, 4> arguments{
                    client, block, static_cast<u32>(status), data.context};
                static_cast<void>(invoke_guest_callback(
                    context, *context.memory, "fs_async", data.callback, arguments));
                return;
            }

            // The queue half: the message carries a pointer to an
            // FSAsyncResult, which is what FSGetAsyncResult hands back. The
            // result also records the message it arrived as, so a title that
            // reads `result->ioMsg` sees what it received.
            Message message;
            message.value = block + fs_async_result_offset;
            if (!write_async_result(
                    *context.memory, message.value, client, block, status, data, message)) {
                return;
            }
            if (!post_message(context, *context.memory, data.queue, message)) {
                // The request is over as far as the driver is concerned, so the
                // command block is free -- but the title is still waiting for a
                // message that is never coming. Naming the block is what turns
                // that into something findable rather than a title that stops.
                WIIU_LOG_WARN(Coreinit,
                              "the completion for FS command block {:#010x} could not be "
                              "delivered to queue {:#010x}; whoever was waiting for it will "
                              "keep waiting",
                              block,
                              data.queue);
            }
        });
}

void fs_open_file_async(Call& call) {
    const u32 client = call.next_word();
    const u32 block = call.next_word();
    const u32 path_address = call.next_word();
    const u32 mode_address = call.next_word();
    const u32 handle_address = call.next_word();
    static_cast<void>(call.next_word()); // error mask
    const u32 async_address = call.next_word();

    if (!claim_fs_block(block)) {
        // The block is carrying another request. Refusing here is what
        // hardware does, and it keeps two completions from arriving through
        // the same structure.
        call.state.gpr[3] = static_cast<u32>(FsStatus::FatalError);
        return;
    }

    s32 status = static_cast<s32>(FsStatus::FatalError);
    if (active() != nullptr) {
        const std::string path = call.read_string(path_address);
        const std::string mode = call.read_string(mode_address, 8);

        u32 handle = 0;
        const FsStatus result = active()->filesystem.open(path, mode, handle);
        status = static_cast<s32>(result);
        if (result == FsStatus::Ok && !call.memory.try_write<u32>(handle_address, handle)) {
            static_cast<void>(active()->filesystem.close(handle));
            status = static_cast<s32>(FsStatus::FatalError);
        }
    }

    complete_async(call, client, block, status, read_async_data(call, async_address));

    // The call itself reports whether the request was *accepted*, not what it
    // found; the status goes to the callback.
    call.state.gpr[3] = 0;
}

void fs_read_file_async(Call& call) {
    const u32 client = call.next_word();
    const u32 block = call.next_word();
    const u32 buffer = call.next_word();
    const u32 size = call.next_word();
    const u32 count = call.next_word();
    const u32 handle = call.next_word();
    static_cast<void>(call.next_word()); // flags
    const u32 async_address = call.next_word();

    if (!claim_fs_block(block)) {
        // The block is carrying another request. Refusing here is what
        // hardware does, and it keeps two completions from arriving through
        // the same structure.
        call.state.gpr[3] = static_cast<u32>(FsStatus::FatalError);
        return;
    }

    s32 status = static_cast<s32>(FsStatus::FatalError);
    if (active() != nullptr) {
        std::vector<u8> data;
        const FsStatus result = active()->filesystem.read(
            handle, static_cast<u32>(static_cast<u64>(size) * count), data);
        if (result == FsStatus::End) {
            status = 0;
        } else if (result != FsStatus::Ok) {
            status = static_cast<s32>(result);
        } else if (!call.memory.write_bytes(buffer, data)) {
            status = static_cast<s32>(FsStatus::FatalError);
        } else {
            status = size == 0 ? 0 : static_cast<s32>(data.size() / size);
        }
    }

    complete_async(call, client, block, status, read_async_data(call, async_address));
    call.state.gpr[3] = 0;
}

void fs_close_file_async(Call& call) {
    const u32 client = call.next_word();
    const u32 block = call.next_word();
    const u32 handle = call.next_word();
    static_cast<void>(call.next_word()); // error mask
    const u32 async_address = call.next_word();

    if (!claim_fs_block(block)) {
        // The block is carrying another request. Refusing here is what
        // hardware does, and it keeps two completions from arriving through
        // the same structure.
        call.state.gpr[3] = static_cast<u32>(FsStatus::FatalError);
        return;
    }

    const s32 status = active() == nullptr ? static_cast<s32>(FsStatus::FatalError)
                                           : static_cast<s32>(active()->filesystem.close(handle));

    complete_async(call, client, block, status, read_async_data(call, async_address));
    call.state.gpr[3] = 0;
}

/// `FSAsyncResult *FSGetAsyncResult(OSMessage *message)` -- the first word of
/// the message the completion was posted as. A one-line function on hardware
/// too; it exists so the title does not have to know that much about the
/// message layout.
void fs_get_async_result(Call& call) {
    const u32 message_address = call.next_word();
    call.state.gpr[3] = read_message(call.memory, message_address).value;
}

} // namespace

void register_all(Registry& registry, Context& context) {
    detail::set_active_context(&context);
    register_threads(registry, context);
    register_time(registry, context);
    register_ios(registry, context);

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

    registry.add("coreinit", "FSOpenFileAsync", fs_open_file_async);
    registry.add("coreinit", "FSReadFileAsync", fs_read_file_async);
    registry.add("coreinit", "FSCloseFileAsync", fs_close_file_async);
    registry.add("coreinit", "FSGetAsyncResult", fs_get_async_result);

    registry.add("coreinit", "DCFlushRange", dc_flush_range);
    registry.add("coreinit", "DCStoreRange", dc_flush_range);
    registry.add("coreinit", "DCInvalidateRange", dc_invalidate_range);
    registry.add("coreinit", "ICInvalidateRange", ic_invalidate_range);

    // Known to exist, not implemented yet. Naming them here means a call
    // reports the function by name instead of as an unresolved import, and
    // the list doubles as the to-do for Phase 6.4 onwards.
    for (const char* name : {
             "OSGetSystemMessageQueue",
             "FSInitCmdBlock",
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
