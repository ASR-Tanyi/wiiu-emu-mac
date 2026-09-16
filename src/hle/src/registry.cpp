#include <wiiu/common/log.hpp>
#include <wiiu/hle/registry.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace wiiu::hle {
namespace {

using wiiu::log::Channel;

/// `sc` — the system call every trampoline contains.
constexpr u32 kSystemCall = 0x44000002;

} // namespace

std::string Call::read_string(ppcaddr_t address, u32 limit) const {
    std::string out;
    out.reserve(32);

    for (u32 i = 0; i < limit; ++i) {
        u8 byte = 0;
        if (!memory.try_read<u8>(address + i, byte) || byte == 0) {
            break;
        }
        out.push_back(static_cast<char>(byte));
    }
    return out;
}

std::string Registry::key(std::string_view library, std::string_view name) {
    return fmt::format("{}.{}", library, name);
}

void Registry::add(std::string library, std::string name, Handler handler) {
    const std::string id = key(library, name);

    Function& function = functions_[id];
    // Keep any trampoline already handed out: a title may have resolved this
    // import before the implementation was registered.
    function.library = std::move(library);
    function.name = std::move(name);
    function.handler = std::move(handler);
    function.is_stub = false;

    if (function.trampoline != 0) {
        by_trampoline_[function.trampoline] = &function;
    }
}

void Registry::add_stub(std::string library, std::string name, u32 return_value) {
    const std::string id = key(library, name);

    // A stub never replaces a real implementation. The lists of "known but not
    // implemented" names are long and maintained by hand, so a name drifting
    // into one after being implemented is inevitable -- and it would silently
    // turn a working function back into a zero-returning placeholder.
    if (const auto existing = functions_.find(id);
        existing != functions_.end() && !existing->second.is_stub) {
        return;
    }

    Function& function = functions_[id];
    function.library = library;
    function.name = name;
    function.is_stub = true;
    function.handler = [id, return_value](Call& call) {
        // Loud on the first call, quiet afterwards: a stub called in a loop
        // would otherwise bury everything else in the log.
        call.state.gpr[3] = return_value;
    };

    if (function.trampoline != 0) {
        by_trampoline_[function.trampoline] = &function;
    }
}

Function* Registry::find(std::string_view library, std::string_view name) {
    const auto it = functions_.find(key(library, name));
    return it == functions_.end() ? nullptr : &it->second;
}

const Function* Registry::find_by_trampoline(ppcaddr_t address) const {
    const auto it = by_trampoline_.find(address);
    return it == by_trampoline_.end() ? nullptr : it->second;
}

ppcaddr_t Registry::resolve(std::string_view library, std::string_view name) {
    const std::string id = key(library, name);

    auto it = functions_.find(id);
    if (it == functions_.end()) {
        // Unimplemented. Registering it anyway means the title loads and the
        // gap shows up at the moment it matters, with the caller's address.
        add_stub(std::string{library}, std::string{name});
        it = functions_.find(id);
    }

    Function& function = it->second;
    if (function.trampoline == 0) {
        if (next_trampoline_ >= trampoline_base + trampoline_size) {
            WIIU_LOG_ERROR(Coreinit, "out of trampoline space resolving {}", id);
            return 0;
        }
        function.trampoline = next_trampoline_;
        next_trampoline_ += 4;
        by_trampoline_[function.trampoline] = &function;
    }
    return function.trampoline;
}

bool Registry::install(memory::Memory& memory, std::string& error) {
    for (const auto& [address, function] : by_trampoline_) {
        if (!memory.try_write<u32>(address, kSystemCall)) {
            error = fmt::format("cannot write trampoline for {}.{} at {:#010x}",
                                function->library,
                                function->name,
                                address);
            return false;
        }
    }
    return true;
}

bool Registry::dispatch(ppcaddr_t address, cpu::CoreState& state, memory::Memory& memory) {
    const auto it = by_trampoline_.find(address);
    if (it == by_trampoline_.end()) {
        return false;
    }

    Function& function = *it->second;
    ++function.call_count;
    ++total_calls_;

    if (trace_capacity_ != 0) {
        if (trace_.size() >= trace_capacity_) {
            trace_.pop_front();
        }
        trace_.push_back(TraceEntry{
            &function, state.lr & ~3U, {state.gpr[3], state.gpr[4], state.gpr[5], state.gpr[6]}});
    }

    if (trace_logging_) {
        WIIU_LOG_DEBUG(Coreinit,
                       "{}.{}({:#x}, {:#x}, {:#x}) from {:#010x}",
                       function.library,
                       function.name,
                       state.gpr[3],
                       state.gpr[4],
                       state.gpr[5],
                       state.lr & ~3U);
    }

    if (function.is_stub && function.call_count == 1) {
        WIIU_LOG_WARN(Coreinit,
                      "unimplemented {}.{} called from {:#010x} (returning 0)",
                      function.library,
                      function.name,
                      state.lr);
    }

    // Set the return address *before* running the handler. A blocking handler
    // switches to another thread, replacing `state` wholesale; if the return
    // address were applied afterwards it would be written into whichever
    // thread happened to be running by then.
    //
    // The same reasoning applies to a blocking handler's own return value: it
    // must be set before it blocks, which is why those are registered as raw
    // handlers rather than wrapped functions.
    state.pc = state.lr & ~3U;

    Call call{.state = state, .memory = memory};
    function.handler(call);
    return true;
}

std::vector<const Function*> Registry::unimplemented_called() const {
    std::vector<const Function*> out;
    for (const auto& [id, function] : functions_) {
        if (function.is_stub && function.call_count > 0) {
            out.push_back(&function);
        }
    }
    std::ranges::sort(out, [](const Function* lhs, const Function* rhs) {
        return lhs->call_count > rhs->call_count;
    });
    return out;
}

} // namespace wiiu::hle

namespace wiiu::hle {

void Registry::set_trace_capacity(std::size_t capacity) {
    trace_capacity_ = capacity;
    trace_.clear();
}

std::vector<TraceEntry> Registry::recent_calls() const {
    return {trace_.begin(), trace_.end()};
}

std::vector<const Function*> Registry::called() const {
    std::vector<const Function*> out;
    for (const auto& [id, function] : functions_) {
        if (function.call_count > 0) {
            out.push_back(&function);
        }
    }
    std::ranges::sort(out, [](const Function* lhs, const Function* rhs) {
        return lhs->call_count > rhs->call_count;
    });
    return out;
}

std::string Registry::format_summary(std::size_t top) const {
    std::string out;

    const auto used = called();
    out += fmt::format("HLE calls: {} total across {} functions\n", total_calls_, used.size());

    for (std::size_t i = 0; i < used.size() && i < top; ++i) {
        out += fmt::format("  {:>8}  {}.{}{}\n",
                           used[i]->call_count,
                           used[i]->library,
                           used[i]->name,
                           used[i]->is_stub ? "  (stub)" : "");
    }

    if (!trace_.empty()) {
        out += "\nmost recent calls:\n";
        for (const TraceEntry& entry : trace_) {
            out += fmt::format("  {}.{}({:#x}, {:#x}) from {:#010x}\n",
                               entry.function->library,
                               entry.function->name,
                               entry.arguments[0],
                               entry.arguments[1],
                               entry.caller);
        }
    }

    return out;
}

} // namespace wiiu::hle
