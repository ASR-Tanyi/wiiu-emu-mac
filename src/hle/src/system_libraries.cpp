#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/system_libraries.hpp>

#include <string>
#include <vector>

namespace wiiu::hle {

namespace {

using wiiu::log::Channel;

[[nodiscard]] coreinit::Context* active() noexcept {
    return coreinit::detail::active_context();
}

// --- nsysnet: sockets with no network behind them --------------------------------
//
// A socket opens, takes its options and sends, and what it sends goes nowhere:
// there is no network here, as on a console with no connection configured.
// What a title sends to port 4405 is logged, though, because that is WHB's UDP
// log -- the way homebrew prints to a host -- and it is the title's own
// account of what it is doing. Receiving, connecting and the rest are left to
// the missing-function policy until a title wants them.

inline constexpr u16 whb_log_port = 4405;
inline constexpr u32 sockaddr_port = 2; ///< sockaddr_in: family, then port, big-endian

/// `int socket(int domain, int type, int protocol)`: a descriptor.
s32 socket_open(Call&, s32 domain, s32 type, s32 protocol) {
    static_cast<void>(domain);
    static_cast<void>(type);
    static_cast<void>(protocol);
    coreinit::Context* const context = active();
    if (context == nullptr) {
        return -1;
    }
    const s32 descriptor = context->next_socket++;
    context->sockets.insert(descriptor);
    return descriptor;
}

[[nodiscard]] bool is_socket(s32 descriptor) {
    return active() != nullptr && active()->sockets.contains(descriptor);
}

/// `int setsockopt(int, int level, int name, const void*, socklen_t)`.
s32 socket_set_option(Call&, s32 descriptor) {
    return is_socket(descriptor) ? 0 : -1;
}

/// `int sendto(int, const void* buffer, size_t length, int flags, const
/// struct sockaddr* to, socklen_t)`: all of it sent, as far as the title
/// can tell.
s32 socket_send_to(Call& call, s32 descriptor, u32 buffer, u32 length, u32 flags, u32 to) {
    static_cast<void>(flags);
    if (!is_socket(descriptor)) {
        return -1;
    }
    u16 port = 0;
    if (to != 0) {
        static_cast<void>(call.memory.try_read<u16>(to + sockaddr_port, port));
    }
    if (port == whb_log_port) {
        std::string text = call.read_string(buffer, length);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
        WIIU_LOG_INFO(Coreinit, "WHB log: {}", text);
    } else {
        WIIU_LOG_DEBUG(Coreinit, "sendto port {}: {} bytes, and no network", port, length);
    }
    return static_cast<s32>(length);
}

/// `int shutdown(int, int how)`.
s32 socket_shutdown(Call&, s32 descriptor) {
    return is_socket(descriptor) ? 0 : -1;
}

/// `int socketclose(int)`.
s32 socket_close(Call&, s32 descriptor) {
    return active() != nullptr && active()->sockets.erase(descriptor) != 0 ? 0 : -1;
}

/// `htons` and friends: network order is big-endian, which the guest is.
u32 identity(Call&, u32 value) {
    return value;
}

// --- nn_ac: the connection manager -------------------------------------------------
//
// Initialising it and asking it to connect succeed -- a console with no
// network configured accepts the request too -- and nothing ever connects.
// Zero is a successful nn::Result.

u32 ac_success(Call&) {
    return 0;
}

void ac_finalize(Call&) {}

} // namespace

void register_system_libraries(Registry& registry) {
    // nsysnet: the BSD sockets layer. `socket_lib_init` returning zero is
    // success, and a title that only initialises the library -- which is most
    // of them, at start-up -- gets past it.
    for (const char* name : {"socket_lib_init", "socket_lib_finish", "set_multicast_state"}) {
        registry.add("nsysnet", name, ac_success);
    }
    registry.add("nsysnet", "socket", socket_open);
    registry.add("nsysnet", "setsockopt", socket_set_option);
    registry.add("nsysnet", "sendto", socket_send_to);
    registry.add("nsysnet", "shutdown", socket_shutdown);
    registry.add("nsysnet", "socketclose", socket_close);
    for (const char* name : {"htons", "htonl", "ntohs", "ntohl"}) {
        registry.add("nsysnet", name, identity);
    }

    // nn_ac: initialising, connecting and closing succeed; nothing connects.
    for (const char* name : {"ACInitialize", "ACConnectAsync", "ACClose"}) {
        registry.add("nn_ac", name, ac_success);
    }
    registry.add("nn_ac", "ACFinalize", ac_finalize);

    // nn_save: save data. Zero is `SAVE_STATUS_OK`. The mount itself is 9.9;
    // until then a title can initialise the library and will be told no when
    // it asks for a directory, rather than being told yes and finding nothing.
    for (const char* name : {"SAVEInit", "SAVEShutdown"}) {
        registry.add_stub("nn_save", name);
    }

    // nn_acp: the application control protocol -- device emulation checks and
    // metadata. Zero is success here too.
    for (const char* name : {"ACPInitialize", "ACPFinalize"}) {
        registry.add_stub("nn_acp", name);
    }

    // nn_act is deliberately absent. Its exports are C++ symbols with mangled
    // names, and a stub registered under a name no title actually imports is
    // dead code that reads like a specification of what the library offers.
    // The automatic policy names them correctly when a real title calls one,
    // which is also the first moment there is anything to check the name
    // against.
}

} // namespace wiiu::hle
