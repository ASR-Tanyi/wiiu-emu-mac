// Phase 1.6 — logging.
//
// One channel per subsystem, each with its own runtime level, plus a
// compile-time floor so disabled levels cost nothing in a release build. A
// ring buffer keeps the last N records in memory for the Phase 10.1 debug UI.
//
//   WIIU_LOG_INFO(Loader, "loaded {} ({} sections)", name, count);
//
// A record below its channel's level costs one relaxed atomic load: the
// arguments are neither evaluated nor formatted.
#pragma once

#include <wiiu/common/types.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace wiiu::log {

/// One per emulated subsystem. Keep in sync with `channel_name`.
enum class Channel : u8 {
    Core,     // startup, config, lifecycle
    Cpu,      // Espresso interpreter / JIT
    Mem,      // address space, MMU, cache model
    Loader,   // RPX/RPL, discs, filesystem
    Coreinit, // coreinit.rpl and kernel HLE
    Ios,      // IPC to IOSU: device opens, ioctls, /dev/fsa
    Gx2,      // GX2 API layer
    Gpu,      // PM4, shader translation, Metal backend
    Ax,       // audio
    Input,    // VPAD / KPAD / controllers
    Frontend, // UI, window, settings
    Count,
};

enum class Level : u8 { Trace, Debug, Info, Warn, Error, Critical, Off };

inline constexpr std::size_t channel_count = static_cast<std::size_t>(Channel::Count);

[[nodiscard]] std::string_view channel_name(Channel channel) noexcept;
[[nodiscard]] std::string_view level_name(Level level) noexcept;

/// Parse a channel or level by name, case-insensitively. Returns false if the
/// name is unknown — used by config parsing and the `--log` flag.
[[nodiscard]] bool parse_channel(std::string_view name, Channel& out) noexcept;
[[nodiscard]] bool parse_level(std::string_view name, Level& out) noexcept;

struct Options {
    Level default_level = Level::Info;
    bool console = true;
    std::string file;                 ///< empty disables the file sink
    std::size_t ring_capacity = 4096; ///< 0 disables the in-memory ring
};

void init(const Options& options = {});
void shutdown();
[[nodiscard]] bool initialized() noexcept;

void set_level(Channel channel, Level level);
void set_all_levels(Level level);
[[nodiscard]] Level level(Channel channel) noexcept;

/// Cheap enough to call on every log site: one relaxed atomic load.
[[nodiscard]] bool enabled(Channel channel, Level level) noexcept;

/// A record retained in the ring buffer for the debug UI.
struct Record {
    Channel channel;
    Level level;
    u64 sequence;
    std::string message;
};

/// Newest-last. `max` of 0 returns everything retained.
[[nodiscard]] std::vector<Record> ring_snapshot(std::size_t max = 0);
void ring_clear();
[[nodiscard]] u64 dropped_count() noexcept;

namespace detail {
void write(Channel channel, Level level, std::string_view message);
}

template <typename... Args>
void emit(Channel channel, Level lvl, fmt::format_string<Args...> format, Args&&... args) {
    if (!enabled(channel, lvl)) {
        return;
    }
    detail::write(channel, lvl, fmt::format(format, std::forward<Args>(args)...));
}

} // namespace wiiu::log

// --- compile-time floor ----------------------------------------------------
// Levels below WIIU_LOG_ACTIVE_LEVEL compile to nothing: no call, no argument
// evaluation. Release keeps Debug and up; override with -DWIIU_LOG_ACTIVE_LEVEL=n.

#define WIIU_LOG_LEVEL_TRACE 0
#define WIIU_LOG_LEVEL_DEBUG 1
#define WIIU_LOG_LEVEL_INFO 2
#define WIIU_LOG_LEVEL_WARN 3
#define WIIU_LOG_LEVEL_ERROR 4
#define WIIU_LOG_LEVEL_CRITICAL 5
#define WIIU_LOG_LEVEL_OFF 6

#ifndef WIIU_LOG_ACTIVE_LEVEL
#ifdef NDEBUG
#define WIIU_LOG_ACTIVE_LEVEL WIIU_LOG_LEVEL_DEBUG
#else
#define WIIU_LOG_ACTIVE_LEVEL WIIU_LOG_LEVEL_TRACE
#endif
#endif

// The level check lives in the macro, not in emit(), so that a gated record
// does not even evaluate its arguments. Formatting a disassembled instruction
// or a surface descriptor costs far more than the log call itself, and in a
// hot path (Phase 4, Phase 8) that difference is the whole point.
#define WIIU_LOG_AT(chan, lvl, ...)                                                                \
    do {                                                                                           \
        if (::wiiu::log::enabled(::wiiu::log::Channel::chan, ::wiiu::log::Level::lvl)) {           \
            ::wiiu::log::emit(::wiiu::log::Channel::chan, ::wiiu::log::Level::lvl, __VA_ARGS__);   \
        }                                                                                          \
    } while (false)

#if WIIU_LOG_ACTIVE_LEVEL <= WIIU_LOG_LEVEL_TRACE
#define WIIU_LOG_TRACE(chan, ...) WIIU_LOG_AT(chan, Trace, __VA_ARGS__)
#else
#define WIIU_LOG_TRACE(chan, ...) ((void)0)
#endif

#if WIIU_LOG_ACTIVE_LEVEL <= WIIU_LOG_LEVEL_DEBUG
#define WIIU_LOG_DEBUG(chan, ...) WIIU_LOG_AT(chan, Debug, __VA_ARGS__)
#else
#define WIIU_LOG_DEBUG(chan, ...) ((void)0)
#endif

#if WIIU_LOG_ACTIVE_LEVEL <= WIIU_LOG_LEVEL_INFO
#define WIIU_LOG_INFO(chan, ...) WIIU_LOG_AT(chan, Info, __VA_ARGS__)
#else
#define WIIU_LOG_INFO(chan, ...) ((void)0)
#endif

#if WIIU_LOG_ACTIVE_LEVEL <= WIIU_LOG_LEVEL_WARN
#define WIIU_LOG_WARN(chan, ...) WIIU_LOG_AT(chan, Warn, __VA_ARGS__)
#else
#define WIIU_LOG_WARN(chan, ...) ((void)0)
#endif

#if WIIU_LOG_ACTIVE_LEVEL <= WIIU_LOG_LEVEL_ERROR
#define WIIU_LOG_ERROR(chan, ...) WIIU_LOG_AT(chan, Error, __VA_ARGS__)
#else
#define WIIU_LOG_ERROR(chan, ...) ((void)0)
#endif

#if WIIU_LOG_ACTIVE_LEVEL <= WIIU_LOG_LEVEL_CRITICAL
#define WIIU_LOG_CRITICAL(chan, ...) WIIU_LOG_AT(chan, Critical, __VA_ARGS__)
#else
#define WIIU_LOG_CRITICAL(chan, ...) ((void)0)
#endif
