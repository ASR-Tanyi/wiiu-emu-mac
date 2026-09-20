#include <wiiu/common/log.hpp>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <deque>
#include <memory>
#include <mutex>

namespace wiiu::log {
namespace {

constexpr std::array<std::string_view, channel_count> kChannelNames{
    "core",
    "cpu",
    "mem",
    "loader",
    "coreinit",
    "ios",
    "gx2",
    "gpu",
    "ax",
    "input",
    "frontend",
};

constexpr std::array<std::string_view, 7> kLevelNames{
    "trace",
    "debug",
    "info",
    "warn",
    "error",
    "critical",
    "off",
};

[[nodiscard]] constexpr std::size_t index_of(Channel channel) noexcept {
    return static_cast<std::size_t>(channel);
}

[[nodiscard]] spdlog::level::level_enum to_spdlog(Level level) noexcept {
    switch (level) {
        case Level::Trace:
            return spdlog::level::trace;
        case Level::Debug:
            return spdlog::level::debug;
        case Level::Info:
            return spdlog::level::info;
        case Level::Warn:
            return spdlog::level::warn;
        case Level::Error:
            return spdlog::level::err;
        case Level::Critical:
            return spdlog::level::critical;
        case Level::Off:
            return spdlog::level::off;
    }
    return spdlog::level::info;
}

/// Fixed-capacity in-memory sink. The debug UI (Phase 10.1) reads this; it is
/// deliberately independent of the console and file sinks so that turning the
/// console off does not blind the UI.
class RingBuffer {
public:
    void configure(std::size_t capacity) {
        const std::lock_guard lock(mutex_);
        capacity_ = capacity;
        records_.clear();
        dropped_ = 0;
    }

    void push(Channel channel, Level level, std::string_view message) {
        const std::lock_guard lock(mutex_);
        if (capacity_ == 0) {
            return;
        }
        while (records_.size() >= capacity_) {
            records_.pop_front();
            ++dropped_;
        }
        records_.push_back(Record{channel, level, sequence_++, std::string{message}});
    }

    [[nodiscard]] std::vector<Record> snapshot(std::size_t max) const {
        const std::lock_guard lock(mutex_);
        const std::size_t count = (max == 0) ? records_.size() : std::min(max, records_.size());
        return std::vector<Record>(records_.end() - static_cast<std::ptrdiff_t>(count),
                                   records_.end());
    }

    void clear() {
        const std::lock_guard lock(mutex_);
        records_.clear();
    }

    [[nodiscard]] u64 dropped() const {
        const std::lock_guard lock(mutex_);
        return dropped_;
    }

private:
    mutable std::mutex mutex_;
    std::deque<Record> records_;
    std::size_t capacity_ = 0;
    u64 sequence_ = 0;
    u64 dropped_ = 0;
};

struct State {
    // Relaxed atomics: the log-site check must be near-free, and a level
    // change racing with a log call may legitimately land either side of it.
    std::array<std::atomic<u8>, channel_count> levels{};
    std::array<std::shared_ptr<spdlog::logger>, channel_count> loggers{};
    RingBuffer ring;
    std::atomic<bool> ready{false};
};

State& state() {
    static State instance;
    return instance;
}

[[nodiscard]] std::string lowercase(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const char c : input) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

} // namespace

std::string_view channel_name(Channel channel) noexcept {
    const auto index = index_of(channel);
    return index < kChannelNames.size() ? kChannelNames[index] : "?";
}

std::string_view level_name(Level level) noexcept {
    const auto index = static_cast<std::size_t>(level);
    return index < kLevelNames.size() ? kLevelNames[index] : "?";
}

bool parse_channel(std::string_view name, Channel& out) noexcept {
    const std::string needle = lowercase(name);
    for (std::size_t i = 0; i < kChannelNames.size(); ++i) {
        if (kChannelNames[i] == needle) {
            out = static_cast<Channel>(i);
            return true;
        }
    }
    return false;
}

bool parse_level(std::string_view name, Level& out) noexcept {
    const std::string needle = lowercase(name);
    for (std::size_t i = 0; i < kLevelNames.size(); ++i) {
        if (kLevelNames[i] == needle) {
            out = static_cast<Level>(i);
            return true;
        }
    }
    return false;
}

void init(const Options& options) {
    State& s = state();
    if (s.ready.load(std::memory_order_acquire)) {
        return;
    }

    s.ring.configure(options.ring_capacity);

    std::vector<spdlog::sink_ptr> sinks;
    if (options.console) {
        auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        console->set_pattern("%^[%L]%$ %-8n %v");
        sinks.push_back(std::move(console));
    }
    if (!options.file.empty()) {
        auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>(options.file, true);
        file->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%L] %-8n %v");
        sinks.push_back(std::move(file));
    }

    for (std::size_t i = 0; i < channel_count; ++i) {
        auto logger = std::make_shared<spdlog::logger>(
            std::string{kChannelNames[i]}, sinks.begin(), sinks.end());
        // Channel gating happens in `enabled()`; spdlog itself stays wide open
        // so a level change never needs to touch the sinks.
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::err);
        s.loggers[i] = std::move(logger);
        s.levels[i].store(static_cast<u8>(options.default_level), std::memory_order_relaxed);
    }

    s.ready.store(true, std::memory_order_release);
}

void shutdown() {
    State& s = state();
    if (!s.ready.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    for (auto& logger : s.loggers) {
        if (logger) {
            logger->flush();
        }
        logger.reset();
    }
    s.ring.clear();
}

bool initialized() noexcept {
    return state().ready.load(std::memory_order_acquire);
}

void set_level(Channel channel, Level lvl) {
    state().levels[index_of(channel)].store(static_cast<u8>(lvl), std::memory_order_relaxed);
}

void set_all_levels(Level lvl) {
    for (auto& slot : state().levels) {
        slot.store(static_cast<u8>(lvl), std::memory_order_relaxed);
    }
}

Level level(Channel channel) noexcept {
    return static_cast<Level>(state().levels[index_of(channel)].load(std::memory_order_relaxed));
}

bool enabled(Channel channel, Level lvl) noexcept {
    const State& s = state();
    if (!s.ready.load(std::memory_order_relaxed)) {
        return false;
    }
    const auto threshold = s.levels[index_of(channel)].load(std::memory_order_relaxed);
    return static_cast<u8>(lvl) >= threshold && lvl != Level::Off;
}

std::vector<Record> ring_snapshot(std::size_t max) {
    return state().ring.snapshot(max);
}

void ring_clear() {
    state().ring.clear();
}

u64 dropped_count() noexcept {
    return state().ring.dropped();
}

namespace detail {

void write(Channel channel, Level lvl, std::string_view message) {
    State& s = state();
    s.ring.push(channel, lvl, message);
    if (auto& logger = s.loggers[index_of(channel)]) {
        logger->log(to_spdlog(lvl), message);
    }
}

} // namespace detail
} // namespace wiiu::log
