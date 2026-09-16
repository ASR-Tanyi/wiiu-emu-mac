#include <wiiu/common/log.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

using wiiu::log::Channel;
using wiiu::log::Level;

namespace {

/// The logger is a process-wide singleton, so tests share one instance and
/// reset state rather than re-initialising it.
struct LogFixture {
    LogFixture() {
        wiiu::log::init({
            .default_level = Level::Trace,
            .console = false, // keep test output readable
            .file = {},
            .ring_capacity = 64,
        });
        wiiu::log::set_all_levels(Level::Trace);
        wiiu::log::ring_clear();
    }
};

} // namespace

TEST_CASE("channel and level names round-trip", "[common][log]") {
    for (std::size_t i = 0; i < wiiu::log::channel_count; ++i) {
        const auto channel = static_cast<Channel>(i);
        const auto name = wiiu::log::channel_name(channel);
        REQUIRE_FALSE(name.empty());

        Channel parsed{};
        REQUIRE(wiiu::log::parse_channel(name, parsed));
        CHECK(parsed == channel);
    }

    for (const auto level : {Level::Trace,
                             Level::Debug,
                             Level::Info,
                             Level::Warn,
                             Level::Error,
                             Level::Critical,
                             Level::Off}) {
        const auto name = wiiu::log::level_name(level);
        REQUIRE_FALSE(name.empty());

        Level parsed{};
        REQUIRE(wiiu::log::parse_level(name, parsed));
        CHECK(parsed == level);
    }
}

TEST_CASE("parsing is case-insensitive and rejects nonsense", "[common][log]") {
    Channel channel{};
    CHECK(wiiu::log::parse_channel("GPU", channel));
    CHECK(channel == Channel::Gpu);
    CHECK(wiiu::log::parse_channel("CoreInit", channel));
    CHECK(channel == Channel::Coreinit);
    CHECK_FALSE(wiiu::log::parse_channel("latte", channel));

    Level level{};
    CHECK(wiiu::log::parse_level("WARN", level));
    CHECK(level == Level::Warn);
    CHECK_FALSE(wiiu::log::parse_level("verbose", level));
}

TEST_CASE_METHOD(LogFixture, "per-channel levels gate independently", "[common][log]") {
    wiiu::log::set_all_levels(Level::Info);
    wiiu::log::set_level(Channel::Gpu, Level::Trace);
    wiiu::log::set_level(Channel::Cpu, Level::Error);

    CHECK(wiiu::log::enabled(Channel::Gpu, Level::Trace));
    CHECK(wiiu::log::enabled(Channel::Gpu, Level::Info));

    CHECK_FALSE(wiiu::log::enabled(Channel::Cpu, Level::Warn));
    CHECK(wiiu::log::enabled(Channel::Cpu, Level::Error));
    CHECK(wiiu::log::enabled(Channel::Cpu, Level::Critical));

    CHECK_FALSE(wiiu::log::enabled(Channel::Mem, Level::Debug));
    CHECK(wiiu::log::enabled(Channel::Mem, Level::Info));

    CHECK(wiiu::log::level(Channel::Gpu) == Level::Trace);
    CHECK(wiiu::log::level(Channel::Cpu) == Level::Error);
}

TEST_CASE_METHOD(LogFixture, "Level::Off silences a channel entirely", "[common][log]") {
    wiiu::log::set_level(Channel::Ax, Level::Off);
    CHECK_FALSE(wiiu::log::enabled(Channel::Ax, Level::Critical));
    CHECK_FALSE(wiiu::log::enabled(Channel::Ax, Level::Trace));
}

TEST_CASE_METHOD(LogFixture, "records reach the ring buffer", "[common][log]") {
    WIIU_LOG_INFO(Loader, "loaded {} ({} sections)", "helloworld.rpx", 12);
    WIIU_LOG_WARN(Gpu, "unimplemented PM4 opcode {:#04x}", 0x42);

    const auto records = wiiu::log::ring_snapshot();
    REQUIRE(records.size() == 2);

    CHECK(records[0].channel == Channel::Loader);
    CHECK(records[0].level == Level::Info);
    CHECK(records[0].message == "loaded helloworld.rpx (12 sections)");

    CHECK(records[1].channel == Channel::Gpu);
    CHECK(records[1].level == Level::Warn);
    CHECK(records[1].message == "unimplemented PM4 opcode 0x42");

    // Sequence numbers are monotonic, so the UI can detect gaps.
    CHECK(records[1].sequence > records[0].sequence);
}

TEST_CASE_METHOD(LogFixture, "a gated record is not formatted or stored", "[common][log]") {
    wiiu::log::set_level(Channel::Cpu, Level::Error);

    int side_effect = 0;
    const auto counting = [&side_effect] {
        ++side_effect;
        return 1;
    };

    WIIU_LOG_DEBUG(Cpu, "should not appear: {}", counting());

    CHECK(side_effect == 0);
    CHECK(wiiu::log::ring_snapshot().empty());
}

TEST_CASE_METHOD(LogFixture, "the ring buffer drops oldest first", "[common][log]") {
    constexpr int kCapacity = 64; // matches the fixture
    constexpr int kWritten = kCapacity + 10;

    for (int i = 0; i < kWritten; ++i) {
        WIIU_LOG_INFO(Core, "record {}", i);
    }

    const auto records = wiiu::log::ring_snapshot();
    REQUIRE(records.size() == kCapacity);

    // The oldest ten are gone; the newest is last.
    CHECK(records.front().message == "record 10");
    CHECK(records.back().message == "record 73");
    CHECK(wiiu::log::dropped_count() >= 10);
}

TEST_CASE_METHOD(LogFixture, "snapshot(max) returns the newest records", "[common][log]") {
    for (int i = 0; i < 10; ++i) {
        WIIU_LOG_INFO(Core, "record {}", i);
    }

    const auto tail = wiiu::log::ring_snapshot(3);
    REQUIRE(tail.size() == 3);
    CHECK(tail[0].message == "record 7");
    CHECK(tail[2].message == "record 9");
}
