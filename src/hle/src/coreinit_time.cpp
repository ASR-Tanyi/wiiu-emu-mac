// Phase 6.7 — time.
//
// The Wii U has two clocks and Cafe OS exposes both. `OSGetSystemTime` is the
// PowerPC timebase: ticks since the console was switched on, at 62.15625 MHz.
// `OSGetTime` is the same counter offset by the real-time clock, so it counts
// from 2000-01-01, and it is the one titles turn into a date.
//
// Only the offset comes from the host, and it is read exactly once. Every
// later call derives from the timebase, which advances by instructions rather
// than by wall-clock time — so a title that asks the time twice a frame gets
// a smoothly advancing clock instead of host jitter, and a run can be
// reproduced exactly by recording that single value (10.4).

#include <wiiu/common/log.hpp>
#include <wiiu/hle/coreinit.hpp>

#include <array>
#include <chrono>
#include <limits>

namespace wiiu::hle::coreinit {
namespace {

using wiiu::log::Channel;

constexpr u64 ticks_per_second = cpu::timebase_hz;
constexpr s64 seconds_per_day = 86400;

/// 1970-01-01 to 2000-01-01, in days. Checked against the civil conversion
/// below rather than trusted.
constexpr s64 epoch_days = 10957;

/// Howard Hinnant's civil-calendar algorithms, which are exact for every year
/// in range and — unlike `gmtime`/`mktime` — have no timezone, locale or
/// thread-safety to get wrong. `days` counts from 1970-01-01.
struct Civil {
    s64 year;
    u32 month; ///< 1-12
    u32 day;   ///< 1-31
};

[[nodiscard]] constexpr Civil civil_from_days(s64 days) noexcept {
    days += 719468;
    const s64 era = (days >= 0 ? days : days - 146096) / 146097;
    const auto doe = static_cast<u64>(days - era * 146097);                // [0, 146096]
    const u64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    const s64 year = static_cast<s64>(yoe) + era * 400;
    const u64 doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    const u64 mp = (5 * doy + 2) / 153;                      // [0, 11]
    const u64 day = doy - (153 * mp + 2) / 5 + 1;            // [1, 31]
    const u64 month = mp < 10 ? mp + 3 : mp - 9;             // [1, 12]
    return {year + static_cast<s64>(month <= 2), static_cast<u32>(month), static_cast<u32>(day)};
}

[[nodiscard]] constexpr s64 days_from_civil(s64 year, u32 month, u32 day) noexcept {
    year -= month <= 2 ? 1 : 0;
    const s64 era = (year >= 0 ? year : year - 399) / 400;
    const auto yoe = static_cast<u64>(year - era * 400);   // [0, 399]
    const u64 shifted = month > 2 ? month - 3 : month + 9; // March-based
    const u64 doy = (153 * shifted + 2) / 5 + day - 1;     // [0, 365]
    const u64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    return era * 146097 + static_cast<s64>(doe) - 719468;
}

static_assert(days_from_civil(2000, 1, 1) == epoch_days);
static_assert(civil_from_days(epoch_days).year == 2000);
static_assert(civil_from_days(epoch_days).month == 1);
static_assert(civil_from_days(epoch_days).day == 1);

/// Floor division, because a date before the epoch must round down rather
/// than toward zero.
[[nodiscard]] constexpr s64 floor_div(s64 value, s64 divisor) noexcept {
    const s64 quotient = value / divisor;
    return (value % divisor != 0 && ((value < 0) != (divisor < 0))) ? quotient - 1 : quotient;
}

[[nodiscard]] constexpr s64 floor_mod(s64 value, s64 divisor) noexcept {
    return value - floor_div(value, divisor) * divisor;
}

/// The host wall clock, as Wii U ticks since 2000-01-01.
[[nodiscard]] u64 host_rtc_ticks() {
    const auto since_1970 = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
    const s64 microseconds = since_1970 - epoch_days * seconds_per_day * 1'000'000;
    if (microseconds <= 0) {
        return 0; // a host clock set before 2000; the epoch itself will do
    }
    // Split rather than multiplying through: microseconds since 2000 times
    // 62 million overflows 64 bits well before the console's own lifetime is
    // up.
    const auto whole = static_cast<u64>(microseconds) / 1'000'000;
    const auto fraction = static_cast<u64>(microseconds) % 1'000'000;
    return whole * ticks_per_second + fraction * ticks_per_second / 1'000'000;
}

// --- the functions themselves ---------------------------------------------

[[nodiscard]] u64 current_time(Call& call) {
    Context* const context = detail::active_context();
    return (context != nullptr ? rtc_epoch_ticks(*context) : 0) + call.state.timebase;
}

u64 os_get_time(Call& call) {
    return current_time(call);
}

u64 os_get_system_time(Call& call) {
    return call.state.timebase;
}

/// The low 32 bits, which is what the OSTick-flavoured calls return. Titles
/// use them for short intervals and rely on the wraparound.
u32 os_get_tick(Call& call) {
    return static_cast<u32>(current_time(call));
}

u32 os_get_system_tick(Call& call) {
    return static_cast<u32>(call.state.timebase);
}

void write_calendar(memory::Memory& memory, ppcaddr_t address, const CalendarTime& time) {
    const std::array<s32, 10> fields{time.second,
                                     time.minute,
                                     time.hour,
                                     time.day,
                                     time.month,
                                     time.year,
                                     time.weekday,
                                     time.yearday,
                                     time.millisecond,
                                     time.microsecond};
    for (u32 i = 0; i < fields.size(); ++i) {
        static_cast<void>(memory.try_write<u32>(address + i * 4, static_cast<u32>(fields[i])));
    }
}

[[nodiscard]] CalendarTime read_calendar(const memory::Memory& memory, ppcaddr_t address) {
    std::array<s32, 10> fields{};
    for (u32 i = 0; i < fields.size(); ++i) {
        u32 word = 0;
        if (memory.try_read<u32>(address + i * 4, word)) {
            fields[i] = static_cast<s32>(word);
        }
    }
    return CalendarTime{fields[0],
                        fields[1],
                        fields[2],
                        fields[3],
                        fields[4],
                        fields[5],
                        fields[6],
                        fields[7],
                        fields[8],
                        fields[9]};
}

void os_ticks_to_calendar_time(Call& call, u64 ticks, u32 out) {
    if (out != 0) {
        write_calendar(call.memory, out, ticks_to_calendar(ticks));
    }
}

u64 os_calendar_time_to_ticks(Call& call, u32 in) {
    return in == 0 ? 0 : calendar_to_ticks(read_calendar(call.memory, in));
}

} // namespace

// --------------------------------------------------------------------------

CalendarTime ticks_to_calendar(u64 ticks) noexcept {
    CalendarTime out;

    const auto seconds = static_cast<s64>(ticks / ticks_per_second);
    const u64 remainder = ticks % ticks_per_second;

    const s64 days = floor_div(seconds, seconds_per_day);
    const s64 in_day = floor_mod(seconds, seconds_per_day);

    const Civil civil = civil_from_days(days + epoch_days);

    out.year = static_cast<s32>(civil.year);
    out.month = static_cast<s32>(civil.month) - 1; // 0-11, as tm_mon is
    out.day = static_cast<s32>(civil.day);
    out.hour = static_cast<s32>(in_day / 3600);
    out.minute = static_cast<s32>((in_day / 60) % 60);
    out.second = static_cast<s32>(in_day % 60);

    // 1970-01-01 was a Thursday, so day 0 is weekday 4 and Sunday is 0.
    out.weekday = static_cast<s32>(floor_mod(days + epoch_days + 4, 7));
    out.yearday = static_cast<s32>(days + epoch_days - days_from_civil(civil.year, 1, 1));

    const u64 microseconds = remainder * 1'000'000 / ticks_per_second;
    out.millisecond = static_cast<s32>(microseconds / 1000);
    out.microsecond = static_cast<s32>(microseconds % 1000);
    return out;
}

u64 calendar_to_ticks(const CalendarTime& time) noexcept {
    // Out-of-range fields normalise rather than being rejected, which is what
    // makes "the same time, plus 90 seconds" a legal thing for a title to
    // build and hand back. Every field is signed and comes from guest memory,
    // so every one of them can arrive negative, the sub-second ones included
    // -- and a negative millisecond that borrows through an unsigned cast
    // lands three and a half days in the *future*, which is a very hard wrong
    // answer to trace back to here.
    const s64 month_index = static_cast<s64>(time.year) * 12 + time.month;
    const s64 year = floor_div(month_index, 12);
    const auto month = static_cast<u32>(floor_mod(month_index, 12)) + 1;

    const s64 sub_microseconds = static_cast<s64>(time.millisecond) * 1000 + time.microsecond;

    const s64 days = days_from_civil(year, month, 1) + (time.day - 1) - epoch_days;
    const s64 seconds = days * seconds_per_day + time.hour * 3600LL + time.minute * 60LL +
                        time.second + floor_div(sub_microseconds, 1'000'000);
    if (seconds < 0) {
        return 0;
    }

    // Past this the tick count does not fit, and wrapping would hand back a
    // plausible-looking date thousands of years out rather than an obvious
    // failure.
    constexpr s64 seconds_limit =
        static_cast<s64>(std::numeric_limits<u64>::max() / ticks_per_second) - 1;
    if (seconds > seconds_limit) {
        return std::numeric_limits<u64>::max();
    }

    const auto within_second = static_cast<u64>(floor_mod(sub_microseconds, 1'000'000));
    return static_cast<u64>(seconds) * ticks_per_second +
           within_second * ticks_per_second / 1'000'000;
}

u64 rtc_epoch_ticks(Context& context) {
    if (context.rtc_valid) {
        return context.rtc_epoch;
    }

    u64 ticks = host_rtc_ticks();
    if (context.recording != nullptr) {
        ticks = context.recording->value(EventKind::HostTime, 0, ticks);
    }

    context.rtc_epoch = ticks;
    context.rtc_valid = true;

    const CalendarTime now = ticks_to_calendar(ticks);
    WIIU_LOG_INFO(Coreinit,
                  "real-time clock: {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                  now.year,
                  now.month + 1,
                  now.day,
                  now.hour,
                  now.minute,
                  now.second);
    return ticks;
}

void register_time(Registry& registry, Context& context) {
    detail::set_active_context(&context);

    registry.add("coreinit", "OSGetTime", os_get_time);
    registry.add("coreinit", "OSGetSystemTime", os_get_system_time);
    registry.add("coreinit", "OSGetTick", os_get_tick);
    registry.add("coreinit", "OSGetSystemTick", os_get_system_tick);
    registry.add("coreinit", "OSTicksToCalendarTime", os_ticks_to_calendar_time);
    registry.add("coreinit", "OSCalendarTimeToTicks", os_calendar_time_to_ticks);

    // Alarms need a callback the scheduler can run on a guest stack, which is
    // the same mechanism the asynchronous FS calls are waiting on (6.9). Named
    // here so a title that uses them says so instead of failing to link.
    for (const char* name : {
             "OSCreateAlarm",
             "OSCreateAlarmEx",
             "OSSetAlarm",
             "OSSetPeriodicAlarm",
             "OSSetAlarmUserData",
             "OSGetAlarmUserData",
             "OSCancelAlarm",
             "OSCancelAlarms",
             "OSWaitAlarm",
         }) {
        registry.add_stub("coreinit", name);
    }
}

} // namespace wiiu::hle::coreinit
