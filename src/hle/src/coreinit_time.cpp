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

#include <fmt/format.h>

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

// --------------------------------------------------------------------------
// Alarms (6.7)
// --------------------------------------------------------------------------
//
// An alarm is a guest callback attached to a point on the timebase. The
// scheduler keeps the deadline, because it is the thing that knows whether
// anything can run before then; when it comes due, the callback runs on its
// own thread, the way hardware runs it on a system thread.
//
// Timebase rather than wall clock is what makes this reproducible: an alarm
// set for a millisecond from now fires after the same number of instructions
// every run, so a replay sees it in the same place (10.4).

/// Alarms register with the scheduler under their guest address. Nothing else
/// uses that number space, and the address is already the alarm's identity.
void arm_alarm(Context& context, u32 handle, u64 deadline);

void fire_alarm(Context& context, u32 handle, u64 now) {
    const auto it = context.alarms.find(handle);
    if (it == context.alarms.end() || !it->second.armed) {
        return;
    }
    Alarm& alarm = it->second;
    ++alarm.fire_count;

    // Periodic alarms re-arm from the deadline, not from now: drifting by the
    // time it took to notice would make a 60 Hz alarm slowly lose frames.
    if (alarm.period != 0) {
        alarm.deadline += alarm.period;
        if (alarm.deadline <= now) {
            // The guest was away long enough to miss one. Hardware does not
            // deliver the backlog either; it carries on from the next slot.
            const u64 missed = ((now - alarm.deadline) / alarm.period) + 1;
            alarm.deadline += missed * alarm.period;
        }
        arm_alarm(context, handle, alarm.deadline);
    } else {
        alarm.armed = false;
    }

    // Anything waiting in OSWaitAlarm wakes whether or not there is a
    // callback: waiting on an alarm with no callback is a legitimate way to
    // use one.
    static_cast<void>(context.scheduler.unblock_all(handle));

    if (alarm.callback == 0) {
        return;
    }

    memory::Memory* const memory = context.memory;
    if (memory == nullptr) {
        WIIU_LOG_ERROR(Coreinit, "alarm {:#010x} fired with no memory to run its callback", handle);
        return;
    }

    // OSAlarmCallback takes the alarm and the interrupted context; nothing
    // here has an OSContext to hand, and a title that dereferences it would
    // fault visibly rather than read something invented.
    const std::array<u32, 2> arguments{handle, 0};
    static_cast<void>(invoke_guest_callback(
        context, *memory, fmt::format("alarm_{:#x}", handle), alarm.callback, arguments));
}

void arm_alarm(Context& context, u32 handle, u64 deadline) {
    context.scheduler.set_timer(
        handle, deadline, [&context, handle](u64 now) { fire_alarm(context, handle, now); });
}

void os_create_alarm(Call& call) {
    const u32 handle = call.next_word();
    Context& context = *detail::active_context();
    context.scheduler.clear_timer(handle);
    context.alarms[handle] = Alarm{};
}

void os_set_alarm(Call& call) {
    const u32 handle = call.next_word();
    const u64 ticks = call.next_long();
    const u32 callback = call.next_word();

    Context& context = *detail::active_context();
    context.memory = &call.memory;

    Alarm& alarm = context.alarms[handle];
    alarm.deadline = call.state.timebase + ticks;
    alarm.period = 0;
    alarm.callback = callback;
    alarm.armed = true;
    arm_alarm(context, handle, alarm.deadline);
    call.state.gpr[3] = 1;
}

void os_set_periodic_alarm(Call& call) {
    const u32 handle = call.next_word();
    const u64 start = call.next_long();
    const u64 interval = call.next_long();
    const u32 callback = call.next_word();

    Context& context = *detail::active_context();
    context.memory = &call.memory;

    Alarm& alarm = context.alarms[handle];

    // OSSetPeriodicAlarm takes an absolute start time, as OSGetTime returns
    // it, and an interval. A start in the past means "begin one interval from
    // now" rather than firing repeatedly to catch up.
    const u64 now = call.state.timebase;
    const u64 epoch = rtc_epoch_ticks(context);
    const u64 start_ticks = start > epoch ? start - epoch : now;

    alarm.deadline = start_ticks > now ? start_ticks : now + interval;
    alarm.period = interval == 0 ? 1 : interval;
    alarm.callback = callback;
    alarm.armed = true;
    arm_alarm(context, handle, alarm.deadline);
    call.state.gpr[3] = 1;
}

void os_cancel_alarm(Call& call) {
    const u32 handle = call.next_word();
    Context& context = *detail::active_context();

    const auto it = context.alarms.find(handle);
    const bool was_armed = it != context.alarms.end() && it->second.armed;
    if (it != context.alarms.end()) {
        it->second.armed = false;
    }
    context.scheduler.clear_timer(handle);

    // A cancelled alarm releases its waiters: they are waiting for something
    // that is now never going to happen.
    static_cast<void>(context.scheduler.unblock_all(handle));
    call.state.gpr[3] = was_armed ? 1U : 0U;
}

void os_cancel_alarms(Call& call) {
    const u32 tag = call.next_word();
    Context& context = *detail::active_context();

    u32 cancelled = 0;
    for (auto& [handle, alarm] : context.alarms) {
        if (!alarm.armed || alarm.tag != tag) {
            continue;
        }
        alarm.armed = false;
        context.scheduler.clear_timer(handle);
        static_cast<void>(context.scheduler.unblock_all(handle));
        ++cancelled;
    }
    call.state.gpr[3] = cancelled;
}

void os_wait_alarm(Call& call) {
    const u32 handle = call.next_word();
    Context& context = *detail::active_context();

    const auto it = context.alarms.find(handle);
    if (it == context.alarms.end() || !it->second.armed) {
        // Nothing to wait for. Returning rather than blocking forever is the
        // difference between a title that carries on and one that hangs.
        call.state.gpr[3] = 0;
        return;
    }

    call.state.gpr[3] = 1;
    context.scheduler.block(call.state, BlockReason::Event, handle);
}

void os_set_alarm_user_data(Call& call) {
    const u32 handle = call.next_word();
    const u32 value = call.next_word();
    detail::active_context()->alarms[handle].user_data = value;
}

void os_get_alarm_user_data(Call& call) {
    const u32 handle = call.next_word();
    Context& context = *detail::active_context();
    const auto it = context.alarms.find(handle);
    call.state.gpr[3] = it == context.alarms.end() ? 0U : it->second.user_data;
}

void os_set_alarm_tag(Call& call) {
    const u32 handle = call.next_word();
    const u32 tag = call.next_word();
    detail::active_context()->alarms[handle].tag = tag;
}

void register_time(Registry& registry, Context& context) {
    detail::set_active_context(&context);

    registry.add("coreinit", "OSGetTime", os_get_time);
    registry.add("coreinit", "OSGetSystemTime", os_get_system_time);
    registry.add("coreinit", "OSGetTick", os_get_tick);
    registry.add("coreinit", "OSGetSystemTick", os_get_system_tick);
    registry.add("coreinit", "OSTicksToCalendarTime", os_ticks_to_calendar_time);
    registry.add("coreinit", "OSCalendarTimeToTicks", os_calendar_time_to_ticks);

    registry.add("coreinit", "OSCreateAlarm", os_create_alarm);
    registry.add("coreinit", "OSCreateAlarmEx", os_create_alarm);
    registry.add("coreinit", "OSSetAlarm", os_set_alarm);
    registry.add("coreinit", "OSSetPeriodicAlarm", os_set_periodic_alarm);
    registry.add("coreinit", "OSSetAlarmUserData", os_set_alarm_user_data);
    registry.add("coreinit", "OSGetAlarmUserData", os_get_alarm_user_data);
    registry.add("coreinit", "OSSetAlarmTag", os_set_alarm_tag);
    registry.add("coreinit", "OSCancelAlarm", os_cancel_alarm);
    registry.add("coreinit", "OSCancelAlarms", os_cancel_alarms);
    registry.add("coreinit", "OSWaitAlarm", os_wait_alarm);
}

} // namespace wiiu::hle::coreinit
