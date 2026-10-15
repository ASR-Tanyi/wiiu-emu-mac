// Phase 8.2 — the GPU's timeline (ADR-004).
//
// GX2 submits command streams and carries on; the GPU gets to them later. A
// title sees "later" three ways -- a retired timestamp it polls, a timestamp
// it waits on, memory the GPU writes when it arrives -- and all three have to
// come out the same on every run, or record/replay (10.4) and the JIT lockstep
// (12.7) stop working. So the GPU runs on the guest's clock:
//
//   * a submission is **copied** when it is made, and given a timestamp and a
//     deadline -- the tick it was submitted at, plus a fixed latency;
//   * it **retires** when the guest's clock reaches its deadline: its packets
//     run, their writes land in guest memory, and the retired timestamp
//     moves. Nothing about it is visible before then;
//   * submissions retire **in order**. A deadline never comes before the one
//     ahead of it, however the ticks they were submitted at compare.
//
// What drives it is outside: the scheduler's timer list asks `next_deadline`
// and calls `retire_until` (8.1 wires that), and a guest thread waiting for a
// timestamp parks until `deadline_of` it, which the idle skip (P2) reaches
// without emulating the wait.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/latte/command_processor.hpp>

#include <deque>
#include <optional>
#include <span>
#include <vector>

namespace wiiu::latte {

class GpuTimeline {
public:
    /// One millisecond at the timebase's 62.15625 MHz. A model, not a
    /// measurement (ADR-004): long enough that a title cannot mistake the GPU
    /// for synchronous, short enough that a frame is a small fraction of it.
    static constexpr u64 default_latency_ticks = 62'156;

    explicit GpuTimeline(CommandProcessor& processor,
                         u64 latency_ticks = default_latency_ticks) noexcept
        : processor_(&processor), latency_(latency_ticks) {}

    /// Queue a stream of host-order words, copied now -- GX2 may reuse the
    /// buffer as soon as this returns. Returns the submission's timestamp,
    /// counting from 1.
    u64 submit(std::span<const u32> words, u64 now_ticks);

    /// Retire every submission whose deadline is at or before `now_ticks`, in
    /// order. Returns how many retired.
    u32 retire_until(u64 now_ticks);

    /// The tick `timestamp` retires at: in the past if it already has, empty
    /// if it was never submitted. A guest thread waiting for it parks until
    /// then.
    [[nodiscard]] std::optional<u64> deadline_of(u64 timestamp) const noexcept;

    /// The next deadline, for the scheduler's timer list. Empty when idle.
    [[nodiscard]] std::optional<u64> next_deadline() const noexcept;

    /// The latest timestamp handed out, and the latest retired. Equal when
    /// the GPU is idle.
    [[nodiscard]] u64 submitted() const noexcept { return submitted_; }

    [[nodiscard]] u64 retired() const noexcept { return retired_; }

    [[nodiscard]] std::size_t pending() const noexcept { return queue_.size(); }

    [[nodiscard]] u64 latency() const noexcept { return latency_; }

private:
    struct Submission {
        u64 timestamp = 0;
        u64 deadline = 0;
        std::vector<u32> words;
    };

    CommandProcessor* processor_;
    u64 latency_;
    std::deque<Submission> queue_;
    u64 submitted_ = 0;
    u64 retired_ = 0;
    u64 last_deadline_ = 0;
    u64 last_retired_deadline_ = 0;
};

} // namespace wiiu::latte
