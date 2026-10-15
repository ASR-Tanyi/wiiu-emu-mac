// Phase 8.2 — see timeline.hpp.

#include <wiiu/latte/timeline.hpp>

#include <algorithm>

namespace wiiu::latte {

u64 GpuTimeline::submit(std::span<const u32> words, u64 now_ticks) {
    Submission submission;
    submission.timestamp = ++submitted_;

    // In order, always: a submission made "earlier" by a clock that went
    // backwards (two cores, one a little behind) still retires after the one
    // ahead of it in the queue, because the GPU reads its ring in order.
    submission.deadline = std::max(now_ticks + latency_, last_deadline_);
    last_deadline_ = submission.deadline;

    submission.words.assign(words.begin(), words.end());
    queue_.push_back(std::move(submission));
    return submitted_;
}

u32 GpuTimeline::retire_until(u64 now_ticks) {
    u32 count = 0;
    while (!queue_.empty() && queue_.front().deadline <= now_ticks) {
        // Popped before running, so a packet that somehow submits more work
        // (nothing does yet) cannot be confused with this one.
        Submission submission = std::move(queue_.front());
        queue_.pop_front();

        // The clock a packet sees is the moment it retires, not the moment
        // whoever called this happened to look.
        processor_->set_clock(submission.deadline);
        static_cast<void>(processor_->execute(submission.words));

        retired_ = submission.timestamp;
        last_retired_deadline_ = submission.deadline;
        ++count;
    }
    return count;
}

std::optional<u64> GpuTimeline::deadline_of(u64 timestamp) const noexcept {
    if (timestamp == 0 || timestamp > submitted_) {
        return std::nullopt;
    }
    if (timestamp <= retired_) {
        return last_retired_deadline_;
    }
    // Pending submissions are in timestamp order, one each.
    return queue_[timestamp - retired_ - 1].deadline;
}

std::optional<u64> GpuTimeline::next_deadline() const noexcept {
    if (queue_.empty()) {
        return std::nullopt;
    }
    return queue_.front().deadline;
}

} // namespace wiiu::latte
