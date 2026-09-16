// Phase 10.4 — deterministic record and replay.
//
// An emulator bug that only shows up after ten minutes of play is close to
// untestable unless the ten minutes can be repeated exactly. Everything in
// this project is already built to make that possible: the scheduler
// interleaves guest threads deterministically, the timebase advances by
// instructions rather than by the host clock, and the interpreter has no
// other source of variation. So two runs of the same title differ only where
// the emulator consults the *host* — and there are very few such places.
//
// This records them. Every host-derived value goes through `exchange`, which
// either appends it to the trace (record) or replaces it with what was
// recorded (replay). Everything else is recomputed, so a trace stays tiny:
// a run is described by the handful of bytes the host actually contributed,
// not by a log of what happened.
//
// The other half is detecting when a replay has stopped matching. Two checks
// do it:
//
//   * every event carries the instruction count it was produced at, so a
//     replay asking for a value at a different point has already diverged and
//     is told so immediately, at the first place it could possibly be known;
//   * periodic checkpoints hash the CPU state (and any memory ranges worth
//     watching), catching divergence that has not yet reached a host call.
//
// The first is the sharp one. The second is what catches a JIT that computes
// a wrong result in a loop that never calls anything (12.7 uses it for
// exactly that).
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/memory/memory.hpp>

#include <cstring>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace wiiu::hle {

/// Why a value could not be recomputed.
enum class EventKind : u8 {
    HostTime = 0,   ///< the real-time clock; read once, at the first OSGetTime
    Random = 1,     ///< host randomness
    Input = 2,      ///< a controller or touch poll (Phase 9)
    HostFile = 3,   ///< host data that may differ between runs
    Checkpoint = 4, ///< a state hash: compared on replay, never consumed
};

[[nodiscard]] std::string_view event_kind_name(EventKind kind) noexcept;

struct Event {
    EventKind kind = EventKind::HostTime;
    u32 channel = 0;      ///< which clock, which controller port
    u64 sequence = 0;     ///< position in the trace, for diagnostics
    u64 instructions = 0; ///< when it happened — the anchor everything hangs on
    std::vector<u8> payload;
};

struct RecordingOptions {
    /// Instructions between automatic checkpoints. Zero disables them, which
    /// is right for a recording meant only to reproduce host input; a
    /// checkpoint costs a state hash and roughly 40 bytes.
    u64 checkpoint_interval = 1'000'000;

    /// Stop the run at the first divergence. Off keeps going, which is useful
    /// when the question is "how far does it get" rather than "where does it
    /// first differ".
    bool stop_on_divergence = true;
};

/// What went wrong, the first time it went wrong. Only the first divergence is
/// kept: everything after it is a consequence, and reporting the consequences
/// buries the cause.
struct Divergence {
    bool detected = false;
    u64 sequence = 0;
    u64 recorded_instructions = 0;
    u64 actual_instructions = 0;
    std::string detail;
};

/// A memory range included in every checkpoint hash.
struct CheckpointRange {
    ppcaddr_t base = 0;
    u32 size = 0;
};

class Recording {
public:
    enum class Mode : u8 { Off, Record, Replay };

    static constexpr u32 format_version = 1;

    // --- lifecycle ---------------------------------------------------------

    /// Begin recording. Watched ranges survive — they describe what to look
    /// at, not what was seen — but everything else starts empty.
    void start_record(const RecordingOptions& options = {});

    /// Replay a trace that has already been loaded. The options come from the
    /// trace itself, so the checkpoints land where they did when it was made.
    void start_replay();

    void stop() noexcept { mode_ = Mode::Off; }

    [[nodiscard]] Mode mode() const noexcept { return mode_; }

    [[nodiscard]] bool active() const noexcept { return mode_ != Mode::Off; }

    /// The instruction counter events are anchored to. The run loop owns it
    /// and lends it; nothing here counts anything itself, because two counters
    /// that are supposed to agree eventually do not.
    void bind_clock(const u64* instructions) noexcept { clock_ = instructions; }

    [[nodiscard]] u64 now() const noexcept { return clock_ != nullptr ? *clock_ : 0; }

    // --- identity ----------------------------------------------------------

    /// What was running. A replay against a different build is the one
    /// divergence that is not worth debugging, so it is refused up front.
    void set_module(std::string name, u64 hash);

    [[nodiscard]] const std::string& module_name() const noexcept { return module_; }

    [[nodiscard]] u64 module_hash() const noexcept { return module_hash_; }

    /// False if `name`/`hash` are not what the trace was made from, with the
    /// reason in `error`. An empty recorded hash matches anything: a trace
    /// made without one is still usable, just less safe.
    [[nodiscard]] bool check_module(const std::string& name, u64 hash, std::string& error) const;

    // --- host values -------------------------------------------------------

    /// Record or replay one host-derived value.
    ///
    /// Record: append `bytes`, return true (the caller keeps what it produced).
    /// Replay: overwrite `bytes` with the recorded payload and return true, or
    ///         flag a divergence and return false, leaving `bytes` untouched.
    /// Off:    return false.
    ///
    /// Returning false rather than throwing means a caller that ignores the
    /// result still behaves correctly — it just is not deterministic, which is
    /// the same position it was in before any of this existed.
    bool exchange(EventKind kind, u32 channel, std::vector<u8>& bytes);

    /// `exchange` for a trivially copyable value.
    ///
    /// Payload bytes are stored exactly as produced, so a trace carries host
    /// byte order; the file header records which, and a reader on the other
    /// endianness refuses it rather than replaying nonsense.
    template <typename T>
    [[nodiscard]] T value(EventKind kind, u32 channel, T produced) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!active()) {
            return produced;
        }
        std::vector<u8> bytes(sizeof(T));
        std::memcpy(bytes.data(), &produced, sizeof(T));
        if (exchange(kind, channel, bytes) && bytes.size() == sizeof(T)) {
            std::memcpy(&produced, bytes.data(), sizeof(T));
        }
        return produced;
    }

    /// Append an event directly, rather than capturing one. This is how a
    /// trace is *authored* instead of recorded — a scripted input sequence, or
    /// a fixed clock so a comparison run always starts from the same date.
    void add_event(EventKind kind, u32 channel, u64 instructions, std::span<const u8> payload);

    // --- checkpoints -------------------------------------------------------

    /// Anything else that belongs to the machine but is not in one core --
    /// the scheduler's saved thread contexts, say. Folded into every
    /// checkpoint hash, and called only when a checkpoint is actually taken,
    /// so it can be as expensive as it needs to be.
    using HashSource = std::function<u64(u64 seed)>;

    void set_hash_source(HashSource source);

    /// Include a memory range in every checkpoint hash. Nothing is included by
    /// default: hashing 2 GiB every million instructions would cost more than
    /// the emulation.
    void watch(ppcaddr_t base, u32 size);

    [[nodiscard]] std::span<const CheckpointRange> watched() const noexcept { return watched_; }

    /// Hash the machine and record or compare it.
    void checkpoint(const cpu::CoreState& state, const memory::Memory* memory = nullptr);

    /// Called once per instruction by the run loop: takes a checkpoint when
    /// the interval has elapsed. Kept inline and branch-light because it sits
    /// in the hot loop.
    void tick(const cpu::CoreState& state, const memory::Memory* memory = nullptr) {
        if (options_.checkpoint_interval == 0 || now() < next_checkpoint_) {
            return;
        }
        checkpoint(state, memory);
    }

    /// Close the run out: records how many instructions it took, or checks
    /// that a replay took the same number. One number, compared once, and it
    /// catches every divergence that never reached a host call or a
    /// checkpoint — including a run that simply stopped earlier.
    void finish(u64 instructions);

    [[nodiscard]] u64 total_instructions() const noexcept { return total_instructions_; }

    /// The hash a checkpoint compares. Exposed because it is the natural way
    /// to compare two cores directly (12.7) as well as two runs.
    [[nodiscard]] static u64 hash_state(const cpu::CoreState& state) noexcept;

    /// FNV-1a. The one hash in the project, used for state, memory and module
    /// identity, so a mismatch is never about which hash someone picked.
    [[nodiscard]] static u64 hash_bytes(std::span<const u8> bytes, u64 seed = 0) noexcept;

    // --- results -----------------------------------------------------------

    [[nodiscard]] const Divergence& divergence() const noexcept { return divergence_; }

    [[nodiscard]] bool diverged() const noexcept { return divergence_.detected; }

    [[nodiscard]] bool stop_requested() const noexcept {
        return divergence_.detected && options_.stop_on_divergence;
    }

    [[nodiscard]] std::span<const Event> events() const noexcept { return events_; }

    [[nodiscard]] std::size_t event_count() const noexcept { return events_.size(); }

    /// Events a replay has not reached yet. Nonzero at the end means the
    /// replay did less than the recording did — a divergence the instruction
    /// anchor cannot catch, because nothing ever asked.
    [[nodiscard]] std::size_t remaining() const noexcept;

    [[nodiscard]] const RecordingOptions& options() const noexcept { return options_; }

    [[nodiscard]] std::string format_summary() const;

    // --- files -------------------------------------------------------------

    [[nodiscard]] bool save(const std::filesystem::path& path, std::string& error) const;
    [[nodiscard]] bool load(const std::filesystem::path& path, std::string& error);

    /// Back to nothing at all, watched ranges included.
    void clear();

private:
    /// Everything about a run, without disturbing the configuration.
    void reset_trace();

    void diverge(u64 sequence, u64 recorded, u64 actual, std::string detail);
    void append(EventKind kind, u32 channel, std::vector<u8> payload);
    void schedule_next_checkpoint();

    Mode mode_ = Mode::Off;
    RecordingOptions options_;

    std::string module_;
    u64 module_hash_ = 0;
    u64 total_instructions_ = 0;

    std::vector<Event> events_;
    std::size_t cursor_ = 0; ///< replay position
    u64 next_checkpoint_ = 0;

    const u64* clock_ = nullptr;
    Divergence divergence_;

    std::vector<CheckpointRange> watched_;
    HashSource hash_source_;
    std::vector<u8> scratch_; ///< reused by checkpoint hashing
};

} // namespace wiiu::hle
