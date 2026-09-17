#include <wiiu/common/log.hpp>
#include <wiiu/hle/recording.hpp>

#include <fmt/format.h>

#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <map>
#include <system_error>

namespace wiiu::hle {
namespace {

using wiiu::log::Channel;

constexpr std::array<char, 8> file_magic{'W', 'I', 'I', 'U', 'R', 'E', 'C', '1'};

/// Bit 0 of the header flags. A trace's payloads are raw host bytes, so a
/// reader on the other endianness must refuse it rather than replay rubbish.
constexpr u32 flag_little_endian = 1U << 0;

constexpr u64 fnv_offset_basis = 0xCBF29CE484222325ULL;
constexpr u64 fnv_prime = 0x100000001B3ULL;

[[nodiscard]] constexpr u64 fold(u64 hash, u64 value) noexcept {
    for (u32 i = 0; i < 8; ++i) {
        hash ^= (value >> (i * 8)) & 0xFF;
        hash *= fnv_prime;
    }
    return hash;
}

void put_u8(std::ostream& out, u8 value) {
    out.put(static_cast<char>(value));
}

void put_u32(std::ostream& out, u32 value) {
    for (u32 i = 0; i < 4; ++i) {
        put_u8(out, static_cast<u8>((value >> (i * 8)) & 0xFF));
    }
}

void put_u64(std::ostream& out, u64 value) {
    for (u32 i = 0; i < 8; ++i) {
        put_u8(out, static_cast<u8>((value >> (i * 8)) & 0xFF));
    }
}

[[nodiscard]] bool get_u8(std::istream& in, u8& value) {
    const int byte = in.get();
    if (byte == std::char_traits<char>::eof()) {
        return false;
    }
    value = static_cast<u8>(byte);
    return true;
}

[[nodiscard]] bool get_u32(std::istream& in, u32& value) {
    value = 0;
    for (u32 i = 0; i < 4; ++i) {
        u8 byte = 0;
        if (!get_u8(in, byte)) {
            return false;
        }
        value |= static_cast<u32>(byte) << (i * 8);
    }
    return true;
}

[[nodiscard]] bool get_u64(std::istream& in, u64& value) {
    value = 0;
    for (u32 i = 0; i < 8; ++i) {
        u8 byte = 0;
        if (!get_u8(in, byte)) {
            return false;
        }
        value |= static_cast<u64>(byte) << (i * 8);
    }
    return true;
}

[[nodiscard]] u64 payload_as_u64(const std::vector<u8>& payload) {
    u64 value = 0;
    std::memcpy(&value, payload.data(), sizeof(value));
    return value;
}

} // namespace

std::string_view event_kind_name(EventKind kind) noexcept {
    switch (kind) {
        case EventKind::HostTime:
            return "host-time";
        case EventKind::Random:
            return "random";
        case EventKind::Input:
            return "input";
        case EventKind::HostFile:
            return "host-file";
        case EventKind::Checkpoint:
            return "checkpoint";
    }
    return "?";
}

// --------------------------------------------------------------------------
// Hashing
// --------------------------------------------------------------------------

u64 Recording::hash_bytes(std::span<const u8> bytes, u64 seed) noexcept {
    u64 hash = seed == 0 ? fnv_offset_basis : seed;
    for (const u8 byte : bytes) {
        hash ^= byte;
        hash *= fnv_prime;
    }
    return hash;
}

u64 Recording::hash_state(const cpu::CoreState& state) noexcept {
    // Field by field rather than over the object's bytes: CoreState is
    // cacheline-aligned and therefore padded, and padding is uninitialised.
    // Hashing it would make two identical machines hash differently, which is
    // the one failure mode a divergence detector must not have.
    u64 hash = fnv_offset_basis;

    for (const u32 gpr : state.gpr) {
        hash = fold(hash, gpr);
    }
    for (const cpu::FpuRegister& fpr : state.fpr) {
        hash = fold(hash, std::bit_cast<u64>(fpr.ps0));
        hash = fold(hash, std::bit_cast<u64>(fpr.ps1));
    }
    for (const cpu::GqrRegister& gqr : state.gqr) {
        hash = fold(hash, gqr.value);
    }

    for (const u32 sprg : state.sprg) {
        hash = fold(hash, sprg);
    }

    // Translation state (Phase 3.4). A run that disagrees about a segment
    // register or a BAT reads different memory through the same address, which
    // is a divergence that would otherwise only surface as wrong data much
    // later.
    for (const u32 segment : state.sr) {
        hash = fold(hash, segment);
    }
    for (u32 index = 0; index < state.ibat_u.size(); ++index) {
        hash = fold(hash, state.ibat_u[index]);
        hash = fold(hash, state.ibat_l[index]);
        hash = fold(hash, state.dbat_u[index]);
        hash = fold(hash, state.dbat_l[index]);
    }
    hash = fold(hash, state.sdr1);
    hash = fold(hash, state.hid4);

    hash = fold(hash, state.pc);
    hash = fold(hash, state.next_pc);
    hash = fold(hash, state.lr);
    hash = fold(hash, state.ctr);
    hash = fold(hash, state.cr.value);
    hash = fold(hash, state.xer.value);
    hash = fold(hash, state.fpscr);
    hash = fold(hash, state.msr);
    hash = fold(hash, state.srr0);
    hash = fold(hash, state.srr1);
    hash = fold(hash, state.dar);
    hash = fold(hash, state.dsisr);
    hash = fold(hash, state.hid0);
    hash = fold(hash, state.hid1);
    hash = fold(hash, state.hid2);
    hash = fold(hash, state.scr);
    hash = fold(hash, state.wpar);
    hash = fold(hash, state.timebase);
    hash = fold(hash, state.decrementer);
    hash = fold(hash, state.core_id);
    hash = fold(hash, state.halted ? 1U : 0U);

    // The lwarx reservation is part of the machine even though no instruction
    // reads it directly: two runs that disagree about whether a reservation is
    // held will disagree about the next stwcx., which is exactly the kind of
    // divergence a checkpoint exists to catch before its consequences appear.
    hash = fold(hash, state.reserved ? 1U : 0U);
    hash = fold(hash, state.reservation_address);
    return hash;
}

// --------------------------------------------------------------------------
// Lifecycle
// --------------------------------------------------------------------------

void Recording::start_record(const RecordingOptions& options) {
    reset_trace();
    options_ = options;
    mode_ = Mode::Record;
    next_checkpoint_ = 0; // the initial state is worth a checkpoint of its own
}

void Recording::start_replay() {
    cursor_ = 0;
    divergence_ = Divergence{};
    mode_ = Mode::Replay;
    schedule_next_checkpoint();
}

void Recording::reset_trace() {
    mode_ = Mode::Off;
    options_ = RecordingOptions{};
    module_.clear();
    module_hash_ = 0;
    total_instructions_ = 0;
    events_.clear();
    cursor_ = 0;
    next_checkpoint_ = 0;
    divergence_ = Divergence{};
}

void Recording::clear() {
    reset_trace();
    watched_.clear();
    hash_source_ = nullptr;
}

void Recording::set_module(std::string name, u64 hash) {
    module_ = std::move(name);
    module_hash_ = hash;
}

bool Recording::check_module(const std::string& name, u64 hash, std::string& error) const {
    if (!module_.empty() && !name.empty() && module_ != name) {
        error = fmt::format("trace was recorded from '{}', not '{}'", module_, name);
        return false;
    }
    // A zero hash on either side means "not known" rather than "does not
    // match"; refusing those would make traces from older tooling unusable
    // for no gain.
    if (module_hash_ != 0 && hash != 0 && module_hash_ != hash) {
        error =
            fmt::format("trace was recorded from a different build of '{}' ({:#018x} vs {:#018x})",
                        name,
                        module_hash_,
                        hash);
        return false;
    }
    return true;
}

void Recording::finish(u64 instructions) {
    if (mode_ == Mode::Record) {
        total_instructions_ = instructions;
        return;
    }
    if (mode_ != Mode::Replay || total_instructions_ == 0 || total_instructions_ == instructions) {
        return;
    }
    diverge(events_.size(),
            total_instructions_,
            instructions,
            fmt::format("the recording ran {} instructions, the replay {}",
                        total_instructions_,
                        instructions));
}

void Recording::set_hash_source(HashSource source) {
    hash_source_ = std::move(source);
}

void Recording::watch(ppcaddr_t base, u32 size) {
    if (size != 0) {
        watched_.push_back({base, size});
    }
}

std::size_t Recording::remaining() const noexcept {
    return cursor_ >= events_.size() ? 0 : events_.size() - cursor_;
}

// --------------------------------------------------------------------------
// Events
// --------------------------------------------------------------------------

void Recording::append(EventKind kind, u32 channel, std::vector<u8> payload) {
    Event event;
    event.kind = kind;
    event.channel = channel;
    event.sequence = events_.size();
    event.instructions = now();
    event.payload = std::move(payload);
    events_.push_back(std::move(event));
}

void Recording::add_event(EventKind kind,
                          u32 channel,
                          u64 instructions,
                          std::span<const u8> payload) {
    Event event;
    event.kind = kind;
    event.channel = channel;
    event.sequence = events_.size();
    event.instructions = instructions;
    event.payload.assign(payload.begin(), payload.end());
    events_.push_back(std::move(event));
}

void Recording::diverge(u64 sequence, u64 recorded, u64 actual, std::string detail) {
    if (divergence_.detected) {
        // Only the first one. Everything after is a consequence, and listing
        // consequences buries the cause.
        return;
    }
    divergence_.detected = true;
    divergence_.sequence = sequence;
    divergence_.recorded_instructions = recorded;
    divergence_.actual_instructions = actual;
    divergence_.detail = std::move(detail);

    // Nothing after this is meaningful, and leaving the checkpoint due would
    // turn the run loop's per-instruction guard into a per-instruction call
    // for the rest of a non-stopping replay.
    next_checkpoint_ = std::numeric_limits<u64>::max();

    WIIU_LOG_ERROR(Core,
                   "replay diverged at event {} ({} instructions in): {}",
                   sequence,
                   actual,
                   divergence_.detail);
}

bool Recording::exchange(EventKind kind, u32 channel, std::vector<u8>& bytes) {
    if (mode_ == Mode::Record) {
        append(kind, channel, bytes);
        return true;
    }
    if (mode_ != Mode::Replay || divergence_.detected) {
        return false;
    }

    if (cursor_ >= events_.size()) {
        diverge(events_.size(),
                0,
                now(),
                fmt::format("the trace has no more events, but a {} value was asked for",
                            event_kind_name(kind)));
        return false;
    }

    const Event& event = events_[cursor_];
    if (event.kind != kind || event.channel != channel) {
        diverge(event.sequence,
                event.instructions,
                now(),
                fmt::format("expected {}[{}], got {}[{}]",
                            event_kind_name(event.kind),
                            event.channel,
                            event_kind_name(kind),
                            channel));
        return false;
    }
    if (event.payload.size() != bytes.size()) {
        // The caller produced a value of a different width than the one
        // recorded. Falling back to the host value here would leave a replay
        // silently non-deterministic while reporting a clean run, which is the
        // one outcome worse than a loud failure.
        diverge(event.sequence,
                event.instructions,
                now(),
                fmt::format("{}[{}] was recorded as {} bytes, asked for as {}",
                            event_kind_name(kind),
                            channel,
                            event.payload.size(),
                            bytes.size()));
        return false;
    }
    if (event.instructions != now()) {
        // The sharpest check there is: the same title asking the host the
        // same question at a different point has already done something
        // different, and this is the earliest moment that can be known.
        diverge(event.sequence,
                event.instructions,
                now(),
                fmt::format("{}[{}] was recorded at instruction {}, replayed at {}",
                            event_kind_name(kind),
                            channel,
                            event.instructions,
                            now()));
        return false;
    }

    bytes = event.payload;
    ++cursor_;
    return true;
}

// --------------------------------------------------------------------------
// Checkpoints
// --------------------------------------------------------------------------

void Recording::schedule_next_checkpoint() {
    if (mode_ == Mode::Record) {
        next_checkpoint_ = now() + options_.checkpoint_interval;
        return;
    }

    // On replay the checkpoints are wherever the recording put them, rather
    // than wherever the interval would land now. Recomputing them would work
    // only for as long as the two agree, which is precisely what is in doubt.
    next_checkpoint_ = std::numeric_limits<u64>::max();
    for (std::size_t i = cursor_; i < events_.size(); ++i) {
        if (events_[i].kind == EventKind::Checkpoint) {
            next_checkpoint_ = events_[i].instructions;
            return;
        }
    }
}

void Recording::checkpoint(const cpu::CoreState& state, const memory::Memory* memory) {
    if (!active() || divergence_.detected) {
        return;
    }

    u64 hash = hash_state(state);
    if (hash_source_) {
        // Whatever else belongs to the machine but is not in this core --
        // notably the saved contexts of threads that are not currently
        // running. Called only when a checkpoint is actually taken.
        hash = hash_source_(hash);
    }
    if (memory != nullptr) {
        for (const CheckpointRange& range : watched_) {
            scratch_.resize(range.size);
            if (memory->read_bytes(range.base, scratch_)) {
                hash = hash_bytes(scratch_, hash);
            } else {
                hash = fold(hash, 0xDEADBEEFULL); // unreadable, but consistently so
            }
        }
    }

    if (mode_ == Mode::Record) {
        std::vector<u8> payload(sizeof(hash));
        std::memcpy(payload.data(), &hash, sizeof(hash));
        append(EventKind::Checkpoint, 0, std::move(payload));
        schedule_next_checkpoint();
        return;
    }

    if (cursor_ >= events_.size()) {
        // Past the end of the trace: the replay is running longer than the
        // recording did. Not itself a divergence — the run may simply have a
        // larger budget — so stop checking rather than complaining.
        next_checkpoint_ = std::numeric_limits<u64>::max();
        return;
    }

    const Event& event = events_[cursor_];
    if (event.kind != EventKind::Checkpoint) {
        diverge(event.sequence,
                event.instructions,
                now(),
                fmt::format("expected {}[{}] at instruction {}, reached a checkpoint instead",
                            event_kind_name(event.kind),
                            event.channel,
                            event.instructions));
        return;
    }

    if (event.instructions != now()) {
        // schedule_next_checkpoint derives the due point from this very event,
        // so this can only fire on a trace whose events are out of order.
        diverge(event.sequence,
                event.instructions,
                now(),
                fmt::format("checkpoint recorded at instruction {}, reached at {}",
                            event.instructions,
                            now()));
        return;
    }

    const u64 recorded = event.payload.size() == sizeof(u64) ? payload_as_u64(event.payload) : 0;
    if (recorded != hash) {
        diverge(
            event.sequence,
            event.instructions,
            now(),
            fmt::format(
                "state hash {:#018x}, recorded {:#018x} (pc {:#010x})", hash, recorded, state.pc));
        return;
    }

    ++cursor_;
    schedule_next_checkpoint();
}

// --------------------------------------------------------------------------
// Reporting
// --------------------------------------------------------------------------

std::string Recording::format_summary() const {
    std::map<EventKind, std::size_t> counts;
    for (const Event& event : events_) {
        ++counts[event.kind];
    }

    std::string out = "recording:\n";
    out += fmt::format("  mode      {}\n",
                       mode_ == Mode::Record   ? "record"
                       : mode_ == Mode::Replay ? "replay"
                                               : "off");
    if (!module_.empty()) {
        out += fmt::format("  module    {} ({:#018x})\n", module_, module_hash_);
    }
    out += fmt::format("  length    {} instructions\n", total_instructions_);
    out += fmt::format("  events    {}\n", events_.size());
    for (const auto& [kind, count] : counts) {
        out += fmt::format("    {:<12} {}\n", event_kind_name(kind), count);
    }
    if (mode_ == Mode::Replay) {
        out += fmt::format("  consumed  {} of {}\n", cursor_, events_.size());
    }
    if (divergence_.detected) {
        out += fmt::format("  DIVERGED at event {} — {}\n"
                           "            recorded at instruction {}, replayed at {}\n",
                           divergence_.sequence,
                           divergence_.detail,
                           divergence_.recorded_instructions,
                           divergence_.actual_instructions);
    }
    return out;
}

// --------------------------------------------------------------------------
// Files
// --------------------------------------------------------------------------

bool Recording::save(const std::filesystem::path& path, std::string& error) const {
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    if (!out) {
        error = fmt::format("{}: cannot open for writing", path.string());
        return false;
    }

    out.write(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    put_u32(out, format_version);
    put_u32(out, std::endian::native == std::endian::little ? flag_little_endian : 0);

    put_u32(out, static_cast<u32>(module_.size()));
    out.write(module_.data(), static_cast<std::streamsize>(module_.size()));
    put_u64(out, module_hash_);

    put_u64(out, options_.checkpoint_interval);
    put_u64(out, total_instructions_);

    // The watched ranges are part of what a checkpoint hash means, so a trace
    // that omits them cannot be replayed: every hash would be computed over
    // different material and the first checkpoint would report a divergence
    // that is really a configuration mismatch.
    put_u32(out, static_cast<u32>(watched_.size()));
    for (const CheckpointRange& range : watched_) {
        put_u32(out, range.base);
        put_u32(out, range.size);
    }

    put_u64(out, events_.size());

    for (const Event& event : events_) {
        put_u8(out, static_cast<u8>(event.kind));
        put_u32(out, event.channel);
        put_u64(out, event.sequence);
        put_u64(out, event.instructions);
        put_u32(out, static_cast<u32>(event.payload.size()));
        if (!event.payload.empty()) {
            out.write(reinterpret_cast<const char*>(event.payload.data()),
                      static_cast<std::streamsize>(event.payload.size()));
        }
    }

    out.flush();
    if (!out) {
        error = fmt::format("{}: write failed", path.string());
        return false;
    }
    return true;
}

bool Recording::load(const std::filesystem::path& path, std::string& error) {
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        error = fmt::format("{}: cannot open", path.string());
        return false;
    }

    // Everything is parsed into locals and only committed once the whole file
    // has been read. A half-loaded trace is worse than none: `start_replay` on
    // a prefix of somebody's run is perfectly legal and quietly wrong.
    std::error_code size_error;
    const auto file_size =
        static_cast<std::streamoff>(std::filesystem::file_size(path, size_error));
    if (size_error) {
        error = fmt::format("{}: cannot size: {}", path.string(), size_error.message());
        return false;
    }

    /// Bytes left in the file. Every allocation is bounded by this, because
    /// the sizes come from the file and a flipped byte should be an error
    /// message rather than a multi-gigabyte reserve.
    const auto remaining = [&]() -> u64 {
        const std::streamoff at = in.tellg();
        return at < 0 || at > file_size ? 0 : static_cast<u64>(file_size - at);
    };

    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != file_magic) {
        error = fmt::format("{}: not a wiiu-emu trace", path.string());
        return false;
    }

    u32 version = 0;
    u32 flags = 0;
    if (!get_u32(in, version) || !get_u32(in, flags)) {
        error = fmt::format("{}: truncated header", path.string());
        return false;
    }
    if (version != format_version) {
        error = fmt::format("{}: trace format version {}, this build reads {}",
                            path.string(),
                            version,
                            format_version);
        return false;
    }

    const bool trace_is_little_endian = (flags & flag_little_endian) != 0;
    if (trace_is_little_endian != (std::endian::native == std::endian::little)) {
        error = fmt::format("{}: recorded on a {}-endian host",
                            path.string(),
                            trace_is_little_endian ? "little" : "big");
        return false;
    }

    u32 module_length = 0;
    if (!get_u32(in, module_length) || module_length > remaining()) {
        error = fmt::format("{}: truncated header", path.string());
        return false;
    }

    std::string module(module_length, '\0');
    if (module_length != 0) {
        in.read(module.data(), static_cast<std::streamsize>(module_length));
    }

    u64 module_hash = 0;
    u64 checkpoint_interval = 0;
    u64 total_instructions = 0;
    u32 watch_count = 0;
    if (!in || !get_u64(in, module_hash) || !get_u64(in, checkpoint_interval) ||
        !get_u64(in, total_instructions) || !get_u32(in, watch_count)) {
        error = fmt::format("{}: truncated header", path.string());
        return false;
    }

    static constexpr u32 watch_entry_size = 8;
    if (static_cast<u64>(watch_count) * watch_entry_size > remaining()) {
        error = fmt::format(
            "{}: claims {} watched ranges, which the file cannot hold", path.string(), watch_count);
        return false;
    }

    std::vector<CheckpointRange> watched;
    watched.reserve(watch_count);
    for (u32 i = 0; i < watch_count; ++i) {
        CheckpointRange range;
        if (!get_u32(in, range.base) || !get_u32(in, range.size)) {
            error = fmt::format("{}: truncated in watched range {}", path.string(), i);
            return false;
        }
        watched.push_back(range);
    }

    u64 event_count = 0;
    if (!get_u64(in, event_count)) {
        error = fmt::format("{}: truncated header", path.string());
        return false;
    }

    // The smallest an event can be on disk, used only to reject a count the
    // file could not possibly contain before anything is allocated for it.
    static constexpr u64 minimum_event_size = 1 + 4 + 8 + 8 + 4;
    if (event_count * minimum_event_size > remaining() ||
        event_count > remaining() / minimum_event_size) {
        error = fmt::format(
            "{}: claims {} events, which the file cannot hold", path.string(), event_count);
        return false;
    }

    std::vector<Event> events;
    events.reserve(static_cast<std::size_t>(event_count));
    for (u64 i = 0; i < event_count; ++i) {
        Event event;
        u8 kind = 0;
        u32 size = 0;
        if (!get_u8(in, kind) || !get_u32(in, event.channel) || !get_u64(in, event.sequence) ||
            !get_u64(in, event.instructions) || !get_u32(in, size)) {
            error = fmt::format("{}: truncated at event {}", path.string(), i);
            return false;
        }
        if (kind > static_cast<u8>(EventKind::Checkpoint)) {
            error = fmt::format("{}: event {} has unknown kind {}", path.string(), i, kind);
            return false;
        }
        if (size > remaining()) {
            error = fmt::format("{}: event {} claims a {}-byte payload the file cannot hold",
                                path.string(),
                                i,
                                size);
            return false;
        }
        // Replay walks the events in order and compares each against the
        // instruction count it was recorded at, so an out-of-order trace would
        // report divergences that are really its own fault.
        if (!events.empty() && event.instructions < events.back().instructions) {
            error = fmt::format("{}: event {} is out of order ({} after {})",
                                path.string(),
                                i,
                                event.instructions,
                                events.back().instructions);
            return false;
        }

        event.kind = static_cast<EventKind>(kind);
        event.payload.resize(size);
        if (size != 0) {
            in.read(reinterpret_cast<char*>(event.payload.data()),
                    static_cast<std::streamsize>(size));
            if (!in) {
                error = fmt::format("{}: truncated in event {}'s payload", path.string(), i);
                return false;
            }
        }
        events.push_back(std::move(event));
    }

    reset_trace();
    module_ = std::move(module);
    module_hash_ = module_hash;
    options_.checkpoint_interval = checkpoint_interval;
    total_instructions_ = total_instructions;
    watched_ = std::move(watched);
    events_ = std::move(events);
    return true;
}

} // namespace wiiu::hle
