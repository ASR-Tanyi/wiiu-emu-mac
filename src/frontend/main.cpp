// wiiu-emu entry point.
//
// Phase 1.2: argument handling, logging bring-up, version banner. The window
// arrives in Phase 1.4, and there is nothing to emulate until Phase 6.

#include <wiiu/common/config.hpp>
#include <wiiu/common/log.hpp>
#include <wiiu/common/version.hpp>
#include <wiiu/cpu/backtrace.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/hle/recording.hpp>
#include <wiiu/loader/rpl.hpp>
#include <wiiu/loader/vfs.hpp>
#include <wiiu/memory/memory.hpp>

#include <fmt/format.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using wiiu::log::Channel;
using wiiu::log::Level;

constexpr std::string_view kUsage = R"(usage: wiiu-emu [options]

options:
  -h, --help              show this message
  -r, --run FILE          load and run an RPX
      --trace             log every HLE call as it happens
      --stats             print HLE call counts when the run ends
      --record FILE       write a deterministic trace of this run
      --replay FILE       re-run a trace and report where it diverges
      --checkpoint N      instructions between replay state hashes
  -V, --version           print the version and exit
  -c, --config PATH       settings file (default: ./wiiu-emu.toml)
  -s, --set KEY=VALUE     override one setting (repeatable)
      --list-settings     print every setting and exit
      --log LEVEL         set every channel to LEVEL
      --log CHAN=LEVEL    set one channel's level (repeatable)
      --log-file PATH     also write the log to PATH
      --quiet             suppress console output

channels: core cpu mem loader coreinit gx2 gpu ax input frontend
levels:   trace debug info warn error critical off
)";

struct Arguments {
    bool help = false;
    bool version = false;
    bool quiet = false;
    bool list_settings = false;
    std::string run_path;
    bool trace = false;
    bool stats = false;
    std::string record_path;
    std::string replay_path;
    wiiu::u64 checkpoint_interval = wiiu::hle::RecordingOptions{}.checkpoint_interval;
    bool checkpoint_given = false;
    std::string log_file;
    std::string config_path = "wiiu-emu.toml";
    std::vector<std::string> overrides;
    Level default_level = Level::Info;
    std::vector<std::pair<Channel, Level>> channel_levels;
};

/// Accepts "debug" (all channels) or "gpu=trace" (one channel).
bool apply_log_spec(std::string_view spec, Arguments& args) {
    const std::size_t sep = spec.find('=');
    if (sep == std::string_view::npos) {
        return wiiu::log::parse_level(spec, args.default_level);
    }

    Channel channel{};
    Level level{};
    if (!wiiu::log::parse_channel(spec.substr(0, sep), channel)) {
        return false;
    }
    if (!wiiu::log::parse_level(spec.substr(sep + 1), level)) {
        return false;
    }
    args.channel_levels.emplace_back(channel, level);
    return true;
}

/// Returns false on a usage error, having already reported it.
bool parse_arguments(std::span<const std::string_view> argv, Arguments& args) {
    for (std::size_t i = 0; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];

        const auto next = [&](std::string_view& out) {
            if (i + 1 >= argv.size()) {
                fmt::print(stderr, "wiiu-emu: {} requires a value\n", arg);
                return false;
            }
            out = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg == "-V" || arg == "--version") {
            args.version = true;
        } else if (arg == "--quiet") {
            args.quiet = true;
        } else if (arg == "-r" || arg == "--run") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            args.run_path = std::string{value};
        } else if (arg == "--trace") {
            args.trace = true;
        } else if (arg == "--stats") {
            args.stats = true;
        } else if (arg == "--record") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            args.record_path = std::string{value};
        } else if (arg == "--replay") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            args.replay_path = std::string{value};
        } else if (arg == "--checkpoint") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            const std::string text{value};
            char* end = nullptr;
            args.checkpoint_interval = std::strtoull(text.c_str(), &end, 0);
            args.checkpoint_given = true;
            if (end == text.c_str() || *end != '\0') {
                fmt::print(stderr, "wiiu-emu: --checkpoint wants a number, got '{}'\n", value);
                return false;
            }
        } else if (arg == "--list-settings") {
            args.list_settings = true;
        } else if (arg == "-c" || arg == "--config") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            args.config_path = std::string{value};
        } else if (arg == "-s" || arg == "--set") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            args.overrides.emplace_back(value);
        } else if (arg == "--log") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            if (!apply_log_spec(value, args)) {
                fmt::print(stderr, "wiiu-emu: unrecognised --log value '{}'\n", value);
                return false;
            }
        } else if (arg == "--log-file") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            args.log_file = std::string{value};
        } else {
            fmt::print(stderr, "wiiu-emu: unrecognised option '{}'\n\n{}", arg, kUsage);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> raw;
    raw.reserve(static_cast<std::size_t>(argc > 1 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) {
        raw.emplace_back(argv[i]);
    }

    Arguments args;
    if (!parse_arguments(raw, args)) {
        return 2;
    }

    if (args.help) {
        fmt::print("{}", kUsage);
        return 0;
    }

    if (args.version) {
        fmt::print("{}\n", wiiu::version::full());
        return 0;
    }

    if (args.list_settings) {
        fmt::print("settings:\n{}", wiiu::config::describe_fields());
        return 0;
    }

    wiiu::log::init({
        .default_level = args.default_level,
        .console = !args.quiet,
        .file = args.log_file,
        .ring_capacity = 4096,
    });

    WIIU_LOG_INFO(Core, "{}", wiiu::version::full());
    WIIU_LOG_INFO(Core, "compiled by {}", wiiu::version::compiler);

    wiiu::config::Config config;
    const auto loaded = config.load(std::filesystem::path{args.config_path});
    if (!loaded.ok) {
        WIIU_LOG_ERROR(Core, "{}: {}", args.config_path, loaded.error);
        wiiu::log::shutdown();
        return 1;
    }
    for (const std::string& warning : loaded.warnings) {
        WIIU_LOG_WARN(Core, "{}: {}", args.config_path, warning);
    }

    for (const std::string& assignment : args.overrides) {
        std::string error;
        if (!config.apply_override(assignment, error)) {
            WIIU_LOG_ERROR(Core, "--set {}: {}", assignment, error);
            wiiu::log::shutdown();
            return 2;
        }
    }

    // The settings file names a log level too; an explicit --log wins, so it
    // is applied after.
    if (Level from_config{}; wiiu::log::parse_level(config.base().log_level, from_config)) {
        wiiu::log::set_all_levels(from_config);
    }
    if (args.default_level != Level::Info) {
        wiiu::log::set_all_levels(args.default_level);
    }
    for (const auto& [channel, level] : args.channel_levels) {
        wiiu::log::set_level(channel, level);
    }

    WIIU_LOG_INFO(Core,
                  "settings: {}{}",
                  args.config_path,
                  loaded.warnings.empty() ? "" : " (with warnings)");

#if WIIU_PLATFORM_APPLE
    WIIU_LOG_INFO(Core, "platform: macOS / Apple silicon");
#else
    WIIU_LOG_WARN(Core,
                  "platform: core-only build — the emulator requires macOS 13+ on Apple silicon");
#endif

    if (args.run_path.empty()) {
        WIIU_LOG_INFO(Core, "nothing to run — pass --run FILE.rpx");
        wiiu::log::shutdown();
        return 0;
    }

    // --- bring up a machine ------------------------------------------------
    //
    // The HLE trampoline area sits outside every Cafe OS region, so it is
    // mapped alongside them rather than being part of the guest's own map.
    std::vector<wiiu::memory::Region> regions(wiiu::memory::default_regions().begin(),
                                              wiiu::memory::default_regions().end());
    regions.insert(regions.begin() + 1,
                   wiiu::memory::Region{"hle",
                                        wiiu::hle::Registry::trampoline_base,
                                        wiiu::hle::Registry::trampoline_size,
                                        0,
                                        wiiu::memory::Backing::None,
                                        wiiu::memory::read_write_execute});

    wiiu::memory::Memory memory;
    std::string error;
    if (!memory.initialize(regions, error)) {
        WIIU_LOG_ERROR(Core, "{}", error);
        wiiu::log::shutdown();
        return 1;
    }

    wiiu::loader::Rpl rpl;
    const auto parsed = wiiu::loader::parse_file(args.run_path, rpl);
    if (!parsed.ok) {
        WIIU_LOG_ERROR(Loader, "{}: {}", args.run_path, parsed.error);
        wiiu::log::shutdown();
        return 1;
    }
    for (const std::string& warning : parsed.warnings) {
        WIIU_LOG_WARN(Loader, "{}", warning);
    }

    wiiu::hle::Registry registry;
    // A ring of recent calls is cheap to keep and is the first thing worth
    // having when a run ends badly.
    registry.set_trace_capacity(32);
    registry.set_trace_logging(args.trace);

    // Both of these are handed to the coreinit context by pointer, so they are
    // declared first and therefore outlive it.
    wiiu::loader::VirtualFileSystem vfs;
    wiiu::hle::Recording recording;

    wiiu::hle::coreinit::Context coreinit;
    wiiu::hle::coreinit::register_all(registry, coreinit);

    if (std::string heap_error; !coreinit.heaps.create_default_heaps(memory, heap_error)) {
        WIIU_LOG_ERROR(Core, "{}", heap_error);
        wiiu::log::shutdown();
        return 1;
    }

    // Content comes from a decrypted title directory beside the executable, if
    // there is one: /code/foo.rpx implies ../content next to it.
    const std::filesystem::path executable{args.run_path};
    if (executable.has_parent_path() && executable.parent_path().filename() == "code") {
        std::string mount_error;
        if (vfs.mount_title_directory(executable.parent_path().parent_path(), mount_error)) {
            coreinit.filesystem.set_filesystem(&vfs);
        } else {
            WIIU_LOG_DEBUG(Loader, "no title directory: {}", mount_error);
        }
    }

    // --- record / replay (10.4) --------------------------------------------
    if (!args.record_path.empty() && !args.replay_path.empty()) {
        WIIU_LOG_ERROR(Core, "--record and --replay are mutually exclusive");
        wiiu::log::shutdown();
        return 2;
    }
    if (args.checkpoint_given && !args.replay_path.empty()) {
        // The interval comes from the trace on replay, because a replay has to
        // look for checkpoints where the recording put them. Silently ignoring
        // the flag would leave someone wondering why it changed nothing.
        WIIU_LOG_ERROR(Core, "--checkpoint applies to --record; a replay uses the trace's own");
        wiiu::log::shutdown();
        return 2;
    }
    if (!args.record_path.empty() || !args.replay_path.empty()) {
        // The module's own bytes identify it. Replaying a trace against a
        // different build is the one divergence not worth debugging, so it is
        // refused rather than reported.
        wiiu::u64 module_hash = 0;
        if (std::ifstream image{args.run_path, std::ios::binary}) {
            const std::vector<wiiu::u8> bytes{std::istreambuf_iterator<char>{image},
                                              std::istreambuf_iterator<char>{}};
            module_hash = wiiu::hle::Recording::hash_bytes(bytes);
        }
        const std::string module_name = std::filesystem::path{args.run_path}.filename().string();

        if (args.replay_path.empty()) {
            recording.start_record({.checkpoint_interval = args.checkpoint_interval});
            recording.set_module(module_name, module_hash);
        } else {
            std::string trace_error;
            if (!recording.load(args.replay_path, trace_error)) {
                WIIU_LOG_ERROR(Core, "{}", trace_error);
                wiiu::log::shutdown();
                return 1;
            }
            if (!recording.check_module(module_name, module_hash, trace_error)) {
                WIIU_LOG_ERROR(Core, "{}", trace_error);
                wiiu::log::shutdown();
                return 1;
            }
            recording.start_replay();
            WIIU_LOG_INFO(Core,
                          "replaying {} ({} events, {} instructions)",
                          args.replay_path,
                          recording.event_count(),
                          recording.total_instructions());
        }
        coreinit.recording = &recording;
    }

    wiiu::cpu::CoreState state;
    wiiu::hle::LoadedModule module;
    const auto load_result = wiiu::hle::load_module(rpl, memory, registry, state, module);
    if (!load_result.ok) {
        WIIU_LOG_ERROR(Loader, "{}", load_result.error);
        wiiu::log::shutdown();
        return 1;
    }
    for (const std::string& warning : load_result.warnings) {
        WIIU_LOG_WARN(Loader, "{}", warning);
    }

    wiiu::cpu::Interpreter interpreter{state, memory};
    // Adopt the entry point as the main thread, so anything it spawns is
    // scheduled alongside it. Without this the scheduler has no record of the
    // caller, and a blocking call quietly does nothing.
    static_cast<void>(coreinit.scheduler.adopt_main(state, wiiu::hle::main_thread_handle));
    coreinit.scheduler.set_exit_callback([&memory](wiiu::s32 value, wiiu::ppcaddr_t out) {
        static_cast<void>(memory.try_write<wiiu::u32>(out, static_cast<wiiu::u32>(value)));
    });

    const wiiu::hle::RunReport report = wiiu::hle::run(
        interpreter, memory, registry, {.scheduler = &coreinit.scheduler, .recording = &recording});

    WIIU_LOG_INFO(Core,
                  "{} after {} instructions and {} HLE calls",
                  wiiu::hle::run_outcome_name(report.outcome),
                  report.instructions,
                  report.hle_calls);

    if (recording.active()) {
        fmt::print("\n{}", recording.format_summary());
        if (recording.mode() == wiiu::hle::Recording::Mode::Record) {
            std::string trace_error;
            if (recording.save(args.record_path, trace_error)) {
                WIIU_LOG_INFO(Core, "wrote {}", args.record_path);
            } else {
                WIIU_LOG_ERROR(Core, "{}", trace_error);
            }
        } else if (recording.remaining() != 0 && !recording.diverged()) {
            // Nothing asked for the rest of the trace, so nothing could
            // compare it. The replay did less than the recording did.
            WIIU_LOG_WARN(Core, "{} recorded events were never reached", recording.remaining());
        }
    }

    for (const wiiu::hle::Function* function : registry.unimplemented_called()) {
        WIIU_LOG_WARN(Core,
                      "unimplemented {}.{} was called {} times",
                      function->library,
                      function->name,
                      function->call_count);
    }

    // A run that ended badly gets the full picture without being asked: where
    // it stopped, the code there, the registers, the call stack, and what it
    // was calling just before. Reproducing a fault by hand to collect this is
    // the slowest part of debugging an emulator.
    if (report.outcome != wiiu::hle::RunOutcome::Returned) {
        fmt::print(stderr, "\n{}\n", wiiu::cpu::format_state_dump(state, memory));
        fmt::print(stderr, "{}\n", registry.format_summary());

        if (report.outcome == wiiu::hle::RunOutcome::Deadlocked) {
            fmt::print(stderr, "threads:\n");
            for (const wiiu::hle::Thread* thread : coreinit.scheduler.threads()) {
                fmt::print(stderr,
                           "  {:#010x} {:<20} {:<10} waiting on {}\n",
                           thread->handle,
                           thread->name,
                           wiiu::hle::thread_state_name(thread->state),
                           wiiu::hle::block_reason_name(thread->block_reason));
            }
        }
    } else if (args.stats) {
        fmt::print("\n{}", registry.format_summary());
    }

    wiiu::log::shutdown();
    return report.outcome == wiiu::hle::RunOutcome::Returned ? 0 : 1;
}
