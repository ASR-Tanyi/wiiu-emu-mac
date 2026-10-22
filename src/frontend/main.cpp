// wiiu-emu entry point.
//
// Phase 1.2: argument handling, logging bring-up, version banner. Phase 1.4
// added the window; Phase 7.6 puts a title's OSScreen output in it.

#include <wiiu/common/config.hpp>
#include <wiiu/common/log.hpp>
#include <wiiu/common/version.hpp>
#include <wiiu/cpu/backtrace.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/loader.hpp>
#include <wiiu/hle/proc_ui.hpp>
#include <wiiu/hle/recording.hpp>
#include <wiiu/hle/system_libraries.hpp>
#include <wiiu/loader/rpl.hpp>
#include <wiiu/loader/vfs.hpp>
#include <wiiu/memory/memory.hpp>

#include "display.hpp"
#include "platform/jit_probe.hpp"

#include <fmt/format.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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
      --window            open a window: alone, a test pattern; with --run,
                          the title's screens (OSScreen today, GX2 later)
      --frames N          with --window alone, stop after N frames
      --gpu-validation    run under Metal's API and shader validation
      --gpu-hud           show Metal's performance HUD over the window
      --capture-frame N   write a GPU capture of frame N (open it in Xcode)
      --capture-path FILE where --capture-frame writes (wiiu-emu.gputrace)
      --jit-check         prove the MAP_JIT + entitlement path and exit
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
    bool window = false;
    bool jit_check = false;
    bool gpu_validation = false;
    bool gpu_hud = false;
    wiiu::u64 capture_frame = 0;
    std::string capture_path = "wiiu-emu.gputrace";
    wiiu::u64 frames = 0; ///< zero means until the window is closed
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
        } else if (arg == "--window") {
            args.window = true;
        } else if (arg == "--frames") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            const std::string text{value};
            char* end = nullptr;
            args.frames = std::strtoull(text.c_str(), &end, 0);
            if (end == text.c_str() || *end != '\0') {
                fmt::print(stderr, "wiiu-emu: --frames wants a number, got '{}'\n", value);
                return false;
            }
        } else if (arg == "--jit-check") {
            args.jit_check = true;
        } else if (arg == "--gpu-validation") {
            args.gpu_validation = true;
        } else if (arg == "--gpu-hud") {
            args.gpu_hud = true;
        } else if (arg == "--capture-frame") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            const std::string text{value};
            char* end = nullptr;
            args.capture_frame = std::strtoull(text.c_str(), &end, 0);
            if (end == text.c_str() || *end != '\0' || args.capture_frame == 0) {
                fmt::print(stderr,
                           "wiiu-emu: --capture-frame wants a frame number from 1, got '{}'\n",
                           value);
                return false;
            }
        } else if (arg == "--capture-path") {
            std::string_view value;
            if (!next(value)) {
                return false;
            }
            args.capture_path = std::string{value};
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

/// Metal reads its tooling switches from the environment when the first
/// device is made, so they are set here, before anything has asked for one
/// (7.8). Existing settings are kept: someone who exported a variable by hand
/// meant it.
void apply_gpu_tooling(const Arguments& args) {
    if (args.gpu_validation) {
        setenv("MTL_DEBUG_LAYER", "1", 0);
        if (args.capture_frame == 0) {
            setenv("MTL_SHADER_VALIDATION", "1", 0);
        } else {
            // Metal will not capture a process running shader validation, so
            // asking for both gets the capture and API validation only.
            WIIU_LOG_WARN(Gpu,
                          "--capture-frame turns shader validation off: Metal cannot "
                          "capture with it on (API validation stays on)");
        }
    }
    if (args.gpu_hud) {
        setenv("MTL_HUD_ENABLED", "1", 0);
    }
    if (args.capture_frame != 0) {
        // Outside Xcode, a process may only capture if it said so up front.
        setenv("MTL_CAPTURE_ENABLED", "1", 0);
    }
}

[[nodiscard]] wiiu::frontend::DisplayOptions display_options(const Arguments& args,
                                                             std::string title) {
    wiiu::frontend::DisplayOptions options;
    options.title = std::move(title);
    options.capture_frame = args.capture_frame;
    options.capture_path = args.capture_path;
    return options;
}

/// A triangle that turns, over a clear that drifts.
///
/// The clear proves frames are reaching the display (1.4); the triangle
/// proves a pipeline compiled, a buffer written by the CPU reached the GPU,
/// and a draw happened (7.6). Its vertices are rewritten every frame into a
/// ring of `frames_in_flight` buffers -- the pattern every per-frame upload
/// follows, and the reason the backend bounds its frames in flight (7.2).
constexpr std::string_view kDemoShader = R"msl(
#include <metal_stdlib>
using namespace metal;

struct Out {
    float4 position [[position]];
    float3 colour;
};

vertex Out vertex_main(uint id [[vertex_id]], const device float2* positions [[buffer(0)]]) {
    Out out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.colour = float3(id == 0 ? 1.0 : 0.0, id == 1 ? 1.0 : 0.0, id == 2 ? 1.0 : 0.0);
    return out;
}

fragment float4 fragment_main(Out in [[stage_in]]) {
    return float4(in.colour, 1.0);
}
)msl";

int run_window(const Arguments& args) {
    wiiu::frontend::Display display;
    std::string error;
    if (!display.open(display_options(args, "wiiu-emu"), error)) {
        WIIU_LOG_ERROR(Gpu, "{}", error);
        return 1;
    }
    wiiu::gpu::GpuBackend& backend = display.backend();

    wiiu::gpu::PipelineDesc desc;
    desc.label = "demo triangle";
    desc.vertex_source = std::string{kDemoShader};
    desc.color_formats[0] = wiiu::gpu::TextureFormat::BGRA8Unorm;
    const wiiu::gpu::PipelineHandle pipeline = backend.create_pipeline(desc, error);
    if (!pipeline.valid()) {
        WIIU_LOG_ERROR(Gpu, "{}", error);
        return 1;
    }

    std::array<wiiu::gpu::BufferHandle, wiiu::gpu::frames_in_flight> vertices{};
    for (wiiu::gpu::BufferHandle& buffer : vertices) {
        wiiu::gpu::BufferDesc buffer_desc;
        buffer_desc.size = 6 * sizeof(float);
        buffer_desc.storage = wiiu::gpu::Storage::Shared;
        buffer_desc.label = "demo vertices";
        buffer = backend.create_buffer(buffer_desc);
    }

    const auto draw = [&](wiiu::gpu::GpuBackend& gpu, const wiiu::gpu::FrameInfo& frame) {
        const auto phase = static_cast<float>(frame.index % 240) / 240.0F;

        // This frame's slot in the ring. The two before it may still be on
        // the GPU; this one cannot be, because the backend lets no more than
        // `frames_in_flight` frames be outstanding.
        const wiiu::gpu::BufferHandle buffer = vertices[frame.index % vertices.size()];
        const float angle = phase * 6.2831853F;
        const float aspect =
            frame.width == 0 ? 1.0F
                             : static_cast<float>(frame.height) / static_cast<float>(frame.width);
        std::array<float, 6> corners{};
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const float theta = angle + (static_cast<float>(corner) * 2.0943951F);
            corners[corner * 2] = 0.6F * std::cos(theta) * aspect;
            corners[(corner * 2) + 1] = 0.6F * std::sin(theta);
        }
        std::array<wiiu::u8, sizeof(corners)> bytes{};
        std::memcpy(bytes.data(), corners.data(), bytes.size());
        static_cast<void>(gpu.write_buffer(buffer, 0, bytes));

        wiiu::gpu::RenderPassDesc pass;
        pass.label = "demo";
        pass.color_count = 1;
        pass.colors[0].texture = frame.drawable;
        pass.colors[0].load = wiiu::gpu::LoadAction::Clear;
        // Nothing reads this back -- it goes straight to the display -- but
        // the display is a reader, so Store it is (P20).
        pass.colors[0].store = wiiu::gpu::StoreAction::Store;
        pass.colors[0].clear = wiiu::gpu::ClearColor{phase, 0.25F, 1.0F - phase, 1.0F};
        if (!gpu.begin_render_pass(pass)) {
            return;
        }
        gpu.set_pipeline(pipeline);
        gpu.set_viewport(wiiu::gpu::Viewport{0.0F,
                                             0.0F,
                                             static_cast<float>(frame.width),
                                             static_cast<float>(frame.height),
                                             0.0F,
                                             1.0F});
        gpu.bind_buffer(wiiu::gpu::ShaderStage::Vertex, 0, buffer, 0);
        wiiu::gpu::DrawCall call;
        call.vertex_count = 3;
        gpu.draw(call);
        gpu.end_render_pass();
    };

    while (display.frame(draw)) {
        if (args.frames != 0 && display.presented() >= args.frames) {
            break;
        }
    }

    WIIU_LOG_INFO(Frontend, "presented {} frames", display.presented());
    for (const wiiu::gpu::BufferHandle buffer : vertices) {
        backend.destroy(buffer);
    }
    backend.destroy(pipeline);
    return 0;
}

/// Hand OSScreen's flips to the display (6.8 -> 7.6), and GX2's.
///
/// A flip copies the visible frame out at once, which is what scan-out is:
/// the title will be drawing into that buffer again two flips from now. A
/// GX2 frame the GPU filled is a texture already, and is shown as one. The
/// TV's flip presents; the GamePad's only updates its image, so a title that
/// flips both every frame does not present twice. A title that only ever
/// uses the GamePad presents on its flips instead.
void connect_screens(wiiu::hle::coreinit::Context& coreinit, wiiu::frontend::Display& display) {
    using wiiu::hle::coreinit::ScreenFlip;
    using wiiu::hle::coreinit::ScreenId;

    coreinit.screens.on_flip = [&display](const ScreenFlip& flip,
                                          const wiiu::memory::Memory& memory) {
        const auto screen = flip.screen == ScreenId::Tv ? wiiu::gpu::PresentScreen::Tv
                                                        : wiiu::gpu::PresentScreen::Drc;
        const wiiu::u32 size = flip.geometry.frame_bytes();
        const wiiu::u8* const host =
            flip.texture.valid() ? nullptr
                                 : memory.translate(flip.address, size, wiiu::memory::Access::Read);

        if (flip.texture.valid()) {
            // A frame the GPU filled (ADR-006): shown where it is.
            display.presenter().show(
                screen,
                wiiu::gpu::ScreenTexture{flip.texture, flip.geometry.width, flip.geometry.height});
        } else if (host != nullptr) {
            display.presenter().update(screen,
                                       wiiu::gpu::ScreenImage{{host, size},
                                                              flip.geometry.width,
                                                              flip.geometry.height,
                                                              flip.geometry.pitch * 4});
        } else {
            WIIU_LOG_WARN(Frontend,
                          "{} flip: the frame at {:#010x} is not readable",
                          wiiu::hle::coreinit::screen_name(flip.screen),
                          flip.address);
        }
        // A disabled screen shows black: not drawn, and its space given up.
        display.presenter().set_shown(screen, flip.enabled);

        if (flip.screen == ScreenId::Drc &&
            display.presenter().has_image(wiiu::gpu::PresentScreen::Tv)) {
            return display.poll();
        }
        return display.present();
    };
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

    if (args.jit_check) {
        const wiiu::platform::JitProbeResult probe = wiiu::platform::probe_jit();
        if (probe.ok()) {
            WIIU_LOG_INFO(Core, "jit check: {}", probe.detail);
        } else {
            WIIU_LOG_ERROR(Core, "jit check: {}", probe.detail);
        }
        wiiu::log::shutdown();
        return probe.ok() ? 0 : 1;
    }

    apply_gpu_tooling(args);

    if (args.window && args.run_path.empty()) {
        const int status = run_window(args);
        wiiu::log::shutdown();
        return status;
    }

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

    // ProcUI drives the title's main loop, and the system libraries are what
    // it touches on the way there (6.11).
    wiiu::hle::procui::State procui;
    wiiu::hle::procui::register_all(registry, coreinit, procui);
    wiiu::hle::register_system_libraries(registry);

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

    // --- the screen (7.6) ---------------------------------------------------
    //
    // Opened before the title starts, so that its first flip has somewhere
    // to go. Declared after the coreinit context: the flip hook refers to it,
    // and it has to be torn down while that hook can no longer be called.
    wiiu::frontend::Display display;
    if (args.window) {
        std::string display_error;
        const std::string title =
            fmt::format("wiiu-emu - {}", std::filesystem::path{args.run_path}.filename().string());
        if (!display.open(display_options(args, title), display_error)) {
            WIIU_LOG_ERROR(Gpu, "{}", display_error);
            wiiu::log::shutdown();
            return 1;
        }
        connect_screens(coreinit, display);
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

    // With a window the title runs until it returns or the window is closed,
    // as a program does. Without one the budget stands, so a run that never
    // finishes still ends and says where it was.
    wiiu::hle::RunOptions run_options{.scheduler = &coreinit.scheduler, .recording = &recording};
    if (args.window) {
        run_options.budget = std::numeric_limits<wiiu::u64>::max();
    }
    const wiiu::hle::RunReport report = wiiu::hle::run(interpreter, memory, registry, run_options);

    // Closing the window halts the core from inside a flip, which looks like
    // any other halt to the run loop. It is the one that is not a problem.
    const bool closed_by_user = coreinit.screens.stop_requested;
    coreinit.screens.on_flip = nullptr;
    if (closed_by_user) {
        WIIU_LOG_INFO(Frontend,
                      "window closed after {} frames and {} instructions",
                      display.presented(),
                      report.instructions);
    }

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
    if (report.outcome != wiiu::hle::RunOutcome::Returned && !closed_by_user) {
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

    display.close();
    wiiu::log::shutdown();
    return report.outcome == wiiu::hle::RunOutcome::Returned || closed_by_user ? 0 : 1;
}
