// Phase 7.6 / 7.7 — see presenter.hpp.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/presenter.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace wiiu::gpu {

namespace {

// One triangle that covers the viewport: the corners past it are clipped
// away, and unlike two triangles there is no diagonal seam for a filter to
// show. Nothing is bound for the vertices -- they come from `vertex_id`.
constexpr std::string_view kShader = R"msl(
#include <metal_stdlib>
using namespace metal;

struct ScreenVertex {
    float4 position [[position]];
    float2 uv;
};

vertex ScreenVertex vertex_main(uint id [[vertex_id]]) {
    const float2 corner = float2(float((id << 1) & 2u), float(id & 2u));
    ScreenVertex out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = float2(corner.x, 1.0 - corner.y);
    return out;
}

fragment float4 fragment_main(ScreenVertex in [[stage_in]],
                              texture2d<float> screen [[texture(0)]],
                              sampler screen_sampler [[sampler(0)]]) {
    // The fourth byte of an OSScreen pixel is ignored by the display, and
    // titles leave anything in it -- WHB's console clears with 0x...FF and
    // others with 0x...00. Opaque, whatever it says.
    return float4(screen.sample(screen_sampler, in.uv).rgb, 1.0);
}
)msl";

[[nodiscard]] std::string_view screen_label(std::size_t index) noexcept {
    return index == 0 ? "tv" : "drc";
}

/// Fit `size` into `area`, aspect kept, centred.
[[nodiscard]] Rect fit(const Rect& area, const Rect& size) noexcept {
    if (size.width == 0 || size.height == 0 || area.width == 0 || area.height == 0) {
        return {};
    }
    // Compare cross-multiplied rather than as ratios: exact in integers, and
    // no rounding decides which axis is the tight one.
    const u64 wide = static_cast<u64>(area.width) * size.height;
    const u64 tall = static_cast<u64>(area.height) * size.width;
    Rect result;
    if (wide >= tall) {
        // The area is wider than the image: full height, bars at the sides.
        result.height = area.height;
        result.width = static_cast<u32>(tall / size.height);
    } else {
        result.width = area.width;
        result.height = static_cast<u32>(wide / size.width);
    }
    result.x = area.x + ((area.width - result.width) / 2);
    result.y = area.y + ((area.height - result.height) / 2);
    return result;
}

} // namespace

PresentLayout layout_screens(u32 width,
                             u32 height,
                             const std::array<Rect, present_screen_count>& sizes,
                             const std::array<bool, present_screen_count>& shown) {
    PresentLayout layout;
    const bool both = shown[0] && shown[1];

    for (std::size_t index = 0; index < present_screen_count; ++index) {
        if (!shown[index]) {
            continue;
        }
        Rect area{0, 0, width, height};
        if (both) {
            // TV above, GamePad below: the order they sit in in a living room.
            const u32 half = height / 2;
            area = index == 0 ? Rect{0, 0, width, half} : Rect{0, half, width, height - half};
        }
        const Rect placed = fit(area, sizes[index]);
        if (placed.width != 0 && placed.height != 0) {
            layout.rects[index] = placed;
            layout.visible[index] = true;
        }
    }
    return layout;
}

bool Presenter::initialize(GpuBackend& backend, TextureFormat target, std::string& error) {
    shutdown();
    backend_ = &backend;
    target_format_ = target;

    PipelineDesc pipeline;
    pipeline.label = "present";
    pipeline.vertex_source = std::string{kShader};
    pipeline.color_formats[0] = target;
    pipeline.color_count = 1;
    pipeline_ = backend.create_pipeline(pipeline, error);
    if (!pipeline_.valid()) {
        error = "the present pipeline did not build: " + error;
        backend_ = nullptr;
        return false;
    }

    // Linear, because the window is almost never an integer multiple of 720p
    // and nearest-neighbour at 1.37x makes text shimmer. Clamped, so the
    // filter never reaches round to the opposite edge.
    SamplerDesc sampler;
    sampler.label = "present";
    sampler.min_filter = Filter::Linear;
    sampler.mag_filter = Filter::Linear;
    sampler.mip_filter = Filter::Nearest;
    sampler.address_u = AddressMode::ClampToEdge;
    sampler.address_v = AddressMode::ClampToEdge;
    sampler.address_w = AddressMode::ClampToEdge;
    sampler_ = backend.create_sampler(sampler);
    if (!sampler_.valid()) {
        error = "the present sampler could not be created";
        shutdown();
        return false;
    }
    return true;
}

void Presenter::shutdown() {
    if (backend_ == nullptr) {
        return;
    }
    for (Screen& screen : screens_) {
        for (TextureHandle& slot : screen.slots) {
            if (slot.valid()) {
                backend_->destroy(slot);
                slot = {};
            }
        }
        screen = Screen{};
    }
    if (pipeline_.valid()) {
        backend_->destroy(pipeline_);
        pipeline_ = {};
    }
    if (sampler_.valid()) {
        backend_->destroy(sampler_);
        sampler_ = {};
    }
    backend_ = nullptr;
}

void Presenter::update(PresentScreen which, const ScreenImage& image) {
    Screen& screen = screens_[static_cast<std::size_t>(which)];
    const std::size_t row = static_cast<std::size_t>(image.width) * 4;
    if (image.width == 0 || image.height == 0 || image.bytes_per_row < row ||
        image.pixels.size() <
            (static_cast<std::size_t>(image.bytes_per_row) * (image.height - 1)) + row) {
        WIIU_LOG_WARN(Gpu,
                      "present: a {}x{} image with {} bytes to a row does not fit in {} bytes",
                      image.width,
                      image.height,
                      image.bytes_per_row,
                      image.pixels.size());
        return;
    }

    // Packed tightly on the way in: the pitch is the source's business, and
    // a copy that keeps it would upload the padding every frame.
    screen.pixels.resize(row * image.height);
    for (u32 y = 0; y < image.height; ++y) {
        const auto source = image.pixels.begin() +
                            static_cast<std::ptrdiff_t>(y * std::size_t{image.bytes_per_row});
        std::copy(source,
                  source + static_cast<std::ptrdiff_t>(row),
                  screen.pixels.begin() + static_cast<std::ptrdiff_t>(y * row));
    }
    screen.width = image.width;
    screen.height = image.height;
    screen.dirty = true;
}

void Presenter::set_shown(PresentScreen which, bool shown) {
    screens_[static_cast<std::size_t>(which)].shown = shown;
}

bool Presenter::has_image(PresentScreen which) const noexcept {
    return !screens_[static_cast<std::size_t>(which)].pixels.empty();
}

bool Presenter::upload(Screen& screen, std::string_view name) {
    // The slot after the one the last draw used. No frame still in flight is
    // reading it: the backend holds at most `frames_in_flight` of them, this
    // one included, and they sampled the current slot or the one before.
    const u32 slot = screen.uploaded ? (screen.current + 1) % frames_in_flight : 0;
    TextureHandle& texture = screen.slots[slot];
    const Rect size{0, 0, screen.width, screen.height};

    if (texture.valid() && !(screen.slot_sizes[slot] == size)) {
        backend_->destroy(texture);
        texture = {};
    }
    if (!texture.valid()) {
        TextureDesc desc;
        desc.width = screen.width;
        desc.height = screen.height;
        desc.format = TextureFormat::RGBA8Unorm;
        desc.usage = TextureUsage::Sampled;
        desc.storage = Storage::Shared; // written by the CPU every flip
        const std::string label = fmt::format("{} screen {}", name, slot);
        desc.label = label;
        texture = backend_->create_texture(desc);
        if (!texture.valid()) {
            return false;
        }
        screen.slot_sizes[slot] = size;
    }

    if (!backend_->write_texture(texture, size, screen.pixels, screen.width * 4)) {
        return false;
    }
    screen.current = slot;
    screen.uploaded = true;
    screen.dirty = false;
    return true;
}

bool Presenter::draw(TextureHandle target, u32 width, u32 height) {
    if (backend_ == nullptr) {
        return false;
    }

    std::array<Rect, present_screen_count> sizes{};
    std::array<bool, present_screen_count> shown{};
    for (std::size_t index = 0; index < present_screen_count; ++index) {
        Screen& screen = screens_[index];
        if (screen.dirty && !upload(screen, screen_label(index))) {
            WIIU_LOG_WARN(Gpu, "present: the {} image could not be uploaded", screen_label(index));
        }
        sizes[index] = Rect{0, 0, screen.width, screen.height};
        shown[index] = screen.shown && screen.uploaded;
    }
    layout_ = layout_screens(width, height, sizes, shown);

    RenderPassDesc pass;
    pass.label = "present";
    pass.color_count = 1;
    pass.colors[0].texture = target;
    // Cleared, not loaded: the bars are black and everything else is drawn
    // over, so last frame's contents are never needed (P20).
    pass.colors[0].load = LoadAction::Clear;
    pass.colors[0].store = StoreAction::Store;
    pass.colors[0].clear = ClearColor{0.0F, 0.0F, 0.0F, 1.0F};
    if (!backend_->begin_render_pass(pass)) {
        return false;
    }

    for (std::size_t index = 0; index < present_screen_count; ++index) {
        if (!layout_.visible[index]) {
            continue;
        }
        const Rect& rect = layout_.rects[index];
        backend_->set_pipeline(pipeline_);
        backend_->set_viewport(Viewport{static_cast<f32>(rect.x),
                                        static_cast<f32>(rect.y),
                                        static_cast<f32>(rect.width),
                                        static_cast<f32>(rect.height),
                                        0.0F,
                                        1.0F});
        backend_->set_scissor(rect);
        backend_->bind_texture(
            ShaderStage::Fragment, 0, screens_[index].slots[screens_[index].current]);
        backend_->bind_sampler(ShaderStage::Fragment, 0, sampler_);

        DrawCall call;
        call.vertex_count = 3;
        backend_->draw(call);
    }

    backend_->end_render_pass();
    return true;
}

} // namespace wiiu::gpu
