// Phase 7.6 / 7.7 / 7.8 — see display.hpp.

#include "display.hpp"

#include <wiiu/common/log.hpp>

namespace wiiu::frontend {

bool Display::open(const DisplayOptions& options, std::string& error) {
    close();
    options_ = options;
    closed_ = false;
    presented_ = 0;

    backend_ = gpu::create_metal_backend();
    if (backend_ == nullptr) {
        error = "this build has no GPU backend: the emulator needs macOS on Apple silicon";
        return false;
    }
    if (!backend_->initialize(error)) {
        // Capability shortfalls land here: ADR-002 refuses with a reason
        // rather than rendering something wrong.
        backend_.reset();
        return false;
    }

    platform::WindowDesc desc;
    desc.title = options.title;
    window_ = platform::create_window(desc, error);
    if (window_ == nullptr) {
        close();
        return false;
    }

    attached_ = window_->surface();
    if (!backend_->attach_surface(attached_, error)) {
        close();
        return false;
    }
    backend_->set_vsync(options.vsync);

    // The drawable is BGRA8; the presenter's pipeline is built for it.
    if (!presenter_.initialize(*backend_, gpu::TextureFormat::BGRA8Unorm, error)) {
        close();
        return false;
    }
    return true;
}

void Display::close() {
    if (capturing_ && backend_ != nullptr) {
        backend_->end_capture();
        capturing_ = false;
    }
    presenter_.shutdown();
    if (backend_ != nullptr) {
        backend_->shutdown();
        backend_.reset();
    }
    window_.reset();
}

bool Display::poll() {
    if (closed_ || window_ == nullptr) {
        return false;
    }
    if (!window_->poll()) {
        closed_ = true;
        return false;
    }

    // Resized, or dragged to a display with a different scale: the drawable
    // has a new size and the backend has to be told before the next frame.
    if (const gpu::NativeSurface surface = window_->surface();
        surface.width != attached_.width || surface.height != attached_.height) {
        std::string error;
        if (!backend_->attach_surface(surface, error)) {
            WIIU_LOG_ERROR(Gpu, "{}", error);
            closed_ = true;
            return false;
        }
        attached_ = surface;
    }
    return true;
}

bool Display::frame(const DrawFunction& draw) {
    if (!poll()) {
        return false;
    }

    // 7.8: the frame asked for, bracketed exactly. The capture has to start
    // *before* `begin_frame`: Metal captures only the command buffers made
    // after it starts, and `begin_frame` is what makes this frame's.
    const u64 number = presented_ + 1;
    if (options_.capture_frame != 0 && number == options_.capture_frame && !capturing_) {
        std::string error;
        capturing_ = backend_->begin_capture(options_.capture_path, error);
        if (!capturing_) {
            WIIU_LOG_ERROR(Gpu, "frame capture: {}", error);
            options_.capture_frame = 0; // said once, not every frame
        }
    }

    gpu::FrameInfo info;
    if (!backend_->begin_frame(info)) {
        // No drawable this time: occluded, or resizing. Skipping is the
        // right answer; waiting for one would stall the whole emulator on
        // the window server. A capture already started stays open for the
        // frame that does get drawn.
        return true;
    }

    draw(*backend_, info);
    backend_->end_frame();
    ++presented_;

    if (capturing_) {
        backend_->end_capture();
        capturing_ = false;
        WIIU_LOG_INFO(
            Frontend, "frame {} captured to {} -- open it in Xcode", number, options_.capture_path);
    }
    return true;
}

bool Display::present() {
    return frame([this](gpu::GpuBackend&, const gpu::FrameInfo& info) {
        if (!presenter_.draw(info.drawable, info.width, info.height)) {
            WIIU_LOG_WARN(Gpu, "present: the frame could not be drawn");
        }
    });
}

} // namespace wiiu::frontend
