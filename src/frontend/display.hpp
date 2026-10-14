// Phase 7.6 / 7.7 / 7.8 — a window with a GPU behind it.
//
// The pieces that have to be put together, in order, for anything to reach
// the screen: a backend that has passed its capability probe, a window, the
// window's surface attached to the backend (again whenever it is resized),
// and a presenter for the Wii U's two screens. And then taken apart in the
// reverse order, because each holds something of the one before it.
//
// Also where a GPU capture of one particular frame is taken (7.8), since a
// frame is the unit a capture is useful in and this is where frames begin
// and end.
#pragma once

#include <wiiu/gpu/backend.hpp>
#include <wiiu/gpu/presenter.hpp>

#include "platform/window.hpp"

#include <functional>
#include <memory>
#include <string>

namespace wiiu::frontend {

struct DisplayOptions {
    std::string title = "wiiu-emu";
    bool vsync = true;

    /// Capture this frame (counting from 1) to `capture_path`, or none if 0.
    u64 capture_frame = 0;
    std::string capture_path = "wiiu-emu.gputrace";
};

class Display {
public:
    Display() = default;

    ~Display() { close(); }

    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    /// Bring everything up, or say which piece would not come.
    [[nodiscard]] bool open(const DisplayOptions& options, std::string& error);

    void close();

    /// One frame, drawn by `draw` into the drawable it is given. False once
    /// the window has been closed -- the caller's cue to stop. A frame with
    /// no drawable to be had (occluded, resizing) is skipped and still
    /// counts as true.
    using DrawFunction = std::function<void(gpu::GpuBackend&, const gpu::FrameInfo&)>;
    [[nodiscard]] bool frame(const DrawFunction& draw);

    /// One frame of whatever the presenter holds.
    [[nodiscard]] bool present();

    /// Handle window events without drawing. False once it has been closed.
    [[nodiscard]] bool poll();

    [[nodiscard]] gpu::GpuBackend& backend() { return *backend_; }

    [[nodiscard]] gpu::Presenter& presenter() { return presenter_; }

    [[nodiscard]] u64 presented() const noexcept { return presented_; }

private:
    DisplayOptions options_;

    // Declared so that destruction, should `close` not have run, takes them
    // apart in the only safe order: the presenter's resources, then the
    // backend that owns them and draws into the window's layer, then the
    // window that owns the layer.
    std::unique_ptr<platform::Window> window_;
    std::unique_ptr<gpu::GpuBackend> backend_;
    gpu::Presenter presenter_;

    gpu::NativeSurface attached_;
    u64 presented_ = 0;
    bool capturing_ = false;
    bool closed_ = false;
};

} // namespace wiiu::frontend
