// Phase 1.4 — the platform layer.
//
// metal-cpp deliberately excludes MetalKit and AppKit, so there is no
// `MTKView` to lean on: the window and its `CAMetalLayer` have to be made by
// hand, in Objective-C++, and handed to the backend as an opaque surface.
// That is the whole job of this header's one implementation file.
//
// Everything above it is written against `Window`, which knows nothing about
// NSWindow or Metal — so the rest of the frontend compiles on any host, and a
// headless build (the compatibility runner of 10.5, or CI) simply has no
// window rather than a different frontend.
#pragma once

#include <wiiu/gpu/backend.hpp>

#include <memory>
#include <string>
#include <string_view>

namespace wiiu::platform {

struct WindowDesc {
    std::string title = "wiiu-emu";

    /// In points, not pixels. The platform layer multiplies by the display's
    /// backing scale to get the drawable size, which on a Retina display is
    /// twice this and is what actually gets rendered.
    u32 width = 1280;
    u32 height = 720;
};

class Window {
public:
    Window() = default;
    virtual ~Window() = default;

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    /// The layer the GPU backend draws into. Its `width`/`height` are the
    /// drawable size in pixels, and they change when the window is resized or
    /// moved between displays of different scales.
    [[nodiscard]] virtual gpu::NativeSurface surface() = 0;

    /// Handle whatever the window system has to say. Returns false once the
    /// user has asked to close, which is the frontend's cue to stop.
    [[nodiscard]] virtual bool poll() = 0;

    virtual void set_title(std::string_view title) = 0;
};

/// Open a window, or explain why not.
///
/// Returns null on a host with no window system of ours — which is not an
/// error worth stopping for, only a reason not to render.
[[nodiscard]] std::unique_ptr<Window> create_window(const WindowDesc& desc, std::string& error);

} // namespace wiiu::platform
