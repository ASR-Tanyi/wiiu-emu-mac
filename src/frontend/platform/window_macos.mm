// Phase 1.4 — NSWindow and CAMetalLayer, by hand.
//
// The only Objective-C++ in the project, and it stays that way: everything it
// knows about leaves through `gpu::NativeSurface`, which is a `void*` and a
// size. The backend casts it back to a `CA::MetalLayer*`; nothing in between
// learns what it is.
//
// Automatic reference counting is **not** on (nothing in the build turns it
// on), so the Objective-C rules apply by hand: anything from `alloc` is owned
// and released once, anything else is autoreleased and must be retained to
// outlive the pool. Both appear below, which is why each is called out where
// it happens.

#include "platform/window.hpp"

#include <wiiu/common/log.hpp>

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#include <string>

namespace wiiu::platform {

namespace {

/// Bring the process up as a GUI application, once.
///
/// A command-line binary is not one by default: without this the window is
/// created but never comes forward, never takes key focus, and looks for all
/// the world like a renderer that drew nothing.
void start_application() {
    static bool started = false;
    if (started) {
        return;
    }
    started = true;

    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp finishLaunching];
    [NSApp activateIgnoringOtherApps:YES];
}

class MacWindow final : public Window {
public:
    ~MacWindow() override {
        if (window_ != nil) {
            [window_ close];
            [window_ release];
        }
        if (layer_ != nil) {
            [layer_ release];
        }
    }

    [[nodiscard]] bool open(const WindowDesc& desc, std::string& error) {
        start_application();

        const NSRect frame = NSMakeRect(0, 0, desc.width, desc.height);
        const NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

        // From `alloc`, so this object is ours and is released in the
        // destructor. `releasedWhenClosed` is turned off because the default
        // is for the close button to release the window out from under us,
        // which is a use-after-free the moment anything polls afterwards.
        window_ = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:style
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
        if (window_ == nil) {
            error = "could not create an NSWindow";
            return false;
        }
        [window_ setReleasedWhenClosed:NO];
        [window_ setTitle:[NSString stringWithUTF8String:desc.title.c_str()]];
        [window_ center];

        // Autoreleased, so it needs a retain to survive this scope.
        layer_ = [[CAMetalLayer layer] retain];
        if (layer_ == nil) {
            error = "could not create a CAMetalLayer";
            return false;
        }

        // The device and pixel format are the backend's to set: this layer is
        // handed over as a surface and the GPU side decides what to put in
        // it. `framebufferOnly` stays at its default of YES -- a drawable
        // nothing reads back is cheaper, and reading one back is what the
        // screenshot path (10.5) will have to ask for explicitly.
        NSView* const view = [window_ contentView];
        [view setWantsLayer:YES];
        [view setLayer:layer_];

        update_drawable_size();

        [window_ makeKeyAndOrderFront:nil];
        WIIU_LOG_INFO(Frontend, "window opened at {}x{} drawable pixels", width_, height_);
        return true;
    }

    [[nodiscard]] gpu::NativeSurface surface() override {
        return gpu::NativeSurface{layer_, width_, height_};
    }

    [[nodiscard]] bool poll() override {
        @autoreleasepool {
            // Drain everything pending and return; blocking here would hand
            // the frame loop to AppKit, which is the opposite of what an
            // emulator wants.
            for (;;) {
                NSEvent* const event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                          untilDate:[NSDate distantPast]
                                                             inMode:NSDefaultRunLoopMode
                                                            dequeue:YES];
                if (event == nil) {
                    break;
                }
                [NSApp sendEvent:event];
            }
        }

        if (![window_ isVisible]) {
            return false;
        }

        update_drawable_size();
        return true;
    }

    void set_title(std::string_view title) override {
        const std::string copy{title};
        [window_ setTitle:[NSString stringWithUTF8String:copy.c_str()]];
    }

private:
    /// Keep the layer's pixel count matched to the window.
    ///
    /// Points are not pixels: a Retina display renders two device pixels per
    /// point, and a window dragged between displays changes scale without
    /// changing size. Getting this wrong is not a crash, it is a blurry
    /// image or a view that renders into a quarter of itself.
    void update_drawable_size() {
        NSView* const view = [window_ contentView];
        const CGFloat scale = [window_ backingScaleFactor];
        const NSRect bounds = [view bounds];

        const auto width = static_cast<u32>(bounds.size.width * scale);
        const auto height = static_cast<u32>(bounds.size.height * scale);
        if (width == width_ && height == height_) {
            return;
        }

        width_ = width;
        height_ = height;
        [layer_ setContentsScale:scale];
        [layer_ setDrawableSize:CGSizeMake(width, height)];
        WIIU_LOG_DEBUG(Frontend, "drawable resized to {}x{} (scale {})", width_, height_, scale);
    }

    NSWindow* window_ = nil;
    CAMetalLayer* layer_ = nil;
    u32 width_ = 0;
    u32 height_ = 0;
};

} // namespace

std::unique_ptr<Window> create_window(const WindowDesc& desc, std::string& error) {
    auto window = std::make_unique<MacWindow>();
    if (!window->open(desc, error)) {
        return nullptr;
    }
    return window;
}

} // namespace wiiu::platform
