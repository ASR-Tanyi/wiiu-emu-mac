// Phase 1.4 — what a window means on a host that has none of ours.
//
// The platform-independent core builds on Linux so the emulator below the
// renderer can be worked on without a Mac. The frontend is part of that, so
// asking for a window has to be a question with an answer.

#include "platform/window.hpp"

namespace wiiu::platform {

std::unique_ptr<Window> create_window(const WindowDesc& desc, std::string& error) {
    static_cast<void>(desc);
    error = "this build has no window system: the emulator's display needs macOS";
    return nullptr;
}

} // namespace wiiu::platform
