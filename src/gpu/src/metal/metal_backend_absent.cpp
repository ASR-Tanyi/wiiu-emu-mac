// Phase 7.1 — what `create_metal_backend` means on a host without Metal.
//
// The platform-independent core builds and tests on Linux (see the root
// CMakeLists) so that everything below the renderer can be worked on and
// verified without a Mac. That only holds if asking for the Metal backend is
// a question with an answer rather than a link error.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/backend.hpp>

namespace wiiu::gpu {

std::unique_ptr<GpuBackend> create_metal_backend() {
    WIIU_LOG_ERROR(Gpu,
                   "this build has no Metal backend: the emulator needs macOS on Apple silicon");
    return nullptr;
}

} // namespace wiiu::gpu
