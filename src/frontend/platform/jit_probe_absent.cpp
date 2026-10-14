// Phase 1.8 — the JIT probe on a host with no MAP_JIT.

#include "platform/jit_probe.hpp"

namespace wiiu::platform {

JitProbeResult probe_jit() {
    JitProbeResult result;
    // Saying "not tested here" rather than "passed" matters: the whole value
    // of this probe is that it fails loudly on the machine that will run the
    // JIT, and a green result from a host that never tried would be worse
    // than no result at all.
    result.detail = "this host has no MAP_JIT; the JIT path is a macOS concern (Phase 12)";
    return result;
}

} // namespace wiiu::platform
