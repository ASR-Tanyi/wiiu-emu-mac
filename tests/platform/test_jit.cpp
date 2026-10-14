// Phase 1.8 — the JIT path, as a test rather than a thing to remember.
//
// Phase 12 needs four things lined up at once, three of which are build
// configuration: `MAP_JIT`, the `com.apple.security.cs.allow-jit`
// entitlement, a signature with the Hardened Runtime, and
// `pthread_jit_write_protect_np` around the writes. The probe checks all four
// by using them, and running it here means the check happens on every build
// rather than when somebody thinks to run `--jit-check`.
//
// **This test binary is signed with the same entitlements as the app** (see
// tests/CMakeLists.txt). Without that it would fail for a reason that has
// nothing to do with the code — which is itself worth knowing, because it is
// exactly how this breaks for a user: the entitlement is on the app and not
// on whatever else tried to compile code.

#include "platform/jit_probe.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace wiiu;

TEST_CASE("code written at runtime can be executed", "[platform][jit]") {
    const platform::JitProbeResult result = platform::probe_jit();

    INFO(result.detail);

    // Reported separately rather than as one boolean, because each failure
    // means something different: no mapping is the entitlement or the
    // signature, no execution is the W^X toggle, and a wrong answer is the
    // instruction cache.
    CHECK(result.mapped);
    CHECK(result.wrote);
    CHECK(result.executed);
    CHECK(result.correct);
    CHECK(result.ok());
}
