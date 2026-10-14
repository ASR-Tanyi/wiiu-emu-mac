// Phase 1.8 — proving the JIT path before anything depends on it.
//
// Phase 12 writes arm64 code at runtime and jumps to it. On Apple silicon
// that needs four things lined up at once, and three of them are build
// configuration rather than code: `MAP_JIT` on the mapping, the
// `com.apple.security.cs.allow-jit` entitlement, a code signature (ad-hoc is
// enough locally) with the Hardened Runtime enabled, and
// `pthread_jit_write_protect_np` around the writes. Miss the entitlement and
// the `mmap` fails outright; miss `sys_icache_invalidate` and the CPU
// executes whatever was in the instruction cache before — which works most of
// the time, which is the worst possible failure mode.
//
// So this probe exists now, eleven phases early, to answer the question while
// it is cheap: it maps a page, writes a two-instruction function into it,
// flips the page to executable, invalidates the instruction cache, calls it,
// and checks the answer. If the signing pipeline is wrong, this says so in a
// sentence instead of Phase 12 discovering it under a recompiler.
#pragma once

#include <string>

namespace wiiu::platform {

struct JitProbeResult {
    bool mapped = false;   ///< MAP_JIT mapping succeeded
    bool wrote = false;    ///< code written with write protection off
    bool executed = false; ///< the generated function ran
    bool correct = false;  ///< ... and returned what it should have
    std::string detail;    ///< what happened, in a sentence

    [[nodiscard]] bool ok() const noexcept { return mapped && wrote && executed && correct; }
};

/// Run the probe. On a platform without `MAP_JIT` this reports that rather
/// than pretending to have tested something.
[[nodiscard]] JitProbeResult probe_jit();

} // namespace wiiu::platform
