// Phase 1.8 — the macOS half of the JIT probe.

#include <wiiu/common/types.hpp>

#include "platform/jit_probe.hpp"

#include <fmt/format.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>

namespace wiiu::platform {

namespace {

/// `mov w0, #42` then `ret`, hand-assembled.
///
/// Two words rather than a compiled stub on purpose: the point is to prove
/// that bytes this process wrote at runtime can be executed, and anything the
/// compiler produced would already be in an executable mapping and would
/// prove nothing.
constexpr u32 kMovW0_42 = 0x52800540;
constexpr u32 kRet = 0xD65F03C0;
constexpr int kExpected = 42;

using ProbeFunction = int (*)();

} // namespace

JitProbeResult probe_jit() {
    JitProbeResult result;

    const std::size_t size = 4096;
    void* const page = ::mmap(nullptr,
                              size,
                              PROT_READ | PROT_WRITE | PROT_EXEC,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT,
                              -1,
                              0);
    if (page == MAP_FAILED) {
        // Almost always the entitlement: MAP_JIT without
        // com.apple.security.cs.allow-jit (and a signature carrying it) fails
        // here and nowhere else.
        result.detail = fmt::format(
            "mmap(MAP_JIT) failed: {}. The binary needs to be signed with the Hardened "
            "Runtime and the com.apple.security.cs.allow-jit entitlement -- the build does "
            "this automatically, so a failure here means the signature did not take.",
            std::strerror(errno));
        return result;
    }
    result.mapped = true;

    // Write protection is per *thread*, not per page: this thread may write
    // while another executes. Phase 12's background compile thread depends on
    // exactly that, which is why the probe uses the real call rather than
    // mprotect.
    ::pthread_jit_write_protect_np(0);
    const std::array<u32, 2> code{kMovW0_42, kRet};
    std::memcpy(page, code.data(), code.size() * sizeof(u32));
    ::pthread_jit_write_protect_np(1);
    result.wrote = true;

    // arm64 has no coherent instruction cache. Without this the CPU may
    // execute whatever was in the I-cache at this address before, which is
    // usually nothing and occasionally something -- the single nastiest bug
    // in a hand-rolled JIT, because it reproduces about one run in fifty.
    ::sys_icache_invalidate(page, size);

    ProbeFunction function = nullptr;
    std::memcpy(&function, &page, sizeof(function));
    const int returned = function();
    result.executed = true;
    result.correct = returned == kExpected;

    ::munmap(page, size);

    result.detail =
        result.correct
            ? "MAP_JIT mapped, written, made executable and called successfully"
            : fmt::format("the generated function returned {} instead of {}", returned, kExpected);
    return result;
}

} // namespace wiiu::platform
