# ADR-002 — Apple silicon only; no Intel Mac support

**Status:** Accepted · 2026-09-15

## Context

Following ADR-001, geometry shaders are emulated with Metal mesh shaders. Mesh
shaders require GPU family **Apple7** (A14/M1 and later). Apple's published
feature-set tables no longer contain a Mac2 (Intel/AMD) table at all, and no
Intel Mac exposes `apple7`.

Intel Macs *do* report Metal 3 support, which makes a naive
`supportsFamily(GPUFamilyMetal3)` check insufficient — Cemu discovered this and
carries an explicit Intel vendor blacklist with the comment "Intel GPUs have
issues with mesh shaders".

Supporting Intel would mean writing a second geometry path as a compute
pre-pass, on a platform that also lacks programmable blending (so no shader
logic ops) and does not guarantee argument buffers Tier 2. macOS 26 Tahoe is
the last release supporting Intel Macs; macOS 27 drops them entirely.

## Decision

Target `arm64` only. `CMAKE_OSX_ARCHITECTURES` is pinned to `arm64` and the
build fails loudly on any other value. Deployment target is macOS 13.0, the
Metal 3 baseline.

At runtime, probe capabilities at startup and refuse to start with a clear
message when mesh shaders are unavailable, rather than silently dropping draws.
(Cemu, lacking a fallback, skips the draw — which presents to the user as
missing geometry with no explanation.)

## Consequences

**Good.** One rendering path instead of two. A meaningful amount of work
avoided on a platform with roughly one OS release of life left. Permission to
use Apple-silicon-only features freely: programmable blending for shader logic
ops, `-mcpu=apple-m1`, unified-memory zero-copy, Tier 2 argument buffers.

**Bad.** Intel Mac owners cannot run this at all. Given the alternative is
running it without geometry, that seems the more honest outcome.

## Revisit if

A compute-based geometry path turns out to be needed anyway — for instance if
it proves faster than mesh shaders for some primitive types — since that would
make Intel support close to free.
