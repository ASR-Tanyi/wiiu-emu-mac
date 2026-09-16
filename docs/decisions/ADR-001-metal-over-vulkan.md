# ADR-001 — Native Metal instead of Vulkan via MoltenVK

**Status:** Accepted · 2026-09-15

## Context

The renderer has to reproduce a 2008-era AMD R700 GPU (Latte). Two backends
were considered: Vulkan through MoltenVK, and native Metal.

Vulkan was the initial choice, on portability grounds — one renderer that could
later serve Linux and Windows. Verification against MoltenVK's current
capabilities killed it:

- **No geometry shaders.** `geometryShader` is false; support has been
  "planned" for years. Latte uses geometry shaders, *and* routes its RECTS
  primitive through the same path, so this is not an exotic-title edge case.
- **No transform feedback.** `VK_EXT_transform_feedback` is unimplemented.
  Latte uses stream-out.
- Also absent: `VK_EXT_custom_border_color`, pipeline-statistics queries.
  Provoking-vertex control and non-seamless cube maps exist only in a
  `MVK_USE_METAL_PRIVATE_API` build.

This is not theoretical. **decaf-emu ships no macOS build**, and its README
cites Vulkan complications; its Vulkan driver depends on exactly those two
missing features. Cemu, which does run on macOS, solved it by writing a native
Metal renderer that emulates geometry shaders with mesh shaders, and reports it
roughly 20–25% faster than its MoltenVK path while running titles MoltenVK
cannot.

## Decision

Write a native Metal backend using `metal-cpp`, with Metal 3 (macOS 13) as the
baseline. Keep it behind an abstract `GPUBackend` interface so the choice is
reversible and a second backend stays possible.

Emit MSL text directly from our own shader IR. Do not route through SPIR-V:
Apple's Metal Shader Converter accepts DXIL rather than SPIR-V, so the only
SPIR-V→MSL path is SPIRV-Cross, whose MSL backend has no mesh shader
support — which would eliminate the geometry-shader path this ADR exists to
enable.

## Consequences

**Good.** Geometry shaders become possible at all. Stream-out works via
`rasterizationEnabled = false` with a `void`-returning vertex function.
Unified memory allows zero-copy aliasing of guest memory (P22). Metal is a
system framework, so there is no dylib to bundle. Native tooling — Xcode frame
capture, the shader debugger, `gpucapture`, the Metal Performance HUD — is
better than the Vulkan equivalent on this platform.

**Bad.** macOS only; a future Linux or Windows port needs a second backend,
which is why the `GPUBackend` abstraction is not optional. Metal is narrower
than Vulkan in the places a 2008 GPU cares about: no provoking-vertex control,
no logic ops, primitive restart cannot be disabled, only three fixed sampler
border colours, always-seamless cube maps, `[0,1]` depth only, no point fill
mode, and a 4-byte vertex attribute alignment requirement that forces manual
in-shader vertex fetch. Each needs shader-side emulation; they are catalogued
in the Phase 7 table in `plan.md`. `metal-cpp` also has no ARC and no MetalKit,
so lifetime management and windowing are ours to write.

**And it forces ADR-002**, because the mesh-shader path requires GPU family
Apple7.

## Revisit if

MoltenVK ships both geometry shaders and transform feedback; or a Linux or
Windows port becomes a real goal, at which point the `GPUBackend` interface
earns its keep and a Vulkan implementation lands beside the Metal one.
