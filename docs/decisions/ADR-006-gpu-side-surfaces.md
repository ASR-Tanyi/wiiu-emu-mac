# ADR-006 — What the GPU draws stays on the GPU

**Status:** Accepted · 2026-10-21

## Context

Until the translator (8.11), every piece of GPU work the emulator did ran on
the CPU, against guest memory: `GX2ClearColor` filled the colour buffer's
bytes, `GX2CopyColorBufferToScanBuffer` detiled them into the scan buffer,
and a flip handed the scan buffer's bytes to the presenter, which uploaded
them. Guest memory was the one copy of every surface, which made the model
simple and the tests easy: run a stream, look at memory.

Draws break that. A draw runs on Metal, and what it renders lands in a Metal
texture, asynchronously, some time after the packet retired (ADR-004). The
next packet in a typical frame is the scan-out copy, which until now read the
colour buffer from guest memory -- where the draw's pixels are not.

There are two ways to close that gap:

- **Write back.** After a draw, or at the points that need it (a scan-out
  copy, a flip), wait for the GPU, read the render target back, re-tile it
  and store it in guest memory. Everything downstream keeps working as it
  does. But it waits for the GPU every frame -- the stall **P4** says is a
  bug -- and re-tiles a whole frame on the CPU every frame, when the CPU
  detiler alone takes tens of milliseconds for a 720p frame in a debug build
  (8.5); it would be replaced the first time a real title ran.
- **Keep it on the GPU.** A surface the GPU renders to lives as a Metal
  texture, and the operations that read it run on the GPU too.

Cemu and decaf both do the second: a texture cache keyed by what defines a
surface, render targets and the scan-out copy on the GPU, guest memory
updated only when the CPU needs the pixels. The plan already assumes it:
**P5** (texture caching with range invalidation), **P22** (zero-copy guest
memory), and 8.13.

## Decision

**A colour buffer the GPU renders to lives in a surface cache as a Metal
texture, and while it does, the texture -- not guest memory -- is that
surface.** The cache is keyed by what the colour-buffer registers say
identifies a surface: CB_COLORn_BASE (the address, with the swizzle),
CB_COLORn_SIZE (pitch and slice in tiles), and from CB_COLORn_INFO the
format, number type, tile mode and component swap, with CB_COLORn_VIEW's
first slice. GX2's own `GX2ColorBuffer` carries the same register values
(`GX2InitColorBufferRegs`, 8.1), so a clear or a scan-out copy from GX2 and a
draw from the register file find the same entry. The texture is the padded
size the registers describe; each consumer uses its own rectangle of it.

**Operations follow the surface.** For a surface in the cache, GX2's clear
becomes a render pass that clears it, the scan-out copy becomes a GPU copy
into the output's scan texture, and the presenter shows that texture. The
CPU implementations stay, for surfaces that are not in the cache -- a title
that writes pixels with the CPU and copies them out -- and as the reference
the GPU paths are tested against on the null backend and on the Mac.

**Coherency, in both directions, is explicit.**

- *CPU to GPU*: guest memory written under a cached surface (a title's
  `GX2Invalidate` naming it, or the cache model's flush tracking, 3.6) makes
  the cached copy stale; it is re-uploaded or dropped. Not needed for the
  first frames, built with 8.13.
- *GPU to CPU*: guest memory under a GPU-resident surface is stale. It is
  brought up to date only when the guest asks for the pixels -- the
  readback ADR-004 already allows for, which waits for the host GPU at the
  retirement of the submission that asked, so it stays deterministic. Not
  built yet; until it is, a title that reads its render target with the CPU
  sees old bytes, and the place that would have needed a readback is logged
  once.

**Hit-rate counters from the first version** (**P5**): a cache that misses
every frame looks like a cache and performs like none.

## Consequences

**Good.**

- Nothing waits for the GPU in a normal frame; drawn pixels never cross to
  the CPU. This is the shape the performance plan needs anyway, rather than
  a version to replace.
- It matches Cemu and decaf, so their findings about surfaces -- aliasing,
  format reinterpretation, which titles read back -- carry over.
- Determinism holds: nothing guest-visible depends on the host GPU except a
  readback, and a readback is pinned to retirement (ADR-004).

**Bad, and worth saying.**

- **Two implementations of clear and copy**, CPU and GPU, that must agree.
  The CPU one is the oracle; tests hold the GPU one to it.
- **Guest memory is no longer the whole truth.** A test that ran a stream and
  looked at memory now has to look at the cache for rendered surfaces, and a
  debugger memory view of a render target shows stale bytes until readback
  exists.
- **Surfaces alias.** A title can render to a buffer and then read the same
  memory as a texture of another format, or render into part of a larger
  surface. The key above treats those as different surfaces; resolving them
  is 8.13's work and will grow the cache's rules, as it did Cemu's.
- The frontend's presenter, today fed bytes, gains a texture path.

## What would make us revisit

- A title that reads its render targets with the CPU every frame, measured --
  at which point that title wants readback scheduled early, or a write-back
  mode behind a per-title setting (14.3).
- Aliasing rules so tangled that a memory-is-truth model with GPU-side
  caching on top turns out simpler than this.
