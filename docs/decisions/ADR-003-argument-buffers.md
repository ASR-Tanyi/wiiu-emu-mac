# ADR-003 — Bindless resource binding via argument buffers (Tier 2)

**Status:** Accepted · 2026-10-14

## Context

Latte binds resources through PM4: `SET_RESOURCE` writes texture descriptors,
`SET_SAMPLER` writes sampler state, `SET_ALU_CONST` writes uniform data. Each
shader stage has its own bank of slots, and a draw can want all of them at
once in all three stages.

Metal's *direct* binding limits are generous for buffers and textures and
tight for samplers:

| Resource | Direct limit |
|---|---|
| Buffers | 31 per stage |
| Textures | 128 per stage (Apple6+) |
| **Samplers** | **16 per stage** |

Sixteen is the number that matters, and it is not sixteen in practice. The
emulator needs slots of its own for the resources no guest draw knows about:
the vertex-fetch buffer the shader reads instead of a vertex descriptor (the
Phase 7 alignment note), the stream-out counter buffers (8.9), the helper
state the object+mesh geometry path needs in every stage (8.8), and the
in-shader emulation of border colours, logic ops and non-seamless cube
sampling catalogued in the Phase 7 table. Cemu solves this by reserving
buffers 28–30, textures 29–30 and samplers 14–15 — which works, and leaves the
guest fourteen sampler slots out of sixteen.

**Argument buffers Tier 2** removes the limit: resources are addressed through
a buffer of descriptors rather than through binding slots, and the count is
bounded by memory rather than by the API. Every Apple silicon GPU supports
Tier 2, and ADR-002 already restricts this project to Apple silicon, so there
is no tier to fall back to and no second path to keep alive.

## Decision

**Bind through argument buffers, Tier 2, for every stage.** One argument
buffer per stage per draw, suballocated from a per-frame ring; direct binding
is used only for the argument buffer itself and for the handful of things that
genuinely are single slots (the vertex stream pointers the mesh path needs
before an argument buffer exists).

The `GPUBackend` interface (7.1) does not expose this. It takes a binding
*index* per stage and the backend decides how that index is reached, so the
PM4 translator in Phase 8 is written against slots and never learns which
mechanism is underneath.

## Consequences

**Good.**

- No slot budget, so no reservation scheme threaded through the shader
  decompiler, and no title that fails because it wanted a fifteenth sampler.
- One encode path. A tier check would otherwise reach pipeline creation,
  resource binding, the MSL emitter and the pipeline cache key.
- Rebinding is a buffer write rather than a sequence of encoder calls, which
  is the cheaper shape when a title changes one texture between draws.
- It composes with `MTLHeap` suballocation (7.4, **P22**): resources that
  already live in a heap for zero-copy reasons are exactly the resources an
  argument buffer wants to point at.

**Bad, and worth saying out loud.**

- **Residency becomes ours.** The GPU does not know what an argument buffer
  points at, so every resource an encoder might touch needs `useResource` or
  `useHeap`. Forget one and the result is a GPU fault or garbage pixels, not a
  validation error — the worst failure shape there is. The mitigation is
  structural: guest-visible memory is allocated from a small number of heaps
  and each encoder calls `useHeap` once per heap, so correctness does not
  depend on tracking individual resources. Debug builds run with Metal API
  validation and shader validation on (7.8), which do catch most of the rest.
- **Frame captures read worse.** Xcode shows direct bindings as named slots
  and argument buffers as a blob. The GPU debugger (8.16) has to print the
  decoded binding table itself; with direct binding it could have leaned on
  Apple's tooling.
- **Reversing this is not free.** The interface hides the mechanism from the
  PM4 translator, but the MSL emitter (8.6) bakes the convention into
  generated source, and the pipeline cache (8.10) is keyed on shaders built
  that way. Changing it later invalidates every cached pipeline.

## Alternatives considered

**Direct binding with reserved slots, as Cemu does.** Proven on this exact
hardware with this exact workload, which is the strongest argument any
alternative here has. Rejected because it spends a permanent design budget —
every future helper resource must find a slot, and every slot taken is one a
title cannot have — to work around a limit that Apple silicon does not impose
on us. Cemu supports Intel through MoltenVK and cannot assume Tier 2; we can.

**A hybrid: direct binding when a draw fits in the slots, argument buffers
when it does not.** Rejected. It is two binding paths, which is two sets of
bugs, and the pipeline cache key grows a mode bit that doubles the cache for
no user-visible gain. The fast path would also be the one that is rarely
exercised on the titles that matter, which is the wrong way round.

## What would make us revisit

- Profiling on real titles showing argument-buffer indirection costing more
  than the reservation scheme would have. This is the one that would actually
  change the answer; it has not been measured, because there is nothing to
  measure yet.
- A residency bug class that the heap-level `useHeap` discipline does not
  prevent, found repeatedly rather than once.
- A future port to a platform without an equivalent — but that is a
  `GPUBackend` implementation question, and the interface was shaped (7.1) so
  that it stays one.
