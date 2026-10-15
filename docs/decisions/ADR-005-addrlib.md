# ADR-005 — Latte's surface layout comes from AMD's addrlib

**Status:** Accepted · 2026-10-15

## Context

Almost every surface Latte touches is tiled: 8x8 micro-tiles, and in the 2D
modes macro-tiles spread across two pipes and four banks with a per-surface
swizzle on top. Every part of the emulator that touches a surface needs the
same arithmetic — how big it is, how it must be aligned, where each element
lives — and needs it to be *exactly* the console's: `GX2CalcSurfaceSizeAndAlignment`
returns sizes titles allocate with, and a texture decoded with one bit of the
bank swizzle wrong is noise.

The plan's risk table already names this: "Latte tiling/swizzle is
under-documented". The arithmetic is AMD's. GX2 itself is built on AMD's
addressing library, addrlib; AMD published addrlib under the MIT licence,
and Mesa carries it — for Evergreen and later, not R600/R700, which is what
Latte is.

The choices were:

1. Write it ourselves from the R600 documentation that exists.
2. Use addrlib with an R600/R700 backend.

## Decision

**Use addrlib, as a pinned third-party dependency, behind one narrow
wrapper.** The R600/R700 backend is decaf's: Mesa's R800 addrlib, adapted to
Latte by decaf's maintainers and published separately under AMD's MIT notice
(`github.com/decaf-emu/addrlib`). It is fetched at a fixed commit, built as
its own target without the project's warning set, and nothing outside
`latte/surface.hpp` includes it. The wrapper knows the Wii U's memory
configuration (two pipes, four banks, 256-byte interleave, 2 KiB rows) and
answers two questions: how is a surface laid out, and where is an element.

It was checked before being relied on: against a second, independently
written implementation of the same addressing (Cemu's, built in a scratch
harness and not part of the repository), 720 surface layouts and 92 million
element addresses, across eight formats, nine sizes, ten tile modes and the
bank/pipe swizzles, came out identical. `tests/latte/test_surface.cpp` pins
values from that run and checks the properties that would break first.

## Consequences

**Good.**

- The arithmetic is the one GX2 is built on, not a reconstruction of it.
- Two independent implementations agree, which is the strongest evidence
  available short of dumping surfaces from hardware.
- It is about five thousand lines we do not write, review or maintain.

**Bad, and worth saying.**

- **The R600 backend is an emulator project's adaptation, not AMD's.** AMD
  never published an R600 addrlib in this form. The agreement with Cemu
  makes a shared mistake unlikely but not impossible: Cemu's authors may have
  looked at the same places.
- It is a dependency with no releases, pinned to a commit, from a project
  whose activity is low. If it needs a fix, the fix is ours to carry.
- Its interface is AMD's C-style one — structs with `size` fields, a global
  handle — which the wrapper exists to keep away from everything else.

## What would make us revisit

- A surface from real hardware (a dump, or a title's output) that disagrees.
- Needing the layout on the GPU (a compute detiler, 8.5) in a form addrlib's
  per-element calls cannot feed fast enough. The answer then is to generate
  tables from it, not to replace it.
