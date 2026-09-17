# wiiu-emu

A Wii U (Cafe) emulator for macOS, written in C++20 with a native Metal renderer.

**Status: M2 reached.** A homebrew RPX loads, runs on the Espresso
interpreter and prints through `OSReport`. There is no renderer yet: M3
(`OSScreen` output through Metal) is the next milestone. See
[`plan.md`](plan.md).

## Requirements

- **macOS 13 (Ventura) or later, on Apple silicon.** Intel Macs are not
  supported and will not be — see
  [ADR-002](docs/decisions/ADR-002-apple-silicon-only.md).
- CMake 3.24+, Ninja, and the Xcode command line tools.

## Build

```sh
scripts/build-macos.sh              # configure, build and test (debug)
scripts/build-macos.sh release      # or: asan, core
scripts/build-macos.sh debug --clean
```

The script prints only what went wrong when something does, and keeps the full
output in `build/<preset>/build.log`. The underlying commands, if you would
rather run them yourself:

```sh
cmake --preset macos-debug
cmake --build --preset macos-debug
ctest --preset macos-debug
./build/macos-debug/bin/wiiu-emu --version
```

The platform-independent subsystems (types, logging, and later the CPU core,
loader and memory) also build on Linux, which keeps them testable in CI
without a Mac:

```sh
cmake --preset core-debug && cmake --build --preset core-debug && ctest --preset core-debug
```

Other presets: `macos-release`, `macos-asan` (ASan + UBSan; not usable with the
JIT once Phase 12 lands), `core-release`.

## Layout

| Path | Contents |
|---|---|
| `src/common` | Types, bit helpers, logging, version |
| `src/memory` | Guest address space, regions, cache model (Phase 3) |
| `src/cpu` | Espresso interpreter and MMU, then JIT (Phases 3.4, 4, 12) |
| `src/loader` | RPX/RPL, discs, filesystem (Phase 5) |
| `src/hle` | Cafe OS libraries (Phase 6) |
| `src/gpu` | PM4, shader translation, Metal backend (Phases 7–8) |
| `src/audio`, `src/input` | AX, VPAD/KPAD (Phase 9) |
| `src/frontend` | Executable, window, UI |
| `tools` | `rpl-dump` and friends (Phase 5.10) |
| `docs` | Architecture, performance reference, decision records |

## Documentation

- [`plan.md`](plan.md) — the 14-phase implementation plan and milestones.
- [`docs/performance.md`](docs/performance.md) — every speed technique (`P1`–`P40`),
  tagged by phase.
- [`docs/decisions/`](docs/decisions/) — architecture decision records.

## Legal

This project emulates hardware. It contains no Nintendo code, no encryption
keys, no system titles and no game data, and it never will. Running anything
requires content you dumped yourself from hardware you own, plus your own key
material, supplied at runtime and stored only on your machine.
