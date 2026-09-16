# Performance Reference

Every technique available to make `wiiu-emu` faster, tagged with the plan phase it lands in.

**How to read the payoff column.** These are order-of-magnitude expectations drawn from emulator practice generally (Dolphin, Cemu, RPCS3, PCSX2), not measurements of this codebase. They exist to rank work, not to predict results. Treat any number here as a hypothesis to be tested against §8.

**The ordering rule.** Items are grouped by expected payoff, not by subsystem. Within a tier, prefer the one whose phase you are already in. Do not implement anything from Tier 4 or below before the Tier 1 items in the same subsystem exist.

**IDs** (`P1`…`P40`) are stable — reference them in commits, issues, and ADRs.

---

## Tier 1 — Multipliers

These change the shape of the performance curve. Everything in later tiers is a percentage adjustment on top of them.

| ID | Technique | Phase | Payoff |
|---|---|---|---|
| **P1** | **JIT recompiler instead of interpreter** | 12 | **5–20×** |
| **P2** | **Idle-loop / spin detection** | 6.3, 12.8 | **1.5–3×** on affected titles |
| **P3** | **Fastmem — host MMU does the bounds checking** | 3.3, 3.5 | 10–40% of CPU time |
| **P4** | **CPU↔GPU sync discipline** | 8.2 | up to **2×** on sync-heavy titles |
| **P5** | **Texture + shader caching with range invalidation** | 8.10, 8.13 | ~10× versus no cache |

### P1 — JIT recompiler
Already Phase 12. Listed here only to anchor the scale: nothing else on this page competes with it.

### P2 — Idle-loop / spin detection
Guest threads spin on `OSWaitEvent`, on polled flags, and on GPU fences. Emulating that spin faithfully burns host cycles producing nothing. Detect the pattern — a short backward branch whose body only loads and compares — and advance emulated time to the next scheduled event instead.

Cheap to implement, large payoff, and it compounds with P1 because spinning is exactly the kind of hot loop a JIT will otherwise compile beautifully and then run millions of times for no reason. **Worth prototyping in the interpreter (Phase 6.3) before the JIT exists**, since it shows up as wall-clock time immediately.

### P3 — Fastmem
Map the guest address space so translation is `host_base + zero_extend32(guest_addr)` with no branch. Catch MMIO and unmapped access with a `SIGSEGV`/`SIGBUS` handler that decodes the faulting host instruction, services it, and resumes.

This removes a compare-and-branch from **every guest load and store**. It is also the item on this page with the strongest ordering constraint: **Phase 3.3 must be designed for it**. Retrofitting fastmem onto an accessor API that assumed explicit checks means rewriting the memory subsystem and every JIT load/store path.

Caveats: the signal handler must be async-signal-safe and must correctly decode arm64 load/store addressing modes; and it interacts with the JIT's `MAP_JIT` regions, so test the two together (Phase 12.3) rather than in isolation.

### P4 — CPU↔GPU sync discipline
GX2 has explicit synchronisation points. A naive implementation adds many implicit ones — waiting on the GPU thread after each draw, each buffer update, each `GX2Flush` — and serialises two pipelines that the hardware ran concurrently.

The rule: the GPU thread consumes the PM4 ring asynchronously (Phase 8.2), and the CPU blocks **only** where the guest actually asked it to. Every other wait is a bug. Instrument this early — a counter of "CPU stalls waiting on GPU per frame" that should be near zero is the cheapest diagnostic on this list.

### P5 — Texture and shader caching
Detiling a texture or translating R700 microcode every time it is referenced is catastrophic rather than merely slow. This is table stakes, listed in Tier 1 because **incorrect invalidation silently reintroduces the full cost** — a cache that misses every frame looks like a cache and performs like none. Pair it with a hit-rate counter from day one.

---

## Tier 2 — JIT code quality

Each of these is roughly 10–30% of CPU time. They multiply with each other and all land in Phase 12.

| ID | Technique | Phase | Note |
|---|---|---|---|
| **P6** | Dead flag elimination | 12.6 | Largest of this tier |
| **P7** | Block linking / direct chaining | 12.6 | |
| **P8** | Superblock / trace formation | 12.1 | |
| **P9** | Register pinning to arm64 GPRs | 12.1 | arm64's 31 registers make this unusually effective |
| **P10** | Inline caches for indirect branches | 12.6 | Plus a return-address stack |
| **P11** | Byte-swap elimination | 12.2 | |
| **P12** | Lazy FPSCR evaluation | 12.5 | |
| **P13** | Tiered compilation | 12.8 | Latency, not throughput |
| **P14** | Persistent on-disk code cache | 12.4 | Second-launch warm-up |
| **P15** | Page-protection SMC detection | 12.4 | Cheaper than checksumming |

### P6 — Dead flag elimination
PowerPC updates CR and XER constantly; most of it is never read. A naive JIT computes all of it. Track liveness across the block and drop the dead computations. **This is the largest single JIT optimisation for PowerPC specifically** and should be the first one implemented.

### P9 — Register pinning
The PPC register file is 32 GPRs; arm64 offers 31. Not a perfect fit, but enough to keep the hot subset resident across a whole block instead of round-tripping through the `PPCCore` struct per instruction. Profile which guest registers are actually hot rather than assuming r1/r2/r13.

### P11 — Byte-swap elimination
`rev` is a single cycle, so this is not about the swap itself — it is about values that get swapped on load, used, and swapped again on store, when the intermediate form was never observed. Track byte-order as a property in the IR and swap only at real observation boundaries.

### P15 — Self-modifying code detection
Page-protect guest code pages and invalidate on write, rather than checksumming block bytes on entry. Note the interaction with P3: both use the `SIGSEGV` path, so they share a handler and must be designed together.

---

## Tier 3 — HLE reach

The best effort-to-payoff ratio on this page. Every guest routine replaced by a native one is pure profit, and each is a few hours of work.

| ID | Technique | Phase | Payoff |
|---|---|---|---|
| **P16** | Native `memcpy`/`memset`/`OSBlockMove`/string funcs | 6.8 | **5–20% overall** |
| **P17** | Native `DCFlushRange`/`DCInvalidateRange` | 3.6, 6.8 | Called per-frame at high frequency |
| **P18** | HLE more libraries (`zlib125`, `nn_ffl`, fonts, math) | 9.9 | Proportional to guest time spent there |
| **P19** | Stub what does not affect output | 6.11, 9.9 | Telemetry, `nn_olv`, account checks |

### P16 — Native memory routines
Games call `coreinit`'s memory and string functions constantly. Running the guest's PowerPC implementation of `memcpy` through the JIT is strictly worse than calling the host's. **Probably the highest return-per-hour item in this document.**

Correctness note: the native implementation must preserve guest-visible byte order and must respect the cache model (P17) — a native `memcpy` that skips a flush the guest relied on produces GPU corruption that looks like a renderer bug.

---

## Tier 4 — Metal and GPU

Apple silicon has genuine structural advantages here; several of these have no equivalent on other backends.

| ID | Technique | Phase | Note |
|---|---|---|---|
| **P20** | **Render pass load/store actions** | 7.7, 8.14 | **10–30% of GPU time** — Apple-specific, easy to get wrong |
| **P21** | Memoryless render targets | 8.14 | Depth/stencil and MSAA never read back |
| **P22** | Zero-copy guest memory via unified memory | 7.4 | Structural Metal advantage |
| **P23** | Bindless via argument buffers Tier 2 | 7.5 | 5–15% CPU-side |
| **P24** | Async shader compilation | 8.10 | Stutter, not throughput |
| **P25** | Ubershader fallback while compiling | 8.10 | Eliminates hitching entirely |
| **P26** | GPU-side detiling via compute | 8.5 | |
| **P27** | Indirect command buffers | 8.2 | Needs Apple9+ for mesh draws |
| **P28** | State dedup + draw batching | 8.11 | |
| **P29** | Asynchronous occlusion queries | 8.16 | Never block on readback |
| **P30** | Skip redundant clears and copies | 8.14 | |
| **P31** | Programmable blending over multi-pass | 8.11 | Apple silicon only — which we are |

### P20 — Render pass load/store actions
The largest Apple-specific item on this page. Apple GPUs are tile-based deferred renderers: each render pass loads tile memory at the start and stores it at the end. Defaulting every attachment to `load`/`store` moves the entire framebuffer through memory twice per pass for no reason.

Use `MTLLoadActionDontCare` where the pass fully overwrites the target, and `MTLStoreActionDontCare` where nothing reads the result. Depth buffers are the common case: written every frame, read back almost never.

This is easy to get wrong precisely because the wrong version is *correct* — it just costs bandwidth. Audit it explicitly rather than expecting a bug report.

### P22 — Zero-copy guest memory
Unified memory means a shared-storage `MTLBuffer` can alias guest RAM directly: no staging buffer, no upload, no copy. This was impossible on the MoltenVK path and is one of the concrete reasons the Metal decision (ADR-001) was right. Design Phase 7.4's allocator around it.

Caveat: aliasing guest memory means the CPU can write while the GPU reads, so this leans on the P5 invalidation tracking and the Phase 3.6 cache model being correct.

### P25 — Ubershader fallback
Dolphin's technique. On a shader cache miss, render immediately with a slow generic shader that interprets the guest shader's parameters, while the specialised variant compiles on a background thread. Swap it in when ready.

Costs throughput on the frames it covers; eliminates compile hitching completely. For an emulator, a consistent 55 fps beats 60 fps with a 300 ms stall every time a new effect appears on screen.

---

## Tier 5 — Threading and host tuning

| ID | Technique | Phase | Note |
|---|---|---|---|
| **P32** | Three guest cores free-running on three host threads | 6.3 | Accuracy/speed dial lives here |
| **P33** | Dedicated GPU and audio threads | 8.2, 9.2 | |
| **P34** | QoS classes — `USER_INTERACTIVE` on hot threads | 6.3 | Keeps macOS off efficiency cores |
| **P35** | Batch the JIT W^X toggle | 12.3 | Or use a dedicated writer thread |
| **P36** | Narrow `sys_icache_invalidate` ranges | 12.3 | Only the bytes actually emitted |
| **P37** | Cache-conscious guest state layout | 4.1 | Avoid false sharing between core threads |
| **P38** | LTO, PGO, `-mcpu=apple-m1`, mimalloc | 1.2 | 5–15% for nearly no work |

### P34 — QoS classes
macOS will happily schedule an emulator thread onto an efficiency core. Explicitly request `QOS_CLASS_USER_INTERACTIVE` for the three CPU threads and the GPU thread. A few lines of code, real effect, frequently forgotten.

### P38 — Build-level wins
Free speed, available from Phase 1.2. PGO needs a representative workload, so revisit it once a title actually boots.

---

## Tier 6 — Accuracy traded for speed

User-facing toggles, all behind the Phase 2.5 per-title config layer so they are opt-in and auditable. None of these are on by default.

| ID | Technique | Phase |
|---|---|---|
| **P39** | Lower internal resolution + MetalFX spatial upscaling | 8.15 |
| **P40** | Relaxed timing, skipped audio effects, per-title hacks | 14.3 |

---

## 7. Traps

**Over-threading the guest cores.** P32 assumes the three Espresso cores can run free. When a title synchronises heavily, the cross-thread coordination cost can exceed what parallelism buys, and a lock-stepped or even single-threaded core scheduler is *faster*. Measure per title; expose it as a setting.

**Building the JIT before the interpreter is bit-exact.** P1 is the biggest win on this page and is deliberately Phase 12 anyway. A JIT built on an interpreter that is not yet trustworthy produces divergence nobody can localise, because there is no reference to diff against. Phase 12.7's lockstep differential testing only works if the interpreter is right.

**Optimising the emulator instead of the emulation.** Time spent micro-tuning C++ that runs once per frame is time not spent on P2 or P16, which run millions of times per frame. Profile first, always.

**Caches that silently miss.** P5, P14, and P24 all fail soft: the emulator stays correct and gets slow. Every cache ships with a hit-rate counter visible in the debug UI.

---

## 8. Measurement

Nothing on this page gets implemented on the strength of its payoff column. The order is: measure, find the actual bottleneck, pick the matching item, measure again.

**Instrumentation to build (Phase 10.3):**
- Per-subsystem frame time breakdown: CPU cores, GPU thread, audio, shader compile.
- CPU stalls waiting on GPU, per frame — the P4 diagnostic. Should be near zero.
- Cache hit rates: texture, shader, pipeline, JIT block.
- JIT statistics: blocks compiled, compile time, bytes emitted, invalidations per second.
- Guest time attribution: which HLE functions and which guest code addresses dominate. This is what tells you whether P16 is worth it *for this title*.

**Host tooling to use rather than rebuild (Phase 10.6):**
- `MTL_HUD_ENABLED=1` — frame time, memory, shader compilation counts, zero code changes.
- Instruments Metal System Trace — CPU/GPU timeline, shows P4 stalls visually.
- `metalperftrace` and `gpudebug` — CLI, no Xcode project needed.
- Xcode GPU frame capture — per-draw cost, shader cost graph, pixel-level heat maps.
- Instruments Time Profiler / `perf` for host-side CPU.

**Comparing two runs honestly (Phase 10.4).** A speed change that also
changes behaviour is not a speed change. `--record` on a known-good run and
`--replay` on the optimised one reduces "did that break anything?" from a
judgement call to a yes or no: every host value is replayed rather than
re-read, and state hashes at fixed instruction counts say where the two first
differ. This is the cheapest guard there is against an optimisation that is
only fast because it is wrong, and it is what Phase 12's JIT lockstep is built
on.

**The five to do first**, if the profile does not say otherwise:

| Order | Item | Phase |
|---|---|---|
| 1 | **P3** Fastmem | 3.3 — *design constraint, must be decided before Phase 3 ships* |
| 2 | **P2** Idle-loop skipping | 6.3 |
| 3 | **P16** Native `memcpy`/`memset` | 6.8 |
| 4 | **P4** CPU↔GPU sync discipline | 8.2 |
| 5 | **P20** Render pass load/store actions | 7.7 |

Together these are most of the achievable speed outside the JIT itself — and unlike P1, all five are reachable before Phase 12.
