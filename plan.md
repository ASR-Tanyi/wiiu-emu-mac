# Wii U Emulator for macOS — Implementation Plan

**Project:** `wiiu-emu` — a Wii U (Cafe) emulator written in modern C++ (C++20), targeting macOS on Apple silicon.
**Renderer:** **native Metal** via `metal-cpp`. Metal 3 baseline (macOS 13+), Apple7 GPU family (M1/A14+) required for the geometry-shader path.
**Status:** planning. Nothing implemented yet.

---

## 0. Ground rules

### Legal / ethical scope
- This project emulates hardware. It ships **no** Nintendo code, **no** encryption keys, **no** system titles, **no** game data.
- The emulator reads content the user dumped from hardware they own. Key material (Wii U common key, per-console OTP, per-disc keys) is supplied by the user at runtime and never committed, embedded, or distributed.
- Anything that cannot be reimplemented cleanly gets a documented HLE stub, not a copied binary.

### Engineering rules
- **Every phase ends in something runnable.** No phase is "write 8000 lines and hope".
- **Tests before breadth.** A CPU instruction is not done until it passes a test; a format is not parsed until a real file round-trips.
- **Big-endian guest, little-endian host.** All guest memory access goes through typed byte-swapping accessors from day one. Retrofitting this later is a rewrite.
- **HLE-first.** Emulate Cafe OS libraries (`coreinit.rpl`, `gx2.rpl`, …) at the function level. LLE of the ARM security processor (Starbuck/IOSU) is explicitly out of scope except where a stub is insufficient.
- **Performance work is tracked separately** in [`docs/performance.md`](docs/performance.md) — 40 techniques (`P1`–`P40`) tagged by phase. Five of them are design constraints rather than later optimisations, and are flagged inline below.
- Prior art to read, not copy: **decaf-emu** (clean HLE architecture, R700 shader translator) and **Cemu** (mature, MPL-2.0, ships the only production Metal backend for this hardware). Both are worth studying; license boundaries respected.

### Platform decision: Apple silicon only
Metal is now the renderer, and the geometry-shader path depends on **mesh shaders**, which require GPU family **Apple7** — Apple silicon only. Intel Macs report Metal 3 but have no mesh shader support (Cemu explicitly vendor-blacklists Intel because its drivers claim Metal 3 and then break). Supporting Intel would mean writing a *second* geometry path as a compute pre-pass, on a platform where macOS 26 Tahoe is the last supported release and macOS 27 drops it entirely. **Target arm64 only.** Revisit only if a compute GS path turns out to be needed anyway for other reasons.

### Reference hardware
| Part | Detail |
|---|---|
| CPU | IBM **Espresso** — 3× PowerPC 750CL-derived cores, 1243.125 MHz (248.625 MHz bus; timebase 62.15625 MHz), 32-bit, **big-endian**, limited out-of-order (in-order retire) |
| L1 / L2 | 32 KB I + 32 KB D per core; L2 **asymmetric**: core0 512 KB, core1 2 MB, core2 512 KB |
| CPU ISA | 750CL + **paired singles** (`ps_*`, `psq_l/psq_st` with GQR quantization). **No AltiVec.** 32-byte cache lines, locked L1D + DMA (HID2, SPR 922/923) |
| Coherency | MERSI across the three L2s with intervention — cores *are* coherent. **Not** coherent with GPU/IO: software `DCFlushRange`/`DCInvalidateRange` required. `lwarx`/`stwcx.` are hardware-buggy; Cafe OS carries workarounds |
| GPU | AMD **Latte**, R700/TeraScale-derived, 549.999 MHz, 16 TMUs, 8 ROPs. (Shader-unit count is disputed: 160 vs 320. Irrelevant to emulation.) Command stream = **PM4 packets**; shaders = **R700 microcode** (CF + ALU/TEX clauses) |
| MEM1 | 32 MB on-die fast memory (1T-SRAM/eDRAM), phys `0x00000000`, virt `0xF4000000` |
| MEM0 | 3 MB SRAM, phys `0x08000000`–`0x082FFFFF` — kernel + IOS images |
| MEM2 | 2 GB DDR3-1600, phys `0x10000000`–`0xCFFFFFFF`. ~1 GB to apps, ~1 GB to OS |
| IOP | **Starbuck**, ARM926EJ-S inside Latte, ~96 KB SRAM, runs IOSU (crypto, FS, ES, USB, net) |
| Audio | DSP in Latte; **AX** API via `sndcore2.rpl`. 96 voices, 32 kHz **or** 48 kHz renderer, 3 ms frame (96/144 samples) |

### Cafe OS virtual memory map (target of Phase 3)
| Virtual | Size | Physical | Contents |
|---|---|---|---|
| `0x01000000` | `0x800000` | `0x32000000` | Loader + system libraries |
| `0x01800000` | `0x20000` | — | Codegen/JIT area (sized by `cos.xml`) |
| `0x02000000` | ≤`0x0E000000` | `0x72000000` | App executable + libraries (ends at `0x10000000`) |
| `0x10000000` | `0x52000000` | `0x20000000` | **App + library data heap** |
| `0xE0000000` | `0x4000000` | `0x14000000` | Hardware / IPC communication area |
| `0xF4000000` | `0x2000000` | `0x00000000` | **MEM1** |
| `0xF8000000` | `0x3000000` | `0x18000000` | Read-only shared data (system fonts) |
| `0xFF200000` | `0x80000` | `0x1B800000` | Kernel heap |

MMIO: PI `0x0C000000`, GPU `0x0C200000`, audio `0x0C280000`, AHB trusted `0x0D000000`.

---

## Phase 1 — Repository and build skeleton

**Goal:** `cmake --build` produces an empty app that opens a Metal-backed window and exits cleanly.

- [x] 1.1 Create repo layout: `src/common`, `src/cpu`, `src/memory`, `src/loader`, `src/hle`, `src/gpu`, `src/audio`, `src/input`, `src/frontend`, `tests/`, `tools/`, `docs/`, `third_party/`.
- [x] 1.2 CMake ≥3.24, C++20, `CMAKE_OSX_ARCHITECTURES=arm64`, `CMAKE_OSX_DEPLOYMENT_TARGET=13.0`. Split into static libs per subsystem plus one executable. Enable Objective-C++ (`.mm`) for the platform layer. Release builds get LTO and `-mcpu=apple-m1`; link mimalloc (**P38**).
- [~] 1.3 Vendor dependencies (fmt, spdlog, Catch2 fetched; the rest declared with pinned tags in `cmake/Dependencies.cmake`, enabled by the phase that needs them): `metal-cpp` (Apple, Apache-2.0, header-only), `fmt`, `spdlog`, `catch2`, `SDL3` (window + input + audio device), `zlib`, `pugixml` (for `cos.xml`/`app.xml`/`meta.xml`), `imgui` (with its Metal backend), `mbedtls` (AES-128-CBC, SHA1/SHA256, RSA verify).
- [ ] 1.4 **Platform layer.** `metal-cpp` deliberately excludes MetalKit and AppKit, so there is no `MTKView` — drive a `CAMetalLayer` directly. Either take SDL3's Metal view or write a thin `.mm` file that creates the window and layer. Wrap `metal-cpp`'s manual `retain`/`release` (no ARC) in an RAII scope helper before writing any Metal code, or leaks are guaranteed.
- [x] 1.5 Clang-format + clang-tidy config; warnings-as-errors; `-fno-strict-aliasing` where guest memory is punned.
- [x] 1.6 Logging facility: per-subsystem channels (`CPU`, `MEM`, `LOADER`, `COREINIT`, `GX2`, `AX`, `GPU`), compile-time level, ring buffer for the debug UI.
- [x] 1.7 Test harness wired into `ctest`; GitHub Actions job on macos-latest (arm64 runner) building + testing.
- [ ] 1.8 **Prove the signing path now, not at release.** A bare `.app` bundle, Hardened Runtime, `com.apple.security.cs.allow-jit`, codesign, notarize, staple — round-trip it once in Phase 1 so Phase 12's JIT doesn't collide with an unknown at the end.
- [x] 1.9 `docs/architecture.md` skeleton, `docs/performance.md` (already written), and `docs/decisions/` (one ADR per significant choice — the Metal-over-Vulkan and arm64-only decisions are ADR-001 and ADR-002).

**Done when:** CI is green, `./wiiu-emu --version` runs, a `CAMetalLayer` window opens, clears to a colour, and closes.

---

## Phase 2 — Common primitives

**Goal:** the endian, type, and container vocabulary the whole emulator uses.

- [x] 2.1 `be<T>` — a byte-swapping storage type with implicit conversion, usable inside guest structs so `be<uint32_t> field;` just works. arm64 `rev`/`rev16` make this nearly free. Verified with static asserts on size/alignment.
- [x] 2.2 Fixed-width guest types: `u8/u16/u32/u64/s8/.../f32/f64`, `ppcaddr_t` (guest virtual, always `uint32_t`), `physaddr_t`.
- [x] 2.3 Bitfield helpers for register decoding (`BitField<pos, len, T>`), used by PPC opcode decode and PM4 register writes.
- [x] 2.4 Small utilities: `align_up`, `bit_cast`, `rotl`, `count_leading_zeros`, span-like `GuestPtr<T>`.
- [x] 2.5 Config system: TOML or JSON settings file + command-line override, hot-reloadable in debug, with per-title override profiles (Phase 14 depends on this).
- [x] 2.6 Unit tests for every one of the above. *(59 tests, including an Elf32 RPX header parsed from a known blob and a per-title settings profile.)*

**Done when:** tests pass, and a hand-written guest struct with mixed `be<>` fields parses a known byte blob correctly. ✅ *Phase 2 complete.*

---

## Phase 3 — Memory subsystem

**Goal:** a faithful guest address space that the CPU and GPU both see.

- [x] 3.1 Reserve a 4 GB `mmap` region as the guest address space; map MEM1/MEM2/MEM0 as sub-mappings so guest virtual → host pointer is a single add.
- [x] 3.2 Implement the Cafe OS virtual map from the table above; represent it as a table of `MemoryRegion { virtual, size, physical, flags, name }`.
- [x] 3.3 Typed accessors: `read<T>(addr)`, `write<T>(addr, value)` performing byte-swap; `translate(addr)` → host pointer. **⚠ Design for fastmem (P3) now.** Translation must reduce to `host_base + zero_extend32(guest_addr)` with no branch, with MMIO and unmapped access caught by a `SIGSEGV`/`SIGBUS` handler that decodes the faulting arm64 instruction. This removes a check from every guest load and store (10–40% of CPU time) and **cannot be retrofitted** onto an accessor API that assumed explicit checks.
- [x] 3.4 MMU model: BAT registers, segment registers, page table search. Start with a flat/identity fast path plus a fallback slow path that honours the real translation, so translation bugs surface but don't cost performance in the common case. *(`src/cpu/mmu.hpp`: eight IBAT/DBAT pairs with the upper four gated by HID4, sixteen segment registers, the hashed page table with both hashes, protection by PP and segment key, SR[N], reference and change bits, and a 256-entry direct-mapped TLB per side. Translation off is one test and the effective address passes through, so nothing is charged to the HLE path. Real addresses are mapped back through the region table, because Phase 3.1 keys the address space by virtual address — documented at the top of `mmu.hpp`, and the one place this deviates from hardware.)*
- [x] 3.5 MMIO dispatch: address ranges routed to device handler objects (GPU, audio, IPC, PI) rather than backing memory — left unmapped so the P3 fault handler routes them. Shares its handler with SMC detection (**P15**), so design the two together.
- [x] 3.6 Cache model: `DCFlushRange` / `DCInvalidateRange` / `DCZeroRange` / `ICInvalidateRange` must invalidate JIT blocks and mark GPU-visible ranges dirty. **Model the CPU↔GPU incoherency explicitly** — games depend on the flushes being real.
- [x] 3.7 Memory watchpoints for the debugger.
- [x] 3.8 Tests: region overlap assertions, read/write round-trips across region boundaries, MMIO routing.

**Done when:** a synthetic test writes through every mapped region and reads back correct big-endian values, and MMIO writes reach a fake device. ✅ *Phase 3 complete.*

---

## Phase 4 — PowerPC interpreter (Espresso)

**Goal:** a correct, slow, single-core interpreter. Correctness now; speed in Phase 12.

- [x] 4.1 `PPCCore` state, cacheline-aligned and laid out hot-fields-first to avoid false sharing between the three core threads (**P37**): 32 GPR, 32 FPR (each 2×f64 for paired singles), CR, XER, LR, CTR, MSR, SRR0/1, HID0-5, GQR0-7, SPRs, TB, DEC, SCR.
- [x] 4.2 Opcode decoder: generated from a declarative instruction table (primary opcode + extended opcode), producing both an interpreter dispatch table and a disassembler from one source of truth.
- [x] 4.3 Disassembler first — it makes every later bug tractable. Validated by **round-trip**: every instruction in the table is disassembled and fed back through binutils' PowerPC assembler, which must produce the identical word (`scripts/ppc_roundtrip.py`, wired into ctest). Stronger than diffing objdump text, since it checks meaning rather than spelling. 187/187 pass. Also ships `ppcdis`.
- [x] 4.4 Integer instructions: arithmetic, logical, compare, rotate/shift, load/store (incl. byte-reversed `lwbrx`/`stwbrx`, multiple `lmw`/`stmw`, string). Condition register ops.
- [x] 4.5 Branch unit: `b/bl/bc/bclr/bcctr`, link stack, `sc` system call trap.
- [x] 4.6 FPU: IEEE-754 double + single ops, `frsp` rounding, FPSCR flags, `fmadd` family with the correct non-fused/fused behaviour, `fres`/`frsqrte` estimate tables matching 750CL precision. *(Estimates are now the 32-entry base/decrement tables, and FPSCR is modelled properly: sticky exception bits with FX meaning "first one", VX and FEX recomputed as summaries rather than stored, FPRF, FI and FR, the enable bits, `mtfsf`/`mtfsb0`/`mtfsb1`/`mtfsfi`/`mcrfs`, and a program exception when an enabled exception fires with MSR[FE0]/[FE1] set. Overflow, underflow and inexact are read back from the host FPU running in the guest's rounding mode, so they come from the arithmetic rather than from a guess about it; FR for a double-precision result is recovered by re-evaluating toward zero. Two gaps remain, both dormant while Cafe OS leaves the enables clear: an enabled exception does not suppress the result write, and MSR[FP] is not consulted, so there is no floating-point-unavailable exception. The estimate tables are transcribed hardware data and worth checking against the 750CL manual — the accuracy tests bound them at 1/256 and 1/32, and measure 1/943 and 1/5461.)*
- [x] 4.7 **Paired singles**: `ps_add/sub/mul/div/madd/msub/sel/merge/neg/abs/res/rsqrte/cmp`, and `psq_l/psq_lu/psq_st/psq_stu` with all GQR quantization types (float, u8, u16, s8, s16) and scale exponents.
- [x] 4.8 Supervisor: `mtmsr/mfmsr`, `mtspr/mfspr`, `rfi`, `tlbie`, cache instructions (`dcbz`, `dcbf`, `dcbst`, `icbi`, `dcbt`). *(Complete with 3.4: `mtsr`/`mfsr`/`mtsrin`/`mfsrin`, `tlbie`, `tlbsync`, the BAT and SDR1 SPRs, and a privilege check — supervisor instructions and SPRs now take a program exception in problem state, without which the MMU would be decorative. No `tlbia`: the 750 does not implement it. Cache instructions translate their effective address like any other access.)*
- [x] 4.9 Reservations: `lwarx`/`stwcx.` per-core reservation granule. Note the real hardware bug — model the *documented* behaviour, and add a compatibility switch if a title turns out to depend on the flaw.
- [x] 4.10 Exception model: decrementer, external interrupt, DSI/ISI, program, floating-point unavailable, system call. Vector table at `0x00000100`-relative offsets honouring `MSR[IP]`. **Note: the Cafe OS map has nothing at low addresses, so 6.2's loader must place a vector page before exception delivery works.**
- [x] 4.11 Timebase and decrementer driven from a host monotonic clock scaled to the timebase rate. *(Currently one tick per instruction — monotonic and proportional, not cycle-accurate, and deterministic, which 10.4 depends on. Hook to a host clock when frame pacing needs it. The rate is 62.15625 MHz: like every PowerPC 750 derivative the timebase counts the **bus** clock / 4, not the core clock / 4 — a factor of five that would have made every emulated second last a fifth as long.)*
- [~] 4.12 **Test suite**: port/adapt an existing PPC instruction test corpus; write a self-checking guest test harness; differential-test the FPU against a software reference. Target: every implemented instruction has at least one test, edge cases (overflow, NaN, denormal, carry) have their own. *(Integer/branch done: 520 golden vectors from an independent Python reference in `scripts/gen_cpu_fixtures.py`, plus six binutils-assembled programs. Extend with the FP work.)*

**Done when:** a hand-assembled PPC program computing a known result runs to completion, and the instruction test suite passes 100%.

**Note on `rfi` and the vector base**, found while testing 3.4 and worth remembering: `MSR[IP]` selects the vector base, so it must survive exception entry — clearing it with the rest of the mask sends every exception to the low addresses whatever the guest configured. And `rfi` restores only the architected MSR bits: SRR1 also carries the exception's cause, and copying it wholesale puts the core into power-saving mode on the way back from a privileged-instruction exception.

---

## Phase 5 — Executable and filesystem formats

**Goal:** turn user-dumped content into loadable code and readable files.

- [x] 5.1 **RPX/RPL parser.** ELF32, PPC, big-endian, `e_type = 0xFF01`, **no program headers** — section headers drive loading. Handle `SHF_DEFLATED = 0x08000000` (zlib inflate per section).
- [x] 5.2 Custom sections: `SHT_RPL_EXPORTS 0x80000001`, `SHT_RPL_IMPORTS 0x80000002`, `SHT_RPL_CRCS 0x80000003`, `SHT_RPL_FILEINFO 0x80000004`. Verify per-section CRC32. Read `RplFileInfo` (0x60 bytes), check `RPL_IS_RPX = 0x2`.
- [~] 5.3 Import/export resolution: PE-DLL-style tables, not ELF dynamic symbols. `.fexports`/`.dexports` (8-byte header, 8-byte entries, nametable) and `.fimport_<lib>`/`.dimport_<lib>`. Loader patches import stubs to point at HLE trampolines. *(Tables are parsed; stub patching belongs with the Phase 6 loader.)*
- [x] 5.4 Relocations: `R_PPC_ADDR32`, `ADDR16_LO/HI/HA`, `REL24`, `REL14`, `DTPMOD32`/`DTPREL32` for TLS.
- [x] 5.5 XML metadata: `cos.xml` (heap sizes, codegen area, permissions), `app.xml`, `meta.xml` (title ID, name, region).
- [x] 5.6 **Loadiine/decrypted layout** first: `/code`, `/content`, `/meta` on disk. This is the cheapest input path and unblocks everything else.
- [ ] 5.7 Then **WUP/NUS**: TMD, ticket (`.tik`), numbered `.app` contents, `.h3` hashes; FST in the first `.app`. AES-128-CBC with title key (decrypted from the ticket using the user-supplied common key), IV derived from content index.
- [ ] 5.8 Then **WUD/WUX** disc images: partition table, SI (ticket+TMD), GM (game), GI/UP (installer/update); disc key handling. WUX = lossless WUD compression. Optionally import Cemu's **WUA** archive format.
- [x] 5.9 Virtual filesystem layer presenting `/vol/content`, `/vol/meta`, `/vol/save`, `/vol/storage_mlc01` regardless of which container the data came from.
- [x] 5.10 `tools/rpl-dump` CLI — dumps sections, imports, exports, disassembly. Invaluable for the rest of the project.
- [x] 5.11 Tests: RPX/RPL fixtures built by `scripts/make_rpx.py` from binutils-assembled code — compressed and uncompressed, executable and library. **Still worth doing:** point `rpl-dump` at a real devkitPro/wut RPX, since these fixtures validate the parser against this project's reading of the format, not the reading itself.

**Done when:** `rpl-dump` prints a correct, `objdump`-matching disassembly and complete import table for a real homebrew RPX.

---

## Phase 6 — Kernel, scheduler, and coreinit HLE

**Goal:** a homebrew `.rpx` reaches `main()` and prints to the log. This is the first "it's alive" moment.

- [x] 6.1 HLE framework: a registry mapping `library.symbol` → native handler; automatic argument marshalling from PPC ABI (r3–r10 int, f1–f13 float, stack overflow) via templates, so a handler is written as an ordinary C++ function.
- [~] 6.2 Loader: resolve RPL dependency graph, map into the `0x02000000` region, allocate heaps per `cos.xml`, build the import trampolines, set up the initial stack and `r2`/`r13` SDA pointers.
- [~] 6.3 Three `PPCCore` instances on three host threads, plus a scheduler owning runnable-thread queues per core. Request `QOS_CLASS_USER_INTERACTIVE` so macOS keeps them off efficiency cores (**P34**). Add **idle-loop / spin detection (P2)** here, in the interpreter, before the JIT exists — guest threads spinning on `OSWaitEvent` or polled flags are pure waste, and skipping ahead is worth 1.5–3× on affected titles. Expose free-running vs lock-stepped cores as a setting (**P32**): free-running is not always faster. *(Threads still interleave deterministically on one host thread — that is what makes 10.4's record/replay and 12.7's JIT lockstep possible, and real parallelism goes behind the same interface when it is worth measuring. **P2 is done**, in both halves: the scheduler keeps every deadline (sleeping threads and alarms) in one list, so when nothing can run the clock jumps to the next one instead of the run loop emulating the wait; and the interpreter recognises a spin — a short backward branch taken repeatedly with no store, no counted branch, no pointer walk and no system call in between — so a thread polling a flag hands the core back in a few hundred instructions rather than at the end of its timeslice. `RunReport` reports both as `skipped_ticks` and `spin_yields`.)*
- [x] 6.4 `coreinit.rpl` — threads: `OSCreateThread`, `OSResumeThread`, `OSJoinThread`, `OSExitThread`, `OSYieldThread`, `OSSetThreadAffinity`, `OSSetThreadPriority`, thread-local storage.
- [x] 6.5 `coreinit.rpl` — synchronisation: `OSMutex`, `OSCond`, `OSEvent`, `OSSemaphore`, `OSRendezvous`, `OSFastMutex`, spinlocks, atomic ops. *(All present. Two things worth remembering: a woken waiter resumes **after** its own lock call — the return address is set before the handler blocks it — so whoever unlocks has to name the new owner rather than leaving the mutex free, and a signalled condition variable has to hand its mutex to one waiter and queue the rest on it, or they resume unprotected. The atomics are plain reads and writes because guest threads interleave on one host thread and a handler runs to completion; they go through one helper, which is where the work lands if cores ever run in parallel (P32).)*
- [~] 6.6 `coreinit.rpl` — memory: `MEMCreateExpHeap`, `MEMCreateFrmHeap`, `MEMCreateUnitHeap`, `MEMAllocFromDefaultHeap`, the foreground/MEM1/MEM2 heap trio, `OSAllocFromSystem`. *(Expanded heaps, default/MEM1 heaps and the MEMAlloc family done. Bookkeeping is host-side rather than in guest block headers — noted in the header, and a contained change if a title turns out to walk them.)*
- [x] 6.7 `coreinit.rpl` — time and alarms: `OSGetTime`, `OSGetSystemTime`, `OSCreateAlarm`, `OSSetPeriodicAlarm`, `OSTicksToCalendarTime`. *(Alarms now run: one-shot and periodic, with user data and tags, cancellation, and `OSWaitAlarm`. They are deadlines on the timebase rather than the host clock, so an alarm fires after the same number of instructions every run and a replay sees it in the same place; a periodic one re-arms from its own deadline, so a 60 Hz alarm does not drift when the clock arrives late. The callback runs on its own thread — see the note below. Time was already done: `OSGetTime`/`OSGetTick` are the real-time clock plus the timebase, `OSGetSystemTime`/`OSGetSystemTick` are the timebase alone, and the calendar conversions go both ways. The real-time clock is read from the host exactly once and everything else derives from the deterministic timebase — which is what makes 10.4's replay reproduce a date rather than re-read the host. Alarms remain: their callbacks need the same guest-stack mechanism the asynchronous FS calls in 6.9 are waiting on.)*
- [~] 6.8 `coreinit.rpl` — misc: `OSReport` (→ our log; this is how homebrew talks to us), `OSFatal`, `OSDynLoad_Acquire`/`FindExport`, `OSScreen*` (the simple framebuffer API — a great first render target), `__os_snprintf`, atomics, `OSGetCoreId`. **Implement `memcpy`/`memset`/`OSBlockMove`/string functions natively (P16)** — best effort-to-payoff ratio in the project, 5–20% overall. They must still respect the Phase 3.6 cache model (**P17**), or they produce GPU corruption that looks like a renderer bug.
- [~] 6.9 `coreinit.rpl` — FS: `FSInit`, `FSAddClient`, `FSOpenFile`, `FSReadFile`, `FSGetStat`, `FSOpenDir`, both sync and async forms, backed by the Phase 5 VFS. *(Sync forms done: open/read/seek/tell/close, directories and stat. `FSOpenFileAsync`, `FSReadFileAsync` and `FSCloseFileAsync` complete through their callback, on a deadline rather than inside the call — a title that starts a read and then sets a flag depends on that order — and a command block carries one request at a time, as hardware requires. The other half of the API, completion posted to an `OSMessageQueue` for `FSGetAsyncResult`, still needs message queues; a request with no callback says so loudly rather than never arriving. Writing is refused rather than ignored; save data is a separate mount in 9.9.)*

**Callbacks from the OS into the title** — alarms, asynchronous completion, and later ProcUI — all run the same way: a guest function on its own scheduler thread, with the PowerPC ABI's argument registers set and a stack from a recycled pool. Hardware uses a system thread per core; a thread per invocation has the same observable shape without a pool whose state could disagree with the scheduler's. Two traps found while building it, both of which look like the emulator hanging: a handle handed back to a new thread while the scheduler still believed that thread was running, and a thread exiting while every other thread is asleep, which leaves the core sitting on the exit sentinel with nothing to switch to.
- [ ] 6.10 IPC stub: intercept PPC→IOSU IPC requests and service them in HLE rather than emulating Starbuck.
- [ ] 6.11 `proc_ui.rpl` (`ProcUIProcessMessages`, foreground/background transitions) and `nsysnet`/`nn_act`/`nn_save`/`nn_acp` stubs that return plausible success.
- [x] 6.12 Missing-function policy: an unimplemented import logs loudly with its library, name, and caller, then returns 0 — so the next gap is always visible.

**Done when:** a "hello world" homebrew RPX built with devkitPro/wut runs to completion and its `OSReport` output appears in our log. **Ship a tagged release here.**

✅ **Reached with a binutils-built RPX** — real symbols, real relocations, `OSReport` output correct. Still to do: run one built with devkitPro/wut, which exercises the real toolchain's section layout and a far larger import surface.

---

## Phase 7 — Metal backend foundation

**Goal:** a Metal device, a swapchain, and the first Wii U pixels on screen.

### What Metal gives you, and what it doesn't

The renderer is native Metal, which is the right call — it's what Cemu ships, it's roughly 20–25% faster than the MoltenVK path, and it runs titles MoltenVK cannot. But Metal is a modern, deliberately narrow API and Latte is a 2008-era GPU. These gaps are **not** blockers, they are **shader-side emulation work** that must be designed in from the start:

| Latte needs | Metal status | Approach |
|---|---|---|
| Geometry shaders | **Absent** | Object + mesh shader pipeline (Apple7+). Also required for Latte **RECTS** primitives, not just real GS |
| Stream-out / transform feedback | **Absent** | `rasterizationEnabled = false` with a `void`-returning vertex function writing to `device` buffers; index output manually from `[[vertex_id]]`/`[[instance_id]]` — there is no hardware output counter. Use compute when a counter is genuinely needed |
| Provoking vertex control | **Absent** (private API only) | Bake flat-shading attribute selection into the shader, or rewrite index buffers |
| Non-seamless cube maps | **Absent** (private API only) | Metal cube sampling is always seamless; emulate in-shader if a title depends on the seam |
| Logic ops | **Absent** (private API only) | Shader logic ops via programmable blending / framebuffer fetch (Apple silicon only — the route Dolphin took) |
| Disabling primitive restart | **Impossible** — `0xFFFF`/`0xFFFFFFFF` are always restart indices | Rewrite the index buffer when Latte uses those as real indices |
| Custom sampler border colour | **Absent** — only `opaqueBlack`/`opaqueWhite`/`transparentBlack` | Emulate arbitrary border colour in-shader |
| Wide lines | **Absent** (line width fixed at 1.0) | Expand to quads if needed |
| Point fill mode | **Absent** (`.lines` and `.fill` only) | Draw points explicitly |
| Depth range `[-1,1]` | Metal is always `[0,1]` | Fold into the projection matrix |
| Vertex attribute alignment | Metal requires 4-byte alignment; Latte does not | **Manual vertex fetch inside the shader.** (Cemu tried a compute realign pre-pass and it was slower.) |

### Tasks

- [ ] 7.1 Define a `GPUBackend` interface — device/queue init, swapchain, buffers, textures, samplers, pipelines, resource binding, render passes, command submission, fences. Metal types do not leak through it. This costs little now and is the only thing that makes a future Vulkan/Linux port or a Metal-4 rewrite tractable.
- [ ] 7.2 Device creation via `metal-cpp`; `CAMetalLayer` swapchain from the Phase 1.4 platform layer; command queue; per-frame command buffers with triple buffering and a semaphore.
- [ ] 7.3 **Capability probe at startup.** Query `supportsFamily(MTL::GPUFamilyMetal3)`, `GPUFamilyApple7` (mesh shaders), `GPUFamilyApple9` (indirect mesh draws), argument buffers tier, programmable blending (`Apple2`+) / tile shaders (`Apple4`+). Log a capability report and refuse to start with a clear message rather than rendering garbage if mesh shaders are missing.
- [ ] 7.4 Memory allocator over `MTLHeap` (**P22** zero-copy guest memory) — suballocate buffers and textures rather than creating them individually. Guest memory backing exploits unified memory: a shared-storage `MTLBuffer` aliasing the guest address space avoids most staging copies. This is a real structural advantage Metal has over the Vulkan path on this hardware.
- [ ] 7.5 **Resource binding design — decide this before Phase 8.** Direct binding limits are 31 buffers, 128 textures (Apple6+), and **16 samplers** — samplers are the binding constraint, and Latte can want more. Cemu works within the limits by reserving slots (buffers 28–30, textures 29–30, samplers 14–15) for helper resources. **Argument buffers Tier 2** (all Apple silicon) lifts this to effectively unbounded and is the cleaner design. Choose tier-2 bindless unless profiling says otherwise; record the decision as an ADR.
- [ ] 7.6 Render a triangle. Then render the `OSScreen` framebuffer from Phase 6.8 as a textured quad — the first real Wii U pixels on screen.
- [ ] 7.7 Frame pacing, vsync via `CAMetalLayer` presentation, resize, and a present path handling both the TV output and the GamePad (DRC) output as two views. **Set render pass load/store actions deliberately from the first pass (P20)** — Apple GPUs are tile-based deferred, and defaulting to `load`/`store` moves the whole framebuffer through memory twice per pass. Worth 10–30% of GPU time, and the wrong version is *correct*, so it never shows up as a bug report.
- [ ] 7.8 Wire up the Metal tooling now: `MTL_HUD_ENABLED=1` overlay, `METAL_DEVICE_WRAPPER_TYPE=1` API validation, `MTL_SHADER_VALIDATION=1`, and confirm Xcode frame capture (or the `gpucapture` CLI) attaches to our process.

**Done when:** an `OSScreen`-based homebrew draws visible output, and a GPU frame capture of it opens cleanly in the Metal debugger.

---

## Phase 8 — GX2 and the Latte command processor

**Goal:** real GPU command streams produce real frames. This is the largest phase; budget accordingly.

- [ ] 8.1 `gx2.rpl` HLE surface: `GX2Init`, `GX2SetupContextStateEx`, `GX2SetContextState`, `GX2DrawEx`, `GX2DrawIndexedEx`, `GX2SwapBuffers`, `GX2WaitForVsync`, `GX2SetColorBuffer`, `GX2SetDepthBuffer`, `GX2SetViewport`, `GX2SetScissor`, `GX2Set{Vertex,Pixel,Geometry}Shader`, `GX2SetVertexUniformBlock`, `GX2InitTextureRegs`, `GX2CopySurface`, `GX2ClearColor`, `GX2ClearDepthStencilEx`.
- [ ] 8.2 **Command buffer model.** GX2 calls append PM4 to a ring buffer; the emulator runs a separate GPU thread consuming it, mirroring hardware asynchrony. Do not shortcut this into synchronous calls — timing-sensitive titles depend on the split. **The CPU blocks only where the guest actually asked it to (P4)**; every other wait is a bug, worth up to 2× on sync-heavy titles. Ship a "CPU stalls waiting on GPU per frame" counter alongside it — it should read near zero.
- [ ] 8.3 **PM4 parser**: Type-0 (base register + count), Type-2 (filler), Type-3 (opcode + size + predicate). Decode `IT_*` opcodes: `SET_CONFIG_REG`, `SET_CONTEXT_REG`, `SET_ALU_CONST`, `SET_RESOURCE`, `SET_SAMPLER`, `DRAW_INDEX_2`, `DRAW_INDEX_AUTO`, `INDEX_TYPE`, `NUM_INSTANCES`, `EVENT_WRITE`, `EVENT_WRITE_EOP`, `SURFACE_SYNC`, `CONTEXT_CONTROL`, `MEM_WRITE`.
- [ ] 8.4 Latte register file: the full context/config register state, decoded into a structured `LatteContext` that pipeline creation reads from.
- [ ] 8.5 **Surface/texture decoder**: Latte tiling modes (linear, 1D-tiled thin/thick, 2D-tiled, micro/macro tiling with bank/pipe swizzle), formats (R8G8B8A8, R5G6B5, BC1–BC5, depth D24S8/D32F, HW-specific ones), mip chains, array slices, 3D textures, cube maps. Detiling on the GPU via compute kernel where possible; CPU fallback for correctness.
- [ ] 8.6 **Shader decompiler.** Parse R700 microcode: CF clauses (ALU, TEX, VTX, EXP, loops, jumps, calls), ALU clauses (5-slot VLIW xyzwt, literals, previous-vector/previous-scalar), TEX/VTX clauses. Lower to an internal IR, then **emit MSL text directly**.
  - **Do not route through SPIR-V.** Apple's Metal Shader Converter takes DXIL, not SPIR-V, so the only SPIR-V→MSL path is SPIRV-Cross — and SPIRV-Cross's MSL backend has **no mesh shader support**, which kills the geometry-shader path outright. Direct MSL emission also keeps naming and binding conventions under our control. Keep the IR backend-agnostic so a SPIR-V emitter can be added later for a Vulkan port.
- [ ] 8.7 Shader analysis pass: detect integer ops, loops, texture dimensionality per sampler, uniform block usage, stream-out writes, and which Metal gaps from the Phase 7 table this shader triggers.
- [ ] 8.8 **Geometry shaders via object + mesh pipeline.** `MTLMeshRenderPipelineDescriptor` with an object function (primitive setup) and mesh function (vertex emission) replacing vertex→GS. Requires Apple7. **Latte RECTS primitives go through this same path** — so this is not an edge case for exotic titles, it's needed early.
- [ ] 8.9 **Stream-out.** Render pipeline with `rasterizationEnabled = false`, vertex function returning `void`, writing to `device` buffers with an explicitly computed output index. Keep the draw alive when `VGT_STRMOUT_EN` is set even with no framebuffer bound.
- [ ] 8.10 **Shader and pipeline cache.** Key by a hash of the microcode plus relevant register state. Use `MTLBinaryArchive` (`addRenderPipelineFunctionsWithDescriptor:` / `addMeshRenderPipelineFunctionsWithDescriptor:`, `serializeToURL:`) **plus** our own title-keyed on-disk cache — `MTLBinaryArchive` is OS/driver-version sensitive and can silently miss, so it cannot be the only layer. A pool of 1–8 background compile threads avoids first-encounter stutter (**P24**); an **ubershader fallback (P25)** that renders immediately with a generic shader while the specialised one compiles removes it entirely. Every cache ships with a visible hit-rate counter — caches fail soft, staying correct while silently costing everything they were meant to save.
- [ ] 8.11 Pipeline state translation: blend, depth/stencil, rasterizer state, `MTLDepthClipMode`, primitive types (Latte quad lists expanded; RECTS via 8.8), vertex fetch shader → **manual in-shader vertex fetch** per the alignment note in Phase 7.
- [ ] 8.12 Uniform handling: GX2 uniform blocks and ALU constants → Metal buffers via a ring allocator, or argument buffers if 7.5 chose bindless.
- [ ] 8.13 Texture and framebuffer cache with memory-range tracking (**P5**), so a CPU write to a texture's backing memory invalidates the GPU copy (this is where the Phase 3.6 cache model pays off).
- [ ] 8.14 Copy/resolve/clear paths, MSAA resolve, depth-to-colour and colour-to-depth copies. Make depth/stencil and MSAA targets that are never read back `memoryless` (**P21**), and skip clears the guest immediately overwrites (**P30**).
- [ ] 8.15 Resolution scaling: render the Wii U framebuffer at an integer multiple of native. **MetalFX spatial upscaling** (`MTLFXSpatialScaler`) is a reasonable final-pass option; MetalFX *temporal* is not usable here, since an emulated GPU's draw stream gives no true motion vectors or jitter control.
- [ ] 8.16 GPU debugger: PM4 packet trace, per-draw state dump, R700 disassembly + generated MSL side by side, render-target viewer, API trace replay.

**Done when:** a GX2 homebrew sample (textured, lit, animated) renders correctly, and a simple 2D game renders recognisably.

---

## Phase 9 — Audio, input, and the rest of the OS

- [ ] 9.1 **AX / `sndcore2.rpl`**: `AXInitWithParams` (32 kHz / 48 kHz renderer, single vs four-stage pipeline), 96 voices, voice formats ADPCM (0x0) / PCM16 (0xA) / PCM8 (0x19), 32 priority levels, 3 aux buses, device types TV (6ch) / DRC (4ch) / RMT (1ch). Frame callback every **3 ms** (144 samples @48 kHz, 96 @32 kHz).
- [ ] 9.2 Mixer + resampler; output via SDL3 audio to CoreAudio. Ring buffer with drift compensation against the emulated clock.
- [ ] 9.3 `AXFX` effects (reverb, chorus, delay) — stub first, implement if titles need them. `MIX`/`snd_user` layers.
- [ ] 9.4 **Input** — `vpad.rpl` (`VPADRead`: GamePad buttons, dual sticks, touchscreen, accelerometer/gyro, battery) mapped from keyboard/mouse and from `GCController` (macOS Game Controller framework) or SDL3 gamepads.
- [ ] 9.5 `padscore.rpl` — `KPAD` (Wii Remote + extensions) and `WPAD`, plus Wii U Pro Controller.
- [ ] 9.6 Configurable input binding profiles per controller type; a settings UI for them.
- [ ] 9.7 Second-screen (DRC) presentation: separate window or split view, with GamePad touch input mapped to mouse/trackpad.
- [ ] 9.8 Save data: `nn_save` / `FSA` paths writing to a per-title save directory; account stubs via `nn_act`.
- [ ] 9.9 Remaining system libraries as needed: `nn_acp`, `nn_boss`, `nn_olv` (stub), `erreula`, `swkbd` (software keyboard — needs a real UI), `nsysnet`/`nlibcurl` (stub or host-backed, off by default), `zlib125`, `nn_ffl`, `camera`, `nfc`.

**Done when:** a homebrew game is playable with sound and controller input.

---

## Phase 10 — Debugging and developer tooling

**Goal:** make the next 90% of compatibility work tractable. Do not defer this to the end.

- [ ] 10.1 ImGui (Metal backend) debug UI: CPU register/thread views, memory viewer/editor, disassembler with breakpoints and stepping, log filter.
- [ ] 10.2 GDB remote stub so real debuggers can attach to the guest.
- [x] 10.3 HLE call tracing with filters; per-function call counts and time. Plus the standing performance instrumentation from `docs/performance.md` §8: per-subsystem frame breakdown, CPU-stalls-on-GPU, cache hit rates, JIT statistics, and guest time attribution by address.
- [x] 10.4 Deterministic record/replay of input + timing, for reproducing bugs and regression-testing. *(`--record FILE` / `--replay FILE`. Only host-derived values are recorded — everything else is recomputed, so a whole run is a handful of bytes. Divergence is caught two ways: every event carries the instruction count it was produced at, so a replay asking the host something at a different point is told immediately; and periodic checkpoints hash the CPU state and any watched memory ranges, which catches a wrong result in a loop that never calls anything — exactly what 12.7 needs. `add_event` also lets a trace be authored rather than captured, which is how a comparison run is pinned to a fixed date, and how scripted input will work in Phase 9.)*
- [ ] 10.5 Automated compatibility runner: boot a list of titles headless for N frames, hash framebuffers, report regressions.
- [ ] 10.6 Lean on the native tooling rather than rebuilding it: Xcode Metal frame capture and shader debugger (live shader edit-and-reload is genuinely useful for the decompiler), plus the CLI tools `gpucapture`, `gpudebug`, and `metalperftrace`, and Metal System Trace in Instruments.
- [~] 10.7 Crash handler: symbolised host backtrace plus full guest state dump. *(Guest side done: a run that ends badly prints where it stopped with disassembly, registers, a back-chain call stack and the recent HLE calls. The host-side handler remains.)*

---

## Phase 11 — Save states

- [ ] 11.1 Versioned serialisation of CPU cores, memory, HLE object tables, scheduler state.
- [ ] 11.2 GPU state: replay-based restore (re-issue tracked state) rather than trying to snapshot `MTLRenderPipelineState` objects.
- [ ] 11.3 Save/load UI, slots, and automatic save-on-exit.
- [ ] 11.4 Tests: save → load → run N frames → framebuffer hash matches an uninterrupted run.

---

## Phase 12 — JIT recompiler

**Goal:** real-time speed. Deliberately last — a JIT built before the interpreter is trustworthy is a debugging nightmare.

- [ ] 12.1 Basic-block discovery, an SSA-ish IR, and guest→host register allocation.
- [ ] 12.2 arm64 backend. Exploit `rev`/`rev16` for cheap byte-swapping and the large register file to pin the hot GPRs.
- [ ] 12.3 **macOS JIT mechanics**, all of which are required: `mmap(..., MAP_JIT, ...)`; the `com.apple.security.cs.allow-jit` entitlement (the mapping fails outright without it); Hardened Runtime + code signing; `pthread_jit_write_protect_np(0)` to write / `(1)` to execute, which is **per-thread**, not per-page — so a background compile thread can hold write permission while execution threads hold execute. **And: call `sys_icache_invalidate()` after emitting code** — arm64 has no coherent I-cache, and skipping this is the classic silent-corruption bug. (Phase 1.8 already proved the signing path.)
- [ ] 12.4 Block cache with invalidation driven by `icbi`/`ICInvalidateRange` and by writes to code pages.
- [ ] 12.5 Paired-singles and FPU codegen — the hard part, especially `frsqrte`/`fres` estimate tables and non-fused multiply-add semantics. NEON maps to paired singles reasonably well.
- [ ] 12.6 Optimisations in order of payoff (`docs/performance.md` Tier 2): **dead flag elimination first (P6)** — PPC CR/XER updates are mostly unused and this is the largest single PowerPC JIT win — then block linking (**P7**), inline caches plus a return-address stack (**P10**), and constant folding.
- [ ] 12.7 **Differential testing**: run interpreter and JIT in lockstep over the instruction test corpus and over real titles, diffing state every block. A JIT without this will produce bugs no one can find.
- [ ] 12.8 Tiered execution: interpret cold blocks, compile hot ones, with a profiling counter.

**Done when:** JIT output matches the interpreter bit-for-bit on the test corpus, and a title runs ≥5× faster.

---

## Phase 13 — Frontend, packaging, and distribution

- [ ] 13.1 Game library UI: scan directories, read `meta.xml` and icons, launch.
- [ ] 13.2 Settings UI: graphics (resolution scale, MetalFX on/off, aspect, vsync), audio, input, paths, per-title overrides.
- [ ] 13.3 First-run key setup: guide the user to supply their own dumped keys; validate and store them in the macOS Keychain. **Ship nothing.**
- [ ] 13.4 macOS `.app` bundle: `Info.plist`, icon set, arm64 binary. No MoltenVK dylib to bundle — Metal is a system framework, which meaningfully simplifies distribution.
- [ ] 13.5 Codesign with Hardened Runtime + the JIT entitlement, notarize, staple — re-running the pipeline first proved in Phase 1.8.
- [ ] 13.6 Auto-update or a clean DMG flow; crash reporting (opt-in).
- [ ] 13.7 User docs: dumping your own content (link to established guides, host nothing), system requirements (**Apple silicon, macOS 13+**), compatibility list.

---

## Phase 14 — Compatibility grind

This phase has no end, only milestones.

- [ ] 14.1 Establish a compatibility database (title ID → status, notes, required hacks).
- [ ] 14.2 Work the list by popularity: fix the top blocker, retest, repeat.
- [ ] 14.3 **Per-title workarounds behind the Phase 2.5 config layer, never hardcoded in engine code.** Expect this to be real: Cemu's Metal renderer carries a forced position-invariance hack for **twelve** titles (Bayonetta 1/2, Disney Planes, LEGO Star Wars TFA, Mario Kart 8, Minecraft Story Mode, Ninja Gaiden 3 RE, Resident Evil Revelations, Star Fox Zero, Breath of the Wild, Wonderful 101) to avoid Z-fighting when vertex positions are recomputed non-invariantly across passes. Build the mechanism before you need it.
- [ ] 14.4 Accuracy hardening: timing, GPU/CPU sync points, edge-case HLE behaviour.
- [ ] 14.5 Performance profiling and optimisation against real titles, working the backlog in [`docs/performance.md`](docs/performance.md) in profile order — never in list order.

---

## Milestones

| M | Deliverable | Phases |
|---|---|---|
| **M0** | Builds, runs, signs, notarizes; tests green | 1–2 |
| **M1** | RPX loads and disassembles correctly | 3–5 |
| **M2** | **Homebrew hello-world prints via `OSReport`** | 6 |
| **M3** | `OSScreen` homebrew renders via Metal | 7 |
| **M4** | GX2 homebrew renders correctly | 8 |
| **M5** | Homebrew game playable with audio + input | 9 |
| **M6** | Debuggable and save-stateable | 10–11 |
| **M7** | Real-time speed via JIT | 12 |
| **M8** | Distributable signed `.app` | 13 |
| **M9** | Commercial titles booting | 14 |

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Metal has no geometry shaders or stream-out | High | Mesh-shader GS path (8.8) and `rasterizationEnabled=false` stream-out (8.9), designed in from Phase 7 rather than bolted on |
| Mesh shaders require Apple7 — no Intel Mac support | Medium | Accepted: arm64-only (ADR-002). Intel is EOL in macOS 27 anyway. A compute-based GS path stays possible if ever needed |
| Metal gaps needing shader-side emulation (provoking vertex, logic ops, primitive restart, border colour, cube seams) | Medium | Catalogued in the Phase 7 table; the shader analysis pass (8.7) flags which shaders need which |
| Latte tiling/swizzle is under-documented | High | Build a tiling test harness early; cross-check against known-good decoders |
| Shader decompiler complexity | High | Incremental: get simple shaders right, add CF/loop/branch support driven by real failures |
| HLE surface is enormous | Medium | Loud logging on unimplemented imports; implement strictly on demand |
| Espresso `lwarx`/`stwcx.` hardware bug | Medium | Implement documented behaviour; add a compat switch if a title depends on the flaw |
| Metal 4 deprecating the legacy tessellation path | Low | Target Metal 3 as baseline, treat Metal 4 as additive; verify before relying on tessellation |
| `MTLBinaryArchive` silently missing across OS versions | Low | Always pair it with our own on-disk cache (8.10) |
| Scope: this is a multi-year project | Certain | Phases 1–6 are genuinely achievable solo; ship at every milestone |

---

## Immediate next actions

1. Phase 1.4 — platform layer: `CAMetalLayer` window plus the `metal-cpp` RAII wrapper. Get a cleared frame on screen. (Needs a Mac; everything below the renderer builds anywhere.)
2. Phase 1.8 — run the codesign + notarize round-trip once, with the JIT entitlement, on the empty app, so Phase 12 does not meet it for the first time at the end.
3. Phase 7.1 and 7.5 — the `GPUBackend` interface and the resource-binding decision (argument buffers tier 2 versus reserved slots), recorded as an ADR before Phase 8 starts depending on it.
4. Message queues (`OSMessageQueue`), which is what the other half of the asynchronous FS API delivers through, and `proc_ui` (6.11) — between them, most of what a title touches before it draws anything.
5. Run an RPX built with devkitPro + `wut` through the loader. The current fixtures validate the parser against this project's reading of the format, not the format itself.
6. Verify the `fres`/`frsqrte` estimate tables against the 750CL manual. They are transcribed hardware data; the accuracy tests bound them but cannot confirm the exact values.
7. Decide what to do about MSR[FP]: nothing consults it, so there is no floating-point-unavailable exception, and a guest doing lazy FPU context switching would silently read stale registers.
