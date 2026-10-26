# Architecture

How the pieces fit together. Filled in as phases land; sections marked
*(not built)* describe intent, not code.

## Shape

```
                 frontend  (window, UI, settings)
                     |
   +-----------+-----+------+-----------+
   |           |            |           |
  cpu        memory        gpu        audio/input
   |           |            |
   +----> hle (Cafe OS libraries) <----+
                 |
              loader  (RPX/RPL, disc, filesystem)
                 |
              common  (types, endian, logging)
```

`common` depends on nothing. Everything depends on `common`. `hle` is the hub:
it is what guest code actually calls, and it drives the other subsystems.

## Principles

**Big-endian guest, little-endian host.** Every guest memory access goes
through a byte-swapping accessor. This is not retrofittable, so it is in place
from Phase 2 before anything reads guest memory.

**HLE-first.** Cafe OS libraries are reimplemented at the function level.
There is no LLE of the ARM security processor (Starbuck/IOSU); its IPC is
intercepted and serviced natively.

**Correctness, then speed.** The interpreter is the reference implementation
and stays in the tree forever — Phase 12's JIT is validated by running in
lockstep against it. Performance work is tracked separately in
[`performance.md`](performance.md), and five of those items are design
constraints rather than later optimisations.

**Everything is observable.** Per-subsystem log channels, an in-memory ring
buffer for the debug UI, cache hit-rate counters, and a loud warning on every
unimplemented HLE import. A cache that silently misses and a stub that silently
returns zero are the two failure modes that cost the most debugging time.

**Deterministic by construction.** Two runs of the same title differ only
where the emulator consults the host, and there are very few such places: the
guest timebase advances by instructions rather than by wall-clock time, the
scheduler interleaves guest threads in a fixed order, and the interpreter has
no other source of variation. So the real-time clock is read once, at the
first `OSGetTime`, and everything else is derived from it. Recording just
those host values is enough to replay a whole run (`--record` / `--replay`,
Phase 10.4), and the same property is what lets Phase 12's JIT be compared
against the interpreter in lockstep. Anything that would introduce
nondeterminism — real thread parallelism, a host-clock-driven timebase —
goes behind an interface that can be turned off, not into the default path.

## Subsystems

### common

Types (`u32`, `ppcaddr_t`, …), bit and alignment helpers, and logging.

Logging is one channel per subsystem, each with an independent runtime level,
behind a compile-time floor. The level check sits in the macro rather than the
function, so a gated record evaluates none of its arguments — formatting a
disassembled instruction costs far more than the call, and in the CPU and GPU
hot paths that difference is the point.

*Phase 2.1 adds `be<T>`, the byte-swapping storage type.*

### memory

A 4 GiB host reservation with the Cafe OS virtual map laid into it, so guest
virtual → host pointer is one add. Fastmem (P3) is a design constraint here:
translation must reduce to `base + zero_extend32(addr)` with no branch, and
MMIO must be caught by a fault handler rather than a per-access check.

The cache model is real, not decorative: the guest CPU is coherent with the
other two cores but *not* with the GPU, so `DCFlushRange` and friends have to
actually invalidate GPU-visible ranges.

Address translation (Phase 3.4) sits in the CPU rather than here, because BATs,
segment registers and SDR1 are CPU registers. With MSR[IR]/[DR] clear — where
the HLE loader leaves the core — the effective address is used unchanged and
translation costs one test. With them set, the BATs and the hashed page table
are walked for real and cached in a TLB. One convention is worth knowing: this
address space is keyed by *virtual* address, since MEM1 is committed where Cafe
OS shows it rather than at physical zero, so a real address produced by a
translation is mapped back through the region table before it is used.

### hle

Cafe OS libraries reimplemented as native functions, reached through one-word
`sc` trampolines that imports resolve to. Threads interleave deterministically
on one host thread, which is what makes record/replay and the eventual JIT
lockstep possible.

Two things are worth knowing before reading it. A handler that blocks switches
the context out from under itself, so anything that can block sets its return
value first — and a thread woken from a blocking call resumes *after* the call,
never inside it, which is why unlocking a mutex has to name the next owner
rather than leaving it free, and why a message queue hands a message to a
waiting receiver rather than queueing it for the receiver to come back for.
And the OS calls back into the title — alarms, asynchronous completion — by
running a guest function on its own scheduler thread, with a stack from a
recycled pool, or by posting to a queue the title is waiting on.

There is a third way into guest code, used by ProcUI: rather than running a
callback on a thread of its own, the handler redirects the thread it is already
on — saving where the title was going, pointing it at the callback with the
return address set to an `sc` trampoline of ours, and putting it back when the
chain ends. That is how a callback runs on the caller's own stack and is
finished before the call returns, which is what the title expects. A title's
start is the same trick before it has begun: an RPX that exports
`__preinit_user` — every one built with wut — is sent through it first, with
the base heaps' handles on a frame of its own and a trampoline as the return
address, and the trampoline applies what it left there and sends the thread on
to the entry point.

Where a structure is the title's to see, it lives in guest memory, as Cafe OS
keeps it: a frame heap's header and recorded states, a spin lock's holder and
count, OSScreen's frames. Where nothing in the title can see it, it is kept on
the host: an expanded heap's block list, a mutex's waiters, a thread's sixteen
specific values (the main thread has no guest `OSThread` to hold them). What
coreinit allocates for itself comes from a system heap of its own, since a
title may take all of MEM2 for its own allocator, and wut's does.

Three properties of the scheduler are load-bearing beyond it. Waiters are
ordered by priority and then by how long they have waited, because Cafe OS
thread queues are FIFO within a level and a title can observe the difference
directly in the order its messages come back. The timebase is machine state
that survives a context switch, not part of the register file a switch
restores: it only goes forward, and every deadline is measured against it. And
a thread has a generation as well as a handle, because a handle is a guest
`OSThread*` that a title reuses — so anything remembering a thread across calls
can tell that the thread it meant is gone and a different one now answers to
its name.

Waiting is not emulated. Every deadline the machine is waiting for, whether a
sleeping thread or an alarm, lives in one list, so when nothing can run the
clock jumps to the next one; and a thread found spinning on a flag hands the
core back immediately rather than at the end of its timeslice (**P2**).

Everything a title cannot do for itself goes through IOSU, and none of IOSU is
emulated. The interception is at the `IOS_*` functions rather than the IPC
doorbell: `ios.hpp` holds a table from `/dev/...` to a native device, so a
request never becomes a packet and Starbuck never has to exist. A device nobody
has written is refused with its path in the log rather than given a handle that
answers everything with zero, which is the missing-function policy one level
down. The asynchronous forms do the work inside the call and schedule only the
reply, through the same timer list, so a title that starts a request and then
sets a flag sees them in that order.

### cpu

An Espresso interpreter: three PowerPC 750CL-derived cores, big-endian, with
paired singles. The decoder, disassembler and interpreter come from one
instruction table so they cannot drift apart, and the JIT (Phase 12) will
dispatch off the same identifiers.

Faults have two modes. `Report` surfaces them to the caller; `Deliver` does
what the hardware does and jumps to the vector. Report is the default because
until Cafe OS installs handlers, delivering an exception means jumping into
zeroed memory, which turns a clear diagnostic into an illegal-instruction loop
somewhere else.

**The exception vectors are not in the Cafe OS memory map.** Nothing is mapped
at low addresses, so the loader (Phase 6.2) has to place a vector page before
Deliver mode is usable at all. This surfaced from a test rather than from
reading the map, and it is easy to miss because Report mode works fine without
it.

### loader *(not built — Phase 5)*

RPX/RPL is ELF32 with custom section types, PE-style import/export tables and
per-section zlib compression. Above it sits a VFS that presents `/vol/content`
and friends identically whether the source was a decrypted folder, a WUP
install or a disc image.

### hle

A registry from `library.symbol` to a native handler, with PPC ABI argument
marshalling generated by templates so a handler is an ordinary C++ function.
Unimplemented imports log their library, name and caller, then return zero.

Two ABI details are worth stating because getting either wrong produces
plausible wrong values rather than a fault: a 64-bit argument occupies a
register pair starting at an **odd** register (r3:r4, r5:r6, r7:r8, r9:r10),
so `f(int, long long)` skips r4; and floating-point arguments take f1–f8
without consuming a GPR slot. Both are checked against what
`powerpc-linux-gnu-gcc -m32` actually emits.

`recording.hpp` holds the determinism layer: every host-derived value passes
through it, and it either appends it to a trace or substitutes the recorded
one. It also hashes machine state at intervals, so a replay that stops
matching is reported at the first point it could be known rather than wherever
the consequences surface.

### gpu *(Phase 7 built — interface, Metal backend, presenter; Phase 8 next)*

One rule holds this subsystem together: **no Metal type crosses
`gpu/backend.hpp`**. Everything above it — and Phase 8's PM4 translator is
all of it — is written against an interface of plain values and opaque
integer handles, so a second backend would be a new file rather than a new
emulator, and so the translator can be tested on a machine with no GPU at
all. `create_null_backend()` is that machine: a backend that records what it
was told, validates everything the real one validates, and draws nothing.

Handles are integers rather than pointers because two later phases want them
to be: a save state (11.2) and an API trace (8.16) can both write one down
and read it back, and a stale one is then a lookup failure instead of a
crash. Zero is reserved, so a zeroed descriptor refers to nothing.

Two decisions are encoded in the types rather than written down somewhere
hopeful. Load and store actions have no default: `Unspecified` is the zero
value and every backend refuses it, because on a tile-based GPU the wrong
action is still *correct* — the image is right, it is merely moving the whole
framebuffer through memory twice — and a mistake that produces correct output
never arrives as a bug report (**P20**). And the binding model is absent from
the interface entirely: callers bind by stage and index, and whether that
reaches the shader through a slot or a descriptor in an argument buffer is
the backend's business ([ADR-003](decisions/ADR-003-argument-buffers.md)).

Until the argument buffers arrive with the MSL emitter (8.6, 8.12), binding is
direct: index `n` is `[[buffer(n)]]`, `[[texture(n)]]` or `[[sampler(n)]]` in
that stage, inside the render pass it was made in. Vertices are never
described to Metal — the shader fetches them from a bound buffer by
`[[vertex_id]]`, because Latte's vertex fetch has no alignment rules and
Metal's has. And shaders compile with fast math off, the GPU-side twin of
`-ffp-contract=off`: Latte's IEEE and DX9 multiplies differ exactly where
fast math takes liberties.

Memory comes from hazard-tracked `MTLHeap`s rather than resource by resource,
so that ADR-003's residency is a few `useHeap` calls an encoder rather than a
list of every resource a draw might touch; guest RAM that the GPU reads can be
wrapped in place (`import_host_memory`, **P22**) rather than copied. No more
than `frames_in_flight` frames are ever between `begin_frame` and the GPU
finishing them, which is what makes a ring of that many per-frame buffers
safe to write from the CPU — the presenter's screen textures are one.

The presenter (`gpu/presenter.hpp`) is the last step to the window: the
newest image of each of the Wii U's two screens, copied when it is handed
over (scan-out is a snapshot), fitted into the drawable with its aspect kept,
TV above GamePad when both are shown. OSScreen (6.8) feeds it today; GX2's
scan-out will feed it the same way.

Latte's own formats — PM4 streams, the register file, its surface layout
(tiling, through AMD's addrlib behind one header —
[ADR-005](decisions/ADR-005-addrlib.md)), and later its shader microcode —
live in `wiiu::latte`, the translator, which turns them into calls on this
interface. The GPU there runs on the guest's
clock ([ADR-004](decisions/ADR-004-gpu-timeline.md)): a submission retires a
fixed latency after it was made, in order, and only then do its packets run
and their writes reach guest memory — so a title polling a GPU timestamp reads
the same value on every run, and a thread waiting for one parks on a deadline
the idle skip jumps to.

GX2 itself is HLE (`hle/gx2.hpp`): its calls append PM4 to a command buffer
the emulator keeps host-side, or to a display list in the title's memory, and
`GX2Flush` hands the buffer to that timeline. The timeline sits on the
scheduler's timer list like an alarm, which is what retires it; nothing reads
it from another thread. Vsync is another timer on the same list — 59.94 Hz of
guest timebase, not the host display's refresh — and a swap flips at the first
vsync by which the submission carrying it has retired, so when a frame flips
is decided by the title's own calls and nothing else. A host thread that encodes Metal work behind the
timeline is ADR-004's later step, for when there is translation worth
overlapping (8.11). R700 shader
microcode is lowered to an internal IR and emitted as MSL — not through
SPIR-V, whose only route to MSL has no mesh shader support, which would kill
the geometry path outright. Geometry shaders and Latte's RECTS become Metal
object+mesh pipelines; stream-out becomes a rasterization-disabled vertex
pass.

See [ADR-001](decisions/ADR-001-metal-over-vulkan.md) for why the backend is
Metal, and the Phase 7 table in `plan.md` for the Metal gaps that need
shader-side emulation.

### frontend / platform

`metal-cpp` excludes MetalKit and AppKit, so there is no `MTKView`: the
window and its `CAMetalLayer` are made by hand in the project's one
Objective-C++ file, and handed to the backend as a `void*` and a size.
Nothing between the two learns what it is.

Two things about that file are worth knowing before reading it. ARC is off,
so Objective-C ownership is manual — objects from `alloc` are owned and
released once, everything else is autoreleased and must be retained to
outlive its pool — and `metal-cpp` has the same rules on the C++ side, which
is what `gpu/metal/raii.hpp` exists to make automatic. And points are not
pixels: the drawable size is the view's bounds times the window's backing
scale, rechecked every poll, because a window dragged between displays
changes scale without changing size.

`frontend/display.hpp` puts the window, the backend and the presenter
together, and takes them apart in the reverse order. OSScreen's flips reach
it from inside the guest's own call, so a title runs on the main thread and
presents from there; closing the window halts the core from inside the next
flip, which the run loop reports as a halt and the frontend recognises as the
user's.

## Conventions

- `wiiu::` for everything; one namespace per subsystem below it.
- `snake_case` functions and variables, `CamelCase` types, trailing `_` on
  private members. Enforced by `.clang-tidy`.
- Targets are `wiiu_<name>` with a `wiiu::<name>` alias, built by
  `wiiu_add_library()` so the warning set and standard are applied uniformly.
- Warnings are errors. A warning that is genuinely wrong gets a suppression
  with a comment, not a relaxed global setting.
