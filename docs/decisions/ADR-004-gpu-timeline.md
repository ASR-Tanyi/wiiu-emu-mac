# ADR-004 — The GPU runs on the guest's clock

**Status:** Accepted · 2026-10-15

## Context

Phase 8.2 says GX2 appends PM4 to a ring and "a separate GPU thread"
consumes it, because on hardware the GPU runs alongside the CPU and titles
depend on that: a title submits a frame, carries on, and later asks whether
the GPU has finished — through a timestamp it polls, a timestamp it waits on,
or memory the GPU writes when it gets there (`EVENT_WRITE_EOP`, `MEM_WRITE`).

The obvious implementation is a host thread reading the ring as fast as the
host allows. It makes every one of those answers depend on host timing: how
far the GPU thread has got when the guest looks is a race, so the value a
title reads from a GPU-written timestamp differs from run to run. That breaks
the two things the CPU side was built deterministic for — record/replay
(10.4), where a run must reproduce exactly, and the JIT lockstep (12.7),
where two runs must agree word for word. The CPU side already solved the same
problem for guest threads: they interleave deterministically on one host
thread, and real parallelism goes behind the interface when it is worth
measuring (6.3).

## Decision

**The GPU has a timeline in guest time, and everything a guest can observe
happens on it.** A submission is queued with the timebase tick it was
submitted at and retires a fixed latency later. Retiring a submission runs
its packets — register writes, draws, nested buffers — and applies their
guest-visible effects: `MEM_WRITE` and `EVENT_WRITE_EOP` land in guest memory,
the retired timestamp advances, interrupts are raised. Retirement happens at
deterministic points: when the scheduler's clock reaches the deadline (it is
one more entry in the timer list alarms and asynchronous FS use), or when a
guest thread waits for it.

A packet's "GPU clock" (`EWP_DATA_CLOCK`, `MW_WRITE_CLOCK`) is the timebase at
retirement, not the host's clock.

**The host GPU thread is an implementation detail behind this.** When the
translator (8.11) has Metal work worth overlapping, encoding moves to a host
thread; guest-visible effects still apply at the retirement point. The one
place host progress becomes guest-visible is a readback (a copy from a GPU
surface into guest memory): retiring a submission that contains one waits for
the host GPU first. That wait is one the guest asked for — it is waiting for
the timestamp — so it is the kind of stall **P4** allows.

## Consequences

**Good.**

- Record/replay and JIT lockstep keep working with a GPU in the picture:
  nothing the guest reads depends on the host.
- Waiting for the GPU costs nothing in host time. A thread that waits for a
  timestamp parks on a deadline, and the idle skip (**P2**) jumps the clock
  straight there.
- The command processor is tested the way the rest of the HLE is: run a
  stream, look at memory, no threads.

**Bad, and worth saying.**

- **The latency is a model, not a measurement.** Real GPU work takes as long
  as it takes; here it takes a constant. A title that measures its own GPU
  time sees the constant. A title whose behaviour depends on the GPU being
  *slow* — a frame-pacing heuristic, say — may behave differently. The
  constant is a setting, so per-title tuning (14.3) can reach it.
- **Packets run late.** Draws are decoded at retirement rather than as soon as
  they are submitted, so host GPU work starts a latency later than it could.
  That is a throughput cost the host thread in 8.11 can win back by running
  ahead on the *host* side, as long as nothing guest-visible moves with it.
- It is one more thing that is not how hardware works, and has to be
  explained to anyone comparing this with decaf or Cemu, both of which use a
  host thread and accept the non-determinism.

## What would make us revisit

- A title that needs the GPU's apparent speed to vary with the work, found
  in practice rather than imagined.
- Profiling that shows retirement-time decoding is the bottleneck even with a
  host encoding thread — at which point decoding, too, can run ahead on the
  host, with only the guest-visible effects pinned to retirement.
