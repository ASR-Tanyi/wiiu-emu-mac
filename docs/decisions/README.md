# Architecture decision records

One file per decision that would otherwise be re-litigated in six months.

Format: context, the decision, consequences (including the bad ones), and what
would make us revisit. A decision that cannot name its own downside has not
been thought through.

| # | Decision | Status |
|---|---|---|
| [001](ADR-001-metal-over-vulkan.md) | Native Metal instead of Vulkan via MoltenVK | Accepted |
| [002](ADR-002-apple-silicon-only.md) | Apple silicon only; no Intel Mac support | Accepted |
| [003](ADR-003-argument-buffers.md) | Bindless resource binding via argument buffers (Tier 2) | Accepted |
| [004](ADR-004-gpu-timeline.md) | The GPU runs on the guest's clock; a host GPU thread is an implementation detail | Accepted |
