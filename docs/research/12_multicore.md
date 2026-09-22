# 12 — Espresso multicore

**Status:** postpone until the single-core interpreter is correct.

## Known fact

Espresso has three documented PowerPC cores, each with one hardware thread.

## Why not start with three cores

Concurrency makes CPU bugs much harder to isolate.

A correct single-core interpreter provides:

- instruction semantics.
- memory access behavior.
- tests.
- a reference backend.

Only then should multicore scheduling and synchronization be introduced.

## Research topics

- which architectural state is per-core.
- shared memory.
- interrupt routing.
- Cafe OS scheduling model.
- PowerPC memory ordering.
- barriers.
- atomics.
- reservation semantics.
- `lwarx` / `stwcx.`.
- cache-management operations that are software-visible.
- inter-core communication/mailboxes if software accesses them directly.

## Practical implementation direction

Do not assume "three Espresso cores" means "three uncontrolled host threads."

A controlled scheduler with deterministic execution slices may be useful early. The actual design should follow software-visible synchronization requirements rather than physical imitation.

## Sources

- WiiUBrew — Wii U Hardware: https://wiiubrew.org/wiki/Wii_U_console
- WiiUBrew — Processor interface: https://wiiubrew.org/wiki/Hardware/Processor_interface
- NXP/Motorola — PowerPC Programming Environments Manual.
