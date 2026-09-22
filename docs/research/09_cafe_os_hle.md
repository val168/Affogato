# 09 — Cafe OS HLE

**Status:** implement only after a real executable reaches system-library calls.

## Goal

Provide the Wii U application-facing behavior of Cafe OS without initially running the complete real Cafe OS implementation.

## Real system

Cafe OS runs on Espresso in Wii U mode.

WiiUBrew describes it as including:

- a PPC kernel,
- executable loader,
- system libraries,
- process isolation,
- memory management,
- interrupt handling,
- IPC with IOSU.

Applications dynamically link against RPL system libraries.

Examples include:

- `coreinit.rpl`
- `gx2.rpl`
- audio/video/input-related libraries.

## Affogato HLE boundary

When a guest application imports a known system function, Affogato can dispatch it to a native implementation.

```text
guest PPC code
    ->
call imported Cafe function
    ->
Affogato HLE dispatcher
    ->
native C++ implementation
    ->
write return values / guest memory
    ->
resume guest PPC code
```

## What must be understood per HLE function

Record:

- library/module name.
- exported function name or symbol identity.
- PowerPC ABI/calling convention.
- arguments.
- pointer/structure layouts.
- return value.
- error codes.
- guest-memory side effects.
- synchronization/blocking semantics.
- whether the function eventually depends on IOSU.

## What does not need to be reproduced

Unless compatibility requires it:

- the internal Cafe implementation.
- identical kernel data structures.
- identical scheduling internals.
- the real executable loader internals.

Accuracy is required at the guest-visible boundary, not behind it.

## First HLE functions

Do not pre-implement hundreds of functions.

Run a small homebrew program, log unresolved imports/calls, and implement the minimum set needed for the next observable milestone.

## Sources

- WiiUBrew — Cafe OS: https://wiiubrew.org/wiki/Cafe_OS
- WiiUBrew — Cafe OS Kernel: https://wiiubrew.org/wiki/Cafe_OS_Kernel
- WiiUBrew — RPL: https://wiiubrew.org/wiki/RPL
