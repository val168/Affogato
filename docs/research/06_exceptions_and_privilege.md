# 06 — Exceptions and privileged state

**Status:** defer until basic user-mode integer programs execute.

## Goal

Add enough exception and privileged behavior for software that genuinely relies on it.

## Research topics

- illegal/unsupported instruction behavior.
- data and instruction access faults.
- alignment exceptions.
- system calls.
- MSR.
- SRR0 / SRR1.
- DAR / DSISR.
- decrementer/time-base interaction.
- privilege checks.
- exception vectors.
- interrupt entry/return.
- address translation state.

WiiUBrew documents Espresso MSR fields and many relevant SPRs. Cafe OS kernel documentation also exposes how Wii U system calls enter the PPC kernel.

## Practical Affogato approach

Early synthetic programs do not need a full exception environment.

Initially:

- unsupported instructions can stop execution with a structured emulator error.
- invalid memory can raise an emulator-side fault.
- tests should clearly expose why execution stopped.

Later, real Cafe OS/homebrew compatibility may require architecturally correct exception behavior.

## Avoid premature implementation

Do not add every privileged SPR to `CpuState` before any code uses it.

Do not reproduce real interrupt-controller timing unless software-visible behavior demands it.

## Sources

- WiiUBrew — Espresso: https://wiiubrew.org/wiki/Espresso
- WiiUBrew — Cafe OS Kernel: https://wiiubrew.org/wiki/Cafe_OS_Kernel
- NXP/Motorola — PowerPC Programming Environments Manual.
