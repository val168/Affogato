# 04 — PowerPC execution semantics

**Status:** research after basic decoding exists.

## Goal

Implement a small, correct interpreter for guest-visible PowerPC behavior.

The interpreter does not emulate Espresso's physical pipeline. It applies the architectural effects of each instruction to `CpuState` and memory.

## Initial instruction families

Start with a deliberately small integer subset.

Useful early categories:

- immediate integer arithmetic.
- basic logical operations.
- comparisons.
- simple branches.
- eventually loads/stores once the memory abstraction exists.

A likely progression is:

```text
addi
addis
ori
oris
simple compare
unconditional branch
basic conditional branch
loads/stores
```

The exact sequence should be chosen based on testability and dependencies.

## Semantics to understand carefully

### rA == 0 special cases

Some immediate-form instructions treat an rA field of zero specially rather than simply reading GPR0. Verify this per instruction instead of adding a global "r0 is always zero" rule.

### CR updates

Record-form operations and comparisons can modify CR fields. Implement exactly the architected effects.

### XER

Carry, overflow, and summary-overflow behavior must be correct when an instruction specifies them.

### Branches

Research:

- relative versus absolute targets.
- AA.
- LK and LR updates.
- CR conditions.
- CTR decrement/test behavior.
- branch-to-LR and branch-to-CTR forms.

### Instruction address progression

The interpreter needs explicit semantics for current instruction address, fall-through, branch targets, and later exceptions. Avoid spreading ad-hoc `+4` logic across instruction handlers.

## Interpreter rule

```text
decode operands
    ->
compute architectural result
    ->
update only specified architectural state
    ->
select next instruction address
```

## Testing rule

Every new instruction should receive small deterministic tests. Do not implement hundreds of instructions and test them only through a game.

## What can be ignored initially

- pipeline latency.
- branch prediction.
- out-of-order/speculative execution.
- cache timing.
- performance-monitor side effects unless software needs them.

## Sources

- NXP/Motorola — PowerPC Programming Environments Manual: https://www.nxp.com/docs/en/user-guide/MPCFPE_AD_R1.pdf
- WiiUBrew — Espresso: https://wiiubrew.org/wiki/Espresso
