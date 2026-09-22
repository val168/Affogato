# 07 — CPU testing and synthetic programs

**Status:** introduce alongside the first CPU implementation.

## Goal

Prove CPU behavior independently of real Wii U executables.

## Test layers

### CpuState tests

Examples:

- deterministic construction/reset.
- GPR access.
- CR field/bit helpers.
- XER helper behavior.

### Decoder tests

Input:

```text
known 32-bit instruction word
```

Output:

```text
expected opcode and operands
```

### Instruction semantic tests

Input:

```text
initial CpuState
decoded instruction
```

Output:

```text
expected CpuState
```

### Synthetic program tests

Place a tiny sequence of PowerPC instructions into guest memory and execute:

```text
fetch
  ->
decode
  ->
execute
  ->
advance/branch
  ->
repeat
```

Then inspect registers and memory.

## Useful termination strategies

For early synthetic programs, use an emulator-controlled stop condition such as:

- execute N instructions,
- stop at a chosen guest address,
- treat a deliberately unsupported/sentinel instruction as test termination.

Do not require a full Cafe OS process-exit implementation just to test arithmetic.

## Testing philosophy

A failing game is a bad unit test.

Keep small tests that isolate decoding, arithmetic, branches, CR/XER changes, endian behavior, and loads/stores. These tests later become the reference suite for the JIT.

## Build integration

Tests should be integrated through CMake/CTest. A test framework such as Catch2 can be introduced once justified by the amount of CPU testing.
