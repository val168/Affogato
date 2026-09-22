# 00 — Practical implementation strategy

## Goal

Affogato should emulate the **Wii U software-visible environment**, not reproduce the console's physical implementation part-for-part.

A game already contains its own gameplay logic, engine, assets, renderer logic, and PowerPC machine code. Affogato's job is to provide enough of the Wii U environment that this software can execute correctly.

## Three implementation styles

### ISA / architectural emulation

Used when the guest program itself consists of machine instructions that must execute.

Primary example: Espresso PowerPC execution.

Affogato initially interprets instructions and modifies architectural state. Later, a JIT can translate hot PowerPC code to native host instructions.

### HLE

Used when software talks to a relatively clean service/API boundary.

Primary examples:

- Cafe OS libraries.
- IOSU-facing services.
- filesystem services.
- timing and many OS facilities.

Affogato can implement the observable function/service behavior directly in host C++ without executing the original service implementation.

### Translation

Used when a guest subsystem maps naturally onto a powerful host subsystem.

Primary example: GX2 / Latte graphics translated to Vulkan.

The host GPU should perform rasterization, sampling, blending, depth tests, and shader execution after Affogato has translated guest state correctly.

## What we intentionally do not emulate first

The initial emulator does not need:

- Espresso pipeline stages.
- branch predictor behavior.
- transistor-level or cycle-accurate CPU execution.
- complete cache simulation.
- the full boot chain.
- Starbuck ARM instruction emulation.
- real IOSU firmware execution.
- Nintendo's actual executable loader implementation.
- physical storage controllers.
- physical controller wireless protocols.
- a cycle-accurate Latte GPU.
- three concurrently running Espresso cores from day one.

These can be revisited only if software-visible compatibility requires them.

## Planned boundaries

```text
Wii U application / homebrew
          |
          +---- PowerPC code ----> Espresso interpreter -> later JIT
          |
          +---- Cafe imports ----> Cafe OS HLE
          |                         |
          |                         +---- IOSU-facing requests -> IOSU HLE
          |                         +---- host filesystem / timers / etc.
          |
          +---- GX2 calls/state --> graphics translation --> Vulkan --> host GPU
```

## Development order

1. One-core Espresso architectural state.
2. PowerPC decoder.
3. Small instruction interpreter.
4. Guest memory abstraction.
5. Synthetic PowerPC programs.
6. ELF understanding/loading.
7. RPX/RPL loading.
8. Small Wii U homebrew.
9. Cafe OS HLE as demanded by software.
10. IOSU service HLE as demanded by software.
11. Basic graphics output.
12. GX2/Latte translation.
13. Multicore semantics.
14. JIT.
15. Remaining subsystems.
