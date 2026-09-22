# Affogato documentation

These notes organize the Wii U research gathered for **Affogato**, an experimental Wii U emulator.

The documentation is arranged around **development phases**. The goal is to learn only enough of each subsystem to unblock the next implementation milestone instead of trying to document the entire console before coding.

## Affogato's practical direction

Affogato does not need to reproduce every internal detail of the physical Wii U. The main rule is:

> Reproduce the behavior visible to Wii U software; use simpler host-side implementations behind that boundary where possible.

Current direction:

- **Espresso / PowerPC:** emulate the guest-visible ISA and architectural state.
- **Cafe OS:** prefer HLE for system/library services.
- **IOSU:** prefer HLE of the PPC-visible service/IPC boundary; do not emulate Starbuck initially.
- **Memory:** begin with a simple guest-memory abstraction; add MMU behavior only when required.
- **GX2 / Latte:** eventually translate guest graphics behavior to a host API such as Vulkan rather than reproducing the physical GPU pipeline.
- **Filesystem / input / audio / networking:** use host services behind Wii U-compatible interfaces where practical.
- **Multicore:** begin with a single Espresso core and add multicore semantics after the basic CPU is correct.
- **JIT:** interpreter first, JIT later. Keep the interpreter as a correctness/reference backend.

## Research phases

1. [Wii U architecture](research/01_wiiu_architecture.md)
2. [Espresso CPU](research/02_espresso_cpu.md)
3. [PowerPC encoding](research/03_powerpc_encoding.md)
4. [PowerPC execution](research/04_powerpc_execution.md)
5. [Memory](research/05_memory.md)
6. [Exceptions and privilege](research/06_exceptions_and_privilege.md)
7. [CPU testing](research/07_cpu_testing.md)
8. [Executable formats](research/08_executable_formats.md)
9. [Cafe OS HLE](research/09_cafe_os_hle.md)
10. [IOSU HLE](research/10_iosu_hle.md)
11. [Graphics](research/11_graphics.md)
12. [Multicore](research/12_multicore.md)
13. [Other subsystems](research/13_other_subsystems.md)

Also see:

- [Implementation strategy](research/00_implementation_strategy.md)
- [Research questions](research/research_questions.md)
- [References](research/references.md)

## Current research cutoff

The immediate implementation target is the **minimal architectural state for one Espresso core**.

Before writing substantial CPU code, the important knowledge is:

- what Espresso is,
- the relevant PowerPC programming model,
- the guest-visible registers,
- instruction size/alignment and byte order,
- how CR/XER/LR/CTR work,
- which state is needed immediately versus later.

Later phase documents are intentionally more checklist-like. Expand them when Affogato reaches those milestones rather than filling them with speculative details now.
