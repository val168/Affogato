# References

These are the main public references currently used for Affogato research.

The list is intentionally biased toward architecture manuals and WiiUBrew/WiiBrew reverse-engineering documentation. Existing emulator codebases are useful as design references, but their code should not be copied blindly and their licenses must be respected.

## Wii U hardware / operating system

### WiiUBrew — Hardware overview

https://wiiubrew.org/wiki/Wii_U_console

Useful for Espresso core count/clock, Latte/GX2/Starbuck overview, MEM1/MEM2 headline sizes, and major hardware components.

### WiiUBrew — Espresso

https://wiiubrew.org/wiki/Espresso

Useful for Espresso identity, documented MSR layout, and documented SPR indices.

### WiiUBrew — Latte

https://wiiubrew.org/wiki/Hardware/Latte

Useful for the relationship between Latte, Starbuck, GX, and GX2.

### WiiUBrew — Starbuck

https://wiiubrew.org/wiki/Hardware/Starbuck

Useful for Starbuck's role and IOSU/boot relationship.

### WiiUBrew — Boot Process

https://wiiubrew.org/wiki/Boot_Process

Useful for the real boot chain, IOSU loading Cafe OS, and Starbuck-to-Espresso startup relationship.

### WiiUBrew — Cafe OS

https://wiiubrew.org/wiki/Cafe_OS

Useful for PPC OS architecture, loader, processes, system libraries, and the IOSU IPC relationship.

### WiiUBrew — Cafe OS Kernel

https://wiiubrew.org/wiki/Cafe_OS_Kernel

Useful for kernel role, syscalls, process isolation, and PPC privilege boundaries.

### WiiUBrew — IOS

https://wiiubrew.org/wiki/IOS

Useful for IOSU kernel/resource managers, `/dev` service model, and open/close/read/write/ioctl/ioctlv concepts.

### WiiUBrew — Latte IPC

https://wiiubrew.org/wiki/Hardware/IPC

Useful for PPC/ARM hardware signalling, IPC register layout, and per-core IPC regions.

### WiiUBrew — Memory Map

https://wiiubrew.org/wiki/Memory_Map

Useful for MEM0/MEM1/MEM2 physical map, application/kernel/IOS regions, and MMIO ranges.

## Executable formats

### WiiUBrew — RPL

https://wiiubrew.org/wiki/RPL

Useful for RPX/RPL as modified ELF, compressed sections, section-header loading, and import/export sections.

## Graphics

### WiiUBrew — GX2 hardware

https://wiiubrew.org/wiki/Hardware/GX2

Useful for GX2 hardware identity, Radeon-family relationship, and MMIO region.

### WiiUBrew — gx2.rpl

https://wiiubrew.org/wiki/Gx2.rpl

Useful for guest-facing GX2 library functions.

## PowerPC

### NXP / Motorola — PowerPC Microprocessor Family: The Programming Environments

https://www.nxp.com/docs/en/user-guide/MPCFPE_AD_R1.pdf

Use as the core generic PowerPC programming-model reference for registers, CR/XER/LR/CTR, instruction forms/semantics, byte ordering, exceptions, and memory/MMU concepts.

Do not automatically assume every optional architecture feature exists on Espresso. Cross-check Wii U/Broadway information when implementation-specific behavior matters.

## Broadway / Nintendo PowerPC lineage

### WiiBrew — Broadway

https://wiibrew.org/wiki/Broadway

Useful as lineage/background for Nintendo's Wii PowerPC.

### WiiBrew — Broadway Registers

https://wiibrew.org/wiki/Broadway/Registers

Useful for Nintendo-specific SPRs and implementation-dependent features.

### WiiBrew — Paired single

https://wiibrew.org/wiki/Paired_single

Useful for paired-single/GQR behavior in the Nintendo PowerPC lineage.

## Existing emulator projects

Potential conceptual references:

- Cemu
- Decaf
- Dolphin

Use them to study abstraction boundaries, CPU/interpreter organization, guest-memory design, HLE versus LLE choices, GPU translation architecture, and testing/debugging strategies.

Do not copy code without understanding license compatibility and provenance.
