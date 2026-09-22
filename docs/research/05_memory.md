# 05 — Memory

**Status:** implement once the interpreter needs instruction/data fetches beyond isolated unit tests.

## Goal

Provide a clean guest-memory abstraction that lets CPU code operate on Wii U addresses without spreading raw host pointers through the emulator.

## Real Wii U information to retain

WiiUBrew documents:

- 32 MiB MEM1.
- 2 GiB MEM2.
- additional MEM0/system regions.
- numerous MMIO ranges.
- distinct regions used by Cafe OS, applications, IOSU, and hardware.

The detailed physical/virtual map becomes important when loading real Wii U software.

## First Affogato memory model

Begin much simpler than the real console.

Needed operations will look conceptually like:

```text
read8(address)
read16(address)
read32(address)
read64(address)

write8(address, value)
write16(address, value)
write32(address, value)
write64(address, value)
```

Requirements:

- explicit guest address type or clear integer convention.
- bounds checking in debug/test paths.
- deterministic invalid-access behavior.
- explicit endian conversion.
- no random direct host pointers embedded throughout the CPU implementation.

## Endianness

The host is x86-64 little-endian while the targeted PowerPC/Wii U software uses big-endian conventions.

Endian conversion should live at well-defined boundaries such as memory access helpers, not be scattered through instruction handlers.

## Flat memory first

Synthetic programs can initially run in a simplified flat address space.

Do not block the first interpreter on:

- complete BAT translation.
- complete page-table/MMU behavior.
- every MMIO device.
- cache simulation.

The memory API should allow address translation to be inserted later behind the same CPU-facing interface.

## When real loading begins

Research and implement:

- executable address ranges.
- code/data permissions as needed.
- MEM1/MEM2 mappings.
- imported libraries.
- relevant MMIO only if accessed.
- virtual/effective/physical address distinctions.

## Sources

- WiiUBrew — Memory map: https://wiiubrew.org/wiki/Memory_Map
- WiiUBrew — Cafe OS: https://wiiubrew.org/wiki/Cafe_OS
- NXP/Motorola — PowerPC Programming Environments Manual.
