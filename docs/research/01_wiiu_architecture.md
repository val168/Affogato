# 01 — Wii U architecture

**Status:** enough information for a high-level architecture map; do not over-research low-level hardware yet.

## Espresso

Espresso is the Wii U's main PowerPC processor. WiiUBrew documents it as a three-core processor, with each core running one thread, clocked at about 1243.125 MHz.

For Affogato, Espresso matters because Wii U applications and Cafe OS execute PowerPC code on it.

## Latte

Latte is the Wii U chipset/package containing graphics hardware and the Starbuck security/I/O processor. The GX2 graphics processor is used directly by Espresso software in Wii U mode.

## Starbuck

Starbuck is the Wii U security/input-output coprocessor. It runs IOSU as well as early boot components.

The real console's boot process begins on Starbuck, which eventually loads Cafe OS and bootstraps Espresso.

## Cafe OS

Cafe OS is the operating system running on the PowerPC side in Wii U mode.

Its responsibilities include process isolation, memory management, interrupt handling, executable loading, system libraries, and communication with IOSU.

Applications dynamically link against RPL system libraries to access services such as memory management, graphics, audio, filesystem, and input.

## IOSU

IOSU runs on Starbuck. It contains a kernel and user-mode resource managers for hardware/device services. These resource managers expose resources using `/dev`-style names and operations conceptually similar to open/close/read/write/ioctl/ioctlv.

## Memory

The major documented memory pools include:

- **MEM1:** 32 MiB.
- **MEM2:** 2 GiB DDR3.
- **MEM0:** legacy/embedded regions used by old GX/system components.

The public Wii U memory map is much more detailed and should be consulted when Affogato starts mapping real executables and MMIO.

## Real boot flow

```text
Starbuck boot0
    ->
Starbuck boot1
    ->
IOSU
    ->
Cafe OS loaded
    ->
Espresso bootstrapped
    ->
Cafe OS runs on Espresso
```

Affogato does **not** need to reproduce this chain initially.

## What guest software can observe

For the first emulator phases, the important observable boundaries are:

- PowerPC instruction behavior.
- guest memory contents and addresses.
- Cafe OS library/API behavior.
- IOSU-facing service behavior.
- GX2-visible graphics state and commands.
- timing/synchronization where software depends on them.

## Affogato plan

### Emulate

- guest-visible Espresso architectural state.
- PowerPC instructions.
- guest memory semantics.
- software-visible synchronization behavior.

### HLE / replace

- Cafe OS services where practical.
- IOSU services instead of emulating Starbuck initially.
- filesystem/storage through host files.
- system timing using an emulator clock backed by host timing where appropriate.

### Translate

- GX2/Latte graphics behavior to Vulkan later.

## What can be deferred

Do not spend early development time on:

- secure boot.
- eFuses.
- NAND/SATA controller internals.
- Wi-Fi/Bluetooth hardware protocols.
- real Starbuck execution.
- exact CPU/GPU bus timing.
- exact cache behavior.
- power management.

## Completion criterion

You should be able to explain this diagram:

```text
                   Wii U application
                          |
                     Cafe OS APIs
                          |
             +------------+------------+
             |                         |
          Espresso                    GX2
       PowerPC execution               |
             |                       Latte
             |                         |
          Memory                  graphics work
             |
             +------ IPC ------> IOSU on Starbuck
```

and identify which parts Affogato intends to emulate, HLE, or translate.

## Sources

- WiiUBrew — Hardware: https://wiiubrew.org/wiki/Wii_U_console
- WiiUBrew — Espresso: https://wiiubrew.org/wiki/Espresso
- WiiUBrew — Latte: https://wiiubrew.org/wiki/Hardware/Latte
- WiiUBrew — Starbuck: https://wiiubrew.org/wiki/Hardware/Starbuck
- WiiUBrew — Boot process: https://wiiubrew.org/wiki/Boot_Process
- WiiUBrew — Cafe OS: https://wiiubrew.org/wiki/Cafe_OS
- WiiUBrew — IOS: https://wiiubrew.org/wiki/IOS
- WiiUBrew — Memory map: https://wiiubrew.org/wiki/Memory_Map
