# 10 — IOSU HLE

**Status:** default plan is HLE; no Starbuck CPU emulator initially.

## Real system

Starbuck is the Wii U security/I/O coprocessor and runs IOSU.

WiiUBrew describes IOSU as providing a kernel plus user-mode resource-manager processes. Resource managers expose `/dev`-style nodes and operations such as:

- open
- close
- read
- write
- seek
- ioctl
- ioctlv

Cafe OS communicates with IOSU through an IPC mechanism.

The real Latte IPC hardware provides signalling between Espresso and Starbuck.

## Affogato plan

Affogato does not initially need:

```text
ARM instruction emulation
    ->
real IOSU kernel
    ->
real resource-manager process
    ->
real controller/device
```

Instead:

```text
guest/Cafe request
    ->
Affogato IOSU-facing dispatcher
    ->
native service implementation
    ->
expected guest-visible result
```

## Research target

For every required IOSU interaction, determine:

- how the Espresso/Cafe side identifies the service.
- request format.
- input/output buffers.
- command/ioctl numbers.
- return/error values.
- asynchronous behavior.
- memory ownership/lifetime.
- synchronization expectations.

## Host replacements

Possible host-backed implementations include:

- filesystem/storage.
- sockets/networking.
- clocks.
- cryptographic operations where needed for legitimate user-owned data.
- device state.

Do not emulate physical NAND/SATA/SD/USB controllers unless a title directly depends on low-level behavior that cannot be represented at the service level.

## Optional far-future LLE

A Starbuck ARM backend plus real IOSU execution could be added much later for system-software accuracy, unusual low-level software, or validation/debugging. It is not a prerequisite for normal Affogato development.

## Sources

- WiiUBrew — Starbuck: https://wiiubrew.org/wiki/Hardware/Starbuck
- WiiUBrew — IOS: https://wiiubrew.org/wiki/IOS
- WiiUBrew — Latte IPC: https://wiiubrew.org/wiki/Hardware/IPC
- WiiUBrew — Boot process: https://wiiubrew.org/wiki/Boot_Process
