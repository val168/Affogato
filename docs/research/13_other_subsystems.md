# 13 — Other subsystems

**Status:** deliberately deferred.

Research these only when software reaches them.

## Audio

Goal: reproduce application-visible audio behavior and send resulting audio to a host backend.

Do not begin by simulating the physical DAC path.

The Wii U contains a DSP related to the Wii DSP, but whether Affogato needs LLE DSP behavior or can HLE/translate higher-level audio work should be decided from compatibility needs.

Reference: https://wiiubrew.org/wiki/Hardware/DSP

## Input

Goal: represent Wii U controller state and map host keyboard/gamepad input into guest-visible controller APIs.

Do not begin with Bluetooth/Wi-Fi radio emulation.

## GamePad

Separate:

- controls/input.
- second-screen video presentation.
- audio.
- camera/microphone/sensors if needed later.
- actual wireless protocol, which can be avoided initially.

## Filesystem/storage

Prefer:

```text
Wii U file API
    ->
Cafe/IOSU HLE
    ->
host filesystem
```

Do not emulate NAND/eMMC/SATA controllers unless low-level software requires them.

## Networking

Prefer host sockets behind Wii U-compatible APIs where possible.

Service compatibility may be a separate problem from hardware emulation.

## Cryptography/security

Secure boot, eFuses, signature enforcement, and physical crypto engines are not initial gameplay requirements.

Implement guest-visible operations only when needed.

## JIT

JIT work comes after the interpreter is stable.

The JIT should translate PowerPC guest code to the host ISA while preserving the interpreter as a reference backend. The interpreter test suite should also validate JIT behavior.
