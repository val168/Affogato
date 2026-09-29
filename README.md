# Affogato

Affogato is an experimental Nintendo Wii U emulator written in modern C++23. It is being developed incrementally, with an emphasis on reproducing guest-visible behavior while keeping early implementations small and testable.

## Current state

Affogato has an early Espresso (PowerPC) interpreter, executable loaders, a small Cafe OS HLE layer, and a minimal desktop frontend. It can load genuine Wii U RPX files and begin executing them, but it is **not yet capable of playing commercial games**. Real titles currently stop at unsupported instructions or unimplemented OS imports; graphics and the rest of the console are not implemented.

The current development focus is following real RPX startup far enough to expose and implement the next narrowly scoped CPU, loader, or HLE requirement. A Wind Waker HD RPX has been used as a diagnostic target and currently reaches an unimplemented `coreinit::__ghsLock` import. That is progress through startup, not a compatibility claim.

## Implemented so far

- A single-core, instruction-interpreted Espresso CPU foundation with general-purpose registers, condition register, LR/CTR, and 64-bit floating-point register storage.
- A growing subset of PowerPC integer instructions, branches, memory operations, function-call mechanics, and scalar `lfs`/`stfs` instructions. Unsupported instructions stop with diagnostics; this is not a complete PowerPC implementation or FPU.
- Guest memory backed by a low flat region and sparse mappings, including accesses that cross backing-region boundaries and runtime fault diagnostics with recent instruction history.
- Minimal big-endian PowerPC ELF and Wii U RPX/RPL loading, including compressed sections, supported relocations, REL24 trampolines, and named import traps for HLE dispatch.
- An `Emulator` session API that creates the core, registers available HLE functions, loads an RPX, runs it, and returns a diagnostic result.
- A deliberately small, single-thread Cafe OS model with a guest-visible synthetic `OSThread` and selected `coreinit` functions. Most Cafe OS and other Wii U services remain unimplemented.
- A basic SDL3 and Dear ImGui frontend for recursively discovering RPX files, remembering game folders, switching between list and grid views, launching on a worker thread, and displaying loader/execution diagnostics. The frontend does not render games.

## Not implemented

Affogato is an early emulator project, not a complete console implementation. In particular, it does not yet provide GX2/Latte graphics or game rendering, audio, controller/input emulation, a complete CPU/FPU, full Cafe OS or IOSU services, multicore scheduling, encrypted-disc or installed-title loading, or broad game compatibility. RPX imports and relocations are supported incrementally; encountering an unimplemented import or instruction is expected.

## Build requirements

The primary development setup is Windows with MinGW-w64 GCC, CMake, and Ninja:

- C++23 compiler (the current project toolchain uses GCC)
- CMake 3.25 or newer
- Ninja 1.11 or newer

SDL3, Dear ImGui, and zlib are kept in `third_party/` and integrated through CMake. Use the project's configure script to discover and configure the local toolchain.

## Configure and build

From PowerShell at the repository root:

```powershell
.\scripts\configure.ps1
cmake --build --preset local-debug
```

Run the test suite with:

```powershell
ctest --test-dir build/debug --output-on-failure
```

The build produces the emulator and, when frontend dependencies are available, `AffogatoFrontend.exe`.
