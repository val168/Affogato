# Affogato

Affogato is an experimental Nintendo Wii U emulator written in modern C++23. It is being developed incrementally, with an emphasis on reproducing guest-visible behavior while keeping early implementations small and testable.

## Current state

Affogato has a partial Espresso (PowerPC) interpreter, an experimental x86-64 recompiler, ELF/RPX loaders, selected Cafe OS and GX2 HLE, and a desktop frontend. It can load Wii U RPX files and execute them to a bounded diagnostic stop, but it is **not yet capable of playing commercial games**. Guest graphics are not rendered end to end. See [the architecture overview](docs/ARCHITECTURE.md) and [development status](docs/DEVELOPMENT_STATUS.md) for the current boundaries.

The current development focus is following real RPX startup far enough to expose and implement the next narrowly scoped CPU, loader, or HLE requirement. Wind Waker HD and Breath of the Wild have been used as bounded diagnostic targets; their latest recorded stops and limits are in [development status](docs/DEVELOPMENT_STATUS.md). Diagnostic progress is not a compatibility claim.

## Implemented so far

- A single-core Espresso interpreter with explicit GPR, CR, XER, LR/CTR, FPSCR, floating-point, and paired-single state, plus a growing subset of PowerPC instructions. Unsupported instructions stop with diagnostics; this is not a complete PowerPC implementation or FPU.
- An experimental AsmJit x86-64 recompiler with a growing native instruction subset, interpreter fallbacks, block caching, and guest-code write-generation validation.
- Guest memory backed by a low flat region and sparse mappings, including accesses that cross backing-region boundaries and runtime fault diagnostics with recent instruction history.
- Minimal big-endian PowerPC ELF and Wii U RPX/RPL loading, including compressed sections, supported relocations, REL24 trampolines, and named import traps for HLE dispatch.
- An `Emulator` session API that creates the core, registers available HLE functions, loads an RPX, runs it, and returns a diagnostic result.
- A deliberately small, single-thread Cafe OS model with a guest-visible synthetic `OSThread` and selected `coreinit` functions. Most Cafe OS and other Wii U services remain unimplemented.
- Selected Cafe OS, GX2, and `nn_*` HLE paths; GX2/Latte command and resource state capture; descriptor decoding and staging for selected linear textures; and a Vulkan swapchain plus opt-in texture upload path. These pieces do not yet render guest games.
- An SDL3 and Dear ImGui frontend for recursively discovering RPX files, remembering game folders, switching between list and grid views, launching on a worker thread, and displaying loader/execution diagnostics.

## Not implemented

Affogato is an early emulator project, not a complete console implementation. It does not yet provide end-to-end GX2/Latte guest rendering, audio, complete controller/input emulation, a complete CPU/FPU, full Cafe OS or IOSU services, multicore scheduling, encrypted-disc or installed-title loading, or broad game compatibility. RPX imports and relocations are supported incrementally; encountering an unimplemented import or instruction is expected.

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
