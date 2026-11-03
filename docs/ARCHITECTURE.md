# Affogato Architecture

This document describes the implementation in the current source tree. Affogato is an experimental Wii U emulator foundation; supported behavior is incremental, and the presence of an HLE handler or graphics helper does not imply broad title compatibility.

## Project and build layout

- `src/cpu/espresso/` contains CPU state, instruction decoding and execution, guest memory, ELF/RPX loading, the recompiler, and Cafe OS/GX2/library HLE.
- `src/gpu/` contains Latte register/resource decoding, texture staging, and the Vulkan host backend.
- `src/emulator.*` provides the session-level API. `src/main.cpp` is a small CPU demonstration executable; the SDL3/ImGui game library and diagnostic frontend is `src/frontend/main.cpp`.
- `tests/` contains CPU and interpreter/recompiler tests, a recompiler benchmark executable, and a minimal WUT RPX fixture.
- CMake defines the CPU static library, the two executables, tests, and benchmark. zlib is a submodule; AsmJit and GLFW are pinned FetchContent dependencies; SDL3 and ImGui are in `third_party/`; Vulkan 1.3 is required.

The repository's `scripts/configure.ps1` discovers MinGW GCC and Ninja and writes local user presets. The checked-in `CMakePresets.json` supplies the underlying `debug` and `release` presets. Follow `AGENTS.md` for the local build workflow.

## Session and execution lifecycle

`Emulator` constructs one `EspressoCore` and registers the available coreinit, GX2, nn_act, nn_olv, and nn_save handlers. `load_rpx` loads an RPX into guest memory, initializes small-data registers, reserves a stack, and creates a synthetic default guest thread and heaps. A session accepts one loaded image. `run` chooses either the interpreter or the opt-in recompiler and returns stop diagnostics, GPR3, HLE statistics, JIT statistics, and queued GX2 commands.

The frontend runs a selected RPX on a worker thread and transfers result/state snapshots to the UI thread. Its normal desktop UI is SDL3 plus Dear ImGui. A separately owned Vulkan window can be opened for backend output. The command-line diagnostic path can run an RPX with an instruction limit, select the recompiler, filter HLE traces, watch guest writes, and dump a bounded guest-memory range.

## Espresso state, decoder, and interpreter

`CpuState` explicitly stores 32 32-bit GPRs, 32 64-bit FPR values, separate paired-single PS1 lanes, eight GQRs, CR, XER, FPSCR, LR, CTR, CIA, and the current single-core reservation. The state is not a complete model of every Espresso privilege, exception, or multicore feature.

`decoder.cpp` maps PowerPC instruction words to the shared `Opcode` enum and a `DecodedInstruction` containing extracted register, immediate, branch, CR, and floating-point fields. The decoder covers a growing subset; unrecognized encodings remain unsupported.

`EspressoCore::step` is the architectural execution path and calls the decoder, manipulates explicit guest state, uses `GuestMemory` for guest accesses, and dispatches HLE imports. `run` repeats steps up to its bound and reports unsupported instructions, unimplemented HLE calls, HLE errors, and memory faults. A bounded instruction-history buffer carries recent decoded operations and selected operand/effect details for diagnostics. Floating-point and paired-single support is partial; this is not a complete PowerPC FPU.

## Guest memory and loading

`GuestMemory` provides big-endian scalar reads/writes over a fixed flat backing vector, additional sparse mapped regions, and address aliases. It checks guest ranges and reports access faults. It tracks write generations per page for compiled-code validation, LL/SC reservation information, optional write-watch events, and supports accesses spanning backing-region boundaries. The flat memory pointer is a read-only JIT fast-path view; callers must validate ranges before using it.

`elf_loader.cpp` handles bounded, 32-bit big-endian PowerPC ELF program segments. `rpx_loader.cpp` loads Wii U RPX sections, inflates supported compressed sections, applies the implemented relocation set (including REL24 trampolines and TLS-related relocations), and binds imported symbols through HLE traps. This does not implement installed-title/disc decryption or a general multi-module runtime loader.

## Recompiler

`Recompiler` is instance-local and uses AsmJit to produce x86-64 host blocks. It decodes blocks, emits native operations for supported instructions, and falls back through `EspressoCore::step` for unsupported native operations and HLE. Guest register values may be held in AsmJit-managed host registers within a block and are flushed to `CpuState` at required boundaries. The cache uses guest entry addresses plus a direct-mapped lookup and recent-block shortcuts; it tracks guest-memory page write generations and memory ownership to reject stale code. The implementation also records native/fallback/dispatch statistics and block execution/compile timings.

The interpreter remains the correctness reference for JIT behavior. Differential tests compare state and memory for selected instructions. Any changes to register synchronization, fallback, guest writes, direct control flow, code invalidation, or instruction counting need particular care. The recompiler is experimental and does not natively lower every interpreted instruction.

## HLE and runtime services

`HleDispatcher` assigns guest addresses to registered function/data imports, maps import addresses to handlers, and records call counts and a bounded recent-call trace. RPX import traps transfer control to this dispatcher; unknown imports stop execution with a diagnostic rather than reporting success.

Cafe OS HLE lives primarily in `cafe_os_hle.cpp`, with guest mutex helpers separated into `guest_mutex.cpp`. Implemented paths model selected thread, memory/heap, filesystem, synchronization, time, ProcUI, input, and diagnostics behavior in a deliberately small single-core runtime. The `nn_*_hle.cpp` files register narrow library-specific behaviors. These handlers are not a complete Cafe OS or IOSU implementation.

## GX2, Latte, and Vulkan

`gx2_hle.cpp` registers GX2 entry points and updates per-core `Gx2RuntimeState`. It records ordered Latte register writes and higher-level commands, including selected draw, shader, texture/sampler, vertex-buffer, color-buffer, depth-buffer, and scan-buffer state. Guest structures are read through checked guest-memory helpers. This command/state layer does not by itself submit a translated guest draw.

`gx2_register_state.hpp` shadows raw Latte registers and retrieves complete texture/sampler resource groups. `latte_texture_resource.hpp` and `latte_sampler.hpp` decode supported words into typed descriptors. `latte_texture_staging.hpp` converts explicitly supported linear 2D UNORM formats into RGBA8 CPU staging data; tiled/compressed/mipped/array and other unsupported cases are rejected.

`VulkanRenderer` owns its window/surface, device, swapchain, frame synchronization, and uploaded texture resources. Its render pass currently clears/presents the host swapchain. It can consume captured GX2 commands and has an opt-in RGBA8 image-upload path, but guest shader translation, general texture binding, guest draw submission, framebuffer emulation, and game-frame presentation are not connected end to end. A Vulkan clear or texture-upload smoke test is backend validation, not guest rendering.

## Tests and key dependencies

CMake registers `affogato_cpu_tests` and `affogato_recompiler_tests` with CTest when `BUILD_TESTING` is enabled. The standalone `affogato_recompiler_benchmark` executable is a measurement tool, not a CTest test. The test sources include CPU-state, decoder, memory, loader, HLE, GX2, and interpreter/recompiler differential coverage. Some long-running title diagnostics described in `docs/development-progress.md` are developer observations, not automated compatibility tests.

The main dependencies are MinGW-w64 GCC/C++23, CMake, Ninja, Vulkan 1.3, zlib, SDL3, GLFW, Dear ImGui, and the pinned AsmJit revision. CMake is the source of truth; do not build source files manually.

## Known boundaries

- One guest Espresso core is scheduled; multicore synchronization and asynchronous hardware agents are not modeled.
- Instruction, exception/privilege, floating-point, HLE, loader, and graphics coverage is partial.
- Full IOSU behavior, encrypted/disc/title installation flows, audio, complete input, shader translation, and playable guest graphics are not implemented.
- RPX execution can make substantial diagnostic progress without demonstrating boot completion or title compatibility.

For current measured status, known stops, and development priorities, see [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) and the detailed historical notes in [development-progress.md](development-progress.md).
