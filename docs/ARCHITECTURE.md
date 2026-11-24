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

`CpuState` explicitly stores 32 32-bit GPRs, 32 64-bit FPR values, separate paired-single PS1 lanes, eight GQRs, CR, XER, FPSCR, LR, CTR, CIA, MSR, SRR0, SRR1, and the current single-core reservation. Loader entry uses MSR zero: supervisor-mode, big-endian, real-address execution. This is an emulator bootstrap context, not a Cafe OS boot state. The decoder supports `mfmsr`, bounded `mtmsr`, SRR0/SRR1 SPR access, and bounded `rfi`; privileged violations enter the modeled program exception path. Other exception classes and illegal-instruction exception entry remain incomplete.

`decoder.cpp` maps PowerPC instruction words to the shared `Opcode` enum and a `DecodedInstruction` containing extracted register, immediate, branch, CR, and floating-point fields. The decoder covers a growing subset; unrecognized encodings remain unsupported.

`EspressoCore::step` is the architectural execution path and calls the decoder, manipulates explicit guest state, uses `GuestMemory` for guest accesses, and dispatches HLE imports. `run` repeats steps up to its bound and reports unsupported instructions, unimplemented HLE calls, HLE errors, and memory faults. A bounded instruction-history buffer carries recent decoded operations and selected operand/effect details for diagnostics. `GuestMemory` is flat and has no MMU translation, so `mtmsr` and `rfi` stop explicitly when asked to enable IR/DR rather than continuing with incorrect address semantics. Floating-point and paired-single support is partial; this is not a complete PowerPC FPU.

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

`gx2_register_state.hpp` shadows raw Latte registers and retrieves complete texture/sampler resource groups. `latte_texture_resource.hpp` and `latte_sampler.hpp` decode supported words into typed descriptors. `latte_texture_staging.hpp` converts selected linear-aligned and 1D thin1 2D UNORM formats into RGBA8 CPU staging data. The 1D thin1 path handles the existing 8-, 16-, and 32-bit format decoders with their element-size pitch alignment; macro-tiled, compressed, mipmapped, array, and other unsupported cases are rejected. This staging helper is not yet connected to guest shader sampling or draw execution.

`Gx2RegisterState::draw_snapshot()` returns a typed, non-owning view that binds one captured draw to the vertex/fetch/program/resource/target/register state from that draw point. The renderer consumes it after applying a command batch. A later `apply()` can reallocate the owning vectors and invalidate the view; it is not a cross-thread ownership mechanism.

`VulkanRenderer` owns its window/surface, device, swapchain, frame synchronization, and uploaded texture resources. At a guest `GX2SwapScanBuffers` request, the frontend snapshots the configured, CPU-visible TV scan buffer from checked guest memory and passes RGBA8 pixels to Vulkan for swapchain presentation. For matching-size, single-sample 2D UNORM RGBA8 surfaces using linear-aligned, 1D thin1, or zero-swizzle 2D thin1 layout, a base-level single-slice view, and TV0, `GX2ClearColor` writes quantized RGBA8 pixels to guest color-buffer memory and `GX2CopyColorBufferToScanBuffer` converts the supported surface layout to tightly packed scanout pixels and writes the next configured TV scan-buffer slot. The 1D thin1 path uses padded 8x8 displayable 32-bit microtiles; the supported 2D thin1 subset uses zero-swizzle 32x16 macrotiles with aligned pitch and height. The following swap selects that slot for presentation. Other macro-tiled/swizzled surfaces, unsupported formats, scaling, multisampling, and DRC targets fail explicitly. These HLE memory operations are synchronous. `--gx2-guest-scanout-smoke-test` runs PPC `bctrl` call stubs into GX2 HLE and checks two changed scanouts before Vulkan presentation; `--gx2-guest-scanout-display-test` shows that controlled output in a visible window. The triangle diagnostic runs `GX2SetAttribBuffer`, `GX2InitFetchShaderEx`, `GX2SetFetchShader`, `GX2DrawEx`, and `GX2DrawIndexedEx` through ordered GX2 command capture. When the explicit `capture_draw_vertex_data` runtime option is enabled, non-indexed draws carry checked, immutable snapshots of their bound vertex-buffer byte ranges, captured during HLE execution so the asynchronous frontend cannot observe later guest writes. Indexed draws additionally snapshot U16_LE, U32_LE, U16, or U32 index data and the contiguous vertex-buffer ranges addressed by those indices. Capture is disabled by default to avoid allocations on ordinary guest execution, and `GX2Init` preserves the host launch policy. The diagnostic renderer supports single-instance, base-instance-zero triangle lists, strips, and fans using Float32x2/x3/x4 or Float16x2 position data selected by the supported fetch-program subset. Strips alternate triangle winding as they are expanded to lists; fans use their first vertex as the hub. Indexed draws expand captured U16/U32 indices in little- or big-endian order into a transient sequential diagnostic vertex list before `vkCmdDraw`; primitive restart is rejected because its active state is not captured. This remains fixed host-shader output to the host swapchain, not Latte shader execution or guest render-target writes. The controlled non-indexed/indexed/strip/fan test was visibly displayed and its Vulkan window closed cleanly; the fixture triangles are spatially separated, so the visible result verifies that captured indices select the intended guest vertices. The display test decodes each draw's HLE-generated VTX fetch program to find location 0's buffer slot, format, and byte offset, then converts captured Float32 or Float16x2 positions to a sequential Float32 list for the explicitly selected diagnostic shader. Indexed positions are resolved through their captured index stream. The diagnostic policy issues one Vulkan `vkCmdDraw` per captured draw in command order. The frontend's opt-in diagnostic GX2 draw mode captures and routes game draw batches through this same policy; it sends diagnostic output to the host swapchain and gives that output precedence over guest scanout while active. Unsupported layouts are reported in the frontend diagnostics. Affogato supports only a narrow fetch-program subset; it does not translate the guest vertex or pixel shader, implement arbitrary GX2 formats, write the guest render target from a draw, or render game draws with game shaders. The controlled non-indexed/indexed/strip/fan diagnostic batch was visibly verified. The renderer scales scanout to the window extent and uses a transfer path when supported; otherwise the existing clear path remains available. Vulkan clear and texture-upload smoke tests are backend validation, not proof of guest draw rendering.

## Tests and key dependencies

CMake registers `affogato_cpu_tests` and `affogato_recompiler_tests` with CTest when `BUILD_TESTING` is enabled. The standalone `affogato_recompiler_benchmark` executable is a measurement tool, not a CTest test. The test sources include CPU-state, decoder, memory, loader, HLE, GX2, and interpreter/recompiler differential coverage. Some long-running title diagnostics described in `docs/development-progress.md` are developer observations, not automated compatibility tests.

The main dependencies are MinGW-w64 GCC/C++23, CMake, Ninja, Vulkan 1.3, zlib, SDL3, GLFW, Dear ImGui, and the pinned AsmJit revision. CMake is the source of truth; do not build source files manually.

## Known boundaries

- One guest Espresso core is scheduled; multicore synchronization and asynchronous hardware agents are not modeled.
- Instruction, exception/privilege, floating-point, HLE, loader, and graphics coverage is partial.
- Full IOSU behavior, encrypted/disc/title installation flows, audio, complete input, shader translation, and playable guest graphics are not implemented.
- RPX execution can make substantial diagnostic progress without demonstrating boot completion or title compatibility.

For current measured status, known stops, and development priorities, see [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) and the detailed historical notes in [development-progress.md](development-progress.md).
