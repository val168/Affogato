# Affogato Development Status

This snapshot is based on the source and repository notes at `f0fbfef` (`main`, 2026-10-25). It distinguishes implemented paths from complete console behavior.

## Current source update (2026-11-03)

The frontend now snapshots a configured TV scan buffer at `GX2SwapScanBuffers` and can present its CPU-visible RGBA8 pixels through the Vulkan swapchain. This requires the guest to have written pixels into the registered memory; Affogato still does not execute Latte draw commands or translate guest shaders, so games whose frames are produced by GX2 draws will not render correctly through this path alone. A failed/unmapped scan-buffer read leaves the previous Vulkan output in place. Debug and Release builds succeeded; Debug CTest passed 2/2, including padded scanout-row staging coverage. The Vulkan RGBA8 upload and scanout presentation smoke test passed on an NVIDIA GeForce RTX 5070 Laptop GPU. No game launch was verified in this update.

## Current implementation

- A single-core Espresso interpreter with a broad but incomplete PowerPC subset, explicit CR/XER/LR/CTR/FPSCR and paired-single state, checked guest memory, bounded instruction history, and stop diagnostics.
- An experimental AsmJit x86-64 recompiler with native lowering for a growing subset, interpreter/HLE fallbacks, code-page write-generation checks, a direct-mapped block lookup, and statistics.
- Big-endian ELF and RPX loading with compressed sections, implemented relocations/trampolines, imported HLE function/data binding, and TLS relocation support.
- A small per-session Cafe OS model and selected coreinit, GX2, nn_act, nn_olv, and nn_save handlers. Unsupported imports remain explicit stops/errors.
- GX2/Latte state and command capture, typed texture/sampler descriptor decoding, CPU staging for selected linear 2D UNORM textures and configured TV scan buffers, and a Vulkan swapchain with guest scan-buffer presentation plus an opt-in RGBA8 upload primitive.
- An SDL3/ImGui RPX library frontend and command-line diagnostic mode.

## Verified boundaries and limitations

- The frontend can load and execute RPX code to a bounded stop. That is not a game-compatibility claim.
- GX2 HLE and GPU state capture are present. Guest scan-buffer pixels can be copied to Vulkan at swap requests, but guest shader translation, draw submission, GPU framebuffer writes, and full guest image production are not connected into an end-to-end renderer.
- CPU, loader, HLE, and graphics support is incremental. Full IOSU, multicore scheduling, complete CPU/FPU behavior, audio, complete input, encrypted-title loading, and broad game compatibility remain out of scope or unimplemented.
- `docs/development-progress.md` records diagnostic runs and measured observations. Its latest notes report WWHD stopping at an unsupported null-target fetch (`0x46524D48`) after 6,081,749 instructions, and BOTW reaching a 50-million instruction cap at CIA `0x2F18DDC`. Neither run reaches a guest draw submission. These figures describe those recorded runs, not current environmental verification or compatibility.
- The repository provides CPU and recompiler test executables plus a benchmark executable. In the current environment (GCC 16.2.0, CMake 4.4.4, Ninja 1.13.2), the Debug and Release presets build successfully and Debug CTest passes 2/2. The Vulkan path was not exercised in a live title during this update.

## Tests and measurements

- `affogato_cpu_tests`: CPU state, decoder, guest memory, loaders, and HLE coverage.
- `affogato_recompiler_tests`: recompiler execution, fallback, invalidation, and selected interpreter/JIT state/memory comparisons.
- `affogato_recompiler_benchmark`: standalone benchmarking; no current-session benchmark was run.
- No current-session performance comparison was made. Existing timing observations and discarded optimization experiments remain in `development-progress.md`; they should not be treated as controlled benchmark claims.

## Development priorities

The detailed queue in `development-progress.md` is the project-specific source for next implementation priorities. The current high-level priorities are:

1. Continue investigating the recorded WWHD/BOTW startup stops through guest initialization and existing HLE behavior; do not fabricate successful callbacks or results.
2. Implement `GX2CopyColorBufferToScanBuffer` and trace the title's actual color-buffer and shader paths. The new scanout upload only displays memory that is already populated; common GX2 flows explicitly copy their color buffer into the scan buffer (for example, [devkitPro's Wii U graphics setup](https://github.com/devkitPro/wut/blob/master/libraries/libwhb/src/gfx.c)).
3. Keep interpreter/recompiler semantics aligned and add native lowerings only with differential tests and repeatable Release measurements where performance is the objective.
4. Extend loaders and HLE only for observed guest requirements, preserving transactional guest-memory effects and explicit unsupported behavior.
5. Keep validating changes with both configured builds and the registered CTest suite; add Vulkan presentation checks that remain usable without a display where practical.

## Next implementation area

Guest color-buffer writes are not emulated yet, and `GX2CopyColorBufferToScanBuffer` is not registered in the current source. The immediate rendering work is to trace the copy API's guest calling convention and surface layout, add focused HLE/state tests, and only then connect a supported color-buffer implementation to the scanout path. Full game rendering additionally requires Latte command execution and guest shader translation.
