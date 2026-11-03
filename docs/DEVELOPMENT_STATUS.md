# Affogato Development Status

This snapshot is based on the source and repository notes at `f0fbfef` (`main`, 2026-10-25). It distinguishes implemented paths from complete console behavior.

## Current implementation

- A single-core Espresso interpreter with a broad but incomplete PowerPC subset, explicit CR/XER/LR/CTR/FPSCR and paired-single state, checked guest memory, bounded instruction history, and stop diagnostics.
- An experimental AsmJit x86-64 recompiler with native lowering for a growing subset, interpreter/HLE fallbacks, code-page write-generation checks, a direct-mapped block lookup, and statistics.
- Big-endian ELF and RPX loading with compressed sections, implemented relocations/trampolines, imported HLE function/data binding, and TLS relocation support.
- A small per-session Cafe OS model and selected coreinit, GX2, nn_act, nn_olv, and nn_save handlers. Unsupported imports remain explicit stops/errors.
- GX2/Latte state and command capture, typed texture/sampler descriptor decoding, CPU staging for selected linear 2D UNORM textures, and a Vulkan swapchain plus opt-in RGBA8 upload primitive.
- An SDL3/ImGui RPX library frontend and command-line diagnostic mode.

## Verified boundaries and limitations

- The frontend can load and execute RPX code to a bounded stop. That is not a game-compatibility claim.
- GX2 HLE and GPU state capture are present, but guest shader translation, draw submission, framebuffer handling, and guest image presentation are not connected into an end-to-end renderer.
- CPU, loader, HLE, and graphics support is incremental. Full IOSU, multicore scheduling, complete CPU/FPU behavior, audio, complete input, encrypted-title loading, and broad game compatibility remain out of scope or unimplemented.
- `docs/development-progress.md` records diagnostic runs and measured observations. Its latest notes report WWHD stopping at an unsupported null-target fetch (`0x46524D48`) after 6,081,749 instructions, and BOTW reaching a 50-million instruction cap at CIA `0x2F18DDC`. Neither run reaches a guest draw submission. These figures describe those recorded runs, not current environmental verification or compatibility.
- The repository provides CPU and recompiler test executables plus a benchmark executable. Historical progress notes report successful Debug/Release builds and test runs, but this workspace's present baseline could not be rebuilt: the configure script could not execute Ninja under the current Windows Application Control policy. `ctest --test-dir build/debug --output-on-failure` found no registered tests in the existing incomplete build directory. See the current session report for that environment-specific result.
- The current environment has GCC 16.2.0 and CMake 4.4.4. `scripts/configure.ps1` found no runnable Ninja >= 1.11, so no fresh baseline build/test result is available. This is a tooling blocker, not a source failure.

## Tests and measurements

- `affogato_cpu_tests`: CPU state, decoder, guest memory, loaders, and HLE coverage.
- `affogato_recompiler_tests`: recompiler execution, fallback, invalidation, and selected interpreter/JIT state/memory comparisons.
- `affogato_recompiler_benchmark`: standalone benchmarking; no current-session benchmark was run.
- No current-session performance comparison was made. Existing timing observations and discarded optimization experiments remain in `development-progress.md`; they should not be treated as controlled benchmark claims.

## Development priorities

The detailed queue in `development-progress.md` is the project-specific source for next implementation priorities. The current high-level priorities are:

1. Continue investigating the recorded WWHD/BOTW startup stops through guest initialization and existing HLE behavior; do not fabricate successful callbacks or results.
2. Audit guest shader, draw, and scan-buffer producers before connecting captured resources to host rendering.
3. Keep interpreter/recompiler semantics aligned and add native lowerings only with differential tests and repeatable Release measurements where performance is the objective.
4. Extend loaders and HLE only for observed guest requirements, preserving transactional guest-memory effects and explicit unsupported behavior.
5. Restore a runnable local GCC/CMake/Ninja toolchain, then configure Debug and Release and run the registered CTest suite before code changes are considered validated.

## Next refactoring area

No source refactor has been made in this documentation checkpoint. After the build tool is runnable, start with a small, reviewable helper/header or memory-related refactor that has focused CPU tests. Keep `interpreter.cpp`, `recompiler.cpp`, and `gx2_hle.cpp` changes scoped to one logical section at a time. Preserve the pre-existing `AGENTS.md` worktree modification.
