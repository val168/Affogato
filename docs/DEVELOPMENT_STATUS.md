# Affogato Development Status

This snapshot is based on the source and repository notes at `f0fbfef` (`main`, 2026-10-25), with verified updates below through 2026-11-22. It distinguishes implemented paths from complete console behavior.

## Current source update (2026-11-03)

The frontend now snapshots a configured TV scan buffer at `GX2SwapScanBuffers` and can present its CPU-visible RGBA8 pixels through the Vulkan swapchain. This requires the guest to have written pixels into the registered memory; Affogato still does not execute Latte draw commands or translate guest shaders, so games whose frames are produced by GX2 draws will not render correctly through this path alone. A failed/unmapped scan-buffer read leaves the previous Vulkan output in place. Debug and Release builds succeeded; Debug CTest passed 2/2, including padded scanout-row staging coverage. The Vulkan RGBA8 upload and scanout presentation smoke test passed on an NVIDIA GeForce RTX 5070 Laptop GPU. No game launch was verified in this update.

## Verified rendering update (2026-11-22)

The controlled diagnostic triangle is now visually verified in the Vulkan window on an NVIDIA GeForce RTX 5070 Laptop GPU. Its guest float2 vertices were displayed with the fixed diagnostic shader; the window remained responsive through maximize/restore and exited cleanly. The Khronos validation layer was installed and enabled by the renderer; no validation warning/error was emitted during the display run. This verifies milestone 1 only, not translated guest shaders or WWHD graphics.

With the explicit `capture_draw_vertex_data` runtime option enabled, non-indexed GX2 draw HLE records carry immutable snapshots of each bound vertex buffer's requested range. Indexed draw HLE also snapshots index bytes for the four WUT `GX2IndexType` values and the contiguous vertex-buffer ranges addressed by those indices. Both paths validate guest-memory ranges at draw time, and `GX2Init` preserves the launch-configured capture policy. Capture is disabled by default to avoid draw-path allocation overhead. The frontend offers an opt-in `Diagnostic GX2 draws (host shader, host window)` mode; it routes ordered draw batches through the renderer's diagnostic policy. The diagnostic renderer now accepts single-instance, base-instance-zero triangle lists and expands captured U16/U32 little- or big-endian indices into sequential diagnostic vertices before Vulkan submission. Primitive restart is rejected because its state is not captured. The visible controlled test submitted one non-indexed and one indexed triangle through guest GX2 HLE and displayed their diagnostic host-shader output; the fixture triangles are spatially separated. This still does not execute Latte shaders or write guest draw results to guest color-buffer memory. Accepted draws use one Vulkan draw range per command, and diagnostic presentation takes precedence over guest scanout while active. Unsupported draws report a graphics diagnostic and are not treated as successfully rendered. The mode still uses fixed host shaders and a host swapchain target; general vertex formats, guest render-target writes, and Latte shaders remain unimplemented. Debug and Release builds, CTest (2/2 in each configuration), and hidden plus visible triangle checks pass. Debug/Release CMake builds emit existing `-Wpedantic` warnings from vendored AsmJit `__int128` and anonymous structs.

The diagnostic consumer now also requires the captured `GX2SetFetchShader` state, decodes the HLE-generated VTX program, selects attribute location 0 and its specified vertex-buffer slot/byte offset, and reads the position from the immutable guest snapshot. Float32x2, Float32x3, and Float32x4 layouts can provide the first two position components; other formats and fetch programs are rejected. The controlled guest test constructs the fetch shader through `GX2InitFetchShaderEx`/`GX2SetFetchShader` and uses vertex-buffer slot 2, a 16-byte stride, and a nonzero 4-byte position offset. Its Vulkan output remains visible, verifying those decoded fields are used. This remains diagnostic-shader output to the host swapchain, not execution of the guest's vertex or pixel shader.

## Current implementation

- A single-core Espresso interpreter with a broad but incomplete PowerPC subset, explicit CR/XER/LR/CTR/FPSCR and paired-single state, checked guest memory, bounded instruction history, and stop diagnostics.
- An experimental AsmJit x86-64 recompiler with native lowering for a growing subset, interpreter/HLE fallbacks, code-page write-generation checks, a direct-mapped block lookup, and statistics.
- Big-endian ELF and RPX loading with compressed sections, implemented relocations/trampolines, imported HLE function/data binding, and TLS relocation support.
- A small per-session Cafe OS model and selected coreinit, GX2, nn_act, nn_olv, and nn_save handlers. Unsupported imports remain explicit stops/errors.
- GX2/Latte state and command capture, typed texture/sampler descriptor decoding, CPU staging for selected linear 2D UNORM textures and configured TV scan buffers, and a Vulkan swapchain with guest scan-buffer presentation plus an opt-in RGBA8 upload primitive.
- An SDL3/ImGui RPX library frontend and command-line diagnostic mode.

## Verified boundaries and limitations

- The frontend can load and execute RPX code to a bounded stop. That is not a game-compatibility claim.
- GX2 HLE supports `GX2ClearColor` and `GX2CopyColorBufferToScanBuffer` for a bounded linear RGBA8 TV0 configuration, with guest-memory writes and ordered command capture. Guest draw execution, shader translation, tiled render targets, and full game frame production remain unimplemented.
- CPU, loader, HLE, and graphics support is incremental. Full IOSU, multicore scheduling, complete CPU/FPU behavior, audio, complete input, encrypted-title loading, and broad game compatibility remain out of scope or unimplemented.
- `docs/development-progress.md` records diagnostic runs and measured observations. Its latest notes report WWHD stopping at an unsupported null-target fetch (`0x46524D48`) after 6,081,749 instructions, and BOTW reaching a 50-million instruction cap at CIA `0x2F18DDC`. Neither run reaches a guest draw submission. These figures describe those recorded runs, not current environmental verification or compatibility.
- The repository provides CPU and recompiler test executables plus a benchmark executable. In the current environment (GCC 16.2.0, CMake 4.4.4, Ninja 1.13.2), the Debug and Release presets build successfully and Debug CTest passes 2/2. The Vulkan path was not exercised in a live title during this update.

## Tests and measurements

- `affogato_cpu_tests`: CPU state, decoder, guest memory, loaders, and HLE coverage.
- `affogato_recompiler_tests`: recompiler execution, fallback, invalidation, and selected interpreter/JIT state/memory comparisons.
- `affogato_recompiler_benchmark`: standalone benchmarking; no current-session benchmark was run.
- `AffogatoFrontend.exe --gx2-guest-scanout-smoke-test`: a controlled PPC `bctrl` → GX2 HLE → guest color-buffer clear → TV scanout copy/swap → Vulkan presentation diagnostic. It checks pixel bytes for two distinct guest clear colors and submits both scanouts, but it does not load a WUT RPX or execute guest draw commands.
- `AffogatoFrontend.exe --gx2-guest-scanout-display-test`: the same controlled operations in a visible window, held open until closed for visual inspection.
- `AffogatoFrontend.exe --gx2-guest-triangle-smoke-test`: captures a guest `GX2DrawEx`, snapshots its non-indexed guest vertex-buffer bytes during HLE execution, and submits the controlled float2 vertices with a diagnostic Vulkan shader. It does not translate Latte shaders or support general guest vertex layouts.
- `AffogatoFrontend.exe --gx2-guest-triangle-display-test`: the same diagnostic path in a visible window; visually verified on 2026-11-22, including maximize/restore and clean close.
- No current-session performance comparison was made. Existing timing observations and discarded optimization experiments remain in `development-progress.md`; they should not be treated as controlled benchmark claims.

## Development priorities

The detailed queue in `development-progress.md` is the project-specific source for next implementation priorities. The current high-level priorities are:

1. Continue investigating the recorded WWHD/BOTW startup stops through guest initialization and existing HLE behavior; do not fabricate successful callbacks or results.
2. Add a controlled guest execution test that drives GX2 setup, clear, copy, and swap through normal HLE dispatch, then verifies the resulting scanout through the existing Vulkan presentation path. Trace title surface layouts before extending format or tiling support.
3. Keep interpreter/recompiler semantics aligned and add native lowerings only with differential tests and repeatable Release measurements where performance is the objective.
4. Extend loaders and HLE only for observed guest requirements, preserving transactional guest-memory effects and explicit unsupported behavior.
5. Keep validating changes with both configured builds and the registered CTest suite; add Vulkan presentation checks that remain usable without a display where practical.

## Next implementation area

The implemented color-buffer clear and copy path is limited to linear-aligned, single-sample 2D UNORM RGBA8 surfaces and matching TV0 scanout. Full game rendering additionally requires tiled color-buffer handling, Latte command execution, guest vertex/index processing, and guest shader translation.

## Verified update (2026-11-22)

The CPU privilege foundation is now implemented in focused changes: `CpuState`
contains MSR/SRR0/SRR1; supported program-exception entry handles privileged
instruction violations; `mfmsr`, bounded `mtmsr`, SRR0/SRR1 `mfspr`/`mtspr`, and
bounded `rfi` execute through the interpreter path. Translation-enabled MSR
states stop explicitly because the flat `GuestMemory` model has no MMU. Cafe OS
thread-context `srr1` restoration remains deferred for the same reason. CPU
and recompiler fallback tests cover reset, privilege violations, supported
state changes, exception saves, and return behavior. Debug and Release builds
and Debug CTest passed 2/2 after the code changes. No live WWHD compatibility
or performance claim was verified.

The initial rendering audit found that the frontend could present
CPU-visible RGBA8 TV scan-buffer memory but the GX2 HLE did not register
`GX2CopyColorBufferToScanBuffer`. Captured draw/clear commands also did not
produce guest framebuffer pixels. The Vulkan window therefore had a
presentation path without an end-to-end guest rendering path. No live title
run was performed during that audit.

## Rendering update (2026-11-22)

Registered `GX2CopyColorBufferToScanBuffer` using the WUT ABI: a color-buffer
descriptor pointer and a scan-target bitmask. The first supported case is TV0,
matching 720p or 1080p RGBA8 dimensions, and a single-sample 2D RGBA8 source in
linear-aligned tile mode with zero swizzle and a base-level single-slice view.
The HLE validates descriptor sizes, pitch, alignment, source/destination guest
ranges, scan-buffer capacity, and the target before changing destination
memory. It strips source row padding and writes the next scan-buffer ring slot;
the following swap request selects that same slot for frontend snapshot and
Vulkan presentation. Copy and swap events remain ordered in GX2 command state.

CPU HLE tests cover padded pitch, destination bytes, ring-slot selection,
unsupported tiled input and DRC targets, undersized output memory, unmapped
source memory, and copy/swap ordering. Debug and Release builds succeeded, and
Debug CTest passed 2/2. `C:\devkitPro` is absent in this environment, so the
WUT RPX fixture could not be compiled and no live guest or Vulkan-window output
was verified.

## Guest color-buffer clear update (2026-11-22)

`GX2ClearColor` now accepts the same bounded linear-aligned, single-sample 2D
RGBA8 base-level/single-slice color-buffer layout. It validates the descriptor
and full guest-memory range, clamps components to UNORM [0, 1], rejects NaN,
quantizes RGBA8, writes active pixels, and preserves row padding. Successful
clears enter the ordered GX2 command stream; unsupported layouts fail before
guest pixels or clear-success state are changed.

CPU HLE tests verify the guest bytes, padded-row preservation, invalid tiled
layout and NaN rejection, and clear → copy → swap order. Debug and Release
builds succeeded, Debug CTest passed 2/2, and the existing Vulkan texture-upload
smoke test passed on an NVIDIA GeForce RTX 5070 Laptop GPU. That smoke test
uses host-provided texture data and does not verify guest-generated output.
The controlled emulator HLE integration diagnostic is recorded below. A WUT
RPX build remains unavailable because `C:\devkitPro` is not installed.

## GX2 scanout integration diagnostic (2026-11-22)

Added `AffogatoFrontend.exe --gx2-guest-scanout-smoke-test`. The diagnostic
executes PPC `bctrl` call stubs through the interpreter into GX2 HLE for
initialization, TV scan-buffer setup, output enable, two differently colored
clears, copies, and swaps. It verifies the first RGBA8 pixel in both captured
scanouts and that the frames differ, then uploads and presents both scanouts
through the Vulkan swapchain. It passed on the NVIDIA GeForce RTX 5070 Laptop
GPU. The visible display variant showed the second guest clear as a solid green
frame in the Vulkan window; that confirms the controlled color-buffer clear,
copy, and scanout route, not game rendering. It is not a WUT-built guest
program, WWHD rendering, or guest draw test.

The next step is to visually inspect the triangle display variant, then extend
the diagnostic path from its known float2 vertex layout to verified guest
fetch formats and shader execution. Add a WUT-built RPX fixture when
devkitPro/WUT is installed. Current clear/copy support still applies only to
the documented linear RGBA8 subset.

## Diagnostic guest triangle (2026-11-22)

Added a controlled PPC `bctrl` test that calls GX2 initialization,
`GX2SetAttribBuffer`, and `GX2DrawEx` through HLE. It verifies the captured
triangle parameters and vertex-buffer state, reads three float2 positions from
guest memory, and submits those positions to a Vulkan triangle-list pipeline
with a fixed diagnostic fragment shader. The hidden-window smoke test passed
on the NVIDIA GeForce RTX 5070 Laptop GPU. This validates guest geometry
submission only; it is not Latte fetch/shader emulation or game draw execution.
