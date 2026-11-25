# Affogato subsystem coverage audit

**Audited commit:** `e69c35d` (`main`, 2026-11-25 project checkpoint).
This is an implementation audit, not a title-compatibility claim. A registered
callback, captured register, or successful backend smoke test does not by itself
mean the guest behavior is implemented.

## Status definitions

- **Implemented and verified** — behavior has a focused deterministic test or
  observed controlled integration result.
- **Implemented, testing incomplete** — code exists, but meaningful coverage or
  behavioral equivalence is missing.
- **Partial** — only a documented subset works; unsupported cases are explicit
  or remain unimplemented.
- **Missing** — no execution path exists.
- **Blocked** — depends on a missing subsystem.
- **Uncertain** — available evidence does not establish correct behavior.

## Espresso CPU

| Feature | Status | Evidence / limitation |
| --- | --- | --- |
| Integer instructions | Partial | Decoder/interpreter and JIT cover a substantial subset; unknown encodings stop. See `decoder.cpp`, `interpreter.cpp`, `recompiler.cpp`, CPU/recompiler tests. |
| Branch instructions | Partial | Common direct/conditional/indirect paths exist; exact workload equivalence is incomplete. `development-progress.md` records title stop at a null indirect target. |
| CR/XER | Partial | Explicit state and selected operations are tested; family-wide instruction coverage is not complete. |
| Floating point | Partial | FPR/FPSCR state and selected operations exist; full IEEE/750CL corner coverage is absent. |
| Paired-single | Partial | Separate PS1 lanes and selected instructions/GQR interactions exist; not a complete extension. |
| Memory operations | Partial | Big-endian bounded `GuestMemory`, aliases, sparse regions, and reservation support exist; no MMU translation and only decoded instruction subset. |
| Atomics/reservations | Partial | LL/SC reservation state and tests exist; full multicore reservation/coherence semantics are not established. |
| Cache operations | Partial/uncertain | Selected cache-related instructions may be represented, but system-wide data/instruction cache visibility is not modeled. |
| Synchronization | Partial | Selected barriers and Cafe synchronization HLE exist; complete hardware ordering and multicore interactions are absent. |
| Privileged state | Partial, verified subset | MSR/SRR0/SRR1, bounded `mfmsr`/`mtmsr`/SPR/rfi and program privilege exception path are tested. Translation-enabled MSR states stop because MMU is absent. |
| Exceptions | Partial | Program exception for supported privilege violations exists; general illegal, alignment, data/instruction storage and external interrupt handling are incomplete. |
| MMU/address translation | Missing | Guest memory is flat/aliased; IR/DR translation is explicitly unsupported. |
| Three-core Espresso scheduling | Missing | Current `EspressoCore` runtime models one core; no coherent inter-core scheduler. |

## Latte GPU and graphics

| Feature | Status | Evidence / limitation |
| --- | --- | --- |
| Raw GPU command ring/PM4 decoder | Partial | A bounded parser handles Type-0 config/context/control writes, Type-2 filler, Type-3 NOP, register-set packets, `INDEX_TYPE`, `NUM_INSTANCES`, and unpredicated auto, memory-indexed, and immediate-indexed draws with supported U16/U32 index formats in direct-call HLE. The supported draw-register subset persists across successful lists and recognized zero-length context restores, with failed-list rollback. Predicate-bit draws, GPU address translation, ring processing, indirect-buffer recursion, and retirement remain unsupported; see [`15_latte_command_processor.md`](15_latte_command_processor.md). |
| GX2 command ordering | Partial, tested | HLE pending command vector and frontend batch handoff preserve selected operation order; not equivalent to asynchronous Latte FIFO/fence execution. |
| Register shadow | Partial | `Gx2RegisterState` retains selected raw context/resource/sampler state and draw-time snapshots. `GX2SetVertexShader` now captures the program start/resource words, primitive/output configuration, declared output IDs, CF offset, stream-out controls, semantic table, and reuse controls; geometry mode selects the ES program range and ring item size. Tests check the command sequence, count bounds, stride conversion, and draw-snapshot persistence. These GX2 effects are reference-derived from WUT/Decaf and related-generation AMD register maps, not hardware-verified; the shadow does not execute guest shader code. Other special Latte register semantics are not generally modeled. |
| GX2 shader resource mode | Partial state model | `GX2SetShaderModeEx` accepts uniform-register/block modes and queues an ordered state operation. The register shadow applies masked `SQ_CONFIG`, GPR/thread/stack allocation, and VGT-mode changes, preserving known prior bits and marking unobserved bits unknown. Geometry/compute modes fail explicitly. This is not physical GPU programming; Latte-specific mode semantics, block-mode bit behavior, ring resources, and synchronization remain unverified. See [`12_latte_shader_isa.md`](12_latte_shader_isa.md). |
| Color/depth render-target descriptors | Partial | HLE reads and records descriptors, binding and selected registers; broad hardware surface semantics are not provided. |
| Surface format/layout/addressing | Partial | CPU staging and GX2 clear/copy support linear-aligned RGBA8, 1D thin1 RGBA8 microtiles, and zero-swizzle 2D thin1 RGBA8 macrotiles; texture staging also supports selected 8/16/32-bit UNORM formats in 1D thin1 and zero-swizzle 2D thin1. Other macro-tiled R600/Latte addressing and general texture/render-target layouts are absent. |
| Clear/copy/scanout | Implemented and verified subset | Linear-aligned RGBA8 clear/copy/scanout was visibly presented; 1D thin1 and zero-swizzle 2D thin1 RGBA8 clear/copy through HLE are covered by CPU tests. Tiled Vulkan presentation, bank-swizzled, other macro-tiled, scaled, multisample, DRC, and general format behavior are unsupported or unverified. |
| Vertex/index capture | Implemented and verified subset | Draw-time immutable guest snapshots support bounded non-indexed and U16/U32 indexed ranges when opt-in capture is enabled. |
| Vertex fetch | Partial, controlled CPU path verified | HLE-generated VTX subset, Float32 scalar/vector, Float16x2, and unsigned UNORM8 scalar/x2/x4 formats with explicit no-swap. `GX2SetVertexShader` writes its descriptor's semantic clear/table values. In the opt-in diagnostic path, draw-time fetch input feeds the supported CPU vertex-shader subset using an explicitly selected GPR base of zero. AMD's related-generation guide says this base is typically zero, but arbitrary Latte values remain unknown. Decaf's fetch generator corroborates the local word pattern (`VTX_INST_SEMANTIC` plus `attrib.location` in word-1 low bits through its GPR field alias), but this does not resolve how Latte interprets the overlap; AMD references also disagree on semantic field width (R6xx: 8, R700 ISA: 9). System-generated inputs, general fetch ISA, other normalized formats, instancing/divisors, and arbitrary attributes remain unsupported. |
| Uniform register uploads | Partial capture and CPU subset integration | `GX2SetVertexUniformReg` and `GX2SetPixelUniformReg` validate and snapshot big-endian guest dwords into separate stage constant banks, retained in ordered commands and draw register snapshots. Ordinary vector-register offsets are supported within the 256-entry bank; loop-constant offsets are explicitly rejected. The vertex CPU test subset maps known dwords from a draw snapshot to absolute constant selectors; unknown components fail on read. This is not normal draw dispatch or Vulkan binding. Uniform blocks remain unimplemented. |
| Latte shader control-flow and clause records | Partial | `latte_shader_decoder.hpp` decodes fixture-verified `CALL_FS`, `EXPORT`, `EXPORT_DONE`, one `ALU` header variant, bounded raw slots, normal export metadata, and no-literal OP2 groups using LAST. Absolute-GPR normal exports can be materialized as typed CPU-side values, but are not applied to graphics stage outputs. Literal clauses and OP3 groups remain unsupported; branch/loop decoding or execution is absent. |
| Shader-stage interface descriptors | Metadata decoding only | `latte_shader_interface.hpp` decodes captured VS `SPI_VS_OUT_ID` words and `VS_PER_COMPONENT`, plus the active PS `SPI_PS_INPUT_CNTL` fields and default values. It does not match producer outputs to fragment inputs, synthesize defaults, choose interpolation, or create SPIR-V stage interfaces. |
| Primitive assembly | Diagnostic partial | Diagnostic route supports triangle lists, strips, fans, and draw-time U16/U32 LE/BE index expansion through `decode_latte_draw_vertex_indices()`; indexed format, first-vertex, base-vertex, invalid snapshot, restart, and overflow cases are CPU-tested. One explicitly selected diagnostic policy feeds these indices through the existing CPU vertex-shader subset; rasterization remains outside normal guest execution. |
| Guest draw dispatch | Diagnostic only | Ordered captured commands can reach the fixed diagnostic host pipeline when explicitly opted in. Normal guest shader/render-target semantics are not executed. |
| Vertex/pixel/geometry shader execution | Experimental CPU subset; explicit diagnostic use | Four-slot no-literal X/Y/Z/W OP2 groups support MOV; floating ADD, IEEE MUL, DX10 MAX/MIN; and integer AND/OR/XOR/NOT/ADD/SUB/MAX/MIN/comparison operations with absolute-GPR, inline-constant, absolute constant-file, or prior-PV sources. A straight-line CPU executor accepts sparse constant vectors and sequences ALU clauses with normal exports through a typed `LatteShaderIrProgram` containing OP2 clauses and normal exports. The controlled Vulkan triangle diagnostic executes its captured guest VS subset per fetched vertex (GPR base zero), emits bounded MOV-to-position SPIR-V, and passes position XY to a fixed host diagnostic fragment shader. A separate constant-only fragment subset translates one synthetic guest pixel shader and passes Vulkan readback. `latte_shader_interface.hpp` also decodes the pixel descriptor's active `SPI_PS_INPUT_CNTL` metadata, but does not link inputs to VS exports or execute interpolants. The explicit smoke tests verify diagnostic or synthetic fixture pixels; they do not execute shaders in ordinary guest draws or write a guest render target. The generic executor rejects `CALL_FS`; the vertex-specific CPU path permits only a canonical leading call after separately processing Affogato's captured fetch subset. Other call forms and general control flow remain unsupported. Floating operations implement source sign modifiers and output scaling; DX10 min/max preserve the manual's ordered selection behavior. MOV applies source-0 absolute/negate as binary32 sign-bit operations; integer operations reject those modifiers. Predicate selection, predicate/mask updates, destination clamp, and nonzero bank swizzles are rejected. Coverage is based on R700-family semantics and assembled fixtures, not Latte hardware validation. Kcache, literals, relative constant addressing, general PS execution, texture execution, general SPIR-V translation and normal guest draw integration remain absent. |
| Texture/sampler decode | Partial | Selected register descriptors are decoded; bounded linear-aligned, 1D thin1, and zero-swizzle 2D thin1 CPU staging exists for selected 8/16/32-bit UNORM formats. Guest shader sampling, depth tiling, mip/array/compressed staging, and other layouts remain absent. |
| Depth/stencil/blend/raster state | Partial capture | Some HLE register state is recorded; Vulkan state mapping and guest draw behavior are absent. |
| GPU memory coherence | Blocked; source-order issue verified | `GX2CopyColorBufferToScanBuffer` reads and writes guest memory inside the CPU HLE call. An opt-in CPU checkpoint and diagnostic-only worker/UI acknowledgement now stop after a GX2 draw is captured and after the UI consumes its batch. This does not wait for GPU completion or write target pixels back. The diagnostic route still targets only the host swapchain. Add verified render-target writes and make them guest-visible before dependent CPU HLE reads; see [`11_graphics.md`](11_graphics.md#guest-draw-guest-memory-and-scanout-ordering). |
| GPU synchronization/fences | Partial capture | Flush/draw-done/fence events are represented; actual completion, wait, and memory visibility semantics are not implemented. |
| Vulkan backend | Implemented and verified backend subset | Window, swapchain, scanout upload, diagnostic shader and texture/clear smoke routes work. The opt-in triangle test copies the rendered swapchain image to host memory and checks a diagnostic pixel. This does not imply normal Latte execution or guest render-target writes. |

## GX2 and Cafe OS

| Feature | Status | Evidence / limitation |
| --- | --- | --- |
| GX2 exports and ABI | Partial | `gx2_hle.cpp` registers numerous functions with argument/descriptor handling; callback presence does not establish full behavior for all modes. Compare each path to WUT declarations and tests. |
| GX2 display-list lifecycle/submission | Partial | `GX2DirectCallDisplayList` and `GX2CallDisplayList` validate/copy one guest stream and queue the supported PM4 subset in order. Nested PM4 indirect-buffer packets, the queued GPU's asynchronous memory visibility, begin/end recording, copy/query APIs, guest shader execution, and GPU retirement remain unsupported. |
| GX2 initialization/state | Partial | Selected runtime defaults, register shadows, surfaces, shaders and scanout state are modeled; no hardware command processor. |
| Guest pointer/structure validation | Implemented in tested paths | HLE uses checked guest memory and big-endian field reads for implemented structures; audit must remain function-specific. |
| Guest threads/synchronization | Partial | Selected coreinit thread/context/mutex/semaphore/event behavior supports a single-core HLE runtime; OS scheduler/interrupt behavior is incomplete. |
| Heap and filesystem | Partial | Selected allocator, file and directory calls exist; this is not Cafe OS or IOSU completeness. |
| Service initialization | Partial | Coreinit/GX2 and selected `nn_*` HLE modules initialize through emulator registration. Full IOSU boot/services are absent. |
| Guest error reporting | Partial | Unknown imports/instructions and HLE execution errors can stop with diagnostics; some asynchronous/runtime errors remain unmodeled. |
| RPX/ELF loading | Partial, tested | Big-endian ELF and bounded RPX sections/implemented relocation set are covered by tests/fixture; installed encrypted title loading and general module runtime are absent. |

## Validation assets and current graphics status

- `affogato_cpu_tests` and `affogato_recompiler_tests` are registered under
  CTest. The last reported configured builds/CTest are 2/2 per build tree; each
  code change must rerun the applicable checks.
- `tests/fixtures/minimal_wut_rpx` is a minimal loader fixture; a built WUT GX2
  end-to-end RPX is not available in this environment because devkitPro is not
  installed (`C:\devkitPro` was absent in the recorded audit).
- `--gx2-guest-scanout-smoke-test` validates guest HLE clear/copy/scanout bytes
  and Vulkan submission. It does not execute a guest draw.
- `--gx2-guest-triangle-smoke-test` exercises controlled PPC/HLE draws and
  guest-backed snapshots with diagnostic shaders. The SPIR-V variant copies
  the swapchain image to host memory and checks an interior pixel for the
  diagnostic fragment color. This verifies diagnostic raster output, not
  visible presentation, guest color-buffer writes, guest pixel shaders, or
  WWHD. An earlier version of the
  visible `--gx2-guest-triangle-display-test` was observed on the NVIDIA
  GeForce RTX 5070 Laptop GPU; its raw-attribute diagnostic geometry remained
  responsive after resizing and closed cleanly with no Vulkan validation
  messages. The current test instead executes a bounded CPU guest-VS subset
  before the fixed host diagnostic shaders. Its hidden smoke passes, but the
  visible window has not been reverified after that change. Neither version
  establishes general Latte shader execution or WWHD output.
- Recorded WWHD/BOTW startup runs did not reach guest draw submission. WWHD's
  shader/surface requirements and game-rendered output are therefore unknown.
- No current performance comparison establishes gains or regressions for the
  graphics changes. Existing recompiler benchmark infrastructure is separate
  from GPU rendering measurement.

## High-value next dependency chain

1. The diagnostic backend now consumes a typed borrowed `Gx2DrawSnapshotView`
   for a draw index. Tests verify the view's captured fields line up and later
   register writes do not alter earlier draw snapshots. The view is invalidated
   by a later `Gx2RegisterState::apply`; it is not a cross-thread ownership
   primitive.
2. Implement the bounded offscreen-target prototype in
   [`11_graphics.md`](11_graphics.md#bounded-offscreen-target-prototype):
   capture supported RT0 RGBA8 guest bytes at HLE draw time, load them into an
   offscreen Vulkan attachment, render only through the explicit diagnostic
   policy, read back after synchronization, and hand bytes to the paused
   emulator worker before dependent HLE copies resume. The existing diagnostic
   UI acknowledgement alone is not GPU completion or memory visibility.
3. Extend the PM4 packet model with verified draw/state operations and
   indirect-buffer address handling while preserving guest order and owned
   command bytes.
4. Add a shared, independently tested surface-address/format subsystem before
   attempting guest draw writes or additional tiled scanout formats.
5. Compare diagnostic readback pixels with CPU reference output, then expand
   the typed IR/backend for verified instruction families.
6. Feed a controlled WUT/Decaf-derived guest draw through captured vertex,
   shader, target, copy, and scanout state; verify pixels by readback.
7. Rerun title workloads and collect actual draw/shader/surface evidence before
   prioritizing WWHD-specific blockers.

## Primary research references

- [WiiUBrew Wii U hardware](https://wiiubrew.org/wiki/Hardware),
  [Espresso](https://wiiubrew.org/wiki/Hardware/Espresso),
  [GX2](https://wiiubrew.org/wiki/Hardware/GX2), and
  [Latte](https://wiiubrew.org/wiki/Hardware/Latte).
- [OpenPOWER ISA specifications](https://openpowerfoundation.org/specifications/isa/)
  and IBM 750CL manual: `https://fail0verflow.com/media/files/ppc_750cl.pdf`.
- [WUT](https://github.com/devkitPro/wut), its `include/gx2/` declarations,
  and `libraries/libwhb/src/gfx_shader.c` helper flow.
- [Cemu](https://github.com/cemu-project/Cemu):
  `src/Cafe/HW/Latte/Core/LatteCommandProcessor.cpp`, `LatteShader.cpp`,
  `LatteAsyncCommands.cpp`, `src/Cafe/HW/Latte/Renderer/Vulkan/`, and
  `src/Cafe/OS/libs/gx2/GX2_Surface_Copy.cpp`.
- [Decaf](https://github.com/decaf-emu/decaf-emu): `src/libgpu/` GPU library;
  [Decaf Wii U tests](https://github.com/decaf-emu/wiiu-tests). GPL-3.0
  reference only; no code copied.
- [AMD R600 ISA manual](https://www.x.org/docs/AMD/old/r600isa.pdf): related
  GPU-family reference; Wii U applicability requires cross-checking.
