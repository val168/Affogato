# Affogato subsystem coverage audit

**Audited commit:** `611238b` (`main`, 2026-11-22 project checkpoint).
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
| Raw GPU command ring/PM4 decoder | Missing | GX2 HLE emits typed `Gx2Command` records; no guest ring-buffer packet processor/retirement path. |
| GX2 command ordering | Partial, tested | HLE pending command vector and frontend batch handoff preserve selected operation order; not equivalent to asynchronous Latte FIFO/fence execution. |
| Register shadow | Partial | `Gx2RegisterState` retains selected raw context/resource/sampler state and draw-time snapshots. Not all Latte registers have typed semantics. |
| Color/depth render-target descriptors | Partial | HLE reads and records descriptors, binding and selected registers; broad hardware surface semantics are not provided. |
| Surface format/layout/addressing | Partial | CPU staging and GX2 clear/copy support linear-aligned RGBA8, 1D thin1 RGBA8 microtiles, and zero-swizzle 2D thin1 RGBA8 macrotiles; texture staging also supports selected 8/16/32-bit UNORM formats in 1D thin1. Other macro-tiled R600/Latte addressing and general texture/render-target layouts are absent. |
| Clear/copy/scanout | Implemented and verified subset | Linear-aligned RGBA8 clear/copy/scanout was visibly presented; 1D thin1 and zero-swizzle 2D thin1 RGBA8 clear/copy through HLE are covered by CPU tests. Tiled Vulkan presentation, bank-swizzled, other macro-tiled, scaled, multisample, DRC, and general format behavior are unsupported or unverified. |
| Vertex/index capture | Implemented and verified subset | Draw-time immutable guest snapshots support bounded non-indexed and U16/U32 indexed ranges when opt-in capture is enabled. |
| Vertex fetch | Partial | HLE-generated VTX subset, Float32 scalar/vector and Float16x2 position formats. General fetch ISA, normalized formats, instancing/divisors, and arbitrary attributes are unsupported. |
| Primitive assembly | Diagnostic partial | Diagnostic route supports triangle lists, strips, fans plus selected index endian forms; not normal guest rasterization. |
| Guest draw dispatch | Diagnostic only | Ordered captured commands can reach the fixed diagnostic host pipeline when explicitly opted in. Normal guest shader/render-target semantics are not executed. |
| Vertex/pixel/geometry shader decode | Missing | GX2 program descriptors/words are snapshotted; [`12_latte_shader_isa.md`](12_latte_shader_isa.md) records current capture boundaries, reference sources, and decoder acceptance requirements. Latte instruction decode/execution/translation is absent. |
| Texture/sampler decode | Partial | Selected register descriptors are decoded; bounded linear-aligned, 1D thin1, and zero-swizzle 2D thin1 CPU staging exists for selected 8/16/32-bit UNORM formats. Guest shader sampling, depth tiling, mip/array/compressed staging, and other layouts remain absent. |
| Depth/stencil/blend/raster state | Partial capture | Some HLE register state is recorded; Vulkan state mapping and guest draw behavior are absent. |
| GPU memory coherence | Blocked | Requires an explicit ownership/ordering model between emulator worker, renderer thread, host Vulkan resources, guest reads and scanout. |
| GPU synchronization/fences | Partial capture | Flush/draw-done/fence events are represented; actual completion, wait, and memory visibility semantics are not implemented. |
| Vulkan backend | Implemented and verified backend subset | Window, swapchain, scanout upload, diagnostic shader and texture/clear smoke routes work; does not imply Latte execution. |

## GX2 and Cafe OS

| Feature | Status | Evidence / limitation |
| --- | --- | --- |
| GX2 exports and ABI | Partial | `gx2_hle.cpp` registers numerous functions with argument/descriptor handling; callback presence does not establish full behavior for all modes. Compare each path to WUT declarations and tests. |
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
  guest-backed snapshots with diagnostic shaders. The Float16x2 extension's
  hidden smoke test passed; its visual rerun was not verified.
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
2. Audit frontend worker/UI synchronization and give command execution, guest
   memory visibility, renderer submission, and readback explicit ownership.
3. Build a documented Latte register/packet model (or prove which typed HLE
   events are sufficient for the target compatibility model).
4. Add a shared, independently tested surface-address/format subsystem before
   attempting guest draw writes or additional tiled scanout formats.
5. Decode a narrow shader family from verified binaries into a small IR and
   test values; then translate that IR to validated SPIR-V.
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
