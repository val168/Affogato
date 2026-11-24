# GX2 and Latte graphics research

**Audit basis:** Affogato commit `611238b` and the source tree at that commit.
This note distinguishes guest/API state capture from execution on the host GPU.

## Purpose and architecture

GX2 is the guest-facing graphics API. Affogato currently handles a selected
set of GX2 calls in HLE, retains an ordered command stream and register shadow,
and has a Vulkan backend for scanout and explicitly opted-in diagnostic draws.
The intended long-term boundary is guest-visible GX2/Latte behavior translated
to Vulkan; the current command records are not a Latte command processor.

The relevant local flow is:

```text
PPC guest -> GX2 HLE -> ordered Gx2Command records
          -> draw-time register/state shadow -> opt-in diagnostic Vulkan draw
          -> (separate path) CPU-visible TV scan buffer -> Vulkan presentation
```

`src/cpu/espresso/gx2_hle.cpp` owns GX2 ABI decoding and guest-visible memory
effects. `src/gpu/gx2_register_state.hpp` applies recorded state in order and
snapshots state at each draw. `src/gpu/vulkan/vulkan_renderer.cpp` implements
diagnostic draw expansion, Vulkan submission, scanout presentation, and texture
upload. `src/frontend/main.cpp` drains GX2 batches and coordinates the UI,
worker, scanout, and optional diagnostic policy.

## Documented and cross-checked behavior

### Documented interface facts

- WUT defines the public GX2 structures, enums, and call signatures used by
  homebrew. Its `GX2InitFetchShaderEx` consumers show that fetch-shader state is
  constructed from attribute location, buffer slot, offset, format, and endian
  mode. WUT is an interface reference; it does not specify every hardware
  detail.
- WiiUBrew describes Latte as the Wii U GPU and GX2 as its guest API/library
  layer. This does not imply that API-level command capture is equivalent to
  executing the GPU's command stream.
- AMD R600-family documents are useful for shader/fetch encoding research, but
  each proposed encoding still needs cross-checking against Wii U-specific
  evidence or a known reference implementation.

### Cross-verified emulator architecture findings

- Cemu has a Latte command processor, shader/cache/decompiler layers, texture
  and surface handling, and backend-specific renderers. Relevant current
  source locations include `src/Cafe/HW/Latte/Core/LatteCommandProcessor.cpp`,
  `LatteShader.cpp`, `LatteAsyncCommands.cpp`, and
  `src/Cafe/HW/Latte/Renderer/Vulkan/`. Its breadth is evidence for subsystem
  boundaries, not proof that every implementation detail applies to this
  experimental emulator.
- Decaf separates GPU work under `src/libgpu/` and maintains the
  `decaf-emu/wiiu-tests` repository for reproducible hardware/behavior tests.
  Its emulator and tests are GPL-3.0; Affogato uses them as behavioral and
  architectural references only. No code is copied by this work.
- WUT's public GX2 declarations and helpers provide independent ABI and enum
  evidence for controlled fixtures. The WUT source is BSD-2-Clause.

## Verified Affogato behavior

### GX2 HLE and state capture

`Gx2RuntimeState` tracks selected TV/DRC scanout, render-target descriptors,
vertex/index bindings, fetch/program shader bindings, selected sampler/texture
register words, blend/depth/scissor/viewport registers, and pending commands.
The implementation is not a complete GX2 export set. Implemented callbacks
must be evaluated by their actual memory/state behavior, not by registration.

Draw records preserve draw arguments and, when the host launch option
`capture_draw_vertex_data` is enabled, copy vertex and index bytes during HLE
execution. That avoids the frontend later reading mutable guest memory for
these captured diagnostic inputs. Capture is disabled by default to avoid
allocations in normal guest execution. `Gx2RegisterState::apply` keeps per-draw
snapshots for vertex bindings, fetch/program shaders, resource words, color and
depth target bindings, and selected raw registers.

### Verified clear/copy/scanout subset

The controlled clear/copy path is a synchronous CPU-memory path for a restricted
linear-aligned, single-sample, 2D RGBA8, base-level/single-slice color buffer
and matching TV0 scanout. `GX2ClearColor` writes guest color-buffer bytes;
`GX2CopyColorBufferToScanBuffer` removes row pitch padding and writes a scanout
ring slot; a later swap selects it for presentation. A controlled PPC-to-HLE
display fixture previously showed guest-origin green pixels in the Vulkan
window. This does not establish draw execution, tiled surfaces, DRC output, or
WWHD rendering.

### Diagnostic draws

The hidden diagnostic path and the visible Float32 tests have exercised
single-instance, base-instance-zero triangle lists, indexed U16/U32 input in
the supported endian modes, strips, and fans. The latest committed feature
adds a narrow Float16x2 fetch/position case (`611238b`); its Debug/Release builds,
CTest, and hidden Vulkan smoke tests passed. The final visible-window rerun for
that Float16x2 update was not verified. The renderer decodes only the
HLE-generated fetch subset needed to select location 0, then expands captured
positions into transient host vertices. The fixed diagnostic host shaders draw
to the host swapchain. These operations do not execute arbitrary guest
attributes or shaders and do not modify the guest render target.

## Important state, formats, and invariants

- `Gx2DrawCall` holds primitive mode, element count, indexed flag, index type,
  element address/offset, instance count, and base instance.
- `Gx2ColorBufferState` and `Gx2DepthBufferState` retain guest descriptor
  fields including dimensions, format, sample mode, pitch, tile mode, swizzle,
  mip offsets, view, and register words. A descriptor being captured does not
  mean its surface addressing is implemented.
- The current diagnostic fetch decoder accepts HLE-generated, non-instanced
  VTX programs and Float32x1/x2/x3/x4 plus Float16x2 in its supported encoding
  combinations. Draw rendering uses two position components. Arbitrary guest
  fetch programs, normalized integer attributes, divisors, and general
  multi-attribute vertex layouts are not supported.
- Diagnostic index expansion supports bounded captured U16/U32 streams in
  little- and big-endian forms. Primitive restart, instancing, and non-triangle
  diagnostic topology are rejected.
- `latte_texture_resource.hpp`, `latte_sampler.hpp`, and
  `latte_texture_staging.hpp` decode selected register words and a constrained
  linear 2D texture subset. This is not general tiled texture sampling.
- GPU fences, flushes, and draw-done calls are represented in command history;
  this is not proof of a hardware-equivalent queue/fence implementation.

## Missing functionality and dependencies

| Capability | Current status | Main dependency |
| --- | --- | --- |
| Raw Latte PM4/ring command processor | Missing; HLE creates typed records | packet/ring ABI, retirement and wait semantics |
| Guest draw route to Vulkan | Diagnostic-only opt-in route exists | ordinary queue integration and unsupported-state policy |
| Fetch shader | Narrow HLE-generated format decoder | verified VTX variants, component/normalization semantics |
| General vertex/index/instance handling | Partial snapshots and diagnostic expansion | fetch semantics, offset/base-instance rules, state capture |
| Latte vertex/pixel shader execution | Missing; program words are captured | verified binary decoder, IR, translation and test corpus |
| Guest render target for draws | Missing; draw diagnostic targets swapchain | surface addressing, Vulkan image cache, guest visibility/readback |
| Color clear/copy | Verified restricted linear RGBA8 path | tiling, additional formats, resolve and sync semantics |
| Depth/stencil and blend pipeline | Registers/descriptors are partly captured | register interpretation and Vulkan state mapping |
| General texture sampling | Resource/sampler word capture and constrained staging | tiled addressing, format coverage, cache coherence |
| GPU ordering/coherence | Typed events and synchronous CPU HLE writes | worker/render ownership, flush/fence and alias rules |
| TV/DRC scanout | TV subset verified; DRC incomplete | separate target semantics and formats |

## Unknown or uncertain behavior

- The local source does not yet execute Wii U PM4 command buffers. A future
  command processor must first establish which software-visible GX2 operations
  require packet-level emulation and which can remain accurately represented
  by HLE state updates.
- Current WWHD runs documented in `docs/development-progress.md` stop before a
  guest draw is captured. Therefore WWHD's actual surface layouts, vertex
  formats, shader ISA mix, and synchronization requirements remain unobserved
  in Affogato.
- Cemu and Decaf differ in architecture and compatibility assumptions. Any
  behavior adopted from them needs cross-checking against WUT declarations,
  hardware docs, fixtures, and the exact Affogato call path.
- Linear-aligned and tiled layout names/encoding must not be conflated. A
  format or layout is supported only when its address calculation and byte
  interpretation are independently tested.

## Proposed implementation architecture

1. Keep GX2 HLE responsible for ABI validation and immutable draw-time guest
   inputs. Keep the renderer independent from guest pointer dereferences.
2. Give each draw one typed, immutable-by-contract view that binds the captured
   command to its draw-time state, instead of relying on frontend-time live
   state or unrelated defaults. Keep current storage allocation behavior
   bounded; consider changing its representation only when consumers need it.
3. Add typed Latte register decoding and a bounded command executor before
   treating raw register state as executable pipeline state.
4. Implement a verified surface-address layer shared by clear, texture staging,
   render-target writes, copy/resolve, and readback; reject unsupported modes.
5. Add shader decoding into a validated IR, then a shader backend and cache.
   Keep diagnostic shaders behind explicit test policy.
6. Give guest-memory writes and Vulkan resources an explicit synchronization
   and ownership model; draw writes must be visible to subsequent guest copy,
   texture reads, and scanout in command order.

This is a proposed direction, not existing functionality. The present frontend
worker/UI threading model must be audited before introducing cross-thread
render-target writes.

## Test strategy

- Keep CPU-only tests for GX2 ABI layouts, register state transitions, draw-time
  snapshot immutability, bounds/overflow, endian handling, and format decoders.
- Use WUT/Decaf-style small guest workloads for API-level operations and keep
  deterministic synthetic fixtures for command processing.
- Validate shader decoders instruction-by-instruction and compare known
  outputs; compiling SPIR-V alone is insufficient.
- Use Vulkan offscreen/readback tests for attachment and pixel behavior where
  available. Keep visible-window checks separate and claim visual verification
  only when observed.
- For title runs, record draw counts, descriptors, shaders, buffer ranges,
  commands, and frame bytes; instruction count alone is not a rendering test.

## Sources and reference locations

- [WiiUBrew GX2 hardware](https://wiiubrew.org/wiki/Hardware/GX2),
  [Latte](https://wiiubrew.org/wiki/Hardware/Latte), and
  [gx2.rpl](https://wiiubrew.org/wiki/Gx2.rpl): hardware/API context.
- [WUT GX2 headers](https://github.com/devkitPro/wut/tree/master/include/gx2)
  and [WUT GX2 helper source](https://github.com/devkitPro/wut/tree/master/libraries/libwhb/src):
  public ABI, enums, and homebrew setup patterns.
- [Cemu Latte command processor](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteCommandProcessor.cpp),
  [shader state/cache](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteShader.cpp),
  [async GPU work](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteAsyncCommands.cpp),
  and the [Vulkan backend](https://github.com/cemu-project/Cemu/tree/main/src/Cafe/HW/Latte/Renderer/Vulkan).
- [Decaf emulator](https://github.com/decaf-emu/decaf-emu), especially
  `src/libgpu/` and `src/libgpu/CMakeLists.txt`, for GPU subsystem boundaries;
  [Decaf Wii U tests](https://github.com/decaf-emu/wiiu-tests) for reproducible
  guest test approaches. Both repositories are GPL-3.0; this note records
  locations and observations, not copied implementation.
- AMD [R600 ISA manual](https://www.x.org/docs/AMD/old/r600isa.pdf): related
  GPU-family reference, not a complete Latte specification.
