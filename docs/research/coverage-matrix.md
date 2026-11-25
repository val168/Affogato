# Subsystem Coverage Matrix

This matrix summarizes current source coverage and evidence. "Implemented and
verified" applies only to the bounded behavior named in the row; it does not
imply title compatibility or full hardware conformance. See
[`ARCHITECTURE.md`](../ARCHITECTURE.md) for data flow and
[`DEVELOPMENT_STATUS.md`](../DEVELOPMENT_STATUS.md) for recent test results.

## Espresso CPU and memory

| Area | Status | Verified boundary and remaining work |
| --- | --- | --- |
| Integer, branch, CR, and memory instruction families | Partially implemented | `decoder.cpp` and `interpreter.cpp` cover a growing subset; unsupported encodings stop execution. Test coverage is instruction-specific, not full ISA conformance. |
| Floating point and paired single | Partially implemented | `CpuState` represents FPRs, PS lanes, GQRs, and FPSCR; arithmetic and edge-case coverage remains incomplete. |
| Atomics and reservations | Partially implemented | `GuestMemory` tracks reservation/write generations and selected LL/SC behavior; multicore reservation semantics are absent. |
| Privilege and exceptions | Partially implemented | MSR, SRR0/SRR1, bounded `mfmsr`/`mtmsr`/`mfspr`/`mtspr`/`rfi`, and program exceptions exist. Other exception classes and translated address execution are unsupported. |
| MMU and address translation | Missing | Guest memory is flat with aliases; enabling instruction/data translation stops explicitly. |
| JIT | Partially implemented | AsmJit blocks, fallback, invalidation, and selected interpreter/JIT differential tests exist. Native lowering and differential coverage are incomplete. |
| Guest memory and loaders | Partially implemented | Bounded big-endian access, sparse regions, aliases, RPX/ELF loading, selected relocations, and loader fixtures exist. IOSU/decryption and general multi-module loading are absent. |

## Latte, GX2, and Vulkan

| Area | Status | Verified boundary and remaining work |
| --- | --- | --- |
| GX2 HLE and ordered state capture | Partially implemented | `gx2_hle.cpp` and `Gx2RegisterState` capture selected API calls, register writes, resources, draw state, and immutable draw-time snapshots. This is not complete GX2/PM4 execution. |
| Surface/texture addressing | Partially implemented | Explicit staging supports selected UNORM formats and linear-aligned, 1D thin1, and narrow 2D thin1 layouts. Other tiling, depth, compressed, mip, array, and swizzle cases reject or remain absent. |
| Guest clear, color copy, and scanout | Implemented and verified for a narrow subset | Matching-size single-sample RGBA8 paths update guest memory and scan buffers; controlled guest scanout tests and display test exist. Unsupported descriptors fail explicitly. |
| GX2 draw capture | Implemented and verified as capture | Selected draw calls preserve vertex/index/program/register/target snapshots. Normal guest draw execution does not follow from capture. |
| Diagnostic draw rendering | Implemented and verified in controlled tests | Bounded triangle topologies, selected vertex formats, optional Latte vertex SPIR-V, fixed diagnostic fragment shader, or the separate constant-only Latte fragment subset are covered by readback tests. This remains opt-in diagnostic output, not ordinary guest rendering or Latte hardware validation. |
| General vertex fetch | Partially implemented | A narrow HLE-generated fetch subset maps selected attributes into shader input registers. Arbitrary guest fetch programs, system inputs, and general instance behavior are absent. |
| Latte shader decoding/execution | Experimental subset | Structural CF/ALU decoding and CPU/SPIR-V subsets cover selected MOV, arithmetic, comparisons, integer logic, min/max, and conditional moves. The fragment path only constant-folds inline-constant MOV groups before MRT0 export, with verified component swizzles and literal zero/one output selectors. Literals, general control flow, textures, full interfaces, and many operations are missing. See [`12_latte_shader_isa.md`](12_latte_shader_isa.md). |
| Guest pixel shaders and texture sampling | Narrow controlled fragment subset verified | A synthetic GX2 pixel shader captured through `GX2SetPixelShader` and draw-time state produces validated fragment SPIR-V and magenta Vulkan readback. Guest fragment inputs, non-constant ALU, other targets, and texture sampling are unsupported. |
| Guest render-target draw writes | Diagnostic route with guest constant fragment subset | The opted-in diagnostic route renders supported triangles into a temporary Vulkan image, reads pixels back into the captured color-buffer layout, and the existing scanout path presents them. It is not normal Latte draw execution; general pixel shaders, target formats, and synchronization remain unsupported. |
| GPU command processor and synchronization | Partially implemented | GX2-originated typed commands are applied in order to a CPU shadow; general PM4 packet execution, asynchronous GPU ordering, and hardware fence behavior are absent. |
| Vulkan presentation | Implemented and verified | Swapchain presentation and guest scanout are covered. Diagnostic pixel readback validates selected generated vertex shaders with a fixed fragment shader and the constant-only Latte fragment subset; no visible window capture was made for the new mode. |

## Cafe OS and runtime

| Area | Status | Verified boundary and remaining work |
| --- | --- | --- |
| HLE dispatch and imported symbols | Partially implemented | Known RPX imports route to registered handlers; unknown imports stop with diagnostics. Coverage is title- and service-incomplete. |
| Threads and synchronization | Partially implemented | A synthetic initial thread, selected thread APIs, mutex helpers, and bounded synchronization behavior exist. Scheduler, multicore, and full lifecycle behavior are incomplete. |
| Filesystem, heaps, and services | Partially implemented | Selected coreinit and `nn_*` service behavior exists; complete Cafe OS/IOSU and title filesystem semantics are absent. |
| Guest application execution | Partially implemented | Local RPX diagnostics execute in interpreter or opt-in JIT with bounded traces. The current WWHD run stops at CIA zero after 6,081,749 instructions and reaches no draw/swap HLE calls. |

## Highest-value next work

The current highest-value graphics dependency is to make ordinary draw
execution consume captured state and produce guest-visible render-target
results without the explicit diagnostic policy or bypassing guest surface
layout rules. Expand fragment translation only with documented operations and
real shader fixtures. Separately, investigate the reproduced WWHD null
indirect-call target from its guest memory/runtime state; do not infer or
fabricate a callback. Neither path currently establishes WWHD rendering.

## Primary implementation references

- CPU and loader: `src/cpu/espresso/{cpu_state,decoder,interpreter,guest_memory,rpx_loader}.hpp/.cpp`
- JIT: `src/cpu/espresso/recompiler.hpp` and `recompiler.cpp`
- Cafe OS and GX2 HLE: `src/cpu/espresso/cafe_os_hle.cpp` and `gx2_hle.cpp`
- Draw/surface/shader subsets: `src/gpu/gx2_register_state.hpp`, `latte_draw.hpp`, `latte_surface_staging.hpp`, `latte_vertex_fetch.hpp`, `latte_shader_decoder.hpp`, `latte_shader_program.hpp`, and `latte_spirv_vertex_subset.hpp`
- Host backend: `src/gpu/vulkan/vulkan_renderer.cpp`
- Tests: `tests/cpu_test.cpp`, `tests/recompiler_test.cpp`, and the CTest targets defined by `CMakeLists.txt`
