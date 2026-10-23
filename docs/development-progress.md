# Development Progress

## WWHD checkpoint

- RPX: `cking.rpx`; entry point `0x028EA9E0`; 24 sections.
- Latest confirmed run: entry `0x028EA9E0`, 24 sections, 6,080,538 guest instructions; stop CIA `0x00000000`; GPR3 `0`; GPR4 `512`; GPR5 `256`; unsupported fetch `0x46524D48` after `bcctr` loads CTR from zero. The 64-instruction trace still shows the surface-copy loop and generic null callback path. The owning object/expected callback remains unidentified; do not synthesize a target. Current native compare emission (`cmpwi`, `cmpw`, `cmplw`) reduces interpreter fallbacks to 1,763,166 (from 1,922,446 before native `cmpwi`) while preserving the stop state. Recent release wall times vary from about 2.6–3.0 seconds, so no speedup is claimed from these runs.
- The direct-mapped compiled-block lookup cache now serves 1,770,516 of 1,790,863 WWHD block-cache hits. The latest run still reaches the same null indirect fetch with the same guest state. Recent host load/RPX/guest timings are unstable (including an 832 ms BOTW RPX load), so the lookup hit rate is recorded as a structural metric, not a demonstrated end-to-end speedup yet.
- Native JIT `fcmpu` handles ordered/unordered comparisons and NaNs; `subfic`, `addic`, and `addic.` emit arithmetic/carry updates natively, and record forms correctly update CR0 with XER.SO. Record-form logical OR/AND/XOR also updates CR0 while preserving other fields. Native `mcrf` and `mfcr` cover CR field copy and full-CR read; differential tests cover all `mcrf` field pairs and every `mfcr` destination GPR. Latest Release runs: WWHD 6,080,538 instructions (4,518,105 native; 1,562,434 fallbacks; 20,651 blocks; 1,704,821 cache hits, 1,684,473 direct hits), same null-target stop/registers; BOTW 860,172 (702,773 native; 157,400 fallbacks; 86,602 blocks; 100,943 cache hits, 98,448 direct hits), same stop. Native `mfcr` reduced BOTW fallbacks by five; WWHD aggregate counters were unchanged. Timings vary and are not treated as benchmark evidence.
- Native JIT `mcrf` now copies one selected four-bit CR field while preserving the other seven; differential tests cover all 64 field pairs over four distinct CR patterns. Debug and Release CTest pass. Current WWHD and BOTW runs retain their prior execution checkpoints and the shared null-CTR stop; no runtime speedup is claimed for this coverage change.
- Latest JIT run after native `lfs`: 4,158,093 native instructions, 1,922,446 interpreter fallbacks, 22,956 compiled blocks, 1,882,460 block-cache hits. Release guest time across three runs: 1.148/1.146/1.182 s versus 1.322/1.257/1.359 s before `lfs`; medians 1.148 s versus 1.322 s (13.2% lower guest time). Host total medians were 1.340 s versus 1.511 s. The 175,096 `lfs` fallbacks now execute natively. Remaining top fallbacks: `fmadd` 149,463; `stfs` 127,192; `stwu` 107,842; `fmr` 98,684; `stfd` 97,997; `cmpwi` 94,017; `slw` 87,439.
- No unimplemented HLE was reached. `GX2InitTextureRegs` ran 13 times, along with the sampler initialization and existing GX2 state HLEs.
- A filtered `coreinit::MEMGetBaseHeapHandle` trace captured all four startup calls: types 1, 8, and 0 return the configured MEM2, foreground, and MEM1 handles respectively. The 25 observed `MEMAllocFromDefaultHeap` calls all returned nonzero addresses. This does not explain the later null member pointer and is evidence against changing these HLE results speculatively.
- The write-watch now snapshots guest GPRs at each matching store. At `0x02785EC4`, the guest stores r3=0 into object+0x10 after calling `0x027538C0`; at the earlier `0x02785D14` initialization store, r3 is the object+0x1C pointer. This is a genuine guest helper return of zero, not an HLE memory side effect. Static helper code reads the still-zero global at `0x10208B64` on one branch; its intended initialization path remains unidentified.

## Secondary BOTW checkpoint

- The loader relocates its HLE data/import windows when an RPX section overlaps the defaults (`a9e1130`). It handles `R_PPC_DTPMOD32` as module 1 and `R_PPC_DTPREL32` as the TLS symbol's module-relative offset (`fc8714a`, `a5f3ed2`). Latest BOTW run: entry `0x03AB62AC`, 26 sections, 860,172 guest instructions, stop CIA `0`, GPR3 `0`, GPR4 `4,194,304`, GPR5 `64`; stop is the known unsupported fetch `0x46524D48` after a null `bcctr`, not a decoded instruction. Current native compare emission reduces interpreter fallbacks to 183,764 (from 218,634 before native `cmpwi`); stop state is unchanged. The direct-mapped lookup served 115,262 of 117,784 compiled-block hits. The faulting callee copies its incoming null `r4` to `r27`, loads a null callback from `[r27+0x0C]+0x34`, and branches through CTR=0. Interpreter and JIT runs agree. The watched `0x041D48FC` address was not the source: it is a reused heap/object field, written four times with valid pointers, and is separate from the null callback argument. `OSEnableOverlayArena` mapped and exposed the guest overlay range at `0xA0000000` (448 MiB). This remains startup/runtime progress, not evidence of game rendering.

## Graphics checkpoint

- Resolved: texture register initialization and sampler descriptor initialization.
- Implemented: pixel texture binding, including Latte resource-word construction and a logical shader-resource command stream (`31c2058`).
- Implemented: shared binding logic for vertex and geometry texture stages, with stage-specific register bases and independent binding state (`c89a6d5`).
- Implemented: three-word sampler binding for pixel, vertex, and geometry stages, with independent per-unit state (`40b1425`).
- Implemented: compute texture and sampler APIs route to the vertex resource banks, matching Cafe's GX2 behavior.
- Implemented sampler border-color state for pixel, vertex, geometry, and compute (compute shares the vertex bank), recording the four FPR float arguments as Latte config-register writes. Register bases and twelve-entry stage banks were checked against Cemu's Latte register definitions.
- GX2 register/resource/sampler writes now transfer incrementally from each `Emulator::run` result to the frontend's Vulkan thread; the renderer maintains a last-write-wins raw Latte register shadow. This is command/state plumbing only, not draw submission or guest rendering.
- The renderer register shadow can now retrieve complete seven-word texture-resource and three-word sampler bindings by shader stage. Partial groups are not exposed; compute correctly resolves through the shared vertex bank. This remains state decoding only and does not yet create Vulkan images or descriptors.
- Added checked Latte texture-word decoding into a typed descriptor: dimensions, encoded pitch, tile mode/type, hardware format, guest image/mipmap addresses, view levels/slices, numeric/endian modes, degamma, and component selectors. Invalid resource types, >32-bit addresses, reserved selectors, and invalid format fields are rejected. This is descriptor interpretation only; tiled guest memory is not yet uploaded to Vulkan.
- Added typed sampler decoding for wrap/filter modes, anisotropy, border/depth state, signed fixed-point LOD bias, and word-2 flags. Ambiguous overlapping word-0 bits remain available through raw words rather than being guessed.
- Added an explicit Vulkan RGBA8 upload path for the existing linear texture staging output: host-visible staging buffer, optimal-tiled sampled image, transfer-to-shader-read layout barriers, and renderer-owned image/memory/view lifetime. Upload is currently an opt-in smoke-tested backend primitive; guest texture binding and draw submission do not invoke it yet.
- Corrected `GX2Set*Texture` to read `GX2Surface.mipmaps` from offset `0x2C`, not `mipmapSize` at `0x28`. A missing mip pointer still aliases the base image even when the size field is nonzero.
- Added: a CPU-side texture staging conversion for single-layer 2D UNORM R8, R4_G4, R8_G8, R5_G6_B5, R5_G5_B5_A1, R4_G4_B4_A4, A1_B5_G5_R5, and R8_G8_B8_A8 in linear-aligned layout. GX2 dimension 1 (2D) is accepted; dimension 2 (3D) is rejected. Guest pitch and component selectors are honored; missing channels expand to zero with alpha one. Packed formats are unpacked from big-endian guest data and expanded to 8-bit channels. Format-appropriate Latte endian metadata is validated while the byte-ordered guest snapshot is kept intact. Tiled, compressed, mipmapped, array, and non-UNORM formats are explicitly rejected rather than misread.
- Actual guest rendering/presentation is not yet confirmed; command submission, shader execution, and framebuffer presentation remain incomplete.

## Completed commits

- `90ab0b0` - `gx2: implement sampler border type`
- `31c2058` - `gx2: implement pixel texture binding`
- `c89a6d5` - `gx2: add vertex and geometry texture binding`
- `40b1425` - `gx2: implement shader sampler binding`
- `41cd53c` - `gx2: bind compute textures and samplers`
- `48b3420` - `gx2: implement sampler border color state`
- `4b4133d` - `gpu: forward GX2 register writes to Vulkan`
- `8903dbb` - `gpu: expose complete Latte texture bindings`
- `ad89b6e` - `gpu: decode Latte texture resource words`
- `cffd6eb` - `gpu: decode Latte sampler state`
- `801f671` - `gx2: fix texture mipmap address field`
- `a6a7b7c` - `gpu: stage linear RGBA8 Latte textures`
- `82688dc` - `gpu: stage linear R8 and RG8 textures`
- `2b76180` - `gpu: honor Latte texture endian modes`
- `8cfea57` - `gpu: stage linear RGB565 Latte textures`
- `2d405d4` - `gpu: stage packed linear Latte formats`
- `eb78c56` - `gpu: stage linear R4G4 textures`
- `adbdc10` - `gpu: correct Latte 2D dimension handling`
- `08225b2` - `jit: add native emission for CR logical operations`
- `e00632e` - `jit: emit native mcrf`
- `c678494` - `jit: report interpreter fallback hot spots`
- `f60e130` - `jit: emit native rlwinm instructions` (pushed HEAD before this `lfd` iteration)
- `1a2a49f` - `jit: emit native lfd loads`
- `a34c5c0` - `jit: report compile and execution timings`
- `35de871` - `jit: emit native lfs loads`
- `e1e3744` - `debug: trace recent HLE calls in RPX diagnostics`
- `10f2751` - `gpu: upload staged RGBA8 textures to Vulkan`
- `79fa40a` - `debug: print full recent HLE trace`
- `9c8742e` - `debug: retain a wider PPC trace`
- `0d7ef50` - `debug: watch guest memory writes`
- `50c38dd` - `debug: dump guest memory in RPX diagnostics`
- `75ef1ec` - `debug: filter recent HLE call trace`
- `0b656bb` - `debug: capture GPRs at watched guest writes`
- `a9e1130` - `loader: relocate HLE ranges around RPX sections`
- `fc8714a` - `loader: implement PPC TLS module ID relocation`
- `a5f3ed2` - `loader: implement PPC TLS symbol offset relocation`
- `5d3387b` - `hle: implement coreinit OSInitEvent`
- `5798d18` - `cpu: implement lwarx and stwcx.`
- `6edcb72` - `hle: implement coreinit OSSignalEvent`
- `33080db` - `hle: implement coreinit OSReport`
- `a34912f` - `hle: implement coreinit OSGetMemBound`
- `5278234` - `hle: implement coreinit OSIsEnabledOverlayArena`
- `3c781d0` - `hle: implement coreinit OSEnableOverlayArena`
- `b99c660` - `jit: emit native cmpwi instructions`
- `4afcb02` - `jit: emit native register compares`
- `72711ac` - `jit: add direct-mapped compiled-block lookup`
- `dfd2db7` - `jit: emit native fcmpu`
- `d543b7f` - `jit: emit native subfic`
- `583b13e` - `jit: emit native addic`
- `72986e4` - `jit: emit record-form logical operations`

## Current test status

- Debug and Release builds completed. Native Windows CTest passed 2/2 in both configurations. Differential tests cover `cmpwi`, `cmpw`, `cmplw`, `fcmpu`, `subfic`, `addic`/`addic.`, `mcrf`, and record-form logical OR/AND/XOR; the direct lookup cache has a two-block loop test and existing self-modifying-code invalidation coverage. Latest WWHD/BOTW counters and stops are recorded above. Neither game showed a semantic regression. Timings remain observations only. Earlier OSEvent, reservation, OSReport, OSGetMemBound, overlay arena, Vulkan upload, and texture-staging tests remain covered as documented above.
- Native `lfs` interpreter/JIT differential tests cover signed zero, finite values, a single subnormal, infinities, qNaN/sNaN, literal-zero `rA`, PS0/PS1 behavior, history, and transactional 4-byte memory faults. Native `lfd` tests cover raw big-endian values, literal-zero `rA`, PS1 preservation, and transactional 8-byte faults.
- RPX diagnostics capture and print the latest 512 HLE calls and 64 PPC instructions. `--trace-hle <library::symbol>` filters the bounded trace to one exact HLE symbol. `--watch-write <guest-address>` opts into a bounded watch of a four-byte word, reporting overlapping writes, writer CIA, resulting big-endian word, local decoded code context, and all 32 GPRs at each matching write. `--dump-memory` provides bounded post-run BE-word inspection. All are diagnostic-only and disabled unless requested.

## Prioritized queue

1. Determine whether the shared `0x46524D48` fetch in BOTW and WWHD is caused by a missing guest initialization path or an incorrect runtime/HLE result; do not fabricate a callback or turn address zero into a valid call.
2. Determine whether the WWHD null indirect callback is caused by a missing guest initialization path or an incorrect runtime/HLE result; do not fabricate a callback or turn address zero into a valid call.
3. Audit actual guest shader, draw, and scan-buffer producers; the current WWHD run stops before any draw or texture-binding HLE is observed.
4. Connect binding-time/command-ordered guest snapshots to the Vulkan texture uploader only when a correct guest use point is available; do not assume a bind-time snapshot remains valid through later writes.
5. Add shader resource descriptors, shader translation, draw submission, and guest framebuffer presentation in independently testable steps. Do not present the clear-test window as game output.
6. Revisit measured JIT fallbacks only with a repeatable Release benchmark: native `fmr`, `stfs`, and `stwu` prototypes each passed targeted differential checks but regressed full WWHD timing, so they were discarded.
7. Measure native compare/branch sequences more directly; `cmpwi` now emits native CR updates, but fused host compare-to-branch is still a separate optimization and must preserve CR consumers and flags.
8. Native `slw` and `srw` prototypes were discarded: `slw` regressed full WWHD timing, and an `srw` lowering using physical ECX/CL crashed in generated code. Revisit shifts only after isolating/fixing the host allocator constraint and demonstrating non-regressing end-to-end time.
9. Rebenchmark the direct-mapped lookup against the same build/configuration under stable host load; if repeatable runtime does not improve, remove it. The next substantial dispatch optimization candidate is safe direct linking for known branch/fallthrough targets, preserving invalidation.
