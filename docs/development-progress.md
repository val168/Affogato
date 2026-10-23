# Development Progress

## WWHD checkpoint

- RPX: `cking.rpx`; entry point `0x028EA9E0`; 24 sections.
- Latest confirmed run: 6,080,538 guest instructions; stop CIA `0x00000000`; GPR3 `0`; stop is an unsupported fetch of `0x46524D48` after `bcctr` loaded CTR from a null callback field. The wider trace shows both the thread-specific candidate and the global object's `+0x10` candidate are null. `0x46524D48` is the synthetic `FRMH` header mirrored at low address zero by the MEM1 alias, so this is a null indirect call obscured by the current flat memory map, not a missing PPC opcode. The caller's object owner is still unidentified; do not synthesize a target.
- JIT run: 3,257,921 native instructions, 2,822,618 interpreter fallbacks, 26,091 compiled blocks, 1,892,364 block-cache hits; latest guest execution about 5.03 s.
- No unimplemented HLE was reached. `GX2InitTextureRegs` ran 13 times, along with the sampler initialization and existing GX2 state HLEs.

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
- `5d43b89` - `gpu: correct Latte 2D dimension handling`

## Current test status

- Debug build: `build/debug-win-native`, Ninja; build is current.
- CTest: 2/2 passed after dimension-code correction and explicit 3D-layout rejection.
- Latest WWHD diagnostic after staging changes: 6,080,538 instructions; stop remains at CIA 0 with null `bcctr` target (`CTR=0`); the staging helper does not affect this guest path. Latest run: RPX load 211 ms; guest execution 5,623 ms.

## Prioritized queue

1. Trace the null `bcctr` caller/object owner further; current evidence shows zero thread-specific and global candidate pointers. Check guest-side initialization and relevant HLE state; do not fabricate a callback.
2. Add a Vulkan image-upload path for the already-supported linear RGBA8 staging representation, with resource lifetime/synchronization tests and no claim that current WWHD has submitted a draw.
3. Audit actual guest shader, draw, and scan-buffer producers; the current WWHD run stops before any draw or texture-binding HLE is observed.
4. Add real render submission and guest framebuffer presentation only after command and resource state are available; do not present the clear-test window as game output.
5. Run a fresh Cemu-vs-Affogato CPU coverage audit if the null guest call remains the only runtime stop.
