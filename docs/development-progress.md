# Development Progress

## WWHD checkpoint

- RPX: `cking.rpx`; entry point `0x028EA9E0`; 24 sections.
- Latest confirmed run: 6,080,538 guest instructions; stop CIA `0x00000000`; GPR3 `0`; stop is an unsupported fetch of `0x46524D48` after `bcctr` loaded CTR from a null callback field. The wider trace shows both the thread-specific candidate and the global object's `+0x10` candidate are null. `0x46524D48` is the synthetic `FRMH` header mirrored at low address zero by the MEM1 alias, so this is a null indirect call obscured by the current flat memory map, not a missing PPC opcode. The caller's object owner is still unidentified; do not synthesize a target.
- JIT run: 3,257,921 native instructions, 2,822,618 interpreter fallbacks, 26,091 compiled blocks, 1,892,364 block-cache hits; guest execution about 4.35 s.
- No unimplemented HLE was reached. `GX2InitTextureRegs` ran 13 times, along with the sampler initialization and existing GX2 state HLEs.

## Graphics checkpoint

- Resolved: texture register initialization and sampler descriptor initialization.
- Implemented: pixel texture binding, including Latte resource-word construction and a logical shader-resource command stream (`31c2058`).
- Implemented: shared binding logic for vertex and geometry texture stages, with stage-specific register bases and independent binding state (`c89a6d5`).
- Implemented: three-word sampler binding for pixel, vertex, and geometry stages, with independent per-unit state (`40b1425`).
- Implemented this iteration: compute texture and sampler APIs route to the vertex resource banks, matching Cafe's GX2 behavior.
- Implemented sampler border-color state for pixel, vertex, geometry, and compute (compute shares the vertex bank), recording the four FPR float arguments as Latte config-register writes. Register bases and twelve-entry stage banks were checked against Cemu's Latte register definitions.
- GX2 register/resource/sampler writes now transfer incrementally from each `Emulator::run` result to the frontend's Vulkan thread; the renderer maintains a last-write-wins raw Latte register shadow. This is command/state plumbing only, not draw submission or guest rendering.
- The renderer register shadow can now retrieve complete seven-word texture-resource and three-word sampler bindings by shader stage. Partial groups are not exposed; compute correctly resolves through the shared vertex bank. This remains state decoding only and does not yet create Vulkan images or descriptors.
- Actual guest rendering/presentation is not yet confirmed; command submission, shader execution, and framebuffer presentation remain incomplete.

## Completed commits

- `90ab0b0` - `gx2: implement sampler border type`
- `31c2058` - `gx2: implement pixel texture binding`
- `c89a6d5` - `gx2: add vertex and geometry texture binding`
- `40b1425` - `gx2: implement shader sampler binding`
- `41cd53c` - `gx2: bind compute textures and samplers`
- `48b3420` - `gx2: implement sampler border color state`
- `4b4133d` - `gpu: forward GX2 register writes to Vulkan`

## Current test status

- Debug build: `build/debug-win-native`, Ninja; build is current.
- CTest after the texture/sampler register-group accessors: 2/2 passed.
- Latest WWHD diagnostic: 6,080,538 instructions; stop remains at CIA 0 with null `bcctr` target (`CTR=0`); the new renderer state accessors do not affect this guest path. Latest run: RPX load 206 ms; guest execution 5,482 ms.

## Prioritized queue

1. Trace the null `bcctr` caller/object owner further; current evidence shows zero thread-specific and global candidate pointers. Check guest-side initialization and relevant HLE state; do not fabricate a callback.
2. Add Latte resource-word decoding and map GX2Texture register fields into validated guest texture descriptors before Vulkan allocation/upload.
3. Audit actual guest shader, draw, and scan-buffer producers; the current WWHD run stops before any draw or texture-binding HLE is observed.
4. Add real render submission and guest framebuffer presentation only after command and resource state are available; do not present the clear-test window as game output.
5. Run a fresh Cemu-vs-Affogato CPU coverage audit if the null guest call remains the only runtime stop.
