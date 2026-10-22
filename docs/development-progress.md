# Development Progress

## WWHD checkpoint

- RPX: `cking.rpx`; entry point `0x028EA9E0`; 24 sections.
- Latest confirmed run: 6,080,538 guest instructions; stop CIA `0x00000000`; GPR3 `0`; stop is an unsupported fetch of `0x46524D48` after `bcctr` loaded CTR from a null callback field. This is a null indirect guest call, not evidence of a missing PPC opcode.
- JIT run: 3,257,921 native instructions, 2,822,618 interpreter fallbacks, 26,091 compiled blocks, 1,892,364 block-cache hits; guest execution about 4.35 s.
- No unimplemented HLE was reached. `GX2InitTextureRegs` ran 13 times, along with the sampler initialization and existing GX2 state HLEs.

## Graphics checkpoint

- Resolved: texture register initialization and sampler descriptor initialization.
- Implemented: pixel texture binding, including Latte resource-word construction and a logical shader-resource command stream (`31c2058`).
- Implemented: shared binding logic for vertex and geometry texture stages, with stage-specific register bases and independent binding state (`c89a6d5`).
- Implemented this iteration: three-word sampler binding for pixel, vertex, and geometry stages, with independent per-unit state.
- Actual guest rendering/presentation is not yet confirmed; command submission, shader execution, and framebuffer presentation remain incomplete.

## Completed commits

- `90ab0b0` - `gx2: implement sampler border type`
- `31c2058` - `gx2: implement pixel texture binding`
- `c89a6d5` - `gx2: add vertex and geometry texture binding`

## Current test status

- Debug build: `build/debug-win-native`, Ninja; build is current.
- CTest after texture and sampler binding: 2/2 passed.

## Prioritized queue

1. Trace the null `bcctr` target to its owner and determine whether initialization, runtime state, or a guest-side condition left the callback unset; do not fabricate a callback.
2. Connect recorded resource/sampler state to command processing and renderer consumption.
3. Audit shader, draw submission, and scan-buffer presentation paths against real guest execution before claiming rendered output.
4. Run a current Cemu-vs-Affogato CPU coverage comparison if the null guest call remains the only runtime stop.
