# Latte command processor research

## Purpose and current boundary

GX2 is the guest API; PM4 is the command packet interface consumed by the
Latte command processor. These are different layers. Affogato currently
implements selected GX2 exports and emits its own typed `Gx2Command` records.
It does not decode a guest PM4 ring buffer. This note records the boundary and
the evidence needed before a PM4 subset is added.

## Documented and reference behavior

AMD's public [R6xx/R7xx 3D programming guide](https://www.x.org/docs/AMD/old/R6xx_R7xx_3D.pdf),
section 5, describes PM4 command packets as a header followed by a variable
body. It documents Type-0 register writes, Type-2 filler packets, and Type-3
operation packets; Type-1 is not used by the R6xx+ packet interface described
there. The R700 family reference is closer to GX2's R7xx lineage, but neither
AMD document specifies every Latte register, custom packet, or Cafe submission
rule. Treat packet fields and supported opcodes as generation-specific until
cross-checked with Wii U evidence.

Reference emulator architecture:

- Cemu processes Latte command buffers in
  [`LatteCommandProcessor.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteCommandProcessor.cpp)
  and maintains GPU register/shadow state separately from GX2 HLE and renderer
  backends.
- Decaf's GPL-3.0 `src/libgpu/src/pm4_processor.cpp` frames Type-0, Type-2,
  and Type-3 packets, converts guest command words to host word order, applies
  register writes, and dispatches supported Type-3 opcodes. Its source is an
  architectural reference only; no code has been copied.
- The register map and known GX2 interfaces are summarized by
  [WiiUBrew Hardware/GX2](https://wiiubrew.org/wiki/Hardware/GX2). That page
  also notes the lack of a perfectly matching public card document.
- AMD's R6xx/R7xx guide, section 5.2.1, defines Type-0 as sequential DWORD
  writes: BASE_INDEX is a DWORD register address and COUNT is the number of
  payload DWORDs minus one. This packet format is used here as family-level
  evidence; the supported destination is deliberately limited to Latte's
  documented context-register window.
- Linux's Radeon R6xx/R7xx packet definitions assign Type-3 opcode `0x10` to
  `NOP` ([`r600d.h`](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/radeon/r600d.h)). Affogato consumes its bounded payload without
  producing register writes. This is family-level evidence, not a
  Latte-specific stream fixture.

These implementations establish useful subsystem boundaries, not proof that
all packet behavior or register side effects match Latte.

### GX2 display-list submission boundary

The public WUT
[`displaylist.h`](https://github.com/devkitPro/wut/blob/master/include/gx2/displaylist.h)
declares `GX2BeginDisplayListEx(void*, uint32_t, BOOL)`,
`GX2EndDisplayList(void*) -> uint32_t`, `GX2CallDisplayList(const void*,
uint32_t)`, `GX2DirectCallDisplayList(const void*, uint32_t)`, and
`GX2CopyDisplayList(const void*, uint32_t)`. These interfaces establish the
ABI surface, but WUT declarations do not specify the underlying command
processor behavior.

Cemu's
[`GX2_Command.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/OS/libs/gx2/GX2_Command.cpp)
documents an implementation distinction:

- `GX2CallDisplayList` emits `INDIRECT_BUFFER_PRIV` with a physical address,
  zero high-address dword, and size in dwords into the current write-gather
  stream.
- `GX2DirectCallDisplayList` submits the provided dword buffer to the TCL
  directly, bypassing write-gather; when not recording a display list, it
  flushes preceding commands first.
- Begin/end recording temporarily redirect per-core write-gather output into
  caller storage; end pads the output and restores the main-core command-buffer
  state.

Decaf's GPL-3.0
[`pm4_processor.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/src/pm4_processor.cpp)
recursively dispatches both indirect-buffer opcodes through a bounded nested
command-buffer path. Its
[`latte_pm4_commands.h`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/latte/latte_pm4_commands.h)
models each indirect-buffer packet with three payload dwords: low address,
unused high address, and dword count. Decaf is architectural reference only;
Affogato does not copy its implementation.

These sources distinguish three behaviors that must not be collapsed into a
single host callback: constructing a guest display list, queuing an indirect
call into a command stream, and submitting a display-list buffer directly.
Affogato currently registers none of the WUT display-list APIs. Its
`GX2SetContextState` parser is a separate bounded restore-list path, not a
general submission path. The pending typed HLE queue therefore cannot observe
guest-authored or guest-mutated PM4 display-list contents.

For Affogato, a submission implementation must define when raw guest dwords are
captured and owned across the frontend-thread handoff, validate guest ranges
and alignment before copying, distinguish direct submission from nested
indirect calls, bound nesting and total decoded dwords, and preserve submission
ordering. `GX2CallDisplayList` cannot be treated as successful by decoding only
its indirect-buffer wrapper: the nested target must be processed. Display-list
recording APIs also require a guest-memory write-gather model; recording only
host-side typed commands would miss guest code that patches recorded packet
payloads in place.

**Evidence classification:** WUT prototypes are documented ABI declarations;
Cemu and Decaf behavior is cross-verified implementation evidence; applying
the same details to all Wii U SDK versions is an inference. Exact Cafe thread
ownership, physical-address translation, cache visibility, display-list
padding requirements, and packet alignment at each API boundary remain to be
verified against a Latte fixture or Wii U behavior.

## Verified Affogato implementation

- `src/cpu/espresso/gx2_hle.cpp::parse_attributes` and
  `initialize_gx2` parse and validate the command-pool base/size, allocate it
  when requested, and record its ownership. The comment in
  `initialize_gx2` explicitly says the pool is registered, not initialized;
  no PM4 packets are emitted or consumed there.
- GX2 HLE callbacks update guest-facing runtime state and append typed records
  such as `set_context_register`, `set_shader_resource`, `draw`, `flush`, and
  `draw_done`. `Gx2CommandType` in `gx2_hle.hpp` defines this host-side queue.
- `GX2Flush` appends an ordered typed boundary. The frontend drains captured
  batches after bounded guest execution; this is not a guest ring submission
  or GPU retirement wait.
- `GX2DrawDone` appends flush and draw-done records. Its current immediate
  retirement behavior is valid only because guest draw work is not submitted
  to the renderer.
- `GX2SetupContextStateEx` zeroes the context storage and leaves its embedded
  PM4 display-list size at zero because Affogato does not generate authentic
  restore packets. `GX2GetContextStateDisplayList` returns that embedded
  buffer and size. `GX2SetContextState` restores Affogato's captured register
  shadow when the size is zero. A bounded nonempty subset decodes Type-2 filler
  and Type-3 `SET_CONTEXT_REG` packets into typed register commands; other
  packet types/opcodes, predication, truncation, and out-of-window writes fail
  explicitly. Queuing these values does not execute GPU work or retire a fence.
- `tests/cpu_test.cpp` covers command-pool attributes, context-state setup,
  zero-length display-list results, shadow restoration, big-endian packet
  playback, Type-3 NOP payload advancement, ordered Type-0 sequential and
  Type-3 context register writes, out-of-window Type-0 rejection, and
  malformed/unsupported-list rejection.
- `Gx2RegisterState::apply` models writes to `SQ_VTX_SEMANTIC_CLEAR` at
  `0x288E0`: each set bit clears the matching entry in the 32-word semantic
  table beginning at `0x28380`, storing `0xFFFFFFFF` in the shadow. The clear
  register itself is write-only and is not retained as readable shadow state.
  The AMD [R6xx 3D register reference](https://www.x.org/docs/AMD/old/R6xx_3D_Registers.pdf)
  identifies the register as a per-entry clear mask; Decaf's
  `src/libgpu/src/pm4_processor.cpp::setRegisters` maps mask bit `i` to
  semantic entry `i`. This is cross-generation reference evidence, not a
  Latte-specific specification. Tests cover selective clearing, write
  ordering, and the `GX2SetContextState` packet-to-register-shadow path; other
  special register side effects remain unsupported.

Consequently, the current `pending_commands` queue is an HLE-produced command
model, not a representation of PM4 dwords. A renderer can execute supported
HLE records without a raw PM4 implementation, but authentic context display
lists and guest-written command streams remain unsupported.

## Coverage and dependencies

| Capability | Status | Evidence / dependency |
| --- | --- | --- |
| GX2 command-pool registration | Partial and tested | Bounds/alignment/ownership checks in `initialize_gx2` |
| GX2 display-list API lifecycle and submission | Missing | WUT declares begin/end, queued/direct call, and copy APIs, but Affogato has no registrations or guest dword ownership model; see the reference comparison above. |
| Typed GX2 HLE event ordering | Partial and tested | `pending_commands`, register shadow and frontend drain |
| PM4 packet framing | Partial and tested | Type-0 payload lengths, Type-2 filler and Type-3 lengths; Type-1 remains unsupported |
| PM4 register writes | Partial and tested | Type-0 sequential writes and Type-3 `SET_CONTEXT_REG` are bounded to the context-register byte range |
| Type-3 operations | Partial and tested | `NOP` and `SET_CONTEXT_REG`; other opcodes are rejected |
| Indirect buffers / nested lists | Missing | Guest bounds, alignment, recursion limit, lifetime and ordering |
| Context display-list generation/decoding | Partial | generation leaves size zero; empty lists restore the register shadow; bounded Type-0 context writes and Type-2/Type-3 `SET_CONTEXT_REG` lists decode into typed writes; GPU packet execution is not implemented |
| GPU completion and waits | Partial / not renderer-backed | Must follow actual submission, fence, and memory-visibility semantics |

## Design constraints for a first implementation

1. Keep packet framing separate from opcode execution. The framing layer should
   consume a bounded `std::span<const std::uint32_t>` already read through
   guest-memory APIs, return typed packets or a structured error, and never
   directly access guest addresses.
2. Explicitly convert each guest big-endian dword before decoding. Validate
   packet body length against remaining dwords before advancing the cursor;
   reject unsupported types/opcodes without partial-success reporting.
3. Execute register packets through one validated register-write path so direct
   GX2 HLE writes and PM4 writes update the same guest GPU shadow and ordered
   records. Model hardware-specific side effects only when verified.
4. Bound indirect-buffer depth, packet count, and total dwords. Preserve packet
   ordering and prevent guest-address overflow before any read.
5. Do not make `GX2DrawDone` report retirement before actual queued GPU work
   completes. Display-list support must preserve the current asynchronous
   frontend ownership boundary.
6. Keep the current parser attached to `GX2SetContextState`; do not present
   this context-list subset as the complete Latte command processor. Expand
   support only when packet fixtures and register side-effect tests verify the
   additional behavior.

## Acceptance tests

- Current fixtures cover Type-0 count/base-index and sequential-write behavior,
  Type-2 filler, and Type-3 NOP payload advancement and
  `SET_CONTEXT_REG` count/opcode/register-offset behavior. The Type-0 packet
  layout is cross-checked against AMD R6xx/R7xx guide section 5.2.1, and the
  NOP opcode against Radeon R6xx/R7xx packet definitions; no independent
  Latte-produced stream fixture is available yet.
- Big-endian dword decoding, empty/truncated streams, count overflow, unknown
  type/opcode, invalid register range, and exact cursor advancement.
- Tests proving malformed packets do not partially mutate the register shadow
  or append commands.
- Context display-list tests that prove supported writes restore the expected
  register words and produce ordered typed state updates.
- Later, integration tests for indirect-buffer recursion, draw ordering,
  fences, and renderer completion before claiming guest command submission.

## Unknowns and next investigation

- The exact GX2/Cafe submission path and PM4/ring-buffer entry points used by
  relevant titles; Affogato currently has no registered `GX2CallDisplayList`
  implementation.
- Which Type-3 opcodes and custom packets are required by real guest workloads.
- Whether context display-list bytes contain native PM4 in all supported GX2
  versions and the exact byte/word order at the guest boundary.
- Register-write masking, context shadow behavior, and wait/fence side effects
  that must be modeled for an implementation beyond packet framing.
- Whether a fixture can be produced with a standalone assembler in this
  environment. The Decaf Wii U test suite requires additional tooling, and the
  current machine has no discovered devkitPro or `latte-assembler` executable.

Do not infer normal guest draw execution from the current HLE record queue.
Before expanding PM4 support, capture a known valid stream or build a
documented controlled fixture. Start with packet framing and context-register
writes, then add execution semantics only with tests for each side effect.
