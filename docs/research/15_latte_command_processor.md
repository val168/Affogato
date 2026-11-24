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

These implementations establish useful subsystem boundaries, not proof that
all packet behavior or register side effects match Latte.

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
  playback, ordered register writes, and malformed/unsupported-list rejection.
- `Gx2RegisterState::apply` models writes to `SQ_VTX_SEMANTIC_CLEAR` at
  `0x288E0`: each set bit clears the matching entry in the 32-word semantic
  table beginning at `0x28380`, storing `0xFFFFFFFF` in the shadow. The clear
  register itself is write-only and is not retained as readable shadow state.
  The AMD [R6xx 3D register reference](https://www.x.org/docs/AMD/old/R6xx_3D_Registers.pdf)
  identifies the register as a per-entry clear mask; Decaf's
  `src/libgpu/src/pm4_processor.cpp::setRegisters` maps mask bit `i` to
  semantic entry `i`. This is cross-generation reference evidence, not a
  Latte-specific specification. Tests cover selective clearing and write
  ordering; other special register side effects remain unsupported.

Consequently, the current `pending_commands` queue is an HLE-produced command
model, not a representation of PM4 dwords. A renderer can execute supported
HLE records without a raw PM4 implementation, but authentic context display
lists and guest-written command streams remain unsupported.

## Coverage and dependencies

| Capability | Status | Evidence / dependency |
| --- | --- | --- |
| GX2 command-pool registration | Partial and tested | Bounds/alignment/ownership checks in `initialize_gx2` |
| Typed GX2 HLE event ordering | Partial and tested | `pending_commands`, register shadow and frontend drain |
| PM4 packet framing | Partial and tested | Type-2 filler and Type-3 lengths; Type-0/Type-1 remain unsupported |
| PM4 register writes | Partial and tested | Type-3 `SET_CONTEXT_REG` maps bounded dword offsets to the context register byte range |
| Type-3 operations | Partial | `SET_CONTEXT_REG` only; all other opcodes are rejected |
| Indirect buffers / nested lists | Missing | Guest bounds, alignment, recursion limit, lifetime and ordering |
| Context display-list generation/decoding | Partial | generation leaves size zero; empty lists restore the register shadow; bounded Type-2/`SET_CONTEXT_REG` lists decode into typed writes; GPU packet execution is not implemented |
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

- Current fixtures cover Type-2 filler and Type-3 count/opcode/register-offset
  behavior. Before Type-0 support is added, add count/base-index fixtures
  cross-checked against AMD's guide and a known Latte stream.
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
