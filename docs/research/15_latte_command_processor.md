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
  payload DWORDs minus one. Section 5.3.3 gives the config/context register
  window bases. These are family-level evidence; the parser intentionally
  accepts only bounded writes within those windows.
- Cemu's
  [`GX2_Draw.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/OS/libs/gx2/GX2_Draw.cpp)
  emits `SET_CONFIG_REG` packets while setting up GX2 draw state. The opcode
  and packet payload shape are reference implementation evidence; Affogato
  does not yet execute the draw packet that follows those setup writes.
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

Cemu's [`GX2_Draw.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/OS/libs/gx2/GX2_Draw.cpp)
shows the ordinary `GX2DrawEx` packet sequence: `SET_CTL_CONST` for the base
vertex, `SET_CONFIG_REG` for `VGT_PRIMITIVE_TYPE`, `INDEX_TYPE`,
`NUM_INSTANCES`, and a predicate-bit `DRAW_INDEX_AUTO`. Its indexed path uses
`DRAW_INDEX_2` and enables the same predicate bit. This establishes a useful
reference sequence. Affogato accepts only an unpredicated
`DRAW_INDEX_AUTO` packet in `GX2DirectCallDisplayList`; predicate-bit Type-3
headers remain rejected. This subset requires primitive mode, instance count,
and base vertex to be written earlier in the same list, and captures the
packet's count and zero draw-initiator word in an ordered `Gx2Command` draw
record; nonzero initiator values fail explicitly.
If enabled, the HLE snapshots non-indexed guest vertex bytes at call time. The
draw record can reach the explicit diagnostic renderer path, but does not
establish guest shader execution or normal Latte render-target behavior.
Cemu's common GX2 draw packets set the predicate bit, so they remain
unsupported until predicate source/state semantics are verified.

AMD's [R7xx/Evergreen/Cayman programming guide, §§9.3.4 and 9.3.9](https://docs.amd.com/api/khub/documents/ApP4PzuRytQl9QLD9lgkmA/content)
describes `DRAW_INDEX_AUTO` as a two-DWORD payload containing index count and
draw initiator. It also says `INDEX_TYPE` belongs to the draw packet sequence
and is not shadowed. This is generation-family evidence, not a Latte-specific
specification. Affogato currently records the packet's draw-initiator word;
the subset does not decode its bitfields. The same guide describes
`NUM_INSTANCES` only as a one-DWORD command packet; it does not establish
Latte's zero-count behavior.

The references differ on the effects used to represent `INDEX_TYPE`:
Cemu's [`LatteCP_itIndexType`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteCommandProcessor.cpp#L405-L409)
writes the value to its `VGT_DMA_INDEX_TYPE` context field, while Decaf's
[`Pm4Processor::indexType`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/src/pm4_processor.cpp#L239-L249)
stores it in both `VGT_INDEX_TYPE` and `VGT_DMA_INDEX_TYPE` register slots.
AMD's [Southern Islands register guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/programmer-references/CIK_3D_registers_v2.pdf)
identifies `VGT_DMA_INDEX_TYPE` at `0x28A7C`; Decaf's
[`latte_registers.h`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/latte/latte_registers.h#L42-L48)
separately identifies `VGT_INDEX_TYPE` at `0x895C`. The R7xx packet guide's
statement that `VGT_INDEX_TYPE` is not shadowed supports treating this
difference as an implementation-model question, not permission to mirror both
writes without a Latte fixture.
Likewise, Cemu normalizes a `NUM_INSTANCES` payload of zero to one, whereas
Decaf stores the raw count. Affogato currently preserves the raw packet value;
its draw path must resolve the zero-count semantics with Latte-specific
evidence before using it to submit instances.

AMD's [Evergreen/Cayman guide §9.1.4](https://www.x.org/docs/AMD/old/evergreen_cayman_programming_guide.pdf)
identifies Type-3 header bit 0 as `PREDICATE` and bit 1 as `SHADER_TYPE`; the
earlier R6xx/R7xx guide documents bit 0 as predication and treats bits 7:1 as
reserved for that generation. Decaf's
[`latte_pm4_commands.h::SetPredication`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/latte/latte_pm4_commands.h)
represents the predicate source address and a control word with clear, ZPASS,
and primitive-count operations. Cemu's
[`LatteCommandProcessor.cpp::LatteCP_itSetPredication`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteCommandProcessor.cpp)
reads those fields and toggles conditional-render state. This cross-generation and
reference-emulator evidence establishes that predicate bit 0 cannot generally
be treated as an unconditional draw. It does not establish Espresso/Latte's
default predicate value or all query synchronization behavior. Decaf's packet
dispatch also does not visibly gate on `HeaderType3::predicate` in
`Pm4Processor::handlePacketType3`, so that code alone is not an independent
verification of the gate's semantics.

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
Affogato registers `GX2DirectCallDisplayList` for a narrow subset: it validates
and copies the guest bytes at call time, then queues supported config/context/
control register writes and unpredicated `DRAW_INDEX_AUTO` records in guest
order. A self-contained auto-draw requires preceding `VGT_PRIMITIVE_TYPE`,
`NUM_INSTANCES`, and base-vertex writes in that list; absent base-instance
state is treated as zero for this bounded GX2DrawEx-shaped form. A zero
instance count is rejected because Cemu normalizes zero to one while Decaf
stores it raw. Non-indexed vertex bytes are captured at HLE call time only
when the existing diagnostic capture option is enabled. Predicated headers,
indexed draws, and all other unsupported opcodes fail explicitly. This is an
ordered command capture path; only the diagnostic policy may submit its
supported float2 subset, and no guest Latte shader or guest render-target
behavior is implied.
`GX2BeginDisplayListEx`,
`GX2EndDisplayList`, `GX2CallDisplayList` and `GX2CopyDisplayList` remain
unregistered; queued indirect calls and guest-authored recording therefore
remain unsupported.

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
  shadow when the size is zero. A bounded nonempty subset decodes Type-2 filler,
  Type-3 NOP, config/context/control Type-0 writes, Type-3 register sets, and
  one-word `INDEX_TYPE` / `NUM_INSTANCES` packets into typed register commands;
  `GX2SetContextState` rejects draw packets. `GX2DirectCallDisplayList` also
  captures an unpredicated two-word `DRAW_INDEX_AUTO` when its required setup
  state occurs earlier in the same list. It rejects predicated headers,
  indexed draws, zero instance counts, truncation, and unsupported operations.
  Vertex bytes are copied at HLE time when diagnostic capture is enabled.
  Queuing/capturing these values does not execute guest Latte shaders or
  retire a fence.
- `tests/cpu_test.cpp` covers command-pool attributes, context-state setup,
  zero-length display-list results, shadow restoration, big-endian packet
  playback, Type-3 NOP payload advancement, ordered Type-0 sequential and
  Type-3 config/context/control register writes, and range and malformed-list
  rejection. The direct-call test also verifies an ordered unpredicated
  `DRAW_INDEX_AUTO`, draw-time guest-byte capture, malformed-payload and
  missing-state atomicity, unsupported-opcode atomicity, alignment and
  guest-memory errors, and empty-list behavior.
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
| GX2 display-list API lifecycle and submission | Partial and tested | `GX2DirectCallDisplayList` copies a bounded guest-memory stream and queues only supported config/context/control register writes; recording, queued indirect call, and copy APIs remain missing. |
| Typed GX2 HLE event ordering | Partial and tested | `pending_commands`, register shadow and frontend drain |
| PM4 packet framing | Partial and tested | Type-0 payload lengths, Type-2 filler and Type-3 lengths; Type-1 remains unsupported |
| PM4 register writes | Partial and tested | Type-0 sequential writes and Type-3 config/context/control register sets are bounded to supported byte windows |
| Type-3 operations | Partial and tested | `NOP`, register sets, `INDEX_TYPE`, `NUM_INSTANCES`, and a self-contained unpredicated `DRAW_INDEX_AUTO`; predicated/indexed draws and other opcodes are rejected |
| Indirect buffers / nested lists | Missing | Guest bounds, alignment, recursion limit, lifetime and ordering |
| Context display-list generation/decoding | Partial | generation leaves size zero; empty lists restore the register shadow; bounded Type-0 config/context/control writes, Type-2/Type-3 register sets, `INDEX_TYPE`, and `NUM_INSTANCES` decode into typed writes; GPU packet execution is not implemented |
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

- Current fixtures cover Type-0 count/base-index and sequential-write behavior
  in config/context windows, Type-2 filler, and Type-3 NOP payload advancement,
  `SET_CONFIG_REG`, and `SET_CONTEXT_REG` count/opcode/register-offset
  behavior. The Type-0 packet layout and config/context register bases are
  cross-checked against AMD R6xx/R7xx guide sections 5.2.1 and 5.3.3. Cemu's
  `src/Cafe/OS/libs/gx2/GX2_Draw.cpp` independently shows GX2 draw setup
  emitting `SET_CONFIG_REG` packets. These are related-family and reference
  implementation evidence; no independent Latte-produced stream fixture is
  available yet.
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
- The PM4 predicate source/state semantics used by the predicate-bit draw
  packets emitted by GX2 `GX2DrawEx`; accepting them without modeling the
  predicate can skip or spuriously execute guest work.
- The default predicate state and when it is reset across display-list and
  command-buffer boundaries; a missing `SET_PREDICATION` cannot yet be assumed
  to mean either true or false.
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
