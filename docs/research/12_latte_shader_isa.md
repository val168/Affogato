# Latte shader ISA and translation research

## Purpose and current boundary

Latte shaders are guest programs associated with GX2 vertex, geometry, and
pixel stages. Affogato currently captures the stage descriptors and bounded
program words when `GX2SetVertexShader`, `GX2SetGeometryShader`, or
`GX2SetPixelShader` runs. `Gx2RegisterState::apply()` snapshots those bindings
at each draw. Structural control-flow decoding recognizes fixture-verified
`CALL_FS`, `EXPORT`, `EXPORT_DONE`, and one ALU-clause header encoding. Raw
ALU slots, OP2/OP3 fields, no-literal OP2/OP3 groups, and narrow four-lane
shader execution subsets are also implemented. A typed IR represents bounded
OP2 or OP3 CPU clauses; an experimental SPIR-V vertex emitter lowers MOV,
binary32 ADD, IEEE multiply, DX10 min/max OP2 groups, and floating-point and
integer conditional-move OP3 groups. The diagnostic Vulkan triangle still
uses an explicit bounded shader test path rather than normal guest programs.

Relevant local implementation:

- `src/cpu/espresso/gx2_hle.cpp::capture_program_shader` reads a checked
  descriptor and program byte range as guest big-endian 32-bit words. It also
  captures secondary programs for the supported geometry-shader ABI.
- `src/cpu/espresso/gx2_hle.hpp::Gx2ProgramShaderBinding` stores stage,
  descriptor words, program address/size/words, and optional secondary program
  data.
- `src/gpu/gx2_register_state.hpp::Gx2RegisterState::apply` snapshots the
  current program bindings for each captured draw. This is state capture, not
  shader execution.
- `src/gpu/latte_vertex_fetch.hpp` decodes a narrow, Affogato-generated fetch
  program subset for diagnostic host vertices, extracts a complete
  `SQ_VTX_SEMANTIC` table from a draw's raw register snapshot, and resolves a
  fetch semantic ID to its VS input GPR. It also applies VTX destination
  selectors for supported scalar/vector formats, including 0.0, 1.0, and
  masked components. `decode_latte_vertex_shader_input()` combines one
  resolved attribute with the immutable draw-time buffer snapshot and returns
  its destination GPR plus optional component bits. It takes the resolved
  vertex index and `VS_GPR_BASE` explicitly. Table extraction rejects
  incomplete, duplicate, or wrongly typed entries. Multiple attributes are
  assembled into a caller-provided register file with transactional writes.
  `execute_latte_vertex_shader_subset()` joins that fetch subset to the
  supported straight-line CPU shader subset for one explicitly selected
  vertex. Its draw-snapshot overload reads the fetch shader, vertex shader,
  semantic registers, and immutable vertex bytes from one
  `Gx2DrawSnapshotView`. This is a deterministic CPU execution path, not
  normal Vulkan draw execution; system inputs are not modeled. The fetch path
  remains separate from general guest vertex/pixel shader execution.
- `GX2InitFetchShaderEx` in `src/cpu/espresso/gx2_hle.cpp` emits one four-dword
  record per attribute. Word 0 selects the semantic VTX form; under that form,
  word 1's overlapping low field is a semantic ID, not a direct destination
  GPR. Affogato decodes it as `semantic_id` for its HLE-generated subset.
  Arbitrary guest fetch programs and the full Latte field width remain
  unsupported.
  `GX2SetFetchShader` captures the guest descriptor and program but does not
  itself emit `SQ_VTX_SEMANTIC` register writes. The CPU subset helper is not
  yet called from normal GX2 draw processing, so captured draw commands do not
  automatically fetch vertices and execute the supported shader subset.
- `src/gpu/latte_shader_decoder.hpp` converts the captured big-endian words
  to the little-endian Latte DWORD values represented in GFD shader code, then
  decodes a narrow control-flow subset, ALU clause ranges, OP2 fields, and
  no-literal OP2 group boundaries.
- `src/gpu/latte_shader_interpreter.hpp` evaluates MOV and selected integer
  OP2 values from absolute GPRs, inline constants, and prior-PV values. Its
  clause executor supports sequences of four-slot X/Y/Z/W groups and commits
  guest-register updates only after all supported groups succeed. It also
  materializes the supported normal-export swizzle form into typed output
  records.
- `src/gpu/latte_shader_program.hpp` executes straight-line OP2-only or
  supported OP3-only ALU clauses and normal exports. It requires
  caller-provided initial GPRs, rejects mixed OP2/OP3 clauses, and explicitly
  rejects fetch-shader calls, branches, texture clauses, and unsupported CF
  operations. It is a testable CPU execution slice, not connected to guest
  draw submission or Vulkan. The SPIR-V emitter still accepts OP2 clauses only.

## Evidence categories

### Documented interfaces

WUT's `include/gx2/shaders.h` and `include/gx2/surface.h` define GX2 shader
descriptors, register storage, and the API boundary. The reconstructed layout
table at `kinnay/Nintendo-File-Formats/src/libs/wiiu/gx2.md` records the
vertex-shader program size/address at offsets `0xD0/0xD4` and pixel-shader
program size/address at `0xA4/0xA8`. Affogato uses these offsets in
`gx2_hle.cpp`; its tests exercise descriptor capture. Interface layouts do
not define the full Latte machine semantics.

### GX2 shader resource mode and register state

WUT's [`include/gx2/shaders.h`](https://github.com/devkitPro/wut/blob/master/include/gx2/shaders.h)
declares `GX2SetShaderModeEx(mode, numVsGpr, numVsStackEntries,
numGsGpr, numGsStackEntries, numPsGpr, numPsStackEntries)` and its inline
`GX2SetShaderMode` wrapper supplies `0x30/0x40/0/0/0xC8/0xC0` for non-geometry
modes or `0x2C/0x20/0x40/0x30/0x4C/0xB0` for geometry mode. WUT's
`GX2ShaderMode` distinguishes uniform-register, uniform-block, geometry, and
compute modes. These are documented public API values, not proof that the
associated device state is emulated.

Decaf's `GX2SetShaderModeEx` in
[`gx2_shaders.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libdecaf/src/cafe/libraries/gx2/gx2_shaders.cpp)
reads existing register values, edits fields, and emits `VGT_GS_MODE` plus six
contiguous config registers beginning at `SQ_CONFIG`. The AMD
[R7xx register reference](https://vgamuseum.info/index.php/companies/item/download/1285_186d50f48f4f4c1398a91e1b07eba33c)
lists `SQ_CONFIG` at `0x8C00`; `SQ_GPR_RESOURCE_MGMT_1/2` at
`0x8C04/0x8C08`; thread management at `0x8C0C`; and stack management at
`0x8C10/0x8C14`. It describes `SQ_CONFIG` as a graphics-idle-only register,
and GPR/stack allocation writes as requiring relevant shader work to be
flushed. Decaf's implementation sets shader priorities, vector-ALU preference,
uniform-register mode, shader GPR/stack allocations, and per-stage thread
allocations. Geometry and compute modes additionally alter VGT mode or require
ring-buffer/resource setup. This is useful cross-checked implementation
behavior, but Decaf is GPL-3.0 and is used here only as a reference; no code is
copied.

Affogato registers `GX2SetShaderModeEx` for uniform-register and uniform-block
modes. It queues a typed state operation; `Gx2RegisterState` applies it in
command order as masked updates to the shader allocation, thread, stack,
`SQ_CONFIG`, and `VGT_GS_MODE` fields. Register snapshot entries carry a known
bit mask, so a partial update does not turn unobserved neighboring bits into
fabricated zeroes. If all prior bits were supplied, updates preserve the
untouched fields. This is a CPU-side state shadow only: it does not program a
physical GPU or establish shader execution. Geometry and compute modes fail
explicitly because their VGT/ring-buffer and scheduling behavior is not
implemented.

The implemented subset validates the public mode values and 8-bit GPR / 12-bit
stack allocation fields, retains all ABI arguments, updates register state in
the ordered draw stream, and rejects geometry/compute modes before queueing a
command. Uniform-block mode clears `DX9_CONSTS` based on the R7xx register
definition that zero selects DX10 constant-cache mode; Decaf only sets this bit
for uniform-register mode and otherwise preserves it. This difference is not
resolved by Wii U hardware evidence. The current tests verify state-shadow
field behavior and command ordering, not Cafe OS hardware register programming.
Latte traces or hardware fixtures are still needed to confirm this mapping,
resource allocation policy, and synchronization behavior before these state
updates can be treated as hardware-verified.

### Uniform-register uploads

WUT declares `GX2SetVertexUniformReg(offset, count, data)` and
`GX2SetPixelUniformReg(offset, count, data)` with 32-bit word pointers. Decaf's
[`gx2_shaders.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libdecaf/src/cafe/libraries/gx2/gx2_shaders.cpp)
maps vertex uploads to `SQ_ALU_CONSTANT0_256` and pixel uploads to
`SQ_ALU_CONSTANT0_0`; the R6xx/R7xx register map places those banks at
`0x31000` and `0x30000`, respectively. Decaf passes `count` words to its
big-endian ALU-constant packet writer. A Wii U trace records a count of 4 when
uploading one float4, consistent with dword-count units. The R7xx register
model lays constant vectors out as four consecutive 32-bit words. These
references establish the current modeled layout, but not all Latte
constant-file edge cases.

Affogato validates the 256-vector bank bounds and guest source range, reads
each dword through guest big-endian access, and appends typed
`set_alu_constant` writes to the ordered GX2 stream. The vertex bank begins at
vector 256 in the shared register map; pixel begins at vector 0. Each write is
retained by `Gx2RegisterState` and draw-time register snapshots. The bounded
vertex CPU subset maps known dwords from its draw snapshot into a sparse
constant file and supplies it to shader execution; unwritten components remain
unknown and reads of them fail. This is a deterministic test path, not normal
GX2 draw dispatch or Vulkan uniform binding. Pixel snapshots can be decoded by
the same address helper, but a pixel shader execution path is not implemented.

The high 16 bits of Decaf's offset select an `SQ_LOOP_CONST_*` entry, while the
lower bits select the ordinary constant register location. Affogato currently
rejects loop-constant uploads explicitly because the per-stage Latte register
mapping and bounds have not been independently confirmed. Uniform-block
uploads are also not implemented; they require bounded guest-buffer capture,
resource descriptors, and draw-time buffer binding. Tests cover stage-bank
separation, word order, draw snapshots, bounds, unmapped pointers, and the
explicit loop-constant limitation.

### Hardware-family references

WiiUBrew's [GX2 hardware page](https://wiiubrew.org/wiki/Hardware/GX2)
identifies the GX2 GPU as an R7xx-family device and notes that no perfectly
matching public card documentation is known. AMD's official [R700 Family
Instruction Set Architecture](https://docs.amd.com/v/u/en-US/R700-Family_Instruction_Set_Architecture)
is therefore a closer primary family reference than the R600 manual. Its
chapters 3–6 describe control flow, ALU, vertex-fetch, and texture-fetch; its
chapters 9–10 provide mnemonics and microcode formats. The R700 manual still
does not specify Latte's customizations, so its fields and semantics are
candidate hypotheses until checked against Wii U shader fixtures and
Latte-focused emulator evidence.

The AMD R700 manual gives a useful structural model for planning a decoder:

- A program's control-flow stream invokes separate ALU, vertex-fetch, and
  texture-fetch clauses; the clause payloads are not one flat instruction
  stream (Chapter 2, §2.4 and Chapter 3, §§3.3–3.4).
- The generic ALU clause header carries a clause address and a slot count
  encoded as `COUNT - 1`; the documented R700 range is 1–128 64-bit slots
  (Chapter 10, §10.1, `CF_ALU_WORD[0,1]`). The same format family includes
  clause control variants such as `ALU_PUSH_BEFORE` and `ALU_POP_AFTER`.
- An ALU instruction occupies two 32-bit dwords. An instruction group has
  vector X/Y/Z/W and transcendental slots, and group boundaries/forwarding
  affect observable register values (Chapter 4, §§4.1–4.4).
- OP2 and OP3 encodings use different field widths; they cannot safely be
  decoded by treating every slot as one uniform opcode layout (Chapter 4,
  §4.5; Chapter 10, §10.2).

The R700 manual's `CALL_FS` entry (Chapter 9, p. 9-10) describes a fetch
subroutine call whose address is relative to a host-configured register; it
also enters fetch-program mode until the corresponding `RETURN`. Only a
vertex shader may make this call, and the instruction includes condition and
nesting fields. Affogato now decodes its relative address, pop count,
condition, and call count. After it separately processes the bound
HLE-generated fetch subset, the vertex-shader CPU path permits only one leading
`CALL_FS` with address 0, pop count 0, active condition, and call count 0;
other call forms are rejected. Cemu's pinned implementation provides an
architectural precedent for lowering the fetch program separately from the
vertex program, but this specific supported call form remains an R700-family
reference plus fixture evidence, not Wii U hardware verification. The
call-target/base relationship remains unverified, so this path is not yet
connected to normal Vulkan draw dispatch.

Cemu at commit
[`2f49c079da1d50b85cea0ed99aedda58846b3523`](https://github.com/cemu-project/Cemu/commit/2f49c079da1d50b85cea0ed99aedda58846b3523)
provides a useful architecture comparison: its
[`LatteDecompiler_ParseCFInstruction`](https://github.com/cemu-project/Cemu/blob/2f49c079da1d50b85cea0ed99aedda58846b3523/src/Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.cpp)
accepts `CALL_FS` as a no-op in the vertex-program lowering, while its shader
context carries a separate parsed fetch-shader object. Its
[`FetchShader.cpp`](https://github.com/cemu-project/Cemu/blob/2f49c079da1d50b85cea0ed99aedda58846b3523/src/Cafe/HW/Latte/Core/FetchShader.cpp)
parses that fetch program independently. This is a translation boundary, not
evidence that the hardware call has no effect. It supports a future design in
which Affogato separately resolves the bound fetch program and only elides a
validated `CALL_FS` after its fetch behavior has been modeled. No Cemu code was
copied, and its treatment does not resolve Latte-specific call-target or
condition behavior.

These are **R700-family reference facts**, not yet verified Latte decoding
rules. In particular, they do not establish how a GX2 shader's guest bytes map
to decoder dwords on this implementation, nor which R700 encodings Latte
changes. Affogato currently reads shader program memory using guest
big-endian `read32_be()` calls. That establishes the capture representation,
not the GPU microcode's field/word ordering. The first shader fixtures must
settle that distinction before any decoder uses these candidate layouts.

The older public [AMD R600 ISA reference](https://www.amd.com/system/files/TechDocs/R600_Instruction_Set_Architecture.pdf)
can help explain shared R6xx/R7xx lineage, but it is a less direct baseline
and is not an Espresso/Latte specification. Any field, opcode, or behavior
used by Affogato must be cross-checked against Latte-focused references and
real shader fixtures before implementation.

The AMD [R6xx/R7xx 3D Engine Programming Guide, revision 1.0](https://www.x.org/docs/AMD/old/R6xx_R7xx_3D.pdf)
describes fetch-shader-to-vertex-shader linkage in Sections 2.5.1 and 2.5.3.
It specifies a 32-entry table with an 8-bit `semanticID`; a match at table
index `n` routes fetched data to `VS_GPR_BASE + 1 + n`. The offset reserves
the first input vector for system-generated vertex and instance IDs. A fetch
semantic absent from the table is unused by the vertex shader and its fetch is
terminated. However, the AMD [R700 ISA manual](https://docs.amd.com/v/u/en-US/R700-Family_Instruction_Set_Architecture),
Chapter 5, describes the fetch instruction's semantic ID as nine bits, while
the [R6xx register reference](https://www.x.org/docs/AMD/old/R6xx_3D_Registers.pdf),
Chapter 4's `SQ_VTX_WORD1_SEM` description, specifies eight bits for that
field. This width disagreement is unresolved for Wii U Latte. The Affogato
resolver currently models only the 8-bit table/register view; it does not
decode a guest fetch instruction's semantic field or prove that the mapping
matches Latte hardware. It also does not model system-generated inputs or
populate draw shader inputs.

Decaf's GX2 reference implementation clarifies the table population path:
[`gx2_shaders.h`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libdecaf/src/cafe/libraries/gx2/gx2_shaders.h)
places `sq_vtx_semantic_clear`, `num_sq_vtx_semantic`, and the 32-entry
`sq_vtx_semantic` array at descriptor offsets `0x3C`, `0x40`, and `0x44`.
Its `GX2SetVertexShader` implementation in
[`gx2_shaders.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libdecaf/src/cafe/libraries/gx2/gx2_shaders.cpp)
emits the clear register followed by the declared table values. Affogato now
captures those descriptor words, rejects a count above 32, and queues the
corresponding register writes after the shader binding. Tests verify command
ordering and that a draw snapshot contains the resulting table. This is
cross-checked reference behavior, not Wii U hardware verification.

### Vertex shader binding register state

WUT's [`GX2VertexShader` definition](https://github.com/devkitPro/wut/blob/master/include/gx2/shaders.h)
lays out the register block at offset `0x00` and statically verifies offsets
including `SQ_PGM_RESOURCES_VS` (`0x00`), output configuration (`0x08`),
output-ID count/table (`0x0C`/`0x10`), primitive output control (`0x38`),
semantic clear/count/table (`0x3C`/`0x40`/`0x44`), stream-out enable (`0xC4`),
reuse controls (`0xC8`/`0xCC`), program size/address (`0xD0`/`0xD4`), mode
(`0xD8`), ring item size (`0x10C`), and stream-out strides (`0x114`).
Affogato now turns the bounded non-geometry VS fields into ordered context
register commands at `GX2SetVertexShader`, validates the declared output-ID
and semantic counts against their fixed descriptor arrays, and retains those
writes in subsequent draw snapshots. Geometry-shader mode uses the ES program
register range (`SQ_PGM_START_ES` at `0x28880`) and captures the ring item size
at the R6xx-reference address `0x288A8`. When stream-out is enabled, the four
byte strides are converted to DWORD units; the enable word is written to
`VGT_STRMOUT_BUFFER_EN` at `0x28B20`. The program start address resolves the
existing guest-memory alias mapping before conversion to the register's
256-byte address units; this is alias resolution only, not MMU translation.

This setup sequence is cross-checked against Decaf's
[`GX2SetVertexShader`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libdecaf/src/cafe/libraries/gx2/gx2_shaders.cpp)
and register names/addresses against the related-generation
[AMD R6xx register reference](https://www.x.org/docs/AMD/old/R6xx_3D_Registers.pdf)
and Linux's [R600 register definitions](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/radeon/r600d.h).
These are reconstructed/reference sources, not a Wii U hardware specification;
the register behavior is not validated on a Latte device. The descriptor and
program capture remains the source of guest shader words. The new register
commands only improve state fidelity: normal draw commands still do not execute
the shader, configure a Vulkan guest pipeline, or produce translated guest
pixels. Loop-constant uploads and broader GPU register state remain separate
work.

For the HLE-generated fetch subset, the overlapping word-1 field is now treated
as a semantic ID because word 0 selects `VTX_INST_SEMANTIC`. This interpretation
matches the alternative semantic and direct-GPR word-1 layouts in AMD's R700
ISA and Decaf's generator, which emits that semantic mode with
`attrib.location` in the overlapping field. The decoded field is named
`semantic_id`, and the diagnostic path searches for its position semantic
explicitly. This resolves the local naming/field interpretation for this
generated subset; it is not direct Wii U hardware validation.

Decaf's
[`GX2InitFetchShaderEx`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libdecaf/src/cafe/libraries/gx2/gx2_fetchshader.cpp)
also emits `VTX_INST_SEMANTIC` in word 0, then sets `attrib.location` through
the overlapping field while filling shared format/selector fields. This
corroborates the HLE-generated word pattern. `GX2SetFetchShader` still captures
guest bytes without executing the fetch program or populating shader GPRs. A
Wii U-produced fixture is still needed before extending this interpretation to
arbitrary guest fetch programs or claiming Latte hardware verification.

### Reference emulator research

- Cemu separates shader capture/cache/decompilation from the renderer. Useful
  source locations are
  [`LatteShader.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteShader.cpp),
  [`LatteShader.h`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteShader.h),
  [`LatteInstructions.h`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/ISA/LatteInstructions.h),
  `src/Cafe/HW/Latte/LegacyShaderDecompiler/`, and
  `src/Cafe/HW/Latte/ISA/`. These files are reference material, not code to
  transplant. The current CF record decoder checks the wider 7-bit opcode
  before falling back to the overlapping 4-bit ALU opcode; its ALU-clause
  count accessor reads bits 24:18 and adds one. This agrees with the R700
  family manual's ALU count description, but remains reference-emulator
  evidence rather than an independent Latte hardware measurement.
  Its `LatteClauseInstruction_ALU` accessors also treat each clause address
  unit as eight bytes and the payload as consecutive 64-bit slots. The
  controlled fixture confirms those offsets and the slot word order for one
  assembled MOV vector group. This supports extracting raw slots, but does not
  establish instruction semantics or all shader byte-order cases.
  Cemu's current `LatteInstructions.h` recognizes ALU clause variants
  `0x48`–`0x4B` and `0x4D`–`0x4F`; Affogato deliberately accepts only fixture-
  verified `0x48`. Its `LatteClauseInstruction_ALU` layout distinguishes OP2
  from OP3 and identifies the final slot in a group with word0 bit 31. The
  legacy decompiler lists `MOV` as OP2 opcode `0x019`. These are useful
  cross-reference points for the next decoder step, not independent Latte
  hardware verification. `LatteCFInstruction_EXPORT_IMPORT` decodes export
  type, array base, source GPR/relative flag, index GPR, burst count, and the
  four swizzle selectors; these fields are cross-checked with the R700 export
  format and the assembled pass-through fixture below.
- Decaf keeps its GPU implementation under
  [`src/libgpu`](https://github.com/decaf-emu/decaf-emu/tree/master/src/libgpu)
  and its GFD raw shader-block reader at
  [`gfd_read.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgfd/src/gfd_read.cpp)
  and publishes a separate
  [Wii U test repository](https://github.com/decaf-emu/wiiu-tests). Its
  architecture and test assumptions need review before any fixture is adapted.
- Decaf's legacy GLSL translator is a useful comparison point at
  `src/libgpu/src/glsl2/glsl2_translate.cpp`; Cemu's current ISA and legacy
  decompiler directories are useful for comparison of decoder organization.
  Implementation disagreement is evidence to investigate, not enough by
  itself to choose hardware behavior.
- Cemu issue [#2122](https://github.com/cemu-project/Cemu/issues/2122)
  documents a concrete VLIW dependency hazard: when vector and transcendental
  ALU units write the same GPR component in a group, previous-value and
  previous-scalar reads must preserve the correct unit's result. This is a
  reference implementation bug report, not an ISA specification, but it
  demonstrates why flattened per-op arithmetic can be incorrect.
- GaryOderNichts' [Wii U shader introduction](https://github.com/GaryOderNichts/wiiu-shaders/blob/master/shaders.md)
  shows a minimal position/color pass-through vertex/pixel shader pair in
  Latte assembly and the `latte-assembler assemble` workflow. The author labels
  the guide incomplete; use it to construct a controlled workload, not as the
  authority for instruction encodings or full GX2 ABI semantics.
- The Decaf Windows release publishes `latte-assembler.exe` alongside the
  emulator ([release assets](https://github.com/decaf-emu/decaf-emu/releases)).
  It was downloaded to a temporary directory and used to assemble the guide's
  minimal pass-through VS/PS source into a GSH. The vertex program is stored
  in a `VertexShaderProgram` GFD block (Decaf's `src/libgfd/src/gfd_read.cpp`
  copies that block as raw bytes). This supplies a reproducible source-to-GSH
  fixture without adding a build dependency or copying third-party code.
  The inspected program block starts with 24 bytes:
  `00 00 00 00 00 00 80 09 3C A0 00 00 88 06 00 94 00 40 01 00 88 06 20 14`.
  Decaf's `assembler_cf.cpp` sets `CALL_FS` in `inst.word1`, and `gfd.cpp`
  copies the `ControlFlowInst` array into the binary. Interpreting consecutive
  DWORDs as little-endian values yields three pairs:
  `(word0=0x00000000, word1=0x09800000)`,
  `(word0=0x0000A03C, word1=0x94000688)`, and
  `(word0=0x00014000, word1=0x14200688)`. Cemu's
  `LatteDecompiler.cpp` reads a `uint32` from offsets +0/+4 as `cfWord0` and
  `cfWord1`, extracts the opcode from bits 29:23, and checks end-of-program at
  bit 21. These pairs decode to `CALL_FS` (0x13), `EXPORT_DONE` (0x28), and
  `EXPORT_DONE` (0x28, EOP set), matching the assembly source. The pairing and
  field order are therefore cross-checked between the generated fixture and
  Cemu's parser.

  Affogato's `read32_be()` capture produces the corresponding values
  `0x00000000, 0x00008009, 0x3CA00000, 0x88060094, 0x00400100, 0x88062014`.
  To obtain the fixture's decoder DWORDs, each captured word must be byte-swapped
  with `std::byteswap`; the resulting sequence matches the pairs above. This
  resolves the captured-word conversion for this controlled GSH fixture. It
  does not establish every Latte CF opcode or family-specific semantic.
- The [Decaf Wii U tests](https://github.com/decaf-emu/wiiu-tests) include
  shader-oriented guest tests, but their README requires `latte-assembler`,
  AMD GPU ShaderAnalyzer, and devkitPro with WUT. The repository is GPL-3.0.
  This environment currently has neither `DEVKITPRO` nor a discoverable
  `latte-assembler`/`devkitPPC` command, so those RPX tests cannot presently be
  built here. Keep it as a later independent conformance source; do not copy
  its implementation into Affogato without license review.
- `decaf-emu/addrlib` is licensed GPL-3.0-or-later and addresses surface
  layout, not shader translation. Shader code reuse requires separate license
  review. No external implementation code has been copied into Affogato.

## Current coverage

| Capability | Status | Evidence / dependency |
| --- | --- | --- |
| GX2 vertex/pixel/geometry descriptor capture | Implemented and tested as capture | HLE descriptor/program range checks and draw-time state snapshots |
| Fetch-shader parsing and semantic linkage | Narrow CPU subset | `latte_vertex_fetch.hpp` recognizes Affogato-generated semantic VTX records, decodes supported formats, applies destination selectors (component swizzles, 0.0, 1.0, and masked writes), extracts a complete correctly typed 32-entry `SQ_VTX_SEMANTIC` table from a draw snapshot, and maps attributes plus immutable vertex snapshots into a caller-provided register file at an explicit vertex index and `VS_GPR_BASE + 1 + table index`. `execute_latte_vertex_shader_subset()` passes this input file to the supported CPU shader subset in one deterministic test path. Writes are validated transactionally. AMD references differ on semantic-field width (R6xx: 8; R700 ISA Chapter 5: 9, while its Chapter 9 semantic-format table says 8); Latte applicability remains unresolved. Arbitrary guest fetch programs are not executed, normal draw dispatch is not connected, and system-generated inputs remain unsupported. No Wii U hardware verification. |
| Latte shader control-flow decoding | Partial | `latte_shader_decoder.hpp` decodes fixture-verified `CALL_FS`, `EXPORT`, `EXPORT_DONE`, and `ALU` clause headers, including EOP, bounded clause ranges, normal export metadata, and `CALL_FS` address/pop/condition/nesting fields. The generic straight-line executor rejects calls. A vertex-specific path accepts one leading canonical call only after separately processing the captured HLE fetch subset. Other call forms, branches, and loops remain unsupported; exports are typed values and are not connected to stage outputs. This behavior is not Wii U hardware-verified. |
| Latte ALU decoding/execution | Experimental subset | OP2/OP3 fields and LAST-delimited no-literal groups are decoded. The CPU clause/program subset executes homogeneous groups and mixed four-lane X/Y/Z/W groups for supported OP2 operations plus floating CMOVE/CMOVGT/CMOVGE and integer CMOVE_INT/CMOVGT_INT/CMOVGE_INT OP3 operations. Sources are read from the pre-group state, writes commit by group, PV forwards between groups, and clause state commits transactionally. Integer conditions use raw-zero equality and signed comparisons; integer source negation is rejected. ALU.Trans/PS, kcache/literal reads in the shader IR, other opcodes, relative addressing, general control flow, normal draw integration, and Latte-hardware validation remain absent. |
| Vertex/pixel shader translation | Experimental vertex subset | `latte_spirv_vertex_subset.hpp` emits SPIR-V for no-literal OP2 MOV, binary32 ADD, IEEE binary32 multiply, and DX10 min/max, plus floating and integer OP3 conditional moves and mixed OP2/OP3 four-lane groups, followed by one identity position export. Generated mixed-group modules pass `spirv-val`; a controlled GX2 guest-word fixture executes an OP2 MOV plus OP3 conditional-move program through Vulkan and confirms red framebuffer pixels using the diagnostic fragment shader. Pixel shader translation, general interfaces, constants, textures, normal draw integration, and direct Latte hardware validation remain missing. |
| Guest shader integration with Vulkan draws | Missing | Shader modules, pipeline cache, descriptor binding, render-target path |
| WWHD shader requirements | Unknown in Affogato | Existing recorded WWHD runs stop before a guest draw is captured |

Do not mark the guest shader milestone complete when programs are merely
captured, parsed, disassembled, or accepted by a host SPIR-V validator.

## Translation design constraints

The implementation should keep the stages explicit:

`guest program bytes → verified word decoder → typed instruction records →
validated IR → SPIR-V → Vulkan shader module`

The decoder must preserve control-flow clauses and instruction grouping.
ALU IR must represent simultaneous VLIW slots and architectural forwarding
rules rather than assuming source-order execution. Register masks, swizzles,
predication, constants, literals, exports, fetches, texture operations, and
stage-specific interfaces need explicit representation. Unsupported opcodes
must produce a structured decode/translation error with stage and word offset;
they must never turn into a successful no-op shader.

Keep CPU shader interpretation as a deterministic test oracle for the first
small subset if it can be made faithful to the verified ISA. The Vulkan
backend should consume validated IR or generated SPIR-V and should not acquire
guest-memory ownership. Captured shader bytes are already immutable in
`Gx2ProgramShaderBinding`; texture bytes and mutable GPU state have separate
snapshot/coherence requirements.

## Acceptance-test plan

1. Add table-driven decoder fixtures for each supported instruction family,
   including invalid encodings, truncation, control-flow bounds, and word
   ordering.
2. Test IR evaluation against independently specified results for normal,
   boundary, NaN/infinity, and forwarding cases where relevant.
3. Validate generated SPIR-V structurally and with `spirv-val` when available.
4. Execute controlled vertex/pixel programs through Vulkan and verify
   framebuffer readback, not only successful module creation.
5. Keep diagnostic host shaders explicitly separate from translated guest
   shaders.
6. Only use a WWHD capture after the game reaches a draw; record shader stage,
   code bytes/hash, bound resources, render-target formats/layouts, and
   observed result without publishing game data into the repository.

## Unknowns and next investigation

- Exact Latte/Espresso differences from the R600 ISA document.
- Instruction stream endianness and pairing as seen by hardware after
  `GX2Set*Shader`/resource initialization.
- Complete CF clause encodings, branch targets, and termination rules.
- ALU vector/transcendental scheduling and all PV/PS forwarding corner cases.
- Texture, uniform, literal, and interpolator encodings required by controlled
  guest workloads.
- Which shader families an available WUT test actually emits and whether the
  current toolchain can build and run that test in this environment.
- Whether real game shader loading yields the same byte representation as the
  controlled GSH fixture, and which CF encodings or semantics differ from the
  R700-family reference.
- Whether the R700 ALU clause address and count interpretation applies without
  Latte-specific adjustment to shader words captured by Affogato.

The non-executing CF record decoder currently supports the pass-through
fixture's `CALL_FS` and export records, plus a separately assembled ALU
fixture's clause header and raw slot extraction. The ALU header uses the upper
four opcode bits combined with `0x40`; the supported fixture resolves to opcode
`0x48`. Its
22-bit `word0` address is measured in eight-byte units, and the seven-bit
`word1` count field is encoded as slot count minus one. The fixture places four
64-bit slots at address `0x20`, ending exactly at the 288-byte program boundary.
This is cross-checked against Cemu's current CF opcode/count/address extraction
and the Decaf assembler output. The decoder checks that the referenced clause
range fits in the captured program but does not decode clause contents.

Tests cover the Affogato-captured guest words, byte-swapped field words, opcode
fields, EOP, incomplete records, unsupported CF and ALU opcodes, missing EOP,
and an out-of-range ALU clause. Do not execute clauses, infer omitted slot
behavior, interpret ALU opcode fields, or generate SPIR-V in this step. Raw
slot extraction validates that its input record is an ALU clause and rechecks
that the clause range fits the captured program.

The OP2 field decoder extracts raw operation ID, source selectors,
channels, relative/negate flags, index mode, destination register/component,
clamp/write-mask, output modifier, and group-end marker. It rejects slots whose
word1 format bits identify OP3. The parser does not classify selector values
or assign operation behavior. A controlled MOV fixture checks source r1 to
destination r3 components and the final-in-group marker against Cemu's
`LatteClauseInstruction_ALU` fields and `MOV` opcode `0x019`.
`evaluate_latte_alu_move_value()` handles the fixture's absolute GPR source
and returns its raw 32-bit component unchanged. The R700 reference describes
MOV as `dst = src0` and also permits constants or previous results; Affogato
currently supports only the GPR source form. The vector-group executor applies
the supported destination writes and returns the group result; other source
modes and previous-scalar state remain unsupported. The supporting R700 reference
is Chapter 9.2, “Copy To GPR” (page 9-92); its applicability to Latte remains a
family-level reference cross-checked by the assembled fixture, not a complete
Latte specification.

`decode_latte_alu_op3_instruction()` now extracts the raw OP3 opcode, the
shared source 0/source 1 fields, source 2 selector/relative/channel/negate
fields, index mode, bank swizzle, destination, predicate selection, clamp,
and group-end state. The R700 manual's Chapter 10 `ALU_DWORD1_OP3` layout
places source 2 in word 1 bits 0–12 and the five-bit opcode in bits 13–17;
its Chapter 9 table identifies `MULADD_IEEE` as OP3 opcode `0x14` and gives
`dst = src0 * src1 + src2`. Cemu's
[`LatteDecompiler_ParseALUClause`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.cpp)
parses the same source-2 fields from word 1. Affogato uses this as a structural
cross-check and does not copy Cemu code. At this stage, the decoder was
structural only; later bounded work adds an OP3-only CPU clause and conditional
move SPIR-V lowering. Other OP3 operations remain unsupported. In particular,
the exact Latte multiply-add rounding/contraction behavior still needs
Latte-specific evidence before choosing `OpFma` versus separate multiply and
add operations. The original constructed-field fixture remains in the CPU
test suite; later execution and SPIR-V checks are recorded in the development
status.

`decode_latte_alu_op3_groups()` now segments OP3-only slot spans at LAST and
rejects literal source selectors and unterminated groups. It preserves each
decoded raw instruction and the first slot's word offset; it does not enforce
four vector lanes, handle ALU.Trans or literals, or mix OP2 and OP3 in a single
group. Tests cover four ordered destination lanes, LAST handling, literal
rejection, and missing final LAST. Debug and Release CTest each passed 18/18.

The CPU reference subset now executes OP3 floating-point `CMOVE` (`0x18`),
`CMOVGT` (`0x19`), and `CMOVGE` (`0x1A`) in four ordered X/Y/Z/W slots. The
R700-family manual, Chapter 9, pages 9-50, 9-52, and 9-54, specifies the
zero comparisons and source1/source2 selection. Ordered C++ float comparisons
make NaN select source2 for all three forms; both signs of zero compare equal
to zero. Source negation toggles the binary32 sign bit, and the selected value
is copied as raw bits. Relative sources, nonzero index mode or bank swizzle,
predication, destination-relative addressing, clamping, and other OP3 opcodes
are rejected. A four-lane group gathers all results before updating guest
registers, so rejected groups do not partially write state. Debug and Release
CTest each passed 18/18. This is based on the R700-family manual, not direct
Latte hardware output; it is not integrated into the typed shader IR or SPIR-V
emitter. Later work integrates these instructions into the CPU shader program
subset and OP3 SPIR-V emitter. A controlled Vulkan test verifies one covered
and one uncovered readback pixel using an OP3 guest vertex program and a
diagnostic host fragment shader. This is not direct Latte hardware output.
See the [AMD R700 ISA manual](https://docs.amd.com/v/u/en-US/R700-Family_Instruction_Set_Architecture).

The integer OP3 subset now additionally executes `CMOVE_INT` (`0x1C`),
`CMOVGT_INT` (`0x1D`), and `CMOVGE_INT` (`0x1E`). AMD's R700-family ISA
manual, Chapter 9, integer OP3 entries, describes equality to zero, signed
greater-than-zero, and signed greater-or-equal-zero respectively. The CPU
reference uses exact 32-bit zero checks and `std::bit_cast<std::int32_t>` for
the signed comparisons; SPIR-V lowers these through signed `OpBitcast`,
`OpIEqual`, `OpSGreaterThan`, and `OpSGreaterThanEqual`, then selects the
original unsigned source bits. Source negation is explicitly rejected for
these integer forms because this bounded subset does not model an integer
negation modifier. Regression tests cover zero, negative, positive, and
`INT_MIN` conditions, SPIR-V validation, and a controlled GX2/Vulkan output
readback. This is still R700-family evidence rather than Latte hardware
verification, and the Vulkan fixture uses a diagnostic host fragment shader.

The typed shader IR now also preserves supported OP2 and OP3 operations in the
same four vector slots. This follows the R700 manual's Chapter 4.3 slot model:
X, Y, Z, and W are independently encoded vector operations sharing a group
boundary. Affogato snapshots source registers before evaluating the four
lanes, applies the supported operations, and commits writes after evaluation;
PV is then forwarded to the next group in that clause. A deterministic
guest-word fixture verifies one OP2 MOV lane alongside three OP3 conditional
moves. Both the CPU executor and SPIR-V emitter now lower the supported mixed
group. The current executor still requires exactly four vector slots, so it
does not implement the manual's optional omitted slots or ALU.Trans slot, and
literal payloads remain rejected. A generated mixed-group SPIR-V module passes
`spirv-val`; the controlled `--gx2-guest-triangle-mixed-alu-spirv-smoke-test`
also executes guest-encoded mixed bytecode through the explicit diagnostic
Vulkan path and confirms red pixels in framebuffer readback. It uses a
diagnostic host fragment shader and is not a normal guest draw or Latte
hardware test. This design is informed by R700 Chapter 4, §4.3, and Cemu's
`LatteDecompiler_ParseALUClause` in the previously linked parser source; no
third-party implementation code was copied. These are not Latte hardware
captures.

The R700 manual's Chapter 4.3 says the LAST bit delimits groups of ALU slots;
Chapter 4.6.2 defines PV/PS as results from the previous group. Chapter 4.9.2
also states that an OP2 write-mask suppresses the GPR write while PV/PS still
update. The implemented executor therefore requires the write mask and applies
destination writes only for that supported form; other write-mask behavior and
PS state are not modeled.

Chapter 4.3 also describes group instruction slots in X, Y, Z, W, Trans order;
literal constants may occupy following 64-bit slots and do not carry LAST bits,
while the ALU clause count is expressed in total slots. Chapter 4.4 describes
how destination components select vector units and how transcendental
operations use ALU.Trans. This is R700-family evidence, so Affogato must still
confirm these details against Latte-focused sources and fixtures. In
particular, a group parser must distinguish instruction slots from literal
payloads before treating each LAST bit as a complete group boundary.

`decode_latte_alu_op2_groups()` now groups OP2-only slots by LAST for clauses
without literal operands. It rejects OP3 encodings, literal selectors, and an
unterminated final group. This does not parse literal payload slots or execute
groups. The pass-through fixture contains one four-slot group and no literals.

`execute_latte_alu_op2_vector_group()` supports four ordered X/Y/Z/W slots with
MOV, integer bitwise AND/OR/XOR/NOT, integer ADD/SUB, signed/unsigned integer
MAX/MIN, and integer set comparisons. Sources are absolute GPRs, explicitly
supplied previous-PV values, or inline constants `248`–`252`; write masks must
be enabled and other modifiers are rejected. The OP2 decoder records
`SRC0_ABS` and `SRC1_ABS` from word1 bits 0 and 1. MOV applies its source
absolute and negate flags as binary32 sign-bit operations, in that order; the
integer evaluator rejects both flags.
This follows the R700 instruction-family semantics and retains all non-sign
bits, but it has not been checked against Latte hardware. The executor gathers
all sources before updating destination lanes and returns the resulting vector
value. A missing PV input is rejected before any destination write. Tests cover
bit-preserving and modified MOV values, same-group source/destination overlap,
PV forwarding, inline constant bit patterns, integer logical truth tables,
wrapping integer arithmetic boundaries, signed/unsigned min/max ordering, and
rejection without partial register writes.
The group behavior follows the R700-family reference and the assembled fixture;
it is not independently verified against Latte hardware. Absolute constant-file
reads are implemented for selectors 256 through 511 using an explicit caller-provided
vector span. PS, literal and kcache reads, relative addressing, other
groups/opcodes and SPIR-V translation remain absent. The pass-through
shader does not establish complete shader decoding.

For the integer logic subset, the R700 manual documents `AND_INT` as
`src0 & src1` (Chapter 9, “AND Bitwise”), `NOT_INT` as `~src0` (Chapter 9,
“Bit-Wise NOT”), and corresponding OR/XOR operations in the integer instruction
tables. Cemu's `LatteClauseInstruction_ALU` opcode definitions map these
instructions to `0x030`–`0x033`. Affogato tests those raw opcode fields through
its decoder and checks the bitwise results. These references are
family/implementation evidence; no Latte hardware output has been captured
for this subset.

The OP2 word1 fields also include `UPDATE_EXECUTE_MASK` (bit 2),
`UPDATE_PRED` (bit 3), and `BANK_SWIZZLE` (bits 18–20), as listed in the R700
Chapter 10 field table. Predicate/mask updates mutate shader execution state,
and bank swizzle controls source read-cycle selection (Chapter 4, Sections
4.7.7 and 4.9.3). Word0 also contains `PRED_SEL` in bits 29–30 (Chapter 10,
page 10-17); it controls per-instruction predicate execution. The bounded
evaluator does not model predicate state or read-cycle selection, so it decodes
these fields and rejects any nonzero value before guest-register writeback.
Only the zero/default bank swizzle and `PRED_SEL_OFF` are accepted.

The R700 manual documents `ADD_INT` as `src0 + src1` (opcode `0x034`) and
`SUB_INT` as integer subtraction (opcode `0x035`) in Chapter 9 and the OP2
opcode table in Chapter 10. Their bounded evaluator uses `std::uint32_t`
addition/subtraction, giving defined modulo-2^32 behavior independent of signed
overflow. Tests include carry out, borrow, and high-bit operands. This follows
the documented signed/unsigned agnostic integer behavior, without Latte
hardware validation.

The OP2 opcode table assigns `MAX_INT`, `MIN_INT`, `MAX_UINT`, and `MIN_UINT`
`0x036` through `0x039`. The R700 instruction descriptions specify signed
ordering for the first pair and unsigned ordering for the second. The
implementation maps signed order to unsigned order by flipping bit 31 in each
operand, then returns the original selected operand. Tests cover values across
the signed boundary and unsigned high-bit cases. This is reference-derived and
not verified against Latte hardware.

The R700 OP2 table assigns floating-point `ADD`, `MUL_IEEE`, `MAX_DX10`, and
`MIN_DX10` opcodes `0x000`, `0x002`, `0x005`, and `0x006` (Chapter 10). The
Chapter 9 descriptions define add and IEEE multiply and specify ordered
selection for DX10 min/max: MAX chooses source 0 on `src0 >= src1`, while MIN
chooses source 0 only on `src0 < src1`; otherwise each selects source 1. These
ordered comparisons also define which input is selected for NaNs and equal
signed zeros. Affogato evaluates those four operations with sign-bit absolute
and negate modifiers and the scalar output multiplier. Tests cover normal
values, NaN selection, signed-zero tie behavior, source modifiers, and OMOD.
This is a bounded R700-family CPU model, not Latte hardware verification.
The reference is AMD's [R700-Family Instruction Set Architecture, Chapters 9
and 10](https://docs.amd.com/v/u/en-US/R700-Family_Instruction_Set_Architecture).

The R700 OP2 table assigns `SETE_INT`, `SETGT_INT`, `SETGE_INT`, `SETNE_INT`,
`SETGT_UINT`, and `SETGE_UINT` opcodes `0x03A`–`0x03F`. The Chapter 9
descriptions define signed or unsigned comparisons and write `0xFFFFFFFF` for
true or zero for false. Affogato implements equality/inequality, signed greater
and greater-or-equal using the same sign-bit order mapping as integer min/max,
and unsigned greater/greater-or-equal using unsigned comparison. Tests exercise
raw opcode decoding, both truth values, and comparisons across the signed and
unsigned boundaries. This is not validated on Latte hardware.

The R700 microcode table assigns selectors `248`–`252` to the inline constants
`0.0`, `1.0`, integer `1`, integer `-1`, and `0.5`; selector `254` is PV and
`255` is PS (Chapter 10, `ALU_WORD0`). Affogato returns the exact 32-bit
patterns for those inline constants and accepts PV only when the caller
supplies the preceding supported vector-group result. The R700 source-address
table assigns selectors `256 through 511` to absolute constant-file vectors
`c[0, 255]` (Chapter 4.7.1). The bounded CPU executor now accepts those vectors
as an explicit caller-provided span and reads the selected component for its
supported OP2 operations. The caller must provide the selected vector; missing
entries fail explicitly. Selector `253` literals, kcache selectors
`128 through 191`, relative addressing, and selector `255` PS remain unsupported.
This absolute constant-file mapping is R700-family behavior and has not been
validated on Latte hardware. Although GX2 uniform-register uploads are now
captured into draw-time raw register snapshots, no adapter yet builds stage
constant vectors from those snapshots or calls the shader subset for normal
guest draws. An opt-in triangle diagnostic does call the draw-snapshot vertex
subset using an explicit GPR base of zero; this is a controlled test path, not
normal guest draw dispatch or a Latte hardware result.

The R700 `ALU_WORD1_OP2` field table places `SRC0_ABS` at bit 0 and `SRC1_ABS`
at bit 1 (Chapter 10, page 10-18). It describes absolute value before
negation and says to use it only for floating-point inputs. The decoder exposes
these bits. The bounded MOV evaluator implements the specified ordering by
clearing the binary32 sign bit for absolute value and toggling it for negation;
the integer logic subset rejects either modifier. Source 1 modifiers for MOV
remain irrelevant because this opcode consumes only source 0.

The R700 manual's Chapter 4.6.2 defines PV as the four vector-unit results of
the previous instruction group and says PV/PS do not survive the end of an ALU
clause. `execute_latte_alu_op2_clause()` applies that bounded rule by forwarding
the prior group's vector result only while evaluating one clause. The executor
supports only four vector slots per group: it does not parse or execute the
fifth ALU.Trans slot, NOP behavior, or mixed-width groups. It evaluates on a
temporary register file and commits only after every group is supported, so an
unsupported later group cannot leave partial host-side test state. Tests cover
PV forwarding across two groups and rejection without partial register writes.
This is R700-family behavior and has not been validated against Latte hardware.

`execute_latte_shader_subset()` connects the tested OP2 and OP3 clause
executors and normal export materializer for a bounded straight-line program.
It processes CF records in guest order, resolves each ALU clause from its
bytecode address, and records `EXPORT`/`EXPORT_DONE` batches in order. A clause
may contain only OP2 or only OP3 slots; mixed clauses are explicitly rejected
until a unified group model can preserve mixed slot semantics. The caller
supplies the initial GPR file, which means fetch-shader input generation is
outside this subset. Unsupported CF records, including `CALL_FS`, fail
explicitly. The fixture test exercises an encoded OP3 clause through program
decoding, CPU execution, and position export. The controlled Vulkan test also
uses the OP3 SPIR-V emitter and verifies framebuffer readback. Normal guest
draws remain capture-only, and this test does not establish general guest
draw or shader integration.

Normal `EXPORT` and `EXPORT_DONE` records use the R700 Chapter 3.4.1
`CF_ALLOC_EXPORT_DWORD0` and `CF_ALLOC_EXPORT_DWORD1_SWIZ` formats. The manual
defines the output type (pixel, position, or parameter), source GPR and optional
relative addressing, burst count, and four component selectors; the last
export for a type is marked by `EXPORT_DONE`. Cemu's
`LatteCFInstruction_EXPORT_IMPORT` accessors use matching fields, and the
assembled pass-through vertex-shader fixture decodes its first export as
position from GPR 1 with identity XYZW selectors. Affogato now exposes those
fields on the decoded CF record. `evaluate_latte_normal_export()` applies the
absolute-GPR source, burst, and component-selector rules to produce typed
CPU-side output records; it rejects relative addressing, buffer encoding, and
invalid GPR ranges. It does not link these records to Vulkan stage outputs or
implement CF export ordering, relative GPR addressing, or memory-buffer export
semantics. R700/Cemu evidence is not independent Latte hardware validation.

## Controlled vertex-subset diagnostic path

The `--gx2-guest-triangle-smoke-test` harness now installs a bounded guest
vertex shader through `GX2SetVertexShader`. It declares all 32 semantic-table
entries, maps semantic 0 to table index 0, and uses the existing fixture's
straight-line MOV program to copy fetched input GPR 1 to GPR 3 and export the
result as position. `GuestDrawPolicy::diagnostic_latte_vertex_position_base_zero`
executes Affogato's HLE fetch subset and supported CPU shader subset from each
draw-time snapshot, then submits exported XY through the fixed diagnostic
Vulkan pipeline. Missing shaders, unsupported shader operations, absent XY
exports, multiple position exports, and out-of-range clip-space XY fail
explicitly. The existing raw-attribute diagnostic policy remains separately
available, and normal frontend execution still uses capture-only unless its
diagnostic draw option is enabled.

The test's GPR base zero is explicit. AMD's related-generation
[R6xx/R7xx 3D Engine Programming Guide, Sections 2.5.1 and 2.5.3](https://www.x.org/docs/AMD/old/R6xx_R7xx_3D.pdf)
describes fetch semantic linkage as `VS_GPR_BASE + 1 + table_index` and says
the base is typically zero. That supports the controlled test setup, but does
not establish the register's value for arbitrary Wii U shaders. The output is
still rendered with host diagnostic shaders, does not write a guest render
target, and does not demonstrate general Latte shader execution or WWHD
rendering.

## Translation architecture audit (2026-11-25)

### Verified Affogato state

`latte_shader_decoder.hpp` produces typed control-flow records, decoded OP2
fields/groups, and export metadata. `latte_shader_program.hpp` now converts
the supported straight-line OP2 clauses and normal exports into a typed
`LatteShaderIrProgram`; the existing CPU reference executor runs this
representation. A leading canonical `CALL_FS` can be omitted from the IR only
for the vertex path that separately handled Affogato's captured fetch subset.
`latte_shader_interpreter.hpp` executes a deliberately bounded CPU subset and
accepts decoded ALU groups, while retaining the raw-slot clause overload for
other callers. `latte_vertex_shader_subset.hpp` connects captured draw-time
fetch inputs, semantic registers, shader words, and vertex constants for one
selected vertex. `Gx2RegisterState::draw_snapshot()` provides a borrowed view
over one draw's captured state. The controlled diagnostic path calls this CPU
subset. A bounded MOV/ADD/MUL/DX10-minmax-to-position module generator now exists. An explicit
diagnostic policy creates a Vulkan pipeline from it and submits controlled
guest triangle draws; there is no normal guest shader cache or renderer path.
The IR currently represents only the existing
supported subset; it does not establish broader Latte shader coverage or
independently verified hardware semantics.

The Vulkan SDK installed for this workspace provides `glslc`,
`glslangValidator`, `spirv-as`, and `spirv-val`, plus glslang and shaderc
libraries. Affogato's CMake currently links Vulkan but does not discover or
link those shader compiler libraries. The diagnostic vertex/fragment SPIR-V
is embedded in `src/gpu/vulkan/diagnostic_shader_spirv.hpp`. This establishes
local validation tools; it does not establish a portable runtime compiler
dependency.

### Bounded SPIR-V vertex emitter

`latte_spirv_vertex_subset.hpp` lowers the shared IR for a single
`float32x4` input mapped to a caller-selected guest GPR and Vulkan location,
four-lane OP2 MOV/arithmetic/min/max groups, OP3 floating conditional-move
groups, and one completed identity `POSITION` export at array base `0x3c`. MOV
reads initialized absolute GPRs or supported inline constants/PV; source
absolute and negate are emitted as 32-bit sign/magnitude bit operations so
NaN payloads and signed zero are preserved. OP3 comparisons use ordered
SPIR-V float comparisons against zero and select raw 32-bit values. ALU
constant-file reads, additional inputs, parameter exports, pixel shaders, and
guest control flow remain unsupported.

The modules use SPIR-V 1.0 and the Vulkan GLSL450 memory model. A CPU-only
CTest generates pass-through, modifier, arithmetic, min/max, and OP3
conditional-move modules; when the SDK tool is available, it runs
`spirv-val --target-env vulkan1.0` on each. The OP3 test checks `OpFOrdEqual`,
`OpFOrdGreaterThan`, `OpFOrdGreaterThanEqual`, and `OpSelect` emission, and
rejects relative sources. `affogato_vulkan_guest_op3_conditional_move_readback`
executes a controlled guest `CMOVGE` vertex shader and verifies that the
translated output covers one readback pixel and leaves another uncovered.
This uses a diagnostic host fragment shader and a test GX2 draw path; it does
not verify the shader on Latte silicon or normal game rendering. `spirv-dis`
confirms the input
load, component bitcasts, built-in position store, and modifier bitwise
operations. This verifies
module validity and encoding of this narrow pass-through shader. The explicit
`--gx2-guest-triangle-spirv-smoke-test` then creates a Vulkan graphics pipeline
from the generated module and submits captured guest draw ranges, using the
fixed diagnostic fragment shader. The Debug run enables the Khronos validation
layer when installed and emitted no validation messages. The explicit test
also copies a swapchain image to host-visible memory and checks an interior
pixel for the diagnostic fragment output. This verifies rasterization of the
captured guest vertex input and bounded MOV position output in that test path;
it is not visible-window verification or CPU-versus-GPU output equivalence.
Normal guest draws remain capture-only, and the fragment shader remains fixed
host diagnostic code.

The [Khronos SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
defines module header and word-count encoding, logical instruction ordering,
entry-point interface declarations, and integer bitwise operations used by
the emitter. The Affogato generator is independently implemented; no
third-party code was copied.

### Reference implementation findings

- Cemu's
  [`LatteShader.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Core/LatteShader.cpp)
  owns stage shader caches and links parsed shader state to renderer shader
  creation. Its shader keys combine guest program data with state that changes
  translation, including fetch-shader state, pixel-shader imports, clip/output
  state, streamout, primitive handling, and selected auxiliary state. Uniform
  values and mutable buffers are not simply treated as shader source bytes.
- Cemu's experimental
  [`LatteTC.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Transcompiler/LatteTC.cpp)
  groups control-flow records into basic-block nodes and emits a `Zir` IR.
  The checked source labels this transcompiler experimental/WIP; it is an
  architectural reference, not evidence of complete shader coverage.
- Cemu's legacy
  [`LatteDecompiler.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.cpp)
  parses CF records, exports, ALU, texture, and fetch records into an
  intermediate representation and generated shader source. Its Vulkan
  renderer compiles generated shader source through glslang, as visible in
  [`VulkanRenderer.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.cpp).
- Decaf's GPL-3.0 reference implementation has a Latte parser under
  `src/libgpu/src/latte/` and a direct SPIR-V backend under
  `src/libgpu/src/spirv/`. The
  [`spirv_transpiler.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/src/spirv/spirv_transpiler.cpp)
  layers on the parser and shader SPIR-V builder; operation families are
  separated into files such as
  [`spirv_alu_op2.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/src/spirv/spirv_alu_op2.cpp),
  [`spirv_export.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/src/spirv/spirv_export.cpp),
  and [`spirv_vtx.cpp`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libgpu/src/spirv/spirv_vtx.cpp).
  These files demonstrate a maintained parser/backend split and per-family
  lowering, but many operations rely on Decaf-specific state and assumptions.
  Decaf's GPL-3.0 code was inspected for architecture only; no implementation
  code is copied into Affogato.

### Design constraints for Affogato

The translator must reuse the current decoder and its explicit rejection
behavior rather than create a second opcode parser. `LatteShaderIrProgram` is
now the initial boundary between bytecode decoding and backends; its CPU
subset covers OP2 clauses, OP3-only conditional-move clauses, and normal
exports. Mixed OP2/OP3 clauses remain unsupported. The CPU subset remains the
semantic reference. The SPIR-V backend lowers no-literal MOV, binary32 ADD,
IEEE binary32 MUL, DX10 MIN/MAX, and OP3 conditional moves to one position
export. For OP2 binary operations it accepts absolute GPR/PV/inline-constant
sources, source absolute/negate modifiers, and unmodified destination writes;
other source modes, modifiers, and outputs are rejected. The emitter uses
SPIR-V `OpFAdd` (opcode 129), `OpFMul` (opcode 133), ordered comparison
(`OpFOrdGreaterThanEqual` 190 for MAX_DX10 and `OpFOrdLessThan` 184 for
MIN_DX10), plus `OpFOrdEqual` (180), `OpFOrdGreaterThan` (186),
`OpFOrdGreaterThanEqual` (190), and `OpSelect` (169) for OP3. These OP3
modules pass `spirv-val`, but have not yet been executed through Vulkan
readback.
AMD's [R700-family ISA manual](https://docs.amd.com/v/u/en-US/R700-Family_Instruction_Set_Architecture),
section 9.2, describes OP2 ADD opcode `0x000` as floating-point
`src0 + src1`; this is family-level evidence and has not been confirmed on
Latte hardware. The same manual, section 9.2, identifies `MUL_IEEE` opcode
`0x002`, uses the same two-source multiply expression, and specifies IEEE
zero-times-infinity behavior. This remains family-level evidence; the module
validator can verify SPIR-V structure but cannot verify that behavior on Latte.
The
[SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
defines the ordered comparisons to return false for unordered operands. This
matches the CPU subset's `src0 >= src1 ? src0 : src1` maximum and
`src0 < src1 ? src0 : src1` minimum selection, including choosing source 1
for NaNs and ties in the minimum case. It defines `OpSelect` as choosing
between supplied values without arithmetic conversion. Continue to reject every
operation, control-flow form, source mode, or export not implemented. Extend
the supported set only with deterministic CPU-reference and `spirv-val`
coverage.

Translation variants must include all shader-affecting captured state, at
minimum stage and bytecode, fetch program, vertex semantic linkage, relevant
output/clip controls, and resource-interface declarations. Constant values,
vertex/index bytes, and image contents should remain draw resources rather
than shader-variant keys unless verified hardware behavior makes them part of
the generated interface. A cache key cannot be finalized until Affogato
captures the required per-stage interface state.

For the first genuine Vulkan shader milestone, require both a supported Latte
vertex program and a supported Latte pixel program to produce the Vulkan stage
modules. A fixed host diagnostic shader may remain available only under its
explicit test policy. A translated shader pair alone is not an end-to-end
rendering milestone: render-target selection, pixel writes, synchronization,
scanout copying, and readback/presentation must also be verified.

### Uncertainties and test plan

The available AMD R6xx/R700 references describe related hardware families,
not the complete Latte ISA. Current fixtures establish only selected word
layouts and operations. Cemu and Decaf differ in code organization and may
contain workarounds or assumptions; neither is treated as hardware proof.
Unknown Latte-specific instruction encodings, input/system GPR initialization,
export layout, precision, and state-dependent translation requirements must
be resolved from Wii U SDK behavior, real guest fixtures, or cross-verified
reference paths before implementation.

Use checked-in deterministic bytecode fixtures and test each supported
decoder/IR operation against the CPU executor. Generate SPIR-V for those same
fixtures, validate modules with `spirv-val` where available, create Vulkan
shader modules, and compare actual shader outputs against CPU reference
results. Then add draw-level tests that verify guest render-target pixels
through scanout. Current tests cover OP2 arithmetic readback and OP3
conditional-move module validation plus a controlled Vulkan readback case;
broader randomized equivalence, guest render-target output, and normal draw
integration tests remain future work.
