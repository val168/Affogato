# Latte shader ISA and translation research

## Purpose and current boundary

Latte shaders are guest programs associated with GX2 vertex, geometry, and
pixel stages. Affogato currently captures the stage descriptors and bounded
program words when `GX2SetVertexShader`, `GX2SetGeometryShader`, or
`GX2SetPixelShader` runs. `Gx2RegisterState::apply()` snapshots those bindings
at each draw. Structural control-flow decoding recognizes fixture-verified
`CALL_FS`, `EXPORT`, `EXPORT_DONE`, and one ALU-clause header encoding. Raw
ALU slots, OP2 fields, no-literal OP2 groups, and a narrow four-lane shader
execution subset are also implemented. There is no shader IR or SPIR-V
generator, and the diagnostic Vulkan triangle uses fixed host shaders rather
than these captured guest programs.

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
  fetch semantic ID to its VS input GPR. Table extraction rejects incomplete,
  duplicate, or wrongly typed entries. The table is not yet consumed to
  populate shader invocation inputs. This fetch path is separate from the
  guest vertex/pixel shader ISA.
- `GX2InitFetchShaderEx` in `src/cpu/espresso/gx2_hle.cpp` emits one four-dword
  record per attribute. Its comments identify word 0 as a semantic fetch but
  word 1's low seven bits as `VTX_WORD1_GPR.DST_GPR`. AMD documents SEM and GPR
  as alternative word-1 formats, so this local combination needs bytecode and
  hardware-family cross-checking before it is used for real shader execution.
  `GX2SetFetchShader` captures the guest descriptor and program but does not
  itself emit `SQ_VTX_SEMANTIC` register writes. The current HLE therefore does
  not establish a complete fetch-to-VS input path.
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
- `src/gpu/latte_shader_program.hpp` executes a straight-line subset containing
  the supported ALU clauses and normal exports. It requires caller-provided
  initial GPRs and explicitly rejects fetch-shader calls, branches, texture
  clauses, and all unsupported CF operations. It is a testable CPU execution
  slice, not connected to guest draw submission or Vulkan.

## Evidence categories

### Documented interfaces

WUT's `include/gx2/shaders.h` and `include/gx2/surface.h` define GX2 shader
descriptors, register storage, and the API boundary. The reconstructed layout
table at `kinnay/Nintendo-File-Formats/src/libs/wiiu/gx2.md` records the
vertex-shader program size/address at offsets `0xD0/0xD4` and pixel-shader
program size/address at `0xA4/0xA8`. Affogato uses these offsets in
`gx2_hle.cpp`; its tests exercise descriptor capture. Interface layouts do
not define the full Latte machine semantics.

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

The fetch-shader instruction encoding remains separate and unresolved. The
R700 ISA describes semantic-ID and direct-GPR word-1 layouts as alternatives.
Decaf's
[`GX2InitFetchShaderEx`](https://github.com/decaf-emu/decaf-emu/blob/master/src/libdecaf/src/cafe/libraries/gx2/gx2_fetchshader.cpp)
also emits `VTX_INST_SEMANTIC` in word 0, then sets `attrib.location` through
the overlapping `SQ_VTX_WORD1_GPR.DST_GPR` field while filling the shared
format/selector fields. Affogato's generated instruction follows the same
word pattern. This corroborates the generator layout but does not establish
whether Latte interprets those low bits as semantic ID, direct GPR, or a
Latte-specific combination; Decaf's field alias is not itself a hardware
specification. Affogato's `GX2SetFetchShader` captures bytes but does not
decode them or populate shader GPRs. A Wii U-produced fixture is still needed
before relying on the interpretation for arbitrary guest fetch programs.

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
| Fetch-shader parsing and semantic linkage | Narrow diagnostic subset | `latte_vertex_fetch.hpp` decodes generated VTX programs, extracts a complete correctly typed 32-entry `SQ_VTX_SEMANTIC` table from a draw snapshot, and resolves 8-bit table IDs to `VS_GPR_BASE + 1 + table index`. AMD's R6xx register reference describes an 8-bit fetch semantic, while the R700 ISA manual says nine bits; applicability to Latte is unresolved. The helper does not decode guest fetch semantic fields or populate shader invocation inputs. System-generated inputs and arbitrary guest fetch programs remain unsupported. No Wii U hardware verification. |
| Latte shader control-flow decoding | Partial | `latte_shader_decoder.hpp` decodes fixture-verified `CALL_FS`, `EXPORT`, `EXPORT_DONE`, and `ALU` clause headers, including EOP, bounded clause ranges, and normal export metadata (type, GPRs, burst count, and component selectors). Unknown opcodes and malformed streams fail explicitly. Export values are not applied to stage outputs; branch/loop execution is absent. |
| Latte ALU decoding/execution | Experimental subset | OP2 fields and LAST-delimited no-literal groups are decoded. A bounded clause executor runs sequential four-slot X/Y/Z/W groups with MOV and selected integer operations, forwards PV between groups within the clause, and leaves external registers unchanged if a later supported-subset check fails. A straight-line shader subset connects ALU clauses to normal export materialization with caller-provided initial GPRs. Fetch shader execution, ALU.Trans/PS, constant-file/literal reads, other opcodes, general control flow, Vulkan use, and Latte-hardware validation are absent. |
| Vertex/pixel shader translation | Missing | Stage interfaces, exports, constants, textures, and SPIR-V backend |
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
it is not independently verified against Latte hardware. PS, literal and
constant-file reads, other groups/opcodes, clause execution, shader IR, and
translation remain absent. The pass-through shader does not establish complete
shader decoding.

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
supplies the preceding supported vector-group result. Selector `253` literals,
constant files, and selector `255` PS remain unsupported; PS requires a scalar
ALU.Trans implementation that Affogato does not yet have.

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

`execute_latte_shader_subset()` connects the tested clause executor and normal
export materializer for a bounded straight-line program. It processes CF
records in guest order, resolves each ALU clause from its bytecode address,
and records `EXPORT`/`EXPORT_DONE` batches in order. The caller supplies the
initial GPR file, which means fetch-shader input generation is outside this
subset. Unsupported CF records, including `CALL_FS`, fail explicitly; this
avoids pretending to execute a fetch shader or control-flow operation. The
fixture test removes the fetch-shader record from an assembled MOV/export
program and verifies the resulting GPR and position export. No shader is yet
executed by the emulator's normal draw path.

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
