# Latte shader ISA and translation research

## Purpose and current boundary

Latte shaders are guest programs associated with GX2 vertex, geometry, and
pixel stages. Affogato currently captures the stage descriptors and bounded
program words when `GX2SetVertexShader`, `GX2SetGeometryShader`, or
`GX2SetPixelShader` runs. `Gx2RegisterState::apply()` snapshots those bindings
at each draw. A small structural control-flow record decoder recognizes
fixture-verified `CALL_FS`, `EXPORT`, `EXPORT_DONE`, and one ALU-clause header
encoding. There is no ALU slot decoder, shader IR, guest shader executor, or
SPIR-V generator. The diagnostic Vulkan triangle uses fixed host shaders and
does not execute these captured guest programs.

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
  program subset for diagnostic host vertices. It is separate from the
  guest vertex/pixel shader ISA.
- `src/gpu/latte_shader_decoder.hpp` converts the captured big-endian words
  to the little-endian Latte DWORD values represented in GFD shader code, then
  decodes a narrow structural control-flow subset. It checks ALU clause header
  ranges but does not decode or execute clause payloads.
- `src/gpu/latte_shader_interpreter.hpp` contains an initial MOV source-value
  helper for one absolute GPR form. It returns raw register bits; it does not
  perform destination writeback, PV/PS updates, or instruction-group execution.

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
  hardware verification.
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
| Fetch-shader parsing | Narrow diagnostic subset | `latte_vertex_fetch.hpp`; generated VTX programs only |
| Latte shader control-flow decoding | Partial | `latte_shader_decoder.hpp` decodes fixture-verified `CALL_FS`, `EXPORT`, `EXPORT_DONE`, and `ALU` clause headers, including EOP and bounded clause ranges. Unknown opcodes and malformed streams fail explicitly. No branch/loop decoding or clause execution. |
| Latte ALU decoding/execution | Partial field/value subset | `decode_latte_alu_op2_instruction()` extracts the OP2 layout; `evaluate_latte_alu_move_value()` returns the source bits for one absolute GPR MOV form. No group scheduling, destination writeback, PV/PS update, predicate, or other opcode semantics. |
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
currently supports only the GPR source. It does not apply destination writes
or update previous-vector/previous-scalar state. The supporting R700 reference
is Chapter 9.2, “Copy To GPR” (page 9-92); its applicability to Latte remains a
family-level reference cross-checked by the assembled fixture, not a complete
Latte specification.

The R700 manual's Chapter 4.3 says the LAST bit delimits groups of ALU slots;
Chapter 4.6.2 defines PV/PS as results from the previous group. Chapter 4.9.2
also states that an OP2 write-mask suppresses the GPR write while PV/PS still
update. Therefore a complete MOV executor cannot be modeled as only a
destination-register copy. Any follow-up execution work must represent grouped
ALU results and their state transitions before claiming instruction execution.

Chapter 4.3 also describes group instruction slots in X, Y, Z, W, Trans order;
literal constants may occupy following 64-bit slots and do not carry LAST bits,
while the ALU clause count is expressed in total slots. Chapter 4.4 describes
how destination components select vector units and how transcendental
operations use ALU.Trans. This is R700-family evidence, so Affogato must still
confirm these details against Latte-focused sources and fixtures. In
particular, a group parser must distinguish instruction slots from literal
payloads before treating each LAST bit as a complete group boundary.

Next implementation work should add semantic execution for a verified ALU
group representation that preserves instruction/literal slot boundaries,
followed by independently tested instruction subsets and explicit guest
register/PV/PS state. The pass-through shader alone does not establish complete
shader decoding. Do not start SPIR-V emission until opcode semantics, slot
grouping, and expected architectural outputs have deterministic tests.
