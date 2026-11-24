# Latte shader ISA and translation research

## Purpose and current boundary

Latte shaders are guest programs associated with GX2 vertex, geometry, and
pixel stages. Affogato currently captures the stage descriptors and bounded
program words when `GX2SetVertexShader`, `GX2SetGeometryShader`, or
`GX2SetPixelShader` runs. `Gx2RegisterState::apply()` snapshots those bindings
at each draw. This preserves evidence for a later translator; no instruction
decoder, shader IR, guest shader executor, or SPIR-V generator exists in the
repository. The diagnostic Vulkan triangle uses fixed host shaders and does
not execute these captured guest programs.

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
  `src/Cafe/HW/Latte/LegacyShaderDecompiler/`, and
  `src/Cafe/HW/Latte/ISA/`. These files are reference material, not code to
  transplant.
- Decaf keeps its GPU implementation under
  [`src/libgpu`](https://github.com/decaf-emu/decaf-emu/tree/master/src/libgpu)
  and publishes a separate
  [Wii U test repository](https://github.com/decaf-emu/wiiu-tests). Its
  architecture and test assumptions need review before any fixture is adapted.
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
| Latte shader control-flow decoding | Missing | CF encoding and clause termination need verified fixtures |
| Latte ALU decoding/execution | Missing | VLIW slot, writeback, predicate, and PV/PS semantics |
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
- Whether a reproducible standalone Latte assembler is available through a
  compatible license and usable without the Wii U RPX build toolchain.

Next implementation work should begin with a source-backed decoder
specification and fixtures for a small, fully understood instruction family.
The introductory pass-through shader is a candidate for end-to-end fixtures
once a suitable assembler is available; it alone does not establish a decoder
specification. Do not start SPIR-V emission until decoding, slot grouping, and
expected architectural outputs have deterministic tests.
