# Affogato Development Status

## Matched varying and guest MOV/ADD/MUL through Vulkan readback (2026-11-25)

The new `--gx2-guest-triangle-matched-varying-move-smoke-test` runs a controlled
guest PPC/GX2 setup whose vertex shader exports a parameter with semantic
`0x31`. Affogato captures the draw-time VS/PS descriptors, links that semantic,
emits the guest VS position and parameter export at matching SPIR-V locations,
and translates the captured guest four-lane MOV group into a Vulkan fragment
shader reading the matched varying. Vulkan interpolates the varying;
deterministic readback verifies a green gradient in the emulated color buffer
and swapchain frame, and the scanout copy path remains in the test. This
executes the tested guest MOV subset, while other guest pixel operations,
ordinary game draw routing, and WWHD rendering remain unsupported.
The matched fragment emitter also supports float ADD and IEEE MUL. Controlled
guest shader fixtures execute MOV, ADD-by-0.5, and MUL-by-0.5, each passing
Vulkan readback for the interpolated gradient. Their generated modules pass
`spirv-val`. Source absolute-value and sign-bit negate modifiers are also
emitted and covered by the ADD module's SPIR-V validation, but not yet by a
Vulkan readback variant. This verifies the bounded operations on these inputs,
not general Latte arithmetic behavior. Full Debug and Release CTest passed
39/39 at commit `64358d1`, before modifier support.

## Fragment varying interface fixture (2026-11-25)

An explicit fragment passthrough module remains as an interface fixture and
accepts only the default perspective interpolation mode; flat, no-perspective,
centroid, sample, and point-sprite inputs fail explicitly. The matched-input
module consumes one linked `float4`, executes supported
four-lane guest MOV, float ADD, or IEEE MUL groups, and applies the guest MRT0
swizzle. Each generated module has a `spirv-val` fixture. The guest integration
tests verify MOV, ADD-by-0.5, and MUL-by-0.5 through Vulkan readback.

## Vertex shader parameter exports to SPIR-V (2026-11-25)

The bounded vertex SPIR-V subset now optionally emits selected identity-
swizzled parameter exports as output vectors at explicit SPIR-V locations.
Tests validate the module with `spirv-val` and reject absent or duplicate
export requests. The later controlled diagnostic maps a captured GX2 semantic
to a shared vertex/fragment location and verifies interpolation through
Vulkan readback. It does not establish ordinary guest draw support.

## Pixel shader defaults through controlled Vulkan guest draw (2026-11-25)

The bounded fragment SPIR-V emitter seeds its GPRs from linked GX2 pixel-input
defaults and folds supported MOV-only shader IR into a constant fragment
output. The explicitly selected diagnostic Vulkan policy now decodes the
captured VS output interface and PS input controls, links their semantics, and
uses that emitter when the PS inputs resolve entirely to `DEFAULT_VAL`. A
controlled GX2 HLE fixture issues a draw with a guest pixel shader that reads
`DEFAULT_VAL=1` (`0,0,0,1`); Vulkan framebuffer readback verifies the resulting
black triangle, and guest color-buffer/scanout checks still pass. Cemu's current
`LatteShader_CreatePSInputTable()` reads this field and asserts values are at
most 1; the other two encodings remain based on related-family documentation
and are not exercised end to end. This constant-folding path still rejects
matched varyings; a separate opt-in integration translates one perspective
varying and its guest MOV group into the fragment stage. The bounded default-
input MOV shader is evaluated on the host with constant inputs, and its color
is emitted as a SPIR-V constant; this does not mean the
Latte MOV executes in a Vulkan fragment shader. This verifies a controlled
guest command and shader path under diagnostic policy; ordinary game draw
routing, general PS ALU, Latte hardware behavior, visible output of this
pixel-default path, and WWHD rendering remain unverified. The current full
Debug and Release CTest suites pass 35/35, including `spirv-val` and the varying
readback integration.

## Visible diagnostic triangle retest (2026-11-25)

The Release `--gx2-guest-triangle-display-test` opened the Vulkan output window
on an NVIDIA GeForce RTX 5070 Laptop GPU. A screen capture showed the guest
buffer geometry as red triangles and rectangles against the diagnostic blue
background; the test's standard output also confirmed its float/indexed,
strip, and fan geometry checks completed. The window stayed responsive during
the capture and accepted a close request. This is diagnostic host-shader output
from controlled test geometry, not a game frame or Latte shader rendering. The
visible run did not independently check resizing or Vulkan validation-layer
messages.

## WWHD startup blocker reproduction (2026-11-25)

A headless interpreter run of the local WWHD RPX, limited to 10,000,000
instructions, stopped after 6,081,749 instructions with CIA `0` and
`unsupported PowerPC instruction 0x46524D48`. The final trace loads guest word
`0x101F8C2C` as pointer `0x07E4F660`, then loads the object's `+0x10` field as
zero, loads a zero target into CTR, and executes `bcctr`. Address zero contains
`0x46524D48` (`FRMH`). A post-run guest-memory dump confirms these values. The
HLE trace reaches GX2 sampler initialization and surface-size calculations,
but contains no GX2 draw or scan-buffer swap calls before stopping. This run
therefore does not exercise the experimental Latte vertex-subset frontend
option and provides no evidence of WWHD rendering. The object's intended owner
and initialization path remain unidentified; no callback or successful
behavior should be fabricated to bypass it.

Repeating the same 10,000,000-instruction diagnostic with the experimental
recompiler produced the same 6,081,749-instruction count, CIA `0`, unsupported
word, recent instruction trace, and final GX2 setup calls. This makes an
interpreter/JIT divergence an unlikely cause of this reproduced stop; it does
not explain the guest helper's null result.

The existing guest-memory write watch on `0x07E4F670` observed two writes,
both storing zero: one at CIA `0x02785D14` from `r30`, and another at
`0x02785EC4` from `r3` after guest helper `0x027538C0` returned. That helper
calls `coreinit::OSLockMutex`, but the current trace does not establish why its
caller stores a null result into the field or what the field represents. This
rules out a never-written field for this run, but does not identify the missing
initialization behavior.

## Frontend Latte vertex-subset diagnostic mode (2026-11-25)

The game-launch UI now offers an explicit experimental mode that routes normal
captured GX2 draw snapshots through the existing bounded Latte vertex-SPIR-V
diagnostic policy. It keeps the existing float2 host-diagnostic mode as a
separate choice and stays disabled by default. This can probe guest draws whose
vertex inputs, fetch state, shader subset, and supported render target match the
current implementation. It still uses a diagnostic fragment shader and CPU
color-buffer rendering; full game compatibility and normal GPU draw execution
are not established. Debug and Release builds passed, and both full CTest
suites passed 27/27. The new option has not yet been run against a game title.

## Integer OP2 shader Vulkan readback (2026-11-25)

Added a second controlled vertex-shader fixture that applies OP2 `AND_INT`
with all bits set, `ADD_INT` with zero, `SUB_INT` from integer one to itself,
and `OR_INT` with zero across the four position components. The resulting
position preserves the triangle and passes the Vulkan framebuffer readback
test. This executes these decoded operations in generated SPIR-V with the
diagnostic fragment shader; it is still not normal guest draw execution or
Latte hardware validation. Debug and Release builds passed; both full CTest
suites passed 27/27.

## Integer comparison shader Vulkan readback (2026-11-25)

Added a controlled guest-word shader path that executes OP2 `SETGT_INT` with
inline integer operands `0` and `1`, writes its false result to the position
Z component, and preserves the guest X/Y/W inputs. The Vulkan test verifies the
expected triangle pixels through framebuffer readback, exercising bytecode
decode, SPIR-V generation, and shader execution. It uses the existing
diagnostic fragment shader and controlled harness; it does not demonstrate
ordinary GX2 draw routing, Latte hardware equivalence, or WWHD rendering.
Debug and Release builds passed; both full CTest suites passed 26/26.

## OP2 integer ALU SPIR-V lowering (2026-11-25)

The experimental vertex SPIR-V emitter now lowers OP2 integer AND/OR/XOR/NOT,
wrapping ADD/SUB, signed/unsigned MIN/MAX, and the six equality/greater-than
comparison forms. The implementation uses 32-bit integer instructions,
explicit signed bitcasts for signed ordering, and select operations for min/max
and comparison masks. The combined generated module covering all 16 operations
passes `spirv-val`; per-operation checks verify each emitted opcode. Debug and
Release builds passed, and both full CTest suites passed 25/25. No Vulkan
Vulkan readback now exercises a selected comparison and AND/ADD/SUB/OR case.
Unary NOT uses an invalid source 1 selector in its emitter fixture to verify
that unused operand is not read. The other integer operations still rely on
module validation and per-opcode emission checks; none are Latte hardware
verified.

## OP2 integer comparison SPIR-V lowering (2026-11-25)

The experimental vertex SPIR-V emitter now lowers OP2 `SETE_INT`,
`SETNE_INT`, `SETGT_INT`, `SETGE_INT`, `SETGT_UINT`, and `SETGE_UINT`.
Signed comparisons bitcast the source bit patterns to signed 32-bit values;
unsigned comparisons retain `uint32` values. Results are expanded to the
32-bit all-ones/zero masks produced by the CPU reference. The deterministic
test emits a module containing all six forms, and `spirv-val` accepts it.
Debug and Release builds passed; Debug CTest passed 25/25. This validates the
backend encoding only: Latte hardware behavior and normal guest shader use are
still unverified.

## Mixed OP2/OP3 CPU shader groups (2026-11-25)

The shader decoder and CPU reference now preserve OP2 and OP3 instruction
types together in a shared four-lane X/Y/Z/W group. The executor reads all
lanes from the pre-group register snapshot, commits lane results together, and
forwards PV between groups in the clause. A decoded guest-word fixture mixes
an OP2 MOV lane with three OP3 conditional moves and checks the resulting
register state. The vertex SPIR-V emitter lowers those lanes through shared
per-operation emitters, and `spirv-val` accepts the generated module.
ALU.Trans, literals, incomplete groups, and Latte hardware behavior remain
unsupported or unverified. A controlled guest GX2/Vulkan fixture executes the
mixed bytecode and confirms red pixels in framebuffer readback. Debug and
Release CTest each passed 24/24. The test uses the diagnostic host fragment
shader and does not exercise ordinary guest draw execution or establish WWHD
rendering.

## OP3 integer conditional moves (2026-11-25)

The experimental CPU shader subset and vertex SPIR-V emitter now support OP3
`CMOVE_INT` (`0x1C`), `CMOVGT_INT` (`0x1D`), and `CMOVGE_INT` (`0x1E`).
Equality compares the raw 32-bit condition to zero; greater-than and
greater-or-equal interpret the condition as signed 32-bit. Integer source
negation is rejected. Tests cover zero, negative, positive, and `INT_MIN`
conditions, select emission, SPIR-V validation, and a controlled guest shader
readback. Debug and Release CTest each passed 22/22. The graphics test uses the
existing diagnostic host fragment shader and controlled GX2 path; this is not
normal guest draw integration, Latte hardware verification, visible-window
verification, or WWHD rendering. The opcode semantics follow AMD's R700-family
manual (Chapter 9, integer OP3 entries); their exact applicability to Latte
remains unverified.

## OP3 conditional moves in SPIR-V (2026-11-25)

The experimental vertex SPIR-V emitter now lowers OP3 `CMOVE`, `CMOVGT`, and
`CMOVGE` groups using ordered float comparisons with zero and raw-bit
`OpSelect`. It rejects unsupported OP3 addressing, modifiers, and mixed OP2/
OP3 clauses. Deterministic tests cover the three comparison opcodes, select
emission, relative-source rejection, and `spirv-val` acceptance. A controlled
GX2/Vulkan test encodes `CMOVGE` guest instructions, captures the guest shader,
translates it, and verifies one covered and one uncovered pixel in Vulkan
readback. Debug and Release CTest each passed 20/20. The test uses a diagnostic
host fragment shader; it does not establish guest pixel-shader execution,
visible-window verification, Latte hardware behavior, or WWHD rendering.
Normal guest draws still do not select these translated shaders.

## OP3-only CPU shader clauses (2026-11-25)

The straight-line CPU shader subset now identifies and decodes ALU clauses
containing OP3 instructions, stores their groups in the existing shader IR,
and executes the supported conditional-move family transactionally. The
decoder rejects clauses containing both OP2 and OP3 slots instead of
misclassifying them. Regression coverage builds a guest-word fixture, decodes
the program, executes it, and verifies register/export results plus mixed
clause rejection. Debug and Release CTest each passed 18/18. The SPIR-V emitter
still rejects OP3 groups, and normal GX2 draws do not invoke this CPU shader
subset. See [`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Bounded Latte OP3 conditional moves (2026-11-25)

Added CPU-reference evaluation for floating-point OP3 `CMOVE` (`0x18`),
`CMOVGT` (`0x19`), and `CMOVGE` (`0x1A`), plus four-lane X/Y/Z/W groups.
The condition uses ordered binary32 comparisons against zero; the chosen
source is returned with its raw bits preserved. Source negate is applied as a
sign-bit toggle. Relative addressing, predicate selection, bank swizzle,
clamping, and other OP3 opcodes are rejected. Group sources are read before
writeback, and failures leave guest registers unchanged. Tests cover signed
zero, NaN, source negation, selected-bit preservation, and unsupported state.
Debug and Release builds and CTest each passed 18/18. This CPU interpreter
subset is not yet integrated into the shader program IR, SPIR-V backend, or
normal guest rendering path; Latte hardware output has not been captured. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Structural OP3 group segmentation (2026-11-25)

Added a bounded OP3-only group parser that segments decoded slots by `LAST`,
preserves slot offsets and raw instruction fields, and rejects literal source
selectors or an unterminated final group. Tests cover the four destination
lanes, group-end state, literal rejection, and missing `LAST`. The parser does
not process mixed OP2/OP3 groups, literal payload slots, ALU.Trans, instruction
semantics, or shader IR/renderer integration. Debug and Release CTest passed
18/18 in both configurations.

## Structural Latte OP3 instruction decoding (2026-11-25)

Added `LatteAluOp3Instruction` and a separate raw-field decoder for OP3
instructions. It captures all three selectors/channels and relative/negate
flags, the raw opcode, index mode, bank swizzle, predicate selection,
destination fields, clamp, and group-end marker. The decoder rejects OP2
encodings; the existing OP2 path continues to reject OP3. Tests construct a
field-rich OP3 word pair and check every decoded field. This is structural
decoding only: no OP3 grouping, CPU interpretation, or SPIR-V lowering is
implemented. The R700 manual describes OP3 `MULADD_IEEE` opcode `0x14`, but
Latte-specific multiply-add rounding behavior remains unknown. Cemu's Latte
parser was consulted as a source-layout cross-check; no implementation code
was copied. Debug and Release builds succeeded, and CTest passed 18/18 in both.
See [`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Bounded Latte DX10 min/max SPIR-V lowering (2026-11-25)

The experimental vertex emitter now lowers `MAX_DX10` and `MIN_DX10` using
ordered SPIR-V floating-point comparisons followed by `OpSelect`. This mirrors
the existing CPU subset's comparison and source-selection rules, including
source 1 selection for unordered comparisons; guest operands remain binary32
and unsupported source modes/modifiers still fail. AMD's R700-family manual
documents the OP2 semantics, while Latte-specific behavior remains
unconfirmed. Deterministic tests check the comparison/select instruction
sequence and `spirv-val` accepts both modules. Debug and Release builds passed;
CTest passed 13/13 in both. Separate Vulkan smoke modes now submit
`MAX_DX10(x, x)` and `MIN_DX10(x, x)` guest shaders and verify guest target and
scanout readback. These identity comparisons keep the test geometry fixed;
additional clamping modes render `MAX_DX10(x, 0)` and `MIN_DX10(x, 0)` through
the generated vertex module and assert red/background pixels at positions that
distinguish the transformed output from a pass-through. Debug and Release
CTest pass 15/15 with those checks. This validates SPIR-V and the controlled
Vulkan path only; Latte hardware behavior and arbitrary guest shader programs
remain unverified. The guest color-buffer diagnostic readback still uses
CPU-decoded geometry and the diagnostic host pipeline; the shader-result
assertion uses the separate Vulkan swapchain draw. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Non-identity Latte arithmetic output checks (2026-11-25)

The controlled Vulkan guest-shader harness now checks non-identity arithmetic
results in swapchain readback. `ADD(x, 1.0)` moves the fixture triangle and is
checked at a pixel outside the unchanged triangle; `MUL_IEEE(x, 0.0)` collapses
its position and is checked at a pixel that the original triangle covers.
These checks exercise generated guest vertex modules through Vulkan rather
than inferring arithmetic from the diagnostic color-buffer path. The generated
geometry still comes from the controlled fixture, and the pixel shader remains
diagnostic, so this does not establish general GX2 draw handling or arbitrary
Latte shader support. Debug and Release CTest each passed 17/17, including the
new ADD and MUL readback cases. Latte-specific instruction behavior remains
unverified on hardware; see [`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

The fixture also emits the documented source-1 negate modifier for its X-lane
`ADD` and verifies the translated triangle at a pixel only the negative offset
covers. This exercises the captured OP2 modifier through decode, SPIR-V
generation, Vulkan execution, and readback. The AMD R700 ISA defines input
negation as an FP source modifier; this remains family-level reference evidence
and has not been confirmed on Latte silicon. Debug and Release CTest each
passed 18/18, including this additional Vulkan readback test.

## Controlled Vulkan run for Latte MUL_IEEE (2026-11-25)

The guest-triangle integration harness now has a separate SPIR-V mode that
captures and translates guest `MUL_IEEE` lanes with a `1.0` second operand,
then submits that generated vertex module through Vulkan and verifies guest
color-buffer and scanout readback. The identity operand keeps the fixture
geometry unchanged; this verifies module routing and execution in the
controlled pipeline, not a non-identity numerical result or Latte hardware
behavior. Debug and Release builds passed; CTest passed 9/9 in both, including
SPIR-V validation and both Vulkan readback modes. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Bounded Latte IEEE multiply SPIR-V lowering (2026-11-25)

The same bounded vertex emitter now lowers OP2 `MUL_IEEE` (`0x002`) lanes to
SPIR-V `OpFMul`, using the previously added binary32 source conversion and
absolute/negate handling. Unsupported modifiers and relative operands still
fail explicitly. The opcode and IEEE zero-times-infinity distinction are
documented in AMD's R700-family ISA manual, section 9.2; Latte-specific
confirmation is not available. The deterministic module test checks `OpFMul`
emission and `spirv-val` accepts the generated module. Debug and Release builds
passed; CTest passed 8/8 in both, including the GPU integration test. The GPU
test at this checkpoint executes the ADD-with-zero fixture, not MUL. A
follow-up controlled MUL run is documented above. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Bounded Latte binary32 ADD SPIR-V lowering (2026-11-25)

The experimental vertex SPIR-V emitter now lowers no-literal OP2 ADD
(`0x000`) lanes in addition to MOV lanes. It preserves bit-level source
absolute/negate modifiers, converts binary32 source bits to SPIR-V floats,
emits `OpFAdd`, and bitcasts the result back for PV/GPR propagation. Relative
sources and other unsupported modifiers remain rejected. The encoding and
`src0 + src1` operation are based on AMD's R700-family ISA manual, section 9.2;
Latte hardware behavior has not been independently verified. CPU-only tests
check emitted operations and modifier handling, and `spirv-val` validates the
generated module. Debug and Release builds passed; CTest passed 7/7 in both,
including the diagnostic Vulkan readback test. Its guest vertex shader now
uses ADD with a zero source, preserving the known geometry while routing the
generated ADD module through Vulkan and checking the resulting frame readback.
This does not yet verify a nonzero arithmetic result against a pixel location,
ordinary guest draw rendering, guest pixel shaders, or WWHD. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Guest draw-to-scanout ordering audit (2026-11-25)

Source inspection found why a Vulkan target alone cannot complete guest draw
rendering. `GX2CopyColorBufferToScanBuffer` reads the source from guest memory
and writes scanout bytes inside its HLE callback. The frontend worker runs up
to one million guest instructions per chunk and captures scanout before
publishing that chunk's GX2 command vector; the UI thread consumes draw
commands afterward. Normal guest execution remains capture-only, while the
diagnostic draw route targets the host swapchain. Therefore a draw and copy in
one chunk leave guest scanout stale even if a later UI submission renders the
draw. No code behavior changed in this audit. Cemu's Latte command processor
and render-target update tracking, and WUT's `libwhb/gfx.c` clear/draw/flush/
copy flow, corroborate the need for ordered command execution and explicit
guest-memory visibility. The next milestone is an ordered worker-to-renderer
handoff with completion/writeback before guest HLE copy reads the target. See
[`research/11_graphics.md`](research/11_graphics.md#guest-draw-guest-memory-and-scanout-ordering).

## Opt-in GX2 draw execution checkpoint (2026-11-25)

`EspressoCore::run`, `Recompiler::run`, and `Emulator::run` can now be asked
to return immediately after a CPU step appends a GX2 draw command. The result
uses `StopReason::gx2_draw_captured`, reports the completed step count, and
leaves CIA at the HLE return address. The default execution path is unchanged.
For this opt-in stop mode, `Recompiler::run` delegates to the interpreter to
preserve the exact HLE boundary. Deterministic CPU and recompiler-facing tests
pass. Debug CTest passed 6/6, including the opt-in Vulkan swapchain readback
test. This does not submit a guest draw automatically, wait for Vulkan
completion, update guest color-buffer memory, or fix scanout ordering. The
frontend worker/UI acknowledgement and guest-memory writeback remain the next
required stages; see
[`research/11_graphics.md`](research/11_graphics.md#guest-draw-guest-memory-and-scanout-ordering).

The explicitly selected diagnostic-draw mode now pauses the worker at each
captured draw until the UI consumes the batch and returns from its Vulkan frame
call. The wait is stop-token aware, and launch shutdown requests stop and wakes
the worker. This behavior is build-checked but has not been visually exercised
with a game. It only orders host diagnostic submission; it does not guarantee
GPU completion, handle unsupported game vertex layouts, update the emulated
color buffer, or fix scanout copies.

## Diagnostic Vulkan pixel readback for the Latte MOV vertex subset (2026-11-25)

The explicit guest-triangle diagnostic can request swapchain transfer-source
usage and copy the rendered image into host-visible memory. The
`--gx2-guest-triangle-spirv-smoke-test` enables this test-only path, waits for
its submitted frame, normalizes the supported BGRA/RGBA byte order, and checks
an interior pixel for the fixed diagnostic fragment color. Debug and Release
builds and CTest passed 6/6 in both configurations, including the opt-in GPU
integration test. The test passed on an NVIDIA GeForce RTX 5070 Laptop GPU.
The Debug renderer enables `VK_LAYER_KHRONOS_validation` when installed and
emitted no validation messages. This verifies that captured guest vertex input
and the bounded MOV vertex module rasterize into the swapchain image. It does
not verify visible-window output, guest color-buffer writes, guest pixel
shaders, or WWHD. Normal rendering does not allocate readback buffers or wait
synchronously. See [`research/11_graphics.md`](research/11_graphics.md).
Enable its CTest registration with
`-DAFFOGATO_ENABLE_VULKAN_INTEGRATION_TESTS=ON`; it is off by default so the
standard suite remains usable without a Vulkan-capable display.

## Diagnostic Vulkan draw using the Latte MOV vertex subset (2026-11-25)

An opt-in `--gx2-guest-triangle-spirv-smoke-test` now takes captured guest
vertex-fetch results, lowers the draw-time vertex shader IR to the bounded
SPIR-V subset, builds a Vulkan graphics pipeline from that module, and submits
the captured triangle-list, strip, and fan draw ranges. The fixed diagnostic
fragment shader is still used. Debug and Release builds passed, CTest passed
5/5 in both configurations, and the hidden smoke passed in both; the Debug
renderer enables `VK_LAYER_KHRONOS_validation` when installed, and no
validation messages were emitted. That stage verified Vulkan pipeline
creation and draw submission. Pixel readback evidence is recorded above;
arbitrary shaders, guest render-target writes, and WWHD remain unsupported.
See [`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Validated SPIR-V vertex MOV subset (2026-11-25)

`emit_latte_spirv_vertex_position_mov_subset()` lowers the shared typed IR's
no-literal MOV groups for one mapped `float32x4` vertex input into a
Vulkan 1.0 vertex module that writes one identity `POSITION` export. It
preserves MOV's raw binary32 values and supports its source absolute/negate
sign-bit modifiers. Other OP2 operations, input interfaces, and export forms
fail explicitly. A CPU-only CTest checks generation and unsupported-op
rejection; when `spirv-val` is available, CTest validates both the plain and
modifier-bearing modules. Debug and Release builds and CTest passed 5/5 in
each configuration. The explicit diagnostic route now consumes this module
for Vulkan vertex processing, while normal guest draws remain capture-only.
No pixel readback or guest render-target output has been verified. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Shared typed IR for the bounded Latte CPU shader subset (2026-11-25)

The straight-line shader subset now decodes supported OP2 clauses and normal
exports into a typed `LatteShaderIrProgram`. The existing CPU reference
executor runs that IR, and ALU execution accepts already-decoded instruction
groups while retaining a raw-slot overload for existing callers. Tests check
the IR's clause groups, export metadata, and the separately handled leading
`CALL_FS` case. Debug and Release builds passed, CTest passed 2/2 in both
configurations, and the hidden diagnostic guest-triangle smoke passed in both.
This is a small CPU/SPIR-V-subset IR foundation only: it does not decode new
shader operations, generate SPIR-V, execute guest shaders in normal draws, or
write guest render targets. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Controlled guest vertex-shader subset in Vulkan diagnostic (2026-11-25)

The controlled GX2 triangle harness now installs a guest vertex shader through
`GX2SetVertexShader` and captures it with the draw's fetch state and guest
vertex bytes. An explicit diagnostic-only policy executes Affogato's existing
CPU fetch and straight-line shader subsets per vertex, consumes the exported
position XY, and submits the resulting geometry through the fixed Vulkan
diagnostic shaders. The test uses GPR base zero, a related-generation AMD
reference default that is not confirmed for arbitrary Latte workloads. The
normal emulator draw path remains capture-only unless diagnostic drawing is
explicitly enabled. The hidden triangle smoke passed in Debug and Release;
Debug and Release builds succeeded and CTest passed 2/2 in both. It confirms
the guest HLE-to-CPU-subset-to-Vulkan submission path, but does not verify
visible window output, Latte shader translation, guest render-target writes,
or WWHD rendering. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Latte absolute constant-file reads in CPU shader subset (2026-11-25)

The bounded Latte OP2 CPU executor now accepts an explicit span of up to 256
constant vectors and resolves R700-family absolute source selectors 256 through 511
to those vectors. Constant values flow through MOV, supported floating/integer
operations, vector groups, clauses, and the straight-line shader subset.
Missing constant vectors fail explicitly. Tests exercise selectors for the
first and last vector, shader output propagation, and missing-file rejection.
The draw-snapshot vertex CPU subset now maps known vertex ALU-constant dwords
into that file, preserving unwritten components as unknown; a regression
executes a constant-fed MOV and checks the exported position. This remains a
CPU test subset, not normal guest draw dispatch or Vulkan rendering. Kcache,
relative addressing, literals, loop constants, and Latte-hardware confirmation
remain outstanding. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## GX2 uniform-register upload capture (2026-11-25)

`GX2SetVertexUniformReg` and `GX2SetPixelUniformReg` validate the selected
256-vector constant bank and guest source range, then snapshot big-endian
dwords as ordered Latte ALU-constant register writes. Draw register snapshots
retain the captured values even if the source guest memory is later changed.
Invalid ranges and loop-constant selectors fail explicitly without appending a
partial command batch. Tests cover stage-bank separation, word order, draw-time
capture, bounds, and unmapped pointers. This is CPU-side state capture only:
automatic shader-subset input mapping, uniform blocks, loop constants, and
Vulkan shader binding remain unsupported. Debug build and CTest passed 2/2; Release
build and CTest also passed 2/2. See
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## GX2 shader-mode state capture (2026-11-25)

`GX2SetShaderModeEx` now accepts uniform-register and uniform-block modes and
queues a typed operation in the normal GX2 command stream. `Gx2RegisterState`
applies the shader-mode GPR, stack, thread, `SQ_CONFIG`, and VGT-mode fields in
order. Register snapshots retain a known-bit mask, so masked updates preserve
previously observed values while leaving unobserved neighboring bits unknown.
Geometry and compute modes fail explicitly because their ring resources and
scheduling are not modeled. Tests cover HLE argument capture, allocation
limits, both uniform modes, field preservation, unknown-bit tracking, and
draw-time ordering. Debug build and CTest pass 2/2. This is CPU-side state
capture only; it does not program a GPU, execute shaders, or prove Latte
hardware semantics. In particular, uniform-block mode clearing `DX9_CONSTS`
differs from Decaf's read/modify/write behavior and remains unverified against
Wii U hardware. See [`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md).

## Current implementation checkpoint (2026-11-25)

Opt-in draw capture snapshots the supported RGBA8 contents of bound GX2 color
target 0 at the HLE draw boundary. Direct GX2 calls and decoded PM4 display-list
draws carry immutable target bytes with each command; capture is disabled by
default and preserved across `GX2Init`. Unsupported descriptors and inaccessible
guest image ranges receive explicit failure statuses.

The explicit diagnostic draw mode now renders captured vertices into an
offscreen `VK_FORMAT_R8G8B8A8_UNORM` target initialized from those bytes. It
reads the image back after a fence, converts through the shared RGBA8 surface
helper, and sends the result to the paused worker. The worker writes guest
memory before resuming guest execution, so a subsequent GX2 color-buffer copy
sees those pixels. A controlled 1280x720 guest HLE workload verifies that
untouched blue pixels survive, diagnostic host-shader pixels appear in the
color buffer, GX2 copies it to TV scanout, and swapchain readback contains the
draw. Debug and Release builds and CTest passed 6/6 in both configurations.
This verifies a controlled diagnostic draw path only; general Latte shader
execution and WWHD rendering remain unverified.

`GX2SetContextState` and `GX2DirectCallDisplayList` share a bounded PM4
decoder. It accepts Type-0 writes in supported config/context and
control-constant windows, Type-2 filler, Type-3 NOP, register-set packets,
`INDEX_TYPE`, `NUM_INSTANCES`, unpredicated `DRAW_INDEX_AUTO`, and a bounded
`DRAW_INDEX_2` subset plus U16/U32 `DRAW_INDEX_IMMD`. The supported draw-register
state (primitive mode, instance count, index type, base vertex, and base
instance) persists across successful direct-call lists, matching the
command-processor register-shadow model. Each list is decoded using a
temporary state copy and commits register changes only after packet and draw
memory validation succeeds. Direct-call input is validated and copied from
guest memory before decoding; malformed or unsupported streams do not append partial
command batches. Indexed draws currently accept only zero-high-byte addresses
that resolve to a bounded range in flat guest physical backing memory, and
reject counts greater than `MAX_SIZE`. Indexed draw records carry base vertex
independently from index-buffer byte offset, matching the
register setup used by Cemu. When draw-byte capture is enabled, the HLE
snapshots index bytes and referenced vertex data at call time. This queues
draw records; it does not establish Latte shader or guest render-target
execution. Predicated draws, GPU address translation, other state operations,
indirect-buffer recursion, list recording/copy APIs, raw ring processing, and
GPU retirement remain unsupported. All supported PM4 draw forms require a
zero draw initiator; other values fail explicitly. Debug and Release builds
passed, with CTest 2/2 in both configurations. The hidden guest-triangle smoke
test passed in Debug and Release. It includes unpredicated auto-index,
memory-indexed, and immediate-indexed display-list draws, validating their
command snapshots and diagnostic Vulkan submission on an NVIDIA GeForce RTX
5070 Laptop GPU. Its output uses
host shaders and does not verify Latte shader execution.

## PM4 draw-state persistence (2026-11-25)

The supported draw-register subset is now retained between direct-call display
lists, so a draw packet can use state written by an earlier list. Invalid
packets and failed draw-memory validation leave both pending commands and the
retained subset unchanged. Tests cover cross-list state reuse and failed-list
rollback. Debug and Release builds succeeded, and CTest passed 2/2 in each
configuration. This improves packet-state continuity; it does not add indirect
buffers, raw ring processing, guest shader execution, or guest render-target
writes.
Zero-length `GX2SetContextState` shadow restores now update the same supported
draw-state subset when they emit one of its register addresses. A regression
test restores primitive mode, then submits a draw-only direct-call list using
that value. The saved-context ranges do not contain every draw-state register;
values absent from the restore stream remain unchanged. Debug and Release
builds and CTest passed 2/2 in both configurations.
The packet research is cross-checked against AMD's R6xx/R7xx family guides,
Cemu, and Decaf; there is still no Latte-produced fixture. References disagree
on whether `INDEX_TYPE` updates one or two shadow slots and whether a zero
`NUM_INSTANCES` becomes one, so Affogato preserves raw packet state pending
Latte-specific evidence. Details are in
[`research/15_latte_command_processor.md`](research/15_latte_command_processor.md).
The PM4 decoder also accepts unpredicated `DRAW_INDEX_IMMD` packets for the
32-bit index forms represented by GX2 index types 1 and 9. Packet structure
and index counts are validated in all modes, while embedded index bytes and
referenced vertex bytes are copied only when diagnostic draw capture is
enabled. The implementation rejects 16-bit immediate indices and
payload/count mismatches. A focused HLE regression test verifies captured
bytes, base-vertex range, capture-disabled behavior, and rejection behavior. This
remains a bounded command-capture feature: it does not execute guest shaders
or render into a guest color buffer.
The current PM4 packet implementation and research details are in
[`research/15_latte_command_processor.md`](research/15_latte_command_processor.md).
Reference research confirms that normal Cemu `GX2DrawEx` uses a predicate-bit
`DRAW_INDEX_AUTO`; predicate evaluation and its reset state remain unsupported,
so that common packet form remains rejected pending verified semantics.

At `be6b69c`, `build/debug/AffogatoFrontend.exe --gx2-guest-triangle-smoke-test`
completed successfully on an NVIDIA GeForce
RTX 5070 Laptop GPU. It submits captured guest-memory geometry through the
diagnostic host shader; this is not guest Latte shader execution. The visible
display test was not verified because the Windows computer-use runtime failed
to initialize, so no claim is made about the triangle's visible output.

## Documentation-driven subsystem audit (2026-11-24)

The first coverage pass is recorded in [`research/14_subsystem_coverage.md`](research/14_subsystem_coverage.md), with the detailed graphics source audit in [`research/11_graphics.md`](research/11_graphics.md). The audit distinguishes verified behavior from partial capture and missing execution. In particular, GX2 draw state and input bytes are captured for the opt-in diagnostic route; a narrow non-executing Latte control-flow record decoder exists, but there is no raw Latte command processor, shader translator, guest draw render-target write, general tiled surface addressing, or GPU/guest-memory coherence model.

`Gx2RegisterState::draw_snapshot()` now gives a draw index a typed, non-owning view over its captured command, vertex bindings, shader bindings/resources, render targets, and raw register words. The diagnostic renderer uses this accessor instead of indexing the fetch-shader snapshot independently. A later `apply()` may invalidate the borrowed view; it is intended for bounded consumption after a command batch is applied, not as cross-thread ownership. Debug and Release builds and CTest passed 2/2 in both configurations; the hidden guest-triangle Vulkan smoke test passed with diagnostic host shaders.

External references consulted for this pass include WiiUBrew's GX2/Latte pages, WUT GX2 declarations/helpers, Cemu's `LatteCommandProcessor.cpp`, `LatteShader.cpp`, `LatteAsyncCommands.cpp` and Vulkan backend, and Decaf's `src/libgpu/` plus its Wii U test repository. No external implementation code was copied. The next architectural dependency is an explicit render/guest-memory synchronization contract; the frontend currently runs guest execution on a worker and Vulkan consumption on the UI thread, so a renderer-thread guest-memory write would race guest execution.

The RGBA8 CPU surface path now also supports 1D tiled thin1 surfaces. Clear and
TV scanout copy share an address calculation for 8x8 displayable 32-bit
microtiles, with padded pitch and height required to be multiples of eight.
Debug and Release builds and CTest passed 2/2 in both configurations. A CPU
HLE integration test clears a tiled guest surface, checks a pixel in the final
microtile, copies it through GX2 scanout HLE, and checks first/last scanout
pixels. This does not add macro-tiling, other formats, Vulkan render targets,
or guest draw writes.

The texture staging helper also uses the researched 1D and zero-swizzle 2D
thin1 address mappings for its existing 8-, 16-, and 32-bit UNORM format
decoders (R8, RG8, packed 16-bit color formats, and RGBA8). It applies the
element-size pitch alignment and rejects depth tiling and unsupported layouts.
Debug and Release builds and CTest passed 2/2 in both configurations. This
produces CPU staging pixels only; the path is not yet connected to translated
guest shaders or normal Vulkan draw submission.

The bounded RGBA8 clear/copy HLE path now also accepts zero-swizzle 2D thin1
surfaces with pitch padded to 32 pixels and height padded to 16 rows. The macro-tile address path
was cross-checked against the R600 Addrlib mapping used by Decaf and
WiiU-Swizzle; tests verify known interleave offsets, full macro-tile address
uniqueness, and guest HLE clear-to-scanout-copy behavior. Debug and Release
builds and CTest passed 2/2 in both configurations. Tiled Vulkan presentation
is not visually verified; bank-swizzled, depth, multisample, non-RGBA8, and
guest draw render-target writes remain unsupported.

This snapshot is based on the source and repository notes at `f0fbfef` (`main`, 2026-10-25), with verified updates below through 2026-11-22. It distinguishes implemented paths from complete console behavior.

## Current source update (2026-11-03)

The frontend now snapshots a configured TV scan buffer at `GX2SwapScanBuffers` and can present its CPU-visible RGBA8 pixels through the Vulkan swapchain. This requires the guest to have written pixels into the registered memory; Affogato still does not execute Latte draw commands or translate guest shaders, so games whose frames are produced by GX2 draws will not render correctly through this path alone. A failed/unmapped scan-buffer read leaves the previous Vulkan output in place. Debug and Release builds succeeded; Debug CTest passed 2/2, including padded scanout-row staging coverage. The Vulkan RGBA8 upload and scanout presentation smoke test passed on an NVIDIA GeForce RTX 5070 Laptop GPU. No game launch was verified in this update.

## Verified rendering update (2026-11-22)

The controlled diagnostic triangle is now visually verified in the Vulkan window on an NVIDIA GeForce RTX 5070 Laptop GPU. Its guest float2 vertices were displayed with the fixed diagnostic shader; the window remained responsive through maximize/restore and exited cleanly. The Khronos validation layer was installed and enabled by the renderer; no validation warning/error was emitted during the display run. This verifies milestone 1 only, not translated guest shaders or WWHD graphics.

With the explicit `capture_draw_vertex_data` runtime option enabled, non-indexed GX2 draw HLE records carry immutable snapshots of each bound vertex buffer's requested range. Indexed draw HLE also snapshots index bytes for the four WUT `GX2IndexType` values and the contiguous vertex-buffer ranges addressed by those indices. Both paths validate guest-memory ranges at draw time, and `GX2Init` preserves the launch-configured capture policy. Capture is disabled by default to avoid draw-path allocation overhead. The frontend offers an opt-in `Diagnostic GX2 draws (host shader, host window)` mode; it routes ordered draw batches through the renderer's diagnostic policy. The diagnostic renderer supports single-instance, base-instance-zero triangle lists, strips, and fans. It expands captured U16/U32 little- or big-endian indices into sequential diagnostic vertices and expands strips/fans into triangle lists before Vulkan submission, preserving alternating strip winding. Primitive restart is rejected because its state is not captured. The visible controlled test submitted Float32 and Float16x2 triangle lists, U16 little-endian and U32 big-endian indexed lists, strips, and fans through guest GX2 HLE; index streams used byte offsets 2 and 4. Distinct triangles and rectangles appeared in the Vulkan window using the diagnostic host shader. This still does not execute Latte shaders or write guest draw results to guest color-buffer memory. Accepted draws use one Vulkan draw range per command, and diagnostic presentation takes precedence over guest scanout while active. Unsupported draws report a graphics diagnostic and are not treated as successfully rendered. The mode still uses fixed host shaders and a host swapchain target; general vertex formats, guest render-target writes, and Latte shaders remain unimplemented. Debug and Release builds, CTest (2/2 in each configuration), and hidden plus visible triangle checks pass. Debug/Release CMake builds emit existing `-Wpedantic` warnings from vendored AsmJit `__int128` and anonymous structs.

The diagnostic consumer now also requires the captured `GX2SetFetchShader` state, decodes the HLE-generated VTX program, selects attribute location 0 and its specified vertex-buffer slot/byte offset, and reads the position from the immutable guest snapshot. Float32x2/x3/x4 and Float16x2 layouts can provide the first two position components. The half-float path converts IEEE 754 binary16 values, including subnormals, to Float32; other formats and fetch programs are rejected. The controlled guest test constructs fetch shaders through `GX2InitFetchShaderEx`/`GX2SetFetchShader` and uses vertex-buffer slot 2, a 16-byte stride, a nonzero 4-byte position offset, and a separately submitted Float16x2 triangle. This specific update passed hidden smoke checks; the visible-window rerun after extracting the conversion helper was interrupted and is not claimed as reverified. Output remains diagnostic-shader output to the host swapchain, not execution of the guest's vertex or pixel shader.

## Current implementation

- A single-core Espresso interpreter with a broad but incomplete PowerPC subset, explicit CR/XER/LR/CTR/FPSCR and paired-single state, checked guest memory, bounded instruction history, and stop diagnostics.
- An experimental AsmJit x86-64 recompiler with native lowering for a growing subset, interpreter/HLE fallbacks, code-page write-generation checks, a direct-mapped block lookup, and statistics.
- Big-endian ELF and RPX loading with compressed sections, implemented relocations/trampolines, imported HLE function/data binding, and TLS relocation support.
- A small per-session Cafe OS model and selected coreinit, GX2, nn_act, nn_olv, and nn_save handlers. Unsupported imports remain explicit stops/errors.
- GX2/Latte state and command capture, typed texture/sampler descriptor decoding, CPU staging for selected linear-aligned, 1D thin1, and zero-swizzle 2D thin1 UNORM textures and configured TV scan buffers, and a Vulkan swapchain with guest scan-buffer presentation plus an opt-in RGBA8 upload primitive. Texture staging is not yet wired to guest shader sampling/draw execution.
- An SDL3/ImGui RPX library frontend and command-line diagnostic mode.

## Verified boundaries and limitations

- The frontend can load and execute RPX code to a bounded stop. That is not a game-compatibility claim.
- GX2 HLE supports `GX2ClearColor` and `GX2CopyColorBufferToScanBuffer` for bounded linear-aligned, 1D thin1, or zero-swizzle 2D thin1 RGBA8 TV0 surfaces, with guest-memory writes and ordered command capture. Other macro-tiled/general surface handling, guest draw execution, shader translation, and full game frame production remain unimplemented.
- CPU, loader, HLE, and graphics support is incremental. Full IOSU, multicore scheduling, complete CPU/FPU behavior, audio, complete input, encrypted-title loading, and broad game compatibility remain out of scope or unimplemented.
- `docs/development-progress.md` records diagnostic runs and measured observations. Its latest notes report WWHD stopping at an unsupported null-target fetch (`0x46524D48`) after 6,081,749 instructions, and BOTW reaching a 50-million instruction cap at CIA `0x2F18DDC`. Neither run reaches a guest draw submission. These figures describe those recorded runs, not current environmental verification or compatibility.
- The repository provides CPU and recompiler test executables plus a benchmark executable. In the current environment (GCC 16.2.0, CMake 4.4.4, Ninja 1.13.2), the Debug and Release presets build successfully and Debug CTest passes 2/2. The Vulkan path was not exercised in a live title during this update.

## Tests and measurements

- `affogato_cpu_tests`: CPU state, decoder, guest memory, loaders, and HLE coverage.
- `affogato_recompiler_tests`: recompiler execution, fallback, invalidation, and selected interpreter/JIT state/memory comparisons.
- `affogato_recompiler_benchmark`: standalone benchmarking; no current-session benchmark was run.
- `AffogatoFrontend.exe --gx2-guest-scanout-smoke-test`: a controlled PPC `bctrl` → GX2 HLE → guest color-buffer clear → TV scanout copy/swap → Vulkan presentation diagnostic. It checks pixel bytes for two distinct guest clear colors and submits both scanouts, but it does not load a WUT RPX or execute guest draw commands.
- `AffogatoFrontend.exe --gx2-guest-scanout-display-test`: the same controlled operations in a visible window, held open until closed for visual inspection.
- `AffogatoFrontend.exe --gx2-guest-triangle-smoke-test`: captures a guest `GX2DrawEx`, snapshots its non-indexed guest vertex-buffer bytes during HLE execution, and submits the controlled float2 vertices with a diagnostic Vulkan shader. It does not translate Latte shaders or support general guest vertex layouts.
- `AffogatoFrontend.exe --gx2-guest-triangle-display-test`: the same diagnostic path in a visible window; visually verified on 2026-11-22, including maximize/restore and clean close.
- No current-session performance comparison was made. Existing timing observations and discarded optimization experiments remain in `development-progress.md`; they should not be treated as controlled benchmark claims.

## Development priorities

The detailed queue in `development-progress.md` is the project-specific source for next implementation priorities. The current high-level priorities are:

1. Continue investigating the recorded WWHD/BOTW startup stops through guest initialization and existing HLE behavior; do not fabricate successful callbacks or results.
2. Extend the verified surface-address layer beyond zero-swizzle 2D thin1 RGBA8 using title captures and deterministic address vectors; keep bank-swizzled, depth, multisample, and other macro-tiled cases unsupported until their inputs are validated.
3. Keep interpreter/recompiler semantics aligned and add native lowerings only with differential tests and repeatable Release measurements where performance is the objective.
4. Extend loaders and HLE only for observed guest requirements, preserving transactional guest-memory effects and explicit unsupported behavior.
5. Keep validating changes with both configured builds and the registered CTest suite; add Vulkan presentation checks that remain usable without a display where practical.

## Next implementation area

The implemented color-buffer clear and copy path supports single-sample 2D UNORM RGBA8 surfaces in linear-aligned, 1D thin1, or zero-swizzle 2D thin1 layout and matching TV0 scanout. Full game rendering additionally requires broader surface support, Latte command execution, guest vertex/index processing, guest draw render-target writes, and guest shader translation.

## Verified update (2026-11-22)

The CPU privilege foundation is now implemented in focused changes: `CpuState`
contains MSR/SRR0/SRR1; supported program-exception entry handles privileged
instruction violations; `mfmsr`, bounded `mtmsr`, SRR0/SRR1 `mfspr`/`mtspr`, and
bounded `rfi` execute through the interpreter path. Translation-enabled MSR
states stop explicitly because the flat `GuestMemory` model has no MMU. Cafe OS
thread-context `srr1` restoration remains deferred for the same reason. CPU
and recompiler fallback tests cover reset, privilege violations, supported
state changes, exception saves, and return behavior. Debug and Release builds
and Debug CTest passed 2/2 after the code changes. No live WWHD compatibility
or performance claim was verified.

The initial rendering audit found that the frontend could present
CPU-visible RGBA8 TV scan-buffer memory but the GX2 HLE did not register
`GX2CopyColorBufferToScanBuffer`. Captured draw/clear commands also did not
produce guest framebuffer pixels. The Vulkan window therefore had a
presentation path without an end-to-end guest rendering path. No live title
run was performed during that audit.

## Rendering update (2026-11-22)

Registered `GX2CopyColorBufferToScanBuffer` using the WUT ABI: a color-buffer
descriptor pointer and a scan-target bitmask. The first supported case is TV0,
matching 720p or 1080p RGBA8 dimensions, and a single-sample 2D RGBA8 source in
linear-aligned tile mode with zero swizzle and a base-level single-slice view.
The HLE validates descriptor sizes, pitch, alignment, source/destination guest
ranges, scan-buffer capacity, and the target before changing destination
memory. It strips source row padding and writes the next scan-buffer ring slot;
the following swap request selects that same slot for frontend snapshot and
Vulkan presentation. Copy and swap events remain ordered in GX2 command state.

CPU HLE tests cover padded pitch, destination bytes, ring-slot selection,
unsupported tiled input and DRC targets, undersized output memory, unmapped
source memory, and copy/swap ordering. Debug and Release builds succeeded, and
Debug CTest passed 2/2. `C:\devkitPro` is absent in this environment, so the
WUT RPX fixture could not be compiled and no live guest or Vulkan-window output
was verified.

## Guest color-buffer clear update (2026-11-22)

`GX2ClearColor` now accepts the same bounded linear-aligned, single-sample 2D
RGBA8 base-level/single-slice color-buffer layout. It validates the descriptor
and full guest-memory range, clamps components to UNORM [0, 1], rejects NaN,
quantizes RGBA8, writes active pixels, and preserves row padding. Successful
clears enter the ordered GX2 command stream; unsupported layouts fail before
guest pixels or clear-success state are changed.

CPU HLE tests verify the guest bytes, padded-row preservation, invalid tiled
layout and NaN rejection, and clear → copy → swap order. Debug and Release
builds succeeded, Debug CTest passed 2/2, and the existing Vulkan texture-upload
smoke test passed on an NVIDIA GeForce RTX 5070 Laptop GPU. That smoke test
uses host-provided texture data and does not verify guest-generated output.
The controlled emulator HLE integration diagnostic is recorded below. A WUT
RPX build remains unavailable because `C:\devkitPro` is not installed.

## GX2 scanout integration diagnostic (2026-11-22)

Added `AffogatoFrontend.exe --gx2-guest-scanout-smoke-test`. The diagnostic
executes PPC `bctrl` call stubs through the interpreter into GX2 HLE for
initialization, TV scan-buffer setup, output enable, two differently colored
clears, copies, and swaps. It verifies the first RGBA8 pixel in both captured
scanouts and that the frames differ, then uploads and presents both scanouts
through the Vulkan swapchain. It passed on the NVIDIA GeForce RTX 5070 Laptop
GPU. The visible display variant showed the second guest clear as a solid green
frame in the Vulkan window; that confirms the controlled color-buffer clear,
copy, and scanout route, not game rendering. It is not a WUT-built guest
program, WWHD rendering, or guest draw test.

The triangle display variant has since been visually inspected in the
2026-11-25 retest above. Next, extend the diagnostic path from its known float2
vertex layout to verified guest fetch formats and shader execution. Add a
WUT-built RPX fixture when
devkitPro/WUT is installed. At that checkpoint, clear/copy support still
applied only to the documented linear RGBA8 subset.

## Diagnostic guest triangle (2026-11-22)

Added a controlled PPC `bctrl` test that calls GX2 initialization,
`GX2SetAttribBuffer`, and `GX2DrawEx` through HLE. It verifies the captured
triangle parameters and vertex-buffer state, reads three float2 positions from
guest memory, and submits those positions to a Vulkan triangle-list pipeline
with a fixed diagnostic fragment shader. The hidden-window smoke test passed
on the NVIDIA GeForce RTX 5070 Laptop GPU. This validates guest geometry
submission only; it is not Latte fetch/shader emulation or game draw execution.

## Latte context display-list register side effect (2026-11-25)

`Gx2RegisterState` now applies the `SQ_VTX_SEMANTIC_CLEAR` write mask: each set
bit clears its corresponding vertex semantic shadow entry to `0xFFFFFFFF`,
and the write-only clear register is not exposed as readable shadow state.
Tests verify selective clearing, preservation of unselected entries, and that
later writes in the same ordered command stream take effect after the clear.
An HLE integration test also decodes a big-endian `GX2SetContextState` packet,
routes its clear-register write through the shared register shadow, and checks
the selected semantic entry.
The behavior is based on AMD's R6xx register reference and cross-checked
against Decaf's PM4 register-write path; the R6xx reference is related-family
evidence rather than a Latte-specific specification. Other special register
side effects remain unmodeled.

Debug and Release builds succeeded, and CTest passed 2/2 in both build trees.
This does not add raw command-ring execution or guest draw submission.

## GX2 display-list submission research (2026-11-25)

WUT's [`include/gx2/displaylist.h`](https://github.com/devkitPro/wut/blob/master/include/gx2/displaylist.h) declares the guest ABI for begin/end recording, queued `GX2CallDisplayList`, direct `GX2DirectCallDisplayList`, and `GX2CopyDisplayList`. Cemu's [`GX2_Command.cpp`](https://github.com/cemu-project/Cemu/blob/main/src/Cafe/OS/libs/gx2/GX2_Command.cpp) distinguishes queued indirect-buffer submission from direct TCL submission; Decaf's PM4 processor recursively processes nested indirect buffers with a depth bound. At the time of this initial audit, Affogato registered none of these exports and could not observe raw guest PM4 dwords.

The initial audit identified owned guest dwords, direct/indirect distinction,
range validation, recursion bounds, and command ordering as requirements.
Since then, `GX2DirectCallDisplayList` has been added for a limited supported
state-write subset. It copies bytes at call time, validates mapped guest
memory, rejects unsupported stateful opcodes, and queues decoded register
updates in order. `GX2CallDisplayList` now snapshots and decodes one target
list through the same ordered path. It does not implement nested indirect
buffers, recording APIs, guest shader execution, guest render-target writes,
or GPU completion. The fixtures are synthetic and
cross-checked against related-family packet definitions; no Latte-produced
stream or WWHD use of these APIs has been verified.

## Queued GX2 display-list subset (2026-11-25)

Registered `GX2CallDisplayList` using the WUT pointer/size ABI. It copies the
target guest bytes at the HLE call point, validates and decodes the same bounded
PM4 subset as `GX2DirectCallDisplayList`, and appends supported commands in
guest order. Tests verify the queued API consumes prior supported PM4 draw
state. This is not full Latte indirect-buffer execution: nested IB packets,
GPU address translation, asynchronous GPU visibility, recording/copy APIs,
and direct-call flush/completion semantics remain unsupported. Debug and
Release builds and CTest passed 2/2; the hidden diagnostic triangle smoke test
passed in both builds using host shaders.

## Diagnostic UNORM8 normalized input family (2026-11-25)

The HLE-generated fetch decoder supports unsigned normalized 8-bit x1/x2/x4
formats (VTX data formats `0x00`, `0x04`, and `0x0A`) when
`GX2_ENDIAN_SWAP_NONE` is selected explicitly. The diagnostic Vulkan path uses
the first two x2/x4 components as position and converts each component to
`byte / 255.0F`; tests check all three descriptor widths, known values, and
truncated vertex bytes. The hidden triangle smoke test includes a guest-issued
UNORM8x4 triangle and passes on the NVIDIA GeForce RTX 5070 Laptop GPU in Debug
and Release. This
validates the diagnostic route only: no guest Latte shader or guest render
target is involved, and visible output for this new format has not been
rechecked.

## PM4 immediate U16 index capture (2026-11-25)

`DRAW_INDEX_IMMD` capture now supports WUT index types 0/4 (U16 little/big
endian) alongside U32 types 1/9. U16 packets require one packed DWORD per pair
of indices; capture reverses the documented pair-halfword swap and removes an
odd trailing pad halfword. Deterministic HLE tests verify guest byte order and
the resulting vertex snapshots for both U16 endian modes. Debug and Release
builds and CTest passed 2/2. The hidden triangle smoke test passed in both
builds. It uses diagnostic host shaders and does not establish guest shader
execution or guest render-target writes.

## Latte shader decoder research checkpoint (2026-11-25)

Updated [`docs/research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md)
with structural facts from AMD's R700-family manual and explicit limits on
applying them to Latte. Shader capture stores checked guest words via
`read32_be()` and snapshots them with draw state. The published Decaf assembler
produced the guide's pass-through GSH fixture. Its 24-byte program block
decodes as three `word0`/`word1` pairs when interpreted little-endian, matching
Cemu's legacy parser (`word1[29:23]` opcode, EOP at bit 21). Affogato's
`read32_be()` snapshot values convert to those fixture words with
`std::byteswap` per DWORD. This establishes the capture conversion for the
controlled fixture; no shader execution capability was added.

## Latte control-flow and ALU-clause header subset (2026-11-25)

Added [`src/gpu/latte_shader_decoder.hpp`](../src/gpu/latte_shader_decoder.hpp)
with a structural decoder for fixture-verified `CALL_FS`, `EXPORT`,
`EXPORT_DONE`, and ALU clause headers. It converts captured big-endian CPU
words to Latte DWORDs, records word offsets/raw fields, extracts ALU clause
addresses in eight-byte units and slot counts, checks clause ranges against the
captured program, and stops at EOP. Tests use Decaf-assembler-produced GSH
fixtures and cover malformed pairs, unsupported opcodes, missing EOP, and an
out-of-range clause. Debug and Release builds succeeded; CTest passed 2/2 in
both configurations. `git diff --check` is clean. At that checkpoint, ALU slot
payloads were not extracted or interpreted; shader execution, SPIR-V, and
guest-shader rendering were absent.

## Latte ALU slot extraction (2026-11-25)

Added `decode_latte_alu_clause_slots()` to extract the bounded raw 64-bit slots
referenced by a decoded ALU clause. It preserves the byte-swapped Latte DWORD
pairing and each slot's location in the captured program, rejects non-ALU
records and out-of-range inputs, and intentionally interprets no instruction
fields. The test materializes the assembled fixture's control-flow and slot
words at their captured offsets in a zero-filled program span, then checks the
first and last slot, rejects a non-ALU record, and covers truncated and
out-of-range input. Debug and Release builds succeeded; CTest passed 2/2 in
both configurations. `git diff --check` is clean. This is structural access
only; ALU opcode semantics and shader execution remain unimplemented.

## Latte ALU OP2 field decoding (2026-11-25)

Added `decode_latte_alu_op2_instruction()` for the common OP2 word layout. It
extracts raw opcode, source selectors/channels and modifiers, destination
fields, output modifier, and last-in-group state without assigning opcode
semantics. The assembled MOV fixture checks r1 to r3 component fields and the
group-ending slot; OP3 layouts are rejected by this entry point. This is still
not shader execution. Debug and Release builds succeeded; CTest passed 2/2 in
both configurations. `git diff --check` is clean.

## Latte MOV source-value subset (2026-11-25)

Added `evaluate_latte_alu_move_value()` for the assembled MOV form that reads an
absolute GPR component. It returns the raw 32-bit source unchanged, preserving
NaN payloads, signed zero, and non-floating register bit patterns. The helper
does not write the destination or update previous-vector/previous-scalar state;
it is not an ALU instruction-group executor. Tests use bit-pattern fixtures
including NaN, negative zero, and infinity. Debug and Release builds succeeded;
CTest passed 2/2 in both configurations. `git diff --check` is clean.

## Latte no-literal OP2 group boundaries (2026-11-25)

Added `decode_latte_alu_op2_groups()` to split decoded OP2 slots at the verified
LAST marker. It rejects OP3 layouts, literal source selectors, and a clause
ending with an unterminated group. Tests confirm the assembled MOV fixture is
one four-slot group and cover those unsupported cases. It does not parse
literal payload slots or execute groups. Debug and Release builds succeeded;
CTest passed 2/2 in both configurations. `git diff --check` is clean.

## Latte four-lane MOV group subset (2026-11-25)

Added `execute_latte_alu_op2_move_vector_group()` for exactly four ordered
X/Y/Z/W MOV operations using supported absolute GPR component reads. It
collects source values before writing any destination lane and returns the
four-lane vector result. Tests cover raw bit preservation, same-group
source/destination overlap, and rejection without partial writes. The behavior
is based on R700-family group semantics and the assembled fixture, not direct
Latte hardware validation. The helper does not implement PS, constants, literal
operands, or general shader execution.

## Latte integer bitwise OP2 vector subset (2026-11-25)

Expanded the four-lane OP2 group executor to support the R700-family AND_INT,
OR_INT, and XOR_INT operations (`0x030`–`0x032`) with two absolute GPR
component sources. The execution path reads every lane before writing any
destination and returns the group's vector result. Tests cover the three
bitwise truth tables and guest destination writes. Debug and Release builds and
CTest passed 2/2 in both configurations, and `git diff --check` is clean. This
is family-reference behavior, not direct Latte hardware verification.

## Latte previous-vector forwarding (2026-11-25)

Added selector `254` support for the R700-family PV source when the caller
supplies the preceding supported group result. MOV and integer bitwise
instructions can read PV components. A missing incoming PV value is rejected
before register writeback. Tests verify a first group feeding a second group
and failure without partial writes. Debug and Release builds succeeded; CTest
passed 2/2 in both configurations, and `git diff --check` is clean. This PV
forwarding subset is based on R700-family documentation and has not been
verified against Latte hardware.

## Latte inline constants and NOT_INT (2026-11-25)

Added the R700-family inline source selectors `248`–`252` and `NOT_INT` opcode
`0x033` to the bounded four-lane OP2 path. The decoder records the source
absolute modifiers at word1 bits 0 and 1. At that stage the raw-bit evaluators
rejected them pending floating-point semantics; the following MOV-modifier
checkpoint added the documented source-0 behavior. Tests check exact constant
bit patterns, decode/execute AND, OR, XOR, and NOT groups, and verify that
unsupported integer modifiers fail before writeback. Debug and Release builds
succeeded; CTest passed 2/2 in each, and `git diff --check` is clean. Behavior
is not verified against Latte hardware.

## Latte MOV source modifiers (2026-11-25)

The bounded MOV evaluator now applies its source-0 absolute and negate flags
using binary32 sign-bit operations in the R700-specified order: absolute value,
then negation. The integer logic subset continues to reject these flags. Tests
cover signed zero, NaN payload preservation, infinity, and combined modifiers.
Debug and Release builds and CTest passed 2/2 in both configurations; the
behavior remains unverified on Latte hardware.

## Latte integer ADD/SUB (2026-11-25)

Added R700-family OP2 `ADD_INT` (`0x034`) and `SUB_INT` (`0x035`). They use
unsigned 32-bit arithmetic, preserving the documented wraparound bit behavior
without signed-overflow undefined behavior. Four-lane group tests cover add
carry, subtract borrow, and high-bit values. Debug and Release builds and CTest
passed 2/2 in both configurations; Latte hardware behavior remains unverified.

## Latte integer min/max (2026-11-25)

Added `MAX_INT`, `MIN_INT`, `MAX_UINT`, and `MIN_UINT` OP2 operations
(`0x036`–`0x039`). Signed comparisons use an order-preserving sign-bit bias on
the unsigned operands, avoiding host signed-conversion assumptions. Group
tests exercise values on both sides of the signed boundary and unsigned
high-bit ordering. Debug and Release builds and CTest passed 2/2 in both
configurations; Latte hardware behavior remains unverified.

## Latte OP2 control-field validation (2026-11-25)

The OP2 decoder now exposes `UPDATE_EXECUTE_MASK`, `UPDATE_PRED`,
`BANK_SWIZZLE`, and `PRED_SEL`. The current vector evaluator rejects
predicate/mask updates, nonzero bank swizzles, and predicated instructions
rather than discarding shader state or read-cycle selection. Tests verify the
field positions and rejection before writeback. These semantics are based on
the R700-family field tables, not Latte hardware captures.

## Latte integer comparisons (2026-11-25)

Added the documented OP2 comparison subset `SETE_INT`, `SETNE_INT`,
`SETGT_INT`, `SETGE_INT`, `SETGT_UINT`, and `SETGE_UINT` (`0x03A`–`0x03F`).
Results use the documented all-ones true mask and zero false mask. Four-lane
tests cover signed and unsigned boundary comparisons. Debug and Release builds
and CTest passed 2/2 in both configurations; Latte hardware behavior remains
unverified.

## Latte bounded OP2 clause execution (2026-11-25)

Commit `dfcef2d` adds `execute_latte_alu_op2_clause()` for clauses composed
only of supported four-slot X/Y/Z/W OP2 groups. PV forwards between groups in
one clause and is reset at the next clause. Register writes are staged so an
unsupported later group cannot leave partial caller state. Tests cover PV
forwarding and rejection without partial writes. Debug and Release builds
succeeded and CTest passed 2/2 in both configurations. This is a CPU-side
experimental helper based on R700-family semantics; it is not wired into
guest shader invocation or verified against Latte hardware.

## Latte normal export decode and materialization (2026-11-25)

Commits `b675e32` and `c58ca2f` decode normal `EXPORT`/`EXPORT_DONE` metadata
and materialize supported absolute-GPR swizzle exports into typed CPU-side
values. The assembled pass-through fixture identifies a position export from
GPR 1 with XYZW selectors. Burst GPR reads, constant-zero/constant-one
selectors, and masked components are tested; relative addressing, buffer
encoding, reserved selectors, and out-of-range GPR access are rejected. Debug
and Release builds succeeded and CTest passed 2/2 in both configurations.
Exports are not yet connected to Vulkan stage outputs or an end-to-end guest
shader executor.

## Visible diagnostic triangle verification (2026-11-25)

Ran `AffogatoFrontend.exe --gx2-guest-triangle-display-test` from the Debug
build on the NVIDIA GeForce RTX 5070 Laptop GPU. The Vulkan window visibly
displayed guest-memory-backed geometry over the controlled green clear,
remained responsive after resizing, and closed cleanly. `VK_LAYER_KHRONOS_validation`
was installed and enabled by the Debug renderer; no validation messages were
emitted during this run. The test completion log confirmed float32, float16,
and UNORM8x4 guest geometry across triangle-list, indexed, strip, and fan
draws, including the supported PM4 draw forms, submitted with the fixed
diagnostic host shader. This verifies the visible diagnostic draw route only;
guest Latte shaders, guest render-target writes from normal draws, and WWHD
graphics remain unverified.

## Straight-line Latte shader subset (2026-11-25)

Added `execute_latte_shader_subset()` in
[`latte_shader_program.hpp`](../src/gpu/latte_shader_program.hpp). It processes
supported ALU clauses and normal export records in CF order, using caller-
supplied initial GPRs, and returns the final GPR file plus ordered export
batches. The assembled fixture verifies a MOV clause writing GPR 3 followed by
a position export from that register. `CALL_FS` is rejected explicitly, as
are other unsupported CF operations by the generic executor; the vertex-
specific executor has a bounded path for a separately processed leading fetch
call. The subset does not translate shaders to SPIR-V or participate in normal
guest draw submission. Debug and Release builds succeeded and CTest passed 2/2
in both configurations.

## Latte vertex semantic linkage (2026-11-25)

Added `resolve_latte_vertex_input_gpr()` in
[`latte_vertex_fetch.hpp`](../src/gpu/latte_vertex_fetch.hpp). It resolves the
8-bit fetch semantic ID against the low byte of up to 32 captured semantic
table entries, ignores the all-ones clear marker, and maps a match at index
`n` to `VS_GPR_BASE + 1 + n`. Duplicate matches, oversized tables, and
destination overflow are rejected. Tests cover a match with unrelated high
bits, absent and cleared entries, duplicate IDs, table bounds, and overflow.
The mapping follows Sections 2.5.1 and 2.5.3 of AMD's R6xx/R7xx 3D Engine
Programming Guide; it remains family-reference behavior rather than direct Wii
U hardware validation. The R6xx register reference describes the fetch
semantic ID as eight bits, while the R700 ISA manual says nine bits; Latte
applicability is unresolved. The resolver does not decode guest fetch fields or
populate shader inputs, and system-generated vertex/instance inputs are not
modeled. Debug and Release builds succeeded and CTest passed 2/2 in both
configurations.

The HLE audit also found that `GX2InitFetchShaderEx` labels its generated
word-0 instruction as semantic while its word-1 comment identifies the low
bits as a direct-GPR destination. AMD's ISA reference treats these as
alternative layouts. The word-0 semantic selector means the overlapping
word-1 field is a semantic ID for this generated instruction form. Decaf's
generator corroborates that encoding. The decoded field and diagnostic lookup
now use the name `semantic_id`; this is a correction to the local model and
does not claim direct Wii U hardware verification. Arbitrary guest fetch
programs are still not executed and shader input GPRs are not populated.

## Draw-time Latte semantic-table capture and GX2 population (2026-11-25)

Added `capture_latte_vertex_semantic_table()` in
[`latte_vertex_fetch.hpp`](../src/gpu/latte_vertex_fetch.hpp). It extracts the
32 `SQ_VTX_SEMANTIC` entries from a raw draw register snapshot using the
existing register base address. It requires every entry to be present as a
context-register write and rejects duplicate or wrongly typed entries instead
of assuming reset values. Tests verify extraction and semantic resolution,
and reject incomplete or wrongly typed snapshots. A follow-up source audit
cross-checked the GX2 descriptor layout and update sequence against Decaf's
`gx2_shaders.h` structure assertions and `GX2SetVertexShader` implementation.
That HLE now queues the semantic-clear register and declared table writes from
descriptor offsets `0x3C`, `0x40`, and `0x44`; it rejects counts above the
32-entry Latte table. Regression coverage checks the emitted command sequence
and extracts the table from a subsequent draw snapshot. This is reference
cross-checking, not Wii U hardware verification. A per-attribute decoder now
combines the semantic table, caller-supplied resolved vertex index and
`VS_GPR_BASE`, and an immutable draw-time vertex-buffer snapshot into one
destination GPR input. It rejects missing or invalid snapshots and returns
masked channels as unwritten. It does not assemble multiple attributes into a
complete invocation, model system-generated inputs, or decode arbitrary guest
fetch programs.

## Bounded Latte floating-point OP2 operations (2026-11-25)

Expanded the no-literal four-lane OP2 CPU executor with R700-family `ADD`,
`MUL_IEEE`, `MAX_DX10`, and `MIN_DX10` operations. The subset honors source
absolute/negate flags and the output multiplier, and rejects destination
clamp, predicate/mask updates, relative addressing, and other unsupported
state before writing registers. Regression coverage checks opcode decoding,
normal arithmetic, NaN selection, signed-zero selection, modifiers, and
transactional rejection. Debug and Release builds succeeded; CTest passed 2/2
in both configurations. These operations follow the R700 ISA manual and are
not verified against Latte hardware; they remain an isolated CPU shader model,
not guest draw execution or Vulkan shader translation.

## Latte VTX destination selectors (2026-11-25)

Added `decode_latte_vertex_attribute_vector()` in
[`latte_vertex_fetch.hpp`](../src/gpu/latte_vertex_fetch.hpp). It applies the
decoded destination selector for each output channel: source components 0–3,
constant 0.0, constant 1.0, or a masked (unwritten) channel. It rejects the
reserved selector and selections outside the format's component count, and
uses the existing endian-aware component decoder. Tests cover source
reordering, constants, masked channels, reserved selectors, and truncated
guest data. The selector behavior is documented in AMD's R700 ISA manual,
Chapter 5, and remains an AMD-family reference rather than Wii U hardware
verification. Arbitrary Latte fetch clauses are not executed.
`decode_latte_vertex_shader_input()`
now combines one decoded attribute with its draw-time snapshot and semantic
table, returning a destination GPR and optional component values. The caller
supplies the resolved vertex index and `VS_GPR_BASE`. Tests cover semantic
lookup, snapshot bounds, component selection, and absent semantics. Debug and
Release builds and both CTest suites passed 2/2. The existing diagnostic
Vulkan smoke test still passes; its pixels use the fixed diagnostic host
shader, not a guest Latte shader.

`apply_latte_vertex_shader_inputs()` now assembles the supported attribute
results into a caller-provided register file. It retains masked channels,
validates all attributes before mutating registers, and uses fixed-capacity
temporary storage bounded by the 32-entry semantic table. Tests verify the
register writes, preserved masked channel, and unchanged state when a later
attribute references a missing draw-time buffer snapshot. Debug and Release
builds succeeded, CTest passed 2/2 in each configuration, and the existing
diagnostic Vulkan triangle smoke test passed. This is input assembly for an
explicit caller-supplied vertex index; it does not execute a Latte vertex
shader or establish Wii U hardware conformance.

## CPU vertex-fetch to shader-subset path (2026-11-25)

Added `execute_latte_vertex_shader_subset()` in
[`latte_vertex_shader_subset.hpp`](../src/gpu/latte_vertex_shader_subset.hpp).
It parses the existing HLE-generated fetch-shader subset, resolves supported
draw-time vertex snapshots and semantics for one explicitly supplied vertex
index, then passes the populated registers into the existing straight-line
CPU shader subset. A deterministic fixture verifies big-endian Float32x2
input, a destination selector with a masked component, MOV execution, and
register output. A draw-snapshot overload obtains the fetch shader, vertex
shader, semantic table, and immutable vertex bytes from one captured
`Gx2DrawSnapshotView`; a regression test runs it through `Gx2RegisterState`.
This composes tested CPU helpers; it is not wired to normal Vulkan draw
submission, does not execute general guest fetch programs, and does not
generate Vulkan shaders. Debug and Release builds succeeded and CTest passed
2/2 in both configurations. The diagnostic Vulkan triangle smoke test passed;
it still uses host diagnostic shaders. The visible-window test was not run
because the Windows capture tool failed during initialization (`failed to
write kernel assets`), so no new visual verification is claimed.

## Bounded Latte fetch-call handoff (2026-11-25)

`decode_latte_control_flow()` now records the `CALL_FS` address in 8-byte
units, pop count, condition, and call count. A vertex-specific shader executor
can consume a single leading call only when its address is zero, its condition
is active, and both pop count and call count are zero. The draw-snapshot path
first processes the bound HLE-generated fetch subset, then executes the vertex
program with that call elided. Tests verify the decoded fields, successful
execution of the captured `CALL_FS` fixture after fetch processing, rejection
by the generic executor, and rejection of conditional calls. This follows the
R700-family reference and Cemu's separate fetch-shader lowering architecture;
the base-register linkage and behavior have not been verified on Wii U
hardware. It remains CPU-only and is not wired to normal Vulkan draws.

## GX2 draw vertex-index resolution (2026-11-25)

Added `decode_latte_draw_vertex_indices()` in
[`latte_draw.hpp`](../src/gpu/latte_draw.hpp) and routed the existing
diagnostic Vulkan path through it. It expands non-indexed first-vertex ranges
and captured U16/U32 index buffers for GX2's little- and big-endian index type
values, then applies the captured base vertex. Invalid snapshots, unsupported
index types, primitive-restart values, and 32-bit range overflow are rejected.
Tests cover all four index encodings, non-indexed ranges, base-vertex addition,
invalid snapshots, restart indices, and overflow. The helper does not execute
vertex shaders, handle instancing, or broaden the diagnostic renderer's
supported topology. WUT's public GX2 enum defines the four index-type values;
its source supports the numeric mapping, while Affogato fixtures verify the
byte-order interpretation. Debug and Release builds succeeded, CTest passed
2/2 in both configurations, and `--gx2-guest-triangle-smoke-test` passed on an
NVIDIA GeForce RTX 5070 Laptop GPU using the diagnostic Vulkan shader. This
does not verify visual presentation or Latte shader execution.

## GX2 vertex-shader register capture (2026-11-25)

Extended `GX2SetVertexShader` from shader/semantic capture to emit an ordered
register-write batch for the descriptor's program setup, vertex output state,
semantic table, and reuse controls. Geometry mode selects the ES program range
and captures its ring item size. Enabled stream-out writes four stride values
in DWORD units. Output-ID and semantic counts are checked before pending
commands or current shader state are modified. Tests cover register addresses
and values, draw-time snapshot retention, stream-out stride conversion,
geometry-mode setup, alias-resolved program starts, and invalid counts. WUT's
descriptor layout and Decaf's GX2 setter were compared; register addresses
were cross-checked against related-generation AMD/Linux references. This is
reference-derived behavior, not Wii U hardware validation. Debug and Release
builds succeeded, and CTest passed 2/2 in each configuration. The diagnostic
Vulkan smoke test passed with its host shader. No guest shader is executed by
the renderer, and this does not complete normal GX2 draw dispatch or establish
guest-generated Vulkan pixels.

## Controlled Latte constant fragment output (2026-11-25)

Added `latte_spirv_fragment_subset.hpp`, which translates four-lane OP2 MOV
groups from inline float zero/one/one-half constants and one completed,
identity-swizzled MRT0 export into a SPIR-V fragment module. Unsupported
operations, modifiers, exports, and unwritten output components are rejected.
The Vulkan diagnostic policy pairs that module with the existing bounded
Latte vertex module for both the test swapchain and the guest color-buffer
rendering/readback path.

The new `--gx2-guest-triangle-pixel-spirv-smoke-test` installs its synthetic
pixel shader using the emulated guest `GX2SetPixelShader` call, issues the
existing guest `GX2DrawEx` calls, captures draw-time shader state, and verifies
the magenta output in color-buffer and presented scanout readback. It is an
explicit controlled test policy; ordinary emulation does not silently replace
unsupported pixel shaders. The generated fragment module passes `spirv-val`,
and the dedicated Vulkan CTest passes on the local NVIDIA GPU. After adding
the test, the full Debug and Release CTest suites both passed 29/29, including
the fragment module validation and guest pixel readback.

This only constant-folds the described shader pattern. Fragment inputs,
general ALU, texture sampling, blending/depth state, general render-target
execution, and WWHD graphics remain unsupported. The AMD R700 export reference
and SPIR-V interface rules are recorded in
[`research/12_latte_shader_isa.md`](research/12_latte_shader_isa.md); the AMD
material is family-level evidence rather than direct Latte verification.

## Latte fragment export swizzles (2026-11-25)

Extended the bounded constant fragment translator to apply pixel export
component selectors for source lanes 0–3 and float literals zero/one. Reserved
and masked selectors remain explicit rejections because the current SPIR-V
output contract requires a defined `vec4`. Tests verify exact generated
constant ordering for swizzled/literal output and rejection of masked output.
Debug and Release builds and full CTest suites passed 29/29 in both
configurations, including fragment SPIR-V validation and controlled guest
pixel readback.
This follows Affogato's existing CPU normal-export selector behavior and the
related-family R700 reference, not a new claim of direct Latte conformance.
