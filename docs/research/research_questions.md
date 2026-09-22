# Research questions by development phase

Use this as a living checklist. Do not answer every question before coding.

## Phase 1 — architecture

- What roles do Espresso, Latte, GX2, Starbuck, Cafe OS, and IOSU play?
- Which component executes application code?
- Which components are directly visible to applications?
- How do Cafe OS and IOSU communicate?
- What major memory pools exist?
- What does the real boot process do?
- Which parts can Affogato skip or HLE?

**Stop condition:** you can draw the console architecture and identify Affogato's emulate/HLE/translate boundaries.

## Phase 2 — one Espresso core

- What is the relevant PowerPC programming model?
- How many GPRs exist and what width are they?
- What are CR, XER, LR, and CTR?
- How should the current instruction address be represented?
- Which state is required for the first integer instructions?
- Which state can wait?
- What is the relevant byte order?
- Which Broadway/Gekko behaviors are confirmed to apply to Espresso, and which are only lineage hints?

**Stop condition:** enough information to implement `CpuState` without guessing.

## Phase 3 — decoding

- Where is the primary opcode?
- Which instruction forms are needed first?
- How are GPR operand fields extracted?
- Which immediates are signed?
- How does sign extension work?
- How are secondary opcodes represented?
- What do Rc/OE/AA/LK mean when encountered?

**Stop condition:** decode one instruction correctly from a raw 32-bit word.

## Phase 4 — execution

- What exactly does each first instruction do?
- Which registers/flags change?
- When is rA==0 special?
- How are CR fields updated?
- How do XER CA/OV/SO work?
- How are branch targets computed?
- How do LR and CTR affect branches?
- How is fall-through/next-address behavior represented?

**Stop condition:** execute a small arithmetic/branch sequence using state only.

## Phase 5 — memory

- How should guest addresses be represented?
- Which widths must be read/written?
- Where should endian conversion happen?
- What alignment behavior is architecturally visible?
- What simple flat mapping is enough for synthetic tests?
- Which real Wii U regions are needed for the first loaded program?
- When does real MMU translation become unavoidable?

**Stop condition:** synthetic code can fetch instructions and manipulate data through `Memory`.

## Phase 6 — exceptions/privilege

- Which exceptions occur in the software being targeted?
- How are SRR0/SRR1/MSR used?
- How are invalid accesses represented?
- How do PPC syscalls enter Cafe OS?
- Which privileged state is required before homebrew runs?

## Phase 7 — validation

- What tiny programs can be hand-verified?
- How should execution terminate in tests?
- How can decoding and semantics be tested independently?
- What tests will later compare interpreter and JIT?

## Phase 8 — loaders

- How are ordinary PPC ELF files represented?
- How do RPX/RPL differ?
- Which sections are compressed?
- How are relocations encoded?
- How are imports/exports represented?
- How is guest memory laid out?
- How is startup/entry state established?

## Phase 9 — Cafe HLE

For each required import:

- library?
- function?
- calling convention?
- arguments?
- structures?
- return value?
- errors?
- memory side effects?
- blocking/thread behavior?
- IOSU dependency?

## Phase 10 — IOSU HLE

For each required service:

- device/service name?
- request operation?
- ioctl/ioctlv number?
- input/output structure?
- guest buffer rules?
- errors?
- async behavior?
- host replacement?

## Phase 11 — graphics

- how are GX2 commands/states submitted?
- buffers?
- surfaces?
- texture formats?
- tiling/swizzling?
- shaders?
- render targets?
- synchronization?
- TV/GamePad presentation?
- best Vulkan equivalent for each observable behavior?

## Phase 12 — multicore

- per-core versus shared state?
- atomics?
- reservation semantics?
- memory barriers/order?
- scheduler expectations?
- interrupts?
- inter-core communication?
- which games/homebrew prove multicore behavior?

## Rule for every phase

Ask four questions:

1. What does the real Wii U do?
2. What can guest software observe?
3. What does Affogato actually need to reproduce?
4. What can Affogato replace with host functionality?
