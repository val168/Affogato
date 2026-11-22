# Espresso Privilege and Exception Audit

This audit records the CPU privilege and exception mechanisms present in the
repository before adding privileged instruction support. It describes source
behavior, not full Wii U compatibility.

## Current execution model

- `CpuState::reset()` clears all modeled architectural state. There is no MSR,
  SRR0, SRR1, segment-register, BAT, or exception-cause state in `CpuState`.
- ELF and RPX loaders copy loadable sections to guest addresses and set `cia`
  directly to the executable entry point. The HLE-oriented startup path does
  not execute a Cafe OS kernel or establish a modeled privilege context first.
- `GuestMemory` provides flat backing memory, explicitly mapped regions, and
  aliases. CPU fetches and data accesses use those guest addresses directly;
  there is no effective-to-physical translation, segment protection, BAT
  lookup, or page-table/TLB walk.
- Therefore loader entry currently behaves as real-address execution with no
  architectural privilege distinction. The emulator must make the intended
  launch context explicit before enabling privilege checks.

## Existing system-register and synchronization support

The decoder recognizes dedicated forms for LR, CTR, XER, GQR0–GQR7, and the
time-base upper/lower reads. It does not preserve a generic SPR number, so
other `mfspr`/`mtspr` encodings decode as unsupported. `isync`, `sync`, and
`lwsync` decode and execute as no-ops. There is no `mfmsr`, `mtmsr`, or `rfi`
support.

Cafe OS thread contexts contain guest `srr0` and `srr1` words. The current
context-switch helpers save the resume CIA to `srr0`, then restore that word to
`cia`; they do not preserve or load an MSR image from `srr1`. Thread creation
initializes the saved `srr1` word to `0x9032`. This context format must be
handled deliberately when MSR state is introduced.

## Existing exception and error handling

- `EspressoCore::step()` returns `unsupported_instruction` for an unsupported
  decode. `run()` converts that into a terminal `StopReason` with the faulting
  CIA and instruction word.
- `GuestMemoryFault` and `HleExecutionError` are host-side exceptions caught by
  `run()` and reported as `memory_fault` or `hle_error`. They do not update
  guest SRRs or vector to guest handlers.
- Cafe OS exception callback registration is HLE bookkeeping for guest OS
  callbacks. It is separate from PowerPC exception entry and return.
- The interpreter is the JIT fallback. Native blocks use generated stores to
  `CpuState` fields and block termination/dispatch rules; context-changing
  instructions must end the current native block and redispatch after state
  changes.

## Unsupported instruction inventory

| Family | Current state | Dependencies / notes | WWHD relevance evidence |
|---|---|---|---|
| `mfmsr`, `mtmsr`, `rfi` | Decode as unsupported | MSR, SRR0/SRR1, privilege checks, exception entry/return, and block redispatch. `mtmsr` can change IR/DR, so correct continuation also requires address translation. | No verified trace evidence yet. |
| Generic `mfspr`, `mtspr` | Only selected SPR aliases decode | Generic SPR decoding and per-SPR access/privilege policy; unsupported or supervisor-only SPRs must raise the correct guest program exception. | No verified trace evidence yet. |
| Segment-register operations (`mfsr`, `mtsr`, `mfsrin`, `mtsrin`) | Decode as unsupported | Segment-register state and translation/protection behavior. | No verified trace evidence yet. |
| `tlbie`, `tlbsync` | Decode as unsupported | TLB state, invalidation and synchronization. Must not be treated as no-ops while translation is active. | No verified trace evidence yet. |
| `dcbi`, `dcbz_l` | Decode as unsupported | Cache-control/locked-cache behavior and alignment/enable rules. | No verified trace evidence yet. |
| `eciwx`, `ecowx` | Decode as unsupported | External-control device interface and its ordering/alignment semantics. | No verified trace evidence yet. |
| Program, syscall, DSI/ISI, alignment, floating-point-unavailable exceptions | Not modeled architecturally | Exception-specific SRR1 fields, vectors, MSR transitions, restart CIA rules, and in some cases MMU or FPSCR integration. | No verified exception-handler trace yet. |
| External interrupts and decrementer | Not modeled | Interrupt sources, priority, masking, time-base/decrementer state, and asynchronous dispatch. | No verified application requirement established. |

The IBM 750CL manual identifies program exceptions at vector offset `0x700`,
states that exceptions clear IR and DR, and describes real addressing as direct
effective-address use when the corresponding translation bit is clear. It also
specifies that `rfi` restores MSR state from SRR1 and resumes at SRR0. These
facts make translation and context synchronization explicit dependencies of
privileged-state support. See the [IBM 750CL User's Manual](https://fail0verflow.com/media/files/ppc_750cl.pdf),
sections 2.3, 4.3–4.5, and 5.

## Implementation order

1. Add explicit MSR/SRR state and define the direct-entry application context.
2. Add a guest exception result/entry path for verified synchronous exception
   classes, preserving host HLE and memory errors as host errors.
3. Add privilege-aware SPR decoding and implement only SPRs with modeled state.
4. Add `mfmsr`, `mtmsr`, and `rfi` with interpreter tests; end native blocks at
   each context-changing operation and initially execute them through the
   interpreter fallback.
5. Keep IR/DR-enabled execution unavailable until address translation exists;
   report this limitation explicitly instead of silently using flat memory.
6. Defer segment/TLB/cache/external-control operations until their backing
   subsystems exist. Prioritize GX2/Vulkan work when remaining privileged
   instructions lack real application traces.
