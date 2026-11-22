# Espresso Privilege and Exception Audit

This audit records the CPU privilege and exception mechanisms currently
present in the repository. It describes source behavior, not full Wii U
compatibility.

## Current execution model

- `CpuState` models MSR, SRR0, and SRR1 in addition to the previously supported
  general CPU state. `CpuState::reset()` clears these fields.
- ELF and RPX loaders copy loadable sections to guest addresses and set `cia`
  directly to the executable entry point. Loader entry starts with MSR zero,
  which is supervisor-mode, big-endian, real-address execution. This is an
  explicit bootstrap model, not a Cafe OS kernel boot or complete Wii U reset
  state.
- `GuestMemory` provides flat backing memory, explicitly mapped regions, and
  aliases. CPU fetches and data accesses use those guest addresses directly;
  there is no effective-to-physical translation, segment protection, BAT
  lookup, or page-table/TLB walk.
- CPU privilege checks are architectural policy and are separate from Cafe OS
  HLE policy.
- Cafe OS thread creation stores `0x9032` in the guest context's `srr1` field,
  but the current scheduler does not restore that field into `CpuState::msr`.
  Its IR/DR bits require address translation, which `GuestMemory` does not
  implement. Thread-context MSR restoration remains unsupported.

## System registers and synchronization

The decoder recognizes dedicated forms for LR, CTR, XER, GQR0-GQR7, and the
time-base upper/lower reads. Generic `mfspr`/`mtspr` support is bounded to
SRR0/SRR1; other SPR encodings remain unsupported. `mfmsr` reads modeled MSR
for supervisor execution. `mtmsr` updates supported MSR state and rejects
attempts to enable instruction or data translation. `rfi` restores SRR0/SRR1
state and likewise stops with an explicit unsupported-state result when the
restored MSR enables translation. These context-changing instructions use the
interpreter fallback and redispatch at the instruction boundary. Existing
`isync`, `sync`, and `lwsync` behavior remains unchanged.

## Exceptions and errors

- Privileged access violations for implemented privileged instructions enter
  a modeled program exception. Entry stores the faulting CIA in SRR0, saves
  supported MSR state and the privilege syndrome in SRR1, clears translation
  and privilege state for this supported path, and vectors to the program
  exception handler at offset `0x700`, with the modeled MSR IP prefix.
- `EspressoCore::step()` still reports an unsupported decode as
  `unsupported_instruction`. Unsupported instructions are not yet converted
  into illegal-instruction program exceptions. `run()` reports them as
  terminal emulator stops with the faulting CIA and instruction word.
- Translation-enabled `mtmsr`/`rfi` requests report an explicit unsupported
  architectural-state stop and leave the current state unchanged; they do not
  silently execute through flat memory.
- `GuestMemoryFault` and `HleExecutionError` remain host-side errors reported
  as `memory_fault` and `hle_error`; they do not update guest SRRs or vector to
  guest handlers.
- Cafe OS exception callback registration is HLE bookkeeping, separate from
  PowerPC exception entry and return.
- The interpreter is the JIT fallback. Context-changing privileged
  instructions use that fallback and terminate execution at the instruction
  boundary so dispatch observes the updated state.

## Remaining unsupported instruction inventory

| Family | Current state | Dependencies / notes | WWHD evidence |
|---|---|---|---|
| `mfmsr` | Implemented for supervisor mode | User-mode access raises a modeled privileged-instruction program exception. | No verified occurrence. |
| `mtmsr` | Implemented for supported non-translated states | User-mode access raises a modeled privileged-instruction program exception. Enabling IR/DR is explicitly unsupported pending MMU translation. | No verified occurrence. |
| `rfi` | Implemented for supported non-translated return states | Privileged; restores modeled MSR from SRR1 and resumes at SRR0. Translation-enabled restore is explicitly unsupported. | No verified occurrence. |
| Generic `mfspr`, `mtspr` | Implemented only for SRR0/SRR1 | Other SPR encodings remain unsupported; full per-SPR policies and invalid-SPR exceptions are not implemented. | No verified occurrence. |
| Segment-register operations | Unsupported | Need segment-register state and translation/protection behavior. | No verified occurrence. |
| `tlbie`, `tlbsync` | Unsupported | Need TLB state, invalidation, and synchronization. | No verified occurrence. |
| `dcbi`, `dcbz_l` | Unsupported | Need cache-control/locked-cache behavior and alignment/enable rules. | No verified occurrence. |
| `eciwx`, `ecowx` | Unsupported | Need external-control device interface and its ordering/alignment semantics. | No verified occurrence. |
| Illegal instruction and other program exceptions | Partial | Privilege violations are modeled; unsupported decodes and other program causes do not yet enter guest exception handlers. | No verified handler trace. |
| Syscall, DSI/ISI, alignment, floating-point-unavailable exceptions | Unsupported | Need exception-specific saved state, vectors, restart rules, and related MMU/FPSCR integration. | No verified handler trace. |
| External interrupts and decrementer | Unsupported | Need interrupt sources, priority, masking, time-base/decrementer state, and asynchronous dispatch. | No verified requirement established. |

The IBM 750CL manual documents program exception vector offset `0x700`,
exception MSR transitions, real addressing when translation is disabled, and
`rfi` restoration from SRR1/SRR0. These hardware rules inform the bounded
implementation above. See the [IBM 750CL User's Manual](https://fail0verflow.com/media/files/ppc_750cl.pdf),
sections 2.3, 4.3-4.5, and 5.

## Remaining work and priority

The initial privilege foundation is implemented and covered by CPU and
interpreter/JIT fallback tests. Further exception classes and handler
execution should be driven by observed title traces. Cafe OS thread-context
MSR restore depends on address translation. Segment/TLB, cache-control, and
external-control operations remain deferred until their backing subsystems
exist.

Rendering is the nearer application blocker: current GX2 HLE captures state
and scan-buffer requests, while the frontend can present CPU-visible RGBA8
scan-buffer memory. The source does not register
`GX2CopyColorBufferToScanBuffer`, execute guest draw commands, or write guest
framebuffer pixels. Continue with the rendering path once its source surface,
copy semantics, and supported tiling can be established from code and tests.
