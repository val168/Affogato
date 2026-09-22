# 02 — Espresso CPU

**Status:** current research priority and next implementation milestone.

## Known high-level facts

Espresso is the Wii U's PowerPC CPU. WiiUBrew documents three cores, one hardware thread per core, and a clock around 1243.125 MHz.

Broadway/Wii research is useful background because Broadway is a Nintendo PowerPC design derived from the PowerPC 750 family. Broadway-derived information is a research aid, not proof that every behavior is identical on Espresso.

## User-visible 32-bit PowerPC programming model

The standard programming model includes:

- 32 general-purpose registers (GPRs), 32 bits each.
- 32 floating-point registers (FPRs), 64 bits each.
- Condition Register (CR), 32 bits.
- Fixed-Point Exception Register (XER), 32 bits.
- Link Register (LR), 32 bits in a 32-bit implementation.
- Count Register (CTR), 32 bits in a 32-bit implementation.
- Floating-Point Status and Control Register (FPSCR), 32 bits.

The privileged environment adds MSR, exception state, MMU-related registers, and other SPRs.

### CR

CR is 32 bits divided into eight 4-bit fields (CR0 through CR7). It is used by compare operations, record-form instructions, and conditional branches.

### XER

XER carries fixed-point status. Important concepts include summary overflow (SO), overflow (OV), and carry (CA). Only implement bits required by instructions as those instructions are added.

### LR

LR is used for branch-and-link behavior and as a branch target for link-register branch instructions. Function calls/returns commonly rely on it.

### CTR

CTR can be used as a branch target and as a loop/decrement condition for conditional branch forms.

### Instruction address

PowerPC documentation often describes current/next instruction address using CIA/NIA/IAR-style terminology rather than requiring a literal hardware register named `PC`.

Affogato still needs an internal field representing where guest execution currently is. Choose a name that makes the semantics clear.

## Espresso-specific documented state

WiiUBrew's Espresso page documents MSR fields and many SPR indices, including:

- XER
- LR
- CTR
- DSISR
- DAR
- DEC
- SDR1
- SRR0 / SRR1
- time-base registers
- SPRG registers
- PVR
- BAT registers

These should **not** all be added to the first `CpuState`.

## Minimal first CpuState

The first implementation should likely focus on state required by early integer instructions:

```text
GPR[32]
CR
XER
LR
CTR
current instruction address
```

Potentially defer until required:

```text
FPRs
FPSCR
MSR
exception registers
BAT/MMU state
time base
decrementer
performance/debug registers
paired-single/GQR state
```

## Instruction size and byte order

PowerPC uses fixed-width, word-aligned 32-bit instructions in the programming model relevant here.

Wii/Wii U documentation uses big-endian guest-facing conventions. Affogato should therefore treat endian handling explicitly and must not assume the little-endian x86-64 host matches guest layout.

## Paired singles

Nintendo's Gekko/Broadway lineage supports paired-single floating-point extensions and Graphics Quantization Registers (GQRs). These are relevant later for real game compatibility but are not required for the first integer interpreter milestone.

## First implementation milestone

```text
construct/reset state
    ->
all selected registers have deterministic values
    ->
write GPR3 = 42
    ->
verify GPR3 == 42
    ->
reset
    ->
verify expected reset state
```

No decoder is required for this milestone.

## Questions to settle before CpuState v1

- What exact name should Affogato use for the current instruction address?
- Which XER bits are needed by the first chosen instructions?
- Which CR operations should be helpers?
- Which reset values are architecture-defined versus emulator-selected test values?
- Which state is per-core versus global/shared?

## Sources

- WiiUBrew — Espresso: https://wiiubrew.org/wiki/Espresso
- WiiUBrew — Wii U Hardware: https://wiiubrew.org/wiki/Wii_U_console
- WiiBrew — Broadway: https://wiibrew.org/wiki/Broadway
- WiiBrew — Broadway registers: https://wiibrew.org/wiki/Broadway/Registers
- WiiBrew — Paired single: https://wiibrew.org/wiki/Paired_single
- NXP/Motorola — PowerPC Programming Environments Manual: https://www.nxp.com/docs/en/user-guide/MPCFPE_AD_R1.pdf
