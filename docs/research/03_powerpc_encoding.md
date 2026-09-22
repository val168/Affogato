# 03 — PowerPC instruction encoding

**Status:** next research phase after the minimal `CpuState`.

## Goal

Understand enough PowerPC instruction encoding to convert one 32-bit guest instruction word into a small decoded representation.

Do **not** try to implement the entire ISA at once.

## Required knowledge

Research and record:

- fixed 32-bit instruction width.
- instruction alignment.
- primary opcode location.
- instruction forms relevant to the first instructions.
- register operand fields such as rD/rS/rA/rB.
- signed and unsigned immediate fields.
- sign extension rules.
- extended/secondary opcode fields.
- control bits such as Rc, OE, AA, LK where applicable.

Instruction forms to understand gradually include D-form, X-form, XO-form, I-form, and B-form.

## Decoder milestone

```text
32-bit instruction word
    ->
extract primary opcode
    ->
recognize one known instruction
    ->
return a decoded representation
```

Then add tests using hard-coded instruction words with independently known fields.

## Decoder design principles

- decoding and execution should remain conceptually separate.
- bit extraction helpers should be explicit and testable.
- avoid giant switch tables generated all at once.
- unknown/unsupported instructions should produce a clear result rather than silently misdecode.
- keep raw instruction bits available for debugging.

## Do not research yet

- exact real Espresso decoder hardware.
- pipeline decode stages.
- issue width.
- physical execution-unit routing.
- instruction timing.

Affogato needs the architectural encoding, not the physical decoder implementation.

## Sources

- NXP/Motorola — PowerPC Programming Environments Manual: https://www.nxp.com/docs/en/user-guide/MPCFPE_AD_R1.pdf
- WiiUBrew — Espresso: https://wiiubrew.org/wiki/Espresso
- WiiBrew — Paired single: https://wiibrew.org/wiki/Paired_single
