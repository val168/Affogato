# 08 — Executable formats: ELF, RPX, RPL

**Status:** research after synthetic PowerPC programs execute correctly.

## Goal

Load actual Wii U code without emulating Nintendo's real loader implementation.

## Cafe OS executable model

WiiUBrew documents Cafe OS applications and libraries as modified ELF files.

RPX/RPL files are modified ELF shared objects used by Cafe OS.

Important documented differences from ordinary ELF include:

- some sections can be zlib-compressed.
- RPX/RPL loading relies on section headers rather than normal ELF program headers.
- dynamic linking uses import/export sections.
- executable code/data can be relocated and linked against RPL libraries.

## Affogato loader responsibilities

Eventually:

```text
open RPX/RPL
    ->
validate format
    ->
read/decompress required sections
    ->
allocate/map guest memory
    ->
apply relocations
    ->
resolve imports/exports
    ->
identify entry point / startup state
    ->
begin guest execution
```

This should be native host code.

Do not emulate Cafe OS's `loader.elf` just to load the first program.

## Research progression

1. Understand ordinary 32-bit PowerPC ELF concepts.
2. Understand exact RPX/RPL section conventions.
3. Implement parsing without execution.
4. Verify section contents and addresses.
5. Map into guest memory.
6. Implement relocations.
7. Implement imports/exports.
8. Try small homebrew.

## Questions to answer

- exact ELF identification fields used by RPX/RPL.
- section types/flags.
- compression flags/format.
- relocation types.
- import/export section naming and contents.
- entry-point/startup conventions.
- module dependencies.
- how `coreinit`, `gx2`, etc. imports are identified.

## Sources

- WiiUBrew — RPL: https://wiiubrew.org/wiki/RPL
- WiiUBrew — Cafe OS: https://wiiubrew.org/wiki/Cafe_OS
