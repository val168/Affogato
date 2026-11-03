Affogato — Local Build Instructions for Codex
Affogato is a C++23 project built with CMake + Ninja using MinGW-w64 GCC on Windows.

The goal of this file is to keep Codex using the correct project build workflow.
Toolchain
Use:
- PowerShell
- GCC / G++ via MinGW-w64
- C++23
- CMake
- Ninja
Current local toolchain:
GCC   16.1.0
CMake 4.3.3
Ninja 1.13.2
Do not assume MSVC or Clang.
Important Rules
CMake is the source of truth for builds.

Do NOT compile source files manually with commands like:
g++ src/main.cpp -o Affogato.exe
Do NOT use VS Code "Build Active File".

Do NOT invent separate build commands if the existing CMake presets work.

Do NOT manually edit generated files in:
build/
Do NOT manually edit:
CMakeUserPresets.json
unless specifically debugging the build configuration.
Configure
From the repository root, use:
.\scripts\configure.ps1
This discovers the local GCC/Ninja toolchain and configures both Debug and Release.

For Debug only:
.\scripts\configure.ps1 -Configuration Debug
For Release only:
.\scripts\configure.ps1 -Configuration Release
If the compiler or toolchain changes, rerun configure.ps1.
Debug Build
Build with:
cmake --build --preset local-debug
Run with:
.\build\debug\Affogato.exe
Release Build
Build with:
cmake --build --preset local-release
Run with:
.\build\release\Affogato.exe
Presets
Available local presets should include:
local-debug
local-release
Check them with:
cmake --list-presets=all
Testing
When tests exist, build Debug first:
cmake --build --preset local-debug
Then run:
ctest --test-dir build/debug --output-on-failure
Before Finishing a Change
At minimum:
cmake --build --preset local-debug
If the change may behave differently under optimization, also run:
cmake --build --preset local-release
If tests exist, run:
ctest --test-dir build/debug --output-on-failure
Do not consider a coding change complete if the relevant build fails.
Git
Do not commit generated files such as:
build/
CMakeUserPresets.json
compile_commands.json
*.exe
*.obj
*.o
Do not automatically commit or push unless explicitly asked.
Code Style and Readability Conventions
These conventions apply to new and substantially modified first-party C++ code. They extend the instructions above; they do not replace them. Follow nearby established patterns when they differ, and avoid unrelated mechanical reformatting. Do not restyle vendored dependencies in third_party/.
General Principles
- Optimize for correctness, clarity, and maintainability first; optimize hot paths based on measurements.
- Prefer straightforward code with visible control flow over clever, compressed expressions.
- Keep changes focused. Do not refactor unrelated code while implementing a feature or fixing a bug.
- Reuse existing helpers and types instead of duplicating logic.
- Avoid speculative abstractions, placeholder interfaces, dead code, and premature general-purpose frameworks.
- Keep emulator architectural state explicit. Do not trade PowerPC/GX2 correctness for cleaner-looking host code.
- Use C++23 features where they improve clarity, portability, or safety; do not introduce features solely for novelty.
Formatting
- Use 4 spaces for indentation. Do not use tabs in C++ source files.
- Use Allman braces: put { on the next line for functions, classes, namespaces, conditionals, and loops.
- Always use braces for if, else, for, while, and do, even for one-line bodies.
- Put one statement per line; do not chain declarations or control-flow statements.
- Add a blank line between logical phases of a function, but avoid gratuitous vertical whitespace.
- Use spaces around binary operators and after commas. Do not pad inside parentheses.
- When an expression or call becomes difficult to scan, wrap it across lines with meaningful indentation.
- Prefer multiline conditions with one logical clause per line when this improves readability.
- Avoid deeply nested ternaries, unnecessarily long one-line lambdas, and multiple side effects in one expression.
- Use a consistent namespace layout matching adjacent code (e.g. namespace affogato::cpu::espresso on its own line, followed by {).
- Keep existing line endings consistent with the file being edited. Avoid wholesale whitespace-only diffs.
[[nodiscard]] bool is_valid_guest_range(std::uint32_t address, std::size_t size)
{
    if (address == 0U)
    {
        return false;
    }

    if (size > maximum_guest_range)
    {
        return false;
    }

    return is_mapped_guest_memory(address, size);
}
Names
Follow the naming conventions already visible in Affogato:
Kind	Convention	Example
Namespace	snake_case	affogato::cpu::espresso
Function / method	snake_case	create_swapchain()
Local variable / parameter	snake_case	guest_address
Free constant / enum value	snake_case	command_buffer_alignment
Class / struct / enum type	PascalCase	VulkanRenderer, Gx2RuntimeState
Private data member	trailing _ when consistent with its class	swapchain_, device_
Source / header file	snake_case	guest_memory.hpp


- Prefer descriptive names that express intent, units, or meaning (guest_address, image_size_bytes, guest_instruction_count).
- Avoid single-letter identifiers except for tight, obvious indices or established architectural names.
- Preserve standard PowerPC register notation (r3, lr, ctr, xer, cr) and established Vulkan/GX2 terminology where helpful.
- Do not rename existing public interfaces merely to satisfy naming conventions.
- Do not use misleading boolean names: prefer has_depth_buffer or is_initialized to ambiguous names like depth or ready_flag.
Types, Constants, and Conversions
- Use std::uint8_t, std::uint16_t, std::uint32_t, and std::uint64_t where guest structures, register widths, and binary formats require exact sizes.
- Use std::size_t for container sizes and host buffer indexing when appropriate. Make guest/host conversions deliberate.
- Prefer constexpr or const to mutable variables when values are invariant.
- Use enum class for new strongly typed state categories where interoperability does not require raw integral constants.
- Avoid magic numbers. Name register offsets, masks, alignments, limits, and format IDs and cite the meaning when non-obvious.
- Keep existing unsigned-literal conventions (0U, 0x20U, ULL) where applicable.
- Prefer explicit, checked conversions for narrowing, signed/unsigned boundaries, and guest-address arithmetic.
- Use static_cast for justified conversions; do not use C-style casts.
- Do not assume host endianness or use unsafe type punning. Prefer existing big-endian memory helpers, std::bit_cast, and std::byteswap when suitable.
- Guard address-plus-length calculations against overflow before accessing guest memory.
- Use std::span, std::array, std::optional, and RAII ownership types when they make contracts clearer; avoid introducing allocations on hot paths without evidence.
Functions and Control Flow
- Give each function one coherent responsibility; extract a named helper when a section performs a separate, reusable task.
- Prefer early returns for invalid arguments and exceptional cases to reduce nesting.
- Keep parsing/validation, architectural state updates, and host-side effects distinguishable.
- Make side effects visible and explicit. Avoid helpers that silently mutate unrelated CPU, GPU, or global state.
- Pass read-only nontrivial objects by const& or appropriate views; pass small scalar types by value.
- Annotate functions with [[nodiscard]] when ignoring their result would likely be a bug.
- Use noexcept only where the operation truly cannot throw.
- Minimize mutable global state; prefer lifecycle-owned state and explicit dependencies.
- Do not suppress errors with empty catch blocks or fabricated success results.
- Use precise error messages that name the subsystem, operation, and relevant value or address.
Header and Include Hygiene
- Include a file's corresponding header first in its .cpp file, then project headers, then standard/third-party headers in a consistent grouping.
- Include the headers for symbols directly used instead of relying on transitive includes.
- Prefer forward declarations in headers when they reduce coupling without obscuring ownership or type requirements.
- Keep implementation-only helpers inside .cpp files or anonymous namespaces.
- Avoid unnecessary dependencies and circular includes.
- Never add using namespace ...; at global scope in project headers.
- Use #pragma once consistently with the existing project.
Comments and Documentation
- Write comments to explain why, hardware semantics, invariants, surprising constraints, or reference-derived behavior—not to narrate obvious statements.
- For PowerPC and Latte register fields, document bit positions, units, encoding assumptions, or specification references when the meaning is not evident.
- Document intentional deviations, unsupported cases, and known limitations honestly.
- Avoid vague comments such as // fix issue, // handle value, or // TODO later without actionable detail.
- Update comments when behavior changes; remove comments made incorrect by a refactor.
- Keep public interface contracts and non-obvious ownership/lifetime rules discoverable near their declarations.
- Avoid large decorative comment banners and unnecessary generated documentation.
Emulator-Specific Correctness
- Treat the interpreter as the architectural reference for JIT behavior, subject to independently verified ISA semantics.
- Do not implement unsupported PowerPC instructions as silent NOPs or silently ignore required state changes.
- Preserve CR, XER, LR, CTR, FPSCR, floating-point edge cases, exception behavior, and memory ordering as applicable.
- Keep guest big-endian representation separate from host representation and clearly name conversions.
- Validate guest pointers and bounds before reads/writes; never directly dereference a raw guest address as a host pointer.
- For GX2/HLE functions, verify the calling convention, structure layout, return values, register encoding, and guest-visible side effects.
- Do not return fabricated HLE success where necessary behavior has not been implemented.
- Distinguish actual guest GPU commands and frames from synthetic Vulkan smoke-test output.
- Preserve JIT code-cache invalidation, fallback behavior, and CPU-state synchronization while optimizing.
- Favor explicit errors for unsupported formats/operations rather than silently producing incorrect output.
Performance-Sensitive Code
- Make the normal execution path easy to identify, especially in the interpreter, JIT dispatcher, GX2 command processing, and Vulkan submission path.
- Avoid repeated heap allocation, formatting/logging, and expensive conversions in hot loops unless justified.
- Keep diagnostics optional or sufficiently cheap for performance runs.
- Use profiling or repeatable benchmarks before claiming speedups.
- Do not replace readable code with micro-optimizations unless measurements show a meaningful benefit.
- Keep fallback paths correct even when optimizing the fast path.
Testing and Change Discipline
- Add focused regression tests for every bug fix and supported instruction/feature where practical.
- For interpreter/JIT changes, compare architectural state and affected memory, not just a final return value.
- Cover normal, boundary, invalid, and relevant endian/overflow cases.
- Verify that CPU-only tests remain usable without a Vulkan-capable display.
- Prefer deterministic tests and bounded execution over timing-sensitive assertions.
- Run the build/test instructions earlier in this file before considering a change complete.
- Inspect git diff --check and review the final diff for unrelated formatting, accidental generated files, and stale comments.
- When a task explicitly requests Git commits or pushes, follow that task's commit/push cadence; otherwise preserve the Git rule above.
Codex Self-Review Checklist
Before completing each independently testable change, check:
1. Does the implementation match existing architecture and naming conventions?
2. Can a reader understand each function without mentally decoding dense expressions?
3. Are non-obvious hardware rules, constants, and assumptions documented?
4. Are guest-memory bounds, endianness, and architectural side effects correct?
5. Is error handling explicit rather than hidden by stubs or silent fallbacks?
6. Are tests meaningful and passing? Are performance claims benchmarked?
7. Is the diff focused, readable, and free of generated or unrelated changes?
Prefer a clean, readable, correct implementation over a larger or more clever one.