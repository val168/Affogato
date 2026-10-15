# Affogato — Local Build Instructions for Codex

Affogato is a C++23 project built with CMake + Ninja using MinGW-w64 GCC on Windows.

The goal of this file is to keep Codex using the correct project build workflow.

## Toolchain

Use:

- PowerShell
- GCC / G++ via MinGW-w64
- C++23
- CMake
- Ninja

Current local toolchain:

```text
GCC   16.1.0
CMake 4.3.3
Ninja 1.13.2
```

Do not assume MSVC or Clang.

## Important Rules

CMake is the source of truth for builds.

Do NOT compile source files manually with commands like:

```powershell
g++ src/main.cpp -o Affogato.exe
```

Do NOT use VS Code "Build Active File".

Do NOT invent separate build commands if the existing CMake presets work.

Do NOT manually edit generated files in:

```text
build/
```

Do NOT manually edit:

```text
CMakeUserPresets.json
```

unless specifically debugging the build configuration.

## Configure

From the repository root, use:

```powershell
.\scripts\configure.ps1
```

This discovers the local GCC/Ninja toolchain and configures both Debug and Release.

For Debug only:

```powershell
.\scripts\configure.ps1 -Configuration Debug
```

For Release only:

```powershell
.\scripts\configure.ps1 -Configuration Release
```

If the compiler or toolchain changes, rerun `configure.ps1`.

## Debug Build

Build with:

```powershell
cmake --build --preset local-debug
```

Run with:

```powershell
.\build\debug\Affogato.exe
```

## Release Build

Build with:

```powershell
cmake --build --preset local-release
```

Run with:

```powershell
.\build\release\Affogato.exe
```

## Presets

Available local presets should include:

```text
local-debug
local-release
```

Check them with:

```powershell
cmake --list-presets=all
```

## Testing

When tests exist, build Debug first:

```powershell
cmake --build --preset local-debug
```

Then run:

```powershell
ctest --test-dir build/debug --output-on-failure
```

## Before Finishing a Change

At minimum:

```powershell
cmake --build --preset local-debug
```

If the change may behave differently under optimization, also run:

```powershell
cmake --build --preset local-release
```

If tests exist, run:

```powershell
ctest --test-dir build/debug --output-on-failure
```

Do not consider a coding change complete if the relevant build fails.

## Git

Do not commit generated files such as:

```text
build/
CMakeUserPresets.json
compile_commands.json
*.exe
*.obj
*.o
```

Do not automatically commit or push unless explicitly asked.