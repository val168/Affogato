[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release", "All")]
    [string]$Configuration = "All"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$MinimumCMake = [Version]"3.25.0"
$MinimumNinja = [Version]"1.11.0"
$MinimumGcc = [Version]"14.0.0"

function Convert-ToVersion {
    param(
        [Parameter(Mandatory)]
        [string]$Text
    )

    $match = [regex]::Match($Text, "\d+\.\d+(?:\.\d+)?")

    if (-not $match.Success) {
        throw "Could not parse version from: $Text"
    }

    $value = $match.Value

    if (($value -split "\.").Count -eq 2) {
        $value += ".0"
    }

    return [Version]$value
}

function Normalize-Path {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    return ($Path -replace "\\", "/").TrimEnd("/").ToLowerInvariant()
}

# ------------------------------------------------------------
# CMake
# ------------------------------------------------------------

$cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue

if (-not $cmakeCommand) {
    throw "CMake was not found in PATH."
}

$cmakePath = $cmakeCommand.Source
$cmakeVersionText = & $cmakePath --version | Select-Object -First 1
$cmakeVersion = Convert-ToVersion $cmakeVersionText

if ($cmakeVersion -lt $MinimumCMake) {
    throw "CMake $MinimumCMake or newer is required. Found $cmakeVersion."
}

# ------------------------------------------------------------
# GCC / G++
# ------------------------------------------------------------

$gppPaths = @()

$gppPaths += Get-Command g++.exe -All -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Source

$wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"

if (Test-Path $wingetRoot) {
    $winlibsPackages = Get-ChildItem `
        -Path $wingetRoot `
        -Directory `
        -Filter "BrechtSanders.WinLibs*" `
        -ErrorAction SilentlyContinue

    foreach ($package in $winlibsPackages) {
        $candidate = Join-Path $package.FullName "mingw64\bin\g++.exe"

        if (Test-Path $candidate) {
            $gppPaths += $candidate
        }
    }
}

$gppPaths = $gppPaths |
    Where-Object { $_ } |
    Sort-Object -Unique

$compatibleCompilers = @()

foreach ($path in $gppPaths) {
    try {
        $versionText = (& $path -dumpfullversion).Trim()
        $target = (& $path -dumpmachine).Trim()
        $version = Convert-ToVersion $versionText

        if (
            $version -ge $MinimumGcc -and
            $target -eq "x86_64-w64-mingw32"
        ) {
            $compatibleCompilers += [PSCustomObject]@{
                Path = $path
                Version = $version
                Target = $target
            }
        }
    }
    catch {
        # Ignore unusable compiler candidates.
    }
}

if ($compatibleCompilers.Count -eq 0) {
    throw "No compatible x86-64 MinGW GCC >= $MinimumGcc was found."
}

$compiler = $compatibleCompilers |
    Sort-Object Version -Descending |
    Select-Object -First 1

$gppPath = $compiler.Path

# ------------------------------------------------------------
# Ninja
# ------------------------------------------------------------

$ninjaPaths = @()

$compilerNinja = Join-Path (Split-Path $gppPath) "ninja.exe"

if (Test-Path $compilerNinja) {
    $ninjaPaths += $compilerNinja
}

$ninjaPaths += Get-Command ninja.exe -All -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Source

$ninjaPaths = $ninjaPaths |
    Where-Object { $_ } |
    Sort-Object -Unique

$compatibleNinja = @()

foreach ($path in $ninjaPaths) {
    try {
        $versionText = (& $path --version).Trim()
        $version = Convert-ToVersion $versionText

        if ($version -ge $MinimumNinja) {
            $compatibleNinja += [PSCustomObject]@{
                Path = $path
                Version = $version
            }
        }
    }
    catch {
        # Ignore unusable Ninja candidates.
    }
}

if ($compatibleNinja.Count -eq 0) {
    throw "No compatible Ninja >= $MinimumNinja was found."
}

$ninja = $compatibleNinja |
    Sort-Object Version -Descending |
    Select-Object -First 1

# ------------------------------------------------------------
# Verify C++23
# ------------------------------------------------------------

$tempBase = Join-Path $env:TEMP "affogato-cxx23-$PID"
$tempSource = "$tempBase.cpp"
$tempObject = "$tempBase.o"

@'
#include <expected>

int main()
{
    std::expected<int, int> value = 42;
    return value.value() == 42 ? 0 : 1;
}
'@ | Set-Content -Path $tempSource -Encoding UTF8

try {
    & $gppPath `
        "-std=c++23" `
        "-c" `
        $tempSource `
        "-o" `
        $tempObject

    if ($LASTEXITCODE -ne 0) {
        throw "The selected compiler failed the C++23 verification."
    }
}
finally {
    Remove-Item $tempSource -Force -ErrorAction SilentlyContinue
    Remove-Item $tempObject -Force -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------
# Generate local CMake presets
# ------------------------------------------------------------

$gppPresetPath = $gppPath -replace "\\", "/"
$ninjaPresetPath = $ninja.Path -replace "\\", "/"

$userPresets = [ordered]@{
    version = 6

    configurePresets = @(
        [ordered]@{
            name = "local-debug"
            displayName = "Local GCC Debug"
            inherits = "debug"

            cacheVariables = [ordered]@{
                CMAKE_CXX_COMPILER = $gppPresetPath
                CMAKE_MAKE_PROGRAM = $ninjaPresetPath
            }
        }

        [ordered]@{
            name = "local-release"
            displayName = "Local GCC Release"
            inherits = "release"

            cacheVariables = [ordered]@{
                CMAKE_CXX_COMPILER = $gppPresetPath
                CMAKE_MAKE_PROGRAM = $ninjaPresetPath
            }
        }
    )

    buildPresets = @(
        [ordered]@{
            name = "local-debug"
            configurePreset = "local-debug"
        }

        [ordered]@{
            name = "local-release"
            configurePreset = "local-release"
        }
    )
}

$userPresetPath = Join-Path $ProjectRoot "CMakeUserPresets.json"

$userPresets |
    ConvertTo-Json -Depth 8 |
    Set-Content -Path $userPresetPath -Encoding UTF8

# ------------------------------------------------------------
# Handle an old compiler cached by CMake
# ------------------------------------------------------------

function Reset-BuildIfCompilerChanged {
    param(
        [Parameter(Mandatory)]
        [string]$BuildDirectory
    )

    $cache = Join-Path $BuildDirectory "CMakeCache.txt"

    if (-not (Test-Path $cache)) {
        return
    }

    $compilerLine = Select-String `
        -Path $cache `
        -Pattern "^CMAKE_CXX_COMPILER:FILEPATH=(.+)$" |
        Select-Object -First 1

    if (-not $compilerLine) {
        return
    }

    $cachedCompiler = $compilerLine.Matches[0].Groups[1].Value

    if (
        (Normalize-Path $cachedCompiler) -ne
        (Normalize-Path $gppPath)
    ) {
        Write-Host "Compiler changed. Recreating $BuildDirectory"
        Remove-Item $BuildDirectory -Recurse -Force
    }
}

Write-Host ""
Write-Host "Affogato toolchain"
Write-Host "------------------"
Write-Host "CMake : $cmakeVersion"
Write-Host "G++   : $($compiler.Version)"
Write-Host "Target: $($compiler.Target)"
Write-Host "Ninja : $($ninja.Version)"
Write-Host ""
Write-Host "Compiler: $gppPath"
Write-Host ""

$presets = switch ($Configuration) {
    "Debug"   { @("local-debug") }
    "Release" { @("local-release") }
    "All"     { @("local-debug", "local-release") }
}

Push-Location $ProjectRoot

try {
    foreach ($preset in $presets) {
        if ($preset -eq "local-debug") {
            $buildDirectory = Join-Path $ProjectRoot "build\debug"
        }
        else {
            $buildDirectory = Join-Path $ProjectRoot "build\release"
        }

        Reset-BuildIfCompilerChanged $buildDirectory

        Write-Host "Configuring $preset..."

        & $cmakePath --preset $preset

        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed for preset '$preset'."
        }
    }
}
finally {
    Pop-Location
}