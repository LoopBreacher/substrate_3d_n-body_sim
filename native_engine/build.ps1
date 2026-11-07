# Configures and builds the native Substrate engine with MSVC + CUDA + Ninja.
# Usage:  .\build.ps1            (Release)
#         .\build.ps1 -Config Debug
#         .\build.ps1 -Clean
param(
    [ValidateSet("Release", "RelWithDebInfo", "Debug")]
    [string]$Config = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$buildDir = Join-Path $root "build"

if ($Clean -and (Test-Path $buildDir)) { Remove-Item -Recurse -Force $buildDir }

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "Visual Studio with C++ tools not found." }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

$cmd = "`"$vcvars`" >nul && " +
       "cmake -S `"$root`" -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=$Config && " +
       "cmake --build `"$buildDir`" --parallel"

cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }

Write-Host "`nBuilt: $buildDir\bin\substrate_native.exe" -ForegroundColor Green

