param(
    [Parameter(Mandatory=$true)][string]$WorkerScript,
    [Parameter(Mandatory=$true)][int]$MemoryMiB,
    [Parameter(Mandatory=$true)][string]$LogPath,
    [switch]$AllowExistingPaging,
    [string[]]$WorkerArguments = @()
)
$ErrorActionPreference = 'Stop'
$rinkRuntimePath = 'D:\CodexData\.codex\visualizations\2026\10\02\private-thread-00008\RinkSpace_tmp\runtime_d_only'
$env:TEMP = Join-Path $rinkRuntimePath 'temp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$rinkProjectRoot = Split-Path -Parent $PSScriptRoot
$rinkPythonPath = Join-Path $rinkProjectRoot '.venv\Scripts\python.exe'
$rinkLauncherPath = Join-Path $PSScriptRoot 'run_rinkspace_d_only.py'
$rinkLauncherArguments = @('--memory-mib', $MemoryMiB, '--log', $LogPath)
if ($AllowExistingPaging) { $rinkLauncherArguments += '--allow-existing-paging' }
$rinkLauncherArguments += $WorkerScript
$rinkLauncherArguments += $WorkerArguments
& $rinkPythonPath -B $rinkLauncherPath @rinkLauncherArguments
exit $LASTEXITCODE
