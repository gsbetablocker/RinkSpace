[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string]$Output,
    [string]$RecoveryBinding,
    [string]$DataRoot
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (!$DataRoot) {
    . (Join-Path $project 'launchers/LOCAL_PYTHON.ps1')
    $basePython = Get-RinkSpacePythonPath $project
    $env:PYTHONPATH = Join-Path $project 'src'
    $DataRoot = (& $basePython -B -X utf8 -c 'import sys; from rinkspace.storage import configured_data_root; print(configured_data_root(sys.argv[1]))' $project | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or !$DataRoot) { throw 'Cannot resolve the RinkSpace Data directory.' }
}
$data = [IO.Path]::GetFullPath($DataRoot).TrimEnd('\','/')
$env:RINKSPACE_DATA_ROOT = $data
$Output = [IO.Path]::GetFullPath($Output)
if (!$Output.StartsWith("$data\", [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Choose an experiment output directory under RinkSpace_Data.'
}
if (Test-Path -LiteralPath $Output) {
    throw 'Choose a new output directory. Existing results remain reviewable.'
}
$python = Join-Path $data 'runtime_tools\vision_cuda\Scripts\python.exe'
if ($RecoveryBinding) {
    $env:RINKSPACE_SKATE_RECOVERY_BINDING = [IO.Path]::GetFullPath($RecoveryBinding)
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
[void][IO.Directory]::CreateDirectory($Output)
$first = Join-Path $Output 'first'
[void][IO.Directory]::CreateDirectory($first)

function Invoke-Stage([string]$File, [string]$Log, [string[]]$Arguments) {
    Write-Output "Run log $Log"
    & $python -B (Join-Path $PSScriptRoot $File) @Arguments *> $Log
    if ($LASTEXITCODE -ne 0) { throw "Stage $File failed. Read $Log." }
}

Invoke-Stage 'projection.py' "$Output\native_closure.log" @('--closure', "$Output\native_closure.json")
Invoke-Stage 'track.py' "$Output\tracking.log" @('--output', "$first\tracked", '--stride', '2')
Invoke-Stage 'recover.py' "$Output\recovery.log" @('--source', $first, '--output', $Output)
Invoke-Stage 'evidence.py' "$Output\evidence.log" @('--output', $Output)
Invoke-Stage 'render_review.py' "$Output\render_image.log" @('--output', $Output)
Invoke-Stage 'bundle.py' "$Output\native_recovery.log" @('--output', $Output, '--native-recovery', '--rounds', '9')
Invoke-Stage 'bundle.py' "$Output\solve.log" @('--output', $Output)
Invoke-Stage 'fusion.py' "$Output\fusion.log" @('--output', $Output)
Invoke-Stage 'render_review.py' "$Output\render_compare.log" @('--output', $Output, '--compare')
