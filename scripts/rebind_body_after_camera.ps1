param(
    [Parameter(Mandatory=$true)][string]$Camera,
    [Parameter(Mandatory=$true)][string]$Scene,
    [Parameter(Mandatory=$true)][string]$ScaleSource,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$V8Directory='D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/body_v8',
    [string]$ClockEvidence='D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/same_exposure_direct_pose_cache_overlay_20261005/portable_body_fit_clock_source_evidence_overlay.json',
    [string]$DirectPoseCache='D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/same_exposure_direct_pose_cache_overlay_20261005/same_exposure_direct_pose_cache.json',
    [string]$Python='',
    [string]$DataRoot='D:/CodexData/RinkSpace_Data',
    [int]$MaxNfev=100
)
$ErrorActionPreference='Stop'
$TaskRepo=Split-Path -Parent $PSScriptRoot
$TaskV8=$V8Directory
if (-not [System.IO.Path]::IsPathRooted($Output) -or -not [System.IO.Path]::IsPathRooted($DataRoot)) {
    throw 'Output and DataRoot must be absolute Data paths.'
}
$TaskDataPrefix=[System.IO.Path]::GetFullPath($DataRoot).TrimEnd([char[]]@('\','/')) + [System.IO.Path]::DirectorySeparatorChar
$TaskOutputPath=[System.IO.Path]::GetFullPath($Output)
if (($TaskOutputPath -split '[\\/]') -contains 'OneDrive') {
    throw 'Body outputs must stay outside OneDrive.'
}
if (-not $TaskOutputPath.StartsWith($TaskDataPrefix,[System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Body outputs must stay inside the configured Data root.'
}
$Output=$TaskOutputPath
$TaskSceneHash=(Get-FileHash -LiteralPath $Scene -Algorithm SHA256).Hash.ToLowerInvariant()
$TaskScale=Get-Content -LiteralPath $ScaleSource -Raw | ConvertFrom-Json
if ($TaskScale.schema_version -ne 'rinkspace.assumed-uniform-metric-scale.v2') {
    throw 'Use the post-geometry uniform closest-60x30 scale manifest.'
}
$TaskMatchingScenes=@($TaskScale.compatible_native_scenes | Where-Object { $_.sha256 -eq $TaskSceneHash })
if ($TaskMatchingScenes.Count -ne 1) { throw 'Scene and uniform-scale SHA binding do not match.' }
if (Test-Path -LiteralPath $Output) { throw 'Preserve existing body runs; select a fresh output.' }
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$TaskRepo/src;$TaskRepo"
$TaskPython=if ([string]::IsNullOrWhiteSpace($Python)) { Join-Path $TaskRepo '.venv/Scripts/python.exe' } else { $Python }
$TaskArguments=@('-B','-m','calibration.program_tracking.body_tracking.fullprogramme_camera_rebind',
    '--v8-directory',$TaskV8,'--camera',$Camera,'--scale-source',$ScaleSource,
    '--body-fit-clock-evidence',$ClockEvidence,
    '--same-exposure-direct-pose-cache',$DirectPoseCache,
    '--output',$Output,'--fit-root','--max-nfev',[string]$MaxNfev)
Push-Location $TaskRepo
try {
    & $TaskPython @TaskArguments
    if ($LASTEXITCODE -ne 0) { throw "Body rebind/root fit failed: $LASTEXITCODE" }
} finally { Pop-Location }
