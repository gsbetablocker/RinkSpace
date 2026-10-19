param(
    [string]$InputConfig = '',
    [string]$Seed = '',
    [string]$Output = '',
    [int]$AcceptedSteps = 0,
    [switch]$VerifyBackendsOnly
)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = $PSScriptRoot
$taskPython = Join-Path $taskProjectRoot '.venv/Scripts/python.exe'
if (-not $InputConfig) { $InputConfig = Join-Path $taskProjectRoot 'configs/solver_runtime/okada_named66.json' }
$taskConfig = Get-Content -LiteralPath $InputConfig -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskConfig.schema -ne 'rinkspace.reusable-solver-inputs.v1') { throw 'Unsupported solver input configuration.' }
if ($AcceptedSteps -lt 0) { throw 'AcceptedSteps must be zero or positive. Zero follows the solver convergence condition.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
& $taskPython -B (Join-Path $taskProjectRoot 'scripts/check_onedrive_paths.py') --root $taskProjectRoot
if ($LASTEXITCODE -ne 0) { throw 'Project paths exceed the Windows sync limit. See the diagnostic above.' }
$env:RINKSPACE_SOLVER_DATA_ROOT = $taskConfig.resource_root
$taskSavedBenchmarkDiagnosticDir = $env:RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR
$taskSavedBenchmarkTimingsPath = $env:RINKSPACE_BENCHMARK_TIMINGS_PATH
$taskSavedStartupDiagnosticEvidence = $env:RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE
Push-Location -LiteralPath $taskProjectRoot
try {
    if (-not $VerifyBackendsOnly) {
        & $taskPython -B -m calibration.solver_runtime.strict_model_scope_inputs --config $InputConfig
        if ($LASTEXITCODE -ne 0) { throw 'The current strict source input branch is not ready for fitting.' }
    }
    if ($taskConfig.startup_diagnostic_evidence) {
        $env:RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE = [string]$taskConfig.startup_diagnostic_evidence
    }
    & $taskPython -B -m calibration.solver_runtime verify-backends
    if ($LASTEXITCODE -ne 0) { throw 'Verified native backends are unavailable. See the diagnostic above.' }
    if ($VerifyBackendsOnly) { return }
    if (-not $Output) { $Output = Join-Path $taskConfig.resource_root ('okada_project_fast_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')) }
    if (Test-Path -LiteralPath $Output) { throw 'Output must be a new directory; existing evidence is preserved.' }
    # Preparation writes before the driver creates Output, so use its sibling.
    $env:RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR = [System.IO.Path]::GetFullPath($Output + '.bootstrap')
    $env:RINKSPACE_BENCHMARK_TIMINGS_PATH = Join-Path $env:RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR 'benchmark_timings_latest.json'
    $taskArguments = @('-B', '-u', '-m', 'calibration.solver_runtime', $taskConfig.driver, '--output', $Output)
    if ($taskConfig.driver -eq 'named66') { $taskArguments += @('--input-config', $InputConfig) }
    $taskHasCameraSeed = $false
    foreach ($taskOption in $taskConfig.source_options.PSObject.Properties) {
        if ($taskOption.Value -is [bool]) {
            if ($taskOption.Value) { $taskArguments += $taskOption.Name }
            continue
        }
        $taskValue = [string]$taskOption.Value
        if ($taskOption.Name -eq '--camera-seed') {
            $taskHasCameraSeed = $true
            if ($Seed) { $taskValue = $Seed }
        }
        $taskArguments += @($taskOption.Name, $taskValue)
    }
    if ($Seed -and -not $taskHasCameraSeed) { $taskArguments += @('--camera-seed', $Seed) }
    if ($AcceptedSteps -gt 0) { $taskArguments += @('--accepted-steps', [string]$AcceptedSteps) }
    Write-Host "Solver source: $taskProjectRoot\calibration\solver_runtime"
    Write-Host "Results: $Output"
    & $taskPython @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Solver exited with $LASTEXITCODE. Preserved results: $Output" }
}
finally {
    $env:RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR = $taskSavedBenchmarkDiagnosticDir
    $env:RINKSPACE_BENCHMARK_TIMINGS_PATH = $taskSavedBenchmarkTimingsPath
    $env:RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE = $taskSavedStartupDiagnosticEvidence
    Pop-Location
}
