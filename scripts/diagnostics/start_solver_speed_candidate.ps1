param(
    [string]$InputConfig = '',
    [string]$Output = '',
    [switch]$DryRun,
    [switch]$OneCompleteCycle,
    [switch]$CapturePriorityDomain,
    [switch]$ReuseAuthenticatedCompleteNormal,
    [switch]$CaptureSOCWitnesses,
    [string]$PivotRetractionBaselineRun = ''
)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $InputConfig) { $InputConfig = Join-Path $taskProjectRoot 'events/ankara_2026/programmes/okada_mei_sp/full_source_speed_candidate_20261010.json' }
$taskConfigPath = [IO.Path]::GetFullPath($InputConfig, $taskProjectRoot)
$taskConfig = Get-Content -LiteralPath $taskConfigPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskConfig.schema -ne 'rinkspace.reusable-solver-inputs.v1' -or $taskConfig.driver -ne 'named66') { throw 'Expected the complete named66 reusable input configuration.' }
$taskResourceRoot = [IO.Path]::GetFullPath([string]$taskConfig.resource_root)
if (-not $Output) { $Output = Join-Path $taskResourceRoot ('okada_full9_review82_speed_candidate_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')) }
$taskOutput = [IO.Path]::GetFullPath($Output, $taskProjectRoot)
$taskBoot = $taskOutput + '.bootstrap'
if (-not $taskOutput.StartsWith($taskResourceRoot.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'New output must remain inside the configured Data resource root.' }
if ((Test-Path -LiteralPath $taskOutput) -or (Test-Path -LiteralPath $taskBoot)) { throw 'Use a new output and bootstrap directory; existing evidence is preserved.' }
$taskPython = Join-Path $taskProjectRoot '.venv/Scripts/python.exe'
$taskPowerShell = (Get-Process -Id $PID).Path
$taskMonitorScript = Join-Path $taskProjectRoot 'scripts/diagnostics/monitor_solver_cpu.ps1'
foreach ($taskRequired in @($taskPython, $taskMonitorScript, $taskConfig.source_options.'--camera-seed', $taskConfig.source_options.'--truth-restoration-state')) {
    if (-not (Test-Path -LiteralPath $taskRequired -PathType Leaf)) { throw "Missing required owned input: $taskRequired" }
}
$taskCache = [string]$taskConfig.source_options.'--verify-source-blocks-from'
if ($taskCache -and -not (Test-Path -LiteralPath (Join-Path $taskCache 'strict_complete_source_replay_inventory.json') -PathType Leaf)) { throw 'Explicit same-state fresh-J verification requires the sealed complete cache.' }
foreach ($taskForbidden in @('--reuse-complete-linearization','--reuse-source-blocks-from','--verified-source-prefix-report','--evaluate-only','--accepted-steps')) {
    if ($taskConfig.source_options.PSObject.Properties[$taskForbidden]) {
        if ($taskForbidden -eq '--reuse-complete-linearization' -and $ReuseAuthenticatedCompleteNormal) { continue }
        throw "Fresh complete continuation cannot include $taskForbidden"
    }
}
if ($ReuseAuthenticatedCompleteNormal) {
    $taskSealedNormal = [string]$taskConfig.source_options.'--reuse-complete-linearization'
    if (-not $taskSealedNormal) { throw 'Explicit sealed-normal continuation requires --reuse-complete-linearization.' }
    foreach ($taskSealFile in @('actual_complete_cache_seal.json','strict_complete_source_replay_inventory.json','actual_initial_frozen_profile_arrays.npz','first_complete_source_normal_numeric.npz')) {
        if (-not (Test-Path -LiteralPath (Join-Path $taskSealedNormal $taskSealFile) -PathType Leaf)) { throw "Missing sealed complete-normal input: $taskSealFile" }
    }
    if ($CapturePriorityDomain) { throw 'Complete-normal continuation must execute the original solver, not a domain capture.' }
}
function ConvertTo-WindowsArgument([string]$Value) {
    # CommandLineToArgvW-compatible quoting, including quotes and a final slash.
    $taskEscaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $taskEscaped = [regex]::Replace($taskEscaped, '(\\+)$', '$1$1')
    return '"' + $taskEscaped + '"'
}
$taskArguments = @('-B','-u','-m','calibration.solver_runtime','named66','--output',$taskOutput)
if ($CapturePriorityDomain) {
    if ($OneCompleteCycle) { throw 'Domain capture ends before solving; do not combine it with a complete-cycle validation.' }
    $taskCaptureEntry = Join-Path $taskProjectRoot 'scripts/diagnostics/capture_priority_domain_current_state.py'
    if (-not (Test-Path -LiteralPath $taskCaptureEntry -PathType Leaf)) { throw 'Missing diagnostic domain-capture entry.' }
    $taskArguments = @('-B','-u',$taskCaptureEntry,'named66','--output',$taskOutput)
}
if ($CaptureSOCWitnesses) {
    if ($CapturePriorityDomain -or $OneCompleteCycle) { throw 'SOC witness capture has its own diagnostic exit; do not combine stop modes.' }
    $taskCaptureEntry = Join-Path $taskProjectRoot 'scripts/diagnostics/capture_current_complete_SOC_face_response.py'
    if (-not (Test-Path -LiteralPath $taskCaptureEntry -PathType Leaf)) { throw 'Missing real production SOC capture entry.' }
    $taskArguments = @('-B','-u',$taskCaptureEntry,'named66','--output',$taskOutput)
}
if ($PivotRetractionBaselineRun) {
    if ($CaptureSOCWitnesses -or $CapturePriorityDomain -or $OneCompleteCycle) { throw 'Retraction comparison continues the original solver; do not combine diagnostic stop modes.' }
    $taskRetractionEntry = Join-Path $taskProjectRoot 'scripts/diagnostics/run_saved_pivot_trial_comparison.py'
    $taskArguments = @('-B','-u',$taskRetractionEntry,'--retraction-baseline-run',[IO.Path]::GetFullPath($PivotRetractionBaselineRun),'named66','--output',$taskOutput)
}
foreach ($taskOption in $taskConfig.source_options.PSObject.Properties) {
    if ($taskOption.Value -is [bool]) {
        if ($taskOption.Value) { $taskArguments += $taskOption.Name }
    } else {
        $taskArguments += @($taskOption.Name, [Convert]::ToString($taskOption.Value, [Globalization.CultureInfo]::InvariantCulture))
    }
}
if ($OneCompleteCycle) { $taskArguments += @('--accepted-steps','1') }
$taskArgumentLine = ($taskArguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
$taskPlan = [ordered]@{schema='rinkspace.hidden-complete-source-launch.v1';config=$taskConfigPath;output=$taskOutput;bootstrap=$taskBoot;python=$taskPython;arguments=$taskArguments;command_line=(ConvertTo-WindowsArgument $taskPython)+' '+$taskArgumentLine;accepted_step_limit_added=[bool]$OneCompleteCycle;one_complete_cycle_validation=[bool]$OneCompleteCycle;runtime_limit_added=$false;fresh_same_q_verification=$taskCache;initial_q_sha256=$taskConfig.continuation.state.q_sha256}
$taskPlan.capture_current_priority_domain_only = [bool]$CapturePriorityDomain
$taskPlan.capture_two_actual_production_SOC_witnesses_only = [bool]$CaptureSOCWitnesses
$taskPlan.saved_full_direction_pivot_retraction_comparison = $PivotRetractionBaselineRun
$taskPlan.authenticated_complete_normal_replay = [bool]$ReuseAuthenticatedCompleteNormal
if ($ReuseAuthenticatedCompleteNormal) {
    $taskPlan.complete_normal_cache = $taskSealedNormal
    $taskPlan.original_source_q_profile_implementation_and_full_seal_gates_required = $true
}
if ($DryRun) { $taskPlan | ConvertTo-Json -Depth 5; return }
[IO.Directory]::CreateDirectory($taskBoot) | Out-Null
Copy-Item -LiteralPath $taskConfigPath -Destination (Join-Path $taskBoot 'input_config.json')
$taskPlan.config_sha256 = (Get-FileHash -LiteralPath $taskConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()
$taskSavedEnvironment = @{}
foreach ($taskName in @('PYTHONDONTWRITEBYTECODE','RINKSPACE_SOLVER_DATA_ROOT','RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR','RINKSPACE_BENCHMARK_TIMINGS_PATH','RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE','RINKSPACE_COMPLETE_CYCLE_VARIANT','RINKSPACE_OBSERVE_COMPLETE_FIRST_CYCLE')) { $taskSavedEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process') }
try {
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:RINKSPACE_SOLVER_DATA_ROOT = $taskResourceRoot
    $env:RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR = $taskBoot
    $env:RINKSPACE_BENCHMARK_TIMINGS_PATH = Join-Path $taskBoot 'benchmark_timings_latest.json'
    $env:RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE = [string]$taskConfig.continuation.startup_diagnostic_evidence
    $env:RINKSPACE_COMPLETE_CYCLE_VARIANT = $null
    $env:RINKSPACE_OBSERVE_COMPLETE_FIRST_CYCLE = '1'
    $taskSolver = Start-Process -FilePath $taskPython -ArgumentList $taskArgumentLine -WorkingDirectory $taskProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskBoot 'stdout.log') -RedirectStandardError (Join-Path $taskBoot 'stderr.log') -PassThru
    $taskSolverIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskSolver.Id)"
    if ($null -eq $taskSolverIdentity) { throw "Solver PID $($taskSolver.Id) exited during launch; inspect preserved bootstrap logs." }
    $taskPlan.started_utc = [DateTime]::UtcNow.ToString('o')
    $taskPlan.solver_process_id = $taskSolver.Id
    $taskPlan.solver_creation_time_utc = $taskSolverIdentity.CreationDate.ToUniversalTime().ToString('o')
    $taskPlan.solver_creation_ticks_utc = $taskSolverIdentity.CreationDate.ToUniversalTime().Ticks
    $taskPlan.creation_identity_required_for_any_future_control = $true
    $taskPlan.monitor_source_sha256 = (Get-FileHash -LiteralPath $taskMonitorScript -Algorithm SHA256).Hash.ToLowerInvariant()
    $taskPlan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBoot 'launch.json') -Encoding utf8
    $taskMonitorArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$taskMonitorScript,'-SolverProcessId',[string]$taskSolver.Id,'-OutputDirectory',$taskBoot)
    $taskMonitorLine = ($taskMonitorArgs | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
    $taskMonitor = Start-Process -FilePath $taskPowerShell -ArgumentList $taskMonitorLine -WorkingDirectory $taskProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskBoot 'monitor_stdout.log') -RedirectStandardError (Join-Path $taskBoot 'monitor_stderr.log') -PassThru
    $taskMonitorIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskMonitor.Id)"
    if ($null -eq $taskMonitorIdentity) { throw "Owned monitor PID $($taskMonitor.Id) exited during launch; solver PID $($taskSolver.Id) remains separately recorded." }
    $taskPlan.monitor_process_id = $taskMonitor.Id
    $taskPlan.monitor_creation_time_utc = $taskMonitorIdentity.CreationDate.ToUniversalTime().ToString('o')
    $taskPlan.monitor_creation_ticks_utc = $taskMonitorIdentity.CreationDate.ToUniversalTime().Ticks
    $taskPlan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBoot 'launch.json') -Encoding utf8
    $taskPlan | ConvertTo-Json -Depth 5
} finally {
    foreach ($taskName in $taskSavedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($taskName, $taskSavedEnvironment[$taskName], 'Process') }
}
