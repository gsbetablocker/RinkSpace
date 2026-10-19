param(
    [ValidateSet('baseline','candidate','both')][string]$Variant = 'candidate',
    [ValidatePattern('^candidate(?:_[A-Za-z0-9_-]+)?$')][string]$CandidateName = 'candidate',
    [string]$OutputRoot = 'D:\CodexData\RinkSpace_Data\events\ankara_2026\calibration\runs\solver_acceleration_20261003\complete_cycle_comparison_20261010',
    [switch]$DryRun,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$taskProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskPython = Join-Path $taskProject '.venv/Scripts/python.exe'
$taskShell = (Get-Process -Id $PID).Path
$taskConfigPath = Join-Path $taskProject 'events/ankara_2026/programmes/okada_mei_sp/full_source_complete_cycle_comparison_20261010.json'
$taskRoot = [IO.Path]::GetFullPath($OutputRoot)
$taskConfig = Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
$taskDataRoot = [IO.Path]::GetFullPath([string]$taskConfig.resource_root)
$taskOldFD = Join-Path $taskDataRoot 'okada_full9_review82_speed_candidate_20261010_101450_850.bootstrap/execution_sources/completion_driven_absolute_fd_runtime.py'
$taskOldSHA = '761fb8a87b612938a57533876c50f6e860106d48bb4b336a579450529262c82c'
$taskMonitorScript = Join-Path $taskProject 'scripts/diagnostics/monitor_solver_cpu.ps1'
if (-not $taskRoot.StartsWith($taskDataRoot.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Comparison output must stay in the existing Data resource root.' }
if ((Get-FileHash -LiteralPath $taskOldFD -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskOldSHA) { throw 'Measured baseline FD snapshot SHA changed.' }
if ((Get-FileHash -LiteralPath $taskConfig.source_options.'--camera-seed' -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskConfig.continuation.checkpoint_file_sha256) { throw 'Shared complete checkpoint bytes changed.' }
if ((Get-FileHash -LiteralPath $taskConfig.source_options.'--truth-restoration-state' -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskConfig.continuation.truth_restoration_state_sha256) { throw 'Shared same-q truth state bytes changed.' }
foreach ($taskForbidden in @('--verify-source-blocks-from','--initial-damping','--reuse-complete-linearization','--reuse-source-blocks-from','--reuse-wide-linearization','--verified-source-prefix-report','--evaluate-only','--reinitialize-invalid-mounts','--reinitialize-invalid-pivot-positions','--absolute-camera-initializer-source')) {
    if ($taskConfig.source_options.PSObject.Properties[$taskForbidden]) { throw "Complete-cycle comparison cannot include $taskForbidden" }
}
if ([string]$taskConfig.source_options.'--accepted-steps' -ne '1') { throw 'The authorized A/B uses exactly one accepted step per variant.' }
function Quote-Argument([string]$Value) {
    $taskEscaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $taskEscaped = [regex]::Replace($taskEscaped, '(\\+)$', '$1$1')
    return '"' + $taskEscaped + '"'
}
$taskVariants = if ($Variant -eq 'both') { @('baseline','candidate') } else { @($Variant) }
$taskSaved = @{}
foreach ($taskName in @('PYTHONDONTWRITEBYTECODE','RINKSPACE_SOLVER_DATA_ROOT','RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR','RINKSPACE_BENCHMARK_TIMINGS_PATH','RINKSPACE_COMPLETE_CYCLE_VARIANT','RINKSPACE_BASELINE_FD_SOURCE','RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE')) { $taskSaved[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process') }
try {
    foreach ($taskVariant in $taskVariants) {
        $taskOut = Join-Path $taskRoot $(if ($taskVariant -eq 'candidate') { $CandidateName } else { 'baseline' })
        $taskBoot = $taskOut + '.bootstrap'
        if ((Test-Path -LiteralPath $taskOut) -or (Test-Path -LiteralPath $taskBoot)) { throw "Preserve existing variant evidence; output already exists: $taskOut" }
        $taskVariantConfig = $taskConfig | ConvertTo-Json -Depth 12 | ConvertFrom-Json
        $taskVariantConfig.source_options.'--geometry-fd-workers' = if ($taskVariant -eq 'candidate') { '20' } else { '8' }
        $taskArgs = @('-B','-u','-m','calibration.solver_runtime','named66','--output',$taskOut)
        foreach ($taskOption in $taskVariantConfig.source_options.PSObject.Properties) {
            if ($taskOption.Value -is [bool]) { if ($taskOption.Value) { $taskArgs += $taskOption.Name } }
            else { $taskArgs += @($taskOption.Name,[Convert]::ToString($taskOption.Value,[Globalization.CultureInfo]::InvariantCulture)) }
        }
        $taskLine = ($taskArgs | ForEach-Object { Quote-Argument $_ }) -join ' '
        $taskPlan = [ordered]@{schema='rinkspace.single-complete-cycle-AB-hidden-launch.v1';variant=$taskVariant;output=$taskOut;bootstrap=$taskBoot;common_input_config=$taskConfigPath;common_input_config_sha256=(Get-FileHash -LiteralPath $taskConfigPath -Algorithm SHA256).Hash.ToLowerInvariant();initial_q_sha256=$taskConfig.continuation.state.q_sha256;checkpoint_sha256=$taskConfig.continuation.checkpoint_file_sha256;seed_checkpoint_damping=$taskConfig.continuation.state.damping;seed_checkpoint_next_fraction=$taskConfig.continuation.state.next_line_search_fraction;actual_driver_start_read_from=(Join-Path $taskBoot 'complete_cycle_comparison.json');actual_driver_damping_inheritance_not_assumed=$true;candidate_start_must_equal_actual_baseline_start=$true;geometry_fd_workers=$taskVariantConfig.source_options.'--geometry-fd-workers';baseline_FD_source=$taskOldFD;baseline_FD_source_sha256=$taskOldSHA;arguments=$taskArgs;no_cached_J_H_or_prefix_imported=$true}
        if ($DryRun) { $taskPlan | ConvertTo-Json -Depth 5; continue }
        [IO.Directory]::CreateDirectory($taskRoot) | Out-Null
        $env:PYTHONDONTWRITEBYTECODE = '1'; $env:RINKSPACE_SOLVER_DATA_ROOT = $taskDataRoot
        $env:RINKSPACE_COMPLETE_CYCLE_VARIANT = $taskVariant; $env:RINKSPACE_BASELINE_FD_SOURCE = $taskOldFD
        $env:RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE = if ($taskVariant -eq 'candidate') { Join-Path $taskRoot 'baseline' } else { $null }
        $taskPlan.startup_diagnostic_evidence = $env:RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE
        # The real formal entry installs every hook and compiles main, then
        # argparse --help exits before prepare builds any source graph.
        $taskPreflight = Join-Path $taskRoot ('preflight_' + $taskVariant)
        [IO.Directory]::CreateDirectory($taskPreflight) | Out-Null
        $env:RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR = $taskPreflight
        $env:RINKSPACE_BENCHMARK_TIMINGS_PATH = Join-Path $taskPreflight 'benchmark_timings_latest.json'
        & $taskPython -B -u -m calibration.solver_runtime named66 --output $taskOut --help 1> (Join-Path $taskPreflight 'stdout.log') 2> (Join-Path $taskPreflight 'stderr.log')
        if ($LASTEXITCODE -ne 0) { throw "Actual $taskVariant entry/AST preflight failed; see $taskPreflight" }
        $taskReceipt = Get-Content -LiteralPath (Join-Path $taskPreflight 'execution_acceleration_receipt.json') -Raw | ConvertFrom-Json
        foreach ($taskHelper in $taskReceipt.execution_helpers) {
            if ((Get-FileHash -LiteralPath $taskHelper.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskHelper.sha256) { throw "Helper changed after installation: $($taskHelper.path)" }
        }
        $taskPlan.static_real_entry_installation_passed = $true
        $taskPlan.preflight_receipt = Join-Path $taskPreflight 'execution_acceleration_receipt.json'
        if ($PrepareOnly) { $taskPlan | ConvertTo-Json -Depth 5; continue }
        if ($taskVariant -eq 'candidate') {
            $taskBaselineBoot = Join-Path $taskRoot 'baseline.bootstrap'
            $taskPrior = Get-Content -LiteralPath (Join-Path $taskBaselineBoot 'launch.json') -Raw | ConvertFrom-Json
            foreach ($taskPair in @(@($taskPrior.solver_process_id,$taskPrior.solver_creation_ticks_utc),@($taskPrior.monitor_process_id,$taskPrior.monitor_creation_ticks_utc))) {
                $taskExisting = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskPair[0])"
                if ($taskExisting -and $taskExisting.CreationDate.ToUniversalTime().Ticks -eq [int64]$taskPair[1]) { throw 'Baseline owned process/monitor is still alive; candidate must run after its full tree exits.' }
            }
            if (-not (Test-Path -LiteralPath (Join-Path $taskBaselineBoot 'complete_cycle_comparison.json'))) { throw 'Baseline has no complete accepted-cycle result; candidate not started.' }
        }
        $taskOldCore = Get-CimInstance Win32_Process -Filter 'ProcessId = 26732'
        if ($taskOldCore -and $taskOldCore.CreationDate.ToUniversalTime().ToString('o') -eq '2026-10-10T02:14:50.9367780Z') { throw 'Previous production graph is still alive; no overlapping full graph is allowed.' }
        [IO.Directory]::CreateDirectory($taskBoot) | Out-Null
        $taskVariantConfig | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $taskBoot 'input_config.json') -Encoding utf8
        $env:RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR = $taskBoot
        $env:RINKSPACE_BENCHMARK_TIMINGS_PATH = Join-Path $taskBoot 'benchmark_timings_latest.json'
        $taskSolver = Start-Process -FilePath $taskPython -ArgumentList $taskLine -WorkingDirectory $taskProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskBoot 'stdout.log') -RedirectStandardError (Join-Path $taskBoot 'stderr.log') -PassThru
        $taskIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskSolver.Id)"
        if (-not $taskIdentity) { throw 'Solver exited during launch; inspect preserved logs.' }
        $taskPlan.started_utc = [DateTime]::UtcNow.ToString('o'); $taskPlan.solver_process_id = $taskSolver.Id
        $taskPlan.solver_creation_time_utc = $taskIdentity.CreationDate.ToUniversalTime().ToString('o'); $taskPlan.solver_creation_ticks_utc = $taskIdentity.CreationDate.ToUniversalTime().Ticks
        $taskPlan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBoot 'launch.json') -Encoding utf8
        $taskMonitorLine = (@('-NoProfile','-ExecutionPolicy','Bypass','-File',$taskMonitorScript,'-SolverProcessId',[string]$taskSolver.Id,'-OutputDirectory',$taskBoot) | ForEach-Object { Quote-Argument $_ }) -join ' '
        $taskMonitor = Start-Process -FilePath $taskShell -ArgumentList $taskMonitorLine -WorkingDirectory $taskProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskBoot 'monitor_stdout.log') -RedirectStandardError (Join-Path $taskBoot 'monitor_stderr.log') -PassThru
        $taskMonitorIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskMonitor.Id)"
        if (-not $taskMonitorIdentity) { throw 'Monitor exited during launch; solver launch identity remains recorded.' }
        $taskPlan.monitor_process_id = $taskMonitor.Id; $taskPlan.monitor_creation_time_utc = $taskMonitorIdentity.CreationDate.ToUniversalTime().ToString('o'); $taskPlan.monitor_creation_ticks_utc = $taskMonitorIdentity.CreationDate.ToUniversalTime().Ticks
        $taskPlan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBoot 'launch.json') -Encoding utf8
        $taskPlan | ConvertTo-Json -Depth 5
        if ($Variant -eq 'both') {
            $taskSolver.WaitForExit(); $taskMonitor.WaitForExit()
            if ($taskSolver.ExitCode -ne 0) { throw "Baseline/variant failed with exit $($taskSolver.ExitCode); next variant not started." }
        }
    }
} finally {
    foreach ($taskName in $taskSaved.Keys) { [Environment]::SetEnvironmentVariable($taskName, $taskSaved[$taskName], 'Process') }
}
