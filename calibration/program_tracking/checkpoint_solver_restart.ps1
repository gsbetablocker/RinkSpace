param(
  [Parameter(Mandatory=$true)][string]$RunDirectory,
  [Parameter(Mandatory=$true)][int]$SolverPid,
  [string]$Reason = 'Resume the same objective after a verified implementation acceleration',
  [switch]$ValidateOnly,
  [switch]$UseSavedCheckpoint,
  [switch]$ReviewBeforeResume
)
$ErrorActionPreference = 'Stop'
$taskRun = (Resolve-Path -LiteralPath $RunDirectory).Path
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$taskDataRoot = [IO.Path]::GetFullPath('D:\CodexData')
$taskInProject = $taskRun.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
$taskInCodexData = $taskRun.StartsWith($taskDataRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
if (-not ($taskInProject -or $taskInCodexData)) { throw 'Run must be a project or CodexData subdirectory' }
$taskTrace = Join-Path $taskRun 'B_solver/solver_trace.json'
$taskReceipt = Join-Path $taskRun 'stopped_checkpoint.json'
if (Test-Path -LiteralPath $taskReceipt) { throw 'Checkpoint stop already recorded' }
function Assert-TaskSolver {
  $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$SolverPid"
  if (-not $taskProcess -or $taskProcess.CommandLine -notmatch 'run_joint_camera_body\.py' -or
      $taskProcess.CommandLine -notmatch [regex]::Escape((Split-Path -Leaf $taskRun))) {
    throw 'PID is not the expected joint camera/body driver'
  }
  return $taskProcess
}
$taskIdentity = Assert-TaskSolver
$taskCode = Get-Content -LiteralPath (Join-Path $taskRun 'code_dependencies.json') -Raw | ConvertFrom-Json
$taskRuntimePath = Join-Path $taskRun 'runtime_identity.json'
$taskConfigPath = Join-Path $taskRun 'run_configuration.json'
if ((Test-Path -LiteralPath $taskRuntimePath) -and (Test-Path -LiteralPath $taskConfigPath)) {
  $taskRuntime = Get-Content -LiteralPath $taskRuntimePath -Raw | ConvertFrom-Json
  $taskConfig = Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
  if ($taskRuntime.pid -ne $SolverPid -or $taskRuntime.executable -ne $taskIdentity.ExecutablePath -or
      [IO.Path]::GetFullPath($taskConfig.output) -ne $taskRun -or
      [Math]::Abs((([DateTimeOffset]$taskRuntime.started_at).UtcDateTime - $taskIdentity.CreationDate.ToUniversalTime()).TotalSeconds) -gt 15) {
    throw 'Runtime identity, process creation time, or exact output binding differs'
  }
  $taskNumericConfig = @{ iterations=[int]$taskConfig.iterations; scale=[double]$taskConfig.scale; body_sigma_px=[double]$taskConfig.body_sigma_px }
  if ($taskNumericConfig.iterations -lt 1 -or $taskNumericConfig.scale -le 0 -or $taskNumericConfig.body_sigma_px -le 0) {
    throw 'Runtime numerical configuration is invalid'
  }
  $taskConfigSource = 'Saved run_configuration.json and runtime_identity.json matched to live executable, PID and creation time'
} else {
  if ($taskCode.sha256.'three_frame_refit\program_tracking\run_joint_camera_body.py' -ne '72c9d8f02027a8032c123a6bcb3b81549f70ae5252be76811cb2e28c3e08ff29' -or
      $taskIdentity.CommandLine -match '--(iterations|scale|body-sigma-px)(\s|=)') {
    throw 'A driver without runtime identity requires the verified legacy defaults'
  }
  $taskNumericConfig = @{ iterations=80; scale=2.853381; body_sigma_px=8.0 }
  $taskConfigSource = 'Original legacy command line and archived driver argparse defaults'
}
$taskBefore = Get-Content -LiteralPath $taskTrace -Raw | ConvertFrom-Json
if ($null -ne $taskBefore.final_cost) { throw 'B already completed; do not stop this phase' }
$taskInitialCount = $taskBefore.steps.Count
if ($ValidateOnly) {
  [ordered]@{ validated_only=$true; pid=$SolverPid; directory=$taskRun; saved_steps=$taskInitialCount; run_configuration=$taskNumericConfig; configuration_source=$taskConfigSource } | ConvertTo-Json -Depth 5
  exit 0
}
if ($UseSavedCheckpoint) {
  if ($taskInitialCount -lt 1 -or $taskBefore.steps[-1].status -ne 'accepted') {
    throw 'No accepted saved checkpoint available'
  }
} else {
  Write-Output "Waiting for next accepted checkpoint after $taskInitialCount saved steps"
  $taskDeadline = (Get-Date).AddMinutes(15)
  do {
    Start-Sleep -Seconds 1
    $taskCurrent = Get-Content -LiteralPath $taskTrace -Raw | ConvertFrom-Json
    if ($null -ne $taskCurrent.final_cost) { throw 'B completed while waiting; no stop performed' }
    if ((Get-Date) -gt $taskDeadline) { throw 'No fresh accepted checkpoint; no stop performed' }
  } until ($taskCurrent.steps.Count -gt $taskInitialCount -and $taskCurrent.steps[-1].status -eq 'accepted')
}
$taskFreshIdentity = Assert-TaskSolver
if ($taskFreshIdentity.CreationDate -ne $taskIdentity.CreationDate) { throw 'Solver PID identity changed while waiting' }
$taskChildren = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $SolverPid })
if (@($taskChildren | Where-Object { $_.Name -notmatch '^python(w)?\.exe$' }).Count) {
  throw 'Unexpected child process; no stop performed'
}
$taskProcessIds = @($SolverPid) + @($taskChildren | ForEach-Object { [int]$_.ProcessId })
Stop-Process -Id $SolverPid -Force
foreach ($taskChild in $taskChildren) {
  Stop-Process -Id $taskChild.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 250
if (@(Get-Process -Id $taskProcessIds -ErrorAction SilentlyContinue).Count) { throw 'Solver tree did not exit' }
$taskFiles = @{}
foreach ($taskName in @('B_solver/accepted_states.npz','B_solver/solver_trace.json',
  'B_initial_states.npz','original_prior_states.npz','original_graph_binding.json',
  'body_inputs.json','sample_inventory.json','source_spin_intervals.json',
  'visibility_attachment.json','visibility_weighting.json','code_dependencies.json')) {
  $taskFiles[$taskName] = (Get-FileHash -LiteralPath (Join-Path $taskRun $taskName) -Algorithm SHA256).Hash.ToLowerInvariant()
}
foreach ($taskName in @('runtime_identity.json','run_configuration.json','resume_field_binding.json')) {
  $taskOptional = Join-Path $taskRun $taskName
  if (Test-Path -LiteralPath $taskOptional) {
    $taskFiles[$taskName] = (Get-FileHash -LiteralPath $taskOptional -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$taskFinal = Get-Content -LiteralPath $taskTrace -Raw | ConvertFrom-Json
$taskRecord = [ordered]@{
  schema='rinkspace.stopped-solver-checkpoint.v1'
  stopped_at=(Get-Date).ToString('o')
  reason=$Reason
  process_tree_pids=$taskProcessIds
  source_driver_command=$taskIdentity.CommandLine
  accepted_steps=$taskFinal.steps.Count
  accepted_cost=$taskFinal.steps[-1].cost
  files_sha256=$taskFiles
  run_configuration=$taskNumericConfig
  configuration_source=$taskConfigSource
  source_directory_preserved=$true
  resume_required=(-not $ReviewBeforeResume)
  geometry_review_before_resume=[bool]$ReviewBeforeResume
}
$taskRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReceipt -Encoding utf8
$taskRecord | ConvertTo-Json -Depth 4
