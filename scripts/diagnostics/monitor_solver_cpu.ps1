param([Parameter(Mandatory=$true)][int]$SolverProcessId,
      [Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$taskLogicalCpus = [Environment]::ProcessorCount
$taskKnown = @{}
$taskPriorCpu = @{}
$taskPriorWall = [Diagnostics.Stopwatch]::GetTimestamp()
$taskRootProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $SolverProcessId"
if ($null -eq $taskRootProcess) { return }
$taskKnown[$SolverProcessId] = $taskRootProcess.CreationDate.ToUniversalTime().Ticks
$taskRootHandle = Get-Process -Id $SolverProcessId
$null = $taskRootHandle.Handle
$taskRootNativeStart = $taskRootHandle.StartTime.ToUniversalTime()
if ([Math]::Abs($taskRootNativeStart.Ticks - $taskKnown[$SolverProcessId]) -gt 9) {
    throw 'Solver process identity changed before its exit-time handle was retained.'
}
$taskStream = Join-Path $OutputDirectory 'process_tree_utilization.jsonl'
$taskLatest = Join-Path $OutputDirectory 'process_tree_utilization_latest.json'
while ($true) {
    $taskProcesses = @(Get-CimInstance Win32_Process)
    $taskById = @{}
    foreach ($taskItem in $taskProcesses) { $taskById[[int]$taskItem.ProcessId] = $taskItem }
    $taskChanged = $true
    while ($taskChanged) {
        $taskChanged = $false
        foreach ($taskItem in $taskProcesses) {
            $taskId = [int]$taskItem.ProcessId
            $taskParentId = [int]$taskItem.ParentProcessId
            $taskCreation = $taskItem.CreationDate.ToUniversalTime().Ticks
            $taskParentOwned = ($taskKnown.ContainsKey($taskParentId) -and $taskById.ContainsKey($taskParentId) -and
                $taskKnown[$taskParentId] -eq $taskById[$taskParentId].CreationDate.ToUniversalTime().Ticks)
            # ParentProcessId survives after the real parent exits. A reused
            # PID must not adopt that earlier parent's surviving children.
            $taskChildOfCurrentParent = ($taskParentOwned -and $taskId -ne $SolverProcessId -and
                $taskCreation -ge $taskById[$taskParentId].CreationDate.ToUniversalTime().Ticks)
            if ($taskChildOfCurrentParent -and (-not $taskKnown.ContainsKey($taskId) -or $taskKnown[$taskId] -ne $taskCreation)) {
                $taskKnown[$taskId] = $taskCreation
                $taskChanged = $true
            }
        }
    }
    $taskCurrent = @($taskProcesses | Where-Object {
        $taskKnown.ContainsKey([int]$_.ProcessId) -and
        $taskKnown[[int]$_.ProcessId] -eq $_.CreationDate.ToUniversalTime().Ticks
    })
    if ($taskCurrent.Count -eq 0) { break }
    $taskNow = [Diagnostics.Stopwatch]::GetTimestamp()
    $taskWall = ($taskNow - $taskPriorWall) / [Diagnostics.Stopwatch]::Frequency
    $taskCpuDelta = 0.0
    $taskRss = 0L
    $taskPrivateCommit = 0L
    $taskDetails = @()
    foreach ($taskItem in $taskCurrent) {
        $taskKey = [string]$taskItem.ProcessId + ':' + [string]$taskItem.CreationDate
        $taskCpu = ([double]$taskItem.KernelModeTime + [double]$taskItem.UserModeTime) / 10000000
        $taskDelta = 0.0
        if ($taskPriorCpu.ContainsKey($taskKey)) { $taskDelta = [Math]::Max(0, $taskCpu - $taskPriorCpu[$taskKey]) }
        $taskPriorCpu[$taskKey] = $taskCpu
        $taskCpuDelta += $taskDelta
        $taskRss += [long]$taskItem.WorkingSetSize
        $taskPrivateCommit += [long]$taskItem.PrivatePageCount
        $taskDetails += @{pid=[int]$taskItem.ProcessId; creation_time_utc=$taskItem.CreationDate.ToUniversalTime().ToString('o'); cpu_seconds_delta=$taskDelta; working_set_bytes=[long]$taskItem.WorkingSetSize; private_committed_bytes=[long]$taskItem.PrivatePageCount}
    }
    $taskRecord = @{timestamp_utc=[DateTime]::UtcNow.ToString('o'); logical_cpu_count=$taskLogicalCpus;
        interval_seconds=$taskWall; cpu_seconds_delta=$taskCpuDelta;
        cpu_percent_of_machine=100*$taskCpuDelta/($taskWall*$taskLogicalCpus);
        effective_busy_cores=$taskCpuDelta/$taskWall; process_count=$taskCurrent.Count;
        summed_working_set_bytes=$taskRss; summed_working_set_may_double_count_shared_pages=$true;
        summed_private_committed_bytes=$taskPrivateCommit; processes=$taskDetails}
    $taskJson = $taskRecord | ConvertTo-Json -Depth 5 -Compress
    [IO.File]::AppendAllText($taskStream, $taskJson + [Environment]::NewLine)
    [IO.File]::WriteAllText($taskLatest, $taskJson)
    $taskPriorWall = $taskNow
    Start-Sleep -Seconds 2
}
if ($taskRootHandle.HasExited) {
    $taskExitUtc = $taskRootHandle.ExitTime.ToUniversalTime()
    $taskExitRecord = @{schema='rinkspace.owned-solver-native-process-exit.v1';
        solver_process_id=$SolverProcessId; solver_creation_ticks_utc=$taskKnown[$SolverProcessId];
        native_start_time_utc=$taskRootNativeStart.ToString('o');
        native_exit_time_utc=$taskExitUtc.ToString('o');
        cold_start_to_native_process_exit_seconds=($taskExitUtc-$taskRootNativeStart).TotalSeconds;
        all_observed_owned_identities_absent_utc=[DateTime]::UtcNow.ToString('o');
        solver_exit_code=$taskRootHandle.ExitCode; known_owned_identity_count=$taskKnown.Count}
    [IO.File]::WriteAllText((Join-Path $OutputDirectory 'complete_process_exit.json'),
        ($taskExitRecord | ConvertTo-Json -Depth 3))
}
$taskRootHandle.Dispose()
