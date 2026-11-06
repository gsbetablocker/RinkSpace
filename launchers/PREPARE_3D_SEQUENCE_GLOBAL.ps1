[CmdletBinding()]
param(
    [ValidateSet('commands','check','register','prepare','static','all')][string]$Action='all',
    [ValidateSet('global','blade_image')][string]$Family='global',
    [string]$Stage='',
    [string]$ParentStage='',
    [string]$ParentParameter='',
    [string]$Parameter='',
    [switch]$IncludeLegacy
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH=(Join-Path $projectRoot 'src')+';'+$projectRoot
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$basePython=Join-Path $projectRoot '.venv\Scripts\python.exe'
$dataRoot=(& $basePython -B -X utf8 -c 'import sys; from rinkspace.storage import configured_data_root; print(configured_data_root(sys.argv[1]))' $projectRoot | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $dataRoot) { throw 'Cannot resolve RinkSpace Data directory.' }
$python=Join-Path $dataRoot 'runtime_tools\vision_cuda\Scripts\python.exe'
$values=@('-B','-X','utf8','-m','calibration.program_tracking.body_tracking.sequence_delivery',$Action,'--project',$projectRoot,'--data-root',$dataRoot,'--family',$Family)
if ($Stage) { $values+=@('--stage',$Stage) }
if ($ParentStage) { $values+=@('--parent-stage',$ParentStage) }
if ($ParentParameter) { $values+=@('--parent-parameter',$ParentParameter) }
if ($Parameter) { $values+=@('--parameter',$Parameter) }
if ($Action -in @('register','all')) { $values+='--publish-config' }
if ($IncludeLegacy) { $values+='--include-legacy' }
& $python @values
if ($LASTEXITCODE -ne 0) { throw 'Sequence delivery preparation failed; no GUI was opened.' }
