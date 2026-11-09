[CmdletBinding()]
param([string]$Variant='programme_repair_1108', [switch]$Fresh, [switch]$Wait)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$reviewConfig=Join-Path $projectRoot 'events/ankara_2026/programmes/okada_mei_sp/viewer3d_programme_repair_1108.json'
& (Join-Path $PSScriptRoot 'START_3D_RESULTS.ps1') -Config $reviewConfig -Variant $Variant -Fresh:$Fresh -Wait:$Wait
