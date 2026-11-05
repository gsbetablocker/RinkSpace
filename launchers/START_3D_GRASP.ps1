[CmdletBinding()]
param([string]$Variant='support_midpoint',[switch]$Fresh,[switch]$Wait,[switch]$Check)
$ErrorActionPreference='Stop'
$configPath=Join-Path (Split-Path -Parent $PSScriptRoot) 'events/ankara_2026/programmes/okada_mei_sp/viewer3d_spin_grasp_1104.json'
if ($Check) {
 $config=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
 if (-not (Test-Path -LiteralPath $config.source_video)) { throw 'Source video missing' }
 foreach ($item in $config.variants) { if (-not (Test-Path -LiteralPath $item.result)) { throw ('Result missing: '+$item.result) }; Write-Output $item.id }
 Write-Output '76-95s, same native PTS; no window opened.'; exit 0
}
& (Join-Path $PSScriptRoot 'START_3D_RESULTS.ps1') -Config $configPath -Variant $Variant -Fresh:$Fresh -Wait:$Wait
exit $LASTEXITCODE
