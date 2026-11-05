[CmdletBinding()]
param([string]$Variant='support_new',[switch]$Fresh,[switch]$Wait,[switch]$Check)
$ErrorActionPreference='Stop'
$configPath=Join-Path (Split-Path -Parent $PSScriptRoot) 'events\ankara_2026\programmes\okada_mei_sp\viewer3d_spin_support_1104.json'
if ($Check) {
    $config=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $config.source_video)) { throw 'Source video missing.' }
    if (-not (Test-Path -LiteralPath $config.support_observations)) { throw 'Source region observations missing.' }
    foreach ($item in $config.variants) {
        $resultPath=Join-Path $config.source_run $item.result
        if (-not (Test-Path -LiteralPath $resultPath)) { throw "Result missing: $resultPath" }
        Write-Output ($item.id+': '+$resultPath)
    }
    Write-Output 'Source clock: native 50Hz, original 87.50-92.50s; core 88.00-92.00s. No GUI opened.'
    exit 0
}
& (Join-Path $PSScriptRoot 'START_3D_RESULTS.ps1') -Config $configPath -Variant $Variant -Fresh:$Fresh -Wait:$Wait
exit $LASTEXITCODE
