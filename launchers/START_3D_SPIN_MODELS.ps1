[CmdletBinding()]
param([string]$Variant='exmr_prefoot',[switch]$Fresh,[switch]$Wait,[switch]$Check)
$ErrorActionPreference='Stop'
$configPath=Join-Path (Split-Path -Parent $PSScriptRoot) 'events\ankara_2026\programmes\okada_mei_sp\viewer3d_spin_models_1104.json'
if ($Check) {
    $config=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $config.source_video)) { throw 'Source video missing.' }
    foreach ($item in $config.variants) {
        $resultPath=if ([IO.Path]::IsPathRooted($item.result)) {$item.result} else {Join-Path $config.source_run $item.result}
        if (-not (Test-Path -LiteralPath $resultPath)) { throw "Result missing: $resultPath" }
        Write-Output ($item.id+': '+$resultPath)
    }
    Write-Output 'Source clock: native 50Hz, full continuous 76.00-95.00s. No GUI opened.'
    exit 0
}
& (Join-Path $PSScriptRoot 'START_3D_RESULTS.ps1') -Config $configPath -Variant $Variant -Fresh:$Fresh -Wait:$Wait
exit $LASTEXITCODE