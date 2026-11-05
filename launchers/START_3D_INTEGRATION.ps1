[CmdletBinding()]
param(
    [ValidateSet('s01','jump','step','spin')][string]$Clip='',
    [string]$Variant='', [switch]$Fresh, [switch]$Wait, [switch]$Check
)
$ErrorActionPreference='Stop'
$programme=Join-Path (Split-Path -Parent $PSScriptRoot) 'events\ankara_2026\programmes\okada_mei_sp'
$configs=@{
    s01='viewer3d_fullbody_1102.json'
    jump='viewer3d_action_jump_s02_1103.json'
    step='viewer3d_action_step_s01_1103.json'
    spin='viewer3d_action_spin_s05_1103.json'
}
if ($Check) {
    foreach ($name in @('s01','jump','step','spin')) {
        $path=Join-Path $programme $configs[$name]
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing comparison config: $path" }
        $config=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Output ("{0}: {1}; default={2}; variants={3}" -f $name,$path,$config.default_variant,($config.comparison_variants -join ','))
    }
    exit 0
}
if (-not $Clip) {
    Write-Host '1  S01 preserved baseline / earlier integration candidate (22.12-44.96s)'
    Write-Host '2  Jump A/B (44.98-52.00s)'
    Write-Host '3  Steps A/B (25.00-35.00s)'
    Write-Host '4  Spin A/B (76.00-95.00s)'
    $choice=Read-Host 'Choose 1-4 (Enter = S01 baseline)'
    $Clip=switch ($choice) { '2' {'jump'} '3' {'step'} '4' {'spin'} default {'s01'} }
}
$path=Join-Path $programme $configs[$Clip]
if (-not (Test-Path -LiteralPath $path)) { throw "Missing comparison config: $path" }
$arguments=@{Config=$path;Fresh=$Fresh;Wait=$Wait}
if ($Variant) {$arguments.Variant=$Variant}
& (Join-Path $PSScriptRoot 'START_3D_RESULTS.ps1') @arguments
exit $LASTEXITCODE
