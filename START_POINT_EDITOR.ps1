[CmdletBinding()]
param(
    [string]$Dataset = '',
    [string]$Media = '',
    [string]$SourceMedia = '',
    [string]$Model = '',
    [ValidateRange(1, 65535)]
    [int]$Port = 48173,
    [switch]$NoOpen
)

# ASCII source keeps this launcher compatible with Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$projectRoot = $PSScriptRoot
$batchSuffix = 'events\ankara_2026\programmes\okada_mei_sp\source_datasets\frame_by_frame_review_20261013\first_shot_remaining_50fps\repair_20261014'
if ([string]::IsNullOrWhiteSpace($Dataset)) {
    $Dataset = Join-Path $projectRoot $batchSuffix
}
if ([string]::IsNullOrWhiteSpace($Media)) {
    $Media = 'D:\CodexData\RinkSpace_Data\events\ankara_2026\programmes\okada_mei_sp\results\frame_by_frame_review_20261013\first_shot_remaining_50fps\repair_20261014'
    if ([string]::IsNullOrWhiteSpace($SourceMedia)) {
        $SourceMedia = Split-Path -Parent $Media
    }
}
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $projectRoot 'events\ankara_2026\models\framewise_8a28_first_shot_20261014\repair_20261014'
}

function Resolve-EditorDirectory([string]$PathValue, [string]$Label) {
    if (-not (Test-Path -LiteralPath $PathValue -PathType Container)) {
        throw "$Label directory does not exist: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).ProviderPath
}

function Same-EditorPath([string]$Left, [string]$Right) {
    if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) {
        return $false
    }
    try {
        $leftFull = [IO.Path]::GetFullPath($Left).TrimEnd('\', '/')
        $rightFull = [IO.Path]::GetFullPath($Right).TrimEnd('\', '/')
        return [string]::Equals($leftFull, $rightFull, [StringComparison]::OrdinalIgnoreCase)
    } catch {
        return $false
    }
}

function Quote-EditorArgument([string]$Value) {
    # Windows command-line quoting: double backslashes before quotes/end quote.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

try {
    $datasetPath = Resolve-EditorDirectory $Dataset 'Dataset'
    $mediaPath = Resolve-EditorDirectory $Media 'Media'
    if ([string]::IsNullOrWhiteSpace($SourceMedia)) { $SourceMedia = $Media }
    $sourceMediaPath = Resolve-EditorDirectory $SourceMedia 'Source media'
    $modelPath = Resolve-EditorDirectory $Model 'Model'
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "Project Python is missing: $pythonPath"
    }
    & $pythonPath -B (Join-Path $projectRoot 'scripts/check_onedrive_paths.py') --root $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Project paths exceed the Windows sync limit. See the diagnostic above.'
    }
    foreach ($required in @(
        (Join-Path $datasetPath 'batch_manifest.json'),
        (Join-Path $sourceMediaPath 'camera_seeds.npz'),
        (Join-Path $modelPath 'scene.json'),
        (Join-Path $modelPath 'feature_registry.json'),
        (Join-Path $modelPath 'ad2_material_template.json')
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required editor input is missing: $required"
        }
    }

    $editorAddress = "http://127.0.0.1:$Port/"
    $runtimePath = Join-Path $mediaPath 'point_editor_runtime.json'
    function Get-MatchingEditor {
        try {
            if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) { return $false }
            $runtime = Get-Content -LiteralPath $runtimePath -Raw -Encoding UTF8 | ConvertFrom-Json
            if (-not (Same-EditorPath $runtime.dataset $datasetPath) -or
                -not (Same-EditorPath $runtime.media $mediaPath) -or
                -not (Same-EditorPath $runtime.model $modelPath)) { return $false }
            if ([string]$runtime.url -ne $editorAddress) { return $false }
            $runtimeSource = $runtime.source_media
            if ([string]::IsNullOrWhiteSpace($runtimeSource)) { $runtimeSource = $runtime.media }
            if (-not (Same-EditorPath $runtimeSource $sourceMediaPath)) { return $false }
            $health = Invoke-RestMethod -Uri ($editorAddress + 'api/health') -Method Get -TimeoutSec 2
            return ($health.status -eq 'ok' -and (Same-EditorPath $health.dataset $datasetPath))
        } catch {
            return $false
        }
    }

    if (Get-MatchingEditor) {
        Write-Output "Reusing point editor: $editorAddress"
        if (-not $NoOpen) { Start-Process -FilePath $editorAddress }
        exit 0
    }

    # Probe the requested loopback port without stopping any existing process.
    $probe = New-Object Net.Sockets.TcpListener -ArgumentList ([Net.IPAddress]::Parse('127.0.0.1')), $Port
    try {
        $probe.Start()
    } catch {
        throw "Port $Port is occupied by an unrelated or unverified service. No process was stopped. Choose -Port explicitly or close that service yourself."
    } finally {
        $probe.Stop()
    }

    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stdoutPath = Join-Path $mediaPath ("point_editor_start_$stamp.stdout.log")
    $stderrPath = Join-Path $mediaPath ("point_editor_start_$stamp.stderr.log")
    $argumentValues = @('-B', '-X', 'utf8', '-m', 'calibration.program_tracking.frame_point_editor',
        '--dataset', $datasetPath, '--media', $mediaPath, '--source-media', $sourceMediaPath, '--model', $modelPath, '--port', [string]$Port)
    $argumentText = ($argumentValues | ForEach-Object { Quote-EditorArgument $_ }) -join ' '
    $editorProcess = Start-Process -FilePath $pythonPath -ArgumentList $argumentText -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    $ready = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Get-MatchingEditor) { $ready = $true; break }
        $editorProcess.Refresh()
        if ($editorProcess.HasExited) {
            $details = ''
            if (Test-Path -LiteralPath $stderrPath) {
                $details = (Get-Content -LiteralPath $stderrPath -Tail 12 -Encoding UTF8) -join [Environment]::NewLine
            }
            throw "Point editor exited before becoming ready. Error log: $stderrPath`n$details"
        }
        Start-Sleep -Milliseconds 200
    }
    if (-not $ready) {
        throw "Point editor is not healthy yet; its process was left running. Check $stdoutPath and $stderrPath."
    }
    Write-Output "Point editor ready: $editorAddress"
    Write-Output "Logs: $stdoutPath ; $stderrPath"
    if (-not $NoOpen) { Start-Process -FilePath $editorAddress }
} catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
