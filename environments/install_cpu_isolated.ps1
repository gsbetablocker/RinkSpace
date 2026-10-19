<#
Create a new CPU-only environment without touching the project's live .venv.
Source staging, wheels, pip cache and reports stay under the specified Data root.
Existing target or staging directories cause a failure; neither is overwritten.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BasePython,
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$EnvironmentDirectory,
    [string]$RequirementsFile = (Join-Path $PSScriptRoot 'requirements-cpu-validated-windows-lock.txt'),
    [string]$OfflineWheelhouse
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\', '/')
$dataDirectory = (Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\', '/')
$environmentPath = [IO.Path]::GetFullPath($EnvironmentDirectory).TrimEnd('\', '/')
$baseInterpreter = (Resolve-Path -LiteralPath $BasePython).Path
$requirementsPath = (Resolve-Path -LiteralPath $RequirementsFile).Path
$separator = [IO.Path]::DirectorySeparatorChar
if (-not $environmentPath.StartsWith($dataDirectory + $separator, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'EnvironmentDirectory must be strictly inside DataRoot.'
}
if ($dataDirectory.Equals($projectRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $dataDirectory.StartsWith($projectRoot + $separator, [StringComparison]::OrdinalIgnoreCase) -or
    $dataDirectory -match '(?i)[\\/]OneDrive(?:[\\/]|$)') {
    throw 'DataRoot must stay outside the OneDrive project.'
}
if (Test-Path -LiteralPath $environmentPath) {
    throw 'The selected environment already exists. Select a new path; it will not be overwritten.'
}
$stagingPath = $environmentPath + '-build'
if (Test-Path -LiteralPath $stagingPath) {
    throw 'The selected build directory already exists. Select a new environment path.'
}
& $baseInterpreter -B -c 'import struct,sys; assert sys.platform == "win32" and sys.version_info[:2] == (3,11) and struct.calcsize("P") == 8, "Use Windows AMD64 Python 3.11"'
if ($LASTEXITCODE -ne 0) { throw 'Base interpreter validation failed.' }
New-Item -ItemType Directory -Path $stagingPath | Out-Null
$wheelDirectory = Join-Path $stagingPath 'wheelhouse'
$sourceDirectory = Join-Path $stagingPath 'source'
New-Item -ItemType Directory -Path $wheelDirectory | Out-Null
New-Item -ItemType Directory -Path (Join-Path $sourceDirectory 'src') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'pyproject.toml') -Destination $sourceDirectory
Copy-Item -LiteralPath (Join-Path $projectRoot 'src/rinkspace') -Destination (Join-Path $sourceDirectory 'src') -Recurse

# Subprocess settings apply only to this independent install session.
$previousCache = $env:PIP_CACHE_DIR
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PIP_CACHE_DIR = Join-Path $stagingPath 'pip_cache'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $baseInterpreter -B -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
    $environmentPython = Join-Path $environmentPath 'Scripts/python.exe'
    $pipArguments = @('-m', 'pip', '--isolated', '--disable-pip-version-check', '--cache-dir', $env:PIP_CACHE_DIR)
    if ($OfflineWheelhouse) {
        $inputWheelDirectory = (Resolve-Path -LiteralPath $OfflineWheelhouse).Path
        $downloadArguments = @('download', '--no-index', '--find-links', $inputWheelDirectory)
    } else {
        $downloadArguments = @('download', '--index-url', 'https://pypi.org/simple', '--retries', '0', '--timeout', '60')
    }
    & $environmentPython @pipArguments @downloadArguments --only-binary=:all: --dest $wheelDirectory -r $requirementsPath *> (Join-Path $stagingPath 'download.log')
    if ($LASTEXITCODE -ne 0) { throw "CPU wheel download failed. See $stagingPath/download.log" }
    & $environmentPython @pipArguments install --no-index --find-links $wheelDirectory --only-binary=:all: -r $requirementsPath *> (Join-Path $stagingPath 'install.log')
    if ($LASTEXITCODE -ne 0) { throw "CPU installation failed. See $stagingPath/install.log" }
    # Build only the small project's Python wheel from a private Data copy.
    & $environmentPython @pipArguments wheel --no-deps --no-build-isolation --wheel-dir $wheelDirectory $sourceDirectory *> (Join-Path $stagingPath 'project-wheel.log')
    if ($LASTEXITCODE -ne 0) { throw "Project wheel build failed. See $stagingPath/project-wheel.log" }
    $projectWheels = @(Get-ChildItem -LiteralPath $wheelDirectory -File -Filter 'rinkspace-*.whl')
    if ($projectWheels.Count -ne 1) { throw 'Expected exactly one newly built RinkSpace wheel.' }
    & $environmentPython @pipArguments install --no-index --no-deps $projectWheels[0].FullName *> (Join-Path $stagingPath 'project-install.log')
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
    & $environmentPython @pipArguments check *> (Join-Path $stagingPath 'pip-check.log')
    if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }
    & $environmentPython -B (Join-Path $PSScriptRoot 'smoke_cpu_environment.py') --environment $environmentPath --out (Join-Path $stagingPath 'cpu-smoke.json')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency/CLI smoke failed.' }
    Get-ChildItem -LiteralPath $wheelDirectory -File -Filter '*.whl' |
        ForEach-Object { [PSCustomObject]@{ filename = $_.Name; size = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() } } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stagingPath 'wheel-manifest.json') -Encoding utf8
    Write-Output "Independent CPU environment: $environmentPath"
    Write-Output "Dependency/CLI report: $stagingPath/cpu-smoke.json"
} finally {
    $env:PIP_CACHE_DIR = $previousCache
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
}
