[CmdletBinding()]
param([ValidateSet('Check','Install','Serve')][string]$LocalPythonAction)

function Get-RinkSpacePythonPath([string]$ProjectRoot, [switch]$InstallLocation) {
    $bootstrap = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $bootstrap -PathType Leaf)) {
        $bootstrap = 'python'
    }
    $resolve = @'
from pathlib import Path
import sys
project = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project / 'src'))
from rinkspace.storage import AssetStore
assets = AssetStore(project)
interpreter = assets.resolve('.venv/Scripts/python.exe')
if sys.argv[2] == 'Install' and assets.configured:
    interpreter = assets.data / 'runtime_tools/local_python/Scripts/python.exe'
print(interpreter)
'@
    $encoding = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = [Text.Encoding]::UTF8
        $mode = if ($InstallLocation) { 'Install' } else { 'Run' }
        $resolved = & $bootstrap -B -X utf8 -c $resolve $ProjectRoot $mode
        if ($LASTEXITCODE -ne 0 -or -not $resolved) {
            throw 'Cannot resolve the RinkSpace Python environment.'
        }
        return $resolved
    } finally {
        [Console]::OutputEncoding = $encoding
    }
}

if ($LocalPythonAction) {
    $ErrorActionPreference = 'Stop'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $python = Get-RinkSpacePythonPath $projectRoot -InstallLocation:($LocalPythonAction -eq 'Install')
    $pathCheck = Join-Path $projectRoot 'scripts/check_onedrive_paths.py'
    $installed = Test-Path -LiteralPath $python -PathType Leaf
    if ($LocalPythonAction -eq 'Serve' -and -not $installed) {
        throw 'Run launchers\INSTALL_LOCAL.cmd first.'
    }
    $checker = if ($installed) { $python } else { 'python' }
    & $checker -B $pathCheck --root $projectRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($LocalPythonAction -eq 'Install') {
        $environment = Split-Path -Parent (Split-Path -Parent $python)
        $runtimeTools = Split-Path -Parent $environment
        if ((Split-Path -Leaf $runtimeTools) -eq 'runtime_tools') {
            $runtimeTemp = Join-Path (Split-Path -Parent $runtimeTools) 'tmp'
            [void][IO.Directory]::CreateDirectory($runtimeTemp)
            $env:TEMP = $runtimeTemp
            $env:TMP = $runtimeTemp
        }
        if (-not $installed) {
            & $checker -B -m venv $environment
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        & $python -B -m pip install --no-compile -e ($projectRoot + '[dev,acquisition]')
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python -B $pathCheck --root $projectRoot
    } elseif ($LocalPythonAction -eq 'Serve') {
        & $python -B -m rinkspace serve --root $projectRoot --port 8765 --open
    }
    exit $LASTEXITCODE
}
