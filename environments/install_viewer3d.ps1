[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH=(Join-Path $projectRoot 'src')+';'+$projectRoot
$env:PYTHONDONTWRITEBYTECODE='1'
$basePython=Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $basePython)) { throw 'Create the project Python environment first.' }
$dataRoot=(& $basePython -B -c 'import sys; from rinkspace.storage import configured_data_root; print(configured_data_root(sys.argv[1]))' $projectRoot | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $dataRoot) { throw 'Cannot resolve the RinkSpace Data directory.' }
$runtime=Join-Path $dataRoot 'runtime_tools\results3d'
$python=Join-Path $runtime 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & $basePython -m venv $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create the dedicated viewer environment.' }
}
& $python -m pip install --upgrade 'pip>=26.0'
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
# CPU-only Torch keeps this viewer independent of the fitting/inference environments.
& $python -m pip install 'https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-win_amd64.whl'
if ($LASTEXITCODE -ne 0) { throw 'CPU Torch installation failed. This environment requires Windows x64 / Python 3.11.' }
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-viewer3d.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Viewer dependency installation failed.' }
$env:OPENBLAS_NUM_THREADS='2';$env:OMP_NUM_THREADS='2';$env:CUDA_VISIBLE_DEVICES=''
& $python -B -m rinkspace.viewer3d.prepare --project $projectRoot --data-root $dataRoot --all
if ($LASTEXITCODE -ne 0) { throw 'Saved-pose display preparation failed. Original results have not been modified.' }
Write-Output 'Ready. Double-click launchers\START_3D_RESULTS.cmd in the project directory.'
