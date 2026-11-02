[CmdletBinding()]
param([string]$Variant='', [switch]$Fresh, [switch]$Wait)
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
$env:PYTHONPATH=(Join-Path $projectRoot 'src')+';'+$projectRoot
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:CUDA_VISIBLE_DEVICES=''
function Quote-PlayerArgument([string]$Value) {
    $escaped=[regex]::Replace($Value,'(\\*)"','$1$1\"')
    $escaped=[regex]::Replace($escaped,'(\\+)$','$1$1')
    return '"'+$escaped+'"'
}
try {
    $basePython=Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $basePython)) { throw 'Project Python is missing.' }
    $dataRoot=(& $basePython -B -c 'import sys; from rinkspace.storage import configured_data_root; print(configured_data_root(sys.argv[1]))' $projectRoot | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $dataRoot) { throw 'Cannot resolve the RinkSpace Data directory.' }
    $python=Join-Path $dataRoot 'runtime_tools\results3d\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'Viewer environment is missing. Run environments\install_viewer3d.ps1 once.' }
    $work=Join-Path $dataRoot 'workbench\results3d'
    [void][IO.Directory]::CreateDirectory($work)
    $values=@('-B','-X','utf8','-m','rinkspace.viewer3d.app','--project',$projectRoot,'--data-root',$dataRoot)
    if ($Variant) { $values+=@('--variant',$Variant) }
    if ($Fresh) { $values+='--fresh' }
    if ($Wait) {
        Push-Location $work
        try {
            & $python @values
            if ($LASTEXITCODE -ne 0) { throw 'The 3D player exited with an error.' }
        } finally { Pop-Location }
    } else {
        $stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $stdout=Join-Path $work ('start_'+$stamp+'.stdout.log')
        $stderr=Join-Path $work ('start_'+$stamp+'.stderr.log')
        $arguments=($values | ForEach-Object { Quote-PlayerArgument $_ }) -join ' '
        $child=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $work -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        Write-Output ('RinkSpace 3D player started. Log: '+$stderr)
        # Catch an immediate import/startup error while allowing the native window to run independently.
        if ($child.WaitForExit(5000) -and $child.ExitCode -ne 0) {
            throw ((Get-Content -LiteralPath $stderr -Raw -Encoding UTF8)+'; Log: '+$stderr)
        }
    }
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [void][System.Windows.Forms.MessageBox]::Show([string]$_,'RinkSpace 3D player - startup failed')
    exit 1
}
