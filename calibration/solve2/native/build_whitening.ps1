param(
 [string]$Repo='D:\OneDrive\codex_projs\RinkSpace',
 [string]$DataRoot='D:\CodexData\RinkSpace_Data',
 [string]$VcVars='D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
)
$ErrorActionPreference='Stop'
$Repo=[IO.Path]::GetFullPath($Repo)
$DataRoot=[IO.Path]::GetFullPath($DataRoot)
$source=Join-Path $Repo 'calibration\solve2\native\factor_assembly.cpp'
$sourceHash=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
$scriptHash=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$buildRoot=Join-Path $DataRoot ('runs\solve2_native\whitening_'+$sourceHash.Substring(0,10)+'_'+$scriptHash.Substring(0,6))
[void](New-Item -ItemType Directory -Path $buildRoot -Force)
$dll=Join-Path $buildRoot 'rinksolve_whitening.dll'
$manifest=Join-Path $buildRoot 'build.json'
if (!(Test-Path -LiteralPath $VcVars)) { throw 'Public MSVC compiler setup missing' }
if ((Test-Path -LiteralPath $dll) -and (Test-Path -LiteralPath $manifest)) {
 $prior=Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
 if($prior.source_sha256 -eq $sourceHash -and $prior.status -eq 'built') {Write-Output $manifest;exit 0}
}
$rsp=Join-Path $buildRoot 'compile.rsp'
$log=Join-Path $buildRoot 'build.log'
$argsList=@('/nologo','/std:c++17','/O2','/fp:precise','/EHsc','/MD','/openmp','/LD','/DNDEBUG','/utf-8',('"'+$source+'"'),('/Fo"'+(Join-Path $buildRoot 'factor_assembly.obj')+'"'),('/Fe"'+$dll+'"'))
[IO.File]::WriteAllLines($rsp,$argsList,[Text.Encoding]::ASCII)
$command='call "'+$VcVars+'" >nul && cl.exe @"'+$rsp+'" >"'+$log+'" 2>&1'
& $env:ComSpec /d /s /c $command
$code=$LASTEXITCODE
$status=if($code -eq 0 -and (Test-Path -LiteralPath $dll)) {'built'} else {'failed'}
$record=[ordered]@{schema='rinksolve.whitening_build.v1';status=$status;source=$source;source_sha256=$sourceHash;build_script_sha256=$scriptHash;binary=$dll;compiler_setup=$VcVars;flags=$argsList;log=$log;exit_code=$code;uses_legacy_solver=$false;abi_version=1;parallel_runtime='MSVC OpenMP';dll_directories=@()}
if($status -eq 'built') {$record.binary_sha256=(Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant()}
$record|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $manifest -Encoding utf8
Write-Output $manifest
if($status -ne 'built'){Get-Content -LiteralPath $log;throw 'New factor assembly build failed'}
