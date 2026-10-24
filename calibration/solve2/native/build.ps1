param(
  [string]$Repo = 'D:\OneDrive\codex_projs\RinkSpace',
  [string]$DataRoot = 'D:\CodexData\RinkSpace_Data',
  [string]$CeresPrefix = '',
  [string]$DepsRoot = '',
  [string]$VcVars = 'D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
)
$ErrorActionPreference='Stop'
$Repo=[IO.Path]::GetFullPath($Repo)
$DataRoot=[IO.Path]::GetFullPath($DataRoot)
if (!$CeresPrefix) { $CeresPrefix=Join-Path $Repo 'environments\cg22p' }
if (!$DepsRoot) { $DepsRoot=Join-Path $DataRoot 'runtime_tools\rf_deps' }
$source=Join-Path $Repo 'calibration\solve2\native\batch_ceres.cpp'
$sourceHash=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
$scriptHash=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$buildRoot=Join-Path $DataRoot ('runs\solve2_native\build_'+$sourceHash.Substring(0,10)+'_'+$scriptHash.Substring(0,6))
[void](New-Item -ItemType Directory -Path $buildRoot -Force)
$dll=Join-Path $buildRoot 'rinksolve_batch.dll'
$manifest=Join-Path $buildRoot 'build.json'
$required=@($source,$VcVars,(Join-Path $CeresPrefix 'lib\ceres.lib'),(Join-Path $CeresPrefix 'bin\ceres.dll'),
            (Join-Path $DepsRoot 'glog\Library\lib\glog.lib'),(Join-Path $DepsRoot 'gflags\Library\lib\gflags.lib'))
foreach($item in $required) { if (!(Test-Path -LiteralPath $item)) { throw ('Missing public dependency: '+$item) } }
if ((Test-Path -LiteralPath $dll) -and (Test-Path -LiteralPath $manifest)) {
  $old=Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
  if($old.source_sha256 -eq $sourceHash -and $old.status -eq 'built') { Write-Output $manifest; exit 0 }
}
$rsp=Join-Path $buildRoot 'compile.rsp'
$log=Join-Path $buildRoot 'build.log'
$argsList=@('/nologo','/std:c++17','/O2','/openmp','/fp:strict','/EHsc','/MD','/c','/DNDEBUG','/DGLOG_NO_ABBREVIATED_SEVERITIES','/DGLOG_USE_GLOG_EXPORT','/utf-8',
('/I"'+(Join-Path $CeresPrefix 'include')+'"'),('/I"'+(Join-Path $DepsRoot 'eigen\Library\include\eigen3')+'"'),
('/I"'+(Join-Path $DepsRoot 'glog\Library\include')+'"'),('/I"'+(Join-Path $DepsRoot 'gflags\Library\include')+'"'),
('"'+$source+'"'),('/Fo"'+(Join-Path $buildRoot 'batch_ceres.obj')+'"'))
$linkRsp=Join-Path $buildRoot 'link.rsp'
$linkArgs=@('/nologo','/DLL',('/OUT:"'+$dll+'"'),('"'+(Join-Path $buildRoot 'batch_ceres.obj')+'"'),
('"'+(Join-Path $CeresPrefix 'lib\ceres.lib')+'"'),('"'+(Join-Path $DepsRoot 'glog\Library\lib\glog.lib')+'"'),
('"'+(Join-Path $DepsRoot 'gflags\Library\lib\gflags.lib')+'"'))
[IO.File]::WriteAllLines($rsp,$argsList,[Text.Encoding]::ASCII)
[IO.File]::WriteAllLines($linkRsp,$linkArgs,[Text.Encoding]::ASCII)
$command='call "'+$VcVars+'" >nul && cl.exe @"'+$rsp+'" >"'+$log+'" 2>&1 && link.exe @"'+$linkRsp+'" >>"'+$log+'" 2>&1'
& $env:ComSpec /d /s /c $command
$code=$LASTEXITCODE
$status=if($code -eq 0 -and (Test-Path -LiteralPath $dll)) {'built'} else {'failed'}
$record=[ordered]@{schema='rinksolve.native_build.v1';status=$status;source=$source;source_sha256=$sourceHash;build_script_sha256=$scriptHash;
  binary=$dll;compiler_setup=$VcVars;ceres_prefix=$CeresPrefix;dependency_root=$DepsRoot;
  dll_directories=@((Join-Path $CeresPrefix 'bin'),(Join-Path $DepsRoot 'glog\Library\bin'),(Join-Path $DepsRoot 'gflags\Library\bin'),'D:\Program Files\CUDA\bin\x64');
  flags=$argsList;link_flags=$linkArgs;log=$log;exit_code=$code;uses_legacy_solver=$false}
if($status -eq 'built') {$record.binary_sha256=(Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant()}
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
Write-Output $manifest
if($status -ne 'built') { Get-Content -LiteralPath $log; throw 'Native Ceres bridge build failed' }
