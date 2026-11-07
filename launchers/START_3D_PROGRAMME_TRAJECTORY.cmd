@echo off
set "RS_PROGRAMME_CONFIG=%~dp0..\events\ankara_2026\programmes\okada_mei_sp\viewer3d_programme_trajectory_1106.json"
if not exist "%RS_PROGRAMME_CONFIG%" (
  echo Register an actual completed programme result with programme_player first.
  exit /b 2
)
if "%~1"=="" (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%RS_PROGRAMME_CONFIG%" -Fresh
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%RS_PROGRAMME_CONFIG%" %*
)
exit /b %ERRORLEVEL%
