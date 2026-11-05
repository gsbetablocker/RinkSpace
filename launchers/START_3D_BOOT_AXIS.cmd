@echo off
if "%~1"=="" (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\events\ankara_2026\programmes\okada_mei_sp\viewer3d_boot_axis_1105.json" -Variant boot_axis_entry
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\events\ankara_2026\programmes\okada_mei_sp\viewer3d_boot_axis_1105.json" %*
)
exit /b %ERRORLEVEL%
