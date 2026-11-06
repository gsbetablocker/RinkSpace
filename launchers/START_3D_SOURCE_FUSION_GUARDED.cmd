@echo off
if "%~1"=="" (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\events\ankara_2026\programmes\okada_mei_sp\viewer3d_source_fusion_guarded_1106.json" -Variant joint_source_guarded -Fresh
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0..\events\ankara_2026\programmes\okada_mei_sp\viewer3d_source_fusion_guarded_1106.json" %*
)
exit /b %ERRORLEVEL%
