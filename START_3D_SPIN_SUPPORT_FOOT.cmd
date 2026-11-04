@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_RESULTS.ps1" -Config "%~dp0events\ankara_2026\programmes\okada_mei_sp\viewer3d_spin_support_foot_1104.json" %*
exit /b %ERRORLEVEL%
