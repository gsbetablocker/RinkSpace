@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_SPIN_MODELS.ps1" %*
if errorlevel 1 pause
