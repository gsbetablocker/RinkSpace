@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_INTEGRATION.ps1" %*
if errorlevel 1 pause
