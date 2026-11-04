@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_SUPPORT.ps1" %*
if errorlevel 1 pause
