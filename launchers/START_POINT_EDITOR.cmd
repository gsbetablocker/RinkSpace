@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_POINT_EDITOR.ps1" %*
exit /b %errorlevel%
