@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0LOCAL_PYTHON.ps1" -LocalPythonAction Serve
set "RINKSPACE_START_EXIT=%ERRORLEVEL%"
pause
exit /b %RINKSPACE_START_EXIT%
