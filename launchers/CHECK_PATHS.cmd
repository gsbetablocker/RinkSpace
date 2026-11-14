@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0LOCAL_PYTHON.ps1" -LocalPythonAction Check
set "RINKSPACE_PATH_CHECK_EXIT=%ERRORLEVEL%"
pause
exit /b %RINKSPACE_PATH_CHECK_EXIT%
