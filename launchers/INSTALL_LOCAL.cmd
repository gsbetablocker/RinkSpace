@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0LOCAL_PYTHON.ps1" -LocalPythonAction Install
set "RINKSPACE_INSTALL_EXIT=%ERRORLEVEL%"
pause
exit /b %RINKSPACE_INSTALL_EXIT%
