@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_3D_GRASP.ps1" %*
set "rinkspace_exit=%ERRORLEVEL%"
if not "%rinkspace_exit%"=="0" (
  echo.
  echo RinkSpace 3D failed. Review the error above.
  pause
)
exit /b %rinkspace_exit%
