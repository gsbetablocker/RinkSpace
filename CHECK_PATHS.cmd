@echo off
set PYTHONDONTWRITEBYTECODE=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B "scripts\check_onedrive_paths.py"
) else (
  python -B "scripts\check_onedrive_paths.py"
)
set "RINKSPACE_PATH_CHECK_EXIT=%ERRORLEVEL%"
pause
exit /b %RINKSPACE_PATH_CHECK_EXIT%
