@echo off
set PYTHONDONTWRITEBYTECODE=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run INSTALL_LOCAL.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B "scripts\check_onedrive_paths.py"
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B -m rinkspace serve --root "%~dp0." --port 8765 --open
pause
