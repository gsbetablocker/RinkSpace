@echo off
set PYTHONDONTWRITEBYTECODE=1
set PIP_NO_COMPILE=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B "scripts\check_onedrive_paths.py"
) else (
  python -B "scripts\check_onedrive_paths.py"
)
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" python -B -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -B -m pip install --no-compile -e ".[dev,acquisition]"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -B "scripts\check_onedrive_paths.py"
if errorlevel 1 exit /b 1
pause
