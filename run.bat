@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_live_local.py %*
) else (
    python run_live_local.py %*
)
pause
