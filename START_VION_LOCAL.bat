@echo off
setlocal
cd /d "%~dp0"

title Vion Local Launcher
echo.
echo ==========================================
echo  Vion local launcher
echo ==========================================
echo.

set "PY_CMD="

python --version >nul 2>&1
if %errorlevel%==0 (
    set "PY_CMD=python"
) else (
    py -3 --version >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    echo Python is not installed or not available.
    echo.
    where winget >nul 2>&1
    if %errorlevel%==0 (
        echo This script can install Python 3.11 with winget.
        choice /M "Install Python 3.11 now"
        if errorlevel 2 goto python_help
        winget install -e --id Python.Python.3.11
        echo.
        echo Python install finished. Please close this window and run START_VION_LOCAL.bat again.
        pause
        exit /b
    )
    goto python_help
)

echo Using Python:
%PY_CMD% --version
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 goto venv_fail
)

echo Installing server packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto deps_fail

echo.
echo Starting YOLO/MSR API server...
echo API: http://localhost:8000/analyze-score
start "Vion YOLO/MSR API" cmd /k ""%CD%\.venv\Scripts\python.exe" -m uvicorn yolo_msr_server:app --host 127.0.0.1 --port 8000"

echo Opening Vion app...
start "" "%CD%\vion.html"
echo.
echo Done. Keep the API server window open while using YOLO/MSR mode.
pause
exit /b

:python_help
echo.
echo Please install Python 3.11 or newer first:
echo https://www.python.org/downloads/
echo.
echo During install, check:
echo [x] Add python.exe to PATH
echo.
pause
exit /b 1

:venv_fail
echo.
echo Failed to create Python virtual environment.
echo Install Python from https://www.python.org/downloads/ and check "Add python.exe to PATH".
pause
exit /b 1

:deps_fail
echo.
echo Failed to install Python packages.
echo Check your internet connection, then run this file again.
pause
exit /b 1
