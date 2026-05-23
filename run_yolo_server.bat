@echo off
cd /d "%~dp0"
echo.
echo [Vion] Installing YOLO/MSR server dependencies...
python -m pip install -r requirements.txt
echo.
echo [Vion] Starting YOLO/MSR API at http://localhost:8000/analyze-score
echo [Vion] Optional: set VION_YOLO_WEIGHTS=E:\dev\best.pt before running this file.
echo.
python -m uvicorn yolo_msr_server:app --host 127.0.0.1 --port 8000
pause
