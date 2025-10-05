@echo off
echo ========================================
echo Transit Detector Web App - Launcher
echo ========================================
echo.

echo [1/3] Installing requirements...
pip install -r ./requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)
echo.

echo [2/3] Starting Transit Detector server...
start /B python ./transit_detector_app_fixed.py

echo [3/3] Waiting for server to start...
timeout /t 5 /nobreak > nul

echo Opening browser...
start http://127.0.0.1:5000

echo.
echo ========================================
echo Transit Detector is now running!
echo Browser should open automatically.
echo Server running at: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server.
echo ========================================
pause
