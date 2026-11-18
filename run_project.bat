@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================================
echo   Power Anomaly ML - Adaptive Electricity Anomaly Detection Launcher
echo ============================================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    echo [1/4] Activating Python virtual environment...
    call ".venv\Scripts\activate.bat"
) else (
    echo [1/4] Using system Python environment...
)

echo [2/4] Validating dataset and running ML pipeline...
python validate_data.py
if errorlevel 1 (
    echo Data validation failed. Exiting.
    pause
    exit /b 1
)

python run_pipeline.py
if errorlevel 1 (
    echo ML pipeline failed. Exiting.
    pause
    exit /b 1
)

if not exist "reports\accuracy_report.json" (
    python generate_accuracy_report.py
)

echo [3/4] Checking React 18 + Webpack 5 frontend build...
cd frontend
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
        echo npm install failed. Exiting.
        pause
        exit /b 1
    )
)

if not exist "dist\index.html" (
    echo Building Webpack 5 production bundle...
    call npm run build
    if errorlevel 1 (
        echo Frontend build failed. Exiting.
        pause
        exit /b 1
    )
)

echo [4/4] Starting Express server at http://localhost:5000 ...
start "" "http://localhost:5000"
node server.js
