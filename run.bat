@echo off
echo ========================================
echo           NeuroSWAY
echo    Fall Detection System
echo    for Parkinson's Patients
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found:
python --version

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo ========================================

REM Install requirements from requirements.txt
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

pip install matplotlib==3.7.2
if errorlevel 1 (
    echo ERROR: Failed to install matplotlib
    pause
    exit /b 1
)


echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.

REM Test installation
echo Testing installation...
python -c "import cv2, mediapipe as mp, numpy as np; print('All modules imported successfully!')"
if errorlevel 1 (
    echo ERROR: Installation test failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting NeuroSWAY
echo ========================================
echo.

REM Run the fall detection system
python realtime_fall_detection.py

echo.
echo Program finished.
pause
