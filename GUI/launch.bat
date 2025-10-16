@echo off
REM ========================================
REM  Auto-start script for YourQualityCheck GUI
REM  Runs on NUC startup
REM ========================================

REM --- Adjust these paths ---
set PYTHON_EXE=C:\Users\Group8\anaconda3\envs\AIENV\python.exe
set APP_DIR=C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\GUI
set SCRIPT=Triple_I_APP.py

REM --- Log file for debugging ---
set LOG_FILE=%APP_DIR%\startup_log.txt

echo [%date% %time%] Starting YourQualityCheck... >> "%LOG_FILE%"

REM --- Move to app directory ---
cd /d "%APP_DIR%"

REM --- Run the GUI silently ---
start "" "%PYTHON_EXE%" "%SCRIPT%" >> "%LOG_FILE%" 2>&1

echo [%date% %time%] YourQualityCheck launched. >> "%LOG_FILE%"
exit
