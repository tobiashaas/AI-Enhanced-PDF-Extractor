@echo off
REM System Status Monitor - Check processing status
REM Auto-generated launcher for Windows

echo Starting System Status Monitor - Check processing status...
python "status.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Python command failed. Trying 'py' command...
    py "status.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Could not run Python script.
    echo Please ensure Python is installed and in your PATH.
    echo.
    pause
)
