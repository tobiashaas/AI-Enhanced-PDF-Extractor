@echo off
REM Setup Wizard - Interactive configuration
REM Auto-generated launcher for Windows

echo Starting Setup Wizard - Interactive configuration...
python "setup_wizard.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Python command failed. Trying 'py' command...
    py "setup_wizard.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Could not run Python script.
    echo Please ensure Python is installed and in your PATH.
    echo.
    pause
)
