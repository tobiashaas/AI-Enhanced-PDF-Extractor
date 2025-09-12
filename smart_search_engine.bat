@echo off
REM Smart Search Engine - Query processed documents
REM Auto-generated launcher for Windows

echo Starting Smart Search Engine - Query processed documents...
python "smart_search_engine.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Python command failed. Trying 'py' command...
    py "smart_search_engine.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Could not run Python script.
    echo Please ensure Python is installed and in your PATH.
    echo.
    pause
)
