@echo off
REM AI PDF Processor - Main extraction system
REM Auto-generated launcher for Windows

echo Starting AI PDF Processor - Main extraction system...
python "ai_pdf_processor.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Python command failed. Trying 'py' command...
    py "ai_pdf_processor.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Could not run Python script.
    echo Please ensure Python is installed and in your PATH.
    echo.
    pause
)
