#!/bin/bash
# AI PDF Processor - Main extraction system
# Auto-generated launcher for macOS/Linux

echo "Starting AI PDF Processor - Main extraction system..."

# Try python3 first (preferred on macOS/Linux)
if command -v python3 &> /dev/null; then
    python3 "ai_pdf_processor.py" "$@"
elif command -v python &> /dev/null; then
    python "ai_pdf_processor.py" "$@"
else
    echo "Error: Python not found in PATH"
    echo "Please install Python 3.x"
    exit 1
fi
