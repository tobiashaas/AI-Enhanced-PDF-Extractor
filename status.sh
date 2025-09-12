#!/bin/bash
# System Status Monitor - Check processing status
# Auto-generated launcher for macOS/Linux

echo "Starting System Status Monitor - Check processing status..."

# Try python3 first (preferred on macOS/Linux)
if command -v python3 &> /dev/null; then
    python3 "status.py" "$@"
elif command -v python &> /dev/null; then
    python "status.py" "$@"
else
    echo "Error: Python not found in PATH"
    echo "Please install Python 3.x"
    exit 1
fi
