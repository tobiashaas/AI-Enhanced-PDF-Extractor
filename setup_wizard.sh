#!/bin/bash
# Setup Wizard - Interactive configuration
# Auto-generated launcher for macOS/Linux

echo "Starting Setup Wizard - Interactive configuration..."

# Try python3 first (preferred on macOS/Linux)
if command -v python3 &> /dev/null; then
    python3 "setup_wizard.py" "$@"
elif command -v python &> /dev/null; then
    python "setup_wizard.py" "$@"
else
    echo "Error: Python not found in PATH"
    echo "Please install Python 3.x"
    exit 1
fi
