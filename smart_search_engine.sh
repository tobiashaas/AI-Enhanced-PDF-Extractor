#!/bin/bash
# Smart Search Engine - Query processed documents
# Auto-generated launcher for macOS/Linux

echo "Starting Smart Search Engine - Query processed documents..."

# Try python3 first (preferred on macOS/Linux)
if command -v python3 &> /dev/null; then
    python3 "smart_search_engine.py" "$@"
elif command -v python &> /dev/null; then
    python "smart_search_engine.py" "$@"
else
    echo "Error: Python not found in PATH"
    echo "Please install Python 3.x"
    exit 1
fi
