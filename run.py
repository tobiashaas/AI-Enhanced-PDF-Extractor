#!/usr/bin/env python3
"""
Universal Cross-Platform Launcher
=================================
Automatically detects platform and runs appropriate Python command
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def find_python():
    """Find the best Python command for this platform"""
    commands = []
    
    if platform.system() == 'Windows':
        commands = ['python', 'py', 'python3']
    else:
        commands = ['python3', 'python']
    
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Found: {cmd} ({version})")
                return cmd
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    return None

def main():
    if len(sys.argv) < 2:
        print("🚀 AI-ENHANCED PDF EXTRACTION SYSTEM")
        print("=" * 50)
        print("Available scripts:")
        print("  ai_pdf_processor.py    - Main extraction system")
        print("  smart_search_engine.py - Query processed documents") 
        print("  status.py             - Check system status")
        print("  setup_wizard.py       - Interactive setup")
        print()
        print("Usage: python run.py <script_name> [args...]")
        print("Example: python run.py ai_pdf_processor.py --help")
        return
    
    script_name = sys.argv[1]
    script_args = sys.argv[2:]
    
    # Check if script exists
    script_path = Path(script_name)
    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return
    
    # Find Python command
    python_cmd = find_python()
    if not python_cmd:
        print("❌ No Python installation found!")
        print("Please install Python 3.x")
        return
    
    # Run the script
    cmd = [python_cmd, script_name] + script_args
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error running script: {e}")

if __name__ == "__main__":
    main()
