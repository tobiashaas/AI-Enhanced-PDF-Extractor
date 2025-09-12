#!/usr/bin/env python3
"""
Universal AI PDF Processor Launcher
===================================
Cross-platform auto-detection and optimal configuration launcher
Detects OS, Hardware, Python version and launches with best settings
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path

class UniversalLauncher:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.python_cmd = self.detect_python()
        self.hardware_info = self.detect_hardware()
        
    def detect_python(self):
        """Detect best Python command for this platform"""
        commands = []
        
        if self.os_type == 'windows':
            commands = ['python', 'py', 'python3']
        else:
            commands = ['python3', 'python']
        
        for cmd in commands:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'Python 3' in result.stdout:
                    version = result.stdout.strip().split()[1]
                    print(f"✅ Found Python {version} via '{cmd}'")
                    return cmd
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        print("❌ No suitable Python 3.x found!")
        return None
    
    def detect_hardware(self):
        """Detect hardware capabilities"""
        info = {
            'cpu_count': os.cpu_count() or 4,
            'has_gpu': False,
            'gpu_type': None,
            'has_metal': False,
            'has_cuda': False,
            'recommended_workers': 4,
            'recommended_batch_size': 100
        }
        
        # macOS Metal detection
        if self.os_type == 'darwin':
            if 'arm64' in self.architecture or 'm1' in self.architecture.lower() or 'm2' in self.architecture.lower():
                info['has_metal'] = True
                info['has_gpu'] = True
                info['gpu_type'] = 'Apple Silicon'
                info['recommended_workers'] = 6
                info['recommended_batch_size'] = 150
                
        # Windows/Linux NVIDIA detection
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=3)
            if result.returncode == 0:
                info['has_cuda'] = True
                info['has_gpu'] = True
                info['gpu_type'] = 'NVIDIA CUDA'
                info['recommended_workers'] = 8
                info['recommended_batch_size'] = 200
        except:
            pass
        
        # Adjust for CPU count
        info['recommended_workers'] = min(info['recommended_workers'], info['cpu_count'])
        
        return info
    
    def print_system_info(self):
        """Display detected system information"""
        print("🖥️  SYSTEM DETECTION")
        print("=" * 50)
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print(f"CPU Cores: {self.hardware_info['cpu_count']}")
        print(f"Python: {self.python_cmd}")
        
        if self.hardware_info['has_gpu']:
            print(f"GPU: ✅ {self.hardware_info['gpu_type']}")
        else:
            print(f"GPU: ❌ CPU only")
            
        print(f"Recommended Workers: {self.hardware_info['recommended_workers']}")
        print(f"Recommended Batch Size: {self.hardware_info['recommended_batch_size']}")
        print()
    
    def check_dependencies(self):
        """Check if all dependencies are available"""
        print("🔍 DEPENDENCY CHECK")
        print("=" * 30)
        
        dependencies = {
            'Ollama': self.check_ollama,
            'Config': self.check_config,
            'Python Packages': self.check_python_packages
        }
        
        all_ok = True
        for name, check_func in dependencies.items():
            try:
                if check_func():
                    print(f"✅ {name}: OK")
                else:
                    print(f"❌ {name}: Missing")
                    all_ok = False
            except Exception as e:
                print(f"⚠️  {name}: Error - {e}")
                all_ok = False
        
        return all_ok
    
    def check_ollama(self):
        """Check Ollama installation and version"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code == 200:
                version = response.json().get('version', 'unknown')
                print(f"   Ollama {version}")
                return True
        except:
            pass
        return False
    
    def check_config(self):
        """Check if config.json exists"""
        return Path('config.json').exists()
    
    def check_python_packages(self):
        """Check critical Python packages"""
        try:
            import fitz, requests, supabase
            return True
        except ImportError:
            return False
    
    def setup_missing_dependencies(self):
        """Guide user through missing dependency setup"""
        print("\n🔧 SETUP MISSING DEPENDENCIES")
        print("=" * 40)
        
        if not self.check_config():
            print("⚠️  No config.json found")
            print("💡 Running setup wizard...")
            self.run_setup_wizard()
        
        if not self.check_ollama():
            print("⚠️  Ollama not running")
            print("💡 Install instructions:")
            if self.os_type == 'windows':
                print("   Windows: Download from https://ollama.ai/download")
            else:
                print("   macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        
        if not self.check_python_packages():
            print("⚠️  Python packages missing")
            print("💡 Install: pip install -r requirements.txt")
    
    def run_setup_wizard(self):
        """Run the setup wizard if available"""
        if Path('setup_wizard.py').exists():
            print("🧙 Starting setup wizard...")
            subprocess.run([self.python_cmd, 'setup_wizard.py'])
        else:
            print("❌ setup_wizard.py not found")
    
    def get_optimal_args(self, script_name, user_args):
        """Generate optimal arguments based on hardware"""
        args = list(user_args)  # Copy user args
        
        # Add hardware-optimized arguments if not already specified
        if '--workers' not in args and '--parallel-workers' not in args:
            args.extend(['--workers', str(self.hardware_info['recommended_workers'])])
        
        if '--batch-size' not in args:
            args.extend(['--batch-size', str(self.hardware_info['recommended_batch_size'])])
        
        # Add GPU acceleration flags
        if self.hardware_info['has_metal'] and '--metal' not in args:
            args.append('--metal')
        
        if self.hardware_info['has_cuda'] and '--cuda' not in args:
            args.append('--cuda')
        
        return args
    
    def launch_script(self, script_name, args=None):
        """Launch a script with optimal configuration"""
        if not self.python_cmd:
            print("❌ Cannot launch: No suitable Python found")
            return False
        
        script_path = Path(script_name)
        if not script_path.exists():
            print(f"❌ Script not found: {script_name}")
            return False
        
        # Get optimal arguments
        user_args = args or []
        optimal_args = self.get_optimal_args(script_name, user_args)
        
        # Construct command
        cmd = [self.python_cmd, script_name] + optimal_args
        
        print(f"\n🚀 LAUNCHING: {script_name}")
        print("=" * 50)
        print(f"Command: {' '.join(cmd)}")
        print(f"Optimized for: {self.hardware_info['gpu_type'] or 'CPU'}")
        print()
        
        try:
            subprocess.run(cmd)
            return True
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Launch error: {e}")
            return False

def main():
    print("🌍 UNIVERSAL AI PDF PROCESSOR LAUNCHER")
    print("=" * 60)
    print("Auto-detecting system and optimizing configuration...")
    print()
    
    launcher = UniversalLauncher()
    launcher.print_system_info()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("📋 AVAILABLE COMMANDS:")
        print("=" * 30)
        print("  process <file>     - Process PDF document")
        print("  search             - Interactive search engine")  
        print("  status             - System status check")
        print("  setup              - Run setup wizard")
        print()
        print("📝 USAGE EXAMPLES:")
        print("  python launch.py process document.pdf")
        print("  python launch.py search")
        print("  python launch.py status")
        print("  python launch.py setup")
        print()
        return
    
    command = sys.argv[1].lower()
    remaining_args = sys.argv[2:]
    
    # Map commands to scripts
    script_mapping = {
        'process': 'ai_pdf_processor.py',
        'search': 'smart_search_engine.py',
        'status': 'status.py', 
        'setup': 'setup_wizard.py'
    }
    
    if command not in script_mapping:
        print(f"❌ Unknown command: {command}")
        return
    
    script_name = script_mapping[command]
    
    # Check dependencies
    if not launcher.check_dependencies():
        print("\n⚠️  Missing dependencies detected!")
        launcher.setup_missing_dependencies()
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    # Launch with optimal configuration
    launcher.launch_script(script_name, remaining_args)

if __name__ == "__main__":
    main()