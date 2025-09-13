#!/usr/bin/env python3
"""
NPU/GPU Hardware Detection und Support Check
"""

import sys
import platform

def check_hardware_acceleration():
    print("🔍 HARDWARE ACCELERATION CHECK")
    print("=" * 50)
    
    # System Info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print()
    
    # PyTorch CUDA
    try:
        import torch
        print("🎮 NVIDIA GPU STATUS:")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA verfügbar: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Geräte: {torch.cuda.device_count()}")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        print()
    except ImportError:
        print("❌ PyTorch nicht installiert")
        print()
    
    # Intel Extension for PyTorch (NPU)
    try:
        import intel_extension_for_pytorch as ipex
        print("🧠 INTEL NPU STATUS:")
        print(f"   Intel Extension: {ipex.__version__}")
        print("   ✅ NPU Support verfügbar")
        print()
    except ImportError:
        print("🧠 INTEL NPU STATUS:")
        print("   ❌ Intel Extension nicht installiert")
        print("   💡 Für NPU Support installieren:")
        print("      pip install intel_extension_for_pytorch")
        print()
    
    # OpenVINO (Intel AI)
    try:
        import openvino as ov
        print("⚡ OPENVINO STATUS:")
        print(f"   OpenVINO Version: {ov.__version__}")
        core = ov.Core()
        devices = core.available_devices
        print(f"   Verfügbare Geräte: {devices}")
        if 'NPU' in devices:
            print("   ✅ NPU Device erkannt")
        if 'GPU' in devices:
            print("   ✅ iGPU Device erkannt")
        print()
    except ImportError:
        print("⚡ OPENVINO STATUS:")
        print("   ❌ OpenVINO nicht installiert")
        print("   💡 Für Intel AI Boost:")
        print("      pip install openvino")
        print()
    
    # DirectML (Windows NPU)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print("🪟 DIRECTML STATUS:")
        print(f"   ONNX Runtime verfügbar: {ort.__version__}")
        if 'DmlExecutionProvider' in providers:
            print("   ✅ DirectML Provider verfügbar")
            print("   ✅ Windows NPU Support möglich")
        else:
            print("   ❌ DirectML Provider nicht verfügbar")
        print(f"   Verfügbare Provider: {providers}")
        print()
    except ImportError:
        print("🪟 DIRECTML STATUS:")
        print("   ❌ ONNX Runtime nicht installiert")
        print("   💡 Für Windows NPU:")
        print("      pip install onnxruntime-directml")
        print()

if __name__ == "__main__":
    check_hardware_acceleration()