#!/usr/bin/env python3
"""
BEAST MODE Live Dashboard - Zeigt die Power in Echtzeit!
"""

import time
import os
import psutil
import torch
from modules.hardware_acceleration import get_device_manager

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_gpu_stats():
    """Get NVIDIA GPU stats"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_name = torch.cuda.get_device_name(0)
        
        return {
            'name': gpu_name,
            'memory_used': memory_allocated,
            'memory_reserved': memory_reserved,
            'memory_total': memory_total,
            'utilization': (memory_allocated / memory_total) * 100 if memory_total > 0 else 0
        }
    return None

def get_system_stats():
    """Get system stats"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        'cpu_percent': cpu_percent,
        'cpu_cores': psutil.cpu_count(),
        'memory_used_gb': (memory.total - memory.available) / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        'memory_percent': memory.percent
    }

def create_bar(percentage, width=20):
    """Create a visual progress bar"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def print_beast_dashboard():
    """Print the live beast dashboard"""
    
    # Get all stats
    dm = get_device_manager()
    gpu_stats = get_gpu_stats()
    sys_stats = get_system_stats()
    
    clear_screen()
    
    print("🦾" + "="*78 + "🦾")
    print("🔥               BEAST MODE - LIVE PERFORMANCE DASHBOARD              🔥")
    print("🦾" + "="*78 + "🦾")
    print()
    
    # System Overview
    print("💻 SYSTEM OVERVIEW:")
    print(f"   CPU Usage:  {create_bar(sys_stats['cpu_percent'], 30)} ({sys_stats['cpu_cores']} cores)")
    print(f"   RAM Usage:  {create_bar(sys_stats['memory_percent'], 30)} ({sys_stats['memory_used_gb']:.1f}/{sys_stats['memory_total_gb']:.1f} GB)")
    print()
    
    # GPU Stats
    if gpu_stats:
        print("🎮 NVIDIA RTX 2000 ADA:")
        print(f"   GPU Memory: {create_bar(gpu_stats['utilization'], 30)} ({gpu_stats['memory_used']:.1f}/{gpu_stats['memory_total']:.1f} GB)")
        print(f"   Reserved:   {gpu_stats['memory_reserved']:.2f} GB")
    else:
        print("🎮 NVIDIA GPU: ❌ Not Available")
    print()
    
    # Multi-Device Status
    print("🚀 MULTI-DEVICE ACCELERATION STATUS:")
    devices = dm.devices
    
    # NVIDIA GPU
    gpu_status = "🟢 READY" if devices['cuda_gpu']['available'] else "🔴 OFFLINE"
    print(f"   🎮 NVIDIA CUDA GPU:     {gpu_status}")
    
    # Intel NPU
    npu_status = "🟢 READY" if devices['intel_npu']['available'] else "🔴 OFFLINE"
    print(f"   🧠 Intel AI Boost NPU: {npu_status} (Power Efficient)")
    
    # Intel iGPU
    igpu_status = "🟢 READY" if devices['intel_igpu']['available'] else "🔴 OFFLINE"
    print(f"   📱 Intel iGPU:         {igpu_status} (Shared Memory)")
    
    # DirectML
    dml_status = "🟢 READY" if devices['directml']['available'] else "🔴 OFFLINE"
    print(f"   🪟 DirectML NPU:       {dml_status} (Windows ML)")
    
    print()
    
    # Performance Configuration
    profile = dm.get_hardware_profile()
    print("⚡ PERFORMANCE CONFIGURATION:")
    print(f"   Parallel Workers:    {profile.optimal_workers}")
    print(f"   GPU Batch Size:      {profile.gpu_batch_size}")
    print(f"   NPU Batch Size:      {profile.npu_batch_size}")
    print(f"   Shared Memory Pool:  {profile.shared_memory_gb:.1f} GB")
    print()
    
    # Workload Distribution
    print("🎯 INTELLIGENT WORKLOAD DISTRIBUTION:")
    workloads = [
        ("Vision Analysis", "🎮 NVIDIA GPU", "cuda_gpu"),
        ("Text Embeddings", "🧠 Intel NPU", "intel_npu"), 
        ("Text Generation", "🎮 NVIDIA GPU", "cuda_gpu"),
        ("Image Processing", "📱 Intel iGPU", "intel_igpu"),
        ("OCR Tasks", "🪟 DirectML", "directml"),
        ("Text Chunking", "💻 CPU Cores", "cpu")
    ]
    
    for task, device, device_key in workloads:
        available = devices.get(device_key, {}).get('available', device_key == 'cpu')
        status = "✅" if available else "❌"
        print(f"   {task:15} → {device:15} {status}")
    
    print()
    
    # Beast Mode Indicator
    active_devices = sum(1 for d in devices.values() if d.get('available', False))
    beast_level = active_devices / 4.0 * 100
    
    print("🦾 BEAST MODE LEVEL:")
    print(f"   {create_bar(beast_level, 40)} ({active_devices}/4 devices)")
    
    if beast_level == 100:
        print("   🔥🔥🔥 MAXIMUM BEAST MODE ACTIVATED! 🔥🔥🔥")
    elif beast_level >= 75:
        print("   💪💪 HIGH PERFORMANCE MODE 💪💪")
    elif beast_level >= 50:
        print("   ⚡ STANDARD ACCELERATION ⚡")
    else:
        print("   📱 BASIC MODE")
    
    print()
    print("🦾" + "="*78 + "🦾")
    print(f"⏰ Last Update: {time.strftime('%H:%M:%S')} | Press Ctrl+C to exit")
    print("🦾" + "="*78 + "🦾")

def run_live_dashboard():
    """Run the live dashboard"""
    print("🚀 Starting BEAST MODE Live Dashboard...")
    print("   Press Ctrl+C to exit")
    time.sleep(2)
    
    try:
        while True:
            print_beast_dashboard()
            time.sleep(2)  # Update every 2 seconds
    except KeyboardInterrupt:
        clear_screen()
        print("🦾 BEAST MODE Dashboard stopped.")
        print("💪 Your system is ready for maximum performance!")

if __name__ == "__main__":
    run_live_dashboard()