#!/usr/bin/env python3
"""
Maximum Performance Hardware Acceleration Manager
GPU + NPU + Shared Memory für ultimative Performance
"""

import os
import torch
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    """Hardware-Profil für optimale Ressourcenverteilung"""
    gpu_memory_gb: float
    cpu_cores: int
    system_ram_gb: float
    npu_available: bool
    igpu_available: bool
    shared_memory_gb: float
    optimal_workers: int
    gpu_batch_size: int
    npu_batch_size: int

class DeviceManager:
    """Intelligenter Device Manager für GPU+NPU Koordination"""
    
    def __init__(self):
        self.devices = {}
        self.workload_distribution = {}
        self._detect_devices()
        self._setup_shared_memory()
        
    def _detect_devices(self):
        """Detaillierte Hardware-Erkennung"""
        self.devices = {
            'cuda_gpu': self._detect_cuda(),
            'intel_npu': self._detect_npu(),
            'intel_igpu': self._detect_igpu(),
            'directml': self._detect_directml()
        }
        
        logger.info("🔍 Hardware Detection Complete:")
        for device, status in self.devices.items():
            logger.info(f"   {device}: {'✅' if status['available'] else '❌'}")
    
    def _detect_cuda(self) -> Dict[str, Any]:
        """NVIDIA CUDA Detection"""
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return {
                    'available': True,
                    'name': torch.cuda.get_device_name(0),
                    'memory_gb': props.total_memory // (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessors': props.multi_processor_count
                }
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")
        
        return {'available': False}
    
    def _detect_npu(self) -> Dict[str, Any]:
        """Intel NPU Detection via OpenVINO"""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            
            if 'NPU' in devices:
                # Try to get NPU properties
                try:
                    npu_props = core.get_property('NPU', ov.properties.supported_properties)
                    return {
                        'available': True,
                        'device': 'NPU',
                        'framework': 'OpenVINO',
                        'properties': str(npu_props)[:200]  # Limit output
                    }
                except Exception:
                    return {
                        'available': True,
                        'device': 'NPU',
                        'framework': 'OpenVINO',
                        'properties': 'Intel AI Boost NPU'
                    }
        except Exception as e:
            logger.warning(f"NPU detection failed: {e}")
        
        return {'available': False}
    
    def _detect_igpu(self) -> Dict[str, Any]:
        """Intel iGPU Detection"""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            
            if 'GPU' in devices:
                return {
                    'available': True,
                    'device': 'GPU',
                    'framework': 'OpenVINO',
                    'type': 'Intel Integrated Graphics'
                }
        except Exception as e:
            logger.warning(f"iGPU detection failed: {e}")
        
        return {'available': False}
    
    def _detect_directml(self) -> Dict[str, Any]:
        """DirectML Detection"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'DmlExecutionProvider' in providers:
                return {
                    'available': True,
                    'providers': providers,
                    'framework': 'DirectML'
                }
        except Exception as e:
            logger.warning(f"DirectML detection failed: {e}")
        
        return {'available': False}
    
    def _setup_shared_memory(self):
        """Setup Windows Shared Memory für GPU+NPU"""
        try:
            # System Memory Info
            memory = psutil.virtual_memory()
            
            # Calculate optimal shared memory
            total_gb = memory.total // (1024**3)
            available_gb = memory.available // (1024**3)
            
            # Reserve 30% of available memory for shared operations
            shared_memory_gb = min(available_gb * 0.3, 20.0)  # Max 20GB for 64GB systems
            
            # Set environment variables for memory optimization
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # DirectML Memory Optimization
            os.environ['ORT_DIRECTML_MEMORY_FRACTION'] = '0.3'  # 30% of GPU memory
            
            # PyTorch Memory Settings
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)  # 80% for PyTorch (increased)
                torch.cuda.empty_cache()
            
            self.shared_memory_config = {
                'total_system_gb': total_gb,
                'available_gb': available_gb,
                'allocated_shared_gb': shared_memory_gb,
                'cuda_fraction': 0.8,
                'directml_fraction': 0.4
            }
            
            logger.info(f"🧠 Shared Memory Setup:")
            logger.info(f"   Total System: {total_gb} GB")
            logger.info(f"   Available: {available_gb} GB") 
            logger.info(f"   Shared Pool: {shared_memory_gb:.1f} GB")
            
        except Exception as e:
            logger.error(f"Shared memory setup failed: {e}")
    
    def get_hardware_profile(self) -> HardwareProfile:
        """Generate optimal hardware profile"""
        cuda_info = self.devices.get('cuda_gpu', {})
        memory = psutil.virtual_memory()
        
        # Calculate optimal workers based on all available devices
        base_workers = psutil.cpu_count()
        
        # Boost workers if multiple accelerators available
        accelerator_count = sum(1 for d in self.devices.values() if d.get('available', False))
        optimal_workers = min(base_workers + (accelerator_count * 2), 16)
        
        # GPU Batch size based on VRAM
        gpu_memory = cuda_info.get('memory_gb', 0)
        gpu_batch_size = min(300 if gpu_memory >= 8 else 200, 500)
        
        # NPU typically works better with smaller batches
        npu_batch_size = 100
        
        return HardwareProfile(
            gpu_memory_gb=gpu_memory,
            cpu_cores=base_workers,
            system_ram_gb=memory.total // (1024**3),
            npu_available=self.devices['intel_npu']['available'],
            igpu_available=self.devices['intel_igpu']['available'],
            shared_memory_gb=self.shared_memory_config.get('allocated_shared_gb', 4.0),
            optimal_workers=optimal_workers,
            gpu_batch_size=gpu_batch_size,
            npu_batch_size=npu_batch_size
        )
    
    def get_optimal_device_assignment(self, workload_type: str) -> Dict[str, str]:
        """Intelligente Workload-Verteilung"""
        assignments = {
            'vision_analysis': 'cuda_gpu',      # GPU für Computer Vision
            'text_embeddings': 'intel_npu',     # NPU für Embeddings
            'text_generation': 'cuda_gpu',      # GPU für LLM
            'image_processing': 'intel_igpu',   # iGPU für Bild-Ops
            'chunking': 'cpu',                  # CPU für Text-Processing
            'ocr': 'directml'                   # DirectML für OCR
        }
        
        # Fallback wenn NPU nicht verfügbar
        if not self.devices['intel_npu']['available']:
            assignments['text_embeddings'] = 'cuda_gpu'
        
        if not self.devices['intel_igpu']['available']:
            assignments['image_processing'] = 'cuda_gpu'
            
        if not self.devices['directml']['available']:
            assignments['ocr'] = 'cuda_gpu'
        
        return assignments.get(workload_type, 'cpu')

# Global instance
device_manager = DeviceManager()

def get_device_manager() -> DeviceManager:
    """Get the global device manager instance"""
    return device_manager

def print_hardware_summary():
    """Print comprehensive hardware summary"""
    dm = get_device_manager()
    profile = dm.get_hardware_profile()
    
    print("\n🚀 MAXIMUM PERFORMANCE CONFIGURATION")
    print("=" * 60)
    
    print(f"💻 System: {profile.cpu_cores} cores, {profile.system_ram_gb} GB RAM")
    print(f"🎮 NVIDIA GPU: {profile.gpu_memory_gb} GB VRAM")
    print(f"🧠 Intel NPU: {'✅ Available' if profile.npu_available else '❌ Not available'}")
    print(f"📱 Intel iGPU: {'✅ Available' if profile.igpu_available else '❌ Not available'}")
    print(f"🔄 Shared Memory Pool: {profile.shared_memory_gb:.1f} GB")
    print(f"⚡ Optimal Workers: {profile.optimal_workers}")
    print(f"📦 GPU Batch Size: {profile.gpu_batch_size}")
    print(f"🧠 NPU Batch Size: {profile.npu_batch_size}")
    
    print("\n🎯 WORKLOAD ASSIGNMENTS:")
    workloads = ['vision_analysis', 'text_embeddings', 'text_generation', 'image_processing', 'chunking', 'ocr']
    for workload in workloads:
        device = dm.get_optimal_device_assignment(workload)
        print(f"   {workload}: {device}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print_hardware_summary()