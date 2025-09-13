#!/usr/bin/env python3
"""
Apple Silicon (M1/M2/M3) Hardware Acceleration Manager
Neural Engine + Metal + Unified Memory für ultimative macOS Performance
"""

import os
import platform
import subprocess
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AppleSiliconProfile:
    """Hardware-Profil für Apple Silicon Chips"""
    chip_model: str
    performance_cores: int
    efficiency_cores: int
    neural_engine_cores: int
    gpu_cores: int
    unified_memory_gb: float
    shared_memory_gb: float
    optimal_workers: int
    metal_batch_size: int
    neural_engine_batch_size: int
    memory_bandwidth_gbps: float

class AppleSiliconManager:
    """Intelligenter Device Manager für Apple Silicon"""
    
    def __init__(self, config_path='config.json'):
        self.devices = {}
        self.workload_distribution = {}
        self._load_config(config_path)
        self._detect_apple_silicon()
        self._setup_metal_acceleration()
        
    def _load_config(self, config_path):
        """Lade Konfiguration aus config.json"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            self.apple_config = self.config.get('apple_silicon', {})
            self.ollama_config = self.config.get('ollama', {})
            logger.info("✅ Konfiguration erfolgreich geladen")
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Konfiguration: {e}")
            self.config = {}
            self.apple_config = {}
            self.ollama_config = {}
    
    def _detect_apple_silicon(self) -> Dict[str, Any]:
        """Detaillierte Apple Silicon Chip-Erkennung"""
        chip_info = {
            'available': False,
            'chip_family': 'unknown',
            'neural_engine_available': False
        }
        
        try:
            # Check if running on macOS
            if platform.system() != 'Darwin':
                logger.info("Nicht auf macOS - keine Apple Silicon Erkennung möglich")
                return chip_info
                
            # Get macOS version
            mac_ver = platform.mac_ver()[0]
            
            # Get processor info
            processor = platform.processor()
            machine = platform.machine()
            
            # Check if running on Apple Silicon
            is_apple_silicon = machine == 'arm64'
            
            if not is_apple_silicon:
                logger.info("Läuft auf Intel Mac, nicht auf Apple Silicon")
                return chip_info
                
            # Try to get detailed chip info using sysctl
            try:
                # Get chip model
                chip_model_cmd = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                              capture_output=True, text=True, check=True)
                chip_model = chip_model_cmd.stdout.strip()
                
                # Determine chip family and specs based on model string
                chip_family = 'unknown'
                neural_engine_cores = 16  # Default for most models
                if 'M1' in chip_model:
                    if 'Max' in chip_model:
                        chip_family = 'M1 Max'
                        performance_cores = 8
                        efficiency_cores = 2
                        gpu_cores = 24
                        memory_bandwidth = 400
                    elif 'Pro' in chip_model:
                        chip_family = 'M1 Pro'
                        performance_cores = 6
                        efficiency_cores = 2
                        gpu_cores = 14
                        memory_bandwidth = 200
                    elif 'Ultra' in chip_model:
                        chip_family = 'M1 Ultra'
                        performance_cores = 16
                        efficiency_cores = 4
                        gpu_cores = 48
                        memory_bandwidth = 800
                    else:
                        chip_family = 'M1'
                        performance_cores = 4
                        efficiency_cores = 4
                        gpu_cores = 7
                        memory_bandwidth = 68
                        neural_engine_cores = 16
                elif 'M2' in chip_model:
                    if 'Max' in chip_model:
                        chip_family = 'M2 Max'
                        performance_cores = 8
                        efficiency_cores = 4
                        gpu_cores = 30
                        memory_bandwidth = 400
                        neural_engine_cores = 16
                    elif 'Pro' in chip_model:
                        chip_family = 'M2 Pro'
                        performance_cores = 6
                        efficiency_cores = 4
                        gpu_cores = 16
                        memory_bandwidth = 200
                        neural_engine_cores = 16
                    else:
                        chip_family = 'M2'
                        performance_cores = 4
                        efficiency_cores = 4
                        gpu_cores = 8
                        memory_bandwidth = 100
                        neural_engine_cores = 16
                elif 'M3' in chip_model:
                    if 'Max' in chip_model:
                        chip_family = 'M3 Max'
                        performance_cores = 10
                        efficiency_cores = 4
                        gpu_cores = 40
                        memory_bandwidth = 800
                        neural_engine_cores = 16
                    elif 'Pro' in chip_model:
                        chip_family = 'M3 Pro'
                        performance_cores = 6
                        efficiency_cores = 6
                        gpu_cores = 18
                        memory_bandwidth = 300
                        neural_engine_cores = 16
                    else:
                        chip_family = 'M3'
                        performance_cores = 4
                        efficiency_cores = 4
                        gpu_cores = 10
                        memory_bandwidth = 100
                        neural_engine_cores = 16
                
                # Get RAM info
                memory = psutil.virtual_memory()
                ram_gb = memory.total / (1024**3)
                
                chip_info = {
                    'available': True,
                    'chip_model': chip_model,
                    'chip_family': chip_family,
                    'performance_cores': performance_cores,
                    'efficiency_cores': efficiency_cores,
                    'total_cores': performance_cores + efficiency_cores,
                    'gpu_cores': gpu_cores,
                    'neural_engine_cores': neural_engine_cores,
                    'neural_engine_available': True,
                    'unified_memory_gb': ram_gb,
                    'memory_bandwidth_gbps': memory_bandwidth,
                    'os_version': mac_ver
                }
                
                # Log detected hardware
                logger.info(f"🍎 Erkannter Apple Silicon Chip: {chip_family}")
                logger.info(f"   CPU: {performance_cores} Performance + {efficiency_cores} Efficiency Kerne")
                logger.info(f"   GPU: {gpu_cores} Kerne")
                logger.info(f"   Neural Engine: {neural_engine_cores} Kerne")
                logger.info(f"   Unified Memory: {ram_gb:.1f} GB")
                logger.info(f"   Memory Bandwidth: {memory_bandwidth} GB/s")
                
            except Exception as e:
                logger.warning(f"Detaillierte Chip-Info konnte nicht ermittelt werden: {e}")
                
                # Fallback: Basic detection
                chip_info = {
                    'available': True,
                    'chip_family': 'Apple Silicon',
                    'neural_engine_available': True,
                }
                
        except Exception as e:
            logger.warning(f"Apple Silicon Erkennung fehlgeschlagen: {e}")
            
        self.devices['apple_silicon'] = chip_info
        return chip_info
    
    def _setup_metal_acceleration(self):
        """Setup Metal Acceleration für maximale Performance"""
        try:
            # Get memory info
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            # Calculate optimal shared memory (60% of available for Apple Silicon)
            shared_memory_gb = min(available_gb * 0.6, 24.0)  # Max 24GB for M1 Max/Ultra
            
            # Set environment variables for Metal optimization
            metal_fraction = self.apple_config.get('metal_memory_fraction', 0.85)
            os.environ['PYTORCH_METAL_MEMORY_FRACTION'] = str(metal_fraction)
            
            # Enable PyTorch Metal backend
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            # Enable ANE acceleration when available
            if self.apple_config.get('ane_acceleration', True):
                os.environ['ENABLE_ANE'] = '1'
            
            # Metal Performance Shaders optimization
            os.environ['MPS_ENABLE_SHARED_MEMORY'] = '1'
            
            # Setup CoreML optimization for vision tasks
            os.environ['COREML_SILENCE_DEPRECATION'] = '1'
            
            self.metal_config = {
                'total_system_gb': total_gb,
                'available_gb': available_gb,
                'allocated_shared_gb': shared_memory_gb,
                'metal_fraction': metal_fraction,
                'ane_enabled': self.apple_config.get('ane_acceleration', True)
            }
            
            logger.info(f"🔥 Metal Acceleration Setup:")
            logger.info(f"   Metal Memory Fraction: {metal_fraction:.2f}")
            logger.info(f"   Shared Memory Pool: {shared_memory_gb:.1f} GB")
            logger.info(f"   ANE Acceleration: {'Enabled' if self.apple_config.get('ane_acceleration', True) else 'Disabled'}")
            
        except Exception as e:
            logger.error(f"Metal Acceleration Setup fehlgeschlagen: {e}")
    
    def get_apple_silicon_profile(self) -> AppleSiliconProfile:
        """Generiere optimales Profil für Apple Silicon"""
        chip_info = self.devices.get('apple_silicon', {})
        memory = psutil.virtual_memory()
        
        # Get chip parameters
        chip_model = chip_info.get('chip_family', 'unknown')
        perf_cores = chip_info.get('performance_cores', 4)
        eff_cores = chip_info.get('efficiency_cores', 4)
        neural_cores = chip_info.get('neural_engine_cores', 16)
        gpu_cores = chip_info.get('gpu_cores', 8)
        memory_gb = memory.total / (1024**3)
        memory_bw = chip_info.get('memory_bandwidth_gbps', 100)
        
        # Calculate optimal workers based on available cores
        # Apple Silicon benefits from slightly more workers due to efficient thread scheduling
        optimal_workers = min(perf_cores * 2 + eff_cores, 16)
        
        # Calculate Metal batch size based on GPU cores and memory
        gpu_factor = gpu_cores / 8  # Normalize against base M1
        metal_batch_size = min(int(200 * gpu_factor), 500)
        
        # Neural Engine batch size (typically smaller batches work better)
        neural_engine_batch_size = 120
        
        # Calculate shared memory allocation (less needed due to unified memory)
        shared_memory_gb = min(memory_gb * 0.3, 12.0)
        
        return AppleSiliconProfile(
            chip_model=chip_model,
            performance_cores=perf_cores,
            efficiency_cores=eff_cores,
            neural_engine_cores=neural_cores,
            gpu_cores=gpu_cores,
            unified_memory_gb=memory_gb,
            shared_memory_gb=self.metal_config.get('allocated_shared_gb', 4.0),
            optimal_workers=optimal_workers,
            metal_batch_size=metal_batch_size,
            neural_engine_batch_size=neural_engine_batch_size,
            memory_bandwidth_gbps=memory_bw
        )
    
    def get_optimal_device_assignment(self) -> Dict[str, str]:
        """Intelligente Workload-Verteilung für Apple Silicon"""
        
        # Get user-defined workload distribution from config if available
        if 'multi_device_pipeline' in self.config and 'workload_distribution' in self.config['multi_device_pipeline']:
            return self.config['multi_device_pipeline']['workload_distribution']
        
        # Default assignments optimized for Apple Silicon
        assignments = {
            'vision_analysis': 'metal',           # Metal für Vision Tasks
            'text_embeddings': 'neural_engine',   # Neural Engine für Embeddings
            'text_generation': 'metal',           # Metal für Text Generation
            'image_processing': 'metal',          # Metal für Bildverarbeitung
            'chunking': 'cpu_performance_cores',  # Performance-Kerne für Textverarbeitung
            'ocr': 'cpu_efficiency_cores'         # Efficiency-Kerne für OCR
        }
        
        return assignments

# Global instance
apple_silicon_manager = AppleSiliconManager()

def get_apple_silicon_manager() -> AppleSiliconManager:
    """Get the global Apple Silicon manager instance"""
    return apple_silicon_manager

def print_apple_silicon_summary():
    """Print comprehensive Apple Silicon hardware summary"""
    asm = get_apple_silicon_manager()
    profile = asm.get_apple_silicon_profile()
    
    print("\n🍎 APPLE SILICON MAXIMUM PERFORMANCE CONFIGURATION")
    print("=" * 70)
    
    print(f"💻 Chip: {profile.chip_model}")
    print(f"🧠 CPU: {profile.performance_cores} Performance + {profile.efficiency_cores} Efficiency Kerne")
    print(f"🎮 GPU: {profile.gpu_cores} Kerne")
    print(f"⚡ Neural Engine: {profile.neural_engine_cores} Kerne")
    print(f"📊 Memory Bandwidth: {profile.memory_bandwidth_gbps} GB/s")
    print(f"🔄 Unified Memory: {profile.unified_memory_gb:.1f} GB")
    print(f"🧠 Shared Memory Pool: {profile.shared_memory_gb:.1f} GB")
    print(f"⚡ Optimal Workers: {profile.optimal_workers}")
    print(f"📦 Metal Batch Size: {profile.metal_batch_size}")
    print(f"🧠 Neural Engine Batch Size: {profile.neural_engine_batch_size}")
    
    print("\n🎯 WORKLOAD ASSIGNMENTS:")
    assignments = asm.get_optimal_device_assignment()
    for workload, device in assignments.items():
        print(f"   {workload}: {device}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print_apple_silicon_summary()