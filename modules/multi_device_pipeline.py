#!/usr/bin/env python3
"""
Multi-Device AI Pipeline für maximale Performance
GPU + NPU + iGPU + DirectML Koordination
"""

import asyncio
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor

from .hardware_acceleration import get_device_manager, HardwareProfile

logger = logging.getLogger(__name__)

@dataclass
class WorkloadTask:
    """Task für die Multi-Device Pipeline"""
    task_id: str
    task_type: str  # 'vision', 'embedding', 'generation', 'processing'
    data: Any
    priority: int = 1
    device_preference: Optional[str] = None
    callback: Optional[callable] = None

class DeviceWorker(ABC):
    """Abstract base class für Device-specific Workers"""
    
    def __init__(self, device_name: str, batch_size: int = 100):
        self.device_name = device_name
        self.batch_size = batch_size
        self.is_busy = False
        self.task_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
    @abstractmethod
    async def process_task(self, task: WorkloadTask) -> Any:
        """Process a single task on this device"""
        pass
    
    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """Check if this worker can handle the task type"""
        pass

class CUDAGPUWorker(DeviceWorker):
    """NVIDIA CUDA GPU Worker für Vision & Text Generation"""
    
    def __init__(self, **kwargs):
        super().__init__("cuda_gpu", **kwargs)
        self.supported_tasks = ['vision_analysis', 'text_generation', 'large_embeddings']
        
    def can_handle(self, task_type: str) -> bool:
        return task_type in self.supported_tasks
        
    async def process_task(self, task: WorkloadTask) -> Any:
        """GPU Processing mit CUDA"""
        try:
            self.is_busy = True
            
            if task.task_type == 'vision_analysis':
                return await self._process_vision(task.data)
            elif task.task_type == 'text_generation':
                return await self._process_text_generation(task.data)
            elif task.task_type == 'large_embeddings':
                return await self._process_large_embeddings(task.data)
                
        except Exception as e:
            logger.error(f"CUDA GPU task failed: {e}")
            return None
        finally:
            self.is_busy = False
    
    async def _process_vision(self, data: Dict) -> Dict:
        """Vision Analysis auf GPU"""
        # Placeholder für Ollama Vision API
        await asyncio.sleep(0.1)  # Simulate processing
        return {"vision_result": f"GPU processed vision for {data.get('image_id', 'unknown')}"}
    
    async def _process_text_generation(self, data: Dict) -> Dict:
        """Text Generation auf GPU"""
        await asyncio.sleep(0.2)  # Simulate processing
        return {"text_result": f"GPU generated text: {data.get('prompt', '')[:50]}..."}
    
    async def _process_large_embeddings(self, data: Dict) -> Dict:
        """Large Embeddings auf GPU (Fallback)"""
        await asyncio.sleep(0.05)  # Simulate processing
        return {"embedding": [0.1] * 768, "device": "cuda_gpu"}

class NPUWorker(DeviceWorker):
    """Intel NPU Worker für Embeddings & kleine AI Tasks"""
    
    def __init__(self, **kwargs):
        super().__init__("intel_npu", **kwargs)
        self.supported_tasks = ['text_embeddings', 'small_ai_tasks', 'inference']
        
    def can_handle(self, task_type: str) -> bool:
        return task_type in self.supported_tasks
        
    async def process_task(self, task: WorkloadTask) -> Any:
        """NPU Processing mit OpenVINO"""
        try:
            self.is_busy = True
            
            if task.task_type == 'text_embeddings':
                return await self._process_embeddings(task.data)
            elif task.task_type == 'small_ai_tasks':
                return await self._process_small_ai(task.data)
                
        except Exception as e:
            logger.error(f"NPU task failed: {e}")
            return None
        finally:
            self.is_busy = False
    
    async def _process_embeddings(self, data: Dict) -> Dict:
        """Text Embeddings auf NPU - Ultra efficient"""
        await asyncio.sleep(0.02)  # NPU is faster for embeddings
        return {"embedding": [0.2] * 768, "device": "intel_npu", "efficiency": "high"}
    
    async def _process_small_ai(self, data: Dict) -> Dict:
        """Kleine AI Tasks auf NPU"""
        await asyncio.sleep(0.03)
        return {"ai_result": f"NPU processed: {data.get('input', '')}", "power_efficient": True}

class iGPUWorker(DeviceWorker):
    """Intel iGPU Worker für Image Processing"""
    
    def __init__(self, **kwargs):
        super().__init__("intel_igpu", **kwargs)
        self.supported_tasks = ['image_processing', 'image_conversion', 'ocr_prep']
        
    def can_handle(self, task_type: str) -> bool:
        return task_type in self.supported_tasks
        
    async def process_task(self, task: WorkloadTask) -> Any:
        """iGPU Processing mit OpenVINO"""
        try:
            self.is_busy = True
            
            if task.task_type == 'image_processing':
                return await self._process_images(task.data)
            elif task.task_type == 'image_conversion':
                return await self._convert_images(task.data)
                
        except Exception as e:
            logger.error(f"iGPU task failed: {e}")
            return None
        finally:
            self.is_busy = False
    
    async def _process_images(self, data: Dict) -> Dict:
        """Image Processing auf iGPU"""
        await asyncio.sleep(0.08)  # Moderate speed
        return {"processed_image": f"iGPU processed {data.get('image_count', 1)} images"}
    
    async def _convert_images(self, data: Dict) -> Dict:
        """Image Format Conversion"""
        await asyncio.sleep(0.05)
        return {"converted": True, "format": data.get('target_format', 'jpeg')}

class DirectMLWorker(DeviceWorker):
    """DirectML Worker für ONNX Models & OCR"""
    
    def __init__(self, **kwargs):
        super().__init__("directml", **kwargs)
        self.supported_tasks = ['ocr', 'onnx_inference', 'specialized_ml']
        
    def can_handle(self, task_type: str) -> bool:
        return task_type in self.supported_tasks
        
    async def process_task(self, task: WorkloadTask) -> Any:
        """DirectML Processing"""
        try:
            self.is_busy = True
            
            if task.task_type == 'ocr':
                return await self._process_ocr(task.data)
            elif task.task_type == 'onnx_inference':
                return await self._onnx_inference(task.data)
                
        except Exception as e:
            logger.error(f"DirectML task failed: {e}")
            return None
        finally:
            self.is_busy = False
    
    async def _process_ocr(self, data: Dict) -> Dict:
        """OCR auf DirectML"""
        await asyncio.sleep(0.06)
        return {"ocr_text": f"DirectML OCR: {data.get('image_id', 'unknown')}", "confidence": 0.95}
    
    async def _onnx_inference(self, data: Dict) -> Dict:
        """ONNX Model Inference"""
        await asyncio.sleep(0.04)
        return {"inference_result": "DirectML ONNX processed", "model": data.get('model_name', 'default')}

class MultiDeviceAIPipeline:
    """Koordiniert alle AI-Devices für maximale Performance"""
    
    def __init__(self):
        self.device_manager = get_device_manager()
        self.hardware_profile = self.device_manager.get_hardware_profile()
        self.workers = {}
        self.task_scheduler = None
        self.performance_monitor = PerformanceMonitor()
        
        self._initialize_workers()
        self._start_scheduler()
    
    def _initialize_workers(self):
        """Initialize alle verfügbare Workers"""
        # CUDA GPU Worker
        if self.device_manager.devices['cuda_gpu']['available']:
            self.workers['cuda_gpu'] = CUDAGPUWorker(
                batch_size=self.hardware_profile.gpu_batch_size
            )
            logger.info("✅ CUDA GPU Worker initialized")
        
        # NPU Worker
        if self.device_manager.devices['intel_npu']['available']:
            self.workers['intel_npu'] = NPUWorker(
                batch_size=self.hardware_profile.npu_batch_size
            )
            logger.info("✅ Intel NPU Worker initialized")
        
        # iGPU Worker
        if self.device_manager.devices['intel_igpu']['available']:
            self.workers['intel_igpu'] = iGPUWorker(batch_size=150)
            logger.info("✅ Intel iGPU Worker initialized")
        
        # DirectML Worker
        if self.device_manager.devices['directml']['available']:
            self.workers['directml'] = DirectMLWorker(batch_size=100)
            logger.info("✅ DirectML Worker initialized")
        
        logger.info(f"🚀 Multi-Device Pipeline: {len(self.workers)} workers active")
    
    def _start_scheduler(self):
        """Start the task scheduler"""
        self.task_queue = asyncio.Queue()
        self.results = {}
        
    async def submit_task(self, task: WorkloadTask) -> str:
        """Submit a task to the pipeline"""
        await self.task_queue.put(task)
        return task.task_id
    
    async def submit_batch(self, tasks: List[WorkloadTask]) -> List[str]:
        """Submit multiple tasks efficiently"""
        task_ids = []
        for task in tasks:
            await self.task_queue.put(task)
            task_ids.append(task.task_id)
        return task_ids
    
    async def get_result(self, task_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Get task result with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if task_id in self.results:
                return self.results.pop(task_id)
            await asyncio.sleep(0.01)
        return None
    
    async def process_pipeline(self) -> None:
        """Main processing loop"""
        while True:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Find optimal worker
                optimal_worker = self._find_optimal_worker(task)
                
                if optimal_worker:
                    # Process task
                    start_time = time.time()
                    result = await optimal_worker.process_task(task)
                    processing_time = time.time() - start_time
                    
                    # Store result
                    self.results[task.task_id] = result
                    
                    # Update performance metrics
                    self.performance_monitor.record_task(
                        task.task_type, 
                        optimal_worker.device_name, 
                        processing_time
                    )
                    
                    # Callback if provided
                    if task.callback:
                        await task.callback(result)
                else:
                    logger.warning(f"No worker available for task {task.task_type}")
                    self.results[task.task_id] = None
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
    
    def _find_optimal_worker(self, task: WorkloadTask) -> Optional[DeviceWorker]:
        """Find the best worker for a task"""
        # Check device preference first
        if task.device_preference and task.device_preference in self.workers:
            worker = self.workers[task.device_preference]
            if worker.can_handle(task.task_type) and not worker.is_busy:
                return worker
        
        # Find optimal device based on workload assignment
        optimal_device = self.device_manager.get_optimal_device_assignment(task.task_type)
        
        if optimal_device in self.workers:
            worker = self.workers[optimal_device]
            if worker.can_handle(task.task_type) and not worker.is_busy:
                return worker
        
        # Fallback: find any available worker that can handle the task
        for worker in self.workers.values():
            if worker.can_handle(task.task_type) and not worker.is_busy:
                return worker
        
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_monitor.get_stats()

class PerformanceMonitor:
    """Monitor performance across all devices"""
    
    def __init__(self):
        self.task_stats = {}
        self.device_stats = {}
        
    def record_task(self, task_type: str, device: str, processing_time: float):
        """Record task performance"""
        if task_type not in self.task_stats:
            self.task_stats[task_type] = []
        
        if device not in self.device_stats:
            self.device_stats[device] = []
        
        self.task_stats[task_type].append(processing_time)
        self.device_stats[device].append(processing_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'task_performance': {},
            'device_performance': {},
            'total_tasks': sum(len(times) for times in self.task_stats.values())
        }
        
        # Task type stats
        for task_type, times in self.task_stats.items():
            if times:
                stats['task_performance'][task_type] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        # Device stats
        for device, times in self.device_stats.items():
            if times:
                stats['device_performance'][device] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'utilization': len(times) / max(len(times), 1)  # Simplified utilization
                }
        
        return stats

# Global pipeline instance
_pipeline_instance = None

def get_ai_pipeline() -> MultiDeviceAIPipeline:
    """Get or create the global AI pipeline"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MultiDeviceAIPipeline()
    return _pipeline_instance

# Convenience functions
async def process_vision_task(image_data: Dict, task_id: str = None) -> Optional[Dict]:
    """Process vision analysis task"""
    pipeline = get_ai_pipeline()
    task_id = task_id or f"vision_{int(time.time() * 1000)}"
    
    task = WorkloadTask(
        task_id=task_id,
        task_type='vision_analysis',
        data=image_data,
        priority=2
    )
    
    await pipeline.submit_task(task)
    return await pipeline.get_result(task_id)

async def process_embedding_task(text_data: Dict, task_id: str = None) -> Optional[Dict]:
    """Process text embedding task"""
    pipeline = get_ai_pipeline()
    task_id = task_id or f"embed_{int(time.time() * 1000)}"
    
    task = WorkloadTask(
        task_id=task_id,
        task_type='text_embeddings',
        data=text_data,
        priority=1
    )
    
    await pipeline.submit_task(task)
    return await pipeline.get_result(task_id)

async def process_generation_task(prompt_data: Dict, task_id: str = None) -> Optional[Dict]:
    """Process text generation task"""
    pipeline = get_ai_pipeline()
    task_id = task_id or f"gen_{int(time.time() * 1000)}"
    
    task = WorkloadTask(
        task_id=task_id,
        task_type='text_generation',
        data=prompt_data,
        priority=2
    )
    
    await pipeline.submit_task(task)
    return await pipeline.get_result(task_id)