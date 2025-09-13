#!/usr/bin/env python3
"""
Real-time Performance Monitor für Multi-Device AI Pipeline
Überwacht GPU + NPU + iGPU + DirectML Performance in Echtzeit
"""

import time
import threading
import psutil
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DeviceMetrics:
    """Metriken für ein einzelnes Device"""
    device_name: str
    device_type: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[float]
    power_draw_w: Optional[float]
    tasks_completed: int
    avg_task_time_ms: float
    current_batch_size: int
    is_busy: bool
    last_update: str

@dataclass 
class SystemMetrics:
    """System-weite Metriken"""
    timestamp: str
    cpu_usage_percent: float
    ram_used_gb: float
    ram_total_gb: float
    shared_memory_used_gb: float
    total_tasks_per_second: float
    active_workers: int
    queue_length: int
    overall_efficiency: float

@dataclass
class PerformanceSnapshot:
    """Kompletter Performance-Snapshot"""
    system: SystemMetrics
    devices: List[DeviceMetrics]
    bottlenecks: List[str]
    recommendations: List[str]
    uptime_seconds: float

class RealTimeMonitor:
    """Echtzeit Performance Monitor"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.is_running = False
        self.start_time = time.time()
        
        # Metrics storage
        self.device_metrics = {}
        self.system_metrics = None
        self.performance_history = []
        self.max_history_length = 300  # 5 Minuten bei 1s Intervall
        
        # Threading
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Device monitors
        self.cuda_monitor = CUDAMonitor() if TORCH_AVAILABLE else None
        self.openvino_monitor = OpenVINOMonitor() if OPENVINO_AVAILABLE else None
        self.system_monitor = SystemMonitor()
        
        # Performance tracking
        self.task_counters = {}
        self.task_timings = {}
        
    def start(self):
        """Start monitoring in background thread"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🔍 Real-time performance monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect metrics from all sources
                self._collect_metrics()
                
                # Analyze for bottlenecks
                self._analyze_bottlenecks()
                
                # Update history
                self._update_history()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self):
        """Collect metrics from all devices"""
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            
            self.system_metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_usage_percent=cpu_percent,
                ram_used_gb=round((memory.total - memory.available) / (1024**3), 2),
                ram_total_gb=round(memory.total / (1024**3), 2),
                shared_memory_used_gb=self._estimate_shared_memory(),
                total_tasks_per_second=self._calculate_throughput(),
                active_workers=len([d for d in self.device_metrics.values() if d.is_busy]),
                queue_length=0,  # Would need to be passed from pipeline
                overall_efficiency=self._calculate_efficiency()
            )
            
            # Device-specific metrics
            if self.cuda_monitor:
                cuda_metrics = self.cuda_monitor.get_metrics()
                if cuda_metrics:
                    self.device_metrics['cuda_gpu'] = cuda_metrics
            
            if self.openvino_monitor:
                npu_metrics = self.openvino_monitor.get_npu_metrics()
                igpu_metrics = self.openvino_monitor.get_igpu_metrics()
                
                if npu_metrics:
                    self.device_metrics['intel_npu'] = npu_metrics
                if igpu_metrics:
                    self.device_metrics['intel_igpu'] = igpu_metrics
    
    def _estimate_shared_memory(self) -> float:
        """Estimate shared memory usage"""
        # Simplified estimation - would need actual shared memory tracking
        return round(psutil.virtual_memory().shared / (1024**3), 2)
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks per second across all devices"""
        total_tasks = sum(counter.get('completed', 0) for counter in self.task_counters.values())
        uptime = time.time() - self.start_time
        return round(total_tasks / max(uptime, 1), 2)
    
    def _calculate_efficiency(self) -> float:
        """Calculate overall system efficiency"""
        if not self.device_metrics:
            return 0.0
        
        busy_devices = sum(1 for device in self.device_metrics.values() if device.is_busy)
        total_devices = len(self.device_metrics)
        
        if total_devices == 0:
            return 0.0
        
        return round((busy_devices / total_devices) * 100, 1)
    
    def _analyze_bottlenecks(self):
        """Analyze system for performance bottlenecks"""
        self.current_bottlenecks = []
        self.current_recommendations = []
        
        if not self.system_metrics:
            return
        
        # CPU bottleneck
        if self.system_metrics.cpu_usage_percent > 85:
            self.current_bottlenecks.append("High CPU usage")
            self.current_recommendations.append("Consider reducing parallel workers")
        
        # Memory bottleneck
        memory_usage = (self.system_metrics.ram_used_gb / self.system_metrics.ram_total_gb) * 100
        if memory_usage > 85:
            self.current_bottlenecks.append("High memory usage")
            self.current_recommendations.append("Reduce batch sizes or enable memory optimization")
        
        # Device-specific bottlenecks
        for device_name, metrics in self.device_metrics.items():
            if metrics.memory_used_mb / metrics.memory_total_mb > 0.9:
                self.current_bottlenecks.append(f"{device_name} memory nearly full")
                self.current_recommendations.append(f"Reduce {device_name} batch size")
        
        # Low efficiency
        if self.system_metrics.overall_efficiency < 50:
            self.current_bottlenecks.append("Low device utilization")
            self.current_recommendations.append("Check task distribution and queue management")
    
    def _update_history(self):
        """Update performance history"""
        snapshot = self.get_current_snapshot()
        self.performance_history.append(snapshot)
        
        # Limit history size
        if len(self.performance_history) > self.max_history_length:
            self.performance_history.pop(0)
    
    def get_current_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        with self.lock:
            return PerformanceSnapshot(
                system=self.system_metrics or SystemMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_usage_percent=0, ram_used_gb=0, ram_total_gb=0,
                    shared_memory_used_gb=0, total_tasks_per_second=0,
                    active_workers=0, queue_length=0, overall_efficiency=0
                ),
                devices=list(self.device_metrics.values()),
                bottlenecks=getattr(self, 'current_bottlenecks', []),
                recommendations=getattr(self, 'current_recommendations', []),
                uptime_seconds=round(time.time() - self.start_time, 1)
            )
    
    def record_task_completion(self, device: str, task_type: str, duration_ms: float):
        """Record task completion for metrics"""
        with self.lock:
            if device not in self.task_counters:
                self.task_counters[device] = {'completed': 0, 'total_time': 0}
            
            self.task_counters[device]['completed'] += 1
            self.task_counters[device]['total_time'] += duration_ms
            
            # Update device metrics
            if device in self.device_metrics:
                metrics = self.device_metrics[device]
                metrics.tasks_completed = self.task_counters[device]['completed']
                metrics.avg_task_time_ms = round(
                    self.task_counters[device]['total_time'] / 
                    self.task_counters[device]['completed'], 2
                )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        snapshot = self.get_current_snapshot()
        
        # Calculate trends from history
        trends = self._calculate_trends()
        
        return {
            'timestamp': snapshot.system.timestamp,
            'uptime_seconds': snapshot.uptime_seconds,
            'system': asdict(snapshot.system),
            'devices': [asdict(device) for device in snapshot.devices],
            'bottlenecks': snapshot.bottlenecks,
            'recommendations': snapshot.recommendations,
            'trends': trends,
            'summary': {
                'total_devices': len(snapshot.devices),
                'active_devices': len([d for d in snapshot.devices if d.is_busy]),
                'total_memory_gb': sum(d.memory_total_mb for d in snapshot.devices) / 1024,
                'total_tasks_completed': sum(d.tasks_completed for d in snapshot.devices),
                'avg_efficiency': snapshot.system.overall_efficiency
            }
        }
    
    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate performance trends"""
        if len(self.performance_history) < 10:
            return {'trend': 'insufficient_data'}
        
        recent = self.performance_history[-5:]
        older = self.performance_history[-10:-5]
        
        recent_avg_efficiency = sum(s.system.overall_efficiency for s in recent) / len(recent)
        older_avg_efficiency = sum(s.system.overall_efficiency for s in older) / len(older)
        
        if recent_avg_efficiency > older_avg_efficiency + 5:
            trend = 'improving'
        elif recent_avg_efficiency < older_avg_efficiency - 5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'efficiency_trend': trend,
            'recent_avg_efficiency': round(recent_avg_efficiency, 1),
            'older_avg_efficiency': round(older_avg_efficiency, 1)
        }
    
    def print_live_dashboard(self):
        """Print live performance dashboard"""
        snapshot = self.get_current_snapshot()
        
        print("\n" + "="*80)
        print("🚀 REAL-TIME PERFORMANCE DASHBOARD")
        print("="*80)
        
        # System overview
        print(f"⏱️  Uptime: {snapshot.uptime_seconds:.1f}s")
        print(f"💻 CPU: {snapshot.system.cpu_usage_percent:.1f}%")
        print(f"🧠 RAM: {snapshot.system.ram_used_gb:.1f}/{snapshot.system.ram_total_gb:.1f} GB")
        print(f"📊 Efficiency: {snapshot.system.overall_efficiency:.1f}%")
        print(f"⚡ Tasks/sec: {snapshot.system.total_tasks_per_second:.2f}")
        
        # Device status
        print("\n📱 DEVICE STATUS:")
        for device in snapshot.devices:
            status = "🟢 BUSY" if device.is_busy else "⚪ IDLE"
            memory_pct = (device.memory_used_mb / device.memory_total_mb) * 100
            print(f"   {device.device_name}: {status} | "
                  f"Mem: {memory_pct:.1f}% | "
                  f"Tasks: {device.tasks_completed} | "
                  f"Avg: {device.avg_task_time_ms:.1f}ms")
        
        # Bottlenecks
        if snapshot.bottlenecks:
            print("\n⚠️  BOTTLENECKS:")
            for bottleneck in snapshot.bottlenecks:
                print(f"   • {bottleneck}")
        
        # Recommendations
        if snapshot.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in snapshot.recommendations:
                print(f"   • {rec}")
        
        print("="*80)

class CUDAMonitor:
    """NVIDIA CUDA GPU Monitor"""
    
    def get_metrics(self) -> Optional[DeviceMetrics]:
        """Get CUDA GPU metrics"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            # GPU memory info
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory
            
            return DeviceMetrics(
                device_name="NVIDIA RTX 2000 Ada",
                device_type="cuda_gpu",
                utilization_percent=self._estimate_utilization(),
                memory_used_mb=memory_allocated / (1024**2),
                memory_total_mb=total_memory / (1024**2),
                temperature_c=None,  # Would need nvidia-ml-py for this
                power_draw_w=None,
                tasks_completed=0,  # Updated by task recorder
                avg_task_time_ms=0.0,
                current_batch_size=0,
                is_busy=memory_allocated > memory_reserved * 0.1,
                last_update=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"CUDA metrics error: {e}")
            return None
    
    def _estimate_utilization(self) -> float:
        """Estimate GPU utilization based on memory usage"""
        if not torch.cuda.is_available():
            return 0.0
        
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        
        if memory_reserved == 0:
            return 0.0
        
        # Simple heuristic: utilization based on memory allocation
        return min((memory_allocated / memory_reserved) * 100, 100.0)

class OpenVINOMonitor:
    """OpenVINO NPU/iGPU Monitor"""
    
    def __init__(self):
        if OPENVINO_AVAILABLE:
            try:
                self.core = ov.Core()
                self.available_devices = self.core.available_devices
            except Exception as e:
                logger.warning(f"OpenVINO monitor init failed: {e}")
                self.core = None
                self.available_devices = []
        else:
            self.core = None
            self.available_devices = []
    
    def get_npu_metrics(self) -> Optional[DeviceMetrics]:
        """Get NPU metrics"""
        if not self.core or 'NPU' not in self.available_devices:
            return None
        
        try:
            return DeviceMetrics(
                device_name="Intel AI Boost NPU",
                device_type="intel_npu",
                utilization_percent=self._estimate_npu_utilization(),
                memory_used_mb=0,  # NPU memory not easily accessible
                memory_total_mb=1024,  # Estimated
                temperature_c=None,
                power_draw_w=3.0,  # NPU typically very low power
                tasks_completed=0,
                avg_task_time_ms=0.0,
                current_batch_size=0,
                is_busy=False,
                last_update=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"NPU metrics error: {e}")
            return None
    
    def get_igpu_metrics(self) -> Optional[DeviceMetrics]:
        """Get iGPU metrics"""
        if not self.core or 'GPU' not in self.available_devices:
            return None
        
        try:
            return DeviceMetrics(
                device_name="Intel Integrated Graphics",
                device_type="intel_igpu",
                utilization_percent=self._estimate_igpu_utilization(),
                memory_used_mb=0,  # Shared with system memory
                memory_total_mb=4096,  # Estimated shared memory allocation
                temperature_c=None,
                power_draw_w=15.0,  # Estimated
                tasks_completed=0,
                avg_task_time_ms=0.0,
                current_batch_size=0,
                is_busy=False,
                last_update=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"iGPU metrics error: {e}")
            return None
    
    def _estimate_npu_utilization(self) -> float:
        """Estimate NPU utilization"""
        # Placeholder - actual utilization would need Intel NPU monitoring tools
        return 0.0
    
    def _estimate_igpu_utilization(self) -> float:
        """Estimate iGPU utilization"""
        # Placeholder - could use Windows Performance Counters
        return 0.0

class SystemMonitor:
    """System-wide monitor"""
    
    def get_shared_memory_usage(self) -> float:
        """Get shared memory usage in GB"""
        try:
            return psutil.virtual_memory().shared / (1024**3)
        except:
            return 0.0

# Global monitor instance
_monitor_instance = None

def get_performance_monitor() -> RealTimeMonitor:
    """Get or create global performance monitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RealTimeMonitor()
    return _monitor_instance

def start_monitoring():
    """Start performance monitoring"""
    monitor = get_performance_monitor()
    monitor.start()
    return monitor

def stop_monitoring():
    """Stop performance monitoring"""
    monitor = get_performance_monitor()
    monitor.stop()

def get_live_report() -> Dict[str, Any]:
    """Get live performance report"""
    monitor = get_performance_monitor()
    return monitor.get_performance_report()

if __name__ == "__main__":
    # Demo
    monitor = start_monitoring()
    
    try:
        for i in range(10):
            time.sleep(2)
            monitor.print_live_dashboard()
    finally:
        stop_monitoring()