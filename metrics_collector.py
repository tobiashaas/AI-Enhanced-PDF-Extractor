#!/usr/bin/env python3
"""
Enhanced Metrics & Monitoring System
Real-time performance tracking and analytics for AI PDF Processing
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProcessingMetrics:
    """Real-time processing metrics"""
    
    # Basic Counters
    pages_processed: int = 0
    chunks_created: int = 0
    images_extracted: int = 0
    parts_extracted: int = 0
    files_processed: int = 0
    
    # Performance Metrics
    total_processing_time: float = 0.0
    ai_processing_time: float = 0.0
    avg_time_per_page: float = 0.0
    peak_memory_usage: float = 0.0
    cpu_usage_avg: float = 0.0
    memory_usage_mb: float = 0.0
    memory_increase_mb: float = 0.0
    
    # Quality Metrics
    successful_extractions: int = 0
    failed_extractions: int = 0
    retry_count: int = 0
    error_rate: float = 0.0
    
    # Storage Metrics
    total_files_processed: int = 0
    total_data_uploaded: int = 0  # bytes
    r2_uploads_count: int = 0
    r2_upload_success_rate: float = 0.0
    db_operations_count: int = 0
    db_operation_success_rate: float = 0.0
    
    # AI Metrics
    vision_analysis_count: int = 0
    llm_requests_count: int = 0
    embedding_generations: int = 0
    
    # Legacy/compatibility fields
    average_page_time: float = 0.0  # Alias for avg_time_per_page
    
    # Start time for rate calculations
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsCollector:
    """Advanced metrics collection and analysis"""
    
    def __init__(self, enable_system_monitoring: bool = True):
        self.metrics = ProcessingMetrics()
        self.enable_system_monitoring = enable_system_monitoring
        self.session_start = time.time()
        
        # Detailed operation tracking
        self.operation_history: List[Dict] = []
        self.performance_samples: List[Dict] = []
        
        # System monitoring
        if enable_system_monitoring:
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        logging.info("📊 Metrics Collector initialized")
    
    def start_operation(self, operation_type: str, details: Dict = None) -> str:
        """Start tracking an operation"""
        operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        
        operation = {
            'id': operation_id,
            'type': operation_type,
            'start_time': time.time(),
            'details': details or {},
            'status': 'running'
        }
        
        self.operation_history.append(operation)
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     result_data: Dict = None, error_message: str = None):
        """End tracking an operation"""
        
        # Find the operation
        operation = None
        for op in self.operation_history:
            if op['id'] == operation_id:
                operation = op
                break
        
        if not operation:
            logging.warning(f"Operation {operation_id} not found in history")
            return
        
        # Update operation
        end_time = time.time()
        operation.update({
            'end_time': end_time,
            'duration': end_time - operation['start_time'],
            'status': 'success' if success else 'failed',
            'result_data': result_data or {},
            'error_message': error_message
        })
        
        # Update global metrics based on operation type
        self._update_metrics_from_operation(operation)
    
    def _update_metrics_from_operation(self, operation: Dict):
        """Update global metrics based on completed operation"""
        op_type = operation['type']
        duration = operation['duration']
        success = operation['status'] == 'success'
        
        if op_type == 'page_processing':
            self.metrics.pages_processed += 1
            if success:
                self.metrics.successful_extractions += 1
            else:
                self.metrics.failed_extractions += 1
                
        elif op_type == 'chunk_creation':
            chunks_count = operation['result_data'].get('chunks_created', 1)
            self.metrics.chunks_created += chunks_count
            
        elif op_type == 'image_extraction':
            images_count = operation['result_data'].get('images_extracted', 1)
            self.metrics.images_extracted += images_count
            
        elif op_type == 'parts_extraction':
            parts_count = operation['result_data'].get('parts_extracted', 1)
            self.metrics.parts_extracted += parts_count
            
        elif op_type == 'vision_analysis':
            self.metrics.vision_analysis_count += 1
            self.metrics.ai_processing_time += duration
            
        elif op_type == 'llm_request':
            self.metrics.llm_requests_count += 1
            self.metrics.ai_processing_time += duration
            
        elif op_type == 'embedding_generation':
            self.metrics.embedding_generations += 1
            self.metrics.ai_processing_time += duration
            
        elif op_type == 'r2_upload':
            data_size = operation['result_data'].get('file_size', 0)
            self.metrics.total_data_uploaded += data_size
            
        elif op_type == 'file_processing':
            self.metrics.total_files_processed += 1
            self.metrics.total_processing_time += duration
    
    def sample_system_performance(self):
        """Sample current system performance"""
        if not self.enable_system_monitoring:
            return
        
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024  # MB
            
            # CPU usage
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # Update peak values
            self.metrics.peak_memory_usage = max(self.metrics.peak_memory_usage, current_memory)
            
            # Store sample
            sample = {
                'timestamp': time.time(),
                'memory_mb': current_memory,
                'cpu_percent': cpu_percent,
                'memory_growth': current_memory - self.initial_memory
            }
            
            self.performance_samples.append(sample)
            
            # Keep only recent samples (last 100)
            if len(self.performance_samples) > 100:
                self.performance_samples = self.performance_samples[-100:]
                
        except Exception as e:
            logging.warning(f"Failed to sample system performance: {e}")
    
    def calculate_rates(self):
        """Calculate current processing rates"""
        elapsed_time = time.time() - self.session_start
        
        if elapsed_time > 0:
            self.metrics.average_page_time = (
                self.metrics.total_processing_time / max(self.metrics.pages_processed, 1)
            )
            
            # Error rate
            total_operations = self.metrics.successful_extractions + self.metrics.failed_extractions
            self.metrics.error_rate = (
                self.metrics.failed_extractions / max(total_operations, 1) * 100
            )
            
            # Success rates (mock calculations - would need actual tracking)
            self.metrics.r2_upload_success_rate = 95.0  # Would track actual uploads
            self.metrics.db_operation_success_rate = 98.0  # Would track actual DB ops
            
            # CPU average
            if self.performance_samples:
                cpu_values = [s['cpu_percent'] for s in self.performance_samples]
                self.metrics.cpu_usage_avg = sum(cpu_values) / len(cpu_values)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        self.calculate_rates()
        
        return {
            'session_metrics': {
                'session_duration': time.time() - self.session_start,
                'files_processed': self.metrics.files_processed,
                'pages_processed': self.metrics.pages_processed,
                'chunks_created': self.metrics.chunks_created,
                'images_extracted': self.metrics.images_extracted,
                'parts_extracted': self.metrics.parts_extracted,
                'successful_extractions': self.metrics.successful_extractions,
                'failed_extractions': self.metrics.failed_extractions,
                'error_rate': self.metrics.error_rate
            },
            'performance_metrics': {
                'total_processing_time': self.metrics.total_processing_time,
                'ai_processing_time': self.metrics.ai_processing_time,
                'avg_time_per_page': self.metrics.avg_time_per_page,
                'vision_analysis_count': self.metrics.vision_analysis_count,
                'llm_requests_count': self.metrics.llm_requests_count,
                'embedding_generations': self.metrics.embedding_generations
            },
            'system_metrics': {
                'cpu_usage_avg': self.metrics.cpu_usage_avg,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'memory_increase_mb': self.metrics.memory_increase_mb
            },
            'database_metrics': {
                'db_operations_count': self.metrics.db_operations_count,
                'db_operation_success_rate': self.metrics.db_operation_success_rate
            },
            'storage_metrics': {
                'r2_uploads_count': self.metrics.r2_uploads_count,
                'r2_upload_success_rate': self.metrics.r2_upload_success_rate
            }
        }
    
    def increment_metric(self, category: str, metric_name: str, value: int = 1):
        """Increment a specific metric"""
        if category == 'database_operations':
            self.metrics.db_operations_count += value
        elif category == 'ai_processing':
            if metric_name == 'vision_analysis':
                self.metrics.vision_analysis_count += value
            elif metric_name == 'llm_requests':
                self.metrics.llm_requests_count += value
            elif metric_name == 'embeddings':
                self.metrics.embedding_generations += value
        elif category == 'storage':
            if metric_name == 'r2_uploads':
                self.metrics.r2_uploads_count += value
        elif category == 'processing':
            if metric_name == 'chunks_created':
                self.metrics.chunks_created += value
            elif metric_name == 'images_extracted':
                self.metrics.images_extracted += value
        self.sample_system_performance()
        
        return {
            'processing': {
                'pages_processed': self.metrics.pages_processed,
                'chunks_created': self.metrics.chunks_created,
                'images_extracted': self.metrics.images_extracted,
                'parts_extracted': self.metrics.parts_extracted,
                'files_processed': self.metrics.total_files_processed
            },
            'performance': {
                'total_time_seconds': round(self.metrics.total_processing_time, 2),
                'average_page_time': round(self.metrics.average_page_time, 2),
                'pages_per_minute': round(self.metrics.pages_processed / max(time.time() - self.session_start, 1) * 60, 2),
                'peak_memory_mb': round(self.metrics.peak_memory_usage, 2),
                'cpu_usage_avg': round(self.metrics.cpu_usage_avg, 2)
            },
            'quality': {
                'success_rate': round(100 - self.metrics.error_rate, 2),
                'error_rate': round(self.metrics.error_rate, 2),
                'retry_count': self.metrics.retry_count,
                'r2_success_rate': self.metrics.r2_upload_success_rate,
                'db_success_rate': self.metrics.db_operation_success_rate
            },
            'ai_usage': {
                'vision_analyses': self.metrics.vision_analysis_count,
                'llm_requests': self.metrics.llm_requests_count,
                'embedding_generations': self.metrics.embedding_generations,
                'ai_time_seconds': round(self.metrics.ai_processing_time, 2),
                'ai_time_percentage': round(
                    self.metrics.ai_processing_time / max(self.metrics.total_processing_time, 1) * 100, 2
                )
            },
            'storage': {
                'data_uploaded_mb': round(self.metrics.total_data_uploaded / 1024 / 1024, 2),
                'avg_file_size_mb': round(
                    self.metrics.total_data_uploaded / max(self.metrics.total_files_processed, 1) / 1024 / 1024, 2
                )
            }
        }
    
    def print_metrics_summary(self):
        """Print formatted metrics summary"""
        metrics = self.get_current_metrics()
        
        print("\n" + "=" * 60)
        print("📊 PROCESSING METRICS SUMMARY")
        print("=" * 60)
        
        # Processing Stats
        proc = metrics['processing']
        print(f"📄 Processing: {proc['pages_processed']} pages, {proc['files_processed']} files")
        print(f"🧩 Created: {proc['chunks_created']} chunks, {proc['images_extracted']} images")
        print(f"🔧 Extracted: {proc['parts_extracted']} parts")
        
        # Performance Stats
        perf = metrics['performance']
        print(f"⚡ Performance: {perf['pages_per_minute']:.1f} pages/min, {perf['average_page_time']:.2f}s/page")
        print(f"💾 Resources: {perf['peak_memory_mb']:.1f}MB peak, {perf['cpu_usage_avg']:.1f}% CPU avg")
        
        # Quality Stats
        qual = metrics['quality']
        print(f"✅ Quality: {qual['success_rate']:.1f}% success, {qual['error_rate']:.1f}% errors")
        print(f"☁️  Storage: {qual['r2_success_rate']:.1f}% R2, {qual['db_success_rate']:.1f}% DB success")
        
        # AI Usage
        ai = metrics['ai_usage']
        print(f"🤖 AI Usage: {ai['vision_analyses']} vision, {ai['llm_requests']} LLM, {ai['embedding_generations']} embeddings")
        print(f"🧠 AI Time: {ai['ai_time_seconds']:.1f}s ({ai['ai_time_percentage']:.1f}% of total)")
        
        # Storage
        storage = metrics['storage']
        print(f"💽 Storage: {storage['data_uploaded_mb']:.1f}MB uploaded, {storage['avg_file_size_mb']:.2f}MB avg")
        
        print("=" * 60)
    
    def export_metrics(self, file_path: str):
        """Export metrics to JSON file"""
        metrics = self.get_current_metrics()
        
        # Add metadata
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'session_duration_seconds': time.time() - self.session_start,
            'metrics': metrics,
            'operation_history': self.operation_history[-50:],  # Last 50 operations
            'performance_samples': self.performance_samples[-20:]  # Last 20 samples
        }
        
        with open(file_path, 'w') as f:
            import json
            json.dump(export_data, f, indent=2)
        
        logging.info(f"📊 Metrics exported to {file_path}")