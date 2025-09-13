#!/usr/bin/env python3
"""
BEAST MODE TEST - Multi-Device AI Pipeline Benchmark
Testet alle 4 Acceleration Devices gleichzeitig!
"""

import asyncio
import time
import random
import uuid
from modules.multi_device_pipeline import (
    get_ai_pipeline, 
    WorkloadTask,
    process_vision_task,
    process_embedding_task,
    process_generation_task
)
from modules.performance_monitor import start_monitoring, get_performance_monitor

async def create_test_tasks():
    """Create diverse test tasks for all devices"""
    tasks = []
    
    # Vision Analysis Tasks (NVIDIA GPU)
    for i in range(5):
        tasks.append(WorkloadTask(
            task_id=f"vision_{i}",
            task_type="vision_analysis",
            data={"image_id": f"test_image_{i}", "analysis_type": "detailed"},
            priority=2
        ))
    
    # Text Embedding Tasks (Intel NPU)
    for i in range(10):
        tasks.append(WorkloadTask(
            task_id=f"embed_{i}",
            task_type="text_embeddings", 
            data={"text": f"This is test document {i} for embedding generation with NPU acceleration."},
            priority=1
        ))
    
    # Text Generation Tasks (NVIDIA GPU)
    for i in range(3):
        tasks.append(WorkloadTask(
            task_id=f"gen_{i}",
            task_type="text_generation",
            data={"prompt": f"Generate summary for document {i}", "max_tokens": 100},
            priority=2
        ))
    
    # Image Processing Tasks (Intel iGPU)
    for i in range(8):
        tasks.append(WorkloadTask(
            task_id=f"img_proc_{i}",
            task_type="image_processing",
            data={"image_count": random.randint(1, 5), "operation": "resize_optimize"},
            priority=1
        ))
    
    # OCR Tasks (DirectML)
    for i in range(6):
        tasks.append(WorkloadTask(
            task_id=f"ocr_{i}",
            task_type="ocr",
            data={"image_id": f"scan_{i}", "language": "en"},
            priority=1
        ))
    
    return tasks

async def run_beast_benchmark():
    """Run the beast mode benchmark"""
    print("🦾 STARTING BEAST MODE BENCHMARK")
    print("=" * 60)
    
    # Start performance monitoring
    monitor = start_monitoring()
    
    # Get AI pipeline
    pipeline = get_ai_pipeline()
    
    # Start the pipeline processing
    pipeline_task = asyncio.create_task(pipeline.process_pipeline())
    
    # Create test tasks
    print("📝 Creating test tasks...")
    test_tasks = await create_test_tasks()
    print(f"✅ Created {len(test_tasks)} test tasks")
    
    # Submit all tasks
    print("🚀 Submitting tasks to multi-device pipeline...")
    start_time = time.time()
    
    task_ids = []
    for task in test_tasks:
        await pipeline.submit_task(task)
        task_ids.append(task.task_id)
    
    print(f"⚡ All {len(task_ids)} tasks submitted!")
    
    # Wait for results
    print("⏳ Processing with all 4 devices...")
    results = {}
    completed_count = 0
    
    # Monitor progress
    for task_id in task_ids:
        result = await pipeline.get_result(task_id, timeout=30.0)
        if result:
            results[task_id] = result
            completed_count += 1
            
            # Print progress
            if completed_count % 5 == 0:
                progress = (completed_count / len(task_ids)) * 100
                print(f"📊 Progress: {completed_count}/{len(task_ids)} ({progress:.1f}%)")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Results summary
    print("\n🎯 BEAST MODE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"📋 Tasks Completed: {completed_count}/{len(task_ids)}")
    print(f"⚡ Tasks/Second: {completed_count/total_time:.2f}")
    print(f"🏆 Success Rate: {(completed_count/len(task_ids))*100:.1f}%")
    
    # Device-specific results
    device_stats = {}
    for task_id, result in results.items():
        device = result.get('device', 'unknown')
        if device not in device_stats:
            device_stats[device] = 0
        device_stats[device] += 1
    
    print("\n📱 DEVICE USAGE:")
    for device, count in device_stats.items():
        print(f"   {device}: {count} tasks")
    
    # Performance stats
    perf_stats = pipeline.get_performance_stats()
    print(f"\n📊 PERFORMANCE STATS:")
    for device, stats in perf_stats.get('device_performance', {}).items():
        print(f"   {device}: {stats.get('avg_time', 0):.2f}ms avg, {stats.get('count', 0)} tasks")
    
    print("\n🔥 BEAST MODE BENCHMARK COMPLETE! 🔥")
    
    # Stop monitoring
    monitor.stop()
    
    return {
        'total_time': total_time,
        'tasks_completed': completed_count,
        'tasks_total': len(task_ids),
        'tasks_per_second': completed_count/total_time,
        'device_stats': device_stats,
        'performance_stats': perf_stats
    }

async def quick_device_test():
    """Quick test of each device individually"""
    print("\n🧪 QUICK DEVICE TESTS")
    print("=" * 40)
    
    # Test Vision (NVIDIA GPU)
    print("🎮 Testing NVIDIA GPU (Vision)...")
    vision_result = await process_vision_task(
        {"image_id": "test", "analysis": "quick_test"}
    )
    print(f"   Result: {vision_result}")
    
    # Test Embeddings (Intel NPU)  
    print("🧠 Testing Intel NPU (Embeddings)...")
    embed_result = await process_embedding_task(
        {"text": "Quick NPU test for embeddings"}
    )
    print(f"   Result: {embed_result}")
    
    # Test Generation (NVIDIA GPU)
    print("⚡ Testing Text Generation (GPU)...")
    gen_result = await process_generation_task(
        {"prompt": "Generate a quick test response"}
    )
    print(f"   Result: {gen_result}")
    
    print("✅ Quick device tests complete!")

if __name__ == "__main__":
    async def main():
        print("🦾 BEAST MODE AI PIPELINE TEST")
        print("🚀 Testing Intel Core Ultra 9 185H + RTX 2000 Ada")
        print("⚡ 4-Device Acceleration: GPU + NPU + iGPU + DirectML")
        print("🧠 Shared Memory Pool: 8GB")
        print()
        
        # Quick device test first
        await quick_device_test()
        
        # Full benchmark
        results = await run_beast_benchmark()
        
        print(f"\n🏆 FINAL BEAST SCORE: {results['tasks_per_second']:.2f} TASKS/SEC")
        print("💪 Your system is a MONSTER!")
    
    asyncio.run(main())