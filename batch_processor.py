#!/usr/bin/env python3
"""
Batch PDF Processing with Full AI Power
Optimiert für Apple Silicon und maximale Performance
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

def find_all_pdfs():
    """Finde alle PDF-Dateien im Documents-Ordner"""
    pdf_files = []
    for root, dirs, files in os.walk('Documents'):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def process_single_pdf(pdf_path):
    """Verarbeite eine einzelne PDF mit launch.py"""
    try:
        print(f"🔄 Processing: {pdf_path}")
        start_time = time.time()
        
        # Verwende launch.py für die Verarbeitung
        result = os.system(f'python3 launch.py process "{pdf_path}"')
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result == 0:
            print(f"✅ Success: {pdf_path} ({duration:.1f}s)")
            return {"status": "success", "file": pdf_path, "duration": duration}
        else:
            print(f"❌ Failed: {pdf_path} ({duration:.1f}s)")
            return {"status": "failed", "file": pdf_path, "duration": duration}
            
    except Exception as e:
        print(f"💥 Error processing {pdf_path}: {str(e)}")
        return {"status": "error", "file": pdf_path, "error": str(e)}

def main():
    print("🚀 BATCH PDF PROCESSING - FULL AI POWER")
    print("=" * 60)
    
    # Finde alle PDFs
    pdf_files = find_all_pdfs()
    print(f"📋 Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        print("❌ No PDF files found in Documents folder")
        return
    
    # Apple Silicon optimiert: 6 Worker für beste Performance
    max_workers = min(6, mp.cpu_count())
    print(f"⚡ Using {max_workers} parallel workers")
    print(f"🖥️  Optimized for Apple Silicon GPU acceleration")
    print("-" * 60)
    
    start_time = time.time()
    results = []
    completed = 0
    
    # Parallele Verarbeitung mit ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Starte alle Tasks
        future_to_pdf = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}
        
        # Verarbeite Ergebnisse sobald sie fertig sind
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Zeige Fortschritt
                progress = (completed / len(pdf_files)) * 100
                elapsed = time.time() - start_time
                avg_time = elapsed / completed if completed > 0 else 0
                remaining = (len(pdf_files) - completed) * avg_time
                
                print(f"📊 Progress: {completed}/{len(pdf_files)} ({progress:.1f}%) | "
                      f"Avg: {avg_time:.1f}s | ETA: {remaining/60:.1f}min")
                
            except Exception as exc:
                print(f"💥 Exception for {pdf}: {exc}")
                results.append({"status": "exception", "file": pdf, "error": str(exc)})
    
    # Finale Statistiken
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    print("\n" + "=" * 60)
    print("📈 BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {successful}/{len(pdf_files)}")
    print(f"❌ Failed: {failed}/{len(pdf_files)}")
    print(f"⚡ Average: {total_time/len(pdf_files):.1f}s per file")
    print(f"🔥 Throughput: {len(pdf_files)/(total_time/60):.1f} files/minute")
    
    # Speichere Ergebnisse
    results_file = f"batch_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "total_time_seconds": total_time,
            "results": results
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()