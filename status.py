#!/usr/bin/env python3
"""
AI-Enhanced PDF Extraction System - Status Monitor
Überwacht Verarbeitungsfortschritt und AI System Status
"""

import json
import requests
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from supabase import create_client

class AISystemMonitor:
    def __init__(self):
        self.config = self.load_config()
        self.supabase = create_client(self.config['supabase_url'], self.config['supabase_key'])
        self.ollama_url = "http://localhost:11434"
        
    def load_config(self):
        config_file = Path("config.json")
        if not config_file.exists():
            print("❌ config.json nicht gefunden!")
            sys.exit(1)
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def check_ollama_status(self):
        """Check Ollama service and models status"""
        print("🤖 OLLAMA AI SERVICE STATUS")
        print("-" * 30)
        
        try:
            # Check service
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama Service: ONLINE")
                
                # Check models
                models_data = response.json()
                models = models_data.get('models', [])
                
                print(f"📊 Installierte Models: {len(models)}")
                
                required_models = ["llama3.1:8b", "llava:13b"]
                for req_model in required_models:
                    found = any(req_model in model['name'] for model in models)
                    status = "✅" if found else "❌"
                    print(f"   {status} {req_model}")
                
            else:
                print("❌ Ollama Service: OFFLINE")
                return False
                
        except requests.exceptions.RequestException:
            print("❌ Ollama Service: NICHT ERREICHBAR")
            return False
        
        print()
        return True
    
    def check_database_status(self):
        """Check database connection and statistics"""
        print("🗄️  DATABASE STATUS")
        print("-" * 30)
        
        try:
            # Test connection
            result = self.supabase.table("chunks").select("id").limit(1).execute()
            print("✅ Supabase: VERBUNDEN")
            
            # Get statistics
            chunks_count = self.supabase.table("chunks").select("id", count="exact").execute()
            images_count = self.supabase.table("images").select("id", count="exact").execute()
            processing_count = self.supabase.table("processing_logs").select("id", count="exact").execute()
            
            print(f"📊 Gesamt Chunks: {chunks_count.count}")
            print(f"📊 Gesamt Images: {images_count.count}")
            print(f"📊 Verarbeitete Dateien: {processing_count.count}")
            
            # Recent activity
            recent = self.supabase.table("processing_logs")\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(5)\
                .execute()
            
            if recent.data:
                print("\n📈 Letzte Aktivitäten:")
                for entry in recent.data:
                    status_icon = "✅" if entry['status'] == 'completed' else "❌" if entry['status'] == 'error' else "⏳"
                    filename = Path(entry['file_path']).name
                    created = entry['created_at'][:19].replace('T', ' ')
                    print(f"   {status_icon} {filename} - {created}")
            
        except Exception as e:
            print(f"❌ Database Error: {e}")
            return False
        
        print()
        return True
    
    def check_file_system(self):
        """Check file system and documents folder"""
        print("📁 FILE SYSTEM STATUS")
        print("-" * 30)
        
        docs_path = Path(self.config['documents_path'])
        
        if docs_path.exists():
            print(f"✅ Documents Folder: {docs_path}")
            
            # Count PDFs
            pdf_files = list(docs_path.glob("**/*.pdf"))
            print(f"📄 PDF Dateien gefunden: {len(pdf_files)}")
            
            if pdf_files:
                print("📋 PDF Dateien:")
                for pdf in pdf_files[:10]:  # Show first 10
                    size_mb = pdf.stat().st_size / 1024 / 1024
                    print(f"   📄 {pdf.name} ({size_mb:.1f} MB)")
                
                if len(pdf_files) > 10:
                    print(f"   ... und {len(pdf_files) - 10} weitere")
        else:
            print(f"❌ Documents Folder nicht gefunden: {docs_path}")
            return False
        
        print()
        return True
    
    def test_ai_processing(self):
        """Test AI processing capabilities"""
        print("🧪 AI PROCESSING TEST")
        print("-" * 30)
        
        # Test text generation
        try:
            text_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.config.get('text_model', 'llama3.1:8b'),
                    "prompt": "Analyze this text: 'Error Code C2557 - Replace toner cartridge'",
                    "stream": False
                },
                timeout=30
            )
            
            if text_response.status_code == 200:
                print("✅ Text Analysis: FUNKTIONIERT")
            else:
                print("❌ Text Analysis: FEHLER")
        except Exception as e:
            print(f"❌ Text Analysis Error: {e}")
        
        # Test vision processing
        try:
            vision_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.config.get('vision_model', 'llava:13b'),
                    "prompt": "What type of document page is this?",
                    "images": ["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="],
                    "stream": False
                },
                timeout=30
            )
            
            if vision_response.status_code == 200:
                print("✅ Vision Analysis: FUNKTIONIERT")
            else:
                print("❌ Vision Analysis: FEHLER")
        except Exception as e:
            print(f"❌ Vision Analysis Error: {e}")
        
        print()
    
    def show_processing_stats(self):
        """Show detailed processing statistics"""
        print("📊 VERARBEITUNGS-STATISTIKEN")
        print("-" * 30)
        
        try:
            # Manufacturer breakdown
            manufacturer_stats = self.supabase.rpc('get_manufacturer_stats').execute()
            
            if manufacturer_stats.data:
                print("📈 Nach Hersteller:")
                for stat in manufacturer_stats.data:
                    print(f"   📊 {stat['manufacturer']}: {stat['count']} Chunks")
            
            # Document type breakdown
            doc_type_stats = self.supabase.rpc('get_document_type_stats').execute()
            
            if doc_type_stats.data:
                print("\n📈 Nach Dokument-Typ:")
                for stat in doc_type_stats.data:
                    print(f"   📊 {stat['document_type']}: {stat['count']} Chunks")
            
            # Error codes found
            error_codes = self.supabase.table("chunks")\
                .select("error_codes")\
                .not_.is_("error_codes", "null")\
                .limit(100)\
                .execute()
            
            all_codes = []
            for chunk in error_codes.data:
                if chunk['error_codes']:
                    all_codes.extend(chunk['error_codes'])
            
            unique_codes = list(set(all_codes))
            print(f"\n🔍 Error Codes gefunden: {len(unique_codes)}")
            if unique_codes:
                print("   Beispiele:", ", ".join(unique_codes[:10]))
            
        except Exception as e:
            print(f"⚠️  Erweiterte Statistiken nicht verfügbar: {e}")
        
        print()
    
    def run_continuous_monitor(self):
        """Run continuous monitoring"""
        print("🔄 KONTINUIERLICHES MONITORING")
        print("-" * 30)
        print("Drücken Sie Ctrl+C zum Beenden...\n")
        
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H")
                
                print("=" * 70)
                print(f"    AI-ENHANCED PDF SYSTEM MONITOR - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 70)
                
                self.check_ollama_status()
                self.check_database_status()
                
                # Show recent activity
                try:
                    recent = self.supabase.table("processing_logs")\
                        .select("*")\
                        .order("updated_at", desc=True)\
                        .limit(3)\
                        .execute()
                    
                    if recent.data:
                        print("🔄 AKTUELLE AKTIVITÄT")
                        print("-" * 30)
                        for entry in recent.data:
                            status_icon = "✅" if entry['status'] == 'completed' else "❌" if entry['status'] == 'error' else "⏳"
                            filename = Path(entry['file_path']).name
                            updated = entry['updated_at'][:19].replace('T', ' ')
                            chunks = entry.get('chunks_created', 0)
                            print(f"   {status_icon} {filename}")
                            print(f"      Status: {entry['status']} | Chunks: {chunks} | {updated}")
                        print()
                except:
                    pass
                
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring beendet")
    
    def run_full_status(self):
        """Run complete status check"""
        print("=" * 70)
        print("    AI-ENHANCED PDF EXTRACTION SYSTEM - STATUS")
        print("=" * 70)
        
        all_good = True
        
        all_good &= self.check_ollama_status()
        all_good &= self.check_database_status()
        all_good &= self.check_file_system()
        
        self.test_ai_processing()
        self.show_processing_stats()
        
        print("=" * 70)
        if all_good:
            print("✅ SYSTEM STATUS: ALLE KOMPONENTEN FUNKTIONSFÄHIG")
        else:
            print("⚠️  SYSTEM STATUS: PROBLEME ERKANNT")
        print("=" * 70)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitor = AISystemMonitor()
        monitor.run_continuous_monitor()
    else:
        monitor = AISystemMonitor()
        monitor.run_full_status()

if __name__ == "__main__":
    main()
