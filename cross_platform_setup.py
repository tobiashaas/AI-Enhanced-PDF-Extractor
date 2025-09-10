#!/usr/bin/env python3
"""
Cross-Platform Setup für AI-Enhanced PDF Extractor
Automatisches Setup für Windows/macOS/Linux
"""

import os
import sys
import json
import platform
from pathlib import Path

def detect_platform():
    """Detect the operating system"""
    system = platform.system().lower()
    if system == "darwin":
        return "macOS"
    elif system == "windows":
        return "Windows"
    elif system == "linux":
        return "Linux"
    else:
        return "Unknown"

def get_default_documents_path():
    """Get default documents path for each platform"""
    system = platform.system().lower()
    home = Path.home()
    
    if system == "darwin":  # macOS
        return str(home / "Documents" / "PDFs")
    elif system == "windows":
        return str(home / "Documents" / "PDFs")
    elif system == "linux":
        return str(home / "Documents" / "PDFs")
    else:
        return "./Documents"

def setup_config():
    """Setup configuration for the current platform"""
    print("🚀 Cross-Platform Setup für AI-Enhanced PDF Extractor")
    print("=" * 60)
    
    # Detect platform
    current_platform = detect_platform()
    print(f"🖥️  Erkanntes System: {current_platform}")
    
    # Check if config already exists
    if os.path.exists('config.json'):
        print("⚠️  config.json bereits vorhanden!")
        overwrite = input("Überschreiben? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("❌ Setup abgebrochen")
            return False
    
    # Load template
    if not os.path.exists('config.example.json'):
        print("❌ config.example.json nicht gefunden!")
        print("   Bitte Repository vollständig klonen")
        return False
    
    print("📋 Lade Konfiguration-Template...")
    with open('config.example.json', 'r') as f:
        config = json.load(f)
    
    # Remove comments (keys starting with _)
    clean_config = {k: v for k, v in config.items() if not k.startswith('_')}
    
    # Set platform-specific defaults
    default_docs_path = get_default_documents_path()
    print(f"📁 Standard Dokumente-Pfad: {default_docs_path}")
    
    # Ask for documents path
    user_docs_path = input(f"PDF-Dokumente Pfad [{default_docs_path}]: ").strip()
    if not user_docs_path:
        user_docs_path = default_docs_path
    
    clean_config['documents_path'] = user_docs_path
    
    # Create documents directory if it doesn't exist
    docs_path = Path(user_docs_path)
    if not docs_path.exists():
        try:
            docs_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Dokumente-Ordner erstellt: {user_docs_path}")
        except Exception as e:
            print(f"⚠️ Konnte Ordner nicht erstellen: {e}")
    
    # Platform-specific optimizations
    if current_platform == "macOS":
        print("🍎 Apple Silicon Optimierungen werden durch performance_optimizer.py gesetzt")
    elif current_platform == "Windows":
        print("🪟 Windows/RTX Optimierungen werden durch performance_optimizer.py gesetzt")
    elif current_platform == "Linux":
        print("🐧 Linux Optimierungen werden durch performance_optimizer.py gesetzt")
    
    # Save config
    with open('config.json', 'w') as f:
        json.dump(clean_config, f, indent=2)
    
    print("✅ config.json erfolgreich erstellt!")
    
    # Show critical information
    print("\\n" + "="*60)
    print("🔴 WICHTIGE INFORMATIONEN:")
    print("="*60)
    print("✅ Supabase & R2 sind bereits konfiguriert")
    print("✅ Alle PCs teilen dieselbe Datenbank")
    print("✅ R2 Public Domain ist korrekt gesetzt")
    print("⚠️  NIEMALS Supabase/R2 Config ändern!")
    print("="*60)
    
    return True

def run_hardware_optimization():
    """Run hardware optimization"""
    print("\\n🔧 Starte Hardware-Optimierung...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'performance_optimizer.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Hardware-Optimierung erfolgreich!")
            print("📊 Optimale Einstellungen wurden automatisch gesetzt")
        else:
            print("⚠️ Hardware-Optimierung fehlgeschlagen:")
            print(result.stderr)
            
    except Exception as e:
        print(f"⚠️ Konnte Hardware-Optimierung nicht ausführen: {e}")
        print("   Führe manuell aus: python3 performance_optimizer.py")

def verify_setup():
    """Verify the setup is correct"""
    print("\\n🔍 Überprüfe Setup...")
    
    # Check config
    if not os.path.exists('config.json'):
        print("❌ config.json nicht gefunden")
        return False
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Check critical values
    required_keys = [
        'supabase_url', 'supabase_key', 
        'r2_account_id', 'r2_public_domain_id',
        'documents_path'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Fehlender Wert: {key}")
            return False
    
    # Check r2_public_domain_id specifically
    if config['r2_public_domain_id'] != '80a63376fddf4b909ed55ee53a401a93':
        print("❌ r2_public_domain_id ist falsch!")
        print(f"   Erwartet: 80a63376fddf4b909ed55ee53a401a93")
        print(f"   Gefunden: {config['r2_public_domain_id']}")
        return False
    
    # Check documents path
    docs_path = Path(config['documents_path'])
    if not docs_path.exists():
        print(f"⚠️ Dokumente-Pfad existiert nicht: {docs_path}")
    
    print("✅ Setup-Verifikation erfolgreich!")
    return True

def main():
    """Main setup function"""
    try:
        # Setup config
        if not setup_config():
            return
        
        # Run hardware optimization
        run_hardware_optimization()
        
        # Verify setup
        if verify_setup():
            print("\\n🎉 SETUP ERFOLGREICH ABGESCHLOSSEN!")
            print("\\n📋 Nächste Schritte:")
            print("1. 📄 PDFs in den Dokumente-Ordner kopieren")
            print("2. 🚀 Verarbeitung starten: python3 ai_pdf_processor.py")
            print("3. 📊 Status prüfen: python3 status.py")
            
            # Show config summary
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            print("\\n📋 Konfiguration-Zusammenfassung:")
            print(f"   📁 Dokumente: {config['documents_path']}")
            print(f"   🧠 Vision Model: {config['vision_model']}")
            print(f"   💾 Database: Shared (alle PCs)")
            print(f"   ☁️  Storage: R2 Public Domain aktiv")
        
    except KeyboardInterrupt:
        print("\\n❌ Setup abgebrochen")
    except Exception as e:
        print(f"❌ Setup-Fehler: {e}")

if __name__ == "__main__":
    main()
