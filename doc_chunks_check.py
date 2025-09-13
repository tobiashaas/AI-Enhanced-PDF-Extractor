#!/usr/bin/env python3
"""
Skript zur Überprüfung der Dokument-Chunks und Dokumentarten in der Datenbank
"""

import os
import sys
import logging
from dotenv import load_dotenv
from supabase import create_client

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("doc_chunks_check.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger initialisieren
logger = logging.getLogger(__name__)

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

def init_supabase():
    """Initialisiert die Supabase-Verbindung"""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        logger.error("SUPABASE_URL oder SUPABASE_KEY fehlen in der Umgebung")
        return None
    
    try:
        client = create_client(url, key)
        logger.info("Supabase-Verbindung hergestellt")
        return client
    except Exception as e:
        logger.error(f"Fehler bei der Supabase-Verbindung: {e}")
        return None

def check_document_tables():
    """Prüft die verschiedenen Dokument-Tabellen"""
    supabase = init_supabase()
    if not supabase:
        logger.error("Keine Verbindung zur Datenbank")
        return
        
    print("=== DOKUMENTEN-TABELLEN ÜBERPRÜFUNG ===")
    
    # Liste der zu überprüfenden Tabellen
    tables = [
        "service_manuals",
        "parts_catalogs",
        "bulletins",
        "cpmd_documents",
        "document_chunks"
    ]
    
    for table in tables:
        try:
            # Prüfe, ob die Tabelle existiert (durch Abruf eines einzelnen Eintrags)
            result = supabase.table(table).select("*").limit(1).execute()
            print(f"✅ {table}: {len(result.data)} Beispieleinträge abgerufen")
                
            # Wenn Einträge vorhanden sind, zeige Beispiel
            if len(result.data) > 0:
                print(f"   Beispiel-ID: {result.data[0].get('id', 'keine ID')}")
                
                # Zeige verfügbare Spalten
                columns = list(result.data[0].keys())
                print(f"   Spalten: {', '.join(columns)}")
                
                # Zähle alle Einträge
                try:
                    count = supabase.table(table).select("id").execute()
                    print(f"   Gesamtanzahl: {len(count.data)} Einträge")
                except Exception as count_err:
                    print(f"   Fehler beim Zählen: {count_err}")
        except Exception as e:
            print(f"❌ {table}: Fehler oder Tabelle nicht vorhanden ({str(e)})")
    
    # Überprüfe die Verarbeitungslogs
    try:
        logs_result = supabase.table("processing_logs").select("*").limit(5).execute()
        if logs_result.data:
            print(f"\n=== LETZTE VERARBEITUNGS-LOGS ({len(logs_result.data)}) ===")
            for log in logs_result.data:
                print(f"- ID: {log.get('id')}, Status: {log.get('status')}")
                print(f"  Datei: {log.get('original_filename', 'unbekannt')}")
                print(f"  Typ: {log.get('document_type', 'unbekannt')}, Chunks: {log.get('chunks_created', 0)}, Bilder: {log.get('images_extracted', 0)}")
                print(f"  Start: {log.get('started_at', 'unbekannt')}, Ende: {log.get('completed_at', 'unbekannt')}")
        else:
            print("\nKeine Verarbeitungs-Logs gefunden")
    except Exception as e:
        print(f"Fehler beim Abrufen der Verarbeitungslogs: {e}")
        
    # Überprüfe die images Tabelle nach Kategorisierung
    try:
        print("\n=== BILDER NACH QUELLE ===")
        result = supabase.table("images").select("source_table, count(*)").execute()
        print("Fehler: count() wird nicht unterstützt")
    except Exception:
        try:
            result = supabase.table("images").select("*").execute()
            print(f"Anzahl Bilder insgesamt: {len(result.data)}")
            
            # Kategorisierung manuell durchführen
            source_tables = {}
            for img in result.data:
                source = img.get("source_table", "unbekannt")
                if source not in source_tables:
                    source_tables[source] = 0
                source_tables[source] += 1
                
            for source, count in source_tables.items():
                print(f"- {source}: {count} Bilder")
                
            # Prüfe, ob Hersteller und Modell gesetzt sind
            manufacturers = {}
            models = {}
            for img in result.data:
                mfr = img.get("manufacturer", "unbekannt")
                if mfr not in manufacturers:
                    manufacturers[mfr] = 0
                manufacturers[mfr] += 1
                
                mdl = img.get("model", "unbekannt")
                if mdl not in models:
                    models[mdl] = 0
                models[mdl] += 1
                
            print("\n=== BILDER NACH HERSTELLER ===")
            for mfr, count in manufacturers.items():
                print(f"- {mfr}: {count} Bilder")
                
            print("\n=== BILDER NACH MODELL ===")
            for mdl, count in models.items():
                print(f"- {mdl}: {count} Bilder")
        except Exception as e:
            print(f"Fehler beim Analysieren der Bilder-Tabelle: {e}")
    
    print("\n=== PRÜFUNG ABGESCHLOSSEN ===")

if __name__ == "__main__":
    check_document_tables()