#!/usr/bin/env python3
"""
Test-Skript für die PDF Textextraktion und Chunk-Verarbeitung gemäß Supabase AI Schema
"""

import os
import sys
import fitz  # PyMuPDF
import logging
import hashlib
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chunk_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger initialisieren
logger = logging.getLogger(__name__)

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

# Import des Dokument-Processors
try:
    from modules.document_processing.processor import DocumentProcessor, ServiceManualProcessor
    from modules.r2_client import R2Client
    logger.info("Module erfolgreich importiert")
except ImportError as e:
    logger.error(f"Fehler beim Import der Module: {e}")
    sys.exit(1)
    
def load_config():
    """Lädt die Konfiguration"""
    import json
    try:
        with open("config.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        return {}

def init_supabase():
    """Initialisiert die Supabase-Verbindung"""
    from supabase import create_client
    
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

def check_tables(supabase):
    """Überprüft die existierenden Tabellen gemäß Supabase AI Dokumentation"""
    try:
        tables = [
            "service_manuals",
            "bulletins",
            "cpmd_documents",
            "parts_catalog",  # Singular laut Supabase AI
            "images"
        ]
        
        print("\n=== ÜBERPRÜFUNG DER TABELLEN ===")
        
        for table in tables:
            try:
                result = supabase.table(table).select("*").limit(1).execute()
                print(f"✅ Tabelle {table}: {len(result.data)} Beispieleinträge")
                
                # Prüfe Struktur, wenn vorhanden
                if len(result.data) > 0:
                    columns = list(result.data[0].keys())
                    print(f"   Spalten: {', '.join(columns)}")
            except Exception as e:
                print(f"❌ Tabelle {table}: Fehler oder nicht vorhanden - {e}")
        
        print("===============================")
    except Exception as e:
        logger.error(f"Fehler bei der Tabellenprüfung: {e}")

def process_document_test(pdf_path):
    """Test der Dokumentenverarbeitung mit Fokus auf Text-Chunks gemäß Supabase AI Schema"""
    logger.info(f"Verarbeite PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF nicht gefunden: {pdf_path}")
        return False
    
    # Konfiguration und Clients initialisieren
    config = load_config()
    supabase = init_supabase()
    r2_client = R2Client(config, supabase)
    
    if not supabase:
        logger.error("Kein Supabase-Client verfügbar")
        return False
        
    # Prüfe zuerst die Tabellen
    check_tables(supabase)
    
    try:
        # Dokument-Prozessor initialisieren
        processor = ServiceManualProcessor(supabase, r2_client, config)
        
        # PDF öffnen
        pdf_document = fitz.open(pdf_path)
        logger.info(f"PDF geöffnet: {len(pdf_document)} Seiten")
        
        # Textextraktion (1-5 Seiten)
        pages_data = []
        max_pages = min(5, len(pdf_document))
        
        for i in range(max_pages):
            page = pdf_document[i]
            text = page.get_text()
            pages_data.append({
                "page_number": i + 1,
                "text": text
            })
            logger.info(f"Seite {i+1}: {len(text)} Zeichen extrahiert")
        
        # Text-Chunks erstellen
        all_text = ""
        for page in pages_data:
            all_text += page["text"] + " "
        
        chunk_size = config.get("processing", {}).get("chunk_size", 1000)
        overlap = config.get("processing", {}).get("chunk_overlap", 200)
        chunks = processor.chunk_text(all_text, chunk_size, overlap)
        
        logger.info(f"{len(chunks)} Text-Chunks erstellt (Chunk-Größe: {chunk_size}, Überlappung: {overlap})")
        
        # Embeddings generieren versuchen
        try:
            # Limitiere auf die ersten 3 Chunks für den Test
            test_chunks = chunks[:3] if len(chunks) > 3 else chunks
            embeddings = processor.generate_embeddings(test_chunks)
            logger.info(f"{len(embeddings)} Embeddings generiert")
            
            if embeddings and len(embeddings[0]) > 0:
                logger.info(f"Embedding-Dimensionalität: {len(embeddings[0])}")
                print(f"✅ Embeddings generiert mit Dimensionalität: {len(embeddings[0])}")
                
                # Gemäß Supabase AI: Embedding-Dimension sollte 768 sein
                if len(embeddings[0]) == 768:
                    print("✅ Korrekte Embedding-Dimensionalität (768 für EmbeddingGemma)")
                else:
                    print(f"⚠️ Unerwartete Embedding-Dimensionalität: {len(embeddings[0])} (soll 768 sein)")
            else:
                logger.warning("Leere Embeddings generiert")
                print("⚠️ Leere Embeddings generiert")
                
        except Exception as embed_err:
            logger.error(f"Fehler bei der Embedding-Generierung: {embed_err}")
            print(f"❌ Embedding-Generierung fehlgeschlagen: {embed_err}")
            embeddings = [[0.0] * 768 for _ in test_chunks]  # Dummy-Embeddings
            
        # Erstelle ein minimales document_data für einen Test nach Supabase AI Schema
        metadata = processor.extract_metadata(pdf_document)
        file_hash = hashlib.sha256(open(pdf_path, 'rb').read()).hexdigest()[:60]
        
        # Teste die Speicherung eines Service Manual Chunks direkt in service_manuals Tabelle
        try:
            if chunks and len(chunks) > 0:
                test_chunk = chunks[0]
                test_embedding = embeddings[0] if embeddings and len(embeddings) > 0 else [0.0] * 768
                
                # Gemäß Supabase AI Schema: direkt in service_manuals einfügen
                result = supabase.table("service_manuals").insert({
                    "content": test_chunk,
                    "chunk_index": 0,
                    "page_number": 1,
                    "file_hash": file_hash,
                    "embedding": test_embedding,
                    "manufacturer": "HP",
                    "model": "X580"
                }).execute()
                
                if result.data:
                    chunk_id = result.data[0].get("id", "unbekannt")
                    logger.info(f"Test-Chunk erfolgreich in service_manuals gespeichert: {chunk_id}")
                    print(f"✅ Test-Chunk erfolgreich in service_manuals gespeichert mit ID: {chunk_id}")
                    
                    # Lösche den Test-Chunk wieder
                    supabase.table("service_manuals").delete().eq("id", chunk_id).execute()
                    logger.info("Test-Chunk wurde gelöscht")
                    print("✅ Test-Chunk erfolgreich gelöscht")
                else:
                    logger.warning("Chunk-Speicherung fehlgeschlagen: Keine Daten zurückgegeben")
                    print("❌ Chunk konnte nicht gespeichert werden (keine Daten zurückgegeben)")
        except Exception as chunk_err:
            logger.error(f"Fehler beim Testen der service_manuals Tabelle: {chunk_err}")
            print(f"❌ Fehler beim Speichern in service_manuals: {chunk_err}")
        
        # PDF schließen
        pdf_document.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Fehler bei der Dokumentenverarbeitung: {e}")
        return False

def verify_chunks_in_tables(supabase):
    """Überprüft, ob Chunks in den korrekten Tabellen gespeichert wurden (gemäß Supabase AI Schema)"""
    try:
        tables = [
            "service_manuals",
            "bulletins",
            "cpmd_documents",
            "parts_catalog"  # Singular laut Supabase AI
        ]
        
        print("\n=== ÜBERPRÜFUNG DER GESPEICHERTEN CHUNKS ===")
        
        for table in tables:
            try:
                result = supabase.table(table).select("*").order("created_at", desc=True).limit(5).execute()
                
                if result.data:
                    print(f"✅ Tabelle {table}: {len(result.data)} neueste Einträge gefunden")
                    
                    # Zeige Beispieldaten des neuesten Eintrags
                    if len(result.data) > 0:
                        entry = result.data[0]
                        content = entry.get("content", "")
                        if content and len(content) > 100:
                            content = content[:100] + "..."
                            
                        print(f"   ID: {entry.get('id', 'unbekannt')}")
                        print(f"   Chunk-Index: {entry.get('chunk_index', 'unbekannt')}")
                        print(f"   Inhalt (Auszug): {content}")
                        
                        # Prüfe, ob ein Embedding vorhanden ist
                        if "embedding" in entry and entry["embedding"]:
                            print(f"   ✅ Embedding vorhanden (Dimension: {len(entry['embedding'])})")
                        else:
                            print(f"   ❌ Kein Embedding vorhanden")
                else:
                    print(f"ℹ️ Tabelle {table}: Keine Einträge gefunden")
            except Exception as e:
                print(f"❓ Tabelle {table}: Fehler bei der Abfrage - {e}")
        
        print("============================================")
    except Exception as e:
        logger.error(f"Fehler bei der Chunk-Überprüfung: {e}")

if __name__ == "__main__":
    # PDF-Pfad
    pdf_path = "Documents/Service_Manuals/HP/X580/HP_X580_SM.pdf"
    
    print("=== DOKUMENT-CHUNKS TEST (SUPABASE AI SCHEMA) ===")
    
    # Supabase initialisieren
    supabase = init_supabase()
    if not supabase:
        print("❌ Test abgebrochen: Keine Verbindung zu Supabase")
        sys.exit(1)
        
    # Tabellen prüfen
    check_tables(supabase)
    
    # Chunks in Tabellen prüfen
    verify_chunks_in_tables(supabase)
    
    # Pfad prüfen
    if not os.path.exists(pdf_path):
        logger.error(f"PDF nicht gefunden: {pdf_path}")
        print(f"❌ PDF nicht gefunden: {pdf_path}")
        sys.exit(1)
    
    # Verarbeitung starten
    if process_document_test(pdf_path):
        print("✅ Test abgeschlossen")
    else:
        print("❌ Test fehlgeschlagen")
    print("===")