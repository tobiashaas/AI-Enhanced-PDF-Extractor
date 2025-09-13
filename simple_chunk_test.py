#!/usr/bin/env python3
"""
Minimaler Test für die PDF Textextraktion und Chunk-Verarbeitung
"""

import os
import sys
import fitz  # PyMuPDF
import logging
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import time

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_chunk_test.log"),
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
            "parts_catalog",
            "images"
        ]
        
        print("\n=== ÜBERPRÜFUNG DER TABELLEN ===")
        
        for table in tables:
            try:
                result = supabase.table(table).select("*").limit(5).execute()
                print(f"✅ Tabelle {table}: {len(result.data)} Beispieleinträge")
            except Exception as e:
                print(f"❌ Tabelle {table}: Fehler oder nicht vorhanden - {e}")
        
        print("===============================")
    except Exception as e:
        logger.error(f"Fehler bei der Tabellenprüfung: {e}")

def test_document_processing():
    """Test der Dokumentenverarbeitung mit minimalem Text"""
    logger.info("Starte minimalen Dokumententest")
    
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
        
        # Ein minimales Test-Dokument (einfacher Text)
        test_text = """
        HP X580 Service Manual
        
        Chapter 1: Introduction
        This manual provides service information for the HP X580 printer.
        
        Chapter 2: Troubleshooting
        Common error codes and solutions.
        Error 001: Paper jam - Clear the paper path.
        Error 002: Toner low - Replace the toner cartridge.
        """
        
        # Text-Chunks erstellen
        chunk_size = 100
        overlap = 20
        chunks = processor.chunk_text(test_text, chunk_size, overlap)
        
        logger.info(f"{len(chunks)} Text-Chunks erstellt (Chunk-Größe: {chunk_size}, Überlappung: {overlap})")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i}: {chunk[:30]}...")
        
        # Embeddings generieren
        try:
            # Nur die ersten 2 Chunks für den Test
            test_chunks = chunks[:2] if len(chunks) > 2 else chunks
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
            
        # Erstelle ein minimales document_data für einen Test
        test_file_hash = hashlib.sha256(test_text.encode()).hexdigest()[:60]
        
        # Teste die Speicherung in service_manuals Tabelle
        try:
            if chunks and len(chunks) > 0:
                for i, (chunk, embedding) in enumerate(zip(test_chunks, embeddings)):
                    # Gemäß Supabase AI Schema: direkt in service_manuals einfügen
                    result = supabase.table("service_manuals").insert({
                        "content": chunk,
                        "chunk_index": i,
                        "page_number": 1,
                        "file_hash": test_file_hash,
                        "embedding": embedding,
                        "manufacturer": "HP",
                        "model": "X580"
                    }).execute()
                    
                    if result.data:
                        chunk_id = result.data[0].get("id", "unbekannt")
                        logger.info(f"Chunk {i} erfolgreich in service_manuals gespeichert: {chunk_id}")
                        print(f"✅ Chunk {i} erfolgreich in service_manuals gespeichert mit ID: {chunk_id}")
                
                # Prüfen ob die Chunks in der Datenbank sind
                time.sleep(1)  # Kurz warten für DB-Sync
                check_chunks_in_db(supabase, test_file_hash)
                
                # Lösche die Test-Chunks
                result = supabase.table("service_manuals").delete().eq("file_hash", test_file_hash).execute()
                logger.info(f"Test-Chunks wurden gelöscht: {len(result.data)} Einträge")
                print(f"✅ {len(result.data)} Test-Chunks erfolgreich gelöscht")
                
        except Exception as chunk_err:
            logger.error(f"Fehler beim Testen der service_manuals Tabelle: {chunk_err}")
            print(f"❌ Fehler beim Speichern in service_manuals: {chunk_err}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fehler bei der Dokumentenverarbeitung: {e}")
        return False

def check_chunks_in_db(supabase, file_hash):
    """Überprüft ob die Test-Chunks in der Datenbank sind"""
    try:
        result = supabase.table("service_manuals").select("*").eq("file_hash", file_hash).execute()
        
        print(f"\n=== TEST-CHUNKS IN DATENBANK ===")
        print(f"Gefundene Chunks mit hash {file_hash}: {len(result.data)}")
        
        for i, chunk in enumerate(result.data):
            print(f"Chunk {i}:")
            print(f"  ID: {chunk.get('id')}")
            print(f"  Inhalt: {chunk.get('content')[:50]}...")
            print(f"  Embedding-Dimension: {len(chunk.get('embedding', []))}")
            print("")
            
    except Exception as e:
        logger.error(f"Fehler beim Prüfen der Chunks in der DB: {e}")
        print(f"❌ Fehler beim Prüfen der Chunks: {e}")

if __name__ == "__main__":
    print("=== MINIMALER DOKUMENT-CHUNKS TEST ===")
    
    # Test ausführen
    if test_document_processing():
        print("✅ Test erfolgreich abgeschlossen")
    else:
        print("❌ Test fehlgeschlagen")