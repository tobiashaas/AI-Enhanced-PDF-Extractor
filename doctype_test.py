#!/usr/bin/env python3
"""
Einfacher Test für die Dokumenttyperkennung ohne Speicherung in der Datenbank
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("doctype_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger initialisieren
logger = logging.getLogger(__name__)

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

def load_config():
    """Lädt die Konfiguration"""
    import json
    try:
        with open("config.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        return {}

def detect_document_type(file_path: Path) -> str:
    """
    Bestimmt den Dokumenttyp basierend auf dem Dateipfad
    
    Args:
        file_path: Pfad zum Dokument
        
    Returns:
        str: Dokumenttyp oder 'unknown'
    """
    path_str = str(file_path).lower()
    
    for folder, doc_type in {
        "service_manuals": "service_manuals",
        "bulletins": "bulletins",
        "parts_catalogs": "parts_catalogs",
        "cpmd": "cpmd_documents",
        "video_tutorials": "video_tutorials"
    }.items():
        if folder.lower() in path_str:
            return doc_type
            
    return "unknown"

def get_table_for_document_type(document_type: str) -> str:
    """
    Gibt die entsprechende Datenbanktabelle für einen Dokumenttyp zurück
    
    Args:
        document_type: Der Dokumenttyp
        
    Returns:
        str: Tabellenname
    """
    # Tabellen-Mapping gemäß Supabase AI Dokumentation
    table_mapping = {
        "service_manual": "service_manuals",
        "bulletin": "bulletins",
        "cpmd": "cpmd_documents",
        "parts_manual": "parts_catalog",  # Gemäß Supabase AI für Parts Catalog Chunks
        "service_manuals": "service_manuals",
        "bulletins": "bulletins",
        "cpmd_documents": "cpmd_documents",
        "parts_catalogs": "parts_catalog",
        "video_tutorials": "video_tutorials"  # Keine 's' am Ende nötig
    }
    
    # Verwende das Mapping oder Standardwert 'unknown'
    return table_mapping.get(document_type, "unknown")

def test_document_paths():
    """Testet die Dokumenttyperkennung für verschiedene Pfade"""
    test_paths = [
        "/Users/tobiashaas/Docker/PDF-Extractor/Documents/Service_Manuals/HP/X580/HP_X580_SM.pdf",
        "/Users/tobiashaas/Docker/PDF-Extractor/Documents/Bulletins/HP/HP_PageWide_Bulletin_2025.pdf",
        "/Users/tobiashaas/Docker/PDF-Extractor/Documents/Parts_Catalogs/HP/E60165/Parts_List.pdf",
        "/Users/tobiashaas/Docker/PDF-Extractor/Documents/CPMD/HP_E60165_CPMD.pdf",
        "/Users/tobiashaas/Docker/PDF-Extractor/Documents/Video_Tutorials/demo_videos.csv",
        "/Users/tobiashaas/Docker/PDF-Extractor/Documents/Unknown/some_document.pdf",
    ]
    
    print("\n=== DOKUMENTTYP-ERKENNUNG TEST ===\n")
    
    for path in test_paths:
        path_obj = Path(path)
        doc_type = detect_document_type(path_obj)
        table_name = get_table_for_document_type(doc_type)
        
        print(f"Pfad: {path}")
        print(f"Erkannter Dokumenttyp: {doc_type}")
        print(f"Zieltabelle für Chunks: {table_name}")
        print("-" * 50)
    
    # Test für edge cases mit ProcessingPipeline-Logik vs DocumentProcessor-Logik
    print("\n=== TABELLEN-MAPPING TEST ===\n")
    
    # Test document types from ProcessingPipeline
    pipeline_types = ["service_manuals", "bulletins", "parts_catalogs", "cpmd_documents", "video_tutorials"]
    for doc_type in pipeline_types:
        table_name = get_table_for_document_type(doc_type)
        print(f"Pipeline Dokumenttyp: {doc_type} -> Tabelle: {table_name}")
    
    # Test document types from DocumentProcessor
    processor_types = ["service_manual", "bulletin", "parts_manual", "cpmd"]
    for doc_type in processor_types:
        table_name = get_table_for_document_type(doc_type)
        print(f"Processor Dokumenttyp: {doc_type} -> Tabelle: {table_name}")

if __name__ == "__main__":
    test_document_paths()