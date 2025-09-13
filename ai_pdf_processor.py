#!/usr/bin/env python3
"""
AI-Enhanced PDF Processing System - Hauptskript
-------------------------------------------

Modulares System zum Extrahieren, Verarbeiten und Indizieren technischer Dokumentation.

Features:
- EmbeddingGemma (768-dimensional) für semantische Suche
- ZERO Conversion Policy für Bilder (bewahrt Originale)
- Dokument-spezifische Verarbeitung (Manuals, Bulletins, Parts, CPMD, Videos)
- Automatische Modell- und Versionserkennung
- Vollständige Supabase-Integration mit RLS

Autor: Tobias Haas
Datum: 13. September 2025
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import requests
from dotenv import load_dotenv
from supabase import create_client

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

# Import der Module
from modules.processing_pipeline.processor import ProcessingPipeline
from modules.chat_memory.processor import MemoryManager

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_processor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AIPDFProcessor:
    """Hauptklasse für die PDF-Dokumentverarbeitung"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialisiert den PDF-Prozessor
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config = self._load_config(config_path)
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Dokumente-Verzeichnis aus Konfiguration oder Umgebungsvariablen
        documents_path = os.environ.get("DOCUMENTS_PATH") or self.config.get("documents_path", "./Documents")
        
        # Wenn relativer Pfad, dann vom Basisverzeichnis ausgehen
        if documents_path.startswith("./"):
            self.documents_dir = self.base_dir / documents_path[2:]
        else:
            self.documents_dir = Path(documents_path)
        
        # Supabase-Verbindung initialisieren
        self.supabase = self._init_supabase()
        
        # Embedding-Client initialisieren
        self.embedding_client = self._init_embedding_client()
        
        # Memory-Manager initialisieren
        self.memory_manager = MemoryManager(self.supabase, self.config)
        
        # R2-Client initialisieren
        from modules.r2_client import R2Client
        self.r2_client = R2Client(self.config, self.supabase)
        
        # Processing-Pipeline initialisieren
        self.pipeline = ProcessingPipeline(self.supabase, self.r2_client, self.embedding_client, self.config)
        
        logger.info("AI PDF Processor initialisiert")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Lädt die Konfigurationsdatei
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            
        Returns:
            Dict[str, Any]: Konfigurationsobjekt
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Konfiguration geladen aus {config_path}")
                return config
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            # Standard-Konfiguration mit Standardwerten
            # Sensible Daten müssen immer aus Umgebungsvariablen kommen
            return {
                "embedding": {
                    "model": "embeddinggemma",
                    "dimensions": 768
                },
                "processing": {
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "use_vision_analysis": False,
                    "chunking_strategy": "intelligent",
                    "max_chunk_size": 600,
                    "min_chunk_size": 200
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "vision_model": "llava:7b",
                    "text_model": "llama3.1:8b",
                    "parallel_workers": 4,
                    "batch_size": 100
                },
                "image_processing": {
                    "keep_originals": True,
                    "zero_conversion": True,
                    "store_original_type": True,
                    "validate_integrity": True
                },
                "memory": {
                    "memory_dir": "MEMORY"
                },
                "documents_path": "./Documents"
            }
    
    def _init_supabase(self):
        """
        Initialisiert die Supabase-Verbindung
        
        Returns:
            Supabase-Client
        """
        try:
            # Ausschließlich Umgebungsvariablen für sensible Daten verwenden
            url = os.environ.get("SUPABASE_URL", "")
            key = os.environ.get("SUPABASE_KEY", "")
            
            if not url or not key:
                logger.error("Supabase-URL oder -Key nicht in Umgebungsvariablen konfiguriert (.env)")
                return None
                
            client = create_client(url, key)
            logger.info("Supabase-Verbindung hergestellt")
            return client
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren der Supabase-Verbindung: {e}")
            return None
    
    def _init_embedding_client(self):
        """
        Initialisiert den Embedding-Client
        
        Returns:
            Embedding-Client
        """
        # Umgebungsvariablen für Modelleinstellungen mit Fallback auf Konfiguration
        model = os.environ.get("EMBEDDING_MODEL") or self.config.get("embedding", {}).get("model", "embeddinggemma")
        dimensions = int(os.environ.get("EMBEDDING_DIMENSIONS") or self.config.get("embedding", {}).get("dimensions", 768))
        
        logger.info(f"Embedding-Client initialisiert mit Modell {model}")
        
        # Dummy-Client (durch eigene Implementierung ersetzen)
        class DummyEmbeddingClient:
            def embed_text(self, text):
                import numpy as np
                return np.random.rand(dimensions)
        
        return DummyEmbeddingClient()
    
    def _check_ollama_version(self):
        """Überprüft die Ollama-Version (mind. 0.11.10 für EmbeddingGemma)"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code == 200:
                version = response.json().get("version", "0.0.0")
                
                # Versionsprüfung - min. 0.11.10
                major, minor, patch = map(int, version.split('.'))
                if (major < 0 or (major == 0 and minor < 11) or 
                    (major == 0 and minor == 11 and patch < 10)):
                    logger.warning(f"⚠️ Ollama Version {version} zu niedrig! Mindestens 0.11.10 erforderlich")
                else:
                    logger.info(f"Ollama Version {version} OK")
                    
                return True
        except Exception as e:
            logger.warning(f"Ollama-Verbindung konnte nicht hergestellt werden: {e}")
            
        return False
    
    def process_all_documents(self) -> Dict[str, Any]:
        """
        Verarbeitet alle Dokumente im Documents-Verzeichnis
        
        Returns:
            Dict[str, Any]: Ergebnis der Verarbeitung
        """
        logger.info(f"Starte die Verarbeitung aller Dokumente in {self.documents_dir}")
        
        if not self.documents_dir.exists():
            logger.error(f"Documents-Verzeichnis nicht gefunden: {self.documents_dir}")
            return {"success": False, "error": "Documents-Verzeichnis nicht gefunden"}
        
        result = self.pipeline.process_all(self.documents_dir)
        
        # Log-Zusammenfassung
        logger.info(f"Verarbeitung abgeschlossen: {result['processed']} verarbeitet, "
                   f"{result['failed']} fehlgeschlagen, {result['skipped']} übersprungen")
        
        return result
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Verarbeitet eine einzelne Datei
        
        Args:
            file_path: Pfad zur zu verarbeitenden Datei
            
        Returns:
            Dict[str, Any]: Ergebnis der Verarbeitung
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Datei nicht gefunden: {path}")
            return {"success": False, "error": "Datei nicht gefunden"}
        
        logger.info(f"Verarbeite Datei: {path}")
        result = self.pipeline.process_document(path)
        
        return result
    
    def update_memory(self, technical_updates: Dict[str, Any] = None, 
                     plan_updates: Dict[str, Any] = None) -> bool:
        """
        Aktualisiert die Speicher-Dateien
        
        Args:
            technical_updates: Updates für das technische Cheat Sheet
            plan_updates: Updates für den Projekt-Masterplan
            
        Returns:
            bool: True, wenn erfolgreich
        """
        success = True
        
        if technical_updates:
            if self.memory_manager.update_technical_cheat_sheet(technical_updates):
                logger.info("Technisches Cheat Sheet aktualisiert")
            else:
                logger.error("Fehler beim Aktualisieren des technischen Cheat Sheets")
                success = False
                
        if plan_updates:
            if self.memory_manager.update_project_master_plan(plan_updates):
                logger.info("Projekt-Masterplan aktualisiert")
            else:
                logger.error("Fehler beim Aktualisieren des Projekt-Masterplans")
                success = False
                
        return success

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="AI PDF Processor")
    parser.add_argument("--config", default="config.json", help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--process-all", action="store_true", help="Alle Dokumente verarbeiten")
    parser.add_argument("--process-file", help="Eine einzelne Datei verarbeiten")
    parser.add_argument("--update-memory", action="store_true", help="Gedächtnis aktualisieren")
    
    args = parser.parse_args()
    
    # Processor initialisieren
    processor = AIPDFProcessor(args.config)
    
    if args.process_all:
        result = processor.process_all_documents()
        if result["success"]:
            print(f"Verarbeitung abgeschlossen: {result['processed']} Dokumente verarbeitet")
        else:
            print(f"Verarbeitung fehlgeschlagen: {result.get('error', 'Unbekannter Fehler')}")
            
    elif args.process_file:
        result = processor.process_file(args.process_file)
        if result["success"]:
            print(f"Datei erfolgreich verarbeitet: {args.process_file}")
        else:
            print(f"Fehler bei der Verarbeitung von {args.process_file}: {result.get('error', 'Unbekannter Fehler')}")
            
    elif args.update_memory:
        # Hier könnten Aktualisierungen aus Dateien oder CLI-Argumenten geladen werden
        print("Gedächtnis-Update ist derzeit nur programmatisch möglich")
        
    else:
        print("Keine Aktion angegeben. Verwende --process-all, --process-file oder --update-memory")
        parser.print_help()

if __name__ == "__main__":
    main()