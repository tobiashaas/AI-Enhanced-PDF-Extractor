#!/usr/bin/env python3
"""
Processing Pipeline Module
-------------------------
Modul für die zentrale Pipeline der Dokumentverarbeitung.
Koordiniert alle Module und protokolliert den Verarbeitungsprozess.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime
import time

# Import der anderen Module
from modules.document_processing.processor import ServiceManualProcessor, BulletinProcessor, PartsManualProcessor, CPMDProcessor
from modules.image_processing.processor import ImageProcessor
from modules.parts_management.processor import PartsManager

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    """Zentrale Verarbeitungspipeline für alle Dokumente"""
    
    def __init__(self, supabase_client, r2_client, embedding_client, config):
        """
        Initialisiert die Verarbeitungspipeline
        
        Args:
            supabase_client: Client für die Supabase-Verbindung
            r2_client: Client für R2 Cloud Storage
            embedding_client: Client für Embeddings
            config: Konfigurationsobjekt
        """
        self.supabase = supabase_client
        self.r2_client = r2_client
        self.embedding_client = embedding_client
        self.config = config
        
        # Vision-Client (optional)
        self.vision_client = None
        if config.get("use_vision_analysis", False):
            # Hier würden wir den Vision-Client initialisieren
            # z.B. für llava:7b oder andere Vision-Modelle
            self.vision_client = "vision_client_placeholder"
        
        # Dokument-Prozessoren initialisieren
        self._init_processors()
        
        # Processing-Metriken
        self.metrics = {
            "processed_documents": 0,
            "failed_documents": 0,
            "extracted_images": 0,
            "processing_time": 0
        }
        
        logger.info("Processing Pipeline initialisiert")
    
    def _init_processors(self):
        """Initialisiert alle Dokumentprozessoren"""
        # Dokument-Prozessoren
        self.service_manual_processor = ServiceManualProcessor(
            self.supabase, self.r2_client, self.config
        )
        
        self.bulletin_processor = BulletinProcessor(
            self.supabase, self.r2_client, self.config
        )
        
        self.parts_catalog_processor = PartsManualProcessor(
            self.supabase, self.r2_client, self.config
        )
        
        self.cpmd_processor = CPMDProcessor(
            self.supabase, self.r2_client, self.config
        )
        
        # Bild-Prozessor
        self.image_processor = ImageProcessor(
            self.supabase, self.vision_client, self.config
        )
        
        # Parts Manager
        self.parts_manager = PartsManager(
            self.supabase, self.config
        )
        
        # Video-Prozessor (optional)
        self.video_processor = None
        try:
            from modules.video_processing.processor import VideoProcessor
            self.video_processor = VideoProcessor()
            logger.info("Video-Prozessor erfolgreich initialisiert")
        except ImportError:
            logger.warning("Video-Prozessor nicht verfügbar (Modul nicht gefunden)")
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Verarbeitet ein einzelnes Dokument
        
        Args:
            file_path: Pfad zum Dokument
            
        Returns:
            Dict[str, Any]: Ergebnis der Verarbeitung
        """
        start_time = time.time()
        result = {
            "success": False,
            "document_type": None,
            "file_path": str(file_path),
            "error": None,
            "log_id": None,
            "processing_time": 0
        }
        
        try:
            # Dateipfad validieren
            if not file_path.exists():
                result["error"] = f"Datei nicht gefunden: {file_path}"
                return result
            
            # Datei-Hash berechnen
            file_hash = self._calculate_file_hash(file_path)
            
            # Überprüfen, ob die Datei bereits verarbeitet wurde
            if self._is_already_processed(file_hash):
                result["error"] = "Datei wurde bereits verarbeitet"
                return result
            
            # Dokumenttyp bestimmen
            doc_type = self._detect_document_type(file_path)
            if not doc_type:
                result["error"] = f"Dokumenttyp konnte nicht bestimmt werden: {file_path}"
                return result
                
            result["document_type"] = doc_type
            
            # Log-Eintrag erstellen
            log_id = self._create_processing_log(file_path, file_hash, doc_type)
            result["log_id"] = log_id
            
            # Dokument verarbeiten
            success = False
            if doc_type == "service_manuals":
                success = self.service_manual_processor.process_document(file_path, file_hash, log_id)
            elif doc_type == "bulletins":
                success = self.bulletin_processor.process_document(file_path, file_hash, log_id)
            elif doc_type == "parts_catalogs":
                success = self.parts_catalog_processor.process_document(file_path, file_hash, log_id)
            elif doc_type == "cpmd_documents":
                success = self.cpmd_processor.process_document(file_path, file_hash, log_id)
            elif doc_type == "video_tutorials" and self.video_processor:
                # Video-CSV verarbeiten
                if file_path.suffix.lower() == ".csv":
                    self.video_processor.process_csv(str(file_path))
                    success = True
            
            # Log aktualisieren
            if success:
                self._update_processing_log(log_id, "completed", "Verarbeitung abgeschlossen")
                self.metrics["processed_documents"] += 1
            else:
                self._update_processing_log(log_id, "error", "Verarbeitung fehlgeschlagen")
                self.metrics["failed_documents"] += 1
            
            result["success"] = success
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {file_path}: {e}")
            result["error"] = str(e)
            if "log_id" in result and result["log_id"]:
                self._update_processing_log(result["log_id"], "error", str(e))
            self.metrics["failed_documents"] += 1
        
        # Verarbeitungszeit messen
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        self.metrics["processing_time"] += processing_time
        
        return result
    
    def process_all(self, documents_path: Path) -> Dict[str, Any]:
        """
        Verarbeitet alle Dokumente im angegebenen Pfad
        
        Args:
            documents_path: Pfad zum Dokumentenverzeichnis
            
        Returns:
            Dict[str, Any]: Ergebnis der Verarbeitung
        """
        results = {
            "success": True,
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "document_types": {
                "service_manuals": 0,
                "bulletins": 0,
                "parts_catalogs": 0,
                "cpmd_documents": 0,
                "video_tutorials": 0
            },
            "processing_time": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            doc_types = {
                "Service_Manuals": "service_manuals",
                "Bulletins": "bulletins",
                "Parts_Catalogs": "parts_catalogs",
                "CPMD": "cpmd_documents",
                "Video_Tutorials": "video_tutorials"
            }
            
            # Für jeden Dokumenttyp
            for folder_name, table_name in doc_types.items():
                folder_path = documents_path / folder_name
                if not folder_path.exists():
                    logger.warning(f"Ordner nicht gefunden: {folder_path}")
                    continue
                    
                logger.info(f"Verarbeite Dokumente in {folder_path}")
                
                # Rekursive Suche nach PDFs/CSVs
                for file_path in folder_path.glob("**/*"):
                    if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.csv']:
                        result = self.process_document(file_path)
                        
                        if result["success"]:
                            results["processed"] += 1
                            if result["document_type"] in results["document_types"]:
                                results["document_types"][result["document_type"]] += 1
                        elif result["error"] == "Datei wurde bereits verarbeitet":
                            results["skipped"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append({
                                "file": str(file_path),
                                "error": result["error"]
                            })
                
                logger.info(f"Verarbeitung von {folder_name} abgeschlossen")
        
        except Exception as e:
            logger.error(f"Fehler bei der Gesamtverarbeitung: {e}")
            results["success"] = False
            results["errors"].append({
                "file": "OVERALL_PROCESSING",
                "error": str(e)
            })
        
        # Verarbeitungszeit messen
        results["processing_time"] = time.time() - start_time
        
        return results
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Berechnet den SHA-256 Hash einer Datei"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _is_already_processed(self, file_hash: str) -> bool:
        """
        Prüft, ob eine Datei bereits verarbeitet wurde
        
        Args:
            file_hash: Hash der Datei
            
        Returns:
            bool: True, wenn die Datei bereits verarbeitet wurde
        """
        result = self.supabase.table("processing_logs").select("id, status") \
                   .eq("file_hash", file_hash) \
                   .execute()
                   
        if result.data:
            status = result.data[0].get("status")
            # Bereits erfolgreich verarbeitet
            return status == "completed"
            
        return False
    
    def _detect_document_type(self, file_path: Path) -> Optional[str]:
        """
        Bestimmt den Dokumenttyp basierend auf dem Dateipfad
        
        Args:
            file_path: Pfad zum Dokument
            
        Returns:
            Optional[str]: Dokumenttyp oder None
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
                
        return None
    
    def _create_processing_log(self, file_path: Path, file_hash: str, doc_type: str) -> str:
        """
        Erstellt einen Verarbeitungslog-Eintrag
        
        Args:
            file_path: Pfad zum Dokument
            file_hash: Hash des Dokuments
            doc_type: Dokumenttyp
            
        Returns:
            str: ID des Log-Eintrags
        """
        # Prüfen, ob ein Eintrag existiert und ggf. löschen
        try:
            existing = self.supabase.table("processing_logs").select("id").eq("file_hash", file_hash).execute()
            if existing.data:
                logger.info(f"Bestehenden Log-Eintrag für {file_path} gefunden, lösche und erstelle neu")
                self.supabase.table("processing_logs").delete().eq("file_hash", file_hash).execute()
        except Exception as e:
            logger.warning(f"Fehler beim Prüfen auf bestehenden Log-Eintrag: {e}")
        
        # Neuen Eintrag erstellen
        try:
            result = self.supabase.table("processing_logs").insert({
                "file_path": str(file_path),
                "file_hash": file_hash,
                "original_filename": file_path.name,
                "status": "processing",
                "processing_stage": "initialized",
                "progress_percentage": 0,
                "document_type": doc_type,
                "started_at": datetime.now().isoformat()
            }).execute()
            
            if result.data:
                return result.data[0]["id"]
            else:
                logger.warning(f"Konnte keinen Log-Eintrag erstellen für {file_path}")
                return None
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Log-Eintrags: {e}")
            return None
    
    def _update_processing_log(self, log_id: str, status: str, stage: str, 
                              progress: int = None, metadata: Dict = None) -> None:
        """
        Aktualisiert einen Verarbeitungslog-Eintrag
        
        Args:
            log_id: ID des Log-Eintrags
            status: Status (processing, completed, error)
            stage: Verarbeitungsschritt
            progress: Fortschritt in Prozent
            metadata: Zusätzliche Metadaten
        """
        if not log_id:
            return
            
        update_data = {
            "status": status,
            "processing_stage": stage,
            "updated_at": datetime.now().isoformat()
        }
        
        if progress is not None:
            update_data["progress_percentage"] = progress
            
        if status == "completed":
            update_data["completed_at"] = datetime.now().isoformat()
            
        if metadata:
            update_data["document_info"] = metadata
            
        self.supabase.table("processing_logs").update(update_data).eq("id", log_id).execute()