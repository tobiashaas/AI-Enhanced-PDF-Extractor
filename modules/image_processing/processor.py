#!/usr/bin/env python3
"""
Image Processing Module
----------------------
Modul für die Extraktion und Verarbeitung von Bildern aus Dokumenten.
Implementiert die ZERO CONVERSION POLICY - Originalformate werden beibehalten.
"""

import os
from pathlib import Path
import hashlib
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import mimetypes
import json
import requests

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Hauptklasse für die Bildverarbeitung mit ZERO CONVERSION POLICY
    """
    
    def __init__(self, supabase_client, vision_client, config):
        """
        Initialisiert den Bildprozessor
        
        Args:
            supabase_client: Client für die Supabase-Verbindung
            vision_client: Client für Vision-AI (falls aktiviert)
            config: Konfigurationsobjekt
        """
        self.supabase = supabase_client
        self.vision_client = vision_client
        self.config = config
        self.use_vision_analysis = config.get("use_vision_analysis", True)
        
        # R2/Cloud Storage Setup
        self.storage_config = config.get("r2", {})
        self.bucket_name = self.storage_config.get("bucket_name")
        self.endpoint = self.storage_config.get("endpoint")
        
        logger.info(f"Image Processor initialisiert. ZERO CONVERSION POLICY aktiv.")
    
    def extract_images(self, file_path: Path, file_hash: str, source_table: str, 
                      source_id: str, manufacturer: str, models: List[str]) -> List[str]:
        """
        Extrahiert Bilder aus einem Dokument mit ZERO CONVERSION POLICY
        
        Args:
            file_path: Pfad zur Datei
            file_hash: Hash der Datei
            source_table: Quelltabelle (service_manuals, bulletins, etc.)
            source_id: ID des Quelldokuments
            manufacturer: Hersteller
            models: Liste von Modellen
            
        Returns:
            List[str]: Liste von Bild-IDs
        """
        # Dateityp bestimmen
        if file_path.suffix.lower() == ".pdf":
            return self._extract_images_from_pdf(file_path, file_hash, source_table, source_id, manufacturer, models)
        else:
            logger.warning(f"Nicht unterstütztes Dateiformat für Bildextraktion: {file_path.suffix}")
            return []
    
    def _extract_images_from_pdf(self, pdf_path: Path, file_hash: str, source_table: str, 
                               source_id: str, manufacturer: str, models: List[str]) -> List[str]:
        """
        Extrahiert Bilder aus einer PDF-Datei
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            file_hash: Hash der Datei
            source_table: Quelltabelle
            source_id: ID des Quelldokuments
            manufacturer: Hersteller
            models: Liste von Modellen
            
        Returns:
            List[str]: Liste von Bild-IDs
        """
        extracted_image_ids = []
        
        # Temporäres Verzeichnis für Bildextraktion - plattformunabhängig
        import tempfile
        temp_base = tempfile.gettempdir()
        temp_dir = Path(temp_base) / f"pdf_images_{file_hash}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # PDF-Bilder extrahieren mit pdfimages oder ähnlichem Tool
            # WICHTIG: ZERO CONVERSION POLICY - Originalformat beibehalten
            
            # Beispiel mit pdfimages (müsste installiert sein)
            # pdfimages -all ${pdf_path} ${temp_dir}/img
            
            # Dies ist ein Platzhalter für den tatsächlichen Extraktionscode
            # In einer echten Implementation würde hier pdfimages oder eine Bibliothek verwendet
            
            # Beispiel: Simuliere extrahierte Bilder für das Beispiel
            for i, img_type in enumerate(['jpg', 'png', 'svg']):
                temp_img_path = temp_dir / f"img_{i}.{img_type}"
                
                # Simuliere das Vorhandensein dieser Datei
                # In der echten Implementation würden diese von pdfimages erstellt
                with open(temp_img_path, 'wb') as f:
                    f.write(b'dummy image data')
                
                # Prozessiere das extrahierte Bild
                image_id = self._process_extracted_image(
                    temp_img_path, i, file_hash, source_table, source_id, 
                    manufacturer, models, i+1
                )
                
                if image_id:
                    extracted_image_ids.append(image_id)
            
        except Exception as e:
            logger.error(f"Fehler bei der Bildextraktion aus PDF {pdf_path}: {e}")
        finally:
            # Temporäres Verzeichnis aufräumen
            # In einer echten Implementation:
            # import shutil
            # shutil.rmtree(temp_dir)
            pass
            
        return extracted_image_ids
    
    def _process_extracted_image(self, image_path: Path, image_index: int, file_hash: str,
                              source_table: str, source_id: str, manufacturer: str, 
                              models: List[str], page_number: int) -> Optional[str]:
        """
        Verarbeitet ein extrahiertes Bild
        
        Args:
            image_path: Pfad zum extrahierten Bild
            image_index: Index des Bildes
            file_hash: Hash der Quelldatei
            source_table: Quelltabelle
            source_id: ID des Quelldokuments
            manufacturer: Hersteller
            models: Liste von Modellen
            page_number: Seitenzahl
            
        Returns:
            Optional[str]: Bild-ID oder None bei Fehler
        """
        try:
            # ZERO CONVERSION POLICY:
            # - Originalformat beibehalten
            # - Keine Konvertierung
            # - Keine Komprimierung oder Größenänderung
            
            # Mime-Type bestimmen
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "application/octet-stream"
                
            # Bildtyp bestimmen (diagram, photo, vector)
            image_type = self._detect_image_type(image_path)
            
            # Bild-Hash generieren
            img_hash = self._calculate_hash(image_path)
            
            # Speicherpfad in R2/Cloud Storage
            storage_path = f"{file_hash}/{os.path.basename(image_path)}"
            
            # Bild in Cloud Storage hochladen
            public_url = self._upload_to_storage(image_path, storage_path, mime_type)
            
            # Vision-Analyse durchführen, wenn aktiviert
            vision_analysis = None
            if self.use_vision_analysis:
                vision_analysis = self._analyze_with_vision_ai(image_path)
            
            # In DB speichern
            image_data = {
                "file_hash": file_hash,
                "source_table": source_table,
                "source_id": source_id,
                "page_number": page_number,
                "image_index": image_index,
                "storage_url": public_url,
                "image_type": image_type,
                "manufacturer": manufacturer,
                "model": models[0] if models else None,
                "hash": img_hash,
                "metadata": {
                    "original_format": os.path.splitext(image_path)[1][1:],  # Format ohne Punkt
                    "extraction_method": "zero_conversion_policy",
                    "mime_type": mime_type
                }
            }
            
            if vision_analysis:
                image_data["vision_analysis"] = vision_analysis
                image_data["description"] = vision_analysis.get("caption", "")
            
            # In Supabase speichern
            result = self.supabase.table("images").insert(image_data).execute()
            
            if result.data:
                return result.data[0]["id"]
            return None
                
        except Exception as e:
            logger.error(f"Fehler bei der Bildverarbeitung {image_path}: {e}")
            return None
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Berechnet den Hash einer Datei"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _detect_image_type(self, image_path: Path) -> str:
        """
        Bestimmt den Bildtyp
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            str: Bildtyp (diagram, photo, vector, etc.)
        """
        ext = image_path.suffix.lower()
        
        # Vektorformate
        if ext in ['.svg', '.eps', '.ai']:
            return "vector"
        
        # Basierend auf Dateiendung für das Beispiel
        if "diagram" in image_path.name.lower():
            return "diagram"
        elif "table" in image_path.name.lower():
            return "table"
        else:
            return "photo"  # Default
    
    def _upload_to_storage(self, file_path: Path, storage_path: str, content_type: str) -> str:
        """
        Lädt eine Datei in den Cloud Storage hoch
        
        Args:
            file_path: Pfad zur lokalen Datei
            storage_path: Pfad im Cloud Storage
            content_type: MIME-Typ der Datei
            
        Returns:
            str: Öffentliche URL zur Datei
        """
        # In einer echten Implementation würde hier der Upload zu R2/S3/etc. erfolgen
        # Für das Beispiel geben wir eine Dummy-URL zurück
        
        # Beispiel für echten R2 Upload:
        """
        import boto3
        s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.storage_config.get("access_key_id"),
            aws_secret_access_key=self.storage_config.get("secret_access_key")
        )
        
        with open(file_path, 'rb') as file:
            s3.upload_fileobj(
                file,
                self.bucket_name,
                storage_path,
                ExtraArgs={'ContentType': content_type}
            )
        
        return f"{self.storage_config.get('public_url')}/{storage_path}"
        """
        
        # Dummy-URL für das Beispiel
        return f"https://storage.example.com/{storage_path}"
    
    def _analyze_with_vision_ai(self, image_path: Path) -> Dict:
        """
        Analysiert ein Bild mit Vision AI
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Dict: Ergebnisse der Vision-Analyse
        """
        if not self.use_vision_analysis:
            return {}
            
        try:
            # In einer echten Implementation würde hier die Vision AI API aufgerufen
            # Beispiel mit dem konfigurierten Vision-Modell:
            
            # Für das Beispiel geben wir simulierte Ergebnisse zurück
            vision_model = self.config.get("vision_model", "llava:7b")
            
            return {
                "model": vision_model,
                "caption": f"Extracted image from technical document",
                "labels": ["printer", "technical", "diagram"],
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Vision-Analyse: {e}")
            return {}