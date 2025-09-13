#!/usr/bin/env python3
"""
Document Processor Module
------------------------
Basisklasse und Implementierungen für alle dokumentspezifischen Verarbeitungsprozesse.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import re
import hashlib

logger = logging.getLogger(__name__)

class DocumentProcessor(ABC):
    """Abstrakte Basisklasse für alle Dokumentprozessoren"""
    
    def __init__(self, supabase_client, embedding_client, config):
        """
        Initialisiert den Dokumentprozessor
        
        Args:
            supabase_client: Client für die Supabase-Verbindung
            embedding_client: Client für das Embedding-Modell
            config: Konfigurationsobjekt
        """
        self.supabase = supabase_client
        self.embedding_client = embedding_client
        self.config = config
        
    @abstractmethod
    def process_document(self, file_path: Path, file_hash: str, log_id: str) -> bool:
        """
        Verarbeitet ein Dokument
        
        Args:
            file_path: Pfad zur Datei
            file_hash: Hash der Datei
            log_id: ID des Verarbeitungslogs
            
        Returns:
            bool: True, wenn die Verarbeitung erfolgreich war
        """
        pass
    
    def extract_text(self, file_path: Path) -> str:
        """
        Extrahiert Text aus einer Datei
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            str: Extrahierter Text
        """
        raise NotImplementedError("Diese Methode muss in der abgeleiteten Klasse implementiert werden")
    
    def extract_chunks(self, text: str) -> List[str]:
        """
        Teilt Text in Chunks
        
        Args:
            text: Zu teilender Text
            
        Returns:
            List[str]: Liste von Text-Chunks
        """
        if self.config.get("chunking_strategy") == "intelligent":
            return self._intelligent_chunking(text)
        else:
            return self._simple_chunking(text)
    
    def _intelligent_chunking(self, text: str) -> List[str]:
        """
        Intelligentes Chunking mit semantischen Grenzen
        
        Args:
            text: Zu teilender Text
            
        Returns:
            List[str]: Liste von Chunks
        """
        chunks = []
        max_chunk_size = self.config.get("max_chunk_size", 600)
        min_chunk_size = self.config.get("min_chunk_size", 200)
        
        # Split bei Überschriften und Abschnitten
        sections = re.split(r'\n\s*#{1,3}\s+|\n\s*[A-Z][A-Z\s]+\n', text)
        
        current_chunk = ""
        for section in sections:
            if len(current_chunk) + len(section) <= max_chunk_size:
                current_chunk += section
            else:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                if len(section) > max_chunk_size:
                    # Teile zu große Abschnitte weiter
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= max_chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if len(current_chunk) >= min_chunk_size:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                else:
                    current_chunk = section
                    
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _simple_chunking(self, text: str) -> List[str]:
        """
        Einfaches Chunking mit fester Größe
        
        Args:
            text: Zu teilender Text
            
        Returns:
            List[str]: Liste von Chunks
        """
        max_chunk_size = self.config.get("max_chunk_size", 600)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_chunk_size):
            chunk = ' '.join(words[i:i+max_chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generiert Embeddings für Chunks
        
        Args:
            chunks: Liste von Text-Chunks
            
        Returns:
            List[List[float]]: Liste von Embedding-Vektoren
        """
        return self.embedding_client.embed_documents(chunks)
    
    def extract_version_info(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Extrahiert Versionsinformationen aus Text und Dateinamen
        
        Args:
            text: Text, aus dem Versionsinfo extrahiert wird
            filename: Dateiname, aus dem Versionsinfo extrahiert wird
            
        Returns:
            Dict: Versionsinformationen
        """
        version_info = {}
        
        # Suche nach Versionsnummern im Format v1.2.3, Version 1.2.3, etc.
        version_patterns = [
            r'[vV]ersion\s+(\d+\.\d+\.\d+)',
            r'[vV](\d+\.\d+\.\d+)',
            r'Rev(?:ision)?\s+([A-Z])',
            r'Release\s+(\d{4}-\d{2})'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text) or re.search(pattern, filename)
            if match:
                version = match.group(1)
                version_info["version"] = version
                break
        
        # Suche nach Datum
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{2}\.\d{2}\.\d{4})',
            r'([A-Z][a-z]{2}\s+\d{4})'  # z.B. "Jan 2023"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text) or re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                version_info["release_date"] = date_str
                break
                
        # Revision/Firmware
        if "Rev" in text or "Revision" in text:
            rev_match = re.search(r'Rev(?:ision)?\s+([A-Z0-9]+)', text)
            if rev_match:
                version_info["revision"] = rev_match.group(1)
                
        return version_info
    
    def extract_model_numbers(self, text: str, manufacturer: str) -> List[str]:
        """
        Extrahiert Modellnummern aus Text basierend auf Hersteller-Mustern
        
        Args:
            text: Text, aus dem Modellnummern extrahiert werden
            manufacturer: Hersteller
            
        Returns:
            List[str]: Liste von Modellnummern
        """
        models = []
        
        # Hersteller-spezifische Muster
        patterns = {
            "HP": [
                r'E\d{5}',  # E50045, E52545
                r'X\d{3}',   # X580
                r'M\d{3}',   # M506
                r'Color\s+LaserJet\s+([A-Z0-9]+)'
            ],
            "Konica_Minolta": [
                r'C\d{4}i?',  # C3350i, C3351
                r'bizhub\s+([A-Z0-9]+)'
            ],
            "Lexmark": [
                r'CX\d{3}[a-z]?',  # CX963, CX963se
                r'MX\d{3}[a-z]?'   # MX910
            ]
        }
        
        # Wende Muster an
        manufacturer_patterns = patterns.get(manufacturer, [r'[A-Z0-9]{4,}'])
        for pattern in manufacturer_patterns:
            matches = re.findall(pattern, text)
            models.extend(matches)
            
        # Entferne Duplikate
        return list(set(models))


class ServiceManualProcessor(DocumentProcessor):
    """Prozessor für Service Manuals"""
    
    def process_document(self, file_path: Path, file_hash: str, log_id: str) -> bool:
        """Verarbeitet ein Service Manual"""
        try:
            # Text extrahieren
            content = self.extract_text(file_path)
            
            # Hersteller und Modelle erkennen
            manufacturer, models = self._detect_manufacturer_models(file_path)
            
            # Version extrahieren
            version_info = self.extract_version_info(content, file_path.name)
            
            # In Chunks aufteilen
            chunks = self.extract_chunks(content)
            
            # Embeddings generieren
            embeddings = self.generate_embeddings(chunks)
            
            # Daten speichern
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Weitere Felder extrahieren
                procedure_type = self._detect_procedure_type(chunk)
                problem_type = self._detect_problem_type(chunk)
                difficulty = self._detect_difficulty(chunk)
                tools = self._extract_tools(chunk)
                safety = self._extract_safety_warnings(chunk)
                
                # In DB speichern
                manual_data = {
                    "content": chunk,
                    "file_hash": file_hash,
                    "page_number": i // 3 + 1,  # Annahme: ca. 3 Chunks pro Seite
                    "chunk_index": i,
                    "manufacturer": manufacturer,
                    "model": models[0] if models else None,  # Hauptmodell
                    "document_version": version_info.get("version", ""),
                    "procedure_type": procedure_type,
                    "problem_type": problem_type,
                    "difficulty_level": difficulty,
                    "tools_required": tools,
                    "safety_warnings": safety,
                    "embedding": embedding,
                    "metadata": {
                        "version_info": version_info,
                        "compatible_models": models,
                        "extraction_date": datetime.now().isoformat()
                    }
                }
                
                self.supabase.table("service_manuals").insert(manual_data).execute()
                
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung des Service Manuals {file_path}: {e}")
            return False
    
    def _detect_manufacturer_models(self, file_path: Path):
        """Erkennt Hersteller und Modelle aus dem Dateipfad"""
        # Bestimme Hersteller aus Ordnerstruktur
        parts = file_path.parts
        
        # Hersteller aus Ordnerpfad bestimmen
        manufacturers = ["HP", "Konica_Minolta", "Brother", "Canon", "Epson", 
                        "Kyocera", "Lexmark", "Oki", "Ricoh", "Samsung", 
                        "Sharp", "Xerox"]
        
        manufacturer = next((m for m in manufacturers if m in parts), "Unknown")
        
        # Modelle aus Dateinamen extrahieren
        models = self.extract_model_numbers(file_path.name, manufacturer)
        
        # Wenn keine Modelle im Dateinamen, versuche im Parent-Ordner zu finden
        if not models and len(parts) > 1:
            models = self.extract_model_numbers(parts[-2], manufacturer)
        
        return manufacturer, models
    
    def _detect_procedure_type(self, text: str) -> str:
        """Erkennt den Verfahrenstyp aus dem Text"""
        procedure_keywords = {
            "maintenance": ["maintenance", "wartung", "pflege", "reinigung"],
            "repair": ["repair", "reparatur", "fix", "troubleshoot", "fehlersuche"],
            "installation": ["install", "setup", "einrichtung", "konfiguration"],
            "operation": ["operation", "usage", "verwendung", "bedienung"],
            "calibration": ["calibration", "kalibrierung", "adjustment", "einstellung"]
        }
        
        text_lower = text.lower()
        for proc_type, keywords in procedure_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return proc_type
        
        return "general"
    
    def _detect_problem_type(self, text: str) -> str:
        """Erkennt den Problemtyp aus dem Text"""
        problem_keywords = {
            "paper_jam": ["paper jam", "papierstau", "jam", "stau"],
            "print_quality": ["quality", "qualität", "faded", "blurry", "streaks"],
            "connectivity": ["network", "connection", "verbindung", "wifi", "ethernet"],
            "error_code": ["error code", "fehlercode", "error", "fehler"],
            "hardware": ["hardware", "mechanical", "mechanisch", "physical", "physisch"]
        }
        
        text_lower = text.lower()
        for prob_type, keywords in problem_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return prob_type
        
        return "other"
    
    def _detect_difficulty(self, text: str) -> str:
        """Erkennt den Schwierigkeitsgrad aus dem Text"""
        difficulty_indicators = {
            "beginner": ["simple", "easy", "einfach", "basic", "schnell"],
            "intermediate": ["moderate", "mittel", "normal", "standard"],
            "advanced": ["complex", "komplex", "advanced", "schwierig", "expert"]
        }
        
        text_lower = text.lower()
        for level, keywords in difficulty_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
                
        # Default basierend auf Textlänge und Komplexität
        if len(text) > 1000 or "technician" in text_lower or "service personnel" in text_lower:
            return "advanced"
        elif len(text) > 500:
            return "intermediate"
        else:
            return "beginner"
    
    def _extract_tools(self, text: str) -> List[str]:
        """Extrahiert benötigte Werkzeuge aus dem Text"""
        tools = []
        tool_keywords = [
            "screwdriver", "schraubendreher", 
            "pliers", "zange", 
            "tweezers", "pinzette",
            "brush", "bürste",
            "alcohol", "alkohol",
            "tool", "werkzeug",
            "gloves", "handschuhe"
        ]
        
        text_lower = text.lower()
        for tool in tool_keywords:
            if tool in text_lower:
                tools.append(tool)
                
        return tools
    
    def _extract_safety_warnings(self, text: str) -> List[str]:
        """Extrahiert Sicherheitshinweise aus dem Text"""
        warnings = []
        warning_patterns = [
            r'(?:Warning|Warnung|Caution|Achtung):\s*([^.!?]+[.!?])',
            r'(?:Warning|Warnung|Caution|Achtung)[!:]\s*([^.!?]+[.!?])'
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            warnings.extend(matches)
            
        return [warning.strip() for warning in warnings]


class BulletinProcessor(DocumentProcessor):
    """Prozessor für Bulletins"""
    
    def process_document(self, file_path: Path, file_hash: str, log_id: str) -> bool:
        """Verarbeitet ein Bulletin"""
        try:
            # Implementation ähnlich wie bei ServiceManualProcessor
            # Bulletin-spezifische Extraktion
            pass
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung des Bulletins {file_path}: {e}")
            return False


class PartsCatalogProcessor(DocumentProcessor):
    """Prozessor für Teilekataloge"""
    
    def process_document(self, file_path: Path, file_hash: str, log_id: str) -> bool:
        """Verarbeitet einen Teilekatalog"""
        try:
            # Implementation ähnlich wie bei ServiceManualProcessor
            # Teilekatalog-spezifische Extraktion mit CSV-Pairing
            pass
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung des Teilekatalogs {file_path}: {e}")
            return False


class CPMDProcessor(DocumentProcessor):
    """Prozessor für HP Control Panel Message Documents"""
    
    def process_document(self, file_path: Path, file_hash: str, log_id: str) -> bool:
        """Verarbeitet ein CPMD-Dokument"""
        try:
            # Implementation ähnlich wie bei ServiceManualProcessor
            # CPMD-spezifische Extraktion (Error Codes etc.)
            pass
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung des CPMD-Dokuments {file_path}: {e}")
            return False