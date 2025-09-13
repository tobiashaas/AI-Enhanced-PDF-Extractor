#!/usr/bin/env python3
"""
Parts Management Module
---------------------
Modul für die Verwaltung von Ersatzteilen und deren Kompatibilität mit verschiedenen Modellen.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class PartsManager:
    """Verwaltet Ersatzteile und deren Modellkompatibilität"""
    
    def __init__(self, supabase_client, config):
        """
        Initialisiert den Parts Manager
        
        Args:
            supabase_client: Client für die Supabase-Verbindung
            config: Konfigurationsobjekt
        """
        self.supabase = supabase_client
        self.config = config
        logger.info("Parts Manager initialisiert")
    
    def process_parts_csv(self, csv_path: Path, manufacturer: str, 
                          models: List[str], doc_version: str) -> Dict[str, Any]:
        """
        Verarbeitet eine CSV mit Ersatzteilen
        
        Args:
            csv_path: Pfad zur CSV-Datei
            manufacturer: Hersteller
            models: Liste kompatibler Modelle
            doc_version: Dokumentversion
            
        Returns:
            Dict[str, Any]: Ergebnisse der Verarbeitung
        """
        results = {
            "processed_parts": 0,
            "added_compatibility": 0,
            "errors": 0
        }
        
        try:
            # CSV einlesen
            df = pd.read_csv(csv_path)
            
            # Prüfe erforderliche Spalten
            required_cols = ["part_number"]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"CSV fehlen erforderliche Spalten: {required_cols}")
                return results
                
            # Optionale Spalten und ihre Standardwerte
            optional_cols = {
                "part_name": "",
                "description": "",
                "category": "other",
                "price": None
            }
            
            # Optionale Spalten mit Standardwerten füllen
            for col, default in optional_cols.items():
                if col not in df.columns:
                    df[col] = default
                    
            # Verarbeite jede Zeile
            for _, row in df.iterrows():
                try:
                    # Prüfe, ob Teil bereits existiert
                    part_number = row["part_number"]
                    result = self.supabase.table("parts_catalog").select("id") \
                                .eq("manufacturer", manufacturer) \
                                .eq("part_number", part_number) \
                                .execute()
                    
                    if result.data:
                        # Teil existiert bereits, aktualisiere es
                        part_id = result.data[0]["id"]
                        
                        # Aktualisiere Teiledaten wenn nötig
                        if any(row[col] for col in optional_cols if col in row and row[col]):
                            self.supabase.table("parts_catalog").update({
                                "part_name": row.get("part_name", ""),
                                "description": row.get("description", ""),
                                "category": row.get("category", "other"),
                                "updated_at": datetime.now().isoformat()
                            }).eq("id", part_id).execute()
                    else:
                        # Neues Teil erstellen
                        part_data = {
                            "part_number": part_number,
                            "part_name": row.get("part_name", ""),
                            "manufacturer": manufacturer,
                            "models_compatible": models,
                            "category": row.get("category", "other"),
                            "description": row.get("description", ""),
                            "source_document_version": doc_version,
                            "metadata": {
                                "csv_source": csv_path.name,
                                "price": row.get("price", None),
                                "import_date": datetime.now().isoformat()
                            }
                        }
                        
                        part_result = self.supabase.table("parts_catalog").insert(part_data).execute()
                        if part_result.data:
                            part_id = part_result.data[0]["id"]
                        else:
                            results["errors"] += 1
                            continue
                            
                    # Kompatibilität für jedes Modell speichern
                    for model in models:
                        # Prüfe, ob Kompatibilität bereits existiert
                        compat_result = self.supabase.table("parts_model_compatibility").select("id") \
                                        .eq("part_id", part_id) \
                                        .eq("model", model) \
                                        .eq("manufacturer", manufacturer) \
                                        .execute()
                                        
                        if not compat_result.data:
                            # Neue Kompatibilität erstellen
                            compatibility_data = {
                                "part_id": part_id,
                                "model": model,
                                "manufacturer": manufacturer,
                                "compatibility_confirmed": True,
                                "source_document": csv_path.name,
                                "document_version": doc_version
                            }
                            
                            self.supabase.table("parts_model_compatibility").insert(compatibility_data).execute()
                            results["added_compatibility"] += 1
                    
                    results["processed_parts"] += 1
                    
                except Exception as row_error:
                    logger.error(f"Fehler bei der Verarbeitung von Zeile {_}: {row_error}")
                    results["errors"] += 1
            
        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der CSV {csv_path}: {e}")
            results["errors"] += 1
            
        return results
    
    def extract_parts_from_text(self, text: str, manufacturer: str, 
                               models: List[str], doc_version: str) -> List[str]:
        """
        Extrahiert Teilenummern aus Text
        
        Args:
            text: Zu durchsuchender Text
            manufacturer: Hersteller
            models: Liste kompatibler Modelle
            doc_version: Dokumentversion
            
        Returns:
            List[str]: Liste gefundener Teilenummern
        """
        import re
        
        # Herstellerspezifische Teilenummern-Muster
        patterns = {
            "HP": [
                r'[A-Z]{1,2}\d{3}[A-Z]{0,2}',      # RC1-0545, RG5-7606-000CN
                r'\d[A-Z]{2}\d{2}-\d{4}'           # 5MX12-3456
            ],
            "Konica_Minolta": [
                r'[A-Z]\d{3}\d{4}',                # A00J563600
                r'\d{3}[A-Z]-\d{4}'                # 423G-7654
            ],
            "Lexmark": [
                r'40X\d{4}',                       # 40X7545
                r'[A-Z]{2}-[A-Z]\d{3}'             # MS-C925
            ]
        }
        
        # Verwende Standardmuster wenn kein herstellerspezifisches vorhanden
        manufacturer_patterns = patterns.get(manufacturer, [r'[A-Z0-9]{5,}'])
        
        # Suche nach Teilenummern
        found_parts = []
        for pattern in manufacturer_patterns:
            matches = re.findall(pattern, text)
            found_parts.extend(matches)
            
        # Entferne Duplikate
        unique_parts = list(set(found_parts))
        
        # Speichere gefundene Teile in der Datenbank
        for part_number in unique_parts:
            self._store_part(part_number, manufacturer, models, doc_version)
            
        return unique_parts
    
    def _store_part(self, part_number: str, manufacturer: str, 
                   models: List[str], doc_version: str) -> Optional[str]:
        """
        Speichert ein Teil in der Datenbank
        
        Args:
            part_number: Teilenummer
            manufacturer: Hersteller
            models: Liste kompatibler Modelle
            doc_version: Dokumentversion
            
        Returns:
            Optional[str]: ID des Teils oder None bei Fehler
        """
        try:
            # Prüfe, ob Teil bereits existiert
            result = self.supabase.table("parts_catalog").select("id") \
                        .eq("manufacturer", manufacturer) \
                        .eq("part_number", part_number) \
                        .execute()
                        
            if result.data:
                # Teil existiert bereits
                part_id = result.data[0]["id"]
            else:
                # Neues Teil erstellen
                part_data = {
                    "part_number": part_number,
                    "manufacturer": manufacturer,
                    "models_compatible": models,
                    "source_document_version": doc_version,
                    "metadata": {
                        "extracted": True,
                        "extraction_date": datetime.now().isoformat(),
                        "confidence": "medium"  # Aus Text extrahiert = mittlere Konfidenz
                    }
                }
                
                part_result = self.supabase.table("parts_catalog").insert(part_data).execute()
                if part_result.data:
                    part_id = part_result.data[0]["id"]
                else:
                    return None
            
            # Kompatibilität für jedes Modell speichern
            for model in models:
                # Prüfe, ob Kompatibilität bereits existiert
                compat_result = self.supabase.table("parts_model_compatibility").select("id") \
                                .eq("part_id", part_id) \
                                .eq("model", model) \
                                .eq("manufacturer", manufacturer) \
                                .execute()
                                
                if not compat_result.data:
                    # Neue Kompatibilität erstellen
                    compatibility_data = {
                        "part_id": part_id,
                        "model": model,
                        "manufacturer": manufacturer,
                        "compatibility_confirmed": False,  # Aus Text extrahiert = unbestätigt
                        "document_version": doc_version
                    }
                    
                    self.supabase.table("parts_model_compatibility").insert(compatibility_data).execute()
            
            return part_id
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Teils {part_number}: {e}")
            return None