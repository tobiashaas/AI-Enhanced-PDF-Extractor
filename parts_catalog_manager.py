#!/usr/bin/env python3
"""
Parts Catalog Manager
====================
Intelligente Verwaltung von Parts Catalogs mit PDF+CSV Pairing und Tracking.

Features:
- Automatisches PDF+CSV Pairing basierend auf Dateinamen
- Metadata Tracking mit processing history
- Validation der Match-Rate zwischen PDF und CSV
- Strukturierte Ordner-Organisation nach Manufacturer/Model
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class PartsCatalogPair:
    """Repräsentiert ein PDF+CSV Paar mit Metadata"""
    manufacturer: str
    model: str
    pdf_path: Path
    csv_path: Path
    metadata_path: Path
    pdf_hash: Optional[str] = None
    csv_hash: Optional[str] = None
    last_processed: Optional[datetime] = None
    processing_status: str = "pending"
    parts_count_csv: Optional[int] = None
    parts_count_pdf: Optional[int] = None
    match_rate: Optional[float] = None

class PartsCatalogManager:
    """Manager für Parts Catalog Organisation und Processing"""
    
    def __init__(self, base_directory: str = "Documents/Parts_Catalogs"):
        self.base_dir = Path(base_directory)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def scan_for_pairs(self) -> List[PartsCatalogPair]:
        """
        Scannt den Parts_Catalogs Ordner nach PDF+CSV Paaren
        
        Returns:
            List[PartsCatalogPair]: Gefundene, gültige Paare
        """
        pairs = []
        
        for manufacturer_dir in self.base_dir.iterdir():
            if not manufacturer_dir.is_dir():
                continue
                
            manufacturer_folder = manufacturer_dir.name
            
            # 🔧 HERSTELLER-MAPPING: Ordnername → Korrekter Name
            manufacturer_mapping = {
                'Konica_Minolta': 'Konica Minolta',
                'Hewlett_Packard': 'HP',
                'HP_Inc': 'HP',
                'Canon_Inc': 'Canon',
                'Xerox_Corporation': 'Xerox',
                'Ricoh_Company': 'Ricoh',
                'Fujifilm_Business': 'Fujifilm',
                'Fuji_Xerox': 'Fujifilm',
                'UTAX_Triumph': 'UTAX',
                'Triumph_Adler': 'UTAX',
                'Kyocera_Document': 'Kyocera',
                'Kyocera_Mita': 'Kyocera',
                'Samsung_Electronics': 'Samsung',
                'Samsung_Print': 'Samsung',
                'Lexmark_International': 'Lexmark',
                'Brother_Industries': 'Brother',
                'Brother_International': 'Brother'
            }
            
            manufacturer = manufacturer_mapping.get(manufacturer_folder, manufacturer_folder.replace('_', ' '))
            
            for model_dir in manufacturer_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model = model_dir.name
                pair = self._find_pdf_csv_pair(manufacturer, model, model_dir)
                
                if pair:
                    pairs.append(pair)
                    self.logger.info(f"✅ Found valid pair: {manufacturer}/{model}")
                else:
                    self.logger.warning(f"⚠️ No valid pair found in: {manufacturer}/{model}")
        
        return pairs
    
    def _find_pdf_csv_pair(self, manufacturer: str, model: str, model_dir: Path) -> Optional[PartsCatalogPair]:
        """Sucht nach PDF+CSV Paar in einem Model-Ordner"""
        
        # Suche nach PDF und CSV Dateien mit ähnlichen Namen
        pdf_files = list(model_dir.glob("*.pdf"))
        csv_files = list(model_dir.glob("*.csv"))
        
        if not pdf_files or not csv_files:
            return None
        
        # Finde passende Paare basierend auf Dateinamen
        for pdf_file in pdf_files:
            pdf_basename = pdf_file.stem  # Dateiname ohne Extension
            
            for csv_file in csv_files:
                csv_basename = csv_file.stem
                
                # Prüfe verschiedene Matching-Strategien
                if self._files_match(pdf_basename, csv_basename):
                    metadata_path = model_dir / "metadata.json"
                    
                    return PartsCatalogPair(
                        manufacturer=manufacturer,
                        model=model,
                        pdf_path=pdf_file,
                        csv_path=csv_file,
                        metadata_path=metadata_path,
                        pdf_hash=self._calculate_file_hash(pdf_file),
                        csv_hash=self._calculate_file_hash(csv_file)
                    )
        
        return None
    
    def _files_match(self, pdf_basename: str, csv_basename: str) -> bool:
        """Prüft ob PDF und CSV zusammengehören"""
        
        # Strategie 1: Identische Basenames
        if pdf_basename == csv_basename:
            return True
        
        # Strategie 2: Ähnliche Namen (ohne _Parts, _Catalog suffixes)
        pdf_clean = pdf_basename.replace("_Parts", "").replace("_Catalog", "").replace("_parts", "")
        csv_clean = csv_basename.replace("_Parts", "").replace("_Catalog", "").replace("_parts", "")
        
        if pdf_clean == csv_clean:
            return True
        
        # Strategie 3: Model-Name ist in beiden enthalten
        # z.B. "C451i_Parts.pdf" und "C451i_PartsList.csv"
        if len(pdf_clean) >= 4 and len(csv_clean) >= 4:
            if pdf_clean in csv_basename or csv_clean in pdf_basename:
                return True
        
        return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Berechnet SHA-256 Hash einer Datei"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def load_metadata(self, pair: PartsCatalogPair) -> Dict:
        """Lädt Metadata aus metadata.json wenn vorhanden"""
        if pair.metadata_path.exists():
            try:
                with open(pair.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metadata for {pair.model}: {e}")
        return {}
    
    def save_metadata(self, pair: PartsCatalogPair):
        """Speichert Metadata in metadata.json"""
        metadata = {
            "manufacturer": pair.manufacturer,
            "model": pair.model,
            "pdf_path": str(pair.pdf_path),
            "csv_path": str(pair.csv_path),
            "pdf_hash": pair.pdf_hash,
            "csv_hash": pair.csv_hash,
            "last_processed": pair.last_processed.isoformat() if pair.last_processed else None,
            "processing_status": pair.processing_status,
            "parts_count": {
                "csv": pair.parts_count_csv,
                "pdf_extracted": pair.parts_count_pdf,
                "match_rate": pair.match_rate
            }
        }
        
        try:
            with open(pair.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"💾 Metadata saved for {pair.manufacturer}/{pair.model}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def get_processing_candidates(self) -> List[PartsCatalogPair]:
        """
        Gibt Liste der Paare zurück, die verarbeitet werden sollten
        
        Kriterien:
        - Beide Dateien (PDF + CSV) existieren
        - Noch nicht verarbeitet ODER Dateien haben sich geändert
        """
        candidates = []
        pairs = self.scan_for_pairs()
        
        for pair in pairs:
            metadata = self.load_metadata(pair)
            
            # Prüfe ob Verarbeitung nötig ist
            needs_processing = False
            
            if not metadata:
                # Keine Metadata = noch nie verarbeitet
                needs_processing = True
                reason = "Never processed"
            elif metadata.get("pdf_hash") != pair.pdf_hash:
                # PDF hat sich geändert
                needs_processing = True
                reason = "PDF file changed"
            elif metadata.get("csv_hash") != pair.csv_hash:
                # CSV hat sich geändert
                needs_processing = True
                reason = "CSV file changed"
            elif metadata.get("processing_status") != "completed":
                # Vorherige Verarbeitung nicht erfolgreich
                needs_processing = True
                reason = "Previous processing incomplete"
            
            if needs_processing:
                self.logger.info(f"📋 Processing candidate: {pair.manufacturer}/{pair.model} - {reason}")
                candidates.append(pair)
            else:
                self.logger.info(f"✅ Up to date: {pair.manufacturer}/{pair.model}")
        
        return candidates
    
    def organize_existing_files(self):
        """
        Organisiert existierende Parts-Dateien in die neue Struktur
        """
        # Prüfe Documents Ordner nach Parts-Dateien
        docs_dir = Path("Documents")
        
        if not docs_dir.exists():
            return
        
        parts_files = []
        parts_files.extend(docs_dir.glob("*Parts*.pdf"))
        parts_files.extend(docs_dir.glob("*Parts*.csv"))
        parts_files.extend(docs_dir.glob("*parts*.pdf"))
        parts_files.extend(docs_dir.glob("*parts*.csv"))
        
        for file_path in parts_files:
            self._organize_single_file(file_path)
    
    def _organize_single_file(self, file_path: Path):
        """Organisiert eine einzelne Datei in die Parts-Struktur"""
        filename = file_path.name
        
        # Versuche Manufacturer und Model aus Dateinamen zu extrahieren
        manufacturer, model = self._extract_manufacturer_model(filename)
        
        if manufacturer and model:
            target_dir = self.base_dir / manufacturer / model
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / filename
            
            if not target_path.exists():
                file_path.rename(target_path)
                self.logger.info(f"📁 Moved: {filename} -> {manufacturer}/{model}/")
            else:
                self.logger.info(f"⚠️ File already exists: {target_path}")
    
    def _extract_manufacturer_model(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """Extrahiert Manufacturer und Model aus Dateinamen"""
        
        # Bekannte Manufacturer-Patterns
        manufacturers = {
            "konica": "Konica_Minolta",
            "minolta": "Konica_Minolta", 
            "hp": "HP",
            "canon": "Canon",
            "xerox": "Xerox",
            "ricoh": "Ricoh"
        }
        
        filename_lower = filename.lower()
        manufacturer = None
        
        for key, value in manufacturers.items():
            if key in filename_lower:
                manufacturer = value
                break
        
        if not manufacturer:
            return None, None
        
        # Versuche Model zu extrahieren
        # Beispiele: "C451i_Parts.pdf", "HP_4000_Parts.csv"
        model = None
        
        if manufacturer == "Konica_Minolta":
            # Suche nach Patterns wie C451i, C552, etc.
            import re
            match = re.search(r'[Cc](\d+[a-zA-Z]*)', filename)
            if match:
                model = match.group(0).upper()
        
        elif manufacturer == "HP":
            # HP Patterns
            match = re.search(r'(LaserJet|OfficeJet|DeskJet)[\s_]*(\w+)', filename, re.IGNORECASE)
            if match:
                model = f"{match.group(1)}_{match.group(2)}"
            else:
                # Einfache Nummer wie "4000"
                match = re.search(r'(\d{4,})', filename)
                if match:
                    model = f"Model_{match.group(1)}"
        
        return manufacturer, model

def main():
    """Demo/Test der Parts Catalog Manager Funktionalität"""
    manager = PartsCatalogManager()
    
    print("🔍 Parts Catalog Manager")
    print("=" * 50)
    
    # 1. Organisiere existierende Dateien
    print("\n1. Organizing existing files...")
    manager.organize_existing_files()
    
    # 2. Scanne nach Paaren
    print("\n2. Scanning for PDF+CSV pairs...")
    pairs = manager.scan_for_pairs()
    print(f"Found {len(pairs)} valid pairs")
    
    # 3. Zeige Processing Candidates
    print("\n3. Checking processing candidates...")
    candidates = manager.get_processing_candidates()
    print(f"Found {len(candidates)} candidates for processing")
    
    # 4. Beispiel-Metadata
    if candidates:
        print("\n4. Example metadata handling...")
        first_candidate = candidates[0]
        first_candidate.last_processed = datetime.now()
        first_candidate.processing_status = "completed"
        first_candidate.parts_count_csv = 1400
        first_candidate.parts_count_pdf = 1390
        first_candidate.match_rate = 99.3
        
        manager.save_metadata(first_candidate)

if __name__ == "__main__":
    main()