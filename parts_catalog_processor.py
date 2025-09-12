#!/usr/bin/env python3
"""
Parts Catalog & CSV Integration System
Specialized processing for Parts Catalogs with Explosion Diagrams + CSV Data
"""

import os
import csv
import json
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
import logging
from dataclasses import dataclass

@dataclass
class PartInfo:
    """Structured part information - Enhanced for Konica Minolta format"""
    part_number: str
    description: str
    manufacturer: str
    model_compatibility: List[str]
    category: str
    price: Optional[float] = None
    availability: Optional[str] = None
    location_in_diagram: Optional[str] = None
    replacement_parts: Optional[List[str]] = None
    
    # Konica Minolta specific fields
    quantity: Optional[int] = None
    ship_unit: Optional[int] = None
    pmn_no: Optional[str] = None
    destinations: Optional[str] = None
    part_class: Optional[str] = None
    page_reference: Optional[int] = None
    key_reference: Optional[str] = None
    
class PartsProcessor:
    """Enhanced processor for Parts Catalogs with CSV and Explosion Diagram support"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parts-specific patterns
        self.part_number_patterns = [
            r'[A-Z]+[-\s]?\d{4,}[-\s]?\w*',  # Standard format: AB-1234-CD
            r'\d{4}[-\s]?\d{3}[-\s]?\d{3}',   # Numeric format: 1234-567-890
            r'[A-Z]{2,4}\d{6,}',              # Combined: ABC123456
        ]
        
        # Explosion diagram indicators
        self.diagram_keywords = [
            'explosion', 'exploded', 'assembly', 'diagram', 'breakdown',
            'parts location', 'component layout', 'service parts'
        ]
    
    def process_parts_catalog(self, pdf_path: str, csv_path: str = None) -> Dict:
        """
        Main processor for Parts Catalogs
        Args:
            pdf_path: Path to PDF with explosion diagrams
            csv_path: Optional CSV with structured parts data
        """
        print(f"🔧 Processing Parts Catalog: {os.path.basename(pdf_path)}")
        
        # 1. Extract manufacturer and model from filename/path
        catalog_info = self.parse_catalog_info(pdf_path)
        
        # 2. Process CSV data if available
        csv_parts = {}
        if csv_path and os.path.exists(csv_path):
            csv_parts = self.process_csv_parts(csv_path)
            print(f"   📊 CSV Parts loaded: {len(csv_parts)} entries")
        
        # 3. Process PDF for explosion diagrams and part references
        pdf_parts = self.process_pdf_diagrams(pdf_path, catalog_info, csv_parts if csv_parts else None)
        
        # 4. Cross-reference and enhance data
        enhanced_parts = self.cross_reference_parts(csv_parts, pdf_parts)
        
        # 5. Create specialized chunks for parts search
        parts_chunks = self.create_parts_chunks(enhanced_parts, catalog_info)
        
        return {
            'catalog_info': catalog_info,
            'parts_data': enhanced_parts,
            'chunks': parts_chunks,
            'csv_count': len(csv_parts),
            'pdf_parts_count': len(pdf_parts)
        }
    
    def parse_catalog_info(self, pdf_path: str) -> Dict:
        """Extract catalog metadata from filename and PDF content"""
        filename = os.path.basename(pdf_path)
        
        # Parse manufacturer and model from filename
        # Example: "C451i_Parts.pdf" -> Konica Minolta C451i
        catalog_info = {
            'filename': filename,
            'document_type': 'Parts Catalog',
            'manufacturer': 'Unknown',
            'model': 'Unknown',
            'catalog_type': 'parts'
        }
        
        # Manufacturer detection patterns (filename-based)
        filename_lower = filename.lower()
        
        if any(x in filename_lower for x in ['hp', 'hewlett', 'packard', 'laserjet', 'officejet']):
            catalog_info['manufacturer'] = 'HP'
        elif any(x in filename_lower for x in ['canon', 'imagerunner', 'pixma']):
            catalog_info['manufacturer'] = 'Canon'
        elif any(x in filename_lower for x in ['xerox', 'workcentre', 'phaser']):
            catalog_info['manufacturer'] = 'Xerox'
        elif any(x in filename_lower for x in ['ricoh', 'aficio', 'mp', 'sp']):
            catalog_info['manufacturer'] = 'Ricoh'
        elif any(x in filename_lower for x in ['fujifilm', 'fuji', 'docucentre', 'docuprint', 'apeos']):
            catalog_info['manufacturer'] = 'Fujifilm'
        elif any(x in filename_lower for x in ['kyocera', 'ecosys', 'taskalfa', 'fs-c', 'fs-', 'km-']):
            catalog_info['manufacturer'] = 'Kyocera'
        elif any(x in filename_lower for x in ['utax', 'triumph']) or (filename_lower.startswith('ta') and len(filename_lower) > 2):
            catalog_info['manufacturer'] = 'UTAX'
        elif any(x in filename_lower for x in ['samsung', 'proxpress', 'xpress', 'clp', 'clx', 'scx', 'sl-', 'ml-']):
            catalog_info['manufacturer'] = 'Samsung'
        elif any(x in filename_lower for x in ['lexmark', 'optra', 'e120', 'e230', 'e240', 'e250', 'e260', 'e340', 'e350', 'e360', 'e450', 'e460', 't640', 't650', 'x264', 'x363', 'x364', 'x463', 'x464', 'x466', 'cx310', 'cx410', 'cx510', 'ms310', 'ms410', 'ms510', 'ms610', 'mx310', 'mx410', 'mx510', 'mx610', 'mx710', 'mx810']):
            catalog_info['manufacturer'] = 'Lexmark'
        elif any(x in filename_lower for x in ['brother', 'dcp-', 'mfc-', 'hl-', 'fax-', 'intellifax', 'p-touch', 'ql-', 'pt-', 'td-', 'rj-', 'pj-', 'mw-', 'justio']):
            catalog_info['manufacturer'] = 'Brother'
        elif any(x in filename_lower for x in ['konica', 'minolta', 'bizhub', 'c451', 'c550', 'c650', 'c3350', 'c3351', 'c361', 'c4050']):
            catalog_info['manufacturer'] = 'Konica Minolta'
        
        # Enhanced PDF-based manufacturer detection
        try:
            import fitz
            doc = fitz.open(pdf_path)
            
            # Check first few pages for manufacturer info
            manufacturer_keywords = {
                'Konica Minolta': [
                    'konica', 'minolta', 'bizhub', 'accurio', 'magicolor', 
                    'pagepro', 'dimage', 'develop'
                ],
                'HP': [
                    'hewlett', 'packard', 'laserjet', 'officejet', 'deskjet', 
                    'envy', 'designjet', 'pagewide', 'indigo', 'latex'
                ],
                'Canon': [
                    'canon', 'imagerunner', 'pixma', 'maxify', 'selphy', 
                    'imageclass', 'imageprograf', 'variprint', 'advance'
                ],
                'Xerox': [
                    'xerox', 'workcentre', 'phaser', 'colorqube', 'versalink', 
                    'altalink', 'primelink', 'iridesse', 'baltoro'
                ],
                'Ricoh': [
                    'ricoh', 'aficio', 'mp', 'sp', 'im c', 'pro c', 'geljet',
                    'nashuatec', 'lanier', 'infotec'
                ],
                'Fujifilm': [
                    'fujifilm', 'fuji xerox', 'docucentre', 'docuprint', 
                    'apeos', 'apeosport', 'xerox docucentre'
                ],
                'UTAX': [
                    'utax', 'triumph-adler', 'ta', 'kyocera utax', 'p-c',
                    'lp-', 'cdc', 'ci-'
                ],
                'Kyocera': [
                    'kyocera', 'ecosys', 'taskalfa', 'fs-c', 'fs-', 'km-c',
                    'km-', 'mita', 'copystar'
                ],
                'Samsung': [
                    'samsung', 'proxpress', 'xpress', 'clp-', 'clx-', 'scx-',
                    'sl-', 'ml-', 'multixpress'
                ],
                'Lexmark': [
                    'lexmark', 'optra', 'e120', 'e230', 'e240', 'e250', 'e260',
                    'e340', 'e350', 'e360', 'e450', 'e460', 't640', 't650',
                    'x264', 'x363', 'x364', 'x463', 'x464', 'x466', 'cx310',
                    'cx410', 'cx510', 'ms310', 'ms410', 'ms510', 'ms610',
                    'mx310', 'mx410', 'mx510', 'mx610', 'mx710', 'mx810'
                ],
                'Brother': [
                    'brother', 'dcp-', 'mfc-', 'hl-', 'fax-', 'intellifax',
                    'p-touch', 'ql-', 'pt-', 'td-', 'rj-', 'pj-', 'mw-',
                    'justio'
                ]
            }
            
            # Search in first 3 pages
            for page_num in range(min(3, doc.page_count)):
                page = doc[page_num]
                text = page.get_text().lower()
                
                for manufacturer, keywords in manufacturer_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        catalog_info['manufacturer'] = manufacturer
                        print(f"   🏭 Manufacturer detected from PDF: {manufacturer}")
                        break
                        
                if catalog_info['manufacturer'] != 'Unknown':
                    break
                    
            doc.close()
            
        except Exception as e:
            print(f"   ⚠️ Could not extract manufacturer from PDF: {e}")
        
        # Model extraction
        model_match = re.search(r'([A-Z]?\d{3,4}[A-Z]?)', filename, re.IGNORECASE)
        if model_match:
            catalog_info['model'] = model_match.group(1).upper()
        
        return catalog_info
    
    def process_csv_parts(self, csv_path: str) -> Dict[str, PartInfo]:
        """Process CSV parts data into structured format - Konica Minolta format"""
        parts = {}
        
        try:
            df = pd.read_csv(csv_path)
            
            # Konica Minolta CSV columns:
            # "Model Name","Page","Key","Parts No.","Parts Name","Destinations","Class","Quantity","Ship Unit","PMN No."
            
            for _, row in df.iterrows():
                # Extract part number (primary key)
                part_number = str(row.get('Parts No.', '')).strip()
                
                if not part_number or part_number == 'nan':
                    continue
                
                # Parse model from "Model Name" column
                model_name = str(row.get('Model Name', '')).strip()
                models = [model_name] if model_name else []
                
                # Parse destinations (regional availability)
                destinations = str(row.get('Destinations', '')).strip()
                if destinations == 'nan' or not destinations:
                    availability = 'Global'
                    destinations = ''
                else:
                    availability = f'Regional: {destinations}'
                
                # Parse class (C=Critical, D=Standard, etc.)
                part_class = str(row.get('Class', '')).strip()
                if part_class == 'nan':
                    part_class = ''
                category = self._map_class_to_category(part_class)
                
                # Create PartInfo object with actual CSV data
                part_info = PartInfo(
                    part_number=part_number,
                    description=str(row.get('Parts Name', '')).strip(),
                    manufacturer='Konica Minolta',  # From filename context
                    model_compatibility=models,
                    category=category,
                    price=None,  # Not available in this CSV format
                    availability=availability,
                    location_in_diagram=f"Page {row.get('Page', 'Unknown')}, Key {row.get('Key', 'Unknown')}"
                )
                
                # Add additional metadata from CSV
                part_info.quantity = row.get('Quantity', 1)
                part_info.ship_unit = row.get('Ship Unit', 1)
                pmn_value = str(row.get('PMN No.', '')).strip()
                part_info.pmn_no = pmn_value if pmn_value != 'nan' else ''
                part_info.destinations = destinations
                part_info.part_class = part_class
                part_info.page_reference = row.get('Page', None)
                part_info.key_reference = row.get('Key', None)
                
                parts[part_number] = part_info
        
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
        
        return parts
    
    def _map_class_to_category(self, part_class: str) -> str:
        """Map Konica Minolta part class to category"""
        class_mapping = {
            'C': 'Critical Component',      # Critical parts
            'D': 'Standard Component',      # Standard parts  
            'M': 'Maintenance Item',        # Maintenance parts
            'S': 'Service Part',           # Service-specific parts
            'O': 'Optional Component'       # Optional parts
        }
        return class_mapping.get(part_class.upper(), 'Unknown')
    
    def process_pdf_diagrams(self, pdf_path: str, catalog_info: Dict, csv_reference: Dict = None) -> Dict[str, Dict]:
        """Extract parts information from PDF explosion diagrams - Enhanced with CSV reference"""
        pdf_parts = {}
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            # If we have CSV reference, use it to improve extraction accuracy
            csv_part_numbers = set()
            if csv_reference:
                csv_part_numbers = set(csv_reference.keys())
                self.logger.info(f"Using CSV reference with {len(csv_part_numbers)} parts for improved PDF extraction")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                
                # Check if page contains explosion diagram
                is_diagram_page = any(keyword in text.lower() for keyword in self.diagram_keywords)
                
                # Extract part numbers from text with CSV validation
                if csv_reference:
                    # Use CSV as reference - only extract parts that exist in CSV
                    part_numbers = self.extract_validated_part_numbers(text, csv_part_numbers)
                else:
                    # Fallback to pattern-based extraction
                    part_numbers = self.extract_part_numbers(text)
                
                # Extract images if it's a diagram page
                images = []
                if is_diagram_page:
                    images = self.extract_page_images(page, page_num)
                
                # Store part references with page context
                for part_num in part_numbers:
                    if part_num not in pdf_parts:
                        pdf_parts[part_num] = {
                            'part_number': part_num,
                            'pages': [],
                            'context': [],
                            'is_in_diagram': False,
                            'diagram_images': []
                        }
                    
                    pdf_parts[part_num]['pages'].append(page_num + 1)
                    pdf_parts[part_num]['context'].append(text[:200] + '...')
                    
                    if is_diagram_page:
                        pdf_parts[part_num]['is_in_diagram'] = True
                        pdf_parts[part_num]['diagram_images'].extend(images)
        
        except Exception as e:
            self.logger.error(f"Error processing PDF diagrams: {e}")
        
        return pdf_parts
    
    def extract_validated_part_numbers(self, text: str, csv_part_numbers: set) -> List[str]:
        """Extract part numbers that are validated against CSV reference"""
        found_parts = []
        
        # Split text into words and clean them
        words = text.replace('\n', ' ').split()
        
        for word in words:
            # Clean the word (remove special characters at edges)
            cleaned_word = word.strip('()[]{},.;:!?-_|\\/')
            
            # Check if this word matches any CSV part number
            if cleaned_word in csv_part_numbers:
                found_parts.append(cleaned_word)
                continue
            
            # Also check for partial matches (in case of formatting differences)
            for csv_part in csv_part_numbers:
                if len(csv_part) > 6:  # Only for longer part numbers
                    # Check if CSV part is contained in the word or vice versa
                    if csv_part in cleaned_word or cleaned_word in csv_part:
                        # Additional validation: similar length
                        if abs(len(csv_part) - len(cleaned_word)) <= 2:
                            found_parts.append(csv_part)  # Use the CSV version
                            break
        
        return list(set(found_parts))  # Remove duplicates
    
    def extract_part_numbers(self, text: str) -> List[str]:
        """Extract part numbers using multiple patterns"""
        part_numbers = []
        
        for pattern in self.part_number_patterns:
            matches = re.findall(pattern, text)
            part_numbers.extend(matches)
        
        # Clean and deduplicate
        cleaned_parts = []
        for part in part_numbers:
            cleaned = re.sub(r'[-\s]+', '-', part.strip())
            if len(cleaned) >= 5 and cleaned not in cleaned_parts:
                cleaned_parts.append(cleaned)
        
        return cleaned_parts
    
    def extract_page_images(self, page, page_num: int) -> List[Dict]:
        """Extract images from a page (for explosion diagrams)"""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # Skip if not RGB/Gray
                    img_data = pix.tobytes("png")
                    
                    images.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'width': pix.width,
                        'height': pix.height,
                        'data': img_data,
                        'type': 'explosion_diagram'
                    })
                
                pix = None
            except Exception as e:
                self.logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
        
        return images
    
    def cross_reference_parts(self, csv_parts: Dict, pdf_parts: Dict) -> Dict:
        """Cross-reference CSV and PDF data to create enhanced parts database"""
        enhanced_parts = {}
        
        # Start with CSV data as base
        for part_num, csv_part in csv_parts.items():
            enhanced_parts[part_num] = {
                'part_number': part_num,
                'description': csv_part.description,
                'manufacturer': csv_part.manufacturer,
                'model_compatibility': csv_part.model_compatibility,
                'category': csv_part.category,
                'price': csv_part.price,
                'availability': csv_part.availability,
                'source': 'csv',
                'has_diagram': False,
                'diagram_pages': [],
                'context': [],
                # Konica Minolta specific fields
                'part_class': getattr(csv_part, 'part_class', ''),
                'quantity': getattr(csv_part, 'quantity', 1),
                'ship_unit': getattr(csv_part, 'ship_unit', 1),
                'pmn_no': getattr(csv_part, 'pmn_no', ''),
                'destinations': getattr(csv_part, 'destinations', ''),
                'page_reference': getattr(csv_part, 'page_reference', None),
                'key_reference': getattr(csv_part, 'key_reference', None),
                'location_in_diagram': getattr(csv_part, 'location_in_diagram', '')
            }
        
        # Enhance with PDF data
        for part_num, pdf_part in pdf_parts.items():
            if part_num in enhanced_parts:
                # Enhance existing CSV entry
                enhanced_parts[part_num]['has_diagram'] = pdf_part['is_in_diagram']
                enhanced_parts[part_num]['diagram_pages'] = pdf_part['pages']
                enhanced_parts[part_num]['context'] = pdf_part['context']
                enhanced_parts[part_num]['source'] = 'csv+pdf'
            else:
                # Create new entry from PDF only
                enhanced_parts[part_num] = {
                    'part_number': part_num,
                    'description': f"Part found in diagram (Page {pdf_part['pages'][0]})",
                    'manufacturer': 'Unknown',
                    'model_compatibility': [],
                    'category': 'Unknown',
                    'price': None,
                    'availability': 'Unknown',
                    'source': 'pdf',
                    'has_diagram': pdf_part['is_in_diagram'],
                    'diagram_pages': pdf_part['pages'],
                    'context': pdf_part['context']
                }
        
        return enhanced_parts
    
    def create_parts_chunks(self, parts_data: Dict, catalog_info: Dict) -> List[Dict]:
        """Create specialized chunks optimized for parts search"""
        chunks = []
        
        for part_num, part_info in parts_data.items():
            # Create comprehensive part description
            chunk_content = self._build_part_chunk_content(part_info)
            
            # Create searchable chunk
            chunk = {
                'content': chunk_content,
                'chunk_type': 'parts_catalog',
                'part_number': part_num,
                'manufacturer': catalog_info['manufacturer'],
                'model': catalog_info['model'],
                'document_type': 'Parts Catalog',
                'category': part_info.get('category', 'Unknown'),
                'has_explosion_diagram': part_info.get('has_diagram', False),
                'diagram_pages': part_info.get('diagram_pages', []),
                'price': part_info.get('price'),
                'availability': part_info.get('availability'),
                'search_keywords': self._generate_search_keywords(part_info),
                'metadata': {
                    'source': part_info.get('source', 'unknown'),
                    'compatibility': part_info.get('model_compatibility', []),
                    'context_pages': part_info.get('diagram_pages', [])
                }
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _build_part_chunk_content(self, part_info: Dict) -> str:
        """Build comprehensive content for part chunk - Konica Minolta enhanced"""
        content_parts = [
            f"PART NUMBER: {part_info['part_number']}"
        ]
        
        if part_info.get('description'):
            content_parts.append(f"DESCRIPTION: {part_info['description']}")
        
        if part_info.get('category'):
            content_parts.append(f"CATEGORY: {part_info['category']}")
        
        if part_info.get('model_compatibility'):
            models = ', '.join(part_info['model_compatibility'])
            content_parts.append(f"COMPATIBLE MODELS: {models}")
        
        # Konica Minolta specific fields
        if part_info.get('part_class'):
            content_parts.append(f"PART CLASS: {part_info['part_class']}")
        
        if part_info.get('quantity'):
            content_parts.append(f"QUANTITY PER UNIT: {part_info['quantity']}")
        
        if part_info.get('ship_unit'):
            content_parts.append(f"SHIPPING UNIT: {part_info['ship_unit']}")
        
        if part_info.get('destinations'):
            content_parts.append(f"REGIONAL AVAILABILITY: {part_info['destinations']}")
        
        if part_info.get('pmn_no'):
            content_parts.append(f"PMN NUMBER: {part_info['pmn_no']}")
        
        if part_info.get('location_in_diagram'):
            content_parts.append(f"DIAGRAM LOCATION: {part_info['location_in_diagram']}")
        
        if part_info.get('availability'):
            content_parts.append(f"AVAILABILITY: {part_info['availability']}")
        
        if part_info.get('has_diagram'):
            pages = ', '.join(map(str, part_info.get('diagram_pages', [])))
            content_parts.append(f"EXPLOSION DIAGRAM: Available on pages {pages}")
        
        if part_info.get('context'):
            content_parts.append(f"TECHNICAL CONTEXT: {part_info['context'][0]}")
        
        return '\n'.join(content_parts)
    
    def _generate_search_keywords(self, part_info: Dict) -> List[str]:
        """Generate search keywords for enhanced findability"""
        keywords = [part_info['part_number']]
        
        if part_info.get('description'):
            keywords.extend(part_info['description'].split())
        
        if part_info.get('category'):
            keywords.append(part_info['category'])
        
        if part_info.get('model_compatibility'):
            keywords.extend(part_info['model_compatibility'])
        
        # Clean and deduplicate
        return list(set([kw.lower().strip() for kw in keywords if len(kw) > 2]))
    
    def _get_column_value(self, row, columns: Dict, possible_names: List[str], default: str) -> str:
        """Get value from row using multiple possible column names"""
        for name in possible_names:
            if name in columns:
                val = row[columns[name]]
                if pd.notna(val):
                    return str(val).strip()
        return default
    
    def _get_float_value(self, row, columns: Dict, possible_names: List[str]) -> Optional[float]:
        """Get float value from row"""
        for name in possible_names:
            if name in columns:
                val = row[columns[name]]
                if pd.notna(val):
                    try:
                        # Remove currency symbols and parse
                        clean_val = re.sub(r'[^\d.]', '', str(val))
                        return float(clean_val)
                    except ValueError:
                        continue
        return None
    
    def _parse_models(self, models_str: str) -> List[str]:
        """Parse model compatibility string into list"""
        if not models_str:
            return []
        
        # Split by common delimiters
        models = re.split(r'[,;|]', models_str)
        return [model.strip().upper() for model in models if model.strip()]

# Integration with existing AI PDF Processor
def integrate_parts_processor():
    """Integration guide for adding parts processing to existing system"""
    integration_code = '''
    # Add to ai_pdf_processor.py in parse_file_path method:
    
    def parse_file_path(self, file_path: str) -> Dict:
        # ... existing code ...
        
        # Detect Parts Catalogs
        if 'parts' in filename.lower() or 'catalog' in filename.lower():
            document_info['document_type'] = 'Parts Catalog'
            document_info['requires_parts_processing'] = True
        
        return document_info
    
    # Add new processing method:
    
    def process_parts_catalog(self, file_path: str, document_info: Dict) -> Dict:
        """Enhanced processing for Parts Catalogs"""
        from parts_catalog_processor import PartsProcessor
        
        parts_processor = PartsProcessor(self.config)
        
        # Look for accompanying CSV file
        csv_path = file_path.replace('.pdf', '.csv').replace('.pdfz', '.csv')
        
        return parts_processor.process_parts_catalog(file_path, csv_path)
    
    # Modify main process_pdf method:
    
    def process_pdf(self, file_path: str) -> bool:
        # ... existing code until document_info parsing ...
        
        # Check if this is a parts catalog
        if document_info.get('requires_parts_processing'):
            print("🔧 Erkannte Parts Catalog - verwende spezialisierte Verarbeitung")
            parts_result = self.process_parts_catalog(file_path, document_info)
            # Store parts data in specialized table/format
            return self.store_parts_data(parts_result)
        
        # ... continue with regular processing ...
    '''
    
    return integration_code

if __name__ == "__main__":
    # Example usage
    config = {}  # Your existing config
    processor = PartsProcessor(config)
    
    # Test with example file
    result = processor.process_parts_catalog("Documents/C451i_Parts.pdf")
    print(f"Processed {len(result['parts_data'])} parts")