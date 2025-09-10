# PDF Extraction System mit AI-Enhanced Chunking - Ollama Integration

## Projektübersicht

Entwickle ein **Python-basiertes PDF-Extraktionssystem** mit **KI-gestütztem Smart Chunking** über **lokales Ollama**. Das System nutzt **Vision AI + LLM-Segmentation** für optimale Verarbeitung von Service Manuals mit komplexen Strukturen.

### Ziel
- **AI-Enhanced Chunking** mit lokalen Ollama Models
- **Vision-Guided Segmentation** für mehrseitige Tabellen und Diagramme
- **Procedure-Aware Splitting** behält Reparaturschritte zusammen
- **Error-Code-Solution Grouping** für bessere Techniker-Antworten
- **Multimodal Document Understanding** für optimale Kontexterhaltung

## AI-Chunking Strategien (Best Practice Research)

### **1. Vision-Guided Multimodal Chunking** (PRIMARY STRATEGY)
**89% Accuracy vs 78% bei traditionellem Chunking**[93][105]

```python
class VisionGuidedChunker:
    def __init__(self, ollama_client):
        self.ollama = ollama_client
        self.vision_model = "llava:13b"  # oder bakllava:7b
        
    def analyze_pdf_page_structure(self, page_image_base64, context=""):
        """
        Vision Model analysiert PDF-Seite visuell für optimale Segmentierung
        """
        prompt = f'''
        Analysiere diese Service Manual Seite visuell und identifiziere:
        
        CRITICAL RULES FOR SERVICE MANUALS:
        1. NEVER split numbered procedure steps
        2. Keep multi-page tables together 
        3. Group error codes with their solutions
        4. Maintain figure-text relationships
        5. Preserve connection point references
        
        Identify these content types:
        - PROCEDURE: Step-by-step repair instructions
        - ERROR_CODE: Error with diagnostic steps
        - TABLE: Multi-column data (may span pages)
        - DIAGRAM: Figure with explanatory text
        - PARTS_LIST: Component listings with numbers
        - CONNECTION: Cable/connector references
        
        Context from previous pages: {context}
        
        Return JSON with optimal chunk boundaries and types.
        '''
        
        response = self.ollama.generate(
            model=self.vision_model,
            prompt=prompt,
            images=[page_image_base64],
            format="json"
        )
        return response['response']
```

### **2. Agentic Content-Type Detection** (INTELLIGENT ROUTING)
**AI Agent wählt optimale Chunking-Strategie basierend auf Content**[85][88]

```python
class AgenticChunkingRouter:
    def __init__(self, ollama_client):
        self.ollama = ollama_client
        self.text_model = "llama3.1:8b"
        
    def determine_chunking_strategy(self, text_content, page_analysis):
        """
        LLM analysiert Content und wählt beste Chunking-Methode
        """
        prompt = f'''
        Analyze this service manual content and determine the optimal chunking strategy:
        
        Content: {text_content[:2000]}
        Visual Analysis: {page_analysis}
        
        Available Strategies:
        1. PROCEDURE_AWARE: Keep all numbered steps together (1., 2., 3...)
        2. TABLE_PRESERVING: Maintain table integrity across pages  
        3. ERROR_CODE_GROUPING: Group error codes with solutions
        4. DIAGRAM_LINKED: Keep figures with explanatory text
        5. SEMANTIC_BOUNDARY: Split at natural topic changes
        
        Manufacturer-Specific Considerations:
        - HP: CPMD documents have linked procedures
        - Konica Minolta: Error codes often have multi-step solutions
        - Canon: Service procedures reference multiple diagrams
        
        Return the best strategy name and reasoning.
        '''
        
        response = self.ollama.generate(
            model=self.text_model,
            prompt=prompt,
            format="json"
        )
        return response['response']
```

### **3. LLM-Guided Semantic Boundary Detection** (PRECISION SPLITTING)
**Findet optimale Trennpunkte ohne Kontext zu verlieren**[86][89]

```python
class SemanticBoundaryDetector:
    def __init__(self, ollama_client):
        self.ollama = ollama_client
        self.text_model = "llama3.1:8b"
        
    def find_optimal_split_points(self, text_blocks, content_type):
        """
        LLM identifiziert beste Stellen zum Splitten ohne Kontext zu verlieren
        """
        prompt = f'''
        Find the optimal split points in this service manual content:
        
        Content Type: {content_type}
        Text Blocks: {text_blocks}
        
        SPLITTING RULES:
        - NEVER split within numbered procedures
        - NEVER separate error codes from solutions  
        - NEVER break table rows
        - NEVER separate figure references from text
        - DO split at topic changes (new error code, new procedure)
        - DO split at section boundaries
        
        Analyze each paragraph transition and rate split-safety (1-10):
        10 = Perfect split point (topic change)
        5 = Neutral (could split if needed)
        1 = Never split (breaks context)
        
        Return array of paragraph indices with split scores.
        '''
        
        response = self.ollama.generate(
            model=self.text_model,
            prompt=prompt,
            format="json"
        )
        return response['response']
```

## Technische Architektur mit AI Integration

### Core Technologies (Enhanced)
- **Python 3.9+** als Hauptsprache
- **PyMuPDF (fitz)** für PDF-Verarbeitung und Bildextraktion
- **Ollama Client** für lokale LLM/Vision AI Integration
- **llava:13b** für Vision-Guided Page Analysis
- **llama3.1:8b** für Text-Based Semantic Analysis
- **Externe Supabase** Vector Database
- **Cloudflare R2** für Bildspeicherung
- **Sentence Transformers** für finale Embeddings

### AI-Enhanced Processing Pipeline
```
PDF Pages → Vision AI Analysis → Content Type Detection → Smart Chunking Strategy → LLM Boundary Detection → Optimized Chunks
```

## Implementierung

### 1. Setup-Wizard (Erweitert mit Ollama)

**`setup_wizard.py` erstellen:**

```python
#!/usr/bin/env python3
"""
AI-Enhanced PDF Extraction System - Setup Wizard
Konfiguration mit Ollama Model Setup
"""

import os
import sys
import json
import getpass
import requests
from pathlib import Path
from supabase import create_client
import boto3
from sentence_transformers import SentenceTransformer

class AISetupWizard:
    def __init__(self):
        self.config = {}
        self.config_file = Path("config.json")
        self.ollama_base_url = "http://localhost:11434"
        
    def welcome_message(self):
        print("=" * 70)
        print("    AI-ENHANCED PDF EXTRACTION SYSTEM - SETUP WIZARD")
        print("=" * 70)
        print("Dieses System nutzt Ollama für intelligentes AI-Chunking")
        print("Bessere Accuracy durch Vision AI und LLM-Segmentation")
        print("=" * 70)
        print()
        
    def check_ollama_installation(self):
        print("🤖 OLLAMA INSTALLATION PRÜFEN")
        print("-" * 30)
        
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                installed_models = [model['name'] for model in models]
                
                print("✅ Ollama läuft erfolgreich!")
                print(f"   Installierte Modelle: {len(installed_models)}")
                for model in installed_models:
                    print(f"   - {model}")
                
                return installed_models
            else:
                print("❌ Ollama läuft, aber API nicht erreichbar")
                return []
                
        except requests.exceptions.RequestException:
            print("❌ Ollama ist nicht verfügbar!")
            print("   Bitte starten Sie Ollama mit: 'ollama serve'")
            return []
    
    def setup_required_models(self, installed_models):
        print("📥 ERFORDERLICHE MODELLE SETUP")
        print("-" * 30)
        
        required_models = {
            "llama3.1:8b": "Text Analysis und Semantic Boundary Detection",
            "llava:13b": "Vision-Guided PDF Page Analysis (empfohlen)",
            "bakllava:7b": "Leichtere Vision Alternative"
        }
        
        missing_models = []
        
        for model, description in required_models.items():
            if not any(model in installed for installed in installed_models):
                print(f"⚠️  Fehlt: {model} - {description}")
                missing_models.append(model)
            else:
                print(f"✅ Verfügbar: {model}")
        
        if missing_models:
            print(f"\\n📥 {len(missing_models)} Modelle müssen heruntergeladen werden")
            
            # Vision Model Auswahl
            if "llava:13b" in missing_models and "bakllava:7b" in missing_models:
                choice = input("Vision Model wählen - (1) llava:13b (besser) oder (2) bakllava:7b (kleiner): ")
                if choice == "2":
                    missing_models.remove("llava:13b")
                    self.config['vision_model'] = "bakllava:7b"
                else:
                    missing_models.remove("bakllava:7b") 
                    self.config['vision_model'] = "llava:13b"
            
            # Text Model ist immer erforderlich
            self.config['text_model'] = "llama3.1:8b"
            
            install_now = input(f"Modelle jetzt herunterladen? (j/n): ")
            if install_now.lower() == 'j':
                return self.download_models(missing_models)
            else:
                print("⚠️  Setup kann ohne Modelle nicht fortgesetzt werden")
                return False
        
        # Setze verfügbare Modelle
        for installed in installed_models:
            if "llava" in installed or "bakllava" in installed:
                self.config['vision_model'] = installed
            elif "llama3.1" in installed:
                self.config['text_model'] = installed
                
        return True
    
    def download_models(self, models):
        print("📥 MODELLE HERUNTERLADEN...")
        print("-" * 30)
        
        for model in models:
            print(f"⏳ Lade {model} herunter...")
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/pull",
                    json={"name": model},
                    timeout=1800  # 30 Minuten Timeout
                )
                
                if response.status_code == 200:
                    print(f"✅ {model} erfolgreich installiert!")
                else:
                    print(f"❌ Fehler beim Download von {model}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Download Fehler für {model}: {e}")
                return False
        
        return True
    
    def test_ollama_models(self):
        print("🧪 MODELLE TESTEN...")
        print("-" * 30)
        
        # Test Text Model
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.config['text_model'],
                    "prompt": "Test: What is a service manual?",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Text Model ({self.config['text_model']}) funktioniert!")
            else:
                print(f"❌ Text Model Test fehlgeschlagen")
                return False
                
        except Exception as e:
            print(f"❌ Text Model Fehler: {e}")
            return False
        
        # Test Vision Model  
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate", 
                json={
                    "model": self.config['vision_model'],
                    "prompt": "Describe this image briefly.",
                    "images": ["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Vision Model ({self.config['vision_model']}) funktioniert!")
            else:
                print(f"❌ Vision Model Test fehlgeschlagen")
                return False
                
        except Exception as e:
            print(f"❌ Vision Model Fehler: {e}")
            return False
        
        return True
    
    def collect_ai_config(self):
        print("⚙️  AI PROCESSING KONFIGURATION")
        print("-" * 30)
        
        chunk_strategy = input("Chunking Strategy [intelligent]: ").strip() or "intelligent"
        self.config['chunking_strategy'] = chunk_strategy
        
        vision_analysis = input("Vision Analysis aktivieren? (j/n) [j]: ").strip() or "j"
        self.config['use_vision_analysis'] = vision_analysis.lower() == 'j'
        
        semantic_boundaries = input("LLM Semantic Boundary Detection? (j/n) [j]: ").strip() or "j"
        self.config['use_semantic_boundaries'] = semantic_boundaries.lower() == 'j'
        
        max_chunk_size = input("Max Chunk Size [600]: ").strip() or "600"
        self.config['max_chunk_size'] = int(max_chunk_size)
        
        min_chunk_size = input("Min Chunk Size [200]: ").strip() or "200"
        self.config['min_chunk_size'] = int(min_chunk_size)
        
        print("✅ AI Konfiguration erfasst")
        print()
    
    # ... [Alle anderen Methoden aus dem vorherigen Setup bleiben gleich]
    # collect_supabase_config, collect_r2_config, etc.
    
    def run_setup(self):
        self.welcome_message()
        
        # Ollama Setup
        installed_models = self.check_ollama_installation()
        if not installed_models and not self.setup_required_models(installed_models):
            print("❌ Setup abgebrochen - Ollama Modelle erforderlich")
            return False
            
        if not self.test_ollama_models():
            print("❌ Setup abgebrochen - Model Tests fehlgeschlagen")
            return False
        
        # AI Configuration
        self.collect_ai_config()
        
        # Standard Konfiguration
        self.collect_supabase_config()
        self.collect_r2_config() 
        self.collect_processing_config()
        
        # ... Rest der Setup-Logik bleibt gleich
        
        return True

def main():
    wizard = AISetupWizard()
    wizard.run_setup()

if __name__ == "__main__":
    main()
```

### 2. AI-Enhanced PDF Processor

**`ai_pdf_processor.py` erstellen:**

```python
#!/usr/bin/env python3
"""
AI-Enhanced PDF Extraction System
Smart Chunking mit Ollama Vision + LLM Integration
"""

import os
import sys
import json
import base64
import hashlib
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from io import BytesIO

import fitz  # PyMuPDF
import numpy as np
import requests
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from supabase import create_client
import boto3
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer

@dataclass
class AIProcessingConfig:
    # Standard Config
    supabase_url: str
    supabase_key: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    documents_path: str
    
    # AI Config
    ollama_base_url: str = "http://localhost:11434"
    vision_model: str = "llava:13b"
    text_model: str = "llama3.1:8b"
    use_vision_analysis: bool = True
    use_semantic_boundaries: bool = True
    chunking_strategy: str = "intelligent"
    max_chunk_size: int = 600
    min_chunk_size: int = 200

class OllamaClient:
    """Client für lokale Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def generate_text(self, model: str, prompt: str, format: str = None) -> str:
        """Generate text response from LLM"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            if format:
                payload["format"] = format
                
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            return response.json().get('response', '')
            
        except Exception as e:
            logging.error(f"Ollama text generation error: {e}")
            return ""
    
    def generate_vision(self, model: str, prompt: str, image_base64: str, format: str = None) -> str:
        """Generate response from vision model with image"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
            
            if format:
                payload["format"] = format
                
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )
            
            response.raise_for_status()
            return response.json().get('response', '')
            
        except Exception as e:
            logging.error(f"Ollama vision generation error: {e}")
            return ""

class VisionGuidedChunker:
    """Vision AI für PDF Page Analysis und Smart Chunking"""
    
    def __init__(self, ollama_client: OllamaClient, config: AIProcessingConfig):
        self.ollama = ollama_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def pdf_page_to_base64(self, page: fitz.Page, dpi=150) -> str:
        """Convert PDF page to base64 image for vision analysis"""
        try:
            # Render page as image
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return img_base64
            
        except Exception as e:
            self.logger.error(f"Error converting page to base64: {e}")
            return ""
    
    def analyze_page_structure(self, page: fitz.Page, page_text: str, context: str = "") -> Dict:
        """Vision analysis of PDF page for optimal chunking"""
        
        if not self.config.use_vision_analysis:
            return {"content_types": ["text"], "split_points": [], "strategy": "semantic"}
        
        page_image = self.pdf_page_to_base64(page)
        if not page_image:
            return {"content_types": ["text"], "split_points": [], "strategy": "semantic"}
        
        prompt = f'''
        Analyze this service manual page visually and identify optimal chunking strategy:
        
        CONTENT TYPES TO IDENTIFY:
        - PROCEDURE: Numbered steps (1., 2., 3... or Step 1, Step 2...)  
        - ERROR_CODE: Error codes with diagnostic/repair steps
        - TABLE: Tabular data (may span multiple pages)
        - DIAGRAM: Figures, charts, technical drawings
        - PARTS_LIST: Component listings with part numbers
        - CONNECTION: Cable/connector references and pinouts
        - WARNING: Safety notices and cautions
        
        TEXT CONTENT: {page_text[:1000]}
        PREVIOUS PAGE CONTEXT: {context}
        
        CRITICAL RULES:
        1. NEVER split numbered procedures - keep all steps together
        2. Group error codes with their complete solutions
        3. Preserve table integrity across pages
        4. Keep figures with their explanatory text
        5. Maintain connection point references together
        
        Return JSON with:
        {{
            "content_types": ["PROCEDURE", "ERROR_CODE", etc],
            "split_safe_points": [paragraph_indices_where_safe_to_split],
            "keep_together_ranges": [[start_para, end_para], ...],
            "recommended_strategy": "procedure_aware|table_preserving|error_grouping|semantic",
            "confidence": 0.0-1.0
        }}
        '''
        
        response = self.ollama.generate_vision(
            model=self.config.vision_model,
            prompt=prompt,
            image_base64=page_image,
            format="json"
        )
        
        try:
            return json.loads(response) if response else {
                "content_types": ["text"], 
                "split_safe_points": [], 
                "recommended_strategy": "semantic"
            }
        except json.JSONDecodeError:
            self.logger.warning(f"Vision analysis returned invalid JSON: {response[:200]}")
            return {"content_types": ["text"], "split_safe_points": [], "recommended_strategy": "semantic"}

class AgenticChunkingRouter:
    """AI Agent für intelligente Chunking-Strategie Auswahl"""
    
    def __init__(self, ollama_client: OllamaClient, config: AIProcessingConfig):
        self.ollama = ollama_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def determine_optimal_strategy(self, text_content: str, vision_analysis: Dict, manufacturer: str) -> Dict:
        """LLM determines best chunking approach based on content analysis"""
        
        prompt = f'''
        Analyze this service manual content and determine optimal chunking strategy:
        
        MANUFACTURER: {manufacturer}
        TEXT SAMPLE: {text_content[:1500]}
        VISION ANALYSIS: {json.dumps(vision_analysis)}
        
        AVAILABLE STRATEGIES:
        1. PROCEDURE_AWARE: Keep numbered procedures intact (best for repair steps)
        2. TABLE_PRESERVING: Maintain table integrity across pages
        3. ERROR_CODE_GROUPING: Group error codes with complete solutions
        4. DIAGRAM_LINKED: Keep figures with explanatory text  
        5. CONNECTION_PRESERVING: Keep connector/pinout info together
        6. SEMANTIC_BOUNDARY: Split at natural topic boundaries
        
        MANUFACTURER-SPECIFIC RULES:
        - HP: CPMD documents have linked multi-step procedures
        - Konica Minolta: Error codes often have 5-10 solution steps
        - Canon: Service procedures reference multiple diagrams
        - Kyocera: Connection points are frequently referenced
        - Brother: Parts lists are integrated with procedures
        
        Analyze content patterns and select best strategy with reasoning.
        
        Return JSON:
        {{
            "primary_strategy": "strategy_name",
            "secondary_strategy": "fallback_strategy", 
            "reasoning": "why this strategy is optimal",
            "estimated_chunk_count": number,
            "special_handling": ["multi_page_table", "figure_refs", etc]
        }}
        '''
        
        response = self.ollama.generate_text(
            model=self.config.text_model,
            prompt=prompt,
            format="json"
        )
        
        try:
            return json.loads(response) if response else {
                "primary_strategy": "semantic_boundary",
                "reasoning": "fallback to semantic chunking"
            }
        except json.JSONDecodeError:
            return {"primary_strategy": "semantic_boundary", "reasoning": "JSON parse error fallback"}

class SemanticBoundaryDetector:
    """LLM-guided precision splitting at optimal boundaries"""
    
    def __init__(self, ollama_client: OllamaClient, config: AIProcessingConfig):
        self.ollama = ollama_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def find_optimal_split_points(self, paragraphs: List[str], strategy_info: Dict) -> List[Dict]:
        """LLM identifies best split points without breaking context"""
        
        if not self.config.use_semantic_boundaries:
            return self.fallback_splitting(paragraphs)
        
        # Prepare paragraph context for analysis
        paragraph_context = []
        for i, para in enumerate(paragraphs[:20]):  # Limit for token efficiency
            paragraph_context.append(f"[{i}] {para[:200]}...")
        
        prompt = f'''
        Find optimal split points in this service manual content:
        
        STRATEGY: {strategy_info.get('primary_strategy', 'semantic')}
        SPECIAL HANDLING: {strategy_info.get('special_handling', [])}
        
        PARAGRAPHS:
        {chr(10).join(paragraph_context)}
        
        SPLITTING SAFETY RULES (Rate each transition 1-10):
        10 = PERFECT split (new topic/section/procedure)
        8-9 = GOOD split (subtopic change, different error code)
        5-7 = NEUTRAL (could split if needed for size)
        3-4 = RISKY (might break context)
        1-2 = NEVER split (breaks procedure steps, table rows, etc)
        
        NEVER SPLIT WITHIN:
        - Numbered procedures (Step 1, Step 2...)
        - Error code solutions 
        - Table rows
        - Figure explanations
        - Connection point descriptions
        
        GOOD SPLIT POINTS:
        - Between different error codes
        - Between different procedures  
        - At major section breaks
        - Between unrelated topics
        
        Return JSON array with split analysis:
        [
            {{
                "paragraph_index": 0,
                "split_score": 7,
                "reasoning": "Topic change from diagnosis to repair",
                "chunk_type": "procedure"
            }}, ...
        ]
        '''
        
        response = self.ollama.generate_text(
            model=self.config.text_model,
            prompt=prompt,
            format="json"
        )
        
        try:
            return json.loads(response) if response else self.fallback_splitting(paragraphs)
        except json.JSONDecodeError:
            self.logger.warning("Semantic boundary detection failed, using fallback")
            return self.fallback_splitting(paragraphs)
    
    def fallback_splitting(self, paragraphs: List[str]) -> List[Dict]:
        """Fallback splitting based on size and simple heuristics"""
        split_points = []
        current_size = 0
        
        for i, para in enumerate(paragraphs):
            current_size += len(para)
            
            # Simple heuristics for safe split points
            is_safe_split = (
                current_size > self.config.min_chunk_size and
                not para.strip().startswith(('1.', '2.', '3.', 'Step', '•')) and
                not re.search(r'(Figure|Table|Error|Code)\s+\d', para)
            )
            
            if is_safe_split and current_size > self.config.max_chunk_size * 0.8:
                split_points.append({
                    "paragraph_index": i,
                    "split_score": 6,
                    "reasoning": "Size-based split with basic safety check",
                    "chunk_type": "text"
                })
                current_size = 0
        
        return split_points

class AIEnhancedPDFProcessor:
    """Main processor with AI-guided chunking capabilities"""
    
    def __init__(self, config: AIProcessingConfig):
        self.config = config
        self.supabase = create_client(config.supabase_url, config.supabase_key)
        self.r2_client = self._init_r2_client()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # AI Components
        self.ollama = OllamaClient(config.ollama_base_url)
        self.vision_chunker = VisionGuidedChunker(self.ollama, config)
        self.agentic_router = AgenticChunkingRouter(self.ollama, config)
        self.boundary_detector = SemanticBoundaryDetector(self.ollama, config)
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_r2_client(self):
        """Initialize Cloudflare R2 client"""
        return boto3.client(
            's3',
            endpoint_url=f'https://{self.config.r2_account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=self.config.r2_access_key_id,
            aws_secret_access_key=self.config.r2_secret_access_key
        )
    
    def ai_enhanced_chunking_process(self, pdf_document: fitz.Document, manufacturer: str, 
                                   document_type: str, file_path: str, file_hash: str) -> List[Dict]:
        """Main AI-enhanced chunking pipeline"""
        
        self.logger.info(f"Starting AI-enhanced chunking for {manufacturer} {document_type}")
        
        all_chunks = []
        document_context = ""
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            
            self.logger.info(f"Processing page {page_num + 1} with AI analysis")
            
            # Step 1: Vision-guided page structure analysis
            vision_analysis = self.vision_chunker.analyze_page_structure(
                page, page_text, document_context
            )
            
            # Step 2: Agentic strategy determination  
            strategy_info = self.agentic_router.determine_optimal_strategy(
                page_text, vision_analysis, manufacturer
            )
            
            # Step 3: Create page chunks using AI insights
            page_chunks = self.create_ai_guided_chunks(
                page_text, page_num + 1, strategy_info, vision_analysis,
                manufacturer, document_type, file_path, file_hash
            )
            
            all_chunks.extend(page_chunks)
            document_context = page_text[-500:]  # Keep context for next page
            
            self.logger.info(f"Page {page_num + 1}: Created {len(page_chunks)} chunks using {strategy_info.get('primary_strategy', 'semantic')} strategy")
        
        self.logger.info(f"AI-enhanced chunking complete: {len(all_chunks)} total chunks")
        return all_chunks
    
    def create_ai_guided_chunks(self, page_text: str, page_num: int, strategy_info: Dict,
                               vision_analysis: Dict, manufacturer: str, document_type: str,
                               file_path: str, file_hash: str) -> List[Dict]:
        """Create chunks using AI guidance and optimal split points"""
        
        # Split into paragraphs for analysis
        paragraphs = [p.strip() for p in page_text.split('\\n\\n') if p.strip()]
        
        if len(paragraphs) < 2:
            # Single paragraph or very short text
            return [self.create_chunk_data(
                page_text, page_num, 0, manufacturer, document_type,
                file_path, file_hash, strategy_info.get('primary_strategy', 'text')
            )]
        
        # Get optimal split points using LLM
        split_analysis = self.boundary_detector.find_optimal_split_points(
            paragraphs, strategy_info
        )
        
        # Create chunks based on AI-identified boundaries
        chunks = []
        current_chunk_paragraphs = []
        chunk_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            current_chunk_paragraphs.append(paragraph)
            
            # Check if this is a good split point
            should_split = False
            chunk_type = "text"
            
            for split_info in split_analysis:
                if split_info.get('paragraph_index') == i:
                    if split_info.get('split_score', 0) >= 7:
                        should_split = True
                        chunk_type = split_info.get('chunk_type', 'text')
                        break
            
            # Also split if chunk gets too large
            current_content = '\\n\\n'.join(current_chunk_paragraphs)
            if len(current_content) > self.config.max_chunk_size:
                should_split = True
            
            # Create chunk if splitting or at end
            if should_split or i == len(paragraphs) - 1:
                if current_chunk_paragraphs:
                    chunk_data = self.create_chunk_data(
                        current_content, page_num, chunk_index, manufacturer,
                        document_type, file_path, file_hash, chunk_type
                    )
                    chunks.append(chunk_data)
                    chunk_index += 1
                    current_chunk_paragraphs = []
        
        return chunks
    
    def create_chunk_data(self, chunk_text: str, page_num: int, chunk_index: int,
                         manufacturer: str, document_type: str, file_path: str,
                         file_hash: str, chunk_type: str) -> Dict:
        """Create comprehensive chunk data with AI-enhanced metadata"""
        
        # Generate embedding
        try:
            embedding = self.embedding_model.encode(chunk_text).tolist()
        except Exception as e:
            self.logger.error(f"Error generating chunk embedding: {e}")
            embedding = None
        
        # Extract enhanced metadata using AI insights
        error_codes = self.extract_error_codes_ai(chunk_text, manufacturer)
        figure_refs = self.extract_figure_references_ai(chunk_text)
        connections = self.extract_connection_points_ai(chunk_text)
        procedures = self.extract_procedures_ai(chunk_text)
        
        return {
            'content': chunk_text,
            'embedding': embedding,
            'manufacturer': manufacturer,
            'document_type': document_type,
            'file_path': file_path,
            'original_filename': os.path.basename(file_path),
            'file_hash': file_hash,
            'chunk_type': chunk_type,
            'page_number': page_num,
            'chunk_index': chunk_index,
            'error_codes': error_codes,
            'figure_references': figure_refs,
            'connection_points': connections,
            'procedures': procedures,
            'metadata': {
                'processing_date': datetime.now(timezone.utc).isoformat(),
                'chunk_length': len(chunk_text),
                'page_context': f"Page {page_num}",
                'ai_enhanced': True,
                'chunking_strategy': 'ai_guided'
            }
        }
    
    def extract_error_codes_ai(self, text: str, manufacturer: str) -> List[str]:
        """AI-enhanced error code extraction with context understanding"""
        
        # First use regex patterns (fast)
        error_patterns = {
            'HP': [r'\\b[0-9A-F]{4}h\\b', r'\\bC[0-9]{4}\\b', r'\\b[0-9]{2}\\.[0-9]{2}\\.[0-9]{2}\\b'],
            'Konica Minolta': [r'\\bC[0-9]{4}\\b'],
            'Kyocera': [r'\\bC[0-9]{4}\\b', r'\\bCF[0-9]{3}\\b'],
            'Canon': [r'\\bE[0-9]{3}\\b'],
            'Brother': [r'\\b[A-Z]{2}-[0-9]{2}\\b'],
            'Xerox': [r'\\b0[0-9]{2}\\b'],
            'Lexmark': [r'\\b[0-9]{3}\\b'],
            'Fujifilm': [r'\\bE[0-9]{3}\\b'],
            'UTAX': [r'\\bERROR [0-9]{2}\\b']
        }
        
        regex_codes = []
        if manufacturer in error_patterns:
            for pattern in error_patterns[manufacturer]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                regex_codes.extend(matches)
        
        # If we found codes with regex, return them (faster)
        if regex_codes:
            return list(set(regex_codes))
        
        # If no regex matches and text mentions errors, use AI
        if any(word in text.lower() for word in ['error', 'fehler', 'code', 'fault']):
            prompt = f'''
            Extract error codes from this {manufacturer} service manual text:
            
            TEXT: {text[:800]}
            
            Look for {manufacturer}-specific error code patterns:
            - HP: C0001, 0001h, 10.20.30 formats
            - Konica Minolta: C2557, C3101 formats  
            - Kyocera: C0030, CF000 formats
            - Canon: E301, E502 formats
            
            Return only valid error codes, one per line.
            '''
            
            response = self.ollama.generate_text(
                model=self.config.text_model,
                prompt=prompt
            )
            
            if response:
                ai_codes = [code.strip() for code in response.split('\\n') if code.strip()]
                return ai_codes[:10]  # Limit to avoid noise
        
        return []
    
    def extract_figure_references_ai(self, text: str) -> List[str]:
        """AI-enhanced figure reference extraction"""
        patterns = [
            r'\\bFigure\\s+[0-9]+(?:-[0-9]+)?(?:\\.[0-9]+)*\\b',
            r'\\bFig\\.?\\s+[0-9]+(?:-[0-9]+)?(?:\\.[0-9]+)*\\b',
            r'\\bAbbildung\\s+[0-9]+(?:-[0-9]+)?(?:\\.[0-9]+)*\\b',
            r'\\bAbb\\.?\\s+[0-9]+(?:-[0-9]+)?(?:\\.[0-9]+)*\\b'
        ]
        
        figure_refs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            figure_refs.extend(matches)
        
        return list(set(figure_refs))
    
    def extract_connection_points_ai(self, text: str) -> List[str]:
        """AI-enhanced connection point extraction"""
        patterns = [
            r'\\b[A-Z]+[0-9]+-[A-Z]+[0-9]+\\b',  # P1-J2 format
            r'\\bConnector\\s+[A-Z]+[0-9]+\\b',
            r'\\bAnschluss\\s+[A-Z]+[0-9]+\\b',
            r'\\b[A-Z]{1,3}[0-9]+\\b'  # CN1, J12 format
        ]
        
        connections = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            connections.extend(matches)
        
        return list(set(connections))
    
    def extract_procedures_ai(self, text: str) -> List[str]:
        """Extract procedure step references"""
        step_patterns = [
            r'\\bStep\\s+[0-9]+\\b',
            r'\\bSchritt\\s+[0-9]+\\b',
            r'\\b[0-9]+\\.\\s+[A-Z]',  # 1. Something
        ]
        
        procedures = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            procedures.extend(matches)
        
        return list(set(procedures))
    
    # ... [Rest der Methoden bleiben gleich: process_pdf, store_chunks, etc.]
    
    def process_pdf(self, file_path: str) -> bool:
        """Main PDF processing with AI enhancement"""
        try:
            self.logger.info(f"Starting AI-enhanced processing: {file_path}")
            
            # Parse manufacturer and document type from folder structure
            manufacturer, document_type = self.parse_file_path(file_path)
            self.logger.info(f"Detected: {manufacturer} - {document_type}")
            
            # Get file info
            file_hash = self.get_file_hash(file_path)
            
            # Check if already processed
            if self.is_already_processed(file_hash):
                self.logger.info(f"File already processed, skipping: {file_path}")
                return True
            
            # Log processing start
            self.log_processing_start(file_path, file_hash)
            
            # Open PDF
            pdf_document = fitz.open(file_path)
            
            # Extract images (same as before)
            images = self.extract_images_from_pdf(pdf_document, file_hash)
            self.logger.info(f"Extracted {len(images)} images")
            
            # AI-Enhanced chunking process
            chunks = self.ai_enhanced_chunking_process(
                pdf_document, manufacturer, document_type, file_path, file_hash
            )
            self.logger.info(f"AI-enhanced chunking created {len(chunks)} chunks")
            
            # Store in database
            self.store_chunks_in_database(chunks, images)
            
            # Update processing log
            self.log_processing_complete(file_hash, len(chunks), len(images))
            
            pdf_document.close()
            self.logger.info(f"Successfully processed with AI enhancement: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in AI-enhanced processing {file_path}: {e}")
            if 'file_hash' in locals():
                self.log_processing_error(file_hash, str(e))
            return False
    
    # ... [Alle anderen Methoden aus dem vorherigen Code bleiben gleich]

# ... [PDFWatcher, load_config, main Funktionen bleiben gleich]

def load_config() -> AIProcessingConfig:
    """Load AI-enhanced configuration from config.json"""
    config_file = Path("config.json")
    
    if not config_file.exists():
        print("❌ config.json nicht gefunden!")
        print("   Führen Sie zuerst 'python setup_wizard.py' aus.")
        sys.exit(1)
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        return AIProcessingConfig(
            supabase_url=config_data['supabase_url'],
            supabase_key=config_data['supabase_key'],
            r2_account_id=config_data['r2_account_id'],
            r2_access_key_id=config_data['r2_access_key_id'],
            r2_secret_access_key=config_data['r2_secret_access_key'],
            r2_bucket_name=config_data['r2_bucket_name'],
            documents_path=config_data['documents_path'],
            vision_model=config_data.get('vision_model', 'llava:13b'),
            text_model=config_data.get('text_model', 'llama3.1:8b'),
            use_vision_analysis=config_data.get('use_vision_analysis', True),
            use_semantic_boundaries=config_data.get('use_semantic_boundaries', True),
            chunking_strategy=config_data.get('chunking_strategy', 'intelligent'),
            max_chunk_size=config_data.get('max_chunk_size', 600),
            min_chunk_size=config_data.get('min_chunk_size', 200)
        )
    except Exception as e:
        print(f"❌ Fehler beim Laden der AI-Konfiguration: {e}")
        sys.exit(1)

def main():
    print("=" * 70)
    print("    AI-ENHANCED PDF EXTRACTION SYSTEM")  
    print("=" * 70)
    
    # Load AI configuration
    config = load_config()
    print(f"✅ AI-Konfiguration geladen")
    print(f"   Vision Model: {config.vision_model}")
    print(f"   Text Model: {config.text_model}")
    print(f"   Strategy: {config.chunking_strategy}")
    
    # Initialize AI-enhanced processor
    processor = AIEnhancedPDFProcessor(config)
    print("✅ AI-Enhanced PDF Processor initialisiert")
    
    # Test Ollama connection
    test_response = processor.ollama.generate_text(
        config.text_model, 
        "Test: Respond with 'AI Ready'"
    )
    
    if "ready" in test_response.lower():
        print("✅ Ollama AI Models bereit")
    else:
        print("⚠️  Ollama Verbindung könnte instabil sein")
    
    # Process existing files
    process_existing_files(processor, config.documents_path)
    
    # Start file watcher
    print("🤖 Starte AI-überwachtes PDF Processing...")
    event_handler = PDFWatcher(processor)
    observer = Observer()
    observer.schedule(event_handler, config.documents_path, recursive=True)
    observer.start()
    
    print(f"✅ AI-Enhanced System läuft! Überwache: {config.documents_path}")
    print("   🧠 Vision AI analysiert PDF Seiten für optimales Chunking")
    print("   🎯 LLM erkennt Verfahrensschritte und Fehlercodes intelligent")
    print("   📊 Semantic Boundary Detection verhindert Kontext-Verlust")
    print("   Drücken Sie Ctrl+C zum Beenden...")
    print("=" * 70)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\\n🛑 AI-Enhanced System wird beendet...")
    
    observer.join()
    print("✅ AI-Enhanced System erfolgreich beendet")

if __name__ == "__main__":
    main()
```

### 3. Erweiterte Dependencies

```
# AI-Enhanced Requirements mit Ollama Support
PyMuPDF==1.23.26
sentence-transformers==2.7.0
numpy==1.26.4
watchdog==4.0.1
boto3==1.34.162
psycopg2-binary==2.9.9
requests==2.31.0
pillow==10.4.0
python-dotenv==1.0.1
```

## Copilot Implementation Steps mit AI Integration

### Phase 1: AI Setup & Validation  
1. **setup_wizard.py** - Ollama Model Download und Testing implementieren
2. **OllamaClient** - Request/Response Handling für lokale API
3. **Model Validation** - Vision + Text Model Funktionalität prüfen

### Phase 2: Vision-Guided Analysis
1. **VisionGuidedChunker** - PDF Page zu Base64 Konvertierung
2. **Page Structure Analysis** - Vision AI für Layout-Erkennung
3. **Content Type Detection** - Procedures, Tables, Error Codes erkennen

### Phase 3: Agentic Intelligence
1. **AgenticChunkingRouter** - Strategie-Auswahl basierend auf Content
2. **Strategy Determination** - Manufacturer-spezifische Regeln
3. **Confidence Scoring** - AI Confidence für Entscheidungen

### Phase 4: Semantic Boundaries
1. **SemanticBoundaryDetector** - LLM-guided Split Point Detection
2. **Context Preservation** - Verfahrensschritte zusammenhalten
3. **Safety Scoring** - Split-Risiko bewerten (1-10 Scale)

### Phase 5: Enhanced Processing Pipeline
1. **AIEnhancedPDFProcessor** - Integration aller AI Components
2. **Multi-Modal Chunking** - Vision + Text Analysis kombinieren
3. **Intelligent Metadata** - AI-extracted Error Codes, Procedures, etc.

## Best Practice Benefits

### **🎯 Vision-Guided Chunking** 
- **89% vs 78% Accuracy** gegenüber traditionellem Chunking
- **Erkennt mehrseitige Tabellen** automatisch
- **Behält Verfahrensschritte zusammen**
- **Versteht Diagramm-Text Beziehungen**

### **🧠 LLM Semantic Analysis**
- **Intelligente Split-Point Detection** 
- **Context-Aware Boundary Recognition**
- **Manufacturer-Specific Rules** (HP CPMD, Konica Error Codes)
- **Procedure Step Preservation**

### **🤖 Agentic Strategy Selection**
- **Content-Type Detection** (Procedure, Error Code, Table, Diagram)
- **Dynamic Strategy Switching** basierend auf Seiteninhalt
- **Confidence-Based Fallbacks** für robuste Verarbeitung

## Performance & Lokale Vorteile

### **💻 Lokale Ollama Integration**
- **Keine API Kosten** für AI-Chunking
- **Datenschutz** - PDFs bleiben lokal
- **Schnelle Response** ohne Network Latency
- **Skalierbar** ohne Rate Limits

### **⚡ Optimierte Models**
- **llava:13b** für Vision Analysis (oder bakllava:7b für weniger RAM)
- **llama3.1:8b** für Text Analysis
- **Sentence Transformers** für finale Embeddings
- **Fallback Strategies** bei AI-Fehlern

**Das AI-Enhanced System ist die modernste Lösung für Service Manual Processing! 🚀**

Die Kombination aus **Vision AI + LLM Guidance** bringt **signifikant bessere Chunking-Qualität** und **robuste Verarbeitung** komplexer technischer Dokumentationen.