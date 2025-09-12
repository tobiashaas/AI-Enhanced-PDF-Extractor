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

# Import refactored modules  
from r2_storage_client import R2StorageClient, R2Config
from database_client import DatabaseClient
from image_processor import ImageProcessor
from metrics_collector import MetricsCollector

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
    
    # R2 Public Access Config
    r2_public_domain_id: str = None  # For public URLs (can differ from account_id)
    
    # AI Config
    ollama_base_url: str = "http://localhost:11434"
    vision_model: str = "llava:7b"
    text_model: str = "llama3.1:8b"
    embedding_model: str = "embeddinggemma"  # Neues Ollama Embedding Model
    embedding_provider: str = "ollama"  # "ollama" oder "sentence_transformers"
    use_vision_analysis: bool = True
    use_semantic_boundaries: bool = True
    chunking_strategy: str = "intelligent"
    max_chunk_size: int = 600
    min_chunk_size: int = 200
    
    # Hardware Acceleration Config
    parallel_workers: int = 4
    batch_size: int = 100
    use_metal_acceleration: bool = False
    use_cuda_acceleration: bool = False
    use_neural_engine: bool = False
    gpu_memory_fraction: float = 0.7
    ollama_gpu_layers: int = -1
    ollama_num_thread: int = 4
    memory_optimization: str = "balanced"

class OllamaClient:
    """Hardware-optimierter Client für lokale Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434", config: AIProcessingConfig = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.config = config
        
        # Hardware-Info für Timeout-Anpassungen
        self.hardware_info = {
            'has_metal': config.use_metal_acceleration if config else False,
            'has_cuda': config.use_cuda_acceleration if config else False,
            'has_neural_engine': config.use_neural_engine if config else False
        }
        
        # Check Ollama version on initialization
        self.check_ollama_version()
        
        # Hardware-spezifische Optimierungen anwenden
        if config:
            self.setup_hardware_acceleration()
    
    def check_ollama_version(self):
        """Check if Ollama version supports embeddinggemma and new API"""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                version = version_info.get('version', 'unknown')
                print(f"🔧 Ollama Version: {version}")
                
                # Check if version is sufficient for embeddinggemma
                if self.is_version_sufficient(version):
                    print("✅ Ollama version supports embeddinggemma model")
                else:
                    print(f"⚠️  Ollama {version} may not support embeddinggemma")
                    print("💡 Recommended: Update to Ollama >= 0.11.10")
            else:
                print("⚠️  Could not verify Ollama version")
        except Exception as e:
            print(f"⚠️  Ollama version check failed: {e}")
    
    def is_version_sufficient(self, version_str):
        """Check if version is >= 0.11.10"""
        try:
            # Parse version like "0.11.10" or "0.3.0"
            version_parts = [int(x) for x in version_str.split('.')]
            
            # Minimum required: 0.11.10
            min_version = [0, 11, 10]
            
            # Compare version parts
            for i in range(min(len(version_parts), len(min_version))):
                if version_parts[i] > min_version[i]:
                    return True
                elif version_parts[i] < min_version[i]:
                    return False
            
            # If all compared parts are equal, check length
            return len(version_parts) >= len(min_version)
            
        except Exception:
            return False  # Assume insufficient if can't parse
        
    def setup_hardware_acceleration(self):
        """Hardware-Beschleunigung konfigurieren"""
        if self.config.use_metal_acceleration:
            # Apple Silicon Metal optimiert
            os.environ["OLLAMA_GPU_LAYERS"] = str(self.config.ollama_gpu_layers)
            os.environ["OLLAMA_NUM_THREAD"] = str(self.config.ollama_num_thread)
            print("   ⚡ Apple Silicon Metal Beschleunigung aktiviert")
            
        elif self.config.use_cuda_acceleration:
            # NVIDIA CUDA optimiert
            os.environ["OLLAMA_GPU_LAYERS"] = str(self.config.ollama_gpu_layers)
            os.environ["OLLAMA_NUM_THREAD"] = str(self.config.ollama_num_thread)
            print(f"   ⚡ NVIDIA CUDA Beschleunigung aktiviert ({self.config.gpu_memory_fraction*100:.0f}% VRAM)")
            
        # Memory Optimization basierend auf Hardware
        if self.config.memory_optimization == "unified_memory":
            # Apple Silicon Unified Memory
            self.session.headers.update({"Connection": "keep-alive"})
            print("   🧠 Unified Memory Optimierung aktiv")
            
        elif self.config.memory_optimization == "gpu_optimized":
            # High-End GPU optimiert
            self.session.timeout = 300  # Längere Timeouts für große Models
            print("   🎮 GPU Memory Optimierung aktiv")
    
    def generate_text(self, model: str, prompt: str, format: str = None) -> str:
        """Hardware-optimiertes Text Generation"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_gpu": self.config.ollama_gpu_layers if self.config else -1,
                    "num_thread": self.config.ollama_num_thread if self.config else 4
                }
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
    
    def generate_vision(self, model: str, prompt: str, image_base64: str, format: str = None, options: dict = None) -> str:
        """Hardware-optimiertes Vision Generation mit erweiterten Timeouts und Fallback"""
        try:
            # Merge options mit defaults
            vision_options = {
                "num_gpu": self.config.ollama_gpu_layers if self.config else -1,
                "num_thread": self.config.ollama_num_thread if self.config else 4,
                "temperature": 0.1,  # Konsistentere Ergebnisse
                "top_p": 0.9
            }
            
            if options:
                vision_options.update(options)
            
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": vision_options
            }
            
            if format:
                payload["format"] = format
            
            # Timeout aus options oder Hardware-basiert
            timeout = options.get('timeout', 360) if options else 360  # Erhöht auf 6 Minuten
            if self.config and getattr(self.config, 'use_metal_acceleration', False):
                timeout = max(timeout, 420)  # 7 Minuten für Metal
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            response.raise_for_status()
            return response.json().get('response', '')
            
        except requests.exceptions.Timeout:
            logging.warning(f"Vision model timeout nach {timeout}s - verwende Fallback")
            return self._fallback_vision_analysis()
        except Exception as e:
            logging.error(f"Ollama vision generation error: {e}")
            return self._fallback_vision_analysis()
    
    def _fallback_vision_analysis(self) -> str:
        """Fallback für Vision AI bei Timeouts oder Fehlern"""
        return json.dumps({
            "content_types": ["text", "procedure"],
            "split_safe_points": [],
            "recommended_strategy": "semantic",
            "confidence": 0.5,
            "analysis_method": "fallback"
        })
    
    def generate_embedding(self, model: str, prompt: str) -> List[float]:
        """Ollama Embedding Generation"""
        try:
            payload = {
                "model": model,
                "prompt": prompt
            }
            
            # Use new API endpoint (/api/embed) - requires Ollama >= 0.3.0
            # embeddinggemma model requires Ollama >= 0.11.10 anyway
            response = self.session.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json().get('embedding', [])
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.error(f"Ollama API endpoint not found. Please update Ollama to version >= 0.11.10")
                logging.error(f"Current endpoint: {self.base_url}/api/embed")
                logging.error(f"Run: curl -fsSL https://ollama.ai/install.sh | sh")
            else:
                logging.error(f"Ollama HTTP error: {e}")
            return []
        except Exception as e:
            logging.error(f"Ollama embedding generation error: {e}")
            return []

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
        """Hardware-optimierte Vision analysis für PDF-Seiten mit Timeout-Handling"""
        
        if not self.config.use_vision_analysis:
            return {"content_types": ["text"], "split_points": [], "strategy": "semantic", "analysis_method": "disabled"}
        
        page_num = page.number + 1
        
        # Performance-Optimierung: Text-only Fallback für einfache Seiten  
        if len(page_text.strip()) < 100:
            return {
                "content_types": ["text"], 
                "split_points": [], 
                "strategy": "semantic",
                "analysis_method": "text_fallback_minimal"
            }
        
        try:
            page_image = self.pdf_page_to_base64(page)
            if not page_image:
                return self._fallback_text_analysis(page_text)

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

            # Hardware-optimierte Timeouts
            timeout_config = {
                'timeout': 300 if self.ollama.hardware_info.get('has_metal') else 240,
                'temperature': 0.3,
                'top_p': 0.8
            }

            response = self.ollama.generate_vision(
                model=self.config.vision_model,
                prompt=prompt,
                image_base64=page_image,
                format="json",
                options=timeout_config
            )

            try:
                result = json.loads(response) if response else self._fallback_text_analysis(page_text)
                result["analysis_method"] = "vision_ai"
                return result
                
            except json.JSONDecodeError:
                self.logger.warning(f"Vision analysis returned invalid JSON for page {page_num}: {response[:200]}")
                return self._fallback_text_analysis(page_text)
                
        except requests.exceptions.Timeout:
            self.logger.warning(f"Vision AI timeout für Seite {page_num} - verwende Text-Fallback")
            return self._fallback_text_analysis(page_text)
            
        except Exception as e:
            self.logger.warning(f"Vision analysis error für Seite {page_num}: {e}")
            return self._fallback_text_analysis(page_text)
    
    def _fallback_text_analysis(self, page_text: str) -> Dict:
        """Text-basierte Fallback-Analyse wenn Vision AI nicht verfügbar"""
        
        content_types = ["text"]
        split_safe_points = []
        
        # Heuristische Pattern Detection
        if re.search(r'^\s*\d+\.\s|\bstep\s+\d+\b', page_text.lower(), re.MULTILINE):
            content_types.append("PROCEDURE")
            strategy = "procedure_aware"
        elif re.search(r'\berror\s+code\b|\bfault\s+code\b', page_text.lower()):
            content_types.append("ERROR_CODE") 
            strategy = "error_grouping"
        elif re.search(r'\btable\b|\bpart\s+number\b', page_text.lower()):
            content_types.append("TABLE")
            strategy = "table_preserving"
        else:
            strategy = "semantic"
        
        # Einfache Split-Points basierend auf Absätzen
        paragraphs = page_text.split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 50 and not re.match(r'^\s*\d+\.', para):
                split_safe_points.append(i)
        
        return {
            "content_types": content_types,
            "split_safe_points": split_safe_points,
            "keep_together_ranges": [],
            "recommended_strategy": strategy,
            "confidence": 0.6,
            "analysis_method": "text_heuristic"
        }

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
    """Hardware-optimierter AI-Enhanced PDF Processor"""
    
    def __init__(self, config: AIProcessingConfig):
        self.config = config
        
        # Initialize refactored clients (backward compatible)
        self.supabase = create_client(config.supabase_url, config.supabase_key)
        self.db_client = DatabaseClient(config.supabase_url, config.supabase_key)
        
        # Original R2 client (keep for compatibility)
        self.r2_client = self._init_r2_client()
        
        # New R2 client (for gradual migration)
        r2_config = R2Config(
            account_id=config.r2_account_id,
            access_key_id=config.r2_access_key_id,
            secret_access_key=config.r2_secret_access_key,
            bucket_name=config.r2_bucket_name
        )
        self.r2_new_client = R2StorageClient(r2_config)
        
        # Image processing with vector and raster support
        image_config = getattr(config, 'image_processing', {})
        self.image_processor = ImageProcessor(
            render_dpi=image_config.get('render_dpi', 144),
            export_vectors=image_config.get('export_vectors', True),
            export_raster=image_config.get('export_raster', True)
        )
        self.r2_client = self._init_r2_client()
        
        # Hardware-optimiertes Embedding Model
        self.embedding_model = self._init_embedding_model()
        
        # AI Components mit Hardware-Optimierung
        self.ollama = OllamaClient(config.ollama_base_url, config)
        self.vision_chunker = VisionGuidedChunker(self.ollama, config)
        self.agentic_router = AgenticChunkingRouter(self.ollama, config)
        self.boundary_detector = SemanticBoundaryDetector(self.ollama, config)
        
        # Metrics and monitoring
        self.metrics_collector = MetricsCollector()
        
        # Enhanced Metrics & Monitoring
        self.metrics = MetricsCollector(enable_system_monitoring=True)
        
        # Legacy performance stats (backward compatibility)
        self.performance_stats = {
            "pages_processed": 0,
            "chunks_created": 0,
            "images_extracted": 0,
            "total_processing_time": 0
        }
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Hardware-Info anzeigen
        self._print_hardware_status()
        
    def _init_embedding_model(self):
        """Hardware-optimiertes Embedding Model laden - Ollama oder SentenceTransformers"""
        embedding_provider = getattr(self.config, 'embedding_provider', 'sentence_transformers')
        model_name = getattr(self.config, 'embedding_model', 'all-MiniLM-L6-v2')
        
        print(f"   🧠 Lade Embedding Model: {model_name} ({embedding_provider})")
        
        if embedding_provider == "ollama":
            # Neues Ollama embeddinggemma Model
            print("   ⚡ Ollama Embedding Model (embeddinggemma)")
            return None  # Kein lokales Model nötig, API-basiert
        
        # Fallback: SentenceTransformers für kompatibilität
        print(f"   🔧 SentenceTransformers Fallback: {model_name}")
        
        # Apple Silicon Optimierung
        if getattr(self.config, 'use_metal_acceleration', False):
            import torch
            if torch.backends.mps.is_available():
                print("   ⚡ Metal Performance Shaders für Embeddings aktiv")
                return SentenceTransformer(model_name, device='mps')
        
        # CUDA Optimierung  
        elif getattr(self.config, 'use_cuda_acceleration', False):
            import torch
            if torch.cuda.is_available():
                print(f"   ⚡ CUDA für Embeddings aktiv (GPU: {torch.cuda.get_device_name()})")
                return SentenceTransformer(model_name, device='cuda')
        
        # CPU Fallback
        print("   💻 CPU Embeddings (Standard)")
        return SentenceTransformer(model_name)
    
    def generate_chunk_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama or SentenceTransformers"""
        embedding_provider = getattr(self.config, 'embedding_provider', 'sentence_transformers')
        
        if embedding_provider == "ollama":
            # Ollama embeddinggemma API
            embedding_model = getattr(self.config, 'embedding_model', 'embeddinggemma')
            return self.ollama.generate_embedding(embedding_model, text)
        
        elif self.embedding_model is not None:
            # SentenceTransformers Fallback
            return self.embedding_model.encode(text).tolist()
        
        else:
            self.logger.warning("Kein Embedding Model verfügbar")
            return None
    
    def _print_hardware_status(self):
        """Hardware-Status anzeigen"""
        if getattr(self.config, 'use_metal_acceleration', False):
            print("   🍎 Apple Silicon Beschleunigung: Aktiv")
            print(f"   ⚡ Metal GPU Layers: {self.config.ollama_gpu_layers}")
            print(f"   🧵 CPU Threads: {self.config.ollama_num_thread}")
            
        elif getattr(self.config, 'use_cuda_acceleration', False):
            print("   🎮 NVIDIA CUDA Beschleunigung: Aktiv")  
            print(f"   ⚡ GPU Layers: {self.config.ollama_gpu_layers}")
            print(f"   📊 VRAM Usage: {self.config.gpu_memory_fraction*100:.0f}%")
            
        else:
            print("   💻 Standard CPU Verarbeitung")
            
        print(f"   🔄 Parallel Workers: {self.config.parallel_workers}")
        print(f"   📦 Batch Size: {self.config.batch_size}")
        
    def _init_r2_client(self):
        """Initialize Cloudflare R2 client"""
        return boto3.client(
            's3',
            endpoint_url=f'https://{self.config.r2_account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=self.config.r2_access_key_id,
            aws_secret_access_key=self.config.r2_secret_access_key
        )
    
    def ai_enhanced_chunking_process(self, pdf_document: fitz.Document, document_info: Dict, 
                                   file_path: str, file_hash: str) -> List[Dict]:
        """Hardware-optimierte AI-enhanced chunking pipeline für große PDFs"""
        
        manufacturer = document_info['manufacturer']
        document_type = document_info['document_type']
        total_pages = len(pdf_document)
        print(f"      📄 Verarbeite {total_pages} Seiten mit AI-Enhanced Chunking...")
        self.logger.info(f"Starting AI-enhanced chunking for {manufacturer} {document_type}")
        
        # Adaptive Strategie für große PDFs
        if total_pages > 1000:
            print(f"      ⚡ Große PDF erkannt ({total_pages} Seiten) - Aktiviere Batch-Verarbeitung")
            return self._process_large_pdf_optimized(pdf_document, document_info, file_path, file_hash)
        
        all_chunks = []
        document_context = ""
        
        # Get already processed pages to avoid reprocessing
        processed_pages = self.get_processed_pages(file_hash)
        if processed_pages:
            print(f"      ♻️  Gefunden: {len(processed_pages)} bereits verarbeitete Seiten")
            print(f"      ⏭️  Überspringe bereits verarbeitete Seiten...")
        
        for page_num in range(total_pages):
            # Skip already processed pages
            if page_num + 1 in processed_pages:
                if page_num % 50 == 0 or page_num < 10:
                    print(f"      ✅ Seite {page_num + 1}/{total_pages}: Bereits verarbeitet, überspringe...")
                continue
                
            page = pdf_document[page_num]
            page_text = page.get_text()
            
            # Progress anzeigen
            if page_num % 50 == 0 or page_num < 10:
                print(f"      🔍 Seite {page_num + 1}/{total_pages}: AI-Analyse läuft...")
            
            self.logger.info(f"Processing page {page_num + 1} with AI analysis")
            
            # Step 1: Vision-guided page structure analysis (mit Timeout-Handling)
            try:
                vision_analysis = self.vision_chunker.analyze_page_structure(
                    page, page_text, document_context
                )
                
                if page_num % 100 == 0:
                    print(f"         👁️  Vision AI analysiert Seitenstruktur...")
                    
            except Exception as e:
                self.logger.warning(f"Vision analysis failed for page {page_num + 1}: {e}")
                vision_analysis = {"content_types": ["text"], "recommended_strategy": "semantic"}
            
            # Step 2: Agentic strategy determination  
            try:
                strategy_info = self.agentic_router.determine_optimal_strategy(
                    page_text, vision_analysis, manufacturer
                )
                
                if page_num % 100 == 0:
                    strategy_name = strategy_info.get('primary_strategy', 'semantic')
                    print(f"         🧠 LLM bestimmt optimale Chunking-Strategie...")
                    print(f"         ✅ Strategie gewählt: {strategy_name}")
                    
            except Exception as e:
                self.logger.warning(f"Strategy determination failed for page {page_num + 1}: {e}")
                strategy_info = {"primary_strategy": "semantic", "confidence": 0.5}
            
            # Step 3: Create page chunks using AI insights
            page_chunks = self.create_ai_guided_chunks(
                page_text, page_num + 1, strategy_info, vision_analysis,
                document_info, file_path, file_hash
            )
            
            all_chunks.extend(page_chunks)
            document_context = page_text[-500:]  # Keep context for next page
            
            if page_num % 100 == 0:
                print(f"         📝 {len(page_chunks)} Chunks erstellt")
            
            # SOFORTIGE SPEICHERUNG alle 10 Seiten!
            if (page_num + 1) % 10 == 0 or page_num == total_pages - 1:
                # Speichere neue Chunks sofort
                new_page_chunks = [chunk for chunk in page_chunks if 'id' not in chunk]
                if new_page_chunks:
                    try:
                        print(f"         💾 Speichere {len(new_page_chunks)} Chunks von Seite {page_num + 1}...")
                        
                        # Use new database client with metrics
                        op_id = self.metrics.start_operation('chunk_creation', {
                            'page_number': page_num + 1,
                            'chunk_count': len(new_page_chunks)
                        })
                        
                        success = self.db_client.insert_chunks_batch(new_page_chunks)
                        
                        self.metrics.end_operation(op_id, success, {
                            'chunks_created': len(new_page_chunks) if success else 0
                        })
                        
                        if success:
                            print(f"         ✅ Chunks für Seite {page_num + 1} gespeichert")
                            self.logger.info(f"Saved chunks for page {page_num + 1}")
                        else:
                            print(f"         ❌ Fehler beim Speichern der Chunks für Seite {page_num + 1}")
                        
                        # Update legacy stats
                        self.performance_stats["chunks_created"] += len(new_page_chunks) if success else 0
                        
                    except Exception as e:
                        print(f"         ⚠️  Speicherfehler Seite {page_num + 1}: {e}")
                        self.logger.error(f"Error saving chunks for page {page_num + 1}: {e}")
            
            # Zwischenspeichern alle 500 Seiten
            if page_num > 0 and page_num % 500 == 0:
                print(f"      💾 Zwischenspeicherung: {len(all_chunks)} Chunks nach {page_num + 1} Seiten")
                self._save_intermediate_chunks(all_chunks, file_hash, page_num)
            
            self.logger.info(f"Page {page_num + 1}: Created {len(page_chunks)} chunks using {strategy_info.get('primary_strategy', 'semantic')} strategy")
        
        print(f"      🎉 AI-Enhanced Chunking abgeschlossen: {len(all_chunks)} Chunks total")
        self.logger.info(f"AI-enhanced chunking complete: {len(all_chunks)} total chunks")
        return all_chunks
    
    def _process_large_pdf_optimized(self, pdf_document: fitz.Document, document_info: Dict, 
                                   file_path: str, file_hash: str) -> List[Dict]:
        """Optimierte Verarbeitung für große PDFs (>1000 Seiten)"""
        total_pages = len(pdf_document)
        all_chunks = []
        
        # Batch-Verarbeitung in 100-Seiten Blöcken
        batch_size = 100
        total_batches = (total_pages + batch_size - 1) // batch_size
        
        print(f"      📦 Batch-Verarbeitung: {total_batches} Batches à {batch_size} Seiten")
        
        for batch_num in range(total_batches):
            start_page = batch_num * batch_size
            end_page = min(start_page + batch_size, total_pages)
            
            print(f"      🔄 Batch {batch_num + 1}/{total_batches}: Seiten {start_page + 1}-{end_page}")
            
            batch_chunks = []
            document_context = ""
            
            for page_num in range(start_page, end_page):
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                # Vereinfachte Analyse für große PDFs (weniger Vision AI)
                if page_num % 20 == 0:  # Nur jede 20. Seite Vision AI statt jede 10.
                    try:
                        vision_analysis = self.vision_chunker.analyze_page_structure(
                            page, page_text, document_context
                        )
                    except:
                        vision_analysis = {"content_types": ["text"], "recommended_strategy": "semantic"}
                else:
                    # Fallback ohne Vision AI für Performance
                    vision_analysis = {"content_types": ["text"], "recommended_strategy": "semantic"}
                
                # Simplified strategy für Performance
                strategy_info = {"primary_strategy": "semantic", "confidence": 0.7}
                
                page_chunks = self.create_ai_guided_chunks(
                    page_text, page_num + 1, strategy_info, vision_analysis,
                    document_info, file_path, file_hash
                )
                
                batch_chunks.extend(page_chunks)
                document_context = page_text[-300:]
            
            all_chunks.extend(batch_chunks)
            print(f"         ✅ Batch {batch_num + 1} abgeschlossen: {len(batch_chunks)} Chunks")
            
            # EXTRACT AND STORE IMAGES FROM THIS BATCH
            batch_images = []
            original_filename = os.path.basename(file_path)  # Define original_filename for image processing
            print(f"         🖼️  Extrahiere Bilder von Batch {batch_num + 1} (Seiten {start_page + 1}-{end_page})...")
            for page_num in range(start_page, end_page):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Convert to PNG
                            img_data = pix.tobytes("png")
                            
                            # Generate image hash for duplicate detection (SHA-256)
                            image_hash = hashlib.sha256(img_data).hexdigest()
                            
                            # Generate R2 key
                            img_hash = hashlib.md5(img_data).hexdigest()[:16]
                            r2_key = f"images/{file_hash}/page_{page_num+1}_img_{img_index+1}_{img_hash}.png"
                            
                            # Check if image already exists in R2
                            try:
                                self.r2_client.head_object(
                                    Bucket=self.config.r2_bucket_name,
                                    Key=r2_key
                                )
                                # Image already exists
                                public_domain_id = getattr(self.config, 'r2_public_domain_id', self.config.r2_account_id)
                                r2_url = f"https://pub-{public_domain_id}.r2.dev/{r2_key}"
                                status = 'reused_existing'
                                
                            except ClientError as e:
                                if e.response['Error']['Code'] == '404':
                                    # Image doesn't exist, upload it
                                    self.r2_client.put_object(
                                        Bucket=self.config.r2_bucket_name,
                                        Key=r2_key,
                                        Body=img_data,
                                        ContentType="image/png"
                                    )
                                    public_domain_id = getattr(self.config, 'r2_public_domain_id', self.config.r2_account_id)
                                    r2_url = f"https://pub-{public_domain_id}.r2.dev/{r2_key}"
                                    status = 'newly_uploaded'
                                else:
                                    self.logger.error(f"Error checking R2 object: {e}")
                                    continue
                            
                            batch_images.append({
                                'image_hash': image_hash,  # SHA-256 des Bildinhalts
                                'file_hash': file_hash,
                                'original_filename': original_filename,
                                'page_number': page_num + 1,
                                'storage_url': r2_url,
                                'storage_bucket': self.config.r2_bucket_name,
                                'storage_key': r2_key,
                                'image_type': 'diagram',  # Default, kann später per AI klassifiziert werden
                                'figure_reference': f"Page {page_num + 1} Image {img_index + 1}",
                                'manufacturer': document_info.get('manufacturer', 'Unknown'),
                                'model': document_info.get('model', 'Unknown'),
                                'document_type': document_info.get('document_type', 'Unknown'),
                                'document_source': 'PDF Extraction',
                                'width': pix.width,
                                'height': pix.height,
                                'file_size_bytes': len(img_data),
                                'mime_type': 'image/png',
                                'processing_status': 'processed',
                                'vision_analysis': {},  # Wird später gefüllt
                                'file_path': file_path
                            })
                        
                        pix = None
                        
                    except Exception as e:
                        self.logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            if batch_images:
                print(f"         🖼️  Gefunden: {len(batch_images)} Bilder in Batch {batch_num + 1}")
            
            # SOFORTIGE SPEICHERUNG jedes Batches (Chunks + Images)!
            new_batch_chunks = [chunk for chunk in batch_chunks if 'id' not in chunk]
            if new_batch_chunks:
                try:
                    print(f"         💾 Speichere {len(new_batch_chunks)} Chunks von Batch {batch_num + 1}...")
                    
                    # Use new database client with metrics
                    op_id = self.metrics.start_operation('chunk_creation', {
                        'batch_number': batch_num + 1,
                        'chunk_count': len(new_batch_chunks)
                    })
                    
                    success = self.db_client.insert_chunks_batch(new_batch_chunks)
                    
                    self.metrics.end_operation(op_id, success, {
                        'chunks_created': len(new_batch_chunks) if success else 0
                    })
                    
                    if success:
                        print(f"         ✅ Batch {batch_num + 1} Chunks gespeichert")
                        self.logger.info(f"Saved batch {batch_num + 1} chunks")
                    else:
                        print(f"         ❌ Fehler beim Speichern von Batch {batch_num + 1}")
                    
                    # Update legacy stats
                    self.performance_stats["chunks_created"] += len(new_batch_chunks) if success else 0
                    
                    # Markiere als gespeichert
                    for i, chunk in enumerate(new_batch_chunks):
                        chunk['id'] = result.data[i]['id']
                    
                    print(f"         ✅ Batch {batch_num + 1} Chunks gespeichert")
                    self.logger.info(f"Saved batch {batch_num + 1} chunks")
                    
                except Exception as e:
                    print(f"         ⚠️  Speicherfehler Batch {batch_num + 1}: {e}")
                    self.logger.error(f"Error saving batch {batch_num + 1} chunks: {e}")
            
            # Store batch images immediately
            if batch_images:
                try:
                    print(f"         💾 Speichere {len(batch_images)} Bild-Metadaten von Batch {batch_num + 1}...")
                    
                    # Clean batch_images for database compatibility and check for duplicates
                    clean_batch_images = []
                    new_images = []
                    
                    for img in batch_images:
                        clean_img = img.copy()
                        clean_img.pop('metadata', None)  # Remove metadata field
                        clean_batch_images.append(clean_img)
                    
                    # Check which images are actually new (batch check for efficiency)
                    if clean_batch_images:
                        # Get all hashes in this batch
                        batch_hashes = [img["image_hash"] for img in clean_batch_images]
                        
                        # Use new database client for batch hash check
                        op_id = self.metrics.start_operation('image_hash_check', {
                            'hash_count': len(batch_hashes)
                        })
                        
                        existing_hashes = self.db_client.check_existing_image_hashes(batch_hashes)
                        existing_hashes_set = set(existing_hashes)
                        
                        self.metrics.end_operation(op_id, True, {
                            'existing_hashes': len(existing_hashes),
                            'new_hashes': len(batch_hashes) - len(existing_hashes)
                        })
                        
                        # Filter out existing images
                        for img in clean_batch_images:
                            if img["image_hash"] not in existing_hashes_set:
                                new_images.append(img)
                    
                    if new_images:
                        # Validate that all required fields are present before insert
                        required_fields = ['image_hash', 'file_hash', 'original_filename', 'page_number', 
                                         'storage_url', 'storage_bucket', 'storage_key', 'image_type',
                                         'figure_reference', 'manufacturer', 'document_type', 'width', 'height']
                        
                        valid_images = []
                        for img in new_images:
                            missing_fields = [field for field in required_fields if field not in img or img[field] is None]
                            if missing_fields:
                                print(f"         ❌ Image {img.get('image_hash', 'unknown')[:16]} missing fields: {missing_fields}")
                                continue
                            valid_images.append(img)
                        
                        if valid_images:
                            try:
                                result = self.db_client.insert_images(valid_images)
                                self.metrics_collector.increment_metric('database_operations', 'images_batch_inserts')
                                print(f"         ✅ {len(valid_images)} neue Bilder von Batch {batch_num + 1} gespeichert ({len(clean_batch_images) - len(valid_images)} übersprungen)")
                                self.logger.info(f"Saved {len(valid_images)} new images from batch {batch_num + 1}")
                            except Exception as insert_error:
                                error_str = str(insert_error)
                                if "duplicate key value violates unique constraint" in error_str or "already exists" in error_str:
                                    print(f"         ⚠️  {len(valid_images)} Bilder bereits vorhanden (DB Constraint) - NORMAL")
                                    self.logger.info(f"Images from batch {batch_num + 1} already exist in database - skipping")
                                else:
                                    print(f"         ❌ INSERT ERROR: {insert_error}")
                                    self.logger.error(f"Insert error for batch {batch_num + 1}: {insert_error}")
                                # Continue processing despite image insert failure
                        else:
                            print(f"         ⚠️  No valid images to insert from batch {batch_num + 1}")
                    else:
                        print(f"         ⚠️  Alle {len(clean_batch_images)} Bilder von Batch {batch_num + 1} bereits vorhanden (Duplikate übersprungen)")
                        self.logger.info(f"All {len(clean_batch_images)} images from batch {batch_num + 1} were duplicates")
                    
                except Exception as e:
                    print(f"         ⚠️  Speicherfehler Bilder Batch {batch_num + 1}: {e}")
                    self.logger.error(f"Error saving batch {batch_num + 1} images: {e}")
            
            # Zwischenspeichern jedes Batches (wird übersprungen da bereits gespeichert)
            if batch_num % 5 == 0:  # Alle 5 Batches = 500 Seiten
                self._save_intermediate_chunks(all_chunks, file_hash, end_page)
        
        return all_chunks
    
    def _save_intermediate_chunks(self, chunks: List[Dict], file_hash: str, page_num: int):
        """Zwischenspeicherung für große PDFs - alle 50 Seiten speichern"""
        try:
            # Speichere alle 50 Seiten oder bei mehr als 100 Chunks
            if page_num % 50 == 0 or len(chunks) > 100:
                if chunks:  # Nur wenn Chunks vorhanden sind
                    print(f"         💾 Zwischenspeicherung: {len(chunks)} Chunks bis Seite {page_num}")
                    
                    # Speichere nur die neuen Chunks (die noch keine ID haben)
                    new_chunks = [chunk for chunk in chunks if 'id' not in chunk]
                    
                    if new_chunks:
                        try:
                            # Store in batches of 50 for intermediate saves
                            batch_size = 50
                            for i in range(0, len(new_chunks), batch_size):
                                batch = new_chunks[i:i + batch_size]
                                result = self.db_client.insert_chunks(batch)
                                self.metrics_collector.increment_metric('database_operations', 'chunks_batch_inserts')
                                
                                # Markiere als gespeichert
                                for j, chunk in enumerate(batch):
                                    chunks[chunks.index(chunk)]['id'] = result.data[j]['id']
                            
                            print(f"         ✅ {len(new_chunks)} neue Chunks zwischengespeichert")
                            self.logger.info(f"Intermediate save: {len(new_chunks)} chunks saved at page {page_num}")
                            
                        except Exception as e:
                            print(f"         ⚠️  Zwischenspeicherung fehlgeschlagen: {e}")
                            self.logger.warning(f"Intermediate save failed at page {page_num}: {e}")
        except Exception as e:
            self.logger.warning(f"Intermediate save failed: {e}")
    
    def create_ai_guided_chunks(self, page_text: str, page_num: int, strategy_info: Dict,
                               vision_analysis: Dict, document_info: Dict,
                               file_path: str, file_hash: str) -> List[Dict]:
        """Create chunks using AI guidance and optimal split points"""
        
        # Split into paragraphs for analysis
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            # Single paragraph or very short text
            return [self.create_chunk_data(
                page_text, page_num, 0, document_info,
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
            current_content = '\n\n'.join(current_chunk_paragraphs)
            if len(current_content) > self.config.max_chunk_size:
                should_split = True
            
            # Create chunk if splitting or at end
            if should_split or i == len(paragraphs) - 1:
                if current_chunk_paragraphs:
                    chunk_data = self.create_chunk_data(
                        current_content, page_num, chunk_index, document_info,
                        file_path, file_hash, chunk_type
                    )
                    chunks.append(chunk_data)
                    chunk_index += 1
                    current_chunk_paragraphs = []
        
        return chunks
    
    def create_chunk_data(self, chunk_text: str, page_num: int, chunk_index: int,
                         document_info: Dict, file_path: str, file_hash: str, chunk_type: str) -> Dict:
        """Create comprehensive chunk data optimized for n8n Vector Store with document types"""
        
        # Extract document information
        manufacturer = document_info['manufacturer']
        document_type = document_info['document_type']
        document_subtype = document_info.get('document_subtype')
        document_priority = document_info.get('document_priority', 'normal')
        document_source = document_info.get('document_source', 'Official Manufacturer')
        
        # Generate embedding for n8n Vector Store
        try:
            embedding = self.generate_chunk_embedding(chunk_text)
            if embedding is None or len(embedding) == 0:
                self.logger.warning(f"Empty embedding generated for chunk, skipping...")
                embedding = None
        except Exception as e:
            self.logger.error(f"Error generating chunk embedding: {e}")
            embedding = None
        
        # Extract RICH METADATA for n8n intelligent search
        error_codes = self.extract_error_codes_ai(chunk_text, manufacturer)
        figure_refs = self.extract_figure_references_ai(chunk_text)
        connections = self.extract_connection_points_ai(chunk_text)
        procedures = self.extract_procedures_ai(chunk_text)
        
        # Extract MODEL from manufacturer string (HP E52645 -> E52645)
        model = self.extract_model_from_manufacturer(manufacturer)
        base_manufacturer = self.extract_base_manufacturer(manufacturer)
        
        # Determine problem type for n8n categorization
        problem_type = self.determine_problem_type(chunk_text, chunk_type)
        procedure_type = self.determine_procedure_type(chunk_text, chunk_type, document_type)
        
        # Create n8n-optimized metadata with document context
        n8n_metadata = {
            'confidence': self.calculate_chunk_confidence(chunk_text, error_codes, procedures),
            'related_models': self.find_related_models(model, base_manufacturer),
            'tools_required': self.extract_tools_required(chunk_text),
            'difficulty_level': self.assess_difficulty_level(chunk_text, procedures),
            'estimated_time': self.estimate_repair_time(chunk_text, procedure_type),
            'safety_warnings': self.extract_safety_warnings(chunk_text),
            'parts_mentioned': self.extract_parts_mentioned(chunk_text),
            'document_context': {
                'subtype': document_subtype,
                'source_reliability': 'high' if document_source == 'Official Manufacturer' else 'medium',
                'content_freshness': self.assess_content_freshness(document_type, document_priority)
            }
        }
        
        return {
            # Core content for n8n Vector Store
            'content': chunk_text,
            'token_count': len(chunk_text.split()),  # Supabase inspired token count
            'embedding': embedding,
            
            # PRIMARY SEARCH FIELDS for n8n
            'manufacturer': base_manufacturer,      # "HP", "Canon" 
            'model': model,                        # "E52645", "X580"
            'error_codes': error_codes,            # ["C0001", "C0010"]
            'problem_type': problem_type,          # "scanner_issue", "paper_jam"
            'procedure_type': procedure_type,      # "troubleshooting", "maintenance"
            
            # ENHANCED DOCUMENT CONTEXT for n8n
            'document_type': document_type,        # "Service Manual", "Bulletin", "CPMD", "Video"
            'document_subtype': document_subtype,  # "Service Bulletin", "Parts Replacement Video"
            'document_priority': document_priority, # "urgent", "normal", "reference"
            'document_source': document_source,    # "Official Manufacturer", "Internal", "Third Party"
            'page_number': page_num,
            'chunk_index': chunk_index,
            
            # TECHNICAL METADATA for n8n filtering
            'figure_references': figure_refs,
            'procedures': procedures,
            
            # RICH METADATA for n8n AI Agent
            'metadata': n8n_metadata,
            
            # SYSTEM FIELDS
            'file_path': file_path,
            'original_filename': os.path.basename(file_path),
            'file_hash': file_hash,
            'page_number': page_num,
            'chunk_index': chunk_index,
            'error_codes': error_codes,
            'figure_references': figure_refs,
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
            'HP': [r'\b[0-9A-F]{4}h\b', r'\bC[0-9]{4}\b', r'\b[0-9]{2}\.[0-9]{2}\.[0-9]{2}\b'],
            'Konica Minolta': [r'\bC[0-9]{4}\b'],
            'Kyocera': [r'\bC[0-9]{4}\b', r'\bCF[0-9]{3}\b'],
            'Canon': [r'\bE[0-9]{3}\b'],
            'Brother': [r'\b[A-Z]{2}-[0-9]{2}\b'],
            'Xerox': [r'\b0[0-9]{2}\b'],
            'Lexmark': [r'\b[0-9]{3}\b'],
            'Fujifilm': [r'\bE[0-9]{3}\b'],
            'UTAX': [r'\bERROR [0-9]{2}\b']
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
                ai_codes = [code.strip() for code in response.split('\n') if code.strip()]
                return ai_codes[:10]  # Limit to avoid noise
        
        return []
    
    def extract_figure_references_ai(self, text: str) -> List[str]:
        """AI-enhanced figure reference extraction"""
        patterns = [
            r'\bFigure\s+[0-9]+(?:-[0-9]+)?(?:\.[0-9]+)*\b',
            r'\bFig\.?\s+[0-9]+(?:-[0-9]+)?(?:\.[0-9]+)*\b',
            r'\bAbbildung\s+[0-9]+(?:-[0-9]+)?(?:\.[0-9]+)*\b',
            r'\bAbb\.?\s+[0-9]+(?:-[0-9]+)?(?:\.[0-9]+)*\b'
        ]
        
        figure_refs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            figure_refs.extend(matches)
        
        return list(set(figure_refs))
    
    def extract_connection_points_ai(self, text: str) -> List[str]:
        """AI-enhanced connection point extraction"""
        patterns = [
            r'\b[A-Z]+[0-9]+-[A-Z]+[0-9]+\b',  # P1-J2 format
            r'\bConnector\s+[A-Z]+[0-9]+\b',
            r'\bAnschluss\s+[A-Z]+[0-9]+\b',
            r'\b[A-Z]{1,3}[0-9]+\b'  # CN1, J12 format
        ]
        
        connections = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            connections.extend(matches)
        
        return list(set(connections))
    
    def extract_procedures_ai(self, text: str) -> List[str]:
        """Extract procedure step references"""
        step_patterns = [
            r'\bStep\s+[0-9]+\b',
            r'\bSchritt\s+[0-9]+\b',
            r'\b[0-9]+\.\s+[A-Z]',  # 1. Something
        ]
        
        procedures = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            procedures.extend(matches)
        
        return list(set(procedures))
    
    def extract_model_from_manufacturer(self, manufacturer: str) -> str:
        """Extract pure model number from manufacturer string"""
        # "HP E52645" -> "E52645"
        parts = manufacturer.split()
        if len(parts) > 1:
            return parts[1]  # Model part
        return manufacturer
    
    def extract_base_manufacturer(self, manufacturer: str) -> str:
        """Extract base manufacturer from manufacturer string"""
        # "HP E52645" -> "HP"
        return manufacturer.split()[0]
    
    def determine_problem_type(self, text: str, chunk_type: str) -> str:
        """Determine problem type for n8n categorization"""
        text_lower = text.lower()
        
        problem_types = {
            'scanner_issue': ['scanner', 'scan', 'calibration', 'adf'],
            'paper_jam': ['paper jam', 'jam', 'paper feed', 'paper path'],
            'toner_issue': ['toner', 'cartridge', 'ink', 'supply'],
            'display_issue': ['display', 'screen', 'panel', 'lcd'],
            'network_issue': ['network', 'wifi', 'ethernet', 'connection'],
            'fuser_issue': ['fuser', 'heating', 'temperature'],
            'motor_issue': ['motor', 'drive', 'gear', 'movement'],
            'sensor_issue': ['sensor', 'detection', 'position'],
            'memory_issue': ['memory', 'ram', 'storage', 'disk'],
            'power_issue': ['power', 'voltage', 'electrical', 'psu']
        }
        
        for problem_type, keywords in problem_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return problem_type
        
        # Fallback based on chunk_type
        if 'error' in chunk_type:
            return 'error_general'
        elif 'maintenance' in chunk_type:
            return 'maintenance'
        
        return 'general'
    
    def determine_procedure_type(self, text: str, chunk_type: str, document_type: str = None) -> str:
        """Determine procedure type for n8n workflow categorization with document context"""
        text_lower = text.lower()
        
        # Document-type specific procedure types
        if document_type == 'CPMD':
            return 'official_repair_procedure'
        elif document_type == 'Service Bulletin':
            return 'urgent_procedure'
        elif document_type == 'Video':
            return 'visual_instruction'
        elif document_type == 'Parts Catalog':
            return 'part_specification'
        
        # Standard procedure type detection
        if any(word in text_lower for word in ['troubleshoot', 'error', 'fault', 'problem', 'diagnos']):
            return 'troubleshooting'
        elif any(word in text_lower for word in ['replace', 'install', 'remove', 'exchange']):
            return 'part_replacement'
        elif any(word in text_lower for word in ['clean', 'maintenance', 'service', 'calibrate']):
            return 'maintenance'
        elif any(word in text_lower for word in ['setup', 'configure', 'initial', 'install']):
            return 'installation'
        elif any(word in text_lower for word in ['test', 'verify', 'check', 'validate']):
            return 'testing'
        
        return 'general_procedure'
    
    def assess_content_freshness(self, document_type: str, document_priority: str) -> str:
        """Assess content freshness based on document type and priority"""
        if document_type in ['Service Bulletin', 'Technical Information', 'CPMD']:
            if document_priority == 'urgent':
                return 'very_fresh'  # Recent urgent bulletin
            return 'fresh'  # Recent technical info
        elif document_type == 'Video':
            return 'fresh'  # Video instructions are usually recent
        elif document_type in ['Service Manual', 'Parts Catalog']:
            return 'stable'  # Standard documentation
        
        return 'unknown'
    
    def calculate_chunk_confidence(self, text: str, error_codes: List[str], procedures: List[str]) -> float:
        """Calculate confidence score for chunk relevance"""
        confidence = 0.5  # Base confidence
        
        # More error codes = higher confidence
        if error_codes:
            confidence += min(len(error_codes) * 0.1, 0.3)
        
        # More procedures = higher confidence  
        if procedures:
            confidence += min(len(procedures) * 0.05, 0.2)
        
        # Technical terms increase confidence
        technical_terms = ['step', 'procedure', 'remove', 'install', 'check', 'verify']
        technical_count = sum(1 for term in technical_terms if term.lower() in text.lower())
        confidence += min(technical_count * 0.02, 0.1)
        
        return min(confidence, 1.0)
    
    def find_related_models(self, model: str, manufacturer: str) -> List[str]:
        """Find related models for cross-reference"""
        if not model:
            return []
        
        # HP model families
        if manufacturer == 'HP':
            if model.startswith('E'):
                # E-Series models
                e_series = ['E50045', 'E52645', 'E55040', 'E57540']
                return [m for m in e_series if m != model]
            elif model.startswith('X'):
                # X-Series models  
                return ['X580']
        
        return []
    
    def extract_tools_required(self, text: str) -> List[str]:
        """Extract required tools from text"""
        tools = []
        tool_patterns = [
            r'screwdriver', r'phillips head', r'flathead',
            r'wrench', r'spanner', r'hex key', r'allen key',
            r'pliers', r'tweezers', r'multimeter',
            r'torque wrench', r'socket', r'ratchet'
        ]
        
        text_lower = text.lower()
        for pattern in tool_patterns:
            if re.search(pattern, text_lower):
                tools.append(pattern.replace(r'\b', '').replace(r'\\', ''))
        
        return list(set(tools))
    
    def assess_difficulty_level(self, text: str, procedures: List[str]) -> str:
        """Assess technical difficulty level"""
        text_lower = text.lower()
        
        # High difficulty indicators
        if any(word in text_lower for word in ['disassemble', 'motherboard', 'firmware', 'calibration', 'alignment']):
            return 'expert'
        
        # Medium difficulty indicators  
        if any(word in text_lower for word in ['replace', 'remove', 'install', 'adjust']):
            return 'intermediate'
        
        # Many steps = higher difficulty
        if len(procedures) > 5:
            return 'intermediate'
        elif len(procedures) > 10:
            return 'expert'
        
        return 'basic'
    
    def estimate_repair_time(self, text: str, procedure_type: str) -> str:
        """Estimate repair time based on procedure complexity"""
        text_lower = text.lower()
        
        # Look for explicit time mentions
        time_match = re.search(r'(\d+)\s*(minute|hour|min|hr)', text_lower)
        if time_match:
            return f"{time_match.group(1)} {time_match.group(2)}"
        
        # Estimate based on procedure type
        time_estimates = {
            'troubleshooting': '15-30 minutes',
            'part_replacement': '30-60 minutes', 
            'maintenance': '20-45 minutes',
            'installation': '45-90 minutes',
            'testing': '10-20 minutes'
        }
        
        return time_estimates.get(procedure_type, '30 minutes')
    
    def extract_safety_warnings(self, text: str) -> List[str]:
        """Extract safety warnings and cautions"""
        warnings = []
        warning_patterns = [
            r'caution[:\s]([^.]+)',
            r'warning[:\s]([^.]+)',
            r'danger[:\s]([^.]+)',
            r'notice[:\s]([^.]+)'
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            warnings.extend([match.strip() for match in matches])
        
        return warnings[:3]  # Limit to 3 most important
    
    def extract_parts_mentioned(self, text: str) -> List[str]:
        """Extract mentioned parts/components"""
        parts = []
        part_patterns = [
            r'fuser', r'scanner', r'adf', r'pickup roller',
            r'toner cartridge', r'drum', r'transfer belt',
            r'mainboard', r'control panel', r'display',
            r'motor', r'sensor', r'cable', r'connector'
        ]
        
        text_lower = text.lower()
        for pattern in part_patterns:
            if re.search(pattern, text_lower):
                parts.append(pattern.replace(r'\b', ''))
        
        return list(set(parts))
    
    def parse_file_path(self, file_path: str) -> Tuple[str, str]:
        """Enhanced filename-based manufacturer and model detection"""
        path_str = str(file_path).lower()
        filename = os.path.basename(file_path).lower()
        
        # Enhanced manufacturer detection with model extraction
        manufacturers = {
            'HP': {
                'keywords': ['hp_', 'hewlett', 'packard'],
                'model_patterns': [
                    r'hp[_\-]([a-z]?\d+[a-z]*(?:[_\-]\d+)*)',  # HP_E50045, HP_X580
                    r'([a-z]?\d+[a-z]*(?:[_\-]\d+)*)_sm',       # E50045_SM
                    r'([a-z]\d+)',                              # E778, X580
                ]
            },
            'Konica Minolta': {
                'keywords': ['konica', 'minolta'],
                'model_patterns': [r'(bizhub[\s\-]?\d+)', r'(magicolor[\s\-]?\d+)']
            },
            'Kyocera': {
                'keywords': ['kyocera'],
                'model_patterns': [r'(ecosys[\s\-]?\w+)', r'(taskalfa[\s\-]?\d+)']
            },
            'Canon': {
                'keywords': ['canon'],
                'model_patterns': [r'(imagerunner[\s\-]?\w+)', r'(ir[\s\-]?\d+)']
            },
            'Brother': {
                'keywords': ['brother'],
                'model_patterns': [r'(dcp[\s\-]?\d+)', r'(mfc[\s\-]?\d+)']
            },
            'Xerox': {
                'keywords': ['xerox'],
                'model_patterns': [r'(workcentre[\s\-]?\d+)', r'(phaser[\s\-]?\d+)']
            },
            'Lexmark': {
                'keywords': ['lexmark'],
                'model_patterns': [r'([a-z]\d+)', r'(optra[\s\-]?\w+)']
            },
            'Ricoh': {
                'keywords': ['ricoh'],
                'model_patterns': [r'(aficio[\s\-]?\w+)', r'(mp[\s\-]?\d+)']
            }
        }
        
        detected_manufacturer = "Unknown"
        detected_model = ""
        
        # Enhanced detection using filename context
        for mfg, info in manufacturers.items():
            keywords = info['keywords']
            model_patterns = info['model_patterns']
            
            # Check if manufacturer keyword present
            if any(keyword in filename or keyword in path_str for keyword in keywords):
                detected_manufacturer = mfg
                
                # Extract specific model from filename
                for pattern in model_patterns:
                    match = re.search(pattern, filename)
                    if match:
                        detected_model = match.group(1).upper()
                        break
                
                # If no specific pattern, try generic extraction
                if not detected_model:
                    # Look for model numbers in filename context
                    model_match = re.search(r'([a-z]?\d+[a-z]*(?:[_\-]\d+)*)', filename)
                    if model_match:
                        detected_model = model_match.group(1).upper()
                
                break
        
        # Document type detection with enhanced context
        doc_types = {
            'Service Manual': ['sm', 'service', 'manual', 'repair'],
            'Parts Catalog': ['parts', 'component', 'fru', 'spare'],
            'Technical Information': ['technical', 'info', 'bulletin', 'advisory'],
            'Service Bulletin': ['bulletin', 'alert', 'notice'],
            'CPMD': ['cpmd'],
            'Video': ['video', 'tutorial', 'instruction'],
            'Installation': ['install', 'setup', 'configuration'],
            'Firmware': ['firmware', 'fw', 'software'],
            'Troubleshooting': ['troubleshoot', 'error', 'diagnostic'],
            'Maintenance': ['maintenance', 'clean', 'replace', 'service']
        }
        
        document_type = "Service Manual"  # Default
        document_subtype = None
        document_priority = "normal"
        document_source = "Official Manufacturer"  # Default für alle Hersteller
        
        for doc_type, keywords in doc_types.items():
            if any(keyword in filename for keyword in keywords):
                document_type = doc_type
                break
        
        # Determine document priority based on type and keywords
        if document_type in ['Service Bulletin', 'Technical Information', 'CPMD']:
            # Check for urgent keywords
            if any(urgent in filename for urgent in ['urgent', 'critical', 'immediate']):
                document_priority = "urgent"
            else:
                document_priority = "normal"
        
        # Determine document subtype
        if document_type == 'Technical Information':
            if 'bulletin' in filename:
                document_subtype = 'Service Bulletin'
            elif 'advisory' in filename:
                document_subtype = 'Technical Advisory'
        elif document_type == 'Video':
            if 'replacement' in filename or 'replace' in filename:
                document_subtype = 'Parts Replacement Video'
            elif 'troubleshoot' in filename:
                document_subtype = 'Troubleshooting Video'
        
        # Enhanced logging with document classification
        full_info = f"{detected_manufacturer} {detected_model}" if detected_model else detected_manufacturer
        self.logger.info(f"Detected: {full_info} - {document_type}")
        if document_subtype:
            self.logger.info(f"Subtype: {document_subtype}")
        if document_priority != "normal":
            self.logger.info(f"Priority: {document_priority}")
        
        return {
            'manufacturer': full_info if detected_model else detected_manufacturer,
            'document_type': document_type,
            'document_subtype': document_subtype,
            'document_priority': document_priority,
            'document_source': document_source
        }
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash of file for duplicate detection"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def is_already_processed(self, file_hash: str) -> bool:
        """Check if file was already processed successfully"""
        try:
            result = self.db_client.get_processing_log_status(file_hash)
            self.metrics_collector.increment_metric('database_operations', 'processing_logs_queries')
            if result.data:
                return result.data[0]["status"] == "completed"
            return False
        except Exception as e:
            self.logger.error(f"Error checking processing status: {e}")
            return False

    def check_processing_status(self, file_hash: str) -> Dict:
        """Enhanced check using new processing_logs table"""
        try:
            result = self.supabase.rpc(
                'check_file_processed', 
                {'input_file_hash': file_hash}
            ).execute()
            
            if result.data and len(result.data) > 0:
                status = result.data[0]
                return {
                    'is_processed': status['is_processed'],
                    'status': status['processing_status'],
                    'chunks_count': status['chunks_count'],
                    'images_count': status['images_count']
                }
            else:
                return {
                    'is_processed': False,
                    'status': 'not_found',
                    'chunks_count': 0,
                    'images_count': 0
                }
                
        except Exception as e:
            self.logger.warning(f"Error checking processing status: {e}")
            return {
                'is_processed': False,
                'status': 'error',
                'chunks_count': 0,
                'images_count': 0
            }

    def start_processing_session(self, file_path: str, file_hash: str, document_info: Dict) -> int:
        """Start new processing session and get log ID"""
        try:
            result = self.supabase.rpc(
                'start_processing',
                {
                    'input_file_path': file_path,
                    'input_file_hash': file_hash,
                    'input_filename': os.path.basename(file_path),
                    'input_document_info': document_info
                }
            ).execute()
            
            return result.data if result.data else None
            
        except Exception as e:
            self.logger.error(f"Error starting processing session: {e}")
            return None

    def update_processing_progress(self, log_id: int, stage: str, progress: int, 
                                 chunks_count: int = None, images_count: int = None):
        """Update processing progress"""
        try:
            self.supabase.rpc(
                'update_processing_progress',
                {
                    'log_id': log_id,
                    'new_stage': stage,
                    'progress': progress,
                    'chunks_count': chunks_count,
                    'images_count': images_count
                }
            ).execute()
            
        except Exception as e:
            self.logger.warning(f"Error updating progress: {e}")

    def complete_processing_session(self, log_id: int, processing_time: int):
        """Mark processing as completed"""
        try:
            self.supabase.rpc(
                'complete_processing',
                {
                    'log_id': log_id,
                    'processing_time': processing_time
                }
            ).execute()
            
        except Exception as e:
            self.logger.warning(f"Error completing processing: {e}")
    
    def get_processed_pages(self, file_hash: str) -> set:
        """Get set of already processed page numbers for this file"""
        try:
            result = self.db_client.get_chunks_page_numbers(file_hash)
            self.metrics_collector.increment_metric('database_operations', 'chunks_page_queries')
            if result.data:
                processed_pages = {chunk["page_number"] for chunk in result.data}
                self.logger.info(f"Found {len(processed_pages)} already processed pages for file {file_hash[:8]}...")
                return processed_pages
            return set()
        except Exception as e:
            self.logger.error(f"Error checking processed pages: {e}")
            return set()
    
    def is_page_already_processed(self, file_hash: str, page_num: int) -> bool:
        """Check if specific page was already processed"""
        try:
            result = self.db_client.get_chunk_id_by_page(file_hash, page_num)
            self.metrics_collector.increment_metric('database_operations', 'chunks_page_checks')
            return len(result.data) > 0
        except Exception as e:
            self.logger.error(f"Error checking page {page_num} status: {e}")
            return False
    
    def log_processing_start(self, file_path: str, file_hash: str):
        """Log start of processing"""
        try:
            # Use upsert to handle existing entries
            self.db_client.upsert_processing_log({
                "file_path": file_path,
                "file_hash": file_hash,
                "status": "processing",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            self.metrics_collector.increment_metric('database_operations', 'processing_logs_upserts')
        except Exception as e:
            self.logger.error(f"Error logging processing start: {e}")
    
    def log_processing_complete(self, file_hash: str, chunk_count: int, image_count: int):
        """Log successful completion"""
        try:
            self.db_client.update_processing_log_by_file_hash(file_hash, {
                "status": "completed",
                "chunks_created": chunk_count,
                "images_extracted": image_count,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            self.metrics_collector.increment_metric('database_operations', 'processing_logs_updates')
        except Exception as e:
            self.logger.error(f"Error logging completion: {e}")
    
    def log_processing_error(self, file_hash: str, error_message: str):
        """Log processing error"""
        try:
            self.db_client.update_processing_log_by_file_hash(file_hash, {
                "status": "error",
                "error_message": error_message,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            self.metrics_collector.increment_metric('database_operations', 'processing_logs_updates')
        except Exception as e:
            self.logger.error(f"Error logging error: {e}")
    
    def extract_images_from_pdf(self, pdf_document: fitz.Document, file_hash: str, original_filename: str, document_info: Dict) -> List[Dict]:
        """Extract and upload images from PDF to R2 storage"""
        images = []
        total_pages = len(pdf_document)
        total_images_found = 0
        
        # First pass: count total images
        print("      🔍 Scanne PDF nach Bildern...")
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            image_list = page.get_images()
            total_images_found += len(image_list)
        
        print(f"      📊 Gefunden: {total_images_found} Bilder auf {total_pages} Seiten")
        
        if total_images_found == 0:
            print("      ⏭️  Keine Bilder zum Extrahieren")
            return images
        
        # Second pass: extract images with progress
        extracted_count = 0
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            image_list = page.get_images()
            
            if len(image_list) > 0:
                print(f"      📄 Seite {page_num + 1}/{total_pages}: {len(image_list)} Bilder")
            
            for img_index, img in enumerate(image_list):
                try:
                    extracted_count += 1
                    print(f"         🖼️  [{extracted_count}/{total_images_found}] Extrahiere Bild {img_index + 1}...")
                    
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    # Try to get original format first (for CMYK and better quality)
                    try:
                        img_dict = pdf_document.extract_image(xref)
                        img_data = img_dict["image"]
                        original_ext = img_dict["ext"]
                        mime_type = f"image/{original_ext}"
                        if original_ext == "jpg":
                            mime_type = "image/jpeg"
                        
                        print(f"         📸 Original Format: {original_ext} ({len(img_data)} bytes)")
                        
                        # Generate R2 key with original extension
                        img_hash = hashlib.md5(img_data).hexdigest()[:16]
                        r2_key = f"images/{file_hash}/page_{page_num+1}_img_{img_index+1}_{img_hash}.{original_ext}"
                        
                    except Exception as extract_error:
                        # Fallback to pixmap conversion for non-standard images
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            original_ext = "png"
                            mime_type = "image/png"
                            
                            print(f"         🔄 Fallback PNG: ({len(img_data)} bytes)")
                            
                            # Generate R2 key
                            img_hash = hashlib.md5(img_data).hexdigest()[:16]
                            r2_key = f"images/{file_hash}/page_{page_num+1}_img_{img_index+1}_{img_hash}.png"
                        else:
                            # Skip CMYK images that can't be converted
                            print(f"         ⚠️  Skipping CMYK image (channels: {pix.n})")
                            pix = None
                            continue
                        
                        # ✅ PRÜFE OB BILD BEREITS IN R2 EXISTIERT
                        try:
                            self.r2_client.head_object(
                                Bucket=self.config.r2_bucket_name,
                                Key=r2_key
                            )
                            # Bild existiert bereits!
                            print(f"         ♻️  Bild bereits in R2, überspringe Upload...")
                            
                            # Generate public URL with correct domain
                            public_domain_id = getattr(self.config, 'r2_public_domain_id', self.config.r2_account_id)
                            r2_url = f"https://pub-{public_domain_id}.r2.dev/{r2_key}"
                            
                            # Generate comprehensive image hash for database
                            image_hash = hashlib.sha256(img_data).hexdigest()
                            
                            images.append({
                                # Required NOT NULL fields
                                'image_hash': image_hash,
                                'file_hash': file_hash,
                                'original_filename': original_filename,
                                'page_number': page_num + 1,
                                'storage_url': r2_url,
                                'storage_bucket': self.config.r2_bucket_name,
                                'storage_key': r2_key,
                                'image_type': 'diagram',  # Default classification
                                'manufacturer': document_info['manufacturer'],
                                'model': document_info.get('model', 'Unknown'),
                                'document_type': document_info['document_type'],
                                
                                # Optional technical fields
                                'width': pix.width,
                                'height': pix.height,
                                'file_size_bytes': len(img_data),
                                'document_source': document_info.get('document_source', 'PDF'),
                                'mime_type': mime_type,
                                
                                # Metadata
                                'metadata': {
                                    'extracted_at': datetime.now(timezone.utc).isoformat(),
                                    'size_bytes': len(img_data),
                                    'status': 'reused_existing',
                                    'original_format': original_ext,
                                    'r2_key': r2_key  # Keep for compatibility
                                }
                            })
                            
                            print(f"         ✅ Existierendes Bild wiederverwendet ({len(img_data)/1024:.1f} KB)")
                            
                        except ClientError as e:
                            if e.response['Error']['Code'] == '404':
                                # Bild existiert nicht, upload durchführen
                                print(f"         ☁️  Uploade zu Cloudflare R2...")
                                
                                try:
                                    self.r2_client.put_object(
                                        Bucket=self.config.r2_bucket_name,
                                        Key=r2_key,
                                        Body=img_data,
                                        ContentType=mime_type
                                    )
                                    
                                    # Generate public URL
                                    public_domain_id = getattr(self.config, 'r2_public_domain_id', self.config.r2_account_id)
                                    r2_url = f"https://pub-{public_domain_id}.r2.dev/{r2_key}"
                                    
                                    size_kb = len(img_data) / 1024
                                    print(f"         ✅ Erfolgreich hochgeladen ({size_kb:.1f} KB)")
                                    
                                    # Generate comprehensive image hash for database
                                    image_hash = hashlib.sha256(img_data).hexdigest()
                                    
                                    images.append({
                                        # Required NOT NULL fields
                                        'image_hash': image_hash,
                                        'file_hash': file_hash,
                                        'original_filename': original_filename,
                                        'page_number': page_num + 1,
                                        'storage_url': r2_url,
                                        'storage_bucket': self.config.r2_bucket_name,
                                        'storage_key': r2_key,
                                        'image_type': 'diagram',  # Default classification
                                        'manufacturer': document_info['manufacturer'],
                                        'model': document_info.get('model', 'Unknown'),
                                        'document_type': document_info['document_type'],
                                        
                                        # Optional technical fields
                                        'width': pix.width,
                                        'height': pix.height,
                                        'file_size_bytes': len(img_data),
                                        'document_source': document_info.get('document_source', 'PDF'),
                                        'mime_type': mime_type,
                                        
                                        # Metadata
                                        'metadata': {
                                            'extracted_at': datetime.now(timezone.utc).isoformat(),
                                            'size_bytes': len(img_data),
                                            'status': 'newly_uploaded',
                                            'original_format': original_ext,
                                            'r2_key': r2_key  # Keep for compatibility
                                        }
                                    })
                                    
                                except ClientError as upload_error:
                                    print(f"         ❌ Fehler beim Upload: {upload_error}")
                                    self.logger.error(f"Error uploading image to R2: {upload_error}")
                            else:
                                # Anderer Fehler bei head_object
                                print(f"         ⚠️  R2-Prüfung fehlgeschlagen: {e}")
                                self.logger.error(f"Error checking R2 object: {e}")
                    
                    pix = None
                    
                except Exception as e:
                    print(f"         ❌ Fehler bei Bildverarbeitung: {e}")
                    self.logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                    continue
        
        print(f"      🎉 Bildextraktion abgeschlossen: {len(images)}/{total_images_found} erfolgreich")
        return images
    
    def store_chunks_and_images_enhanced(self, chunks: List[Dict], images: List[Dict], processing_log_id: int):
        """Enhanced storage with normalized many-to-many image relationships"""
        
        # First, store images and get their IDs
        image_ids = []
        image_page_mapping = {}  # Map page numbers to image IDs
        
        if images:
            try:
                print(f"      🖼️  Speichere {len(images)} Bilder...")
                
                # Prepare images for database (remove non-DB fields)
                db_images = []
                for img in images:
                    db_img = img.copy()
                    # Remove fields that don't exist in database
                    db_img.pop('image_data', None)
                    db_img.pop('pix', None)
                    db_img.pop('format', None)  # format column doesn't exist in schema
                    db_img.pop('metadata', None)  # metadata column doesn't exist in schema
                    db_images.append(db_img)
                
                # Store images in batches with duplicate checking
                batch_size = 50
                for i in range(0, len(db_images), batch_size):
                    batch = db_images[i:i + batch_size]
                    
                    # Check for duplicates before inserting (batch check for efficiency)
                    new_batch = []
                    if batch:
                        # Get all hashes in this batch  
                        batch_hashes = [img["image_hash"] for img in batch]
                        
                        # Single query to check all hashes at once
                        existing_hashes_result = self.db_client.check_existing_image_hashes(batch_hashes)
                        existing_hashes = set(existing_hashes_result)
                        self.metrics_collector.increment_metric('database_operations', 'images_hash_checks')
                        
                        # Filter out existing images
                        for img in batch:
                            if img["image_hash"] not in existing_hashes:
                                new_batch.append(img)
                    
                    if new_batch:
                        try:
                            result = self.db_client.insert_images(new_batch)
                            self.metrics_collector.increment_metric('database_operations', 'images_batch_inserts')
                            if result.data:
                                for j, img_data in enumerate(result.data):
                                    img_id = img_data['id']
                                    img_page = img_data['page_number']
                                    image_ids.append(img_id)
                                    
                                    # Build page mapping for chunk linking
                                    if img_page not in image_page_mapping:
                                        image_page_mapping[img_page] = []
                        except Exception as insert_error:
                            error_str = str(insert_error)
                            if "duplicate key value violates unique constraint" in error_str or "already exists" in error_str:
                                print(f"         ⚠️  Batch images already exist (DB Constraint) - continuing")
                                self.logger.info(f"Batch images already exist in database - skipping")
                            else:
                                print(f"         ❌ Image batch insert error: {insert_error}")
                                self.logger.error(f"Image batch insert error: {insert_error}")
                            # Continue processing despite image insert failure
                                image_page_mapping[img_page].append(img_id)
                    
                    # Also get IDs for existing duplicates
                    for img in batch:
                        if img not in new_batch:
                            existing_img = self.db_client.get_image_by_hash(img["image_hash"])
                            self.metrics_collector.increment_metric('database_operations', 'images_individual_queries')
                            if existing_img.data:
                                img_id = existing_img.data[0]['id']
                                img_page = existing_img.data[0]['page_number']
                                image_ids.append(img_id)
                                
                                if img_page not in image_page_mapping:
                                    image_page_mapping[img_page] = []
                                image_page_mapping[img_page].append(img_id)
                
                print(f"      ✅ {len(image_ids)} Bilder gespeichert")
                
            except Exception as e:
                print(f"      ❌ Fehler beim Speichern der Bilder: {e}")
                self.logger.error(f"Error storing images: {e}")
        
        # Then store chunks with processing_log_id
        chunk_ids = []
        if chunks:
            try:
                print(f"      📝 Speichere {len(chunks)} Chunks...")
                
                # Add processing_log_id to chunks (remove related_images field)
                db_chunks = []
                for chunk in chunks:
                    db_chunk = chunk.copy()
                    
                    # Remove id if present (auto-generated by database)
                    if 'id' in db_chunk:
                        del db_chunk['id']
                    
                    db_chunk['processing_log_id'] = processing_log_id
                    # Remove the old related_images field - we'll use junction table
                    db_chunk.pop('related_images', None)
                    db_chunks.append(db_chunk)
                
                # Store chunks in batches
                batch_size = 100
                total_batches = (len(db_chunks) + batch_size - 1) // batch_size
                
                for i in range(0, len(db_chunks), batch_size):
                    batch_num = (i // batch_size) + 1
                    batch = db_chunks[i:i + batch_size]
                    
                    print(f"         📦 Batch {batch_num}/{total_batches}: {len(batch)} Chunks")
                    result = self.db_client.insert_chunks(batch)
                    self.metrics_collector.increment_metric('database_operations', 'chunks_batch_inserts')
                    
                    if result.data:
                        for chunk_data in result.data:
                            chunk_ids.append(chunk_data['id'])
                
                print(f"      ✅ Alle {len(chunks)} Chunks erfolgreich gespeichert")
                
            except Exception as e:
                print(f"      ❌ Fehler beim Speichern der Chunks: {e}")
                self.logger.error(f"Error storing chunks: {e}")
                raise
        
        # Finally, create chunk-image relationships in junction table
        if chunk_ids and image_page_mapping:
            try:
                print(f"      🔗 Erstelle Chunk-Image Verknüpfungen...")
                
                chunk_image_relations = []
                for i, chunk in enumerate(chunks):
                    if i < len(chunk_ids):
                        chunk_id = chunk_ids[i]
                        chunk_page = chunk.get('page_number', 0)
                        
                        # Find images on the same page
                        if chunk_page in image_page_mapping:
                            for image_id in image_page_mapping[chunk_page]:
                                chunk_image_relations.append({
                                    'chunk_id': chunk_id,
                                    'image_id': image_id
                                })
                
                # Store relations in batches
                if chunk_image_relations:
                    batch_size = 200
                    for i in range(0, len(chunk_image_relations), batch_size):
                        batch = chunk_image_relations[i:i + batch_size]
                        self.db_client.insert_chunk_images(batch)
                        self.metrics_collector.increment_metric('database_operations', 'chunk_images_inserts')
                    
                    print(f"      ✅ {len(chunk_image_relations)} Chunk-Image Verknüpfungen erstellt")
                
            except Exception as e:
                print(f"      ⚠️  Warnung bei Chunk-Image Verknüpfungen: {e}")
                self.logger.warning(f"Warning creating chunk-image relations: {e}")
        
        self.logger.info(f"Stored {len(chunks)} chunks, {len(image_ids)} images, and {len(chunk_image_relations) if 'chunk_image_relations' in locals() else 0} relations")
        """Store chunks and images in Supabase database"""
        
        print("   💾 Speichere Daten in Supabase...")
        
        # Store chunks
        if chunks:
            try:
                # Filter out chunks that are already saved (have an 'id')
                new_chunks = [chunk for chunk in chunks if 'id' not in chunk]
                
                if new_chunks:
                    print(f"      📝 Speichere {len(new_chunks)} neue Chunks ({len(chunks) - len(new_chunks)} bereits gespeichert)...")
                    
                    # Store in batches of 100 for better performance
                    batch_size = 100
                    total_batches = (len(new_chunks) + batch_size - 1) // batch_size
                    
                    for i in range(0, len(new_chunks), batch_size):
                        batch_num = (i // batch_size) + 1
                        batch = new_chunks[i:i + batch_size]
                        
                        print(f"         📦 Batch {batch_num}/{total_batches}: {len(batch)} Chunks")
                        result = self.db_client.insert_chunks(batch)
                        self.metrics_collector.increment_metric('database_operations', 'chunks_batch_inserts')
                    
                    print(f"      ✅ Alle {len(new_chunks)} neuen Chunks erfolgreich gespeichert")
                    self.logger.info(f"Stored {len(new_chunks)} new chunks in database")
                else:
                    print(f"      ✅ Alle {len(chunks)} Chunks bereits gespeichert")
                    
            except Exception as e:
                print(f"      ❌ Fehler beim Speichern der Chunks: {e}")
                self.logger.error(f"Error storing chunks: {e}")
        
        # Store images
        if images:
            try:
                print(f"      🖼️  Speichere {len(images)} Bild-Metadaten...")
                
                # Clean images for database compatibility and check duplicates
                clean_images = []
                new_images = []
                
                for img in images:
                    clean_img = img.copy()
                    clean_img.pop('metadata', None)  # Remove metadata field
                    clean_images.append(clean_img)
                
                # Check which images are actually new (batch check for efficiency)
                new_images = []
                if clean_images:
                    # Get all hashes in this batch
                    image_hashes = [img["image_hash"] for img in clean_images]
                    
                    # Single query to check all hashes at once  
                    existing_hashes_result = self.db_client.check_existing_image_hashes(image_hashes)
                    existing_hashes = set(existing_hashes_result)
                    self.metrics_collector.increment_metric('database_operations', 'images_hash_checks')
                    
                    # Filter out existing images
                    for img in clean_images:
                        if img["image_hash"] not in existing_hashes:
                            new_images.append(img)
                
                if new_images:
                    try:
                        result = self.db_client.insert_images(new_images)
                        self.metrics_collector.increment_metric('database_operations', 'images_batch_inserts')
                        print(f"      ✅ {len(new_images)} neue Bild-Metadaten gespeichert ({len(clean_images) - len(new_images)} Duplikate übersprungen)")
                        self.logger.info(f"Stored {len(new_images)} new images in database")
                    except Exception as insert_error:
                        error_str = str(insert_error)
                        if "duplicate key value violates unique constraint" in error_str or "already exists" in error_str:
                            print(f"      ⚠️  Images already exist (DB Constraint) - continuing")
                            self.logger.info(f"Images already exist in database - skipping")
                        else:
                            print(f"      ❌ Image insert error: {insert_error}")
                            self.logger.error(f"Image insert error: {insert_error}")
                else:
                    print(f"      ⚠️  Alle {len(clean_images)} Bilder bereits vorhanden (Duplikate übersprungen)")
                    self.logger.info(f"All {len(clean_images)} images were duplicates")
                    
            except Exception as e:
                print(f"      ❌ Fehler beim Speichern der Bild-Metadaten: {e}")
                self.logger.error(f"Error storing images: {e}")
        
        print("   🎉 Datenbank-Speicherung abgeschlossen!")
    
    def process_pdf(self, file_path: str) -> bool:
        """Main PDF processing with AI enhancement and enhanced logging"""
        start_time = time.time()
        processing_log_id = None
        
        try:
            print(f"🤖 Starting AI-enhanced processing...")
            self.logger.info(f"Starting AI-enhanced processing: {file_path}")
            
            # Parse manufacturer and document type from folder structure
            document_info = self.parse_file_path(file_path)
            manufacturer = document_info['manufacturer']
            document_type = document_info['document_type']
            print(f"   🏭 Hersteller: {manufacturer}")
            print(f"   📋 Typ: {document_type}")
            if document_info.get('document_subtype'):
                print(f"   📂 Subtyp: {document_info['document_subtype']}")
            if document_info.get('document_priority', 'normal') != 'normal':
                print(f"   ⚡ Priorität: {document_info['document_priority']}")
            self.logger.info(f"Detected: {manufacturer} - {document_type}")
            
            # Get file info
            print("   🔍 Berechne Datei-Hash...")
            file_hash = self.get_file_hash(file_path)
            original_filename = os.path.basename(file_path)
            
            # Enhanced duplicate check
            print("   📊 Prüfe Verarbeitungsstatus...")
            status_info = self.check_processing_status(file_hash)
            if status_info['is_processed']:
                print(f"   ⏭️  Datei bereits verarbeitet ({status_info['chunks_count']} Chunks, {status_info['images_count']} Bilder)")
                self.logger.info(f"File already processed, skipping: {file_path}")
                return True
            
            # Start processing session
            print("   📝 Erstelle Verarbeitungsprotokoll...")
            processing_log_id = self.start_processing_session(file_path, file_hash, document_info)
            if not processing_log_id:
                raise Exception("Failed to start processing session")
            
            # Open PDF
            print("   📄 Öffne PDF-Datei...")
            pdf_document = fitz.open(file_path)
            total_pages = len(pdf_document)
            print(f"   📃 Seiten gesamt: {total_pages}")
            
            # Update progress
            self.update_processing_progress(processing_log_id, "parsing", 10)
            
            # Check if this is a large PDF - if so, skip upfront image extraction
            # Large PDFs will handle images in batches during processing
            images = []
            if total_pages <= 1000:
                # Extract images using new ImageProcessor (includes vector graphics)
                print("   🖼️  Extrahiere Bilder...")
                print("      🔍 Scanne PDF nach Bildern...")
                self.update_processing_progress(processing_log_id, "image_extraction", 20)
                
                # Use the new ImageProcessor with vector support
                images = []
                for page_num in range(total_pages):
                    page = pdf_document[page_num]
                    page_images = self.image_processor.extract_images_from_page(page, file_hash, page_num + 1)
                    images.extend(page_images)
                
                print(f"      📊 Gefunden: {len(images)} Bilder auf {total_pages} Seiten")
                if len(images) > 0:
                    print(f"   ✅ {len(images)} Bilder extrahiert")
                else:
                    print("   ⏭️  Keine Bilder zum Extrahieren")
                self.logger.info(f"Extracted {len(images)} images (including vector graphics)")
            else:
                print("   🖼️  Große PDF erkannt - Bilder werden während Batch-Verarbeitung extrahiert")
                self.logger.info(f"Large PDF detected ({total_pages} pages) - images will be extracted during batch processing")
            
            # AI-Enhanced chunking process
            print("   🧠 Starte AI-Enhanced Chunking...")
            print("      • Vision AI analysiert Seitenstrukturen")
            print("      • LLM erkennt Verfahrensschritte")
            print("      • Semantic Boundary Detection aktiv")
            
            self.update_processing_progress(processing_log_id, "chunking", 40)
            chunks = self.ai_enhanced_chunking_process(
                pdf_document, document_info, file_path, file_hash
            )
            print(f"   ✅ {len(chunks)} intelligente Chunks erstellt")
            self.logger.info(f"AI-enhanced chunking created {len(chunks)} chunks")
            
            # Store in database
            print("   💾 Speichere in Supabase Datenbank...")
            self.update_processing_progress(processing_log_id, "storing", 80)
            self.store_chunks_and_images_enhanced(chunks, images, processing_log_id)
            print("   ✅ Daten erfolgreich gespeichert")
            
            # Complete processing session
            processing_time = int(time.time() - start_time)
            self.complete_processing_session(processing_log_id, processing_time)
            
            pdf_document.close()
            print(f"   🎉 AI-Enhanced Verarbeitung erfolgreich abgeschlossen! ({processing_time}s)")
            self.logger.info(f"Successfully processed with AI enhancement: {file_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ FEHLER: {str(e)}")
            self.logger.error(f"Error in AI-enhanced processing {file_path}: {e}")
            
            if processing_log_id:
                try:
                    # Mark as failed in processing log
                    self.db_client.update_processing_log_by_id(processing_log_id, {
                        "status": "failed",
                        "error_message": str(e),
                        "updated_at": "now()"
                    })
                    self.metrics_collector.increment_metric('database_operations', 'processing_logs_updates')
                except:
                    pass
                    
            return False

class PDFWatcher(FileSystemEventHandler):
    """File system watcher for automatic PDF processing"""
    
    def __init__(self, processor: AIEnhancedPDFProcessor):
        self.processor = processor
        self.logger = logging.getLogger(__name__)
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.logger.info(f"New PDF detected: {event.src_path}")
            time.sleep(2)  # Wait for file to be fully written
            self.processor.process_pdf(event.src_path)
    
    def on_moved(self, event):
        if not event.is_directory and event.dest_path.lower().endswith('.pdf'):
            self.logger.info(f"PDF moved to: {event.dest_path}")
            time.sleep(2)
            self.processor.process_pdf(event.dest_path)

def load_config() -> AIProcessingConfig:
    """Load AI-enhanced configuration from config.json"""
    config_file = Path("config.json")
    
    if not config_file.exists():
        print("❌ config.json nicht gefunden!")
        print("   Führen Sie zuerst 'python launch.py' aus.")
        sys.exit(1)
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        # Validate that config is not empty
        if not config_data or len(config_data) == 0:
            print("❌ config.json ist leer!")
            print("   Führen Sie 'python launch.py' aus, um die Konfiguration neu zu erstellen.")
            sys.exit(1)
            
        # Check for required fields
        required_fields = ['supabase_url', 'supabase_key', 'r2_account_id', 'documents_path']
        missing_fields = [field for field in required_fields if field not in config_data]
        if missing_fields:
            print(f"❌ config.json unvollständig! Fehlende Felder: {missing_fields}")
            print("   Führen Sie 'python launch.py' aus, um die Konfiguration zu vervollständigen.")
            sys.exit(1)
            
        config = AIProcessingConfig(
            supabase_url=config_data['supabase_url'],
            supabase_key=config_data['supabase_key'],
            r2_account_id=config_data['r2_account_id'],
            r2_access_key_id=config_data['r2_access_key_id'],
            r2_secret_access_key=config_data['r2_secret_access_key'],
            r2_bucket_name=config_data['r2_bucket_name'],
            documents_path=config_data['documents_path'],
            vision_model=config_data.get('vision_model', 'llava:7b'),
            text_model=config_data.get('text_model', 'llama3.1:8b'),
            embedding_model=config_data.get('embedding_model', 'embeddinggemma'),
            embedding_provider=config_data.get('embedding_provider', 'ollama'),
            use_vision_analysis=config_data.get('use_vision_analysis', True),
            use_semantic_boundaries=config_data.get('use_semantic_boundaries', True),
            chunking_strategy=config_data.get('chunking_strategy', 'intelligent'),
            max_chunk_size=config_data.get('max_chunk_size', 600),
            min_chunk_size=config_data.get('min_chunk_size', 200)
        )
        
        # Add image processing configuration as attribute
        config.image_processing = config_data.get('image_processing', {
            'render_dpi': 144,
            'export_vectors': True,
            'export_raster': True,
            'vector_format': 'svg',
            'raster_format': 'png'
        })
        
        return config
    except Exception as e:
        print(f"❌ Fehler beim Laden der AI-Konfiguration: {e}")
        sys.exit(1)

def process_existing_files(processor: AIEnhancedPDFProcessor, documents_path: str):
    """Process existing PDF files in the documents directory"""
    print("🔍 Suche nach existierenden PDF-Dateien...")
    
    pdf_files = []
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if pdf_files:
        print(f"📄 Gefunden: {len(pdf_files)} PDF-Dateien")
        print("=" * 50)
        
        for i, pdf_file in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_file)
            print(f"\n🔄 [{i}/{len(pdf_files)}] Verarbeite: {filename}")
            print(f"   📁 Pfad: {pdf_file}")
            
            # Check file size
            file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
            print(f"   📊 Größe: {file_size:.1f} MB")
            
            start_time = time.time()
            success = processor.process_pdf(pdf_file)
            end_time = time.time()
            
            if success:
                print(f"   ✅ Erfolgreich verarbeitet in {end_time - start_time:.1f}s")
            else:
                print(f"   ❌ Fehler bei der Verarbeitung")
            
            print("-" * 50)
        
        print(f"\n🎉 Verarbeitung abgeschlossen! {len(pdf_files)} Dateien verarbeitet")
    else:
        print("📄 Keine PDF-Dateien im Documents/ Ordner gefunden")
        print(f"   Überprüfen Sie den Pfad: {documents_path}")

def main():
    import sys
    
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
    
    # Check if specific file provided
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        if os.path.exists(pdf_file):
            print(f"🎯 Verarbeite spezifische Datei: {pdf_file}")
            success = processor.process_pdf(pdf_file)
            if success:
                print("✅ Verarbeitung erfolgreich!")
            else:
                print("❌ Verarbeitung fehlgeschlagen!")
            return
        else:
            print(f"❌ Datei nicht gefunden: {pdf_file}")
            return
    
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
        print("\n🛑 AI-Enhanced System wird beendet...")
    
    observer.join()
    print("✅ AI-Enhanced System erfolgreich beendet")

if __name__ == "__main__":
    main()
