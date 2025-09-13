import uuid
import uuid
import os
import fitz  # PyMuPDF
import json
import logging
import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import requests
import base64
from io import BytesIO
from PIL import Image
import tempfile
import olla                          # Bild-Daten für die Datenbank vorbereiten
            db_data = {
                "source_table": source_table,
                "source_id": source_id,
                "file_hash": image_data.get("file_hash", ""),
                "page_number": image_data.get("page_number", 0),
                "image_index": image_data.get("image_index", 0),
                "storage_url": image_data.get("url", ""),
                "image_type": "photo",  # Standard-Typ
                "manufacturer": manufacturer,
                "model": model,
                "hash": image_data.get("hash", ""),
                "metadata": {-Daten für die Datenbank vorbereiten
            db_data = {
                "source_table": source_table,
                "source_id": source_id,
                "file_hash": image_data.get("file_hash", ""),
                "page_number": image_data.get("page_number", 0),
                "image_index": image_data.get("image_index", 0),
                "storage_url": image_data.get("url", ""),
                "image_type": "photo",  # Standard-Typ
                "manufacturer": manufacturer,
                "model": model,
                "hash": image_data.get("hash", ""),
                "metadata": { logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingClient:
    """Abstract base class for embedding clients"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a text string"""
        pass
    
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        pass

class OllamaEmbeddingClient(EmbeddingClient):
    """Ollama-based embedding client"""
    
    def __init__(self, model_name: str = "embeddinggemma"):
        self.model_name = model_name
        self.client = ollama
        # Test connection
        try:
            self.client.embeddings(model=self.model_name, prompt="test")
            logger.info(f"Successfully connected to Ollama with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a text string using Ollama"""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents using Ollama"""
        embeddings = []
        for doc in documents:
            embeddings.append(self.embed_text(doc))
        return embeddings

class DummyEmbeddingClient(EmbeddingClient):
    """Dummy embedding client for testing"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        logger.info(f"Initialized Dummy Embedding Client with dimension {dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate dummy embeddings for a text string"""
        import hashlib
        import numpy as np
        
        # Create a deterministic but dummy embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        np.random.seed(hash_int)
        
        # Generate a dummy embedding with the specified dimension
        embedding = np.random.normal(0, 1, self.dimension).tolist()
        return embedding
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate dummy embeddings for multiple documents"""
        return [self.embed_text(doc) for doc in documents]

class DocumentProcessor(ABC):
    """Base class for document processors"""
    
    def __init__(self, supabase_client, r2_client, config: Dict[str, Any]):
        self.supabase = supabase_client
        self.r2 = r2_client
        self.config = config
        
        # Initialize embedding client based on configuration
        embedding_type = config.get("embedding_client", {}).get("type", "ollama")
        if embedding_type == "ollama":
            model_name = config.get("embedding_client", {}).get("model", "embeddinggemma")
            try:
                self.embedding_client = OllamaEmbeddingClient(model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama embedding client: {e}. Falling back to dummy client.")
                self.embedding_client = DummyEmbeddingClient()
        else:
            logger.info("Using dummy embedding client")
            self.embedding_client = DummyEmbeddingClient()
    
    @abstractmethod
    def process_document(self, file_path: str, file_hash: str = None, log_id: str = None) -> bool:
        """Process a document and store its data in the database"""
        pass
    
    def extract_text(self, pdf_document) -> List[Dict[str, Any]]:
        """Extract text from a PDF document"""
        pages = []
        for i, page in enumerate(pdf_document):
            text = page.get_text()
            pages.append({
                "page_number": i + 1,
                "text": text
            })
        return pages
    
    def extract_images(self, pdf_document, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from a PDF document and upload to R2"""
        images_data = []
        doc_name = os.path.basename(file_path)
        
        for i, page in enumerate(pdf_document):
            image_list = page.get_images()
            
            # No images on this page
            if not image_list:
                continue
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    
                    if not base_image:
                        continue
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create a unique filename for the image
                    image_filename = f"{doc_name}_page{i+1}_img{img_index+1}.{image_ext}"
                    
                    # Berechne den Hash der Bilddaten für die Deduplizierung
                    import hashlib
                    image_hash = hashlib.sha256(image_bytes).hexdigest()
                    
                    # Prüfe, ob das Bild bereits hochgeladen wurde
                    if hasattr(self.r2, 'hash_exists') and self.r2.hash_exists(image_hash):
                        # Bild bereits vorhanden, hole die URL
                        r2_url = self.r2.get_url_for_hash(image_hash)
                        logger.info(f"Bild bereits hochgeladen, verwende vorhandenes Bild: {r2_url}")
                    else:
                        # Bild noch nicht hochgeladen, speichere und lade hoch
                        # Create a temporary file to save the image
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_ext}") as tmp_file:
                            tmp_file.write(image_bytes)
                            tmp_file_path = tmp_file.name
                        
                        # Upload to R2
                        try:
                            # BytesIO für die Daten erstellen
                            image_fileobj = BytesIO(image_bytes)
                            
                            # Hochladen mit Hash-Information
                            if hasattr(self.r2, 'upload_fileobj') and 'image_hash' in self.r2.upload_fileobj.__code__.co_varnames:
                                # Wenn der R2-Client den Hash-Parameter unterstützt
                                r2_url = self.r2.upload_fileobj(image_fileobj, image_filename, image_hash=image_hash)
                            else:
                                # Fallback für ältere R2-Client-Versionen
                                r2_url = self.upload_to_r2(tmp_file_path, image_filename)
                            
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                    
                    # Save image metadata
                    images_data.append({
                        "page_number": i + 1,
                        "image_index": img_index + 1,
                        "filename": image_filename,
                        "url": r2_url,
                        "width": base_image.get("width", 0),
                        "height": base_image.get("height", 0),
                        "content_type": f"image/{image_ext}",
                        "hash": image_hash,  # Speichere den Hash in den Metadaten
                        "storage_url": r2_url  # Storage URL für die Datenbank
                    })
                
                except Exception as e:
                    logger.error(f"Error extracting image {img_index} from page {i+1}: {str(e)}")
        
        
        # Speichere die Bilder in der Datenbank (wenn Supabase verfügbar)
        if hasattr(self, 'supabase') and self.supabase:
            source_table = "service_manuals"  # Standard-Wert, sollte überschrieben werden
            source_id = str(uuid.uuid4())  # Generiere eine UUID für die Quelle
            manufacturer = "unknown"
            model = "unknown"
            
            # Für jedes extrahierte Bild
            for img_data in images_data:
                self.store_image_in_db(img_data, source_table, source_id, manufacturer, model)
                
        return images_data
    
    def upload_to_r2(self, file_path: str, object_name: str) -> str:
        """Upload a file to R2 and return the URL"""
        try:
            with open(file_path, 'rb') as file_data:
                # Use R2 client to upload file
                response = self.r2.upload_file(file_path, object_name)
                
                # Return the public URL
                base_url = self.config.get("r2", {}).get("public_url", "")
                if base_url:
                    return f"{base_url.rstrip('/')}/{object_name}"
                return object_name
        except Exception as e:
            logger.error(f"Error uploading to R2: {str(e)}")
            return ""
    
    def extract_metadata(self, pdf_document) -> Dict[str, Any]:
        """Extract metadata from a PDF document"""
        metadata = pdf_document.metadata
        if metadata:
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": len(pdf_document)
            }
        else:
            return {
                "page_count": len(pdf_document)
            }
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                # Try to find a natural breaking point (period, newline)
                if end < len(text):
                    # Look for the last period or newline in the chunk
                    last_period = text.rfind('.', start, end)
                    last_newline = text.rfind('\n', start, end)
                    break_point = max(last_period, last_newline)
                    
                    # If found a natural break point, use it
                    if break_point > start:
                        end = break_point + 1  # Include the period or newline
                
                # Add the chunk
                chunks.append(text[start:end])
                
                # Move to next chunk with overlap
                start = end - overlap
                if start < 0:
                    start = 0
        
        return chunks
        
    def store_image_in_db(self, image_data, source_table, source_id, manufacturer, model):
        """Speichert ein Bild in der Datenbank"""
        try:
            # Wenn source_id kein UUID ist, erstelle ein neues
            try:
                uuid.UUID(source_id)
            except (ValueError, TypeError, AttributeError):
                source_id = str(uuid.uuid4())
                
            # Bild-Daten für die Datenbank vorbereiten
            db_data = {
                "source_table": source_table,
                "source_id": source_id,
                "file_hash": image_data.get("file_hash", ""),
                "page_number": image_data.get("page_number", 0),
                "image_index": image_data.get("image_index", 0),
                "storage_url": image_data.get("url", ""),
                "image_type": "photo",  # Standard-Typ
                "manufacturer": manufacturer,
                "model": model,
                "metadata": {
                    "width": image_data.get("width", 0),
                    "height": image_data.get("height", 0),
                    "mime_type": image_data.get("content_type", ""),
                    "original_format": image_data.get("filename", "").split('.')[-1] if "." in image_data.get("filename", "") else ""
                }
            }
            
            # In Supabase speichern
            result = self.supabase.table("images").insert(db_data).execute()
            
            if result.data:
                logger.info(f"Bild in DB gespeichert mit ID: {result.data[0]['id']}")
                return result.data[0]["id"]
            return None
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Bildes in der Datenbank: {e}")
            return None
    
    
    def _store_image_in_db_duplicate(self, image_data, source_table, source_id, manufacturer, model):
        """Speichert ein Bild in der Datenbank - Duplikat"""
        try:
            # Wenn source_id kein UUID ist, erstelle ein neues
            try:
                uuid.UUID(source_id)
            except (ValueError, TypeError, AttributeError):
                source_id = str(uuid.uuid4())
                
            # Bild-Daten für die Datenbank vorbereiten
            db_data = {
                "source_table": source_table,
                "source_id": source_id,
                "file_hash": image_data.get("file_hash", ""),
                "page_number": image_data.get("page_number", 0),
                "image_index": image_data.get("image_index", 0),
                "storage_url": image_data.get("url", ""),
                "image_type": "photo",  # Standard-Typ
                "manufacturer": manufacturer,
                "model": model,
                "metadata": {
                    "width": image_data.get("width", 0),
                    "height": image_data.get("height", 0),
                    "mime_type": image_data.get("content_type", ""),
                    "original_format": image_data.get("filename", "").split('.')[-1] if "." in image_data.get("filename", "") else ""
                }
            }
            
            # In Supabase speichern
            result = self.supabase.table("images").insert(db_data).execute()
            
            if result.data:
                return result.data[0]["id"]
            return None
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Bildes in der Datenbank: {e}")
            return None
    
    def generate_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        try:
            # Generate embeddings using the embedding client
            embeddings = self.embedding_client.embed_documents(text_chunks)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [[] for _ in text_chunks]  # Return empty embeddings on error
    
    def store_document_data(self, document_data: Dict[str, Any]) -> bool:
        """Store document data in the database"""
        try:
            # Bestimme die Zieltabelle basierend auf dem Dokumenttyp
            document_type = document_data.get("document_type", "unknown")
            target_table = document_type + "s" if document_type != "parts_manual" else "parts_catalogs"
            
            # Erstelle eine UUID für das Dokument
            doc_id = str(uuid.uuid4())
            
            # Erstelle einen Log-Eintrag in processing_logs
            try:
                log_data = {
                    "file_path": document_data.get("file_path", ""),
                    "file_hash": document_data.get("file_hash", ""),
                    "original_filename": os.path.basename(document_data.get("file_path", "")),
                    "status": "completed",
                    "document_type": document_type,
                    "manufacturer": document_data.get("metadata", {}).get("manufacturer", "unknown"),
                    "model": document_data.get("metadata", {}).get("model", "unknown"),
                    "document_title": document_data.get("title", ""),
                    "document_version": document_data.get("metadata", {}).get("document_version", ""),
                    "chunks_created": len(document_data.get("chunks", [])),
                    "images_extracted": len(document_data.get("images", [])),
                    "started_at": datetime.datetime.now().isoformat(),
                    "completed_at": datetime.datetime.now().isoformat()
                }
                
                self.supabase.table("processing_logs").insert(log_data).execute()
                logger.info(f"Processing log für {document_data.get('title', '')} erstellt")
            except Exception as log_err:
                logger.warning(f"Fehler beim Erstellen des Processing Logs: {log_err}")
            
            # Insert images
            for image in document_data.get("images", []):
                self.supabase.table("images").insert(
                    {
                        "source_table": document_data.get("document_type", "unknown"),
                        "source_id": doc_id,
                        "page_number": image.get("page_number", 0),
                        "image_index": image.get("image_index", 0),
                        "storage_url": image.get("url", ""),
                        "hash": image.get("hash", ""),  # Speichere den Bild-Hash in der DB
                        "manufacturer": document_data.get("metadata", {}).get("manufacturer", ""),
                        "model": document_data.get("metadata", {}).get("model", ""),
                        "metadata": {
                            "filename": image.get("filename", ""),
                            "width": image.get("width", 0),
                            "height": image.get("height", 0),
                            "content_type": image.get("content_type", "")
                        }
                    }
                ).execute()
            
            # Insert chunks with embeddings
            chunks = document_data.get("chunks", [])
            embeddings = document_data.get("embeddings", [])
            
            # Bestimme die Zieltabelle basierend auf dem Dokumenttyp
            document_type = document_data.get("document_type", "unknown")
            
            # Tabellen-Mapping gemäß Supabase AI Dokumentation
            table_mapping = {
                "service_manual": "service_manuals",
                "bulletin": "bulletins",
                "cpmd": "cpmd_documents",
                "parts_manual": "parts_catalog"  # Gemäß Supabase AI für Parts Catalog Chunks
            }
            
            target_table = table_mapping.get(document_type, document_type + "s")
            logger.info(f"Chunks werden in Tabelle '{target_table}' gespeichert (Dokumenttyp: {document_type})")
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Skip if embedding is empty (error case)
                if not embedding:
                    continue
                
                # Bereite die grundlegenden Chunk-Daten vor (für alle Tabellentypen)
                chunk_data = {
                    "content": chunk,
                    "chunk_index": i,
                    "file_hash": document_data.get("file_hash", ""),
                    "embedding": embedding,
                    "manufacturer": document_data.get("metadata", {}).get("manufacturer", ""),
                    "model": document_data.get("metadata", {}).get("model", ""),
                    "page_number": i // 5  # Einfache Schätzung der Seitenzahl
                }
                
                # Je nach Ziel-Tabelle spezifische Felder hinzufügen
                if target_table == "service_manuals":
                    chunk_data.update({
                        "document_version": document_data.get("metadata", {}).get("document_version", "")
                    })
                elif target_table == "bulletins":
                    chunk_data.update({
                        "document_version": document_data.get("metadata", {}).get("document_version", ""),
                        "models_affected": [] # Sollte ein Array sein, aber im aktuellen Kontext nicht verfügbar
                    })
                elif target_table == "cpmd_documents":
                    chunk_data.update({
                        "document_version": document_data.get("metadata", {}).get("document_version", ""),
                        "message_type": "general"
                    })
                
                # In die entsprechende Tabelle einfügen
                try:
                    result = self.supabase.table(target_table).insert(chunk_data).execute()
                    if result.data:
                        logger.info(f"Chunk {i} in Tabelle {target_table} gespeichert mit ID: {result.data[0].get('id', 'unbekannt')}")
                    else:
                        logger.warning(f"Chunk {i} in Tabelle {target_table} gespeichert, aber keine Daten zurückgegeben")
                except Exception as chunk_err:
                    logger.error(f"Fehler beim Speichern von Chunk {i} in {target_table}: {chunk_err}")
            
            logger.info(f"Successfully stored document data for {document_data.get('title', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document data: {str(e)}")
            return False

class ServiceManualProcessor(DocumentProcessor):
    """Processor for service manuals"""
    
    def process_document(self, file_path: str, file_hash: str = None, log_id: str = None) -> bool:
        """Process a service manual document"""
        try:
            # Open the PDF document
            pdf_document = fitz.open(file_path)
            
            # Extract data
            pages_data = self.extract_text(pdf_document)
            images_data = self.extract_images(pdf_document, file_path)
            metadata = self.extract_metadata(pdf_document)
            
            # Combine all text for chunking
            all_text = ""
            for page in pages_data:
                all_text += page["text"] + " "
            
            # Chunk the text
            chunk_size = self.config.get("processing", {}).get("chunk_size", 1000)
            overlap = self.config.get("processing", {}).get("chunk_overlap", 200)
            chunks = self.chunk_text(all_text, chunk_size, overlap)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare document data
            document_data = {
                "title": os.path.basename(file_path),
                "file_path": file_path,
                "document_type": "service_manual",
                "metadata": metadata,
                "pages": pages_data,
                "images": images_data,
                "chunks": chunks,
                "embeddings": embeddings,
                "file_hash": file_hash
            }
            
            # Store in database
            success = self.store_document_data(document_data)
            
            # Close the PDF document
            pdf_document.close()
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing service manual {file_path}: {str(e)}")
            return False

class PartsManualProcessor(DocumentProcessor):
    """Processor for parts manuals"""
    
    def process_document(self, file_path: str, file_hash: str = None, log_id: str = None) -> bool:
        """Process a parts manual document"""
        try:
            # Open the PDF document
            pdf_document = fitz.open(file_path)
            
            # Extract data
            pages_data = self.extract_text(pdf_document)
            images_data = self.extract_images(pdf_document, file_path)
            metadata = self.extract_metadata(pdf_document)
            
            # Look for associated CSV file
            base_name = os.path.splitext(file_path)[0]
            csv_file = f"{base_name}.csv"
            parts_data = []
            
            if os.path.exists(csv_file):
                # Process CSV data if exists
                # This would typically contain structured parts data
                import csv
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        parts_data.append(row)
                
                # Add parts data to metadata
                metadata["parts_data"] = parts_data
            
            # Combine all text for chunking
            all_text = ""
            for page in pages_data:
                all_text += page["text"] + " "
            
            # Chunk the text
            chunk_size = self.config.get("processing", {}).get("chunk_size", 1000)
            overlap = self.config.get("processing", {}).get("chunk_overlap", 200)
            chunks = self.chunk_text(all_text, chunk_size, overlap)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare document data
            document_data = {
                "title": os.path.basename(file_path),
                "file_path": file_path,
                "document_type": "parts_manual",
                "metadata": metadata,
                "pages": pages_data,
                "images": images_data,
                "chunks": chunks,
                "embeddings": embeddings,
                "file_hash": file_hash
            }
            
            # Store in database
            success = self.store_document_data(document_data)
            
            # Close the PDF document
            pdf_document.close()
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing parts manual {file_path}: {str(e)}")
            return False

class BulletinProcessor(DocumentProcessor):
    """Processor for bulletins"""
    
    def process_document(self, file_path: str, file_hash: str = None, log_id: str = None) -> bool:
        """Process a bulletin document"""
        try:
            # Open the PDF document
            pdf_document = fitz.open(file_path)
            
            # Extract data
            pages_data = self.extract_text(pdf_document)
            images_data = self.extract_images(pdf_document, file_path)
            metadata = self.extract_metadata(pdf_document)
            
            # Combine all text for chunking
            all_text = ""
            for page in pages_data:
                all_text += page["text"] + " "
            
            # Chunk the text
            chunk_size = self.config.get("processing", {}).get("chunk_size", 1000)
            overlap = self.config.get("processing", {}).get("chunk_overlap", 200)
            chunks = self.chunk_text(all_text, chunk_size, overlap)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare document data
            document_data = {
                "title": os.path.basename(file_path),
                "file_path": file_path,
                "document_type": "bulletin",
                "metadata": metadata,
                "pages": pages_data,
                "images": images_data,
                "chunks": chunks,
                "embeddings": embeddings,
                "file_hash": file_hash
            }
            
            # Store in database
            success = self.store_document_data(document_data)
            
            # Close the PDF document
            pdf_document.close()
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing bulletin {file_path}: {str(e)}")
            return False

class CPMDProcessor(DocumentProcessor):
    """Processor for CPMD documents"""
    
    def process_document(self, file_path: str, file_hash: str = None, log_id: str = None) -> bool:
        """Process a CPMD document"""
        try:
            # Open the PDF document
            pdf_document = fitz.open(file_path)
            
            # Extract data
            pages_data = self.extract_text(pdf_document)
            images_data = self.extract_images(pdf_document, file_path)
            metadata = self.extract_metadata(pdf_document)
            
            # Combine all text for chunking
            all_text = ""
            for page in pages_data:
                all_text += page["text"] + " "
            
            # Chunk the text
            chunk_size = self.config.get("processing", {}).get("chunk_size", 1000)
            overlap = self.config.get("processing", {}).get("chunk_overlap", 200)
            chunks = self.chunk_text(all_text, chunk_size, overlap)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare document data
            document_data = {
                "title": os.path.basename(file_path),
                "file_path": file_path,
                "document_type": "cpmd",
                "metadata": metadata,
                "pages": pages_data,
                "images": images_data,
                "chunks": chunks,
                "embeddings": embeddings,
                "file_hash": file_hash
            }
            
            # Store in database
            success = self.store_document_data(document_data)
            
            # Close the PDF document
            pdf_document.close()
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing CPMD document {file_path}: {str(e)}")
            return False

class DocumentProcessorFactory:
    """Factory for creating document processors"""
    
    @staticmethod
    def create_processor(document_type: str, supabase_client, r2_client, config: Dict[str, Any]) -> DocumentProcessor:
        """Create a document processor based on document type"""
        if document_type == "service_manual":
            return ServiceManualProcessor(supabase_client, r2_client, config)
        elif document_type == "parts_manual":
            return PartsManualProcessor(supabase_client, r2_client, config)
        elif document_type == "bulletin":
            return BulletinProcessor(supabase_client, r2_client, config)
        elif document_type == "cpmd":
            return CPMDProcessor(supabase_client, r2_client, config)
        else:
            # Default to service manual processor
            logger.warning(f"Unknown document type: {document_type}. Using ServiceManualProcessor as default.")
            return ServiceManualProcessor(supabase_client, r2_client, config)