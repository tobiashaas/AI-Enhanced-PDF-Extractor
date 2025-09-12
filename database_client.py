#!/usr/bin/env python3
"""
Database Operations Client
Handles all Supabase database operations for the AI PDF Processing System
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from supabase import create_client, Client

# Schema compatibility functions (inline)
def map_chunks_data_to_schema(data):
    """Map chunks data to current schema"""
    return data

def map_images_data_to_schema(data):
    """Map images data to current schema"""
    return data

def get_existing_image_hashes_compatible(client, hashes):
    """Get existing image hashes compatible with current schema"""
    try:
        result = client.table('images').select('hash').in_('hash', hashes).execute()
        return [row['hash'] for row in result.data]
    except Exception as e:
        logger.error(f"Error getting image hashes: {e}")
        return []

def get_image_by_hash_compatible(client, hash_value):
    """Get image by hash compatible with current schema"""
    try:
        result = client.table('images').select('*').eq('hash', hash_value).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error getting image by hash: {e}")
        return None

# Initialize logger
logger = logging.getLogger(__name__)


class DatabaseClient:
    """Supabase Database Client with optimized operations"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.supabase: Client = self.client  # Alias for backward compatibility
        
    # ========== Processing Logs Operations ==========
    
    def create_processing_log(self, file_path: str, file_hash: str, original_filename: str) -> Optional[Dict]:
        """Create new processing log entry"""
        try:
            log_data = {
                'file_path': file_path,
                'file_hash': file_hash,
                'original_filename': original_filename,
                'status': 'processing',
                'processing_stage': 'started',
                'progress_percentage': 0,
                'chunks_created': 0,
                'images_extracted': 0,
                'processing_time_seconds': 0,
                'retry_count': 0,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table('processing_logs').insert(log_data).execute()
            logging.info(f"✅ Created processing log for: {original_filename}")
            return result.data[0] if result.data else None
        except Exception as e:
            logging.error(f"❌ Failed to create processing log: {e}")
            return None
    
    def update_processing_log(self, file_hash: str, updates: Dict[str, Any]) -> bool:
        """Update processing log with new data"""
        try:
            updates['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            result = self.client.table('processing_logs').update(updates).eq('file_hash', file_hash).execute()
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"❌ Failed to update processing log: {e}")
            return False
    
    def complete_processing_log(self, file_hash: str, success: bool, error_message: Optional[str] = None) -> bool:
        """Mark processing log as completed or failed"""
        try:
            updates = {
                'status': 'completed' if success else 'failed',
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if error_message:
                updates['error_message'] = error_message
                
            result = self.client.table('processing_logs').update(updates).eq('file_hash', file_hash).execute()
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"❌ Failed to complete processing log: {e}")
            return False
    
    # ========== Images Operations ==========
    
    def insert_image(self, image_data: Dict[str, Any]) -> bool:
        """Insert image record into database"""
        try:
            result = self.client.table('images').insert(image_data).execute()
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"❌ Failed to insert image: {e}")
            return False
    
    def get_images_by_hash(self, file_hash: str) -> List[Dict]:
        """Get all images for a specific file hash"""
        try:
            result = self.client.table('images').select('*').eq('file_hash', file_hash).execute()
            return result.data
        except Exception as e:
            logging.error(f"❌ Failed to get images: {e}")
            return []
    
    # ========== Chunks Operations ==========
    
    def insert_chunk(self, chunk_data: Dict[str, Any]) -> bool:
        """Insert chunk record into database"""
        try:
            # Map to schema-compatible format
            mapped_data = map_chunks_data_to_schema(chunk_data)
            result = self.client.table('chunks').insert(mapped_data).execute()
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"❌ Failed to insert chunk: {e}")
            return False
    
    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> bool:
        """Insert multiple chunks in batch"""
        try:
            # Map all chunks to schema-compatible format
            mapped_chunks = [map_chunks_data_to_schema(chunk) for chunk in chunks]
            result = self.client.table('chunks').insert(mapped_chunks).execute()
            logging.info(f"✅ Inserted {len(chunks)} chunks")
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"❌ Failed to insert chunks batch: {e}")
            return False
    
    # ========== Parts Catalog Operations ==========
    
    def insert_parts_batch(self, parts: List[Dict[str, Any]]) -> bool:
        """Insert multiple parts in batch"""
        try:
            result = self.client.table('parts_catalog').insert(parts).execute()
            logging.info(f"✅ Inserted {len(parts)} parts")
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"❌ Failed to insert parts batch: {e}")
            return False
    
    def search_parts(self, query: str, manufacturer: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Search parts catalog with optional filters"""
        try:
            query_builder = self.client.table('parts_catalog').select('*')
            
            if manufacturer:
                query_builder = query_builder.eq('manufacturer', manufacturer)
                
            # Text search across multiple fields
            query_builder = query_builder.or_(
                f'part_number.ilike.%{query}%,'
                f'part_name.ilike.%{query}%,'
                f'description.ilike.%{query}%'
            )
            
            result = query_builder.limit(limit).execute()
            return result.data
        except Exception as e:
            logging.error(f"❌ Failed to search parts: {e}")
            return []
    
    # ========== Utility Operations ==========
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get overall processing statistics"""
        try:
            # Get total counts from each table
            parts_count = self.client.table('parts_catalog').select('id', count='exact').execute().count
            images_count = self.client.table('images').select('id', count='exact').execute().count  
            chunks_count = self.client.table('chunks').select('id', count='exact').execute().count
            logs_count = self.client.table('processing_logs').select('id', count='exact').execute().count
            
            return {
                'total_parts': parts_count or 0,
                'total_images': images_count or 0,
                'total_chunks': chunks_count or 0,
                'total_logs': logs_count or 0
            }
        except Exception as e:
            logging.error(f"❌ Failed to get processing stats: {e}")
            return {'total_parts': 0, 'total_images': 0, 'total_chunks': 0, 'total_logs': 0}
    
    def check_file_processed(self, file_hash: str) -> bool:
        """Check if file has already been processed"""
        try:
            result = self.client.table('processing_logs').select('id').eq('file_hash', file_hash).eq('status', 'completed').execute()
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"❌ Failed to check file processed status: {e}")
            return False
    
    def check_existing_image_hashes(self, image_hashes):
        """Check which image hashes already exist in the database"""
        try:
            # Use compatibility function for schema mismatch
            return get_existing_image_hashes_compatible(self, image_hashes)
        except Exception as e:
            logger.error(f"Error checking image hashes: {e}")
            return []  # Return empty list on error instead of raising
    
    def insert_images(self, images_data):
        """Insert multiple images into the database"""
        try:
            # Map all images to schema-compatible format
            mapped_images = [map_images_data_to_schema(img) for img in images_data]
            result = self.supabase.table("images").insert(mapped_images).execute()
            return result
        except Exception as e:
            logger.error(f"Error inserting images: {e}")
            raise
    
    def insert_chunks(self, chunks_data):
        """Insert multiple chunks into the database"""
        try:
            # Map all chunks to schema-compatible format
            mapped_chunks = [map_chunks_data_to_schema(chunk) for chunk in chunks_data]
            result = self.supabase.table("chunks").insert(mapped_chunks).execute()
            return result
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            raise
    
    def get_chunks_page_numbers(self, file_hash):
        """Get page numbers for existing chunks"""
        try:
            result = self.supabase.table("chunks").select("page_number").eq("file_hash", file_hash).execute()
            return result
        except Exception as e:
            logger.error(f"Error getting chunk page numbers: {e}")
            raise
    
    def get_chunk_id_by_page(self, file_hash, page_num):
        """Get chunk ID by file hash and page number"""
        try:
            result = self.supabase.table("chunks").select("id").eq("file_hash", file_hash).eq("page_number", page_num).limit(1).execute()
            return result
        except Exception as e:
            logger.error(f"Error getting chunk ID: {e}")
            raise
    
    def get_processing_log_status(self, file_hash):
        """Get processing log status for file"""
        try:
            result = self.supabase.table("processing_logs").select("status").eq("file_hash", file_hash).execute()
            return result
        except Exception as e:
            logger.error(f"Error getting processing log status: {e}")
            raise
    
    def update_processing_log(self, file_hash, update_data):
        """Update processing log for file"""
        try:
            result = self.supabase.table("processing_logs").update(update_data).eq("file_hash", file_hash).execute()
            return result
        except Exception as e:
            logger.error(f"Error updating processing log: {e}")
            raise
    
    def update_processing_log_by_id(self, log_id, update_data):
        """Update processing log by ID"""
        try:
            result = self.supabase.table("processing_logs").update(update_data).eq("id", log_id).execute()
            return result
        except Exception as e:
            logger.error(f"Error updating processing log by ID: {e}")
            raise
    
    def upsert_processing_log(self, log_data):
        """Upsert processing log entry"""
        try:
            result = self.supabase.table("processing_logs").upsert(log_data).execute()
            return result
        except Exception as e:
            logger.error(f"Error upserting processing log: {e}")
            raise
    
    def update_processing_log_by_file_hash(self, file_hash, update_data):
        """Update processing log by file hash"""
        try:
            result = self.supabase.table("processing_logs").update(update_data).eq("file_hash", file_hash).execute()
            return result
        except Exception as e:
            logger.error(f"Error updating processing log by file hash: {e}")
            raise
    
    def get_image_by_hash(self, image_hash):
        """Get image details by hash"""
        try:
            # Use compatibility function for schema mismatch
            return get_image_by_hash_compatible(self, image_hash)
        except Exception as e:
            logger.error(f"Error getting image by hash: {e}")
            return {'data': []}  # Return empty result on error
    
    def insert_chunk_images(self, chunk_images_data):
        """Insert chunk-images relationships"""
        try:
            result = self.supabase.table("chunk_images").insert(chunk_images_data).execute()
            return result
        except Exception as e:
            logger.error(f"Error inserting chunk images: {e}")
            raise