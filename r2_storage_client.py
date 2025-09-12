#!/usr/bin/env python3
"""
Cloudflare R2 Storage Client
Handles all file uploads and URL generation for the AI PDF Processing System
"""

import boto3
from botocore.exceptions import ClientError
import logging
from typing import Optional
from dataclasses import dataclass


@dataclass 
class R2Config:
    """Configuration for Cloudflare R2 Storage"""
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    public_access_url: Optional[str] = None


class R2StorageClient:
    """Cloudflare R2 Storage Client with optimized uploads and public URLs"""
    
    def __init__(self, config: R2Config):
        self.config = config
        self.client = self._init_client()
        
    def _init_client(self):
        """Initialize R2 client with credentials"""
        try:
            return boto3.client(
                's3',
                endpoint_url=f'https://{self.config.account_id}.r2.cloudflarestorage.com',
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name='auto'
            )
        except Exception as e:
            logging.error(f"Failed to initialize R2 client: {e}")
            raise
    
    def upload_file(self, file_data: bytes, key: str, content_type: str = 'application/octet-stream') -> bool:
        """Upload file to R2 storage"""
        try:
            self.client.put_object(
                Bucket=self.config.bucket_name,
                Key=key,
                Body=file_data,
                ContentType=content_type
            )
            logging.info(f"✅ Uploaded to R2: {key}")
            return True
        except ClientError as e:
            logging.error(f"❌ R2 upload failed for {key}: {e}")
            return False
    
    def generate_public_url(self, key: str) -> str:
        """Generate public URL for R2 object"""
        if self.config.public_access_url:
            return f"{self.config.public_access_url}/{key}"
        
        # Fallback to default R2 dev URL format
        return f"https://pub-{self.config.account_id}.r2.dev/{key}"
    
    def file_exists(self, key: str) -> bool:
        """Check if file exists in R2"""
        try:
            self.client.head_object(Bucket=self.config.bucket_name, Key=key)
            return True
        except ClientError:
            return False
    
    def delete_file(self, key: str) -> bool:
        """Delete file from R2"""
        try:
            self.client.delete_object(Bucket=self.config.bucket_name, Key=key)
            logging.info(f"🗑️ Deleted from R2: {key}")
            return True
        except ClientError as e:
            logging.error(f"❌ R2 deletion failed for {key}: {e}")
            return False