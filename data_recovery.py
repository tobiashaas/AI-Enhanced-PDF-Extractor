#!/usr/bin/env python3
"""
Data Recovery Script: Import orphaned R2 images to Supabase
Recovers images that exist in R2 but missing from database
"""

import boto3
import json
import re
from datetime import datetime, timezone
from supabase import create_client
from botocore.exceptions import ClientError

class DataRecovery:
    def __init__(self):
        # Load config
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize clients
        self.supabase = create_client(self.config['supabase_url'], self.config['supabase_key'])
        self.r2_client = boto3.client(
            's3',
            endpoint_url=f'https://{self.config["r2_account_id"]}.r2.cloudflarestorage.com',
            aws_access_key_id=self.config['r2_access_key_id'],
            aws_secret_access_key=self.config['r2_secret_access_key']
        )
    
    def analyze_orphaned_images(self):
        """Analyze which R2 images are missing from database"""
        print("🔍 Analysiere verwaiste R2-Bilder...")
        
        # Get all R2 images
        r2_images = []
        try:
            response = self.r2_client.list_objects_v2(
                Bucket=self.config['r2_bucket_name'], 
                Prefix='images/'
            )
            
            if 'Contents' in response:
                r2_images = [obj['Key'] for obj in response['Contents']]
                print(f"📊 R2 Images gefunden: {len(r2_images)}")
            else:
                print("❌ Keine R2 Images gefunden")
                return []
                
        except Exception as e:
            print(f"❌ Fehler beim R2-Abruf: {e}")
            return []
        
        # Get all database images
        db_keys = set()
        try:
            result = self.supabase.table('images').select('r2_key').execute()
            db_keys = {img['r2_key'] for img in result.data}
            print(f"📊 DB Images gefunden: {len(db_keys)}")
            
        except Exception as e:
            print(f"❌ Fehler beim DB-Abruf: {e}")
            return []
        
        # Find orphaned images
        orphaned = [key for key in r2_images if key not in db_keys]
        print(f"🔍 Verwaiste Bilder: {len(orphaned)}")
        
        return orphaned
    
    def parse_r2_key(self, r2_key):
        """Parse R2 key to extract metadata"""
        # Pattern: images/{file_hash}/page_{page_num}_img_{img_index}_{img_hash}.png
        pattern = r'images/([^/]+)/page_(\d+)_img_(\d+)_([^.]+)\.png'
        match = re.match(pattern, r2_key)
        
        if match:
            file_hash, page_num, img_index, img_hash = match.groups()
            return {
                'file_hash': file_hash,
                'page_number': int(page_num),
                'image_index': int(img_index),
                'image_hash': img_hash
            }
        return None
    
    def get_image_dimensions(self, r2_key):
        """Get image dimensions from R2"""
        try:
            response = self.r2_client.head_object(
                Bucket=self.config['r2_bucket_name'],
                Key=r2_key
            )
            # We can't get dimensions from head_object, use defaults
            return {'width': 100, 'height': 100, 'size': response['ContentLength']}
        except:
            return {'width': 100, 'height': 100, 'size': 1024}
    
    def recover_orphaned_images(self, orphaned_keys, batch_size=100):
        """Import orphaned R2 images to database"""
        print(f"🔄 Starte Recovery von {len(orphaned_keys)} Bildern...")
        
        total_recovered = 0
        failed_count = 0
        
        for i in range(0, len(orphaned_keys), batch_size):
            batch = orphaned_keys[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(orphaned_keys) + batch_size - 1) // batch_size
            
            print(f"📦 Batch {batch_num}/{total_batches}: {len(batch)} Bilder")
            
            batch_data = []
            for r2_key in batch:
                # Parse metadata from R2 key
                metadata = self.parse_r2_key(r2_key)
                if not metadata:
                    print(f"   ⚠️ Ungültiger R2-Key: {r2_key}")
                    failed_count += 1
                    continue
                
                # Get image info
                img_info = self.get_image_dimensions(r2_key)
                
                # Generate public URL
                r2_url = f"https://pub-{self.config['r2_account_id']}.r2.dev/{r2_key}"
                
                # Create database record
                image_record = {
                    'file_hash': metadata['file_hash'],
                    'page_number': metadata['page_number'],
                    'image_index': metadata['image_index'],
                    'r2_key': r2_key,
                    'r2_url': r2_url,
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'format': 'PNG',
                    'metadata': {
                        'recovered_at': datetime.now(timezone.utc).isoformat(),
                        'size_bytes': img_info['size'],
                        'status': 'recovered_orphan',
                        'original_hash': metadata['image_hash']
                    }
                }
                
                batch_data.append(image_record)
            
            # Insert batch
            if batch_data:
                try:
                    result = self.supabase.table('images').insert(batch_data).execute()
                    recovered_count = len(result.data)
                    total_recovered += recovered_count
                    print(f"   ✅ {recovered_count} Bilder erfolgreich importiert")
                    
                except Exception as e:
                    print(f"   ❌ Batch-Import fehlgeschlagen: {e}")
                    failed_count += len(batch_data)
        
        print(f"🎉 Recovery abgeschlossen:")
        print(f"   ✅ Erfolgreich: {total_recovered}")
        print(f"   ❌ Fehlgeschlagen: {failed_count}")
        print(f"   📊 Erfolgsquote: {total_recovered/(total_recovered+failed_count)*100:.1f}%")
        
        return total_recovered, failed_count
    
    def run_full_recovery(self):
        """Run complete data recovery process"""
        print("🚀 Starte vollständige Datenwiederherstellung...")
        print("=" * 60)
        
        # Step 1: Analyze
        orphaned = self.analyze_orphaned_images()
        
        if not orphaned:
            print("✅ Keine verwaisten Bilder gefunden - alles in Ordnung!")
            return
        
        print(f"📋 Gefunden: {len(orphaned)} verwaiste Bilder")
        print("\n🔍 Beispiele:")
        for key in orphaned[:5]:
            print(f"   {key}")
        
        # Ask for confirmation
        response = input(f"\n❓ Sollen {len(orphaned)} Bilder wiederhergestellt werden? (y/N): ")
        if response.lower() != 'y':
            print("❌ Wiederherstellung abgebrochen")
            return
        
        # Step 2: Recover
        print(f"\n🔄 Starte Wiederherstellung...")
        recovered, failed = self.recover_orphaned_images(orphaned)
        
        print(f"\n🎉 Wiederherstellung abgeschlossen!")
        return recovered, failed

if __name__ == "__main__":
    recovery = DataRecovery()
    recovery.run_full_recovery()
