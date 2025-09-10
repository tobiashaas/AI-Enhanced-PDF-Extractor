#!/usr/bin/env python3
"""
R2 Public URL Tester
Testet ob R2 Bucket korrekt für Public Access konfiguriert ist
"""

import boto3
import json
import requests
from botocore.config import Config

def main():
    print("🌐 R2 Public Access Tester")
    print("=" * 40)
    
    try:
        # Load config
        with open('config.json') as f:
            config = json.load(f)
        
        # Setup R2 client with SigV4
        r2_client = boto3.client('s3',
            endpoint_url=f'https://{config["r2_account_id"]}.r2.cloudflarestorage.com',
            aws_access_key_id=config['r2_access_key_id'],
            aws_secret_access_key=config['r2_secret_access_key'],
            config=Config(signature_version='s3v4', region_name='auto')
        )
        
        bucket_name = config['r2_bucket_name']
        account_id = config['r2_account_id']
        
        # Get a sample image
        print("🔍 Suche Beispiel-Bild...")
        objects = r2_client.list_objects_v2(Bucket=bucket_name, Prefix='images/', MaxKeys=1)
        
        if 'Contents' not in objects or len(objects['Contents']) == 0:
            print("❌ Keine Bilder in R2 gefunden")
            return
        
        sample_key = objects['Contents'][0]['Key']
        print(f"📁 Test-Bild: {sample_key}")
        
        # Test different URL formats
        url_formats = [
            # Standard R2 public URL
            f"https://pub-{account_id}.r2.dev/{sample_key}",
            # Alternative format
            f"https://{bucket_name}.{account_id}.r2.cloudflarestorage.com/{sample_key}",
            # Custom domain format (if configured)
            f"https://{bucket_name}.r2.dev/{sample_key}"
        ]
        
        print("\\n🧪 Teste verschiedene Public URL Formate...")
        
        working_url = None
        for i, url in enumerate(url_formats, 1):
            print(f"\\n{i}. {url[:80]}{'...' if len(url) > 80 else ''}")
            
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                status = response.status_code
                
                if status == 200:
                    print(f"   ✅ Status: {status} - PUBLIC ACCESS FUNKTIONIERT!")
                    working_url = url
                    break
                elif status == 403:
                    print(f"   ❌ Status: {status} - Forbidden (Public Access nicht aktiviert)")
                elif status == 404:
                    print(f"   ⚠️  Status: {status} - Not Found (Falsches URL Format)")
                else:
                    print(f"   ⚠️  Status: {status} - {response.reason}")
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Connection Error: {e}")
        
        if working_url:
            print(f"\\n🎉 SUCCESS! Public URL funktioniert:")
            print(f"✅ {working_url}")
            
            # Test full GET request
            try:
                response = requests.get(working_url, timeout=10)
                if response.status_code == 200:
                    print(f"📊 Bild-Größe: {len(response.content):,} bytes")
                    print(f"📋 Content-Type: {response.headers.get('content-type', 'unknown')}")
                    
                    # Save sample for verification
                    with open('test_image_download.png', 'wb') as f:
                        f.write(response.content)
                    print(f"💾 Test-Bild gespeichert: test_image_download.png")
                    
            except Exception as e:
                print(f"⚠️ Download-Test fehlgeschlagen: {e}")
        else:
            print("\\n❌ KEIN PUBLIC ACCESS GEFUNDEN!")
            print("\\n🔧 Nächste Schritte:")
            print("1. 🌐 Gehe zu Cloudflare Dashboard")
            print("2. 📁 R2 Object Storage → Dein Bucket")
            print("3. ⚙️ Settings → Public Access")
            print("4. ✅ Enable Public Access")
            print("5. 🔄 Führe dieses Script erneut aus")
            
            print("\\n🔐 Alternative: Presigned URLs verwenden:")
            print("python3 test_presigned_url.py")
        
        # Generate working presigned URL as backup
        print("\\n🔐 Backup: Presigned URL (immer verfügbar):")
        try:
            presigned_url = r2_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': sample_key},
                ExpiresIn=3600,
                HttpMethod='GET'
            )
            print(f"🔗 {presigned_url[:100]}...")
            print("⏰ Gültig für 1 Stunde")
            
        except Exception as e:
            print(f"❌ Presigned URL Fehler: {e}")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()
