#!/usr/bin/env python3
"""
R2 Presigned URL Test
Generiert eine temporäre URL für ein zufälliges Bild aus R2
"""

import boto3
import json
from datetime import datetime, timedelta

def main():
    print("🔗 R2 Presigned URL Generator")
    print("=" * 40)
    
    try:
        # Load config
        with open('config.json') as f:
            config = json.load(f)
        
        # Setup R2 client with explicit SigV4 for R2 compatibility
        from botocore.config import Config
        r2_client = boto3.client('s3',
            endpoint_url=f'https://{config["r2_account_id"]}.r2.cloudflarestorage.com',
            aws_access_key_id=config['r2_access_key_id'],
            aws_secret_access_key=config['r2_secret_access_key'],
            config=Config(signature_version='s3v4', region_name='auto')
        )
        
        bucket_name = config['r2_bucket_name']
        
        # Get a random image
        print("🔍 Suche zufälliges Bild in R2...")
        objects = r2_client.list_objects_v2(Bucket=bucket_name, Prefix='images/', MaxKeys=1)
        
        if 'Contents' not in objects or len(objects['Contents']) == 0:
            print("❌ Keine Bilder in R2 gefunden")
            return
        
        sample_key = objects['Contents'][0]['Key']
        print(f"📁 Gefunden: {sample_key}")
        
        # Generate presigned URL (1 hour expiry)
        expiration = 3600  # 1 hour
        
        print(f"⏰ Generiere Presigned URL (gültig für {expiration//60} Minuten)...")
        
        presigned_url = r2_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': sample_key},
            ExpiresIn=expiration,
            HttpMethod='GET'
        )
        
        expires_at = datetime.now() + timedelta(seconds=expiration)
        
        print("\n" + "="*60)
        print("✅ PRESIGNED URL GENERIERT:")
        print("="*60)
        print(f"🔗 URL: {presigned_url}")
        print(f"⏰ Gültig bis: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        print("\n💡 Diese URL ist temporär zugänglich!")
        print("   - Kann in Browser/Apps verwendet werden")
        print("   - Expiriert automatisch nach 1 Stunde")
        print("   - Keine weitere Authentifizierung nötig")
        
        # Test the URL
        test = input("\n❓ URL in Browser öffnen? (y/N): ").lower().strip()
        if test == 'y':
            import webbrowser
            webbrowser.open(presigned_url)
            print("🌐 URL im Browser geöffnet")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()
