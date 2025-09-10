#!/usr/bin/env python3
"""
R2 Access Configuration Utility
Konfiguriert Public Access oder generiert Presigned URLs für private R2 Images
"""

import boto3
import json
from datetime import datetime, timedelta
from supabase import create_client

def load_config():
    """Load configuration"""
    with open('config.json') as f:
        return json.load(f)

def setup_r2_client(config):
    """Setup R2 client"""
    return boto3.client('s3',
        endpoint_url=f'https://{config["r2_account_id"]}.r2.cloudflarestorage.com',
        aws_access_key_id=config['r2_access_key_id'],
        aws_secret_access_key=config['r2_secret_access_key']
    )

def check_current_access(r2_client, bucket_name, account_id):
    """Check current bucket access configuration"""
    print("🔍 Überprüfe aktuelle R2 Bucket Konfiguration...")
    
    # Test public access with a sample image
    try:
        objects = r2_client.list_objects_v2(Bucket=bucket_name, Prefix='images/', MaxKeys=1)
        if 'Contents' in objects and len(objects['Contents']) > 0:
            sample_key = objects['Contents'][0]['Key']
            public_url = f"https://pub-{account_id}.r2.dev/{sample_key}"
            
            print(f"📁 Beispiel Bild: {sample_key}")
            print(f"🌐 Public URL: {public_url}")
            
            # Test if URL is accessible (basic check)
            import requests
            try:
                response = requests.head(public_url, timeout=10)
                if response.status_code == 200:
                    print("✅ Public Access: AKTIV")
                    return True
                else:
                    print(f"❌ Public Access: NICHT AKTIV (Status: {response.status_code})")
                    return False
            except Exception as e:
                print(f"❌ Public Access: NICHT AKTIV (Error: {e})")
                return False
        else:
            print("📁 Keine Bilder im Bucket gefunden")
            return False
    except Exception as e:
        print(f"⚠️ Konnte Bucket nicht prüfen: {e}")
        return False

def option1_configure_public_access(r2_client, bucket_name, account_id):
    """Option 1: Configure public access for R2 bucket"""
    print("\n🌐 Option 1: Public Access konfigurieren")
    print("=" * 50)
    
    print("⚠️  ACHTUNG: Dies macht ALLE Bilder öffentlich zugänglich!")
    print("🔒 Für Produktionsumgebungen nur mit Vorsicht verwenden!")
    
    confirm = input("\n❓ Public Access aktivieren? (y/N): ").lower().strip()
    if confirm != 'y':
        print("❌ Public Access nicht aktiviert")
        return False
    
    try:
        # Cloudflare R2 Public Access über Dashboard konfigurieren
        print("\n📝 MANUELLE SCHRITTE für Public Access:")
        print("=" * 40)
        print("1. 🌐 Gehe zu Cloudflare Dashboard > R2 Object Storage")
        print(f"2. 📁 Wähle Bucket: {bucket_name}")
        print("3. ⚙️ Gehe zu Settings > Public Access")
        print("4. ✅ Aktiviere 'Enable Public Access'")
        print("5. 🔗 Konfiguriere Custom Domain (optional)")
        print(f"6. 🌍 Public URL Format: https://pub-{account_id}.r2.dev/")
        
        print("\n💡 Alternative: R2 Custom Domain für bessere Performance")
        print("   - Konfiguriere eigene Domain (z.B. images.yourdomain.com)")
        print("   - Bessere Caching und Performance")
        print("   - Mehr Kontrolle über Access")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Public Access Konfiguration: {e}")
        return False

def option2_generate_presigned_urls(r2_client, bucket_name, config):
    """Option 2: Generate presigned URLs for private access"""
    print("\n🔐 Option 2: Presigned URLs für privaten Zugriff")
    print("=" * 50)
    
    try:
        # Get sample images
        objects = r2_client.list_objects_v2(Bucket=bucket_name, Prefix='images/', MaxKeys=5)
        
        if 'Contents' not in objects or len(objects['Contents']) == 0:
            print("❌ Keine Bilder gefunden")
            return False
        
        print(f"📊 Generiere Presigned URLs für {len(objects['Contents'])} Beispiel-Bilder...")
        
        # Generate presigned URLs (24 hours expiry)
        expiration = 3600 * 24  # 24 hours
        
        for obj in objects['Contents'][:3]:  # Show first 3
            key = obj['Key']
            
            try:
                presigned_url = r2_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': key},
                    ExpiresIn=expiration
                )
                
                print(f"\n📁 {key}")
                print(f"🔗 Presigned URL (24h): {presigned_url[:100]}...")
                
            except Exception as e:
                print(f"❌ Fehler bei URL Generation für {key}: {e}")
        
        print(f"\n✅ Presigned URLs generiert (Gültig für {expiration//3600} Stunden)")
        
        # Update database with presigned URLs option
        update_db = input("\n❓ Sollen alle Datenbank URLs zu Presigned URLs aktualisiert werden? (y/N): ").lower().strip()
        if update_db == 'y':
            update_database_with_presigned_urls(config, r2_client, bucket_name, expiration)
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Presigned URL Generation: {e}")
        return False

def update_database_with_presigned_urls(config, r2_client, bucket_name, expiration):
    """Update database with presigned URLs"""
    print("\n🔄 Aktualisiere Datenbank mit Presigned URLs...")
    
    try:
        supabase = create_client(config['supabase_url'], config['supabase_key'])
        
        # Get all images from database
        result = supabase.table('images').select('id, r2_key').execute()
        images = result.data
        
        if not images:
            print("❌ Keine Bilder in Datenbank gefunden")
            return False
        
        print(f"📊 Aktualisiere {len(images)} Bilder...")
        
        # Update in batches of 50
        batch_size = 50
        updated_count = 0
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            updates = []
            for image in batch:
                try:
                    presigned_url = r2_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket_name, 'Key': image['r2_key']},
                        ExpiresIn=expiration
                    )
                    
                    updates.append({
                        'id': image['id'],
                        'r2_url': presigned_url,
                        'url_expires_at': (datetime.now() + timedelta(seconds=expiration)).isoformat()
                    })
                    
                except Exception as e:
                    print(f"⚠️ Fehler bei {image['r2_key']}: {e}")
                    continue
            
            if updates:
                # Bulk update
                for update in updates:
                    supabase.table('images').update({
                        'r2_url': update['r2_url'],
                        'metadata': {'url_expires_at': update['url_expires_at']}
                    }).eq('id', update['id']).execute()
                
                updated_count += len(updates)
                print(f"   📦 Batch {(i//batch_size)+1}: {len(updates)} URLs aktualisiert")
        
        print(f"\n✅ {updated_count} Presigned URLs erfolgreich in Datenbank gespeichert")
        print(f"⏰ URLs gültig bis: {(datetime.now() + timedelta(seconds=expiration)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Datenbank Update: {e}")
        return False

def option3_hybrid_approach(r2_client, bucket_name, config):
    """Option 3: Hybrid approach with access tiers"""
    print("\n🔀 Option 3: Hybrid Ansatz (Empfohlen)")
    print("=" * 50)
    
    print("💡 Intelligenter Zugriff basierend auf Use Case:")
    print("   🌐 Public: Für häufig genutzte Bilder (Thumbnails, Previews)")
    print("   🔐 Private: Für vertrauliche Dokumente (Full-Size, Original)")
    print("   ⚡ CDN: Mit Cloudflare für optimale Performance")
    
    print("\n📋 Implementierungs-Empfehlungen:")
    print("1. 📁 Separate Buckets/Prefixes:")
    print("   - public/thumbnails/ → Public Access")
    print("   - private/originals/ → Presigned URLs")
    
    print("\n2. 🔄 Automatische Tier-Zuweisung:")
    print("   - Kleine Bilder (<1MB) → Public")
    print("   - Große Bilder (>1MB) → Private")
    
    print("\n3. 🚀 Performance Optimierung:")
    print("   - Cloudflare CDN für Public Images")
    print("   - Presigned URLs mit Cache Headers")
    
    return True

def main():
    """Main function"""
    print("🚀 R2 Access Configuration Utility")
    print("=" * 50)
    
    try:
        config = load_config()
        r2_client = setup_r2_client(config)
        bucket_name = config['r2_bucket_name']
        account_id = config['r2_account_id']
        
        # Check current access
        is_public = check_current_access(r2_client, bucket_name, account_id)
        
        if is_public:
            print("\n✅ Bucket ist bereits public zugänglich!")
            print("🔗 Bilder sind über Public URLs erreichbar")
            return
        
        print("\n📋 Verfügbare Optionen:")
        print("1. 🌐 Public Access konfigurieren (Alle Bilder öffentlich)")
        print("2. 🔐 Presigned URLs generieren (Privater Zugriff)")
        print("3. 🔀 Hybrid Ansatz (Empfohlen)")
        print("4. ❌ Abbrechen")
        
        choice = input("\n❓ Wähle Option (1-4): ").strip()
        
        if choice == '1':
            option1_configure_public_access(r2_client, bucket_name, account_id)
        elif choice == '2':
            option2_generate_presigned_urls(r2_client, bucket_name, config)
        elif choice == '3':
            option3_hybrid_approach(r2_client, bucket_name, config)
        elif choice == '4':
            print("❌ Abgebrochen")
        else:
            print("❌ Ungültige Auswahl")
    
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()
