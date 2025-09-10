#!/usr/bin/env python3
"""
Update Database URLs to R2 Public Format
Aktualisiert alle image URLs in der Datenbank auf das korrekte R2 Public Format
"""

import json
from supabase import create_client

def main():
    print("🔄 Database URL Update zu R2 Public Format")
    print("=" * 50)
    
    try:
        # Load config
        with open('config.json') as f:
            config = json.load(f)
        
        # Connect to Supabase
        supabase = create_client(config['supabase_url'], config['supabase_key'])
        
        # Get current images
        print("📊 Lade aktuelle Bild-URLs aus Datenbank...")
        result = supabase.table('images').select('id, r2_key, r2_url').execute()
        images = result.data
        
        if not images:
            print("❌ Keine Bilder in der Datenbank gefunden")
            return
        
        print(f"📁 Gefunden: {len(images)} Bilder")
        
        # Show current URL format
        sample_current = images[0]['r2_url'] if images[0]['r2_url'] else 'Keine URL'
        print(f"🔍 Aktuelles Format: {sample_current[:80]}...")
        
        # Define the working public domain
        working_public_domain = "pub-80a63376fddf4b909ed55ee53a401a93.r2.dev"
        print(f"🎯 Ziel-Format: https://{working_public_domain}/images/...")
        
        # Ask for confirmation
        print(f"\\n❓ Sollen {len(images)} URLs aktualisiert werden?")
        print("   ✅ Von: Presigned/Private URLs")
        print(f"   ✅ Zu: https://{working_public_domain}/[r2_key]")
        
        confirm = input("\\nFortfahren? (y/N): ").lower().strip()
        if confirm != 'y':
            print("❌ Abgebrochen")
            return
        
        # Update URLs
        print("\\n🔄 Aktualisiere URLs...")
        updated_count = 0
        batch_size = 50
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(images) + batch_size - 1) // batch_size
            
            print(f"📦 Batch {batch_num}/{total_batches}: {len(batch)} URLs")
            
            for image in batch:
                try:
                    # Generate new public URL
                    new_url = f"https://{working_public_domain}/{image['r2_key']}"
                    
                    # Update in database
                    supabase.table('images').update({
                        'r2_url': new_url
                    }).eq('id', image['id']).execute()
                    
                    updated_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Fehler bei Bild {image['id']}: {e}")
            
            print(f"   ✅ Batch {batch_num} abgeschlossen")
        
        print(f"\\n🎉 URL Update abgeschlossen!")
        print(f"✅ {updated_count} URLs erfolgreich aktualisiert")
        
        # Test a sample URL
        if updated_count > 0:
            print("\\n🧪 Teste aktualisierte URL...")
            test_result = supabase.table('images').select('r2_url').limit(1).execute()
            if test_result.data:
                test_url = test_result.data[0]['r2_url']
                print(f"🔗 Beispiel: {test_url}")
                
                # Test accessibility
                import requests
                try:
                    response = requests.head(test_url, timeout=10)
                    if response.status_code == 200:
                        print("✅ URL ist öffentlich zugänglich!")
                    else:
                        print(f"⚠️ Status: {response.status_code}")
                except Exception as e:
                    print(f"⚠️ Test fehlgeschlagen: {e}")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()
