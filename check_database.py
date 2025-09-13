#!/usr/bin/env python3
"""
Database Check Script - Prüft ob alle Daten korrekt in Supabase abgelegt wurden
"""
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime

# Load environment variables
load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("🔍 DATABASE CHECK - AI-Enhanced PDF Extractor")
print("=" * 50)
print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🌐 Supabase URL: {os.getenv('SUPABASE_URL')}")
print("=" * 50)

def check_table(table_name, description):
    """Prüfe eine Tabelle und zeige Statistiken"""
    try:
        # Count total entries
        result = supabase.table(table_name).select("*", count="exact").execute()
        total_count = result.count if hasattr(result, 'count') else len(result.data)
        
        print(f"\n📊 {description} ({table_name}):")
        print(f"   Total Entries: {total_count}")
        
        if result.data and len(result.data) > 0:
            # Show latest 3 entries
            recent = supabase.table(table_name).select("*").order("created_at", desc=True).limit(3).execute()
            print("   📅 Latest Entries:")
            for i, entry in enumerate(recent.data[:3], 1):
                entry_id = entry.get('id', 'N/A')
                created = entry.get('created_at', entry.get('started_at', 'N/A'))
                if created != 'N/A' and 'T' in str(created):
                    created = created.split('T')[0]  # Just date part
                
                # Table specific info
                if table_name == 'processing_logs':
                    status = entry.get('status', 'N/A')
                    filename = entry.get('original_filename', 'N/A')
                    print(f"     {i}. {filename} | Status: {status} | Date: {created}")
                elif table_name == 'service_manuals':
                    manufacturer = entry.get('manufacturer', 'N/A')
                    model = entry.get('model', 'N/A')
                    chunk_text = entry.get('content', '')[:50] + '...' if entry.get('content') else 'N/A'
                    print(f"     {i}. {manufacturer} {model} | Content: {chunk_text} | Date: {created}")
                elif table_name == 'images':
                    storage_url = entry.get('storage_url', 'N/A')
                    image_type = entry.get('image_type', 'N/A')
                    filename = storage_url.split('/')[-1] if storage_url != 'N/A' else 'N/A'
                    print(f"     {i}. {filename} | Type: {image_type} | Date: {created}")
                else:
                    print(f"     {i}. ID: {entry_id} | Date: {created}")
        else:
            print("   ⚠️  No entries found")
            
        return total_count
        
    except Exception as e:
        print(f"   ❌ Error checking {table_name}: {e}")
        return 0

# Check all major tables
tables_to_check = [
    ("processing_logs", "📋 Processing Logs"),
    ("service_manuals", "📖 Service Manual Chunks"),
    ("images", "🖼️  Extracted Images"),
    ("bulletins", "📰 Bulletins"),
    ("parts_catalog", "🔧 Parts Catalog"),
    ("cpmd_documents", "📄 CPMD Documents"),
    ("video_tutorials", "🎥 Video Tutorials"),
    ("n8n_chat_memory", "💭 Chat Memory"),
    ("parts_model_compatibility", "🔗 Parts Compatibility"),
]

total_entries = 0
for table_name, description in tables_to_check:
    count = check_table(table_name, description)
    total_entries += count

print("\n" + "=" * 50)
print(f"📈 SUMMARY:")
print(f"   Total Database Entries: {total_entries}")

# Check for recent processing activity
print(f"\n🕐 RECENT ACTIVITY CHECK:")
try:
    # Check processing logs from today
    today = datetime.now().strftime('%Y-%m-%d')
    recent_logs = supabase.table("processing_logs").select("*").gte("started_at", f"{today}T00:00:00").execute()
    
    if recent_logs.data:
        print(f"   ✅ {len(recent_logs.data)} processing jobs today")
        for log in recent_logs.data:
            filename = log.get('original_filename', 'Unknown')
            status = log.get('status', 'Unknown')
            started = log.get('started_at', '')
            if 'T' in started:
                time_part = started.split('T')[1][:8]  # HH:MM:SS
            else:
                time_part = 'Unknown'
            print(f"     • {filename} | {status} | {time_part}")
    else:
        print(f"   ⚠️  No processing activity today")

    # Check if images are being stored
    recent_images = supabase.table("images").select("*").order("created_at", desc=True).limit(5).execute()
    if recent_images.data:
        print(f"   ✅ {len(recent_images.data)} recent images in database")
        for img in recent_images.data[:3]:
            url = img.get('storage_url', '')
            filename = url.split('/')[-1] if url else 'Unknown'
            print(f"     • {filename}")
    else:
        print(f"   ⚠️  No images in database")

except Exception as e:
    print(f"   ❌ Error checking recent activity: {e}")

print("\n" + "=" * 50)
print("✅ Database check complete!")