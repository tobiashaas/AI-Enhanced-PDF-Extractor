#!/usr/bin/env python3
"""
Detailed Database Analysis
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("🔍 DETAILED DATABASE ANALYSIS")
print("=" * 50)

# 1. Check processing logs in detail
print("\n📋 PROCESSING LOGS DETAILS:")
try:
    logs = supabase.table("processing_logs").select("*").order("started_at", desc=True).limit(5).execute()
    for log in logs.data:
        print(f"   📄 {log.get('original_filename', 'N/A')}")
        print(f"      Status: {log.get('status', 'N/A')}")
        print(f"      Started: {log.get('started_at', 'N/A')}")
        print(f"      Stage: {log.get('processing_stage', 'N/A')}")
        print(f"      Progress: {log.get('progress_percentage', 'N/A')}%")
        print(f"      Chunks Created: {log.get('chunks_created', 'N/A')}")
        print(f"      Images Extracted: {log.get('images_extracted', 'N/A')}")
        print(f"      Error: {log.get('error_message', 'None')}")
        print()
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Check images in detail
print("\n🖼️ IMAGES DETAILS:")
try:
    images = supabase.table("images").select("*").order("created_at", desc=True).limit(3).execute()
    for img in images.data:
        print(f"   🖼️ Image ID: {img.get('id', 'N/A')}")
        print(f"      URL: {img.get('storage_url', 'N/A')}")
        print(f"      Type: {img.get('image_type', 'N/A')}")
        print(f"      Hash: {img.get('hash', 'N/A')}")
        print(f"      Source Table: {img.get('source_table', 'N/A')}")
        print(f"      Source ID: {img.get('source_id', 'N/A')}")
        print(f"      Created: {img.get('created_at', 'N/A')}")
        metadata = img.get('metadata', {})
        if metadata:
            print(f"      Metadata: {metadata}")
        print()
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Check what tables actually exist
print("\n📊 CHECKING ACTUAL TABLE STRUCTURE:")
tables_to_try = [
    "service_manuals",
    "bulletins", 
    "parts_catalog",  # Singular version
    "parts_catalogs", # Plural version
    "cpmd_documents",
    "video_tutorials"
]

for table in tables_to_try:
    try:
        result = supabase.table(table).select("*").limit(1).execute()
        print(f"   ✅ {table}: EXISTS ({len(result.data)} sample entries)")
        if result.data:
            print(f"      Sample columns: {list(result.data[0].keys())}")
    except Exception as e:
        print(f"   ❌ {table}: {e}")

# 4. Check if service manual processing is actually happening
print("\n🔍 SERVICE MANUAL PROCESSING CHECK:")
try:
    # Check if there are any entries in service_manuals table
    result = supabase.table("service_manuals").select("*").execute()
    print(f"   Service manual entries: {len(result.data)}")
    
    if len(result.data) > 0:
        print("   ✅ Service manuals are being processed!")
        for entry in result.data[:3]:
            print(f"      • {entry.get('manufacturer', 'N/A')} {entry.get('model', 'N/A')}")
            print(f"        Content: {entry.get('content', '')[:100]}...")
    else:
        print("   ⚠️  No service manual entries found")
        print("   💡 This might indicate processing is still ongoing or failed")

except Exception as e:
    print(f"   ❌ Error checking service_manuals: {e}")

print("\n" + "=" * 50)
print("🎯 ANALYSIS COMPLETE")