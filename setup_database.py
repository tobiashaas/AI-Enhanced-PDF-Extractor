#!/usr/bin/env python3
"""
Supabase Database Setup
Erstellt alle benötigten Tabellen für das AI-Enhanced PDF System
"""

import json
from supabase import create_client

def setup_database():
    print("🗄️  SUPABASE DATABASE SETUP")
    print("=" * 50)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Connect to Supabase
    supabase = create_client(config['supabase_url'], config['supabase_key'])
    
    # SQL Commands
    tables_sql = [
        # Vector extension
        "CREATE EXTENSION IF NOT EXISTS vector;",
        
        # Chunks table
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(384),
            manufacturer TEXT,
            document_type TEXT,
            file_path TEXT,
            original_filename TEXT,
            file_hash TEXT,
            chunk_type TEXT,
            page_number INTEGER,
            chunk_index INTEGER,
            error_codes TEXT[],
            figure_references TEXT[],
            connection_points TEXT[],
            procedures TEXT[],
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """,
        
        # Images table
        """
        CREATE TABLE IF NOT EXISTS images (
            id BIGSERIAL PRIMARY KEY,
            file_hash TEXT NOT NULL,
            page_number INTEGER,
            image_index INTEGER,
            r2_key TEXT NOT NULL,
            r2_url TEXT,
            width INTEGER,
            height INTEGER,
            format TEXT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """,
        
        # Processing log table
        """
        CREATE TABLE IF NOT EXISTS processing_log (
            id BIGSERIAL PRIMARY KEY,
            file_path TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL,
            chunks_created INTEGER DEFAULT 0,
            images_extracted INTEGER DEFAULT 0,
            error_message TEXT,
            processing_time_seconds REAL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """,
        
        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_chunks_manufacturer ON chunks(manufacturer);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);",
        "CREATE INDEX IF NOT EXISTS idx_processing_log_status ON processing_log(status);"
    ]
    
    # Execute SQL commands
    for i, sql in enumerate(tables_sql, 1):
        try:
            print(f"📝 Erstelle Tabelle/Index {i}/{len(tables_sql)}...")
            result = supabase.rpc('exec_sql', {'sql': sql}).execute()
            print(f"✅ SQL {i} erfolgreich ausgeführt")
        except Exception as e:
            print(f"❌ Fehler bei SQL {i}: {e}")
            # Try alternative method
            try:
                # Direct SQL execution (newer Supabase versions)
                result = supabase.postgrest.rpc('exec_sql', {'sql': sql}).execute()
                print(f"✅ SQL {i} erfolgreich ausgeführt (alternative Methode)")
            except Exception as e2:
                print(f"❌ Auch alternative Methode fehlgeschlagen: {e2}")
                print("   Bitte führen Sie das SQL manuell im Supabase Dashboard aus")
    
    # Test tables
    print("\n🧪 TESTE TABELLEN...")
    try:
        # Test chunks table
        result = supabase.table("chunks").select("*").limit(1).execute()
        print("✅ Chunks Tabelle verfügbar")
        
        # Test images table
        result = supabase.table("images").select("*").limit(1).execute()
        print("✅ Images Tabelle verfügbar")
        
        # Test processing_log table
        result = supabase.table("processing_log").select("*").limit(1).execute()
        print("✅ Processing Log Tabelle verfügbar")
        
        print("\n🎉 DATENBANK SETUP ERFOLGREICH!")
        print("   Alle Tabellen sind bereit für AI-Enhanced PDF Processing")
        
    except Exception as e:
        print(f"❌ Tabellen-Test fehlgeschlagen: {e}")
        print("   Bitte prüfen Sie Ihre Supabase Konfiguration")

if __name__ == "__main__":
    setup_database()
