#!/usr/bin/env python3
"""
Skript zum Erstellen der document_chunks Tabelle in der Datenbank
"""

import os
import sys
import logging
from dotenv import load_dotenv
from supabase import create_client

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger initialisieren
logger = logging.getLogger(__name__)

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

def init_supabase():
    """Initialisiert die Supabase-Verbindung"""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        logger.error("SUPABASE_URL oder SUPABASE_KEY fehlen in der Umgebung")
        return None
    
    try:
        client = create_client(url, key)
        logger.info("Supabase-Verbindung hergestellt")
        return client
    except Exception as e:
        logger.error(f"Fehler bei der Supabase-Verbindung: {e}")
        return None

def create_document_chunks_table():
    """Erstellt die document_chunks Tabelle in der Datenbank"""
    supabase = init_supabase()
    if not supabase:
        logger.error("Keine Verbindung zur Datenbank")
        return False
    
    try:
        # SQL-Befehl zum Erstellen der Tabelle
        sql = """
        CREATE TABLE IF NOT EXISTS public.document_chunks (
          id uuid PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
          document_id uuid NOT NULL,
          chunk_index integer NOT NULL,
          content text NOT NULL,
          embedding extensions.vector(768),
          metadata jsonb,
          created_at timestamp with time zone DEFAULT now()
        );
        
        -- Index für document_id
        CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON public.document_chunks (document_id);
        
        -- Index für embedding (HNSW)
        CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON public.document_chunks USING hnsw (embedding extensions.vector_cosine_ops);
        
        -- RLS aktivieren
        ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;
        
        -- Policy für Service Key Only
        CREATE POLICY "service_key_only_document_chunks" 
        ON public.document_chunks FOR ALL TO public 
        USING (auth.role() = 'service_role');
        """
        
        # SQL-Befehl ausführen
        result = supabase.rpc("custom_sql", {"query_text": sql}).execute()
        
        logger.info("Document_chunks Tabelle wurde erstellt")
        print("✅ Document_chunks Tabelle wurde erfolgreich erstellt")
        
        return True
    except Exception as e:
        logger.error(f"Fehler beim Erstellen der document_chunks Tabelle: {e}")
        print(f"❌ Fehler beim Erstellen der document_chunks Tabelle: {e}")
        return False

if __name__ == "__main__":
    print("=== DOCUMENT_CHUNKS TABELLE SETUP ===")
    if create_document_chunks_table():
        print("✅ Setup erfolgreich abgeschlossen")
    else:
        print("❌ Setup fehlgeschlagen")
    print("====================================")