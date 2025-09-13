#!/usr/bin/env python3
"""
Dieses Skript löscht alle Daten in den relevanten Tabellen der Datenbank
und (optional) alle Dateien im R2-Bucket.
"""

import os
import sys
import logging
from dotenv import load_dotenv
import boto3
from supabase import create_client

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_reset.log"),
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

def clear_database_tables(supabase):
    """Löscht alle Einträge aus den relevanten Tabellen"""
    tables = [
        "service_manuals",
        "bulletins",
        "cpmd_documents",
        "parts_catalog",  # Singular laut Supabase AI
        "images",
        "processing_logs",
        "n8n_chat_memory",
        "parts_model_compatibility"
    ]
    
    print("\n=== LÖSCHEN DER DATENBANK-TABELLEN ===")
    
    for table in tables:
        try:
            result = supabase.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            deleted_count = len(result.data) if hasattr(result, 'data') and result.data else 0
            print(f"✅ Tabelle {table}: {deleted_count} Einträge gelöscht")
        except Exception as e:
            print(f"❌ Tabelle {table}: Fehler beim Löschen - {e}")
    
    print("=========================================")

def init_r2_client():
    """Initialisiert den R2-Client"""
    account_id = os.environ.get("R2_ACCOUNT_ID", "")
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    bucket_name = os.environ.get("R2_BUCKET_NAME", "")
    
    if not all([account_id, access_key, secret_key, bucket_name]):
        logger.error("R2 Zugangsdaten fehlen in der Umgebung")
        return None, None
    
    try:
        # R2-Endpunkt URL erstellen
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        # S3-Client für R2 erstellen
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto"
        )
        
        logger.info(f"R2-Client für Bucket '{bucket_name}' initialisiert")
        return s3_client, bucket_name
    except Exception as e:
        logger.error(f"Fehler beim Initialisieren des R2-Clients: {e}")
        return None, None

def clear_r2_bucket(s3_client, bucket_name):
    """Löscht alle Dateien im R2-Bucket"""
    if not s3_client or not bucket_name:
        print("❌ R2-Client oder Bucket-Name nicht verfügbar")
        return
    
    print("\n=== LÖSCHEN ALLER DATEIEN IM R2-BUCKET ===")
    
    try:
        # Paginator für große Buckets erstellen
        paginator = s3_client.get_paginator('list_objects_v2')
        total_deleted = 0
        
        # Iteration über alle Seiten von Objekten
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' not in page:
                continue
                
            # Löschoperationen für diese Seite vorbereiten
            delete_list = [{'Key': obj['Key']} for obj in page['Contents']]
            
            if not delete_list:
                continue
            
            # Löschen der Objekte dieser Seite
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': delete_list}
            )
            
            total_deleted += len(delete_list)
            print(f"  - {len(delete_list)} Dateien gelöscht (Gesamt: {total_deleted})")
        
        if total_deleted > 0:
            print(f"✅ Insgesamt {total_deleted} Dateien aus dem R2-Bucket '{bucket_name}' gelöscht")
        else:
            print(f"ℹ️ Keine Dateien im Bucket '{bucket_name}' gefunden")
            
    except Exception as e:
        print(f"❌ Fehler beim Löschen der R2-Dateien: {e}")
    
    print("=========================================")

def main():
    print("=== DATENBANK UND R2 ZURÜCKSETZEN ===")
    
    # Prüfe, ob wir im automatischen Modus sind
    force_reset = os.environ.get("FORCE_DB_RESET", "").lower() == "true"
    
    # Supabase initialisieren
    supabase = init_supabase()
    if not supabase:
        print("❌ Konnte keine Verbindung zu Supabase herstellen")
        return
    
    if not force_reset:
        # Bestätigung vom Benutzer einholen
        confirm = input("⚠️ WARNUNG: Dies wird ALLE Daten in der Datenbank löschen. Fortfahren? (j/n): ")
        if confirm.lower() != 'j':
            print("Operation abgebrochen.")
            return
    else:
        print("⚠️ Automatischer Modus: Bestätigung übersprungen")
    
    # Datenbank-Tabellen leeren
    clear_database_tables(supabase)
    
    # R2-Client initialisieren
    s3_client, bucket_name = init_r2_client()
    
    if s3_client and bucket_name:
        if not force_reset:
            # Bestätigung vom Benutzer einholen
            confirm_r2 = input("⚠️ WARNUNG: Dies wird ALLE Dateien im R2-Bucket löschen. Fortfahren? (j/n): ")
            proceed_r2 = confirm_r2.lower() == 'j'
        else:
            print("⚠️ Automatischer Modus: R2-Bestätigung übersprungen")
            proceed_r2 = True
            
        if proceed_r2:
            clear_r2_bucket(s3_client, bucket_name)
        else:
            print("R2-Bucket-Löschung übersprungen.")
    
    print("\n✅ Zurücksetzen abgeschlossen!")
    print("Sie können jetzt von Null anfangen mit der Dokumentverarbeitung.")

if __name__ == "__main__":
    main()