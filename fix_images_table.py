#!/usr/bin/env python3
"""
Dieses Skript korrigiert die images-Tabelle in der Datenbank,
indem es sie neu erstellt mit der korrekten extensions.uuid_generate_v4() Referenz.

Vor allem für Windows-Benutzer, die Probleme mit dieser Tabelle haben.
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
        logging.FileHandler("fix_images_table.log"),
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

def read_sql_file(file_path):
    """Liest eine SQL-Datei und gibt den Inhalt zurück"""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Fehler beim Lesen der SQL-Datei: {e}")
        return None

def execute_sql(supabase, sql):
    """Führt SQL-Code aus"""
    try:
        # Führe SQL direkt aus über das PostgreSQL-Interface
        # Hinweis: Dies erfordert entsprechende Berechtigungen
        result = supabase.rpc('exec_sql', {'query': sql}).execute()
        return True
    except Exception as e:
        logger.error(f"Fehler bei der SQL-Ausführung: {e}")
        return False

def main():
    print("=== KORREKTUR DER IMAGES-TABELLE ===")
    
    # Supabase initialisieren
    supabase = init_supabase()
    if not supabase:
        print("❌ Konnte keine Verbindung zu Supabase herstellen")
        return
    
    # Bestätigung vom Benutzer einholen
    confirm = input("⚠️ WARNUNG: Dies wird die images-Tabelle neu erstellen. Alle Daten in dieser Tabelle gehen verloren. Fortfahren? (j/n): ")
    if confirm.lower() != 'j':
        print("Operation abgebrochen.")
        return
    
    # SQL-Datei lesen
    sql_content = read_sql_file("fix_images_table.sql")
    if not sql_content:
        print("❌ Konnte die SQL-Datei nicht lesen")
        return
    
    print("\n=== FÜHRE SQL AUS ===")
    
    # SQL ausführen
    if execute_sql(supabase, sql_content):
        print("✅ images-Tabelle erfolgreich korrigiert")
    else:
        print("❌ Fehler bei der Korrektur der images-Tabelle")
    
    # Als Alternative, wenn execute_sql nicht funktioniert:
    print("\nAlternative Methode (manuell):")
    print("1. Öffnen Sie die Supabase SQL-Editor")
    print("2. Kopieren Sie den Inhalt der Datei 'fix_images_table.sql'")
    print("3. Führen Sie den SQL-Code aus")
    
    print("\n✅ Skript beendet")

if __name__ == "__main__":
    main()