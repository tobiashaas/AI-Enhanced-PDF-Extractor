# 🔧 SUPABASE SCHEMA UPDATE ANLEITUNG

## Problem
Der AI PDF Processor kann nicht in die Datenbank schreiben, weil Spalten in der `chunks` Tabelle fehlen.

**Fehlermeldungen:**
- `Could not find the 'connection_points' column`
- `Could not find the 'document_priority' column`
- `null value in column "chunk_index" violates not-null constraint`

## ✅ Lösung

### Schritt 1: Supabase SQL Editor öffnen
1. Gehen Sie zu: https://supabase.com/dashboard/project/xvqsvrxyjjunbsdudfly/sql
2. Klicken Sie auf "New Query"

### Schritt 2: SQL Schema Update ausführen
1. Kopieren Sie den Inhalt von `supabase_schema_update.sql`
2. Fügen Sie ihn in den SQL Editor ein
3. Klicken Sie auf "Run"

### Schritt 3: Erfolg überprüfen
Das Script sollte erfolgreich ausgeführt werden und zeigt am Ende die neue Tabellenstruktur.

## 📋 Was wird hinzugefügt

### Neue Spalten:
- `connection_points` - TEXT[] für Verbindungspunkte
- `document_priority` - TEXT für Dokumentpriorität  
- `document_subtype` - TEXT für Dokumentuntertyp
- `document_source` - TEXT für Dokumentquelle
- `chunk_index` - INTEGER für Chunk-Index (nullable)
- `figure_references` - TEXT[] für Abbildungsreferenzen
- `procedures` - TEXT[] für Verfahrensschritte
- `error_codes` - TEXT[] für Fehlercodes

### Performance Indizes:
- `idx_chunks_manufacturer_model` - Schnelle Hersteller/Modell-Suche
- `idx_chunks_document_type` - Dokumenttyp-Filter
- `idx_chunks_page_number` - Seitennummern-Index
- `idx_chunks_file_hash` - Datei-Hash-Index

## 🧪 Nach dem Update testen

```bash
python3 simple_pdf_test.py  # Basic funktioniert bereits
python3 launch.py process "Documents/Parts_Catalogs/Konica_Minolta/C451i/C451i_Parts.pdf"  # Sollte jetzt funktionieren
```

## 🚀 Dann Batch Processing starten

```bash
python3 batch_processor.py  # Alle 15 PDFs mit voller AI Power
```