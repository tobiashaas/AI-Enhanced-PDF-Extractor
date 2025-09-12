# 🤖 SUPABASE AI AGENT - DATABASE SCHEMA REQUEST

## 📋 **KONTEXT**
Ich brauche ein **komplettes PostgreSQL Schema** für ein **AI-Enhanced PDF Processing System** das Service Manuals, Teile-Kataloge und technische Dokumentation verarbeitet.

## 🎯 **HAUPTFUNKTIONEN**
1. **PDF Chunking** - Intelligente Text-Segmentierung mit AI
2. **Vision AI** - Extraktion von Bildern und Diagrammen
3. **Parts Catalog** - Ersatzteil-Verwaltung mit Deduplizierung
4. **Smart Search** - Vector-basierte semantische Suche
5. **Processing Logs** - Vollständige Verarbeitungsprotokollierung

## 📊 **BENÖTIGTE TABELLEN**

### 1. **`chunks`** - Haupttabelle für AI-verarbeitete Textblöcke
**Zweck:** Speichert intelligente Text-Segmente mit AI-Metadaten
```
- id (Primary Key, UUID)
- content (TEXT) - Der Textinhalt
- file_hash (TEXT) - Eindeutige Datei-Identifikation
- page_number (INTEGER) - Seitennummer
- chunk_index (INTEGER) - Position im Dokument
- manufacturer (TEXT) - Hersteller (HP, Canon, etc.)
- model (TEXT) - Gerätemodell
- document_type (TEXT) - "Service Manual", "Parts Catalog", etc.
- problem_type (TEXT) - "scanner_issue", "paper_jam", etc.
- procedure_type (TEXT) - "troubleshooting", "maintenance", etc.
- embedding (VECTOR) - 768-dimensional für Supabase Vector Search
- metadata (JSONB) - Flexible Zusatzdaten
- created_at (TIMESTAMP)

// AI-Enhanced Felder:
- connection_points (TEXT[]) - Erkannte Verbindungsstellen
- document_priority (TEXT) - "urgent", "normal", "reference"
- document_subtype (TEXT) - Detailtyp
- document_source (TEXT) - Quelle der Information
- figure_references (TEXT[]) - Referenzierte Abbildungen
- procedures (TEXT[]) - Erkannte Verfahrensschritte
- error_codes (TEXT[]) - Erkannte Fehlercodes
```

### 2. **`images`** - Bild-Metadaten aus PDFs
**Zweck:** Verwaltet extrahierte Bilder mit Vision AI Analyse
```
- id (Primary Key)
- file_hash (TEXT) - Verknüpfung zur Quelldatei
- page_number (INTEGER)
- image_index (INTEGER) - Position auf der Seite
- storage_url (TEXT) - R2/Cloud Storage URL
- image_type (TEXT) - "diagram", "photo", "table", etc.
- description (TEXT) - AI-generierte Beschreibung
- manufacturer (TEXT)
- model (TEXT)
- hash (TEXT) - Bild-Hash für Deduplizierung
- metadata (JSONB)
- created_at (TIMESTAMP)

// Vision AI Felder:
- document_source (TEXT) - Quelle
- vision_analysis (JSONB) - Vollständige Vision AI Ergebnisse
```

### 3. **`parts_catalog`** - Ersatzteil-Verwaltung
**Zweck:** Deduplizierte Ersatzteil-Datenbank
```
- id (Primary Key)
- part_number (TEXT UNIQUE) - Eindeutige Teilenummer
- part_name (TEXT)
- manufacturer (TEXT)
- models_compatible (TEXT[]) - Kompatible Gerätemodelle
- category (TEXT) - Teilekategorie
- description (TEXT)
- metadata (JSONB)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
```

### 4. **`processing_logs`** - Verarbeitungsprotokoll
**Zweck:** Vollständige Nachverfolgung aller PDF-Verarbeitungen
```
- id (Primary Key)
- file_path (TEXT)
- file_hash (TEXT UNIQUE)
- original_filename (TEXT)
- status (TEXT) - "processing", "completed", "failed"
- processing_stage (TEXT) - Aktuelle Phase
- progress_percentage (INTEGER)
- chunks_created (INTEGER)
- images_extracted (INTEGER)
- manufacturer (TEXT)
- model (TEXT)
- document_type (TEXT)
- document_info (JSONB)
- document_title (TEXT)
- document_version (TEXT)
- error_message (TEXT)
- started_at (TIMESTAMP)
- completed_at (TIMESTAMP)
- processing_time_seconds (INTEGER)
- retry_count (INTEGER DEFAULT 0)
- updated_at (TIMESTAMP)
```

## ⚡ **PERFORMANCE-ANFORDERUNGEN**

### **Indizes für hohe Performance:**
```sql
-- Chunks Tabelle (Hauptsuche)
CREATE INDEX idx_chunks_manufacturer_model ON chunks(manufacturer, model);
CREATE INDEX idx_chunks_document_type ON chunks(document_type);
CREATE INDEX idx_chunks_page_number ON chunks(page_number);
CREATE INDEX idx_chunks_file_hash ON chunks(file_hash);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- Images Tabelle
CREATE INDEX idx_images_file_hash ON images(file_hash);
CREATE INDEX idx_images_manufacturer_model ON images(manufacturer, model);

-- Parts Catalog
CREATE INDEX idx_parts_part_number ON parts_catalog(part_number);
CREATE INDEX idx_parts_manufacturer ON parts_catalog(manufacturer);

-- Processing Logs
CREATE INDEX idx_processing_logs_status ON processing_logs(status);
CREATE INDEX idx_processing_logs_file_hash ON processing_logs(file_hash);
```

## 🎯 **SPEZIELLE ANFORDERUNGEN**

1. **Vector Search Support:** chunks.embedding als 768-dimensional vector für Supabase Vector
2. **Flexible Metadaten:** JSONB Felder für erweiterte AI-Daten
3. **Deduplizierung:** parts_catalog.part_number als UNIQUE
4. **Performance:** Optimiert für 10,000+ chunks mit schnellen Suchen
5. **Monitoring:** processing_logs für vollständige Nachverfolgung

## 🔧 **ERWARTETES ERGEBNIS**
**Komplettes SQL-Schema** mit:
- ✅ Alle Tabellen mit korrekten Datentypen
- ✅ Performance-Indizes
- ✅ Foreign Key Constraints wo sinnvoll  
- ✅ Supabase Vector Support
- ✅ Produktionsreif für AI PDF Processing

**Ziel:** Schema das sofort in Supabase SQL Editor ausgeführt werden kann für ein funktionierendes AI PDF Processing System.