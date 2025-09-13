# 🧠 MEMORY - PROJECT MASTER PLAN
**AI-Enhanced PDF Processing System - Refactor Branch**
*Erstellt: 13. September 2025*

## 📋 **PROJEKT-ÜBERBLICK**

### 🎯 **MISSION:**
Modulares AI-Enhanced PDF Processing System für Service Manuals, Parts Catalogs und technische Dokumentation mit semantischer Suche.

### 🏗️ **ARCHITEKTUR-PRINZIPIEN:**
1. **Modular Design** - Jede Funktionalität als separates Modul
2. **Clean Code** - Klare Trennung von Verantwortlichkeiten  
3. **Wartbarkeit** - Jedes Modul einzeln testbar und updatebar
4. **Skalierbarkeit** - Neue Features durch Module hinzufügbar
5. **Erweiterbarkeit** - Neue Dokumenttypen/Hersteller einfach integrierbar

### 🗄️ **DATENBANK-ARCHITEKTUR:**
- **10 Haupttabellen** mit UUID Primary Keys
- **RLS Security** - Service Key Only Access
- **5 Vector Embeddings** für semantische Suche
- **Document-Type Separation** für optimierte Suchen
- **Multi-Manufacturer Support** mit Parts-Kompatibilität
- **Version Control** - Dokument-Versionen tracken
- **Model Mapping** - Speichert kompatible Modelle als Arrays
- **Strukturierte Kategorien** - Mapped auf `/Documents` Ordner

---

## 🔧 **TECHNOLOGIE-STACK**

### 🤖 **AI/ML:**
- **Embedding Model:** EmbeddingGemma (Google, 768-dimensional)
- **Ollama Version:** >= 0.11.10 
- **API:** `/api/embed` (neue API, nicht legacy)
- **Vector Database:** Supabase pgvector

### 🗄️ **Database:**
- **Platform:** Supabase PostgreSQL
- **Vector Dimensions:** 768 (für EmbeddingGemma)
- **Security:** Row Level Security (RLS) aktiviert
- **Access:** Service Key Only - keine öffentlichen Policies
- **Version Control:** Versionsnummern aus Dokumenten extrahieren
- **Model Arrays:** Kompatible Modelle als ARRAY-Typ speichern
- **Categories:** Nutzt vorhandene `/Documents` Struktur:
  - 📄 Bulletins (pro Hersteller)
  - 📱 CPMD (HP only)
  - 🔧 Parts Catalogs (pro Hersteller)
  - 📚 Service Manuals (pro Hersteller)

### 🖼️ **Image Processing:**
- **✅ ZERO Conversion Policy:**
  - Alle Formate bleiben original (SVG, PNG, JPG, BMP, etc.)
  - Keine Konvertierung zwischen Formaten
  - Keine "Optimierung" der Originale
  - PDFs behalten ihre Original-Bilder

- **✅ Universal Format Support:**
  - Vektorgrafiken (SVG, AI, EPS, PDF)
  - Rasterbilder (PNG, JPG, BMP, TIFF)
  - Technische Zeichnungen in Originalformat
  - Fotos in Originalformat

- **✅ Storage Strategie:**
  - R2/Cloud Storage mit exakten Originalen
  - Byte-für-Byte identische Kopien
  - Content-Type aus Original übernehmen
  - Hash-Validierung für Integrität

---

## 📊 **HAUPT-TABELLEN (10)**

### 📄 **Document Processing Tables:**
1. **`service_manuals`** - Reparatur-/Wartungsanleitungen (chunked)
   ```sql
   -- Zusätzliche Felder:
   file_hash text NOT NULL,         -- PDF Quell-Hash
   original_filename text NOT NULL,  -- Originaler Dateiname
   file_size bigint NOT NULL,       -- Dateigröße in Bytes
   total_pages integer NOT NULL,    -- Seitenanzahl
   chunking_strategy text NOT NULL  -- intelligent/semantic/fixed
   ```

2. **`bulletins`** - Technische Mitteilungen und Updates
   ```sql
   -- Zusätzliche Felder:
   bulletin_type text NOT NULL,     -- security/maintenance/update
   severity level text,             -- high/medium/low
   release_date timestamp           -- Veröffentlichungsdatum
   ```

3. **`parts_catalogs`** - Ersatzteil-Dokumentation (chunked)
   ```sql
   -- Zusätzliche Felder:
   has_csv boolean DEFAULT false,   -- CSV Pairing Check
   price_currency text,             -- EUR/USD etc.
   last_price_update timestamp,     -- Letztes Preisupdate
   parts_count integer             -- Anzahl Teile im Katalog
   ```

4. **`cpmd_documents`** - HP Control Panel Messages
   ```sql
   -- Zusätzliche Felder:
   firmware_version text[],         -- Kompatible Firmware
   error_code_range text,          -- z.B. "C4000-C4999"
   control_panel_type text         -- Touch/LCD/LED
   ```

5. **`video_tutorials`** - Video-Anleitungen
   ```sql
   -- Zusätzliche Felder:
   video_duration interval,         -- Länge des Videos
   resolution text,                -- z.B. "1920x1080"
   has_audio boolean,              -- Audio vorhanden?
   language text[]                 -- Verfügbare Sprachen
   ```

### 🔧 **Support Tables:**
6. **`images`** - Bild-Metadaten mit Vision AI (alle Dokumenttypen)
7. **`parts_catalog`** - Master Parts Database (dedupliziert)
8. **`parts_model_compatibility`** - Parts-Model Mapping (Many-to-Many)

### 💬 **System Tables:**
9. **`n8n_chat_memory`** - Chat Memory für Context-Aware AI
10. **`processing_logs`** - Vollständige Verarbeitungsprotokolle

---

## 🎯 **WICHTIGE REGELN**

### 🏗️ **Architektur:**
- Modular bauen - jede Funktionalität als Modul
- Clean Code Prinzipien einhalten
- Neue Features durch Module hinzufügbar
- Jedes Modul einzeln testbar

### 🖼️ **Image Processing:**
- **NIEMALS** Bilder konvertieren - Original beibehalten
- Vektorgrafiken (SVG, AI, EPS) separat extrahieren
- Beide Formate parallel unterstützen
- `image_type`: "diagram", "photo", "vector", "flowchart", "table"

### 🗄️ **Database:**
- Alle Primary Keys sind UUIDs
- Part Numbers nur per Hersteller unique (nicht global!)
- Embeddings immer 768-dimensional für EmbeddingGemma
- RLS auf allen Tabellen - nur Service Key Access
- JSONB für flexible Metadaten
- Dokument-Versionen tracken:
  ```sql
  version_info: {
    version: string,          -- "1.2.3"
    release_date: timestamp,  -- wenn verfügbar
    revision: string         -- "Rev A" etc.
  }
  ```
- Modell-Kompatibilität als Arrays:
  ```sql
  compatible_models: string[], -- ["E55040", "E57540"]
  model_series: string[],     -- ["E-Series", "PageWide"]
  ```
- Kategorien aus Ordnerstruktur:
  ```sql
  document_category: enum,    -- Von /Documents Struktur
  manufacturer: enum,         -- Aus Unterordnern
  subcategory: string        -- Zusätzliche Gruppierung
  ```

### 🔌 **Embedding & Processing:**
- Ollama >= 0.11.10 für EmbeddingGemma
- Neue `/api/embed` API verwenden (nicht legacy)
- Model: `embeddinggemma` (Google, 768-dim)
- Batch Processing für Performance
- **Chunking Strategy:**
  ```json
  {
    "chunking_strategy": "intelligent",
    "max_chunk_size": 600,
    "min_chunk_size": 200,
    "use_semantic_boundaries": true
  }
  ```
- **Vision Analysis:**
  ```json
  {
    "vision_model": "llava:7b",
    "use_vision_analysis": true,
    "render_dpi": 144,
    "export_vectors": true,
    "vector_format": "svg",
    "raster_format": "png"
  }
  ```
- **Performance:**
  ```json
  {
    "use_metal_acceleration": true,
    "parallel_workers": 8,
    "batch_size": 150,
    "memory_optimization": "unified_memory"
  }
  ```

---

## 🚀 **USE CASES**

### 📋 **Typische Anfragen:**
1. **"HP E55040 Toner wechseln"** → `service_manuals` + `video_tutorials`
2. **"Control Panel Fehlercode C4051"** → `cpmd_documents` + `service_manuals`  
3. **"Welche Teile für C3350i Scanner?"** → `parts_catalogs` + `bulletins`
4. **Chat Memory:** Context aus vorherigen Gesprächen → `n8n_chat_memory`

### 🔍 **Semantic Search Beispiel:**
```sql
-- Query: "Scanner Problem HP"
-- Findet auch: "Scanner jam troubleshooting", "Scanner error messages"
SELECT content FROM service_manuals 
WHERE manufacturer = 'HP' 
ORDER BY embedding <-> $1::vector 
LIMIT 10;
```

---

## 📁 **MODUL-STRUKTUR**

```
Core Modules:
├── Document Processing
│   ├── Service Manuals Handler
│   ├── Bulletins Handler  
│   ├── Parts Catalogs Handler
│   ├── CPMD Documents Handler
│   └── Video Tutorials Handler
├── Image Processing
│   ├── Vision AI Analyzer
│   ├── Vector Graphics Extractor
│   └── Original Format Preserver
├── Parts Management
│   ├── Parts Master Database
│   └── Model Compatibility Manager
├── Chat Memory
│   └── n8n Integration Module
└── Processing Pipeline
    └── Logs & Monitoring
```

---

## ⚠️ **CRITICAL SUCCESS FACTORS**

### 🎯 **Performance:**
- Vector Search mit HNSW Indizes optimiert
- Batch Processing für Embeddings
- Document-type spezifische Suchen
- 25+ Performance-Indizes

### 🔒 **Security:**
- RLS auf allen 10 Tabellen aktiviert
- Nur Service Key Access - keine anonymen Policies
- Sichere API-Endpunkte

### 🏗️ **Modularity:**
- Jedes Feature als separates Modul
- Klare Interfaces zwischen Modulen
- Einfache Erweiterbarkeit für neue Dokumenttypen
- Wartbare Code-Architektur

### 📊 **Data Quality:**
- Original-Formate bewahren (Images)
- Metadaten in JSONB für Flexibilität
- Document Versioning Support
- Multi-Manufacturer Compatibility

---

## 📚 **REFERENZ-DOKUMENTE**

1. **DATABASE_STRUCTURE_GUIDE.md** - Komplette DB-Architektur
2. **TECHNICAL_REQUIREMENTS.md** - Ollama & API Setup
3. **SUPABASE_AI_AGENT_SCHEMA_REQUEST.md** - Schema-Spezifikation
4. **SQL_Befehl.md** - Production-ready SQL Schema

---

**🎯 MERKE: Immer modular denken, Originale bewahren, EmbeddingGemma nutzen, RLS Security!**