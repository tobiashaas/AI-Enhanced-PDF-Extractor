# 🗄️ DATABASE STRUCTURE GUIDE
**AI-Enhanced PDF Processing System - Refactor Branch**
*Letzte Aktualisierung: 13. September 2025*

## 📋 **ÜBERBLICK**
- **Total Tabellen:** 10 Haupttabellen
- **Primary Keys:** Alle UUID-basiert 
- **Secu### 🔍 **WICHTIGE INDIZES**

### Vector Search (HNSW):
```sql
-- Mit Extensions Schema
service_manuals.embedding USING hnsw (emb### 🗄️ **Database Regeln:**
13. **Part Numbers:** Nicht global unique! Nur per Hersteller unique
14. **UUIDs:** Alle Primary Keys sind UUIDs (extensions.uuid_generate_v4())
15. **Embeddings:** Immer 768-dimensional für EmbeddingGemma
16. **Extensions Schema:** Extensions in separatem `extensions` Schema
17. **RLS:** Service Key Only - keine öffentlichen Policies
18. **Arrays:** TEXT[] für Multi-Value Felder (models_compatible, etc.)
19. **JSONB:** Für flexible Metadaten und Vision AI Ergebnisse
20. **Timestamps:** Immer mit timezone (timestamp with time zone)xtensions.vector_cosine_ops)
bulletins.embedding USING hnsw (embedding extensions.vector_cosine_ops)
parts_catalogs.embedding USING hnsw (embedding extensions.vector_cosine_ops) 
cpmd_documents.embedding USING hnsw (embedding extensions.vector_cosine_ops)
video_tutorials.embedding USING hnsw (embedding extensions.vector_cosine_ops)
```S auf allen Tabellen (Service Key Only)
- **Vector Search:** 5 Embedding-Spalten f### 🔌 **E### 🗄️ **Database Regeln:### 🔌 **Ollama Setup:**
- [ ] Ollama Version >= 0.11.10 installiert
- [ ] embeddinggemma Model heruntergeladen
- [ ] Neue `/api/embed` API Endpoint getestet
- [ ] Batch Processing für Embedding Performance optimiert. **Part Numbers:** Nicht global unique! Nur per Hersteller unique
14. **UUIDs:** Alle Primary Keys sind UUIDs (uuid_generate_v4())
15. **Embeddings:** Immer 768-dimensional für EmbeddingGemma
16. **RLS:** Service Key Only - keine öffentlichen Policies
17. **Arrays:** TEXT[] für Multi-Value Felder (models_compatible, etc.)
18. **JSONB:** Für flexible Metadaten und Vision AI Ergebnisse
19. **Timestamps:** Immer mit timezone (timestamp with time zone) Regeln:**
9. **Ollama Version:** Minimum 0.11.10 für EmbeddingGemma support
10. **API Endpoint:** Neue `/api/embed` API verwenden (nicht legacy)
11. **Model:** embeddinggemma für 768-dimensional embeddings
12. **Batch Processing:** Chunks in Batches für Performancemantische Suche
- **Performance:** 25+ optimierte Indizes
- **Embeddings:** EmbeddingGemma via Ollama (768-dimensional)

## 🔧 **SYSTEM REQUIREMENTS**

### 📦 **Ollama Requirements:**
- **Version:** Minimum Ollama 0.11.10 (für EmbeddingGemma support)
- **Model:** embeddinggemma (768-dimensional embeddings)
- **API:** Neue Embedding API `/api/embed` verwenden (nicht legacy)
- **Performance:** Optimiert für batch processing von PDF chunks

### 🗃️ **Database Extensions Requirements:**
- **Schema:** Extensions in separatem `extensions` Schema (nicht `public`)
- **Extensions:** uuid-ossp und vector in `extensions` Schema 
- **Search Path:** `public, extensions` für alle Roles
- **Query:** `SELECT e.extname AS extension, n.nspname AS schema FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'vector';`

### 🔌 **API Integration:**
```bash
# Neue Ollama Embedding API
POST http://localhost:11434/api/embed
{
  "model": "embeddinggemma",
  "input": "Text to embed..."
}
```

## 🏛️ **ARCHITEKTUR-PRINZIPIEN**

### 🧩 **MODULAR DESIGN**
- **Wartbarkeit:** Jedes Modul ist eigenständig und testbar
- **Skalierbarkeit:** Neue Features durch Module hinzufügbar
- **Clean Code:** Klare Trennung von Verantwortlichkeiten
- **Erweiterbarkeit:** Neue Dokumenttypen/Hersteller einfach integrierbar

### 📐 **MODUL-STRUKTUR**
```
Core Modules:
├── Document Processing (service_manuals, bulletins, parts_catalogs, cpmd_documents, video_tutorials)
├── Image Processing (images + vision analysis)
├── Parts Management (parts_catalog, parts_model_compatibility)
├── Chat Memory (n8n_chat_memory)
└── Processing Pipeline (processing_logs)
```

### 🖼️ **IMAGE PROCESSING REQUIREMENTS**
- **❌ KEINE Konvertierung:** Originale bleiben unverändert
- **✅ Vektorgrafiken:** SVG, AI, EPS Extraktion erforderlich
- **✅ Raster + Vektor:** Beide Formate parallel unterstützen
- **✅ Metadaten:** Typ-spezifische Behandlung (diagram, photo, vector, etc.)

---

## 🏗️ **HAUPT-TABELLEN**

### 1. **`service_manuals`** - Service Manual Chunks
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: keine
```
**Zweck:** Reparatur- und Wartungsanleitungen (chunked)
**Key Felder:**
- `content` (TEXT) - Textinhalt
- `manufacturer` (TEXT) - HP, Canon, etc.
- `model` (TEXT) - Gerätemodell
- `embedding` (VECTOR 768) - Für semantische Suche
- `procedure_type` - "troubleshooting", "maintenance", "repair"
- `problem_type` - "scanner_issue", "paper_jam", etc.

### 2. **`bulletins`** - Technical Bulletins
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: keine
```
**Zweck:** Technische Mitteilungen und Updates
**Key Felder:**
- `content` (TEXT) - Textinhalt
- `manufacturer` (TEXT)
- `models_affected` (TEXT[]) - Array betroffener Modelle
- `embedding` (VECTOR 768)
- `bulletin_type` - "urgent", "advisory", "recall"
- `priority_level` - "critical", "high", "medium", "low"

### 3. **`parts_catalogs`** - Parts Catalog Chunks
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: keine
```
**Zweck:** Ersatzteil-Dokumentation (chunked)
**Key Felder:**
- `content` (TEXT) - Textinhalt
- `manufacturer` (TEXT)
- `model` (TEXT)
- `embedding` (VECTOR 768)
- `part_category` - "toner", "drum", "fuser", etc.
- `part_numbers_mentioned` (TEXT[]) - Erkannte Teilenummern

### 4. **`cpmd_documents`** - HP Control Panel Messages
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: keine
```
**Zweck:** HP-spezifische Control Panel Nachrichten
**Key Felder:**
- `content` (TEXT) - Textinhalt
- `manufacturer` (TEXT) - DEFAULT 'HP'
- `model` (TEXT) - HP Modell
- `embedding` (VECTOR 768)
- `message_code` (TEXT) - Control Panel Code
- `message_type` - "error", "warning", "info"

### 5. **`video_tutorials`** - Video Tutorial Information
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: keine
```
**Zweck:** Video-Anleitungen für Reparaturen
**Key Felder:**
- `content` (TEXT) - Transkript
- `video_url` (TEXT) - URL zum Video
- `manufacturer` (TEXT)
- `model` (TEXT)
- `embedding` (VECTOR 768)
- `tutorial_type` - "repair", "maintenance", etc.
- `duration_minutes` (INTEGER)

---

## 🖼️ **SUPPORT-TABELLEN**

### 6. **`images`** - Bild-Metadaten
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: keine (soft reference über source_table + source_id)
```
**Zweck:** Verwaltet extrahierte Bilder mit Vision AI (KEINE Konvertierung!)
**Key Felder:**
- `source_table` (TEXT) - Referenz zu Quelltabelle
- `source_id` (UUID) - ID in der Quelltabelle
- `storage_url` (TEXT) - R2/Cloud Storage URL (Original Format!)
- `image_type` (TEXT) - "diagram", "photo", "vector", "flowchart", "table"
- `vision_analysis` (JSONB) - AI-Analyse Ergebnisse

**⚠️ WICHTIG:**
- Originale NICHT konvertieren (PNG, JPG, SVG, AI, EPS bleiben original)
- Vektorgrafiken (SVG, AI, EPS) separat extrahieren und speichern
- Beide Formate parallel unterstützen für optimale Qualität

### 7. **`parts_catalog`** - Master Parts Database
```sql
PRIMARY KEY: id (UUID)
UNIQUE: (manufacturer, part_number)
FOREIGN KEYS: keine
```
**Zweck:** Deduplizierte Ersatzteil-Datenbank
**Key Felder:**
- `part_number` (TEXT) - Teilenummer (nicht unique global!)
- `manufacturer` (TEXT) - Hersteller
- `models_compatible` (TEXT[]) - Array kompatibler Modelle
- `category` (TEXT) - Teilekategorie

### 8. **`parts_model_compatibility`** - Parts-Model Mapping
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: part_id → parts_catalog(id) ON DELETE CASCADE
```
**Zweck:** Many-to-Many zwischen Parts und Models
**Key Felder:**
- `part_id` (UUID) - FK zu parts_catalog
- `model` (TEXT) - Gerätemodell
- `manufacturer` (TEXT)
- `compatibility_confirmed` (BOOLEAN)

---

## 💬 **SYSTEM-TABELLEN**

### 9. **`n8n_chat_memory`** - Chat Memory für n8n
```sql
PRIMARY KEY: id (UUID)
UNIQUE: keine
FOREIGN KEYS: keine
```
**Zweck:** Einfacher Chat-Verlauf für Context-Aware AI
**Key Felder:**
- `session_id` (TEXT) - Eindeutige Session
- `message_type` - "user", "assistant", "system"
- `message_content` (TEXT)
- `manufacturer_context` (TEXT) - Aktueller Hersteller
- `model_context` (TEXT) - Aktuelles Modell
- `workflow_execution_id` (TEXT) - n8n Workflow ID

### 10. **`processing_logs`** - Verarbeitungsprotokoll
```sql
PRIMARY KEY: id (UUID)
UNIQUE: file_hash
FOREIGN KEYS: keine
```
**Zweck:** Vollständige Nachverfolgung aller PDF-Verarbeitungen
**Key Felder:**
- `file_hash` (TEXT UNIQUE) - Eindeutige Datei-ID
- `status` - "processing", "completed", "failed"
- `chunks_created` (INTEGER)
- `images_extracted` (INTEGER)
- `document_type` (TEXT)

---

## 🔍 **WICHTIGE INDIZES**

### Vector Search (HNSW):
```sql
service_manuals.embedding
bulletins.embedding  
parts_catalogs.embedding
cpmd_documents.embedding
video_tutorials.embedding
```

### Performance Indizes:
```sql
-- Manufacturer/Model Suchen
idx_service_manuals_manufacturer_model
idx_parts_catalogs_manufacturer_model
idx_video_tutorials_manufacturer_model

-- Parts System
idx_parts_manufacturer_part_number (manufacturer, part_number)
idx_parts_compatibility_part_id

-- Chat Memory
idx_n8n_chat_session_id
idx_n8n_chat_manufacturer_model

-- File Processing
idx_processing_logs_file_hash
idx_*_file_hash (alle document tables)
```

---

## 🔐 **SECURITY (RLS)**

**Alle Tabellen haben RLS aktiviert:**
```sql
ALTER TABLE <table_name> ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_<table_name>" 
ON <table_name> FOR ALL TO public 
USING (auth.role() = 'service_role');
```

**Zugriff nur über Service Key - Keine anonymen/authenticated Policies**

---

## 🎯 **VERBINDUNGEN & BEZIEHUNGEN**

### Document → Images (Soft Reference):
```
images.source_table = 'service_manuals' 
images.source_id = service_manuals.id
```

### Parts System (Hard FK):
```
parts_model_compatibility.part_id → parts_catalog.id (CASCADE DELETE)
```

### File Processing:
```
*.file_hash → processing_logs.file_hash (via UNIQUE constraint)
```

---

## 🚀 **USE CASES & QUERIES**

### 1. Semantic Search in Service Manuals:
```sql
SELECT * FROM service_manuals 
WHERE manufacturer = 'HP' AND model = 'E55040'
ORDER BY embedding <-> $1 LIMIT 10;
```

### 2. Find Compatible Parts:
```sql
SELECT pc.* FROM parts_catalog pc
JOIN parts_model_compatibility pmc ON pc.id = pmc.part_id
WHERE pmc.model = 'C3350i' AND pmc.manufacturer = 'Konica_Minolta';
```

### 3. Chat Memory Context:
```sql
SELECT * FROM n8n_chat_memory 
WHERE session_id = $1 
ORDER BY message_timestamp DESC LIMIT 20;
```

### 4. Processing Status:
```sql
SELECT * FROM processing_logs 
WHERE status = 'processing' OR status = 'failed';
```

---

## ⚠️ **WICHTIGE REGELN**

### 🏗️ **Architektur-Regeln:**
1. **Modular Design:** Jede Funktionalität als separates Modul
2. **Clean Code:** Klare Trennung von Verantwortlichkeiten
3. **Erweiterbarkeit:** Neue Features durch Module hinzufügbar
4. **Wartbarkeit:** Jedes Modul einzeln testbar und updatebar

### 🖼️ **Image Processing Regeln:**
5. **Original Format:** Bilder NIEMALS konvertieren - Original beibehalten
6. **Vektorgrafiken:** SVG, AI, EPS separat extrahieren und speichern
7. **Dual Support:** Raster UND Vektor parallel unterstützen
8. **Format Detection:** Automatische Erkennung von Bild-/Vektor-Typen

### � **Embedding Regeln:**
9. **Ollama Version:** Minimum 0.11.10 für embeddinggemma support
10. **API Endpoint:** Neue `/api/embed` API verwenden (nicht legacy)
11. **Model:** mxbai-embed-large für 768-dimensional embeddings
12. **Batch Processing:** Chunks in Batches für Performance

### �🗄️ **Database Regeln:**
13. **Part Numbers:** Nicht global unique! Nur per Hersteller unique
14. **UUIDs:** Alle Primary Keys sind UUIDs (uuid_generate_v4())
15. **Embeddings:** Immer 768-dimensional für mxbai-embed-large
16. **RLS:** Service Key Only - keine öffentlichen Policies
17. **Arrays:** TEXT[] für Multi-Value Felder (models_compatible, etc.)
18. **JSONB:** Für flexible Metadaten und Vision AI Ergebnisse
19. **Timestamps:** Immer mit timezone (timestamp with time zone)

---

## 🔧 **MAINTENANCE CHECKLIST**

### 🏗️ **Modulare Entwicklung:**
- [ ] Neue Features als separate Module implementieren
- [ ] Module-Dependencies klar dokumentieren
- [ ] Jedes Modul einzeln testbar machen
- [ ] Clean Code Prinzipien einhalten

### � **Ollama Setup:**
- [ ] Ollama Version >= 0.11.10 installiert
- [ ] mxbai-embed-large Model heruntergeladen
- [ ] Neue `/api/embed` API Endpoint getestet
- [ ] Batch Processing für Embedding Performance optimiert

### �🖼️ **Image Processing:**
- [ ] Original-Formate niemals konvertieren
- [ ] Vektorgrafik-Extraktion (SVG, AI, EPS) prüfen
- [ ] Duale Format-Unterstützung testen
- [ ] Vision AI für alle Bildtypen validieren

### 🗄️ **Database Maintenance:**
- [ ] Vector Indizes regelmäßig REINDEX
- [ ] processing_logs für alte Einträge aufräumen
- [ ] n8n_chat_memory Session Cleanup
- [ ] Orphaned Images in storage prüfen
- [ ] RLS Policies testen bei Schema-Updates
- [ ] Extensions Schema Berechtigungen prüfen
- [ ] Search Path für neue Roles konfigurieren

---

**✅ Diese modulare Struktur ist Production-Ready für AI PDF Processing mit maximaler Erweiterbarkeit!**