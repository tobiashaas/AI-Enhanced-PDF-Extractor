# 🤖 SUPABASE AI AGENT - DATABASE SCHEMA REQUEST

## 📋 **KONTEXT**
Ich brauche ein **komplettes PostgreSQL Schema** für ein **AI-Enhanced PDF Processin3. **Flexible Metadaten:** JSONB Felder für erweiterte AI-Daten
4. **Deduplizierung:** parts_catalog.part_number als UNIQUE
5. **Multi-Model Support:** Ein Teil kann mehrere Modelle unterstützen (C3350i, C3351i, C4050i)
6. **Versionierung:** Dokument-Versionen für bessere Nachverfolgung ("September 2025")
7. **Performance:** Optimiert für 10,000+ chunks mit schnellen Suchen
8. **Monitoring:** processing_logs für vollständige Nachverfolgungstem** das Service Manuals, Teile-Kataloge und technische Dokumentation verarbeitet.

## 🎯 **HAUPTFUNKTIONEN**
1. **Specialized Document Processing** - Separate Tabellen für optimierte Suche
2. **Vision AI** - Extraktion von Bildern und Diagrammen  
3. **Parts Catalog** - Ersatzteil-Verwaltung mit Multi-Model Support
4. **Smart Search** - Document-spezifische semantische Suche
5. **Processing Logs** - Vollständige Verarbeitungsprotokollierung

## 📊 **BENÖTIGTE TABELLEN**

### 1. **`service_manuals`** - Service Manual Chunks
**Zweck:** Reparatur- und Wartungsanleitungen
```
- id (Primary Key, UUID)
- content (TEXT) - Der Textinhalt
- file_hash (TEXT) - Eindeutige Datei-Identifikation
- page_number (INTEGER) - Seitennummer
- chunk_index (INTEGER) - Position im Dokument
- manufacturer (TEXT) - Hersteller (HP, Canon, etc.)
- model (TEXT) - Gerätemodell
- document_version (TEXT) - "September 2025", "Rev 1.2", etc.
- procedure_type (TEXT) - "troubleshooting", "maintenance", "repair"
- problem_type (TEXT) - "scanner_issue", "paper_jam", "toner_replace"
- difficulty_level (TEXT) - "basic", "intermediate", "advanced"
- estimated_time (TEXT) - "15 minutes", "1 hour"
- tools_required (TEXT[]) - Benötigte Werkzeuge
- safety_warnings (TEXT[]) - Sicherheitshinweise
- embedding (VECTOR) - 768-dimensional für Supabase Vector Search
- metadata (JSONB) - Flexible Zusatzdaten
- created_at (TIMESTAMP)

// AI-Enhanced Felder:
- connection_points (TEXT[]) - Erkannte Verbindungsstellen
- figure_references (TEXT[]) - Referenzierte Abbildungen
- procedures (TEXT[]) - Erkannte Verfahrensschritte
- error_codes (TEXT[]) - Erkannte Fehlercodes
```

### 2. **`bulletins`** - Technical Bulletins
**Zweck:** Technische Mitteilungen und Updates
```
- id (Primary Key, UUID)
- content (TEXT) - Der Textinhalt
- file_hash (TEXT) - Eindeutige Datei-Identifikation
- page_number (INTEGER) - Seitennummer
- chunk_index (INTEGER) - Position im Dokument
- manufacturer (TEXT) - Hersteller
- models_affected (TEXT[]) - Betroffene Modelle
- document_version (TEXT) - Version/Datum
- bulletin_type (TEXT) - "urgent", "advisory", "recall", "update"
- priority_level (TEXT) - "critical", "high", "medium", "low"
- issue_category (TEXT) - "hardware", "software", "safety", "performance"
- resolution_status (TEXT) - "active", "superseded", "resolved"
- effective_date (DATE) - Wann tritt es in Kraft
- embedding (VECTOR) - 768-dimensional für Supabase Vector Search
- metadata (JSONB)
- created_at (TIMESTAMP)

// AI-Enhanced Felder:
- affected_components (TEXT[]) - Betroffene Komponenten
- symptoms (TEXT[]) - Erkannte Symptome
- root_cause (TEXT) - Grundursache
- solution_steps (TEXT[]) - Lösungsschritte
```

### 3. **`parts_catalogs`** - Parts Catalog Information
**Zweck:** Ersatzteil-Informationen und Kompatibilität
```
- id (Primary Key, UUID)
- content (TEXT) - Der Textinhalt
- file_hash (TEXT) - Eindeutige Datei-Identifikation
- page_number (INTEGER) - Seitennummer
- chunk_index (INTEGER) - Position im Dokument
- manufacturer (TEXT) - Hersteller
- model (TEXT) - Gerätemodell
- document_version (TEXT) - Catalog Version
- part_category (TEXT) - "toner", "drum", "fuser", "sensor", etc.
- availability_status (TEXT) - "available", "discontinued", "special_order"
- embedding (VECTOR) - 768-dimensional für Supabase Vector Search
- metadata (JSONB)
- created_at (TIMESTAMP)

// AI-Enhanced Felder:
- part_numbers_mentioned (TEXT[]) - Erkannte Teilenummern
- compatibility_info (TEXT[]) - Kompatibilitätsinformationen
- replacement_parts (TEXT[]) - Alternative Teile
- installation_notes (TEXT[]) - Einbauhinweise
```

### 4. **`cpmd_documents`** - HP Control Panel Message Documents
**Zweck:** HP-spezifische Control Panel Nachrichten
```
- id (Primary Key, UUID)
- content (TEXT) - Der Textinhalt
- file_hash (TEXT) - Eindeutige Datei-Identifikation
- page_number (INTEGER) - Seitennummer
- chunk_index (INTEGER) - Position im Dokument
- manufacturer (TEXT) - "HP" (immer)
- model (TEXT) - HP Gerätemodell
- document_version (TEXT) - Version/Datum
- message_code (TEXT) - Control Panel Code
- message_type (TEXT) - "error", "warning", "info", "maintenance"
- user_action_required (BOOLEAN) - Benutzeraktion erforderlich
- technician_action_required (BOOLEAN) - Technikeraktion erforderlich
- embedding (VECTOR) - 768-dimensional für Supabase Vector Search
- metadata (JSONB)
- created_at (TIMESTAMP)

// AI-Enhanced Felder:
- display_message (TEXT) - Angezeigte Nachricht
- possible_causes (TEXT[]) - Mögliche Ursachen
- resolution_steps (TEXT[]) - Lösungsschritte
- related_error_codes (TEXT[]) - Verwandte Fehlercodes
```

### 5. **`video_tutorials`** - Video Tutorial Information
**Zweck:** Video-Anleitungen für Reparaturen und Wartung
```
- id (Primary Key, UUID)
- content (TEXT) - Transkript oder Beschreibung
- file_hash (TEXT) - Video-Hash
- video_url (TEXT) - URL zum Video
- manufacturer (TEXT) - Hersteller
- model (TEXT) - Gerätemodell
- document_version (TEXT) - Video Version/Datum
- tutorial_type (TEXT) - "repair", "maintenance", "troubleshooting", "installation"
- procedure_name (TEXT) - "Replace Toner", "Clean Fuser", etc.
- difficulty_level (TEXT) - "beginner", "intermediate", "expert"
- duration_minutes (INTEGER) - Video-Länge
- language (TEXT) - Sprache des Videos
- quality_rating (DECIMAL) - Bewertung 1-5
- embedding (VECTOR) - 768-dimensional für Supabase Vector Search
- metadata (JSONB)
- created_at (TIMESTAMP)

// AI-Enhanced Felder:
- tools_shown (TEXT[]) - Im Video gezeigte Werkzeuge
- parts_demonstrated (TEXT[]) - Demonstrierte Teile
- key_steps (TEXT[]) - Wichtige Schritte
- common_mistakes (TEXT[]) - Häufige Fehler
```

### 6. **`images`** - Bild-Metadaten aus allen Dokumenttypen
**Zweck:** Verwaltet extrahierte Bilder mit Vision AI Analyse
```
- id (Primary Key)
- file_hash (TEXT) - Verknüpfung zur Quelldatei
- source_table (TEXT) - "service_manuals", "bulletins", "parts_catalogs", "cpmd_documents", "video_tutorials"
- page_number (INTEGER)
- image_index (INTEGER) - Position auf der Seite
- storage_url (TEXT) - R2/Cloud Storage URL
- image_type (TEXT) - "diagram", "photo", "table", "flowchart", etc.
- description (TEXT) - AI-generierte Beschreibung
- document_version (TEXT) - Version des Quelldokuments
- manufacturer (TEXT)
- model (TEXT)
- hash (TEXT) - Bild-Hash für Deduplizierung
- metadata (JSONB)
- created_at (TIMESTAMP)

// Vision AI Felder:
- document_source (TEXT) - Quelle
- vision_analysis (JSONB) - Vollständige Vision AI Ergebnisse
```

### 7. **`parts_catalog`** - Ersatzteil-Master-Datenbank
**Zweck:** Deduplizierte Ersatzteil-Datenbank mit Multi-Model Support
```
- id (Primary Key, UUID) - Eindeutige ID für jeden Part
- part_number (TEXT) - Teilenummer (kann bei verschiedenen Herstellern gleich sein)
- part_name (TEXT)
- manufacturer (TEXT)
- models_compatible (TEXT[]) - Array: ["C3350i", "C3351i", "C4050i", "C4051i"]
- category (TEXT) - Teilekategorie
- description (TEXT)
- source_document_version (TEXT) - Version des Parts Catalogs
- metadata (JSONB)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)

// Unique Constraint für Hersteller + Part Number Kombination
UNIQUE(manufacturer, part_number)
```

### 8. **`parts_model_compatibility`** - Ersatzteil-Model Zuordnung
**Zweck:** Flexible Many-to-Many Beziehung zwischen Parts und Models
```
- id (Primary Key)
- part_id (UUID) - Foreign Key zu parts_catalog.id
- model (TEXT) - Gerätemodell (C3350i, C3351i, etc.)
- manufacturer (TEXT) - Hersteller
- compatibility_confirmed (BOOLEAN) - Bestätigte Kompatibilität
- source_document (TEXT) - Welches Dokument die Kompatibilität bestätigt
- document_version (TEXT) - Version des bestätigenden Dokuments
- created_at (TIMESTAMP)
```

### 9. **`n8n_chat_memory`** - Chat Konversations-Speicher
**Zweck:** Einfacher Chat-Verlauf für Context-Aware AI Conversations
```
- id (Primary Key, UUID)
- session_id (TEXT) - Eindeutige Session-ID für Konversation
- message_type (TEXT) - "user", "assistant", "system"
- message_content (TEXT) - Nachrichteninhalt
- message_timestamp (TIMESTAMP) - Wann wurde die Nachricht gesendet
- metadata (JSONB) - Einfache Zusatzdaten
- created_at (TIMESTAMP)

// Minimal AI Context (optional):
- manufacturer_context (TEXT) - Aktueller Hersteller im Gespräch
- model_context (TEXT) - Aktuelles Gerätemodell im Gespräch
- workflow_execution_id (TEXT) - n8n Workflow Execution ID (optional)
```

### 10. **`processing_logs`** - Verarbeitungsprotokoll
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
-- Service Manuals
CREATE INDEX idx_service_manuals_manufacturer_model ON service_manuals(manufacturer, model);
CREATE INDEX idx_service_manuals_procedure_type ON service_manuals(procedure_type);
CREATE INDEX idx_service_manuals_problem_type ON service_manuals(problem_type);
CREATE INDEX idx_service_manuals_file_hash ON service_manuals(file_hash);
CREATE INDEX idx_service_manuals_embedding ON service_manuals USING hnsw (embedding vector_cosine_ops);

-- Bulletins
CREATE INDEX idx_bulletins_manufacturer ON bulletins(manufacturer);
CREATE INDEX idx_bulletins_bulletin_type ON bulletins(bulletin_type);
CREATE INDEX idx_bulletins_priority_level ON bulletins(priority_level);
CREATE INDEX idx_bulletins_file_hash ON bulletins(file_hash);
CREATE INDEX idx_bulletins_embedding ON bulletins USING hnsw (embedding vector_cosine_ops);

-- Parts Catalogs
CREATE INDEX idx_parts_catalogs_manufacturer_model ON parts_catalogs(manufacturer, model);
CREATE INDEX idx_parts_catalogs_part_category ON parts_catalogs(part_category);
CREATE INDEX idx_parts_catalogs_file_hash ON parts_catalogs(file_hash);
CREATE INDEX idx_parts_catalogs_embedding ON parts_catalogs USING hnsw (embedding vector_cosine_ops);

-- CPMD Documents
CREATE INDEX idx_cpmd_model ON cpmd_documents(model);
CREATE INDEX idx_cpmd_message_code ON cpmd_documents(message_code);
CREATE INDEX idx_cpmd_message_type ON cpmd_documents(message_type);
CREATE INDEX idx_cpmd_file_hash ON cpmd_documents(file_hash);
CREATE INDEX idx_cpmd_embedding ON cpmd_documents USING hnsw (embedding vector_cosine_ops);

-- Video Tutorials
CREATE INDEX idx_video_tutorials_manufacturer_model ON video_tutorials(manufacturer, model);
CREATE INDEX idx_video_tutorials_procedure_name ON video_tutorials(procedure_name);
CREATE INDEX idx_video_tutorials_tutorial_type ON video_tutorials(tutorial_type);
CREATE INDEX idx_video_tutorials_embedding ON video_tutorials USING hnsw (embedding vector_cosine_ops);

-- Images Tabelle
CREATE INDEX idx_images_file_hash ON images(file_hash);
CREATE INDEX idx_images_source_table ON images(source_table);
CREATE INDEX idx_images_manufacturer_model ON images(manufacturer, model);

-- Parts Catalog
CREATE INDEX idx_parts_id ON parts_catalog(id);
CREATE INDEX idx_parts_manufacturer_part_number ON parts_catalog(manufacturer, part_number);
CREATE INDEX idx_parts_manufacturer ON parts_catalog(manufacturer);
CREATE INDEX idx_parts_document_version ON parts_catalog(source_document_version);

-- Parts Model Compatibility
CREATE INDEX idx_parts_compatibility_part_id ON parts_model_compatibility(part_id);
CREATE INDEX idx_parts_compatibility_model ON parts_model_compatibility(model);
CREATE INDEX idx_parts_compatibility_manufacturer ON parts_model_compatibility(manufacturer);

-- n8n Chat Memory (simplified)
CREATE INDEX idx_n8n_chat_session_id ON n8n_chat_memory(session_id);
CREATE INDEX idx_n8n_chat_timestamp ON n8n_chat_memory(message_timestamp);
CREATE INDEX idx_n8n_chat_manufacturer_model ON n8n_chat_memory(manufacturer_context, model_context);
CREATE INDEX idx_n8n_chat_workflow_execution ON n8n_chat_memory(workflow_execution_id);

-- Processing Logs
CREATE INDEX idx_processing_logs_status ON processing_logs(status);
CREATE INDEX idx_processing_logs_file_hash ON processing_logs(file_hash);
```

## 🎯 **SPEZIELLE ANFORDERUNGEN**

1. **Document-Specific Search:** Separate Tabellen für optimierte Suche je Dokumenttyp
2. **Vector Search Support:** Jede Tabelle hat eigenes embedding für semantische Suche
3. **Flexible Metadaten:** JSONB Felder für erweiterte AI-Daten
4. **Parts mit UUID:** parts_catalog.id ist UUID Primary Key, part_number mit UNIQUE(manufacturer, part_number)
5. **Multi-Model Support:** Ein Teil kann mehrere Modelle unterstützen (C3350i, C3351i, C4050i)
6. **Versionierung:** Dokument-Versionen für bessere Nachverfolgung ("September 2025")
7. **HP CPMD Support:** Spezielle Tabelle für HP Control Panel Messages
8. **Video Integration:** Support für Video-Tutorials mit Metadaten
9. **Performance:** Optimiert für schnelle, typ-spezifische Suchen
10. **Cross-Reference:** Images Tabelle verknüpft alle Dokumenttypen
11. **n8n Chat Memory:** Einfacher Konversations-Speicher (nicht überkompliziert)

## 🔒 **SICHERHEITSANFORDERUNGEN**

### **Row Level Security (RLS):**
```sql
-- Aktiviere RLS für alle Tabellen
ALTER TABLE service_manuals ENABLE ROW LEVEL SECURITY;
ALTER TABLE bulletins ENABLE ROW LEVEL SECURITY;
ALTER TABLE parts_catalogs ENABLE ROW LEVEL SECURITY;
ALTER TABLE cpmd_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE video_tutorials ENABLE ROW LEVEL SECURITY;
ALTER TABLE images ENABLE ROW LEVEL SECURITY;
ALTER TABLE parts_catalog ENABLE ROW LEVEL SECURITY;
ALTER TABLE parts_model_compatibility ENABLE ROW LEVEL SECURITY;
ALTER TABLE n8n_chat_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Nur Service Key hat Zugriff
CREATE POLICY "Service key access only" ON service_manuals FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON bulletins FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON parts_catalogs FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON cpmd_documents FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON video_tutorials FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON images FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON parts_catalog FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON parts_model_compatibility FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON n8n_chat_memory FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service key access only" ON processing_logs FOR ALL USING (auth.role() = 'service_role');
```

### **Security Features:**
- ✅ **RLS aktiviert** auf allen Tabellen
- ✅ **Service Key Only** - Kein anonymer Zugriff
- ✅ **Production Ready** - Keine öffentlichen Policies
- ✅ **Secret Key Protection** - Nur authentifizierte API Calls

## 🔧 **ERWARTETES ERGEBNIS**
**Komplettes SQL-Schema** mit:
- ✅ Alle Tabellen mit korrekten Datentypen (10 Haupttabellen)
- ✅ Performance-Indizes für schnelle, typ-spezifische Suchen
- ✅ Foreign Key Constraints wo sinnvoll  
- ✅ Supabase Vector Support für semantische Suche (5 Vector embeddings)
- ✅ Multi-Model Support für Parts (C3350i, C3351i, C4050i können gleiche Teile haben)
- ✅ Dokument-Versionierung ("September 2025", "Rev 1.2", etc.)
- ✅ **RLS Security** - Service Key Only Access
- ✅ **n8n Chat Memory** - Context-Aware Conversations
- ✅ **Production Ready** - Sichere, skalierbare Architektur

**Beispiel Use Cases:** 
- **Techniker fragt "Toner wechseln HP E55040"** → Suche in `service_manuals` + `video_tutorials`
- **"Control Panel zeigt Fehlercode C4051"** → Direkte Suche in `cpmd_documents` + `service_manuals`
- **"Welche Teile für C3350i Scanner Problem?"** → `parts_catalogs` + `bulletins`
- **Chat Memory:** "Wir sprachen vorhin über Scanner-Probleme am HP E55040" → Context aus `n8n_chat_memory`
- **n8n Workflow:** Speichert Conversation State zwischen verschiedenen AI-Nodes
- **Ersatzteil "Toner Cartridge TN-328K"** von Canon ist kompatibel mit ["C3350i", "C3351i", "C4050i", "C4051i"]
- **HP Part "123456"** und Canon Part "123456"** können koexistieren dank UNIQUE(manufacturer, part_number)
- **Verschiedene Dokument-Versionen:** "July 2024", "V1.2", "Rev 2.1"

**Ziel:** Schema das sofort in Supabase SQL Editor ausgeführt werden kann für ein funktionierendes AI PDF Processing System mit **document-type optimized search**, **RLS Security**, **n8n Chat Memory**, und vollständiger Parts-Model Kompatibilität.