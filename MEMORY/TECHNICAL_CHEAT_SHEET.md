# 🔧 TECHNICAL CHEAT SHEET
**Schnelle Referenz für Development**

## 🤖 **OLLAMA SETUP**
```bash
# Version prüfen (min. 0.11.10)
ollama --version

# EmbeddingGemma downloaden
ollama pull embeddinggemma

# Test API Call
curl http://localhost:11434/api/embed -d '{
  "model": "embeddinggemma", 
  "input": "Test text for embedding"
}'
```

## 🗄️ **DATABASE QUICK COMMANDS**
```sql
-- Vector Search Template
SELECT content, manufacturer, model,
       (1 - (embedding <-> $1::vector)) as similarity
FROM service_manuals 
WHERE manufacturer = 'HP'
  AND (1 - (embedding <-> $1::vector)) > 0.7
ORDER BY similarity DESC
LIMIT 10;

-- Check Vector Dimensions
SELECT vector_dims(embedding) FROM service_manuals LIMIT 1;
-- Should return: 768

-- RLS Status Check
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
  AND rowsecurity = true;

-- Extension Setup für UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA extensions;

-- Prüfen, wo Extensions installiert sind
SELECT e.extname AS extension, n.nspname AS schema 
FROM pg_extension e 
JOIN pg_namespace n ON e.extnamespace = n.oid 
WHERE e.extname IN ('uuid-ossp', 'vector');
```

## 📊 **10 MAIN TABLES CHECKLIST**
- [ ] `service_manuals` 
  ```sql
  version_info jsonb,
  compatible_models text[],
  manufacturer text
  ```
- [ ] `bulletins`
  ```sql
  version_info jsonb,
  affected_models text[],
  manufacturer text
  ```
- [ ] `parts_catalogs`
  ```sql
  version_info jsonb,
  compatible_models text[],
  manufacturer text
  ```
- [ ] `cpmd_documents`
  ```sql
  version_info jsonb,
  compatible_models text[],
  manufacturer text DEFAULT 'HP'
  ```
- [ ] `video_tutorials`
  ```sql
  version_info jsonb,
  model_series text[],
  manufacturer text
  ```
- [ ] `images`
  ```sql
  document_type text,
  manufacturer text,
  model_reference text[]
  ```
- [ ] `parts_catalog`
  ```sql
  manufacturer text,
  part_number text,
  compatible_models text[]
  ```
- [ ] `parts_model_compatibility`
  ```sql
  part_id uuid REFERENCES parts_catalog,
  model text,
  manufacturer text
  ```
- [ ] `n8n_chat_memory` - Chat Kontext
- [ ] `processing_logs`
  ```sql
  document_category text,
  manufacturer text,
  success boolean
  ```

## 🔍 **EMBEDDING MODEL SPECS**
```json
{
  "model": "embeddinggemma",
  "provider": "Google",
  "parameters": "300M",
  "dimensions": 768,
  "api_endpoint": "/api/embed",
  "batch_support": true,
  "multilingual": true
}
```

## 🖼️ **ZERO CONVERSION POLICY**
```bash
# ✅ RICHTIG:
cp original.* storage/     # Jedes Format erlaubt
sha256sum original.*      # Hash zur Validierung
file -i original.*        # Content-Type checken

# ❌ FALSCH:
convert                   # Keine Konvertierung!
mogrify                   # Keine Modifikation!
optimize                  # Keine Optimierung!

# 📝 SUPPORTED FORMATS:
# - Vektor: .svg, .ai, .eps, .pdf
# - Raster: .png, .jpg, .bmp, .tiff
# - Compound: .pdf (mit eingebetteten Bildern)
```

## 🏗️ **MODULE TEMPLATE**
```typescript
// Module Structure Template
interface DocumentProcessor {
  processDocument(file: File): Promise<ProcessResult>
  extractChunks(content: string): Chunk[]
  generateEmbeddings(chunks: Chunk[]): Promise<Vector[]>
  storeInDatabase(data: ProcessedData): Promise<void>
}

class ServiceManualProcessor implements DocumentProcessor {
  // Implement interface...
}
```

## 🔒 **SECURITY CHECKLIST**
- [ ] RLS aktiviert auf allen Tabellen
- [ ] Service Key Only Policies
- [ ] Keine anonymen/authenticated Policies  
- [ ] API Endpoints authentifiziert
- [ ] Environment Variables gesichert

## ⚡ **PERFORMANCE MONITORING**
```sql
-- Vector Index Performance
EXPLAIN ANALYZE 
SELECT * FROM service_manuals 
ORDER BY embedding <-> $1::vector 
LIMIT 10;

-- Table Sizes
SELECT 
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## 🚨 **COMMON PITFALLS**
1. **❌ Image Conversion** - Originale NIEMALS konvertieren
2. **❌ Global Part Numbers** - Nur per Hersteller unique
3. **❌ Legacy API** - `/api/embeddings` nicht verwenden
4. **❌ Public Policies** - Nur Service Key Access
5. **❌ Wrong Dimensions** - Immer 768 für EmbeddingGemma

## 📱 **EMERGENCY COMMANDS**
```bash
# Ollama Restart
brew services restart ollama

# Check Model Status  
ollama list
ollama ps

# Database Connection Test
psql "postgresql://[supabase-connection-string]"

# Clear Ollama Models (if needed)
ollama rm embeddinggemma
ollama pull embeddinggemma

## 🪟 **WINDOWS-KOMPATIBILITÄT**
```sql
-- Windows: Überprüfen der Extension-Konfiguration
-- Stellen Sie sicher, dass die uuid-ossp Extension im extensions-Schema installiert ist
SELECT e.extname AS extension, n.nspname AS schema 
FROM pg_extension e 
JOIN pg_namespace n ON e.extnamespace = n.oid 
WHERE e.extname = 'uuid-ossp';

-- Falls nötig, verschieben/installieren Sie die Extension im extensions-Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA extensions;

-- Beim Erstellen von Tabellen IMMER das Schema für uuid_generate_v4() angeben
-- Beispiel:
CREATE TABLE example_table (
  id uuid PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
  name text,
  created_at timestamp with time zone DEFAULT now()
);

-- Um Windows-kompatibel zu bleiben, prüfen Sie die SQL-Befehle
-- für alle Tabellen, ob extensions.uuid_generate_v4() verwendet wird:
SELECT column_name, column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
AND column_default LIKE '%uuid_generate_v4%'
AND column_default NOT LIKE '%extensions.uuid_generate_v4%';
```
```
```