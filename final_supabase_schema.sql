-- ============================================
-- FINALES SUPABASE SCHEMA (EmbeddingGemma 768D) 
-- Optimiert von Supabase AI + angepasst für AI-Enhanced PDF Extractor
-- Features: 768D EmbeddingGemma, R2 Images, Processing Logs, n8n Integration
-- ============================================

-- Hinweis: Die Datenbank hat die "vector" Extension bereits installiert
-- Falls die Extension nicht vorhanden wäre:
-- CREATE EXTENSION IF NOT EXISTS vector;

-- -------------------------------------------------
-- 1) HAUPTTABELLEN mit Identity PKs und Normalisierung
--    - Alle Tabellen haben id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY
--    - RLS wird am Ende aktiviert; Policies werden angelegt
--    - Fremdschlüssel-Constraints werden inline definiert
-- -------------------------------------------------

CREATE TABLE IF NOT EXISTS images (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

  -- IMAGE IDENTIFICATION & HASH
  image_hash text NOT NULL UNIQUE,          -- SHA-256 Hash des Bildinhalts für Duplikat-Detection
  file_hash text NOT NULL,                  -- Hash der ursprünglichen PDF-Datei
  original_filename text NOT NULL,
  page_number integer NOT NULL,

  -- R2/S3 STORAGE INFO
  storage_url text NOT NULL,                -- Vollständige öffentliche URL: "https://pub-xyz.r2.dev/images/HP_E52645_page5_fig1.jpg"
  storage_bucket text NOT NULL,             -- "pdf-images"
  storage_key text NOT NULL,                -- "HP/E52645/page_5_figure_1.jpg"
  
  -- ALTERNATIVE URLs für verschiedene Zugriffe
  presigned_url text,                       -- Temporäre presigned URL (optional)
  public_url text,                          -- Alternative öffentliche URL
  cdn_url text,                             -- CDN-optimierte URL (optional)

  -- IMAGE CLASSIFICATION
  image_type text NOT NULL,                 -- "diagram", "photo", "chart", "table", "schematic"
  figure_reference text,                    -- "Figure 5-1", "Diagram 2-3"

  -- DOCUMENT CONTEXT
  manufacturer text NOT NULL,
  model text NOT NULL,
  document_type text NOT NULL,
  document_source text,

  -- TECHNICAL METADATA
  width integer,
  height integer,
  file_size_bytes bigint,
  mime_type text DEFAULT 'image/jpeg',
  
  -- IMAGE QUALITY & PROCESSING
  image_quality text DEFAULT 'medium',      -- "low", "medium", "high"
  compression_ratio real,                   -- Kompressionsgrad
  processing_status text DEFAULT 'pending', -- "pending", "processed", "optimized"

  -- AI ANALYSIS RESULTS
  vision_analysis jsonb DEFAULT '{}'::jsonb, -- {description, detected_objects, confidence, text_regions}
  extracted_text text,                      -- OCR results
  
  -- SEARCHABILITY
  searchable_content text,                  -- Kombinierter durchsuchbarer Text
  tags text[],                              -- Manuelle/AI Tags ["electrical", "maintenance", "safety"]

  -- SYSTEM FIELDS
  file_path text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),

  -- CONSTRAINTS
  CONSTRAINT images_image_hash_unique UNIQUE (image_hash),                    -- Verhindert identische Bilder
  CONSTRAINT images_filehash_page_fig_unique UNIQUE (file_hash, page_number, figure_reference) -- Verhindert Duplikate pro PDF-Seite
);

CREATE TABLE IF NOT EXISTS processing_logs (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

  -- FILE IDENTIFICATION
  file_path text NOT NULL,
  file_hash text NOT NULL UNIQUE,
  original_filename text NOT NULL,

  -- PROCESSING STATUS
  status text NOT NULL DEFAULT 'pending',    -- allowed: pending, processing, completed, failed, skipped
  processing_stage text,                     -- parsing, chunking, embedding, storing
  progress_percentage integer DEFAULT 0,

  -- DOCUMENT METADATA
  manufacturer text,
  model text,
  document_type text,
  document_info jsonb,

  -- PROCESSING RESULTS
  total_pages integer,
  chunks_created integer DEFAULT 0,
  images_extracted integer DEFAULT 0,
  processing_time_seconds integer,

  -- ERROR HANDLING
  error_message text,
  retry_count integer DEFAULT 0,
  last_error_at timestamp with time zone,

  -- TIMESTAMPS
  started_at timestamp with time zone DEFAULT now(),
  completed_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

  -- CONTENT & EMBEDDING (768D für EmbeddingGemma)
  content text NOT NULL,
  token_count integer,
  embedding vector(768),                    -- WICHTIG: 768D für EmbeddingGemma

  -- PRIMARY SEARCH FIELDS (n8n Filter)
  manufacturer text NOT NULL,
  model text NOT NULL,
  error_codes text[],
  problem_type text,
  procedure_type text,

  -- DOCUMENT CONTEXT
  document_type text NOT NULL,
  document_subtype text,
  document_priority text DEFAULT 'normal',
  document_source text,
  page_number integer,
  chunk_type text,
  chunk_index integer,

  -- TECHNICAL METADATA
  figure_references text[],
  connection_points text[],
  procedures text[],

  -- FOREIGN KEY RELATIONSHIPS
  processing_log_id bigint REFERENCES processing_logs(id) ON DELETE SET NULL,

  -- RICH METADATA
  metadata jsonb DEFAULT '{}'::jsonb,

  -- SYSTEM FIELDS
  file_path text,
  original_filename text,
  file_hash text REFERENCES processing_logs(file_hash) ON DELETE SET NULL,
  created_at timestamp with time zone DEFAULT now()
);

-- JUNCTION TABLE für Many-to-Many Relationship (Chunks <-> Images)
CREATE TABLE IF NOT EXISTS chunk_images (
  chunk_id bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  image_id bigint NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  PRIMARY KEY (chunk_id, image_id)
);

-- -------------------------------------------------
-- 2) PERFORMANCE INDIZES (optimiert für n8n)
-- -------------------------------------------------

-- Vector Index für Embedding-Suche (IVFFLAT). Tune "lists" nach Datenmenge.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
  ON chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Manufacturer + Model Index
CREATE INDEX IF NOT EXISTS idx_chunks_manufacturer_model
  ON chunks (manufacturer, model);

-- Error Codes Index
CREATE INDEX IF NOT EXISTS idx_chunks_error_codes_gin
  ON chunks USING GIN (error_codes);

-- Kategoriale Indizes
CREATE INDEX IF NOT EXISTS idx_chunks_problem_type
  ON chunks (problem_type);
CREATE INDEX IF NOT EXISTS idx_chunks_procedure_type
  ON chunks (procedure_type);

-- Document Type + Priority
CREATE INDEX IF NOT EXISTS idx_chunks_document_priority
  ON chunks (document_type, document_priority);

CREATE INDEX IF NOT EXISTS idx_chunks_document_source
  ON chunks (document_source);

CREATE INDEX IF NOT EXISTS idx_chunks_document_context
  ON chunks (document_type, chunk_type);

-- Processing Log Relationship
CREATE INDEX IF NOT EXISTS idx_chunks_processing_log_id
  ON chunks (processing_log_id);

-- File Hash Index
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash
  ON chunks (file_hash);

-- Full-Text Search Index
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
  ON chunks USING gin(to_tsvector('english', content));

-- Metadata JSONB Index
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
  ON chunks USING GIN (metadata jsonb_path_ops);

-- Images Indizes
CREATE INDEX IF NOT EXISTS idx_images_image_hash
  ON images (image_hash);                           -- Für Duplikat-Detection
CREATE INDEX IF NOT EXISTS idx_images_file_hash
  ON images (file_hash);                           -- Für PDF-bezogene Suchen
CREATE INDEX IF NOT EXISTS idx_images_manufacturer_model
  ON images (manufacturer, model);                 -- Für n8n Filter
CREATE INDEX IF NOT EXISTS idx_images_storage_key
  ON images (storage_key);                         -- Für R2 Zugriff
CREATE INDEX IF NOT EXISTS idx_images_document_type_image_type
  ON images (document_type, image_type);           -- Für kategorisierte Suchen
CREATE INDEX IF NOT EXISTS idx_images_processing_status
  ON images (processing_status);                   -- Für Verarbeitungsstatus
CREATE INDEX IF NOT EXISTS idx_images_tags_gin
  ON images USING GIN (tags);                      -- Für Tag-basierte Suchen
CREATE INDEX IF NOT EXISTS idx_images_vision_analysis_gin
  ON images USING GIN (vision_analysis);           -- Für AI-Analysis Suchen

-- Processing Logs Indizes
CREATE INDEX IF NOT EXISTS idx_processing_logs_status
  ON processing_logs (status);
CREATE INDEX IF NOT EXISTS idx_processing_logs_original_filename
  ON processing_logs (original_filename);

-- Junction Table Index
CREATE INDEX IF NOT EXISTS idx_chunk_images_image_id
  ON chunk_images (image_id);

-- -------------------------------------------------
-- 3) VIEWS (security_invoker für RLS-Kompatibilität)
-- -------------------------------------------------

CREATE OR REPLACE VIEW processing_status
WITH (security_invoker = on) AS
SELECT
  pl.id,
  pl.original_filename,
  pl.status,
  pl.processing_stage,
  pl.progress_percentage,
  pl.manufacturer,
  pl.model,
  pl.document_type,
  pl.chunks_created,
  pl.images_extracted,
  pl.processing_time_seconds,
  pl.started_at,
  pl.completed_at,
  pl.error_message
FROM processing_logs pl
ORDER BY pl.started_at DESC;

CREATE OR REPLACE VIEW failed_processing
WITH (security_invoker = on) AS
SELECT * FROM processing_logs 
WHERE status = 'failed' 
ORDER BY started_at DESC;

CREATE OR REPLACE VIEW resume_candidates
WITH (security_invoker = on) AS
SELECT * FROM processing_logs
WHERE status IN ('pending','processing')
  AND started_at < NOW() - INTERVAL '1 hour'
ORDER BY started_at ASC;

CREATE OR REPLACE VIEW error_procedures
WITH (security_invoker = on) AS
SELECT 
    c.id, c.content, c.manufacturer, c.model, c.error_codes, 
    c.problem_type, c.procedures, c.metadata, c.document_type, c.document_priority,
    (c.metadata->>'confidence')::float as confidence_score
FROM chunks c
WHERE c.procedure_type = 'troubleshooting' 
AND array_length(c.error_codes, 1) > 0
ORDER BY 
    CASE c.document_priority 
        WHEN 'urgent' THEN 1 
        WHEN 'normal' THEN 2 
        ELSE 3 
    END,
    (c.metadata->>'confidence')::float DESC;

CREATE OR REPLACE VIEW urgent_bulletins
WITH (security_invoker = on) AS
SELECT 
    c.id, c.content, c.manufacturer, c.model, c.document_type, c.document_subtype,
    c.error_codes, c.problem_type, c.metadata->>'confidence' as confidence,
    c.created_at
FROM chunks c
WHERE c.document_type IN ('Bulletin', 'Technical Information', 'CPMD')
AND c.document_priority = 'urgent'
ORDER BY c.created_at DESC;

-- -------------------------------------------------
-- 4) HELPER FUNCTIONS (verbessert und RLS-kompatibel)
-- -------------------------------------------------

-- Function: check_file_processed (Duplicate Detection)
CREATE OR REPLACE FUNCTION check_file_processed(input_file_hash text)
RETURNS TABLE (
  is_processed boolean,
  processing_status text,
  chunks_count bigint,
  images_count bigint
) AS $$
DECLARE
  pl_status text;
BEGIN
  SELECT status INTO pl_status 
  FROM processing_logs 
  WHERE file_hash = input_file_hash 
  LIMIT 1;
  
  IF pl_status IS NULL THEN
    RETURN QUERY SELECT false, 'not_found'::text, 0::bigint, 0::bigint;
    RETURN;
  END IF;

  RETURN QUERY
  SELECT (pl_status = 'completed')::boolean,
         pl_status,
         (SELECT COUNT(*) FROM chunks WHERE file_hash = input_file_hash),
         (SELECT COUNT(*) FROM images WHERE file_hash = input_file_hash);
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Function: start_processing (Session Management)
CREATE OR REPLACE FUNCTION start_processing(
  input_file_path text,
  input_file_hash text,
  input_filename text,
  input_document_info jsonb
) RETURNS bigint AS $$
DECLARE
  log_id bigint;
BEGIN
  INSERT INTO processing_logs (
    file_path, file_hash, original_filename,
    status, processing_stage, manufacturer, model, document_type, document_info,
    started_at, updated_at
  ) VALUES (
    input_file_path, input_file_hash, input_filename,
    'processing', 'parsing', 
    input_document_info->>'manufacturer',
    input_document_info->>'model', 
    input_document_info->>'document_type',
    input_document_info, now(), now()
  )
  ON CONFLICT (file_hash) DO UPDATE SET
    status = 'processing',
    processing_stage = 'parsing',
    started_at = now(),
    retry_count = COALESCE(processing_logs.retry_count,0) + 1,
    updated_at = now()
  RETURNING id INTO log_id;

  RETURN log_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: update_processing_progress (Progress Tracking)
CREATE OR REPLACE FUNCTION update_processing_progress(
  log_id bigint,
  new_stage text,
  progress integer,
  chunks_count integer DEFAULT NULL,
  images_count integer DEFAULT NULL
) RETURNS void AS $$
BEGIN
  UPDATE processing_logs
  SET processing_stage = new_stage,
      progress_percentage = progress,
      chunks_created = COALESCE(chunks_count, chunks_created),
      images_extracted = COALESCE(images_count, images_extracted),
      updated_at = now()
  WHERE id = log_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: complete_processing (Session Completion)
CREATE OR REPLACE FUNCTION complete_processing(
  log_id bigint,
  processing_time integer
) RETURNS void AS $$
BEGIN
  UPDATE processing_logs
  SET status = 'completed',
      processing_stage = 'completed',
      progress_percentage = 100,
      processing_time_seconds = processing_time,
      completed_at = now(),
      updated_at = now()
  WHERE id = log_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: Image Management & URL Generation
CREATE OR REPLACE FUNCTION get_image_by_hash(input_image_hash text)
RETURNS TABLE(
  id bigint,
  storage_url text,
  public_url text,
  cdn_url text,
  image_type text,
  figure_reference text,
  width integer,
  height integer,
  file_size_bytes bigint,
  vision_analysis jsonb
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    i.id,
    i.storage_url,
    i.public_url,
    i.cdn_url,
    i.image_type,
    i.figure_reference,
    i.width,
    i.height,
    i.file_size_bytes,
    i.vision_analysis
  FROM images i
  WHERE i.image_hash = input_image_hash;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: Bulk Image URL Update (für R2 Domain Changes)
CREATE OR REPLACE FUNCTION update_image_urls(
  old_domain text,
  new_domain text
)
RETURNS integer AS $$
DECLARE
  updated_count integer;
BEGIN
  UPDATE images 
  SET 
    storage_url = replace(storage_url, old_domain, new_domain),
    public_url = replace(public_url, old_domain, new_domain),
    updated_at = now()
  WHERE storage_url LIKE '%' || old_domain || '%'
     OR public_url LIKE '%' || old_domain || '%';
  
  GET DIAGNOSTICS updated_count = ROW_COUNT;
  RETURN updated_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: Enhanced Vector Search mit Images (768D EmbeddingGemma)
CREATE OR REPLACE FUNCTION enhanced_vector_search_with_images(
  query_embedding vector(768),
  search_manufacturer text DEFAULT NULL,
  search_model text DEFAULT NULL,
  search_problem_type text DEFAULT NULL,
  preferred_document_types text[] DEFAULT NULL,
  include_images boolean DEFAULT true,
  match_threshold float DEFAULT 0.3,
  match_count integer DEFAULT 10
) RETURNS TABLE (
  chunk_id bigint,
  content text,
  manufacturer text,
  model text,
  document_type text,
  similarity float,
  related_images_count integer,
  image_urls text[]
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.id,
    c.content,
    c.manufacturer,
    c.model,
    c.document_type,
    1 - (c.embedding <=> query_embedding) AS similarity,
    COALESCE(ci.count_images, 0) AS related_images_count,
    COALESCE(imgs.urls, ARRAY[]::text[]) AS image_urls
  FROM chunks c
  LEFT JOIN LATERAL (
    SELECT COUNT(*)::integer AS count_images 
    FROM chunk_images chi 
    WHERE chi.chunk_id = c.id
  ) ci ON true
  LEFT JOIN LATERAL (
    SELECT ARRAY_AGG(
      CASE 
        WHEN i.cdn_url IS NOT NULL THEN i.cdn_url      -- Bevorzuge CDN URL
        WHEN i.public_url IS NOT NULL THEN i.public_url -- Dann public URL  
        ELSE i.storage_url                              -- Fallback auf storage URL
      END
    ) AS urls
    FROM chunk_images chi
    JOIN images i ON i.id = chi.image_id
    WHERE chi.chunk_id = c.id
    LIMIT 50
  ) imgs ON include_images
  WHERE (c.embedding IS NOT NULL AND (1 - (c.embedding <=> query_embedding)) > match_threshold)
    AND (search_manufacturer IS NULL OR c.manufacturer = search_manufacturer)
    AND (search_model IS NULL OR c.model = search_model)
    AND (search_problem_type IS NULL OR c.problem_type = search_problem_type)
    AND (preferred_document_types IS NULL OR c.document_type = ANY(preferred_document_types))
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Function: Simplified Vector Search (LangChain Kompatibilität)
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(768),
  match_count integer DEFAULT 10,
  filter jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.id,
    c.content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM chunks c
  WHERE c.metadata @> filter
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- -------------------------------------------------
-- 5) ROW LEVEL SECURITY (Production-Ready Policies)
-- -------------------------------------------------

-- Aktivieren der RLS (danach sind Policies erforderlich)
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE images ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_images ENABLE ROW LEVEL SECURITY;

-- Policies: konservative Beispiele für n8n (öffentlicher Lesezugriff, geschriebene ops nur für "authenticated")
CREATE POLICY "public_select_chunks" ON chunks 
  FOR SELECT USING (true);
CREATE POLICY "authenticated_insert_chunks" ON chunks 
  FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "authenticated_update_chunks" ON chunks 
  FOR UPDATE TO authenticated USING (true);

CREATE POLICY "public_select_images" ON images 
  FOR SELECT USING (true);
CREATE POLICY "authenticated_insert_images" ON images 
  FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "public_select_processing_logs" ON processing_logs 
  FOR SELECT USING (true);
CREATE POLICY "authenticated_insert_processing_logs" ON processing_logs 
  FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "authenticated_update_processing_logs" ON processing_logs 
  FOR UPDATE TO authenticated USING (true);

CREATE POLICY "public_select_chunk_images" ON chunk_images 
  FOR SELECT USING (true);
CREATE POLICY "authenticated_insert_chunk_images" ON chunk_images 
  FOR INSERT TO authenticated WITH CHECK (true);

-- WICHTIG: Die obigen Policies sind pragmatisch für Integrationen wie n8n.
-- Für produktive Systeme sollten Policies auf Basis von auth.uid() und tenant-Claims verfeinert werden.

-- -------------------------------------------------
-- 6) STATISTIKEN AKTUALISIEREN
-- -------------------------------------------------
ANALYZE chunks;
ANALYZE images;
ANALYZE processing_logs;
ANALYZE chunk_images;

-- ============================================
-- INSTALLATION FERTIG
-- Kurze Zusammenfassung (auf Deutsch):
-- - Vector-Extension: bereits installiert (public, v0.8.0).
-- - Embedding-Dimension: 768 (EmbeddingGemma) in allen relevanten Funktionen und Spalten.
-- - RLS: aktiviert; Beispiel-Policies gesetzt — bitte an Produktionsanforderungen anpassen.
-- - Indizes: IVFFLAT für vector-Search + mehrere Filter-Indizes.
-- - Views & Funktionen: bereit für n8n-Integration und LangChain-ähnliche Abfragen.
-- ============================================