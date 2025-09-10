-- AI-Enhanced PDF Extraction System - Supabase Tabellen
-- Kopieren Sie diesen gesamten Code in den Supabase SQL Editor

-- 1. Vector Extension aktivieren (für AI Embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Chunks Tabelle (Hauptdaten der verarbeiteten PDF-Segmente)
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    manufacturer TEXT,
    document_type TEXT,
    file_path TEXT,
    original_filename TEXT,
    file_hash TEXT,
    chunk_type TEXT,
    page_number INTEGER,
    chunk_index INTEGER,
    error_codes TEXT[],
    figure_references TEXT[],
    connection_points TEXT[],
    procedures TEXT[],
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Images Tabelle (Extrahierte Bilder aus PDFs)
CREATE TABLE IF NOT EXISTS images (
    id BIGSERIAL PRIMARY KEY,
    file_hash TEXT NOT NULL,
    page_number INTEGER,
    image_index INTEGER,
    r2_key TEXT NOT NULL,
    r2_url TEXT,
    width INTEGER,
    height INTEGER,
    format TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Processing Log Tabelle (Verarbeitungsprotokoll)
CREATE TABLE IF NOT EXISTS processing_log (
    id BIGSERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    chunks_created INTEGER DEFAULT 0,
    images_extracted INTEGER DEFAULT 0,
    error_message TEXT,
    processing_time_seconds REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Indexes für bessere Performance
CREATE INDEX IF NOT EXISTS idx_chunks_manufacturer ON chunks(manufacturer);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_content_gin ON chunks USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_processing_log_status ON processing_log(status);
CREATE INDEX IF NOT EXISTS idx_processing_log_file_hash ON processing_log(file_hash);

-- 6. Row Level Security (RLS) Policies
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Enable read access for all users" ON chunks FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON chunks FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON chunks FOR UPDATE USING (true);

ALTER TABLE images ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Enable read access for all users" ON images FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON images FOR INSERT WITH CHECK (true);

ALTER TABLE processing_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Enable read access for all users" ON processing_log FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON processing_log FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON processing_log FOR UPDATE USING (true);

-- 7. Helper Functions für Statistiken
CREATE OR REPLACE FUNCTION get_manufacturer_stats()
RETURNS TABLE(manufacturer TEXT, count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT c.manufacturer, COUNT(*) as count
    FROM chunks c
    WHERE c.manufacturer IS NOT NULL
    GROUP BY c.manufacturer
    ORDER BY count DESC;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_document_type_stats()
RETURNS TABLE(document_type TEXT, count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT c.document_type, COUNT(*) as count
    FROM chunks c
    WHERE c.document_type IS NOT NULL
    GROUP BY c.document_type
    ORDER BY count DESC;
END;
$$ LANGUAGE plpgsql;
