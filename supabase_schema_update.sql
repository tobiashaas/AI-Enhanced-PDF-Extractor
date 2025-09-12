-- ============================================
-- SUPABASE SCHEMA UPDATE FOR AI PDF PROCESSOR
-- Adds missing columns to chunks table for enhanced AI processing
-- Execute this in Supabase SQL Editor: https://supabase.com/dashboard
-- ============================================

-- Add missing columns to chunks table
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS connection_points TEXT[];
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_priority TEXT DEFAULT 'normal';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_subtype TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_source TEXT DEFAULT 'PDF Extract';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_references TEXT[];
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS procedures TEXT[];
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS error_codes TEXT[];

-- Add performance indices for fast queries
CREATE INDEX IF NOT EXISTS idx_chunks_manufacturer_model ON chunks(manufacturer, model);
CREATE INDEX IF NOT EXISTS idx_chunks_document_type ON chunks(document_type);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON chunks(page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);

-- Fix NOT NULL constraints where needed
ALTER TABLE chunks ALTER COLUMN chunk_index DROP NOT NULL;

-- Add documentation
COMMENT ON TABLE chunks IS 'Enhanced chunks table with AI processing support for PDF extraction system';

-- Verify the update worked
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'chunks' 
ORDER BY ordinal_position;
