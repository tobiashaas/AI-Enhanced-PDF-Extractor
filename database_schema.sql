-- ============================================
-- POSTGRESQL DATABASE SCHEMA (EmbeddingGemma 768D) 
-- Compatible with PostgreSQL databases including Supabase, AWS RDS, etc.
-- Features: 768D EmbeddingGemma vectors, Cloud storage integration, Processing logs
-- ============================================

-- Note: Requires the "vector" extension for pgvector support
-- Install if not available:
CREATE EXTENSION IF NOT EXISTS vector;

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  
  -- CONTENT
  content TEXT NOT NULL,
  embedding vector(768),  -- EmbeddingGemma produces 768-dimensional vectors
  
  -- DOCUMENT METADATA
  manufacturer TEXT,
  document_type TEXT,
  file_path TEXT,
  original_filename TEXT,
  file_hash TEXT,
  
  -- CHUNK METADATA
  chunk_type TEXT,
  page_number INTEGER,
  chunk_index INTEGER,
  
  -- EXTRACTED FEATURES
  error_codes TEXT[],
  figure_references TEXT[],
  connection_points TEXT[],
  procedures TEXT[],
  
  -- SYSTEM
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create images table
CREATE TABLE IF NOT EXISTS images (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

  -- IMAGE IDENTIFICATION & HASH
  image_hash text NOT NULL UNIQUE,
  file_hash text NOT NULL,
  original_filename text NOT NULL,
  page_number integer NOT NULL,

  -- CLOUD STORAGE INFO (R2/S3 compatible)
  storage_url text NOT NULL,
  storage_bucket text NOT NULL,
  storage_key text NOT NULL,

  -- IMAGE METADATA
  width integer,
  height integer,
  format text,
  file_size_bytes bigint,

  -- SYSTEM
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create processing_log table  
CREATE TABLE IF NOT EXISTS processing_log (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  
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

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_manufacturer ON chunks(manufacturer);
CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_processing_log_status ON processing_log(status);
