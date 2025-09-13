-- AI-Enhanced PDF Processing Schema (Production-Ready)
-- Generated from SUPABASE_AI_AGENT_SCHEMA_REQUEST.md
-- Date: 13. September 2025
-- 
-- Prerequisites: 
-- - Extension "uuid-ossp" installed
-- - Extension "vector" installed for embeddings
--
-- Tables: 10 main tables for AI PDF processing
-- Security: RLS enabled with service key only access

BEGIN;

-- 1) Schemas
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE SCHEMA IF NOT EXISTS private;

-- 2) Extensions in separates Schema verschieben
-- Extension-Suche (Befehl zum Überprüfen des aktuellen Schemas einer Extension)
-- SELECT e.extname AS extension, n.nspname AS schema FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'vector';

-- Extensions installieren (oder verschieben)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA extensions;

-- Falls vector bereits im public Schema existiert, ins extensions Schema verschieben
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        ALTER EXTENSION vector SET SCHEMA extensions;
    ELSE
        CREATE EXTENSION IF NOT EXISTS "vector" SCHEMA extensions;
    END IF;
END
$$;

-- 3) Zugriffsrechte für Extensions Schema setzen
GRANT USAGE ON SCHEMA extensions TO postgres, service_role, anon;

-- 4) Extension-Schema zur Suche hinzufügen
ALTER ROLE postgres SET search_path TO public, extensions;
ALTER ROLE service_role SET search_path TO public, extensions;
ALTER ROLE anon SET search_path TO public, extensions;

-- 2) Tables

-- service_manuals
CREATE TABLE IF NOT EXISTS public.service_manuals (
  id uuid PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
  content text,
  file_hash text,
  page_number integer,
  chunk_index integer,
  manufacturer text,
  model text,
  document_version text,
  procedure_type text,
  problem_type text,
  difficulty_level text,
  estimated_time text,
  tools_required text[],
  safety_warnings text[],
  embedding extensions.vector(768),
  metadata jsonb,
  created_at timestamp with time zone DEFAULT now(),
  connection_points text[],
  figure_references text[],
  procedures text[],
  error_codes text[]
);

-- bulletins
CREATE TABLE IF NOT EXISTS public.bulletins (
  id uuid PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
  content text,
  file_hash text,
  page_number integer,
  chunk_index integer,
  manufacturer text,
  models_affected text[],
  document_version text,
  bulletin_type text,
  priority_level text,
  issue_category text,
  resolution_status text,
  effective_date date,
  embedding extensions.vector(768),
  metadata jsonb,
  created_at timestamp with time zone DEFAULT now(),
  affected_components text[],
  symptoms text[],
  root_cause text,
  solution_steps text[]
);

-- parts_catalogs (document chunks for parts catalogs)
CREATE TABLE IF NOT EXISTS public.parts_catalogs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  content text,
  file_hash text,
  page_number integer,
  chunk_index integer,
  manufacturer text,
  model text,
  document_version text,
  part_category text,
  availability_status text,
  embedding vector(768),
  metadata jsonb,
  created_at timestamp with time zone DEFAULT now(),
  part_numbers_mentioned text[],
  compatibility_info text[],
  replacement_parts text[],
  installation_notes text[]
);

-- cpmd_documents
CREATE TABLE IF NOT EXISTS public.cpmd_documents (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  content text,
  file_hash text,
  page_number integer,
  chunk_index integer,
  manufacturer text DEFAULT 'HP',
  model text,
  document_version text,
  message_code text,
  message_type text,
  user_action_required boolean,
  technician_action_required boolean,
  embedding vector(768),
  metadata jsonb,
  created_at timestamp with time zone DEFAULT now(),
  display_message text,
  possible_causes text[],
  resolution_steps text[],
  related_error_codes text[]
);

-- video_tutorials
CREATE TABLE IF NOT EXISTS public.video_tutorials (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  content text,
  file_hash text,
  video_url text,
  manufacturer text,
  model text,
  document_version text,
  tutorial_type text,
  procedure_name text,
  difficulty_level text,
  duration_minutes integer,
  language text,
  quality_rating numeric,
  embedding vector(768),
  metadata jsonb,
  created_at timestamp with time zone DEFAULT now(),
  tools_shown text[],
  parts_demonstrated text[],
  key_steps text[],
  common_mistakes text[]
);

-- images (shared)
CREATE TABLE IF NOT EXISTS public.images (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  file_hash text,
  source_table text,
  source_id uuid,
  page_number integer,
  image_index integer,
  storage_url text,
  image_type text,
  description text,
  document_version text,
  manufacturer text,
  model text,
  hash text,
  metadata jsonb,
  vision_analysis jsonb,
  created_at timestamp with time zone DEFAULT now()
);

-- parts_catalog (master parts)
CREATE TABLE IF NOT EXISTS public.parts_catalog (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  part_number text NOT NULL,
  part_name text,
  manufacturer text NOT NULL,
  models_compatible text[],
  category text,
  description text,
  source_document_version text,
  metadata jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT parts_manufacturer_partnumber_unique UNIQUE (manufacturer, part_number)
);

-- parts_model_compatibility
CREATE TABLE IF NOT EXISTS public.parts_model_compatibility (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  part_id uuid NOT NULL REFERENCES public.parts_catalog(id) ON DELETE CASCADE,
  model text NOT NULL,
  manufacturer text,
  compatibility_confirmed boolean DEFAULT false,
  source_document text,
  document_version text,
  created_at timestamp with time zone DEFAULT now()
);

-- n8n_chat_memory
CREATE TABLE IF NOT EXISTS public.n8n_chat_memory (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id text NOT NULL,
  message_type text,
  message_content text,
  message_timestamp timestamp with time zone DEFAULT now(),
  metadata jsonb,
  created_at timestamp with time zone DEFAULT now(),
  manufacturer_context text,
  model_context text,
  workflow_execution_id text
);

-- processing_logs
CREATE TABLE IF NOT EXISTS public.processing_logs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  file_path text,
  file_hash text UNIQUE,
  original_filename text,
  status text,
  processing_stage text,
  progress_percentage integer,
  chunks_created integer,
  images_extracted integer,
  manufacturer text,
  model text,
  document_type text,
  document_info jsonb,
  document_title text,
  document_version text,
  error_message text,
  started_at timestamp with time zone,
  completed_at timestamp with time zone,
  processing_time_seconds integer,
  retry_count integer DEFAULT 0,
  updated_at timestamp with time zone DEFAULT now()
);

-- 3) Indexes for performance (including vector HNSW)

CREATE INDEX IF NOT EXISTS idx_service_manuals_manufacturer_model ON public.service_manuals (manufacturer, model);
CREATE INDEX IF NOT EXISTS idx_service_manuals_procedure_type ON public.service_manuals (procedure_type);
CREATE INDEX IF NOT EXISTS idx_service_manuals_problem_type ON public.service_manuals (problem_type);
CREATE INDEX IF NOT EXISTS idx_service_manuals_file_hash ON public.service_manuals (file_hash);
CREATE INDEX IF NOT EXISTS idx_service_manuals_embedding ON public.service_manuals USING hnsw (embedding extensions.vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_bulletins_manufacturer ON public.bulletins (manufacturer);
CREATE INDEX IF NOT EXISTS idx_bulletins_bulletin_type ON public.bulletins (bulletin_type);
CREATE INDEX IF NOT EXISTS idx_bulletins_priority_level ON public.bulletins (priority_level);
CREATE INDEX IF NOT EXISTS idx_bulletins_file_hash ON public.bulletins (file_hash);
CREATE INDEX IF NOT EXISTS idx_bulletins_embedding ON public.bulletins USING hnsw (embedding extensions.vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_parts_catalogs_manufacturer_model ON public.parts_catalogs (manufacturer, model);
CREATE INDEX IF NOT EXISTS idx_parts_catalogs_part_category ON public.parts_catalogs (part_category);
CREATE INDEX IF NOT EXISTS idx_parts_catalogs_file_hash ON public.parts_catalogs (file_hash);
CREATE INDEX IF NOT EXISTS idx_parts_catalogs_embedding ON public.parts_catalogs USING hnsw (embedding extensions.vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_cpmd_model ON public.cpmd_documents (model);
CREATE INDEX IF NOT EXISTS idx_cpmd_message_code ON public.cpmd_documents (message_code);
CREATE INDEX IF NOT EXISTS idx_cpmd_message_type ON public.cpmd_documents (message_type);
CREATE INDEX IF NOT EXISTS idx_cpmd_file_hash ON public.cpmd_documents (file_hash);
CREATE INDEX IF NOT EXISTS idx_cpmd_embedding ON public.cpmd_documents USING hnsw (embedding extensions.vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_video_tutorials_manufacturer_model ON public.video_tutorials (manufacturer, model);
CREATE INDEX IF NOT EXISTS idx_video_tutorials_procedure_name ON public.video_tutorials (procedure_name);
CREATE INDEX IF NOT EXISTS idx_video_tutorials_tutorial_type ON public.video_tutorials (tutorial_type);
CREATE INDEX IF NOT EXISTS idx_video_tutorials_embedding ON public.video_tutorials USING hnsw (embedding extensions.vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_images_file_hash ON public.images (file_hash);
CREATE INDEX IF NOT EXISTS idx_images_source_table ON public.images (source_table);
CREATE INDEX IF NOT EXISTS idx_images_manufacturer_model ON public.images (manufacturer, model);

CREATE INDEX IF NOT EXISTS idx_parts_id ON public.parts_catalog (id);
CREATE INDEX IF NOT EXISTS idx_parts_manufacturer_part_number ON public.parts_catalog (manufacturer, part_number);
CREATE INDEX IF NOT EXISTS idx_parts_manufacturer ON public.parts_catalog (manufacturer);
CREATE INDEX IF NOT EXISTS idx_parts_document_version ON public.parts_catalog (source_document_version);

CREATE INDEX IF NOT EXISTS idx_parts_compatibility_part_id ON public.parts_model_compatibility (part_id);
CREATE INDEX IF NOT EXISTS idx_parts_compatibility_model ON public.parts_model_compatibility (model);
CREATE INDEX IF NOT EXISTS idx_parts_compatibility_manufacturer ON public.parts_model_compatibility (manufacturer);

CREATE INDEX IF NOT EXISTS idx_n8n_chat_session_id ON public.n8n_chat_memory (session_id);
CREATE INDEX IF NOT EXISTS idx_n8n_chat_timestamp ON public.n8n_chat_memory (message_timestamp);
CREATE INDEX IF NOT EXISTS idx_n8n_chat_manufacturer_model ON public.n8n_chat_memory (manufacturer_context, model_context);
CREATE INDEX IF NOT EXISTS idx_n8n_chat_workflow_execution ON public.n8n_chat_memory (workflow_execution_id);

CREATE INDEX IF NOT EXISTS idx_processing_logs_status ON public.processing_logs (status);
CREATE INDEX IF NOT EXISTS idx_processing_logs_file_hash ON public.processing_logs (file_hash);

-- 4) Enable RLS and create policies (Service key only access)

ALTER TABLE public.service_manuals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_service_manuals" ON public.service_manuals FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.bulletins ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_bulletins" ON public.bulletins FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.parts_catalogs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_parts_catalogs" ON public.parts_catalogs FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.cpmd_documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_cpmd_documents" ON public.cpmd_documents FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.video_tutorials ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_video_tutorials" ON public.video_tutorials FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.images ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_images" ON public.images FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.parts_catalog ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_parts_catalog" ON public.parts_catalog FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.parts_model_compatibility ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_parts_model_compatibility" ON public.parts_model_compatibility FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.n8n_chat_memory ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_n8n_chat_memory" ON public.n8n_chat_memory FOR ALL TO public USING (auth.role() = 'service_role');

ALTER TABLE public.processing_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_key_only_processing_logs" ON public.processing_logs FOR ALL TO public USING (auth.role() = 'service_role');

-- 5) Foreign key index for parts_model_compatibility.part_id
CREATE INDEX IF NOT EXISTS idx_parts_model_compatibility_part_id_fk ON public.parts_model_compatibility (part_id);

-- 6) Transaction commit
COMMIT;

-- Schema creation completed successfully!
-- Total tables: 10
-- Total indexes: 25+
-- Security: RLS enabled on all tables
-- Vector search: 5 embedding columns optimized

-----------------------------
-- EXECUTION NOTES
-----------------------------
-- This schema is production-ready for AI PDF processing
-- All tables have UUID primary keys
-- RLS policies restrict access to service key only
-- Vector search enabled for semantic document search
-- Multi-manufacturer part compatibility supported
-- Document versioning and n8n chat memory included
