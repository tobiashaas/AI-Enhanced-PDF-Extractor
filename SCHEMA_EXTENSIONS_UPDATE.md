-- AI-Enhanced PDF Processing Schema mit Extensions Schema
-- Aktualisiert nach Supabase Best Practices
-- Datum: 13. September 2025
-- 
-- Änderungen:
-- - Extension "vector" in separatem Schema "extensions"
-- - Verbesserter Setup-Prozess für Extensions
-- - Alle anderen Einstellungen bleiben gleich

BEGIN;

-- 1) Schemas
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE SCHEMA IF NOT EXISTS private;

-- 2) Extensions in separates Schema verschieben
-- Extension-Suche
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

-- 5) Tabellen (wie in der vorherigen Version)

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

-- Ähnliche Anpassungen für die anderen Tabellen...

-- 6) Indizes (nutzen extensions.vector)
CREATE INDEX IF NOT EXISTS idx_service_manuals_embedding ON public.service_manuals USING hnsw (embedding extensions.vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_bulletins_embedding ON public.bulletins USING hnsw (embedding extensions.vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_parts_catalogs_embedding ON public.parts_catalogs USING hnsw (embedding extensions.vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_cpmd_embedding ON public.cpmd_documents USING hnsw (embedding extensions.vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_video_tutorials_embedding ON public.video_tutorials USING hnsw (embedding extensions.vector_cosine_ops);

-- Rest der Tabellen und Indizes wie in Ihrer ursprünglichen Datei...

COMMIT;