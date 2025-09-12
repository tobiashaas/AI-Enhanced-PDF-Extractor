-- ============================================
-- FUNCTION CLEANUP - Entfernt Function-Konflikte
-- WICHTIG: In Supabase SQL Editor ausführen BEVOR final_supabase_schema.sql
-- ============================================

-- Drop alle alten Function-Versionen mit falschen Parameter-Typen
DROP FUNCTION IF EXISTS public.check_file_processed(input_file_hash character varying);
DROP FUNCTION IF EXISTS public.start_processing(input_file_path text, input_file_hash character varying, input_filename text, input_document_info jsonb);
DROP FUNCTION IF EXISTS public.update_processing_progress(log_id bigint, new_stage text, progress integer, chunks_count integer, images_count integer);
DROP FUNCTION IF EXISTS public.complete_processing(log_id bigint, processing_time integer);
DROP FUNCTION IF EXISTS public.enhanced_vector_search_with_images(query_embedding vector, search_manufacturer text, search_model text, search_problem_type text, preferred_document_types text[], include_images boolean, match_threshold float, match_count integer);

-- Drop auch potentiell vorhandene alte Tabellen-Strukturen falls vorhanden
DROP TABLE IF EXISTS public.chunk_images CASCADE;

-- Drop alte Array-basierte Spalten falls vorhanden
ALTER TABLE IF EXISTS public.chunks DROP COLUMN IF EXISTS related_images;

-- Cleanup abgeschlossen
SELECT 'Function cleanup completed - jetzt final_supabase_schema.sql ausführen' as status;