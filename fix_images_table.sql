-- Fix für die images-Tabelle
-- Erstellt die Tabelle korrekt mit extensions.uuid_generate_v4()
-- Datum: 13. September 2025

-- Prüfe, ob die Tabelle bereits existiert
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables 
               WHERE table_schema = 'public' 
               AND table_name = 'images') THEN
        -- Tabelle löschen, wenn sie bereits existiert
        DROP TABLE IF EXISTS public.images;
    END IF;
END $$;

-- Erstelle die Tabelle mit korrekter UUID-Extension-Referenz
CREATE TABLE IF NOT EXISTS public.images (
  id uuid PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
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

-- Erstelle die notwendigen Indizes
CREATE INDEX IF NOT EXISTS idx_images_file_hash ON public.images (file_hash);
CREATE INDEX IF NOT EXISTS idx_images_source_table ON public.images (source_table);
CREATE INDEX IF NOT EXISTS idx_images_manufacturer_model ON public.images (manufacturer, model);

-- Berechtigungen setzen
ALTER TABLE public.images ENABLE ROW LEVEL SECURITY;
GRANT ALL ON public.images TO postgres;
GRANT ALL ON public.images TO service_role;