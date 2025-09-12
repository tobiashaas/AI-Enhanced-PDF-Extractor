-- ============================================
-- AI AGENT PERFORMANCE OPTIMIZATION - SAFE EXECUTION
-- Schritt-für-Schritt Database Optimierung
-- ============================================

-- STEP 1: TSV Column für Full-Text Search
-- (Manuell im Supabase Dashboard ausführen)
/*
ALTER TABLE public.parts_catalog 
ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', 
        COALESCE(part_number, '') || ' ' || 
        COALESCE(description, '') || ' ' || 
        COALESCE(category, '') || ' ' ||
        COALESCE(part_name, '') || ' ' ||
        array_to_string(COALESCE(model_compatibility, '{}'), ' ')
    )
) STORED;
*/

-- STEP 2: Basic Performance Indizes (sicher ausführbar)
CREATE INDEX IF NOT EXISTS idx_chunks_manufacturer_model 
ON public.chunks (manufacturer, model) 
WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_content_gin 
ON public.chunks USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_chunk_images_chunk_id 
ON public.chunk_images (chunk_id);

CREATE INDEX IF NOT EXISTS idx_chunk_images_image_id 
ON public.chunk_images (image_id);

CREATE INDEX IF NOT EXISTS idx_images_file_hash_page 
ON public.images (file_hash, page_number);

-- STEP 3: Parts Catalog Basic Indizes
CREATE INDEX IF NOT EXISTS idx_parts_model_compatibility 
ON public.parts_catalog USING gin(model_compatibility);

CREATE INDEX IF NOT EXISTS idx_parts_category_manufacturer 
ON public.parts_catalog (category, manufacturer);

CREATE INDEX IF NOT EXISTS idx_parts_part_number 
ON public.parts_catalog (part_number);

-- STEP 4: AI Agent Optimized Views (ohne Price/Availability)
CREATE OR REPLACE VIEW ai_agent_search_view AS
SELECT 
    c.id as chunk_id,
    c.content,
    c.manufacturer,
    c.model,
    c.page_number,
    c.metadata,
    
    -- Image Information
    i.id as image_id,
    i.storage_url,
    i.public_url,
    i.width,
    i.height,
    
    -- Optimized Parts Information (ohne Price/Availability)
    pc.part_number,
    pc.part_name,
    pc.description as part_description,
    pc.category,
    pc.model_compatibility,
    
    -- Part Match Quality für AI Agent
    CASE 
        WHEN c.model = ANY(pc.model_compatibility) THEN 100
        WHEN c.manufacturer = pc.manufacturer THEN 80
        ELSE 60
    END as part_relevance_score
    
FROM public.chunks c
LEFT JOIN public.chunk_images ci ON c.id = ci.chunk_id
LEFT JOIN public.images i ON ci.image_id = i.id
LEFT JOIN public.parts_catalog pc ON c.manufacturer = pc.manufacturer
WHERE pc.part_number IS NOT NULL;

-- STEP 5: Fast Parts Lookup View (ohne Price/Availability)
CREATE OR REPLACE VIEW parts_lookup_optimized AS
SELECT 
    part_number,
    manufacturer,
    part_name,
    description,
    category,
    model_compatibility,
    
    -- Quality ranking ohne Price-Abhängigkeit
    CASE 
        WHEN description IS NOT NULL AND part_name IS NOT NULL 
             AND array_length(model_compatibility, 1) > 0 THEN 1
        WHEN description IS NOT NULL AND part_name IS NOT NULL THEN 2
        WHEN description IS NOT NULL OR part_name IS NOT NULL THEN 3
        ELSE 4
    END as quality_rank,
    
    -- Model count for prioritization
    array_length(model_compatibility, 1) as model_count,
    
    -- Search helper (ohne Price)
    part_number || ' ' || 
    COALESCE(part_name, '') || ' ' ||
    COALESCE(description, '') || ' ' || 
    COALESCE(category, '') as search_text
    
FROM public.parts_catalog
WHERE part_number IS NOT NULL
ORDER BY quality_rank, manufacturer, part_number;

-- STEP 6: Helper Functions (PostgreSQL Functions)
CREATE OR REPLACE FUNCTION search_parts_simple(
    search_term text
)
RETURNS TABLE (
    part_number text,
    description text,
    category text,
    manufacturer text,
    best_price numeric
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pc.part_number,
        pc.description,
        pc.category,
        pc.manufacturer,
        COALESCE(pc.price, pc.price_msrp, pc.price_dealer) as best_price
    FROM public.parts_catalog pc
    WHERE 
        pc.part_number ILIKE '%' || search_term || '%' OR
        pc.description ILIKE '%' || search_term || '%' OR
        pc.category ILIKE '%' || search_term || '%'
    ORDER BY 
        CASE 
            WHEN pc.part_number ILIKE search_term || '%' THEN 1
            WHEN pc.part_number ILIKE '%' || search_term || '%' THEN 2
            WHEN pc.description ILIKE '%' || search_term || '%' THEN 3
            ELSE 4
        END,
        pc.manufacturer,
        pc.part_number
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- STEP 7: Performance Monitoring Table
CREATE TABLE IF NOT EXISTS ai_agent_query_performance (
    id bigserial PRIMARY KEY,
    query_type text NOT NULL,
    execution_time_ms integer NOT NULL,
    result_count integer NOT NULL,
    search_parameters jsonb,
    created_at timestamp with time zone DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_query_performance_type_time 
ON ai_agent_query_performance (query_type, created_at);

-- Comments for documentation
COMMENT ON VIEW ai_agent_search_view IS 'Optimized view for AI Agent searches combining chunks, images, and parts (without embedding dependency)';
COMMENT ON VIEW parts_lookup_optimized IS 'Fast parts lookup with quality ranking for AI Agent';
COMMENT ON FUNCTION search_parts_simple IS 'Simple text-based parts search for AI Agent fallback';
COMMENT ON TABLE ai_agent_query_performance IS 'Performance monitoring for AI Agent database queries';