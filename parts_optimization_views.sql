
-- AI Agent Parts Lookup (optimiert, ohne Price/Availability)
CREATE OR REPLACE VIEW ai_parts_lookup_clean AS
SELECT 
    part_number,
    manufacturer,
    part_name,
    description,
    category,
    model_compatibility,
    
    -- Model Count für Prioritisierung
    array_length(model_compatibility, 1) as model_count,
    
    -- Search Priority basierend auf Datenqualität
    CASE 
        WHEN description IS NOT NULL AND part_name IS NOT NULL 
             AND array_length(model_compatibility, 1) > 0 THEN 1
        WHEN description IS NOT NULL OR part_name IS NOT NULL THEN 2
        ELSE 3
    END as search_priority,
    
    -- Combined search text
    part_number || ' ' || 
    COALESCE(part_name, '') || ' ' ||
    COALESCE(description, '') || ' ' || 
    COALESCE(category, '') as search_text
    
FROM public.parts_catalog
WHERE part_number IS NOT NULL
ORDER BY search_priority, manufacturer, part_number;

-- Fast Part Number Lookup
CREATE OR REPLACE FUNCTION get_part_by_number(search_pn text)
RETURNS TABLE (
    part_number text,
    manufacturer text,
    description text,
    compatible_models text[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pc.part_number,
        pc.manufacturer,
        pc.description,
        pc.model_compatibility
    FROM public.parts_catalog pc
    WHERE pc.part_number = search_pn
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Model Compatibility Search
CREATE OR REPLACE FUNCTION find_parts_for_model(target_model text)
RETURNS TABLE (
    part_number text,
    manufacturer text,
    description text,
    category text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pc.part_number,
        pc.manufacturer,
        pc.description,
        pc.category
    FROM public.parts_catalog pc
    WHERE target_model = ANY(pc.model_compatibility)
    ORDER BY pc.manufacturer, pc.part_number
    LIMIT 100;
END;
$$ LANGUAGE plpgsql;
