
"""
Parts Catalog Helper Functions - Optimiert für AI Agent
Entfernt Price/Availability Abhängigkeiten
"""

def get_part_by_number(db, part_number: str) -> dict:
    """Hole Part Details nach Part Number (eindeutige Referenz)"""
    result = db.supabase.table('parts_catalog')\
        .select('part_number, manufacturer, part_name, description, category, model_compatibility')\
        .eq('part_number', part_number)\
        .execute()
    
    return result.data[0] if result.data else None

def find_parts_by_model(db, model: str, limit: int = 50) -> list:
    """Finde alle Parts für ein spezifisches Model"""
    result = db.supabase.table('parts_catalog')\
        .select('part_number, manufacturer, description, category, model_compatibility')\
        .contains('model_compatibility', [model])\
        .limit(limit)\
        .execute()
    
    return result.data

def search_parts_optimized(db, search_term: str, limit: int = 20) -> list:
    """Optimierte Parts-Suche ohne Price/Availability"""
    # Exact match zuerst
    exact = db.supabase.table('parts_catalog')\
        .select('part_number, manufacturer, part_name, description, category')\
        .eq('part_number', search_term)\
        .execute()
    
    if exact.data:
        return exact.data
    
    # Fuzzy search
    fuzzy = db.supabase.table('parts_catalog')\
        .select('part_number, manufacturer, part_name, description, category')\
        .ilike('part_number', f'%{search_term}%')\
        .limit(limit)\
        .execute()
    
    return fuzzy.data

def get_parts_quality_stats(db) -> dict:
    """Analysiere Parts-Datenqualität"""
    result = db.supabase.table('parts_catalog')\
        .select('part_number, description, part_name, model_compatibility')\
        .execute()
    
    stats = {
        'total_parts': len(result.data),
        'with_description': 0,
        'with_part_name': 0,
        'with_models': 0,
        'complete_parts': 0
    }
    
    for part in result.data:
        if part.get('description'):
            stats['with_description'] += 1
        if part.get('part_name'):
            stats['with_part_name'] += 1
        if part.get('model_compatibility') and len(part.get('model_compatibility', [])) > 0:
            stats['with_models'] += 1
        if (part.get('description') and part.get('part_name') and 
            part.get('model_compatibility') and len(part.get('model_compatibility', [])) > 0):
            stats['complete_parts'] += 1
    
    return stats
