# Dokumenttypen und Tabellennamen - Empfehlung

## Problem: Inkonsistente Benennung zwischen Pipeline und Processor

In der aktuellen Implementierung gibt es eine Inkonsistenz zwischen den Dokumenttypen, die von der Pipeline erkannt werden, und den Tabellennamen, die in der Datenbank verwendet werden:

1. `ProcessingPipeline._detect_document_type()` verwendet:
   - "service_manuals"
   - "bulletins"
   - "parts_catalogs" (Plural)
   - "cpmd_documents"
   - "video_tutorials"

2. Aber die Datenbanktabellen heißen:
   - "service_manuals"
   - "bulletins"
   - "parts_catalog" (Singular!)
   - "cpmd_documents"
   - "video_tutorials"

3. Und im `DocumentProcessor` wird das Mapping wie folgt durchgeführt:
   ```python
   table_mapping = {
       "service_manual": "service_manuals",
       "bulletin": "bulletins",
       "cpmd": "cpmd_documents",
       "parts_manual": "parts_catalog"  # Gemäß Supabase AI für Parts Catalog Chunks
   }
   ```

## Lösung: Einheitliches Mapping für alle Module

Hier ist die empfohlene Implementierung für ein einheitliches Mapping zwischen Dokumenttypen und Tabellennamen:

```python
# In modules/document_processing/processor.py anpassen:
def get_table_for_document_type(document_type: str) -> str:
    """
    Gibt die entsprechende Datenbanktabelle für einen Dokumenttyp zurück
    
    Args:
        document_type: Der Dokumenttyp
        
    Returns:
        str: Tabellenname
    """
    # Tabellen-Mapping für alle Dokumenttypen
    # Umfasst sowohl Pipeline als auch Processor Nomenklatur
    table_mapping = {
        # Pipeline-Typen (aus dem Pfad)
        "service_manuals": "service_manuals",
        "bulletins": "bulletins",
        "parts_catalogs": "parts_catalog",  # Wichtig: Plural zu Singular
        "cpmd_documents": "cpmd_documents",
        "video_tutorials": "video_tutorials",
        
        # Processor-Typen (interne Klassifikation)
        "service_manual": "service_manuals",
        "bulletin": "bulletins",
        "parts_manual": "parts_catalog",
        "cpmd": "cpmd_documents",
    }
    
    # Verwende das Mapping oder den ursprünglichen Typ als Fallback
    return table_mapping.get(document_type, document_type)
```

## Empfehlungen für das Processing Pipeline Modul

In `modules/processing_pipeline/processor.py` sollte die `_detect_document_type` Methode angepasst werden:

```python
def _detect_document_type(self, file_path: Path) -> Optional[str]:
    """
    Bestimmt den Dokumenttyp basierend auf dem Dateipfad
    
    Args:
        file_path: Pfad zum Dokument
            
    Returns:
        Optional[str]: Dokumenttyp oder None
    """
    path_str = str(file_path).lower()
    
    # Mapping zwischen Ordnernamen und Dokumenttypen
    folder_doc_types = {
        "service_manuals": "service_manuals",
        "bulletins": "bulletins",
        "parts_catalogs": "parts_catalogs",  # Behalte hier Plural
        "cpmd": "cpmd_documents",
        "video_tutorials": "video_tutorials"
    }
    
    for folder, doc_type in folder_doc_types.items():
        if folder.lower() in path_str:
            return doc_type
                
    return None
```

## Vorteile dieser Lösung

1. **Konsistenz:** Klare Zuordnung zwischen Pfad, Dokumenttyp und Datenbanktabelle
2. **Trennung der Zuständigkeiten:** Pipeline erkennt Dokumenttypen, Processor kümmert sich um die Tabellenzuordnung
3. **Flexibilität:** Neue Dokumenttypen können einfach hinzugefügt werden
4. **Robustheit:** Funktioniert mit allen bisher implementierten Dokumenttypen
5. **Wartbarkeit:** Zentrales Mapping für alle Tabellennamen

Diese Änderungen garantieren, dass die Dokumenttypen korrekt erkannt werden und alle Chunks in den richtigen Tabellen landen.