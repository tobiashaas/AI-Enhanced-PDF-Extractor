# Parts Catalogs Organization

## Folder Structure:
```
Parts_Catalogs/
├── Konica_Minolta/
│   ├── C451i/
│   │   ├── C451i_Parts.pdf
│   │   ├── C451i_Parts.csv
│   │   └── metadata.json
│   ├── C552/
│   │   ├── C552_Parts.pdf
│   │   ├── C552_Parts.csv
│   │   └── metadata.json
├── HP/
│   ├── LaserJet_Pro_4000/
│   │   ├── HP_4000_Parts.pdf
│   │   ├── HP_4000_Parts.csv
│   │   └── metadata.json
└── Canon/
    └── imageRUNNER_ADVANCE/
        ├── ...
```

## Naming Convention:
- **PDF**: `{Model}_Parts.pdf`
- **CSV**: `{Model}_Parts.csv` (exact same basename)
- **Metadata**: `metadata.json` (optional, for tracking)

## Processing Rules:
1. **Pairing Required**: Only process if BOTH PDF + CSV exist
2. **Validation**: CSV serves as "ground truth" for PDF extraction
3. **Tracking**: Use metadata.json for processing history
4. **Error Handling**: Log mismatches, missing files

## Example metadata.json:
```json
{
  "manufacturer": "Konica Minolta",
  "model": "C451i",
  "catalog_version": "2024-03",
  "last_processed": "2024-09-12T10:30:00Z",
  "pdf_hash": "abc123...",
  "csv_hash": "def456...",
  "parts_count": {
    "csv": 1400,
    "pdf_extracted": 1390,
    "match_rate": 99.3
  },
  "processing_status": "completed"
}
```