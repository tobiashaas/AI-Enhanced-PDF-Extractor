# Document Processing Module

This module handles the processing of different document types:
- Service manuals
- Technical bulletins
- Parts catalogs
- CPMD (Control Panel Message Documents)

## Key Features
- Document type detection
- Automatic version extraction
- Model compatibility identification
- Intelligent chunking
- Text extraction with metadata

## Classes
- `DocumentProcessor`: Abstract base class
- `ServiceManualProcessor`: For service and repair manuals
- `BulletinProcessor`: For technical bulletins
- `PartsCatalogProcessor`: For parts catalogs 
- `CPMDProcessor`: For control panel message documents

## Integration
This module integrates with:
- Image processing (for image extraction)
- Parts management (for parts catalogs)
- Processing pipeline (for orchestration)