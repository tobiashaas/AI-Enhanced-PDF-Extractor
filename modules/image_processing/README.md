# Image Processing Module

This module handles the extraction and processing of images from documents, implementing the ZERO CONVERSION POLICY.

## Key Features
- Zero conversion policy (preserves original image formats)
- Support for both raster and vector graphics
- Optional vision analysis with AI models
- Metadata extraction from images
- Storage in R2/S3 compatible storage

## ZERO CONVERSION POLICY
All images are stored in their original format without conversion, ensuring maximum quality preservation:
- Raster formats: PNG, JPG, TIFF, BMP
- Vector formats: SVG, EPS, AI

## Classes
- `ImageProcessor`: Main class for image extraction and processing

## Integration
This module integrates with:
- Document processing (for source documents)
- Vision AI (optional for image analysis)
- Supabase/R2 storage for image archiving