# 📁 Documents Directory

This directory is where you place your PDF files for processing.

## Structure
```
Documents/
├── .gitkeep                    # Keeps this directory in git
├── README.md                   # This file
├── your-pdf-file-1.pdf         # Your PDF files go here
├── your-pdf-file-2.pdf         # (ignored by git)
└── subfolder/
    └── more-pdfs.pdf           # Subdirectories supported
```

## Usage

1. **Place PDF files here** that you want to process
2. **Run the processor**: `python3 ai_pdf_processor.py`
3. **PDFs are automatically detected** and processed with AI enhancement

## Notes

- ✅ **PDF files are gitignored** - they won't be committed to the repository
- ✅ **Directory structure preserved** - this folder exists in all clones
- ✅ **Recursive processing** - subdirectories are also scanned
- ✅ **Progress saving** - processing can be interrupted and resumed

## Cross-Platform Paths

The `documents_path` in config.json should point here:

- **macOS**: `/Users/YourName/path/to/AI-Enhanced-PDF-Extractor/Documents`
- **Windows**: `C:\Users\YourName\path\to\AI-Enhanced-PDF-Extractor\Documents`  
- **Linux**: `/home/yourname/path/to/AI-Enhanced-PDF-Extractor/Documents`

The `cross_platform_setup.py` script automatically sets the correct path for your system.
